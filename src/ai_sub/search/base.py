"""Base data structures, HTTP client base, and Retry-After parser for search providers."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import string
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Self

import logfire
from httpx import AsyncClient, HTTPStatusError, Response, TimeoutException, TransportError
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import SearchSettings


class WebSearchResult(BaseModel):
    """A single result snippet returned by a web search provider."""

    title: str = Field(description="Display title of the search result.")
    url: HttpUrl = Field(description="Destination URL of the webpage.")
    content: str = Field(description="Extracted textual content relevant to the search query.")


class WebQueryResult(BaseModel):
    """Search results grouped with their originating query string."""

    query: str = Field(description="The original search query.")
    results: list[WebSearchResult] = Field(description="Retrieved search result items.")


def normalize_query(query: str) -> str:
    """Strips punctuation and normalizes query case for deduplicated caching.

    Args:
        query: Raw search query string.

    Returns:
        str: Normalized query key.
    """
    translator = str.maketrans("", "", string.punctuation)
    return query.translate(translator).casefold()


def parse_retry_after(header_value: str | None) -> float | None:
    """Parses an HTTP Retry-After header value into a delay in seconds.

    Supports both delay-seconds and HTTP-date formats according to RFC 9110 §10.2.3.

    Args:
        header_value: Raw header value string.

    Returns:
        float | None: Delay in seconds if parseable, or None if missing or invalid.
    """
    if not header_value:
        return None
    cleaned = header_value.strip()
    try:
        return max(0.0, float(cleaned))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


class BaseSearchProvider(ABC):
    """Abstract base class for all web search and lyrics providers."""

    settings: SearchSettings
    provider_name: str
    _cache: dict[str, Any]

    def __init__(self, settings: SearchSettings, provider_name: str) -> None:
        """Initializes the search provider.

        Args:
            settings: Search configuration settings.
            provider_name: Identifier name of the provider.
        """
        self.settings = settings
        self.provider_name = provider_name
        self._cache = {}

    async def __aenter__(self) -> Self:
        """Initializes the provider async context.

        Returns:
            Self: Active provider instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Closes the provider async context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if raised.
        """
        return None

    @abstractmethod
    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables or tool specification dictionaries for model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound tool callables or Interactions tool dictionaries.
        """
        ...

    @abstractmethod
    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single search query returning normalized search results.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """
        ...

    async def _gather_multi_search(
        self,
        queries: list[str],
        search_fn: Callable[[str], Any] | None = None,
        max_queries: int | None = None,
    ) -> list[dict[str, Any]]:
        """Gathers search results for multiple queries concurrently into standardized query-result dicts.

        Args:
            queries: A list of search query strings to execute concurrently.
            search_fn: Optional custom callable taking a single query. Defaults to self.search_single.
            max_queries: Optional limit on the number of queries to process. Defaults to None.

        Returns:
            list[dict[str, Any]]: List of serialized WebQueryResult dictionaries.
        """
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if max_queries is not None:
            clean_queries = clean_queries[:max_queries]
        if not clean_queries:
            return []

        async def _run(q: str) -> dict[str, Any]:
            if search_fn is not None:
                res = search_fn(q)
                if inspect.isawaitable(res):
                    res = await res
            else:
                res = await self.search_single(q)
            return WebQueryResult.model_validate({"query": q, "results": res}).model_dump(mode="json")

        return await asyncio.gather(*(_run(q) for q in clean_queries))


class BaseHttpSearchProvider(BaseSearchProvider, ABC):
    """Abstract base class managing rate limits, HTTP sessions, retries, and caching for HTTP providers."""

    requires_api_key: bool
    _client: AsyncClient | None
    _limiter: Limiter

    def __init__(self, settings: SearchSettings, provider_name: str, requires_api_key: bool = True) -> None:
        """Initializes the base HTTP search provider.

        Args:
            settings: Search configuration settings.
            provider_name: Name identifier of the search provider.
            requires_api_key: Whether an API key is mandatory to initialize the client session.
        """
        super().__init__(settings, provider_name=provider_name)
        self.requires_api_key = requires_api_key
        self._client = None

        if self.settings.qps < 1:
            rate = 1
            duration = int(math.ceil(1 / self.settings.qps)) * Duration.SECOND
        else:
            rate = int(self.settings.qps)
            duration = Duration.SECOND

        self._limiter = limiter_factory.create_inmemory_limiter(rate_per_duration=rate, duration=duration)

    def _get_api_key(self) -> SecretStr | None:
        """Resolves the API key for this provider from settings.

        Returns:
            SecretStr | None: Secret key or None.
        """
        return self.settings.get_key_for_provider(self.provider_name)

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Constructs provider-specific HTTP headers including authorization.

        Returns:
            dict[str, str]: Header key-value mapping.
        """
        ...

    async def __aenter__(self) -> Self:
        """Initializes the authenticated HTTP client session.

        Returns:
            Self: Active provider instance.

        Raises:
            ValueError: If the required API key is missing.
        """
        if self.requires_api_key and self._get_api_key() is None:
            raise ValueError(f"{self.provider_name.capitalize()} API key is required but not configured.")

        headers = self._get_headers()
        self._client = AsyncClient(headers=headers, timeout=self.settings.timeout)
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Closes the HTTP client session.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if raised.
        """
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    async def _request(self, method: str, url: str, **kwargs: Any) -> Response:
        """Sends an HTTP request with rate limiting, Retry-After compliance, and exponential backoff.

        Args:
            method: HTTP method (e.g. 'GET', 'POST').
            url: Destination endpoint URL.
            **kwargs: Extra parameters passed to the HTTP client (e.g. params, json).

        Returns:
            Response: Validated HTTP response.

        Raises:
            RuntimeError: If client is uninitialized or retries exhausted.
            HTTPStatusError: If unretryable status code occurs or retries exhausted.
            TransportError: If network transport fails and retries exhausted.
            TimeoutException: If request times out and retries exhausted.
            ConnectionError: If connection fails and retries exhausted.
            TimeoutError: If async task times out and retries exhausted.
        """
        if self._client is None:
            raise RuntimeError(
                f"{self.provider_name.capitalize()} client session is not active. Use within 'async with'."
            )

        max_attempts = self.settings.retries + 1
        for attempt in range(max_attempts):
            await self._limiter.try_acquire_async(self.provider_name, blocking=True)
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (
                HTTPStatusError,
                TransportError,
                TimeoutException,
                ConnectionError,
                asyncio.TimeoutError,
            ) as e:
                is_retryable = (
                    e.response.status_code in (429, 500, 502, 503, 504) if isinstance(e, HTTPStatusError) else True
                )
                if is_retryable and attempt < self.settings.retries:
                    retry_after: float | None = None
                    if isinstance(e, HTTPStatusError):
                        retry_after = parse_retry_after(e.response.headers.get("retry-after"))

                    exp_backoff = self.settings.min_wait_seconds * (self.settings.multiplier**attempt)
                    effective_backoff = max(exp_backoff, retry_after or 0.0)
                    backoff = min(self.settings.max_wait_seconds, effective_backoff) + random.uniform(0, 1)

                    logfire.warning(
                        f"{self.provider_name.capitalize()} {method} to {url} failed with {e} "
                        f"(attempt {attempt + 1}/{max_attempts}). Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise RuntimeError(f"{self.provider_name.capitalize()} request to {url} failed after {max_attempts} attempts.")
