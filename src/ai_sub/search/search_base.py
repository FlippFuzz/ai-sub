"""Base data structures, HTTP client base, and Retry-After parser for search providers."""

from __future__ import annotations

import asyncio
import inspect
import math
import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from fractions import Fraction
from typing import Any, LiteralString, Self, TypeVar

import logfire
from httpx import AsyncClient, HTTPStatusError, Response, TimeoutException, TransportError
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import SearchSettings

T = TypeVar("T")


class _BoundedCache(OrderedDict[str, Any]):
    """Thread-safe bounded in-memory cache with FIFO eviction."""

    max_size: int

    def __init__(self, max_size: int = 1000) -> None:
        """Initializes the bounded cache.

        Args:
            max_size: Maximum entries before evicting oldest item. Defaults to 1000.
        """
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a key, moving it to the end or evicting oldest when capacity is exceeded.

        Args:
            key: Cache key string.
            value: Value to store.
        """
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


class WebSearchResult(BaseModel):
    """A single result snippet returned by a web search provider."""

    title: str = Field(description="Display title of the search result.")
    url: HttpUrl = Field(description="Destination URL of the webpage.")
    content: str = Field(description="Extracted textual content relevant to the search query.")


class WebQueryResult(BaseModel):
    """Search results grouped with their originating query string."""

    query: str = Field(description="The original search query.")
    results: list[WebSearchResult] = Field(description="Retrieved search result items.")
    error: str | None = Field(default=None, description="Error message if this query failed.")


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
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


class BaseSearchProvider(ABC):
    """Abstract base class for all web search and lyrics providers."""

    settings: SearchSettings
    provider_name: str
    _cache: _BoundedCache
    _limiter: Limiter

    def __init__(self, settings: SearchSettings, provider_name: str) -> None:
        """Initializes the search provider.

        Args:
            settings: Search configuration settings.
            provider_name: Identifier name of the provider.
        """
        self.settings = settings
        self.provider_name = provider_name
        self._cache = _BoundedCache(max_size=1000)
        self._limiter = self._create_rate_limiter()

    def _create_rate_limiter(self) -> Limiter:
        """Creates an in-memory QPS rate limiter preserving fractional rates from search settings.

        Returns:
            Limiter: Configured rate limiter instance.
        """
        qps = self.settings.qps
        second_ms = int(Duration.SECOND.value)
        if qps < 1.0:
            rate = 1
            duration_ms = max(1, math.ceil(second_ms / qps))
        elif float(qps).is_integer():
            rate = int(qps)
            duration_ms = second_ms
        else:
            frac = Fraction(str(qps)).limit_denominator(20)
            rate = frac.numerator
            duration_ms = frac.denominator * second_ms
        return limiter_factory.create_inmemory_limiter(rate_per_duration=rate, duration=duration_ms)

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
        """Closes the provider async context and underlying rate limiter.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if raised.
        """
        self._limiter.close()

    @abstractmethod
    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables or tool specification dictionaries for model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound tool callables or Interactions tool dictionaries.
        """
        ...

    def _to_web_results(self, entries: Iterable[tuple[Any, Any, Any]]) -> list[WebSearchResult]:
        """Normalizes raw (title, url, content) tuples into validated WebSearchResult items.

        Args:
            entries: Iterable of raw (title, url, content) tuples.

        Returns:
            list[WebSearchResult]: Validated and truncated search result models.
        """
        max_len = self.settings.maxlength
        results: list[WebSearchResult] = []
        for title, url, content in entries:
            if title and url and content:
                try:
                    results.append(
                        WebSearchResult(
                            title=str(title),
                            url=url,
                            content=str(content)[:max_len],
                        )
                    )
                except Exception:
                    continue
        return results

    async def _execute_with_retry(
        self,
        action_desc: str,
        operation: Callable[[], Awaitable[T]],
        is_retryable: Callable[[Exception], bool] | None = None,
        get_retry_after: Callable[[Exception], float | None] | None = None,
    ) -> T:
        """Executes an operation under rate limiting with exponential backoff retries.

        Args:
            action_desc: Description of the request for warning logs.
            operation: Zero-argument async callable executing the lookup.
            is_retryable: Predicate determining whether a raised exception should be retried.
                Defaults to retrying all exceptions.
            get_retry_after: Optional extractor returning Retry-After seconds from an exception.

        Returns:
            T: Result produced by the operation.

        Raises:
            RuntimeError: If retries are unexpectedly exhausted without returning or raising.
        """
        max_attempts = self.settings.retries + 1
        for attempt in range(max_attempts):
            await self._limiter.try_acquire_async(self.provider_name, blocking=True)
            try:
                return await operation()
            except Exception as e:
                can_retry = is_retryable(e) if is_retryable is not None else True
                if can_retry and attempt < self.settings.retries:
                    retry_after = get_retry_after(e) if get_retry_after is not None else None
                    backoff = self.settings.backoff.compute_delay(attempt, retry_after=retry_after)
                    logfire.warning(
                        f"{action_desc} failed with {e} (attempt {attempt + 1}/{max_attempts}). "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise RuntimeError(f"{action_desc} failed after {max_attempts} attempts.")

    async def _cached_search(
        self,
        cache_key: str,
        span_template: LiteralString,
        fetcher: Callable[[], Awaitable[Any]],
        cache_empty: bool = False,
        **span_attrs: Any,
    ) -> Any:
        """Executes an async lookup with bounded caching and Logfire span tracing.

        Args:
            cache_key: Normalized key for cache lookup and storage.
            span_template: Static Logfire span template string.
            fetcher: Zero-argument async callable returning search results.
            cache_empty: Whether to store empty result lists in cache. Defaults to False.
            **span_attrs: Span attributes passed to Logfire.

        Returns:
            Any: Cached or newly fetched results.
        """
        if cache_key in self._cache:
            logfire.debug(
                "Using cached {provider} result for {cache_key=}",
                provider=self.provider_name,
                cache_key=cache_key,
            )
            return self._cache[cache_key]

        with logfire.span(span_template, **span_attrs):
            results = await fetcher()

        if results or cache_empty:
            self._cache[cache_key] = results
        return results

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single search query with caching and telemetry spans.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """
        return await self._cached_search(
            cache_key=normalize_query(query),
            span_template="{provider}_search: {query=}",
            fetcher=lambda: self._do_search(query),
            provider=self.provider_name,
            query=query,
        )

    async def _do_search(self, query: str) -> list[WebSearchResult]:
        """Executes provider-specific single search query. Subclasses override this.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """
        return []

    async def _gather_multi_search(
        self,
        queries: list[str],
        search_fn: Callable[[str], Any] | None = None,
        max_queries: int | None = None,
    ) -> list[dict[str, Any]]:
        """Gathers search results for multiple queries concurrently into standardized query-result dicts.

        Individual query failures are isolated per item so successful results in the same
        batch are still returned immediately.

        Args:
            queries: A list of search query strings to execute concurrently.
            search_fn: Optional custom callable taking a single query. Defaults to None.
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
            try:
                if search_fn is not None:
                    res = search_fn(q)
                    if inspect.isawaitable(res):
                        res = await res
                else:
                    res = await self.search_single(q)
                return WebQueryResult.model_validate({"query": q, "results": res}).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            except Exception as e:
                logfire.warning(f"{self.provider_name.capitalize()} query '{q}' failed: {e}")
                return WebQueryResult(query=q, results=[], error=str(e)).model_dump(mode="json", exclude_none=True)

        return await asyncio.gather(*(_run(q) for q in clean_queries))


class BaseHttpSearchProvider(BaseSearchProvider, ABC):
    """Abstract base class managing HTTP sessions, rate limits, retries, and caching for HTTP providers."""

    requires_api_key: bool
    _client: AsyncClient | None

    def __init__(self, settings: SearchSettings, provider_name: str, requires_api_key: bool = True) -> None:
        """Initializes the base HTTP search provider.

        Args:
            settings: Search configuration settings.
            provider_name: Name identifier of the search provider.
            requires_api_key: Whether an API key is mandatory. Defaults to True.
        """
        super().__init__(settings, provider_name=provider_name)
        self.requires_api_key = requires_api_key
        self._client = None

    def _get_api_key(self) -> SecretStr | None:
        """Resolves the API key for this provider from settings.

        Returns:
            SecretStr | None: Secret key or None.
        """
        return self.settings.get_key_for_provider(self.provider_name)

    def _bearer_headers(self) -> dict[str, str]:
        """Builds a standard Bearer Authorization header using this provider's API key.

        Returns:
            dict[str, str]: Authorization header mapping.

        Raises:
            ValueError: If the required API key is not configured.
        """
        key = self._get_api_key()
        if key is None:
            raise ValueError(f"{self.provider_name.capitalize()} API key is required but not configured.")
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

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
        self._client = AsyncClient(headers=self._get_headers(), timeout=self.settings.timeout.total_seconds())
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Closes the HTTP client session and rate limiter.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if raised.
        """
        try:
            if self._client:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
                self._client = None
        finally:
            await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _request(self, method: str, url: str, **kwargs: Any) -> Response:
        """Sends an HTTP request with rate limiting, Retry-After compliance, and exponential backoff.

        Args:
            method: HTTP method (e.g. 'GET', 'POST').
            url: Destination endpoint URL.
            **kwargs: Extra parameters passed to the HTTP client.

        Returns:
            Response: Validated HTTP response.

        Raises:
            RuntimeError: If client session is not active.
        """
        client = self._client
        if client is None:
            raise RuntimeError(
                f"{self.provider_name.capitalize()} client session is not active. Use within 'async with'."
            )

        async def _send() -> Response:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        def _is_retryable(e: Exception) -> bool:
            if isinstance(e, HTTPStatusError):
                return e.response.status_code in (429, 500, 502, 503, 504)
            return isinstance(e, (TransportError, TimeoutException, ConnectionError, asyncio.TimeoutError))

        def _get_retry_after(e: Exception) -> float | None:
            if isinstance(e, HTTPStatusError):
                return parse_retry_after(e.response.headers.get("retry-after"))
            return None

        return await self._execute_with_retry(
            action_desc=f"{self.provider_name.capitalize()} {method} to {url}",
            operation=_send,
            is_retryable=_is_retryable,
            get_retry_after=_get_retry_after,
        )
