"""Base data structures and shared HTTP client base for search providers."""

from __future__ import annotations

import asyncio
import math
import random
import string
from abc import ABC, abstractmethod
from typing import Any, Callable, Self

import logfire
from httpx import AsyncClient, HTTPStatusError, Response, TimeoutException, TransportError
from pydantic import BaseModel, Field, HttpUrl
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


class BaseHttpSearchProvider(ABC):
    """Abstract base class managing rate limits, HTTP sessions, retries, and caching for search providers."""

    settings: SearchSettings
    provider_name: str
    _client: AsyncClient | None
    _cache: dict[str, Any]
    _limiter: Limiter

    def __init__(self, settings: SearchSettings, provider_name: str) -> None:
        """Initializes the base HTTP search provider.

        Args:
            settings: Search configuration settings.
            provider_name: Name identifier of the search provider.
        """
        self.settings = settings
        self.provider_name = provider_name
        self._cache = {}
        self._client = None

        if self.settings.qps < 1:
            rate = 1
            duration = int(math.ceil(1 / self.settings.qps)) * Duration.SECOND
        else:
            rate = int(self.settings.qps)
            duration = Duration.SECOND

        self._limiter = limiter_factory.create_inmemory_limiter(rate_per_duration=rate, duration=duration)

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Constructs provider-specific HTTP headers including authorization.

        Returns:
            dict[str, str]: Header key-value mapping.
        """
        ...

    @abstractmethod
    def get_tools(self) -> list[Callable[..., Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any]]: Bound tool methods.
        """
        ...

    async def __aenter__(self) -> Self:
        """Initializes the authenticated HTTP client session.

        Returns:
            Self: Active provider instance.

        Raises:
            ValueError: If the required API key is missing.
        """
        if self.settings.key is None:
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
        """Sends an HTTP request with rate limiting and exponential backoff retries.

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
                    backoff = min(
                        self.settings.max_wait_seconds,
                        (self.settings.min_wait_seconds * (self.settings.multiplier**attempt)) + random.uniform(0, 1),
                    )
                    logfire.warning(
                        f"{self.provider_name.capitalize()} {method} to {url} failed with {e} "
                        f"(attempt {attempt + 1}/{max_attempts}). Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise RuntimeError(f"{self.provider_name.capitalize()} request to {url} failed after {max_attempts} attempts.")
