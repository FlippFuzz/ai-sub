"""Unified web search models and dependency container.

This module defines the data models for search results and the core
dependency container used to orchestrate web search operations across
different providers.
"""

import asyncio
import math
import random
import string
from typing import Any, Self

import logfire
from httpx import AsyncClient, HTTPStatusError, Response, TimeoutException, TransportError
from pydantic import BaseModel, Field, HttpUrl
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import WebSearchSettings


class WebSearchResult(BaseModel):
    """A single result returned by a web search API."""

    title: str = Field(description="The display title of the search result")
    url: HttpUrl = Field(description="The destination URL the search result links to.")
    content: str = Field(description="The page content relevant to the query")


class WebQueryResult(BaseModel):
    """Groups search results with their original query."""

    query: str = Field(description="The original search query")
    results: list[WebSearchResult] = Field(description="The list of search results for this query")


class WebSearchDeps:
    """Dependency container for web search operations.

    Manages the HTTP client, rate limiting, and caching for web search providers.
    """

    _settings: WebSearchSettings
    _client: AsyncClient
    _cache: dict[str, list[WebSearchResult]]
    _limiter: Limiter
    _provider: str

    def __init__(self, settings: WebSearchSettings, provider: str):
        """Initializes the WebSearchDeps.

        Args:
            settings: A WebSearchSettings instance.
            provider: The name of the provider (e.g., 'ollama', 'langsearch') for rate limiting.
        """
        self._settings = settings
        self._provider = provider
        self._cache = {}

        # Handle fractional QPS by scaling the duration.
        # e.g., 0.5 QPS becomes 1 query per 2 seconds.
        if self._settings.qps < 1:
            rate = 1
            # Convert to float to resolve Pylance operator issues and round up with math.ceil.
            duration = int(math.ceil(1 / self._settings.qps)) * Duration.SECOND
        else:
            rate = int(self._settings.qps)
            duration = Duration.SECOND

        self._limiter = limiter_factory.create_inmemory_limiter(
            rate_per_duration=rate,
            duration=duration,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize a query by removing punctuation and case-folding.

        Args:
            query: The raw search query string.

        Returns:
            A normalized version of the query suitable for cache-key comparison.
        """
        translator = str.maketrans("", "", string.punctuation)
        return query.translate(translator).casefold()

    async def __aenter__(self) -> Self:
        """Initialize the underlying httpx.AsyncClient and enter its context.

        Returns:
            The initialized session instance.

        Raises:
            ValueError: If the API key is not configured.
        """
        if self._settings.key is None:
            raise ValueError(f"{self._provider.capitalize()} API key is not configured")
        headers = {"Authorization": f"Bearer {self._settings.key.get_secret_value()}"}
        self._client = AsyncClient(headers=headers, timeout=self._settings.timeout)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Close the underlying httpx.AsyncClient when exiting the context.

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The traceback, if an exception was raised.
        """
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def post(self, url: str, json: dict[str, Any]) -> Response:
        """Send a POST request to the server with retries and exponential backoff.

        Args:
            url: The destination URL.
            json: The JSON payload for the request body.

        Returns:
            The HTTP response object.

        Raises:
            httpx.HTTPStatusError: If an unretryable HTTP error occurs or retries are exhausted.
            httpx.TransportError: If a transport error occurs and retries are exhausted.
            httpx.TimeoutException: If a request times out and retries are exhausted.
            ConnectionError: If a connection error occurs and retries are exhausted.
            asyncio.TimeoutError: If an async operation times out and retries are exhausted.
        """
        max_attempts = self._settings.retries + 1
        for attempt in range(max_attempts):
            await self._limiter.try_acquire_async(self._provider, blocking=True)
            try:
                response = await self._client.post(url, json=json)
                response.raise_for_status()
                return response
            except (
                HTTPStatusError,
                TransportError,
                TimeoutException,
                ConnectionError,
                asyncio.TimeoutError,
            ) as e:
                is_retryable = False
                if isinstance(e, HTTPStatusError):
                    status_code = e.response.status_code
                    is_retryable = status_code in (429, 500, 502, 503, 504)
                else:
                    is_retryable = True

                if is_retryable and attempt < self._settings.retries:
                    backoff = min(
                        self._settings.max_wait_seconds,
                        (self._settings.min_wait_seconds * (self._settings.multiplier**attempt)) + random.uniform(0, 1),
                    )
                    logfire.warning(
                        f"Web search POST to {url} failed with {e} (attempt {attempt + 1}/{max_attempts}). "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        assert False, "Unreachable code path"
