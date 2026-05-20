"""Unified web search models and dependency container.

This module defines the data models for search results and the core
dependency container used to orchestrate web search operations across
different providers.
"""

import math
import string
from typing import Any, Self

from httpx import AsyncClient, Response
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
        self._client = AsyncClient(headers=headers)
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
        """Send a POST request to the server.

        Args:
            url: The destination URL.
            json: The JSON payload for the request body.

        Returns:
            The HTTP response object.
        """
        # Enforce rate limit
        await self._limiter.try_acquire_async(self._provider, blocking=True)
        return await self._client.post(url, json=json)
