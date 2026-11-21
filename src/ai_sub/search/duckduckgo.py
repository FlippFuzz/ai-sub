"""DuckDuckGo web search provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseSearchProvider, WebSearchResult


def _fetch_ddgs(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Executes a synchronous DuckDuckGo search query.

    Treats normal empty-result errors from ddgs as an empty list to match HTTP search providers,
    while propagating rate-limit, timeout, and unexpected backend errors for retry handling.

    Args:
        query: Search query string.
        max_results: Maximum results to retrieve. Defaults to 10.

    Returns:
        list[dict[str, Any]]: Raw response dictionaries, or an empty list if no results are found.

    Raises:
        DDGSException: If DuckDuckGo encounters rate-limiting, a timeout, or a backend error.
    """
    with DDGS() as ddgs:
        try:
            return list(ddgs.text(query, max_results=max_results))
        except DDGSException as e:
            if not isinstance(e, (RatelimitException, TimeoutException)) and "no results found" in str(e).lower():
                return []
            raise


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """Encapsulates DuckDuckGo search operations with caching and rate limiting."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the DuckDuckGo provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="duckduckgo")

    async def _do_search(self, query: str) -> list[WebSearchResult]:
        """Performs a DuckDuckGo search query.

        Args:
            query: Search query string.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """

        def _is_retryable(e: Exception) -> bool:
            return isinstance(e, (DDGSException, ConnectionError, asyncio.TimeoutError, OSError))

        raw_results = await self._execute_with_retry(
            action_desc=f"DuckDuckGo search for '{query}'",
            operation=lambda: asyncio.to_thread(_fetch_ddgs, query, 10),
            is_retryable=_is_retryable,
        )
        return self._to_web_results((item.get("title"), item.get("href"), item.get("body")) for item in raw_results)

    async def web_search_duckduckgo_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using DuckDuckGo.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """
        return await self._gather_multi_search(queries)

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound search methods.
        """
        return [self.web_search_duckduckgo_multi]
