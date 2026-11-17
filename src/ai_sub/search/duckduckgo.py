"""DuckDuckGo web search provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ddgs import DDGS

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseSearchProvider, WebSearchResult


def _fetch_ddgs(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Executes a synchronous DuckDuckGo search query.

    Args:
        query: Search query string.
        max_results: Maximum results to retrieve. Defaults to 10.

    Returns:
        list[dict[str, Any]]: Raw response dictionaries.
    """
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


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
        raw_results = await self._execute_with_retry(
            action_desc=f"DuckDuckGo search for '{query}'",
            operation=lambda: asyncio.to_thread(_fetch_ddgs, query, 10),
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
