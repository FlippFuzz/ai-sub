"""Langsearch web search API provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseHttpSearchProvider, WebSearchResult


class LangsearchSearchProvider(BaseHttpSearchProvider):
    """Encapsulates Langsearch API operations using the shared HTTP search base."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the Langsearch provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="langsearch")

    def _get_headers(self) -> dict[str, str]:
        """Constructs authorization headers for Langsearch API requests.

        Returns:
            dict[str, str]: HTTP request headers.
        """
        return self._bearer_headers()

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound search methods.
        """
        return [self.web_search_langsearch_multi]

    async def _do_search(self, query: str) -> list[WebSearchResult]:
        """Performs a single web search against the Langsearch API.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search results.
        """
        payload = {
            "query": query,
            "count": 5,
            "contents": {"text": {"maxCharacters": self.settings.maxlength}},
            "freshness": "noLimit",
        }
        response = await self._request("POST", "https://api.langsearch.com/v1/web-search", json=payload)
        pages = response.json().get("data", {}).get("webPages", {}).get("value", [])
        return self._to_web_results(
            (item.get("name"), item.get("url"), item.get("text") or item.get("summary")) for item in pages
        )

    async def web_search_langsearch_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using Langsearch.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """
        return await self._gather_multi_search(queries)
