"""Ollama web search API provider."""

from __future__ import annotations

from typing import Any, Callable

from ai_sub.config import SearchSettings
from ai_sub.search.base import BaseHttpSearchProvider, WebSearchResult


class OllamaSearchProvider(BaseHttpSearchProvider):
    """Encapsulates Ollama web search operations using the shared HTTP search base."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the Ollama provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="ollama")

    def _get_headers(self) -> dict[str, str]:
        """Constructs authorization headers for Ollama API requests.

        Returns:
            dict[str, str]: HTTP request headers.
        """
        key = self._get_api_key()
        assert key is not None
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound search methods.
        """
        return [self.web_search_ollama_multi]

    async def _do_search(self, query: str) -> list[WebSearchResult]:
        """Performs a single web search against the Ollama API.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search results.
        """
        response = await self._request("POST", "https://ollama.com/api/web_search", json={"query": query})
        data = response.json()

        results: list[WebSearchResult] = []
        for item in data.get("results") or []:
            title = item.get("title")
            url = item.get("url")
            content = item.get("content")
            if title and url and content:
                try:
                    results.append(WebSearchResult(title=title, url=url, content=content[: self.settings.max_length]))
                except Exception:
                    continue
        return results

    async def web_search_ollama_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using Ollama.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """
        return await self._gather_multi_search(queries)
