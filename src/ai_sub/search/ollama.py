"""Ollama web search API provider."""

from __future__ import annotations

from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import (
    BaseHttpSearchProvider,
    WebSearchResult,
    normalize_query,
)


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

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single web search against Ollama API.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search results.
        """
        cache_key = normalize_query(query)
        if cache_key in self._cache:
            logfire.debug("Using cached Ollama query result for {query=}", query=query)
            return self._cache[cache_key]

        with logfire.span("ollama_search: {query=}", query=query):
            response = await self._request("POST", "https://ollama.com/api/web_search", json={"query": query})
            data = response.json()

            results: list[WebSearchResult] = []
            for item in data.get("results") or []:
                title = item.get("title")
                url = item.get("url")
                content = item.get("content")
                if title and url and content:
                    try:
                        results.append(
                            WebSearchResult(title=title, url=url, content=content[: self.settings.max_length])
                        )
                    except Exception:
                        continue

            self._cache[cache_key] = results
            return results

    async def web_search_ollama_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using Ollama.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """
        return await self._gather_multi_search(queries)
