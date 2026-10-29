"""Langsearch web search API provider."""

from __future__ import annotations

from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import (
    BaseHttpSearchProvider,
    WebSearchResult,
    normalize_query,
)


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
        key = self._get_api_key()
        assert key is not None
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound search methods.
        """
        return [self.web_search_langsearch_multi]

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single web search against the Langsearch API.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search results.
        """
        cache_key = normalize_query(query)
        if cache_key in self._cache:
            logfire.debug("Using cached Langsearch query result for {query=}", query=query)
            return self._cache[cache_key]

        max_length = self.settings.max_length
        payload = {
            "query": query,
            "count": 5,
            "contents": {"text": {"maxCharacters": max_length}},
            "freshness": "noLimit",
        }

        with logfire.span("langsearch_search: {query=}", query=query) as span:
            response = await self._request("POST", "https://api.langsearch.com/v1/web-search", json=payload)
            data = response.json()

            if usage := data.get("usage"):
                span.set_attribute("langsearch.usage", usage)
                if input_tokens := usage.get("input_tokens"):
                    span.set_attribute("langsearch.input_tokens", input_tokens)
                if output_tokens := usage.get("output_tokens"):
                    span.set_attribute("langsearch.output_tokens", output_tokens)

            raw_results = data.get("data", {}).get("webPages", {}).get("value", [])

            results: list[WebSearchResult] = []
            for item in raw_results:
                name = item.get("name")
                url = item.get("url")
                raw_content = item.get("text") or item.get("summary")
                if name and url and raw_content:
                    content = str(raw_content)[:max_length]
                    try:
                        results.append(WebSearchResult(title=name, url=url, content=content))
                    except Exception:
                        continue

            self._cache[cache_key] = results
            return results

    async def web_search_langsearch_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using Langsearch.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """
        return await self._gather_multi_search(queries)
