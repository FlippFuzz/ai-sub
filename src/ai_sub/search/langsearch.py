"""Langsearch web search API provider."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import (
    BaseHttpSearchProvider,
    WebQueryResult,
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
        assert self.settings.key is not None
        return {"Authorization": f"Bearer {self.settings.key.get_secret_value()}"}

    def get_tools(self) -> list[Callable[..., Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any]]: Bound search methods.
        """
        return [self.web_search_langsearch_multi]

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single web search against the Langsearch API.

        Args:
            query: Search query string.

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
                    results.append(WebSearchResult(title=name, url=url, content=content))

            self._cache[cache_key] = results
            return results

    async def web_search_langsearch_multi(self, queries: list[str]) -> list[dict[str, Any]]:
        """Perform multiple web search queries in parallel using Langsearch.

        Args:
            queries: A list of search query strings to execute concurrently.

        Returns:
            list[dict[str, Any]]: List of search results grouped by their original queries.
        """

        async def _run(q: str) -> dict[str, Any]:
            res = await self.search_single(q)
            return WebQueryResult.model_validate({"query": q, "results": res}).model_dump(mode="json")

        return await asyncio.gather(*(_run(q) for q in queries))
