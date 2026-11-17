"""Tinyfish search and webpage content extraction API provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseHttpSearchProvider, WebSearchResult, normalize_query


class TinyfishSearchProvider(BaseHttpSearchProvider):
    """Encapsulates Tinyfish Search and Fetch operations using the shared HTTP search base."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the Tinyfish search provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="tinyfish")

    def _get_headers(self) -> dict[str, str]:
        """Constructs authorization and content-type headers for Tinyfish API requests.

        Returns:
            dict[str, str]: HTTP request headers.

        Raises:
            ValueError: If the Tinyfish API key is not configured.
        """
        key = self._get_api_key()
        if key is None:
            raise ValueError("Tinyfish API key is required but not configured.")
        return {"X-API-Key": key.get_secret_value(), "Content-Type": "application/json"}

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns search and fetch tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Both tinyfish_search_multi and tinyfish_fetch methods.
        """
        return [self.tinyfish_search_multi, self.tinyfish_fetch]

    async def search_single(
        self,
        query: str,
        purpose: str = "Search the web for factual information, background context, or media metadata",
        language: str | None = None,
    ) -> list[WebSearchResult]:
        """Performs a single search query returning normalized search results.

        Args:
            query: Search query keywords.
            purpose: A statement explaining the task or goal. Defaults to general web search.
            language: Optional ISO language code to tailor results.

        Returns:
            list[WebSearchResult]: Formatted search result items containing title, url, and content.
        """
        params: dict[str, str] = {"query": query, "purpose": purpose}
        if language:
            params["language"] = language

        async def _fetch() -> list[WebSearchResult]:
            response = await self._request("GET", "https://api.search.tinyfish.ai", params=params)
            raw_results = response.json().get("results", [])
            return self._to_web_results(
                (item.get("title"), item.get("url"), item.get("snippet")) for item in raw_results
            )

        return await self._cached_search(
            cache_key=f"search:{normalize_query(query)}:{purpose}:{language}",
            span_template="tinyfish_search: {query=}",
            fetcher=_fetch,
            cache_empty=True,
            query=query,
            purpose=purpose,
            language=language or "auto",
        )

    async def tinyfish_search_multi(
        self,
        queries: list[str],
        purpose: str = "Search the web for factual information, background context, or media metadata",
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute up to 5 web search queries simultaneously in a single turn using Tinyfish Search API.

        Always bundle all related terms, people, mechanics, or lore items into the queries list at once
        rather than searching one by one across multiple turns.

        Args:
            queries: List of search query strings to execute concurrently (up to 5 queries).
            purpose: A shared statement explaining the task or goal (e.g. 'Identify VTuber lore and terminology').
            language: Optional ISO language code to tailor results (e.g. 'ja', 'en', 'ko', 'zh').

        Returns:
            list[dict[str, Any]]: List of search results grouped by their originating queries.
        """
        return await self._gather_multi_search(
            queries=queries,
            search_fn=lambda q: self.search_single(query=q, purpose=purpose, language=language),
            max_queries=5,
        )

    async def tinyfish_fetch(
        self,
        urls: list[str],
        purpose: str = "Fetch and extract full web page content in Markdown format",
    ) -> list[dict[str, Any]]:
        """Fetch full extracted page content (Markdown) from specific URLs using the Tinyfish Fetch API.

        Use this tool when search snippets are incomplete or truncated, and you need to read the full,
        untruncated text from specific web pages found in search results.

        Args:
            urls: List of web page URLs to fetch and extract content from (maximum 10 URLs).
            purpose: A short statement explaining the goal of fetching these pages (e.g. 'Read full article').

        Returns:
            list[dict[str, Any]]: List of extracted web pages containing markdown text, title, url, and language.
        """
        valid_urls = [u for u in urls if u and u.startswith(("http://", "https://"))][:10]
        if not valid_urls:
            return []

        payload = {"urls": valid_urls, "purpose": purpose, "format": "markdown"}

        async def _fetch() -> list[dict[str, Any]]:
            response = await self._request("POST", "https://api.fetch.tinyfish.ai", json=payload)
            raw_results = response.json().get("results", [])

            max_len = self.settings.maxlength * 2
            return [
                {
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "language": page.get("language"),
                    "text": str(page.get("text") or "")[:max_len],
                }
                for page in raw_results
            ]

        return await self._cached_search(
            cache_key=f"fetch:{','.join(sorted(valid_urls))}:{purpose}",
            span_template="tinyfish_fetch: {urls=}",
            fetcher=_fetch,
            cache_empty=True,
            urls=valid_urls,
            purpose=purpose,
        )
