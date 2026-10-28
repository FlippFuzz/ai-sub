"""Tinyfish search and webpage content extraction API provider."""

from __future__ import annotations

from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import (
    BaseHttpSearchProvider,
    WebSearchResult,
    normalize_query,
)


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
        """
        key = self._get_api_key()
        assert key is not None
        return {
            "X-API-Key": key.get_secret_value(),
            "Content-Type": "application/json",
        }

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
        cache_key = f"search:{normalize_query(query)}:{purpose}:{language}"
        if cache_key in self._cache:
            logfire.debug("Using cached Tinyfish search result for {query=}", query=query)
            return self._cache[cache_key]

        params: dict[str, str] = {
            "query": query,
            "purpose": purpose,
        }
        if language:
            params["language"] = language

        with logfire.span(
            "tinyfish_search: {query=}",
            query=query,
            purpose=purpose,
            language=language or "auto",
        ):
            response = await self._request("GET", "https://api.search.tinyfish.ai", params=params)
            data = response.json()
            raw_results = data.get("results", [])

            results: list[WebSearchResult] = []
            max_len = self.settings.max_length
            for item in raw_results:
                title = item.get("title")
                url = item.get("url")
                raw_snippet = item.get("snippet")
                if title and url and raw_snippet:
                    content = str(raw_snippet)[:max_len]
                    try:
                        results.append(WebSearchResult(title=title, url=url, content=content))
                    except Exception:
                        continue

            self._cache[cache_key] = results
            return results

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

        cache_key = f"fetch:{','.join(sorted(valid_urls))}:{purpose}"
        if cache_key in self._cache:
            logfire.debug("Using cached Tinyfish fetch results for {urls=}", urls=valid_urls)
            return self._cache[cache_key]

        payload = {
            "urls": valid_urls,
            "purpose": purpose,
            "format": "markdown",
        }

        with logfire.span("tinyfish_fetch: {urls=}", urls=valid_urls, purpose=purpose):
            response = await self._request("POST", "https://api.fetch.tinyfish.ai", json=payload)
            data = response.json()
            raw_results = data.get("results", [])

            results: list[dict[str, Any]] = []
            max_len = self.settings.max_length * 2  # Allow generous length for full articles and lyric pages
            for page in raw_results:
                text = str(page.get("text") or "")[:max_len]
                results.append(
                    {
                        "url": page.get("url"),
                        "title": page.get("title"),
                        "language": page.get("language"),
                        "text": text,
                    }
                )

            self._cache[cache_key] = results
            return results
