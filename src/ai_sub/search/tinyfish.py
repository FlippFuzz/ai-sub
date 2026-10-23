"""Tinyfish search and webpage content extraction API provider."""

from __future__ import annotations

from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import BaseHttpSearchProvider, WebSearchResult, normalize_query


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
            list[Callable[..., Any] | dict[str, Any]]: Both tinyfish_search and tinyfish_fetch methods.
        """
        return [self.tinyfish_search, self.tinyfish_fetch]

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Performs a single search query returning normalized search results.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """
        raw_results = await self.tinyfish_search(query)
        results: list[WebSearchResult] = []
        for item in raw_results:
            title = item.get("title")
            url = item.get("url")
            snippet = item.get("snippet")
            if title and url and snippet:
                results.append(WebSearchResult(title=title, url=url, content=snippet))
        return results

    async def tinyfish_search(
        self,
        query: str,
        purpose: str = "Search the web for factual information, background context, or media metadata",
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the web for information, context, terminology, or lyrics using the Tinyfish Search API.

        Args:
            query: Search query keywords (e.g. topics, people, terminology, song titles, or composers).
            purpose: A statement explaining the task or goal (e.g. 'Identify video lore', 'Find song lyrics').
            language: Optional ISO language code to tailor results (e.g. 'ja', 'en', 'ko', 'zh').

        Returns:
            list[dict[str, Any]]: List of ranked search results containing title, url, snippet, date, and site_name.
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

            results: list[dict[str, Any]] = []
            for item in raw_results:
                snippet = str(item.get("snippet") or "")[: self.settings.max_length]
                results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": snippet,
                        "site_name": item.get("site_name"),
                        "date": item.get("date"),
                    }
                )

            self._cache[cache_key] = results
            return results

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
