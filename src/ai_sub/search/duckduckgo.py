"""DuckDuckGo web search provider."""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Callable

import logfire
from ddgs import DDGS
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import SearchSettings
from ai_sub.search.base import BaseSearchProvider, WebSearchResult


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

    _limiter: Limiter

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the DuckDuckGo provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="duckduckgo")
        rate = 1 if self.settings.qps < 1 else int(self.settings.qps)
        duration = int(math.ceil(1 / self.settings.qps)) * Duration.SECOND if self.settings.qps < 1 else Duration.SECOND
        self._limiter = limiter_factory.create_inmemory_limiter(rate_per_duration=rate, duration=duration)

    async def _do_search(self, query: str) -> list[WebSearchResult]:
        """Performs a DuckDuckGo search query.

        Args:
            query: Search query string.

        Returns:
            list[WebSearchResult]: Formatted search result items.
        """
        max_attempts = self.settings.retries + 1
        raw_results: list[dict[str, Any]] = []

        for attempt in range(max_attempts):
            await self._limiter.try_acquire_async("duckduckgo", blocking=True)
            try:
                raw_results = await asyncio.to_thread(_fetch_ddgs, query, 10)
                break
            except Exception as e:
                if attempt < self.settings.retries:
                    backoff = min(
                        self.settings.max_wait_seconds,
                        (self.settings.min_wait_seconds * (self.settings.multiplier**attempt)) + random.uniform(0, 1),
                    )
                    logfire.warning(
                        f"DuckDuckGo search for '{query}' failed with {e} (attempt {attempt + 1}/{max_attempts}). "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                logfire.error(f"DuckDuckGo search for '{query}' exhausted all {max_attempts} attempts: {e}")
                return []

        results: list[WebSearchResult] = []
        for item in raw_results:
            title = item.get("title")
            url = item.get("href")
            body = item.get("body")
            if title and url and body:
                try:
                    results.append(WebSearchResult(title=title, url=url, content=str(body)[: self.settings.max_length]))
                except Exception:
                    continue
        return results

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
