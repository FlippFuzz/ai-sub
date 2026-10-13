"""DuckDuckGo web search API client.

Provides tools for performing single and multi-query web searches
using the DuckDuckGo search library.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import logfire
from ddgs import DDGS

from ai_sub.web_search import WebQueryResult, WebSearchDeps, WebSearchResult


def _fetch_ddgs_results(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetches search results synchronously using DuckDuckGo search.

    Args:
        query: The search query string.
        max_results: Maximum results to retrieve.

    Returns:
        list[dict[str, Any]]: Raw search results from DuckDuckGo.
    """
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


@logfire.instrument("duckduckgo_web_search: {query=}")
async def web_search_duckduckgo_single(deps: WebSearchDeps, query: str) -> list[WebSearchResult]:
    """Perform a single web search query using DuckDuckGo.

    Args:
        deps: The web search dependencies.
        query: The search query string.

    Returns:
        list[WebSearchResult]: A list of search results.
    """
    cache_key = deps._normalize_query(query)
    if cache_key in deps._cache:
        logfire.debug("Using cached query.")
        return deps._cache[cache_key]

    max_attempts = deps._settings.retries + 1
    raw_results: list[dict[str, Any]] = []

    for attempt in range(max_attempts):
        await deps._limiter.try_acquire_async("duckduckgo", blocking=True)
        try:
            raw_results = await asyncio.to_thread(_fetch_ddgs_results, query, 10)
            break
        except Exception as e:
            if attempt < deps._settings.retries:
                backoff = min(
                    deps._settings.max_wait_seconds,
                    (deps._settings.min_wait_seconds * (deps._settings.multiplier**attempt)) + random.uniform(0, 1),
                )
                logfire.warning(
                    f"DuckDuckGo search for '{query}' failed with {e} (attempt {attempt + 1}/{max_attempts}). "
                    f"Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                continue
            logfire.error(f"DuckDuckGo search for '{query}' failed after {max_attempts} attempts: {e}")
            raw_results = []

    results = [
        WebSearchResult(
            title=title,
            url=url,
            content=body,
        )
        for item in raw_results
        if (title := item.get("title"))
        and (url := item.get("href"))
        and (body := item.get("body"))
        and len(body) <= deps._settings.max_length
    ]
    deps._cache[cache_key] = results
    return results


async def web_search_duckduckgo_multi(deps: WebSearchDeps, queries: list[str]) -> list[WebQueryResult]:
    """Perform multiple web search queries using DuckDuckGo.

    Args:
        deps: The web search dependencies.
        queries: A list of search query strings.

    Returns:
        list[WebQueryResult]: A list of results grouped by their original queries.
    """

    async def _search(query: str) -> WebQueryResult:
        results = await web_search_duckduckgo_single(deps, query)
        return WebQueryResult.model_validate({"query": query, "results": results})

    return await asyncio.gather(*[_search(q) for q in queries])


async def duckduckgo_search(deps: WebSearchDeps, query: str) -> list[WebSearchResult]:
    """Single query DuckDuckGo search tool alias.

    Args:
        deps: The web search dependencies.
        query: The search query string.

    Returns:
        list[WebSearchResult]: A list of search results.
    """
    return await web_search_duckduckgo_single(deps, query)
