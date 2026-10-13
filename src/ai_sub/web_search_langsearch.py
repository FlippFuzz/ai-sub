"""Langsearch web search API client.

Provides tools for performing single and multi-query web searches
using Langsearch's web search endpoint.
"""

from __future__ import annotations

import asyncio

import logfire

from ai_sub.web_search import WebQueryResult, WebSearchDeps, WebSearchResult


@logfire.instrument("langsearch_web_search: {query=}")
async def web_search_langsearch_single(deps: WebSearchDeps, query: str) -> list[WebSearchResult]:
    """Perform a single web search query against Langsearch's API.

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

    payload = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": 10,
    }
    response = await deps.post("https://api.langsearch.com/v1/web-search", json=payload)

    data = response.json()
    # The results are nested within data.webPages.value, requiring manual mapping
    raw_results = data.get("data", {}).get("webPages", {}).get("value", [])
    results = [
        WebSearchResult(
            title=name,
            url=url,
            content=summary,
        )
        for item in raw_results
        if (name := item.get("name"))
        and (url := item.get("url"))
        and (summary := item.get("summary"))
        and len(summary) <= deps._settings.max_length
    ]
    deps._cache[cache_key] = results
    return results


async def web_search_langsearch_multi(deps: WebSearchDeps, queries: list[str]) -> list[WebQueryResult]:
    """Perform multiple web search queries against Langsearch's API.

    Args:
        deps: The web search dependencies.
        queries: A list of search query strings.

    Returns:
        list[WebQueryResult]: A list of results grouped by their original queries.
    """

    async def _search(query: str) -> WebQueryResult:
        results = await web_search_langsearch_single(deps, query)
        return WebQueryResult.model_validate({"query": query, "results": results})

    return await asyncio.gather(*[_search(q) for q in queries])
