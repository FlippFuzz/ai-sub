"""Ollama web search API client.

Provides tools for performing single and multi-query web searches
using Ollama's web search endpoint.
"""

from __future__ import annotations

import asyncio

import logfire

from ai_sub.web_search import WebQueryResult, WebSearchDeps, WebSearchResult


@logfire.instrument("ollama_web_search: {query=}")
async def web_search_ollama_single(deps: WebSearchDeps, query: str) -> list[WebSearchResult]:
    """Perform a single web search query against Ollama's web search API.

    Args:
        deps: The web search dependencies.
        query: The search query string.

    Returns:
        list[WebSearchResult]: A list of WebSearchResult objects containing the search results.
    """
    cache_key = deps._normalize_query(query)
    if cache_key in deps._cache:
        logfire.debug("Using cached query.")
        return deps._cache[cache_key]

    response = await deps.post("https://ollama.com/api/web_search", json={"query": query})

    data = response.json()
    results = [
        WebSearchResult(
            title=title,
            url=url,
            content=content,
        )
        for result in (data.get("results") or [])
        if (title := result.get("title"))
        and (url := result.get("url"))
        and (content := result.get("content"))
        and len(content) <= deps._settings.max_length
    ]
    deps._cache[cache_key] = results
    return results


async def web_search_ollama_multi(deps: WebSearchDeps, queries: list[str]) -> list[WebQueryResult]:
    """Perform multiple web search queries against Ollama's web search API.

    Args:
        deps: The web search dependencies.
        queries: A list of search query strings.

    Returns:
        list[WebQueryResult]: A list of QueryResult objects, each containing the original query and its search results.
    """

    async def _search(query: str) -> WebQueryResult:
        results = await web_search_ollama_single(deps, query)
        return WebQueryResult.model_validate({"query": query, "results": results})

    return await asyncio.gather(*[_search(q) for q in queries])
