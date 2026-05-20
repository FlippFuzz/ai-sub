"""Ollama web search API client.

Provides tools for performing single and multi-query web searches
using Ollama's web search endpoint.
"""

import asyncio

import logfire
from pydantic_ai import RunContext

from ai_sub.data_models import AgentDeps
from ai_sub.web_search import WebQueryResult, WebSearchResult


@logfire.instrument("ollama_web_search: {query=}")
async def web_search_ollama_single(ctx: RunContext[AgentDeps], query: str) -> list[WebSearchResult]:
    """Perform a single web search query against Ollama's web search API.

    Args:
        ctx: The run context containing AgentDeps with ollama_search dependency.
        query: The search query string.

    Returns:
        A list of WebSearchResult objects containing the search results.
    """
    deps = ctx.deps.web_search
    assert deps is not None
    cache_key = deps._normalize_query(query)
    if cache_key in deps._cache:
        logfire.debug("Using cached query.")
        return deps._cache[cache_key]

    response = await deps.post("https://ollama.com/api/web_search", json={"query": query})
    response.raise_for_status()

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


async def web_search_ollama_multi(ctx: RunContext[AgentDeps], queries: list[str]) -> list[WebQueryResult]:
    """Perform multiple web search queries against Ollama's web search API.

    Args:
        ctx: The run context containing AgentDeps with ollama_search dependency.
        queries: A list of search query strings.

    Returns:
        A list of QueryResult objects, each containing the original query and its search results.
    """

    async def _search(query: str) -> WebQueryResult:
        results = await web_search_ollama_single(ctx, query)
        return WebQueryResult.model_validate({"query": query, "results": results})

    return await asyncio.gather(*[_search(q) for q in queries])
