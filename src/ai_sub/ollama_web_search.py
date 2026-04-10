"""Ollama web search API client.

Provides tools for performing single and multi-query web searches
using Ollama's web search endpoint.
"""

import asyncio
from typing import Any, Self

import logfire
from httpx import AsyncClient, Response
from pydantic import BaseModel, Field, HttpUrl
from pydantic_ai import RunContext

from ai_sub.config import OllamaSearchSettings


class OllamaSearchResult(BaseModel):
    """A single result returned by Ollama's web search API."""

    title: str = Field(description="The display title of the search result")
    url: HttpUrl = Field(description="The destination URL the search result links to.")
    content: str = Field(description="The page content relevant to the query")


class QueryResult(BaseModel):
    """Groups search results with their original query."""

    query: str = Field(description="The original search query")
    results: list[OllamaSearchResult] = Field(description="The list of search results for this query")


class OllamaWebSearchDeps:
    """Dependency container for Ollama web search operations.

    Manages the HTTP client and API key configuration for Ollama's
    web search endpoint.
    """

    _settings: OllamaSearchSettings
    _client: AsyncClient

    def __init__(self, settings: OllamaSearchSettings):
        """Initializes the OllamaWebSearchDeps.

        Args:
            settings: An OllamaSearchSettings instance containing the API key.
        """
        self._settings = settings

    async def __aenter__(self) -> Self:
        """Initialize the underlying httpx.AsyncClient and enter its context.

        Returns:
            The initialized session instance.

        Raises:
            ValueError: If the Ollama API key is not configured.
        """
        if self._settings.key is None:
            raise ValueError("Ollama API key is not configured")
        headers = {"Authorization": f"Bearer {self._settings.key.get_secret_value()}"}
        self._client = AsyncClient(headers=headers)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Close the underlying httpx.AsyncClient when exiting the context."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def post(self, url: str, json: dict[str, Any]) -> Response:
        """Send a POST request to the given URL with the provided JSON payload.

        Args:
            url: The URL to send the POST request to.
            json: The JSON payload to include in the request body.

        Returns:
            The HTTP response from the server.
        """
        return await self._client.post(url, json=json)


@logfire.instrument("ollama_web_search: {query=}")
async def ollama_web_search_single(ctx: RunContext[OllamaWebSearchDeps], query: str) -> list[OllamaSearchResult]:
    """Perform a single web search query against Ollama's web search API.

    Args:
        ctx: The run context containing the OllamaWebSearchDeps.
        query: The search query string.

    Returns:
        A list of OllamaSearchResult objects containing the search results.
    """
    deps = ctx.deps
    response = await deps.post("https://ollama.com/api/web_search", json={"query": query})
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    return [OllamaSearchResult.model_validate(result) for result in results]


async def ollama_web_search_multi(ctx: RunContext[OllamaWebSearchDeps], queries: list[str]) -> list[QueryResult]:
    """Perform multiple web search queries against Ollama's web search API.

    Args:
        ctx: The run context containing the OllamaWebSearchDeps.
        queries: A list of search query strings.

    Returns:
        A list of QueryResult objects, each containing the original query and its search results.
    """

    async def _search(query: str) -> QueryResult:
        results = await ollama_web_search_single(ctx, query)
        return QueryResult.model_validate({"query": query, "results": results})

    return await asyncio.gather(*[_search(q) for q in queries])
