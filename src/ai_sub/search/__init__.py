"""Auxiliary web search providers, shared data types, and provider factory."""

from __future__ import annotations

from typing import Any

from google.genai import types

from ai_sub.config import SearchSettings
from ai_sub.search.base import WebQueryResult, WebSearchResult, normalize_query
from ai_sub.search.duckduckgo import DuckDuckGoSearchProvider
from ai_sub.search.langsearch import LangsearchSearchProvider
from ai_sub.search.ollama import OllamaSearchProvider
from ai_sub.search.tinyfish import TinyfishSearchProvider


def create_search_provider(settings: SearchSettings) -> tuple[Any | None, list[Any]]:
    """Instantiates the configured search provider and extracts its Gemini tool callables.

    Args:
        settings: Search configuration settings.

    Returns:
        tuple[Any | None, list[Any]]: A tuple of (provider_instance, tools_list).
            provider_instance is an async context manager if it manages an HTTP session, or None.
            tools_list is the list of callables or types.Tool objects ready for Gemini registration.
    """
    tool_type = settings.tool

    if tool_type == "tinyfish":
        provider = TinyfishSearchProvider(settings)
        return provider, provider.get_tools()

    if tool_type == "langsearch":
        provider = LangsearchSearchProvider(settings)
        return provider, provider.get_tools()

    if tool_type == "ollama":
        provider = OllamaSearchProvider(settings)
        return provider, provider.get_tools()

    if tool_type == "duckduckgo":
        provider = DuckDuckGoSearchProvider(settings)
        return provider, provider.get_tools()

    if tool_type == "builtin":
        return None, [types.Tool(google_search=types.GoogleSearch())]

    return None, []


__all__ = [
    "DuckDuckGoSearchProvider",
    "LangsearchSearchProvider",
    "OllamaSearchProvider",
    "TinyfishSearchProvider",
    "WebQueryResult",
    "WebSearchResult",
    "create_search_provider",
    "normalize_query",
]
