"""Auxiliary web search providers, shared data types, and provider factory."""

from __future__ import annotations

from ai_sub.config import SearchSettings
from ai_sub.search.base import (
    BaseHttpSearchProvider,
    BaseSearchProvider,
    WebQueryResult,
    WebSearchResult,
    normalize_query,
    parse_retry_after,
)
from ai_sub.search.builtin import GoogleBuiltinSearchProvider
from ai_sub.search.duckduckgo import DuckDuckGoSearchProvider
from ai_sub.search.langsearch import LangsearchSearchProvider
from ai_sub.search.lrclib import LrclibSearchProvider
from ai_sub.search.ollama import OllamaSearchProvider
from ai_sub.search.tinyfish import TinyfishSearchProvider


def create_search_provider(tool_name: str, settings: SearchSettings) -> BaseSearchProvider:
    """Instantiates a single search provider by name.

    Args:
        tool_name: Name of the search provider ('builtin', 'duckduckgo', 'ollama', 'langsearch', 'tinyfish', 'lrclib').
        settings: Search configuration settings.

    Returns:
        BaseSearchProvider: Configured search provider instance.

    Raises:
        ValueError: If tool_name is unknown.
    """
    clean_name = tool_name.strip().lower()
    if clean_name == "tinyfish":
        return TinyfishSearchProvider(settings)
    if clean_name == "langsearch":
        return LangsearchSearchProvider(settings)
    if clean_name == "ollama":
        return OllamaSearchProvider(settings)
    if clean_name == "duckduckgo":
        return DuckDuckGoSearchProvider(settings)
    if clean_name == "lrclib":
        return LrclibSearchProvider(settings)
    if clean_name == "builtin":
        return GoogleBuiltinSearchProvider(settings)

    raise ValueError(
        f"Unknown search provider: '{tool_name}'. "
        "Supported providers are: 'builtin', 'duckduckgo', 'ollama', 'langsearch', 'tinyfish', 'lrclib'."
    )


def create_search_providers(
    settings: SearchSettings,
    tools: list[str] | None = None,
) -> list[BaseSearchProvider]:
    """Instantiates the configured search providers in order with deduplication.

    Args:
        settings: Search configuration settings.
        tools: Optional explicit list of tool names to instantiate. Defaults to settings.tools.

    Returns:
        list[BaseSearchProvider]: Strictly-typed list of instantiated search provider instances.
    """
    target_tools = tools if tools is not None else settings.tools
    providers: list[BaseSearchProvider] = []
    seen: set[str] = set()

    for tool in target_tools:
        clean_name = tool.strip().lower()
        if not clean_name or clean_name in seen:
            continue
        seen.add(clean_name)
        providers.append(create_search_provider(clean_name, settings))

    return providers


__all__ = [
    "BaseHttpSearchProvider",
    "BaseSearchProvider",
    "DuckDuckGoSearchProvider",
    "GoogleBuiltinSearchProvider",
    "LangsearchSearchProvider",
    "LrclibSearchProvider",
    "OllamaSearchProvider",
    "TinyfishSearchProvider",
    "WebQueryResult",
    "WebSearchResult",
    "create_search_provider",
    "create_search_providers",
    "normalize_query",
    "parse_retry_after",
]
