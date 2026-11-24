"""Auxiliary web search provider factory and ordering utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ai_sub.config import SearchSettings
from ai_sub.search.builtin import GoogleBuiltinSearchProvider
from ai_sub.search.duckduckgo import DuckDuckGoSearchProvider
from ai_sub.search.langsearch import LangsearchSearchProvider
from ai_sub.search.lrclib import LrclibSearchProvider
from ai_sub.search.ollama import OllamaSearchProvider
from ai_sub.search.search_base import BaseSearchProvider
from ai_sub.search.tinyfish import TinyfishSearchProvider

_SEARCH_PROVIDER_REGISTRY: dict[str, Callable[[SearchSettings], BaseSearchProvider]] = {
    "builtin": GoogleBuiltinSearchProvider,
    "duckduckgo": DuckDuckGoSearchProvider,
    "langsearch": LangsearchSearchProvider,
    "lrclib": LrclibSearchProvider,
    "ollama": OllamaSearchProvider,
    "tinyfish": TinyfishSearchProvider,
}


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

    Raises:
        ValueError: If any configured tool name is unknown.
    """
    target_tools = tools if tools is not None else settings.tools
    providers: list[BaseSearchProvider] = []
    seen: set[str] = set()

    for tool in target_tools:
        clean = tool.strip().lower()
        if not clean or clean in seen:
            continue
        provider_cls = _SEARCH_PROVIDER_REGISTRY.get(clean)
        if provider_cls is None:
            supported = ", ".join(repr(k) for k in sorted(_SEARCH_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown search provider: '{tool}'. Supported providers are: {supported}.")
        seen.add(clean)
        providers.append(provider_cls(settings))

    return providers


def order_lyrics_providers(providers: Sequence[BaseSearchProvider]) -> list[BaseSearchProvider]:
    """Orders search providers for lyrics refinement so LRCLIB precedes general web search.

    Args:
        providers: Configured search providers.

    Returns:
        list[BaseSearchProvider]: Providers ordered with LRCLIB entries first.
    """
    lrclib = [p for p in providers if p.provider_name == "lrclib"]
    others = [p for p in providers if p.provider_name != "lrclib"]
    return [*lrclib, *others]


__all__ = [
    "BaseSearchProvider",
    "create_search_providers",
    "order_lyrics_providers",
]
