"""Gemini built-in Google Search grounding provider."""

from __future__ import annotations

from typing import Any, Callable

from ai_sub.config import SearchSettings
from ai_sub.search.base import BaseSearchProvider, WebSearchResult


class GoogleBuiltinSearchProvider(BaseSearchProvider):
    """Encapsulates Gemini native Google Search grounding."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the built-in search provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="builtin")

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns the Interactions API declaration for Gemini native Google Search grounding.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: The Interactions API Google Search tool specification.
        """
        return [{"type": "google_search"}]

    async def search_single(self, query: str) -> list[WebSearchResult]:
        """Built-in Google search is an LLM grounding tool and does not support standalone Python pre-search.

        Args:
            query: Search query keywords.

        Returns:
            list[WebSearchResult]: Always returns an empty list.
        """
        return []
