"""Gemini built-in Google Search grounding provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseSearchProvider


class GoogleBuiltinSearchProvider(BaseSearchProvider):
    """Encapsulates Gemini native Google Search grounding."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the built-in search provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="builtin", enable_rate_limit=False)

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns the Interactions API declaration for Gemini native Google Search grounding.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: The Interactions API Google Search tool specification.
        """
        return [{"type": "google_search"}]
