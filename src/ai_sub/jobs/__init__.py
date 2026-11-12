"""AI Sub: AI-Powered Subtitle Generation with Translation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_sub.pipeline import Pipeline

__all__ = ["Pipeline"]


def __getattr__(name: str) -> Any:
    """Lazy import for public attributes to prevent importing heavy dependencies at module init.

    Args:
        name: Name of the attribute to look up.

    Returns:
        Any: The resolved module attribute.

    Raises:
        AttributeError: If attribute name is not recognized.
    """
    if name == "Pipeline":
        from ai_sub.pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
