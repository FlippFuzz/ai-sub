"""Compact shortcode generation and model tag formatting."""

from __future__ import annotations

import re

from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION


def generate_model_shortcode(model_name: str) -> str:
    """Generates an ultra-compact filename shortcode for a model identifier.

    Transforms Gemini models into 'g<version><tier>[variant]' (e.g., 'g38f', 'g35l').
    Non-Gemini models are sanitized to safe kebab-case strings.

    Args:
        model_name: Model identifier (e.g., 'google-gla:gemini-3.8-flash').

    Returns:
        str: Sanitized shortcode string.
    """
    clean = model_name.split(":", 1)[-1].lower()
    if not clean.startswith("gemini"):
        return re.sub(r"[^a-zA-Z0-9]+", "-", clean.replace(".", "")).strip("-") or "model"

    rem = clean[6:].lstrip("-_")
    v_match = re.search(r"(\d+(?:\.\d+)*)", rem)
    v_code = v_match.group(1).replace(".", "") if v_match else ""
    rem = rem[v_match.end() :].lstrip("-_") if v_match else rem

    tier = "g"
    for name, code in (("flash-lite", "l"), ("lite", "l"), ("flash", "f"), ("pro", "p")):
        if rem.startswith(name):
            tier = code
            rem = rem[len(name) :].lstrip("-_")
            break

    variant = {"preview": "p"}.get(rem, re.sub(r"[^a-zA-Z0-9]+", "-", rem.replace(".", "")).strip("-"))
    base = f"g{v_code}{tier}"
    return f"{base}{variant}" if variant == "p" else (f"{base}-{variant}" if variant else base)


def generate_summary_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
) -> str:
    """Combines model shortcode and 2-digit summary prompt version tag.

    Args:
        model_name: Model identifier string.
        summary_prompt_version: Prompt version number. Defaults to SUMMARY_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<summary_v>' (e.g., 'g35l-01').
    """
    return f"{generate_model_shortcode(model_name)}-{summary_prompt_version:02d}"


def generate_subtitles_shortcode(
    model_name: str,
    summary_shortcode: str | None = None,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode and 2-digit subtitles prompt version tag with optional summary lineage.

    Args:
        model_name: Model identifier string.
        summary_shortcode: Optional upstream summary tag. Defaults to None.
        subtitles_prompt_version: Prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '[<summary_code>_]<model_code>-<subtitles_v>'.
    """
    sub_tag = f"{generate_model_shortcode(model_name)}-{subtitles_prompt_version:02d}"
    return f"{summary_shortcode}_{sub_tag}" if summary_shortcode else sub_tag


def generate_lyrics_shortcode(
    lyrics_model_name: str,
    subtitles_shortcode: str,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
) -> str:
    """Combines upstream subtitles shortcode and lyrics model tag with prompt version.

    Args:
        lyrics_model_name: Lyrics model identifier string.
        subtitles_shortcode: Full upstream subtitles shortcode representing transcription lineage.
        lyrics_prompt_version: Lyrics prompt version number. Defaults to LYRICS_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<subtitles_shortcode>_<lyrics_code>-<lyrics_v>'.
    """
    return f"{subtitles_shortcode}_{generate_model_shortcode(lyrics_model_name)}-{lyrics_prompt_version:02d}"


def generate_full_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode with composite 6-digit prompt version suffix for final output.

    Args:
        model_name: Model identifier string.
        summary_prompt_version: Summary prompt version number. Defaults to SUMMARY_PROMPT_VERSION.
        lyrics_prompt_version: Lyrics prompt version number. Defaults to LYRICS_PROMPT_VERSION.
        subtitles_prompt_version: Subtitle prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<summary_v><lyrics_v><sub_v>' (e.g., 'g38f-011019').
    """
    code = generate_model_shortcode(model_name)
    return f"{code}-{summary_prompt_version:02d}{lyrics_prompt_version:02d}{subtitles_prompt_version:02d}"
