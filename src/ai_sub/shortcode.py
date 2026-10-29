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
    clean_name = model_name.split(":", 1)[-1].lower()

    if clean_name.startswith("gemini"):
        rem = clean_name[len("gemini") :].lstrip("-_")

        version_match = re.search(r"(\d+(?:\.\d+)*)", rem)
        if version_match:
            version_str = version_match.group(1)
            version_code = version_str.replace(".", "")
            rem = rem.replace(version_str, "", 1).lstrip("-_")
        else:
            version_code = ""

        tier_code = "g"
        for tier_name, code in [
            ("flash-lite", "l"),
            ("lite", "l"),
            ("flash", "f"),
            ("pro", "p"),
        ]:
            if rem == tier_name or rem.startswith(tier_name + "-") or rem.startswith(tier_name + "_"):
                tier_code = code
                rem = rem[len(tier_name) :].lstrip("-_")
                break

        base_code = f"g{version_code}{tier_code}"
        if rem:
            variant_map = {"preview": "p"}
            if rem in variant_map:
                return f"{base_code}{variant_map[rem]}"

            variant = rem.replace(".", "")
            variant = re.sub(r"[^a-zA-Z0-9]+", "-", variant).strip("-")
            if variant:
                return f"{base_code}-{variant}"

        return base_code

    sanitized = clean_name.replace(".", "")
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", sanitized).strip("-")
    return sanitized or "model"


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
    model_code = generate_model_shortcode(model_name)
    return f"{model_code}-{summary_prompt_version:02d}"


def generate_subtitles_shortcode(
    model_name: str,
    summary_shortcode: str | None = None,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode and 2-digit subtitles prompt version tag with optional summary lineage.

    If summary_shortcode is provided, prepends it so changes to the upstream summary model
    automatically invalidate downstream subtitle caches.

    Args:
        model_name: Model identifier string.
        summary_shortcode: Optional upstream summary tag (e.g. 'g35l-01').
        subtitles_prompt_version: Prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '[<summary_code>_]<model_code>-<subtitles_v>'
            (e.g., 'g35l-01_g38f-19' or 'g38f-19').
    """
    model_code = generate_model_shortcode(model_name)
    sub_tag = f"{model_code}-{subtitles_prompt_version:02d}"
    if summary_shortcode:
        return f"{summary_shortcode}_{sub_tag}"
    return sub_tag


def generate_lyrics_shortcode(
    lyrics_model_name: str,
    subtitles_shortcode: str,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
) -> str:
    """Combines upstream subtitles shortcode and lyrics model tag with prompt version.

    Args:
        lyrics_model_name: Lyrics model identifier string.
        subtitles_shortcode: Full upstream subtitles shortcode representing the transcription lineage.
        lyrics_prompt_version: Lyrics prompt version number. Defaults to LYRICS_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<subtitles_shortcode>_<lyrics_code>-<lyrics_v>'
            (e.g., 'g35l-01_g38f-19_g35l-10' or 'g38f-19_g35l-10').
    """
    lyrics_code = generate_model_shortcode(lyrics_model_name)
    return f"{subtitles_shortcode}_{lyrics_code}-{lyrics_prompt_version:02d}"


def generate_full_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode with a composite 6-digit prompt version suffix for final output.

    Args:
        model_name: Model identifier string.
        summary_prompt_version: Summary prompt version number. Defaults to SUMMARY_PROMPT_VERSION.
        lyrics_prompt_version: Lyrics prompt version number. Defaults to LYRICS_PROMPT_VERSION.
        subtitles_prompt_version: Subtitle prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<summary_v><lyrics_v><sub_v>' (e.g., 'g38f-011019').
    """
    model_code = generate_model_shortcode(model_name)
    suffix = f"-{summary_prompt_version:02d}{lyrics_prompt_version:02d}{subtitles_prompt_version:02d}"
    return f"{model_code}{suffix}"
