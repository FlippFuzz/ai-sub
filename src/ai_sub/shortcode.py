"""Compact shortcode generation and model tag formatting."""

from __future__ import annotations

import re

from ai_sub.jobs.music_refine import MUSIC_REFINE_PROMPT_VERSION
from ai_sub.jobs.subtitle import SUBTITLES_PROMPT_VERSION
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


def generate_music_refine_shortcode(
    model_name: str,
    music_refine_prompt_version: int = MUSIC_REFINE_PROMPT_VERSION,
) -> str:
    """Combines model shortcode and 2-digit music refinement prompt version tag.

    Args:
        model_name: Model identifier string.
        music_refine_prompt_version: Prompt version number. Defaults to MUSIC_REFINE_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<refine_v>' (e.g., 'g35l-10').
    """
    model_code = generate_model_shortcode(model_name)
    return f"{model_code}-{music_refine_prompt_version:02d}"


def generate_full_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
    music_refine_prompt_version: int = MUSIC_REFINE_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode with a 6-digit prompt version suffix.

    Args:
        model_name: Model identifier string.
        summary_prompt_version: Summary prompt version number. Defaults to SUMMARY_PROMPT_VERSION.
        music_refine_prompt_version: Refinement prompt version number. Defaults to MUSIC_REFINE_PROMPT_VERSION.
        subtitles_prompt_version: Subtitle prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<summary_v><refine_v><sub_v>' (e.g., 'g38f-011019').
    """
    model_code = generate_model_shortcode(model_name)
    suffix = f"-{summary_prompt_version:02d}{music_refine_prompt_version:02d}{subtitles_prompt_version:02d}"
    return f"{model_code}{suffix}"
