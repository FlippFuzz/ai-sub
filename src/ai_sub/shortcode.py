"""Shortcode generation and model string formatting for AI Sub."""

import re

from ai_sub.prompt import LYRICS_PROMPT_VERSION, SUBTITLES_PROMPT_VERSION


def generate_model_shortcode(model_name: str) -> str:
    """Generates a model shortcode for filename usage.

    Shortcodes are currently formatted specifically for Google Gemini models, as this project
    is primarily tested and designed to work with Gemini models. For Gemini models, an
    ultra-compact tag in the format `g<version><tier>[<variant>]` is generated. Known variant
    suffixes (such as 'preview') map to single letters, while unhandled suffixes are sanitized
    and appended with a hyphen (e.g., 'g25f-exp').

    For all other (non-Gemini) models, the model name is sanitized to be filename-safe
    by stripping provider prefixes, removing dots, and replacing invalid characters with hyphens.

    Args:
        model_name: Full model identifier string (e.g., "google-gla:gemini-3.5-flash-lite"
            or "openai:gpt-4.0").

    Returns:
        A shortcode string for Gemini models (e.g., "g35l", "g36f", "g31p", "g25fp", or "g25f-exp"),
        or a filename-sanitized string for non-Gemini models (e.g., "gpt-40", "claude-35-sonnet").

    Gemini Shortcode Rules:
        1. Provider Prefix Stripping:
           Strips provider prefixes separated by a colon if present
           (e.g., "google-gla:gemini-3.5-flash-lite" -> "gemini-3.5-flash-lite").

        2. Family Code:
           Always "g" for Gemini models.

        3. Version Digits:
           Extracts numeric version digits and strips decimal points
           (e.g., "3.5" -> "35", "3.6" -> "36", "3.1" -> "31", "2.5" -> "25").

        4. Tier Code:
           Maps Gemini tier descriptors to single letters:
           - "flash-lite" -> "l"
           - "flash"      -> "f"
           - "pro"        -> "p"

        5. Variant Suffixes:
           Known variant suffixes map to single letters (e.g., "preview" -> "p", so
           "gemini-2.5-flash-preview" -> "g25fp"). Unhandled variant suffixes fallback to
           being sanitized and appended with a hyphen (e.g., "gemini-2.5-flash-exp" -> "g25f-exp").

    Examples:
        >>> generate_model_shortcode("gemini-3.5-flash-lite")
        'g35l'

        >>> generate_model_shortcode("google-gla:gemini-3.6-flash")
        'g36f'

        >>> generate_model_shortcode("google-gla:gemini-3.1-pro")
        'g31p'

        >>> generate_model_shortcode("gemini-2.5-flash-preview")
        'g25fp'

        >>> generate_model_shortcode("gemini-2.5-flash-exp")
        'g25f-exp'

        >>> generate_model_shortcode("claude-3.5-sonnet")
        'claude-35-sonnet'
    """
    clean_name = model_name.split(":", 1)[-1].lower()

    # Handle Gemini models specifically
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
            variant_map = {
                "preview": "p",
            }
            if rem in variant_map:
                return f"{base_code}{variant_map[rem]}"

            variant = rem.replace(".", "")
            variant = re.sub(r"[^a-zA-Z0-9]+", "-", variant).strip("-")
            if variant:
                return f"{base_code}-{variant}"
        return base_code

    # For non-Gemini models: sanitize to be filename-safe and drop dots
    sanitized = clean_name.replace(".", "")
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", sanitized).strip("-")
    return sanitized or "model"


def generate_lyrics_shortcode(
    model_name: str,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
) -> str:
    """Generates a shortcode tag combining the model shortcode and lyrics prompt version.

    Calls `generate_model_shortcode` to format the base model identifier, then appends
    a hyphenated 2-digit version suffix (`-<lyrics_v>`).

    Args:
        model_name: Full model identifier string (e.g., "google-gla:gemini-3.5-flash-lite").
        lyrics_prompt_version: Version integer for the lyrics prompt (defaults to current prompt version).

    Returns:
        A lyrics shortcode string in the format `<model_code>-<lyrics_v>`
        (e.g., "g35l-09" for Gemini, or "claude-35-sonnet-01" for non-Gemini models).

    Examples:
        >>> generate_lyrics_shortcode("google-gla:gemini-3.5-flash-lite", 9)
        'g35l-09'

        >>> generate_lyrics_shortcode("google-gla:gemini-3.6-flash", 8)
        'g36f-08'

        >>> generate_lyrics_shortcode("claude-3.5-sonnet", 1)
        'claude-35-sonnet-01'
    """
    model_code = generate_model_shortcode(model_name)
    return f"{model_code}-{lyrics_prompt_version:02d}"


def generate_full_shortcode(
    model_name: str,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Generates a full shortcode tag combining the model shortcode and prompt version numbers.

    Calls `generate_model_shortcode` to format the base model identifier, then appends
    a hyphenated 4-digit suffix containing the zero-padded lyrics prompt version and
    subtitles prompt version (`-<lyrics_v><sub_v>`).

    Args:
        model_name: Full model identifier string (e.g., "google-gla:gemini-3.5-flash-lite").
        lyrics_prompt_version: Version integer for the lyrics prompt (defaults to current prompt version).
        subtitles_prompt_version: Version integer for the subtitle prompt (defaults to current prompt version).

    Returns:
        A full shortcode string in the format `<model_code>-<lyrics_v><sub_v>`
        (e.g., "g35l-0918" for Gemini, or "claude-35-sonnet-0918" for non-Gemini models).

    Examples:
        >>> generate_full_shortcode("google-gla:gemini-3.5-flash-lite", 9, 18)
        'g35l-0918'

        >>> generate_full_shortcode("google-gla:gemini-3.6-flash", 8, 14)
        'g36f-0814'

        >>> generate_full_shortcode("claude-3.5-sonnet", 1, 3)
        'claude-35-sonnet-0103'
    """
    model_code = generate_model_shortcode(model_name)
    suffix = f"-{lyrics_prompt_version:02d}{subtitles_prompt_version:02d}"
    return f"{model_code}{suffix}"
