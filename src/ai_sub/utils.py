"""Utility functions for shortcode generation and model string formatting."""

import re

from ai_sub.prompt import LYRICS_PROMPT_VERSION, SUBTITLES_PROMPT_VERSION

FAMILY_MAP = {
    "gemini": "g",
    "claude": "c",
    "gpt": "gpt",
    "whisper": "wh",
    "deepseek": "ds",
    "llama": "lm",
}

TIER_MAP = {
    "flash": "f",
    "lite": "l",
    "pro": "p",
    "sonnet": "s",
    "opus": "o",
    "haiku": "h",
    "mini": "m",
    "nano": "n",
    "ultra": "u",
    "large": "lg",
}


def generate_model_shortcode(model_name: str) -> str:
    """Generates an ultra-compact, deterministic model shortcode.

    Strips provider prefixes, extracts the model family, version digits, and tier
    descriptors, and combines them into a concise model code.

    Args:
        model_name: Full model identifier string (e.g., "google-gla:gemini-3.5-flash-lite").

    Returns:
        A compact model shortcode string in the format `<family_code><version_code><tier_code>`
        (e.g., "g35fl").

    Generation Rules:
        1. Provider Prefix Stripping:
           Strips provider prefixes separated by a colon if present
           (e.g., "google-gla:gemini-3.5-flash-lite" -> "gemini-3.5-flash-lite").

        2. Family Prefix (1-3 chars):
           - "gemini"   -> "g"
           - "claude"   -> "c"
           - "gpt"      -> "gpt"
           - "whisper"  -> "wh"
           - "deepseek" -> "ds"
           - "llama"    -> "lm"
           - Fallback: First 3 letters of the family token in lowercase.

        3. Version Number Extraction:
           - Extracts version digits and strips decimal points.
           - Examples: "3.5" -> "35", "1.5" -> "15", "4" -> "4", "3.6" -> "36".

        4. Tier Codes:
           Maps individual descriptor terms to single or double letters:
           - "flash" -> "f", "lite" -> "l", "pro" -> "p", "sonnet" -> "s"
           - "opus"  -> "o", "haiku" -> "h", "mini" -> "m", "nano" -> "n"
           - "ultra" -> "u", "large" -> "lg"
           - Concatenates compound tiers in order (e.g., "flash-lite" -> "fl").
           - Fallback: First letter of the descriptor term in lowercase.

    Examples:
        >>> generate_model_shortcode("google-gla:gemini-3.5-flash-lite")
        'g35fl'

        >>> generate_model_shortcode("gemini-3.6-flash")
        'g36f'

        >>> generate_model_shortcode("claude-3.5-sonnet")
        'c35s'
    """
    clean_name = model_name.split(":", 1)[-1]
    tokens = [t for t in re.split(r"[^a-zA-Z0-9.]+", clean_name) if t]

    if not tokens:
        return "mod"

    family_token = tokens[0].lower()
    family_code = FAMILY_MAP.get(family_token, family_token[:3])

    version_code = ""
    tier_terms: list[str] = []

    for token in tokens[1:]:
        match = re.match(r"^v?(\d+(?:\.\d+)*)(.*)$", token, re.IGNORECASE)
        if match and match.group(1):
            version_code += match.group(1).replace(".", "")
            remainder = match.group(2)
            if remainder:
                tier_terms.append(remainder)
        else:
            tier_terms.append(token)

    tier_code_parts: list[str] = []
    for term in tier_terms:
        term_lower = term.lower()
        if term_lower in TIER_MAP:
            tier_code_parts.append(TIER_MAP[term_lower])
        elif term:
            tier_code_parts.append(term[0].lower())

    tier_code = "".join(tier_code_parts)
    return f"{family_code}{version_code}{tier_code}"


def generate_full_shortcode(
    model_name: str,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Generates an ultra-compact tag combining model shortcode and prompt version numbers.

    Calls `generate_model_shortcode` to format the base model identifier, then appends
    a hyphenated 4-digit suffix containing the zero-padded lyrics prompt version and
    subtitles prompt version.

    Args:
        model_name: Full model identifier string (e.g., "google-gla:gemini-3.5-flash-lite").
        lyrics_prompt_version: Version integer for the lyrics prompt (defaults to current prompt version).
        subtitles_prompt_version: Version integer for the subtitle prompt (defaults to current prompt version).

    Returns:
        A compact shortcode string in the format `<model_code>-<lyrics_v><sub_v>`
        (e.g., "g35fl-0918").

    Generation Rules:
        1. Model Code:
           Derived via `generate_model_shortcode(model_name)` (e.g., "g35fl").

        2. Prompt Version Suffix:
           Directly concatenates the zero-padded 2-digit lyrics prompt version and
           subtitles prompt version, separated from the model code by a hyphen
           (e.g., lyrics version 9 and subtitles version 18 -> "-0918").

    Examples:
        >>> generate_full_shortcode("google-gla:gemini-3.5-flash-lite", 9, 18)
        'g35fl-0918'

        >>> generate_full_shortcode("gemini-3.6-flash", 8, 14)
        'g36f-0814'

        >>> generate_full_shortcode("claude-3.5-sonnet", 1, 3)
        'c35s-0103'
    """
    model_code = generate_model_shortcode(model_name)
    suffix = f"-{lyrics_prompt_version:02d}{subtitles_prompt_version:02d}"
    return f"{model_code}{suffix}"
