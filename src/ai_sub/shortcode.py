"""Compact shortcode generation, model tag formatting, and subtitle track ranking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION


def generate_model_shortcode(model_name: str) -> str:
    """Generates an ultra-compact filename shortcode for a model identifier.

    Transforms Gemini models into 'g<version><tier>[variant]' (e.g., 'g38f', 'g37f', 'g35l').
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


@dataclass(frozen=True)
class SubtitleTrackInfo:
    """Structured metadata and ranking indicators parsed from a subtitle shortcode or file path."""

    model: str
    display_label: str
    shortcode: str
    is_refined: bool
    tier_rank: int
    version_tuple: tuple[int, ...]
    summary_version: int
    lyrics_version: int
    subtitles_version: int

    @property
    def sort_key(self) -> tuple[int, int, tuple[int, ...], int, int, int]:
        """Sort key tuple for ordering tracks from best to worst.

        Hierarchy:
        1. is_refined (1 for refined/primary, 0 for raw .int/)
        2. tier_rank (Pro=3 > Flash=2 > Flash-Lite=1 > Other=0)
        3. version_tuple (e.g., (3, 10) > (3, 9) > (3, 8) > (3, 7))
        4. subtitles_version (higher prompt version)
        5. lyrics_version (higher prompt version)
        6. summary_version (higher prompt version)

        Returns:
            tuple[int, int, tuple[int, ...], int, int, int]: Comparable sort key.
        """
        return (
            1 if self.is_refined else 0,
            self.tier_rank,
            self.version_tuple,
            self.subtitles_version,
            self.lyrics_version,
            self.summary_version,
        )


def parse_subtitle_track(path_or_name: str | Path) -> SubtitleTrackInfo:
    """Parses model, prompt lineage, refinement status, and ranking metadata from a subtitle filename or path.

    Handles standard full shortcodes (e.g., 'g38f-011019'), legacy shortcodes without summary stages
    (e.g., 'g35f-0918'), raw dialogue-only tracks (e.g., 'raw-g38f-011019' or '.int/...'), and multi-stage
    lineage strings by taking the final model stage.

    Args:
        path_or_name: Subtitle file path, filename, or shortcode string.

    Returns:
        SubtitleTrackInfo: Structured track metadata and sorting attributes.
    """
    raw_str = str(path_or_name).replace("\\", "/")
    is_refined = not (
        ".raw-" in raw_str
        or "/raw-" in raw_str
        or raw_str.startswith("raw-")
        or "/.int/" in raw_str
        or raw_str.startswith(".int/")
    )

    leaf = raw_str.split("/")[-1]
    if leaf.endswith(".srt"):
        leaf = leaf[:-4]

    # Extract shortcode suffix from filename (e.g. 'Title [id].g38f-011019' or 'Title.raw-g38f-011019')
    if ".raw-" in leaf:
        leaf = leaf.split(".raw-")[-1]
    elif leaf.startswith("raw-"):
        leaf = leaf[4:]
    elif "." in leaf:
        leaf = leaf.split(".")[-1]

    # If an intermediate lineage shortcode is encountered (e.g. 'g35l-01_g38f-19'), evaluate the final stage
    if "_" in leaf:
        leaf = leaf.split("_")[-1]

    shortcode = leaf.strip()

    # Split into model shortcode and prompt version suffix
    parts = shortcode.split("-")
    prompt_digits = ""
    if len(parts) > 1 and parts[-1].isdigit():
        prompt_digits = parts[-1]
        model_code = "-".join(parts[:-1])
    else:
        model_code = shortcode

    # Decompose prompt version suffix
    summary_v = 0
    lyrics_v = 0
    subtitles_v = 0
    if len(prompt_digits) >= 6:
        summary_v = int(prompt_digits[-6:-4])
        lyrics_v = int(prompt_digits[-4:-2])
        subtitles_v = int(prompt_digits[-2:])
    elif len(prompt_digits) == 4:
        lyrics_v = int(prompt_digits[-4:-2])
        subtitles_v = int(prompt_digits[-2:])
    elif len(prompt_digits) == 2:
        subtitles_v = int(prompt_digits[-2:])

    # Decompose model code (e.g. 'g38f', 'g37f', 'g36f', 'g35l', 'g310f', 'g30p', 'g3p')
    tier_map = {"p": ("pro", 3), "f": ("flash", 2), "l": ("flash-lite", 1)}
    match = re.match(r"^g(\d+)([lfp])(.*)$", model_code.lower())

    if match:
        v_code = match.group(1)
        tier_char = match.group(2)
        variant = match.group(3).lstrip("-_")

        if len(v_code) == 1:
            v_str = v_code
            v_tuple = (int(v_code), 0)
        elif len(v_code) == 2:
            v_str = f"{v_code[0]}.{v_code[1]}"
            v_tuple = (int(v_code[0]), int(v_code[1]))
        else:
            # Handles multi-digit minor versions (e.g. '310' -> 3.10)
            v_str = f"{v_code[0]}.{v_code[1:]}"
            v_tuple = (int(v_code[0]), int(v_code[1:]))

        tier_name, tier_rank = tier_map[tier_char]
        model_name = f"gemini-{v_str}-{tier_name}"
        if variant:
            model_name = f"{model_name}-{variant}"

        tier_label = "lite" if tier_char == "l" else tier_name
        base_label = f"{v_str}-{tier_label}"
        display_label = base_label if is_refined else f"{base_label} (Draft)"
    else:
        model_name = model_code
        tier_rank = 0
        v_tuple = (0, 0)
        display_label = model_code if is_refined else f"{model_code} (Draft)"

    return SubtitleTrackInfo(
        model=model_name,
        display_label=display_label,
        shortcode=shortcode,
        is_refined=is_refined,
        tier_rank=tier_rank,
        version_tuple=v_tuple,
        summary_version=summary_v,
        lyrics_version=lyrics_v,
        subtitles_version=subtitles_v,
    )


def subtitle_track_sort_key(
    path_or_name: str | Path,
) -> tuple[int, int, tuple[int, ...], int, int, int]:
    """Generates a comparable sort tuple for ordering subtitle files from best to worst.

    Args:
        path_or_name: Subtitle file path, filename, or shortcode string.

    Returns:
        tuple[int, int, tuple[int, ...], int, int, int]: Sort tuple
        (is_refined, tier, version_tuple, sub_v, lyrics_v, summary_v).
    """
    return parse_subtitle_track(path_or_name).sort_key
