"""Compact shortcode generation, model tag formatting, and subtitle track ranking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION

# Single ranking & shortcode definition (ordered highest -> lowest).
# Format: {shortcode_char: tier_name}. Unlisted tiers ("Others") receive rank 0.
_TIERS: dict[str, str] = {
    "c": "carbon",
    "a": "argon",
    "h": "hydrogen",
    "p": "pro",
    "f": "flash",
    "l": "flash-lite",
}

_TIER_RANKS: dict[str, int] = {name: len(_TIERS) - idx for idx, name in enumerate(_TIERS.values())}

# Longest prefix first so e.g. "flash-lite" matches before "flash" regardless of rank order.
_TIER_ENCODE_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((name, code) for code, name in _TIERS.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def generate_model_shortcode(model_name: str) -> str:
    """Generates an ultra-compact filename shortcode for a model identifier.

    Transforms Gemini models into 'g<version><tier>[variant]' (e.g., 'g4c', 'g4a', 'g4h', 'g38f', 'g35l'),
    or 'g<version>-<name>' for unrecognized tiers (e.g., 'g4-neon').
    Non-Gemini models are sanitized to safe kebab-case strings.

    Args:
        model_name: Model identifier (e.g., 'google-gla:gemini-4-hydrogen', 'google-gla:gemini-3.8-flash').

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

    tier = ""
    for name, code in _TIER_ENCODE_PREFIXES:
        if rem.startswith(name):
            tier = code
            rem = rem[len(name) :].lstrip("-_")
            break

    variant = re.sub(r"[^a-zA-Z0-9]+", "-", rem.replace(".", "")).strip("-")
    base = f"g{v_code}{tier}"
    return f"{base}-{variant}" if variant else base


def generate_summary_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
) -> str:
    """Combines model shortcode and 2-digit summary prompt version tag.

    Args:
        model_name: Model identifier string.
        summary_prompt_version: Prompt version number. Defaults to SUMMARY_PROMPT_VERSION.

    Returns:
        str: Tag formatted as '<model_code>-<summary_v>' (e.g., 'g4a-01', 'g35l-01').
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
        str: Tag formatted as '<model_code>-<summary_v><lyrics_v><sub_v>' (e.g., 'g4c-011019', 'g4h-011019').
    """
    code = generate_model_shortcode(model_name)
    return f"{code}-{summary_prompt_version:02d}{lyrics_prompt_version:02d}{subtitles_prompt_version:02d}"


@dataclass(frozen=True)
class SubtitleTrackInfo:
    """Structured metadata and ranking indicators parsed from a subtitle shortcode or file path."""

    model: str
    track_name: str
    display_label: str
    prompt_version_str: str
    shortcode: str
    is_refined: bool
    tier_rank: int
    version_tuple: tuple[int, ...]
    summary_version: int
    lyrics_version: int
    subtitles_version: int

    @property
    def sort_key(self) -> tuple[int, tuple[int, ...], int, str, int, int, int]:
        """Sort key tuple for ordering tracks from best to worst.

        Hierarchy:
        1. is_refined (1 for refined/primary, 0 for raw .int/)
        2. version_tuple (e.g., (4, 0) > (3, 10) > (3, 8) > (3, 5))
        3. tier_rank (derived from _TIERS; 0 for Others)
        4. model (alphabetical text tie-breaker)
        5. subtitles_version (higher prompt version)
        6. lyrics_version (higher prompt version)
        7. summary_version (higher prompt version)

        Returns:
            tuple[int, tuple[int, ...], int, str, int, int, int]: Comparable sort key.
        """
        return (
            1 if self.is_refined else 0,
            self.version_tuple,
            self.tier_rank,
            self.model,
            self.subtitles_version,
            self.lyrics_version,
            self.summary_version,
        )


def _parse_version_code(v_code: str) -> str:
    """Converts a compact numeric version code into a dotted version string.

    Args:
        v_code: Compact numeric version string (e.g., '4', '38', '310').

    Returns:
        str: Dotted version string (e.g., '4', '3.8', '3.10').
    """
    if len(v_code) == 1:
        return v_code
    if len(v_code) == 2:
        return f"{v_code[0]}.{v_code[1]}"
    return f"{v_code[0]}.{v_code[1:]}"


def _decode_model_code(model_code: str) -> str:
    """Expands a compact model shortcode back into its canonical model name.

    Args:
        model_code: Compact model code (e.g., 'g4h', 'g38f', 'g4-neon').

    Returns:
        str: Reconstructed model name (e.g., 'gemini-4-hydrogen', 'gemini-3.8-flash').
    """
    match = re.match(r"^g(\d+)(.*)$", model_code.lower())
    if not match:
        return model_code

    v_str = _parse_version_code(match.group(1))
    rest = match.group(2)

    if rest and rest[0] in _TIERS:
        tier_name = _TIERS[rest[0]]
        variant = rest[1:].lstrip("-_")
        return f"gemini-{v_str}-{tier_name}-{variant}" if variant else f"gemini-{v_str}-{tier_name}"

    variant = rest.lstrip("-_")
    return f"gemini-{v_str}-{variant}" if variant else f"gemini-{v_str}"


def _rank_model(model_name: str) -> tuple[tuple[int, ...], int]:
    """Extracts version tuple and tier rank by splitting 'gemini-<version>-<text>'.

    Args:
        model_name: Canonical model name (e.g., 'gemini-4-hydrogen', 'gemini-3.8-flash').

    Returns:
        tuple[tuple[int, ...], int]: (version_tuple, tier_rank) where tier_rank is 0 for unlisted tiers.
    """
    match = re.match(r"^gemini-(\d+(?:\.\d+)*)(?:-(.*))?$", model_name.lower())
    if not match:
        return (0, 0), 0

    parts = [int(p) for p in match.group(1).split(".")]
    v_tuple = (parts[0], 0) if len(parts) == 1 else tuple(parts)
    text = match.group(2) or ""

    for prefix, _ in _TIER_ENCODE_PREFIXES:
        if text == prefix or text.startswith(f"{prefix}-"):
            return v_tuple, _TIER_RANKS[prefix]

    return v_tuple, 0


def parse_subtitle_track(path_or_name: str | Path) -> SubtitleTrackInfo:
    """Parses model, prompt lineage, refinement status, and ranking metadata from a subtitle filename or path.

    Handles standard full shortcodes (e.g., 'g4c-011019', 'g4h-011019', 'g38f-011019'), legacy shortcodes
    without summary stages (e.g., 'g35f-0918'), undefined tiers (e.g., 'g4-neon-011019'), raw dialogue-only
    tracks (e.g., 'raw-g4c-011019' or '.int/...'), and multi-stage lineage strings by taking the final stage.

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

    if ".raw-" in leaf:
        leaf = leaf.split(".raw-")[-1]
    elif leaf.startswith("raw-"):
        leaf = leaf[4:]
    elif "." in leaf:
        leaf = leaf.split(".")[-1]

    if "_" in leaf:
        leaf = leaf.split("_")[-1]

    shortcode = leaf.strip()

    parts = shortcode.split("-")
    prompt_digits = ""
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) in (2, 4, 6):
        prompt_digits = parts[-1]
        model_code = "-".join(parts[:-1])
    else:
        model_code = shortcode

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

    model_name = _decode_model_code(model_code)
    v_tuple, tier_rank = _rank_model(model_name)

    base_track_name = f"{model_name} ({prompt_digits})" if prompt_digits else model_name
    track_name = base_track_name if is_refined else f"{base_track_name} (Draft)"
    display_label = model_name if is_refined else f"{model_name} (Draft)"

    return SubtitleTrackInfo(
        model=model_name,
        track_name=track_name,
        display_label=display_label,
        prompt_version_str=prompt_digits,
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
) -> tuple[int, tuple[int, ...], int, str, int, int, int]:
    """Generates a comparable sort tuple for ordering subtitle files from best to worst.

    Args:
        path_or_name: Subtitle file path, filename, or shortcode string.

    Returns:
        tuple[int, tuple[int, ...], int, str, int, int, int]: Sort tuple
        (is_refined, version_tuple, tier_rank, model, sub_v, lyrics_v, summary_v).
    """
    return parse_subtitle_track(path_or_name).sort_key
