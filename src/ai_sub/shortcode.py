"""Compact shortcode generation, model tag formatting, and subtitle track ranking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION


@dataclass(frozen=True)
class _Tier:
    """Gemini model tier definition with accepted shortcodes (first is canonical)."""

    name: str
    codes: list[str]


# Ordered lowest -> highest priority; list position determines tier rank.
_TIERS: list[_Tier] = [
    _Tier("flash-lite", ["l"]),
    _Tier("flash", ["f"]),
    _Tier("pro", ["p"]),
    _Tier("hydrogen", ["h", "hy"]),
    _Tier("argon", ["a"]),
    _Tier("carbon", ["c"]),
]

_TIER_ENCODE: dict[str, str] = {t.name: t.codes[0] for t in _TIERS}
_TIER_DECODE: dict[str, tuple[str, int]] = {code: (t.name, rank) for rank, t in enumerate(_TIERS) for code in t.codes}

_MODEL_RE = re.compile(r"^gemini-(\d(?:\.\d)?)-([a-z-]+)$")
_TRACK_RE = re.compile(r"(?:^|[./])(raw-)?(g(\d{1,2})([a-z]{1,2})-(\d+))$")


def generate_model_shortcode(model_name: str) -> str:
    """Generates a compact shortcode for a supported Gemini model identifier.

    Transforms Gemini models into 'g<version><tier>' (e.g., 'g4c', 'g4a', 'g4h', 'g38f', 'g35l').

    Args:
        model_name: Gemini model identifier (e.g., 'gemini-3.8-flash', 'gemini-4-hydrogen').

    Returns:
        str: Compact shortcode string.

    Raises:
        ValueError: If model_name is not a recognized Gemini model tier.
    """
    match = _MODEL_RE.fullmatch(model_name.strip().lower())
    if not match or (tier_code := _TIER_ENCODE.get(match.group(2))) is None:
        raise ValueError(f"Unsupported Gemini model name for shortcode: '{model_name}'")

    version_code = match.group(1).replace(".", "")
    return f"g{version_code}{tier_code}"


def generate_full_shortcode(
    model_name: str,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
) -> str:
    """Combines model shortcode with composite 6-digit prompt version suffix for final output.

    Args:
        model_name: Gemini model identifier string.
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
    """Structured metadata and ranking score parsed from a subtitle shortcode or file path."""

    model: str
    prompt_version_str: str
    shortcode: str
    is_refined: bool
    score: int

    @property
    def display_label(self) -> str:
        """Human-readable model label with draft indicator when unrefined.

        Returns:
            str: Display label string.
        """
        return self.model if self.is_refined else f"{self.model} (Draft)"

    @property
    def track_name(self) -> str:
        """Full subtitle track name including prompt version and draft indicator.

        Returns:
            str: Track name string.
        """
        base = f"{self.model} ({self.prompt_version_str})" if self.prompt_version_str else self.model
        return base if self.is_refined else f"{base} (Draft)"


def parse_subtitle_track(path_or_name: str | Path) -> SubtitleTrackInfo:
    """Parses model, prompt version, refinement status, and sort score from a subtitle path or shortcode.

    Args:
        path_or_name: Subtitle file path, filename, or shortcode string.

    Returns:
        SubtitleTrackInfo: Structured track metadata and numeric sort score.
    """
    raw = str(path_or_name).replace("\\", "/")
    stem = raw[:-4] if raw.lower().endswith(".srt") else raw
    is_int_path = "/.int/" in raw or raw.startswith(".int/")

    match = _TRACK_RE.search(stem)
    if match:
        raw_prefix = match.group(1)
        shortcode, ver_code, tier_code, prompt_version = (
            match.group(2),
            match.group(3),
            match.group(4),
            match.group(5),
        )
        tier_info = _TIER_DECODE.get(tier_code)
        if tier_info is not None:
            tier_name, tier_rank = tier_info
            ver_display = ver_code if len(ver_code) == 1 else f"{ver_code[0]}.{ver_code[1:]}"
            ver_score = int(ver_code.ljust(2, "0"))
            is_refined = not (bool(raw_prefix) or is_int_path)
            score = (
                (1 if is_refined else 0) * 10_000_000_000
                + ver_score * 100_000_000
                + tier_rank * 1_000_000
                + int(prompt_version)
            )
            return SubtitleTrackInfo(
                model=f"gemini-{ver_display}-{tier_name}",
                prompt_version_str=prompt_version,
                shortcode=shortcode,
                is_refined=is_refined,
                score=score,
            )

    fallback = stem.rsplit("/", 1)[-1]
    return SubtitleTrackInfo(
        model=fallback,
        prompt_version_str="",
        shortcode="",
        is_refined=not is_int_path,
        score=0,
    )


def subtitle_track_sort_key(path_or_name: str | Path) -> int:
    """Returns a numeric sort score for ordering subtitle files from highest to lowest quality.

    Args:
        path_or_name: Subtitle file path, filename, or shortcode string.

    Returns:
        int: Numeric score combining refinement status, Gemini version, tier rank, and prompt version.
    """
    return parse_subtitle_track(path_or_name).score
