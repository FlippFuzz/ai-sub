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
    _Tier("hydrogen", ["h"]),
    _Tier("argon", ["a"]),
    _Tier("carbon", ["c"]),
]

_TIER_ENCODE: dict[str, str] = {t.name: t.codes[0] for t in _TIERS}
_TIER_DECODE: dict[str, tuple[str, int]] = {code: (t.name, rank) for rank, t in enumerate(_TIERS) for code in t.codes}

_MODEL_RE = re.compile(r"^gemini-(\d(?:\.\d)?)-([a-z-]+)$")
_CODE_RE = re.compile(r"^g(\d{1,2})([a-z]{1,2})$")
_TRACK_RE = re.compile(r"(?:^|[./])(raw-)?((?:g\d{1,2}[a-z]{1,2}-){1,3}\d{6})$")


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


def _decode_model_code(code: str) -> tuple[str, int] | None:
    """Decodes a single model shortcode into its full model name and numeric rank score.

    Args:
        code: Shortcode token (e.g., 'g38f', 'g4h').

    Returns:
        tuple[str, int] | None: (full_model_name, model_rank_score) or None if invalid.
    """
    match = _CODE_RE.fullmatch(code)
    if not match:
        return None
    ver_code, tier_code = match.group(1), match.group(2)
    tier_info = _TIER_DECODE.get(tier_code)
    if tier_info is None:
        return None
    tier_name, tier_rank = tier_info
    ver_display = ver_code if len(ver_code) == 1 else f"{ver_code[0]}.{ver_code[1:]}"
    ver_score = int(ver_code.ljust(2, "0"))
    return f"gemini-{ver_display}-{tier_name}", ver_score * 100 + tier_rank


def generate_full_shortcode(
    subtitles_model: str,
    summary_model: str | None = None,
    lyrics_model: str | None = None,
    summary_prompt_version: int = SUMMARY_PROMPT_VERSION,
    subtitles_prompt_version: int = SUBTITLES_PROMPT_VERSION,
    lyrics_prompt_version: int = LYRICS_PROMPT_VERSION,
) -> str:
    """Builds a full multi-stage lineage shortcode and 6-digit prompt version suffix.

    Missing or disabled stages omit their model code and record '00' in their 2-digit slot.

    Args:
        subtitles_model: Gemini model identifier for subtitle transcription.
        summary_model: Optional Gemini model identifier for summary (None when disabled).
        lyrics_model: Optional Gemini model identifier for lyrics refinement (None when disabled).
        summary_prompt_version: Summary prompt version number. Defaults to SUMMARY_PROMPT_VERSION.
        subtitles_prompt_version: Subtitle prompt version number. Defaults to SUBTITLES_PROMPT_VERSION.
        lyrics_prompt_version: Lyrics prompt version number. Defaults to LYRICS_PROMPT_VERSION.

    Returns:
        str: Lineage tag formatted as '<models>-<sum_v><sub_v><lyr_v>' (e.g., 'g35l-g4h-g35l-011910').

    Raises:
        ValueError: If any prompt version is outside valid bounds for 2-digit formatting.
    """
    if not (1 <= subtitles_prompt_version <= 99):
        raise ValueError(f"subtitles_prompt_version must be between 1 and 99, got {subtitles_prompt_version}.")
    if summary_model is not None and not (0 <= summary_prompt_version <= 99):
        raise ValueError(f"summary_prompt_version must be between 0 and 99, got {summary_prompt_version}.")
    if lyrics_model is not None and not (0 <= lyrics_prompt_version <= 99):
        raise ValueError(f"lyrics_prompt_version must be between 0 and 99, got {lyrics_prompt_version}.")

    model_codes: list[str] = []
    if summary_model is not None and summary_prompt_version > 0:
        model_codes.append(generate_model_shortcode(summary_model))
        sum_v = summary_prompt_version
    else:
        sum_v = 0

    model_codes.append(generate_model_shortcode(subtitles_model))

    if lyrics_model is not None and lyrics_prompt_version > 0:
        model_codes.append(generate_model_shortcode(lyrics_model))
        lyr_v = lyrics_prompt_version
    else:
        lyr_v = 0

    return f"{'-'.join(model_codes)}-{sum_v:02d}{subtitles_prompt_version:02d}{lyr_v:02d}"


@dataclass(frozen=True)
class SubtitleTrackInfo:
    """Structured metadata and ranking score parsed from a subtitle lineage shortcode or file path."""

    model: str
    summary_model: str | None
    lyrics_model: str | None
    models_shortcode: str
    prompt_version_str: str
    shortcode: str
    is_refined: bool
    score: int

    @property
    def display_label(self) -> str:
        """Compact model lineage label with draft indicator when unrefined.

        Returns:
            str: Display label string (e.g., 'g35l-g4h-g35l' or 'g35l-g4h-g35l [Draft]').
        """
        base = self.models_shortcode if self.shortcode else self.model
        return base if self.is_refined else f"{base} [Draft]"

    @property
    def track_name(self) -> str:
        """Full subtitle track name including lineage shortcode and draft indicator.

        Returns:
            str: Track name string (e.g., 'gemini-4-hydrogen [g35l-g4h-g35l-011910]').
        """
        base = f"{self.model} [{self.shortcode}]" if self.shortcode else self.model
        return base if self.is_refined else f"{base} [Draft]"


def parse_subtitle_track(path_or_name: str | Path) -> SubtitleTrackInfo:
    """Parses model lineage, prompt versions, refinement status, and sort score from a subtitle path or shortcode.

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
        shortcode = match.group(2)
        models_shortcode, prompt_version_str = shortcode.rsplit("-", 1)
        sum_v = int(prompt_version_str[0:2])
        sub_v = int(prompt_version_str[2:4])
        lyr_v = int(prompt_version_str[4:6])

        code_strs = models_shortcode.split("-")
        active_stages = [
            stage for stage, ver in (("summary", sum_v), ("subtitles", sub_v), ("lyrics", lyr_v)) if ver > 0
        ]

        if sub_v > 0 and len(code_strs) == len(active_stages):
            decoded = [_decode_model_code(c) for c in code_strs]
            if all(item is not None for item in decoded):
                stage_map = dict(zip(active_stages, [item for item in decoded if item is not None], strict=True))
                sub_model, sub_score = stage_map["subtitles"]
                sum_model, sum_score = stage_map.get("summary", (None, 0))
                lyr_model, lyr_score = stage_map.get("lyrics", (None, 0))
                is_refined = not (bool(raw_prefix) or is_int_path)

                score = (
                    (1 if is_refined else 0) * 10**18
                    + sub_score * 10**14
                    + sum_score * 10**10
                    + lyr_score * 10**6
                    + sub_v * 10**4
                    + lyr_v * 10**2
                    + sum_v
                )
                return SubtitleTrackInfo(
                    model=sub_model,
                    summary_model=sum_model,
                    lyrics_model=lyr_model,
                    models_shortcode=models_shortcode,
                    prompt_version_str=prompt_version_str,
                    shortcode=shortcode,
                    is_refined=is_refined,
                    score=score,
                )

    fallback = stem.rsplit("/", 1)[-1]
    return SubtitleTrackInfo(
        model=fallback,
        summary_model=None,
        lyrics_model=None,
        models_shortcode=fallback,
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
        int: Numeric score combining refinement status, stage models, and prompt versions.
    """
    return parse_subtitle_track(path_or_name).score
