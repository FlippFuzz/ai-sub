"""Unified subtitle stitching, offset shifting, and metadata embedding across all execution modes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysubs2 import SSAEvent, SSAFile

from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION, LyricsJob
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION, SubtitlesJob
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION
from ai_sub.telemetry import get_app_version


@dataclass
class StitchResult:
    """Encapsulates the output files and status of a subtitle stitching run."""

    generated_files: list[Path]
    total_events: int
    duration_ms: int
    lyrics_refined: bool
    status: str
    any_failed: bool
    max_retries_exceeded: bool
    any_pending: bool


def format_subtitle_metadata(metadata: dict[str, Any]) -> str:
    """Formats subtitle metadata dictionary into safely escaped JSON for SSA comments.

    Args:
        metadata: Key-value metadata dictionary.

    Returns:
        str: Encoded JSON string with curly braces escaped as parentheses.
    """
    return json.dumps(metadata, indent=2, ensure_ascii=False).replace("{", "(").replace("}", ")")


def apply_metadata_header(subs: SSAFile, metadata: dict[str, Any]) -> None:
    """Inserts a 1ms metadata comment event at index 0 of an SSAFile.

    Args:
        subs: The SSAFile to mutate.
        metadata: Metadata dictionary to serialize.
    """
    info_str = format_subtitle_metadata(metadata)
    subs.insert(0, SSAEvent(start=0, end=1, text=info_str))
    if len(subs) > 1 and subs[1].start < 1:
        subs[1].start = 1


def stitch_subtitles(
    segments: Sequence[tuple[str, int]],
    state_dir: Path,
    output_primary_path: Path,
    output_raw_path: Path | None,
    subtitles_shortcode: str,
    lyrics_shortcode: str,
    validation_buffer_ms: int,
    settings_dump: dict[str, Any],
    max_runs: int = 3,
    status_override: str | None = None,
    summary_available: bool = True,
    initial_offset_ms: int = 0,
    use_lyrics: bool = True,
) -> StitchResult:
    """Stitches segment subtitle and lyrics state files into unified, offset-shifted SRT files.

    Produces the primary subtitle file and, if musical performances were refined,
    an additional raw dialogue subtitles file.

    Args:
        segments: Sequence of (segment_stem, segment_duration_ms) pairs in order.
        state_dir: Directory containing intermediate YAML state files.
        output_primary_path: Filesystem path for the primary stitched SRT.
        output_raw_path: Filesystem path for the raw dialogue SRT if singing was refined.
        subtitles_shortcode: Model shortcode used for subtitle YAML state files.
        lyrics_shortcode: Model shortcode used for lyrics YAML state files.
        validation_buffer_ms: Validation buffer tolerance in milliseconds.
        settings_dump: Serialized configuration settings to embed in Event 0.
        max_runs: Maximum attempt limit for retries. Defaults to 3.
        status_override: Explicit status string ('Completed', 'RetriesExceeded', 'InProgress'). Defaults to None.
        summary_available: Whether whole-video summary was active and successful. Defaults to True.
        initial_offset_ms: Starting time offset in milliseconds. Defaults to 0.
        use_lyrics: Whether lyrics refinement was enabled. Defaults to True.

    Returns:
        StitchResult: Generated file paths and completion metrics.
    """
    all_subs = SSAFile()
    generated_files: list[Path] = []
    current_offset_ms = initial_offset_ms
    any_failed = False
    max_retries_exceeded = False
    any_pending = False
    lyrics_refined_any = False

    for stem, seg_dur_ms in segments:
        lyrics_file = state_dir / f"{stem}.lyrics.{lyrics_shortcode}.yaml"
        sub_file = state_dir / f"{stem}.subtitles.{subtitles_shortcode}.yaml"

        lyrics_job = LyricsJob.load(lyrics_file, validation_buffer_ms)
        sub_job = SubtitlesJob.load(sub_file, validation_buffer_ms)

        lyrics_required = bool(
            use_lyrics and sub_job and sub_job.response and sub_job.response.detected_musical_performances
        )

        active_resp = None
        if lyrics_required:
            if lyrics_job and lyrics_job.response:
                active_resp = lyrics_job.response
                lyrics_refined_any = True
        else:
            if sub_job and sub_job.response:
                active_resp = sub_job.response

        if active_resp:
            seg_subs = active_resp.get_ssafile()
            seg_subs.shift(ms=current_offset_ms)
            all_subs += seg_subs
        else:
            all_subs.append(
                SSAEvent(
                    start=current_offset_ms,
                    end=current_offset_ms + seg_dur_ms,
                    text="Error processing subtitles for this segment.",
                )
            )
            any_failed = True
            if lyrics_required:
                attempts = lyrics_job.total_attempts if lyrics_job else 0
            else:
                attempts = sub_job.total_attempts if sub_job else 0

            if attempts >= max_runs:
                max_retries_exceeded = True
            else:
                any_pending = True

        current_offset_ms += seg_dur_ms

    if status_override:
        final_status = status_override
    elif not any_failed:
        final_status = "Completed"
    elif max_retries_exceeded:
        final_status = "RetriesExceeded"
    else:
        final_status = "InProgress"

    metadata = {
        "ai_sub_version": get_app_version(),
        "generated_at": datetime.now().isoformat(),
        "summary_prompt_version": SUMMARY_PROMPT_VERSION if summary_available else None,
        "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
        "lyrics_prompt_version": LYRICS_PROMPT_VERSION,
        "status": final_status,
        "settings": settings_dump,
    }

    apply_metadata_header(all_subs, metadata)

    output_primary_path.parent.mkdir(parents=True, exist_ok=True)
    all_subs.save(str(output_primary_path))
    generated_files.append(output_primary_path)

    # Generate raw pre-lyrics subtitle track if musical performances were refined
    if lyrics_refined_any and output_raw_path is not None:
        raw_subs = SSAFile()
        raw_offset_ms = initial_offset_ms
        for stem, seg_dur_ms in segments:
            sub_file = state_dir / f"{stem}.subtitles.{subtitles_shortcode}.yaml"
            sub_job = SubtitlesJob.load(sub_file, validation_buffer_ms)
            if sub_job and sub_job.response:
                seg_raw_subs = sub_job.response.get_ssafile()
                seg_raw_subs.shift(ms=raw_offset_ms)
                raw_subs += seg_raw_subs
            else:
                raw_subs.append(
                    SSAEvent(
                        start=raw_offset_ms,
                        end=raw_offset_ms + seg_dur_ms,
                        text="Error processing subtitles for this segment.",
                    )
                )
            raw_offset_ms += seg_dur_ms

        apply_metadata_header(raw_subs, metadata)
        output_raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_subs.save(str(output_raw_path))
        generated_files.append(output_raw_path)

    return StitchResult(
        generated_files=generated_files,
        total_events=len(all_subs),
        duration_ms=current_offset_ms,
        lyrics_refined=lyrics_refined_any,
        status=final_status,
        any_failed=any_failed,
        max_retries_exceeded=max_retries_exceeded,
        any_pending=any_pending,
    )
