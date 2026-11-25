"""Unified video processing pipeline managing segmentation, compression, AI DAG stages, and stitching."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import logfire
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt
from pysubs2 import SSAEvent, SSAFile

from ai_sub.config import (
    ReencodeSettings,
    RetryPolicy,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
    StageExecutionConfig,
    StageReencodeSettings,
    atomic_write_text,
)
from ai_sub.gemini_executor import calculate_sha256_hex
from ai_sub.jobs.job_base import AiJob, BaseJob, TaskType, parse_timestamp_string_ms
from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION, LyricsAiResponse, LyricsJob
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION, SubtitlesAiResponse, SubtitlesJob
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION, SummaryAiResponse, SummaryJob
from ai_sub.shortcode import generate_full_shortcode
from ai_sub.telemetry import get_app_version
from ai_sub.video import (
    compute_file_id_hash,
    get_keyframe_fps,
    get_video_duration_ms,
    get_video_fps,
    has_audio_stream,
    is_valid_reencoded_file,
    sanitize_filename,
    split_video,
)

_MD_ESCAPE_RE = re.compile(r"([\\*_`\[\]<>~])")


@dataclass(frozen=True)
class FailureOutcome:
    """Result returned when recording a non-retryable task or re-encode failure."""

    action: Literal["retry", "fallback_reencode", "exhausted", "ignored"]
    fallback_tier: int = 0
    target_fps: float | None = None
    tier_attempts: int = 0


@dataclass(frozen=True)
class FallbackProbeSpec:
    """Descriptor specifying which FFprobe checks should run off-lock before advancing fallback tiers."""

    source_path: Path
    keyframe_media: Path | None = None
    probe_keyframe_fps: bool = False
    probe_audio: bool = False


class MediaStageState(BaseModel):
    """Shared media preparation and FPS fallback state for a summary or segment unit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    media_ready: bool = Field(default=False, description="Whether media is ready for AI inference.")
    media_path: Path | None = Field(default=None, description="Path to active compressed or raw media.")
    target_fps: NonNegativeFloat | Literal["keyframes"] | None = Field(
        default=None,
        description="Target FPS (0 for audio-only) or 'keyframes' for re-encoding, or None if using raw media.",
    )
    fallback_tier: NonNegativeInt = Field(default=0, description="Active fallback FPS tier index.")


class SummaryState(MediaStageState):
    """Per-unit execution and media state for the Phase 1 whole-video summary."""

    tier_attempts: NonNegativeInt = Field(default=0, description="Non-retryable failures at the current FPS tier.")
    response: SummaryAiResponse | None = Field(default=None, description="Completed summary response.")
    exhausted: bool = Field(default=False, description="Whether summary exhausted all retries and fallback tiers.")


class SegmentState(MediaStageState):
    """Per-unit execution, media, and AI stage state for an individual video chunk."""

    index: int = Field(description="Zero-based segment index.")
    raw_path: Path = Field(description="Path to the split uncompressed segment chunk.")
    duration_ms: PositiveInt = Field(description="Duration of this segment in milliseconds.")
    offset_ms: int = Field(description="Absolute start offset of this segment in milliseconds.")

    subtitles_tier_attempts: NonNegativeInt = Field(
        default=0,
        description="Non-retryable subtitle failures at the current FPS tier.",
    )
    subtitles_response: SubtitlesAiResponse | None = Field(default=None, description="Completed subtitles response.")
    subtitles_exhausted: bool = Field(
        default=False,
        description="Whether subtitles exhausted all retries and fallback tiers.",
    )

    lyrics_tier_attempts: NonNegativeInt = Field(default=0, description="Non-retryable lyrics failures.")
    lyrics_response: LyricsAiResponse | None = Field(default=None, description="Completed lyrics response.")
    lyrics_exhausted: bool = Field(default=False, description="Whether lyrics exhausted retries.")


def _float_fps(val: float | Literal["keyframes"] | None) -> float | None:
    """Converts a numeric FPS value to float, returning None for 'keyframes' or None.

    Args:
        val: FPS target value.

    Returns:
        float | None: Numeric FPS or None.
    """
    return float(val) if isinstance(val, (int, float)) else None


def _content_dump(model: BaseModel | None) -> dict[str, Any] | None:
    """Serializes a response model excluding internal reasoning traces for upstream checks.

    Args:
        model: Pydantic model instance or None.

    Returns:
        dict[str, Any] | None: Serialized dictionary without 'thoughts', or None.
    """
    if model is None:
        return None
    return model.model_dump(mode="json", exclude={"thoughts"})


def _apply_metadata_header(subs: SSAFile, metadata: dict[str, Any]) -> None:
    """Sorts events chronologically and inserts a 1ms metadata comment event at index 0.

    Shifts any events starting at 0ms forward to 1ms so Event 0 stays cleanly ordered.

    Args:
        subs: The SSAFile to mutate.
        metadata: Metadata dictionary to serialize.
    """
    subs.sort()
    for ev in subs:
        if ev.start < 1:
            ev.start = 1
            if ev.end <= ev.start:
                ev.end = ev.start + 1
    info_str = json.dumps(metadata, indent=2, ensure_ascii=False).replace("{", "(").replace("}", ")")
    subs.insert(0, SSAEvent(start=0, end=1, text=info_str))


def _append_segment_events(
    target: SSAFile,
    resp: SubtitlesAiResponse | LyricsAiResponse | None,
    seg: SegmentState,
    error_text: str = "Error processing subtitles for this segment.",
) -> None:
    """Appends time-shifted subtitle events for a segment, or an error placeholder if missing.

    Args:
        target: Destination SSAFile accumulating stitched subtitle events.
        resp: Completed subtitle or lyrics response, or None if unavailable.
        seg: Segment metadata providing offset and duration.
        error_text: Placeholder text when resp is None.
    """
    if resp is not None:
        seg_subs = resp.get_ssafile()
        seg_subs.shift(ms=seg.offset_ms)
        target += seg_subs
    else:
        target.append(
            SSAEvent(
                start=seg.offset_ms,
                end=seg.offset_ms + seg.duration_ms,
                text=error_text,
            )
        )


def _collapse_ws(text: str) -> str:
    """Collapses internal whitespace and line breaks into single spaces.

    Args:
        text: Input string to normalize.

    Returns:
        str: Trimmed single-line text.
    """
    return " ".join(text.split())


def _escape_md(text: str) -> str:
    """Escapes inline Markdown and HTML-triggering characters for Jellyfin overview rendering.

    Args:
        text: Raw text string.

    Returns:
        str: Markdown-escaped string.
    """
    return _MD_ESCAPE_RE.sub(r"\\\1", text)


def _build_nfo_plot(summary: SummaryAiResponse) -> str:
    """Formats all global summary sections into a Markdown Jellyfin `<plot>` string.

    Args:
        summary: Phase 1 global video summary containing overview, chapters, entities, and glossary.

    Returns:
        str: Formatted plot text, or empty string if all sections are empty.
    """
    sections: list[str] = []

    raw_summary = summary.overall_summary.replace("\r\n", "\n").replace("\r", "\n").strip()
    if raw_summary:
        # Preserve paragraph breaks while escaping inline syntax and accidental ATX headings.
        lines = [line.strip() for line in raw_summary.splitlines()]
        normalized = re.sub(r"\n{2,}", "\n\n", "\n".join(lines))
        clean_summary = re.sub(r"(?m)^(#{1,6})(?=\s|$)", r"\\\1", _escape_md(normalized))
        sections.append(clean_summary)

    chapter_lines: list[str] = []
    for ch in summary.chapters:
        ch_title = _escape_md(_collapse_ws(ch.title))
        if not ch_title:
            continue
        start = _escape_md(_collapse_ws(ch.start))
        end = _escape_md(_collapse_ws(ch.end))
        header = f"**{start}–{end} — {ch_title}**" if (start and end) else f"**{ch_title}**"

        song_title = _escape_md(_collapse_ws(ch.song_title)) if ch.song_title else ""
        song_artist = _escape_md(_collapse_ws(ch.song_artist)) if ch.song_artist else ""
        if song_title and song_artist:
            song_part = f" [🎵 {song_title} — {song_artist}]"
        elif song_title or song_artist:
            song_part = f" [🎵 {song_title or song_artist}]"
        else:
            song_part = ""

        desc = _escape_md(_collapse_ws(ch.description))
        chapter_lines.append(f"- {header}{song_part}: {desc}" if desc else f"- {header}{song_part}")

    if chapter_lines:
        sections.append("\n".join([f"### Chapters ({len(chapter_lines)})", *chapter_lines]))

    entity_lines: list[str] = []
    entity_count = 0
    for ent in summary.entities:
        name = _escape_md(_collapse_ws(ent.name))
        if not name:
            continue
        entity_count += 1
        head = f"- **{name}**"
        if ent.original_name and (orig := _escape_md(_collapse_ws(ent.original_name))):
            head += f" ({orig})"
        if ent.role_or_affiliation and (role := _escape_md(_collapse_ws(ent.role_or_affiliation))):
            head += f" — {role}"
        entity_lines.append(head)

        for label, raw_val in (
            ("Lore", ent.persona_lore),
            ("Speech", ent.speech_patterns_or_ticks),
            ("Visuals", ent.visual_cues),
        ):
            if raw_val and (val := _escape_md(_collapse_ws(raw_val))):
                entity_lines.append(f"  - {label}: {val}")

    if entity_count > 0:
        sections.append("\n".join([f"### Entities ({entity_count})", *entity_lines]))

    glossary_lines: list[str] = []
    for item in summary.domain_glossary:
        term = _escape_md(_collapse_ws(item.term))
        meaning = _escape_md(_collapse_ws(item.meaning))
        if term and meaning:
            glossary_lines.append(f"- **{term}**: {meaning}")
        elif term or meaning:
            glossary_lines.append(f"- {term or meaning}")

    if glossary_lines:
        sections.append("\n".join([f"### Domain Glossary ({len(glossary_lines)})", *glossary_lines]))

    return "\n\n".join(sections)


def _build_nfo_xml(summary: SummaryAiResponse, title: str | None = None) -> str:
    """Builds a Jellyfin-compatible Movie NFO XML string from a global summary response.

    Args:
        summary: Phase 1 global video summary containing plot and chapter timeline.
        title: Optional video title to expose in `<title>`. Omitted when None.

    Returns:
        str: Formatted UTF-8 XML document string.
    """
    root = ET.Element("movie")

    if title and (clean_title := title.strip()):
        title_el = ET.SubElement(root, "title")
        title_el.text = clean_title

    if plot_text := _build_nfo_plot(summary):
        plot_el = ET.SubElement(root, "plot")
        plot_el.text = plot_text

    chapter_entries: list[tuple[str, int]] = []
    prev_start_s: int | None = None
    for ch in summary.chapters:
        ch_title = ch.title.strip()
        if not ch_title:
            continue
        try:
            start_ms = parse_timestamp_string_ms(ch.start)
        except ValueError:
            continue

        raw_s = start_ms // 1000
        if prev_start_s is not None and raw_s <= prev_start_s:
            start_s = max(prev_start_s + 1, math.ceil(start_ms / 1000))
        else:
            start_s = raw_s

        prev_start_s = start_s
        chapter_entries.append((ch_title, start_s))

    if chapter_entries:
        chapters_el = ET.SubElement(root, "chapters")
        for ch_title, start_s in chapter_entries:
            ET.SubElement(chapters_el, "chapter", name=ch_title, start=str(start_s))

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    return f'<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n{body}\n'


def _resolve_target_fps(
    current_target: float | Literal["keyframes"] | None,
    fallback_tier: int,
    cfg: StageReencodeSettings,
    source_path: Path,
) -> float | Literal["keyframes"] | None:
    """Determines target compression FPS from fallback state or size threshold configuration.

    Args:
        current_target: Persisted fallback target FPS if already in fallback, else None.
        fallback_tier: Active fallback tier index (0 = initial, 1+ = fallback).
        cfg: Re-encoding settings for this stage.
        source_path: Input media file path to evaluate against threshold.

    Returns:
        float | Literal["keyframes"] | None: Target FPS (0 for audio-only) or 'keyframes', or None for raw.
    """
    if fallback_tier > 0 and current_target is not None:
        return current_target
    if cfg.threshold > 0 and source_path.stat().st_size >= cfg.threshold:
        return cfg.fps
    return None


async def probe_keyframe_base_fps(
    source_path: Path,
    keyframe_media: Path | None = None,
) -> float:
    """Probes the effective keyframe rate from an encoded keyframe proxy or the raw source video.

    Args:
        source_path: Uncompressed source video path.
        keyframe_media: Optional already-encoded keyframes-only media file from tier 0.

    Returns:
        float: Positive keyframe rate in FPS (defaults to 1.0 if probing fails).
    """
    try:
        if keyframe_media is not None and keyframe_media.is_file():
            probed_fps = await get_video_fps(keyframe_media)
        else:
            probed_fps = await get_keyframe_fps(source_path)
        if probed_fps > 0:
            return probed_fps
        logfire.warning(
            f"Probed non-positive keyframe rate ({probed_fps}) for {source_path.name}; defaulting base FPS to 1.0."
        )
    except Exception as e:
        logfire.warning(f"Failed to probe keyframe rate for {source_path.name} ({e}); defaulting base FPS to 1.0.")
    return 1.0


async def probe_fallback_info(spec: FallbackProbeSpec) -> tuple[float | None, bool | None]:
    """Runs FFprobe keyframe-FPS and audio-stream checks concurrently off-lock.

    Args:
        spec: Probe descriptor produced by Pipeline.get_fallback_probe_spec().

    Returns:
        tuple[float | None, bool | None]: (probed_base_fps, has_audio) matching the requested flags.
    """
    if spec.probe_keyframe_fps and spec.probe_audio:
        base_fps, has_audio = await asyncio.gather(
            probe_keyframe_base_fps(spec.source_path, keyframe_media=spec.keyframe_media),
            has_audio_stream(spec.source_path),
        )
        return base_fps, has_audio
    if spec.probe_keyframe_fps:
        return await probe_keyframe_base_fps(spec.source_path, keyframe_media=spec.keyframe_media), None
    if spec.probe_audio:
        return None, await has_audio_stream(spec.source_path)
    return None, None


async def _compute_next_fallback_fps(
    cfg: StageReencodeSettings,
    current_tier: int,
    current_target: float | Literal["keyframes"] | None,
    existing_media: Path | None,
    source_path: Path,
    probed_base_fps: float | None = None,
    has_audio: bool | None = None,
) -> float | None:
    """Computes the target FPS for the next fallback tier, snapping below minfps to 0.0 (audio-only).

    Args:
        cfg: Stage re-encoding settings containing base FPS, minfps floor, and divisor.
        current_tier: Current fallback tier index before incrementing.
        current_target: Active FPS or 'keyframes' target at the current tier.
        existing_media: Compressed media path from the current tier, if present.
        source_path: Uncompressed source video path.
        probed_base_fps: Optional pre-probed base keyframe FPS computed outside locks. Defaults to None.
        has_audio: Optional pre-probed audio-stream presence flag computed outside locks. Defaults to None.

    Returns:
        float | None: Next FPS target (0.0 for audio-only), or None if no audio stream exists for 0 fps.
    """
    if probed_base_fps is not None:
        base_fps = probed_base_fps
    elif current_tier > 0 and isinstance(current_target, (int, float)):
        base_fps = float(current_target)
    elif cfg.fps == "keyframes":
        keyframe_media = existing_media if current_target == "keyframes" else None
        base_fps = await probe_keyframe_base_fps(source_path, keyframe_media=keyframe_media)
    else:
        base_fps = float(cfg.fps)

    candidate_fps = base_fps / float(cfg.fallback.divisor)
    if candidate_fps >= cfg.fallback.minfps:
        return candidate_fps

    audio_available = has_audio if has_audio is not None else await has_audio_stream(source_path)
    return 0.0 if audio_available else None


class Pipeline(BaseModel):
    """Manages the full lifecycle, media preparation, on-disk checkpoints, and DAG state for a video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_video_path: Path = Field(description="Filesystem path to the raw input video.")
    job_dir: Path = Field(description="Working directory containing state files and segments for this video.")
    video_id: str = Field(description="Unique video identifier.")
    media_tag: str = Field(description="11-character video ID or filename hash used in intermediate filenames.")
    title: str = Field(description="Display title of the video.")
    uploader: str = Field(description="Channel or uploader identifier.")
    channel_name: str | None = Field(default=None, description="Human-readable channel display name.")
    filename_stem: str = Field(description="Filesystem-safe base name for output files.")
    source_type: Literal["local", "youtube"] = Field(default="local", description="Ingestion source type.")
    timestamp: int = Field(default=0, description="Video publication or intake Unix timestamp.")
    user_context: str | None = Field(default=None, description="Background user context string.")
    duration_ms: int = Field(description="Total duration of the input video in milliseconds.")
    validation_buffer_ms: int = Field(default=2000, description="Timestamp validation tolerance in milliseconds.")
    summary_enabled: bool = Field(default=True, description="Whether whole-video summary stage is enabled.")
    lyrics_enabled: bool = Field(default=True, description="Whether musical lyrics refinement is enabled.")

    summary_config: StageExecutionConfig = Field(description="Model, tier, and agentic config for summary.")
    subtitles_config: StageExecutionConfig = Field(description="Model, tier, and agentic config for subtitles.")
    lyrics_config: StageExecutionConfig = Field(description="Model, tier, and agentic config for lyrics.")

    summary_shortcode: str = Field(description="Model shortcode for summary.")
    subtitles_shortcode: str = Field(description="Model lineage shortcode for subtitles.")
    lyrics_shortcode: str = Field(description="Model lineage shortcode for lyrics.")
    full_shortcode: str = Field(description="Combined full lineage shortcode for final output.")

    reencode_settings: ReencodeSettings = Field(description="Re-encoding configuration settings.")
    search_tools: list[str] = Field(default_factory=list, description="Enabled search provider names.")
    search_settings: SearchSettings | None = Field(
        default=None,
        description="Search configuration settings when running standalone.",
    )
    split_settings: SplittingSettings = Field(description="Splitting configuration settings.")
    retry_settings: RetrySettings = Field(description="Retry configuration settings.")

    summary: SummaryState = Field(default_factory=SummaryState, description="Whole-video summary unit state.")
    segments: list[SegmentState] = Field(default_factory=list, description="Ordered per-segment state machines.")

    def get_retry_settings(self, task_type: TaskType) -> RetryPolicy:
        """Resolves the effective retry policy for a pipeline task type based on its configured model.

        Args:
            task_type: AI task type (SUMMARY, SUBTITLES, or LYRICS).

        Returns:
            RetryPolicy: Effective retry policy for the stage's model.
        """
        if task_type == TaskType.SUMMARY:
            model = self.summary_config.model
        elif task_type == TaskType.SUBTITLES:
            model = self.subtitles_config.model
        else:
            model = self.lyrics_config.model
        return self.retry_settings.resolve_for_model(model)

    @property
    def summary_aborted(self) -> bool:
        """Whether the summary stage permanently failed and aborted pipeline execution.

        Returns:
            bool: True if summary is enabled and exhausted all retries, False otherwise.
        """
        return self.summary_enabled and self.summary.exhausted

    def segment_subtitles_concluded(self, seg: SegmentState) -> bool:
        """Checks whether subtitle transcription for a segment has completed or exhausted retries.

        Args:
            seg: Target segment state.

        Returns:
            bool: True if subtitles completed or exhausted, False otherwise.
        """
        return seg.subtitles_response is not None or seg.subtitles_exhausted

    def segment_needs_lyrics(self, seg: SegmentState) -> bool:
        """Checks whether a segment requires lyrics refinement based on non-empty draft subtitles and singing.

        Args:
            seg: Target segment state.

        Returns:
            bool: True if lyrics refinement is enabled and segment has both subtitles and detected singing.
        """
        return bool(
            self.lyrics_enabled
            and seg.subtitles_response is not None
            and seg.subtitles_response.subtitles
            and seg.subtitles_response.detected_musical_performances
        )

    def segment_lyrics_concluded(self, seg: SegmentState) -> bool:
        """Checks whether lyrics refinement for a segment is unnecessary or has finished/exhausted.

        Args:
            seg: Target segment state.

        Returns:
            bool: True if subtitles exhausted, lyrics not required, or lyrics completed/exhausted.
        """
        if seg.subtitles_exhausted:
            return True
        if seg.subtitles_response is None:
            return False
        if not self.segment_needs_lyrics(seg):
            return True
        return seg.lyrics_response is not None or seg.lyrics_exhausted

    @classmethod
    async def create(
        cls,
        raw_video_path: Path,
        job_dir: Path,
        title: str,
        uploader: str,
        summary_config: StageExecutionConfig,
        subtitles_config: StageExecutionConfig,
        lyrics_config: StageExecutionConfig,
        reencode_settings: ReencodeSettings,
        split_settings: SplittingSettings,
        retry_settings: RetrySettings,
        search_settings: SearchSettings | None = None,
        search_tools: list[str] | None = None,
        channel_name: str | None = None,
        video_id: str | None = None,
        media_tag: str | None = None,
        output_stem: str | None = None,
        source_type: Literal["local", "youtube"] = "local",
        timestamp: int = 0,
        user_context: str | None = None,
        validation_buffer_ms: int = 2000,
        summary_enabled: bool = True,
        lyrics_enabled: bool = True,
    ) -> Self:
        """Asynchronously initializes the pipeline, probes media, splits segments, and resumes checkpoints.

        Loads YAML state checkpoints before validating compressed media on disk so any unit
        in a fallback FPS tier validates against its persisted target FPS.

        Args:
            raw_video_path: Path to the raw input video.
            job_dir: Working directory containing state files and segments for this video.
            title: Display title of the video.
            uploader: Channel or uploader identifier.
            summary_config: Summary stage model, tier, and agentic configuration.
            subtitles_config: Subtitles stage model, tier, and agentic configuration.
            lyrics_config: Lyrics stage model, tier, and agentic configuration.
            reencode_settings: Re-encoding configuration settings.
            split_settings: Splitting configuration settings.
            retry_settings: Retry configuration settings.
            search_settings: Optional standalone search configuration settings. Defaults to None.
            search_tools: Optional explicit search tool names (used in server mode). Defaults to None.
            channel_name: Human-readable channel display name. Defaults to None.
            video_id: Unique video identifier. Defaults to raw_video_path.stem.
            media_tag: Compact identifier (YouTube ID or 11-char filename hash) for intermediate files.
                Defaults to video_id for YouTube or SHA-256 filename hash for local files.
            output_stem: Base filename stem used for finalized SRT outputs. Defaults to video_id for
                YouTube or sanitized raw_video_path.stem for local files.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication or intake Unix timestamp. Defaults to 0.
            user_context: Background user context string. Defaults to None.
            validation_buffer_ms: Timestamp validation tolerance in milliseconds. Defaults to 2000.
            summary_enabled: Whether the whole-video summary stage is enabled. Defaults to True.
            lyrics_enabled: Whether musical lyrics refinement is enabled. Defaults to True.

        Returns:
            Pipeline: Initialized pipeline instance with media prepared and checkpoints loaded.

        Raises:
            RuntimeError: If no valid positive-duration video segments could be created.
            ValueError: If split offset exceeds or equals the total video duration.
        """
        raw_resolved = raw_video_path.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        vid = video_id or raw_resolved.stem
        resolved_tag = media_tag or (vid if source_type == "youtube" else compute_file_id_hash(raw_resolved.name))
        default_stem = vid if source_type == "youtube" else raw_resolved.stem
        filename_stem = sanitize_filename(output_stem if output_stem is not None else default_stem)

        # 1. Probe raw video duration and validate split offset before splitting
        total_duration_ms = await get_video_duration_ms(raw_resolved)
        start_offset_s = int(split_settings.offset.total_seconds())
        if start_offset_s * 1000 >= total_duration_ms:
            raise ValueError(
                f"split.offset ({start_offset_s}s) exceeds or equals total "
                f"video duration ({total_duration_ms / 1000:.1f}s)."
            )

        # 2. Split video into uniform segments
        split_duration_s = max(1, int(split_settings.duration.total_seconds()))
        split_tolerance_ms = int(split_settings.tolerance.total_seconds() * 1000)
        all_segments = await split_video(
            input_video=raw_resolved,
            output_dir=job_dir,
            split_duration_s=split_duration_s,
            media_tag=resolved_tag,
            duration_tolerance_ms=split_tolerance_ms,
        )
        if not all_segments:
            raise RuntimeError(f"No video segments produced for {raw_resolved.name}.")

        all_durations = await asyncio.gather(*(get_video_duration_ms(p) for p in all_segments))

        # 3. Deliberately round split.offset down to whole segment chunks (no mid-file trimming)
        # so stream-copied chunks remain aligned with uniform part_%03d segment indices.
        if start_offset_s > 0:
            skip_count = start_offset_s // split_duration_s
            remainder_s = start_offset_s % split_duration_s
            if skip_count >= len(all_segments):
                raise ValueError(
                    f"split.offset ({start_offset_s}s) skips all {len(all_segments)} segment(s) "
                    f"for video duration ({total_duration_ms / 1000:.1f}s)."
                )
            if remainder_s > 0:
                logfire.info(
                    f"split.offset={start_offset_s}s rounded down to {skip_count * split_duration_s}s "
                    f"({skip_count} whole segment(s) of {split_duration_s}s)."
                )
            initial_offset_ms = sum(all_durations[:skip_count])
            segment_paths = all_segments[skip_count:]
            segment_durations_ms = all_durations[skip_count:]
            if skip_count > 0:
                logfire.info(
                    f"Skipping first {skip_count} segment(s) ({initial_offset_ms / 60000:.1f}m) "
                    f"due to split.offset={start_offset_s}s."
                )
        else:
            initial_offset_ms = 0
            segment_paths = all_segments
            segment_durations_ms = all_durations

        # 4. Build ordered SegmentState entries with cumulative timeline offsets
        segments: list[SegmentState] = []
        current_offset_ms = initial_offset_ms
        for idx, (seg_p, seg_dur_ms) in enumerate(zip(segment_paths, segment_durations_ms, strict=True)):
            if seg_dur_ms <= 0:
                raise RuntimeError(f"Segment '{seg_p.name}' has non-positive duration ({seg_dur_ms}ms).")
            segments.append(
                SegmentState(
                    index=idx,
                    raw_path=seg_p,
                    duration_ms=seg_dur_ms,
                    offset_ms=current_offset_ms,
                )
            )
            current_offset_ms += seg_dur_ms

        # 5. Generate lineage shortcodes and derive stage prefixes directly from full_shortcode
        full_shortcode = generate_full_shortcode(
            subtitles_model=subtitles_config.model,
            summary_model=summary_config.model if summary_enabled else None,
            lyrics_model=lyrics_config.model if lyrics_enabled else None,
        )
        codes = full_shortcode.rsplit("-", 1)[0].split("-")
        summary_shortcode = codes[0] if summary_enabled else ""
        subtitles_shortcode = "-".join(codes[: 2 if summary_enabled else 1])
        lyrics_shortcode = "-".join(codes) if lyrics_enabled else ""

        resolved_search_tools = (
            list(search_tools)
            if search_tools is not None
            else ([str(t) for t in search_settings.tools] if search_settings is not None else [])
        )

        instance = cls(
            raw_video_path=raw_resolved,
            job_dir=job_dir,
            video_id=vid,
            media_tag=resolved_tag,
            title=title,
            uploader=uploader,
            channel_name=channel_name,
            filename_stem=filename_stem,
            source_type=source_type,
            timestamp=timestamp,
            user_context=user_context,
            duration_ms=total_duration_ms,
            validation_buffer_ms=validation_buffer_ms,
            summary_enabled=summary_enabled,
            lyrics_enabled=lyrics_enabled,
            summary_config=summary_config,
            subtitles_config=subtitles_config,
            lyrics_config=lyrics_config,
            summary_shortcode=summary_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            lyrics_shortcode=lyrics_shortcode,
            full_shortcode=full_shortcode,
            reencode_settings=reencode_settings,
            search_tools=resolved_search_tools,
            search_settings=search_settings,
            split_settings=split_settings,
            retry_settings=retry_settings,
            segments=segments,
        )

        # 6. Load YAML checkpoints first so fallback FPS targets are known before media validation
        instance._discover_on_disk_state()
        await instance._prepare_media_states()
        return instance

    def _segment_yaml_stem(self, seg: SegmentState) -> str:
        """Returns the segment YAML checkpoint stem without the video media tag suffix.

        Args:
            seg: Target segment state.

        Returns:
            str: Segment prefix (e.g. 'part_000').
        """
        return seg.raw_path.stem.removesuffix(f"_{self.media_tag}")

    def _reencoded_output_path(
        self,
        target_fps: float | Literal["keyframes"] | None,
        seg: SegmentState | None = None,
    ) -> Path:
        """Returns the compressed media file path (.m4a for audio-only 0 fps, .mp4 otherwise).

        Args:
            target_fps: Target FPS, 'keyframes', or None.
            seg: Target segment state, or None for the summary proxy. Defaults to None.

        Returns:
            Path: Destination path for the compressed media output.
        """
        ext = ".m4a" if target_fps == 0 else ".mp4"
        if seg is None:
            return self.job_dir / f"reencoded_summary_{self.media_tag}{ext}"
        return self.job_dir / f"reencoded_{seg.raw_path.stem}{ext}"

    def _fallback_stage_context(
        self,
        task_type: TaskType,
        seg: SegmentState | None,
    ) -> tuple[MediaStageState, StageReencodeSettings, Path, str, int, bool]:
        """Resolves media state, re-encode rules, source path, label, attempts, and conclusion flag for a stage.

        Args:
            task_type: Target AI stage (SUMMARY or SUBTITLES).
            seg: Target segment state when task_type is SUBTITLES.

        Returns:
            tuple[MediaStageState, StageReencodeSettings, Path, str, int, bool]:
                (media_state, reencode_cfg, source_path, label, tier_attempts, is_concluded).
        """
        if task_type == TaskType.SUMMARY:
            s = self.summary
            return (
                s,
                self.reencode_settings.summary,
                self.raw_video_path,
                "Summary",
                s.tier_attempts,
                s.response is not None or s.exhausted,
            )
        assert seg is not None
        return (
            seg,
            self.reencode_settings.segment,
            seg.raw_path,
            f"Subtitles for '{self._segment_yaml_stem(seg)}'",
            seg.subtitles_tier_attempts,
            seg.subtitles_response is not None or seg.subtitles_exhausted,
        )

    def _restore_failure_state(
        self,
        job: AiJob,
        fallback_enabled: bool,
        max_runs: int,
    ) -> tuple[int, int, float | Literal["keyframes"] | None, bool]:
        """Resolves fallback tier, attempt count, target FPS, and exhaustion from an incomplete checkpoint.

        Args:
            job: Loaded incomplete AiJob checkpoint (response is None).
            fallback_enabled: Whether FPS fallback is enabled for this stage.
            max_runs: Maximum non-retryable attempts per FPS tier.

        Returns:
            tuple[int, int, float | Literal["keyframes"] | None, bool]:
                (fallback_tier, tier_attempts, target_fps, exhausted).
        """
        at_terminal_tier = (not fallback_enabled) or (job.target_fps == 0)
        was_exhausted = job.exhausted or (job.tier_attempts >= max_runs and at_terminal_tier)
        return job.fallback_tier, job.tier_attempts, job.target_fps, was_exhausted

    def _checkpoint_matches(
        self,
        job: AiJob | None,
        label: str,
        expected_model: str,
        expected_duration_ms: int,
        expected_offset_ms: int | None = None,
    ) -> bool:
        """Validates that a loaded YAML checkpoint matches the current model and media bounds.

        Args:
            job: Loaded checkpoint job instance, or None.
            label: Human-readable stage label for mismatch logs.
            expected_model: Configured Gemini model identifier for this stage.
            expected_duration_ms: Expected media or segment duration in milliseconds.
            expected_offset_ms: Expected segment start offset in milliseconds, if applicable. Defaults to None.

        Returns:
            bool: True if checkpoint exists and matches model and media bounds.
        """
        if job is None:
            return False
        if job.model_name != expected_model:
            logfire.info(
                f"Model mismatch for {label} (file: {job.model_name}, current: {expected_model}). Re-processing."
            )
            return False
        if job.media_tag != self.media_tag or job.video_duration_ms != expected_duration_ms:
            logfire.info(f"Source media mismatch for {label} checkpoint. Re-processing.")
            return False
        if expected_offset_ms is not None and getattr(job, "segment_offset_ms", None) != expected_offset_ms:
            logfire.info(f"Segment boundary mismatch for {label} checkpoint. Re-processing.")
            return False
        return True

    def _discover_on_disk_state(self) -> None:
        """Inspects job_dir for existing valid YAML checkpoints and restores unit states.

        Deliberately preserves existing checkpoints when only `user_context` or `agentic_video`
        changes between runs; invalidation is scoped to prompt versions, stage model names,
        source video identity/duration, segment boundaries, and upstream stage outputs.
        """
        summary_fallback_enabled = self.reencode_settings.summary.fallback.enabled
        segment_fallback_enabled = self.reencode_settings.segment.fallback.enabled
        summary_max_runs = self.get_retry_settings(TaskType.SUMMARY).maxruns
        subtitles_max_runs = self.get_retry_settings(TaskType.SUBTITLES).maxruns
        lyrics_max_runs = self.get_retry_settings(TaskType.LYRICS).maxruns

        if self.summary_enabled:
            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            loaded_summary = SummaryJob.load(summary_file, self.validation_buffer_ms)
            if self._checkpoint_matches(loaded_summary, "summary", self.summary_config.model, self.duration_ms):
                assert loaded_summary is not None
                if loaded_summary.response is not None:
                    self.summary.response = loaded_summary.response
                    self.summary.tier_attempts = loaded_summary.tier_attempts
                    self.summary.fallback_tier = loaded_summary.fallback_tier
                    self.summary.target_fps = loaded_summary.target_fps
                    logfire.debug(f"Loaded existing summary from {summary_file.name}")
                else:
                    tier, attempts, fps, exhausted = self._restore_failure_state(
                        loaded_summary, summary_fallback_enabled, summary_max_runs
                    )
                    self.summary.fallback_tier = tier
                    self.summary.tier_attempts = attempts
                    self.summary.target_fps = fps
                    self.summary.exhausted = exhausted
                    if exhausted:
                        logfire.warning("Summary stage previously exhausted all retries and fallback tiers.")

            # Downstream subtitles and lyrics are invalid until the current summary completes.
            if self.summary.response is None:
                return

        expected_summary_dump = _content_dump(self.summary.response) if self.summary_enabled else None

        for seg in self.segments:
            yaml_stem = self._segment_yaml_stem(seg)
            sub_file = self.job_dir / f"{yaml_stem}.subtitles.{self.subtitles_shortcode}.yaml"
            loaded_sub = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            if not self._checkpoint_matches(
                loaded_sub,
                f"{yaml_stem} subtitles",
                self.subtitles_config.model,
                seg.duration_ms,
                seg.offset_ms,
            ):
                continue
            assert loaded_sub is not None
            if self.summary_enabled and _content_dump(loaded_sub.global_summary) != expected_summary_dump:
                logfire.info(f"Upstream summary changed for {yaml_stem}. Re-processing subtitles.")
                continue

            if loaded_sub.response is not None:
                seg.subtitles_response = loaded_sub.response
                seg.subtitles_tier_attempts = loaded_sub.tier_attempts
                seg.fallback_tier = loaded_sub.fallback_tier
                seg.target_fps = loaded_sub.target_fps

                if self.segment_needs_lyrics(seg):
                    lyr_file = self.job_dir / f"{yaml_stem}.lyrics.{self.lyrics_shortcode}.yaml"
                    loaded_lyr = LyricsJob.load(lyr_file, self.validation_buffer_ms)
                    if not self._checkpoint_matches(
                        loaded_lyr,
                        f"{yaml_stem} lyrics",
                        self.lyrics_config.model,
                        seg.duration_ms,
                        seg.offset_ms,
                    ):
                        continue
                    assert loaded_lyr is not None
                    if (
                        loaded_lyr.draft_subtitles != loaded_sub.response.subtitles
                        or loaded_lyr.detected_performances != loaded_sub.response.detected_musical_performances
                        or (self.summary_enabled and _content_dump(loaded_lyr.global_summary) != expected_summary_dump)
                    ):
                        logfire.info(f"Upstream subtitles or summary changed for {yaml_stem}. Re-processing lyrics.")
                        continue

                    if loaded_lyr.response is not None:
                        seg.lyrics_response = loaded_lyr.response
                        seg.lyrics_tier_attempts = loaded_lyr.tier_attempts
                    else:
                        _, attempts, _, exhausted = self._restore_failure_state(loaded_lyr, False, lyrics_max_runs)
                        seg.lyrics_tier_attempts = attempts
                        seg.lyrics_exhausted = exhausted
            else:
                tier, attempts, fps, exhausted = self._restore_failure_state(
                    loaded_sub, segment_fallback_enabled, subtitles_max_runs
                )
                seg.fallback_tier = tier
                seg.target_fps = fps
                seg.subtitles_tier_attempts = attempts
                seg.subtitles_exhausted = exhausted

    def _is_segment_ai_concluded(self, seg: SegmentState) -> bool:
        """Checks whether all AI tasks required for a segment have completed or exhausted.

        Args:
            seg: Target segment state.

        Returns:
            bool: True if no further AI processing is needed for this segment.
        """
        return self.segment_subtitles_concluded(seg) and self.segment_lyrics_concluded(seg)

    async def _link_raw_summary_proxy(self) -> Path:
        """Links or copies the raw video to a managed 'summary_<tag>' path, verifying file identity.

        Returns:
            Path: Path to the validated linked or copied summary proxy file.
        """
        proxy_path = self.job_dir / f"summary_{self.media_tag}{self.raw_video_path.suffix}"
        sidecar_path = proxy_path.with_name(f".{proxy_path.name}.sha256")

        if proxy_path.is_symlink() and not proxy_path.exists():
            proxy_path.unlink(missing_ok=True)
        elif proxy_path.is_file():
            is_same = False
            try:
                is_same = os.path.samefile(proxy_path, self.raw_video_path)
            except OSError:
                is_same = False

            if is_same:
                sidecar_path.unlink(missing_ok=True)
                return proxy_path

            if sidecar_path.is_file() and proxy_path.stat().st_size == self.raw_video_path.stat().st_size:
                try:
                    recorded_hash = sidecar_path.read_text(encoding="utf-8").strip()
                    raw_hash, proxy_hash = await asyncio.gather(
                        asyncio.to_thread(calculate_sha256_hex, self.raw_video_path),
                        asyncio.to_thread(calculate_sha256_hex, proxy_path),
                    )
                    if recorded_hash and recorded_hash == raw_hash and proxy_hash == raw_hash:
                        return proxy_path
                except OSError:
                    pass

            proxy_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)

        try:
            os.link(self.raw_video_path, proxy_path)
        except OSError:
            try:
                os.symlink(self.raw_video_path, proxy_path)
            except OSError:
                raw_hash = await asyncio.to_thread(calculate_sha256_hex, self.raw_video_path)
                await asyncio.to_thread(shutil.copyfile, self.raw_video_path, proxy_path)
                atomic_write_text(sidecar_path, raw_hash)

        return proxy_path

    async def _prepare_media_states(self) -> None:
        """Evaluates media compression needs and validates existing encoded files against target FPS."""

        async def _check_stage_media(
            state: MediaStageState,
            cfg: StageReencodeSettings,
            source_path: Path,
            seg: SegmentState | None = None,
        ) -> None:
            is_summary = seg is None
            desired_fps = _resolve_target_fps(state.target_fps, state.fallback_tier, cfg, source_path)
            if desired_fps == 0 and state.fallback_tier == 0 and not await has_audio_stream(source_path):
                logfire.warning(
                    f"Cannot use 0 fps (audio-only) for {source_path.name}: no audio stream found. Falling back to raw."
                )
                desired_fps = None

            state.target_fps = desired_fps
            if desired_fps is None:
                state.media_path = await self._link_raw_summary_proxy() if is_summary else source_path
                state.media_ready = True
                return

            encoded_path = self._reencoded_output_path(desired_fps, seg)
            if await is_valid_reencoded_file(
                input_path=source_path,
                output_path=encoded_path,
                fps=desired_fps,
                duration_tolerance_ms=int(cfg.tolerance.total_seconds() * 1000),
            ):
                state.media_path = encoded_path
                state.media_ready = True
            else:
                state.media_path = None
                state.media_ready = False

        if self.summary_enabled:
            if self.summary.response is not None or self.summary.exhausted:
                self.summary.media_ready = True
            else:
                await _check_stage_media(
                    self.summary,
                    self.reencode_settings.summary,
                    self.raw_video_path,
                )

        seg_cfg = self.reencode_settings.segment
        probe_sem = asyncio.Semaphore(8)

        async def _check_segment(seg: SegmentState) -> None:
            async with probe_sem:
                if self._is_segment_ai_concluded(seg):
                    out_p = self._reencoded_output_path(seg.target_fps, seg)
                    seg.media_path = out_p if out_p.is_file() else seg.raw_path
                    seg.media_ready = True
                    return
                await _check_stage_media(seg, seg_cfg, seg.raw_path, seg=seg)

        await asyncio.gather(*(_check_segment(seg) for seg in self.segments))

    def is_summary_complete(self) -> bool:
        """Checks if the whole-video summary stage is complete or disabled.

        Returns:
            bool: True if summary is complete or disabled, False otherwise.
        """
        if not self.summary_enabled:
            return True
        return self.summary.response is not None

    def is_done(self) -> bool:
        """Checks if the entire video has completed all stages or aborted on summary exhaustion.

        Returns:
            bool: True if pipeline execution has concluded, False otherwise.
        """
        if self.summary_aborted:
            return True
        return self.is_summary_complete() and all(self._is_segment_ai_concluded(seg) for seg in self.segments)

    def has_failures(self) -> bool:
        """Checks whether any stage in the pipeline permanently exhausted retries.

        Returns:
            bool: True if summary, any subtitle segment, or any lyrics segment exhausted retries.
        """
        if self.summary_aborted:
            return True
        for seg in self.segments:
            if seg.subtitles_exhausted:
                return True
            if self.lyrics_enabled and seg.lyrics_exhausted:
                return True
        return False

    def get_job_media_path(self, job: BaseJob) -> Path:
        """Resolves the active media file path for an executable AI job.

        Args:
            job: Job instance returned by get_pending_jobs().

        Returns:
            Path: Compressed proxy/segment path when ready, or raw media fallback.
        """
        if isinstance(job, SummaryJob):
            return self.summary.media_path or self.raw_video_path
        if isinstance(job, (SubtitlesJob, LyricsJob)) and job.segment_index is not None:
            seg = self.segments[job.segment_index]
            return seg.media_path or seg.raw_path
        return self.raw_video_path

    def get_fallback_probe_spec(
        self,
        task_type: TaskType,
        segment_index: int | None = None,
    ) -> FallbackProbeSpec | None:
        """Builds an off-lock FFprobe spec if the next failure triggers keyframe or audio checks.

        Args:
            task_type: AI stage about to record a non-retryable failure.
            segment_index: Zero-based segment index for segment stages. Defaults to None.

        Returns:
            FallbackProbeSpec | None: Probe specification to execute outside coordinator locks, or None.
        """
        if task_type == TaskType.SUMMARY:
            seg = None
        elif task_type == TaskType.SUBTITLES and segment_index is not None:
            seg = self.segments[segment_index]
        else:
            return None

        media_state, cfg, source_path, _, attempts, concluded = self._fallback_stage_context(task_type, seg)
        max_runs = self.get_retry_settings(task_type).maxruns
        if concluded or attempts + 1 < max_runs or not cfg.fallback.enabled or media_state.target_fps == 0:
            return None

        need_kf = media_state.fallback_tier == 0 and cfg.fps == "keyframes"
        if need_kf:
            keyframe_media = (
                media_state.media_path
                if media_state.target_fps == "keyframes" and media_state.media_path is not None
                else None
            )
            return FallbackProbeSpec(
                source_path=source_path,
                keyframe_media=keyframe_media,
                probe_keyframe_fps=True,
                probe_audio=True,
            )

        if media_state.fallback_tier > 0 and isinstance(media_state.target_fps, (int, float)):
            known_base = float(media_state.target_fps)
        else:
            known_base = float(cfg.fps)

        if (known_base / float(cfg.fallback.divisor)) < cfg.fallback.minfps:
            return FallbackProbeSpec(source_path=source_path, probe_audio=True)

        return None

    def _build_reencode_job(
        self,
        *,
        name: str,
        input_file: Path,
        output_file: Path,
        media_state: MediaStageState,
        cfg: StageReencodeSettings,
        is_summary_proxy: bool,
        segment_index: int | None = None,
    ) -> ReencodeJob:
        """Constructs a ReencodeJob from stage media state and re-encode settings.

        Args:
            name: Unique job identifier.
            input_file: Source raw video or segment path.
            output_file: Destination path for compressed output.
            media_state: Summary or segment media stage state.
            cfg: Re-encoding settings for this stage.
            is_summary_proxy: Whether this job encodes the summary proxy.
            segment_index: Zero-based segment index when encoding a segment. Defaults to None.

        Returns:
            ReencodeJob: Configured FFmpeg re-encoding job.
        """
        assert media_state.target_fps is not None
        return ReencodeJob(
            name=name,
            input_file=input_file,
            output_file=output_file,
            fps=media_state.target_fps,
            height=cfg.height,
            bitrate=int(cfg.bitrate),
            duration_tolerance_ms=int(cfg.tolerance.total_seconds() * 1000),
            encoder=self.reencode_settings.encoder,
            keyframe_only=cfg.fps == "keyframes",
            segment_index=segment_index,
            is_summary_proxy=is_summary_proxy,
            fallback_tier=media_state.fallback_tier,
        )

    def _build_ai_job(self, task_type: TaskType, seg: SegmentState | None = None) -> AiJob:
        """Constructs a typed AiJob reflecting current pipeline state and stage configuration.

        Args:
            task_type: AI stage to construct (SUMMARY, SUBTITLES, or LYRICS).
            seg: SegmentState required when task_type is SUBTITLES or LYRICS. Defaults to None.

        Returns:
            AiJob: Configured SummaryJob, SubtitlesJob, or LyricsJob instance.
        """
        if task_type == TaskType.SUMMARY:
            return SummaryJob(
                name="summary",
                video_duration_ms=self.duration_ms,
                media_tag=self.media_tag,
                user_context=self.user_context,
                model_name=self.summary_config.model,
                service_tier=self.summary_config.tier,
                agentic_video=self.summary_config.agentic,
                tier_attempts=self.summary.tier_attempts,
                fallback_tier=self.summary.fallback_tier,
                target_fps=self.summary.target_fps,
                exhausted=self.summary.exhausted,
                response=self.summary.response,
            )

        assert seg is not None
        common_seg: dict[str, Any] = {
            "name": self._segment_yaml_stem(seg),
            "video_duration_ms": seg.duration_ms,
            "media_tag": self.media_tag,
            "segment_offset_ms": seg.offset_ms,
            "global_summary": self.summary.response,
            "user_context": self.user_context,
            "segment_index": seg.index,
            "fallback_tier": seg.fallback_tier,
            "target_fps": seg.target_fps,
        }

        if task_type == TaskType.SUBTITLES:
            return SubtitlesJob(
                model_name=self.subtitles_config.model,
                service_tier=self.subtitles_config.tier,
                agentic_video=self.subtitles_config.agentic,
                tier_attempts=seg.subtitles_tier_attempts,
                exhausted=seg.subtitles_exhausted,
                response=seg.subtitles_response,
                **common_seg,
            )

        sub_resp = seg.subtitles_response
        return LyricsJob(
            draft_subtitles=sub_resp.subtitles if sub_resp else [],
            detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
            model_name=self.lyrics_config.model,
            service_tier=self.lyrics_config.tier,
            agentic_video=self.lyrics_config.agentic,
            tier_attempts=seg.lyrics_tier_attempts,
            exhausted=seg.lyrics_exhausted,
            response=seg.lyrics_response,
            **common_seg,
        )

    def get_pending_jobs(self) -> list[BaseJob]:
        """Evaluates DAG dependencies and returns all jobs currently ready to execute.

        Summary re-encoding is emitted first, followed by segment re-encodes in segment order.
        AI jobs for any segment emit independently as soon as summary is complete and that
        segment's media is ready.

        Returns:
            list[BaseJob]: Ordered list of ready ReencodeJob, SummaryJob, SubtitlesJob, and LyricsJob items.
        """
        if self.summary_aborted:
            return []

        jobs: list[BaseJob] = []

        # 1. Summary proxy re-encode job (Priority 1)
        if (
            self.summary_enabled
            and self.summary.response is None
            and not self.summary.exhausted
            and not self.summary.media_ready
            and self.summary.target_fps is not None
        ):
            jobs.append(
                self._build_reencode_job(
                    name="summary_proxy",
                    input_file=self.raw_video_path,
                    output_file=self._reencoded_output_path(self.summary.target_fps),
                    media_state=self.summary,
                    cfg=self.reencode_settings.summary,
                    is_summary_proxy=True,
                )
            )

        # 2. Segment re-encode jobs (Priority 2, normal segment index order)
        seg_cfg = self.reencode_settings.segment
        for seg in self.segments:
            if not self._is_segment_ai_concluded(seg) and not seg.media_ready and seg.target_fps is not None:
                jobs.append(
                    self._build_reencode_job(
                        name=f"{self._segment_yaml_stem(seg)}_reencode",
                        input_file=seg.raw_path,
                        output_file=self._reencoded_output_path(seg.target_fps, seg),
                        media_state=seg,
                        cfg=seg_cfg,
                        is_summary_proxy=False,
                        segment_index=seg.index,
                    )
                )

        # 3. Whole-video Summary AI job
        if self.summary_enabled and self.summary.response is None and not self.summary.exhausted:
            if self.summary.media_ready and self.summary.media_path is not None:
                jobs.append(self._build_ai_job(TaskType.SUMMARY))
            return jobs

        # 4. Per-segment Subtitles and Lyrics AI jobs (independent per segment once summary completes)
        for seg in self.segments:
            if not seg.media_ready or seg.media_path is None:
                continue

            if seg.subtitles_response is None and not seg.subtitles_exhausted:
                jobs.append(self._build_ai_job(TaskType.SUBTITLES, seg))
            elif self.segment_needs_lyrics(seg) and seg.lyrics_response is None and not seg.lyrics_exhausted:
                jobs.append(self._build_ai_job(TaskType.LYRICS, seg))

        return jobs

    def _save_checkpoint(self, task_type: TaskType, seg: SegmentState | None = None) -> None:
        """Persists the current state of an AI stage into its YAML checkpoint file.

        Args:
            task_type: AI stage to persist (SUMMARY, SUBTITLES, or LYRICS).
            seg: SegmentState required when task_type is SUBTITLES or LYRICS. Defaults to None.
        """
        job = self._build_ai_job(task_type, seg)
        if task_type == TaskType.SUMMARY:
            job.save(self.job_dir / f"summary.{self.summary_shortcode}.yaml")
            return

        assert seg is not None
        yaml_stem = self._segment_yaml_stem(seg)
        if task_type == TaskType.SUBTITLES:
            job.save(self.job_dir / f"{yaml_stem}.subtitles.{self.subtitles_shortcode}.yaml")
        elif task_type == TaskType.LYRICS:
            job.save(self.job_dir / f"{yaml_stem}.lyrics.{self.lyrics_shortcode}.yaml")

    async def _advance_or_exhaust_stage(
        self,
        task_type: TaskType,
        seg: SegmentState | None = None,
        log_context: str | None = None,
        fallback_base_fps: float | None = None,
        has_audio: bool | None = None,
    ) -> FailureOutcome:
        """Advances a stage to the next fallback FPS tier (down to 0 fps audio-only) or marks it exhausted.

        Args:
            task_type: Target AI stage (SUMMARY or SUBTITLES).
            seg: Target segment state when task_type is SUBTITLES. Defaults to None.
            log_context: Optional failure threshold description to log. Defaults to None.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.
            has_audio: Optional pre-probed audio-stream presence flag. Defaults to None.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        media_state, cfg, source_path, label, tier_attempts, _ = self._fallback_stage_context(task_type, seg)
        is_summary = task_type == TaskType.SUMMARY
        current_tier = media_state.fallback_tier
        current_target = media_state.target_fps
        existing_media = media_state.media_path
        max_runs = self.get_retry_settings(task_type).maxruns

        next_fps: float | None = None
        if cfg.fallback.enabled and current_target != 0:
            next_fps = await _compute_next_fallback_fps(
                cfg=cfg,
                current_tier=current_tier,
                current_target=current_target,
                existing_media=existing_media,
                source_path=source_path,
                probed_base_fps=fallback_base_fps,
                has_audio=has_audio,
            )

        if next_fps is not None:
            # Remove stale compressed output so the next fallback tier always encodes cleanly
            if (
                existing_media is not None
                and existing_media != source_path
                and existing_media.name.startswith("reencoded_")
                and existing_media.is_file()
            ):
                try:
                    existing_media.unlink()
                except OSError:
                    pass

            next_tier = current_tier + 1
            media_state.fallback_tier = next_tier
            media_state.target_fps = next_fps
            media_state.media_path = None
            media_state.media_ready = False
            if is_summary:
                self.summary.tier_attempts = 0
            else:
                assert seg is not None
                seg.subtitles_tier_attempts = 0

            self._save_checkpoint(task_type, seg)
            if log_context:
                target_desc = "0 fps (audio-only)" if next_fps == 0 else f"{next_fps:.4g} fps"
                logfire.warning(f"{label} {log_context}; falling back to {target_desc} (tier {next_tier}).")
            return FailureOutcome(
                action="fallback_reencode",
                fallback_tier=next_tier,
                target_fps=next_fps,
                tier_attempts=0,
            )

        if is_summary:
            self.summary.exhausted = True
        else:
            assert seg is not None
            seg.subtitles_exhausted = True

        self._save_checkpoint(task_type, seg)
        if log_context:
            suffix = " Aborting." if is_summary else ""
            logfire.error(f"{label} exhausted all {max_runs} run(s) across all FPS tiers.{suffix}")
        return FailureOutcome(
            action="exhausted",
            fallback_tier=current_tier,
            target_fps=_float_fps(current_target),
            tier_attempts=tier_attempts,
        )

    def complete_reencode(self, job: ReencodeJob, media_path: Path) -> None:
        """Marks a summary proxy or segment re-encode job complete and unlocks downstream AI work.

        Args:
            job: Completed ReencodeJob instance.
            media_path: Resulting media file path.
        """
        media_state: MediaStageState | None = (
            self.summary
            if job.is_summary_proxy
            else (self.segments[job.segment_index] if job.segment_index is not None else None)
        )
        if media_state is not None:
            media_state.media_path = media_path
            media_state.media_ready = True

    async def fail_reencode(self, job: ReencodeJob) -> FailureOutcome:
        """Handles a failed FFmpeg re-encode job.

        On initial encoding (fallback_tier == 0), falls back safely to the raw input file.
        On fallback encoding (fallback_tier > 0), advances to the next lower FPS tier if available
        or marks the unit exhausted without reusing previously failed media.

        Args:
            job: Failed ReencodeJob instance.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        if job.is_summary_proxy:
            media_state: MediaStageState = self.summary
            task_type = TaskType.SUMMARY
            seg: SegmentState | None = None
        elif job.segment_index is not None:
            seg = self.segments[job.segment_index]
            media_state = seg
            task_type = TaskType.SUBTITLES
        else:
            return FailureOutcome(action="ignored")

        if job.fallback_tier == 0:
            media_state.media_path = await self._link_raw_summary_proxy() if job.is_summary_proxy else job.input_file
            media_state.media_ready = True
            media_state.target_fps = None
            return FailureOutcome(action="retry")

        return await self._advance_or_exhaust_stage(task_type, seg=seg)

    def _commit_stage_checkpoint(
        self,
        state: SummaryState | SegmentState,
        resp_attr: str,
        exhausted_attr: str,
        resp: BaseModel,
        task_type: TaskType,
        seg: SegmentState | None = None,
    ) -> None:
        """Commits a validated AI response and persists its YAML checkpoint with rollback on failure.

        Args:
            state: Target SummaryState or SegmentState instance.
            resp_attr: Attribute name holding the response object.
            exhausted_attr: Attribute name holding the exhausted flag.
            resp: Validated AI response model.
            task_type: Completed AI stage type.
            seg: Segment state for segment stages. Defaults to None.
        """
        prev = (getattr(state, resp_attr), getattr(state, exhausted_attr))
        setattr(state, resp_attr, resp)
        setattr(state, exhausted_attr, False)
        try:
            self._save_checkpoint(task_type, seg)
        except Exception:
            setattr(state, resp_attr, prev[0])
            setattr(state, exhausted_attr, prev[1])
            raise

    def complete_ai_task(
        self,
        task_type: TaskType,
        result_data: Any,
        segment_index: int | None = None,
    ) -> None:
        """Validates AI output, persists YAML checkpoints atomically, and advances unit state.

        Restores previous in-memory fields if saving the checkpoint fails so state remains
        consistent and completion stays retryable.

        Args:
            task_type: Completed AI stage type (SUMMARY, SUBTITLES, or LYRICS).
            result_data: Output Pydantic model or dictionary returned by the executor/worker.
            segment_index: Zero-based segment index for SUBTITLES or LYRICS. Defaults to None.

        Raises:
            ValueError: If segment_index is missing or task_type is unsupported.
        """
        val_ctx: dict[str, Any] = {"validation_buffer_ms": self.validation_buffer_ms}

        if task_type == TaskType.SUMMARY:
            val_ctx["video_duration_ms"] = self.duration_ms
            resp_cls: type[BaseModel] = SummaryAiResponse
            state: SummaryState | SegmentState = self.summary
            resp_attr, exhausted_attr = "response", "exhausted"
            seg = None
        else:
            if segment_index is None:
                raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")
            seg = self.segments[segment_index]
            state = seg
            val_ctx["video_duration_ms"] = seg.duration_ms
            if task_type == TaskType.SUBTITLES:
                resp_cls = SubtitlesAiResponse
                resp_attr, exhausted_attr = "subtitles_response", "subtitles_exhausted"
            elif task_type == TaskType.LYRICS:
                val_ctx["require_subtitles"] = bool(seg.subtitles_response and seg.subtitles_response.subtitles)
                resp_cls = LyricsAiResponse
                resp_attr, exhausted_attr = "lyrics_response", "lyrics_exhausted"
            else:
                raise ValueError(f"Unsupported task_type: '{task_type.value}'.")

        resp = (
            result_data if isinstance(result_data, resp_cls) else resp_cls.model_validate(result_data, context=val_ctx)
        )
        self._commit_stage_checkpoint(state, resp_attr, exhausted_attr, resp, task_type, seg)

    def complete_job(self, job: BaseJob, result_data: Any) -> None:
        """Completes a typed Job instance returned by get_pending_jobs().

        Args:
            job: Executed BaseJob subclass instance.
            result_data: Resulting Path (for ReencodeJob) or AI response payload.

        Raises:
            TypeError: If job is an unsupported BaseJob subclass.
        """
        if isinstance(job, ReencodeJob):
            self.complete_reencode(job, Path(result_data))
        elif isinstance(job, AiJob):
            self.complete_ai_task(job.task_type, result_data, segment_index=job.segment_index)
        else:
            raise TypeError(f"Unsupported job type: {type(job).__name__}")

    async def record_job_failure(self, job: BaseJob) -> FailureOutcome:
        """Records a non-retryable or re-encode failure for a typed Job instance.

        Args:
            job: Failed BaseJob subclass instance.

        Returns:
            FailureOutcome: Resulting state transition.

        Raises:
            TypeError: If job is an unsupported BaseJob subclass.
        """
        if isinstance(job, ReencodeJob):
            return await self.fail_reencode(job)
        if isinstance(job, AiJob):
            return await self.record_failure(job.task_type, segment_index=job.segment_index)
        raise TypeError(f"Unsupported job type: {type(job).__name__}")

    async def record_failure(
        self,
        task_type: TaskType,
        segment_index: int | None = None,
        fallback_base_fps: float | None = None,
        has_audio: bool | None = None,
    ) -> FailureOutcome:
        """Records a non-retryable AI failure, advancing to fallback FPS or marking exhausted.

        Args:
            task_type: AI stage that encountered a non-retryable failure.
            segment_index: Zero-based segment index for SUBTITLES or LYRICS. Defaults to None.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.
            has_audio: Optional pre-probed audio-stream presence flag. Defaults to None.

        Returns:
            FailureOutcome: State transition indicating whether to retry, re-encode at lower FPS, or exhaust.

        Raises:
            ValueError: If segment_index is missing for a segment task.
        """
        max_runs = self.get_retry_settings(task_type).maxruns
        seg: SegmentState | None = None
        if task_type in (TaskType.SUBTITLES, TaskType.LYRICS):
            if segment_index is None:
                raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")
            seg = self.segments[segment_index]

        if task_type == TaskType.LYRICS:
            assert seg is not None
            if seg.lyrics_response is not None or seg.lyrics_exhausted:
                return FailureOutcome(action="ignored")
            seg.lyrics_tier_attempts += 1
            target_fps = _float_fps(seg.target_fps)
            if seg.lyrics_tier_attempts >= max_runs:
                seg.lyrics_exhausted = True
                self._save_checkpoint(TaskType.LYRICS, seg)
                logfire.error(
                    f"Lyrics for '{self._segment_yaml_stem(seg)}' exhausted all {max_runs} run(s); "
                    "emitting error in refined SRT while preserving raw subtitles."
                )
                return FailureOutcome(
                    action="exhausted",
                    fallback_tier=seg.fallback_tier,
                    target_fps=target_fps,
                    tier_attempts=seg.lyrics_tier_attempts,
                )
            self._save_checkpoint(TaskType.LYRICS, seg)
            return FailureOutcome(
                action="retry",
                fallback_tier=seg.fallback_tier,
                target_fps=target_fps,
                tier_attempts=seg.lyrics_tier_attempts,
            )

        if task_type not in (TaskType.SUMMARY, TaskType.SUBTITLES):
            return FailureOutcome(action="ignored")

        media_state, _, _, _, prev_attempts, concluded = self._fallback_stage_context(task_type, seg)
        if concluded:
            return FailureOutcome(action="ignored")

        tier_attempts = prev_attempts + 1
        if task_type == TaskType.SUMMARY:
            self.summary.tier_attempts = tier_attempts
        else:
            assert seg is not None
            seg.subtitles_tier_attempts = tier_attempts

        if tier_attempts >= max_runs:
            return await self._advance_or_exhaust_stage(
                task_type,
                seg=seg,
                log_context=f"reached {max_runs} non-retryable failure(s)",
                fallback_base_fps=fallback_base_fps,
                has_audio=has_audio,
            )

        self._save_checkpoint(task_type, seg)
        return FailureOutcome(
            action="retry",
            fallback_tier=media_state.fallback_tier,
            target_fps=_float_fps(media_state.target_fps),
            tier_attempts=tier_attempts,
        )

    def _derive_status(self) -> str:
        """Derives completion status based on session progress and recorded failures.

        Returns:
            str: 'RetriesExceeded' if any stage failed, 'Completed' if finished, or 'InProgress'.
        """
        if self.has_failures():
            return "RetriesExceeded"
        return "Completed" if self.is_done() else "InProgress"

    def _build_settings_metadata(self) -> dict[str, Any]:
        """Constructs settings metadata dictionary for embedding into subtitle Event 0 comments.

        Returns:
            dict[str, Any]: Settings dictionary.
        """
        if self.search_settings is not None:
            search_dump: dict[str, Any] = self.search_settings.model_dump(
                mode="json",
                exclude={"tinyfishkey": True, "langsearchkey": True, "ollamakey": True},
            )
        else:
            search_dump = {"tools": self.search_tools}

        retry_exclude = None if self.retry_settings.models else {"models": True}

        return {
            "ai": {
                "model_summary": self.summary_config.model if self.summary_enabled else None,
                "model_subtitles": self.subtitles_config.model,
                "model_lyrics": self.lyrics_config.model if self.lyrics_enabled else None,
                "service_tier_summary": self.summary_config.tier if self.summary_enabled else None,
                "service_tier_subtitles": self.subtitles_config.tier,
                "service_tier_lyrics": self.lyrics_config.tier if self.lyrics_enabled else None,
                "search": search_dump,
                "agentic_video_summary": self.summary_config.agentic if self.summary_enabled else False,
                "agentic_video_subtitles": self.subtitles_config.agentic,
                "agentic_video_lyrics": self.lyrics_config.agentic if self.lyrics_enabled else False,
                "validation_buffer_ms": self.validation_buffer_ms,
            },
            "reencode": self.reencode_settings.model_dump(mode="json"),
            "split": self.split_settings.model_dump(mode="json"),
            "retry": self.retry_settings.model_dump(mode="json", exclude=retry_exclude),
            "context": self.user_context,
        }

    def write_nfo(
        self,
        output_dir: Path | None = None,
        filename_stem: str | None = None,
    ) -> Path | None:
        """Writes a Jellyfin-compatible .nfo metadata file from the Phase 1 global summary.

        Args:
            output_dir: Destination directory for the .nfo file. Defaults to job_dir.
            filename_stem: Base filename stem for the .nfo output. Defaults to self.filename_stem.

        Returns:
            Path | None: Path to the written .nfo file, or None if summary data is unavailable.
        """
        if self.summary.response is None:
            logfire.warning(f"Skipping .nfo generation for '{self.video_id}': global summary is unavailable.")
            return None

        target_dir = output_dir if output_dir is not None else self.job_dir
        stem = filename_stem if filename_stem is not None else self.filename_stem
        nfo_path = target_dir / f"{stem}.nfo"

        xml_content = _build_nfo_xml(
            summary=self.summary.response,
            title=self.title if self.source_type == "youtube" else None,
        )
        atomic_write_text(nfo_path, xml_content)
        return nfo_path

    def finalize(
        self,
        output_dir: Path | None = None,
        filename_stem: str | None = None,
        allow_partial: bool = False,
    ) -> list[Path]:
        """Stitches all segment responses into synchronized SRT files with timeline offset shifting.

        When lyrics refinement is required but fails or is unavailable, the refined output emits
        an error placeholder for affected segments while still writing the raw subtitle track so
        users can switch tracks in the player.

        Args:
            output_dir: Destination directory for generated subtitle files. Defaults to job_dir.
            filename_stem: Base filename stem for SRT outputs. Defaults to self.filename_stem.
            allow_partial: Whether to stitch partial subtitles when processing is incomplete. Defaults to False.

        Returns:
            list[Path]: List of generated subtitle file paths.

        Raises:
            RuntimeError: If video processing is incomplete and allow_partial is False.
        """
        if not self.is_done() and not allow_partial:
            raise RuntimeError(f"Cannot finalize subtitles: processing for '{self.video_id}' is incomplete.")

        target_dir = output_dir if output_dir is not None else self.job_dir
        stem = filename_stem if filename_stem is not None else self.filename_stem
        primary_path = target_dir / f"{stem}.{self.full_shortcode}.srt"
        raw_path = target_dir / f"{stem}.raw-{self.full_shortcode}.srt" if self.lyrics_enabled else None

        all_subs = SSAFile()
        raw_subs = SSAFile() if raw_path is not None else None
        lyrics_needed_any = False

        for seg in self.segments:
            sub_resp = seg.subtitles_response
            lyr_resp = seg.lyrics_response
            lyrics_required = self.segment_needs_lyrics(seg)

            if lyrics_required:
                lyrics_needed_any = True
                active_resp: SubtitlesAiResponse | LyricsAiResponse | None = (
                    lyr_resp if (lyr_resp is not None and lyr_resp.subtitles) else None
                )
                _append_segment_events(
                    all_subs,
                    active_resp,
                    seg,
                    error_text="Error refining lyrics for this segment.",
                )
            else:
                _append_segment_events(all_subs, sub_resp, seg)

            if raw_subs is not None:
                _append_segment_events(raw_subs, sub_resp, seg)

        metadata = {
            "ai_sub_version": get_app_version(),
            "generated_at": datetime.now().isoformat(),
            "summary_prompt_version": (
                SUMMARY_PROMPT_VERSION if self.summary_enabled and self.summary.response is not None else None
            ),
            "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
            "lyrics_prompt_version": LYRICS_PROMPT_VERSION if self.lyrics_enabled else None,
            "status": self._derive_status(),
            "settings": self._build_settings_metadata(),
        }

        generated_files: list[Path] = []
        _apply_metadata_header(all_subs, metadata)
        target_dir.mkdir(parents=True, exist_ok=True)
        all_subs.save(str(primary_path))
        generated_files.append(primary_path)

        if lyrics_needed_any and raw_subs is not None and raw_path is not None:
            _apply_metadata_header(raw_subs, metadata)
            raw_subs.save(str(raw_path))
            generated_files.append(raw_path)

        return generated_files
