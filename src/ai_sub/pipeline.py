"""Unified video processing pipeline managing segmentation, compression, AI DAG stages, and stitching."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import logfire
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat
from pysubs2 import SSAEvent, SSAFile

from ai_sub.config import (
    ReencodeSettings,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
    StageExecutionConfig,
    StageReencodeSettings,
)
from ai_sub.jobs.job_base import AiJob, BaseJob, TaskType
from ai_sub.jobs.lyrics import LYRICS_PROMPT_VERSION, LyricsAiResponse, LyricsJob
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import SUBTITLES_PROMPT_VERSION, SubtitlesAiResponse, SubtitlesJob
from ai_sub.jobs.summary import SUMMARY_PROMPT_VERSION, SummaryAiResponse, SummaryJob
from ai_sub.shortcode import generate_full_shortcode, generate_model_shortcode
from ai_sub.telemetry import get_app_version
from ai_sub.video import (
    compute_file_id_hash,
    get_keyframe_fps,
    get_video_duration_ms,
    get_video_fps,
    is_valid_reencoded_file,
    sanitize_filename,
    split_video,
)


@dataclass(frozen=True)
class FailureOutcome:
    """Result returned when recording a non-retryable task or re-encode failure."""

    action: Literal["retry", "fallback_reencode", "exhausted", "ignored"]
    fallback_tier: int = 0
    target_fps: float | None = None
    tier_attempts: int = 0


class MediaStageState(BaseModel):
    """Shared media preparation and FPS fallback state for a summary or segment unit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    media_ready: bool = Field(default=False, description="Whether media is ready for AI inference.")
    media_path: Path | None = Field(default=None, description="Path to active compressed or raw media.")
    target_fps: PositiveFloat | Literal["keyframes"] | None = Field(
        default=None,
        description="Target FPS or 'keyframes' for re-encoding, or None if using raw media.",
    )
    fallback_tier: NonNegativeInt = Field(default=0, description="Active fallback FPS tier index.")


class SummaryState(MediaStageState):
    """Per-unit execution and media state for the Phase 1 whole-video summary."""

    tier_attempts: NonNegativeInt = Field(default=0, description="Non-retryable failures at the current FPS tier.")
    total_attempts: NonNegativeInt = Field(default=0, description="Cumulative attempts across all executions.")
    response: SummaryAiResponse | None = Field(default=None, description="Completed summary response.")
    exhausted: bool = Field(default=False, description="Whether summary exhausted all retries and fallback tiers.")


class SegmentState(MediaStageState):
    """Per-unit execution, media, and AI stage state for an individual video chunk."""

    index: int = Field(description="Zero-based segment index.")
    raw_path: Path = Field(description="Path to the split uncompressed segment chunk.")
    duration_ms: int = Field(description="Duration of this segment in milliseconds.")
    offset_ms: int = Field(description="Absolute start offset of this segment in milliseconds.")

    subtitles_tier_attempts: NonNegativeInt = Field(
        default=0,
        description="Non-retryable subtitle failures at the current FPS tier.",
    )
    subtitles_total_attempts: NonNegativeInt = Field(
        default=0,
        description="Cumulative subtitle attempts across all executions.",
    )
    subtitles_response: SubtitlesAiResponse | None = Field(default=None, description="Completed subtitles response.")
    subtitles_exhausted: bool = Field(
        default=False,
        description="Whether subtitles exhausted all retries and fallback tiers.",
    )

    lyrics_tier_attempts: NonNegativeInt = Field(default=0, description="Non-retryable lyrics failures.")
    lyrics_total_attempts: NonNegativeInt = Field(default=0, description="Cumulative lyrics attempts.")
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
) -> None:
    """Appends time-shifted subtitle events for a segment, or an error placeholder if missing.

    Args:
        target: Destination SSAFile accumulating stitched subtitle events.
        resp: Completed subtitle or lyrics response, or None if unavailable.
        seg: Segment metadata providing offset and duration.
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
                text="Error processing subtitles for this segment.",
            )
        )


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
        float | Literal["keyframes"] | None: Target FPS or 'keyframes' if re-encoding is needed, or None for raw.
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


async def _compute_next_fallback_fps(
    cfg: StageReencodeSettings,
    current_tier: int,
    current_target: float | Literal["keyframes"] | None,
    existing_media: Path | None,
    source_path: Path,
    probed_base_fps: float | None = None,
) -> float:
    """Computes the target FPS for the next fallback re-encoding tier.

    Args:
        cfg: Stage re-encoding settings containing base FPS and fallback multiplier.
        current_tier: Current fallback tier index before incrementing.
        current_target: Active FPS or 'keyframes' target at the current tier.
        existing_media: Compressed media path from the current tier, if present.
        source_path: Uncompressed source video path.
        probed_base_fps: Optional pre-probed base keyframe FPS computed outside locks. Defaults to None.

    Returns:
        float: Target FPS for the next fallback tier.
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

    return base_fps / float(cfg.fallback.multiplier)


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
            RuntimeError: If no video segments could be created.
            ValueError: If split offset exceeds the total video duration.
        """
        raw_resolved = raw_video_path.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        vid = video_id or raw_resolved.stem
        resolved_tag = media_tag or (vid if source_type == "youtube" else compute_file_id_hash(raw_resolved.name))
        default_stem = vid if source_type == "youtube" else raw_resolved.stem
        filename_stem = sanitize_filename(output_stem if output_stem is not None else default_stem)

        # 1. Probe raw video duration
        total_duration_ms = await get_video_duration_ms(raw_resolved)

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
        start_offset_s = int(split_settings.offset.total_seconds())
        if start_offset_s > 0:
            skip_count = start_offset_s // split_duration_s
            remainder_s = start_offset_s % split_duration_s
            if skip_count >= len(all_segments):
                raise ValueError(
                    f"split.offset ({start_offset_s}s) exceeds or equals total "
                    f"video duration ({total_duration_ms / 1000:.1f}s)."
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
            segments.append(
                SegmentState(
                    index=idx,
                    raw_path=seg_p,
                    duration_ms=seg_dur_ms,
                    offset_ms=current_offset_ms,
                )
            )
            current_offset_ms += seg_dur_ms

        # 5. Generate lineage shortcodes
        summary_shortcode = generate_model_shortcode(summary_config.model) if summary_enabled else ""
        subtitles_code = generate_model_shortcode(subtitles_config.model)
        subtitles_shortcode = f"{summary_shortcode}-{subtitles_code}" if summary_enabled else subtitles_code
        lyrics_shortcode = (
            f"{subtitles_shortcode}-{generate_model_shortcode(lyrics_config.model)}" if lyrics_enabled else ""
        )
        full_shortcode = generate_full_shortcode(
            subtitles_model=subtitles_config.model,
            summary_model=summary_config.model if summary_enabled else None,
            lyrics_model=lyrics_config.model if lyrics_enabled else None,
        )

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

    def _restore_failure_state(
        self,
        job: AiJob,
        fallback_attempts: int,
    ) -> tuple[int, int, float | Literal["keyframes"] | None, bool]:
        """Resolves fallback tier, attempt count, target FPS, and exhaustion from an incomplete checkpoint.

        Args:
            job: Loaded incomplete AiJob checkpoint (response is None).
            fallback_attempts: Configured fallback tier count for this stage.

        Returns:
            tuple[int, int, float | Literal["keyframes"] | None, bool]:
                (fallback_tier, tier_attempts, target_fps, exhausted).
        """
        was_exhausted = job.exhausted or (
            job.tier_attempts >= self.retry_settings.maxruns and job.fallback_tier >= fallback_attempts
        )
        return job.fallback_tier, job.tier_attempts, job.target_fps, was_exhausted

    def _discover_on_disk_state(self) -> None:
        """Inspects job_dir for existing valid YAML checkpoints and restores unit states.

        Deliberately preserves existing checkpoints when only `user_context` or `agentic_video`
        changes between runs; invalidation is scoped to prompt versions, stage model names,
        and upstream stage outputs.
        """
        summary_fallbacks = self.reencode_settings.summary.fallback.attempts
        segment_fallbacks = self.reencode_settings.segment.fallback.attempts

        if self.summary_enabled:
            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            loaded_summary = SummaryJob.load(summary_file, self.validation_buffer_ms)
            if loaded_summary and loaded_summary.model_name != self.summary_config.model:
                logfire.info(
                    f"Model mismatch for summary (file: {loaded_summary.model_name}, "
                    f"current: {self.summary_config.model}). Re-processing."
                )
                loaded_summary = None

            if loaded_summary:
                self.summary.total_attempts = loaded_summary.total_attempts
                if loaded_summary.response is not None:
                    self.summary.response = loaded_summary.response
                    self.summary.tier_attempts = loaded_summary.tier_attempts
                    self.summary.fallback_tier = loaded_summary.fallback_tier
                    self.summary.target_fps = loaded_summary.target_fps
                    logfire.debug(f"Loaded existing summary from {summary_file.name}")
                else:
                    tier, attempts, fps, exhausted = self._restore_failure_state(loaded_summary, summary_fallbacks)
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
            sub_file = self.job_dir / f"{seg.raw_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
            loaded_sub = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            if not loaded_sub:
                continue
            if loaded_sub.model_name != self.subtitles_config.model:
                logfire.info(
                    f"Model mismatch for {seg.raw_path.stem} subtitles "
                    f"(file: {loaded_sub.model_name}, current: {self.subtitles_config.model}). Re-processing."
                )
                continue
            if self.summary_enabled and _content_dump(loaded_sub.global_summary) != expected_summary_dump:
                logfire.info(f"Upstream summary changed for {seg.raw_path.stem}. Re-processing subtitles.")
                continue

            seg.subtitles_total_attempts = loaded_sub.total_attempts
            if loaded_sub.response is not None:
                seg.subtitles_response = loaded_sub.response
                seg.subtitles_tier_attempts = loaded_sub.tier_attempts
                seg.fallback_tier = loaded_sub.fallback_tier
                seg.target_fps = loaded_sub.target_fps

                if self.segment_needs_lyrics(seg):
                    lyr_file = self.job_dir / f"{seg.raw_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
                    loaded_lyr = LyricsJob.load(lyr_file, self.validation_buffer_ms)
                    if not loaded_lyr:
                        continue
                    if loaded_lyr.model_name != self.lyrics_config.model:
                        logfire.info(
                            f"Model mismatch for {seg.raw_path.stem} lyrics "
                            f"(file: {loaded_lyr.model_name}, current: {self.lyrics_config.model}). Re-processing."
                        )
                        continue
                    if (
                        loaded_lyr.draft_subtitles != loaded_sub.response.subtitles
                        or loaded_lyr.detected_performances != loaded_sub.response.detected_musical_performances
                        or (self.summary_enabled and _content_dump(loaded_lyr.global_summary) != expected_summary_dump)
                    ):
                        logfire.info(
                            f"Upstream subtitles or summary changed for {seg.raw_path.stem}. Re-processing lyrics."
                        )
                        continue

                    seg.lyrics_total_attempts = loaded_lyr.total_attempts
                    if loaded_lyr.response is not None:
                        seg.lyrics_response = loaded_lyr.response
                        seg.lyrics_tier_attempts = loaded_lyr.tier_attempts
                    else:
                        _, attempts, _, exhausted = self._restore_failure_state(loaded_lyr, 0)
                        seg.lyrics_tier_attempts = attempts
                        seg.lyrics_exhausted = exhausted
            else:
                tier, attempts, fps, exhausted = self._restore_failure_state(loaded_sub, segment_fallbacks)
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

    def _link_raw_summary_proxy(self) -> Path:
        """Links or copies the raw video to a managed 'summary_<tag>' path for Gemini Files cleanup.

        Returns:
            Path: Path to the linked or copied summary proxy file.
        """
        proxy_path = self.job_dir / f"summary_{self.media_tag}{self.raw_video_path.suffix}"
        if proxy_path.is_symlink() and not proxy_path.exists():
            try:
                proxy_path.unlink()
            except OSError:
                pass
        elif proxy_path.is_file() and proxy_path.stat().st_size != self.raw_video_path.stat().st_size:
            try:
                proxy_path.unlink()
            except OSError:
                pass
        if not proxy_path.is_file():
            try:
                os.link(self.raw_video_path, proxy_path)
            except OSError:
                try:
                    os.symlink(self.raw_video_path, proxy_path)
                except OSError:
                    shutil.copyfile(self.raw_video_path, proxy_path)
        return proxy_path

    async def _validate_encoded_media(
        self,
        source_path: Path,
        encoded_path: Path,
        fps: float | Literal["keyframes"],
        duration_tolerance_ms: int,
    ) -> tuple[Path | None, bool]:
        """Validates an existing compressed file on disk against expected duration and FPS.

        Args:
            source_path: Uncompressed source file path.
            encoded_path: Candidate re-encoded file path.
            fps: Expected target framerate or 'keyframes'.
            duration_tolerance_ms: Allowed duration tolerance in milliseconds.

        Returns:
            tuple[Path | None, bool]: (encoded_path, True) if valid, else (None, False).
        """
        if await is_valid_reencoded_file(
            input_path=source_path,
            output_path=encoded_path,
            fps=fps,
            duration_tolerance_ms=duration_tolerance_ms,
        ):
            return encoded_path, True
        return None, False

    async def _prepare_media_states(self) -> None:
        """Evaluates media compression needs and validates existing encoded files against target FPS."""
        if self.summary_enabled:
            if self.summary.response is not None or self.summary.exhausted:
                self.summary.media_ready = True
            else:
                sum_cfg = self.reencode_settings.summary
                desired_sum_fps = _resolve_target_fps(
                    self.summary.target_fps,
                    self.summary.fallback_tier,
                    sum_cfg,
                    self.raw_video_path,
                )
                self.summary.target_fps = desired_sum_fps
                if desired_sum_fps is None:
                    self.summary.media_path = self._link_raw_summary_proxy()
                    self.summary.media_ready = True
                else:
                    self.summary.media_path, self.summary.media_ready = await self._validate_encoded_media(
                        self.raw_video_path,
                        self.job_dir / f"reencoded_summary_{self.media_tag}.mp4",
                        desired_sum_fps,
                        int(sum_cfg.tolerance.total_seconds() * 1000),
                    )

        seg_cfg = self.reencode_settings.segment
        seg_tolerance_ms = int(seg_cfg.tolerance.total_seconds() * 1000)
        probe_sem = asyncio.Semaphore(8)

        async def _check_segment(seg: SegmentState) -> None:
            async with probe_sem:
                out_p = self.job_dir / f"reencoded_{seg.raw_path.stem}.mp4"
                if self._is_segment_ai_concluded(seg):
                    seg.media_path = out_p if out_p.is_file() else seg.raw_path
                    seg.media_ready = True
                    return

                desired_fps = _resolve_target_fps(
                    seg.target_fps,
                    seg.fallback_tier,
                    seg_cfg,
                    seg.raw_path,
                )
                seg.target_fps = desired_fps
                if desired_fps is None:
                    seg.media_path = seg.raw_path
                    seg.media_ready = True
                else:
                    seg.media_path, seg.media_ready = await self._validate_encoded_media(
                        seg.raw_path,
                        out_p,
                        desired_fps,
                        seg_tolerance_ms,
                    )

        await asyncio.gather(*(_check_segment(seg) for seg in self.segments))

    def is_summary_complete(self) -> bool:
        """Checks if the whole-video summary stage is complete or disabled.

        Returns:
            bool: True if summary is complete or disabled, False otherwise.
        """
        if not self.summary_enabled:
            return True
        return self.summary.response is not None

    def is_subtitles_complete(self) -> bool:
        """Checks if all subtitle chunks have concluded (succeeded or exhausted).

        Returns:
            bool: True if all chunks have completed or exhausted subtitles, False otherwise.
        """
        return all(self.segment_subtitles_concluded(seg) for seg in self.segments)

    def is_lyrics_complete(self) -> bool:
        """Checks if all segments requiring musical lyrics refinement have concluded.

        Returns:
            bool: True if lyrics refinement is disabled or all segments have concluded lyrics, False otherwise.
        """
        if not self.lyrics_enabled:
            return True
        return all(self.segment_lyrics_concluded(seg) for seg in self.segments)

    def is_done(self) -> bool:
        """Checks if the entire video has completed all stages or aborted on summary exhaustion.

        Returns:
            bool: True if pipeline execution has concluded, False otherwise.
        """
        if self.summary_aborted:
            return True
        return self.is_summary_complete() and self.is_subtitles_complete() and self.is_lyrics_complete()

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

    def get_fallback_probe_paths(
        self,
        task_type: TaskType,
        segment_index: int | None = None,
    ) -> tuple[Path, Path | None] | None:
        """Returns (source_path, keyframe_media) if the next failure triggers keyframe FPS probing.

        Args:
            task_type: AI stage about to record a non-retryable failure.
            segment_index: Zero-based segment index for segment stages. Defaults to None.

        Returns:
            tuple[Path, Path | None] | None: Paths to probe outside coordinator locks, or None.
        """
        if task_type == TaskType.SUMMARY:
            if self.summary.response is not None or self.summary.exhausted:
                return None
            if self.summary.tier_attempts + 1 < self.retry_settings.maxruns:
                return None
            cfg = self.reencode_settings.summary
            media_state: MediaStageState = self.summary
            source_path = self.raw_video_path
        elif task_type == TaskType.SUBTITLES and segment_index is not None:
            seg = self.segments[segment_index]
            if seg.subtitles_response is not None or seg.subtitles_exhausted:
                return None
            if seg.subtitles_tier_attempts + 1 < self.retry_settings.maxruns:
                return None
            cfg = self.reencode_settings.segment
            media_state = seg
            source_path = seg.raw_path
        else:
            return None

        if media_state.fallback_tier != 0 or cfg.fallback.attempts <= 0 or cfg.fps != "keyframes":
            return None

        keyframe_media = (
            media_state.media_path
            if media_state.target_fps == "keyframes" and media_state.media_path is not None
            else None
        )
        return source_path, keyframe_media

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
                user_context=self.user_context,
                model_name=self.summary_config.model,
                service_tier=self.summary_config.tier,
                agentic_video=self.summary_config.agentic,
                total_attempts=self.summary.total_attempts,
                tier_attempts=self.summary.tier_attempts,
                fallback_tier=self.summary.fallback_tier,
                target_fps=self.summary.target_fps,
                exhausted=self.summary.exhausted,
                response=self.summary.response,
            )

        assert seg is not None
        common_seg: dict[str, Any] = {
            "name": seg.raw_path.stem,
            "video_duration_ms": seg.duration_ms,
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
                total_attempts=seg.subtitles_total_attempts,
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
            total_attempts=seg.lyrics_total_attempts,
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
            sum_cfg = self.reencode_settings.summary
            jobs.append(
                ReencodeJob(
                    name="summary_proxy",
                    input_file=self.raw_video_path,
                    output_file=self.job_dir / f"reencoded_summary_{self.media_tag}.mp4",
                    fps=self.summary.target_fps,
                    height=sum_cfg.height,
                    bitrate=int(sum_cfg.bitrate),
                    duration_tolerance_ms=int(sum_cfg.tolerance.total_seconds() * 1000),
                    encoder=self.reencode_settings.encoder,
                    keyframe_only=sum_cfg.fps == "keyframes",
                    is_summary_proxy=True,
                    fallback_tier=self.summary.fallback_tier,
                )
            )

        # 2. Segment re-encode jobs (Priority 2, normal segment index order)
        seg_cfg = self.reencode_settings.segment
        seg_tolerance_ms = int(seg_cfg.tolerance.total_seconds() * 1000)
        for seg in self.segments:
            if not self._is_segment_ai_concluded(seg) and not seg.media_ready and seg.target_fps is not None:
                jobs.append(
                    ReencodeJob(
                        name=f"{seg.raw_path.stem}_reencode",
                        input_file=seg.raw_path,
                        output_file=self.job_dir / f"reencoded_{seg.raw_path.stem}.mp4",
                        fps=seg.target_fps,
                        height=seg_cfg.height,
                        bitrate=int(seg_cfg.bitrate),
                        duration_tolerance_ms=seg_tolerance_ms,
                        encoder=self.reencode_settings.encoder,
                        keyframe_only=seg_cfg.fps == "keyframes",
                        segment_index=seg.index,
                        is_summary_proxy=False,
                        fallback_tier=seg.fallback_tier,
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
        if task_type == TaskType.SUBTITLES:
            job.save(self.job_dir / f"{seg.raw_path.stem}.subtitles.{self.subtitles_shortcode}.yaml")
        elif task_type == TaskType.LYRICS:
            job.save(self.job_dir / f"{seg.raw_path.stem}.lyrics.{self.lyrics_shortcode}.yaml")

    async def _advance_or_exhaust_stage(
        self,
        task_type: TaskType,
        seg: SegmentState | None = None,
        log_context: str | None = None,
        fallback_base_fps: float | None = None,
    ) -> FailureOutcome:
        """Advances a stage to the next fallback FPS tier or marks it exhausted, persisting to disk.

        Args:
            task_type: Target AI stage (SUMMARY or SUBTITLES).
            seg: Target segment state when task_type is SUBTITLES. Defaults to None.
            log_context: Optional failure threshold description to log. Defaults to None.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        is_summary = task_type == TaskType.SUMMARY
        if is_summary:
            media_state: MediaStageState = self.summary
            cfg = self.reencode_settings.summary
            source_path = self.raw_video_path
            tier_attempts = self.summary.tier_attempts
            label = "Summary"
        else:
            assert seg is not None
            media_state = seg
            cfg = self.reencode_settings.segment
            source_path = seg.raw_path
            tier_attempts = seg.subtitles_tier_attempts
            label = f"Subtitles for '{seg.raw_path.stem}'"

        current_tier = media_state.fallback_tier
        current_target = media_state.target_fps
        existing_media = media_state.media_path

        if current_tier < cfg.fallback.attempts:
            next_fps = await _compute_next_fallback_fps(
                cfg=cfg,
                current_tier=current_tier,
                current_target=current_target,
                existing_media=existing_media,
                source_path=source_path,
                probed_base_fps=fallback_base_fps,
            )
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
                logfire.warning(f"{label} {log_context}; falling back to {next_fps:.4g} fps (tier {next_tier}).")
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
            logfire.error(f"{label} exhausted all {self.retry_settings.maxruns} run(s) across all FPS tiers.{suffix}")
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
            media_state.media_path = self._link_raw_summary_proxy() if job.is_summary_proxy else job.input_file
            media_state.media_ready = True
            media_state.target_fps = None
            return FailureOutcome(action="retry")

        return await self._advance_or_exhaust_stage(task_type, seg=seg)

    def complete_ai_task(
        self,
        task_type: TaskType,
        result_data: Any,
        segment_index: int | None = None,
    ) -> None:
        """Validates AI output, persists YAML checkpoints, and advances unit state.

        Args:
            task_type: Completed AI stage type (SUMMARY, SUBTITLES, or LYRICS).
            result_data: Output Pydantic model or dictionary returned by the executor/worker.
            segment_index: Zero-based segment index for SUBTITLES or LYRICS. Defaults to None.

        Raises:
            ValueError: If segment_index is missing or lyrics output is empty when draft subtitles exist.
        """
        val_ctx: dict[str, Any] = {"validation_buffer_ms": self.validation_buffer_ms}

        if task_type == TaskType.SUMMARY:
            val_ctx["video_duration_ms"] = self.duration_ms
            resp = (
                result_data
                if isinstance(result_data, SummaryAiResponse)
                else SummaryAiResponse.model_validate(result_data, context=val_ctx)
            )
            self.summary.response = resp
            self.summary.exhausted = False
            self.summary.total_attempts += 1
            self._save_checkpoint(TaskType.SUMMARY)
            return

        if segment_index is None:
            raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")

        seg = self.segments[segment_index]
        val_ctx["video_duration_ms"] = seg.duration_ms

        if task_type == TaskType.SUBTITLES:
            seg.subtitles_response = (
                result_data
                if isinstance(result_data, SubtitlesAiResponse)
                else SubtitlesAiResponse.model_validate(result_data, context=val_ctx)
            )
            seg.subtitles_exhausted = False
            seg.subtitles_total_attempts += 1
            self._save_checkpoint(TaskType.SUBTITLES, seg)
        elif task_type == TaskType.LYRICS:
            require_subtitles = bool(seg.subtitles_response and seg.subtitles_response.subtitles)
            val_ctx["require_subtitles"] = require_subtitles
            lyr_resp = (
                result_data
                if isinstance(result_data, LyricsAiResponse)
                else LyricsAiResponse.model_validate(result_data, context=val_ctx)
            )
            if require_subtitles and not lyr_resp.subtitles:
                raise ValueError("Refined lyrics output cannot be empty when draft subtitles exist.")
            seg.lyrics_response = lyr_resp
            seg.lyrics_exhausted = False
            seg.lyrics_total_attempts += 1
            self._save_checkpoint(TaskType.LYRICS, seg)

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
    ) -> FailureOutcome:
        """Records a non-retryable AI failure, advancing to fallback FPS or marking exhausted.

        Args:
            task_type: AI stage that encountered a non-retryable failure.
            segment_index: Zero-based segment index for SUBTITLES or LYRICS. Defaults to None.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.

        Returns:
            FailureOutcome: State transition indicating whether to retry, re-encode at lower FPS, or exhaust.

        Raises:
            ValueError: If segment_index is missing for a segment task.
        """
        max_runs = self.retry_settings.maxruns
        seg: SegmentState | None = None
        if task_type in (TaskType.SUBTITLES, TaskType.LYRICS):
            if segment_index is None:
                raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")
            seg = self.segments[segment_index]

        if task_type == TaskType.LYRICS:
            assert seg is not None
            if seg.lyrics_response is not None or seg.lyrics_exhausted:
                return FailureOutcome(action="ignored")
            seg.lyrics_total_attempts += 1
            seg.lyrics_tier_attempts += 1
            target_fps = _float_fps(seg.target_fps)
            if seg.lyrics_tier_attempts >= max_runs:
                seg.lyrics_exhausted = True
                self._save_checkpoint(TaskType.LYRICS, seg)
                logfire.error(
                    f"Lyrics for '{seg.raw_path.stem}' exhausted all {max_runs} run(s); "
                    "falling back to draft subtitles."
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

        if task_type == TaskType.SUMMARY:
            if self.summary.response is not None or self.summary.exhausted:
                return FailureOutcome(action="ignored")
            self.summary.total_attempts += 1
            self.summary.tier_attempts += 1
            tier_attempts = self.summary.tier_attempts
            media_state: MediaStageState = self.summary
        elif task_type == TaskType.SUBTITLES:
            assert seg is not None
            if seg.subtitles_response is not None or seg.subtitles_exhausted:
                return FailureOutcome(action="ignored")
            seg.subtitles_total_attempts += 1
            seg.subtitles_tier_attempts += 1
            tier_attempts = seg.subtitles_tier_attempts
            media_state = seg
        else:
            return FailureOutcome(action="ignored")

        if tier_attempts >= max_runs:
            return await self._advance_or_exhaust_stage(
                task_type,
                seg=seg,
                log_context=f"reached {max_runs} non-retryable failure(s)",
                fallback_base_fps=fallback_base_fps,
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
            "retry": self.retry_settings.model_dump(mode="json"),
            "context": self.user_context,
        }

    def finalize(
        self,
        output_primary_path: Path | None = None,
        output_raw_path: Path | None = None,
        status_override: str | None = None,
        allow_partial: bool = False,
    ) -> list[Path]:
        """Stitches all segment responses into synchronized SRT files with timeline offset shifting.

        When lyrics refinement is required but fails or is unavailable, the primary output
        falls back to the completed draft subtitles rather than emitting an error placeholder.

        Args:
            output_primary_path: Destination path for primary SRT file. Defaults to auto-generated path in job_dir.
            output_raw_path: Optional destination path for raw dialogue SRT. Defaults to auto-generated path.
            status_override: Explicit completion status override string. Defaults to None.
            allow_partial: Whether to stitch partial subtitles when processing is incomplete. Defaults to False.

        Returns:
            list[Path]: List of generated subtitle file paths.

        Raises:
            RuntimeError: If video processing is incomplete and allow_partial is False.
        """
        if not self.is_done() and not allow_partial:
            raise RuntimeError(f"Cannot finalize subtitles: processing for '{self.video_id}' is incomplete.")

        primary_path = (
            output_primary_path
            if output_primary_path is not None
            else self.job_dir / f"{self.filename_stem}.{self.full_shortcode}.srt"
        )
        raw_path = (
            output_raw_path
            if output_raw_path is not None
            else (self.job_dir / f"{self.filename_stem}.raw-{self.full_shortcode}.srt" if self.lyrics_enabled else None)
        )

        all_subs = SSAFile()
        raw_subs = SSAFile() if raw_path is not None else None
        lyrics_refined_any = False

        for seg in self.segments:
            sub_resp = seg.subtitles_response
            lyr_resp = seg.lyrics_response
            lyrics_required = self.segment_needs_lyrics(seg)

            active_resp: SubtitlesAiResponse | LyricsAiResponse | None = None
            if lyrics_required and lyr_resp is not None and lyr_resp.subtitles:
                active_resp = lyr_resp
                lyrics_refined_any = True
            elif sub_resp is not None:
                active_resp = sub_resp

            _append_segment_events(all_subs, active_resp, seg)
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
            "status": status_override or self._derive_status(),
            "settings": self._build_settings_metadata(),
        }

        generated_files: list[Path] = []
        _apply_metadata_header(all_subs, metadata)
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        all_subs.save(str(primary_path))
        generated_files.append(primary_path)

        if lyrics_refined_any and raw_subs is not None and raw_path is not None:
            _apply_metadata_header(raw_subs, metadata)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_subs.save(str(raw_path))
            generated_files.append(raw_path)

        return generated_files
