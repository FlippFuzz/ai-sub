"""Unified video processing pipeline managing segmentation, compression, AI DAG stages, and stitching."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import logfire
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    SegmentReEncodeSettings,
    ServiceTier,
    SplittingSettings,
    SummaryReEncodeSettings,
)
from ai_sub.jobs.base import AiJob, BaseJob
from ai_sub.jobs.lyrics import LyricsAiResponse, LyricsJob
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import SubtitlesAiResponse, SubtitlesJob
from ai_sub.jobs.summary import SummaryAiResponse, SummaryJob
from ai_sub.protocol import TaskType
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_subtitles_shortcode,
    generate_summary_shortcode,
)
from ai_sub.stitching import stitch_subtitles
from ai_sub.video import (
    get_video_duration_ms,
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


class SummaryState(BaseModel):
    """Per-unit execution and media state for the Phase 1 whole-video summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    media_ready: bool = Field(default=False, description="Whether summary media is ready for AI inference.")
    media_path: Path | None = Field(default=None, description="Path to active summary media file.")
    target_fps: PositiveFloat | None = Field(
        default=None,
        description="Target FPS for re-encoding, or None if using raw video.",
    )
    fallback_tier: NonNegativeInt = Field(default=0, description="Active fallback FPS tier index.")
    tier_attempts: NonNegativeInt = Field(default=0, description="Non-retryable failures at the current FPS tier.")
    total_attempts: NonNegativeInt = Field(default=0, description="Cumulative attempts across all executions.")
    response: SummaryAiResponse | None = Field(default=None, description="Completed summary response.")
    exhausted: bool = Field(default=False, description="Whether summary exhausted all retries and fallback tiers.")


class SegmentState(BaseModel):
    """Per-unit execution, media, and AI stage state for an individual video chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(description="Zero-based segment index.")
    raw_path: Path = Field(description="Path to the split uncompressed segment chunk.")
    duration_ms: int = Field(description="Duration of this segment in milliseconds.")
    offset_ms: int = Field(description="Absolute start offset of this segment in milliseconds.")

    media_ready: bool = Field(default=False, description="Whether segment media is ready for AI inference.")
    media_path: Path | None = Field(default=None, description="Path to active compressed or raw segment media.")
    target_fps: PositiveFloat | None = Field(
        default=None,
        description="Target FPS for re-encoding, or None if using raw segment.",
    )
    fallback_tier: NonNegativeInt = Field(default=0, description="Active fallback FPS tier index.")

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


def _resolve_target_fps(
    current_target: float | None,
    cfg: SummaryReEncodeSettings | SegmentReEncodeSettings,
    source_path: Path,
) -> float | None:
    """Determines target compression FPS from fallback state or size threshold configuration.

    Args:
        current_target: Persisted fallback target FPS if already in fallback, else None.
        cfg: Re-encoding settings for this stage.
        source_path: Input media file path to evaluate against threshold_mb.

    Returns:
        float | None: Target FPS if re-encoding is needed, or None to use raw media.
    """
    if current_target is not None:
        return current_target
    file_mb = source_path.stat().st_size / (1024 * 1024)
    if cfg.enabled and (cfg.threshold_mb == 0 or file_mb >= cfg.threshold_mb):
        return cfg.fps
    return None


class Pipeline(BaseModel):
    """Manages the full lifecycle, media preparation, on-disk checkpoints, and DAG state for a video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_video_path: Path = Field(description="Filesystem path to the raw input video.")
    job_dir: Path = Field(description="Working directory containing state files and segments for this video.")
    video_id: str = Field(description="Unique video identifier.")
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
    enforce_max_runs: bool = Field(
        default=True,
        description="Whether to enforce RetrySettings.max_runs per FPS tier across runs.",
    )

    summary_model: str = Field(description="Gemini model identifier for summary.")
    subtitles_model: str = Field(description="Gemini model identifier for subtitles.")
    lyrics_model: str = Field(description="Gemini model identifier for lyrics.")
    service_tier_summary: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier override for summary.",
    )
    service_tier_subtitles: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier override for subtitles.",
    )
    service_tier_lyrics: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier override for lyrics.",
    )
    agentic_video_summary: bool = Field(
        default=True,
        description="Whether agentic video understanding is enabled for summary.",
    )
    agentic_video_subtitles: bool = Field(
        default=False,
        description="Whether agentic video understanding is enabled for subtitles.",
    )
    agentic_video_lyrics: bool = Field(
        default=True,
        description="Whether agentic video understanding is enabled for lyrics.",
    )

    summary_shortcode: str = Field(description="Model shortcode for summary.")
    subtitles_shortcode: str = Field(description="Model shortcode for subtitles.")
    lyrics_shortcode: str = Field(description="Model shortcode for lyrics.")
    full_shortcode: str = Field(description="Combined full shortcode for final output.")

    reencode_settings: ReEncodeSettings = Field(description="Re-encoding configuration settings.")
    search_settings: SearchSettings = Field(description="Search configuration settings.")
    split_settings: SplittingSettings = Field(description="Splitting configuration settings.")
    retry_settings: RetrySettings = Field(description="Retry configuration settings.")

    initial_offset_ms: int = Field(default=0, description="Initial start offset in milliseconds from start_offset_min.")
    summary: SummaryState = Field(default_factory=SummaryState, description="Whole-video summary unit state.")
    segments: list[SegmentState] = Field(default_factory=list, description="Ordered per-segment state machines.")

    @property
    def summary_aborted(self) -> bool:
        """Whether the summary stage permanently failed and aborted pipeline execution."""
        return self.summary_enabled and self.summary.exhausted

    @property
    def _effective_max_runs(self) -> int:
        """Maximum non-retryable failures per FPS tier before fallback or exhaustion."""
        return self.retry_settings.max_runs if self.enforce_max_runs else 1

    @classmethod
    async def create(
        cls,
        raw_video_path: Path,
        job_dir: Path,
        title: str,
        uploader: str,
        summary_model: str,
        subtitles_model: str,
        lyrics_model: str,
        reencode_settings: ReEncodeSettings,
        search_settings: SearchSettings,
        split_settings: SplittingSettings,
        retry_settings: RetrySettings,
        channel_name: str | None = None,
        video_id: str | None = None,
        source_type: Literal["local", "youtube"] = "local",
        timestamp: int = 0,
        user_context: str | None = None,
        validation_buffer_ms: int = 2000,
        summary_enabled: bool = True,
        lyrics_enabled: bool = True,
        enforce_max_runs: bool = True,
        service_tier_summary: ServiceTier | None = None,
        service_tier_subtitles: ServiceTier | None = None,
        service_tier_lyrics: ServiceTier | None = None,
        agentic_video_summary: bool = True,
        agentic_video_subtitles: bool = False,
        agentic_video_lyrics: bool = True,
    ) -> Self:
        """Asynchronously initializes the pipeline, probes media, splits segments, and resumes checkpoints.

        Loads YAML state checkpoints before validating compressed media on disk so any unit
        in a fallback FPS tier validates against its persisted target FPS.

        Args:
            raw_video_path: Path to the raw input video.
            job_dir: Working directory containing state files and segments for this video.
            title: Display title of the video.
            uploader: Channel or uploader identifier.
            summary_model: Gemini model identifier for summary.
            subtitles_model: Gemini model identifier for subtitles.
            lyrics_model: Gemini model identifier for lyrics.
            reencode_settings: Re-encoding configuration settings.
            search_settings: Search configuration settings.
            split_settings: Splitting configuration settings.
            retry_settings: Retry configuration settings.
            channel_name: Human-readable channel display name. Defaults to None.
            video_id: Unique video identifier. Defaults to raw_video_path.stem.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication or intake Unix timestamp. Defaults to 0.
            user_context: Background user context string. Defaults to None.
            validation_buffer_ms: Timestamp validation tolerance in milliseconds. Defaults to 2000.
            summary_enabled: Whether the whole-video summary stage is enabled. Defaults to True.
            lyrics_enabled: Whether musical lyrics refinement is enabled. Defaults to True.
            enforce_max_runs: Whether to enforce RetrySettings.max_runs per FPS tier across runs. Defaults to True.
            service_tier_summary: Optional Google Gemini service tier override for summary. Defaults to None.
            service_tier_subtitles: Optional Google Gemini service tier override for subtitles. Defaults to None.
            service_tier_lyrics: Optional Google Gemini service tier override for lyrics. Defaults to None.
            agentic_video_summary: Whether agentic video is enabled for summary. Defaults to True.
            agentic_video_subtitles: Whether agentic video is enabled for subtitles. Defaults to False.
            agentic_video_lyrics: Whether agentic video is enabled for lyrics. Defaults to True.

        Returns:
            Pipeline: Initialized pipeline instance with media prepared and checkpoints loaded.

        Raises:
            RuntimeError: If no video segments could be created.
            ValueError: If start_offset_min exceeds the total video duration.
        """
        raw_resolved = raw_video_path.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "reencoded").mkdir(parents=True, exist_ok=True)

        vid = video_id or raw_resolved.stem
        filename_stem = sanitize_filename(raw_resolved.stem)

        # 1. Probe raw video duration
        total_duration_ms = await get_video_duration_ms(raw_resolved)

        # 2. Split video into uniform segments
        all_segments = await split_video(
            input_video=raw_resolved,
            output_dir=job_dir,
            split_duration_s=split_settings.max_seconds,
            output_pattern="part_%03d",
            duration_tolerance_ms=split_settings.duration_tolerance_ms,
        )
        if not all_segments:
            raise RuntimeError(f"No video segments produced for {raw_resolved.name}.")

        all_durations = await asyncio.gather(*(get_video_duration_ms(p) for p in all_segments))

        # 3. Apply start_offset_min using actual durations of skipped segments
        if split_settings.start_offset_min > 0:
            start_offset_s = split_settings.start_offset_min * 60
            skip_count = start_offset_s // split_settings.max_seconds
            if skip_count >= len(all_segments):
                raise ValueError(
                    f"start_offset_min ({split_settings.start_offset_min}m) exceeds or equals total "
                    f"video duration ({total_duration_ms / 60000:.1f}m)."
                )
            initial_offset_ms = sum(all_durations[:skip_count])
            segment_paths = all_segments[skip_count:]
            segment_durations_ms = all_durations[skip_count:]
            logfire.info(
                f"Skipping first {skip_count} segment(s) ({initial_offset_ms / 60000:.1f}m) "
                f"due to start_offset_min={split_settings.start_offset_min}."
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

        # 5. Generate shortcodes
        summary_shortcode = generate_summary_shortcode(summary_model) if summary_enabled else ""
        subtitles_shortcode = generate_subtitles_shortcode(
            model_name=subtitles_model,
            summary_shortcode=summary_shortcode if summary_enabled else None,
        )
        lyrics_shortcode = generate_lyrics_shortcode(
            lyrics_model_name=lyrics_model,
            subtitles_shortcode=subtitles_shortcode,
        )
        full_shortcode = generate_full_shortcode(subtitles_model)

        instance = cls(
            raw_video_path=raw_resolved,
            job_dir=job_dir,
            video_id=vid,
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
            enforce_max_runs=enforce_max_runs,
            summary_model=summary_model,
            subtitles_model=subtitles_model,
            lyrics_model=lyrics_model,
            service_tier_summary=service_tier_summary,
            service_tier_subtitles=service_tier_subtitles,
            service_tier_lyrics=service_tier_lyrics,
            agentic_video_summary=agentic_video_summary,
            agentic_video_subtitles=agentic_video_subtitles,
            agentic_video_lyrics=agentic_video_lyrics,
            summary_shortcode=summary_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            lyrics_shortcode=lyrics_shortcode,
            full_shortcode=full_shortcode,
            reencode_settings=reencode_settings,
            search_settings=search_settings,
            split_settings=split_settings,
            retry_settings=retry_settings,
            initial_offset_ms=initial_offset_ms,
            segments=segments,
        )

        # 6. Load YAML checkpoints first so fallback FPS targets are known before media validation
        instance._discover_on_disk_state()
        await instance._prepare_media_states()
        return instance

    def _restore_failure_state(self, job: AiJob, fallback_count: int) -> tuple[int, int, float | None, bool]:
        """Resolves fallback tier, attempt count, target FPS, and exhaustion from an incomplete checkpoint.

        In standalone mode (enforce_max_runs=False), previously exhausted jobs reset to initial settings
        so users can re-run freely.

        Args:
            job: Loaded incomplete AiJob checkpoint (response is None).
            fallback_count: Number of configured fallback FPS tiers for this stage.

        Returns:
            tuple[int, int, float | None, bool]: (fallback_tier, tier_attempts, target_fps, exhausted).
        """
        was_exhausted = job.exhausted or (
            job.tier_attempts >= self.retry_settings.max_runs and job.fallback_tier >= fallback_count
        )
        if not self.enforce_max_runs and was_exhausted:
            return 0, 0, None, False
        return job.fallback_tier, job.tier_attempts, job.target_fps, was_exhausted

    def _discover_on_disk_state(self) -> None:
        """Inspects job_dir for existing valid YAML checkpoints and restores unit states."""
        summary_fallbacks = len(self.reencode_settings.summary.fallback_fps)
        segment_fallbacks = len(self.reencode_settings.segment.fallback_fps)

        if self.summary_enabled:
            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            loaded_summary = SummaryJob.load(summary_file, self.validation_buffer_ms)
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

        for seg in self.segments:
            sub_file = self.job_dir / f"{seg.raw_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
            loaded_sub = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            if not loaded_sub:
                continue

            seg.subtitles_total_attempts = loaded_sub.total_attempts
            if loaded_sub.response is not None:
                seg.subtitles_response = loaded_sub.response
                seg.subtitles_tier_attempts = loaded_sub.tier_attempts
                seg.fallback_tier = loaded_sub.fallback_tier
                seg.target_fps = loaded_sub.target_fps

                if self.lyrics_enabled and loaded_sub.response.detected_musical_performances:
                    lyr_file = self.job_dir / f"{seg.raw_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
                    loaded_lyr = LyricsJob.load(lyr_file, self.validation_buffer_ms)
                    if loaded_lyr:
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
        if seg.subtitles_exhausted:
            return True
        if seg.subtitles_response is None:
            return False
        if self.lyrics_enabled and seg.subtitles_response.detected_musical_performances:
            return seg.lyrics_response is not None or seg.lyrics_exhausted
        return True

    async def _validate_encoded_media(
        self,
        source_path: Path,
        encoded_path: Path,
        fps: float,
        duration_tolerance_ms: int,
    ) -> tuple[Path | None, bool]:
        """Validates an existing compressed file on disk against expected duration and FPS.

        Args:
            source_path: Uncompressed source file path.
            encoded_path: Candidate re-encoded file path.
            fps: Expected target framerate.
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
                desired_sum_fps = _resolve_target_fps(self.summary.target_fps, sum_cfg, self.raw_video_path)
                self.summary.target_fps = desired_sum_fps
                if desired_sum_fps is None:
                    proxy_path = self.job_dir / f"summary_proxy{self.raw_video_path.suffix}"
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
                    self.summary.media_path = proxy_path
                    self.summary.media_ready = True
                else:
                    self.summary.media_path, self.summary.media_ready = await self._validate_encoded_media(
                        self.raw_video_path,
                        self.job_dir / "summary_proxy.mp4",
                        desired_sum_fps,
                        sum_cfg.duration_tolerance_ms,
                    )

        seg_cfg = self.reencode_settings.segment
        reencode_dir = self.job_dir / "reencoded"
        probe_sem = asyncio.Semaphore(8)

        async def _check_segment(seg: SegmentState) -> None:
            async with probe_sem:
                out_p = reencode_dir / f"{seg.raw_path.stem}.mp4"
                if self._is_segment_ai_concluded(seg):
                    seg.media_path = out_p if out_p.is_file() else seg.raw_path
                    seg.media_ready = True
                    return

                desired_fps = _resolve_target_fps(seg.target_fps, seg_cfg, seg.raw_path)
                seg.target_fps = desired_fps
                if desired_fps is None:
                    seg.media_path = seg.raw_path
                    seg.media_ready = True
                else:
                    seg.media_path, seg.media_ready = await self._validate_encoded_media(
                        seg.raw_path,
                        out_p,
                        desired_fps,
                        seg_cfg.duration_tolerance_ms,
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
        return all(seg.subtitles_response is not None or seg.subtitles_exhausted for seg in self.segments)

    def is_lyrics_complete(self) -> bool:
        """Checks if all segments with detected musical performances have concluded lyrics.

        Returns:
            bool: True if all singing segments have resolved or exhausted lyrics, False otherwise.
        """
        if not self.lyrics_enabled:
            return True
        if not self.is_subtitles_complete():
            return False
        for seg in self.segments:
            if seg.subtitles_response and seg.subtitles_response.detected_musical_performances:
                if seg.lyrics_response is None and not seg.lyrics_exhausted:
                    return False
        return True

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
                    output_file=self.job_dir / "summary_proxy.mp4",
                    fps=self.summary.target_fps,
                    height=sum_cfg.height,
                    bitrate_kb=sum_cfg.bitrate_kb,
                    duration_tolerance_ms=sum_cfg.duration_tolerance_ms,
                    encoder=self.reencode_settings.encoder,
                    keyframe_only=sum_cfg.keyframe_only,
                    is_summary_proxy=True,
                    fallback_tier=self.summary.fallback_tier,
                )
            )

        # 2. Segment re-encode jobs (Priority 2, normal segment index order)
        seg_cfg = self.reencode_settings.segment
        reencode_dir = self.job_dir / "reencoded"
        for seg in self.segments:
            if not self._is_segment_ai_concluded(seg) and not seg.media_ready and seg.target_fps is not None:
                jobs.append(
                    ReencodeJob(
                        name=f"{seg.raw_path.stem}_reencode",
                        input_file=seg.raw_path,
                        output_file=reencode_dir / f"{seg.raw_path.stem}.mp4",
                        fps=seg.target_fps,
                        height=seg_cfg.height,
                        bitrate_kb=seg_cfg.bitrate_kb,
                        duration_tolerance_ms=seg_cfg.duration_tolerance_ms,
                        encoder=self.reencode_settings.encoder,
                        keyframe_only=seg_cfg.keyframe_only,
                        segment_index=seg.index,
                        is_summary_proxy=False,
                        fallback_tier=seg.fallback_tier,
                    )
                )

        # 3. Whole-video Summary AI job
        if self.summary_enabled and self.summary.response is None and not self.summary.exhausted:
            if self.summary.media_ready and self.summary.media_path is not None:
                jobs.append(
                    SummaryJob(
                        name="summary",
                        video_duration_ms=self.duration_ms,
                        user_context=self.user_context,
                        media_path=self.summary.media_path,
                        model_name=self.summary_model,
                        service_tier=self.service_tier_summary,
                        agentic_video=self.agentic_video_summary,
                        retry_settings=self.retry_settings,
                        total_attempts=self.summary.total_attempts,
                        tier_attempts=self.summary.tier_attempts,
                        fallback_tier=self.summary.fallback_tier,
                        target_fps=self.summary.target_fps,
                    )
                )
            return jobs

        # 4. Per-segment Subtitles and Lyrics AI jobs (independent per segment once summary completes)
        for seg in self.segments:
            if not seg.media_ready or seg.media_path is None:
                continue

            if seg.subtitles_response is None and not seg.subtitles_exhausted:
                jobs.append(
                    SubtitlesJob(
                        name=seg.raw_path.stem,
                        video_duration_ms=seg.duration_ms,
                        segment_offset_ms=seg.offset_ms,
                        global_summary=self.summary.response,
                        user_context=self.user_context,
                        media_path=seg.media_path,
                        model_name=self.subtitles_model,
                        service_tier=self.service_tier_subtitles,
                        agentic_video=self.agentic_video_subtitles,
                        retry_settings=self.retry_settings,
                        segment_index=seg.index,
                        total_attempts=seg.subtitles_total_attempts,
                        tier_attempts=seg.subtitles_tier_attempts,
                        fallback_tier=seg.fallback_tier,
                        target_fps=seg.target_fps,
                    )
                )
            elif self.lyrics_enabled and seg.subtitles_response is not None:
                sub_resp = seg.subtitles_response
                if sub_resp.detected_musical_performances and seg.lyrics_response is None and not seg.lyrics_exhausted:
                    jobs.append(
                        LyricsJob(
                            name=seg.raw_path.stem,
                            video_duration_ms=seg.duration_ms,
                            segment_offset_ms=seg.offset_ms,
                            draft_subtitles=sub_resp.subtitles,
                            detected_performances=sub_resp.detected_musical_performances,
                            global_summary=self.summary.response,
                            user_context=self.user_context,
                            media_path=seg.media_path,
                            model_name=self.lyrics_model,
                            service_tier=self.service_tier_lyrics,
                            agentic_video=self.agentic_video_lyrics,
                            retry_settings=self.retry_settings,
                            segment_index=seg.index,
                            total_attempts=seg.lyrics_total_attempts,
                            tier_attempts=seg.lyrics_tier_attempts,
                            fallback_tier=seg.fallback_tier,
                            target_fps=seg.target_fps,
                        )
                    )

        return jobs

    def _save_checkpoint(self, task_type: TaskType, seg: SegmentState | None = None) -> None:
        """Persists the current state of an AI stage into its YAML checkpoint file.

        Args:
            task_type: AI stage to persist (SUMMARY, SUBTITLES, or LYRICS).
            seg: SegmentState required when task_type is SUBTITLES or LYRICS. Defaults to None.
        """
        if task_type == TaskType.SUMMARY:
            SummaryJob(
                name="summary",
                video_duration_ms=self.duration_ms,
                user_context=self.user_context,
                service_tier=self.service_tier_summary,
                agentic_video=self.agentic_video_summary,
                total_attempts=self.summary.total_attempts,
                tier_attempts=self.summary.tier_attempts,
                fallback_tier=self.summary.fallback_tier,
                target_fps=self.summary.target_fps,
                exhausted=self.summary.exhausted,
                response=self.summary.response,
            ).save(self.job_dir / f"summary.{self.summary_shortcode}.yaml")
            return

        assert seg is not None
        if task_type == TaskType.SUBTITLES:
            SubtitlesJob(
                name=seg.raw_path.stem,
                video_duration_ms=seg.duration_ms,
                segment_offset_ms=seg.offset_ms,
                global_summary=self.summary.response,
                user_context=self.user_context,
                service_tier=self.service_tier_subtitles,
                agentic_video=self.agentic_video_subtitles,
                segment_index=seg.index,
                total_attempts=seg.subtitles_total_attempts,
                tier_attempts=seg.subtitles_tier_attempts,
                fallback_tier=seg.fallback_tier,
                target_fps=seg.target_fps,
                exhausted=seg.subtitles_exhausted,
                response=seg.subtitles_response,
            ).save(self.job_dir / f"{seg.raw_path.stem}.subtitles.{self.subtitles_shortcode}.yaml")
        elif task_type == TaskType.LYRICS:
            sub_resp = seg.subtitles_response
            LyricsJob(
                name=seg.raw_path.stem,
                video_duration_ms=seg.duration_ms,
                segment_offset_ms=seg.offset_ms,
                draft_subtitles=sub_resp.subtitles if sub_resp else [],
                detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
                global_summary=self.summary.response,
                user_context=self.user_context,
                service_tier=self.service_tier_lyrics,
                agentic_video=self.agentic_video_lyrics,
                segment_index=seg.index,
                total_attempts=seg.lyrics_total_attempts,
                tier_attempts=seg.lyrics_tier_attempts,
                fallback_tier=seg.fallback_tier,
                target_fps=seg.target_fps,
                exhausted=seg.lyrics_exhausted,
                response=seg.lyrics_response,
            ).save(self.job_dir / f"{seg.raw_path.stem}.lyrics.{self.lyrics_shortcode}.yaml")

    def _advance_or_exhaust_summary(self, log_context: str | None = None) -> FailureOutcome:
        """Advances summary to the next fallback FPS tier or marks it exhausted, persisting to disk.

        Args:
            log_context: Optional failure threshold description to log. Defaults to None.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        fallbacks = self.reencode_settings.summary.fallback_fps
        if self.summary.fallback_tier < len(fallbacks):
            self.summary.fallback_tier += 1
            next_fps = fallbacks[self.summary.fallback_tier - 1]
            self.summary.target_fps = next_fps
            self.summary.tier_attempts = 0
            self.summary.media_path = None
            self.summary.media_ready = False
            self._save_checkpoint(TaskType.SUMMARY)
            if log_context:
                logfire.warning(
                    f"Summary {log_context}; falling back to {next_fps} fps (tier {self.summary.fallback_tier})."
                )
            return FailureOutcome(
                action="fallback_reencode",
                fallback_tier=self.summary.fallback_tier,
                target_fps=next_fps,
                tier_attempts=0,
            )

        self.summary.exhausted = True
        self._save_checkpoint(TaskType.SUMMARY)
        if log_context:
            logfire.error(
                f"Summary exhausted all {self._effective_max_runs} attempt(s) across all FPS tiers. Aborting."
            )
        return FailureOutcome(
            action="exhausted",
            fallback_tier=self.summary.fallback_tier,
            target_fps=self.summary.target_fps,
            tier_attempts=self.summary.tier_attempts,
        )

    def _advance_or_exhaust_segment(self, seg: SegmentState, log_context: str | None = None) -> FailureOutcome:
        """Advances a segment to the next fallback FPS tier or marks subtitles exhausted, persisting to disk.

        Args:
            seg: Target segment state.
            log_context: Optional failure threshold description to log. Defaults to None.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        fallbacks = self.reencode_settings.segment.fallback_fps
        if seg.fallback_tier < len(fallbacks):
            seg.fallback_tier += 1
            next_fps = fallbacks[seg.fallback_tier - 1]
            seg.target_fps = next_fps
            seg.subtitles_tier_attempts = 0
            seg.media_path = None
            seg.media_ready = False
            self._save_checkpoint(TaskType.SUBTITLES, seg)
            if log_context:
                logfire.warning(
                    f"Subtitles for '{seg.raw_path.stem}' {log_context}; "
                    f"falling back to {next_fps} fps (tier {seg.fallback_tier})."
                )
            return FailureOutcome(
                action="fallback_reencode",
                fallback_tier=seg.fallback_tier,
                target_fps=next_fps,
                tier_attempts=0,
            )

        seg.subtitles_exhausted = True
        self._save_checkpoint(TaskType.SUBTITLES, seg)
        if log_context:
            logfire.error(
                f"Subtitles for '{seg.raw_path.stem}' exhausted all "
                f"{self._effective_max_runs} attempt(s) across all FPS tiers."
            )
        return FailureOutcome(
            action="exhausted",
            fallback_tier=seg.fallback_tier,
            target_fps=seg.target_fps,
            tier_attempts=seg.subtitles_tier_attempts,
        )

    def complete_reencode(self, job: ReencodeJob, media_path: Path) -> None:
        """Marks a summary proxy or segment re-encode job complete and unlocks downstream AI work.

        Args:
            job: Completed ReencodeJob instance.
            media_path: Resulting media file path.
        """
        if job.is_summary_proxy:
            self.summary.media_path = media_path
            self.summary.media_ready = True
        elif job.segment_index is not None:
            seg = self.segments[job.segment_index]
            seg.media_path = media_path
            seg.media_ready = True

    def fail_reencode(self, job: ReencodeJob) -> FailureOutcome:
        """Handles a failed FFmpeg re-encode job.

        On initial encoding (fallback_tier == 0), falls back safely to the raw input file.
        On fallback encoding (fallback_tier > 0), advances to the next lower FPS tier if available
        or marks the unit exhausted without reusing previously failed media.

        Args:
            job: Failed ReencodeJob instance.

        Returns:
            FailureOutcome: Resulting state transition.
        """
        if job.fallback_tier == 0:
            if job.is_summary_proxy:
                self.summary.media_path = job.input_file
                self.summary.media_ready = True
                self.summary.target_fps = None
            elif job.segment_index is not None:
                seg = self.segments[job.segment_index]
                seg.media_path = job.input_file
                seg.media_ready = True
                seg.target_fps = None
            return FailureOutcome(action="retry")

        if job.is_summary_proxy:
            return self._advance_or_exhaust_summary()
        if job.segment_index is not None:
            return self._advance_or_exhaust_segment(self.segments[job.segment_index])
        return FailureOutcome(action="ignored")

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
            ValueError: If segment_index is missing for a segment task.
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
            self.summary.tier_attempts += 1
            self._save_checkpoint(TaskType.SUMMARY)
            return

        if segment_index is None:
            raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")

        seg = self.segments[segment_index]
        val_ctx["video_duration_ms"] = seg.duration_ms

        if task_type == TaskType.SUBTITLES:
            payload = (
                result_data.model_dump(mode="json") if isinstance(result_data, SubtitlesAiResponse) else result_data
            )
            seg.subtitles_response = SubtitlesAiResponse.model_validate(payload, context=val_ctx)
            seg.subtitles_exhausted = False
            seg.subtitles_total_attempts += 1
            seg.subtitles_tier_attempts += 1
            self._save_checkpoint(TaskType.SUBTITLES, seg)
        elif task_type == TaskType.LYRICS:
            payload = result_data.model_dump(mode="json") if isinstance(result_data, LyricsAiResponse) else result_data
            seg.lyrics_response = LyricsAiResponse.model_validate(payload, context=val_ctx)
            seg.lyrics_exhausted = False
            seg.lyrics_total_attempts += 1
            seg.lyrics_tier_attempts += 1
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
        elif isinstance(job, SummaryJob):
            self.complete_ai_task(TaskType.SUMMARY, result_data)
        elif isinstance(job, SubtitlesJob):
            self.complete_ai_task(TaskType.SUBTITLES, result_data, segment_index=job.segment_index)
        elif isinstance(job, LyricsJob):
            self.complete_ai_task(TaskType.LYRICS, result_data, segment_index=job.segment_index)
        else:
            raise TypeError(f"Unsupported job type: {type(job).__name__}")

    def record_failure(
        self,
        task_type: TaskType,
        segment_index: int | None = None,
    ) -> FailureOutcome:
        """Records a non-retryable AI failure, advancing to fallback FPS or marking exhausted.

        Args:
            task_type: AI stage that encountered a non-retryable failure.
            segment_index: Zero-based segment index for SUBTITLES or LYRICS. Defaults to None.

        Returns:
            FailureOutcome: State transition indicating whether to retry, re-encode at lower FPS, or exhaust.

        Raises:
            ValueError: If segment_index is missing for a segment task.
        """
        max_runs = self._effective_max_runs
        log_reason = f"reached {max_runs} non-retryable failure(s)"

        if task_type == TaskType.SUMMARY:
            if self.summary.response is not None or self.summary.exhausted:
                return FailureOutcome(action="ignored")
            self.summary.total_attempts += 1
            self.summary.tier_attempts += 1
            if self.summary.tier_attempts >= max_runs:
                return self._advance_or_exhaust_summary(log_context=log_reason)
            self._save_checkpoint(TaskType.SUMMARY)
            return FailureOutcome(
                action="retry",
                fallback_tier=self.summary.fallback_tier,
                target_fps=self.summary.target_fps,
                tier_attempts=self.summary.tier_attempts,
            )

        if segment_index is None:
            raise ValueError(f"segment_index is required for task_type '{task_type.value}'.")

        seg = self.segments[segment_index]

        if task_type == TaskType.SUBTITLES:
            if seg.subtitles_response is not None or seg.subtitles_exhausted:
                return FailureOutcome(action="ignored")
            seg.subtitles_total_attempts += 1
            seg.subtitles_tier_attempts += 1
            if seg.subtitles_tier_attempts >= max_runs:
                return self._advance_or_exhaust_segment(seg, log_context=log_reason)
            self._save_checkpoint(TaskType.SUBTITLES, seg)
            return FailureOutcome(
                action="retry",
                fallback_tier=seg.fallback_tier,
                target_fps=seg.target_fps,
                tier_attempts=seg.subtitles_tier_attempts,
            )

        if task_type == TaskType.LYRICS:
            if seg.lyrics_response is not None or seg.lyrics_exhausted:
                return FailureOutcome(action="ignored")
            seg.lyrics_total_attempts += 1
            seg.lyrics_tier_attempts += 1
            if seg.lyrics_tier_attempts >= max_runs:
                seg.lyrics_exhausted = True
                self._save_checkpoint(TaskType.LYRICS, seg)
                logfire.error(
                    f"Lyrics for '{seg.raw_path.stem}' exhausted all {max_runs} attempt(s); "
                    "falling back to draft subtitles."
                )
                return FailureOutcome(
                    action="exhausted",
                    fallback_tier=seg.fallback_tier,
                    target_fps=seg.target_fps,
                    tier_attempts=seg.lyrics_tier_attempts,
                )
            self._save_checkpoint(TaskType.LYRICS, seg)
            return FailureOutcome(
                action="retry",
                fallback_tier=seg.fallback_tier,
                target_fps=seg.target_fps,
                tier_attempts=seg.lyrics_tier_attempts,
            )

        return FailureOutcome(action="ignored")

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
        search_dump = self.search_settings.model_dump(
            mode="json",
            exclude={"key": True, "tinyfish_key": True, "langsearch_key": True, "ollama_key": True},
        )
        return {
            "ai": {
                "model_summary": self.summary_model,
                "model_subtitles": self.subtitles_model,
                "model_lyrics": self.lyrics_model,
                "service_tier_summary": self.service_tier_summary,
                "service_tier_subtitles": self.service_tier_subtitles,
                "service_tier_lyrics": self.service_tier_lyrics,
                "search": search_dump,
                "agentic_video_summary": self.agentic_video_summary,
                "agentic_video_subtitles": self.agentic_video_subtitles,
                "agentic_video_lyrics": self.agentic_video_lyrics,
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
        """Stitches all segment YAML files into synchronized SRT files with timeline offset shifting.

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

        segments = [(seg.raw_path.stem, seg.duration_ms) for seg in self.segments]

        result = stitch_subtitles(
            segments=segments,
            state_dir=self.job_dir,
            output_primary_path=primary_path,
            output_raw_path=raw_path,
            subtitles_shortcode=self.subtitles_shortcode,
            lyrics_shortcode=self.lyrics_shortcode,
            validation_buffer_ms=self.validation_buffer_ms,
            settings_dump=self._build_settings_metadata(),
            status_override=status_override or self._derive_status(),
            summary_available=self.summary.response is not None,
            initial_offset_ms=self.initial_offset_ms,
            use_lyrics=self.lyrics_enabled,
        )
        return result.generated_files
