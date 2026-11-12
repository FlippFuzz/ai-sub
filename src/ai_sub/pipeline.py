"""Unified video processing pipeline managing video splitting, compression, AI DAG stages, and stitching."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, Self

import logfire
from pydantic import BaseModel, ConfigDict, Field
from yt_dlp.utils import sanitize_filename

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
)
from ai_sub.jobs.base import BaseJob
from ai_sub.jobs.lyrics import (
    LyricsAiResponse,
    LyricsJob,
)
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import (
    SubtitlesAiResponse,
    SubtitlesJob,
)
from ai_sub.jobs.summary import (
    SummaryAiResponse,
    SummaryJob,
)
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
    split_video,
)


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

    summary_model: str = Field(description="Gemini model identifier for summary.")
    subtitles_model: str = Field(description="Gemini model identifier for subtitles.")
    lyrics_model: str = Field(description="Gemini model identifier for lyrics.")
    summary_shortcode: str = Field(description="Model shortcode for summary.")
    subtitles_shortcode: str = Field(description="Model shortcode for subtitles.")
    lyrics_shortcode: str = Field(description="Model shortcode for lyrics.")
    full_shortcode: str = Field(description="Combined full shortcode for final output.")

    reencode_settings: ReEncodeSettings = Field(description="Re-encoding configuration settings.")
    search_settings: SearchSettings = Field(description="Search configuration settings.")
    split_settings: SplittingSettings = Field(description="Splitting configuration settings.")
    retry_settings: RetrySettings = Field(description="Retry configuration settings.")

    initial_offset_ms: int = Field(default=0, description="Initial start offset in milliseconds from start_offset_min.")
    segment_paths: list[Path] = Field(default_factory=list, description="Ordered paths to active video chunks.")
    segment_durations_ms: list[int] = Field(default_factory=list, description="Durations of active chunks.")

    summary_proxy_needed: bool = Field(default=False, description="Whether summary proxy re-encoding is required.")
    summary_proxy_path: Path | None = Field(default=None, description="Path to the summary proxy video if ready.")
    reencoded_segment_paths: dict[int, Path] = Field(
        default_factory=dict, description="Mapping of segment indices to compressed media paths."
    )

    summary_response: SummaryAiResponse | None = Field(default=None, description="Summary AI response model.")
    summary_failed: bool = Field(default=False, description="Whether summary stage exhausted max runs.")
    summary_attempts: int = Field(default=0, description="Attempts recorded for summary.")

    subtitles_responses: dict[int, SubtitlesAiResponse] = Field(
        default_factory=dict, description="Subtitles responses by segment index."
    )
    failed_subtitles: set[int] = Field(default_factory=set, description="Segment indices with exhausted subtitles.")
    subtitles_attempts: dict[int, int] = Field(default_factory=dict, description="Subtitle attempts by segment index.")

    lyrics_responses: dict[int, LyricsAiResponse] = Field(
        default_factory=dict, description="Lyrics responses by segment index."
    )
    failed_lyrics: set[int] = Field(default_factory=set, description="Segment indices with exhausted lyrics.")
    lyrics_attempts: dict[int, int] = Field(default_factory=dict, description="Lyrics attempts by segment index.")

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
    ) -> Self:
        """Asynchronously initializes the pipeline, probes media, splits segments, and discovers on-disk state.

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

        Returns:
            Pipeline: Initialized pipeline instance with media prepared and state discovered.

        Raises:
            RuntimeError: If no video segments could be created.
            ValueError: If start_offset_min exceeds the total video duration.
        """
        raw_resolved = raw_video_path.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

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

        # 3. Apply start_offset_min (rounded down to nearest segment duration)
        if split_settings.start_offset_min > 0:
            start_offset_s = split_settings.start_offset_min * 60
            skip_count = start_offset_s // split_settings.max_seconds
            if skip_count >= len(all_segments):
                raise ValueError(
                    f"start_offset_min ({split_settings.start_offset_min}m) exceeds or equals total "
                    f"video duration ({total_duration_ms / 60000:.1f}m)."
                )
            initial_offset_ms = skip_count * split_settings.max_seconds * 1000
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

        # 4. Evaluate summary proxy re-encoding
        sum_cfg = reencode_settings.summary
        raw_size_mb = raw_resolved.stat().st_size / (1024 * 1024)
        summary_proxy_needed = (
            summary_enabled and sum_cfg.enabled and (sum_cfg.threshold_mb == 0 or raw_size_mb >= sum_cfg.threshold_mb)
        )
        proxy_path = job_dir / "summary_proxy.mp4"
        if not summary_proxy_needed:
            actual_proxy_path: Path | None = raw_resolved
        elif proxy_path.is_file():
            actual_proxy_path = proxy_path
        else:
            actual_proxy_path = None

        # 5. Evaluate segment re-encoding requirements
        seg_cfg = reencode_settings.segment
        reencode_dir = job_dir / "reencoded"
        reencode_dir.mkdir(parents=True, exist_ok=True)
        reencoded_segment_paths: dict[int, Path] = {}
        for idx, seg_p in enumerate(segment_paths):
            file_mb = seg_p.stat().st_size / (1024 * 1024)
            needs_reenc = seg_cfg.enabled and (seg_cfg.threshold_mb == 0 or file_mb >= seg_cfg.threshold_mb)
            if not needs_reenc:
                reencoded_segment_paths[idx] = seg_p
            else:
                out_p = reencode_dir / f"{seg_p.stem}.mp4"
                if out_p.is_file():
                    reencoded_segment_paths[idx] = out_p

        # 6. Generate shortcodes
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
            summary_model=summary_model,
            subtitles_model=subtitles_model,
            lyrics_model=lyrics_model,
            summary_shortcode=summary_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            lyrics_shortcode=lyrics_shortcode,
            full_shortcode=full_shortcode,
            reencode_settings=reencode_settings,
            search_settings=search_settings,
            split_settings=split_settings,
            retry_settings=retry_settings,
            initial_offset_ms=initial_offset_ms,
            segment_paths=segment_paths,
            segment_durations_ms=segment_durations_ms,
            summary_proxy_needed=summary_proxy_needed,
            summary_proxy_path=actual_proxy_path,
            reencoded_segment_paths=reencoded_segment_paths,
        )
        instance._discover_on_disk_state()
        return instance

    def _discover_on_disk_state(self) -> None:
        """Inspects job_dir for existing valid checkpoints and tracks completed or failed stages."""
        max_runs = self.retry_settings.max_runs

        if self.summary_enabled:
            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            loaded_summary = SummaryJob.load(summary_file, self.validation_buffer_ms)
            if loaded_summary:
                self.summary_attempts = loaded_summary.total_attempts
                if loaded_summary.response:
                    self.summary_response = loaded_summary.response
                    logfire.debug(f"Loaded existing summary from {summary_file.name}")
                elif loaded_summary.total_attempts >= max_runs:
                    self.summary_failed = True
                    logfire.warning(f"Summary stage previously exhausted all {max_runs} runs.")

        for idx, seg_path in enumerate(self.segment_paths):
            sub_file = self.job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
            loaded_sub = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            if loaded_sub:
                self.subtitles_attempts[idx] = loaded_sub.total_attempts
                if loaded_sub.response:
                    self.subtitles_responses[idx] = loaded_sub.response

                    if self.lyrics_enabled and loaded_sub.response.detected_musical_performances:
                        lyr_file = self.job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
                        loaded_lyr = LyricsJob.load(lyr_file, self.validation_buffer_ms)
                        if loaded_lyr:
                            self.lyrics_attempts[idx] = loaded_lyr.total_attempts
                            if loaded_lyr.response:
                                self.lyrics_responses[idx] = loaded_lyr.response
                            elif loaded_lyr.total_attempts >= max_runs:
                                self.failed_lyrics.add(idx)
                elif loaded_sub.total_attempts >= max_runs:
                    self.failed_subtitles.add(idx)

    def is_summary_complete(self) -> bool:
        """Checks if the whole-video summary stage is complete or bypassed.

        Returns:
            bool: True if summary is complete or bypassed, False otherwise.
        """
        if not self.summary_enabled:
            return True
        return self.summary_response is not None

    def is_subtitles_complete(self) -> bool:
        """Checks if all subtitle chunks have finished processing (succeeded or exhausted retries).

        Returns:
            bool: True if all chunks have completed subtitles, False otherwise.
        """
        concluded = len(self.subtitles_responses) + len(self.failed_subtitles)
        return concluded >= len(self.segment_paths)

    def is_lyrics_complete(self) -> bool:
        """Checks if all segments with detected musical performances have resolved lyrics.

        Returns:
            bool: True if all singing segments have completed lyrics or lyrics are disabled, False otherwise.
        """
        if not self.lyrics_enabled:
            return True
        if not self.is_subtitles_complete():
            return False
        for idx, sub_resp in self.subtitles_responses.items():
            if sub_resp.detected_musical_performances:
                if idx not in self.lyrics_responses and idx not in self.failed_lyrics:
                    return False
        return True

    def is_done(self) -> bool:
        """Checks if the entire video has completed all pipeline stages and is ready for finalization.

        Returns:
            bool: True if summary, subtitles, and required lyrics are complete, False otherwise.
        """
        # A summary proxy is only required if summary is still pending
        if not self.is_summary_complete() and not self.summary_failed and self.summary_proxy_needed:
            if self.summary_proxy_path is None:
                return False

        if len(self.reencoded_segment_paths) < len(self.segment_paths):
            return False

        if self.summary_enabled and self.summary_failed:
            return True

        return self.is_summary_complete() and self.is_subtitles_complete() and self.is_lyrics_complete()

    def get_pending_jobs(self) -> list[BaseJob]:
        """Evaluates DAG dependencies and returns all jobs currently ready to execute.

        Returns:
            list[BaseJob]: List of ready jobs (ReencodeJob, SummaryJob, SubtitlesJob, LyricsJob).
        """
        jobs: list[BaseJob] = []

        # 1. Summary proxy re-encode job (only if summary still needs to be run)
        if (
            self.summary_enabled
            and not self.is_summary_complete()
            and not self.summary_failed
            and self.summary_proxy_needed
            and self.summary_proxy_path is None
        ):
            sum_cfg = self.reencode_settings.summary
            proxy_dest = self.job_dir / "summary_proxy.mp4"
            jobs.append(
                ReencodeJob(
                    name="summary_proxy",
                    input_file=self.raw_video_path,
                    output_file=proxy_dest,
                    fps=sum_cfg.fps,
                    height=sum_cfg.height,
                    bitrate_kb=sum_cfg.bitrate_kb,
                    duration_tolerance_ms=sum_cfg.duration_tolerance_ms,
                    encoder=self.reencode_settings.encoder,
                    keyframe_only=sum_cfg.keyframe_only,
                    is_summary_proxy=True,
                )
            )

        # 2. Segment chunk re-encode jobs
        if self.reencode_settings.segment.enabled:
            seg_cfg = self.reencode_settings.segment
            reencode_dir = self.job_dir / "reencoded"
            for idx, seg_p in enumerate(self.segment_paths):
                if idx not in self.reencoded_segment_paths:
                    out_p = reencode_dir / f"{seg_p.stem}.mp4"
                    jobs.append(
                        ReencodeJob(
                            name=f"{seg_p.stem}_reencode",
                            input_file=seg_p,
                            output_file=out_p,
                            fps=seg_cfg.fps,
                            height=seg_cfg.height,
                            bitrate_kb=seg_cfg.bitrate_kb,
                            duration_tolerance_ms=seg_cfg.duration_tolerance_ms,
                            encoder=self.reencode_settings.encoder,
                            keyframe_only=seg_cfg.keyframe_only,
                            segment_index=idx,
                            is_summary_proxy=False,
                        )
                    )

        # 3. Whole-video Summary job
        if self.summary_enabled and not self.is_summary_complete() and not self.summary_failed:
            if self.summary_proxy_path is not None:
                jobs.append(
                    SummaryJob(
                        name="summary",
                        video_duration_ms=self.duration_ms,
                        user_context=self.user_context,
                        media_path=self.summary_proxy_path,
                        model_name=self.summary_model,
                        retry_settings=self.retry_settings,
                        total_attempts=self.summary_attempts,
                    )
                )
            return jobs

        if self.summary_enabled and self.summary_failed:
            return jobs

        # 4. Subtitle transcription and lyrics refinement jobs
        current_offset_ms = self.initial_offset_ms
        for idx, (seg_path, seg_dur_ms) in enumerate(zip(self.segment_paths, self.segment_durations_ms, strict=True)):
            media_file = self.reencoded_segment_paths.get(idx)
            if media_file is None:
                current_offset_ms += seg_dur_ms
                continue

            if idx not in self.subtitles_responses and idx not in self.failed_subtitles:
                jobs.append(
                    SubtitlesJob(
                        name=seg_path.stem,
                        video_duration_ms=seg_dur_ms,
                        segment_offset_ms=current_offset_ms,
                        global_summary=self.summary_response,
                        user_context=self.user_context,
                        media_path=media_file,
                        model_name=self.subtitles_model,
                        retry_settings=self.retry_settings,
                        segment_index=idx,
                        total_attempts=self.subtitles_attempts.get(idx, 0),
                    )
                )
            elif self.lyrics_enabled and idx in self.subtitles_responses:
                sub_resp = self.subtitles_responses[idx]
                if (
                    sub_resp.detected_musical_performances
                    and idx not in self.lyrics_responses
                    and idx not in self.failed_lyrics
                ):
                    jobs.append(
                        LyricsJob(
                            name=seg_path.stem,
                            video_duration_ms=seg_dur_ms,
                            segment_offset_ms=current_offset_ms,
                            draft_subtitles=sub_resp.subtitles,
                            detected_performances=sub_resp.detected_musical_performances,
                            global_summary=self.summary_response,
                            user_context=self.user_context,
                            media_path=media_file,
                            model_name=self.lyrics_model,
                            retry_settings=self.retry_settings,
                            segment_index=idx,
                            total_attempts=self.lyrics_attempts.get(idx, 0),
                        )
                    )

            current_offset_ms += seg_dur_ms

        return jobs

    def complete_job(
        self,
        job_or_name: BaseJob | str,
        result_data: Any = None,
        segment_index: int | None = None,
    ) -> None:
        """Validates job output, persists checkpoints to disk, and updates session state.

        Args:
            job_or_name: The completed Job instance or string name identifier.
            result_data: Output data returned by the executor (model response, dictionary, or Path).
            segment_index: Chunk index if applicable. Defaults to None.
        """
        job_name = job_or_name.name if isinstance(job_or_name, BaseJob) else str(job_or_name)

        # Handle ReencodeJob (by instance or string name)
        is_reencode = isinstance(job_or_name, ReencodeJob) or (
            isinstance(job_or_name, str) and ("reencode" in job_name or "proxy" in job_name)
        )
        if is_reencode:
            out_path = result_data if isinstance(result_data, Path) else getattr(job_or_name, "output_file", None)
            is_proxy = getattr(job_or_name, "is_summary_proxy", False) or "proxy" in job_name
            idx = segment_index if segment_index is not None else getattr(job_or_name, "segment_index", None)

            if is_proxy and out_path:
                self.summary_proxy_path = out_path
            elif idx is not None and out_path:
                self.reencoded_segment_paths[idx] = out_path
            return

        val_ctx: dict[str, Any] = {"validation_buffer_ms": self.validation_buffer_ms}

        is_summary = isinstance(job_or_name, SummaryJob) or job_name == "summary"
        is_lyrics = isinstance(job_or_name, LyricsJob) or "lyrics" in job_name
        is_subtitles = (
            isinstance(job_or_name, SubtitlesJob)
            or "subtitles" in job_name
            or job_name.startswith("part_")
            or not (is_summary or is_lyrics)
        )

        if is_summary:
            val_ctx["video_duration_ms"] = self.duration_ms
            if isinstance(result_data, SummaryAiResponse):
                resp = result_data
            else:
                resp = SummaryAiResponse.model_validate(result_data, context=val_ctx)
            self.summary_response = resp
            self.summary_failed = False

            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            existing = SummaryJob.load(summary_file, self.validation_buffer_ms)
            attempts = max(existing.total_attempts if existing else 0, self.summary_attempts, 1)
            self.summary_attempts = attempts

            summary_job = SummaryJob(
                name="summary",
                video_duration_ms=self.duration_ms,
                user_context=self.user_context,
                total_attempts=attempts,
                response=resp,
            )
            summary_job.save(summary_file)

        elif is_lyrics:
            idx = segment_index if segment_index is not None else getattr(job_or_name, "segment_index", None)
            assert idx is not None
            seg_dur_ms = self.segment_durations_ms[idx]
            val_ctx["video_duration_ms"] = seg_dur_ms

            payload_lyr = (
                result_data.model_dump(mode="json") if isinstance(result_data, LyricsAiResponse) else result_data
            )
            resp_lyr = LyricsAiResponse.model_validate(payload_lyr, context=val_ctx)

            self.lyrics_responses[idx] = resp_lyr
            self.failed_lyrics.discard(idx)

            seg_path = self.segment_paths[idx]
            lyr_file = self.job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
            existing = LyricsJob.load(lyr_file, self.validation_buffer_ms)
            attempts = max(existing.total_attempts if existing else 0, self.lyrics_attempts.get(idx, 0), 1)
            self.lyrics_attempts[idx] = attempts

            offset_ms = self.initial_offset_ms + sum(self.segment_durations_ms[:idx])
            sub_resp = self.subtitles_responses.get(idx)
            lyrics_job = LyricsJob(
                name=seg_path.stem,
                video_duration_ms=seg_dur_ms,
                segment_offset_ms=offset_ms,
                draft_subtitles=sub_resp.subtitles if sub_resp else [],
                detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
                global_summary=self.summary_response,
                user_context=self.user_context,
                total_attempts=attempts,
                response=resp_lyr,
            )
            lyrics_job.save(lyr_file)

        elif is_subtitles:
            idx = segment_index if segment_index is not None else getattr(job_or_name, "segment_index", None)
            assert idx is not None
            seg_dur_ms = self.segment_durations_ms[idx]
            val_ctx["video_duration_ms"] = seg_dur_ms

            payload_sub = (
                result_data.model_dump(mode="json") if isinstance(result_data, SubtitlesAiResponse) else result_data
            )
            resp_sub = SubtitlesAiResponse.model_validate(payload_sub, context=val_ctx)

            self.subtitles_responses[idx] = resp_sub
            self.failed_subtitles.discard(idx)

            seg_path = self.segment_paths[idx]
            sub_file = self.job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
            existing = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            attempts = max(existing.total_attempts if existing else 0, self.subtitles_attempts.get(idx, 0), 1)
            self.subtitles_attempts[idx] = attempts

            offset_ms = self.initial_offset_ms + sum(self.segment_durations_ms[:idx])
            sub_job = SubtitlesJob(
                name=seg_path.stem,
                video_duration_ms=seg_dur_ms,
                segment_offset_ms=offset_ms,
                global_summary=self.summary_response,
                user_context=self.user_context,
                total_attempts=attempts,
                response=resp_sub,
            )
            sub_job.save(sub_file)

    def record_failure(
        self,
        task_type: TaskType | str,
        segment_index: int | None = None,
        attempts: int = 1,
    ) -> None:
        """Persists task failure attempt counts to on-disk YAML checkpoints.

        Args:
            task_type: Type of task that failed.
            segment_index: Chunk index if applicable. Defaults to None.
            attempts: Total attempts recorded so far. Defaults to 1.
        """
        max_runs = self.retry_settings.max_runs
        clean_type = task_type.value if isinstance(task_type, TaskType) else str(task_type).lower()

        if clean_type == "summary":
            summary_file = self.job_dir / f"summary.{self.summary_shortcode}.yaml"
            existing = SummaryJob.load(summary_file, self.validation_buffer_ms)
            if existing and existing.response is not None:
                return

            effective_attempts = max(existing.total_attempts if existing else 0, self.summary_attempts, attempts)
            self.summary_attempts = effective_attempts
            if effective_attempts >= max_runs:
                self.summary_failed = True

            summary_job = SummaryJob(
                name="summary",
                video_duration_ms=self.duration_ms,
                user_context=self.user_context,
                total_attempts=effective_attempts,
                response=None,
            )
            summary_job.save(summary_file)

        elif clean_type == "subtitles":
            assert segment_index is not None
            idx = segment_index
            seg_path = self.segment_paths[idx]
            sub_file = self.job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
            existing = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
            if existing and existing.response is not None:
                return

            effective_attempts = max(
                existing.total_attempts if existing else 0,
                self.subtitles_attempts.get(idx, 0),
                attempts,
            )
            self.subtitles_attempts[idx] = effective_attempts
            if effective_attempts >= max_runs:
                self.failed_subtitles.add(idx)

            offset_ms = self.initial_offset_ms + sum(self.segment_durations_ms[:idx])
            sub_job = SubtitlesJob(
                name=seg_path.stem,
                video_duration_ms=self.segment_durations_ms[idx],
                segment_offset_ms=offset_ms,
                global_summary=self.summary_response,
                user_context=self.user_context,
                total_attempts=effective_attempts,
                response=None,
            )
            sub_job.save(sub_file)

        elif clean_type == "lyrics":
            assert segment_index is not None
            idx = segment_index
            seg_path = self.segment_paths[idx]
            lyr_file = self.job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
            existing = LyricsJob.load(lyr_file, self.validation_buffer_ms)
            if existing and existing.response is not None:
                return

            effective_attempts = max(
                existing.total_attempts if existing else 0,
                self.lyrics_attempts.get(idx, 0),
                attempts,
            )
            self.lyrics_attempts[idx] = effective_attempts
            if effective_attempts >= max_runs:
                self.failed_lyrics.add(idx)

            offset_ms = self.initial_offset_ms + sum(self.segment_durations_ms[:idx])
            sub_resp = self.subtitles_responses.get(idx)
            lyrics_job = LyricsJob(
                name=seg_path.stem,
                video_duration_ms=self.segment_durations_ms[idx],
                segment_offset_ms=offset_ms,
                draft_subtitles=sub_resp.subtitles if sub_resp else [],
                detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
                global_summary=self.summary_response,
                user_context=self.user_context,
                total_attempts=effective_attempts,
                response=None,
            )
            lyrics_job.save(lyr_file)

    def _derive_status(self) -> str:
        """Derives completion status based on session progress and recorded failures.

        Returns:
            str: 'RetriesExceeded' if any stage failed, 'Completed' if finished, or 'InProgress'.
        """
        if self.summary_failed or self.failed_subtitles or (self.lyrics_enabled and self.failed_lyrics):
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
                "search": search_dump,
                "agentic_video_summary": True,
                "agentic_video_subtitles": False,
                "agentic_video_lyrics": True,
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

        segments = [
            (seg_path.stem, seg_dur_ms)
            for seg_path, seg_dur_ms in zip(self.segment_paths, self.segment_durations_ms, strict=True)
        ]

        result = stitch_subtitles(
            segments=segments,
            state_dir=self.job_dir,
            output_primary_path=primary_path,
            output_raw_path=raw_path,
            subtitles_shortcode=self.subtitles_shortcode,
            lyrics_shortcode=self.lyrics_shortcode,
            validation_buffer_ms=self.validation_buffer_ms,
            settings_dump=self._build_settings_metadata(),
            max_runs=self.retry_settings.max_runs,
            status_override=status_override or self._derive_status(),
            summary_available=self.summary_response is not None,
            initial_offset_ms=self.initial_offset_ms,
            use_lyrics=self.lyrics_enabled,
        )
        return result.generated_files
