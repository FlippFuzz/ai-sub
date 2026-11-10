"""Pipeline coordinator managing 1 active video lifecycle, task creation, and state resumption."""

from __future__ import annotations

import asyncio
import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import logfire
from yt_dlp.utils import sanitize_filename

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
)
from ai_sub.jobs.summary import SummaryJob
from ai_sub.pipeline_session import PipelineSession
from ai_sub.protocol import TaskType
from ai_sub.server.config import PipelineConfig
from ai_sub.server.models import ActiveVideo, ServerTask, TaskStatus
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_subtitles_shortcode,
    generate_summary_shortcode,
)
from ai_sub.video import (
    get_video_duration_ms,
    reencode_video,
    split_video,
)


class PipelineStatus(str, Enum):
    """Lifecycle execution status of a combination pipeline."""

    IDLE = "idle"
    PREPARING = "preparing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    PUSH_FAILED = "push_failed"


class ComboPipeline:
    """Manages an independent combination pipeline restricted to 1 active video."""

    config: PipelineConfig
    base_work_dir: Path
    work_dir: Path
    validation_buffer_ms: int
    reencode_settings: ReEncodeSettings
    search_settings: SearchSettings
    split_settings: SplittingSettings
    retry_settings: RetrySettings

    summary_shortcode: str
    subtitles_shortcode: str
    lyrics_shortcode: str
    full_shortcode: str

    status: PipelineStatus
    active_video: ActiveVideo | None
    preparing_video_info: dict[str, Any] | None
    tasks: dict[str, ServerTask]
    _reencode_task: asyncio.Task[None] | None

    def __init__(
        self,
        config: PipelineConfig,
        base_work_dir: Path,
        reencode_settings: ReEncodeSettings,
        search_settings: SearchSettings,
        split_settings: SplittingSettings,
        retry_settings: RetrySettings,
        validation_buffer_ms: int = 2000,
    ) -> None:
        """Initializes the combination pipeline runner.

        Args:
            config: Model pipeline configuration.
            base_work_dir: Root working directory containing per-pipeline folders.
            reencode_settings: Video compression and proxy rules.
            search_settings: Search tool configuration.
            split_settings: Video chunk duration and tolerance rules.
            retry_settings: Task attempt limits and backoff rules.
            validation_buffer_ms: Allowed timestamp tolerance in milliseconds. Defaults to 2000.
        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.work_dir = base_work_dir / config.name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.validation_buffer_ms = validation_buffer_ms

        self.reencode_settings = reencode_settings
        self.search_settings = search_settings
        self.split_settings = split_settings
        self.retry_settings = retry_settings

        self.summary_shortcode = generate_summary_shortcode(config.summary_model)
        self.subtitles_shortcode = generate_subtitles_shortcode(
            model_name=config.subtitles_model,
            summary_shortcode=self.summary_shortcode,
        )
        self.lyrics_shortcode = generate_lyrics_shortcode(
            lyrics_model_name=config.lyrics_model,
            subtitles_shortcode=self.subtitles_shortcode,
        )
        self.full_shortcode = generate_full_shortcode(config.subtitles_model)

        self.status = PipelineStatus.IDLE
        self.active_video = None
        self.preparing_video_info = None
        self.tasks = {}
        self._reencode_task = None

    @property
    def summary_response(self) -> Any:
        """Summary response from the active session if available."""
        if self.active_video and self.active_video.session:
            return self.active_video.session.summary_response
        return None

    def is_idle(self) -> bool:
        """Checks if the pipeline is idle and ready to accept a new video.

        Returns:
            bool: True if pipeline status is IDLE, False otherwise.
        """
        return self.status == PipelineStatus.IDLE

    def set_preparing(
        self,
        video_id: str,
        title: str,
        uploader: str,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "youtube",
    ) -> None:
        """Marks pipeline as PREPARING immediately while media is being downloaded or chunked.

        Args:
            video_id: Unique video identifier.
            title: Title of the video.
            uploader: Channel or uploader handle identifier.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'youtube'.
        """
        self.status = PipelineStatus.PREPARING
        self.preparing_video_info = {
            "video_id": video_id,
            "title": title,
            "uploader": uploader,
            "channel_name": channel_name,
            "source_type": source_type,
        }

    def clear_preparing(self) -> None:
        """Resets preparing state back to IDLE if download or chunk generation fails."""
        if not self.active_video:
            self.status = PipelineStatus.IDLE
        self.preparing_video_info = None

    def is_task_ready_for_claim(self, task: ServerTask) -> bool:
        """Determines if a task is eligible to be claimed by a worker.

        Args:
            task: Candidate server task.

        Returns:
            bool: True if task is ready for pickup, False otherwise.
        """
        if task.task_type == TaskType.SUBTITLES and self.reencode_settings.segment.enabled:
            if task.segment_index is not None and self.active_video is not None:
                return task.segment_index in self.active_video.reencoded_segment_paths
        return True

    def _build_user_context(self, active: ActiveVideo | None = None) -> str | None:
        """Constructs background user context from video metadata.

        Args:
            active: Optional active video model. Defaults to self.active_video.

        Returns:
            str | None: Formatted context string or None.
        """
        video = active or self.active_video
        if not video:
            return None
        if video.source_type == "youtube":
            ch = video.channel_name or video.uploader
            return f"This is a YouTube video from {ch} channel. The video title is '{video.title}'."
        return f"The video title is '{video.title}'."

    def _get_active_search_tools(self) -> list[str]:
        """Resolves active search tools from pipeline override or global search settings.

        Returns:
            list[str]: Active search tools list.
        """
        return self.config.search_tools if self.config.search_tools is not None else self.search_settings.tools

    async def _reencode_segments_background(self) -> None:
        """Background loop re-encoding segments sequentially using the serial FFmpeg lock."""
        if not self.active_video:
            return

        active = self.active_video
        reencode_dir = active.job_dir / "reencoded"
        reencode_dir.mkdir(parents=True, exist_ok=True)
        seg_cfg = self.reencode_settings.segment

        # If segment re-encoding is disabled, mark all segments as ready immediately
        if not seg_cfg.enabled:
            for idx, seg_path in enumerate(active.segment_paths):
                active.reencoded_segment_paths[idx] = seg_path
            return

        with logfire.span(
            "Background Re-encoding Loop: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            enabled=seg_cfg.enabled,
            threshold_mb=seg_cfg.threshold_mb,
            total_chunks=len(active.segment_paths),
            _level="debug",
        ) as loop_span:
            reencoded_count = 0
            skipped_count = 0

            for idx, seg_path in enumerate(active.segment_paths):
                # Skip if already marked ready (e.g. below size threshold)
                if idx in active.reencoded_segment_paths:
                    skipped_count += 1
                    continue

                out_file = reencode_dir / f"{seg_path.stem}.mp4"
                if out_file.is_file():
                    active.reencoded_segment_paths[idx] = out_file
                    reencoded_count += 1
                    continue

                file_size_mb = seg_path.stat().st_size / (1024 * 1024)
                should_reencode = seg_cfg.threshold_mb == 0 or file_size_mb >= seg_cfg.threshold_mb

                if not should_reencode:
                    active.reencoded_segment_paths[idx] = seg_path
                    skipped_count += 1
                    continue

                with logfire.span(
                    "Re-encoding Chunk: {chunk}",
                    chunk=seg_path.name,
                    combo=self.config.name,
                    video_id=active.video_id,
                    size_mb=round(file_size_mb, 2),
                    _level="debug",
                ) as chunk_span:
                    try:
                        await reencode_video(
                            input_path=seg_path,
                            output_path=out_file,
                            fps=seg_cfg.fps,
                            height=seg_cfg.height,
                            bitrate_kb=seg_cfg.bitrate_kb,
                            encoder=self.reencode_settings.encoder or "libx264",
                            duration_tolerance_ms=seg_cfg.duration_tolerance_ms,
                            keyframe_only=seg_cfg.keyframe_only,
                        )
                        active.reencoded_segment_paths[idx] = out_file
                        reencoded_count += 1
                        out_size_mb = out_file.stat().st_size / (1024 * 1024)
                        chunk_span.set_attribute("result", "reencoded")
                        chunk_span.set_attribute("output_size_mb", round(out_size_mb, 2))

                        task_id = f"{self.config.name}::{active.video_id}::subtitles::{idx:03d}"
                        if task_id in self.tasks:
                            self.tasks[task_id].media_path = out_file
                    except Exception as e:
                        chunk_span.set_attribute("result", "failed")
                        chunk_span.set_attribute("error", str(e))
                        logfire.warning(f"Re-encode failed for chunk {seg_path.name}: {e}. Falling back to raw chunk.")
                        active.reencoded_segment_paths[idx] = seg_path

            loop_span.set_attribute("reencoded_chunks", reencoded_count)
            loop_span.set_attribute("skipped_chunks", skipped_count)

    def _sync_tasks_from_session(self) -> None:
        """Synchronizes pending task specifications from PipelineSession into ServerTask instances."""
        active = self.active_video
        session = active.session if active else None
        if not active or session is None:
            return

        pending_specs = session.get_pending_task_specs()
        for spec in pending_specs:
            full_task_id = f"{self.config.name}::{spec.task_id}"
            if full_task_id in self.tasks:
                if spec.segment_index is not None and spec.segment_index in active.reencoded_segment_paths:
                    self.tasks[full_task_id].media_path = active.reencoded_segment_paths[spec.segment_index]
                continue

            if spec.task_type == TaskType.SUMMARY:
                model = self.config.summary_model
                tier = self.config.service_tier_summary
                tools = [t for t in self._get_active_search_tools() if t != "lrclib"]
                agentic = True
                prior_attempts = session.summary_attempts
            elif spec.task_type == TaskType.SUBTITLES:
                model = self.config.subtitles_model
                tier = self.config.service_tier_subtitles
                tools = []
                agentic = False
                prior_attempts = (
                    session.subtitles_attempts.get(spec.segment_index, 0) if spec.segment_index is not None else 0
                )
            else:  # TaskType.LYRICS
                model = self.config.lyrics_model
                tier = self.config.service_tier_lyrics
                tools = self._get_active_search_tools()
                agentic = True
                prior_attempts = (
                    session.lyrics_attempts.get(spec.segment_index, 0) if spec.segment_index is not None else 0
                )

            self.tasks[full_task_id] = ServerTask(
                task_id=full_task_id,
                task_type=spec.task_type,
                pipeline_name=self.config.name,
                video_id=active.video_id,
                segment_index=spec.segment_index,
                model=model,
                service_tier=tier,
                media_path=spec.media_path,
                video_duration_ms=spec.video_duration_ms,
                segment_offset_ms=spec.segment_offset_ms,
                user_context=spec.user_context,
                global_summary=spec.global_summary,
                draft_subtitles=spec.draft_subtitles,
                detected_performances=spec.detected_performances,
                search_tools=tools,
                agentic_video=agentic,
                attempts=prior_attempts,
                status=TaskStatus.PENDING,
            )

    async def prepare_and_set_video(
        self,
        video_id: str,
        uploader: str,
        title: str,
        raw_video_path: Path,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "local",
        timestamp: int = 0,
    ) -> None:
        """Prepares a video inside work/<combo>/<video_id>/, saving metadata and splitting chunks.

        Args:
            video_id: Unique video identifier.
            uploader: Channel or uploader handle identifier.
            title: Title of the video.
            raw_video_path: Path to the local raw video file in this combo directory.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication or intake Unix timestamp. Defaults to 0.
        """
        self.status = PipelineStatus.PREPARING
        self.preparing_video_info = {
            "video_id": video_id,
            "uploader": uploader,
            "channel_name": channel_name,
            "title": title,
            "source_type": source_type,
        }
        job_dir = self.work_dir / video_id
        job_dir.mkdir(parents=True, exist_ok=True)
        filename_stem = sanitize_filename(raw_video_path.stem)

        with logfire.span(
            "Prepare Video: {video_id}",
            combo=self.config.name,
            video_id=video_id,
            title=title,
            uploader=uploader,
        ) as prep_span:
            total_duration_ms = await get_video_duration_ms(raw_video_path)
            raw_size_mb = raw_video_path.stat().st_size / (1024 * 1024)

            meta_file = job_dir / "video_meta.json"
            meta_data = {
                "video_id": video_id,
                "uploader": uploader,
                "channel_name": channel_name,
                "title": title,
                "filename_stem": filename_stem,
                "raw_filename": raw_video_path.name,
                "raw_path": str(raw_video_path.resolve()),
                "source_type": source_type,
                "timestamp": timestamp,
                "duration_ms": total_duration_ms,
                "raw_size_mb": round(raw_size_mb, 2),
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)

            sum_cfg = self.reencode_settings.summary
            should_reencode_summary = sum_cfg.enabled and (
                sum_cfg.threshold_mb == 0 or raw_size_mb >= sum_cfg.threshold_mb
            )

            with logfire.span(
                "Check Summary Proxy: {video_id}",
                combo=self.config.name,
                video_id=video_id,
                raw_size_mb=round(raw_size_mb, 2),
                threshold_mb=sum_cfg.threshold_mb,
                enabled=sum_cfg.enabled,
                _level="debug",
            ) as proxy_span:
                if should_reencode_summary:
                    proxy_path = job_dir / "summary_proxy.mp4"
                    await reencode_video(
                        input_path=raw_video_path,
                        output_path=proxy_path,
                        fps=sum_cfg.fps,
                        height=sum_cfg.height,
                        bitrate_kb=sum_cfg.bitrate_kb,
                        encoder=self.reencode_settings.encoder or "libx264",
                        duration_tolerance_ms=sum_cfg.duration_tolerance_ms,
                        keyframe_only=sum_cfg.keyframe_only,
                    )
                    proxy_size_mb = proxy_path.stat().st_size / (1024 * 1024)
                    proxy_span.set_attribute("result", "reencoded")
                    proxy_span.set_attribute("proxy_size_mb", round(proxy_size_mb, 2))
                else:
                    proxy_path = raw_video_path
                    proxy_span.set_attribute("result", "skipped")

            with logfire.span(
                "Split Video into Segments: {video_id}",
                combo=self.config.name,
                video_id=video_id,
                split_duration_s=self.split_settings.max_seconds,
            ) as split_span:
                segment_paths = await split_video(
                    input_video=raw_video_path,
                    output_dir=job_dir,
                    split_duration_s=self.split_settings.max_seconds,
                    output_pattern="part_%03d",
                    duration_tolerance_ms=self.split_settings.duration_tolerance_ms,
                )
                durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_paths))
                split_span.set_attribute("chunks_created", len(segment_paths))

            # Pre-populate reencoded_segment_paths for chunks that do not require re-encoding
            seg_cfg = self.reencode_settings.segment
            initial_reencoded_paths: dict[int, Path] = {}
            for idx, p in enumerate(segment_paths):
                file_mb = p.stat().st_size / (1024 * 1024)
                needs_reenc = seg_cfg.enabled and (seg_cfg.threshold_mb == 0 or file_mb >= seg_cfg.threshold_mb)
                if not needs_reenc:
                    initial_reencoded_paths[idx] = p

            active = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                filename_stem=filename_stem,
                source_type=source_type,
                timestamp=timestamp,
                job_dir=job_dir,
                raw_path=raw_video_path,
                duration_ms=total_duration_ms,
                summary_proxy_path=proxy_path,
                segment_paths=segment_paths,
                reencoded_segment_paths=initial_reencoded_paths,
                segment_durations_ms=durations,
            )
            active.user_context = self._build_user_context(active)

            # Initialize session sharing the exact same dictionary in memory
            session = PipelineSession(
                job_dir=job_dir,
                video_id=video_id,
                title=title,
                uploader=uploader,
                channel_name=channel_name,
                duration_ms=total_duration_ms,
                segment_paths=segment_paths,
                segment_durations_ms=durations,
                summary_model=self.config.summary_model,
                subtitles_model=self.config.subtitles_model,
                lyrics_model=self.config.lyrics_model,
                reencode_settings=self.reencode_settings,
                search_settings=self.search_settings,
                split_settings=self.split_settings,
                retry_settings=self.retry_settings,
                source_type=source_type,
                timestamp=timestamp,
                user_context=active.user_context,
                validation_buffer_ms=self.validation_buffer_ms,
                summary_proxy_path=proxy_path,
                reencoded_segment_paths=active.reencoded_segment_paths,
            )
            active.session = session
            self.active_video = active

            self.tasks.clear()
            self.preparing_video_info = None
            self.status = PipelineStatus.PROCESSING

            self._sync_tasks_from_session()
            self._reencode_task = asyncio.create_task(self._reencode_segments_background())

            prep_span.set_attribute("result", "ready")
            prep_span.set_attribute("chunk_count", len(segment_paths))

    def on_task_completed(self, task_id: str, result_data: dict[str, Any]) -> None:
        """Handles task completion, validating output and updating session checkpoints.

        Args:
            task_id: Unique task identifier.
            result_data: Validated structured dictionary received from worker.
        """
        task = self.tasks.get(task_id)
        active = self.active_video
        session = active.session if active else None
        if not task or not active or session is None:
            return

        with logfire.span(
            "Task Cascade Complete: {task_id}",
            combo=self.config.name,
            task_id=task_id,
            task_type=task.task_type.value,
        ):
            session.complete_task(
                task_type=task.task_type,
                result_data=result_data,
                segment_index=task.segment_index,
            )
            task.status = TaskStatus.COMPLETED
            task.result = result_data
            self._sync_tasks_from_session()

    def on_task_failed(self, task_id: str, attempts: int) -> None:
        """Handles task failure, updating session state and persisting attempt counts to disk.

        Args:
            task_id: Unique task identifier.
            attempts: Total attempts recorded so far for this task.
        """
        task = self.tasks.get(task_id)
        active = self.active_video
        session = active.session if active else None
        if not task or not active or session is None:
            return

        session.record_task_failure(
            task_type=task.task_type,
            segment_index=task.segment_index,
            attempts=attempts,
        )

    def is_video_complete(self) -> bool:
        """Checks if all segments for the active video have finished processing (succeeded or failed).

        Returns:
            bool: True if video processing has concluded across all stages, False otherwise.
        """
        if not self.active_video or self.status == PipelineStatus.PREPARING:
            return False

        if self.status in (PipelineStatus.FINALIZING, PipelineStatus.PUSH_FAILED):
            return True

        summary_task_id = f"{self.config.name}::{self.active_video.video_id}::summary"
        summary_task = self.tasks.get(summary_task_id)
        if summary_task and summary_task.status == TaskStatus.FAILED:
            return True

        # If any generated task is still actively being processed, the video is not complete
        if any(t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED) for t in self.tasks.values()):
            return False

        session = self.active_video.session
        if session is None:
            return False

        # Summary must either be completed or aborted
        if not session.is_summary_complete() and not session.summary_failed:
            return False

        # All subtitle chunks must have tasks generated and concluded
        sub_tasks = [t for t in self.tasks.values() if t.task_type == TaskType.SUBTITLES]
        if len(sub_tasks) < len(self.active_video.segment_paths):
            return False

        return True

    def stitch_final_subtitles(self, output_path: Path) -> list[Path]:
        """Stitches completed segment subtitles into unified SRT files.

        Args:
            output_path: Destination path for the primary stitched SRT file.

        Returns:
            list[Path]: List of generated subtitle file paths.

        Raises:
            RuntimeError: If video processing is incomplete or active video is missing.
        """
        active = self.active_video
        session = active.session if active else None
        if active is None or session is None or not self.is_video_complete():
            raise RuntimeError(f"Cannot stitch subtitles: {self.config.name} video processing is incomplete.")

        raw_output_path = output_path.parent / f"{active.filename_stem}.raw-{self.full_shortcode}.srt"
        has_failures = any(t.status == TaskStatus.FAILED for t in self.tasks.values())
        status_str = "RetriesExceeded" if has_failures else "Completed"

        with logfire.span(
            "Stitch Subtitles: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            output_file=output_path.name,
        ):
            return session.stitch(
                output_primary_path=output_path,
                output_raw_path=raw_output_path,
                status_override=status_str,
                allow_partial=True,
            )

    async def resume_from_disk(self) -> bool:
        """Inspects work/<combo>/ to restore in-progress video subdirectories upon server boot.

        Returns:
            bool: True if an active video was successfully restored, False otherwise.
        """
        valid_extensions = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}

        with logfire.span("Disk Resume Check: {combo}", combo=self.config.name) as resume_span:
            job_dirs = [d for d in self.work_dir.iterdir() if d.is_dir() and d.name != "reencoded"]
            if not job_dirs:
                self.status = PipelineStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "idle")
                return False

            job_dir = job_dirs[0]
            meta_file = job_dir / "video_meta.json"
            if not meta_file.is_file():
                shutil.rmtree(job_dir, ignore_errors=True)
                self.status = PipelineStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "purged_invalid_folder")
                return False

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                video_id = str(meta.get("video_id", job_dir.name))
                uploader = str(meta.get("uploader", "unknown"))
                channel_name = meta.get("channel_name")
                title = str(meta.get("title", job_dir.name))
                filename_stem = str(meta.get("filename_stem", ""))
                meta_duration_ms = int(meta.get("duration_ms", 0))
                source_type = meta.get("source_type", "youtube")
                timestamp = int(meta.get("timestamp", 0))
            except Exception:
                self.status = PipelineStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "corrupt_metadata")
                return False

            segment_files = sorted(
                [p for p in job_dir.glob("part_*") if p.is_file() and p.suffix.lower() in valid_extensions]
            )
            raw_meta_str = str(meta.get("raw_path", ""))
            raw_meta = Path(raw_meta_str) if raw_meta_str else None
            raw_video_path = raw_meta if raw_meta and raw_meta.is_file() else None

            # Validate whether existing segment files cover the full original video duration
            durations: list[int] = []
            if segment_files:
                durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_files))
                total_seg_duration = sum(durations)
                tolerance = self.split_settings.duration_tolerance_ms * len(segment_files)

                if meta_duration_ms > 0 and abs(total_seg_duration - meta_duration_ms) > tolerance:
                    if raw_video_path:
                        logfire.warning(f"Incomplete split detected for {video_id} on resume. Re-splitting...")
                        for p in segment_files:
                            try:
                                p.unlink()
                            except OSError:
                                pass
                        # Clear stale re-encoded media and outdated segment checkpoints
                        shutil.rmtree(job_dir / "reencoded", ignore_errors=True)
                        for stale_yaml in job_dir.glob("part_*.yaml"):
                            try:
                                stale_yaml.unlink()
                            except OSError:
                                pass
                        for stale_srt in job_dir.glob("part_*.srt"):
                            try:
                                stale_srt.unlink()
                            except OSError:
                                pass

                        segment_files = await split_video(
                            input_video=raw_video_path,
                            output_dir=job_dir,
                            split_duration_s=self.split_settings.max_seconds,
                            output_pattern="part_%03d",
                            duration_tolerance_ms=self.split_settings.duration_tolerance_ms,
                        )
                        durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_files))
                    else:
                        logfire.warning(f"Incomplete split detected for {video_id} but raw video missing. Purging job.")
                        shutil.rmtree(job_dir, ignore_errors=True)
                        self.status = PipelineStatus.IDLE
                        self.preparing_video_info = None
                        return False
            elif raw_video_path:
                shutil.rmtree(job_dir / "reencoded", ignore_errors=True)
                for stale_yaml in job_dir.glob("part_*.yaml"):
                    try:
                        stale_yaml.unlink()
                    except OSError:
                        pass
                for stale_srt in job_dir.glob("part_*.srt"):
                    try:
                        stale_srt.unlink()
                    except OSError:
                        pass

                segment_files = await split_video(
                    input_video=raw_video_path,
                    output_dir=job_dir,
                    split_duration_s=self.split_settings.max_seconds,
                    output_pattern="part_%03d",
                    duration_tolerance_ms=self.split_settings.duration_tolerance_ms,
                )
                durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_files))
            else:
                shutil.rmtree(job_dir, ignore_errors=True)
                self.status = PipelineStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "no_chunks_or_raw")
                return False

            summary_checkpoint = SummaryJob.load(
                job_dir / f"summary.{self.summary_shortcode}.yaml", self.validation_buffer_ms
            )
            summary_concluded = bool(
                summary_checkpoint
                and (summary_checkpoint.response or summary_checkpoint.total_attempts >= self.retry_settings.max_runs)
            )

            proxy_path = job_dir / "summary_proxy.mp4"
            if not proxy_path.is_file():
                if raw_video_path:
                    proxy_path = raw_video_path
                else:
                    if not summary_concluded:
                        logfire.warning(
                            f"Summary pending for {video_id} but raw media and proxy are missing. Purging stale job."
                        )
                        shutil.rmtree(job_dir, ignore_errors=True)
                        self.status = PipelineStatus.IDLE
                        self.preparing_video_info = None
                        return False
                    proxy_path = segment_files[0]

            if not filename_stem:
                filename_stem = sanitize_filename(segment_files[0].stem)

            reencode_dir = job_dir / "reencoded"
            seg_cfg = self.reencode_settings.segment
            reencoded_paths: dict[int, Path] = {}
            for idx, p in enumerate(segment_files):
                out_p = reencode_dir / f"{p.stem}.mp4"
                if out_p.is_file():
                    reencoded_paths[idx] = out_p
                else:
                    file_mb = p.stat().st_size / (1024 * 1024)
                    needs_reenc = seg_cfg.enabled and (seg_cfg.threshold_mb == 0 or file_mb >= seg_cfg.threshold_mb)
                    if not needs_reenc:
                        reencoded_paths[idx] = p

            active = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                filename_stem=filename_stem,
                source_type=source_type,
                timestamp=timestamp,
                job_dir=job_dir,
                raw_path=raw_video_path or segment_files[0],
                duration_ms=sum(durations),
                summary_proxy_path=proxy_path,
                segment_paths=segment_files,
                reencoded_segment_paths=reencoded_paths,
                segment_durations_ms=durations,
            )
            active.user_context = self._build_user_context(active)

            session = PipelineSession(
                job_dir=job_dir,
                video_id=video_id,
                title=title,
                uploader=uploader,
                channel_name=channel_name,
                duration_ms=sum(durations),
                segment_paths=segment_files,
                segment_durations_ms=durations,
                summary_model=self.config.summary_model,
                subtitles_model=self.config.subtitles_model,
                lyrics_model=self.config.lyrics_model,
                reencode_settings=self.reencode_settings,
                search_settings=self.search_settings,
                split_settings=self.split_settings,
                retry_settings=self.retry_settings,
                source_type=source_type,
                timestamp=timestamp,
                user_context=active.user_context,
                validation_buffer_ms=self.validation_buffer_ms,
                summary_proxy_path=proxy_path,
                reencoded_segment_paths=active.reencoded_segment_paths,
            )
            session.discover_on_disk_state()
            active.session = session
            self.active_video = active

            self.tasks.clear()
            self.preparing_video_info = None

            # Seed completed or permanently failed tasks from existing on-disk responses
            sum_id = f"{self.config.name}::{video_id}::summary"
            if session.summary_response:
                self.tasks[sum_id] = ServerTask(
                    task_id=sum_id,
                    task_type=TaskType.SUMMARY,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    model=self.config.summary_model,
                    media_path=proxy_path,
                    video_duration_ms=active.duration_ms,
                    status=TaskStatus.COMPLETED,
                    attempts=session.summary_attempts,
                    result=session.summary_response.model_dump(mode="json"),
                )
            elif session.summary_failed:
                self.tasks[sum_id] = ServerTask(
                    task_id=sum_id,
                    task_type=TaskType.SUMMARY,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    model=self.config.summary_model,
                    media_path=proxy_path,
                    video_duration_ms=active.duration_ms,
                    status=TaskStatus.FAILED,
                    attempts=max(session.summary_attempts, self.retry_settings.max_runs),
                )

            for idx, sub_resp in session.subtitles_responses.items():
                sub_id = f"{self.config.name}::{video_id}::subtitles::{idx:03d}"
                self.tasks[sub_id] = ServerTask(
                    task_id=sub_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    media_path=reencoded_paths.get(idx) or segment_files[idx],
                    video_duration_ms=durations[idx],
                    status=TaskStatus.COMPLETED,
                    attempts=session.subtitles_attempts.get(idx, 1),
                    result=sub_resp.model_dump(mode="json"),
                )

            for idx in session.failed_subtitles:
                sub_id = f"{self.config.name}::{video_id}::subtitles::{idx:03d}"
                self.tasks[sub_id] = ServerTask(
                    task_id=sub_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    media_path=reencoded_paths.get(idx) or segment_files[idx],
                    video_duration_ms=durations[idx],
                    status=TaskStatus.FAILED,
                    attempts=max(session.subtitles_attempts.get(idx, 0), self.retry_settings.max_runs),
                )

            for idx, lyr_resp in session.lyrics_responses.items():
                lyr_id = f"{self.config.name}::{video_id}::lyrics::{idx:03d}"
                self.tasks[lyr_id] = ServerTask(
                    task_id=lyr_id,
                    task_type=TaskType.LYRICS,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.lyrics_model,
                    media_path=reencoded_paths.get(idx) or segment_files[idx],
                    video_duration_ms=durations[idx],
                    status=TaskStatus.COMPLETED,
                    attempts=session.lyrics_attempts.get(idx, 1),
                    result=lyr_resp.model_dump(mode="json"),
                )

            for idx in session.failed_lyrics:
                lyr_id = f"{self.config.name}::{video_id}::lyrics::{idx:03d}"
                self.tasks[lyr_id] = ServerTask(
                    task_id=lyr_id,
                    task_type=TaskType.LYRICS,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.lyrics_model,
                    media_path=reencoded_paths.get(idx) or segment_files[idx],
                    video_duration_ms=durations[idx],
                    status=TaskStatus.FAILED,
                    attempts=max(session.lyrics_attempts.get(idx, 0), self.retry_settings.max_runs),
                )

            self._sync_tasks_from_session()
            self._reencode_task = asyncio.create_task(self._reencode_segments_background())

            if self.is_video_complete():
                self.status = PipelineStatus.FINALIZING
                resume_span.set_attribute("result", "resumed_complete")
            else:
                self.status = PipelineStatus.PROCESSING
                pending_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
                resume_span.set_attribute("result", "resumed_processing")
                resume_span.set_attribute("pending_tasks", pending_count)

            return True

    async def clear_active_video(self) -> None:
        """Purges working directory files and clears the active video state for this pipeline."""
        with logfire.span("Clear Active Video: {combo}", combo=self.config.name) as clear_span:
            if self._reencode_task and not self._reencode_task.done():
                self._reencode_task.cancel()
                try:
                    await self._reencode_task
                except asyncio.CancelledError:
                    pass
                self._reencode_task = None

            purged_path = None
            if self.active_video and self.active_video.job_dir.exists():
                purged_path = str(self.active_video.job_dir)
                try:
                    shutil.rmtree(self.active_video.job_dir)
                except OSError as e:
                    logfire.warning(f"Could not purge job directory {self.active_video.job_dir.name}: {e}")

            clear_span.set_attribute("purged_directory", purged_path or "none")
            self.active_video = None
            self.preparing_video_info = None
            self.tasks.clear()
            self.status = PipelineStatus.IDLE
