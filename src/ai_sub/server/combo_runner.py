"""Combination runner managing 1 active video lifecycle, task creation, and state resumption."""

from __future__ import annotations

import asyncio
import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import logfire

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
)
from ai_sub.jobs.base import BaseJob
from ai_sub.jobs.lyrics import LyricsJob
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import SubtitlesJob
from ai_sub.jobs.summary import SummaryJob
from ai_sub.pipeline import Pipeline
from ai_sub.protocol import TaskType
from ai_sub.server.config import PipelineConfig
from ai_sub.server.models import ActiveVideo, ServerTask, TaskStatus
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_subtitles_shortcode,
    generate_summary_shortcode,
)


class RunnerStatus(str, Enum):
    """Lifecycle execution status of a combination runner."""

    IDLE = "idle"
    PREPARING = "preparing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    PUSH_FAILED = "push_failed"


class ComboRunner:
    """Manages an independent combination runner restricted to 1 active video."""

    config: PipelineConfig
    base_work_dir: Path
    work_dir: Path
    reencode_semaphore: asyncio.Semaphore
    coordinator_lock: asyncio.Lock | None
    validation_buffer_ms: int
    reencode_settings: ReEncodeSettings
    default_search_tools: list[str]
    split_settings: SplittingSettings
    retry_settings: RetrySettings

    summary_shortcode: str
    subtitles_shortcode: str
    lyrics_shortcode: str
    full_shortcode: str

    status: RunnerStatus
    active_video: ActiveVideo | None
    preparing_video_info: dict[str, Any] | None
    tasks: dict[str, ServerTask]
    _reencode_task: asyncio.Task[None] | None

    def __init__(
        self,
        config: PipelineConfig,
        base_work_dir: Path,
        reencode_semaphore: asyncio.Semaphore,
        reencode_settings: ReEncodeSettings,
        default_search_tools: list[str],
        split_settings: SplittingSettings,
        retry_settings: RetrySettings,
        coordinator_lock: asyncio.Lock | None = None,
        validation_buffer_ms: int = 2000,
    ) -> None:
        """Initializes the combination runner.

        Args:
            config: Model pipeline configuration.
            base_work_dir: Root working directory containing per-pipeline folders.
            reencode_semaphore: Server-wide concurrency semaphore for FFmpeg compression.
            reencode_settings: Video compression and proxy rules.
            default_search_tools: Default search tools list when not overridden per pipeline.
            split_settings: Video chunk duration and tolerance rules.
            retry_settings: Task attempt limits and backoff rules.
            coordinator_lock: Optional coordinator lock to synchronize task mutations. Defaults to None.
            validation_buffer_ms: Allowed timestamp tolerance in milliseconds. Defaults to 2000.
        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.work_dir = base_work_dir / config.name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.reencode_semaphore = reencode_semaphore
        self.coordinator_lock = coordinator_lock
        self.validation_buffer_ms = validation_buffer_ms

        self.reencode_settings = reencode_settings
        self.default_search_tools = default_search_tools
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

        self.status = RunnerStatus.IDLE
        self.active_video = None
        self.preparing_video_info = None
        self.tasks = {}
        self._reencode_task = None

    @property
    def summary_response(self) -> Any:
        """Summary response from the active pipeline if available."""
        if self.active_video and self.active_video.pipeline:
            return self.active_video.pipeline.summary_response
        return None

    def is_idle(self) -> bool:
        """Checks if the runner is idle and ready to accept a new video.

        Returns:
            bool: True if runner status is IDLE, False otherwise.
        """
        return self.status == RunnerStatus.IDLE

    def set_preparing(
        self,
        video_id: str,
        title: str,
        uploader: str,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "youtube",
    ) -> None:
        """Marks runner as PREPARING immediately while media is being downloaded or chunked.

        Args:
            video_id: Unique video identifier.
            title: Title of the video.
            uploader: Channel or uploader handle identifier.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'youtube'.
        """
        self.status = RunnerStatus.PREPARING
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
            self.status = RunnerStatus.IDLE
        self.preparing_video_info = None

    def is_task_ready_for_claim(self, task: ServerTask) -> bool:
        """Determines if a task is eligible to be claimed by a remote worker.

        Args:
            task: Candidate server task.

        Returns:
            bool: True if task is ready for pickup, False otherwise.
        """
        if task.task_type == TaskType.SUBTITLES and self.reencode_settings.segment.enabled:
            if task.segment_index is not None and self.active_video is not None:
                pipeline = self.active_video.pipeline
                return bool(pipeline and task.segment_index in pipeline.reencoded_segment_paths)
        return True

    def _build_user_context(
        self,
        title: str,
        uploader: str,
        channel_name: str | None = None,
        source_type: str = "youtube",
    ) -> str:
        """Constructs background user context from video metadata.

        Args:
            title: Video title.
            uploader: Channel or uploader handle.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Source category ('local' or 'youtube'). Defaults to 'youtube'.

        Returns:
            str: Formatted context string.
        """
        if source_type == "youtube":
            ch = channel_name or uploader
            return f"This is a YouTube video from {ch} channel. The video title is '{title}'."
        return f"The video title is '{title}'."

    def _get_active_search_tools(self) -> list[str]:
        """Resolves active search tools from pipeline override or server default search tools.

        Returns:
            list[str]: Active search tools list.
        """
        return self.config.search_tools if self.config.search_tools is not None else self.default_search_tools

    async def _process_local_reencode_jobs(self) -> None:
        """Background loop executing ReencodeJobs locally under the shared server semaphore."""
        while self.active_video and not self.is_video_complete():
            pipeline = self.active_video.pipeline
            if not pipeline:
                break

            all_chunks_ready = len(pipeline.reencoded_segment_paths) == len(pipeline.segment_paths)
            proxy_resolved = not pipeline.summary_proxy_needed or pipeline.summary_proxy_path is not None
            if all_chunks_ready and proxy_resolved:
                break

            pending = pipeline.get_pending_jobs()
            reenc_jobs = [j for j in pending if isinstance(j, ReencodeJob)]
            if not reenc_jobs:
                await asyncio.sleep(2.0)
                continue

            for job in reenc_jobs:
                if not self.active_video or self.active_video.pipeline is not pipeline:
                    break
                try:
                    async with self.reencode_semaphore:
                        out_path = await job.execute()

                    if self.coordinator_lock:
                        async with self.coordinator_lock:
                            pipeline.complete_job(job, out_path)
                            self.sync_tasks_from_pipeline()
                    else:
                        pipeline.complete_job(job, out_path)
                        self.sync_tasks_from_pipeline()

                except Exception as e:
                    logfire.warning(
                        f"Local re-encoding job '{job.name}' failed: {e}. Falling back to uncompressed input."
                    )
                    # Clean up partial or corrupt output file to prevent lingering artifacts
                    if job.output_file.exists():
                        try:
                            job.output_file.unlink()
                        except OSError:
                            pass

                    if self.coordinator_lock:
                        async with self.coordinator_lock:
                            pipeline.complete_job(job, job.input_file)
                            self.sync_tasks_from_pipeline()
                    else:
                        pipeline.complete_job(job, job.input_file)
                        self.sync_tasks_from_pipeline()

    def sync_tasks_from_pipeline(self) -> None:
        """Synchronizes pending AI jobs from Pipeline into ServerTask instances."""
        active = self.active_video
        pipeline = active.pipeline if active else None
        if not active or pipeline is None:
            return

        pending_jobs: list[BaseJob] = pipeline.get_pending_jobs()
        for job in pending_jobs:
            if isinstance(job, ReencodeJob):
                continue

            if isinstance(job, SummaryJob):
                full_task_id = f"{self.config.name}::{active.video_id}::summary"
                if full_task_id in self.tasks:
                    continue
                self.tasks[full_task_id] = ServerTask(
                    task_id=full_task_id,
                    task_type=TaskType.SUMMARY,
                    pipeline_name=self.config.name,
                    video_id=active.video_id,
                    model=self.config.summary_model,
                    service_tier=self.config.service_tier_summary,
                    media_path=job.media_path or pipeline.summary_proxy_path or pipeline.raw_video_path,
                    video_duration_ms=job.video_duration_ms,
                    segment_offset_ms=0,
                    user_context=job.user_context,
                    search_tools=[t for t in self._get_active_search_tools() if t != "lrclib"],
                    agentic_video=True,
                    attempts=job.total_attempts,
                    status=TaskStatus.PENDING,
                )

            elif isinstance(job, SubtitlesJob):
                idx = job.segment_index
                assert idx is not None
                full_task_id = f"{self.config.name}::{active.video_id}::subtitles::{idx:03d}"
                media_file = pipeline.reencoded_segment_paths.get(idx) or job.media_path or pipeline.segment_paths[idx]
                if full_task_id in self.tasks:
                    self.tasks[full_task_id].media_path = media_file
                    continue

                self.tasks[full_task_id] = ServerTask(
                    task_id=full_task_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=active.video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    service_tier=self.config.service_tier_subtitles,
                    media_path=media_file,
                    video_duration_ms=job.video_duration_ms,
                    segment_offset_ms=job.segment_offset_ms,
                    user_context=job.user_context,
                    global_summary=job.global_summary,
                    search_tools=[],
                    agentic_video=False,
                    attempts=job.total_attempts,
                    status=TaskStatus.PENDING,
                )

            elif isinstance(job, LyricsJob):
                idx = job.segment_index
                assert idx is not None
                full_task_id = f"{self.config.name}::{active.video_id}::lyrics::{idx:03d}"
                media_file = pipeline.reencoded_segment_paths.get(idx) or job.media_path or pipeline.segment_paths[idx]
                if full_task_id in self.tasks:
                    self.tasks[full_task_id].media_path = media_file
                    continue

                self.tasks[full_task_id] = ServerTask(
                    task_id=full_task_id,
                    task_type=TaskType.LYRICS,
                    pipeline_name=self.config.name,
                    video_id=active.video_id,
                    segment_index=idx,
                    model=self.config.lyrics_model,
                    service_tier=self.config.service_tier_lyrics,
                    media_path=media_file,
                    video_duration_ms=job.video_duration_ms,
                    segment_offset_ms=job.segment_offset_ms,
                    user_context=job.user_context,
                    global_summary=job.global_summary,
                    draft_subtitles=job.draft_subtitles,
                    detected_performances=job.detected_performances,
                    search_tools=self._get_active_search_tools(),
                    agentic_video=True,
                    attempts=job.total_attempts,
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
        coordinator_lock: asyncio.Lock | None = None,
    ) -> None:
        """Prepares media using Pipeline, saves video metadata, and begins processing under lock.

        Args:
            video_id: Unique video identifier.
            uploader: Channel or uploader handle identifier.
            title: Title of the video.
            raw_video_path: Path to the local raw video file.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication or intake Unix timestamp. Defaults to 0.
            coordinator_lock: Optional coordinator lock to synchronize task mutation. Defaults to None.
        """
        self.status = RunnerStatus.PREPARING
        self.preparing_video_info = {
            "video_id": video_id,
            "uploader": uploader,
            "channel_name": channel_name,
            "title": title,
            "source_type": source_type,
        }
        job_dir = self.work_dir / video_id
        job_dir.mkdir(parents=True, exist_ok=True)
        user_context = self._build_user_context(title, uploader, channel_name, source_type)

        with logfire.span(
            "Prepare Video for Runner: {video_id}",
            combo=self.config.name,
            video_id=video_id,
            title=title,
        ) as prep_span:
            pipeline = await Pipeline.create(
                raw_video_path=raw_video_path,
                job_dir=job_dir,
                title=title,
                uploader=uploader,
                channel_name=channel_name,
                video_id=video_id,
                summary_model=self.config.summary_model,
                subtitles_model=self.config.subtitles_model,
                lyrics_model=self.config.lyrics_model,
                service_tier_summary=self.config.service_tier_summary,
                service_tier_subtitles=self.config.service_tier_subtitles,
                service_tier_lyrics=self.config.service_tier_lyrics,
                reencode_settings=self.reencode_settings,
                search_settings=SearchSettings(tools=self._get_active_search_tools()),
                split_settings=self.split_settings,
                retry_settings=self.retry_settings,
                source_type=source_type,
                timestamp=timestamp,
                user_context=user_context,
                validation_buffer_ms=self.validation_buffer_ms,
            )

            meta_file = job_dir / "video_meta.json"
            meta_data = {
                "video_id": video_id,
                "uploader": uploader,
                "channel_name": channel_name,
                "title": title,
                "filename_stem": pipeline.filename_stem,
                "raw_filename": raw_video_path.name,
                "raw_path": str(raw_video_path.resolve()),
                "source_type": source_type,
                "timestamp": timestamp,
                "duration_ms": pipeline.duration_ms,
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)

            active = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                filename_stem=pipeline.filename_stem,
                source_type=source_type,
                timestamp=timestamp,
                user_context=user_context,
                job_dir=job_dir,
                raw_path=raw_video_path,
                duration_ms=pipeline.duration_ms,
                pipeline=pipeline,
            )

            lock_to_use = coordinator_lock or self.coordinator_lock
            if lock_to_use:
                async with lock_to_use:
                    self.active_video = active
                    self.tasks.clear()
                    self.preparing_video_info = None
                    self.status = RunnerStatus.PROCESSING
                    self.sync_tasks_from_pipeline()
            else:
                self.active_video = active
                self.tasks.clear()
                self.preparing_video_info = None
                self.status = RunnerStatus.PROCESSING
                self.sync_tasks_from_pipeline()

            self._reencode_task = asyncio.create_task(self._process_local_reencode_jobs())

            prep_span.set_attribute("result", "ready")
            prep_span.set_attribute("chunks", len(pipeline.segment_paths))

    def on_task_completed(self, task_id: str, result_data: dict[str, Any]) -> None:
        """Handles task completion, validating output and updating pipeline state.

        Args:
            task_id: Unique task identifier.
            result_data: Validated structured dictionary received from worker.
        """
        task = self.tasks.get(task_id)
        active = self.active_video
        pipeline = active.pipeline if active else None
        if not task or not active or pipeline is None:
            return

        with logfire.span(
            "Task Cascade Complete: {task_id}",
            combo=self.config.name,
            task_id=task_id,
            task_type=task.task_type.value,
        ):
            pipeline.complete_job(
                job_or_name=task.task_type.value,
                result_data=result_data,
                segment_index=task.segment_index,
                task_type=task.task_type,
            )
            task.status = TaskStatus.COMPLETED
            task.result = result_data
            self.sync_tasks_from_pipeline()

    def on_task_failed(self, task_id: str, attempts: int) -> None:
        """Handles task failure, updating pipeline state and persisting attempt counts.

        Args:
            task_id: Unique task identifier.
            attempts: Total attempts recorded so far for this task.
        """
        task = self.tasks.get(task_id)
        active = self.active_video
        pipeline = active.pipeline if active else None
        if not task or not active or pipeline is None:
            return

        pipeline.record_failure(
            task_type=task.task_type,
            segment_index=task.segment_index,
            attempts=attempts,
        )
        self.sync_tasks_from_pipeline()

    def is_video_complete(self) -> bool:
        """Checks if all segments for the active video have concluded.

        Returns:
            bool: True if video processing has concluded across all stages, False otherwise.
        """
        if not self.active_video or self.status == RunnerStatus.PREPARING:
            return False

        if self.status in (RunnerStatus.FINALIZING, RunnerStatus.PUSH_FAILED):
            return True

        if self.active_video.pipeline:
            return self.active_video.pipeline.is_done()

        return False

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
        pipeline = active.pipeline if active else None
        if active is None or pipeline is None or not self.is_video_complete():
            raise RuntimeError(f"Cannot stitch subtitles: {self.config.name} video processing is incomplete.")

        raw_output_path = output_path.parent / f"{active.filename_stem}.raw-{self.full_shortcode}.srt"

        with logfire.span(
            "Stitch Subtitles: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            output_file=output_path.name,
        ):
            return pipeline.finalize(
                output_primary_path=output_path,
                output_raw_path=raw_output_path,
                allow_partial=True,
            )

    async def resume_from_disk(self) -> bool:
        """Inspects work/<combo>/ to restore in-progress video directories upon server boot.

        Selects the newest valid job directory with ground-truth metadata and existing media,
        while purging invalid folders and deferring removal of older directories until
        the candidate pipeline successfully initializes.

        Returns:
            bool: True if an active video was successfully restored, False otherwise.
        """
        with logfire.span("Disk Resume Check: {combo}", combo=self.config.name) as resume_span:
            job_dirs = [d for d in self.work_dir.iterdir() if d.is_dir() and d.name != "reencoded"]
            if not job_dirs:
                self.status = RunnerStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "idle")
                return False

            # Sort candidate job directories newest-first based on video_meta.json or directory mtime
            def _get_mtime(d: Path) -> float:
                meta = d / "video_meta.json"
                if meta.is_file():
                    try:
                        return meta.stat().st_mtime
                    except OSError:
                        pass
                try:
                    return d.stat().st_mtime
                except OSError:
                    return 0.0

            valid_candidates: list[tuple[Path, dict[str, Any], Path]] = []
            for d in job_dirs:
                meta_file = d / "video_meta.json"
                if not meta_file.is_file():
                    logfire.info(f"Purging stale folder without metadata: {d.name} in {self.config.name}")
                    shutil.rmtree(d, ignore_errors=True)
                    continue

                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    raw_path_str = str(meta.get("raw_path", ""))
                    raw_path = Path(raw_path_str) if raw_path_str else None
                    if not raw_path or not raw_path.is_file():
                        logfire.warning(f"Purging job folder with missing raw video: {d.name} in {self.config.name}")
                        shutil.rmtree(d, ignore_errors=True)
                        continue

                    valid_candidates.append((d, meta, raw_path))
                except Exception as e:
                    logfire.warning(f"Purging job folder with corrupt metadata: {d.name} ({e})")
                    shutil.rmtree(d, ignore_errors=True)
                    continue

            valid_candidates.sort(key=lambda item: _get_mtime(item[0]), reverse=True)

            resumed_pipeline: Pipeline | None = None
            resumed_dir: Path | None = None
            resumed_meta: dict[str, Any] | None = None
            resumed_raw: Path | None = None
            failed_resume_dirs: set[Path] = set()

            for job_dir, meta, raw_path in valid_candidates:
                video_id = str(meta.get("video_id", job_dir.name))
                uploader = str(meta.get("uploader", "unknown"))
                channel_name = meta.get("channel_name")
                title = str(meta.get("title", job_dir.name))
                source_type = meta.get("source_type", "youtube")
                timestamp = int(meta.get("timestamp", 0))

                user_context = self._build_user_context(title, uploader, channel_name, source_type)

                try:
                    pipeline = await Pipeline.create(
                        raw_video_path=raw_path,
                        job_dir=job_dir,
                        title=title,
                        uploader=uploader,
                        channel_name=channel_name,
                        video_id=video_id,
                        summary_model=self.config.summary_model,
                        subtitles_model=self.config.subtitles_model,
                        lyrics_model=self.config.lyrics_model,
                        service_tier_summary=self.config.service_tier_summary,
                        service_tier_subtitles=self.config.service_tier_subtitles,
                        service_tier_lyrics=self.config.service_tier_lyrics,
                        reencode_settings=self.reencode_settings,
                        search_settings=SearchSettings(tools=self._get_active_search_tools()),
                        split_settings=self.split_settings,
                        retry_settings=self.retry_settings,
                        source_type=source_type,
                        timestamp=timestamp,
                        user_context=user_context,
                        validation_buffer_ms=self.validation_buffer_ms,
                    )
                    resumed_pipeline = pipeline
                    resumed_dir = job_dir
                    resumed_meta = meta
                    resumed_raw = raw_path
                    break
                except Exception as e:
                    # Do not delete directory on initialization errors; keep it intact and check next candidate
                    logfire.warning(
                        f"Failed to initialize pipeline during resume for {job_dir.name}: {e}. Preserving job folder."
                    )
                    failed_resume_dirs.add(job_dir)
                    continue

            if resumed_pipeline is None or resumed_dir is None or resumed_meta is None or resumed_raw is None:
                self.status = RunnerStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "no_valid_video_found")
                return False

            # Purge older job directories now that resumed pipeline has succeeded,
            # excluding candidates that failed initialization to preserve their checkpoints
            for job_dir, _, _ in valid_candidates:
                if job_dir != resumed_dir and job_dir not in failed_resume_dirs:
                    logfire.info(f"Purging older job folder: {job_dir.name} in {self.config.name}")
                    shutil.rmtree(job_dir, ignore_errors=True)

            job_dir = resumed_dir
            meta = resumed_meta
            raw_path = resumed_raw
            pipeline = resumed_pipeline
            video_id = str(meta.get("video_id", job_dir.name))
            uploader = str(meta.get("uploader", "unknown"))
            channel_name = meta.get("channel_name")
            title = str(meta.get("title", job_dir.name))
            source_type = meta.get("source_type", "youtube")
            timestamp = int(meta.get("timestamp", 0))
            user_context = self._build_user_context(title, uploader, channel_name, source_type)

            active = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                filename_stem=pipeline.filename_stem,
                source_type=source_type,
                timestamp=timestamp,
                user_context=user_context,
                job_dir=job_dir,
                raw_path=raw_path,
                duration_ms=pipeline.duration_ms,
                pipeline=pipeline,
            )
            self.active_video = active
            self.tasks.clear()
            self.preparing_video_info = None

            # Seed completed or failed tasks from pipeline
            sum_id = f"{self.config.name}::{video_id}::summary"
            if pipeline.summary_response:
                self.tasks[sum_id] = ServerTask(
                    task_id=sum_id,
                    task_type=TaskType.SUMMARY,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    model=self.config.summary_model,
                    service_tier=self.config.service_tier_summary,
                    media_path=pipeline.summary_proxy_path or raw_path,
                    video_duration_ms=pipeline.duration_ms,
                    status=TaskStatus.COMPLETED,
                    attempts=pipeline.summary_attempts,
                    result=pipeline.summary_response.model_dump(mode="json"),
                )
            elif pipeline.summary_failed:
                self.tasks[sum_id] = ServerTask(
                    task_id=sum_id,
                    task_type=TaskType.SUMMARY,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    model=self.config.summary_model,
                    service_tier=self.config.service_tier_summary,
                    media_path=pipeline.summary_proxy_path or raw_path,
                    video_duration_ms=pipeline.duration_ms,
                    status=TaskStatus.FAILED,
                    attempts=pipeline.summary_attempts,
                )

            for idx, sub_resp in pipeline.subtitles_responses.items():
                sub_id = f"{self.config.name}::{video_id}::subtitles::{idx:03d}"
                self.tasks[sub_id] = ServerTask(
                    task_id=sub_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    service_tier=self.config.service_tier_subtitles,
                    media_path=pipeline.reencoded_segment_paths.get(idx) or pipeline.segment_paths[idx],
                    video_duration_ms=pipeline.segment_durations_ms[idx],
                    status=TaskStatus.COMPLETED,
                    attempts=pipeline.subtitles_attempts.get(idx, 1),
                    result=sub_resp.model_dump(mode="json"),
                )

            for idx in pipeline.failed_subtitles:
                sub_id = f"{self.config.name}::{video_id}::subtitles::{idx:03d}"
                self.tasks[sub_id] = ServerTask(
                    task_id=sub_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    service_tier=self.config.service_tier_subtitles,
                    media_path=pipeline.reencoded_segment_paths.get(idx) or pipeline.segment_paths[idx],
                    video_duration_ms=pipeline.segment_durations_ms[idx],
                    status=TaskStatus.FAILED,
                    attempts=pipeline.subtitles_attempts.get(idx, self.retry_settings.max_runs),
                )

            for idx, lyr_resp in pipeline.lyrics_responses.items():
                lyr_id = f"{self.config.name}::{video_id}::lyrics::{idx:03d}"
                self.tasks[lyr_id] = ServerTask(
                    task_id=lyr_id,
                    task_type=TaskType.LYRICS,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.lyrics_model,
                    service_tier=self.config.service_tier_lyrics,
                    media_path=pipeline.reencoded_segment_paths.get(idx) or pipeline.segment_paths[idx],
                    video_duration_ms=pipeline.segment_durations_ms[idx],
                    status=TaskStatus.COMPLETED,
                    attempts=pipeline.lyrics_attempts.get(idx, 1),
                    result=lyr_resp.model_dump(mode="json"),
                )

            for idx in pipeline.failed_lyrics:
                lyr_id = f"{self.config.name}::{video_id}::lyrics::{idx:03d}"
                self.tasks[lyr_id] = ServerTask(
                    task_id=lyr_id,
                    task_type=TaskType.LYRICS,
                    pipeline_name=self.config.name,
                    video_id=video_id,
                    segment_index=idx,
                    model=self.config.lyrics_model,
                    service_tier=self.config.service_tier_lyrics,
                    media_path=pipeline.reencoded_segment_paths.get(idx) or pipeline.segment_paths[idx],
                    video_duration_ms=pipeline.segment_durations_ms[idx],
                    status=TaskStatus.FAILED,
                    attempts=pipeline.lyrics_attempts.get(idx, self.retry_settings.max_runs),
                )

            self.sync_tasks_from_pipeline()
            self._reencode_task = asyncio.create_task(self._process_local_reencode_jobs())

            if self.is_video_complete():
                self.status = RunnerStatus.FINALIZING
                resume_span.set_attribute("result", "resumed_complete")
            else:
                self.status = RunnerStatus.PROCESSING
                resume_span.set_attribute("result", "resumed_processing")

            return True

    async def clear_active_video(self) -> None:
        """Purges working directory files and clears the active video state for this runner."""
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
            self.status = RunnerStatus.IDLE
