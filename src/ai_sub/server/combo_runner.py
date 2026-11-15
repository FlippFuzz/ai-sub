"""Combination runner managing 1 active video lifecycle, task creation, and state resumption."""

from __future__ import annotations

import asyncio
import inspect
import json
import shutil
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import logfire

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    ServiceTier,
    SplittingSettings,
)
from ai_sub.jobs.lyrics import LyricsJob
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitles import DetectedMusicalPerformance, Subtitles, SubtitlesJob
from ai_sub.jobs.summary import SummaryAiResponse, SummaryJob
from ai_sub.pipeline import FailureOutcome, Pipeline
from ai_sub.protocol import TaskType
from ai_sub.server.models import ActiveVideo, ServerTask, TaskStatus
from ai_sub.server.server_config import PipelineConfig
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
    max_reencode_concurrency: int
    coordinator_lock: asyncio.Lock
    on_pipeline_done: Callable[[], Any] | None
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
    _reencode_wake_event: asyncio.Event

    def __init__(
        self,
        config: PipelineConfig,
        base_work_dir: Path,
        reencode_semaphore: asyncio.Semaphore,
        reencode_settings: ReEncodeSettings,
        default_search_tools: list[str],
        split_settings: SplittingSettings,
        retry_settings: RetrySettings,
        max_reencode_concurrency: int = 1,
        coordinator_lock: asyncio.Lock | None = None,
        on_pipeline_done: Callable[[], Any] | None = None,
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
            max_reencode_concurrency: Maximum concurrent local re-encodes scheduled per runner. Defaults to 1.
            coordinator_lock: Optional coordinator lock to synchronize task mutations. Defaults to None.
            on_pipeline_done: Optional callback triggered when background re-encoding concludes the pipeline.
            validation_buffer_ms: Allowed timestamp tolerance in milliseconds. Defaults to 2000.
        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.work_dir = base_work_dir / config.name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.reencode_semaphore = reencode_semaphore
        self.max_reencode_concurrency = max(1, max_reencode_concurrency)
        self.coordinator_lock = coordinator_lock or asyncio.Lock()
        self.on_pipeline_done = on_pipeline_done
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
        self._reencode_wake_event = asyncio.Event()

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
        """Determines if a task's media is ready to be claimed by a remote worker.

        Args:
            task: Candidate server task.

        Returns:
            bool: True if task is ready for pickup, False otherwise.
        """
        if not self.active_video or not self.active_video.pipeline:
            return False
        pipeline = self.active_video.pipeline
        if task.task_type == TaskType.SUMMARY:
            return pipeline.summary.media_ready
        if task.segment_index is not None and 0 <= task.segment_index < len(pipeline.segments):
            return pipeline.segments[task.segment_index].media_ready
        return False

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

    def _get_task_spec(self, task_type: TaskType) -> tuple[str, ServiceTier | None, list[str], bool]:
        """Resolves model, service tier, search tools, and agentic flag for a task type.

        Args:
            task_type: Target AI task category.

        Returns:
            tuple[str, ServiceTier | None, list[str], bool]:
                (model, service_tier, search_tools, agentic_video).
        """
        if task_type == TaskType.SUMMARY:
            tools = [t for t in self._get_active_search_tools() if t != "lrclib"]
            return self.config.summary_model, self.config.service_tier_summary, tools, True
        if task_type == TaskType.SUBTITLES:
            return self.config.subtitles_model, self.config.service_tier_subtitles, [], False
        return self.config.lyrics_model, self.config.service_tier_lyrics, self._get_active_search_tools(), True

    def _upsert_server_task(
        self,
        task_id: str,
        task_type: TaskType,
        video_id: str,
        media_path: Path,
        video_duration_ms: int,
        attempts: int,
        status: TaskStatus = TaskStatus.PENDING,
        segment_index: int | None = None,
        segment_offset_ms: int = 0,
        user_context: str | None = None,
        global_summary: SummaryAiResponse | None = None,
        draft_subtitles: list[Subtitles] | None = None,
        detected_performances: list[DetectedMusicalPerformance] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Creates or updates a pending ServerTask entry in self.tasks.

        Args:
            task_id: Unique server task identifier.
            task_type: Type of AI task.
            video_id: Owning video identifier.
            media_path: Path to the segment or summary media file.
            video_duration_ms: Media duration in milliseconds.
            attempts: Current attempt count for this FPS tier.
            status: Task status. Defaults to TaskStatus.PENDING.
            segment_index: Zero-based segment index, if applicable. Defaults to None.
            segment_offset_ms: Segment timeline offset in milliseconds. Defaults to 0.
            user_context: Optional background user context string. Defaults to None.
            global_summary: Optional Phase 1 summary context. Defaults to None.
            draft_subtitles: Optional draft subtitles for lyrics tasks. Defaults to None.
            detected_performances: Optional detected performances for lyrics tasks. Defaults to None.
            result: Optional completed result payload. Defaults to None.
        """
        if task_id in self.tasks:
            existing = self.tasks[task_id]
            if existing.status == TaskStatus.PENDING:
                existing.media_path = media_path
                existing.attempts = attempts
            return

        model, service_tier, search_tools, agentic_video = self._get_task_spec(task_type)
        self.tasks[task_id] = ServerTask(
            task_id=task_id,
            task_type=task_type,
            pipeline_name=self.config.name,
            video_id=video_id,
            segment_index=segment_index,
            model=model,
            service_tier=service_tier,
            media_path=media_path,
            video_duration_ms=video_duration_ms,
            segment_offset_ms=segment_offset_ms,
            user_context=user_context,
            global_summary=global_summary,
            draft_subtitles=draft_subtitles or [],
            detected_performances=detected_performances or [],
            search_tools=search_tools,
            agentic_video=agentic_video,
            attempts=attempts,
            status=status,
            result=result,
        )

    async def _create_pipeline(
        self,
        raw_video_path: Path,
        job_dir: Path,
        video_id: str,
        title: str,
        uploader: str,
        channel_name: str | None,
        source_type: Literal["local", "youtube"],
        timestamp: int,
        user_context: str,
    ) -> Pipeline:
        """Initializes a Pipeline instance using this runner's model and processing configuration.

        Args:
            raw_video_path: Path to the local raw video file.
            job_dir: Working directory for segments and checkpoints.
            video_id: Unique video identifier.
            title: Video display title.
            uploader: Channel or uploader handle.
            channel_name: Optional human-readable channel name.
            source_type: Ingestion source category ('local' or 'youtube').
            timestamp: Video publication or intake Unix timestamp.
            user_context: Resolved user background context string.

        Returns:
            Pipeline: Initialized pipeline instance.
        """
        return await Pipeline.create(
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
            enforce_max_runs=True,
        )

    def _handle_reencode_failure_locked(self, job: ReencodeJob, pipeline: Pipeline) -> None:
        """Applies re-encode failure state transitions and updates corresponding ServerTask if exhausted.

        Args:
            job: Failed ReencodeJob instance.
            pipeline: Active Pipeline instance owning the job.
        """
        outcome = pipeline.fail_reencode(job)
        if outcome.action == "exhausted" and self.active_video:
            vid = self.active_video.video_id
            if job.is_summary_proxy:
                task_id = f"{self.config.name}::{vid}::summary"
            elif job.segment_index is not None:
                task_id = f"{self.config.name}::{vid}::subtitles::{job.segment_index:03d}"
            else:
                task_id = None

            if task_id and task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.FAILED
                self.tasks[task_id].attempts = outcome.tier_attempts

            if pipeline.is_done():
                self._reencode_wake_event.set()

        self.sync_tasks_from_pipeline()

    async def _run_single_reencode(self, job: ReencodeJob, pipeline: Pipeline) -> None:
        """Executes one local FFmpeg re-encode job under the shared server semaphore.

        Args:
            job: Target ReencodeJob instance.
            pipeline: Active Pipeline instance owning the job.
        """
        try:
            async with self.reencode_semaphore:
                out_path = await job.execute()

            async with self.coordinator_lock:
                if self.active_video and self.active_video.pipeline is pipeline:
                    pipeline.complete_reencode(job, out_path)
                    self.sync_tasks_from_pipeline()
        except Exception as e:
            logfire.warning(f"Local re-encoding job '{job.name}' (tier {job.fallback_tier}) failed: {e}")
            async with self.coordinator_lock:
                if self.active_video and self.active_video.pipeline is pipeline:
                    self._handle_reencode_failure_locked(job, pipeline)

    async def _reencode_loop(self) -> None:
        """Event-driven background loop managing initial and fallback ReencodeJobs for the active video.

        Raises:
            asyncio.CancelledError: If the background re-encoding loop is cancelled.
        """
        in_flight: dict[tuple[bool, int | None], asyncio.Task[None]] = {}
        try:
            while self.active_video and self.active_video.pipeline:
                pipeline = self.active_video.pipeline

                # Prune completed tasks from tracking dictionary
                for key, task in list(in_flight.items()):
                    if task.done():
                        del in_flight[key]
                        if not task.cancelled():
                            exc = task.exception()
                            if exc is not None:
                                logfire.warning(f"Unexpected re-encode background task error: {exc}")

                if pipeline.is_done():
                    break

                # Schedule ready ReencodeJobs in priority order (summary first, then ascending segment index)
                if len(in_flight) < self.max_reencode_concurrency:
                    pending_reencodes = [j for j in pipeline.get_pending_jobs() if isinstance(j, ReencodeJob)]
                    for job in pending_reencodes:
                        if len(in_flight) >= self.max_reencode_concurrency:
                            break
                        key = (job.is_summary_proxy, job.segment_index)
                        if key not in in_flight:
                            in_flight[key] = asyncio.create_task(self._run_single_reencode(job, pipeline))

                self._reencode_wake_event.clear()

                wake_task = asyncio.create_task(self._reencode_wake_event.wait())
                waitables: set[asyncio.Task[Any]] = set(in_flight.values()) | {wake_task}
                _, pending = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)
                if wake_task in pending:
                    wake_task.cancel()
                try:
                    await wake_task
                except asyncio.CancelledError:
                    pass

            # Cancel remaining re-encodes if video aborted or concluded
            for t in in_flight.values():
                if not t.done():
                    t.cancel()
            if in_flight:
                await asyncio.gather(*in_flight.values(), return_exceptions=True)
                in_flight.clear()

            if self.active_video and self.active_video.pipeline and self.active_video.pipeline.is_done():
                if self.on_pipeline_done is not None:
                    res = self.on_pipeline_done()
                    if inspect.isawaitable(res):
                        await res

        except asyncio.CancelledError:
            for t in in_flight.values():
                if not t.done():
                    t.cancel()
            if in_flight:
                await asyncio.gather(*in_flight.values(), return_exceptions=True)
            raise

    def sync_tasks_from_pipeline(self) -> None:
        """Synchronizes pending AI jobs from Pipeline into ServerTask instances."""
        active = self.active_video
        pipeline = active.pipeline if active else None
        if not active or pipeline is None:
            return

        for job in pipeline.get_pending_jobs():
            if isinstance(job, ReencodeJob):
                continue

            if isinstance(job, SummaryJob):
                task_id = f"{self.config.name}::{active.video_id}::summary"
                media = job.media_path or pipeline.summary.media_path or pipeline.raw_video_path
                self._upsert_server_task(
                    task_id=task_id,
                    task_type=TaskType.SUMMARY,
                    video_id=active.video_id,
                    media_path=media,
                    video_duration_ms=job.video_duration_ms,
                    attempts=job.tier_attempts,
                    user_context=job.user_context,
                )
            elif isinstance(job, (SubtitlesJob, LyricsJob)):
                idx = job.segment_index
                assert idx is not None
                task_type = TaskType.SUBTITLES if isinstance(job, SubtitlesJob) else TaskType.LYRICS
                task_id = f"{self.config.name}::{active.video_id}::{task_type.value}::{idx:03d}"
                seg = pipeline.segments[idx]
                media = job.media_path or seg.media_path or seg.raw_path
                self._upsert_server_task(
                    task_id=task_id,
                    task_type=task_type,
                    video_id=active.video_id,
                    media_path=media,
                    video_duration_ms=job.video_duration_ms,
                    attempts=job.tier_attempts,
                    segment_index=idx,
                    segment_offset_ms=job.segment_offset_ms,
                    user_context=job.user_context,
                    global_summary=job.global_summary,
                    draft_subtitles=job.draft_subtitles if isinstance(job, LyricsJob) else None,
                    detected_performances=job.detected_performances if isinstance(job, LyricsJob) else None,
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
            pipeline = await self._create_pipeline(
                raw_video_path=raw_video_path,
                job_dir=job_dir,
                video_id=video_id,
                title=title,
                uploader=uploader,
                channel_name=channel_name,
                source_type=source_type,
                timestamp=timestamp,
                user_context=user_context,
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
            async with lock_to_use:
                self.active_video = active
                self.tasks.clear()
                self.preparing_video_info = None
                self.status = RunnerStatus.PROCESSING
                self._reencode_wake_event.clear()
                self.sync_tasks_from_pipeline()

            self._reencode_task = asyncio.create_task(self._reencode_loop())

            prep_span.set_attribute("result", "ready")
            prep_span.set_attribute("chunks", len(pipeline.segments))

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
            pipeline.complete_ai_task(
                task_type=task.task_type,
                result_data=result_data,
                segment_index=task.segment_index,
            )
            task.status = TaskStatus.COMPLETED
            task.result = result_data
            self.sync_tasks_from_pipeline()
            if pipeline.is_done():
                self._reencode_wake_event.set()

    def on_task_failed(self, task_id: str) -> FailureOutcome:
        """Handles a non-retryable task failure, transitioning pipeline state or triggering FPS fallback.

        Args:
            task_id: Unique task identifier.

        Returns:
            FailureOutcome: Resulting state transition from Pipeline.record_failure().
        """
        task = self.tasks.get(task_id)
        active = self.active_video
        pipeline = active.pipeline if active else None
        if not task or not active or pipeline is None:
            return FailureOutcome(action="ignored")

        outcome = pipeline.record_failure(
            task_type=task.task_type,
            segment_index=task.segment_index,
        )

        if outcome.action == "fallback_reencode":
            task.status = TaskStatus.PENDING
            task.attempts = 0
            task.result = None
            self._reencode_wake_event.set()
        elif outcome.action == "exhausted":
            task.status = TaskStatus.FAILED
            task.attempts = outcome.tier_attempts
            if pipeline.is_done():
                self._reencode_wake_event.set()
        elif outcome.action == "retry":
            task.status = TaskStatus.PENDING
            task.attempts = outcome.tier_attempts

        self.sync_tasks_from_pipeline()
        return outcome

    def is_video_complete(self) -> bool:
        """Checks if all stages for the active video have concluded or aborted.

        Returns:
            bool: True if video processing has concluded, False otherwise.
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

    def _seed_resumed_tasks(self, pipeline: Pipeline, video_id: str, raw_path: Path) -> None:
        """Seeds completed and exhausted ServerTask records from resumed Pipeline state.

        Args:
            pipeline: Restored Pipeline instance.
            video_id: Resumed video identifier.
            raw_path: Raw video path fallback.
        """
        if pipeline.summary.response is not None or pipeline.summary.exhausted:
            sum_status = TaskStatus.COMPLETED if pipeline.summary.response is not None else TaskStatus.FAILED
            sum_result = (
                pipeline.summary.response.model_dump(mode="json") if pipeline.summary.response is not None else None
            )
            self._upsert_server_task(
                task_id=f"{self.config.name}::{video_id}::summary",
                task_type=TaskType.SUMMARY,
                video_id=video_id,
                media_path=pipeline.summary.media_path or raw_path,
                video_duration_ms=pipeline.duration_ms,
                attempts=pipeline.summary.tier_attempts,
                status=sum_status,
                result=sum_result,
            )

        for seg in pipeline.segments:
            seg_media = seg.media_path or seg.raw_path
            for task_type, resp, exhausted, attempts in (
                (TaskType.SUBTITLES, seg.subtitles_response, seg.subtitles_exhausted, seg.subtitles_tier_attempts),
                (TaskType.LYRICS, seg.lyrics_response, seg.lyrics_exhausted, seg.lyrics_tier_attempts),
            ):
                if resp is not None or exhausted:
                    self._upsert_server_task(
                        task_id=f"{self.config.name}::{video_id}::{task_type.value}::{seg.index:03d}",
                        task_type=task_type,
                        video_id=video_id,
                        media_path=seg_media,
                        video_duration_ms=seg.duration_ms,
                        attempts=attempts,
                        status=TaskStatus.COMPLETED if resp is not None else TaskStatus.FAILED,
                        segment_index=seg.index,
                        result=resp.model_dump(mode="json") if resp is not None else None,
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

            def _get_mtime(d: Path) -> float:
                meta = d / "video_meta.json"
                target = meta if meta.is_file() else d
                try:
                    return target.stat().st_mtime
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

            valid_candidates.sort(key=lambda item: _get_mtime(item[0]), reverse=True)

            resumed: tuple[Pipeline, Path, dict[str, Any], Path, str] | None = None
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
                    pipeline = await self._create_pipeline(
                        raw_video_path=raw_path,
                        job_dir=job_dir,
                        video_id=video_id,
                        title=title,
                        uploader=uploader,
                        channel_name=channel_name,
                        source_type=source_type,
                        timestamp=timestamp,
                        user_context=user_context,
                    )
                    resumed = (pipeline, job_dir, meta, raw_path, user_context)
                    break
                except Exception as e:
                    logfire.warning(
                        f"Failed to initialize pipeline during resume for {job_dir.name}: {e}. Preserving job folder."
                    )
                    failed_resume_dirs.add(job_dir)

            if resumed is None:
                self.status = RunnerStatus.IDLE
                self.preparing_video_info = None
                resume_span.set_attribute("result", "no_valid_video_found")
                return False

            pipeline, job_dir, meta, raw_path, user_context = resumed
            for cand_dir, _, _ in valid_candidates:
                if cand_dir != job_dir and cand_dir not in failed_resume_dirs:
                    logfire.info(f"Purging older job folder: {cand_dir.name} in {self.config.name}")
                    shutil.rmtree(cand_dir, ignore_errors=True)

            video_id = str(meta.get("video_id", job_dir.name))
            self.active_video = ActiveVideo(
                video_id=video_id,
                uploader=str(meta.get("uploader", "unknown")),
                channel_name=meta.get("channel_name"),
                title=str(meta.get("title", job_dir.name)),
                filename_stem=pipeline.filename_stem,
                source_type=meta.get("source_type", "youtube"),
                timestamp=int(meta.get("timestamp", 0)),
                user_context=user_context,
                job_dir=job_dir,
                raw_path=raw_path,
                duration_ms=pipeline.duration_ms,
                pipeline=pipeline,
            )
            self.tasks.clear()
            self.preparing_video_info = None
            self._reencode_wake_event.clear()

            self._seed_resumed_tasks(pipeline, video_id, raw_path)
            self.sync_tasks_from_pipeline()

            if self.is_video_complete():
                self.status = RunnerStatus.FINALIZING
                resume_span.set_attribute("result", "resumed_complete")
            else:
                self._reencode_task = asyncio.create_task(self._reencode_loop())
                self.status = RunnerStatus.PROCESSING
                resume_span.set_attribute("result", "resumed_processing")

            return True

    async def stop_background_tasks(self) -> None:
        """Cancels and awaits any active local re-encoding background loop without purging disk state."""
        self._reencode_wake_event.set()
        if self._reencode_task and not self._reencode_task.done():
            self._reencode_task.cancel()
            try:
                await self._reencode_task
            except asyncio.CancelledError:
                pass
        self._reencode_task = None

    async def clear_active_video(self, purge_disk: bool = True) -> None:
        """Clears the active video state for this runner, optionally purging working directory files.

        Args:
            purge_disk: Whether to remove the active video's working directory on disk. Defaults to True.
        """
        with logfire.span("Clear Active Video: {combo}", combo=self.config.name) as clear_span:
            await self.stop_background_tasks()

            purged_path = None
            if purge_disk and self.active_video and self.active_video.job_dir.exists():
                purged_path = str(self.active_video.job_dir)
                try:
                    shutil.rmtree(self.active_video.job_dir)
                except OSError as e:
                    logfire.warning(f"Could not purge job directory {self.active_video.job_dir.name}: {e}")

            clear_span.set_attribute("purged_directory", purged_path or "none")
            self.active_video = None
            self.preparing_video_info = None
            self.tasks.clear()
            self._reencode_wake_event.clear()
            self.status = RunnerStatus.IDLE
