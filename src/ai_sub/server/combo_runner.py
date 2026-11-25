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
    ReencodeSettings,
    RetrySettings,
    SplittingSettings,
    StageExecutionConfig,
    atomic_write_text,
)
from ai_sub.jobs.job_base import AiJob, TaskType
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.pipeline import FailureOutcome, Pipeline
from ai_sub.server.models import ServerTask, TaskStatus
from ai_sub.server.server_config import PipelineConfig
from ai_sub.shortcode import generate_full_shortcode
from ai_sub.video import compute_file_id_hash


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
    reencode_settings: ReencodeSettings
    split_settings: SplittingSettings
    retry_settings: RetrySettings
    full_shortcode: str

    status: RunnerStatus
    pipeline: Pipeline | None
    preparing_video_info: dict[str, Any] | None
    tasks: dict[str, ServerTask]
    active_summary_reencode: bool
    active_reencode_segments: set[int]
    _reencode_task: asyncio.Task[None] | None
    _reencode_wake_event: asyncio.Event

    def __init__(
        self,
        config: PipelineConfig,
        base_work_dir: Path,
        reencode_semaphore: asyncio.Semaphore,
        reencode_settings: ReencodeSettings,
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
        self.split_settings = split_settings
        self.retry_settings = retry_settings
        self.full_shortcode = generate_full_shortcode(
            subtitles_model=config.subtitles.model,
            summary_model=config.summary.model,
            lyrics_model=config.lyrics.model,
        )

        self.status = RunnerStatus.IDLE
        self.pipeline = None
        self.preparing_video_info = None
        self.tasks = {}
        self.active_summary_reencode = False
        self.active_reencode_segments = set()
        self._reencode_task = None
        self._reencode_wake_event = asyncio.Event()

    def get_job_dir(self, source_type: Literal["local", "youtube"], video_id: str) -> Path:
        """Returns the source-scoped working directory for a video inside this runner.

        Args:
            source_type: Ingestion source type ('local' or 'youtube').
            video_id: Unique video identifier.

        Returns:
            Path: Working directory path under work/<combo>/<source_type>/<video_id>.
        """
        return self.work_dir / source_type / video_id

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
        if not self.pipeline:
            self.status = RunnerStatus.IDLE
        self.preparing_video_info = None

    def is_task_ready_for_claim(self, task: ServerTask) -> bool:
        """Determines if a task's media is ready to be claimed by a remote worker.

        Args:
            task: Candidate server task.

        Returns:
            bool: True if task is ready for pickup, False otherwise.
        """
        if not self.pipeline:
            return False
        if task.task_type == TaskType.SUMMARY:
            return self.pipeline.summary.media_ready
        if task.segment_index is not None and 0 <= task.segment_index < len(self.pipeline.segments):
            return self.pipeline.segments[task.segment_index].media_ready
        return False

    def search_tools_for_task(self, task_type: TaskType) -> list[str]:
        """Returns permitted search tools for a task type.

        Args:
            task_type: Target AI task category.

        Returns:
            list[str]: Permitted search tool names.
        """
        if task_type == TaskType.SUMMARY:
            return [str(t) for t in self.config.searchtools if t != "lrclib"]
        if task_type == TaskType.SUBTITLES:
            return []
        return [str(t) for t in self.config.searchtools]

    def _format_task_id(
        self,
        task_type: TaskType,
        video_id: str,
        segment_index: int | None = None,
    ) -> str:
        """Formats a unique server task ID for a pipeline stage or segment.

        Args:
            task_type: AI task type.
            video_id: Active video identifier.
            segment_index: Zero-based segment index for segment stages. Defaults to None.

        Returns:
            str: Canonical task ID string.
        """
        if task_type == TaskType.SUMMARY:
            return f"{self.config.name}::{video_id}::summary"
        assert segment_index is not None
        return f"{self.config.name}::{video_id}::{task_type.value}::{segment_index:03d}"

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

    def _stage_config(self, task_type: TaskType) -> StageExecutionConfig:
        """Returns stage execution configuration for a pipeline task type.

        Args:
            task_type: Target AI task category.

        Returns:
            StageExecutionConfig: Model, tier, and agentic flag for the stage.
        """
        if task_type == TaskType.SUMMARY:
            return self.config.summary
        if task_type == TaskType.SUBTITLES:
            return self.config.subtitles
        return self.config.lyrics

    def _upsert_server_task(
        self,
        task_id: str,
        task_type: TaskType,
        video_id: str,
        media_path: Path,
        segment_index: int | None = None,
    ) -> None:
        """Creates or updates a pending ServerTask entry in self.tasks.

        Args:
            task_id: Unique server task identifier.
            task_type: Type of AI task.
            video_id: Owning video identifier.
            media_path: Path to the segment or summary media file.
            segment_index: Zero-based segment index, if applicable. Defaults to None.
        """
        if task_id in self.tasks:
            existing = self.tasks[task_id]
            if existing.status == TaskStatus.PENDING:
                existing.media_path = media_path
            return

        self.tasks[task_id] = ServerTask(
            task_id=task_id,
            task_type=task_type,
            pipeline_name=self.config.name,
            video_id=video_id,
            segment_index=segment_index,
            model=self._stage_config(task_type).model,
            media_path=media_path,
            status=TaskStatus.PENDING,
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
        media_tag = video_id if source_type == "youtube" else compute_file_id_hash(raw_video_path.name)
        return await Pipeline.create(
            raw_video_path=raw_video_path,
            job_dir=job_dir,
            title=title,
            uploader=uploader,
            channel_name=channel_name,
            video_id=video_id,
            media_tag=media_tag,
            output_stem=video_id,
            summary_config=self._stage_config(TaskType.SUMMARY),
            subtitles_config=self._stage_config(TaskType.SUBTITLES),
            lyrics_config=self._stage_config(TaskType.LYRICS),
            reencode_settings=self.reencode_settings,
            search_tools=[str(t) for t in self.config.searchtools],
            split_settings=self.split_settings,
            retry_settings=self.retry_settings,
            source_type=source_type,
            timestamp=timestamp,
            user_context=user_context,
            validation_buffer_ms=self.validation_buffer_ms,
        )

    async def _handle_reencode_failure_locked(self, job: ReencodeJob, pipeline: Pipeline) -> None:
        """Applies re-encode failure state transitions and updates corresponding ServerTask if exhausted.

        Args:
            job: Failed ReencodeJob instance.
            pipeline: Active Pipeline instance owning the job.
        """
        outcome = await pipeline.fail_reencode(job)
        if outcome.action == "exhausted" and self.pipeline is pipeline:
            if job.is_summary_proxy:
                task_id: str | None = self._format_task_id(TaskType.SUMMARY, pipeline.video_id)
            elif job.segment_index is not None:
                task_id = self._format_task_id(TaskType.SUBTITLES, pipeline.video_id, job.segment_index)
            else:
                task_id = None

            if task_id and task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.FAILED

        self.sync_tasks_from_pipeline()

    async def _run_single_reencode(self, job: ReencodeJob, pipeline: Pipeline) -> None:
        """Executes one local FFmpeg re-encode job under the shared server semaphore.

        Args:
            job: Target ReencodeJob instance.
            pipeline: Active Pipeline instance owning the job.
        """
        try:
            async with self.reencode_semaphore:
                if job.is_summary_proxy:
                    self.active_summary_reencode = True
                elif job.segment_index is not None:
                    self.active_reencode_segments.add(job.segment_index)
                try:
                    out_path = await job.execute()
                finally:
                    if job.is_summary_proxy:
                        self.active_summary_reencode = False
                    elif job.segment_index is not None:
                        self.active_reencode_segments.discard(job.segment_index)

            async with self.coordinator_lock:
                if self.pipeline is pipeline:
                    pipeline.complete_reencode(job, out_path)
                    self.sync_tasks_from_pipeline()
        except Exception as e:
            logfire.warning(f"Local re-encoding job '{job.name}' (tier {job.fallback_tier}) failed: {e}")
            async with self.coordinator_lock:
                if self.pipeline is pipeline:
                    await self._handle_reencode_failure_locked(job, pipeline)

    async def _reencode_loop(self) -> None:
        """Event-driven background loop managing initial and fallback ReencodeJobs for the active video.

        Raises:
            asyncio.CancelledError: If the background re-encoding loop is cancelled.
        """
        in_flight: dict[tuple[bool, int | None], asyncio.Task[None]] = {}
        try:
            while self.pipeline:
                pipeline = self.pipeline

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
                try:
                    await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not wake_task.done():
                        wake_task.cancel()
                        await asyncio.gather(wake_task, return_exceptions=True)

            # Cancel remaining re-encodes if video aborted or concluded
            for t in in_flight.values():
                if not t.done():
                    t.cancel()
            if in_flight:
                await asyncio.gather(*in_flight.values(), return_exceptions=True)
                in_flight.clear()

            if self.pipeline and self.pipeline.is_done():
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
        pipeline = self.pipeline
        if pipeline is None:
            return

        for job in pipeline.get_pending_jobs():
            if not isinstance(job, AiJob):
                continue

            media = pipeline.get_job_media_path(job)
            task_id = self._format_task_id(job.task_type, pipeline.video_id, job.segment_index)

            self._upsert_server_task(
                task_id=task_id,
                task_type=job.task_type,
                video_id=pipeline.video_id,
                media_path=media,
                segment_index=job.segment_index,
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
        """Prepares media using Pipeline, saves video metadata, and begins processing under lock.

        Args:
            video_id: Unique video identifier.
            uploader: Channel or uploader handle identifier.
            title: Title of the video.
            raw_video_path: Path to the local raw video file.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication or intake Unix timestamp. Defaults to 0.
        """
        self.set_preparing(
            video_id=video_id,
            title=title,
            uploader=uploader,
            channel_name=channel_name,
            source_type=source_type,
        )
        job_dir = self.get_job_dir(source_type, video_id)
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
                "raw_path": str(raw_video_path.resolve()),
                "source_type": source_type,
                "timestamp": timestamp,
            }
            await asyncio.to_thread(atomic_write_text, meta_file, json.dumps(meta_data, indent=2, ensure_ascii=False))

            async with self.coordinator_lock:
                self.pipeline = pipeline
                self.tasks.clear()
                self.active_summary_reencode = False
                self.active_reencode_segments.clear()
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
        pipeline = self.pipeline
        if not task or pipeline is None:
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
            self.sync_tasks_from_pipeline()
            if pipeline.is_done():
                self._reencode_wake_event.set()

    async def on_task_failed(
        self,
        task_id: str,
        fallback_base_fps: float | None = None,
        has_audio: bool | None = None,
    ) -> FailureOutcome:
        """Handles a non-retryable task failure, transitioning pipeline state or triggering FPS fallback.

        Args:
            task_id: Unique task identifier.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.
            has_audio: Optional pre-probed audio-stream presence flag. Defaults to None.

        Returns:
            FailureOutcome: Resulting state transition from Pipeline.record_failure().
        """
        task = self.tasks.get(task_id)
        pipeline = self.pipeline
        if not task or pipeline is None:
            return FailureOutcome(action="ignored")

        outcome = await pipeline.record_failure(
            task_type=task.task_type,
            segment_index=task.segment_index,
            fallback_base_fps=fallback_base_fps,
            has_audio=has_audio,
        )

        if outcome.action == "retry":
            task.status = TaskStatus.PENDING
        elif outcome.action == "fallback_reencode":
            task.status = TaskStatus.PENDING
            self._reencode_wake_event.set()
        elif outcome.action == "exhausted":
            task.status = TaskStatus.FAILED
            if pipeline.is_done():
                self._reencode_wake_event.set()

        self.sync_tasks_from_pipeline()
        return outcome

    def is_video_complete(self) -> bool:
        """Checks if all stages for the active video have concluded or aborted.

        Returns:
            bool: True if video processing has concluded, False otherwise.
        """
        if not self.pipeline or self.status == RunnerStatus.PREPARING:
            return False

        if self.status in (RunnerStatus.FINALIZING, RunnerStatus.PUSH_FAILED):
            return True

        return self.pipeline.is_done()

    async def resume_from_disk(self) -> bool:
        """Inspects work/<combo>/{local,youtube}/ to restore in-progress video directories upon server boot.

        Selects the newest valid job directory with ground-truth metadata and existing media,
        while purging invalid folders and deferring removal of older directories until
        the candidate pipeline successfully initializes.

        Returns:
            bool: True if an active video was successfully restored, False otherwise.
        """

        async def _purge_dir(target: Path, message: str, info: bool = False) -> None:
            if info:
                logfire.info(message)
            else:
                logfire.warning(message)
            await asyncio.to_thread(shutil.rmtree, target, ignore_errors=True)

        with logfire.span("Disk Resume Check: {combo}", combo=self.config.name) as resume_span:
            job_dirs: list[tuple[Literal["local", "youtube"], Path]] = []
            for stype in ("local", "youtube"):
                source_dir = self.work_dir / stype
                if not source_dir.is_dir():
                    continue
                for d in source_dir.iterdir():
                    if d.is_dir():
                        job_dirs.append((stype, d))

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
            for folder_source_type, d in job_dirs:
                meta_file = d / "video_meta.json"
                if not meta_file.is_file():
                    await _purge_dir(
                        d,
                        f"Purging stale folder without metadata: {folder_source_type}/{d.name} in {self.config.name}",
                        info=True,
                    )
                    continue

                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if not isinstance(meta, dict):
                        await _purge_dir(d, f"Purging job folder with invalid metadata: {folder_source_type}/{d.name}")
                        continue

                    raw_path_str = str(meta.get("raw_path") or "")
                    raw_path = Path(raw_path_str) if raw_path_str else None
                    if not raw_path or not raw_path.is_file():
                        await _purge_dir(
                            d,
                            f"Purging job folder with missing raw video: {folder_source_type}/{d.name} "
                            f"in {self.config.name}",
                        )
                        continue

                    raw_ts = meta.get("timestamp", 0)
                    if isinstance(raw_ts, bool):
                        await _purge_dir(d, f"Purging job folder with invalid timestamp: {folder_source_type}/{d.name}")
                        continue
                    timestamp = int(raw_ts)

                    source_type = meta.get("source_type", folder_source_type)
                    if source_type not in ("local", "youtube"):
                        await _purge_dir(
                            d,
                            f"Purging job folder with invalid source_type {source_type!r}: "
                            f"{folder_source_type}/{d.name}",
                        )
                        continue

                    channel_raw = meta.get("channel_name")
                    parsed_meta = {
                        "video_id": str(meta.get("video_id") or d.name),
                        "uploader": str(meta.get("uploader") or "unknown"),
                        "channel_name": str(channel_raw) if channel_raw is not None else None,
                        "title": str(meta.get("title") or d.name),
                        "source_type": source_type,
                        "timestamp": timestamp,
                    }
                    valid_candidates.append((d, parsed_meta, raw_path))
                except Exception as e:
                    await _purge_dir(
                        d,
                        f"Purging job folder with corrupt metadata: {folder_source_type}/{d.name} ({e})",
                    )

            valid_candidates.sort(key=lambda item: _get_mtime(item[0]), reverse=True)

            resumed: tuple[Pipeline, Path] | None = None
            failed_resume_dirs: set[Path] = set()

            for job_dir, meta, raw_path in valid_candidates:
                video_id = meta["video_id"]
                uploader = meta["uploader"]
                channel_name = meta["channel_name"]
                title = meta["title"]
                source_type = meta["source_type"]
                timestamp = meta["timestamp"]
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
                    resumed = (pipeline, job_dir)
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

            pipeline, job_dir = resumed
            for cand_dir, _, _ in valid_candidates:
                if cand_dir != job_dir and cand_dir not in failed_resume_dirs:
                    await _purge_dir(
                        cand_dir,
                        f"Purging older job folder: {cand_dir.parent.name}/{cand_dir.name} in {self.config.name}",
                        info=True,
                    )

            self.pipeline = pipeline
            self.tasks.clear()
            self.active_summary_reencode = False
            self.active_reencode_segments.clear()
            self.preparing_video_info = None
            self._reencode_wake_event.clear()

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

    def detach_active_state(self, purge_disk: bool = True) -> Path | None:
        """Clears active in-memory pipeline state synchronously and returns any directory to purge.

        Args:
            purge_disk: Whether the active job directory should be purged by the caller. Defaults to True.

        Returns:
            Path | None: Job directory path to delete asynchronously, or None.
        """
        purge_dir = self.pipeline.job_dir if (purge_disk and self.pipeline is not None) else None
        self.pipeline = None
        self.preparing_video_info = None
        self.tasks.clear()
        self.active_summary_reencode = False
        self.active_reencode_segments.clear()
        self._reencode_wake_event.clear()
        return purge_dir
