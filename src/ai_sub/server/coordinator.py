"""Central server coordinator managing combination runners, worker task leases, and telemetry."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import urllib.parse
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Literal

import logfire

from ai_sub.protocol import (
    TaskClaimResponse,
    TaskType,
)
from ai_sub.server.combo_runner import ComboRunner, RunnerStatus
from ai_sub.server.config import ServerSettings
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.models import ActivityEvent, ServerTask, TaskStatus


def _format_logfire_query(task: ServerTask) -> str:
    """Formats the Logfire search query for a server task using its unique task ID.

    Args:
        task: Target server task instance.

    Returns:
        str: Formatted Logfire search query string.
    """
    return f"attributes->>'task_id' = '{task.task_id}'"


class ComboRunnersCoordinator:
    """Orchestrates in-memory combination runners, worker task leases, and dashboard telemetry."""

    settings: ServerSettings
    git_sync: GitSyncManager | None
    runners: dict[str, ComboRunner]
    runner_activity_logs: dict[str, deque[ActivityEvent]]
    reencode_semaphore: asyncio.Semaphore
    _lock: asyncio.Lock
    _finalizing_runners: set[str]
    _sweep_task: asyncio.Task[None] | None
    _background_tasks: set[asyncio.Task[Any]]
    on_runners_finalized: Callable[[], Any] | None

    def __init__(self, settings: ServerSettings, git_sync: GitSyncManager | None = None) -> None:
        """Initializes the central combination runners coordinator.

        Args:
            settings: Server settings model.
            git_sync: Optional Git synchronization manager. Defaults to None.
        """
        self.settings = settings
        self.git_sync = git_sync
        self.runners = {}
        self.runner_activity_logs = {
            p_cfg.name: deque(maxlen=settings.server.activity_log_maxlen) for p_cfg in settings.pipelines
        }
        self.reencode_semaphore = asyncio.Semaphore(settings.server.reencode_concurrency)
        self._lock = asyncio.Lock()
        self._finalizing_runners = set()
        self._sweep_task = None
        self._background_tasks = set()
        self.on_runners_finalized = None

        work_base = settings.server.work_dir
        for p_cfg in settings.pipelines:
            self.runners[p_cfg.name] = ComboRunner(
                config=p_cfg,
                base_work_dir=work_base,
                reencode_semaphore=self.reencode_semaphore,
                reencode_settings=settings.reencode,
                default_search_tools=settings.default_search_tools,
                split_settings=settings.split,
                retry_settings=settings.retry,
                coordinator_lock=self._lock,
                validation_buffer_ms=settings.server.validation_buffer_ms,
            )

    @property
    def pipelines(self) -> dict[str, ComboRunner]:
        """Provides backwards-compatible dictionary property mapping to active runners."""
        return self.runners

    def _log_activity(
        self,
        event_type: Any,
        task: ServerTask,
        worker_id: str,
        duration_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        """Appends an event to the per-runner circular activity buffer with Logfire query string on errors.

        Args:
            event_type: Event classification ('claim', 'complete', 'fail', 'timeout').
            task: Affected server task.
            worker_id: Worker identifier involved.
            duration_s: Execution duration in seconds if complete. Defaults to None.
            detail: Optional extra details or error traceback. Defaults to None.
        """
        now = time()
        time_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        query_str = _format_logfire_query(task) if event_type in ("fail", "timeout") else None

        event = ActivityEvent(
            timestamp=now,
            time_str=time_str,
            event_type=event_type,
            task_id=task.task_id,
            pipeline=task.pipeline_name,
            worker_id=worker_id,
            model=task.model,
            video_id=task.video_id,
            duration_s=round(duration_s, 1) if duration_s is not None else None,
            detail=detail,
            logfire_query=query_str,
        )

        runner_log = self.runner_activity_logs.setdefault(
            task.pipeline_name, deque(maxlen=self.settings.server.activity_log_maxlen)
        )
        runner_log.appendleft(event)

    async def acquire_idle_runner(
        self,
        runner_name: str,
        video_id: str,
        title: str,
        uploader: str,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "youtube",
    ) -> bool:
        """Atomically transitions an idle runner to PREPARING status under lock.

        Args:
            runner_name: Name of the target combination runner.
            video_id: Unique video identifier.
            title: Title of the video.
            uploader: Channel or uploader handle identifier.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'youtube'.

        Returns:
            bool: True if the runner was idle and acquired; False otherwise.
        """
        # Reject invalid or path-traversal video IDs that could corrupt filesystem cleanup
        if video_id in (".", "..") or not video_id.strip() or "/" in video_id or "\\" in video_id:
            logfire.warning(f"Rejected invalid or dangerous video ID for acquisition: {video_id!r}")
            return False

        async with self._lock:
            runner = self.runners.get(runner_name)
            if not runner or not runner.is_idle():
                return False
            runner.set_preparing(
                video_id=video_id,
                title=title,
                uploader=uploader,
                channel_name=channel_name,
                source_type=source_type,
            )
            return True

    async def release_preparing_runner(self, runner_name: str) -> None:
        """Atomically resets a preparing runner back to IDLE status if preparation fails.

        Args:
            runner_name: Name of the combination runner to reset.
        """
        async with self._lock:
            runner = self.runners.get(runner_name)
            if runner:
                runner.clear_preparing()

    async def assign_prepared_video(
        self,
        runner_name: str,
        video_id: str,
        uploader: str,
        title: str,
        raw_video_path: Path,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "local",
        timestamp: int = 0,
    ) -> bool:
        """Prepares media and sets video into the runner, synchronizing state safely under lock.

        Args:
            runner_name: Target runner name.
            video_id: Video identifier string.
            uploader: Channel or uploader handle.
            title: Video title.
            raw_video_path: Local media file path.
            channel_name: Channel display name. Defaults to None.
            source_type: Ingestion category ('local' or 'youtube'). Defaults to 'local'.
            timestamp: Video publication Unix timestamp. Defaults to 0.

        Returns:
            bool: True if video was prepared and assigned; False otherwise.
        """
        # Reject invalid or path-traversal video IDs
        if video_id in (".", "..") or not video_id.strip() or "/" in video_id or "\\" in video_id:
            logfire.warning(f"Rejected invalid or dangerous video ID for assignment: {video_id!r}")
            return False

        runner = self.runners.get(runner_name)
        if not runner:
            return False

        await runner.prepare_and_set_video(
            video_id=video_id,
            uploader=uploader,
            title=title,
            raw_video_path=raw_video_path,
            channel_name=channel_name,
            source_type=source_type,
            timestamp=timestamp,
            coordinator_lock=self._lock,
        )
        return True

    async def start(self) -> None:
        """Starts background lease monitoring and attempts disk resumption on all runners."""
        with logfire.span("Coordinator Startup", runner_count=len(self.runners)) as start_span:
            resumed_count = 0
            for name, runner in self.runners.items():
                try:
                    resumed = await runner.resume_from_disk()
                    if resumed:
                        resumed_count += 1
                        logfire.info(f"Runner '{name}' resumed active video from disk.")
                except Exception as e:
                    logfire.exception(f"Error resuming runner '{name}' from disk: {e}")
                    try:
                        await runner.clear_active_video()
                    except Exception as clear_err:
                        logfire.warning(f"Error clearing active video for failed runner '{name}': {clear_err}")

            start_span.set_attribute("resumed_runners", resumed_count)

            try:
                await self.check_and_finalize_completed_runners()
            except Exception as e:
                logfire.exception(f"Error checking and finalizing completed runners on startup: {e}")

            self._sweep_task = asyncio.create_task(self._lease_sweeper_loop())

    async def stop(self) -> None:
        """Stops background sweeper tasks gracefully."""
        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

        for bg_task in list(self._background_tasks):
            bg_task.cancel()
        self._background_tasks.clear()

    async def _lease_sweeper_loop(self) -> None:
        """Continuously checks for expired worker leases every 15 seconds."""
        timeout = self.settings.server.heartbeat_timeout_seconds
        max_runs = self.settings.retry.max_runs

        while True:
            await asyncio.sleep(15)
            now = time()
            async with self._lock:
                with logfire.span("Lease Sweeper Check", timeout_seconds=timeout, _level="debug") as sweep_span:
                    expired_count = 0
                    active_leases = 0

                    for runner in self.runners.values():
                        # Iterate over a snapshot list to avoid RuntimeError if failure handling modifies tasks
                        for task in list(runner.tasks.values()):
                            if task.status == TaskStatus.CLAIMED:
                                active_leases += 1
                                reference_time = task.last_heartbeat or task.claimed_at or 0.0
                                elapsed = now - reference_time
                                if elapsed > timeout:
                                    expired_count += 1
                                    worker_id = task.worker_id or "unknown"
                                    total_active_s = round(now - (task.claimed_at or now), 1)

                                    diag_msg = (
                                        f"Lease timed out after {timeout}s without heartbeat "
                                        f"(active: {total_active_s}s)."
                                    )

                                    task.worker_id = None
                                    task.claimed_at = None
                                    task.last_heartbeat = None
                                    task.attempts += 1

                                    try:
                                        runner.on_task_failed(task.task_id, task.attempts)
                                    except Exception as e:
                                        logfire.exception(f"Failed to persist failure for '{task.task_id}': {e}")

                                    if task.attempts >= max_runs:
                                        task.status = TaskStatus.FAILED
                                        logfire.error(
                                            f"Task '{task.task_id}' lease expired ({elapsed:.1f}s). Marked FAILED."
                                        )
                                        self._log_activity(
                                            event_type="fail",
                                            task=task,
                                            worker_id=worker_id,
                                            detail=f"{diag_msg} (max runs exhausted)",
                                        )
                                    else:
                                        task.status = TaskStatus.PENDING
                                        logfire.warning(
                                            f"Task '{task.task_id}' lease expired (worker: {worker_id}). "
                                            "Set to PENDING."
                                        )
                                        self._log_activity(
                                            event_type="timeout",
                                            task=task,
                                            worker_id=worker_id,
                                            detail=diag_msg,
                                        )

                    sweep_span.set_attribute("active_leases", active_leases)
                    sweep_span.set_attribute("expired_leases", expired_count)

            try:
                await self.check_and_finalize_completed_runners()
            except Exception as e:
                logfire.exception(f"Error checking and finalizing completed runners in lease sweeper: {e}")

    async def claim_task(
        self,
        worker_id: str,
        models: list[str],
        search_tools: list[str],
    ) -> TaskClaimResponse | None:
        """Matches a worker's available models and search capabilities to an eligible pending task.

        Args:
            worker_id: Unique worker identifier.
            models: List of unexhausted model identifiers offered by the worker.
            search_tools: List of search tools configured on the worker.

        Returns:
            TaskClaimResponse | None: Assigned task details or None if no matching tasks are available.
        """
        async with self._lock:
            with logfire.span(
                "Evaluate Task Claim: {worker_id}",
                worker_id=worker_id,
                offered_models=models,
                search_tools=search_tools,
                _level="debug",
            ) as claim_span:
                available_models = {m.strip().lower() for m in models}
                worker_tools = {t.strip().lower() for t in search_tools}

                for runner in self.runners.values():
                    for task in runner.tasks.values():
                        if task.status != TaskStatus.PENDING or not runner.is_task_ready_for_claim(task):
                            continue

                        if task.model.lower() not in available_models:
                            continue

                        if task.task_type in (TaskType.SUMMARY, TaskType.LYRICS):
                            if task.search_tools and not all(t.lower() in worker_tools for t in task.search_tools):
                                continue

                        now = time()
                        task.status = TaskStatus.CLAIMED
                        task.worker_id = worker_id
                        task.claimed_at = now
                        task.last_heartbeat = now

                        with logfire.span(
                            "Server Task Claimed: {task_id}",
                            task_id=task.task_id,
                            worker_id=worker_id,
                            model=task.model,
                            task_type=task.task_type.value,
                        ):
                            self._log_activity(event_type="claim", task=task, worker_id=worker_id)
                            claim_span.set_attribute("result", "claimed")
                            claim_span.set_attribute("task_id", task.task_id)
                            claim_span.set_attribute("task_type", task.task_type.value)
                            claim_span.set_attribute("model", task.model)

                            logfire.info(
                                f"Task '{task.task_id}' claimed by worker '{worker_id}' "
                                f"(model: {task.model}, type: {task.task_type.value})."
                            )

                        safe_task_id = urllib.parse.quote(task.task_id, safe=":")
                        safe_filename = urllib.parse.quote(task.media_path.name)
                        media_url = f"/api/v1/tasks/{safe_task_id}/media/{safe_filename}"
                        return task.to_claim_response(media_url=media_url, retry=self.settings.retry)

                claim_span.set_attribute("result", "none_available")
                return None

    async def record_heartbeat(
        self,
        task_id: str,
        worker_id: str,
    ) -> bool:
        """Renews a task's lease timestamp upon receiving a worker heartbeat.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker claiming the lease.

        Returns:
            bool: True if lease was successfully renewed, False otherwise.
        """
        now = time()
        async with self._lock:
            for runner in self.runners.values():
                if task_id in runner.tasks:
                    task = runner.tasks[task_id]
                    if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                        task.last_heartbeat = now
                        return True
                    else:
                        logfire.warning(
                            f"Heartbeat rejected for '{task_id}': status={task.status.value}, "
                            f"assigned_worker={task.worker_id}, requesting_worker={worker_id}"
                        )
                        return False

        logfire.warning(f"Heartbeat rejected: task '{task_id}' not found.")
        return False

    def _fail_task_locked(
        self,
        task: ServerTask,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
        is_transient: bool = False,
    ) -> bool:
        """Processes task failure and updates lease state while lock is held.

        Args:
            task: Affected server task.
            worker_id: Worker reporting the failure.
            reason: Error code or exception description.
            error_message: Optional error details. Defaults to None.
            is_transient: Whether failure is transient and exempt from attempt counting. Defaults to False.

        Returns:
            bool: True indicating the task failure was processed.
        """
        if error_message and error_message not in reason:
            detail = f"[{reason}] {error_message}"
        else:
            detail = reason or error_message or "Unknown failure"

        self._log_activity(event_type="fail", task=task, worker_id=worker_id, detail=detail)
        task.worker_id = None
        task.claimed_at = None
        task.last_heartbeat = None

        if not is_transient:
            task.attempts += 1
            max_runs = self.settings.retry.max_runs

            runner = self.runners.get(task.pipeline_name)
            if runner:
                try:
                    runner.on_task_failed(task.task_id, task.attempts)
                except Exception as e:
                    logfire.exception(f"Failed to persist failure for '{task.task_id}': {e}")

            if task.attempts >= max_runs:
                task.status = TaskStatus.FAILED
                logfire.error(f"Task '{task.task_id}' marked FAILED after {task.attempts}/{max_runs} runs.")
                return True

        task.status = TaskStatus.PENDING
        logfire.warning(
            f"Task '{task.task_id}' failed by worker '{worker_id}' (reason: {reason}, "
            f"transient: {is_transient}). Reset to PENDING."
        )
        return True

    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: dict[str, Any],
    ) -> bool:
        """Records task completion and hands the result to the runner.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker completing the task.
            result: Validated AI result dictionary.

        Returns:
            bool: True if completion was accepted, False otherwise.
        """
        async with self._lock:
            for runner in self.runners.values():
                if task_id in runner.tasks:
                    task = runner.tasks[task_id]
                    if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                        with logfire.span(
                            "Process Complete Task: {task_id}", task_id=task_id, worker_id=worker_id
                        ) as comp_span:
                            duration = time() - (task.claimed_at or time())
                            try:
                                runner.on_task_completed(task_id, result)
                            except Exception as val_err:
                                logfire.warning(f"Task '{task_id}' validation failed: {val_err}. Routing to fail.")
                                comp_span.set_attribute("result", "validation_failed")
                                comp_span.set_attribute("error", str(val_err))
                                self._fail_task_locked(
                                    task=task,
                                    worker_id=worker_id,
                                    reason="EXECUTION_ERROR",
                                    error_message=f"Server validation error: {val_err}",
                                    is_transient=False,
                                )
                                return False

                            detail = None
                            if task.task_type == TaskType.SUMMARY:
                                ch_len = len(result.get("chapters", []))
                                ent_len = len(result.get("entities", []))
                                detail = f"Summary: {ch_len} chapters, {ent_len} entities extracted"
                            elif task.task_type == TaskType.SUBTITLES:
                                sub_len = len(result.get("subs", result.get("subtitles", [])))
                                music_len = len(result.get("detected_musical_performances", []))
                                detail = f"Subtitles: {sub_len} lines transcribed ({music_len} songs detected)"
                            elif task.task_type == TaskType.LYRICS:
                                lyr_len = len(result.get("subs", result.get("subtitles", [])))
                                detail = f"Lyrics: {lyr_len} lines refined"

                            self._log_activity(
                                event_type="complete",
                                task=task,
                                worker_id=worker_id,
                                duration_s=duration,
                                detail=detail,
                            )
                            comp_span.set_attribute("result", "accepted")
                            comp_span.set_attribute("duration_s", round(duration, 1))
                            logfire.info(f"Task '{task_id}' COMPLETED by '{worker_id}' ({duration:.1f}s).")
                            return True

            with logfire.span("Process Complete Task Rejected: {task_id}", task_id=task_id, worker_id=worker_id):
                logfire.warning(f"Task '{task_id}' completion from worker '{worker_id}' was rejected.")
            return False

    async def fail_task(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
        is_transient: bool = False,
    ) -> bool:
        """Handles task failure, releasing the lease or marking permanently failed.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker reporting the failure.
            reason: Error code or exception description.
            error_message: Optional error details. Defaults to None.
            is_transient: Whether failure is transient and exempt from attempt counting. Defaults to False.

        Returns:
            bool: True if failure was recorded, False otherwise.
        """
        async with self._lock:
            for runner in self.runners.values():
                if task_id in runner.tasks:
                    task = runner.tasks[task_id]
                    if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                        with logfire.span(
                            "Process Task Failure: {task_id}",
                            task_id=task_id,
                            worker_id=worker_id,
                            reason=reason,
                        ) as fail_span:
                            success = self._fail_task_locked(
                                task=task,
                                worker_id=worker_id,
                                reason=reason,
                                error_message=error_message,
                                is_transient=is_transient,
                            )
                            fail_span.set_attribute("result", task.status.value)
                            fail_span.set_attribute("attempts", task.attempts)
                            return success

            with logfire.span("Process Task Failure Rejected: {task_id}", task_id=task_id, worker_id=worker_id):
                logfire.warning(f"Task '{task_id}' failure report from worker '{worker_id}' was rejected (not found).")
            return False

    def get_task_media_path(self, task_id: str) -> Path | None:
        """Retrieves the local media file path for an active task.

        Args:
            task_id: Unique task identifier.

        Returns:
            Path | None: File path if task exists and file is present, None otherwise.
        """
        for runner in self.runners.values():
            if task_id in runner.tasks:
                task = runner.tasks[task_id]
                if task.media_path.is_file():
                    return task.media_path
        return None

    async def finalize_runner_if_complete(self, runner: ComboRunner) -> bool:
        """Checks if a runner has finished all segments, stitches subtitles, and pushes to Git.

        Args:
            runner: Combination runner instance to check and finalize.

        Returns:
            bool: True if runner was completed and finalized, False otherwise.
        """
        async with self._lock:
            if runner.config.name in self._finalizing_runners:
                return False
            if runner.is_idle() or not runner.active_video:
                return False
            if runner.status != RunnerStatus.PUSH_FAILED and not runner.is_video_complete():
                return False

            self._finalizing_runners.add(runner.config.name)
            runner.status = RunnerStatus.FINALIZING

        try:
            vid = runner.active_video
            assert vid is not None
            logfire.info(f"Runner '{runner.config.name}' finalizing for '{vid.title}'...")

            srt_filename = f"{vid.filename_stem}.{runner.full_shortcode}.srt"
            temp_srt = vid.job_dir / srt_filename

            try:
                generated_srts = await asyncio.to_thread(runner.stitch_final_subtitles, temp_srt)
            except Exception as e:
                logfire.error(f"Subtitle stitching failed for '{runner.config.name}': {e}")
                async with self._lock:
                    runner.status = RunnerStatus.PUSH_FAILED
                return False

            yaml_files = list(vid.job_dir.glob("*.yaml"))
            meta_file = vid.job_dir / "video_meta.json"
            if self.git_sync:
                try:
                    await self.git_sync.finalize_and_push(
                        uploader=vid.uploader,
                        video_id=vid.video_id,
                        full_shortcode=runner.full_shortcode,
                        srt_files=generated_srts,
                        yaml_files=yaml_files,
                        meta_file=meta_file if meta_file.is_file() else None,
                    )
                except Exception as e:
                    logfire.error(f"Git push failed for '{runner.config.name}': {e}. Setting PUSH_FAILED.")
                    async with self._lock:
                        runner.status = RunnerStatus.PUSH_FAILED
                    return False
            else:
                # When Git sync is disabled, copy generated subtitles to durable local output directory
                output_dir = self.settings.server.work_dir / "output" / vid.uploader
                output_dir.mkdir(parents=True, exist_ok=True)
                for srt_path in generated_srts:
                    dest = output_dir / srt_path.name
                    shutil.copy2(srt_path, dest)
                    logfire.info(f"Saved durable subtitle output to {dest}")

            async with self._lock:
                await runner.clear_active_video()
                logfire.info(f"Runner '{runner.config.name}' cleared and ready for next video.")
                return True
        except Exception as e:
            logfire.exception(f"Unexpected error finalizing runner '{runner.config.name}': {e}")
            async with self._lock:
                runner.status = RunnerStatus.PUSH_FAILED
            return False
        finally:
            async with self._lock:
                self._finalizing_runners.discard(runner.config.name)

    @staticmethod
    async def _run_intake_callback(coro: Any) -> None:
        """Executes the intake callback in the background and reports any failures.

        Args:
            coro: Awaitable intake coroutine.
        """
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logfire.exception(f"Background intake callback failed: {e}")

    async def check_and_finalize_completed_runners(self) -> list[ComboRunner]:
        """Inspects all runners, finalizing any that have reached completion.

        Returns:
            list[ComboRunner]: List of runners that completed and were reset.
        """
        finalized: list[ComboRunner] = []
        for runner in list(self.runners.values()):
            if await self.finalize_runner_if_complete(runner):
                finalized.append(runner)

        if finalized and self.on_runners_finalized is not None:
            try:
                res = self.on_runners_finalized()
                if inspect.isawaitable(res):
                    bg_task = asyncio.create_task(self._run_intake_callback(res))
                    self._background_tasks.add(bg_task)
                    bg_task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                logfire.exception(f"Error scheduling on_runners_finalized callback: {e}")

        return finalized

    def get_dashboard_data(
        self,
        pipeline_queues: dict[str, dict[str, Any]],
        youtube_cooldown: dict[str, Any] | None = None,
        total_tracked_videos: int = 0,
    ) -> dict[str, Any]:
        """Gathers macroscopic metrics and active states for the live WebUI dashboard.

        Args:
            pipeline_queues: Mapping of runner names to their outstanding queue dictionaries.
            youtube_cooldown: Optional YouTube intake cooldown status dictionary.
            total_tracked_videos: Count of all media files and streams tracked in memory. Defaults to 0.

        Returns:
            dict[str, Any]: Complete dashboard state dictionary.
        """
        now = time()
        pipeline_data = []

        for name, runner in self.runners.items():
            active_info = None
            stages_info = None

            if runner.active_video and runner.active_video.pipeline:
                pipe = runner.active_video.pipeline
                total_chunks = len(pipe.segment_paths)
                reenc_count = len(pipe.reencoded_segment_paths)
                pending_reenc = max(0, total_chunks - reenc_count)

                active_info = {
                    "video_id": runner.active_video.video_id,
                    "title": runner.active_video.title,
                    "uploader": runner.active_video.uploader,
                    "channel_name": runner.active_video.channel_name,
                    "duration_ms": runner.active_video.duration_ms,
                    "source_type": runner.active_video.source_type,
                }

                summary_task = next((t for t in runner.tasks.values() if t.task_type == TaskType.SUMMARY), None)
                sub_tasks = [t for t in runner.tasks.values() if t.task_type == TaskType.SUBTITLES]
                lyr_tasks = [t for t in runner.tasks.values() if t.task_type == TaskType.LYRICS]

                sum_done = (
                    1
                    if runner.summary_response or (summary_task and summary_task.status == TaskStatus.COMPLETED)
                    else 0
                )
                sum_prog = 1 if summary_task and summary_task.status == TaskStatus.CLAIMED else 0

                sub_done = sum(1 for t in sub_tasks if t.status == TaskStatus.COMPLETED)
                sub_prog = sum(1 for t in sub_tasks if t.status == TaskStatus.CLAIMED)
                sub_concluded = sum(1 for t in sub_tasks if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED))

                lyr_done = sum(1 for t in lyr_tasks if t.status == TaskStatus.COMPLETED)
                lyr_prog = sum(1 for t in lyr_tasks if t.status == TaskStatus.CLAIMED)

                active_lyr_tasks = [
                    {
                        "task_id": t.task_id,
                        "worker_id": t.worker_id,
                        "elapsed_s": int(now - t.claimed_at) if t.claimed_at else None,
                    }
                    for t in lyr_tasks
                    if t.status == TaskStatus.CLAIMED
                ]

                has_singing = bool(lyr_tasks)
                if sub_concluded < total_chunks:
                    lyr_status = "in_progress" if lyr_prog > 0 else "pending"
                else:
                    if lyr_tasks:
                        lyr_status = (
                            "completed"
                            if lyr_done == len(lyr_tasks)
                            else ("in_progress" if lyr_prog > 0 else "pending")
                        )
                    else:
                        lyr_status = "skipped"

                stages_info = {
                    "summary": {
                        "total": 1,
                        "done": sum_done,
                        "in_progress": sum_prog,
                        "status": "completed" if sum_done else ("claimed" if sum_prog else "pending"),
                        "worker_id": summary_task.worker_id if summary_task else None,
                        "elapsed_s": int(now - summary_task.claimed_at)
                        if summary_task and summary_task.claimed_at
                        else None,
                    },
                    "reencoding": {
                        "total": total_chunks,
                        "done": reenc_count,
                        "in_progress": 1 if pending_reenc > 0 else 0,
                        "pending": pending_reenc,
                    },
                    "subtitles": {
                        "total": total_chunks,
                        "done": sub_done,
                        "in_progress": sub_prog,
                    },
                    "lyrics": {
                        "total": len(lyr_tasks),
                        "done": lyr_done,
                        "in_progress": lyr_prog,
                        "has_singing": has_singing,
                        "status": lyr_status,
                        "active_leases": active_lyr_tasks,
                    },
                }

            tasks_list = [
                {
                    "task_id": t.task_id,
                    "type": t.task_type.value,
                    "status": t.status.value,
                    "worker_id": t.worker_id,
                    "attempts": t.attempts,
                    "elapsed_s": int(now - t.claimed_at) if t.claimed_at else None,
                    "logfire_query": _format_logfire_query(t),
                }
                for t in runner.tasks.values()
            ]

            q_info = pipeline_queues.get(name, {"total_count": 0, "items": []})
            queue_total = q_info.get("total_count", 0)
            next_in_queue = q_info.get("items", [])

            pipeline_data.append(
                {
                    "name": name,
                    "status": runner.status.value,
                    "models": {
                        "summary": runner.config.summary_model,
                        "subtitles": runner.config.subtitles_model,
                        "lyrics": runner.config.lyrics_model,
                    },
                    "idle": runner.is_idle(),
                    "active_video": active_info,
                    "preparing_video": runner.preparing_video_info,
                    "stages": stages_info,
                    "tasks": tasks_list,
                    "queue_total": queue_total,
                    "next_in_queue": next_in_queue,
                    "recent_activity": [event.model_dump() for event in self.runner_activity_logs.get(name, [])],
                }
            )

        return {
            "server_time": now,
            "server_time_str": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "youtube_cooldown": youtube_cooldown or {"in_cooldown": False},
            "total_tracked_videos": total_tracked_videos,
            "pipelines": pipeline_data,
        }
