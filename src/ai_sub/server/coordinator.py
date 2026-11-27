"""Central server coordinator managing combination runners and worker task leases."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import urllib.parse
from collections import deque
from collections.abc import Callable
from pathlib import Path
from time import monotonic, time
from typing import Any, Literal

import logfire
from pydantic import ValidationError

from ai_sub.nfo import write_nfo
from ai_sub.pipeline import FallbackProbeSpec, probe_fallback_info
from ai_sub.protocol import TaskClaimResponse, TaskType
from ai_sub.server.combo_runner import ComboRunner, RunnerStatus
from ai_sub.server.dashboard import build_dashboard_data, format_logfire_query
from ai_sub.server.git_sync import GitSyncManager, redact_url_credentials
from ai_sub.server.models import ActivityEvent, ServerTask, TaskStatus
from ai_sub.server.server_config import ServerSettings

_PUSH_RETRY_COOLDOWN_S = 60.0
_ProbeContext = tuple[FallbackProbeSpec, ServerTask, int, int | None]


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Formats a count with singular or plural noun inflection.

    Args:
        count: Numeric count.
        singular: Singular noun form.
        plural: Optional irregular plural form (defaults to '<singular>s').

    Returns:
        str: Formatted '<count> <noun>' string.
    """
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def is_valid_video_id(video_id: str) -> bool:
    """Checks that a video ID is non-empty and free of path traversal characters.

    Args:
        video_id: Candidate video identifier string.

    Returns:
        bool: True if safe for directory path construction, False otherwise.
    """
    return video_id.strip() not in ("", ".", "..") and "/" not in video_id and "\\" not in video_id


def _release_task_lease(task: ServerTask) -> None:
    """Clears active worker lease metadata from a server task.

    Args:
        task: Target server task to release.
    """
    task.worker_id = None
    task.claimed_at = None
    task.last_heartbeat = None


def _build_completion_detail(task_type: TaskType, result: dict[str, Any]) -> str | None:
    """Builds a concise activity detail summary from a completed task result payload.

    Args:
        task_type: Completed AI task category.
        result: Validated AI result dictionary.

    Returns:
        str | None: Human-readable completion summary, or None.
    """
    if task_type == TaskType.SUMMARY:
        ch_len = len(result.get("chapters", []))
        ent_len = len(result.get("entities", []))
        return f"Summary: {_plural(ch_len, 'chapter')}, {_plural(ent_len, 'entity', 'entities')} extracted"
    if task_type == TaskType.SUBTITLES:
        sub_len = len(result.get("subs", result.get("subtitles", [])))
        music_len = len(result.get("detected_musical_performances", []))
        return f"Subtitles: {_plural(sub_len, 'line')} transcribed ({_plural(music_len, 'song')} detected)"
    if task_type == TaskType.LYRICS:
        lyr_len = len(result.get("subs", result.get("subtitles", [])))
        return f"Lyrics: {_plural(lyr_len, 'line')} refined"
    return None


class ComboRunnersCoordinator:
    """Orchestrates in-memory combination runners, worker task leases, and dashboard telemetry."""

    settings: ServerSettings
    git_sync: GitSyncManager
    runners: dict[str, ComboRunner]
    runner_activity_logs: dict[str, deque[ActivityEvent]]
    reencode_semaphore: asyncio.Semaphore
    _lock: asyncio.Lock
    _finalizing_runners: set[str]
    _push_retry_after: dict[str, float]
    _sweep_task: asyncio.Task[None] | None
    _finalize_task: asyncio.Task[None] | None
    _finalize_rerun_requested: bool
    _background_tasks: set[asyncio.Task[Any]]
    _stopping: bool
    on_runners_finalized: Callable[[], Any] | None
    on_video_aborted: Callable[[str, Literal["local", "youtube"], str, str], None] | None

    def __init__(self, settings: ServerSettings, git_sync: GitSyncManager) -> None:
        """Initializes the central combination runners coordinator.

        Args:
            settings: Server settings model.
            git_sync: Git synchronization and output repository manager.
        """
        self.settings = settings
        self.git_sync = git_sync
        self.runners = {}
        self.runner_activity_logs = {
            p_cfg.name: deque(maxlen=settings.coordinator.activitymaxlen) for p_cfg in settings.pipelines
        }
        self.reencode_semaphore = asyncio.Semaphore(settings.reencode.concurrency)
        self._lock = asyncio.Lock()
        self._finalizing_runners = set()
        self._push_retry_after = {}
        self._sweep_task = None
        self._finalize_task = None
        self._finalize_rerun_requested = False
        self._background_tasks = set()
        self._stopping = False
        self.on_runners_finalized = None
        self.on_video_aborted = None

        work_base = settings.coordinator.workdir
        val_buffer_ms = int(settings.coordinator.validationbuffer.total_seconds() * 1000)
        for p_cfg in settings.pipelines:
            self.runners[p_cfg.name] = ComboRunner(
                config=p_cfg,
                base_work_dir=work_base,
                reencode_semaphore=self.reencode_semaphore,
                reencode_settings=settings.reencode,
                split_settings=settings.split,
                retry_settings=settings.retry,
                max_reencode_concurrency=settings.reencode.concurrency,
                coordinator_lock=self._lock,
                on_pipeline_done=self.schedule_finalize_check,
                validation_buffer_ms=val_buffer_ms,
            )

    def _find_task(self, task_id: str) -> tuple[ComboRunner, ServerTask] | None:
        """Locates a ServerTask and its owning ComboRunner by task ID.

        Args:
            task_id: Unique task identifier.

        Returns:
            tuple[ComboRunner, ServerTask] | None: (runner, task) if found, else None.
        """
        for runner in self.runners.values():
            if task := runner.tasks.get(task_id):
                return runner, task
        return None

    def _get_claimed_task(
        self,
        task_id: str,
        worker_id: str,
        action: str,
        expected_generation: int | None = None,
        expected_task: ServerTask | None = None,
    ) -> tuple[ComboRunner, ServerTask] | None:
        """Locates a task and verifies that it is actively claimed by the specified worker.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker identifier asserting lease ownership.
            action: Operation label used in rejection warning logs.
            expected_generation: Optional lease generation that must still match. Defaults to None.
            expected_task: Optional ServerTask instance that must still match by identity. Defaults to None.

        Returns:
            tuple[ComboRunner, ServerTask] | None: (runner, task) if validly claimed, else None.
        """
        found = self._find_task(task_id)
        if not found:
            logfire.warning(f"{action} rejected for task '{task_id}' from worker '{worker_id}': task not found.")
            return None

        runner, task = found
        if expected_task is not None and task is not expected_task:
            logfire.warning(f"{action} rejected for task '{task_id}': task instance was replaced.")
            return None

        if task.status != TaskStatus.CLAIMED or task.worker_id != worker_id:
            logfire.warning(
                f"{action} rejected for task '{task_id}': status={task.status.value}, "
                f"assigned_worker={task.worker_id}, requesting_worker={worker_id}"
            )
            return None

        if expected_generation is not None and task.claim_generation != expected_generation:
            logfire.warning(
                f"{action} rejected for task '{task_id}': claim generation changed "
                f"(expected={expected_generation}, current={task.claim_generation})."
            )
            return None

        return runner, task

    def schedule_finalize_check(self) -> None:
        """Schedules a coalesced asynchronous completion check across all runners."""
        if self._stopping:
            return
        if self._finalize_task is not None and not self._finalize_task.done():
            self._finalize_rerun_requested = True
            return

        async def _run() -> None:
            try:
                while not self._stopping:
                    self._finalize_rerun_requested = False
                    await self.check_and_finalize_completed_runners()
                    if not self._finalize_rerun_requested:
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logfire.exception(f"Background finalization check error: {e}")
            finally:
                self._finalize_task = None

        bg_task = asyncio.create_task(_run())
        self._finalize_task = bg_task
        self._background_tasks.add(bg_task)
        bg_task.add_done_callback(self._background_tasks.discard)

    def _log_activity(
        self,
        event_type: Literal["complete", "retry", "fail", "timeout"],
        task: ServerTask,
        worker_id: str,
        duration_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        """Appends an event to the per-runner circular activity buffer with Logfire query string.

        Args:
            event_type: Event classification ('complete', 'retry', 'fail', 'timeout').
            task: Affected server task.
            worker_id: Worker identifier involved.
            duration_s: Execution duration in seconds if available. Defaults to None.
            detail: Optional extra details or error traceback. Defaults to None.
        """
        event = ActivityEvent(
            timestamp=time(),
            event_type=event_type,
            task_id=task.task_id,
            worker_id=worker_id,
            video_id=task.video_id,
            duration_s=round(duration_s, 1) if duration_s is not None else None,
            detail=detail,
            logfire_query=format_logfire_query(task),
        )

        runner_log = self.runner_activity_logs.setdefault(
            task.pipeline_name, deque(maxlen=self.settings.coordinator.activitymaxlen)
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
        if not is_valid_video_id(video_id):
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
            if runner := self.runners.get(runner_name):
                runner.clear_preparing()

    async def start(self) -> None:
        """Starts background lease monitoring and attempts disk resumption on all runners."""
        self._stopping = False
        with logfire.span("Coordinator Startup", runner_count=len(self.runners)) as start_span:
            resumed_count = 0
            for name, runner in self.runners.items():
                try:
                    if await runner.resume_from_disk():
                        resumed_count += 1
                        logfire.info(f"Runner '{name}' resumed active video from disk.")
                except Exception as e:
                    logfire.exception(f"Error resuming runner '{name}' from disk: {e}")
                    try:
                        # Preserve on-disk checkpoints when startup recovery raises unexpectedly
                        await self._clear_runner_after_finalize(runner, purge_disk=False)
                    except Exception as clear_err:
                        logfire.warning(f"Error resetting in-memory state for failed runner '{name}': {clear_err}")

            start_span.set_attribute("resumed_runners", resumed_count)

            try:
                await self.check_and_finalize_completed_runners()
            except Exception as e:
                logfire.exception(f"Error checking and finalizing completed runners on startup: {e}")

            self._sweep_task = asyncio.create_task(self._lease_sweeper_loop())

    async def stop(self) -> None:
        """Stops background sweeper and runner re-encoders, draining active finalizations gracefully."""
        self._stopping = True

        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

        for runner in self.runners.values():
            await runner.stop_background_tasks()

        # Drain in-flight finalizations/Git pushes cleanly instead of interrupting mid-push
        while self._background_tasks:
            pending = list(self._background_tasks)
            results = await asyncio.gather(*pending, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logfire.warning(f"Background task error during shutdown: {res}")

        # Re-stop runner re-encoders in case a background intake task spawned one while draining
        for runner in self.runners.values():
            await runner.stop_background_tasks()

    async def _lease_sweeper_loop(self) -> None:
        """Continuously checks for expired worker leases every 15 seconds.

        Heartbeat timeouts are treated as retryable failures and reset tasks to PENDING
        without incrementing non-retryable attempt counts toward maxruns.
        """
        timeout = self.settings.coordinator.heartbeattimeout.total_seconds()

        while True:
            await asyncio.sleep(15)
            now = time()
            async with self._lock:
                with logfire.span("Lease Sweeper Check", timeout_seconds=timeout, _level="debug") as sweep_span:
                    expired_count = 0
                    active_leases = 0

                    for runner in self.runners.values():
                        for task in list(runner.tasks.values()):
                            if task.status != TaskStatus.CLAIMED:
                                continue
                            active_leases += 1
                            reference_time = task.last_heartbeat or task.claimed_at or 0.0
                            if (now - reference_time) > timeout:
                                expired_count += 1
                                worker_id = task.worker_id or "unknown"
                                total_active_s = round(now - (task.claimed_at or now), 1)
                                diag_msg = (
                                    f"Lease timed out after {int(timeout)}s without heartbeat "
                                    f"(active: {total_active_s}s)."
                                )
                                _release_task_lease(task)
                                task.status = TaskStatus.PENDING
                                logfire.warning(
                                    f"Task '{task.task_id}' lease expired (worker: {worker_id}). "
                                    "Reset to PENDING (retryable)."
                                )
                                self._log_activity(
                                    event_type="timeout",
                                    task=task,
                                    worker_id=worker_id,
                                    duration_s=total_active_s,
                                    detail=diag_msg,
                                )

                    sweep_span.set_attribute("active_leases", active_leases)
                    sweep_span.set_attribute("expired_leases", expired_count)

            # Schedule finalization asynchronously so Git retries never block lease sweeping
            self.schedule_finalize_check()

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
                    if runner.status != RunnerStatus.PROCESSING or runner.pipeline is None:
                        continue
                    for task in runner.tasks.values():
                        if task.status != TaskStatus.PENDING or not runner.is_task_ready_for_claim(task):
                            continue
                        if task.model.lower() not in available_models:
                            continue

                        required_tools = runner.search_tools_for_task(task.task_type)
                        if required_tools and not all(t.lower() in worker_tools for t in required_tools):
                            continue

                        now = time()
                        task.status = TaskStatus.CLAIMED
                        task.worker_id = worker_id
                        task.claimed_at = now
                        task.last_heartbeat = now
                        task.claim_generation += 1

                        with logfire.span(
                            "Server Task Claimed: {task_id}",
                            task_id=task.task_id,
                            worker_id=worker_id,
                            model=task.model,
                            task_type=task.task_type.value,
                        ):
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
                        return task.to_claim_response(
                            pipeline=runner.pipeline,
                            search_tools=required_tools,
                            media_url=media_url,
                            retry=self.settings.retry,
                            validation_buffer=self.settings.coordinator.validationbuffer,
                        )

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
            claimed = self._get_claimed_task(task_id, worker_id, action="Heartbeat")
            if not claimed:
                return False
            _, task = claimed
            task.last_heartbeat = now
            return True

    async def _fail_task_locked(
        self,
        task: ServerTask,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
        is_retryable: bool = False,
        fallback_base_fps: float | None = None,
        has_audio: bool | None = None,
    ) -> bool:
        """Processes task failure and updates lease state while lock is held.

        Args:
            task: Affected server task.
            worker_id: Worker reporting the failure.
            reason: Error code or exception description.
            error_message: Optional error details. Defaults to None.
            is_retryable: Whether failure is retryable and exempt from attempt counting. Defaults to False.
            fallback_base_fps: Optional pre-probed base FPS for tier-0 keyframes fallback. Defaults to None.
            has_audio: Optional pre-probed audio-stream presence flag. Defaults to None.

        Returns:
            bool: True indicating the task failure was processed.
        """
        with logfire.span(
            "Process Task Failure: {task_id}",
            task_id=task.task_id,
            worker_id=worker_id,
            reason=reason,
        ) as fail_span:
            if error_message and error_message not in reason:
                detail = f"[{reason}] {error_message}"
            else:
                detail = reason or error_message or "Unknown failure"

            duration_s = (time() - task.claimed_at) if task.claimed_at else None
            _release_task_lease(task)

            if is_retryable:
                task.status = TaskStatus.PENDING
                self._log_activity(
                    event_type="retry",
                    task=task,
                    worker_id=worker_id,
                    duration_s=duration_s,
                    detail=detail,
                )
                logfire.warning(
                    f"Task '{task.task_id}' failed by worker '{worker_id}' (retryable: {reason}). Reset to PENDING."
                )
                fail_span.set_attribute("result", task.status.value)
                return True

            runner = self.runners.get(task.pipeline_name)
            if not runner:
                task.status = TaskStatus.FAILED
                self._log_activity(
                    event_type="fail",
                    task=task,
                    worker_id=worker_id,
                    duration_s=duration_s,
                    detail=detail,
                )
                fail_span.set_attribute("result", task.status.value)
                return True

            outcome = await runner.on_task_failed(
                task.task_id,
                fallback_base_fps=fallback_base_fps,
                has_audio=has_audio,
            )
            max_runs = self.settings.retry.resolve_for_model(task.model).maxruns

            if outcome.action == "fallback_reencode":
                if outcome.encoding == "audio" or outcome.fps == 0:
                    fps_str = "0 fps (audio-only)"
                elif outcome.fps is not None:
                    fps_str = f"{outcome.fps:.4g} fps"
                else:
                    fps_str = "unknown fps"
                detail = f"{detail}\n-> Falling back to {fps_str} (tier {outcome.fallback_tier})"
            elif outcome.action == "exhausted":
                detail = f"{detail}\n-> Exhausted all {max_runs} run(s) and fallback FPS tiers."
                logfire.error(f"Task '{task.task_id}' marked FAILED after exhausting retries and fallback tiers.")
            elif outcome.action == "retry":
                logfire.warning(
                    f"Task '{task.task_id}' failed by worker '{worker_id}' "
                    f"(attempt {outcome.tier_attempts}/{max_runs}). Reset to PENDING."
                )

            self._log_activity(
                event_type="fail",
                task=task,
                worker_id=worker_id,
                duration_s=duration_s,
                detail=detail,
            )
            fail_span.set_attribute("result", task.status.value)
            return True

    @staticmethod
    def _prepare_probe_context(runner: ComboRunner, task: ServerTask) -> _ProbeContext | None:
        """Captures off-lock probe spec, task instance, claim generation, and stage tier while refreshing heartbeat.

        Args:
            runner: Owning ComboRunner instance.
            task: Claimed ServerTask instance.

        Returns:
            _ProbeContext | None: (probe_spec, task, claim_generation, current_tier) if probing is needed, else None.
        """
        if runner.pipeline is None:
            return None
        probe_spec = runner.pipeline.get_fallback_probe_spec(task.task_type, task.segment_index)
        if probe_spec is None:
            return None
        stage_state = runner.pipeline.get_stage_media_state(task.task_type, task.segment_index)
        expected_tier = stage_state.fallback_tier if stage_state is not None else None
        # Refresh heartbeat so the lease sweeper does not reclaim the task while probing off-lock
        task.last_heartbeat = time()
        return probe_spec, task, task.claim_generation, expected_tier

    async def _finish_failure_with_probe(
        self,
        task_id: str,
        worker_id: str,
        action: str,
        reason: str,
        error_message: str | None,
        probe_ctx: _ProbeContext,
    ) -> bool:
        """Probes keyframe FPS and audio presence off-lock, re-verifies lease ownership, and records task failure.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker reporting the failure.
            action: Operation label for stale-probe rejection logs.
            reason: Error code or exception description.
            error_message: Optional error details.
            probe_ctx: Tuple of (probe_spec, expected_task, expected_generation, expected_tier).

        Returns:
            bool: True if failure was recorded, False if the lease or tier became stale while probing.
        """
        probe_spec, expected_task, expected_generation, expected_tier = probe_ctx
        fallback_base_fps, has_audio = await probe_fallback_info(probe_spec)

        async with self._lock:
            claimed = self._get_claimed_task(
                task_id,
                worker_id,
                action=action,
                expected_generation=expected_generation,
                expected_task=expected_task,
            )
            if not claimed:
                return False

            runner, task = claimed
            stage_state = (
                runner.pipeline.get_stage_media_state(task.task_type, task.segment_index)
                if runner.pipeline is not None
                else None
            )
            if stage_state is None or stage_state.fallback_tier != expected_tier:
                logfire.warning(f"Discarding stale {action.lower()} probe for '{task_id}': stage tier changed.")
                return False

            return await self._fail_task_locked(
                task=task,
                worker_id=worker_id,
                reason=reason,
                error_message=error_message,
                is_retryable=False,
                fallback_base_fps=fallback_base_fps,
                has_audio=has_audio,
            )

    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Records task completion and persists the result to disk.

        Checkpoint persistence errors (OSError) propagate without failing or revoking the lease so
        workers can retry submission without rerunning inference. Fallback probing triggered by
        validation failures runs outside self._lock and revalidates claim generation and tier.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker completing the task.
            result: Validated AI result dictionary.

        Returns:
            tuple[bool, str | None]: (accepted, validation_error_message). Returns (True, None) on
            acceptance, (False, None) when the lease is invalid/stale, or (False, error_detail) when
            server-side validation rejects the payload.

        Raises:
            OSError: If saving the durable YAML checkpoint fails on disk.
        """
        probe_ctx: _ProbeContext | None = None
        val_error_detail: str | None = None
        val_error_msg = ""

        async with self._lock:
            claimed = self._get_claimed_task(task_id, worker_id, action="Completion")
            if not claimed:
                return False, None

            runner, task = claimed
            with logfire.span("Process Complete Task: {task_id}", task_id=task_id, worker_id=worker_id) as comp_span:
                duration = time() - (task.claimed_at or time())
                try:
                    runner.on_task_completed(task_id, result)
                except OSError as io_err:
                    task.last_heartbeat = time()
                    comp_span.set_attribute("result", "persistence_error")
                    comp_span.set_attribute("error", str(io_err))
                    logfire.error(f"Checkpoint write failed for task '{task_id}': {io_err}")
                    raise
                except (ValidationError, ValueError, TypeError) as val_err:
                    logfire.warning(f"Task '{task_id}' validation failed: {val_err}. Routing to fail.")
                    comp_span.set_attribute("result", "validation_failed")
                    comp_span.set_attribute("error", str(val_err))
                    val_error_detail = str(val_err)
                    val_error_msg = f"Server validation error: {val_error_detail}"
                    if (probe_ctx := self._prepare_probe_context(runner, task)) is None:
                        await self._fail_task_locked(
                            task=task,
                            worker_id=worker_id,
                            reason="EXECUTION_ERROR",
                            error_message=val_error_msg,
                            is_retryable=False,
                        )
                        return False, val_error_detail
                else:
                    self._log_activity(
                        event_type="complete",
                        task=task,
                        worker_id=worker_id,
                        duration_s=duration,
                        detail=_build_completion_detail(task.task_type, result),
                    )
                    comp_span.set_attribute("result", "accepted")
                    comp_span.set_attribute("duration_s", round(duration, 1))
                    logfire.info(f"Task '{task.task_id}' COMPLETED by '{worker_id}' ({duration:.1f}s).")
                    return True, None

        assert probe_ctx is not None and val_error_detail is not None
        recorded = await self._finish_failure_with_probe(
            task_id=task_id,
            worker_id=worker_id,
            action="Validation failure",
            reason="EXECUTION_ERROR",
            error_message=val_error_msg,
            probe_ctx=probe_ctx,
        )
        return False, (val_error_detail if recorded else None)

    async def fail_task(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
        is_retryable: bool = False,
    ) -> bool:
        """Handles task failure, releasing the lease or marking permanently failed.

        Any fallback probing needed is executed outside self._lock and revalidates claim
        generation and tier before applying the outcome.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker reporting the failure.
            reason: Error code or exception description.
            error_message: Optional error details. Defaults to None.
            is_retryable: Whether failure is retryable and exempt from attempt counting. Defaults to False.

        Returns:
            bool: True if failure was recorded, False otherwise.
        """
        async with self._lock:
            claimed = self._get_claimed_task(task_id, worker_id, action="Failure report")
            if not claimed:
                return False

            runner, task = claimed
            if is_retryable or (probe_ctx := self._prepare_probe_context(runner, task)) is None:
                return await self._fail_task_locked(
                    task=task,
                    worker_id=worker_id,
                    reason=reason,
                    error_message=error_message,
                    is_retryable=is_retryable,
                )

        return await self._finish_failure_with_probe(
            task_id=task_id,
            worker_id=worker_id,
            action="Failure report",
            reason=reason,
            error_message=error_message,
            probe_ctx=probe_ctx,
        )

    def get_task_media_path(self, task_id: str) -> Path | None:
        """Retrieves the local media file path for an active task.

        Args:
            task_id: Unique task identifier.

        Returns:
            Path | None: File path if task exists and file is present, None otherwise.
        """
        found = self._find_task(task_id)
        if found and found[1].media_path.is_file():
            return found[1].media_path
        return None

    async def _clear_runner_after_finalize(self, runner: ComboRunner, purge_disk: bool = True) -> None:
        """Stops runner tasks, detaches state under lock, deletes job_dir off-lock if requested, and sets IDLE.

        Args:
            runner: Combination runner to clear and reset.
            purge_disk: Whether the active job directory should be purged on disk. Defaults to True.
        """
        await runner.stop_background_tasks()
        async with self._lock:
            self._push_retry_after.pop(runner.config.name, None)
            purge_dir = runner.detach_active_state(purge_disk=purge_disk)

        try:
            if purge_dir is not None and purge_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, purge_dir)
                except OSError as e:
                    logfire.warning(f"Could not purge job directory {purge_dir.name}: {e}")
        finally:
            async with self._lock:
                runner.status = RunnerStatus.IDLE

    async def finalize_runner_if_complete(self, runner: ComboRunner) -> bool:
        """Checks if a runner has finished all segments, stitches subtitles, and pushes to Git.

        If the summary stage permanently aborted across all fallback tiers, skips Git push,
        records the abort with ingestion so the runner does not loop, and clears the workspace.

        Args:
            runner: Combination runner instance to check and finalize.

        Returns:
            bool: True if runner was completed or aborted and cleared, False otherwise.
        """
        async with self._lock:
            if runner.config.name in self._finalizing_runners:
                return False
            if runner.is_idle() or not runner.pipeline:
                return False
            if not runner.is_video_complete():
                return False

            if runner.status == RunnerStatus.PUSH_FAILED and monotonic() < self._push_retry_after.get(
                runner.config.name, 0.0
            ):
                return False

            self._finalizing_runners.add(runner.config.name)
            runner.status = RunnerStatus.FINALIZING

        stage = "Finalization"
        try:
            pipe = runner.pipeline
            assert pipe is not None

            if pipe.summary_aborted:
                logfire.error(
                    f"Runner '{runner.config.name}' aborted '{pipe.title}' ({pipe.video_id}) "
                    "due to Summary exhaustion. Skipping Git push and holding from automatic retry."
                )
                if self.on_video_aborted is not None:
                    try:
                        self.on_video_aborted(runner.config.name, pipe.source_type, pipe.video_id, "Summary failed")
                    except Exception as cb_err:
                        logfire.warning(f"Error invoking on_video_aborted callback: {cb_err}")

                await self._clear_runner_after_finalize(runner)
                return True

            logfire.info(f"Runner '{runner.config.name}' finalizing for '{pipe.title}'...")

            stage = "Output finalization"
            with logfire.span(
                "Stitch Subtitles: {video_id}",
                combo=runner.config.name,
                video_id=pipe.video_id,
            ):
                generated_srts = await asyncio.to_thread(pipe.finalize, allow_partial=True)
                nfo_file = await asyncio.to_thread(write_nfo, pipe) if self.settings.coordinator.nfo else None

            yaml_files = list(pipe.job_dir.glob("*.yaml"))
            meta_file = pipe.job_dir / "video_meta.json"
            stage = "Git push"
            await self.git_sync.finalize_and_push(
                uploader=pipe.uploader,
                video_id=pipe.video_id,
                full_shortcode=runner.full_shortcode,
                srt_files=generated_srts,
                yaml_files=yaml_files,
                meta_file=meta_file if meta_file.is_file() else None,
                nfo_file=nfo_file,
            )

            stage = "Runner cleanup"
            await self._clear_runner_after_finalize(runner)
            logfire.info(f"Runner '{runner.config.name}' cleared and ready for next video.")
            return True
        except Exception as e:
            if stage in ("Output finalization", "Git push"):
                logfire.error(f"{stage} failed for '{runner.config.name}': {e}. Setting PUSH_FAILED.")
            else:
                logfire.exception(f"Unexpected error during {stage.lower()} for '{runner.config.name}': {e}")
            async with self._lock:
                runner.status = RunnerStatus.PUSH_FAILED
                self._push_retry_after[runner.config.name] = monotonic() + _PUSH_RETRY_COOLDOWN_S
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

        if finalized and not self._stopping and self.on_runners_finalized is not None:
            try:
                res = self.on_runners_finalized()
                if inspect.isawaitable(res):
                    bg_task = asyncio.create_task(self._run_intake_callback(res))
                    self._background_tasks.add(bg_task)
                    bg_task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                logfire.exception(f"Error scheduling on_runners_finalized callback: {e}")

        return finalized

    def _get_public_git_repo_url(self) -> str | None:
        """Returns the sanitized remote Git repository URL without embedded credentials, if configured.

        Returns:
            str | None: Clean remote repository URL or None.
        """
        if not self.git_sync.config or self.git_sync.config.repourl is None:
            return None
        return redact_url_credentials(str(self.git_sync.config.repourl))

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
        return build_dashboard_data(
            runners=self.runners,
            runner_activity_logs=self.runner_activity_logs,
            pipeline_queues=pipeline_queues,
            git_repo_url=self._get_public_git_repo_url(),
            youtube_cooldown=youtube_cooldown,
            total_tracked_videos=total_tracked_videos,
        )
