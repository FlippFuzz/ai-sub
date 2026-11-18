"""Core worker execution loop, per-model slot management, and concurrent task processing."""

from __future__ import annotations

import asyncio
import urllib.parse
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import Any, TypeVar

import logfire
from httpx import HTTPStatusError

from ai_sub.gemini_executor import GeminiExecutor, QuotaExceededError, is_retryable_error
from ai_sub.jobs import (
    LyricsJob,
    SubtitlesJob,
    SummaryJob,
    WorkerContext,
)
from ai_sub.protocol import TaskClaimResponse, TaskType
from ai_sub.search import BaseSearchProvider, create_search_providers
from ai_sub.worker.client import VersionMismatchError, WorkerClient
from ai_sub.worker.worker_config import WorkerSettings

T = TypeVar("T")


class LeaseRevokedError(Exception):
    """Raised when a task lease is revoked or rejected by the server."""


async def _run_cancellable(
    coro: Coroutine[Any, Any, T],
    task_id: str,
    stage: str,
    lease_revoked: asyncio.Event,
    set_active_subtask: Callable[[asyncio.Task[Any] | None], None],
) -> T:
    """Executes a coroutine as a cancellable subtask, translating heartbeat revocations into LeaseRevokedError.

    Args:
        coro: Coroutine to schedule as an asyncio task.
        task_id: Unique task identifier for diagnostics.
        stage: Human-readable stage description (e.g. 'media download').
        lease_revoked: Event indicating whether the server rejected the task lease.
        set_active_subtask: Callback registering or clearing the active subtask handle.

    Returns:
        T: Result produced by the wrapped coroutine.

    Raises:
        asyncio.CancelledError: If the subtask is cancelled while the lease is still active.
        LeaseRevokedError: If the lease was revoked before or during execution.
    """
    if lease_revoked.is_set():
        coro.close()
        raise LeaseRevokedError(f"Lease for task '{task_id}' was rejected before {stage}.")

    subtask = asyncio.create_task(coro)
    set_active_subtask(subtask)
    try:
        return await subtask
    except asyncio.CancelledError:
        if lease_revoked.is_set():
            raise LeaseRevokedError(f"Lease for task '{task_id}' was rejected during {stage}.")
        raise
    finally:
        set_active_subtask(None)


class WorkerRunner:
    """Manages the lifecycle, per-model execution slots, and in-memory quota states of a worker node."""

    settings: WorkerSettings
    client: WorkerClient
    _cond: asyncio.Condition
    _shutdown: bool
    _exhausted_models: set[str]
    _tasks_completed_per_model: dict[str, int]
    _tasks_in_flight_per_model: dict[str, int]
    _last_busy_timestamp: float
    _deadline: float | None
    _version_mismatch: bool
    _fatal_error: bool

    def __init__(self, settings: WorkerSettings) -> None:
        """Initializes the worker runner.

        Args:
            settings: Worker settings configuration.
        """
        self.settings = settings
        self.client = WorkerClient(
            base_url=str(settings.serverurl),
            auth=settings.basic_auth,
        )
        unique_models = list(dict.fromkeys(settings.models))
        self._cond = asyncio.Condition()
        self._shutdown = False
        self._exhausted_models = set()
        self._tasks_completed_per_model = {model: 0 for model in unique_models}
        self._tasks_in_flight_per_model = {model: 0 for model in unique_models}
        self._last_busy_timestamp = monotonic()
        self._deadline = None
        self._version_mismatch = False
        self._fatal_error = False

    def _is_model_done(self, model: str) -> bool:
        """Checks if a model has exhausted its quota or reached its task limit.

        Args:
            model: Model identifier.

        Returns:
            bool: True if the model is done, False otherwise.
        """
        if model in self._exhausted_models:
            return True
        if 0 < self.settings.maxtasks <= self._tasks_completed_per_model.get(model, 0):
            return True
        return False

    def _all_models_finished(self) -> bool:
        """Checks whether all configured models have finished execution.

        Returns:
            bool: True if all models are done, False otherwise.
        """
        unique_models = set(self.settings.models)
        return bool(unique_models and all(self._is_model_done(m) for m in unique_models))

    def _should_stop(self, model: str) -> bool:
        """Checks whether a slot for this model should stop claiming tasks (caller holds _cond).

        Args:
            model: Model identifier.

        Returns:
            bool: True if global shutdown, model completion, or maxduration deadline was reached.
        """
        return (
            self._shutdown
            or self._is_model_done(model)
            or (self._deadline is not None and monotonic() >= self._deadline)
        )

    async def _heartbeat_loop(
        self,
        task_id: str,
        stop_event: asyncio.Event,
        lease_revoked: asyncio.Event,
        on_lease_rejected: Callable[[], None],
    ) -> None:
        """Background task sending periodic heartbeats to maintain the task lease.

        Args:
            task_id: Unique task identifier.
            stop_event: Event signaling when task execution and submission finish normally.
            lease_revoked: Event set when the server rejects a heartbeat lease renewal.
            on_lease_rejected: Callback to cancel any currently active work immediately.
        """
        interval = self.settings.heartbeatinterval.total_seconds()
        hb_num = 0

        with logfire.span("Heartbeats", task_id=task_id, _level="debug"):
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    if not stop_event.is_set():
                        hb_num += 1
                        with logfire.span("Heartbeat Ping #{num}", num=hb_num, task_id=task_id, _level="debug"):
                            acked = await self.client.send_heartbeat(
                                task_id=task_id,
                                worker_id=self.settings.id,
                            )
                            if acked is None:
                                logfire.warning(
                                    f"Heartbeat #{hb_num} for '{task_id}' was unacknowledged (retryable error)."
                                )
                                continue
                            if not acked:
                                logfire.warning(
                                    f"Heartbeat #{hb_num} for task '{task_id}' was rejected. "
                                    "Aborting execution immediately."
                                )
                                lease_revoked.set()
                                on_lease_rejected()
                                break

    async def _execute_task(
        self,
        task: TaskClaimResponse,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
        lease_revoked: asyncio.Event,
        set_active_subtask: Callable[[asyncio.Task[Any] | None], None],
    ) -> dict[str, Any]:
        """Downloads media, runs the canonical Job instance, and returns validated output.

        Args:
            task: Assigned task details from the server including retry policy.
            executor: Configured GeminiExecutor instance.
            search_providers: Active search provider instances.
            lease_revoked: Event signaling whether the server revoked the task lease.
            set_active_subtask: Callback registering the currently running cancellable subtask.

        Returns:
            dict[str, Any]: Validated output data dictionary.

        Raises:
            ValueError: If task type is unknown or structured input is invalid.
        """
        raw_name = Path(urllib.parse.unquote(task.media_url)).name
        if not Path(raw_name).suffix:
            raw_name = f"{raw_name}.mp4"

        chunk_dir = self.settings.workdir / task.pipeline_name / task.video_id
        chunk_path = chunk_dir / raw_name

        try:
            with logfire.span("Downloading Task Media", url=task.media_url):
                await _run_cancellable(
                    coro=self.client.download_media(task.media_url, chunk_path),
                    task_id=task.task_id,
                    stage="media download",
                    lease_revoked=lease_revoked,
                    set_active_subtask=set_active_subtask,
                )

            active_tools = [p for p in search_providers if p.provider_name in task.search_tools]
            ctx = WorkerContext(
                executor=executor,
                media_path=chunk_path,
                search_providers=active_tools,
            )
            validation_buffer_ms = int(task.validation_buffer.total_seconds() * 1000)

            if task.task_type == TaskType.SUMMARY:
                job = SummaryJob(
                    name="summary",
                    video_duration_ms=task.video_duration_ms,
                    user_context=task.user_context,
                    media_path=chunk_path,
                    model_name=task.model,
                    service_tier=task.service_tier,
                    agentic_video=task.agentic_video,
                    retry_settings=task.retry,
                    validation_buffer_ms=validation_buffer_ms,
                )
            elif task.task_type == TaskType.SUBTITLES:
                job = SubtitlesJob(
                    name=task.task_id,
                    video_duration_ms=task.video_duration_ms,
                    segment_offset_ms=task.segment_offset_ms,
                    global_summary=task.global_summary,
                    user_context=task.user_context,
                    media_path=chunk_path,
                    model_name=task.model,
                    service_tier=task.service_tier,
                    agentic_video=task.agentic_video,
                    retry_settings=task.retry,
                    validation_buffer_ms=validation_buffer_ms,
                )
            elif task.task_type == TaskType.LYRICS:
                if not task.draft_subtitles:
                    raise ValueError(
                        f"Lyrics task '{task.task_id}' rejected: missing draft_subtitles required for refinement."
                    )
                job = LyricsJob(
                    name=task.task_id,
                    video_duration_ms=task.video_duration_ms,
                    segment_offset_ms=task.segment_offset_ms,
                    draft_subtitles=task.draft_subtitles,
                    detected_performances=task.detected_performances,
                    global_summary=task.global_summary,
                    user_context=task.user_context,
                    media_path=chunk_path,
                    model_name=task.model,
                    service_tier=task.service_tier,
                    agentic_video=task.agentic_video,
                    retry_settings=task.retry,
                    validation_buffer_ms=validation_buffer_ms,
                )
            else:
                raise ValueError(f"Unknown task type: '{task.task_type}'")

            res = await _run_cancellable(
                coro=job.execute(ctx),
                task_id=task.task_id,
                stage="inference",
                lease_revoked=lease_revoked,
                set_active_subtask=set_active_subtask,
            )
            return res.model_dump(mode="json")

        finally:
            if chunk_path.is_file():
                try:
                    chunk_path.unlink()
                    parent = chunk_path.parent
                    while parent != self.settings.workdir and parent.exists():
                        try:
                            parent.rmdir()
                            parent = parent.parent
                        except OSError:
                            break
                except OSError:
                    pass

    async def _process_claimed_task(
        self,
        claim: TaskClaimResponse,
        model: str,
        slot_id: int,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
    ) -> None:
        """Processes a claimed task, maintaining heartbeats across execution and completion submission.

        Args:
            claim: Assigned task payload from the server.
            model: Model identifier bound to this slot.
            slot_id: Concurrency slot index.
            executor: Configured Gemini executor.
            search_providers: Active search provider instances.

        Raises:
            asyncio.CancelledError: If the worker slot task itself is cancelled.
        """
        slot_label = f"{model}#{slot_id}"
        stop_heartbeat = asyncio.Event()
        lease_revoked = asyncio.Event()
        active_subtask: asyncio.Task[Any] | None = None

        def _set_active(subtask: asyncio.Task[Any] | None) -> None:
            nonlocal active_subtask
            active_subtask = subtask

        def _cancel_active() -> None:
            if active_subtask and not active_subtask.done():
                active_subtask.cancel()

        with logfire.span(
            "Worker Task: {task_id}",
            task_id=claim.task_id,
            pipeline=claim.pipeline_name,
            video_id=claim.video_id,
            model=claim.model,
            task_type=claim.task_type.value,
            worker_id=self.settings.id,
            slot=slot_id,
        ):
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(
                    task_id=claim.task_id,
                    stop_event=stop_heartbeat,
                    lease_revoked=lease_revoked,
                    on_lease_rejected=_cancel_active,
                )
            )

            try:
                try:
                    result_dict = await self._execute_task(
                        task=claim,
                        executor=executor,
                        search_providers=search_providers,
                        lease_revoked=lease_revoked,
                        set_active_subtask=_set_active,
                    )
                    accepted = await _run_cancellable(
                        coro=self.client.complete_task(
                            task_id=claim.task_id,
                            worker_id=self.settings.id,
                            result=result_dict,
                        ),
                        task_id=claim.task_id,
                        stage="completion submission",
                        lease_revoked=lease_revoked,
                        set_active_subtask=_set_active,
                    )
                finally:
                    stop_heartbeat.set()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                if accepted:
                    async with self._cond:
                        count = self._tasks_completed_per_model.get(model, 0) + 1
                        self._tasks_completed_per_model[model] = count
                        if count == self.settings.maxtasks:
                            logfire.info(
                                f"[{slot_label}] Reached max tasks limit ({self.settings.maxtasks}) "
                                f"for model '{model}'. Stopping new task claims for this model."
                            )
                        if self._all_models_finished():
                            logfire.info(
                                "All configured models have reached their task limits or exhausted quotas. "
                                "Shutting down worker."
                            )
                            self._shutdown = True
                        self._cond.notify_all()
                    logfire.info(f"[{slot_label}] Task '{claim.task_id}' processed and accepted by server.")
                else:
                    logfire.warning(f"[{slot_label}] Task '{claim.task_id}' completion was not accepted by server.")

            except LeaseRevokedError as e:
                logfire.warning(f"[{slot_label}] {e}")

            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                logfire.warning(
                    f"[{slot_label}] Task '{claim.task_id}' was cancelled (lease revoked by server or slot draining)."
                )

            except QuotaExceededError as e:
                logfire.warning(f"[{slot_label}] Daily quota reached for model '{claim.model}': {e}")
                err_msg = str(e).strip() or "Daily quota limit reached"
                await self.client.fail_task(
                    task_id=claim.task_id,
                    worker_id=self.settings.id,
                    reason=f"QUOTA_EXHAUSTED: {err_msg}",
                    error_message=err_msg,
                    is_retryable=True,
                )
                async with self._cond:
                    self._exhausted_models.add(model)
                    if self._all_models_finished():
                        logfire.info(
                            "All configured models have exhausted quotas or reached task limits. Shutting down worker."
                        )
                        self._shutdown = True
                    self._cond.notify_all()

            except Exception as e:
                is_retryable = is_retryable_error(e)
                err_msg = str(e).strip() or type(e).__name__
                prefix = "RETRYABLE_ERROR" if is_retryable else "EXECUTION_ERROR"
                fail_reason = f"[{prefix}] {err_msg}"
                logfire.exception(f"[{slot_label}] Execution error on task '{claim.task_id}': {e}")
                await self.client.fail_task(
                    task_id=claim.task_id,
                    worker_id=self.settings.id,
                    reason=fail_reason,
                    error_message=err_msg,
                    is_retryable=is_retryable,
                )

    async def _wait_for_condition(self, timeout_s: float, model: str) -> None:
        """Sleeps while holding _cond until timeout expires, slot should stop, or maxduration deadline arrives.

        Args:
            timeout_s: Maximum duration to wait in seconds.
            model: Model identifier bound to this slot.
        """
        deadline = monotonic() + timeout_s
        if self._deadline is not None:
            deadline = min(deadline, self._deadline)

        while not self._should_stop(model):
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    await self._cond.wait()
            except TimeoutError:
                break

    async def _model_slot_loop(
        self,
        model: str,
        slot_id: int,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
    ) -> None:
        """Independent execution loop dedicated to a single model concurrency slot.

        Args:
            model: Target Gemini model identifier assigned to this slot.
            slot_id: Concurrency index for this model.
            executor: Configured Gemini executor.
            search_providers: List of initialized search providers.
        """
        slot_label = f"{model}#{slot_id}"
        idle_timeout_seconds = self.settings.idletimeout.total_seconds()
        poll_seconds = self.settings.pollinterval.total_seconds()

        while True:
            async with self._cond:
                while not self._should_stop(model) and 0 < self.settings.maxtasks <= (
                    self._tasks_completed_per_model.get(model, 0) + self._tasks_in_flight_per_model.get(model, 0)
                ):
                    await self._cond.wait()

                if self._deadline is not None and monotonic() >= self._deadline and not self._shutdown:
                    max_dur_m = self.settings.maxduration.total_seconds() / 60.0
                    logfire.info(
                        f"[{slot_label}] Maximum run duration ({max_dur_m:.1f}m) reached. Draining slot and exiting."
                    )
                    self._shutdown = True
                    self._cond.notify_all()

                if self._should_stop(model):
                    if model in self._exhausted_models:
                        logfire.info(f"[{slot_label}] Model quota exhausted. Exiting slot.")
                    elif 0 < self.settings.maxtasks <= self._tasks_completed_per_model.get(model, 0):
                        logfire.info(
                            f"[{slot_label}] Reached max tasks limit ({self.settings.maxtasks}) "
                            f"for model '{model}'. Exiting slot."
                        )
                    break

                self._tasks_in_flight_per_model[model] = self._tasks_in_flight_per_model.get(model, 0) + 1

            claim = None
            claim_failed = False
            try:
                claim = await self.client.claim_task(
                    worker_id=self.settings.id,
                    models=[model],
                    search_tools=[str(t) for t in self.settings.ai.search.tools],
                )
            except VersionMismatchError as e:
                logfire.error(
                    f"[{slot_label}] ai-sub package version mismatch: {e}. "
                    "Shutting down worker so worker.sh can update to latest version."
                )
                async with self._cond:
                    self._tasks_in_flight_per_model[model] = max(0, self._tasks_in_flight_per_model[model] - 1)
                    self._version_mismatch = True
                    self._shutdown = True
                    self._cond.notify_all()
                break
            except HTTPStatusError as e:
                if e.response.status_code in (401, 403, 404):
                    logfire.error(
                        f"[{slot_label}] Server rejected task claim with HTTP {e.response.status_code}: {e}. "
                        "Check serverurl and serverauth. Shutting down worker."
                    )
                    async with self._cond:
                        self._tasks_in_flight_per_model[model] = max(0, self._tasks_in_flight_per_model[model] - 1)
                        self._fatal_error = True
                        self._shutdown = True
                        self._cond.notify_all()
                    break
                claim_failed = True
                logfire.warning(f"[{slot_label}] Failed to claim task from server: {e}. Retrying in 5s...")
            except Exception as e:
                claim_failed = True
                logfire.warning(f"[{slot_label}] Failed to claim task from server: {e}. Retrying in 5s...")

            if claim_failed:
                async with self._cond:
                    self._tasks_in_flight_per_model[model] = max(0, self._tasks_in_flight_per_model[model] - 1)
                    if sum(self._tasks_in_flight_per_model.values()) == 0:
                        idle_duration = monotonic() - self._last_busy_timestamp
                        error_timeout = idle_timeout_seconds if idle_timeout_seconds > 0 else 15.0
                        if idle_duration >= error_timeout:
                            logfire.error(
                                f"[{slot_label}] Server unreachable or failing claims for {idle_duration:.1f}s. "
                                "Shutting down worker."
                            )
                            self._fatal_error = True
                            self._shutdown = True
                            self._cond.notify_all()
                            break
                    self._cond.notify_all()
                    await self._wait_for_condition(5.0, model)
                continue

            if claim is not None:
                async with self._cond:
                    self._last_busy_timestamp = monotonic()
                try:
                    await self._process_claimed_task(claim, model, slot_id, executor, search_providers)
                finally:
                    async with self._cond:
                        self._tasks_in_flight_per_model[model] = max(0, self._tasks_in_flight_per_model[model] - 1)
                        self._last_busy_timestamp = monotonic()
                        self._cond.notify_all()
                continue

            # claim is None (HTTP 204): release reservation and check idle timeout atomically under _cond
            async with self._cond:
                self._tasks_in_flight_per_model[model] = max(0, self._tasks_in_flight_per_model[model] - 1)
                if sum(self._tasks_in_flight_per_model.values()) == 0:
                    idle_duration = monotonic() - self._last_busy_timestamp
                    if idle_duration >= idle_timeout_seconds:
                        logfire.info(
                            f"[{slot_label}] All slots have been idle for {idle_duration / 60:.1f}m "
                            f"(timeout: {idle_timeout_seconds / 60:.1f}m). Shutting down worker."
                        )
                        self._shutdown = True
                        self._cond.notify_all()
                        break

                    wait_s = min(poll_seconds, max(1.0, idle_timeout_seconds - idle_duration))
                else:
                    wait_s = poll_seconds

                self._cond.notify_all()
                await self._wait_for_condition(wait_s, model)

    async def run(self) -> int:
        """Executes concurrent per-model worker slots until queues are drained or quotas are exhausted.

        Returns:
            int: Application exit code (0 for clean finish, 1 for abnormal exit or update required).
        """
        if not self.settings.ai.key or not self.settings.ai.key.get_secret_value().strip():
            logfire.error(
                "Gemini API key is missing in worker configuration. "
                "Specify ai.key in worker YAML or AISUBWORKER_AI_KEY / GEMINI_API_KEY environment variable."
            )
            return 1

        models = list(dict.fromkeys(self.settings.models))
        if not models:
            logfire.info("No configured models available. Exiting cleanly.")
            return 0

        self.settings.workdir.mkdir(parents=True, exist_ok=True)
        self._last_busy_timestamp = monotonic()

        concurrency = self.settings.concurrency
        total_slots = len(models) * concurrency
        limit_desc = (
            f"up to {self.settings.maxtasks} tasks per model"
            if self.settings.maxtasks > 0
            else "unlimited tasks per model"
        )
        logfire.info(
            f"Starting worker '{self.settings.id}' with {len(models)} model(s), "
            f"{concurrency} slot(s) per model ({total_slots} total concurrent slot(s)), and {limit_desc}."
        )

        max_duration_s = self.settings.maxduration.total_seconds()
        if max_duration_s > 0:
            self._deadline = monotonic() + max_duration_s
            logfire.info(
                f"Worker session time limit set to {max_duration_s / 60:.1f} minute(s). "
                "New task claims will stop after deadline."
            )

        async with AsyncExitStack() as stack:
            search_providers = create_search_providers(self.settings.ai.search)
            for sp in search_providers:
                await stack.enter_async_context(sp)

            executor = GeminiExecutor(self.settings.ai)
            stack.push_async_callback(executor.aclose)

            slots = [
                asyncio.create_task(self._model_slot_loop(model, slot_id, executor, search_providers))
                for model in models
                for slot_id in range(concurrency)
            ]
            await asyncio.gather(*slots)

        await self.client.close()

        if self._version_mismatch:
            logfire.warning("Worker exited due to an ai-sub package version mismatch. Update required.")
            return 1

        if self._fatal_error:
            logfire.error("Worker exited due to a server authentication or connection error.")
            return 1

        total_completed = sum(self._tasks_completed_per_model.values())
        details = ", ".join(f"{m}: {c}" for m, c in self._tasks_completed_per_model.items())
        logfire.info(f"Worker finished cleanly. Total tasks completed: {total_completed} ({details}).")
        return 0
