"""Core worker execution loop, per-model slot management, and concurrent task processing."""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from time import time
from typing import Any, cast

import logfire
from pydantic import BaseModel

from ai_sub.config import AiSettings, ServiceTier
from ai_sub.gemini_executor import GeminiExecutor, QuotaExceededError
from ai_sub.jobs.base import Prompt
from ai_sub.jobs.lyrics import LyricsAiResponse, prefetch_lyrics
from ai_sub.jobs.subtitles import DetectedMusicalPerformance, SubtitlesAiResponse
from ai_sub.jobs.summary import SummaryAiResponse
from ai_sub.protocol import TaskClaimResponse, TaskType
from ai_sub.search import BaseSearchProvider, create_search_providers
from ai_sub.worker.client import WorkerClient
from ai_sub.worker.config import WorkerSettings

TASK_RESPONSE_SCHEMAS: dict[TaskType, type[BaseModel]] = {
    TaskType.SUMMARY: SummaryAiResponse,
    TaskType.SUBTITLES: SubtitlesAiResponse,
    TaskType.LYRICS: LyricsAiResponse,
}


class WorkerRunner:
    """Manages the lifecycle, per-model execution slots, and in-memory quota states of a worker node."""

    settings: WorkerSettings
    client: WorkerClient
    _lock: asyncio.Lock
    _shutdown_event: asyncio.Event
    _model_stop_events: dict[str, asyncio.Event]
    _exhausted_models: set[str]
    _capped_models: set[str]
    _active_tasks: int
    _in_flight_claims: int
    _tasks_completed_per_model: dict[str, int]
    _last_busy_timestamp: float
    _deadline: float | None

    def __init__(self, settings: WorkerSettings) -> None:
        """Initializes the worker runner.

        Args:
            settings: Worker settings configuration.
        """
        self.settings = settings
        self.client = WorkerClient(
            base_url=str(settings.server_url),
            auth=settings.basic_auth,
        )
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._model_stop_events = {model: asyncio.Event() for model in dict.fromkeys(settings.models)}
        self._exhausted_models = set()
        self._capped_models = set()
        self._active_tasks = 0
        self._in_flight_claims = 0
        self._tasks_completed_per_model = {model: 0 for model in dict.fromkeys(settings.models)}
        self._last_busy_timestamp = time()
        self._deadline = None

    @property
    def _tasks_completed(self) -> int:
        """Total tasks completed across all models.

        Returns:
            int: Aggregate count of completed tasks.
        """
        return sum(self._tasks_completed_per_model.values())

    def _is_model_done(self, model: str) -> bool:
        """Checks if a model has exhausted its quota or reached its task limit.

        Args:
            model: Model identifier.

        Returns:
            bool: True if the model is done, False otherwise.
        """
        if model in self._exhausted_models or model in self._capped_models:
            return True
        if 0 < self.settings.max_tasks <= self._tasks_completed_per_model.get(model, 0):
            return True
        return False

    def _all_models_finished(self) -> bool:
        """Checks whether all configured models have finished execution.

        Returns:
            bool: True if all models are done, False otherwise.
        """
        unique_models = set(self.settings.models)
        return bool(unique_models and all(self._is_model_done(m) for m in unique_models))

    def _trigger_shutdown(self) -> None:
        """Signals global shutdown across all model slots."""
        self._shutdown_event.set()
        for ev in self._model_stop_events.values():
            ev.set()

    async def _heartbeat_loop(
        self,
        task_id: str,
        stop_event: asyncio.Event,
        execution_task: asyncio.Task[Any],
    ) -> None:
        """Background task sending periodic heartbeats to maintain the task lease.

        Args:
            task_id: Unique task identifier.
            stop_event: Event signaling when task execution finishes normally.
            execution_task: Active Gemini execution task to cancel if lease is revoked.
        """
        interval = self.settings.heartbeat_interval_seconds
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    acked = await self.client.send_heartbeat(task_id, self.settings.worker_id)
                    if acked is None:
                        continue
                    if not acked:
                        logfire.warning(f"Heartbeat for task '{task_id}' was rejected. Aborting execution immediately.")
                        execution_task.cancel()
                        break

    async def _run_inference(
        self,
        task: TaskClaimResponse,
        executor: GeminiExecutor,
        chunk_path: Path,
        search_providers: list[BaseSearchProvider],
    ) -> dict[str, Any]:
        """Runs the Gemini multimodal inference task and returns structured output.

        Args:
            task: Assigned task details from the server.
            executor: Configured Gemini executor.
            chunk_path: Local path to the media chunk file.
            search_providers: Active search provider instances.

        Returns:
            dict[str, Any]: Validated output data dictionary.

        Raises:
            ValueError: If target task type response schema is unknown.
        """
        response_cls = TASK_RESPONSE_SCHEMAS.get(task.task_type)
        if response_cls is None:
            raise ValueError(f"Unknown task type: '{task.task_type}'")

        active_tools = [p for p in search_providers if p.provider_name in task.search_tools]

        prompt_user = task.prompt_user
        if task.task_type == TaskType.LYRICS and task.detected_performances:
            try:
                performances = [DetectedMusicalPerformance.model_validate(p) for p in task.detected_performances]
                prefetched = await prefetch_lyrics(performances, active_tools)
                if prefetched and prefetched.strip():
                    search_block = f"<initial_search_results>\n  {prefetched.strip()}\n</initial_search_results>\n"
                    if "<draft_subtitles_to_refine>" in prompt_user:
                        prompt_user = prompt_user.replace(
                            "<draft_subtitles_to_refine>",
                            f"{search_block}<draft_subtitles_to_refine>",
                            1,
                        )
                    else:
                        prompt_user += f"\n{search_block}"
            except Exception as e:
                logfire.warning(f"Worker lyrics pre-fetch error: {e}")

        res = await executor.run_structured(
            prompt=Prompt(system=task.prompt_system, user=prompt_user),
            media_path=chunk_path,
            video_duration_ms=task.video_duration_ms,
            response_type=response_cls,
            model_name=task.model,
            service_tier=cast(ServiceTier | None, task.service_tier),
            tools=active_tools if active_tools else None,
            agentic_video=task.agentic_video,
            task_label=task.task_type.value,
            retry_settings=task.retry,
        )
        return res.model_dump(mode="json")

    async def _execute_task(
        self,
        task: TaskClaimResponse,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
    ) -> dict[str, Any]:
        """Downloads media, runs GeminiExecutor with lease-revocation monitoring, and returns validated output.

        Args:
            task: Assigned task details from the server including retry policy.
            executor: Configured GeminiExecutor instance.
            search_providers: Active search provider instances.

        Returns:
            dict[str, Any]: Validated output data dictionary.
        """
        raw_name = Path(task.media_url).name
        if not Path(raw_name).suffix:
            raw_name = f"{raw_name}.mp4"

        safe_task_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", task.task_id)
        chunk_path = self.settings.temp_dir / f"{safe_task_id}_{raw_name}"
        await self.client.download_media(task.media_url, chunk_path)

        stop_heartbeat = asyncio.Event()
        current_coro = self._run_inference(task, executor, chunk_path, search_providers)
        inference_task = asyncio.create_task(current_coro)
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(task.task_id, stop_heartbeat, inference_task))

        try:
            return await inference_task
        finally:
            stop_heartbeat.set()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            if chunk_path.is_file():
                try:
                    chunk_path.unlink()
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
        """Processes a claimed task, reporting results or failure back to the server.

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
        carrier = {"traceparent": claim.traceparent} if claim.traceparent else None
        trace_ctx = logfire.attach_context(carrier) if carrier else nullcontext()

        with trace_ctx:
            with logfire.span(
                "Worker Task: {task_id}",
                task_id=claim.task_id,
                pipeline=claim.pipeline_name,
                video_id=claim.video_id,
                model=claim.model,
                task_type=claim.task_type.value,
                worker_id=self.settings.worker_id,
                slot=slot_id,
            ):
                try:
                    result_dict = await self._execute_task(claim, executor, search_providers)
                    accepted = await self.client.complete_task(
                        task_id=claim.task_id,
                        worker_id=self.settings.worker_id,
                        task_type=claim.task_type,
                        result=result_dict,
                    )
                    if accepted:
                        async with self._lock:
                            count = self._tasks_completed_per_model.get(model, 0) + 1
                            self._tasks_completed_per_model[model] = count
                            if 0 < self.settings.max_tasks <= count and model not in self._capped_models:
                                self._capped_models.add(model)
                                logfire.info(
                                    f"[{slot_label}] Reached max tasks limit ({self.settings.max_tasks}) "
                                    f"for model '{model}'. Stopping new task claims for this model."
                                )
                                if model in self._model_stop_events:
                                    self._model_stop_events[model].set()
                                if self._all_models_finished():
                                    logfire.info(
                                        "All configured models have reached their task limits or exhausted quotas. "
                                        "Shutting down worker."
                                    )
                                    self._trigger_shutdown()
                        logfire.info(f"[{slot_label}] Task '{claim.task_id}' processed and accepted by server.")
                    else:
                        logfire.warning(f"[{slot_label}] Task '{claim.task_id}' completion was not accepted by server.")

                except asyncio.CancelledError:
                    # Distinguish slot task cancellation (Ctrl-C/shutdown) from inference subtask lease revocation
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    logfire.warning(
                        f"[{slot_label}] Task '{claim.task_id}' was cancelled "
                        "(lease revoked by server or slot draining)."
                    )

                except QuotaExceededError as e:
                    logfire.warning(f"[{slot_label}] Daily quota reached for model '{claim.model}': {e}")
                    await self.client.fail_task(
                        task_id=claim.task_id,
                        worker_id=self.settings.worker_id,
                        reason="QUOTA_EXHAUSTED",
                        model=claim.model,
                        error_message=str(e),
                    )
                    async with self._lock:
                        self._exhausted_models.add(model)
                        if model in self._model_stop_events:
                            self._model_stop_events[model].set()
                        if self._all_models_finished():
                            logfire.info(
                                "All configured models have exhausted quotas or reached task limits. "
                                "Shutting down worker."
                            )
                            self._trigger_shutdown()

                except Exception as e:
                    logfire.exception(f"[{slot_label}] Execution error on task '{claim.task_id}': {e}")
                    await self.client.fail_task(
                        task_id=claim.task_id,
                        worker_id=self.settings.worker_id,
                        reason="EXECUTION_ERROR",
                        model=claim.model,
                        error_message=str(e),
                    )

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
        stop_event = self._model_stop_events.get(model, self._shutdown_event)

        while not stop_event.is_set():
            async with self._lock:
                if model in self._exhausted_models:
                    logfire.info(f"[{slot_label}] Model quota exhausted. Exiting slot.")
                    break

                if model in self._capped_models or (
                    0 < self.settings.max_tasks <= self._tasks_completed_per_model.get(model, 0)
                ):
                    logfire.info(
                        f"[{slot_label}] Reached max tasks limit ({self.settings.max_tasks}) "
                        f"for model '{model}'. Exiting slot."
                    )
                    break

                if self._deadline is not None and time() >= self._deadline:
                    logfire.info(
                        f"[{slot_label}] Maximum run duration ({self.settings.max_duration_minutes:.1f}m) "
                        "reached. Draining slot and exiting."
                    )
                    self._trigger_shutdown()
                    break

                self._in_flight_claims += 1

            claim = None
            claim_failed = False
            try:
                claim = await self.client.claim_task(
                    worker_id=self.settings.worker_id,
                    models=[model],
                    search_tools=self.settings.search.tools,
                )
            except Exception as e:
                claim_failed = True
                logfire.warning(f"[{slot_label}] Failed to claim task from server: {e}. Retrying in 5s...")
            finally:
                async with self._lock:
                    self._in_flight_claims -= 1

            if claim_failed:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                continue

            if claim is not None:
                async with self._lock:
                    self._active_tasks += 1
                    self._last_busy_timestamp = time()

                try:
                    await self._process_claimed_task(claim, model, slot_id, executor, search_providers)
                finally:
                    async with self._lock:
                        self._active_tasks -= 1
                        self._last_busy_timestamp = time()
                continue

            # claim is None (HTTP 204: no task available for this model)
            async with self._lock:
                active_or_claiming = self._active_tasks + self._in_flight_claims
                idle_duration = time() - self._last_busy_timestamp
                idle_timeout_seconds = self.settings.idle_timeout_minutes * 60.0

                if active_or_claiming == 0 and idle_duration >= idle_timeout_seconds:
                    logfire.info(
                        f"[{slot_label}] All slots have been idle for {idle_duration / 60:.1f}m "
                        f"(timeout: {self.settings.idle_timeout_minutes:.1f}m). Shutting down worker."
                    )
                    self._trigger_shutdown()
                    break

                poll_seconds = self.settings.claim_poll_interval_minutes * 60.0
                if active_or_claiming == 0:
                    remaining_idle = max(1.0, idle_timeout_seconds - idle_duration)
                    sleep_duration = min(poll_seconds, remaining_idle)
                else:
                    sleep_duration = poll_seconds

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_duration)
                break
            except asyncio.TimeoutError:
                pass

    async def run(self) -> int:
        """Executes concurrent per-model worker slots until queues are drained or quotas are exhausted.

        Returns:
            int: Application exit code (0 for clean finish).
        """
        if not self.settings.gemini_key:
            logfire.error("Gemini API key is missing in worker configuration. Specify gemini_key in worker YAML.")
            return 1

        models = list(dict.fromkeys(self.settings.models))
        if not models:
            logfire.info("No configured models available. Exiting cleanly.")
            return 0

        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        ai_settings = AiSettings(key=self.settings.gemini_key, search=self.settings.search)
        executor = GeminiExecutor(ai_settings)

        concurrency = self.settings.concurrency
        total_slots = len(models) * concurrency
        limit_desc = (
            f"up to {self.settings.max_tasks} tasks per model"
            if self.settings.max_tasks > 0
            else "unlimited tasks per model"
        )
        logfire.info(
            f"Starting worker '{self.settings.worker_id}' with {len(models)} model(s), "
            f"{concurrency} slot(s) per model ({total_slots} total concurrent slot(s)), and {limit_desc}."
        )

        if self.settings.max_duration_minutes > 0:
            self._deadline = time() + (self.settings.max_duration_minutes * 60.0)
            logfire.info(
                f"Worker session time limit set to {self.settings.max_duration_minutes:.1f} minute(s). "
                "New task claims will stop after deadline."
            )

        async with AsyncExitStack() as stack:
            search_providers = create_search_providers(self.settings.search)
            for sp in search_providers:
                await stack.enter_async_context(sp)

            slots = [
                asyncio.create_task(self._model_slot_loop(model, slot_id, executor, search_providers))
                for model in models
                for slot_id in range(concurrency)
            ]
            await asyncio.gather(*slots)

        await self.client.close()
        total_completed = self._tasks_completed
        details = ", ".join(f"{m}: {c}" for m, c in self._tasks_completed_per_model.items())
        logfire.info(f"Worker finished cleanly. Total tasks completed: {total_completed} ({details}).")
        return 0
