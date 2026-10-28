"""Core worker execution loop, distributed trace context attachment, and concurrent task processing."""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from time import time
from typing import Any

import logfire

from ai_sub.config import AiSettings
from ai_sub.gemini_executor import GeminiExecutor, QuotaExceededError
from ai_sub.jobs.base import Prompt
from ai_sub.jobs.lyrics import LyricsAiResponse
from ai_sub.jobs.subtitles import SubtitlesAiResponse
from ai_sub.jobs.summary import SummaryAiResponse
from ai_sub.protocol import TaskClaimResponse
from ai_sub.search import BaseSearchProvider, create_search_providers
from ai_sub.worker.client import WorkerClient
from ai_sub.worker.config import WorkerSettings
from ai_sub.worker.quota import WorkerQuotaTracker


class WorkerRunner:
    """Manages the lifecycle, concurrent execution slots, and quota states of a worker node."""

    settings: WorkerSettings
    client: WorkerClient
    tracker: WorkerQuotaTracker
    usable_models: list[str]
    _deadline: float | None
    _lock: asyncio.Lock
    _task_completed_event: asyncio.Event
    _active_tasks: int
    _tasks_completed: int

    def __init__(self, settings: WorkerSettings) -> None:
        """Initializes the worker runner.

        Args:
            settings: Worker settings configuration.
        """
        self.settings = settings
        self.client = WorkerClient(
            base_url=str(settings.server_url),
            auth=settings.get_basic_auth_tuple(),
        )
        self.tracker = WorkerQuotaTracker(settings.quota_file)
        self.usable_models = self.tracker.filter_available_models(settings.models)
        self._deadline = None
        self._lock = asyncio.Lock()
        self._task_completed_event = asyncio.Event()
        self._active_tasks = 0
        self._tasks_completed = 0

    async def _heartbeat_loop(
        self,
        task_id: str,
        stop_event: asyncio.Event,
        execution_task: asyncio.Task[Any],
    ) -> None:
        """Background task sending periodic heartbeats to maintain the task lease.

        Aborts execution immediately if the lease was revoked or rejected by the server.

        Args:
            task_id: Unique task identifier.
            stop_event: Event signaling when task execution finishes normally.
            execution_task: The active Gemini execution task to cancel if lease is revoked.
        """
        interval = self.settings.heartbeat_interval_seconds
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    acked = await self.client.send_heartbeat(task_id, self.settings.worker_id)
                    if acked is None:
                        continue  # transient error; lease still valid server-side
                    if not acked:
                        logfire.warning(
                            f"Heartbeat for task '{task_id}' was rejected (lease revoked or expired). "
                            "Aborting in-flight execution immediately."
                        )
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
            ValueError: If target response schema is unknown.
        """
        prompt = Prompt(system=task.prompt_system, user=task.prompt_user)

        if task.response_schema_name == "SummaryAiResponse":
            response_cls = SummaryAiResponse
        elif task.response_schema_name == "SubtitlesAiResponse":
            response_cls = SubtitlesAiResponse
        elif task.response_schema_name == "LyricsAiResponse":
            response_cls = LyricsAiResponse
        else:
            raise ValueError(f"Unknown response schema: '{task.response_schema_name}'")

        active_tools = [p for p in search_providers if p.provider_name in task.search_tools]

        res = await executor.run_structured(
            prompt=prompt,
            media_path=chunk_path,
            video_duration_ms=task.video_duration_ms,
            response_type=response_cls,
            model_name=task.model,
            service_tier=task.service_tier,  # type: ignore[arg-type]
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

        # Unique task filename to prevent collisions between concurrent worker slots
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

    async def _slot_loop(
        self,
        slot_id: int,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
    ) -> None:
        """Independent task-claiming and execution loop for a single concurrent worker slot.

        Args:
            slot_id: Identifier index of this concurrency slot.
            executor: Configured Gemini executor.
            search_providers: List of initialized search providers.
        """
        while True:
            # 1. Exit if worker task completion cap reached
            if 0 < self.settings.max_tasks <= self._tasks_completed:
                logfire.debug(f"[Slot {slot_id}] Reached max tasks limit ({self.settings.max_tasks}). Exiting.")
                break

            # 2. Exit if maximum session duration reached
            if self._deadline is not None and time() >= self._deadline:
                logfire.info(
                    f"[Slot {slot_id}] Maximum run duration ({self.settings.max_duration_minutes:.1f}m) "
                    f"reached. Draining slot and exiting."
                )
                break

            # 3. Re-filter usable models against quota state
            async with self._lock:
                self.usable_models = self.tracker.filter_available_models(self.usable_models)
                current_models = list(self.usable_models)

            if not current_models:
                logfire.info(f"[Slot {slot_id}] No usable models remaining (all quota exhausted). Exiting.")
                break

            # 4. Synchronously claim next task under lock to avoid racing
            async with self._lock:
                if 0 < self.settings.max_tasks <= self._tasks_completed:
                    break
                if self._deadline is not None and time() >= self._deadline:
                    break
                claim = await self.client.claim_task(
                    worker_id=self.settings.worker_id,
                    models=current_models,
                    search_tools=self.settings.search.tools,
                )
                if claim is not None:
                    self._active_tasks += 1
                    self._task_completed_event.clear()

            # 5. Handle no matching tasks available
            if claim is None:
                async with self._lock:
                    active = self._active_tasks

                if active > 0:
                    poll_timeout = self.settings.claim_poll_interval_minutes * 60.0
                    try:
                        await asyncio.wait_for(self._task_completed_event.wait(), timeout=poll_timeout)
                    except asyncio.TimeoutError:
                        pass
                    continue
                else:
                    logfire.info(f"[Slot {slot_id}] No matching tasks on server and no active slots. Exiting.")
                    break

            # 6. Process claimed task
            carrier = {"traceparent": claim.traceparent} if claim.traceparent else None
            trace_ctx = (
                logfire.attach_context(carrier) if carrier and hasattr(logfire, "attach_context") else nullcontext()
            )

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
                                self._tasks_completed += 1
                            logfire.info(f"[Slot {slot_id}] Task '{claim.task_id}' processed and accepted by server.")
                        else:
                            logfire.warning(
                                f"[Slot {slot_id}] Task '{claim.task_id}' completion was not accepted by server."
                            )

                    except asyncio.CancelledError:
                        logfire.warning(
                            f"[Slot {slot_id}] Task '{claim.task_id}' was cancelled "
                            "(lease revoked by server or slot draining)."
                        )

                    except QuotaExceededError as e:
                        logfire.warning(f"[Slot {slot_id}] Daily quota reached for model '{claim.model}': {e}")
                        self.tracker.mark_exhausted(claim.model)
                        await self.client.fail_task(
                            task_id=claim.task_id,
                            worker_id=self.settings.worker_id,
                            reason="QUOTA_EXHAUSTED",
                            model=claim.model,
                            error_message=str(e),
                        )
                        async with self._lock:
                            if claim.model in self.usable_models:
                                self.usable_models.remove(claim.model)

                    except Exception as e:
                        logfire.exception(f"[Slot {slot_id}] Execution error on task '{claim.task_id}': {e}")
                        await self.client.fail_task(
                            task_id=claim.task_id,
                            worker_id=self.settings.worker_id,
                            reason="EXECUTION_ERROR",
                            model=claim.model,
                            error_message=str(e),
                        )

                    finally:
                        async with self._lock:
                            self._active_tasks -= 1
                            self._task_completed_event.set()

        # Wake up any other slots waiting so they can re-evaluate termination
        self._task_completed_event.set()

    async def run(self) -> int:
        """Executes concurrent worker slots until the queue is drained or quotas are exhausted.

        Returns:
            int: Application exit code (0 for clean finish).
        """
        gemini_key = self.settings.get_gemini_api_key()
        if not gemini_key:
            logfire.error("Gemini API key is missing in worker configuration. Specify gemini_key in worker YAML.")
            return 1

        if not self.usable_models:
            logfire.info("All configured models are quota-exhausted until 00:00 PT. Exiting cleanly.")
            return 0

        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)

        ai_settings = AiSettings(key=gemini_key, search=self.settings.search)
        executor = GeminiExecutor(ai_settings)

        concurrency = self.settings.concurrency
        logfire.info(f"Starting worker '{self.settings.worker_id}' with {concurrency} concurrent execution slot(s).")

        if self.settings.max_duration_minutes > 0:
            self._deadline = time() + (self.settings.max_duration_minutes * 60.0)
            logfire.info(
                f"Worker session time limit set to {self.settings.max_duration_minutes:.1f} minute(s). "
                f"New task claims will stop after deadline."
            )

        async with AsyncExitStack() as stack:
            search_providers = create_search_providers(self.settings.search)
            for sp in search_providers:
                await stack.enter_async_context(sp)

            slots = [
                asyncio.create_task(self._slot_loop(slot_id, executor, search_providers))
                for slot_id in range(concurrency)
            ]
            await asyncio.gather(*slots)

        await self.client.close()
        logfire.info(f"Worker finished cleanly. Total tasks completed: {self._tasks_completed}.")
        return 0
