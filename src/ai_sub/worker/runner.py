"""Core worker execution loop, distributed trace context attachment, and task processing."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import Any

import logfire

from ai_sub.config import AiSettings, RetrySettings
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
    """Manages the lifecycle, task execution, and quota states of a worker node."""

    settings: WorkerSettings
    client: WorkerClient
    tracker: WorkerQuotaTracker
    usable_models: list[str]

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

    async def _heartbeat_loop(self, task_id: str, stop_event: asyncio.Event) -> None:
        """Background task sending periodic heartbeats to maintain the task lease.

        Args:
            task_id: Unique task identifier.
            stop_event: Event signaling when task execution finishes.
        """
        interval = self.settings.heartbeat_interval_seconds
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    acked = await self.client.send_heartbeat(task_id, self.settings.worker_id)
                    if not acked:
                        logfire.warning(f"Heartbeat for task '{task_id}' was rejected by the server.")

    async def _execute_task(
        self,
        task: TaskClaimResponse,
        executor: GeminiExecutor,
        search_providers: list[BaseSearchProvider],
    ) -> dict[str, Any]:
        """Downloads media, runs GeminiExecutor, and returns validated output dictionary.

        Args:
            task: Assigned task details from the server.
            executor: Configured GeminiExecutor instance.
            search_providers: Active search provider instances.

        Returns:
            dict[str, Any]: Validated output data dictionary.

        Raises:
            ValueError: If target response schema is unknown.
        """
        raw_name = Path(task.media_url).name
        if not Path(raw_name).suffix:
            raw_name = f"{raw_name}.mp4"

        chunk_path = self.settings.temp_dir / f"{task.video_id}_{raw_name}"
        await self.client.download_media(task.media_url, chunk_path)

        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(task.task_id, stop_heartbeat))

        try:
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
            )

            return res.model_dump(mode="json")
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

    async def run(self) -> int:
        """Executes the main worker loop until the queue is drained or quotas are exhausted.

        Returns:
            int: Application exit code (0 for clean finish).
        """
        gemini_key = self.settings.get_gemini_api_key()
        if not gemini_key:
            logfire.error("Gemini API key is missing on worker. Set GEMINI_API_KEY environment variable.")
            return 1

        if not self.usable_models:
            logfire.info("All configured models are quota-exhausted until 00:00 PT. Exiting cleanly.")
            return 0

        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)

        ai_settings = AiSettings(key=gemini_key, search=self.settings.search)
        retry_settings = RetrySettings(per_run=3)
        executor = GeminiExecutor(ai_settings, retry_settings=retry_settings)

        tasks_completed = 0
        max_tasks = self.settings.max_tasks

        async with AsyncExitStack() as stack:
            search_providers = create_search_providers(self.settings.search)
            for sp in search_providers:
                await stack.enter_async_context(sp)

            while self.usable_models:
                if 0 < max_tasks <= tasks_completed:
                    logfire.info(f"Reached max tasks limit ({max_tasks}). Exiting cleanly.")
                    break

                claim = await self.client.claim_task(
                    worker_id=self.settings.worker_id,
                    models=self.usable_models,
                    search_tools=self.settings.search.tools,
                )
                if not claim:
                    logfire.info("No matching tasks available on the server. Exiting cleanly.")
                    break

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
                                tasks_completed += 1
                                logfire.info(f"Task '{claim.task_id}' processed and accepted by server.")
                            else:
                                logfire.warning(f"Task '{claim.task_id}' completion was not accepted by server.")

                        except QuotaExceededError as e:
                            logfire.warning(f"Daily quota reached for model '{claim.model}': {e}")
                            self.tracker.mark_exhausted(claim.model)
                            await self.client.fail_task(
                                task_id=claim.task_id,
                                worker_id=self.settings.worker_id,
                                reason="QUOTA_EXHAUSTED",
                                model=claim.model,
                                error_message=str(e),
                            )
                            if claim.model in self.usable_models:
                                self.usable_models.remove(claim.model)

                            if not self.usable_models:
                                logfire.info("All models now exhausted for today (reset at 00:00 PT). Exiting cleanly.")
                                break

                        except Exception as e:
                            logfire.exception(f"Execution error on task '{claim.task_id}': {e}")
                            await self.client.fail_task(
                                task_id=claim.task_id,
                                worker_id=self.settings.worker_id,
                                reason="EXECUTION_ERROR",
                                model=claim.model,
                                error_message=str(e),
                            )

        await self.client.close()
        return 0
