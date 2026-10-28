"""Central server coordinator managing dynamic pipelines, task dispatch, worker leases, and activity logs."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import logfire

from ai_sub.protocol import (
    TaskClaimResponse,
    TaskType,
)
from ai_sub.server.config import ServerSettings
from ai_sub.server.models import ActivityEvent, ServerTask, TaskStatus
from ai_sub.server.pipeline import ComboPipeline


class PipelineCoordinator:
    """Orchestrates in-memory pipeline runners, worker task leases, and dashboard telemetry."""

    settings: ServerSettings
    pipelines: dict[str, ComboPipeline]
    activity_log: deque[ActivityEvent]
    _lock: asyncio.Lock
    _sweep_task: asyncio.Task[None] | None

    def __init__(self, settings: ServerSettings) -> None:
        """Initializes the central pipeline coordinator.

        Args:
            settings: Server settings model.
        """
        self.settings = settings
        self.pipelines = {}
        self.activity_log = deque(maxlen=50)
        self._lock = asyncio.Lock()
        self._sweep_task = None

        work_base = settings.server.work_dir
        for p_cfg in settings.pipelines:
            self.pipelines[p_cfg.name] = ComboPipeline(
                config=p_cfg,
                base_work_dir=work_base,
                reencode_settings=settings.reencode,
                search_settings=settings.search,
                split_settings=settings.split,
                retry_settings=settings.retry,
                validation_buffer_ms=settings.server.validation_buffer_ms,
            )

    def _log_activity(
        self,
        event_type: Any,
        task: ServerTask,
        worker_id: str,
        duration_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        """Appends an event to the circular activity buffer.

        Args:
            event_type: Event classification ('claim', 'complete', 'fail', 'timeout').
            task: Affected server task.
            worker_id: Worker identifier involved.
            duration_s: Execution duration in seconds if complete. Defaults to None.
            detail: Optional extra details or error traceback. Defaults to None.
        """
        now = time()
        time_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.activity_log.appendleft(
            ActivityEvent(
                timestamp=now,
                time_str=time_str,
                event_type=event_type,
                task_id=task.task_id,
                pipeline=task.pipeline_name,
                worker_id=worker_id,
                model=task.model,
                duration_s=round(duration_s, 1) if duration_s is not None else None,
                detail=detail,
            )
        )

    async def start(self) -> None:
        """Starts background lease monitoring and attempts disk resumption on all pipelines."""
        with logfire.span("Coordinator Startup", pipeline_count=len(self.pipelines)) as start_span:
            resumed_count = 0
            for name, pipe in self.pipelines.items():
                resumed = await pipe.resume_from_disk()
                if resumed:
                    resumed_count += 1
                    logfire.info(f"Pipeline '{name}' resumed active video from disk.")

            start_span.set_attribute("resumed_pipelines", resumed_count)
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

    async def _lease_sweeper_loop(self) -> None:
        """Continuously checks for expired worker leases every 15 seconds."""
        timeout = self.settings.server.heartbeat_timeout_seconds
        while True:
            await asyncio.sleep(15)
            now = time()
            async with self._lock:
                with logfire.span("Lease Sweeper Check", timeout_seconds=timeout, _level="debug") as sweep_span:
                    expired_count = 0
                    active_leases = 0

                    for pipe in self.pipelines.values():
                        for task in pipe.tasks.values():
                            if task.status == TaskStatus.CLAIMED and task.last_heartbeat is not None:
                                active_leases += 1
                                elapsed = now - task.last_heartbeat
                                if elapsed > timeout:
                                    expired_count += 1
                                    logfire.warning(
                                        f"Task '{task.task_id}' lease expired (worker: {task.worker_id}, "
                                        f"last heartbeat: {elapsed:.1f}s ago). Returning to PENDING."
                                    )
                                    self._log_activity(
                                        event_type="timeout",
                                        task=task,
                                        worker_id=task.worker_id or "unknown",
                                        detail=f"Lease timed out after {timeout}s without heartbeat",
                                    )
                                    task.status = TaskStatus.PENDING
                                    task.worker_id = None
                                    task.claimed_at = None
                                    task.last_heartbeat = None
                                    task.attempts += 1

                    sweep_span.set_attribute("active_leases", active_leases)
                    sweep_span.set_attribute("expired_leases", expired_count)

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

                for pipe in self.pipelines.values():
                    for task in pipe.tasks.values():
                        if task.status != TaskStatus.PENDING:
                            continue

                        if task.model.lower() not in available_models:
                            continue

                        if task.task_type in (TaskType.SUMMARY, TaskType.LYRICS):
                            if task.search_tools and not any(t.lower() in worker_tools for t in task.search_tools):
                                continue

                        now = time()
                        task.status = TaskStatus.CLAIMED
                        task.worker_id = worker_id
                        task.claimed_at = now
                        task.last_heartbeat = now

                        self._log_activity(event_type="claim", task=task, worker_id=worker_id)
                        claim_span.set_attribute("result", "claimed")
                        claim_span.set_attribute("task_id", task.task_id)
                        claim_span.set_attribute("task_type", task.task_type.value)
                        claim_span.set_attribute("model", task.model)

                        logfire.info(
                            f"Task '{task.task_id}' claimed by worker '{worker_id}' "
                            f"(model: {task.model}, type: {task.task_type.value})."
                        )

                        media_filename = task.media_path.name
                        return TaskClaimResponse(
                            task_id=task.task_id,
                            task_type=task.task_type,
                            pipeline_name=task.pipeline_name,
                            video_id=task.video_id,
                            model=task.model,
                            media_url=f"/api/v1/tasks/{task.task_id}/media/{media_filename}",
                            video_duration_ms=task.video_duration_ms,
                            prompt_system=task.prompt.system,
                            prompt_user=task.prompt.user,
                            response_schema_name=task.response_schema_name,
                            traceparent=task.traceparent,
                            search_tools=task.search_tools,
                            agentic_video=task.agentic_video,
                            detected_performances=task.detected_performances,
                            # Pass server-configured retry and backoff timing to worker
                            retry=self.settings.retry,
                        )

                claim_span.set_attribute("result", "none_available")
                return None

    async def record_heartbeat(self, task_id: str, worker_id: str) -> bool:
        """Renews a task's lease timestamp upon receiving a worker heartbeat.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker claiming the lease.

        Returns:
            bool: True if lease was successfully renewed; False if task was revoked or completed.
        """
        async with self._lock:
            for pipe in self.pipelines.values():
                if task_id in pipe.tasks:
                    task = pipe.tasks[task_id]
                    if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                        task.last_heartbeat = time()
                        return True
        return False

    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: dict[str, Any],
    ) -> bool:
        """Records task completion and hands the result to the pipeline runner.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker completing the task.
            result: Validated AI result dictionary.

        Returns:
            bool: True if completion was accepted; False otherwise.
        """
        async with self._lock:
            with logfire.span("Process Complete Task: {task_id}", task_id=task_id, worker_id=worker_id) as comp_span:
                for pipe in self.pipelines.values():
                    if task_id in pipe.tasks:
                        task = pipe.tasks[task_id]
                        if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                            duration = time() - (task.claimed_at or time())
                            pipe.on_task_completed(task_id, result)

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
                                detail = f"Lyrics: {lyr_len} lines refined with song lyrics"

                            self._log_activity(
                                event_type="complete",
                                task=task,
                                worker_id=worker_id,
                                duration_s=duration,
                                detail=detail,
                            )
                            comp_span.set_attribute("result", "accepted")
                            comp_span.set_attribute("duration_s", round(duration, 1))
                            logfire.info(
                                f"Task '{task_id}' marked COMPLETED by worker '{worker_id}' (elapsed: {duration:.1f}s)."
                            )
                            return True

                comp_span.set_attribute("result", "rejected_not_claimed")
                logfire.warning(f"Task '{task_id}' completion from worker '{worker_id}' was rejected.")
                return False

    async def fail_task(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
    ) -> bool:
        """Handles task failure, releasing the lease or marking permanently failed if max_runs reached.

        Args:
            task_id: Unique task identifier.
            worker_id: Worker reporting the failure.
            reason: Error code (e.g. 'QUOTA_EXHAUSTED' or 'EXECUTION_ERROR').
            error_message: Optional error details.

        Returns:
            bool: True if failure was recorded; False otherwise.
        """
        async with self._lock:
            with logfire.span(
                "Process Task Failure: {task_id}",
                task_id=task_id,
                worker_id=worker_id,
                reason=reason,
            ) as fail_span:
                for pipe in self.pipelines.values():
                    if task_id in pipe.tasks:
                        task = pipe.tasks[task_id]
                        if task.status == TaskStatus.CLAIMED and task.worker_id == worker_id:
                            detail = f"[{reason}] {error_message or 'No message provided'}"
                            self._log_activity(
                                event_type="fail",
                                task=task,
                                worker_id=worker_id,
                                detail=detail,
                            )
                            task.worker_id = None
                            task.claimed_at = None
                            task.last_heartbeat = None

                            # Non-quota errors burn an attempt toward max_runs
                            if reason != "QUOTA_EXHAUSTED":
                                task.attempts += 1
                                max_runs = self.settings.retry.max_runs
                                if task.attempts >= max_runs:
                                    task.status = TaskStatus.FAILED
                                    fail_span.set_attribute("result", "permanently_failed")
                                    fail_span.set_attribute("attempts", task.attempts)
                                    logfire.error(
                                        f"Task '{task_id}' marked FAILED after exhausting "
                                        f"{task.attempts}/{max_runs} runs. Reason: {reason}."
                                    )
                                    return True

                            task.status = TaskStatus.PENDING
                            fail_span.set_attribute("result", "returned_to_pending")
                            fail_span.set_attribute("attempts", task.attempts)
                            logfire.warning(
                                f"Task '{task_id}' failed by worker '{worker_id}' (reason: {reason}). "
                                f"Details: {error_message or 'none'}. Returning to PENDING."
                            )
                            return True

                fail_span.set_attribute("result", "task_not_found")
                return False

    def get_task_media_path(self, task_id: str) -> Path | None:
        """Retrieves the local media file path for an active task.

        Args:
            task_id: Unique task identifier.

        Returns:
            Path | None: File path if task exists and file is present; None otherwise.
        """
        for pipe in self.pipelines.values():
            if task_id in pipe.tasks:
                task = pipe.tasks[task_id]
                if task.media_path.is_file():
                    return task.media_path
        return None

    def get_dashboard_data(self, pipeline_queues: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Gathers macroscopic metrics and active states for the live WebUI dashboard.

        Args:
            pipeline_queues: Mapping of pipeline names to their outstanding queued videos.

        Returns:
            dict[str, Any]: Complete dashboard state dictionary.
        """
        now = time()
        pipeline_data = []

        for name, pipe in self.pipelines.items():
            active_info = None
            stages_info = None

            if pipe.active_video:
                total_chunks = len(pipe.active_video.segment_paths)
                reenc_count = len(pipe.active_video.reencoded_segment_paths)
                pending_reenc = max(0, total_chunks - reenc_count)

                active_info = {
                    "video_id": pipe.active_video.video_id,
                    "title": pipe.active_video.title,
                    "uploader": pipe.active_video.uploader,
                    "channel_name": pipe.active_video.channel_name,
                    "duration_ms": pipe.active_video.duration_ms,
                }

                summary_task = next((t for t in pipe.tasks.values() if t.task_type == TaskType.SUMMARY), None)
                sub_tasks = [t for t in pipe.tasks.values() if t.task_type == TaskType.SUBTITLES]
                lyr_tasks = [t for t in pipe.tasks.values() if t.task_type == TaskType.LYRICS]

                sub_completed = sum(1 for t in sub_tasks if t.status == TaskStatus.COMPLETED)
                sub_claimed = sum(1 for t in sub_tasks if t.status == TaskStatus.CLAIMED)
                sub_pct = int((sub_completed / len(sub_tasks) * 100)) if sub_tasks else 0

                lyr_completed = sum(1 for t in lyr_tasks if t.status == TaskStatus.COMPLETED)
                lyr_claimed = sum(1 for t in lyr_tasks if t.status == TaskStatus.CLAIMED)

                active_lyr_tasks = [
                    {
                        "task_id": t.task_id,
                        "worker_id": t.worker_id,
                        "elapsed_s": int(now - t.claimed_at) if t.claimed_at else None,
                    }
                    for t in lyr_tasks
                    if t.status == TaskStatus.CLAIMED
                ]

                lyr_status = "skipped"
                if lyr_tasks:
                    if lyr_completed == len(lyr_tasks):
                        lyr_status = "completed"
                    elif lyr_claimed > 0:
                        lyr_status = "in_progress"
                    else:
                        lyr_status = "pending"

                stages_info = {
                    "summary": {
                        "status": "completed"
                        if pipe.summary_response
                        else (summary_task.status.value if summary_task else "pending"),
                        "worker_id": summary_task.worker_id if summary_task else None,
                        "elapsed_s": int(now - summary_task.claimed_at)
                        if summary_task and summary_task.claimed_at
                        else None,
                    },
                    "reencoding": {
                        "total_chunks": total_chunks,
                        "reencoded": reenc_count,
                        "pending": pending_reenc,
                    },
                    "subtitles": {
                        "total": len(sub_tasks),
                        "completed": sub_completed,
                        "claimed": sub_claimed,
                        "percentage": sub_pct,
                    },
                    "lyrics": {
                        "has_singing": len(lyr_tasks) > 0,
                        "status": lyr_status,
                        "total": len(lyr_tasks),
                        "completed": lyr_completed,
                        "claimed": lyr_claimed,
                        "active_leases": active_lyr_tasks,
                    },
                }

            tasks_list = []
            for t in pipe.tasks.values():
                tasks_list.append(
                    {
                        "task_id": t.task_id,
                        "type": t.task_type.value,
                        "status": t.status.value,
                        "worker_id": t.worker_id,
                        "attempts": t.attempts,
                        "elapsed_s": int(now - t.claimed_at) if t.claimed_at else None,
                    }
                )

            pipeline_data.append(
                {
                    "name": name,
                    "models": {
                        "summary": pipe.config.summary_model,
                        "subtitles": pipe.config.subtitles_model,
                        "lyrics": pipe.config.lyrics_model,
                    },
                    "idle": pipe.is_idle(),
                    "active_video": active_info,
                    "stages": stages_info,
                    "tasks": tasks_list,
                    "next_in_queue": pipeline_queues.get(name, []),
                }
            )

        return {
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipelines": pipeline_data,
            "recent_activity": [event.model_dump() for event in self.activity_log],
        }
