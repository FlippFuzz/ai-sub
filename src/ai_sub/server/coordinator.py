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
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.models import ActivityEvent, ServerTask, TaskStatus
from ai_sub.server.pipeline import ComboPipeline, PipelineStatus


class PipelineCoordinator:
    """Orchestrates in-memory pipeline runners, worker task leases, and dashboard telemetry."""

    settings: ServerSettings
    git_sync: GitSyncManager | None
    pipelines: dict[str, ComboPipeline]
    activity_log: deque[ActivityEvent]
    _lock: asyncio.Lock
    _finalizing_pipes: set[str]
    _sweep_task: asyncio.Task[None] | None

    def __init__(self, settings: ServerSettings, git_sync: GitSyncManager | None = None) -> None:
        """Initializes the central pipeline coordinator.

        Args:
            settings: Server settings model.
            git_sync: Optional Git synchronization manager for pushing completed subtitles. Defaults to None.
        """
        self.settings = settings
        self.git_sync = git_sync
        self.pipelines = {}
        self.activity_log = deque(maxlen=50)
        self._lock = asyncio.Lock()
        self._finalizing_pipes = set()
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

            # Immediately flush and push any videos found 100% complete on disk during boot
            try:
                await self.check_and_finalize_completed_pipelines()
            except Exception as e:
                logfire.exception(f"Error checking and finalizing completed pipelines on startup: {e}")

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
        max_runs = self.settings.retry.max_runs

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
                                    worker_id = task.worker_id or "unknown"
                                    task.worker_id = None
                                    task.claimed_at = None
                                    task.last_heartbeat = None
                                    task.attempts += 1

                                    if task.attempts >= max_runs:
                                        task.status = TaskStatus.FAILED
                                        logfire.error(
                                            f"Task '{task.task_id}' lease expired ({elapsed:.1f}s ago). "
                                            f"Max runs ({task.attempts}/{max_runs}) reached. Marked FAILED."
                                        )
                                        self._log_activity(
                                            event_type="fail",
                                            task=task,
                                            worker_id=worker_id,
                                            detail=f"Lease timed out after {timeout}s: max runs ({max_runs}) exhausted",
                                        )
                                    else:
                                        task.status = TaskStatus.PENDING
                                        logfire.warning(
                                            f"Task '{task.task_id}' lease expired (worker: {worker_id}, "
                                            f"last heartbeat: {elapsed:.1f}s ago). Returning to PENDING."
                                        )
                                        self._log_activity(
                                            event_type="timeout",
                                            task=task,
                                            worker_id=worker_id,
                                            detail=f"Lease timed out after {timeout}s without heartbeat",
                                        )

                    sweep_span.set_attribute("active_leases", active_leases)
                    sweep_span.set_attribute("expired_leases", expired_count)

            # Check if any pipeline reached completion due to expired/failed tasks
            try:
                await self.check_and_finalize_completed_pipelines()
            except Exception as e:
                logfire.exception(f"Error checking and finalizing completed pipelines in lease sweeper: {e}")

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

                        # Check if chunk re-encoding is still pending for this task
                        if not pipe.is_task_ready_for_claim(task):
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
                            service_tier=task.service_tier,
                            detected_performances=task.detected_performances,
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

    def _fail_task_locked(
        self,
        task: ServerTask,
        worker_id: str,
        reason: str,
        error_message: str | None = None,
    ) -> bool:
        """Processes task failure and updates lease state while lock is held.

        Args:
            task: Affected server task.
            worker_id: Worker reporting the failure.
            reason: Error code (e.g. 'QUOTA_EXHAUSTED' or 'EXECUTION_ERROR').
            error_message: Optional error details. Defaults to None.

        Returns:
            bool: True indicating the task failure was processed.
        """
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

        if reason != "QUOTA_EXHAUSTED":
            task.attempts += 1
            max_runs = self.settings.retry.max_runs
            if task.attempts >= max_runs:
                task.status = TaskStatus.FAILED
                logfire.error(
                    f"Task '{task.task_id}' marked FAILED after exhausting "
                    f"{task.attempts}/{max_runs} runs. Reason: {reason}."
                )
                return True

        task.status = TaskStatus.PENDING
        logfire.warning(
            f"Task '{task.task_id}' failed by worker '{worker_id}' (reason: {reason}). "
            f"Details: {error_message or 'none'}. Returning to PENDING."
        )
        return True

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
                            try:
                                pipe.on_task_completed(task_id, result)
                            except Exception as val_err:
                                logfire.warning(
                                    f"Task '{task_id}' completion validation failed: {val_err}. Routing to fail."
                                )
                                comp_span.set_attribute("result", "validation_failed")
                                comp_span.set_attribute("error", str(val_err))
                                self._fail_task_locked(
                                    task=task,
                                    worker_id=worker_id,
                                    reason="EXECUTION_ERROR",
                                    error_message=f"Server validation error: {val_err}",
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
                            success = self._fail_task_locked(
                                task=task,
                                worker_id=worker_id,
                                reason=reason,
                                error_message=error_message,
                            )
                            fail_span.set_attribute("result", task.status.value)
                            fail_span.set_attribute("attempts", task.attempts)
                            return success

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

    async def finalize_pipeline_if_complete(self, pipe: ComboPipeline) -> bool:
        """Checks if a pipeline has finished all segments, stitches subtitles, pushes to Git, and clears it.

        Args:
            pipe: Pipeline instance to check and finalize.

        Returns:
            bool: True if pipeline was completed and finalized; False otherwise.
        """
        async with self._lock:
            if pipe.config.name in self._finalizing_pipes:
                return False
            if pipe.is_idle() or not pipe.active_video:
                return False
            if pipe.status != PipelineStatus.PUSH_FAILED and not pipe.is_video_complete():
                return False

            self._finalizing_pipes.add(pipe.config.name)
            pipe.status = PipelineStatus.FINALIZING

        try:
            vid = pipe.active_video
            assert vid is not None
            logfire.info(f"Pipeline '{pipe.config.name}' finalizing for '{vid.title}'...")

            srt_filename = f"{vid.filename_stem}.{pipe.full_shortcode}.srt"
            temp_srt = vid.job_dir / srt_filename

            # Run synchronous subtitle stitching off the event loop
            try:
                generated_srts = await asyncio.to_thread(pipe.stitch_final_subtitles, temp_srt)
            except Exception as e:
                logfire.error(f"Subtitle stitching failed for '{pipe.config.name}' ({vid.video_id}): {e}")
                async with self._lock:
                    pipe.status = PipelineStatus.PUSH_FAILED
                return False

            yaml_files = list(vid.job_dir.glob("*.yaml"))
            if self.git_sync:
                try:
                    await self.git_sync.finalize_and_push(
                        uploader=vid.uploader,
                        video_id=vid.video_id,
                        full_shortcode=pipe.full_shortcode,
                        srt_files=generated_srts,
                        yaml_files=yaml_files,
                    )
                except Exception as e:
                    logfire.error(
                        f"Git push failed for '{pipe.config.name}' ({vid.video_id}): {e}. Setting PUSH_FAILED."
                    )
                    async with self._lock:
                        pipe.status = PipelineStatus.PUSH_FAILED
                    return False

            async with self._lock:
                pipe.clear_active_video()
                logfire.info(f"Pipeline '{pipe.config.name}' cleared and ready for next video.")
                return True
        except Exception as e:
            logfire.exception(f"Unexpected error finalizing pipeline '{pipe.config.name}': {e}")
            async with self._lock:
                pipe.status = PipelineStatus.PUSH_FAILED
            return False
        finally:
            async with self._lock:
                self._finalizing_pipes.discard(pipe.config.name)

    async def check_and_finalize_completed_pipelines(self) -> list[ComboPipeline]:
        """Inspects all pipelines, finalizing any that have reached completion.

        Returns:
            list[ComboPipeline]: List of pipelines that completed and were reset.
        """
        finalized: list[ComboPipeline] = []
        for pipe in list(self.pipelines.values()):
            if await self.finalize_pipeline_if_complete(pipe):
                finalized.append(pipe)
        return finalized

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
                    "source_type": pipe.active_video.source_type,
                }

                summary_task = next((t for t in pipe.tasks.values() if t.task_type == TaskType.SUMMARY), None)
                sub_tasks = [t for t in pipe.tasks.values() if t.task_type == TaskType.SUBTITLES]
                lyr_tasks = [t for t in pipe.tasks.values() if t.task_type == TaskType.LYRICS]

                sum_done = (
                    1
                    if pipe.summary_response
                    else (1 if summary_task and summary_task.status == TaskStatus.COMPLETED else 0)
                )
                sum_prog = 1 if summary_task and summary_task.status == TaskStatus.CLAIMED else 0

                sub_done = sum(1 for t in sub_tasks if t.status == TaskStatus.COMPLETED)
                sub_prog = sum(1 for t in sub_tasks if t.status == TaskStatus.CLAIMED)

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

                # Accurate state progression: lyrics cannot be marked "skipped" while subtitles are still running
                if sub_done < total_chunks:
                    if lyr_tasks:
                        has_singing = True
                        lyr_status = (
                            "in_progress"
                            if lyr_prog > 0
                            else ("completed" if lyr_done == len(lyr_tasks) else "pending")
                        )
                    else:
                        has_singing = False
                        lyr_status = "pending"
                else:
                    if lyr_tasks:
                        has_singing = True
                        if lyr_done == len(lyr_tasks):
                            lyr_status = "completed"
                        elif lyr_prog > 0:
                            lyr_status = "in_progress"
                        else:
                            lyr_status = "pending"
                    else:
                        has_singing = False
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
                    "status": pipe.status.value,
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
