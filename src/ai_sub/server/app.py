"""FastAPI application providing worker task coordination endpoints and live WebUI dashboard."""

from __future__ import annotations

import asyncio
import mimetypes
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import logfire
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse

from ai_sub.protocol import (
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCompleteRequest,
    TaskFailRequest,
    TaskHeartbeatRequest,
    TaskHeartbeatResponse,
)
from ai_sub.server.config import ServerSettings
from ai_sub.server.coordinator import PipelineCoordinator
from ai_sub.server.dashboard import DASHBOARD_HTML
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.ingestion import IngestionManager


def create_app(settings: ServerSettings) -> FastAPI:
    """Factory creating and configuring the FastAPI server application.

    Args:
        settings: Server settings model.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    git_sync = GitSyncManager(settings.git, settings.server.work_dir)
    coordinator = PipelineCoordinator(settings, git_sync=git_sync)
    ingestion = IngestionManager(settings.sources, coordinator, git_sync, settings.server.work_dir)
    background_tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Manages server startup and shutdown lifecycles.

        Args:
            app: The FastAPI application instance.

        Yields:
            None: Context during application execution.
        """
        await git_sync.ensure_repo()
        await coordinator.start()
        ingestion_task = asyncio.create_task(ingestion.start())
        logfire.info(f"ai-sub-server started on {settings.server.host}:{settings.server.port}")
        yield
        if not ingestion_task.done():
            ingestion_task.cancel()
            try:
                await ingestion_task
            except asyncio.CancelledError:
                pass
        await ingestion.stop()
        await coordinator.stop()
        logfire.info("ai-sub-server shut down gracefully.")

    app = FastAPI(
        title="ai-sub-server",
        description="Central task coordinator, media distributor, and live observability dashboard for AI Sub",
        version="4.0.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Serves the live WebUI dashboard.

        Returns:
            HTMLResponse: Live WebUI HTML page.
        """
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/api/v1/dashboard/data")
    async def get_dashboard_data() -> dict[str, Any]:
        """Returns live dashboard telemetry, active pipelines, queue, and recent event logs.

        Returns:
            dict[str, Any]: Complete dashboard state dictionary.
        """
        pipeline_queues = ingestion.get_all_pipeline_queues()
        return coordinator.get_dashboard_data(pipeline_queues)

    @app.post(
        "/api/v1/tasks/claim",
        response_model=TaskClaimResponse | None,
        responses={204: {"description": "No pending tasks available matching worker capabilities"}},
    )
    async def claim_task(req: TaskClaimRequest) -> TaskClaimResponse | Response:
        """Claims an available pending task matching the worker's models and search capabilities.

        Args:
            req: Worker task claim request containing worker ID, supported models, and search tools.

        Returns:
            TaskClaimResponse | Response: Assigned task details or 204 No Content response if none available.
        """
        task = await coordinator.claim_task(
            worker_id=req.worker_id,
            models=req.models,
            search_tools=req.search_tools,
        )
        if not task:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return task

    @app.post("/api/v1/tasks/{task_id}/heartbeat", response_model=TaskHeartbeatResponse)
    async def heartbeat(task_id: str, req: TaskHeartbeatRequest) -> TaskHeartbeatResponse:
        """Renews an active task lease timestamp.

        Args:
            task_id: Unique task identifier.
            req: Heartbeat request containing worker ID.

        Returns:
            TaskHeartbeatResponse: Acknowledgment status confirming whether lease was renewed.
        """
        acknowledged = await coordinator.record_heartbeat(task_id, req.worker_id)
        return TaskHeartbeatResponse(acknowledged=acknowledged)

    @app.get("/api/v1/tasks/{task_id}/media")
    @app.get("/api/v1/tasks/{task_id}/media/{filename}")
    async def get_task_media(task_id: str, filename: str | None = None) -> FileResponse:
        """Streams the video media chunk associated with an active task.

        Args:
            task_id: Unique task identifier.
            filename: Optional filename in URL path for client-side extension preservation. Defaults to None.

        Returns:
            FileResponse: Media file stream response.

        Raises:
            HTTPException: If the media chunk file is not found on disk.
        """
        media_path = coordinator.get_task_media_path(task_id)
        if not media_path or not media_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media chunk file not found.")

        content_type, _ = mimetypes.guess_type(str(media_path))

        return FileResponse(
            path=media_path,
            media_type=content_type or "video/mp4",
            filename=media_path.name,
        )

    @app.post("/api/v1/tasks/{task_id}/complete")
    async def complete_task(task_id: str, req: TaskCompleteRequest) -> dict[str, str]:
        """Submits completed and validated structured AI results for a task.

        Args:
            task_id: Unique task identifier.
            req: Completion request containing task result data.

        Returns:
            dict[str, str]: Operation status confirmation dictionary.

        Raises:
            HTTPException: If the task was not claimed by this worker or has already completed.
        """
        success = await coordinator.complete_task(task_id, req.worker_id, req.result)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Task was not assigned to this worker or has already completed.",
            )

        # Run finalization and feeding in background so worker response returns immediately
        async def _finalize_and_feed() -> None:
            try:
                finalized = await coordinator.check_and_finalize_completed_pipelines()
                if finalized:
                    await ingestion.feed_idle_pipelines()
            except Exception as e:
                logfire.exception(f"Background finalization and pipeline feed error: {e}")

        bg_task = asyncio.create_task(_finalize_and_feed())
        background_tasks.add(bg_task)
        bg_task.add_done_callback(background_tasks.discard)

        return {"status": "ok"}

    @app.post("/api/v1/tasks/{task_id}/fail")
    async def fail_task(task_id: str, req: TaskFailRequest) -> dict[str, str]:
        """Reports a task execution failure or quota exhaustion to release the lease.

        Args:
            task_id: Unique task identifier.
            req: Failure report request containing reason and optional error details.

        Returns:
            dict[str, str]: Operation status confirmation dictionary.

        Raises:
            HTTPException: If the task could not be found or was not held by this worker.
        """
        success = await coordinator.fail_task(
            task_id=task_id,
            worker_id=req.worker_id,
            reason=req.reason,
            error_message=req.error_message,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task could not be found or was not held by this worker.",
            )
        return {"status": "ok"}

    @app.get("/api/v1/status")
    async def get_server_status() -> dict[str, Any]:
        """Returns macroscopic pipeline and in-memory queue status.

        Returns:
            dict[str, Any]: Mapping of server status and pipeline queue metrics.
        """
        pipeline_status = {}
        for name, pipe in coordinator.pipelines.items():
            pipeline_status[name] = {
                "status": pipe.status.value,
                "idle": pipe.is_idle(),
                "active_video": pipe.active_video.video_id if pipe.active_video else None,
                "total_tasks": len(pipe.tasks),
                "pending_tasks": sum(1 for t in pipe.tasks.values() if t.status.value == "pending"),
                "claimed_tasks": sum(1 for t in pipe.tasks.values() if t.status.value == "claimed"),
                "completed_tasks": sum(1 for t in pipe.tasks.values() if t.status.value == "completed"),
            }

        return {
            "status": "online",
            "pipelines": pipeline_status,
        }

    return app
