"""HTTP client for communicating with ai-sub-server."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

import logfire
from httpx import AsyncClient, HTTPError, Response

from ai_sub.protocol import (
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCompleteRequest,
    TaskFailRequest,
    TaskHeartbeatRequest,
    TaskHeartbeatResponse,
    TaskType,
)


class WorkerClient:
    """Async client managing HTTP communication with the central server."""

    base_url: str
    auth: tuple[str, str] | None
    client: AsyncClient

    def __init__(
        self,
        base_url: str,
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initializes the worker HTTP client.

        Args:
            base_url: Base URL of the ai-sub-server.
            auth: Optional (username, password) tuple for Basic Auth. Defaults to None.
            timeout: HTTP request timeout in seconds. Defaults to 60.0.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.client = AsyncClient(
            base_url=self.base_url,
            auth=auth,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        await self.client.aclose()

    async def claim_task(
        self,
        worker_id: str,
        models: list[str],
        search_tools: list[str],
    ) -> TaskClaimResponse | None:
        """Requests an available pending task matching the worker's specified models.

        Args:
            worker_id: Unique identifier of this worker.
            models: List of model identifiers offered for this claim request.
            search_tools: List of search tools configured on the worker.

        Returns:
            TaskClaimResponse | None: Claimed task details if available, or None if queue is empty.
        """
        req = TaskClaimRequest(
            worker_id=worker_id,
            models=models,
            search_tools=search_tools,
        )
        resp: Response = await self.client.post("/api/v1/tasks/claim", json=req.model_dump(mode="json"))

        if resp.status_code == 204:
            return None

        resp.raise_for_status()
        return TaskClaimResponse.model_validate(resp.json())

    async def send_heartbeat(
        self,
        task_id: str,
        worker_id: str,
    ) -> bool | None:
        """Sends a heartbeat ping to renew an active task lease.

        Args:
            task_id: Unique task identifier.
            worker_id: Unique identifier of this worker.

        Returns:
            bool | None: True if acknowledged; False if lease expired; None if a transient transport error occurred.
        """
        encoded_task_id = urllib.parse.quote(task_id, safe=":")
        req = TaskHeartbeatRequest(worker_id=worker_id)
        try:
            resp: Response = await self.client.post(
                f"/api/v1/tasks/{encoded_task_id}/heartbeat",
                json=req.model_dump(mode="json"),
            )
            if resp.status_code == 200:
                data = TaskHeartbeatResponse.model_validate(resp.json())
                return data.acknowledged
            if resp.status_code >= 500:
                logfire.warning(f"Heartbeat received server error {resp.status_code} for task '{task_id}'")
                return None
            logfire.warning(f"Heartbeat returned unexpected status {resp.status_code} for task '{task_id}'")
            return False
        except Exception as e:
            logfire.warning(f"Heartbeat network error for task '{task_id}': {e}")
            return None

    async def download_media(self, media_url: str, destination: Path) -> Path:
        """Downloads a media chunk file from the server.

        Args:
            media_url: Relative media endpoint URL on the server or absolute URL.
            destination: Local path where the file should be saved.

        Returns:
            Path: Path to the validated downloaded file.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)

        with logfire.span("Downloading media chunk: {url}", url=media_url):
            async with self.client.stream("GET", media_url) as stream_resp:
                stream_resp.raise_for_status()
                with open(destination, "wb") as f:
                    async for chunk in stream_resp.aiter_bytes():
                        f.write(chunk)

        return destination

    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        task_type: TaskType,
        result: dict[str, Any],
    ) -> bool:
        """Submits completed and validated structured AI results to the server.

        Args:
            task_id: Unique task identifier.
            worker_id: Unique identifier of this worker.
            task_type: Type of task completed.
            result: Validated AI result data dictionary.

        Returns:
            bool: True if completion was accepted; False otherwise.
        """
        encoded_task_id = urllib.parse.quote(task_id, safe=":")
        req = TaskCompleteRequest(
            worker_id=worker_id,
            task_id=task_id,
            task_type=task_type,
            result=result,
        )
        try:
            resp: Response = await self.client.post(
                f"/api/v1/tasks/{encoded_task_id}/complete",
                json=req.model_dump(mode="json"),
            )
            return resp.status_code == 200
        except HTTPError as e:
            logfire.error(f"Failed to submit completed task '{task_id}': {e}")
            return False

    async def fail_task(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
        model: str | None = None,
        error_message: str | None = None,
        is_transient: bool = False,
    ) -> bool:
        """Notifies the server of a task failure or daily quota limit exhaustion.

        Args:
            task_id: Unique task identifier.
            worker_id: Unique identifier of this worker.
            reason: Failure code or exception description.
            model: Gemini model identifier that failed, if applicable. Defaults to None.
            error_message: Optional error description or traceback. Defaults to None.
            is_transient: Whether the failure is transient and should not penalize server attempts. Defaults to False.

        Returns:
            bool: True if failure was acknowledged; False otherwise.
        """
        encoded_task_id = urllib.parse.quote(task_id, safe=":")
        req = TaskFailRequest(
            worker_id=worker_id,
            task_id=task_id,
            reason=reason,
            model=model,
            error_message=error_message,
            is_transient=is_transient,
        )
        try:
            resp: Response = await self.client.post(
                f"/api/v1/tasks/{encoded_task_id}/fail",
                json=req.model_dump(mode="json"),
            )
            return resp.status_code == 200
        except Exception as e:
            logfire.warning(f"Failed to report task failure for '{task_id}': {e}")
            return False
