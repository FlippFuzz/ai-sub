"""Wire schemas and data contracts for client-server task communication."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of distributed tasks executed by workers."""

    SUMMARY = "summary"
    SUBTITLES = "subtitles"
    LYRICS = "lyrics"


class TaskClaimRequest(BaseModel):
    """Payload sent by a worker to claim an available task."""

    worker_id: str = Field(description="Unique identifier for the worker.")
    models: list[str] = Field(description="List of unexhausted model identifiers supported by the worker.")
    search_tools: list[str] = Field(
        default_factory=list,
        description="List of search tools available on the worker (e.g. ['tinyfish', 'lrclib']).",
    )


class TaskClaimResponse(BaseModel):
    """Task details returned to a worker upon successful claim."""

    task_id: str = Field(description="Unique task identifier (e.g. 'summary' or 'part_001').")
    task_type: TaskType = Field(description="Type of task to execute.")
    pipeline_name: str = Field(description="Name of the combination pipeline (e.g. 'combo_38f').")
    video_id: str = Field(description="Video identifier string.")
    model: str = Field(description="Gemini model identifier to employ.")
    media_url: str = Field(description="Relative media download endpoint on the server.")
    video_duration_ms: int = Field(description="Video segment duration in milliseconds.")
    prompt_system: str = Field(description="System instruction prompt for the Gemini model.")
    prompt_user: str = Field(description="User prompt instructions and input context.")
    response_schema_name: str = Field(
        description="Target response schema ('SummaryAiResponse', 'SubtitlesAiResponse', or 'LyricsAiResponse')."
    )
    traceparent: str | None = Field(default=None, description="W3C traceparent header for distributed Logfire tracing.")
    search_tools: list[str] = Field(default_factory=list, description="Search tools permitted for this task.")
    agentic_video: bool = Field(default=False, description="Whether to enable agentic video inspection.")
    service_tier: str | None = Field(default=None, description="Optional Google Gemini service tier.")
    detected_performances: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized list of detected singing performances for lyrics refinement.",
    )


class TaskHeartbeatRequest(BaseModel):
    """Heartbeat ping payload sent by a worker to maintain its task lease."""

    worker_id: str = Field(description="Unique identifier of the worker holding the lease.")


class TaskHeartbeatResponse(BaseModel):
    """Heartbeat response confirming lease renewal."""

    acknowledged: bool = Field(description="True if lease was renewed; False if task timed out or expired.")


class TaskCompleteRequest(BaseModel):
    """Payload sent by a worker upon successful execution of a task."""

    worker_id: str = Field(description="Unique identifier of the worker.")
    task_id: str = Field(description="Unique task identifier.")
    task_type: TaskType = Field(description="Task type completed.")
    result: dict[str, Any] = Field(description="Parsed and validated AI output dictionary.")


class TaskFailRequest(BaseModel):
    """Payload sent by a worker when task execution fails."""

    worker_id: str = Field(description="Unique identifier of the worker.")
    task_id: str = Field(description="Unique task identifier.")
    reason: str = Field(description="Failure code (e.g. 'QUOTA_EXHAUSTED' or 'EXECUTION_ERROR').")
    model: str | None = Field(default=None, description="Model identifier that failed, if applicable.")
    error_message: str | None = Field(default=None, description="Optional error description or traceback.")
