"""Wire schemas and data contracts for client-server task communication."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_sub.config import RetrySettings
from ai_sub.jobs.subtitles import DetectedMusicalPerformance, Subtitles
from ai_sub.jobs.summary import SummaryAiResponse


class TaskType(str, Enum):
    """Types of distributed tasks executed by workers."""

    SUMMARY = "summary"
    SUBTITLES = "subtitles"
    LYRICS = "lyrics"


class TaskClaimRequest(BaseModel):
    """Payload sent by a worker to claim an available task."""

    worker_id: str = Field(description="Unique identifier for the worker.")
    client_version: str = Field(description="Installed ai-sub package version of the worker node.")
    models: list[str] = Field(description="List of unexhausted model identifiers supported by the worker.")
    search_tools: list[str] = Field(
        default_factory=list,
        description="List of search tools available on the worker (e.g. ['tinyfish', 'lrclib']).",
    )


class TaskClaimResponse(BaseModel):
    """Task details and structured domain parameters returned to a worker upon successful claim."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier (e.g. 'part_001').")
    task_type: TaskType = Field(description="Type of task to execute.")
    pipeline_name: str = Field(description="Name of the combination pipeline (e.g. 'combo_38f').")
    video_id: str = Field(description="Video identifier string.")
    model: str = Field(description="Gemini model identifier to employ.")
    media_url: str = Field(description="Relative media download endpoint on the server.")
    video_duration_ms: int = Field(description="Video segment duration in milliseconds.")
    segment_offset_ms: int = Field(
        default=0,
        description="Start offset of this segment in milliseconds from the beginning of the full video.",
    )
    user_context: str | None = Field(
        default=None,
        description="Resolved user background context or stream lore.",
    )
    global_summary: SummaryAiResponse | None = Field(
        default=None,
        description="Structured global video summary from phase 1.",
    )
    draft_subtitles: list[Subtitles] = Field(
        default_factory=list,
        description="Draft subtitles to refine for musical lyrics tasks.",
    )
    detected_performances: list[DetectedMusicalPerformance] = Field(
        default_factory=list,
        description="Detected singing performances passed to lyrics refinement.",
    )
    search_tools: list[str] = Field(
        default_factory=list,
        description="Search tools permitted for this task.",
    )
    agentic_video: bool = Field(
        default=False,
        description="Whether to enable agentic video inspection.",
    )
    service_tier: str | None = Field(
        default=None,
        description="Optional Google Gemini service tier.",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Task execution retry and backoff policy configured by the server.",
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
    reason: str = Field(
        description="Failure code (e.g. 'QUOTA_EXHAUSTED', 'TRANSIENT_API_ERROR', or 'EXECUTION_ERROR')."
    )
    model: str | None = Field(default=None, description="Model identifier that failed, if applicable.")
    error_message: str | None = Field(default=None, description="Optional error description or traceback.")
    is_transient: bool = Field(
        default=False,
        description=(
            "Whether the failure is transient (e.g. upstream 5xx, timeout) and should not penalize attempt counts."
        ),
    )
