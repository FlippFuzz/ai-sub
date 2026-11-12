"""Internal data models for server tasks, active videos, worker leases, and activity events."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_sub.config import RetrySettings
from ai_sub.jobs.subtitles import DetectedMusicalPerformance, Subtitles
from ai_sub.jobs.summary import SummaryAiResponse
from ai_sub.pipeline import Pipeline
from ai_sub.protocol import TaskClaimResponse, TaskType


class TaskStatus(str, Enum):
    """Execution state of a task in the server's in-memory queue."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class ServerTask(BaseModel):
    """Represents an atomic unit of AI generation work managed in server memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier across all pipelines.")
    task_type: TaskType = Field(description="Type of task (summary, subtitles, or lyrics).")
    pipeline_name: str = Field(description="Identifier of the owning pipeline.")
    video_id: str = Field(description="Video identifier.")
    segment_index: int | None = Field(default=None, description="Index of the 5-minute video segment, if applicable.")
    model: str = Field(description="Target Gemini model identifier.")
    media_path: Path = Field(description="Local filesystem path to the compressed media file.")
    video_duration_ms: int = Field(description="Duration of the segment in milliseconds.")
    segment_offset_ms: int = Field(default=0, description="Start offset from beginning of video in milliseconds.")
    user_context: str | None = Field(default=None, description="Resolved background context.")
    global_summary: SummaryAiResponse | None = Field(default=None, description="Summary context model.")

    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task state.")
    worker_id: str | None = Field(default=None, description="Identifier of the worker holding the lease.")
    claimed_at: float | None = Field(default=None, description="Timestamp when task was claimed.")
    last_heartbeat: float | None = Field(default=None, description="Timestamp of the most recent worker heartbeat.")
    attempts: int = Field(default=0, description="Total attempts recorded for this task.")

    search_tools: list[str] = Field(default_factory=list, description="Permitted search tool names.")
    agentic_video: bool = Field(default=False, description="Whether agentic video inspection is enabled.")
    service_tier: str | None = Field(default=None, description="Optional Google Gemini service tier.")

    detected_performances: list[DetectedMusicalPerformance] = Field(
        default_factory=list,
        description="Detected singing performances passed to lyrics refinement.",
    )
    draft_subtitles: list[Subtitles] = Field(
        default_factory=list,
        description="Draft subtitles passed to lyrics refinement.",
    )
    result: dict[str, Any] | None = Field(default=None, description="Validated output received from worker.")

    def to_claim_response(self, media_url: str, retry: RetrySettings) -> TaskClaimResponse:
        """Serializes this task into the typed claim response sent to workers.

        Args:
            media_url: Relative URL endpoint where the worker downloads media.
            retry: Server retry policy.

        Returns:
            TaskClaimResponse: Formatted response model.
        """
        return TaskClaimResponse(
            task_id=self.task_id,
            task_type=self.task_type,
            pipeline_name=self.pipeline_name,
            video_id=self.video_id,
            model=self.model,
            media_url=media_url,
            video_duration_ms=self.video_duration_ms,
            segment_offset_ms=self.segment_offset_ms,
            user_context=self.user_context,
            global_summary=self.global_summary,
            draft_subtitles=self.draft_subtitles,
            detected_performances=self.detected_performances,
            search_tools=self.search_tools,
            agentic_video=self.agentic_video,
            service_tier=self.service_tier,
            retry=retry,
        )


class ActiveVideo(BaseModel):
    """Metadata, filesystem paths, and session tracking for the video active in a pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: str = Field(description="Video identifier.")
    uploader: str = Field(description="Channel or uploader handle/identifier.")
    channel_name: str | None = Field(default=None, description="Display name of the channel.")
    title: str = Field(description="Display title of the video.")
    filename_stem: str = Field(description="Filesystem-safe base name for output files.")
    source_type: Literal["local", "youtube"] = Field(default="local", description="Ingestion source type.")
    timestamp: int = Field(default=0, description="Upload or file modification Unix timestamp.")
    user_context: str | None = Field(default=None, description="Generated context string passed to LLM prompts.")
    job_dir: Path = Field(description="Subdirectory containing all files for this video job.")
    raw_path: Path = Field(description="Filesystem path to the original raw video.")
    duration_ms: int = Field(description="Total video duration in milliseconds.")
    pipeline: Pipeline | None = Field(
        default=None,
        description="Active pipeline state machine managing this video.",
    )


class ActivityEvent(BaseModel):
    """Log record of worker task events for dashboard observability."""

    timestamp: float = Field(description="Unix timestamp of the event.")
    time_str: str = Field(description="Human-readable event time string (HH:MM:SS).")
    event_type: Literal["claim", "complete", "fail", "timeout"] = Field(description="Event classification.")
    task_id: str = Field(description="Task identifier.")
    pipeline: str = Field(description="Pipeline name.")
    worker_id: str = Field(description="Worker identifier.")
    model: str = Field(description="Target model.")
    video_id: str | None = Field(default=None, description="Video identifier associated with this task.")
    duration_s: float | None = Field(default=None, description="Execution duration in seconds if complete.")
    detail: str | None = Field(default=None, description="Optional extra detail or error string.")
    logfire_query: str | None = Field(
        default=None,
        description="Dedicated Logfire search query string for tracing this task attempt.",
    )
