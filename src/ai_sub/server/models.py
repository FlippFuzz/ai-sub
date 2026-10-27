"""Internal data models for server tasks, active videos, worker leases, and activity events."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_sub.jobs.base import Prompt
from ai_sub.protocol import TaskType


class TaskStatus(str, Enum):
    """Execution state of a task in the server's in-memory queue."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


def generate_w3c_traceparent() -> str:
    """Generates a standard W3C traceparent header for distributed Logfire tracing.

    Returns:
        str: W3C traceparent string formatted as '00-{trace_id}-{span_id}-01'.
    """
    trace_id = os.urandom(16).hex()
    span_id = os.urandom(8).hex()
    return f"00-{trace_id}-{span_id}-01"


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
    prompt: Prompt = Field(description="System and user prompt pair.")
    response_schema_name: str = Field(description="Expected response schema class name.")

    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task state.")
    worker_id: str | None = Field(default=None, description="Identifier of the worker holding the lease.")
    claimed_at: float | None = Field(default=None, description="Timestamp when task was claimed.")
    last_heartbeat: float | None = Field(default=None, description="Timestamp of the most recent worker heartbeat.")
    attempts: int = Field(default=0, description="Total attempts recorded for this task.")

    search_tools: list[str] = Field(default_factory=list, description="Permitted search tool names.")
    agentic_video: bool = Field(default=False, description="Whether agentic video inspection is enabled.")
    traceparent: str = Field(default_factory=generate_w3c_traceparent, description="Distributed traceparent header.")

    detected_performances: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected singing performances passed to lyrics refinement.",
    )
    draft_subtitles: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Draft subtitles passed to lyrics refinement.",
    )
    result: dict[str, Any] | None = Field(default=None, description="Validated output received from worker.")


class ActiveVideo(BaseModel):
    """Metadata and filesystem paths for the video currently active in a pipeline."""

    video_id: str = Field(description="Video identifier.")
    uploader: str = Field(description="Channel or uploader handle/identifier.")
    channel_name: str | None = Field(default=None, description="Display name of the channel.")
    title: str = Field(description="Display title of the video.")
    filename_stem: str = Field(description="Filesystem-safe base name for output files.")
    source_type: Literal["local", "youtube"] = Field(default="local", description="Ingestion source type.")
    user_context: str | None = Field(default=None, description="Generated context string passed to LLM prompts.")
    job_dir: Path = Field(description="Subdirectory containing all files for this video job.")
    raw_path: Path = Field(description="Filesystem path to the original raw video.")
    duration_ms: int = Field(description="Total video duration in milliseconds.")
    summary_proxy_path: Path | None = Field(default=None, description="Path to the whole-video summary proxy.")
    segment_paths: list[Path] = Field(default_factory=list, description="Ordered paths to 5-minute chunk files.")
    reencoded_segment_paths: dict[int, Path] = Field(
        default_factory=dict,
        description="Mapping of segment indices to their compressed low-bitrate media files.",
    )
    segment_durations_ms: list[int] = Field(default_factory=list, description="Durations for each 5-minute chunk.")


class ActivityEvent(BaseModel):
    """Log record of worker task events for dashboard observability."""

    timestamp: float = Field(description="Unix timestamp of the event.")
    time_str: str = Field(description="Human-readable event time string (HH:MM:SS).")
    event_type: Literal["claim", "complete", "fail", "timeout"] = Field(description="Event classification.")
    task_id: str = Field(description="Task identifier.")
    pipeline: str = Field(description="Pipeline name.")
    worker_id: str = Field(description="Worker identifier.")
    model: str = Field(description="Target model.")
    duration_s: float | None = Field(default=None, description="Execution duration in seconds if complete.")
    detail: str | None = Field(default=None, description="Optional extra detail or error string.")
