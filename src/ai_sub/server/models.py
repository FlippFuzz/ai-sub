"""Internal data models for server task leases and worker activity telemetry."""

from __future__ import annotations

import uuid
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_sub.config import RetrySettings
from ai_sub.pipeline import Pipeline
from ai_sub.protocol import TaskClaimResponse, TaskType


class TaskStatus(str, Enum):
    """Execution state of a task in the server's in-memory queue."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class ServerTask(BaseModel):
    """Tracks worker lease state and routing metadata for an active pipeline task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier across all pipelines.")
    task_type: TaskType = Field(description="Type of task (summary, subtitles, or lyrics).")
    pipeline_name: str = Field(description="Identifier of the owning pipeline.")
    video_id: str = Field(description="Video identifier.")
    segment_index: int | None = Field(default=None, description="Index of the video segment, if applicable.")
    model: str = Field(description="Target Gemini model identifier.")
    media_path: Path = Field(description="Local filesystem path to the compressed media file.")

    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task state.")
    worker_id: str | None = Field(default=None, description="Identifier of the worker holding the lease.")
    claimed_at: float | None = Field(default=None, description="Timestamp when task was claimed.")
    last_heartbeat: float | None = Field(default=None, description="Timestamp of the most recent worker heartbeat.")
    claim_generation: int = Field(
        default=0,
        description="Monotonic claim generation counter incremented each time this task is leased.",
    )

    def to_claim_response(
        self,
        pipeline: Pipeline,
        search_tools: list[str],
        media_url: str,
        retry: RetrySettings,
        validation_buffer: timedelta,
    ) -> TaskClaimResponse:
        """Serializes this task and its authoritative pipeline context into a claim response.

        Args:
            pipeline: Active Pipeline owning this task.
            search_tools: Search tools permitted for this task type.
            media_url: Relative URL endpoint where the worker downloads media.
            retry: Server retry policy.
            validation_buffer: Subtitle timestamp overflow tolerance buffer.

        Returns:
            TaskClaimResponse: Formatted response model.
        """
        common: dict[str, Any] = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "pipeline_name": self.pipeline_name,
            "video_id": self.video_id,
            "model": self.model,
            "media_url": media_url,
            "user_context": pipeline.user_context,
            "search_tools": search_tools,
            "validation_buffer": validation_buffer,
            "retry": retry.resolve_for_model(self.model),
        }

        if self.task_type == TaskType.SUMMARY:
            return TaskClaimResponse(
                video_duration_ms=pipeline.duration_ms,
                agentic_video=pipeline.summary_config.agentic,
                service_tier=pipeline.summary_config.tier,
                target_fps=pipeline.summary.target_fps,
                **common,
            )

        assert self.segment_index is not None
        seg = pipeline.segments[self.segment_index]
        common["video_duration_ms"] = seg.duration_ms
        common["segment_offset_ms"] = seg.offset_ms
        common["global_summary"] = pipeline.summary.response
        common["target_fps"] = seg.target_fps

        if self.task_type == TaskType.SUBTITLES:
            return TaskClaimResponse(
                agentic_video=pipeline.subtitles_config.agentic,
                service_tier=pipeline.subtitles_config.tier,
                **common,
            )

        sub_resp = seg.subtitles_response
        return TaskClaimResponse(
            draft_subtitles=sub_resp.subtitles if sub_resp else [],
            detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
            agentic_video=pipeline.lyrics_config.agentic,
            service_tier=pipeline.lyrics_config.tier,
            **common,
        )


class ActivityEvent(BaseModel):
    """Log record of worker task events for dashboard observability."""

    event_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique identifier for this activity event.",
    )
    timestamp: float = Field(description="Unix timestamp of the event.")
    event_type: Literal["complete", "retry", "fail", "timeout"] = Field(description="Event classification.")
    task_id: str = Field(description="Task identifier.")
    worker_id: str = Field(description="Worker identifier.")
    video_id: str | None = Field(default=None, description="Video identifier associated with this task.")
    duration_s: float | None = Field(default=None, description="Execution duration in seconds if available.")
    detail: str | None = Field(default=None, description="Optional extra detail or error string.")
    logfire_query: str | None = Field(
        default=None,
        description="Dedicated Logfire search query string for tracing this task attempt.",
    )
