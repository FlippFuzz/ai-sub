"""Internal data models for server task leases and worker activity telemetry."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

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
        if self.task_type == TaskType.SUMMARY:
            return TaskClaimResponse(
                task_id=self.task_id,
                task_type=self.task_type,
                pipeline_name=self.pipeline_name,
                video_id=self.video_id,
                model=self.model,
                media_url=media_url,
                video_duration_ms=pipeline.duration_ms,
                segment_offset_ms=0,
                user_context=pipeline.user_context,
                search_tools=search_tools,
                agentic_video=pipeline.summary_config.agentic,
                service_tier=pipeline.summary_config.tier,
                validation_buffer=validation_buffer,
                retry=retry,
            )

        assert self.segment_index is not None
        seg = pipeline.segments[self.segment_index]

        if self.task_type == TaskType.SUBTITLES:
            return TaskClaimResponse(
                task_id=self.task_id,
                task_type=self.task_type,
                pipeline_name=self.pipeline_name,
                video_id=self.video_id,
                model=self.model,
                media_url=media_url,
                video_duration_ms=seg.duration_ms,
                segment_offset_ms=seg.offset_ms,
                user_context=pipeline.user_context,
                global_summary=pipeline.summary.response,
                search_tools=search_tools,
                agentic_video=pipeline.subtitles_config.agentic,
                service_tier=pipeline.subtitles_config.tier,
                validation_buffer=validation_buffer,
                retry=retry,
            )

        sub_resp = seg.subtitles_response
        return TaskClaimResponse(
            task_id=self.task_id,
            task_type=self.task_type,
            pipeline_name=self.pipeline_name,
            video_id=self.video_id,
            model=self.model,
            media_url=media_url,
            video_duration_ms=seg.duration_ms,
            segment_offset_ms=seg.offset_ms,
            user_context=pipeline.user_context,
            global_summary=pipeline.summary.response,
            draft_subtitles=sub_resp.subtitles if sub_resp else [],
            detected_performances=sub_resp.detected_musical_performances if sub_resp else [],
            search_tools=search_tools,
            agentic_video=pipeline.lyrics_config.agentic,
            service_tier=pipeline.lyrics_config.tier,
            validation_buffer=validation_buffer,
            retry=retry,
        )


class ActivityEvent(BaseModel):
    """Log record of worker task events for dashboard observability."""

    timestamp: float = Field(description="Unix timestamp of the event.")
    time_str: str = Field(description="Human-readable event time string (HH:MM:SS).")
    event_type: Literal["complete", "fail", "timeout"] = Field(description="Event classification.")
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
