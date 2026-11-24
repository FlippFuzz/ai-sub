"""Wire schemas and data contracts for client-server task communication."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_sub.config import RetrySettings, ServiceTier
from ai_sub.jobs.job_base import TaskType
from ai_sub.jobs.lyrics import LyricsJob
from ai_sub.jobs.subtitles import DetectedMusicalPerformance, Subtitles, SubtitlesJob
from ai_sub.jobs.summary import SummaryAiResponse, SummaryJob


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
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier.",
    )
    validation_buffer: timedelta = Field(
        default=timedelta(seconds=2),
        description="Allowed subtitle timestamp overflow tolerance buffer configured by the server.",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Task execution retry and backoff policy configured by the server.",
    )

    def to_job(self) -> SummaryJob | SubtitlesJob | LyricsJob:
        """Constructs the typed AI job instance corresponding to this claimed task.

        Returns:
            SummaryJob | SubtitlesJob | LyricsJob: Configured job ready for execution.

        Raises:
            ValueError: If task_type is unknown or a lyrics task lacks draft_subtitles.
        """
        common: dict[str, Any] = {
            "video_duration_ms": self.video_duration_ms,
            "user_context": self.user_context,
            "model_name": self.model,
            "service_tier": self.service_tier,
            "agentic_video": self.agentic_video,
        }
        if self.task_type == TaskType.SUMMARY:
            return SummaryJob(name="summary", **common)
        if self.task_type == TaskType.SUBTITLES:
            return SubtitlesJob(
                name=self.task_id,
                segment_offset_ms=self.segment_offset_ms,
                global_summary=self.global_summary,
                **common,
            )
        if self.task_type == TaskType.LYRICS:
            if not self.draft_subtitles:
                raise ValueError(
                    f"Lyrics task '{self.task_id}' rejected: missing draft_subtitles required for refinement."
                )
            return LyricsJob(
                name=self.task_id,
                segment_offset_ms=self.segment_offset_ms,
                draft_subtitles=self.draft_subtitles,
                detected_performances=self.detected_performances,
                global_summary=self.global_summary,
                **common,
            )
        raise ValueError(f"Unknown task type: '{self.task_type}'")


class TaskHeartbeatRequest(BaseModel):
    """Heartbeat ping payload sent by a worker to maintain its task lease."""

    worker_id: str = Field(description="Unique identifier of the worker holding the lease.")


class TaskHeartbeatResponse(BaseModel):
    """Heartbeat response confirming lease renewal."""

    acknowledged: bool = Field(description="True if lease was renewed; False if task timed out or expired.")


class TaskCompleteRequest(BaseModel):
    """Payload sent by a worker upon successful execution of a task."""

    worker_id: str = Field(description="Unique identifier of the worker.")
    result: dict[str, Any] = Field(description="Parsed and validated AI output dictionary.")


class TaskFailRequest(BaseModel):
    """Payload sent by a worker when task execution fails."""

    worker_id: str = Field(description="Unique identifier of the worker.")
    reason: str = Field(description="Failure code (e.g. 'QUOTA_EXHAUSTED', 'RETRYABLE_ERROR', or 'EXECUTION_ERROR').")
    error_message: str | None = Field(default=None, description="Optional error description or traceback.")
    is_retryable: bool = Field(
        default=False,
        description=(
            "Whether the failure is retryable (e.g. upstream 5xx, timeout, quota) and should not count toward max_runs."
        ),
    )
