"""Dashboard telemetry builders and fallback-tier formatting for ai-sub-server."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from time import time
from typing import Any

from ai_sub.config import StageReencodeSettings
from ai_sub.jobs.job_base import EncodingMode, TaskType
from ai_sub.pipeline import MediaStageState, Pipeline, SegmentState
from ai_sub.server.combo_runner import ComboRunner
from ai_sub.server.models import ActivityEvent, ServerTask, TaskStatus
from ai_sub.telemetry import get_app_version


def _escape_sql_literal(value: str) -> str:
    """Escapes single quotes for SQL string literals.

    Args:
        value: Raw string value.

    Returns:
        str: SQL-escaped string content without surrounding quotes.
    """
    return value.replace("'", "''")


def format_logfire_query(task: ServerTask) -> str:
    """Formats the Logfire search query for a server task using its unique task ID.

    Args:
        task: Target server task instance.

    Returns:
        str: Formatted Logfire search query string.
    """
    return f"attributes->>'task_id' = '{_escape_sql_literal(task.task_id)}'"


def _format_chunk_logfire_query(*tasks: ServerTask | None) -> str | None:
    """Formats a Logfire search query matching all existing tasks for a chunk.

    Args:
        *tasks: Candidate ServerTask instances associated with the chunk.

    Returns:
        str | None: Formatted Logfire search query string, or None if no tasks exist.
    """
    existing = [t for t in tasks if t is not None]
    if not existing:
        return None
    if len(existing) == 1:
        return format_logfire_query(existing[0])
    quoted = ", ".join(f"'{_escape_sql_literal(t.task_id)}'" for t in existing)
    return f"attributes->>'task_id' IN ({quoted})"


def build_fallback_tiers(
    state: MediaStageState,
    cfg: StageReencodeSettings,
    source_path: Path,
    exhausted: bool = False,
) -> list[dict[str, Any]]:
    """Builds the ordered fallback tier progression for dashboard display when in fallback.

    Runs synchronously using cached state and config math without spawning subprocesses.

    Args:
        state: Active summary or segment media state.
        cfg: Re-encoding settings for this stage.
        source_path: Uncompressed source media path for tier-0 threshold check.
        exhausted: Whether this stage has exhausted all retries and fallback tiers. Defaults to False.

    Returns:
        list[dict[str, Any]]: Ordered list of tier dicts, or empty list when not in fallback.
    """
    if state.fallback_tier <= 0:
        return []

    divisor = float(cfg.fallback.divisor)
    minfps = float(cfg.fallback.minfps)

    try:
        was_raw = cfg.threshold == 0 or source_path.stat().st_size < cfg.threshold
    except OSError:
        was_raw = cfg.threshold == 0

    base_fps: float | None = None
    if cfg.fps != "keyframes":
        base_fps = float(cfg.fps)
    elif state.base_fps is not None and state.base_fps > 0:
        base_fps = float(state.base_fps)
    elif state.fps is not None and state.fps > 0:
        base_fps = float(state.fps) * (divisor**state.fallback_tier)

    tiers: list[dict[str, Any]] = []

    def _status(tier_idx: int) -> str:
        if tier_idx < state.fallback_tier:
            return "failed"
        if tier_idx == state.fallback_tier:
            return "exhausted" if exhausted else "active"
        return "upcoming"

    tier0_encoding: EncodingMode = "raw" if was_raw else ("keyframes" if cfg.fps == "keyframes" else "reencoded")
    tiers.append(
        {
            "tier": 0,
            "encoding": tier0_encoding,
            "fps": None if was_raw else base_fps,
            "status": _status(0),
        }
    )

    if base_fps is not None and base_fps > 0:
        for tier_idx in range(1, 21):
            candidate = base_fps / (divisor**tier_idx)
            if candidate >= minfps:
                tiers.append(
                    {
                        "tier": tier_idx,
                        "encoding": "reencoded",
                        "fps": candidate,
                        "status": _status(tier_idx),
                    }
                )
            else:
                if not (exhausted and state.fallback_tier < tier_idx):
                    tiers.append(
                        {
                            "tier": tier_idx,
                            "encoding": "audio",
                            "fps": 0.0,
                            "status": _status(tier_idx),
                        }
                    )
                break
    else:
        for tier_idx in range(1, state.fallback_tier):
            tiers.append(
                {
                    "tier": tier_idx,
                    "encoding": "reencoded",
                    "fps": None,
                    "status": "failed",
                }
            )
        tiers.append(
            {
                "tier": state.fallback_tier,
                "encoding": state.encoding,
                "fps": state.fps,
                "status": _status(state.fallback_tier),
            }
        )

    return tiers


def _get_summary_status(pipe: Pipeline, is_claimed: bool, is_encoding: bool) -> str:
    """Computes the dashboard display status for the whole-video summary stage.

    Args:
        pipe: Active Pipeline state machine.
        is_claimed: Whether a summary task is currently leased to a worker.
        is_encoding: Whether the summary proxy is currently compressing locally.

    Returns:
        str: Canonical summary status label.
    """
    if not pipe.summary_enabled:
        return "disabled"
    if pipe.summary.response is not None:
        return "completed"
    if pipe.summary.exhausted:
        return "failed"
    if is_claimed:
        return "claimed"
    if is_encoding:
        return "encoding"
    return "pending"


def _get_segment_statuses(
    pipe: Pipeline,
    seg: SegmentState,
    is_encoding: bool,
    active_task_type: TaskType | None,
) -> tuple[str, str, str]:
    """Derives matrix, subtitles, and lyrics dashboard statuses for a segment.

    Args:
        pipe: Owning Pipeline state machine.
        seg: Target segment state.
        is_encoding: Whether the segment is currently compressing locally.
        active_task_type: Currently leased AI task type for this segment, if any.

    Returns:
        tuple[str, str, str]: (matrix_status, subtitles_status, lyrics_status).
    """
    sub_claimed = active_task_type == TaskType.SUBTITLES
    lyr_claimed = active_task_type == TaskType.LYRICS
    needs_lyrics = pipe.segment_needs_lyrics(seg)

    if seg.subtitles_exhausted:
        matrix_status = "failed"
    elif sub_claimed:
        matrix_status = "subtitles_running"
    elif lyr_claimed:
        matrix_status = "lyrics_running"
    elif seg.subtitles_response is not None:
        if not needs_lyrics or seg.lyrics_response is not None:
            matrix_status = "done"
        elif seg.lyrics_exhausted:
            matrix_status = "draft_fallback"
        else:
            matrix_status = "lyrics_pending"
    elif is_encoding:
        matrix_status = "encoding"
    elif seg.media_ready:
        matrix_status = "ready"
    else:
        matrix_status = "pending"

    sub_status = (
        "completed"
        if seg.subtitles_response is not None
        else ("failed" if seg.subtitles_exhausted else ("in_progress" if sub_claimed else "pending"))
    )

    if not pipe.lyrics_enabled:
        lyr_status = "disabled"
    elif seg.subtitles_exhausted or (seg.subtitles_response is not None and not needs_lyrics):
        lyr_status = "skipped"
    elif seg.subtitles_response is None:
        lyr_status = "waiting"
    elif seg.lyrics_response is not None:
        lyr_status = "completed"
    elif seg.lyrics_exhausted:
        lyr_status = "draft_fallback"
    elif lyr_claimed:
        lyr_status = "in_progress"
    else:
        lyr_status = "pending"

    return matrix_status, sub_status, lyr_status


def _build_segment_info(
    seg: SegmentState,
    pipe: Pipeline,
    seg_tasks: dict[tuple[TaskType, int], ServerTask],
    active_reencode_segments: set[int],
    subtitles_max_runs: int,
    lyrics_max_runs: int,
    now: float,
) -> dict[str, Any]:
    """Builds dashboard inspection and matrix telemetry for a single video segment.

    Args:
        seg: Target segment state machine.
        pipe: Owning pipeline state machine.
        seg_tasks: Lookup table of segment ServerTasks keyed by (task_type, segment_index).
        active_reencode_segments: Indices of segments actively running FFmpeg compression.
        subtitles_max_runs: Configured maximum non-retryable runs per FPS tier for subtitles.
        lyrics_max_runs: Configured maximum non-retryable runs for lyrics refinement.
        now: Current Unix epoch timestamp.

    Returns:
        dict[str, Any]: Structured segment matrix and inspector dictionary.
    """
    sub_task = seg_tasks.get((TaskType.SUBTITLES, seg.index))
    lyr_task = seg_tasks.get((TaskType.LYRICS, seg.index))
    is_encoding = seg.index in active_reencode_segments
    sub_claimed = sub_task is not None and sub_task.status == TaskStatus.CLAIMED
    lyr_claimed = lyr_task is not None and lyr_task.status == TaskStatus.CLAIMED
    active_task = sub_task if sub_claimed else (lyr_task if lyr_claimed else None)

    matrix_status, sub_status, lyr_status = _get_segment_statuses(
        pipe=pipe,
        seg=seg,
        is_encoding=is_encoding,
        active_task_type=active_task.task_type if active_task else None,
    )

    detected_songs: list[dict[str, Any]] = []
    sub_line_count: int | None = None
    if seg.subtitles_response is not None:
        sub_line_count = len(seg.subtitles_response.subtitles)
        detected_songs = [
            {
                "title": perf.title,
                "artist": perf.artist,
                "start": perf.start,
                "end": perf.end,
            }
            for perf in seg.subtitles_response.detected_musical_performances
        ]

    lyr_line_count: int | None = len(seg.lyrics_response.subtitles) if seg.lyrics_response is not None else None

    return {
        "index": seg.index,
        "offset_ms": seg.offset_ms,
        "duration_ms": seg.duration_ms,
        "matrix_status": matrix_status,
        "media_ready": seg.media_ready,
        "is_encoding": is_encoding,
        "fps": seg.fps,
        "encoding": seg.encoding,
        "fallback_tier": seg.fallback_tier,
        "fallback_tiers": build_fallback_tiers(
            seg,
            pipe.reencode_settings.segment,
            seg.raw_path,
            exhausted=seg.subtitles_exhausted,
        ),
        "subtitles_status": sub_status,
        "subtitles_tier_attempts": seg.subtitles_tier_attempts,
        "subtitles_line_count": sub_line_count,
        "detected_songs": detected_songs,
        "lyrics_status": lyr_status,
        "lyrics_tier_attempts": seg.lyrics_tier_attempts,
        "lyrics_line_count": lyr_line_count,
        "max_runs": subtitles_max_runs,
        "lyrics_max_runs": lyrics_max_runs,
        "active_task_type": active_task.task_type.value if active_task else None,
        "active_worker_id": active_task.worker_id if active_task else None,
        "active_elapsed_s": int(now - active_task.claimed_at) if active_task and active_task.claimed_at else None,
        "logfire_query": _format_chunk_logfire_query(sub_task, lyr_task),
    }


def _build_stages_info(runner: ComboRunner, pipe: Pipeline, now: float) -> dict[str, Any]:
    """Computes per-stage progress counters, summary preview, and segment matrix for a dashboard card.

    Args:
        runner: Active ComboRunner instance.
        pipe: Active Pipeline state machine.
        now: Current Unix epoch timestamp.

    Returns:
        dict[str, Any]: Stage progress mapping and per-segment details.
    """
    summary_max_runs = pipe.get_retry_settings(TaskType.SUMMARY).maxruns
    subtitles_max_runs = pipe.get_retry_settings(TaskType.SUBTITLES).maxruns
    lyrics_max_runs = pipe.get_retry_settings(TaskType.LYRICS).maxruns

    summary_task: ServerTask | None = None
    seg_tasks: dict[tuple[TaskType, int], ServerTask] = {}
    sub_prog = 0
    lyr_prog = 0

    for t in runner.tasks.values():
        if t.task_type == TaskType.SUMMARY:
            summary_task = t
        elif t.segment_index is not None:
            seg_tasks[(t.task_type, t.segment_index)] = t
            if t.status == TaskStatus.CLAIMED:
                if t.task_type == TaskType.SUBTITLES:
                    sub_prog += 1
                elif t.task_type == TaskType.LYRICS:
                    lyr_prog += 1

    sum_state = pipe.summary
    sum_done = 1 if sum_state.response is not None else 0
    sum_claimed = summary_task is not None and summary_task.status == TaskStatus.CLAIMED
    summary_preview = (
        sum_state.response.model_dump(
            mode="json",
            exclude={
                "thoughts": True,
                "entities": {"__all__": {"persona_lore", "visual_cues"}},
            },
        )
        if sum_state.response is not None
        else None
    )

    total_chunks = len(pipe.segments)
    reencode_done = sum(1 for seg in pipe.segments if seg.media_ready)
    reencode_fallback_count = sum(1 for seg in pipe.segments if seg.fallback_tier > 0)
    sub_done = sum(1 for seg in pipe.segments if seg.subtitles_response is not None)
    sub_failed = sum(1 for seg in pipe.segments if seg.subtitles_exhausted)
    lyr_total = sum(1 for seg in pipe.segments if pipe.segment_needs_lyrics(seg))
    lyr_done = sum(1 for seg in pipe.segments if seg.lyrics_response is not None)
    lyr_concluded = sum(
        1 for seg in pipe.segments if pipe.segment_needs_lyrics(seg) and pipe.segment_lyrics_concluded(seg)
    )

    segments_info = [
        _build_segment_info(
            seg=seg,
            pipe=pipe,
            seg_tasks=seg_tasks,
            active_reencode_segments=runner.active_reencode_segments,
            subtitles_max_runs=subtitles_max_runs,
            lyrics_max_runs=lyrics_max_runs,
            now=now,
        )
        for seg in pipe.segments
    ]

    return {
        "summary": {
            "enabled": pipe.summary_enabled,
            "done": sum_done,
            "in_progress": 1 if sum_claimed else 0,
            "is_encoding": runner.active_summary_reencode,
            "exhausted": sum_state.exhausted,
            "status": _get_summary_status(
                pipe,
                is_claimed=sum_claimed,
                is_encoding=runner.active_summary_reencode,
            ),
            "worker_id": summary_task.worker_id if sum_claimed and summary_task else None,
            "elapsed_s": (
                int(now - summary_task.claimed_at) if sum_claimed and summary_task and summary_task.claimed_at else None
            ),
            "logfire_query": format_logfire_query(summary_task) if summary_task else None,
            "fps": sum_state.fps,
            "encoding": sum_state.encoding,
            "fallback_tier": sum_state.fallback_tier,
            "fallback_tiers": build_fallback_tiers(
                sum_state,
                pipe.reencode_settings.summary,
                pipe.raw_video_path,
                exhausted=sum_state.exhausted,
            ),
            "tier_attempts": sum_state.tier_attempts,
            "max_runs": summary_max_runs,
            "preview": summary_preview,
        },
        "reencoding": {
            "total": total_chunks,
            "done": reencode_done,
            "in_progress": len(runner.active_reencode_segments),
            "fallback_count": reencode_fallback_count,
        },
        "subtitles": {
            "total": total_chunks,
            "done": sub_done,
            "failed": sub_failed,
            "in_progress": sub_prog,
        },
        "lyrics": {
            "enabled": pipe.lyrics_enabled,
            "total": lyr_total,
            "done": lyr_done,
            "failed": max(0, lyr_concluded - lyr_done),
            "in_progress": lyr_prog,
            "skipped": pipe.lyrics_enabled and (sub_done + sub_failed) == total_chunks and lyr_total == 0,
        },
        "segments": segments_info,
    }


def build_dashboard_data(
    runners: dict[str, ComboRunner],
    runner_activity_logs: dict[str, deque[ActivityEvent]],
    pipeline_queues: dict[str, dict[str, Any]],
    git_repo_url: str | None,
    youtube_cooldown: dict[str, Any] | None = None,
    total_tracked_videos: int = 0,
) -> dict[str, Any]:
    """Gathers macroscopic metrics, active stages, queues, and activity logs for the live WebUI.

    Args:
        runners: Active combination runners keyed by pipeline name.
        runner_activity_logs: Per-pipeline bounded activity event buffers.
        pipeline_queues: Mapping of runner names to their outstanding queue dictionaries.
        git_repo_url: Sanitized remote Git repository URL, or None.
        youtube_cooldown: Optional YouTube intake cooldown status dictionary. Defaults to None.
        total_tracked_videos: Count of all media files and streams tracked in memory. Defaults to 0.

    Returns:
        dict[str, Any]: Complete dashboard state dictionary.
    """
    now = time()
    pipeline_data = []

    for name, runner in runners.items():
        active_info = None
        stages_info = None
        pipe = runner.pipeline

        if pipe is not None:
            active_info = {
                "video_id": pipe.video_id,
                "title": pipe.title,
                "uploader": pipe.uploader,
                "channel_name": pipe.channel_name,
                "duration_ms": pipe.duration_ms,
                "source_type": pipe.source_type,
            }
            stages_info = _build_stages_info(runner, pipe, now)

        tasks_list: list[dict[str, Any]] = []
        for t in runner.tasks.values():
            if t.status != TaskStatus.CLAIMED:
                continue
            stage_state = pipe.get_stage_media_state(t.task_type, t.segment_index) if pipe is not None else None
            tasks_list.append(
                {
                    "task_id": t.task_id,
                    "type": t.task_type.value,
                    "worker_id": t.worker_id,
                    "elapsed_s": int(now - t.claimed_at) if t.claimed_at else None,
                    "fallback_tier": stage_state.fallback_tier if stage_state is not None else 0,
                    "fps": stage_state.fps if stage_state is not None else None,
                    "encoding": stage_state.encoding if stage_state is not None else "raw",
                    "logfire_query": format_logfire_query(t),
                }
            )

        q_info = pipeline_queues.get(name, {"total_count": 0, "items": []})
        pipeline_data.append(
            {
                "name": name,
                "status": runner.status.value,
                "models": {
                    "summary": runner.config.summary.model,
                    "subtitles": runner.config.subtitles.model,
                    "lyrics": runner.config.lyrics.model,
                },
                "active_video": active_info,
                "preparing_video": runner.preparing_video_info,
                "stages": stages_info,
                "tasks": tasks_list,
                "queue_total": q_info.get("total_count", 0),
                "next_in_queue": q_info.get("items", []),
                "recent_activity": [event.model_dump() for event in runner_activity_logs.get(name, [])],
            }
        )

    return {
        "server_version": get_app_version(),
        "git_repo_url": git_repo_url,
        "youtube_cooldown": youtube_cooldown or {"in_cooldown": False},
        "total_tracked_videos": total_tracked_videos,
        "pipelines": pipeline_data,
    }
