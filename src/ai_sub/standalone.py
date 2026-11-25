"""Standalone pipeline orchestrator and CLI entrypoint for AI Sub."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

import logfire
from pydantic_settings import CliApp
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ai_sub.gemini_executor import GeminiExecutor, QuotaExceededError, is_retryable_error
from ai_sub.jobs import (
    AiJob,
    BaseJob,
    LyricsJob,
    ReencodeJob,
    SubtitlesJob,
    SummaryJob,
    WorkerContext,
)
from ai_sub.pipeline import Pipeline
from ai_sub.search import BaseSearchProvider, create_search_providers, order_lyrics_providers
from ai_sub.standalone_config import StandaloneSettings
from ai_sub.telemetry import setup_logging
from ai_sub.video import init_ffmpeg_engine


class AiSubResult(IntEnum):
    """Application exit status codes."""

    COMPLETED = 0
    IN_PROGRESS = -1
    RETRIES_EXCEEDED = -2


JobKey = tuple[str, int | None]
JobTask = asyncio.Task[tuple[BaseJob, Any, Exception | None]]


def _get_job_key(job: BaseJob) -> JobKey:
    """Computes a unique in-flight tracking key for a job instance.

    Args:
        job: Job instance returned by Pipeline.get_pending_jobs().

    Returns:
        JobKey: Tuple of (job_class_name, segment_index).
    """
    return (type(job).__name__, getattr(job, "segment_index", None))


def _job_model(job: BaseJob) -> str | None:
    """Returns the target Gemini model name for AI jobs, or None for local media jobs.

    Args:
        job: Job instance.

    Returns:
        str | None: Gemini model name or None.
    """
    return job.model_name if isinstance(job, AiJob) else None


def _sync_progress(
    progress: Progress,
    pipeline: Pipeline,
    summary_tid: TaskID | None,
    reencode_tid: TaskID | None,
    subtitles_tid: TaskID,
    lyrics_tid: TaskID | None,
) -> None:
    """Synchronizes Rich progress bar counters directly from current pipeline unit states.

    Args:
        progress: Active Rich Progress instance.
        pipeline: Pipeline holding authoritative unit states.
        summary_tid: Optional summary progress task ID.
        reencode_tid: Optional segment re-encoding progress task ID.
        subtitles_tid: Subtitles progress task ID.
        lyrics_tid: Optional lyrics refinement progress task ID.
    """
    total_chunks = len(pipeline.segments)

    if summary_tid is not None:
        sum_done = 1 if (pipeline.is_summary_complete() or pipeline.summary.exhausted) else 0
        progress.update(summary_tid, completed=sum_done)

    if reencode_tid is not None:
        reenc_done = sum(1 for seg in pipeline.segments if seg.media_ready)
        progress.update(reencode_tid, completed=reenc_done, total=total_chunks)

    sub_done = sum(1 for seg in pipeline.segments if pipeline.segment_subtitles_concluded(seg))
    progress.update(subtitles_tid, completed=sub_done, total=total_chunks)

    if lyrics_tid is not None:
        lyr_done = sum(1 for seg in pipeline.segments if pipeline.segment_lyrics_concluded(seg))
        progress.update(lyrics_tid, completed=lyr_done, total=total_chunks)


async def _run_job(
    job: BaseJob,
    pipeline: Pipeline,
    executor: GeminiExecutor,
    web_search_providers: list[BaseSearchProvider],
    lyrics_search_providers: list[BaseSearchProvider],
    reencode_sem: asyncio.Semaphore,
    summary_sem: asyncio.Semaphore,
    subtitles_sem: asyncio.Semaphore,
    lyrics_sem: asyncio.Semaphore,
) -> tuple[BaseJob, Any, Exception | None]:
    """Executes a single job unit under its respective concurrency semaphore.

    Args:
        job: The job instance to execute.
        pipeline: Active Pipeline instance.
        executor: Configured Gemini executor.
        web_search_providers: Providers available for summary search.
        lyrics_search_providers: Providers available for lyrics search.
        reencode_sem: Concurrency semaphore for FFmpeg re-encoding.
        summary_sem: Concurrency semaphore for summary generation.
        subtitles_sem: Concurrency semaphore for subtitle generation.
        lyrics_sem: Concurrency semaphore for lyrics refinement.

    Returns:
        tuple[BaseJob, Any, Exception | None]: (job, output_result, exception_if_any).
    """
    try:
        if isinstance(job, ReencodeJob):
            async with reencode_sem:
                return job, await job.execute(), None

        if isinstance(job, SummaryJob):
            sem, providers = summary_sem, web_search_providers
        elif isinstance(job, SubtitlesJob):
            sem, providers = subtitles_sem, []
        elif isinstance(job, LyricsJob):
            sem, providers = lyrics_sem, lyrics_search_providers
        else:
            return job, None, ValueError(f"Unknown job type: {type(job)}")

        async with sem:
            ctx = WorkerContext(
                executor=executor,
                media_path=pipeline.get_job_media_path(job),
                search_providers=providers,
                retry_settings=pipeline.get_retry_settings(job.task_type),
                validation_buffer_ms=pipeline.validation_buffer_ms,
            )
            return job, await job.execute(ctx), None

    except Exception as e:
        return job, None, e


async def run_pipeline(settings: StandaloneSettings) -> AiSubResult:
    """Executes the subtitle generation pipeline driven by Pipeline.

    Args:
        settings: Configured standalone application settings.

    Returns:
        AiSubResult: Overall status code.
    """
    console = Console(stderr=True)
    setup_logging(settings.log, console=console)
    input_video_path = cast(Path, settings.video)
    tmp_dir = cast(Path, settings.dir.tmp)
    out_dir = cast(Path, settings.dir.out)
    user_context = settings.get_resolved_user_context()

    with logfire.span("Pipeline: {video_file}", video_file=input_video_path.name):
        if user_context:
            logfire.info(f"Loaded user context ({len(user_context)} characters).")

        use_summary = settings.concurrency.summary > 0
        use_lyrics = settings.concurrency.lyrics > 0

        if settings.nfo and not use_summary:
            logfire.warning(
                "NFO generation requires summary to be enabled (concurrency.summary > 0); skipping .nfo output."
            )

        sum_reencode = settings.reencode.summary
        seg_reencode = settings.reencode.segment
        reencode_possible = bool(
            (use_summary and ((sum_reencode.threshold > 0 and sum_reencode.fps != 0) or sum_reencode.fallback.enabled))
            or (seg_reencode.threshold > 0 and seg_reencode.fps != 0)
            or seg_reencode.fallback.enabled
        )
        if reencode_possible:
            _, _, encoder = await init_ffmpeg_engine(settings.reencode.encoder)
            settings.reencode.encoder = encoder

        pipeline = await Pipeline.create(
            raw_video_path=input_video_path,
            job_dir=tmp_dir,
            title=input_video_path.stem,
            uploader="local",
            summary_config=settings.ai.summary,
            subtitles_config=settings.ai.subtitles,
            lyrics_config=settings.ai.lyrics,
            reencode_settings=settings.reencode,
            search_settings=settings.ai.search,
            split_settings=settings.split,
            retry_settings=settings.retry,
            user_context=user_context,
            validation_buffer_ms=int(settings.ai.buffer.total_seconds() * 1000),
            summary_enabled=use_summary,
            lyrics_enabled=use_lyrics,
        )

        async with AsyncExitStack() as stack:
            web_search_providers: list[BaseSearchProvider] = []
            lyrics_search_providers: list[BaseSearchProvider] = []

            if use_summary or use_lyrics:
                configured = create_search_providers(settings.ai.search)
                for provider in configured:
                    await stack.enter_async_context(provider)
                web_search_providers = [p for p in configured if p.provider_name != "lrclib"]

                if use_lyrics:
                    lyrics_search_providers = order_lyrics_providers(configured)

            executor = GeminiExecutor(settings.ai, settings.retry)
            stack.push_async_callback(executor.aclose)

            reencode_sem = asyncio.Semaphore(settings.concurrency.reencode)
            summary_sem = asyncio.Semaphore(max(1, settings.concurrency.summary))
            subtitles_sem = asyncio.Semaphore(settings.concurrency.subtitles)
            lyrics_sem = asyncio.Semaphore(max(1, settings.concurrency.lyrics))

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=not settings.log.progress,
            )

            total_chunks = len(pipeline.segments)
            segment_reencode_active = any(seg.target_fps is not None for seg in pipeline.segments)
            with progress:
                summary_tid: TaskID | None = progress.add_task("[yellow]Summary", total=1) if use_summary else None
                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=total_chunks) if segment_reencode_active else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles", total=total_chunks)
                lyrics_tid: TaskID | None = (
                    progress.add_task("[magenta]Lyrics", total=total_chunks) if use_lyrics else None
                )

                _sync_progress(progress, pipeline, summary_tid, reencode_tid, subtitles_tid, lyrics_tid)

                in_flight: dict[JobTask, tuple[JobKey, str | None]] = {}
                active_keys: set[JobKey] = set()
                blocked_this_run: set[JobKey] = set()
                exhausted_models: set[str] = set()

                async def _handle_job_outcome(
                    job: BaseJob,
                    result: Any,
                    exc: Exception | None,
                ) -> str | None:
                    if exc is None:
                        pipeline.complete_job(job, result)
                        return None

                    if isinstance(exc, QuotaExceededError):
                        job_model = _job_model(job)
                        model_label = job_model or "unknown"
                        logfire.warning(
                            f"Daily quota reached for '{job.name}' ({model_label}): {exc}. "
                            "Skipping remaining jobs for this model without consuming run attempts."
                        )
                        if job_model:
                            exhausted_models.add(job_model)
                            return job_model
                        return None

                    if isinstance(job, ReencodeJob):
                        logfire.warning(f"Re-encode failed for '{job.name}': {exc}")
                        outcome = await pipeline.record_job_failure(job)
                        if outcome.action != "retry":
                            blocked_this_run.add(_get_job_key(job))
                    elif is_retryable_error(exc):
                        logfire.warning(
                            f"Job '{job.name}' failed with retryable error: {exc}. "
                            "Skipping for this run without consuming run attempts."
                        )
                        blocked_this_run.add(_get_job_key(job))
                    else:
                        logfire.error(f"Job '{job.name}' failed with non-retryable error: {exc}")
                        outcome = await pipeline.record_job_failure(job)

                        # Defer retries and newly triggered fallback re-encodes to subsequent invocations
                        blocked_this_run.add(_get_job_key(job))
                        if outcome.action == "fallback_reencode":
                            seg_idx = getattr(job, "segment_index", None)
                            blocked_this_run.add(("ReencodeJob", seg_idx))

                    return None

                async def _drain_tasks(tasks_to_drain: list[JobTask]) -> None:
                    if not tasks_to_drain:
                        return
                    drain_results = await asyncio.gather(*tasks_to_drain, return_exceptions=True)
                    for task_obj, res in zip(tasks_to_drain, drain_results, strict=True):
                        if popped := in_flight.pop(task_obj, None):
                            active_keys.discard(popped[0])
                        if isinstance(res, tuple):
                            drained_job, drained_out, drained_exc = res
                            await _handle_job_outcome(drained_job, drained_out, drained_exc)
                    _sync_progress(progress, pipeline, summary_tid, reencode_tid, subtitles_tid, lyrics_tid)

                try:
                    # Consumer loop pulling ready DAG jobs directly from the pipeline
                    while not pipeline.is_done():
                        for job in pipeline.get_pending_jobs():
                            job_key = _get_job_key(job)
                            job_model = _job_model(job)
                            if job_key in active_keys or job_key in blocked_this_run:
                                continue
                            if job_model and job_model in exhausted_models:
                                continue

                            task = asyncio.create_task(
                                _run_job(
                                    job=job,
                                    pipeline=pipeline,
                                    executor=executor,
                                    web_search_providers=web_search_providers,
                                    lyrics_search_providers=lyrics_search_providers,
                                    reencode_sem=reencode_sem,
                                    summary_sem=summary_sem,
                                    subtitles_sem=subtitles_sem,
                                    lyrics_sem=lyrics_sem,
                                )
                            )
                            in_flight[task] = (job_key, job_model)
                            active_keys.add(job_key)

                        if not in_flight:
                            break

                        done, _ = await asyncio.wait(list(in_flight), return_when=asyncio.FIRST_COMPLETED)
                        newly_exhausted: set[str] = set()

                        for completed_task in done:
                            if popped := in_flight.pop(completed_task, None):
                                active_keys.discard(popped[0])

                            if completed_task.cancelled():
                                continue

                            job, result, exc = completed_task.result()
                            if exhausted_model := await _handle_job_outcome(job, result, exc):
                                newly_exhausted.add(exhausted_model)

                            _sync_progress(progress, pipeline, summary_tid, reencode_tid, subtitles_tid, lyrics_tid)

                        if pipeline.summary_aborted:
                            break

                        if newly_exhausted:
                            cancelled_tasks = [t for t, (_, model) in in_flight.items() if model in newly_exhausted]
                            for pending_task in cancelled_tasks:
                                if not pending_task.done():
                                    pending_task.cancel()
                            await _drain_tasks(cancelled_tasks)

                    if not pipeline.summary_aborted:
                        await _drain_tasks(list(in_flight))
                finally:
                    if in_flight:
                        remaining_tasks = list(in_flight)
                        for pending_task in remaining_tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        await asyncio.gather(*remaining_tasks, return_exceptions=True)
                        in_flight.clear()
                        active_keys.clear()

        if pipeline.summary_aborted:
            logfire.error("Whole-video summary failed across all fallback FPS tiers. Aborting without generating SRT.")
            return AiSubResult.RETRIES_EXCEEDED

        if not pipeline.is_summary_complete():
            logfire.info("Whole-video summary incomplete; skipping subtitle generation for this run.")
            return AiSubResult.IN_PROGRESS

        # Finalize and stitch synchronized subtitle tracks
        generated = pipeline.finalize(
            output_dir=out_dir,
            filename_stem=input_video_path.stem,
            allow_partial=True,
        )
        for g in generated:
            logfire.info(f"Subtitles written to {g.name}")

        if settings.nfo and use_summary:
            if nfo_path := pipeline.write_nfo(output_dir=out_dir, filename_stem=input_video_path.stem):
                logfire.info(f"NFO written to {nfo_path.name}")

        if not pipeline.is_done():
            return AiSubResult.IN_PROGRESS

        if pipeline.has_failures():
            return AiSubResult.RETRIES_EXCEEDED

        return AiSubResult.COMPLETED


def main() -> None:
    """CLI entrypoint for the AI Sub standalone subtitle generation pipeline."""
    settings = CliApp.run(StandaloneSettings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
