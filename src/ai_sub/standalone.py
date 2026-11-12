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

from ai_sub.config import Settings
from ai_sub.gemini_executor import GeminiExecutor, QuotaExceededError
from ai_sub.jobs import (
    BaseJob,
    LyricsJob,
    ReencodeJob,
    SubtitlesAiResponse,
    SubtitlesJob,
    SummaryJob,
    WorkerContext,
)
from ai_sub.pipeline import Pipeline
from ai_sub.protocol import TaskType
from ai_sub.search import (
    BaseSearchProvider,
    create_search_provider,
    create_search_providers,
)
from ai_sub.telemetry import setup_logging
from ai_sub.video import init_ffmpeg_engine


class AiSubResult(IntEnum):
    """Application exit status codes."""

    COMPLETED = 0
    IN_PROGRESS = -1
    RETRIES_EXCEEDED = -2


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
                out_path = await job.execute()
                return job, out_path, None

        elif isinstance(job, SummaryJob):
            async with summary_sem:
                media = job.media_path or pipeline.summary_proxy_path or pipeline.raw_video_path
                ctx = WorkerContext(
                    executor=executor,
                    media_path=media,
                    search_providers=web_search_providers,
                )
                res = await job.execute(ctx)
                return job, res, None

        elif isinstance(job, SubtitlesJob):
            async with subtitles_sem:
                media = job.media_path or pipeline.raw_video_path
                ctx = WorkerContext(
                    executor=executor,
                    media_path=media,
                    search_providers=[],
                )
                res = await job.execute(ctx)
                return job, res, None

        elif isinstance(job, LyricsJob):
            async with lyrics_sem:
                media = job.media_path or pipeline.raw_video_path
                ctx = WorkerContext(
                    executor=executor,
                    media_path=media,
                    search_providers=lyrics_search_providers,
                )
                res = await job.execute(ctx)
                return job, res, None

        return job, None, ValueError(f"Unknown job type: {type(job)}")

    except Exception as e:
        return job, None, e


async def run_pipeline(settings: Settings) -> AiSubResult:
    """Executes the subtitle generation pipeline driven by Pipeline.

    Args:
        settings: Configured application settings.

    Returns:
        AiSubResult: Overall status code.
    """
    console = Console(stderr=True)
    setup_logging(settings, console=console)
    input_video_path = cast(Path, settings.input_video_file)
    user_context = settings.get_resolved_user_context()

    with logfire.span("Pipeline: {video_file}", video_file=input_video_path.name):
        if user_context:
            logfire.info(f"Loaded user context ({len(user_context)} characters).")

        if settings.reencode.summary.enabled or settings.reencode.segment.enabled:
            _, _, encoder = await init_ffmpeg_engine(settings.reencode.encoder)
            settings.reencode.encoder = encoder

        use_summary = settings.concurrency.summary > 0
        use_lyrics = settings.concurrency.lyrics > 0

        # Initialize the unified pipeline: probes duration, splits, applies offset, checks re-encoding
        pipeline = await Pipeline.create(
            raw_video_path=input_video_path,
            job_dir=settings.dir.tmp,
            title=input_video_path.stem,
            uploader="local",
            summary_model=settings.ai.model_summary,
            subtitles_model=settings.ai.model_subtitles,
            lyrics_model=settings.ai.model_lyrics,
            service_tier_summary=settings.ai.service_tier_summary,
            service_tier_subtitles=settings.ai.service_tier_subtitles,
            service_tier_lyrics=settings.ai.service_tier_lyrics,
            agentic_video_summary=settings.ai.agentic_video_summary,
            agentic_video_subtitles=settings.ai.agentic_video_subtitles,
            agentic_video_lyrics=settings.ai.agentic_video_lyrics,
            reencode_settings=settings.reencode,
            search_settings=settings.ai.search,
            split_settings=settings.split,
            retry_settings=settings.retry,
            user_context=user_context,
            validation_buffer_ms=settings.ai.validation_buffer_ms,
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
                    lrclib_candidates = [p for p in configured if p.provider_name == "lrclib"]
                    if lrclib_candidates:
                        lrclib_provider = lrclib_candidates[0]
                    else:
                        lrclib_provider = create_search_provider("lrclib", settings.ai.search)
                        await stack.enter_async_context(lrclib_provider)
                    lyrics_search_providers = [lrclib_provider, *web_search_providers]

            executor = GeminiExecutor(settings)
            reencode_sem = asyncio.Semaphore(settings.concurrency.re_encode)
            summary_sem = asyncio.Semaphore(settings.concurrency.summary if settings.concurrency.summary > 0 else 1)
            subtitles_sem = asyncio.Semaphore(settings.concurrency.subtitles)
            lyrics_sem = asyncio.Semaphore(settings.concurrency.lyrics if settings.concurrency.lyrics > 0 else 1)

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

            total_chunks = len(pipeline.segment_paths)
            with progress:
                summary_tid: TaskID | None = progress.add_task("[yellow]Summary", total=1) if use_summary else None
                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=total_chunks)
                    if settings.reencode.segment.enabled
                    else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles", total=total_chunks)
                lyrics_tid: TaskID | None = (
                    progress.add_task("[magenta]Lyrics", total=total_chunks) if use_lyrics else None
                )

                # Initialize progress for previously completed disk checkpoints
                if summary_tid is not None and pipeline.is_summary_complete():
                    progress.update(summary_tid, completed=1)
                for idx in range(total_chunks):
                    if reencode_tid is not None and idx in pipeline.reencoded_segment_paths:
                        progress.update(reencode_tid, advance=1)
                    if idx in pipeline.subtitles_responses:
                        progress.update(subtitles_tid, advance=1)
                        if lyrics_tid is not None:
                            sub_r = pipeline.subtitles_responses[idx]
                            if not sub_r.detected_musical_performances or idx in pipeline.lyrics_responses:
                                progress.update(lyrics_tid, advance=1)

                in_flight: dict[str, asyncio.Task[tuple[BaseJob, Any, Exception | None]]] = {}
                quota_exhausted = False

                # Consumer loop pulling ready DAG jobs directly from the pipeline
                while not pipeline.is_done():
                    pending_jobs = pipeline.get_pending_jobs()
                    for job in pending_jobs:
                        if job.name not in in_flight:
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
                            in_flight[job.name] = task

                    if not in_flight:
                        break

                    done, _ = await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED)
                    for completed_task in done:
                        # Find and delete matching in_flight entry before checking cancellation
                        job_name = next((k for k, v in in_flight.items() if v is completed_task), None)
                        if job_name:
                            del in_flight[job_name]

                        if completed_task.cancelled():
                            continue

                        job, result, exc = completed_task.result()

                        if exc is not None:
                            if isinstance(exc, QuotaExceededError):
                                logfire.warning(
                                    f"Daily quota limit reached during '{job.name}': {exc}. "
                                    "Suspending remaining jobs without consuming retry limits."
                                )
                                quota_exhausted = True
                                continue

                            logfire.error(f"Job '{job.name}' failed: {exc}")

                            if isinstance(job, SummaryJob):
                                pipeline.record_failure(TaskType.SUMMARY, attempts=job.total_attempts + 1)
                                if summary_tid is not None:
                                    progress.update(summary_tid, completed=1)
                            elif isinstance(job, SubtitlesJob):
                                pipeline.record_failure(
                                    TaskType.SUBTITLES,
                                    segment_index=job.segment_index,
                                    attempts=job.total_attempts + 1,
                                )
                                progress.update(subtitles_tid, advance=1)
                            elif isinstance(job, LyricsJob):
                                pipeline.record_failure(
                                    TaskType.LYRICS,
                                    segment_index=job.segment_index,
                                    attempts=job.total_attempts + 1,
                                )
                                if lyrics_tid is not None:
                                    progress.update(lyrics_tid, advance=1)
                            elif isinstance(job, ReencodeJob):
                                logfire.warning(f"Re-encode failed for chunk '{job.name}'. Falling back to raw file.")
                                pipeline.complete_job(job, job.input_file)
                                if not job.is_summary_proxy and reencode_tid is not None:
                                    progress.update(reencode_tid, advance=1)
                        else:
                            pipeline.complete_job(job, result)
                            if isinstance(job, SummaryJob) and summary_tid is not None:
                                progress.update(summary_tid, completed=1)
                            elif isinstance(job, ReencodeJob) and not job.is_summary_proxy and reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)
                            elif isinstance(job, SubtitlesJob):
                                progress.update(subtitles_tid, advance=1)
                                if (
                                    lyrics_tid is not None
                                    and isinstance(result, SubtitlesAiResponse)
                                    and not result.detected_musical_performances
                                ):
                                    progress.update(lyrics_tid, advance=1)
                            elif isinstance(job, LyricsJob) and lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)

                    if quota_exhausted:
                        for pending_task in in_flight.values():
                            pending_task.cancel()
                        await asyncio.gather(*in_flight.values(), return_exceptions=True)
                        in_flight.clear()
                        break

                # Await any remaining in-flight tasks if loop ended early without quota exhaustion
                if in_flight:
                    drain_results = await asyncio.gather(*in_flight.values(), return_exceptions=True)
                    for res in drain_results:
                        if isinstance(res, tuple):
                            job, out, exc = res
                            if exc is None:
                                pipeline.complete_job(job, out)

        # Finalize and stitch synchronized subtitle tracks
        out_srt = settings.dir.out / f"{input_video_path.stem}.{pipeline.full_shortcode}.srt"
        raw_out_srt = settings.dir.out / f"{input_video_path.stem}.raw-{pipeline.full_shortcode}.srt"

        generated = pipeline.finalize(
            output_primary_path=out_srt,
            output_raw_path=raw_out_srt if use_lyrics else None,
            allow_partial=True,
        )
        for g in generated:
            logfire.info(f"Subtitles written to {g.name}")

        has_failures = bool(
            pipeline.summary_failed or pipeline.failed_subtitles or (use_lyrics and pipeline.failed_lyrics)
        )
        if pipeline.is_done() and not has_failures:
            return AiSubResult.COMPLETED

        if has_failures:
            return AiSubResult.RETRIES_EXCEEDED

        return AiSubResult.IN_PROGRESS


def main() -> None:
    """CLI entrypoint for the AI Sub standalone subtitle generation pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
