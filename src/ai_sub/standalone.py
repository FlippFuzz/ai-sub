"""Standalone pipeline orchestrator and CLI entrypoint for AI Sub."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

import logfire
from pydantic_settings import CliApp
from pysubs2 import SSAEvent, SSAFile
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
    LYRICS_PROMPT_VERSION,
    SUBTITLES_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    DetectedMusicalPerformance,
    LyricsJob,
    ReencodeJob,
    Subtitles,
    SubtitlesJob,
    SummaryAiResponse,
    SummaryJob,
    WorkerContext,
)
from ai_sub.search import (
    BaseSearchProvider,
    create_search_provider,
    create_search_providers,
)
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_subtitles_shortcode,
    generate_summary_shortcode,
)
from ai_sub.telemetry import get_app_version, setup_logging
from ai_sub.video import (
    get_video_duration_ms,
    init_ffmpeg_engine,
    reencode_video,
    split_video,
)


class AiSubResult(IntEnum):
    """Application exit status codes."""

    COMPLETE = 0
    """All video segments processed successfully."""

    INCOMPLETE = -1
    """One or more segments failed and can be retried on a subsequent run."""

    MAX_RETRIES_EXHAUSTED = -2
    """One or more segments exhausted the persistent retry limit."""


@dataclass
class SegmentTask:
    """Internal task tracking container for a single video segment."""

    index: int
    raw_path: Path
    duration_ms: int
    stem: str
    media_path: Path
    offset_ms: int = 0
    draft_subtitles: list[Subtitles] | None = None
    detected_performances: list[DetectedMusicalPerformance] | None = None
    subtitles_complete: bool = False


def stitch_subtitles(
    video_splits: list[tuple[Path, int]],
    settings: Settings,
    lyrics_shortcode: str,
    subtitles_shortcode: str,
    full_shortcode: str,
    summary_available: bool,
) -> AiSubResult:
    """Assembles segment subtitles (lyrics-refined or initial) into the final merged SRT file.

    Args:
        video_splits: Ordered list of segment paths and durations in milliseconds.
        settings: Application settings.
        lyrics_shortcode: Model shortcode used for lyrics state files.
        subtitles_shortcode: Model shortcode used for subtitle state files.
        full_shortcode: Full model shortcode used for final output SRT filename.
        summary_available: True if global summary succeeded.

    Returns:
        AiSubResult: Pipeline result status code.
    """
    with logfire.span("Stitching final SRT file"):
        all_subs = SSAFile()
        chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
        offset_ms = sum(d for _, d in video_splits[:chunks_to_skip])

        use_lyrics = settings.concurrency.lyrics > 0
        complete = True
        max_retries_exceeded = False
        any_pending = False

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            lyrics_job_path = settings.dir.tmp / f"{video_path.stem}.lyrics.{lyrics_shortcode}.yaml"
            sub_job_path = settings.dir.tmp / f"{video_path.stem}.subtitles.{subtitles_shortcode}.yaml"

            lyrics_job = LyricsJob.load(lyrics_job_path, settings.ai.validation_buffer_ms)
            sub_job = SubtitlesJob.load(sub_job_path, settings.ai.validation_buffer_ms)

            # Determine whether lyrics refinement was required for this segment
            lyrics_required = bool(
                use_lyrics and sub_job and sub_job.response and sub_job.response.detected_musical_performances
            )

            active_response = None
            if lyrics_required:
                # If lyrics was required, ONLY accept a successful lyrics refinement response.
                # If lyrics failed or is missing, do NOT fall back to draft subtitles.
                if lyrics_job and lyrics_job.response:
                    active_response = lyrics_job.response
            else:
                # Lyrics was not required; use initial subtitle response if present.
                if sub_job and sub_job.response:
                    active_response = sub_job.response

            if active_response:
                segment_subs = active_response.get_ssafile()
                segment_subs.shift(ms=offset_ms)
                all_subs += segment_subs
            else:
                all_subs.append(
                    SSAEvent(
                        start=offset_ms,
                        end=offset_ms + video_duration_ms,
                        text="Error processing subtitles for this segment.",
                    )
                )
                complete = False
                attempts = max(
                    lyrics_job.total_attempts if lyrics_job else 0,
                    sub_job.total_attempts if sub_job else 0,
                )

                if attempts >= settings.retry.max_runs:
                    max_retries_exceeded = True
                else:
                    any_pending = True

            offset_ms += video_duration_ms

        settings_dump = settings.model_dump(
            mode="json",
            exclude={
                "input_video_file": True,
                "context_file": True,
                "dir": True,
                "ai": {"key": True, "base_url": True, "search": {"key": True}},
            },
        )
        settings_dump["context"] = settings.get_resolved_user_context()

        metadata = {
            "ai_sub_version": get_app_version(),
            "generated_at": datetime.now().isoformat(),
            "summary_prompt_version": SUMMARY_PROMPT_VERSION if summary_available else None,
            "lyrics_prompt_version": LYRICS_PROMPT_VERSION,
            "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
            "complete": complete,
            "max_retries_exceeded": max_retries_exceeded,
            "any_pending": any_pending,
            "settings": settings_dump,
        }
        info_str = json.dumps(metadata, indent=2).replace("{", "(").replace("}", ")")
        all_subs.insert(0, SSAEvent(start=0, end=1, text=info_str))

        if len(all_subs) > 1 and all_subs[1].start < 1:
            all_subs[1].start = 1

        input_video_path = cast(Path, settings.input_video_file)
        out_srt = settings.dir.out / f"{input_video_path.stem}.{full_shortcode}.srt"
        all_subs.save(str(out_srt))
        logfire.info(f"Subtitles written to {out_srt.name}")

        if any_pending:
            return AiSubResult.INCOMPLETE
        if max_retries_exceeded:
            return AiSubResult.MAX_RETRIES_EXHAUSTED
        return AiSubResult.COMPLETE


async def run_pipeline(settings: Settings) -> AiSubResult:
    """Executes the subtitle generation pipeline.

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
        summary_shortcode = generate_summary_shortcode(settings.ai.model_summary)
        subtitles_shortcode = generate_subtitles_shortcode(
            model_name=settings.ai.model_subtitles,
            summary_shortcode=summary_shortcode if use_summary else None,
        )
        lyrics_shortcode = generate_lyrics_shortcode(
            lyrics_model_name=settings.ai.model_lyrics,
            subtitles_shortcode=subtitles_shortcode,
        )
        full_shortcode = generate_full_shortcode(settings.ai.model_subtitles)

        reencode_dir = settings.dir.tmp / "reencoded"
        if settings.reencode.segment.enabled:
            reencode_dir.mkdir(exist_ok=True)

        async with AsyncExitStack() as stack:
            web_search_providers: list[BaseSearchProvider] = []
            lyrics_search_providers: list[BaseSearchProvider] = []

            if use_summary or use_lyrics:
                configured_providers = create_search_providers(settings.ai.search)
                for provider in configured_providers:
                    await stack.enter_async_context(provider)

                # Summary phase: general web search providers only (never lrclib)
                web_search_providers = [p for p in configured_providers if p.provider_name != "lrclib"]

                if use_lyrics:
                    lrclib_candidates = [p for p in configured_providers if p.provider_name == "lrclib"]
                    if lrclib_candidates:
                        lrclib_provider = lrclib_candidates[0]
                    else:
                        lrclib_provider = create_search_provider("lrclib", settings.ai.search)
                        await stack.enter_async_context(lrclib_provider)

                    lyrics_search_providers = [lrclib_provider, *web_search_providers]

            executor = GeminiExecutor(settings)

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

            summary_response: SummaryAiResponse | None = None

            with progress:
                # ------------------------------------------------------------------
                # Summary: Macro Summary & Chapter Indexing
                # ------------------------------------------------------------------
                if use_summary:
                    summary_tid = progress.add_task("[yellow]Summary", total=1)
                    summary_file = settings.dir.tmp / f"summary.{summary_shortcode}.yaml"
                    summary_job = SummaryJob.load(summary_file, settings.ai.validation_buffer_ms)
                    if not summary_job:
                        summary_job = SummaryJob(
                            name="summary",
                            video_duration_ms=await get_video_duration_ms(input_video_path),
                            user_context=user_context,
                        )
                    else:
                        summary_job.user_context = user_context

                    if summary_job.response:
                        summary_response = summary_job.response
                        progress.update(summary_tid, completed=1)
                        logfire.info("Reusing existing valid global summary.")
                    elif summary_job.total_attempts < settings.retry.max_runs:
                        summary_media = input_video_path
                        should_reencode_summary = False

                        if settings.reencode.summary.enabled:
                            if settings.reencode.summary.threshold_mb == 0:
                                should_reencode_summary = True
                            else:
                                mb = input_video_path.stat().st_size / (1024 * 1024)
                                should_reencode_summary = mb >= settings.reencode.summary.threshold_mb

                        if should_reencode_summary:
                            proxy_output = settings.dir.tmp / "summary_proxy.mp4"
                            with logfire.span("Compressing whole-video summary proxy"):
                                await reencode_video(
                                    input_path=input_video_path,
                                    output_path=proxy_output,
                                    fps=settings.reencode.summary.fps,
                                    height=settings.reencode.summary.height,
                                    bitrate_kb=settings.reencode.summary.bitrate_kb,
                                    encoder=settings.reencode.encoder or "libx264",
                                    duration_tolerance_ms=settings.reencode.summary.duration_tolerance_ms,
                                    keyframe_only=settings.reencode.summary.keyframe_only,
                                )
                                summary_media = proxy_output

                        summary_duration_ms = await get_video_duration_ms(summary_media)
                        summary_job.video_duration_ms = summary_duration_ms

                        try:
                            with logfire.span("Executing Summary"):
                                ctx = WorkerContext(
                                    executor=executor,
                                    media_path=summary_media,
                                    search_providers=web_search_providers,
                                )
                                summary_response = await summary_job.execute(ctx)
                                logfire.info("Summary completed successfully.")
                        except QuotaExceededError:
                            logfire.warning("Quota exceeded during summary stage.")
                        except Exception as e:
                            summary_job.total_attempts += 1
                            logfire.error(f"Summary stage failed after retries: {e}")
                        finally:
                            await asyncio.to_thread(summary_job.save, summary_file)
                            progress.update(summary_tid, completed=1)
                    else:
                        logfire.warning("Skipping summary: max attempts reached.")
                        progress.update(summary_tid, completed=1)

                    # Rule: If summary was enabled and failed to produce a response, stop the pipeline immediately.
                    if summary_response is None:
                        logfire.error("Summary stage failed or incomplete. Halting pipeline execution.")
                        return (
                            AiSubResult.MAX_RETRIES_EXHAUSTED
                            if summary_job.total_attempts >= settings.retry.max_runs
                            else AiSubResult.INCOMPLETE
                        )

                # ------------------------------------------------------------------
                # Video Segmentation & Segment Setup
                # ------------------------------------------------------------------
                video_splits_paths = await split_video(
                    input_video=input_video_path,
                    output_dir=settings.dir.tmp,
                    split_duration_s=settings.split.max_seconds,
                    output_pattern="part_%03d",
                    duration_tolerance_ms=settings.split.duration_tolerance_ms,
                )

                semaphore = asyncio.Semaphore(8)

                async def _probe(p: Path) -> int:
                    async with semaphore:
                        return await get_video_duration_ms(p)

                durations = await asyncio.gather(*(_probe(p) for p in video_splits_paths))
                video_splits = list(zip(video_splits_paths, durations, strict=True))

                chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
                splits_to_process = video_splits[chunks_to_skip:]
                skipped_ms = sum(d for _, d in video_splits[:chunks_to_skip])
                if chunks_to_skip > 0:
                    logfire.info(f"Skipping first {chunks_to_skip} chunks ({skipped_ms}ms) due to start offset.")

                reencode_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()
                subtitles_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()
                lyrics_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()

                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=len(splits_to_process))
                    if settings.reencode.segment.enabled
                    else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles", total=len(splits_to_process))
                lyrics_tid: TaskID | None = (
                    progress.add_task("[magenta]Lyrics", total=len(splits_to_process)) if use_lyrics else None
                )

                current_offset_ms = skipped_ms
                for idx, (path, duration_ms) in enumerate(splits_to_process):
                    sub_file = settings.dir.tmp / f"{path.stem}.subtitles.{subtitles_shortcode}.yaml"
                    lyrics_file = settings.dir.tmp / f"{path.stem}.lyrics.{lyrics_shortcode}.yaml"

                    existing_lyrics = LyricsJob.load(lyrics_file, settings.ai.validation_buffer_ms)
                    existing_sub = SubtitlesJob.load(sub_file, settings.ai.validation_buffer_ms)

                    task_offset_ms = current_offset_ms
                    current_offset_ms += duration_ms

                    is_done = False
                    if existing_sub and existing_sub.response:
                        if use_lyrics and existing_sub.response.detected_musical_performances:
                            is_done = bool(existing_lyrics and existing_lyrics.response)
                        else:
                            is_done = True

                    if is_done:
                        if reencode_tid is not None:
                            progress.update(reencode_tid, advance=1)
                        progress.update(subtitles_tid, advance=1)
                        if lyrics_tid is not None:
                            progress.update(lyrics_tid, advance=1)
                        continue

                    task = SegmentTask(
                        index=idx,
                        raw_path=path,
                        duration_ms=duration_ms,
                        stem=path.stem,
                        media_path=path,
                        offset_ms=task_offset_ms,
                    )

                    should_reencode = False
                    if settings.reencode.segment.enabled:
                        if settings.reencode.segment.threshold_mb == 0:
                            should_reencode = True
                        else:
                            mb = path.stat().st_size / (1024 * 1024)
                            should_reencode = mb >= settings.reencode.segment.threshold_mb

                    if should_reencode:
                        await reencode_queue.put(task)
                    elif existing_sub and existing_sub.response:
                        task.draft_subtitles = existing_sub.response.subtitles
                        task.detected_performances = existing_sub.response.detected_musical_performances
                        task.subtitles_complete = True
                        progress.update(subtitles_tid, advance=1)
                        if use_lyrics and task.detected_performances:
                            await lyrics_queue.put(task)
                        elif lyrics_tid is not None:
                            progress.update(lyrics_tid, advance=1)
                    else:
                        await subtitles_queue.put(task)

                async def _reencode_worker() -> None:
                    while True:
                        task = await reencode_queue.get()
                        try:
                            with logfire.span("Reencode: {segment}", segment=task.stem):
                                out_file = reencode_dir / task.raw_path.with_suffix(".mp4").name
                                job = ReencodeJob(
                                    name=task.stem,
                                    input_file=task.raw_path,
                                    output_file=out_file,
                                    fps=settings.reencode.segment.fps,
                                    height=settings.reencode.segment.height,
                                    bitrate_kb=settings.reencode.segment.bitrate_kb,
                                    duration_tolerance_ms=settings.reencode.segment.duration_tolerance_ms,
                                    encoder=settings.reencode.encoder,
                                    keyframe_only=settings.reencode.segment.keyframe_only,
                                    user_context=user_context,
                                )
                                compressed_path = await job.execute()
                                task.media_path = compressed_path
                                task.duration_ms = await get_video_duration_ms(compressed_path)

                            if reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)

                            await subtitles_queue.put(task)
                        except Exception as e:
                            logfire.warning(f"Re-encoding failed for {task.stem}: {e}. Falling back to raw segment.")
                            task.media_path = task.raw_path
                            if reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)
                            await subtitles_queue.put(task)
                        finally:
                            reencode_queue.task_done()

                async def _subtitles_worker() -> None:
                    while True:
                        task = await subtitles_queue.get()
                        try:
                            with logfire.span("Subtitles: {segment}", segment=task.stem):
                                sub_file = settings.dir.tmp / f"{task.stem}.subtitles.{subtitles_shortcode}.yaml"
                                job = SubtitlesJob.load(sub_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = SubtitlesJob(
                                        name=task.stem,
                                        video_duration_ms=task.duration_ms,
                                        segment_offset_ms=task.offset_ms,
                                        global_summary=summary_response,
                                        user_context=user_context,
                                    )
                                else:
                                    job.user_context = user_context
                                    job.segment_offset_ms = task.offset_ms
                                    job.global_summary = summary_response

                                if not job.response and job.total_attempts < settings.retry.max_runs:
                                    try:
                                        ctx = WorkerContext(
                                            executor=executor,
                                            media_path=task.media_path,
                                            search_providers=[],
                                        )
                                        res = await job.execute(ctx)
                                        task.draft_subtitles = res.subtitles
                                        task.detected_performances = res.detected_musical_performances
                                        task.subtitles_complete = True

                                        srt_path = settings.dir.tmp / f"{task.stem}.subtitles-{subtitles_shortcode}.srt"
                                        await asyncio.to_thread(job.save_srt, srt_path)
                                    except QuotaExceededError:
                                        logfire.warning(f"Quota exceeded during subtitles pass for {task.stem}.")
                                    except Exception as e:
                                        job.total_attempts += 1
                                        logfire.exception(f"Subtitles job error for {task.stem}: {e}")
                                    finally:
                                        await asyncio.to_thread(job.save, sub_file)
                                elif job.response:
                                    task.draft_subtitles = job.response.subtitles
                                    task.detected_performances = job.response.detected_musical_performances
                                    task.subtitles_complete = True
                                elif job.total_attempts >= settings.retry.max_runs:
                                    logfire.warning(f"Skipping subtitles for {task.stem}: max attempts reached.")

                            progress.update(subtitles_tid, advance=1)

                            # Rule: If subtitles failed for this part, DO NOT continue processing this part.
                            if task.subtitles_complete and use_lyrics and task.detected_performances:
                                await lyrics_queue.put(task)
                            elif lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)
                        finally:
                            subtitles_queue.task_done()

                async def _lyrics_worker() -> None:
                    while True:
                        task = await lyrics_queue.get()
                        try:
                            with logfire.span("Lyrics: {segment}", segment=task.stem):
                                lyrics_file = settings.dir.tmp / f"{task.stem}.lyrics.{lyrics_shortcode}.yaml"
                                job = LyricsJob.load(lyrics_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = LyricsJob(
                                        name=task.stem,
                                        video_duration_ms=task.duration_ms,
                                        segment_offset_ms=task.offset_ms,
                                        draft_subtitles=task.draft_subtitles or [],
                                        detected_performances=task.detected_performances or [],
                                        global_summary=summary_response,
                                        user_context=user_context,
                                    )
                                else:
                                    job.user_context = user_context
                                    job.segment_offset_ms = task.offset_ms
                                    job.global_summary = summary_response
                                    if task.draft_subtitles:
                                        job.draft_subtitles = task.draft_subtitles
                                    if task.detected_performances:
                                        job.detected_performances = task.detected_performances

                                if (
                                    not job.response
                                    and job.total_attempts < settings.retry.max_runs
                                    and job.draft_subtitles
                                    and job.detected_performances
                                ):
                                    try:
                                        ctx = WorkerContext(
                                            executor=executor,
                                            media_path=task.media_path,
                                            search_providers=lyrics_search_providers,
                                        )
                                        await job.execute(ctx)

                                        srt_path = settings.dir.tmp / f"{task.stem}.lyrics-{lyrics_shortcode}.srt"
                                        await asyncio.to_thread(job.save_srt, srt_path)
                                    except QuotaExceededError:
                                        logfire.warning(f"Quota exceeded during lyrics pass for {task.stem}.")
                                    except Exception as e:
                                        job.total_attempts += 1
                                        logfire.exception(f"Lyrics job error for {task.stem}: {e}")
                                    finally:
                                        await asyncio.to_thread(job.save, lyrics_file)
                                elif job.total_attempts >= settings.retry.max_runs:
                                    logfire.warning(f"Skipping lyrics for {task.stem}: max attempts reached.")

                            if lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)
                        finally:
                            lyrics_queue.task_done()

                worker_tasks: list[asyncio.Task[Any]] = []
                try:
                    if settings.reencode.segment.enabled:
                        for _ in range(settings.concurrency.re_encode):
                            worker_tasks.append(asyncio.create_task(_reencode_worker()))
                    for _ in range(settings.concurrency.subtitles):
                        worker_tasks.append(asyncio.create_task(_subtitles_worker()))
                    if use_lyrics:
                        for _ in range(settings.concurrency.lyrics):
                            worker_tasks.append(asyncio.create_task(_lyrics_worker()))

                    if settings.reencode.segment.enabled:
                        await reencode_queue.join()
                    await subtitles_queue.join()
                    if use_lyrics:
                        await lyrics_queue.join()
                finally:
                    for wt in worker_tasks:
                        wt.cancel()
                    if worker_tasks:
                        await asyncio.gather(*worker_tasks, return_exceptions=True)

        return stitch_subtitles(
            video_splits=video_splits,
            settings=settings,
            lyrics_shortcode=lyrics_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            full_shortcode=full_shortcode,
            summary_available=summary_response is not None,
        )


def main() -> None:
    """CLI entrypoint for the AI Sub subtitle generation pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
