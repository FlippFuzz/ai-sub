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
    LyricsSceneAiResponse,
    LyricsSceneJob,
    ReencodeJob,
    SubtitleAiResponse,
    SubtitleJob,
    SummaryAiResponse,
    SummaryJob,
    get_lyrics_scenes_prompt,
    get_subtitle_prompt,
    get_summary_prompt,
)
from ai_sub.search import create_search_provider
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_summary_shortcode,
)
from ai_sub.telemetry import get_app_version, setup_logging
from ai_sub.video import (
    get_video_duration_ms,
    get_working_encoder,
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
    lyrics_response: LyricsSceneAiResponse | None = None
    subtitles_complete: bool = False


def stitch_subtitles(
    video_splits: list[tuple[Path, int]],
    settings: Settings,
    summary_shortcode: str,
    lyrics_shortcode: str,
    subtitles_shortcode: str,
    summary_available: bool,
) -> AiSubResult:
    """Assembles individual segment subtitle files into the final merged SRT file.

    Args:
        video_splits: Ordered list of segment paths and durations in milliseconds.
        settings: Application settings.
        summary_shortcode: Model shortcode used for summary state files.
        lyrics_shortcode: Model shortcode used for lyrics state files.
        subtitles_shortcode: Model shortcode used for subtitle state files.
        summary_available: True if Phase 1 global summary succeeded.

    Returns:
        AiSubResult: Pipeline result status code.
    """
    with logfire.span("Stitching final SRT file"):
        all_subs = SSAFile()
        chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
        offset_ms = sum(d for _, d in video_splits[:chunks_to_skip])

        complete = True
        max_retries_exceeded = False
        any_pending = False

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            job_path = settings.dir.tmp / f"{video_path.stem}.subtitles.{subtitles_shortcode}.yaml"
            sub_job = SubtitleJob.load(job_path, settings.ai.validation_buffer_ms)

            if sub_job and sub_job.response:
                segment_subs = sub_job.response.get_ssafile()
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
                sub_attempts = sub_job.total_attempts if sub_job else 0

                if sub_attempts >= settings.retry.max_runs:
                    max_retries_exceeded = True
                else:
                    any_pending = True

            offset_ms += video_duration_ms

        settings_dump = settings.model_dump(
            mode="json",
            exclude={
                "input_video_file": True,
                "dir": True,
                "ai": {"key": True, "base_url": True, "search": {"key": True}},
            },
        )
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
        out_srt = settings.dir.out / f"{input_video_path.stem}.{subtitles_shortcode}.srt"
        all_subs.save(str(out_srt))
        logfire.info(f"Subtitles written to {out_srt.name}")

        if any_pending:
            return AiSubResult.INCOMPLETE
        if max_retries_exceeded:
            return AiSubResult.MAX_RETRIES_EXHAUSTED
        return AiSubResult.COMPLETE


async def run_pipeline(settings: Settings) -> AiSubResult:
    """Executes the 3-phase video subtitle generation pipeline.

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

        summary_reencode_cfg = settings.reencode.resolve_summary()
        segment_reencode_cfg = settings.reencode.resolve_segment()

        if (summary_reencode_cfg.enabled or segment_reencode_cfg.enabled) and not settings.reencode.encoder:
            settings.reencode.encoder = await get_working_encoder()
            summary_reencode_cfg.encoder = settings.reencode.encoder
            segment_reencode_cfg.encoder = settings.reencode.encoder
            logfire.info(f"Using encoder: {settings.reencode.encoder}")

        use_summary = settings.concurrency.summary > 0
        use_lyrics = settings.concurrency.lyrics > 0
        summary_shortcode = generate_summary_shortcode(settings.ai.model_summary)
        lyrics_shortcode = generate_lyrics_shortcode(settings.ai.model_lyrics)
        subtitles_shortcode = generate_full_shortcode(settings.ai.model_subtitles)

        reencode_dir = settings.dir.tmp / "reencoded"
        if segment_reencode_cfg.enabled:
            reencode_dir.mkdir(exist_ok=True)

        async with AsyncExitStack() as stack:
            search_tools: list[Any] = []
            if use_summary or use_lyrics:
                search_provider, search_tools = create_search_provider(settings.ai.search)
                if search_provider is not None and hasattr(search_provider, "__aenter__"):
                    await stack.enter_async_context(search_provider)

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
                # Phase 1: Whole-Video Summary & Knowledge Base (Global Barrier)
                # ------------------------------------------------------------------
                if use_summary:
                    summary_tid = progress.add_task("[yellow]Global Summary", total=1)
                    summary_file = settings.dir.tmp / f"summary.{summary_shortcode}.yaml"
                    existing_summary = SummaryJob.load(summary_file, settings.ai.validation_buffer_ms)

                    if existing_summary and existing_summary.response:
                        summary_response = existing_summary.response
                        progress.update(summary_tid, completed=1)
                        logfire.info("Reusing existing valid global summary.")
                    else:
                        summary_job = SummaryJob(
                            name="summary",
                            video_duration_ms=await get_video_duration_ms(input_video_path),
                        )
                        summary_media = input_video_path
                        should_reencode_summary = False

                        if summary_reencode_cfg.enabled:
                            if summary_reencode_cfg.threshold_mb == 0:
                                should_reencode_summary = True
                            else:
                                mb = input_video_path.stat().st_size / (1024 * 1024)
                                should_reencode_summary = mb >= summary_reencode_cfg.threshold_mb

                        if should_reencode_summary:
                            proxy_output = settings.dir.tmp / f"summary_proxy{input_video_path.suffix}"
                            with logfire.span("Compressing whole-video summary proxy"):
                                await reencode_video(
                                    input_path=input_video_path,
                                    output_path=proxy_output,
                                    fps=summary_reencode_cfg.fps,
                                    height=summary_reencode_cfg.height,
                                    bitrate_kb=summary_reencode_cfg.bitrate_kb,
                                    encoder=summary_reencode_cfg.encoder or "libx264",
                                    duration_tolerance_ms=summary_reencode_cfg.duration_tolerance_ms,
                                )
                                summary_media = proxy_output

                        summary_duration_ms = await get_video_duration_ms(summary_media)
                        summary_job.video_duration_ms = summary_duration_ms

                        try:
                            with logfire.span("Executing Phase 1 Summary"):
                                prompt = get_summary_prompt(user_context=user_context)
                                res = await executor.run_structured(
                                    prompt=prompt,
                                    media_path=summary_media,
                                    video_duration_ms=summary_duration_ms,
                                    response_type=SummaryAiResponse,
                                    model_name=settings.ai.model_summary,
                                    service_tier=settings.ai.service_tier_summary,
                                    tools=search_tools,
                                    retry_policy=settings.retry,
                                    agentic_video=True,
                                )
                                summary_job.response = res
                                summary_job.total_attempts += 1
                                summary_response = res
                                await asyncio.to_thread(summary_job.save, summary_file)
                                logfire.info("Phase 1 global summary completed successfully.")
                        except QuotaExceededError:
                            logfire.warning("Quota exceeded during Phase 1 summary pass.")
                        except Exception as e:
                            summary_job.total_attempts += 1
                            logfire.warning(f"Phase 1 summary failed: {e}. Proceeding without global summary.")
                        finally:
                            progress.update(summary_tid, completed=1)

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
                lyrics_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()
                subtitles_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()

                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=len(splits_to_process))
                    if segment_reencode_cfg.enabled
                    else None
                )
                lyrics_tid: TaskID | None = (
                    progress.add_task("[magenta]Lyrics & Scenes", total=len(splits_to_process)) if use_lyrics else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles", total=len(splits_to_process))

                current_offset_ms = skipped_ms
                for idx, (path, duration_ms) in enumerate(splits_to_process):
                    sub_file = settings.dir.tmp / f"{path.stem}.subtitles.{subtitles_shortcode}.yaml"
                    existing_sub = SubtitleJob.load(sub_file, settings.ai.validation_buffer_ms)

                    task_offset_ms = current_offset_ms
                    current_offset_ms += duration_ms

                    if existing_sub and existing_sub.response:
                        if reencode_tid is not None:
                            progress.update(reencode_tid, advance=1)
                        if lyrics_tid is not None:
                            progress.update(lyrics_tid, advance=1)
                        progress.update(subtitles_tid, advance=1)
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
                    if segment_reencode_cfg.enabled:
                        if segment_reencode_cfg.threshold_mb == 0:
                            should_reencode = True
                        else:
                            mb = path.stat().st_size / (1024 * 1024)
                            should_reencode = mb >= segment_reencode_cfg.threshold_mb

                    if should_reencode:
                        await reencode_queue.put(task)
                    elif use_lyrics:
                        lyr_file = settings.dir.tmp / f"{path.stem}.lyrics.{lyrics_shortcode}.yaml"
                        existing_lyr = LyricsSceneJob.load(lyr_file, settings.ai.validation_buffer_ms)
                        if existing_lyr and existing_lyr.response:
                            task.lyrics_response = existing_lyr.response
                            if lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)
                            await subtitles_queue.put(task)
                        else:
                            await lyrics_queue.put(task)
                    else:
                        await subtitles_queue.put(task)

                async def _reencode_worker() -> None:
                    while True:
                        task = await reencode_queue.get()
                        try:
                            with logfire.span("Reencode: {segment}", segment=task.stem):
                                out_file = reencode_dir / task.raw_path.with_suffix(".mov").name
                                job = ReencodeJob(
                                    name=task.stem,
                                    input_file=task.raw_path,
                                    output_file=out_file,
                                    fps=segment_reencode_cfg.fps,
                                    height=segment_reencode_cfg.height,
                                    bitrate_kb=segment_reencode_cfg.bitrate_kb,
                                    duration_tolerance_ms=segment_reencode_cfg.duration_tolerance_ms,
                                    encoder=segment_reencode_cfg.encoder,
                                )
                                compressed_path = await job.execute()
                                task.media_path = compressed_path
                                task.duration_ms = await get_video_duration_ms(compressed_path)

                            if reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)

                            if use_lyrics:
                                lyr_file = settings.dir.tmp / f"{task.stem}.lyrics.{lyrics_shortcode}.yaml"
                                existing_lyr = LyricsSceneJob.load(lyr_file, settings.ai.validation_buffer_ms)
                                if existing_lyr and existing_lyr.response:
                                    task.lyrics_response = existing_lyr.response
                                    if lyrics_tid is not None:
                                        progress.update(lyrics_tid, advance=1)
                                    await subtitles_queue.put(task)
                                else:
                                    await lyrics_queue.put(task)
                            else:
                                await subtitles_queue.put(task)
                        except Exception as e:
                            logfire.warning(f"Re-encoding failed for {task.stem}: {e}. Falling back to raw segment.")
                            task.media_path = task.raw_path
                            if reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)
                            if use_lyrics:
                                lyr_file = settings.dir.tmp / f"{task.stem}.lyrics.{lyrics_shortcode}.yaml"
                                existing_lyr = LyricsSceneJob.load(lyr_file, settings.ai.validation_buffer_ms)
                                if existing_lyr and existing_lyr.response:
                                    task.lyrics_response = existing_lyr.response
                                    if lyrics_tid is not None:
                                        progress.update(lyrics_tid, advance=1)
                                    await subtitles_queue.put(task)
                                else:
                                    await lyrics_queue.put(task)
                            else:
                                await subtitles_queue.put(task)
                        finally:
                            reencode_queue.task_done()

                async def _lyrics_worker() -> None:
                    while True:
                        task = await lyrics_queue.get()
                        try:
                            with logfire.span("Lyrics & Scenes: {segment}", segment=task.stem):
                                lyr_file = settings.dir.tmp / f"{task.stem}.lyrics.{lyrics_shortcode}.yaml"
                                job = LyricsSceneJob.load(lyr_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = LyricsSceneJob(name=task.stem, video_duration_ms=task.duration_ms)

                                if job.response:
                                    task.lyrics_response = job.response
                                elif job.total_attempts < settings.retry.max_runs:
                                    try:
                                        prompt = get_lyrics_scenes_prompt(
                                            user_context=user_context,
                                            global_summary=summary_response,
                                            segment_offset_ms=task.offset_ms,
                                            segment_duration_ms=task.duration_ms,
                                        )
                                        res = await executor.run_structured(
                                            prompt=prompt,
                                            media_path=task.media_path,
                                            video_duration_ms=task.duration_ms,
                                            response_type=LyricsSceneAiResponse,
                                            model_name=settings.ai.model_lyrics,
                                            service_tier=settings.ai.service_tier_lyrics,
                                            tools=search_tools,
                                            retry_policy=settings.retry,
                                            agentic_video=False,
                                        )
                                        job.response = res
                                        task.lyrics_response = res
                                        job.total_attempts += 1
                                    except QuotaExceededError:
                                        logfire.warning(f"Quota exceeded during lyrics pass for {task.stem}.")
                                    except Exception as e:
                                        job.total_attempts += 1
                                        logfire.exception(f"Lyrics job error for {task.stem}: {e}")
                                    finally:
                                        await asyncio.to_thread(job.save, lyr_file)
                                else:
                                    logfire.warning(f"Skipping lyrics for {task.stem}: max attempts reached.")

                            if lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)

                            await subtitles_queue.put(task)
                        finally:
                            lyrics_queue.task_done()

                async def _subtitles_worker() -> None:
                    while True:
                        task = await subtitles_queue.get()
                        try:
                            with logfire.span("Subtitles: {segment}", segment=task.stem):
                                sub_file = settings.dir.tmp / f"{task.stem}.subtitles.{subtitles_shortcode}.yaml"
                                job = SubtitleJob.load(sub_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = SubtitleJob(name=task.stem, video_duration_ms=task.duration_ms)

                                if not job.response and job.total_attempts < settings.retry.max_runs:
                                    try:
                                        prompt = get_subtitle_prompt(
                                            user_context=user_context,
                                            global_summary=summary_response,
                                            scene_response=task.lyrics_response,
                                            segment_offset_ms=task.offset_ms,
                                            segment_duration_ms=task.duration_ms,
                                        )
                                        res = await executor.run_structured(
                                            prompt=prompt,
                                            media_path=task.media_path,
                                            video_duration_ms=task.duration_ms,
                                            response_type=SubtitleAiResponse,
                                            model_name=settings.ai.model_subtitles,
                                            service_tier=settings.ai.service_tier_subtitles,
                                            tools=None,
                                            retry_policy=settings.retry,
                                            agentic_video=False,
                                        )
                                        job.response = res
                                        job.total_attempts += 1
                                        srt_path = settings.dir.tmp / f"{task.stem}.{subtitles_shortcode}.srt"
                                        await asyncio.to_thread(job.save_srt, srt_path)
                                    except QuotaExceededError:
                                        logfire.warning(f"Quota exceeded during subtitles pass for {task.stem}.")
                                    except Exception as e:
                                        job.total_attempts += 1
                                        logfire.exception(f"Subtitles job error for {task.stem}: {e}")
                                    finally:
                                        await asyncio.to_thread(job.save, sub_file)
                                elif job.total_attempts >= settings.retry.max_runs:
                                    logfire.warning(f"Skipping subtitles for {task.stem}: max attempts reached.")

                            progress.update(subtitles_tid, advance=1)
                        finally:
                            subtitles_queue.task_done()

                worker_tasks: list[asyncio.Task[Any]] = []
                try:
                    if segment_reencode_cfg.enabled:
                        for _ in range(settings.concurrency.re_encode):
                            worker_tasks.append(asyncio.create_task(_reencode_worker()))
                    if use_lyrics:
                        for _ in range(settings.concurrency.lyrics):
                            worker_tasks.append(asyncio.create_task(_lyrics_worker()))
                    for _ in range(settings.concurrency.subtitles):
                        worker_tasks.append(asyncio.create_task(_subtitles_worker()))

                    if segment_reencode_cfg.enabled:
                        await reencode_queue.join()
                    if use_lyrics:
                        await lyrics_queue.join()
                    await subtitles_queue.join()
                finally:
                    for wt in worker_tasks:
                        wt.cancel()
                    if worker_tasks:
                        await asyncio.gather(*worker_tasks, return_exceptions=True)

        return stitch_subtitles(
            video_splits=video_splits,
            settings=settings,
            summary_shortcode=summary_shortcode,
            lyrics_shortcode=lyrics_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            summary_available=summary_response is not None,
        )


def main() -> None:
    """CLI entrypoint for the AI Sub subtitle generation pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
