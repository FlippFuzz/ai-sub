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
from ai_sub.stitching import stitch_subtitles as stitch_subtitles_engine
from ai_sub.telemetry import setup_logging
from ai_sub.video import (
    get_video_duration_ms,
    init_ffmpeg_engine,
    reencode_video,
    set_ffmpeg_concurrency,
    split_video,
)


class AiSubResult(IntEnum):
    """Application exit status codes."""

    COMPLETED = 0
    IN_PROGRESS = -1
    RETRIES_EXCEEDED = -2


class SegmentTask:
    """Internal task tracking container for a single video segment."""

    index: int
    raw_path: Path
    duration_ms: int
    stem: str
    media_path: Path
    offset_ms: int
    draft_subtitles: list[Subtitles] | None
    detected_performances: list[DetectedMusicalPerformance] | None
    subtitles_complete: bool

    def __init__(
        self,
        index: int,
        raw_path: Path,
        duration_ms: int,
        stem: str,
        media_path: Path,
        offset_ms: int = 0,
        draft_subtitles: list[Subtitles] | None = None,
        detected_performances: list[DetectedMusicalPerformance] | None = None,
        subtitles_complete: bool = False,
    ) -> None:
        """Initializes a segment task tracker.

        Args:
            index: Segment sequence index.
            raw_path: Path to the raw video chunk.
            duration_ms: Segment duration in milliseconds.
            stem: File stem identifier.
            media_path: Active media path (raw or re-encoded).
            offset_ms: Start time offset in milliseconds. Defaults to 0.
            draft_subtitles: Initial subtitles generated in subtitles pass. Defaults to None.
            detected_performances: Identified musical singing events. Defaults to None.
            subtitles_complete: Whether subtitle transcription finished. Defaults to False.
        """
        self.index = index
        self.raw_path = raw_path
        self.duration_ms = duration_ms
        self.stem = stem
        self.media_path = media_path
        self.offset_ms = offset_ms
        self.draft_subtitles = draft_subtitles
        self.detected_performances = detected_performances
        self.subtitles_complete = subtitles_complete


def stitch_subtitles(
    video_splits: list[tuple[Path, int]],
    settings: Settings,
    lyrics_shortcode: str,
    subtitles_shortcode: str,
    full_shortcode: str,
    summary_available: bool,
) -> AiSubResult:
    """Assembles segment subtitles into final merged SRT files using the unified stitching engine.

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
        chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
        offset_ms = sum(d for _, d in video_splits[:chunks_to_skip])
        splits_to_process = video_splits[chunks_to_skip:]

        settings_dump = settings.model_dump(
            mode="json",
            exclude={
                "input_video_file": True,
                "context_file": True,
                "dir": True,
                "ai": {
                    "key": True,
                    "base_url": True,
                    "search": {"key": True, "tinyfish_key": True, "langsearch_key": True, "ollama_key": True},
                },
            },
        )
        settings_dump["context"] = settings.get_resolved_user_context()

        input_video_path = cast(Path, settings.input_video_file)
        out_srt = settings.dir.out / f"{input_video_path.stem}.{full_shortcode}.srt"
        raw_out_srt = settings.dir.out / f"{input_video_path.stem}.raw-{full_shortcode}.srt"

        segments = [(p.stem, d) for p, d in splits_to_process]
        result = stitch_subtitles_engine(
            segments=segments,
            state_dir=settings.dir.tmp,
            output_primary_path=out_srt,
            output_raw_path=raw_out_srt,
            subtitles_shortcode=subtitles_shortcode,
            lyrics_shortcode=lyrics_shortcode,
            validation_buffer_ms=settings.ai.validation_buffer_ms,
            settings_dump=settings_dump,
            max_runs=settings.retry.max_runs,
            summary_available=summary_available,
            initial_offset_ms=offset_ms,
            use_lyrics=settings.concurrency.lyrics > 0,
        )

        logfire.info(f"Subtitles written to {out_srt.name}")
        if result.lyrics_refined and raw_out_srt.is_file():
            logfire.info(f"Raw subtitles written to {raw_out_srt.name}")

        if result.any_pending:
            return AiSubResult.IN_PROGRESS
        if result.max_retries_exceeded:
            return AiSubResult.RETRIES_EXCEEDED
        return AiSubResult.COMPLETED


async def run_pipeline(settings: Settings) -> AiSubResult:
    """Executes the subtitle generation pipeline.

    Args:
        settings: Configured application settings.

    Returns:
        AiSubResult: Overall status code.

    Raises:
        RuntimeError: If writing state files to disk fails fatally.
    """
    console = Console(stderr=True)
    setup_logging(settings, console=console)
    input_video_path = cast(Path, settings.input_video_file)
    user_context = settings.get_resolved_user_context()

    set_ffmpeg_concurrency(settings.concurrency.re_encode)

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
                        raw_mb = input_video_path.stat().st_size / (1024 * 1024)
                        sum_cfg = settings.reencode.summary
                        should_reencode = sum_cfg.enabled and (
                            sum_cfg.threshold_mb == 0 or raw_mb >= sum_cfg.threshold_mb
                        )

                        if should_reencode:
                            proxy_output = settings.dir.tmp / "summary_proxy.mp4"
                            await reencode_video(
                                input_path=input_video_path,
                                output_path=proxy_output,
                                fps=sum_cfg.fps,
                                height=sum_cfg.height,
                                bitrate_kb=sum_cfg.bitrate_kb,
                                encoder=settings.reencode.encoder or "libx264",
                                duration_tolerance_ms=sum_cfg.duration_tolerance_ms,
                                keyframe_only=sum_cfg.keyframe_only,
                            )
                            summary_media = proxy_output

                        summary_job.video_duration_ms = await get_video_duration_ms(summary_media)
                        try:
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
                            logfire.error(f"Summary stage failed: {e}")
                        finally:
                            try:
                                await asyncio.to_thread(summary_job.save, summary_file)
                            except Exception as save_err:
                                summary_response = None
                                logfire.error(f"Failed to save summary YAML: {save_err}")
                                raise RuntimeError(
                                    f"Fatal storage error saving {summary_file}: {save_err}"
                                ) from save_err
                            progress.update(summary_tid, completed=1)
                    else:
                        progress.update(summary_tid, completed=1)

                    if summary_response is None:
                        logfire.error("Summary stage failed. Halting pipeline execution.")
                        return (
                            AiSubResult.RETRIES_EXCEEDED
                            if summary_job.total_attempts >= settings.retry.max_runs
                            else AiSubResult.IN_PROGRESS
                        )

                video_splits_paths = await split_video(
                    input_video=input_video_path,
                    output_dir=settings.dir.tmp,
                    split_duration_s=settings.split.max_seconds,
                    output_pattern="part_%03d",
                    duration_tolerance_ms=settings.split.duration_tolerance_ms,
                )

                sem = asyncio.Semaphore(8)

                async def _probe(p: Path) -> int:
                    async with sem:
                        return await get_video_duration_ms(p)

                durations = await asyncio.gather(*(_probe(p) for p in video_splits_paths))
                video_splits = list(zip(video_splits_paths, durations, strict=True))

                chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
                splits_to_process = video_splits[chunks_to_skip:]
                skipped_ms = sum(d for _, d in video_splits[:chunks_to_skip])

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

                    seg_cfg = settings.reencode.segment
                    raw_mb = path.stat().st_size / (1024 * 1024)
                    should_reencode = seg_cfg.enabled and (seg_cfg.threshold_mb == 0 or raw_mb >= seg_cfg.threshold_mb)

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
                        except Exception as e:
                            logfire.warning(f"Re-encoding failed for {task.stem}: {e}. Using raw segment.")
                            task.media_path = task.raw_path
                        finally:
                            if reencode_tid is not None:
                                progress.update(reencode_tid, advance=1)
                            await subtitles_queue.put(task)
                            reencode_queue.task_done()

                async def _subtitles_worker() -> None:
                    while True:
                        task = await subtitles_queue.get()
                        advanced_sub = False
                        advanced_lyr = False
                        try:
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
                                        executor=executor, media_path=task.media_path, search_providers=[]
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
                                    try:
                                        await asyncio.to_thread(job.save, sub_file)
                                    except Exception as save_err:
                                        task.subtitles_complete = False
                                        task.draft_subtitles = None
                                        task.detected_performances = None
                                        logfire.error(f"Failed to save subtitles YAML for {task.stem}: {save_err}")
                                        raise RuntimeError(
                                            f"Fatal storage error saving {sub_file}: {save_err}"
                                        ) from save_err
                            elif job.response:
                                task.draft_subtitles = job.response.subtitles
                                task.detected_performances = job.response.detected_musical_performances
                                task.subtitles_complete = True

                            progress.update(subtitles_tid, advance=1)
                            advanced_sub = True

                            if task.subtitles_complete and use_lyrics and task.detected_performances:
                                await lyrics_queue.put(task)
                            elif lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)
                                advanced_lyr = True
                        except Exception as worker_err:
                            logfire.exception(f"Unhandled error in subtitles worker for {task.stem}: {worker_err}")
                            if not advanced_sub:
                                progress.update(subtitles_tid, advance=1)
                            if lyrics_tid is not None and not advanced_lyr:
                                progress.update(lyrics_tid, advance=1)
                        finally:
                            subtitles_queue.task_done()

                async def _lyrics_worker() -> None:
                    while True:
                        task = await lyrics_queue.get()
                        advanced_lyr = False
                        try:
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
                                    try:
                                        await asyncio.to_thread(job.save, lyrics_file)
                                    except Exception as save_err:
                                        logfire.error(f"Failed to save lyrics YAML for {task.stem}: {save_err}")
                                        raise RuntimeError(
                                            f"Fatal storage error saving {lyrics_file}: {save_err}"
                                        ) from save_err

                            if lyrics_tid is not None:
                                progress.update(lyrics_tid, advance=1)
                                advanced_lyr = True
                        except Exception as worker_err:
                            logfire.exception(f"Unhandled error in lyrics worker for {task.stem}: {worker_err}")
                            if lyrics_tid is not None and not advanced_lyr:
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
    """CLI entrypoint for the AI Sub standalone subtitle generation pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
