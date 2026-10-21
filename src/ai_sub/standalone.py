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
    MUSIC_REFINE_PROMPT_VERSION,
    SUBTITLES_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    DetectedMusicalPerformance,
    MusicRefineAiResponse,
    MusicRefineJob,
    ReencodeJob,
    SubtitleAiResponse,
    SubtitleJob,
    Subtitles,
    SummaryAiResponse,
    SummaryJob,
    get_music_refine_prompt,
    get_subtitle_prompt,
    get_summary_prompt,
)
from ai_sub.search import LrclibSearchProvider, create_search_provider
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_music_refine_shortcode,
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
    draft_subtitles: list[Subtitles] | None = None
    detected_performances: list[DetectedMusicalPerformance] | None = None
    subtitles_complete: bool = False


async def prefetch_lyrics(
    performances: list[DetectedMusicalPerformance],
    lrclib_provider: LrclibSearchProvider,
    search_provider: Any | None,
) -> str:
    """Pre-fetches lyrics from LRCLIB and web search in Python to avoid burning LLM search turns.

    Args:
        performances: List of detected musical performances.
        lrclib_provider: LRCLIB search provider instance.
        search_provider: Optional web search provider instance.

    Returns:
        str: Formatted initial search results text block.
    """
    results_parts: list[str] = []

    for perf in performances:
        if not perf.title or not perf.title.strip():
            continue

        clean_title = perf.title.strip()
        clean_artist = perf.artist.strip() if perf.artist else None
        logfire.info(f"Pre-fetching lyrics for: '{clean_title}' by '{clean_artist or 'Unknown'}'")

        try:
            lrclib_results = await lrclib_provider.search_lrclib_lyrics_database(
                track_title=clean_title,
                artist_name=clean_artist,
            )
            if lrclib_results:
                best = lrclib_results[0]
                results_parts.append(
                    f"Official Lyrics for '{best.get('track_name')}' by {best.get('artist_name')}:\n"
                    f"{best.get('plain_lyrics')}"
                )
        except Exception as e:
            logfire.warning(f"LRCLIB pre-fetch error for '{clean_title}': {e}")

        if not results_parts and search_provider and hasattr(search_provider, "search_single"):
            try:
                query = f"{clean_title} {clean_artist or ''} lyrics 歌詞"
                search_res = await search_provider.search_single(query.strip())
                if search_res:
                    snippets = "\n---\n".join(f"[{r.title}]: {r.content}" for r in search_res[:2])
                    results_parts.append(f"Web Search Results for '{clean_title}':\n{snippets}")
            except Exception as e:
                logfire.debug(f"Web search pre-fetch error for '{clean_title}': {e}")

    return "\n\n".join(results_parts)


def stitch_subtitles(
    video_splits: list[tuple[Path, int]],
    settings: Settings,
    summary_shortcode: str,
    music_refine_shortcode: str,
    subtitles_shortcode: str,
    summary_available: bool,
) -> AiSubResult:
    """Assembles segment subtitles (refined or draft) into the final merged SRT file.

    Args:
        video_splits: Ordered list of segment paths and durations in milliseconds.
        settings: Application settings.
        summary_shortcode: Model shortcode used for summary state files.
        music_refine_shortcode: Model shortcode used for refinement state files.
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
            # Prefer refined job if music was detected and refined; otherwise use draft subtitle job
            refine_job_path = settings.dir.tmp / f"{video_path.stem}.refine.{music_refine_shortcode}.yaml"
            sub_job_path = settings.dir.tmp / f"{video_path.stem}.subtitles.{subtitles_shortcode}.yaml"

            refine_job = MusicRefineJob.load(refine_job_path, settings.ai.validation_buffer_ms)
            sub_job = SubtitleJob.load(sub_job_path, settings.ai.validation_buffer_ms)

            active_response = (
                refine_job.response
                if (refine_job and refine_job.response)
                else (sub_job.response if (sub_job and sub_job.response) else None)
            )

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
                    refine_job.total_attempts if refine_job else 0,
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
            "music_refine_prompt_version": MUSIC_REFINE_PROMPT_VERSION,
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
    """Executes the optimized 2-pass video subtitle generation pipeline.

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

        if (settings.reencode.summary.enabled or settings.reencode.segment.enabled) and not settings.reencode.encoder:
            settings.reencode.encoder = await get_working_encoder()
            logfire.info(f"Using encoder: {settings.reencode.encoder}")

        use_summary = settings.concurrency.summary > 0
        use_refine = settings.concurrency.music_refine > 0
        summary_shortcode = generate_summary_shortcode(settings.ai.model_summary)
        music_refine_shortcode = generate_music_refine_shortcode(settings.ai.model_music_refine)
        subtitles_shortcode = generate_full_shortcode(settings.ai.model_subtitles)

        reencode_dir = settings.dir.tmp / "reencoded"
        if settings.reencode.segment.enabled:
            reencode_dir.mkdir(exist_ok=True)

        async with AsyncExitStack() as stack:
            base_search_tools: list[Any] = []
            search_provider: Any | None = None
            if use_summary or use_refine:
                search_provider, base_search_tools = create_search_provider(settings.ai.search)
                if search_provider is not None and hasattr(search_provider, "__aenter__"):
                    await stack.enter_async_context(search_provider)

            lrclib_provider = LrclibSearchProvider(settings.ai.search)
            await stack.enter_async_context(lrclib_provider)

            refine_search_tools: list[Any] = list(base_search_tools)
            if use_refine:
                refine_search_tools.extend(lrclib_provider.get_tools())

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
                # Phase 1: Macro Summary & Chapter Indexing
                # ------------------------------------------------------------------
                if use_summary:
                    summary_tid = progress.add_task("[yellow]Global Summary", total=1)
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
                            proxy_output = settings.dir.tmp / f"summary_proxy{input_video_path.suffix}"
                            with logfire.span("Compressing whole-video summary proxy"):
                                await reencode_video(
                                    input_path=input_video_path,
                                    output_path=proxy_output,
                                    fps=settings.reencode.summary.fps,
                                    height=settings.reencode.summary.height,
                                    bitrate_kb=settings.reencode.summary.bitrate_kb,
                                    encoder=settings.reencode.encoder or "libx264",
                                    duration_tolerance_ms=settings.reencode.summary.duration_tolerance_ms,
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
                                    tools=base_search_tools,
                                    retry_policy=settings.retry,
                                    agentic_video=True,
                                )
                                summary_job.response = res
                                summary_job.total_attempts += 1
                                summary_response = res
                                logfire.info("Phase 1 global summary completed successfully.")
                        except QuotaExceededError:
                            logfire.warning("Quota exceeded during Phase 1 summary pass.")
                        except Exception as e:
                            summary_job.total_attempts += 1
                            logfire.warning(f"Phase 1 summary failed: {e}. Proceeding without global summary.")
                        finally:
                            await asyncio.to_thread(summary_job.save, summary_file)
                            progress.update(summary_tid, completed=1)
                    else:
                        logfire.warning("Skipping Phase 1 summary: max attempts reached.")
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
                subtitles_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()
                refine_queue: asyncio.Queue[SegmentTask] = asyncio.Queue()

                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=len(splits_to_process))
                    if settings.reencode.segment.enabled
                    else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles (Pass A)", total=len(splits_to_process))
                refine_tid: TaskID | None = (
                    progress.add_task("[magenta]Music Refine (Pass B)", total=len(splits_to_process))
                    if use_refine
                    else None
                )

                current_offset_ms = skipped_ms
                for idx, (path, duration_ms) in enumerate(splits_to_process):
                    sub_file = settings.dir.tmp / f"{path.stem}.subtitles.{subtitles_shortcode}.yaml"
                    refine_file = settings.dir.tmp / f"{path.stem}.refine.{music_refine_shortcode}.yaml"

                    existing_refine = MusicRefineJob.load(refine_file, settings.ai.validation_buffer_ms)
                    existing_sub = SubtitleJob.load(sub_file, settings.ai.validation_buffer_ms)

                    task_offset_ms = current_offset_ms
                    current_offset_ms += duration_ms

                    # Check if already completed (either refined or draft when no music occurred)
                    is_done = False
                    if existing_refine and existing_refine.response:
                        is_done = True
                    elif (
                        existing_sub
                        and existing_sub.response
                        and not existing_sub.response.detected_musical_performances
                    ):  # noqa: E501
                        is_done = True

                    if is_done:
                        if reencode_tid is not None:
                            progress.update(reencode_tid, advance=1)
                        progress.update(subtitles_tid, advance=1)
                        if refine_tid is not None:
                            progress.update(refine_tid, advance=1)
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
                        # Pass A was already done, but needs Pass B
                        task.draft_subtitles = existing_sub.response.subtitles
                        task.detected_performances = existing_sub.response.detected_musical_performances
                        progress.update(subtitles_tid, advance=1)
                        if use_refine and task.detected_performances:
                            await refine_queue.put(task)
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
                                    fps=settings.reencode.segment.fps,
                                    height=settings.reencode.segment.height,
                                    bitrate_kb=settings.reencode.segment.bitrate_kb,
                                    duration_tolerance_ms=settings.reencode.segment.duration_tolerance_ms,
                                    encoder=settings.reencode.encoder,
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
                            with logfire.span("Subtitles (Pass A): {segment}", segment=task.stem):
                                sub_file = settings.dir.tmp / f"{task.stem}.subtitles.{subtitles_shortcode}.yaml"
                                job = SubtitleJob.load(sub_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = SubtitleJob(
                                        name=task.stem,
                                        video_duration_ms=task.duration_ms,
                                        user_context=user_context,
                                    )
                                else:
                                    job.user_context = user_context

                                if not job.response and job.total_attempts < settings.retry.max_runs:
                                    try:
                                        prompt = get_subtitle_prompt(
                                            user_context=user_context,
                                            global_summary=summary_response,
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
                                        task.draft_subtitles = res.subtitles
                                        task.detected_performances = res.detected_musical_performances

                                        # Save draft SRT
                                        srt_path = settings.dir.tmp / f"{task.stem}.{subtitles_shortcode}.srt"
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
                                elif job.total_attempts >= settings.retry.max_runs:
                                    logfire.warning(f"Skipping subtitles for {task.stem}: max attempts reached.")

                            progress.update(subtitles_tid, advance=1)

                            # Route to Pass B only if music was detected and refinement is enabled
                            if use_refine and task.detected_performances:
                                await refine_queue.put(task)
                            elif refine_tid is not None:
                                progress.update(refine_tid, advance=1)
                        finally:
                            subtitles_queue.task_done()

                async def _refine_worker() -> None:
                    while True:
                        task = await refine_queue.get()
                        try:
                            with logfire.span("Music Refine (Pass B): {segment}", segment=task.stem):
                                refine_file = settings.dir.tmp / f"{task.stem}.refine.{music_refine_shortcode}.yaml"
                                job = MusicRefineJob.load(refine_file, settings.ai.validation_buffer_ms)
                                if not job:
                                    job = MusicRefineJob(
                                        name=task.stem,
                                        video_duration_ms=task.duration_ms,
                                        user_context=user_context,
                                    )
                                else:
                                    job.user_context = user_context

                                if (
                                    not job.response
                                    and job.total_attempts < settings.retry.max_runs
                                    and task.draft_subtitles
                                    and task.detected_performances
                                ):
                                    try:
                                        # Pre-fetch lyrics in Python before LLM call to save search turns
                                        prefetched = await prefetch_lyrics(
                                            performances=task.detected_performances,
                                            lrclib_provider=lrclib_provider,
                                            search_provider=search_provider,
                                        )

                                        prompt = get_music_refine_prompt(
                                            draft_subtitles=task.draft_subtitles,
                                            detected_performances=task.detected_performances,
                                            initial_search_results=prefetched,
                                            user_context=user_context,
                                            global_summary=summary_response,
                                            segment_offset_ms=task.offset_ms,
                                            segment_duration_ms=task.duration_ms,
                                        )

                                        res = await executor.run_structured(
                                            prompt=prompt,
                                            media_path=task.media_path,
                                            video_duration_ms=task.duration_ms,
                                            response_type=MusicRefineAiResponse,
                                            model_name=settings.ai.model_music_refine,
                                            service_tier=settings.ai.service_tier_music_refine,
                                            tools=refine_search_tools,
                                            retry_policy=settings.retry,
                                            agentic_video=True,
                                        )
                                        job.response = res
                                        job.total_attempts += 1
                                        srt_path = settings.dir.tmp / f"{task.stem}.{subtitles_shortcode}.srt"
                                        await asyncio.to_thread(job.save_srt, srt_path)
                                    except QuotaExceededError:
                                        logfire.warning(f"Quota exceeded during refine pass for {task.stem}.")
                                    except Exception as e:
                                        job.total_attempts += 1
                                        logfire.exception(f"Music refine job error for {task.stem}: {e}")
                                    finally:
                                        await asyncio.to_thread(job.save, refine_file)
                                elif job.total_attempts >= settings.retry.max_runs:
                                    logfire.warning(f"Skipping refine for {task.stem}: max attempts reached.")

                            if refine_tid is not None:
                                progress.update(refine_tid, advance=1)
                        finally:
                            refine_queue.task_done()

                worker_tasks: list[asyncio.Task[Any]] = []
                try:
                    if settings.reencode.segment.enabled:
                        for _ in range(settings.concurrency.re_encode):
                            worker_tasks.append(asyncio.create_task(_reencode_worker()))
                    for _ in range(settings.concurrency.subtitles):
                        worker_tasks.append(asyncio.create_task(_subtitles_worker()))
                    if use_refine:
                        for _ in range(settings.concurrency.music_refine):
                            worker_tasks.append(asyncio.create_task(_refine_worker()))

                    if settings.reencode.segment.enabled:
                        await reencode_queue.join()
                    await subtitles_queue.join()
                    if use_refine:
                        await refine_queue.join()
                finally:
                    for wt in worker_tasks:
                        wt.cancel()
                    if worker_tasks:
                        await asyncio.gather(*worker_tasks, return_exceptions=True)

        return stitch_subtitles(
            video_splits=video_splits,
            settings=settings,
            summary_shortcode=summary_shortcode,
            music_refine_shortcode=music_refine_shortcode,
            subtitles_shortcode=subtitles_shortcode,
            summary_available=summary_response is not None,
        )


def main() -> None:
    """CLI entrypoint for the AI Sub subtitle generation pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(run_pipeline(settings)).value)


if __name__ == "__main__":
    main()
