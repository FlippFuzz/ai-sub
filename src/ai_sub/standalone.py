"""Standalone pipeline orchestrator and CLI entrypoint for AI Sub."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from enum import IntEnum
from pathlib import Path
from typing import cast

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
    LyricsJob,
    SubtitlesJob,
    SummaryJob,
    WorkerContext,
)
from ai_sub.pipeline_session import PipelineSession
from ai_sub.protocol import TaskType
from ai_sub.search import (
    BaseSearchProvider,
    create_search_provider,
    create_search_providers,
)
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


async def _probe_duration(semaphore: asyncio.Semaphore, path: Path) -> int:
    """Probes video duration in milliseconds under semaphore control.

    Args:
        semaphore: Concurrency limiter for ffprobe processes.
        path: Video file path.

    Returns:
        int: Duration in milliseconds.
    """
    async with semaphore:
        return await get_video_duration_ms(path)


async def _reencode_summary_proxy_if_needed(
    input_video_path: Path,
    total_duration_ms: int,
    settings: Settings,
) -> Path | None:
    """Re-encodes a low-bitrate summary proxy if video size exceeds threshold.

    Validates that any existing proxy matches the current input video in duration
    before reusing it.

    Args:
        input_video_path: Path to the raw input video.
        total_duration_ms: Total duration of the input video in milliseconds.
        settings: Application settings.

    Returns:
        Path | None: Path to verified summary proxy video, or None if skipped.
    """
    sum_cfg = settings.reencode.summary
    raw_mb = input_video_path.stat().st_size / (1024 * 1024)
    should_reencode = sum_cfg.enabled and (sum_cfg.threshold_mb == 0 or raw_mb >= sum_cfg.threshold_mb)
    if not should_reencode:
        return None

    proxy_output = settings.dir.tmp / f"{input_video_path.stem}.summary_proxy.mp4"
    if proxy_output.is_file():
        try:
            proxy_dur = await get_video_duration_ms(proxy_output)
            effective_tolerance = sum_cfg.duration_tolerance_ms + (int(1000 / sum_cfg.fps) if sum_cfg.fps < 1.0 else 0)
            if abs(total_duration_ms - proxy_dur) < effective_tolerance:
                logfire.info(f"Reusing existing valid summary proxy: {proxy_output.name}")
                return proxy_output
        except Exception:
            pass

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
    return proxy_output


async def run_pipeline(settings: Settings) -> AiSubResult:
    """Executes the subtitle generation pipeline driven by PipelineSession.

    Args:
        settings: Configured application settings.

    Returns:
        AiSubResult: Overall status code.
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

        # Split input video into uniform segments
        video_splits_paths = await split_video(
            input_video=input_video_path,
            output_dir=settings.dir.tmp,
            split_duration_s=settings.split.max_seconds,
            output_pattern="part_%03d",
            duration_tolerance_ms=settings.split.duration_tolerance_ms,
        )

        probe_sem = asyncio.Semaphore(8)
        durations = await asyncio.gather(*(_probe_duration(probe_sem, p) for p in video_splits_paths))
        total_duration_ms = sum(durations)

        reencode_dir = settings.dir.tmp / "reencoded"
        if settings.reencode.segment.enabled:
            reencode_dir.mkdir(exist_ok=True)

        # Initialize shared pipeline session and discover any existing checkpoints on disk
        session = PipelineSession(
            job_dir=settings.dir.tmp,
            video_id=input_video_path.stem,
            title=input_video_path.stem,
            uploader="local",
            duration_ms=total_duration_ms,
            segment_paths=video_splits_paths,
            segment_durations_ms=durations,
            summary_model=settings.ai.model_summary,
            subtitles_model=settings.ai.model_subtitles,
            lyrics_model=settings.ai.model_lyrics,
            reencode_settings=settings.reencode,
            search_settings=settings.ai.search,
            split_settings=settings.split,
            retry_settings=settings.retry,
            user_context=user_context,
            validation_buffer_ms=settings.ai.validation_buffer_ms,
            summary_enabled=use_summary,
            lyrics_enabled=use_lyrics,
            summary_proxy_path=None,
        )
        session.discover_on_disk_state()

        # Only compute summary proxy if summary is enabled and not already complete
        summary_proxy_path: Path | None = None
        if use_summary and not session.is_summary_complete() and not session.summary_failed:
            summary_proxy_path = await _reencode_summary_proxy_if_needed(
                input_video_path=input_video_path,
                total_duration_ms=total_duration_ms,
                settings=settings,
            )
            session.summary_proxy_path = summary_proxy_path

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

            with progress:
                summary_tid: TaskID | None = progress.add_task("[yellow]Summary", total=1) if use_summary else None
                reencode_tid: TaskID | None = (
                    progress.add_task("[cyan]Re-encoding", total=len(video_splits_paths))
                    if settings.reencode.segment.enabled
                    else None
                )
                subtitles_tid: TaskID = progress.add_task("[green]Subtitles", total=len(video_splits_paths))
                lyrics_tid: TaskID | None = (
                    progress.add_task("[magenta]Lyrics", total=len(video_splits_paths)) if use_lyrics else None
                )

                # Advance progress bars for previously completed checkpoints
                if summary_tid is not None and session.is_summary_complete():
                    progress.update(summary_tid, completed=1)
                for idx in range(len(video_splits_paths)):
                    if reencode_tid is not None:
                        out_f = reencode_dir / f"{video_splits_paths[idx].stem}.mp4"
                        if out_f.is_file():
                            session.reencoded_segment_paths[idx] = out_f
                            progress.update(reencode_tid, advance=1)
                    if idx in session.subtitles_responses:
                        progress.update(subtitles_tid, advance=1)
                        if lyrics_tid is not None:
                            sub_r = session.subtitles_responses[idx]
                            if not sub_r.detected_musical_performances or idx in session.lyrics_responses:
                                progress.update(lyrics_tid, advance=1)

                # Phase 1: Whole-Video Summary
                if use_summary and session.summary_failed:
                    logfire.error("Summary stage previously exhausted all runs. Halting pipeline execution.")
                    return AiSubResult.RETRIES_EXCEEDED

                if use_summary and not session.is_summary_complete():
                    summary_file = settings.dir.tmp / f"summary.{session.summary_shortcode}.yaml"
                    summary_job = SummaryJob.load(summary_file, settings.ai.validation_buffer_ms)
                    if not summary_job:
                        summary_job = SummaryJob(
                            name="summary",
                            video_duration_ms=total_duration_ms,
                            user_context=user_context,
                        )

                    summary_media = summary_proxy_path or input_video_path
                    summary_job.video_duration_ms = await get_video_duration_ms(summary_media)

                    # Only execute if attempts have not exhausted max runs
                    if summary_job.total_attempts < settings.retry.max_runs:
                        try:
                            ctx = WorkerContext(
                                executor=executor,
                                media_path=summary_media,
                                search_providers=web_search_providers,
                            )
                            sum_res = await summary_job.execute(ctx)
                            session.complete_task(TaskType.SUMMARY, sum_res.model_dump(mode="json"))
                            logfire.info("Summary completed successfully.")
                        except QuotaExceededError:
                            logfire.warning("Quota exceeded during summary stage.")
                        except Exception as e:
                            logfire.error(f"Summary stage failed: {e}")
                            session.record_task_failure(
                                task_type=TaskType.SUMMARY,
                                attempts=summary_job.total_attempts + 1,
                            )
                        finally:
                            if summary_tid is not None:
                                progress.update(summary_tid, completed=1)

                    if not session.is_summary_complete():
                        logfire.error("Summary stage failed. Halting pipeline execution.")
                        return AiSubResult.RETRIES_EXCEEDED if session.summary_failed else AiSubResult.IN_PROGRESS

                # Phase 2: Per-Chunk Processing with Concurrency Semaphores
                reencode_sem = asyncio.Semaphore(settings.concurrency.re_encode)
                subtitles_sem = asyncio.Semaphore(settings.concurrency.subtitles)
                lyrics_sem = asyncio.Semaphore(settings.concurrency.lyrics if settings.concurrency.lyrics > 0 else 1)

                async def _process_segment(idx: int, path: Path, dur_ms: int) -> None:
                    # 1. Re-encode chunk
                    seg_cfg = settings.reencode.segment
                    raw_mb = path.stat().st_size / (1024 * 1024)
                    should_reenc = seg_cfg.enabled and (seg_cfg.threshold_mb == 0 or raw_mb >= seg_cfg.threshold_mb)

                    if should_reenc and idx not in session.reencoded_segment_paths:
                        out_file = reencode_dir / f"{path.stem}.mp4"
                        if not out_file.is_file():
                            async with reencode_sem:
                                try:
                                    await reencode_video(
                                        input_path=path,
                                        output_path=out_file,
                                        fps=seg_cfg.fps,
                                        height=seg_cfg.height,
                                        bitrate_kb=seg_cfg.bitrate_kb,
                                        encoder=settings.reencode.encoder or "libx264",
                                        duration_tolerance_ms=seg_cfg.duration_tolerance_ms,
                                        keyframe_only=seg_cfg.keyframe_only,
                                    )
                                    session.reencoded_segment_paths[idx] = out_file
                                except Exception as e:
                                    logfire.warning(f"Re-encode failed for {path.stem}: {e}. Using raw.")
                                    session.reencoded_segment_paths[idx] = path
                        else:
                            session.reencoded_segment_paths[idx] = out_file

                        if reencode_tid is not None:
                            progress.update(reencode_tid, advance=1)
                    elif reencode_tid is not None and not should_reenc:
                        progress.update(reencode_tid, advance=1)

                    active_media = session.reencoded_segment_paths.get(idx) or path

                    # 2. Subtitles chunk
                    if idx not in session.subtitles_responses and idx not in session.failed_subtitles:
                        sub_file = settings.dir.tmp / f"{path.stem}.subtitles.{session.subtitles_shortcode}.yaml"
                        sub_job = SubtitlesJob.load(sub_file, settings.ai.validation_buffer_ms)
                        offset_ms = sum(durations[:idx])
                        if not sub_job:
                            sub_job = SubtitlesJob(
                                name=path.stem,
                                video_duration_ms=dur_ms,
                                segment_offset_ms=offset_ms,
                                global_summary=session.summary_response,
                                user_context=user_context,
                            )
                        else:
                            sub_job.segment_offset_ms = offset_ms
                            sub_job.global_summary = session.summary_response

                        if not sub_job.response and sub_job.total_attempts < settings.retry.max_runs:
                            async with subtitles_sem:
                                try:
                                    ctx = WorkerContext(
                                        executor=executor,
                                        media_path=active_media,
                                        search_providers=[],
                                    )
                                    sub_res = await sub_job.execute(ctx)
                                    session.complete_task(
                                        TaskType.SUBTITLES,
                                        sub_res.model_dump(mode="json"),
                                        segment_index=idx,
                                    )
                                    srt_p = (
                                        settings.dir.tmp / f"{path.stem}.subtitles-{session.subtitles_shortcode}.srt"
                                    )
                                    await asyncio.to_thread(sub_job.save_srt, srt_p)
                                except QuotaExceededError:
                                    logfire.warning(f"Quota exceeded during subtitles pass for {path.stem}.")
                                except Exception as e:
                                    logfire.exception(f"Subtitles job error for {path.stem}: {e}")
                                    session.record_task_failure(
                                        task_type=TaskType.SUBTITLES,
                                        segment_index=idx,
                                        attempts=sub_job.total_attempts + 1,
                                    )

                        progress.update(subtitles_tid, advance=1)

                    # 3. Lyrics chunk (if singing detected and lyrics enabled)
                    sub_resp = session.subtitles_responses.get(idx)
                    has_singing = bool(sub_resp and sub_resp.detected_musical_performances)

                    if (
                        use_lyrics
                        and has_singing
                        and idx not in session.lyrics_responses
                        and idx not in session.failed_lyrics
                    ):
                        lyr_file = settings.dir.tmp / f"{path.stem}.lyrics.{session.lyrics_shortcode}.yaml"
                        lyr_job = LyricsJob.load(lyr_file, settings.ai.validation_buffer_ms)
                        offset_ms = sum(durations[:idx])
                        assert sub_resp is not None
                        if not lyr_job:
                            lyr_job = LyricsJob(
                                name=path.stem,
                                video_duration_ms=dur_ms,
                                segment_offset_ms=offset_ms,
                                draft_subtitles=sub_resp.subtitles,
                                detected_performances=sub_resp.detected_musical_performances,
                                global_summary=session.summary_response,
                                user_context=user_context,
                            )

                        if not lyr_job.response and lyr_job.total_attempts < settings.retry.max_runs:
                            async with lyrics_sem:
                                try:
                                    ctx = WorkerContext(
                                        executor=executor,
                                        media_path=active_media,
                                        search_providers=lyrics_search_providers,
                                    )
                                    lyr_res = await lyr_job.execute(ctx)
                                    session.complete_task(
                                        TaskType.LYRICS,
                                        lyr_res.model_dump(mode="json"),
                                        segment_index=idx,
                                    )
                                    srt_p = settings.dir.tmp / f"{path.stem}.lyrics-{session.lyrics_shortcode}.srt"
                                    await asyncio.to_thread(lyr_job.save_srt, srt_p)
                                except QuotaExceededError:
                                    logfire.warning(f"Quota exceeded during lyrics pass for {path.stem}.")
                                except Exception as e:
                                    logfire.exception(f"Lyrics job error for {path.stem}: {e}")
                                    session.record_task_failure(
                                        task_type=TaskType.LYRICS,
                                        segment_index=idx,
                                        attempts=lyr_job.total_attempts + 1,
                                    )

                        if lyrics_tid is not None:
                            progress.update(lyrics_tid, advance=1)
                    elif lyrics_tid is not None and not has_singing:
                        progress.update(lyrics_tid, advance=1)

                await asyncio.gather(
                    *(_process_segment(i, p, d) for i, (p, d) in enumerate(zip(video_splits_paths, durations)))
                )

        # Phase 3: Final Subtitle Stitching
        out_srt = settings.dir.out / f"{input_video_path.stem}.{session.full_shortcode}.srt"
        raw_out_srt = settings.dir.out / f"{input_video_path.stem}.raw-{session.full_shortcode}.srt"

        generated = session.stitch(
            output_primary_path=out_srt,
            output_raw_path=raw_out_srt if use_lyrics else None,
            allow_partial=True,
        )
        for g in generated:
            logfire.info(f"Subtitles written to {g.name}")

        has_failures = bool(
            session.summary_failed or session.failed_subtitles or (use_lyrics and session.failed_lyrics)
        )
        if session.is_complete() and not has_failures:
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
