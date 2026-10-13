"""Main entry point for the AI Sub subtitle generation pipeline.

Orchestrates video splitting, re-encoding, uploading, and AI transcription.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO, cast

import logfire
from google.genai.types import File
from pydantic_settings import CliApp
from pysubs2 import SSAEvent, SSAFile
from tqdm.auto import tqdm

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import LoggingSettings, Settings
from ai_sub.data_models import (
    AgentDeps,
    AiSubResult,
    LyricsSceneAiResponse,
    LyricsSceneJob,
    ReEncodingJob,
    SegmentJobs,
    SubtitleAiResponse,
    SubtitleJob,
    UploadFileJob,
)
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.job_runner import JobRunner
from ai_sub.prompt import (
    LYRICS_PROMPT_VERSION,
    SUBTITLES_PROMPT_VERSION,
    get_lyrics_scenes_prompt,
    get_subtitle_prompt,
    get_verification_prompt,
)
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
)
from ai_sub.video import (
    get_video_duration_ms,
    get_working_encoder,
    reencode_video,
    split_video,
)
from ai_sub.web_search import WebSearchDeps


class TqdmWriteWrapper(io.TextIOBase):
    """Redirects writes to tqdm.write to prevent progress bar interference."""

    def write(self, message: str) -> int:
        """Writes a message to the tqdm console, avoiding empty lines.

        Args:
            message: The string message to write.

        Returns:
            int: The length of the message.
        """
        if cleaned := message.rstrip("\r\n"):
            tqdm.write(cleaned)
        return len(message)

    def flush(self) -> None:
        """Flushes the writer. This is a no-op for tqdm."""
        pass


class ReEncodeJobRunner(JobRunner):
    """Worker that re-encodes video segments to a lower quality/different format.

    This is done to reduce file size before uploading to the Gemini Files API,
    saving bandwidth and processing time.
    """

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "reencode",
    ) -> None:
        """Initializes the ReEncodeJobRunner.

        Args:
            settings: The application's configuration settings.
            max_workers: The maximum number of concurrent tasks.
            on_complete: Callback executed upon successful completion of a job.
            name: The name of the runner.
        """
        super().__init__(settings, max_workers, on_complete, name=name)

    async def process(self, job: SegmentJobs) -> None:
        """Re-encodes the video file specified in the job.

        Args:
            job: The segment job container containing re-encoding details.
        """
        reencode_job = job.reencode
        assert reencode_job is not None
        await reencode_video(
            reencode_job.input_file,
            reencode_job.output_file,
            reencode_job.fps,
            reencode_job.height,
            reencode_job.bitrate_kb,
            self.settings.split.re_encode.encoder or "libx264",
            reencode_job.duration_tolerance_ms,
        )

        logfire.info(f"{reencode_job.name} re-encoded to {reencode_job.output_file.name}")


class UploadJobRunner(JobRunner):
    """Worker that uploads video files to the Gemini Files API."""

    uploader: GeminiFileUploader

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        uploader: GeminiFileUploader,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "upload",
    ) -> None:
        """Initializes the UploadJobRunner.

        Args:
            settings: The application's configuration settings.
            max_workers: The maximum number of concurrent tasks.
            uploader: The Gemini file uploader instance to use.
            on_complete: Callback executed upon successful completion of a job.
            name: The name of the runner.
        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.uploader = uploader

    async def process(self, job: SegmentJobs) -> Any:
        """Uploads the specified file using the GeminiFileUploader.

        Args:
            job: The segment job container containing file upload details.

        Returns:
            Any: The uploaded file metadata object.
        """
        upload_job = job.upload
        assert upload_job is not None
        file = await self.uploader.upload_file(upload_job.python_file)
        logfire.info(f"{upload_job.name} uploaded")
        logfire.debug(f"File: {file}")
        return file


class LyricsSceneJobRunner(JobRunner):
    """Worker that executes the AI agent to detect lyrics and scenes."""

    agent: RateLimitedAgentWrapper
    shortcode: str

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "lyrics",
    ) -> None:
        """Initializes the LyricsSceneJobRunner.

        Args:
            settings: The application's configuration settings.
            max_workers: The maximum number of concurrent tasks.
            agent: The AI agent instance for processing.
            on_complete: Callback executed upon successful completion of a job.
            name: The name of the runner.
        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.shortcode = generate_lyrics_shortcode(self.agent.model_name)

    async def process(self, job: SegmentJobs) -> None:
        """Invokes the AI agent to detect lyrics and scenes.

        Args:
            job: The segment job container containing lyrics/scene detection details.
        """
        lyrics_job = job.lyrics
        assert lyrics_job is not None
        if lyrics_job.response:
            logfire.info(f"Skipping lyrics generation for {lyrics_job.name} as valid response exists.")
            return

        assert lyrics_job.file is not None
        response = await self.agent.run(
            get_lyrics_scenes_prompt(),
            lyrics_job.file,
            lyrics_job.video_duration_ms,
            LyricsSceneAiResponse,
        )
        if response:
            lyrics_job.response = response

    async def post_process(self, job: SegmentJobs) -> None:
        """Saves the result of the lyrics detection to disk.

        Args:
            job: The segment job container with the processed response.
        """
        lyrics_job = job.lyrics
        assert lyrics_job is not None

        job_state_path = self.settings.dir.tmp / f"{lyrics_job.name}.lyrics.{self.shortcode}.yaml"
        await asyncio.to_thread(lyrics_job.save, job_state_path)


class SubtitleJobRunner(JobRunner):
    """Worker that executes the AI agent to generate subtitles."""

    agent: RateLimitedAgentWrapper
    shortcode: str

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]] | None = None,
        name: str = "subtitles",
    ) -> None:
        """Initializes the SubtitleJobRunner.

        Args:
            settings: The application's configuration settings.
            max_workers: The maximum number of concurrent tasks.
            agent: The AI agent instance for processing.
            on_complete: Optional callback executed upon successful completion of a job.
            name: The name of the runner.
        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.shortcode = generate_full_shortcode(self.agent.model_name)

    async def process(self, job: SegmentJobs) -> None:
        """Invokes the AI agent to generate subtitles.

        Args:
            job: The segment job container containing subtitle generation details.
        """
        subtitle_job = job.subtitles
        assert subtitle_job is not None

        lyrics_job = job.lyrics
        scene_response = lyrics_job.response if lyrics_job else None
        base_prompt = get_subtitle_prompt(scene_response)

        # 1. Run the initial subtitle generation if no response exists
        if subtitle_job.response is None:
            assert subtitle_job.file is not None
            response = await self.agent.run(
                base_prompt,
                subtitle_job.file,
                subtitle_job.video_duration_ms,
                SubtitleAiResponse,
            )
            if response:
                subtitle_job.response = response
                job_state_path = self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.shortcode}.yaml"
                await asyncio.to_thread(subtitle_job.save, job_state_path)

        # 2. Check for large gaps and trigger verification runs as needed
        gap_threshold_s = self.settings.ai.verification_gap_seconds
        gap_verification_retries = self.settings.ai.gap_verification_retries
        while gap_verification_retries > 0 and not subtitle_job.is_complete(
            gap_threshold_s,
            gap_verification_retries,
            is_first_segment=job.is_first_segment,
            is_last_segment=job.is_last_segment,
            ignore_start=self.settings.ai.gap_verification_ignore_start,
            ignore_end=self.settings.ai.gap_verification_ignore_end,
        ):
            attempt_num = len(subtitle_job.responses)
            logfire.warning(
                f"Large gap(s) (>= {gap_threshold_s}s) detected in '{subtitle_job.name}'. "
                f"Triggering verification re-run (attempt {attempt_num})..."
            )

            verification_prompt = get_verification_prompt(base_prompt, subtitle_job.video_duration_ms)
            assert subtitle_job.file is not None

            new_response = await self.agent.run(
                verification_prompt,
                subtitle_job.file,
                subtitle_job.video_duration_ms,
                SubtitleAiResponse,
            )
            if new_response:
                subtitle_job.response = new_response
                logfire.info(f"Verification re-run completed for {subtitle_job.name}.")

                job_state_path = self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.shortcode}.yaml"
                await asyncio.to_thread(subtitle_job.save, job_state_path)
            else:
                break

    async def post_process(self, job: SegmentJobs) -> None:
        """Saves the result (or partial state) to disk.

        Args:
            job: The segment job container with the processed subtitles.
        """
        subtitle_job = job.subtitles
        assert subtitle_job is not None

        job_state_path = self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.shortcode}.yaml"
        await asyncio.to_thread(subtitle_job.save, job_state_path)

        if subtitle_job.response is not None:

            def _save_ssa(response: SubtitleAiResponse, path: str) -> None:
                response.get_ssafile().save(path)

            await asyncio.to_thread(
                _save_ssa,
                subtitle_job.response,
                str(self.settings.dir.tmp / f"{subtitle_job.name}.{self.shortcode}.srt"),
            )


def stitch_subtitles(video_splits: list[tuple[Path, int]], settings: Settings) -> AiSubResult:
    """Assembles the final subtitle file from processed segments.

    Args:
        video_splits: A list of tuples containing the path and duration of each video segment.
        settings: The application's configuration settings.

    Returns:
        AiSubResult: The overall result status of subtitle generation.
    """
    with logfire.span("Producing final SRT file"):
        all_subtitles = SSAFile()
        lyrics_shortcode = generate_lyrics_shortcode(settings.ai.model_lyrics)
        subtitles_shortcode = generate_full_shortcode(settings.ai.model_subtitles)

        chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
        offset_ms = sum(duration for _, duration in video_splits[:chunks_to_skip])

        complete = True
        max_retries_exceeded = False
        any_pending = False

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            job_path = settings.dir.tmp / f"{video_path.stem}.subtitles.{subtitles_shortcode}.yaml"
            job = SubtitleJob.load(job_path, settings.ai.validation_buffer_ms)
            if job and job.response:
                current_subtitles = job.response.get_ssafile()
                current_subtitles.shift(ms=offset_ms)
                all_subtitles += current_subtitles
            else:
                all_subtitles.append(
                    SSAEvent(
                        start=offset_ms,
                        end=offset_ms + video_duration_ms,
                        text="Error processing subtitles for this segment.",
                    )
                )
                complete = False

                sub_attempts = job.total_attempts if job else 0
                lyrics_attempts = 0
                if settings.thread.lyrics > 0:
                    lyrics_path = settings.dir.tmp / f"{video_path.stem}.lyrics.{lyrics_shortcode}.yaml"
                    lyrics_job = LyricsSceneJob.load(lyrics_path, settings.ai.validation_buffer_ms)
                    lyrics_attempts = lyrics_job.total_attempts if lyrics_job else 0

                if sub_attempts >= settings.retry.max_runs or lyrics_attempts >= settings.retry.max_runs:
                    max_retries_exceeded = True
                else:
                    any_pending = True

            offset_ms += video_duration_ms

        settings_dict = settings.model_dump(
            mode="json",
            exclude={
                "input_video_file": True,
                "dir": True,
                "ai": {
                    "google": {"key": True, "base_url": True},
                    "search": {"key": True},
                },
            },
        )
        state_info = {
            "ai_sub_version": version("ai-sub"),
            "generated_at": datetime.now().isoformat(),
            "lyrics_prompt_version": LYRICS_PROMPT_VERSION,
            "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
            "complete": complete,
            "max_retries_exceeded": max_retries_exceeded,
            "any_pending": any_pending,
            "settings": settings_dict,
        }
        info_text = json.dumps(state_info, indent=2).replace("{", "(").replace("}", ")")
        all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info_text))

        if len(all_subtitles) > 1 and all_subtitles[1].start < 1:
            all_subtitles[1].start = 1

        input_video_path = cast(Path, settings.input_video_file)
        all_subtitles.save(str(settings.dir.out / f"{input_video_path.stem}.{subtitles_shortcode}.srt"))

        if any_pending:
            return AiSubResult.INCOMPLETE
        if max_retries_exceeded:
            return AiSubResult.MAX_RETRIES_EXHAUSTED
        return AiSubResult.COMPLETE


def setup_logging(settings: Settings | LoggingSettings) -> None:
    """Sets up Logfire configuration with Google Gen AI instrumentation.

    Args:
        settings: The configuration settings containing logging options.
    """
    log_settings = settings.log if isinstance(settings, Settings) else settings

    output = cast(TextIO, TqdmWriteWrapper()) if log_settings.progress_bars else None
    logfire.configure(
        console=logfire.ConsoleOptions(
            output=output,
            min_log_level=log_settings.level,
            include_timestamps=log_settings.timestamps,
        ),
        service_name=socket.gethostname(),
        service_version=version("ai-sub"),
        send_to_logfire="if-token-present",
        scrubbing=None if log_settings.scrub else False,
    )
    no_console_logfire = logfire.configure(
        local=True,
        console=False,
        send_to_logfire="if-token-present",
        scrubbing=None if log_settings.scrub else False,
    )

    if hasattr(logfire, "instrument_google_genai"):
        try:
            logfire.instrument_google_genai()
        except Exception as e:
            logfire.debug(f"Failed to instrument Google Gen AI: {e}")

    no_console_logfire.instrument_httpx(capture_all=True)


async def ai_sub(settings: Settings, configure_logging: bool = True) -> AiSubResult:
    """Orchestrates the subtitle generation pipeline.

    Args:
        settings: The application configuration.
        configure_logging: If True, configures Logfire for observability.

    Returns:
        AiSubResult: An enum indicating the final status (COMPLETE, INCOMPLETE, etc.).
    """
    if configure_logging:
        setup_logging(settings)

    input_video_path = cast(Path, settings.input_video_file)

    with logfire.span("Generating subtitles for {video_file}", video_file=input_video_path.name):
        if settings.split.re_encode.enabled and not settings.split.re_encode.encoder:
            with logfire.span("Detecting hardware encoder"):
                settings.split.re_encode.encoder = await get_working_encoder()
                logfire.info(f"Using encoder: {settings.split.re_encode.encoder}")

        use_lyrics = settings.thread.lyrics > 0
        search_tool = settings.ai.search.web_search_tool
        use_custom_search = use_lyrics and search_tool in ("duckduckgo", "ollama", "langsearch")

        async with AsyncExitStack() as stack:
            agent_deps = AgentDeps(validation_buffer_ms=settings.ai.validation_buffer_ms)
            if use_custom_search:
                search_deps = WebSearchDeps(settings.ai.search, provider=search_tool)
                await stack.enter_async_context(search_deps)
                agent_deps.web_search = search_deps

            agent_subtitles = RateLimitedAgentWrapper(
                settings,
                settings.ai.model_subtitles,
                service_tier=settings.ai.service_tier_subtitles,
            )
            agent_scene = (
                RateLimitedAgentWrapper(
                    settings,
                    settings.ai.model_lyrics,
                    use_web_search=True,
                    deps=agent_deps,
                    service_tier=settings.ai.service_tier_lyrics,
                )
                if use_lyrics
                else None
            )

            lyrics_shortcode = generate_lyrics_shortcode(settings.ai.model_lyrics)
            subtitles_shortcode = generate_full_shortcode(settings.ai.model_subtitles)

            bars: dict[str, Any] = {}

            async def refresh_bars_loop() -> None:
                """Periodically refreshes bars to handle window resizing during long waits."""
                if not settings.log.progress_bars:
                    return
                try:
                    while True:
                        await asyncio.sleep(settings.log.progress_bar_refresh_seconds)
                        with tqdm.get_lock():
                            for bar in list(bars.values()):
                                bar.refresh()
                except asyncio.CancelledError:
                    pass

            def mark_done(stage: str) -> None:
                if settings.log.progress_bars and stage in bars:
                    bars[stage].update(1)

            use_reencode = settings.split.re_encode.enabled

            def create_bars(total: int) -> None:
                if not settings.log.progress_bars:
                    return
                common_kwargs: dict[str, Any] = {
                    "total": total,
                    "unit": "part",
                    "leave": True,
                    "dynamic_ncols": False,
                    "ncols": settings.log.progress_bar_width,
                    "bar_format": "{desc:<9}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                }
                if use_reencode:
                    bars["reencode"] = tqdm(desc="Reencodes", position=len(bars), **common_kwargs)
                bars["upload"] = tqdm(desc="Uploads", position=len(bars), **common_kwargs)
                if use_lyrics:
                    bars["lyrics"] = tqdm(desc="Lyrics", position=len(bars), **common_kwargs)
                bars["subtitles"] = tqdm(desc="Subtitles", position=len(bars), **common_kwargs)

            def sync_progress(entry_stage: str) -> None:
                for s in ["reencode", "upload", "lyrics", "subtitles"]:
                    if s == entry_stage:
                        break
                    mark_done(s)

            # Step 1: Split video into smaller segments
            video_splits_paths = await split_video(
                input_video_path,
                settings.dir.tmp,
                settings.split.max_seconds,
                output_pattern="part_%03d",
                duration_tolerance_ms=settings.split.re_encode.duration_tolerance_ms,
            )

            semaphore = asyncio.Semaphore(8)

            async def probe_with_sema(path: Path) -> int:
                async with semaphore:
                    return await get_video_duration_ms(path)

            tasks = [probe_with_sema(path) for path in video_splits_paths]
            durations = await asyncio.gather(*tasks)
            video_splits: list[tuple[Path, int]] = list(zip(video_splits_paths, durations, strict=True))

            chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
            splits_to_process = video_splits
            if chunks_to_skip > 0:
                skipped_splits = video_splits[:chunks_to_skip]
                initial_offset_ms = sum(duration for _, duration in skipped_splits)
                splits_to_process = video_splits[chunks_to_skip:]
                logfire.info(
                    f"Skipping first {chunks_to_skip} chunks ({len(skipped_splits)} segments, "
                    f"{initial_offset_ms}ms) due to start_offset_min={settings.split.start_offset_min}"
                )

            create_bars(len(video_splits))
            for _ in range(len(video_splits) - len(splits_to_process)):
                sync_progress("done")

            # Step 2: Configure the job processing pipeline
            reencode_runner: ReEncodeJobRunner | None = None
            upload_runner: UploadJobRunner | None = None
            scene_detection_runner: LyricsSceneJobRunner | None = None
            subtitle_runner: SubtitleJobRunner | None = None

            async def on_stage_complete(stage: str, job: SegmentJobs, result: Any) -> None:
                """Handles transition between pipeline stages."""
                mark_done(stage)

                file_handle: Any = result
                duration_ms: int = 0
                name: str = ""

                if stage == "reencode":
                    assert job.reencode is not None
                    name = job.reencode.name
                    file_handle = job.reencode.output_file
                    duration_ms = await get_video_duration_ms(file_handle)
                elif stage == "upload":
                    assert job.upload is not None
                    name = job.upload.name
                    duration_ms = job.upload.video_duration_ms
                elif stage == "lyrics":
                    assert job.lyrics is not None
                    if not job.lyrics.response:
                        return
                    name = job.lyrics.name
                    file_handle = file_handle or job.lyrics.file
                    duration_ms = job.lyrics.video_duration_ms

                next_stage = None
                if stage == "reencode":
                    next_stage = "upload"
                elif stage == "upload":
                    next_stage = "lyrics" if use_lyrics else "subtitles"
                elif stage == "lyrics":
                    next_stage = "subtitles"

                if not next_stage:
                    return

                if next_stage == "lyrics" and job.lyrics and job.lyrics.response:
                    await on_stage_complete("lyrics", job, file_handle)
                    return

                if next_stage == "subtitles" and job.subtitles and job.subtitles.response:
                    if job.subtitles.is_complete(
                        settings.ai.verification_gap_seconds,
                        settings.ai.gap_verification_retries,
                        is_first_segment=job.is_first_segment,
                        is_last_segment=job.is_last_segment,
                        ignore_start=settings.ai.gap_verification_ignore_start,
                        ignore_end=settings.ai.gap_verification_ignore_end,
                    ):
                        mark_done("subtitles")
                        return

                if next_stage == "upload":
                    job.upload = UploadFileJob(
                        name=name,
                        python_file=file_handle,
                        video_duration_ms=duration_ms,
                    )
                    if upload_runner:
                        await upload_runner.add_job(job)

                elif next_stage == "lyrics":
                    existing_lyrics = job.lyrics
                    new_lyrics_job = LyricsSceneJob(
                        name=name,
                        file=cast(File, file_handle),
                        video_duration_ms=duration_ms,
                        total_attempts=existing_lyrics.total_attempts if existing_lyrics else 0,
                    )
                    if existing_lyrics and existing_lyrics.response:
                        new_lyrics_job.response = existing_lyrics.response

                    job.lyrics = new_lyrics_job
                    if scene_detection_runner:
                        await scene_detection_runner.add_job(job)

                elif next_stage == "subtitles":
                    existing_subs = job.subtitles
                    new_subs_job = SubtitleJob(
                        name=name,
                        file=cast(File, file_handle),
                        video_duration_ms=duration_ms,
                        total_attempts=existing_subs.total_attempts if existing_subs else 0,
                    )
                    if existing_subs:
                        new_subs_job.responses = list(existing_subs.responses)

                    job.subtitles = new_subs_job
                    if subtitle_runner:
                        await subtitle_runner.add_job(job)

            if use_reencode:
                reencode_runner = ReEncodeJobRunner(
                    settings,
                    settings.thread.re_encode,
                    on_complete=partial(on_stage_complete, "reencode"),
                )

            upload_runner = UploadJobRunner(
                settings,
                settings.thread.uploads,
                uploader=GeminiFileUploader(settings),
                on_complete=partial(on_stage_complete, "upload"),
            )

            if use_lyrics:
                assert agent_scene is not None
                scene_detection_runner = LyricsSceneJobRunner(
                    settings,
                    settings.thread.lyrics,
                    agent_scene,
                    on_complete=partial(on_stage_complete, "lyrics"),
                )

            async def on_subtitles_complete(job: SegmentJobs, result: Any) -> None:
                mark_done("subtitles")

            subtitle_runner = SubtitleJobRunner(
                settings,
                settings.thread.subtitles,
                agent_subtitles,
                on_complete=on_subtitles_complete,
            )

            # Step 3: Populate the job queues
            reencode_dir = settings.dir.tmp / "reencoded"
            if use_reencode:
                reencode_dir.mkdir(exist_ok=True)

            total_splits = len(splits_to_process)
            for idx, (split, duration) in enumerate(splits_to_process):
                is_first = idx == 0
                is_last = idx == total_splits - 1

                job_state = SegmentJobs(
                    is_first_segment=is_first,
                    is_last_segment=is_last,
                )

                lyrics_job_path = settings.dir.tmp / f"{split.stem}.lyrics.{lyrics_shortcode}.yaml"
                lyrics_job = await asyncio.to_thread(
                    LyricsSceneJob.load, lyrics_job_path, settings.ai.validation_buffer_ms
                )
                if lyrics_job:
                    job_state.lyrics = lyrics_job

                subtitle_job_path = settings.dir.tmp / f"{split.stem}.subtitles.{subtitles_shortcode}.yaml"
                subtitle_job = await asyncio.to_thread(
                    SubtitleJob.load, subtitle_job_path, settings.ai.validation_buffer_ms
                )
                if subtitle_job:
                    job_state.subtitles = subtitle_job

                is_complete = subtitle_job is not None and subtitle_job.is_complete(
                    settings.ai.verification_gap_seconds,
                    settings.ai.gap_verification_retries,
                    is_first_segment=is_first,
                    is_last_segment=is_last,
                    ignore_start=settings.ai.gap_verification_ignore_start,
                    ignore_end=settings.ai.gap_verification_ignore_end,
                )
                if is_complete:
                    sync_progress("done")
                    continue

                if subtitle_job is not None and subtitle_job.response:
                    logfire.warning(
                        f"Large gap(s) (>= {settings.ai.verification_gap_seconds}s) detected in "
                        f"'{split.stem}' from previous run. Triggering verification re-run..."
                    )

                should_reencode = False
                if use_reencode:
                    if settings.split.re_encode.threshold_mb == 0:
                        should_reencode = True
                    else:
                        file_size_mb = split.stat().st_size / (1024 * 1024)
                        should_reencode = file_size_mb >= settings.split.re_encode.threshold_mb

                if should_reencode:
                    sync_progress("reencode")
                    output_file = reencode_dir / split.with_suffix(".mov").name
                    job_state.reencode = ReEncodingJob(
                        name=split.stem,
                        input_file=split,
                        output_file=output_file,
                        fps=settings.split.re_encode.fps,
                        height=settings.split.re_encode.height,
                        bitrate_kb=settings.split.re_encode.bitrate_kb,
                        duration_tolerance_ms=settings.split.re_encode.duration_tolerance_ms,
                    )
                    if reencode_runner:
                        await reencode_runner.add_job(job_state)
                    continue

                # Upload is mandatory for all segments
                sync_progress("upload")
                job_state.upload = UploadFileJob(
                    name=split.stem,
                    python_file=split,
                    video_duration_ms=duration,
                )
                await upload_runner.add_job(job_state)

            # Step 4: Start all runners and process queues
            refresh_task = asyncio.create_task(refresh_bars_loop())
            runners = [r for r in [reencode_runner, upload_runner, scene_detection_runner, subtitle_runner] if r]
            try:
                for runner in runners:
                    await runner.start()

                for runner in runners:
                    await runner.join()

                # Step 5: Assemble the final subtitle file
                result = await asyncio.to_thread(stitch_subtitles, video_splits, settings)
                logfire.info(f"Done - {result.name}")
                return result
            finally:
                for runner in runners:
                    await runner.shutdown()
                refresh_task.cancel()
                for b in bars.values():
                    b.close()


def main() -> None:
    """Parses CLI arguments and runs the main ai_sub pipeline."""
    settings = CliApp.run(Settings)
    sys.exit(asyncio.run(ai_sub(settings)).value)


if __name__ == "__main__":
    main()
