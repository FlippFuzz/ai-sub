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

# Resolve forward references in AgentDeps now that all modules are loaded.
# This prevents PydanticUserError when AgentDeps is instantiated.
from ai_sub.prompt import (
    SUBTITLES_PROMPT_VERSION,
    get_lyrics_scenes_prompt,
    get_subtitle_prompt,
    get_verification_prompt,
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
            The length of the message.
        """
        # tqdm.write handles clearing and redrawing bars automatically.
        # We remove trailing newlines because tqdm.write appends its own.
        if cleaned := message.rstrip("\r\n"):
            tqdm.write(cleaned)
        return len(message)

    def flush(self) -> None:
        """Flushes the writer. This is a no-op for tqdm."""
        pass


class ReEncodeJobRunner(JobRunner):
    """Worker that re-encodes video segments to a lower quality/different format.

    This is typically done to reduce file size before uploading to an API,
    saving bandwidth and potentially processing time.
    """

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "reencode",
    ):
        """Initializes the ReEncodeJobRunner.

        Args:
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of concurrent tasks.
            on_complete (Callable[[SegmentJobs, Any], Awaitable[None]]): Callback executed
                upon successful completion of a job.
            name (str): The name of the runner.

        """
        super().__init__(settings, max_workers, on_complete, name=name)

    async def process(self, job: SegmentJobs) -> None:
        """Re-encodes the video file specified in the job.

        Args:
            job (SegmentJobs): The segment job container containing re-encoding details.
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
    """Worker that uploads video files to the Gemini Files API.

    This runner is used when the AI model requires the file to be hosted
    on Google's servers (e.g., for Gemini models).
    """

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        uploader: GeminiFileUploader,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "upload",
    ):
        """Initializes the UploadJobRunner.

        Args:
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of concurrent tasks.
            uploader (GeminiFileUploader): The uploader instance to use.
            on_complete (Callable[[SegmentJobs, Any], Awaitable[None]]): Callback executed
                upon successful completion of a job.
            name (str): The name of the runner.

        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.uploader = uploader

    async def process(self, job: SegmentJobs) -> Any:
        """Uploads the specified file using the GeminiFileUploader.

        Args:
            job: The segment job container containing file upload details.

        Returns:
            The uploaded file metadata object.
        """
        upload_job = job.upload
        assert upload_job is not None
        # Perform the file upload.
        file = await self.uploader.upload_file(upload_job.python_file)
        logfire.info(f"{upload_job.name} uploaded")
        logfire.debug(f"File: {file}")
        return file


class LyricsSceneJobRunner(JobRunner):
    """Worker that executes the AI agent to detect lyrics and scenes."""

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]],
        name: str = "lyrics",
    ):
        """Initializes the LyricsSceneJobRunner.

        Args:
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of concurrent tasks.
            agent (RateLimitedAgentWrapper): The AI agent instance for processing.
            on_complete (Callable[[SegmentJobs, Any], Awaitable[None]]): Callback executed
                upon successful completion of a job.
            name (str): The name of the runner.

        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(self.agent.model_name)

    async def process(self, job: SegmentJobs) -> None:
        """Invokes the AI agent to detect lyrics and scenes.

        Args:
            job (SegmentJobs): The segment job container containing lyrics/scene detection details.
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
            job (SegmentJobs): The segment job container with the processed response.
        """
        lyrics_job = job.lyrics
        assert lyrics_job is not None

        job_state_path = self.settings.dir.tmp / f"{lyrics_job.name}.lyrics.{self.sanitized_model_name}.json"
        await asyncio.to_thread(lyrics_job.save, job_state_path)


class SubtitleJobRunner(JobRunner):
    """Worker that executes the AI agent to generate subtitles."""

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]] | None = None,
        name: str = "subtitles",
    ):
        """Initializes the SubtitleJobRunner.

        Args:
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of concurrent tasks.
            agent (RateLimitedAgentWrapper): The AI agent instance for processing.
            on_complete (Callable[[SegmentJobs, Any], Awaitable[None]] | None): Optional callback
                executed upon successful completion of a job.
            name (str): The name of the runner.

        """
        super().__init__(settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(self.agent.model_name)

    async def process(self, job: SegmentJobs) -> None:
        """Invokes the AI agent to generate subtitles.

        Args:
            job (SegmentJobs): The segment job container containing subtitle generation details.
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
                # Explicitly checkpoint the initial pass before entering the verification block
                job_state_path = (
                    self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.sanitized_model_name}.json"
                )
                await asyncio.to_thread(subtitle_job.save, job_state_path)

        # 2. Check for ANY large gap and trigger verification runs as needed
        gap_threshold_s = self.settings.ai.verification_gap_seconds
        gap_verification_retries = self.settings.ai.gap_verification_retries
        while gap_verification_retries > 0 and not subtitle_job.is_complete(gap_threshold_s, gap_verification_retries):
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

                # Checkpoint after each successful verification pass
                job_state_path = (
                    self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.sanitized_model_name}.json"
                )
                await asyncio.to_thread(subtitle_job.save, job_state_path)
            else:
                break

    async def post_process(self, job: SegmentJobs) -> None:
        """Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.

        Args:
            job (SegmentJobs): The segment job container with the processed subtitles.
        """
        # Save the completed job state to a JSON file for persistence.
        subtitle_job = job.subtitles
        assert subtitle_job is not None

        # Always save the job state to persist retry counts across runs.
        job_state_path = self.settings.dir.tmp / f"{subtitle_job.name}.subtitles.{self.sanitized_model_name}.json"
        await asyncio.to_thread(subtitle_job.save, job_state_path)

        # Also generate a subtitle file for this job for the user to view.
        if subtitle_job.response is not None:
            sanitized_model = self.settings.ai.get_sanitized_model_name(self.settings.ai.model_subtitles)

            def _save_ssa(response: SubtitleAiResponse, path: str) -> None:
                response.get_ssafile().save(path)

            await asyncio.to_thread(
                _save_ssa,
                subtitle_job.response,
                str(self.settings.dir.tmp / f"{subtitle_job.name}.{sanitized_model}.srt"),
            )


def stitch_subtitles(video_splits: list[tuple[Path, int]], settings: Settings) -> AiSubResult:
    """Assembles the final subtitle file from processed segments.

    It iterates through the expected video segments, loads the corresponding
    processed subtitle jobs, and concatenates them. Timestamps are shifted
    appropriately to match the original video's timeline.

    Args:
        video_splits: A list of tuples containing the path and duration of each video segment.
        settings: The application's configuration settings.

    Returns:
        AiSubResult: The overall result status of the subtitle generation (COMPLETE, INCOMPLETE, etc.).

    """
    with logfire.span("Producing final SRT file"):
        all_subtitles = SSAFile()
        sanitized_lyrics_model = settings.ai.get_sanitized_model_name(settings.ai.model_lyrics)
        sanitized_subtitles_model = settings.ai.get_sanitized_model_name(settings.ai.model_subtitles)

        chunks_to_skip = int((settings.split.start_offset_min * 60) / settings.split.max_seconds)
        offset_ms = sum(duration for _, duration in video_splits[:chunks_to_skip])

        complete = True
        max_retries_exceeded = False
        any_pending = False

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            # Load the job result from the temporary JSON file.
            job_path = settings.dir.tmp / f"{video_path.stem}.subtitles.{sanitized_subtitles_model}.json"
            job = SubtitleJob.load(job_path, settings.ai.validation_buffer_ms)
            if job and job.response:
                current_subtitles = job.response.get_ssafile()
                # Shift the timestamps of the current subtitle segment by the
                # cumulative duration of all previous segments.
                current_subtitles.shift(ms=offset_ms)
                all_subtitles += current_subtitles
            else:
                # If a segment failed processing, insert an error message
                # into the subtitles for that time range.
                all_subtitles.append(
                    SSAEvent(
                        start=offset_ms,
                        end=offset_ms + video_duration_ms,
                        text="Error processing subtitles for this segment.",
                    )
                )
                complete = False

                # Check status
                sub_attempts = job.total_attempts if job else 0
                lyrics_attempts = 0
                if settings.thread.lyrics > 0:
                    lyrics_path = settings.dir.tmp / f"{video_path.stem}.lyrics.{sanitized_lyrics_model}.json"
                    lyrics_job = LyricsSceneJob.load(lyrics_path, settings.ai.validation_buffer_ms)
                    lyrics_attempts = lyrics_job.total_attempts if lyrics_job else 0

                if sub_attempts >= settings.retry.max_runs or lyrics_attempts >= settings.retry.max_runs:
                    max_retries_exceeded = True
                else:
                    any_pending = True

            # Add the duration of the current segment to the offset for the next one.
            offset_ms += video_duration_ms

        # Insert version and config, as a single SSAEvent at the beginning (0-1ms)
        # JSON curly braces {} are treated as formatting codes in SRT, so replace them.
        # Also exclude sensitive fields from being displayed
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
            "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
            "complete": complete,
            "max_retries_exceeded": max_retries_exceeded,
            "any_pending": any_pending,
            "settings": settings_dict,
        }
        info_text = json.dumps(state_info, indent=2).replace("{", "(").replace("}", ")")
        all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info_text))

        # Make sure that the info_text don't overlap with the first actual subtitle
        if len(all_subtitles) > 1 and all_subtitles[1].start < 1:
            all_subtitles[1].start = 1

        input_video_path = cast(Path, settings.input_video_file)
        all_subtitles.save(str(settings.dir.out / f"{input_video_path.stem}.{sanitized_subtitles_model}.srt"))

        if any_pending:
            return AiSubResult.INCOMPLETE
        if max_retries_exceeded:
            return AiSubResult.MAX_RETRIES_EXHAUSTED
        return AiSubResult.COMPLETE


def setup_logging(settings: Settings | LoggingSettings) -> None:
    """Sets up default Logfire configuration for standalone execution."""
    log_settings = settings.log if isinstance(settings, Settings) else settings

    # Use TqdmWriteWrapper only if bars are enabled to prevent unnecessary interception
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
    no_console_logfire.instrument_pydantic_ai()
    no_console_logfire.instrument_httpx(capture_all=True)


async def ai_sub(settings: Settings, configure_logging: bool = True) -> AiSubResult:
    """Orchestrates the subtitle generation pipeline.

    The pipeline consists of four sequential stages for each video segment:
    1.  **Re-encode (Optional):** Compresses the video segment if it exceeds size thresholds.
    2.  **Upload (Optional):** Uploads the file to the AI provider (e.g., Gemini Files API).
    3.  **Lyrics/Scene Detection (Optional):** Analyzes the video for context and song lyrics.
    4.  **Subtitle Generation:** Generates the final subtitles using context from previous steps.

    **Resumption Logic:**
    The system is designed to be idempotent and resumable. For each segment, it checks
    existing state files to determine if a stage is already complete. It queues the
    segment at the *earliest incomplete stage*.

    **Job Chaining:**
    Completion of one stage triggers the next stage automatically via callbacks
    (e.g., `on_reencode_complete` queues the `Upload` job).

    Args:
        settings (Settings): The application configuration.
        configure_logging (bool): If True, configures Logfire for observability.
                                  Set to False if the calling application
                                  manages its own logging.

    Returns:
        AiSubResult: An enum indicating the final status (COMPLETE, INCOMPLETE, etc.).

    """
    if configure_logging:
        setup_logging(settings)

    input_video_path = cast(Path, settings.input_video_file)

    # Start the main application logic within a Logfire span for better tracing.
    with logfire.span("Generating subtitles for {video_file}", video_file=input_video_path.name):
        if settings.split.re_encode.enabled and not settings.split.re_encode.encoder:
            with logfire.span("Detecting hardware encoder"):
                settings.split.re_encode.encoder = await get_working_encoder()
                logfire.info(f"Using encoder: {settings.split.re_encode.encoder}")

        # Initialize the AI Agent.
        # A custom wrapper is used to make handling rate limits and differences in models more cleanly
        use_lyrics = settings.thread.lyrics > 0
        use_ollama_search = settings.ai.search.web_search_tool == "ollama" and use_lyrics
        use_langsearch = settings.ai.search.web_search_tool == "langsearch" and use_lyrics

        async with AsyncExitStack() as stack:
            agent_deps = AgentDeps(validation_buffer_ms=settings.ai.validation_buffer_ms)
            if use_ollama_search or use_langsearch:
                provider = "ollama" if use_ollama_search else "langsearch"
                search_deps = WebSearchDeps(settings.ai.search, provider=provider)
                await stack.enter_async_context(search_deps)
                agent_deps.web_search = search_deps

            agent_subtitles = RateLimitedAgentWrapper(settings, settings.ai.model_subtitles)
            agent_scene = (
                RateLimitedAgentWrapper(
                    settings,
                    settings.ai.model_lyrics,
                    use_web_search=True,
                    deps=agent_deps,
                )
                if use_lyrics
                else None
            )

            sanitized_lyrics_model = settings.ai.get_sanitized_model_name(settings.ai.model_lyrics)
            sanitized_subtitles_model = settings.ai.get_sanitized_model_name(settings.ai.model_subtitles)

            # Initialize progress bars and a background task to handle UI refreshes (resizing)
            bars: dict[str, Any] = {}

            async def refresh_bars_loop():
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

            def mark_done(stage: str):
                if settings.log.progress_bars and stage in bars:
                    bars[stage].update(1)

            # Determine stage visibility
            use_reencode = settings.split.re_encode.enabled
            is_google_sub = agent_subtitles.is_google()
            is_google_scene = agent_scene.is_google() if agent_scene else False
            use_upload = (is_google_sub or is_google_scene) and settings.ai.google.use_files_api

            def create_bars(total: int):
                if not settings.log.progress_bars:
                    return
                # leave=True ensures bars stay in place when finished, preventing vertical shifts.
                # dynamic_ncols is set to False to prevent layout corruption in VS Code/PuTTY;
                # we use a fixed ncols value defined in the configuration instead.
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
                if use_upload:
                    bars["upload"] = tqdm(desc="Uploads", position=len(bars), **common_kwargs)
                if use_lyrics:
                    bars["lyrics"] = tqdm(desc="Lyrics", position=len(bars), **common_kwargs)
                bars["subtitles"] = tqdm(desc="Subtitles", position=len(bars), **common_kwargs)

            # Helper to sync progress when skipping stages
            def sync_progress(entry_stage: str):
                for s in ["reencode", "upload", "lyrics", "subtitles"]:
                    if s == entry_stage:
                        break
                    mark_done(s)

            # Step 1: Split the input video into smaller segments.
            video_splits_paths = await split_video(
                input_video_path,
                settings.dir.tmp,
                settings.split.max_seconds,
                output_pattern="part_%03d",
                duration_tolerance_ms=settings.split.re_encode.duration_tolerance_ms,
            )

            # Get durations in parallel
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
            # Mark segments that are completely skipped (already finished) as done for all bars
            for _ in range(len(video_splits) - len(splits_to_process)):
                sync_progress("done")  # Matches no stage, updates everything

            # Step 2: Configure the job processing pipeline.

            reencode_runner: ReEncodeJobRunner | None = None
            upload_runner: UploadJobRunner | None = None
            scene_detection_runner: LyricsSceneJobRunner | None = None
            subtitle_runner: SubtitleJobRunner | None = None

            # Define callbacks
            # These functions handle the transition between pipeline stages.
            # When a job completes, the next required job is created and queued.
            async def on_stage_complete(stage: str, job: SegmentJobs, result: Any) -> None:
                """Handles the transition between pipeline stages."""
                mark_done(stage)

                file_handle: Any = result
                duration_ms: int = 0
                name: str = ""

                # Extract data based on the completed stage
                if stage == "reencode":
                    assert job.reencode is not None
                    name = job.reencode.name
                    file_handle = job.reencode.output_file
                    duration_ms = await get_video_duration_ms(file_handle)
                elif stage == "upload":
                    assert job.upload is not None
                    name = job.upload.name
                    # file_handle is already the result (the uploaded file object)
                    duration_ms = job.upload.video_duration_ms
                elif stage == "lyrics":
                    assert job.lyrics is not None
                    if not job.lyrics.response:
                        return  # Should not happen if on_complete is called after success
                    name = job.lyrics.name
                    file_handle = file_handle or job.lyrics.file
                    duration_ms = job.lyrics.video_duration_ms

                    # Logic for Subtitles file source:
                    # If we are using a non-Google model for subtitles, but we uploaded the file
                    # (e.g. for Google Lyrics), we might need to fallback to the local file.
                    if not agent_subtitles.is_google() and job.upload:
                        file_handle = job.upload.python_file

                # Determine next stage
                next_stage = None
                if stage == "reencode":
                    if use_upload:
                        next_stage = "upload"
                    elif use_lyrics:
                        next_stage = "lyrics"
                    else:
                        next_stage = "subtitles"
                elif stage == "upload":
                    if use_lyrics:
                        next_stage = "lyrics"
                    else:
                        next_stage = "subtitles"
                elif stage == "lyrics":
                    next_stage = "subtitles"

                if not next_stage:
                    return

                # Skip stages that are already complete to avoid incrementing retry counters.
                if next_stage == "lyrics" and job.lyrics and job.lyrics.response:
                    # Transition immediately to the stage following lyrics
                    await on_stage_complete("lyrics", job, file_handle)
                    return

                if next_stage == "subtitles" and job.subtitles and job.subtitles.response:
                    if job.subtitles.is_complete(
                        settings.ai.verification_gap_seconds,
                        settings.ai.gap_verification_retries,
                    ):
                        mark_done("subtitles")
                        return

                # Queue next job
                if next_stage == "upload":
                    # Retries are tracked per-stage. Re-encoding and uploading are
                    # idempotent, so resetting the counter has no adverse effect.
                    job.upload = UploadFileJob(
                        name=name,
                        python_file=file_handle,
                        video_duration_ms=duration_ms,
                    )
                    if upload_runner:
                        await upload_runner.add_job(job)

                elif next_stage == "lyrics":
                    existing_lyrics = job.lyrics
                    # Retries are tracked per-stage.
                    new_lyrics_job = LyricsSceneJob(
                        name=name,
                        file=file_handle,
                        video_duration_ms=duration_ms,
                        total_attempts=existing_lyrics.total_attempts if existing_lyrics else 0,
                    )
                    if existing_lyrics:
                        if existing_lyrics.response:
                            new_lyrics_job.response = existing_lyrics.response

                    job.lyrics = new_lyrics_job
                    if scene_detection_runner:
                        await scene_detection_runner.add_job(job)

                elif next_stage == "subtitles":
                    existing_subs = job.subtitles
                    # Retries are tracked per-stage.
                    new_subs_job = SubtitleJob(
                        name=name,
                        file=file_handle,
                        video_duration_ms=duration_ms,
                        total_attempts=existing_subs.total_attempts if existing_subs else 0,
                    )
                    if existing_subs:
                        new_subs_job.responses = list(existing_subs.responses)

                    job.subtitles = new_subs_job
                    if subtitle_runner:
                        await subtitle_runner.add_job(job)

            # Setup reencode_runner
            if use_reencode:
                reencode_runner = ReEncodeJobRunner(
                    settings,
                    settings.thread.re_encode,
                    on_complete=partial(on_stage_complete, "reencode"),
                )

            # Setup upload_runner
            if use_upload:
                upload_runner = UploadJobRunner(
                    settings,
                    settings.thread.uploads,
                    uploader=GeminiFileUploader(settings),
                    on_complete=partial(on_stage_complete, "upload"),
                )

            # Setup scene_detection_runner
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

            # Setup subtitle_runner
            subtitle_runner = SubtitleJobRunner(
                settings,
                settings.thread.subtitles,
                agent_subtitles,
                on_complete=on_subtitles_complete,
            )

            # Step 3: Populate the job queues.

            # Create a directory for re-encoded files to avoid name collisions
            # and preserve the file stem for stitching.
            reencode_dir = settings.dir.tmp / "reencoded"
            if use_reencode:
                reencode_dir.mkdir(exist_ok=True)

            # Iterate through all video segments to determine their starting point in the pipeline.
            for split, duration in splits_to_process:
                job_state = SegmentJobs()

                # Load jobs if they exist and populate the in-memory JobState
                lyrics_job_path = settings.dir.tmp / f"{split.stem}.lyrics.{sanitized_lyrics_model}.json"
                lyrics_job = await asyncio.to_thread(
                    LyricsSceneJob.load, lyrics_job_path, settings.ai.validation_buffer_ms
                )
                if lyrics_job:
                    job_state.lyrics = lyrics_job

                subtitle_job_path = settings.dir.tmp / f"{split.stem}.subtitles.{sanitized_subtitles_model}.json"
                subtitle_job = await asyncio.to_thread(
                    SubtitleJob.load, subtitle_job_path, settings.ai.validation_buffer_ms
                )
                if subtitle_job:
                    job_state.subtitles = subtitle_job

                # Check if the final stage (Subtitles) is already complete.
                is_complete = subtitle_job is not None and subtitle_job.is_complete(
                    settings.ai.verification_gap_seconds,
                    settings.ai.gap_verification_retries,
                )
                if is_complete:
                    sync_progress("done")
                    continue

                if subtitle_job is not None:
                    if subtitle_job.response:
                        logfire.warning(
                            f"Large gap(s) (>= {settings.ai.verification_gap_seconds}s) detected in "
                            f"'{split.stem}' from previous run. Triggering verification re-run..."
                        )

                # Determine the entry point for this segment.
                # We check requirements in order: Re-encode -> Upload -> Lyrics -> Subtitles.
                # If a step is required, we queue the job and 'continue' to the next segment.
                # The completion of that job will trigger the subsequent steps via the
                # callbacks defined above.

                # 1. Check Re-encode
                should_reencode = False
                if use_reencode:
                    if settings.split.re_encode.threshold_mb == 0:
                        should_reencode = True
                    else:
                        file_size_mb = split.stat().st_size / (1024 * 1024)
                        should_reencode = file_size_mb >= settings.split.re_encode.threshold_mb

                # Determine entry point and sync progress bars for skipped stages
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

                if use_upload:
                    sync_progress("upload")
                    job_state.upload = UploadFileJob(
                        name=split.stem,
                        python_file=split,
                        video_duration_ms=duration,
                    )
                    if upload_runner:
                        await upload_runner.add_job(job_state)
                    continue

                if use_lyrics:
                    if not job_state.lyrics or not job_state.lyrics.response:
                        sync_progress("lyrics")
                        job_state.lyrics = job_state.lyrics or LyricsSceneJob(
                            name=split.stem,
                            file=split,
                            video_duration_ms=duration,
                        )
                        job_state.lyrics.file = split
                        if scene_detection_runner:
                            await scene_detection_runner.add_job(job_state)
                        continue

                sync_progress("subtitles")
                job_state.subtitles = job_state.subtitles or SubtitleJob(
                    name=split.stem,
                    file=split,
                    video_duration_ms=duration,
                )
                job_state.subtitles.file = split
                if subtitle_runner:
                    await subtitle_runner.add_job(job_state)

            # Step 4: Start all runners and wait for them to complete
            refresh_task = asyncio.create_task(refresh_bars_loop())
            runners = [r for r in [reencode_runner, upload_runner, scene_detection_runner, subtitle_runner] if r]
            try:
                # Start all runners
                for runner in runners:
                    await runner.start()

                # Wait for all runners to complete their respective queues
                for runner in runners:
                    await runner.join()

                # Step 5: Assemble the final subtitle file.
                result = await asyncio.to_thread(stitch_subtitles, video_splits, settings)
                logfire.info(f"Done - {result.name}")
                return result
            finally:
                # Ensure all runners are shut down, even on cancellation
                for runner in runners:
                    await runner.shutdown()
                refresh_task.cancel()
                for b in bars.values():
                    b.close()


def main() -> None:
    """Parses CLI arguments and runs the main `ai_sub` function.

    This is the primary entry point for the command-line application. It uses
    `pydantic-settings.CliApp` to build a `Settings` object from command-line
    arguments, environment variables, and .env files, then executes the main
    pipeline and exits with the appropriate status code.
    """
    # Parse settings from CLI arguments, environment variables, and .env file.
    settings = CliApp.run(Settings)

    sys.exit(asyncio.run(ai_sub(settings)).value)


if __name__ == "__main__":
    main()
