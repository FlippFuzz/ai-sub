from __future__ import annotations

import asyncio
import json
import socket
import sys
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import logfire
from pydantic_settings import CliApp
from pysubs2 import SSAEvent, SSAFile

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import Settings
from ai_sub.data_models import (
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
)
from ai_sub.video import (
    get_video_duration_ms,
    get_working_encoder,
    reencode_video,
    split_video,
)


class ReEncodeJobRunner(JobRunner):
    """
    Worker that re-encodes video segments to a lower quality/different format.

    This is typically done to reduce file size before uploading to an API,
    saving bandwidth and potentially processing time.
    """

    def __init__(
        self,
        queue: asyncio.Queue[SegmentJobs],
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[SegmentJobs, Any], None],
        name: str = "reencode",
    ):
        super().__init__(queue, settings, max_workers, on_complete, name=name)

    async def process(self, job: SegmentJobs) -> None:
        """
        Re-encodes the video file specified in the job.

        After re-encoding, it returns None, triggering the on_complete callback in the runner.
        """
        reencode_job = job.reencode
        assert reencode_job is not None
        await asyncio.to_thread(
            reencode_video,
            reencode_job.input_file,
            reencode_job.output_file,
            reencode_job.fps,
            reencode_job.height,
            reencode_job.bitrate_kb,
            self.settings.split.re_encode.encoder or "libx264",
            reencode_job.duration_tolerance_ms,
        )

        logfire.info(
            f"{reencode_job.name} re-encoded to {reencode_job.output_file.name}"
        )


class UploadJobRunner(JobRunner):
    """
    Worker that uploads video files to the Gemini Files API.

    This runner is used when the AI model requires the file to be hosted
    on Google's servers (e.g., for Gemini models).
    """

    def __init__(
        self,
        queue: asyncio.Queue[SegmentJobs],
        settings: Settings,
        max_workers: int,
        uploader: GeminiFileUploader,
        on_complete: Callable[[SegmentJobs, Any], None],
        name: str = "upload",
    ):
        super().__init__(queue, settings, max_workers, on_complete, name=name)
        self.uploader = uploader

    async def process(self, job: SegmentJobs) -> Any:
        """
        Uploads the specified file using the `GeminiFileUploader`.

        Upon successful upload, it returns the file object, triggering the on_complete callback.
        """
        upload_job = job.upload
        assert upload_job is not None
        # Perform the file upload. This is a blocking operation.
        file = await asyncio.to_thread(
            self.uploader.upload_file, upload_job.python_file
        )
        logfire.info(f"{upload_job.name} uploaded")
        logfire.debug(f"File: {file}")
        return file


class LyricsSceneJobRunner(JobRunner):
    """
    Worker that executes the AI agent to detect lyrics and scenes.
    """

    def __init__(
        self,
        queue: asyncio.Queue[SegmentJobs],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], None],
        name: str = "lyrics",
    ):
        super().__init__(queue, settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(
            self.agent.model_name
        )

    async def process(self, job: SegmentJobs) -> None:
        """
        Invokes the AI agent to detect scenes.
        """
        lyrics_job = job.lyrics
        assert lyrics_job is not None
        if lyrics_job.response:
            logfire.info(
                f"Skipping lyrics generation for {lyrics_job.name} as valid response exists."
            )
            return

        assert lyrics_job.file is not None
        # agent.run uses nest_asyncio and run_sync, so we wrap it in a thread
        # to avoid blocking the main event loop.
        lyrics_job.response = await asyncio.to_thread(
            self.agent.run,
            get_lyrics_scenes_prompt(),
            lyrics_job.file,
            lyrics_job.video_duration_ms,
            LyricsSceneAiResponse,
        )

    async def post_process(self, job: SegmentJobs) -> None:
        """Saves the result to disk."""
        lyrics_job = job.lyrics
        assert lyrics_job is not None
        if lyrics_job.response:
            lyrics_job.lyrics_prompt_version = LYRICS_PROMPT_VERSION
            job_state_path = (
                self.settings.dir.tmp
                / f"{lyrics_job.name}.lyrics.{self.sanitized_model_name}.json"
            )
            await asyncio.to_thread(lyrics_job.save, job_state_path)


class SubtitleJobRunner(JobRunner):
    """
    Worker that executes the AI agent to generate subtitles.
    """

    def __init__(
        self,
        queue: asyncio.Queue[SegmentJobs],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SegmentJobs, Any], None] | None = None,
        name: str = "subtitles",
    ):
        super().__init__(queue, settings, max_workers, on_complete, name=name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(
            self.agent.model_name
        )

    async def process(self, job: SegmentJobs) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        subtitle_job = job.subtitles
        assert subtitle_job is not None
        if subtitle_job.response:
            logfire.info(
                f"Skipping subtitle generation for {subtitle_job.name} as valid response exists."
            )
            return

        lyrics_job = job.lyrics
        scene_response = lyrics_job.response if lyrics_job else None

        prompt = get_subtitle_prompt(scene_response)
        assert subtitle_job.file is not None
        subtitle_job.response = await asyncio.to_thread(
            self.agent.run,
            prompt,
            subtitle_job.file,
            subtitle_job.video_duration_ms,
            SubtitleAiResponse,
        )

    async def post_process(self, job: SegmentJobs) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        # Save the completed job state to a JSON file for persistence.
        subtitle_job = job.subtitles
        assert subtitle_job is not None

        # Also generate a subtitle file for this job for the user to view.
        if subtitle_job.response is not None:
            subtitle_job.subtitles_prompt_version = SUBTITLES_PROMPT_VERSION
            job_state_path = (
                self.settings.dir.tmp
                / f"{subtitle_job.name}.subtitles.{self.sanitized_model_name}.json"
            )
            await asyncio.to_thread(subtitle_job.save, job_state_path)
            sanitized_model = self.settings.ai.get_sanitized_model_name(
                self.settings.ai.model_subtitles
            )
            await asyncio.to_thread(
                subtitle_job.response.get_ssafile().save,
                str(
                    self.settings.dir.tmp / f"{subtitle_job.name}.{sanitized_model}.srt"
                ),
            )


def stitch_subtitles(
    video_splits: list[tuple[Path, int]], settings: Settings
) -> AiSubResult:
    """
    Assembles the final subtitle file from processed segments.

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
        sanitized_lyrics_model = settings.ai.get_sanitized_model_name(
            settings.ai.model_lyrics
        )
        sanitized_subtitles_model = settings.ai.get_sanitized_model_name(
            settings.ai.model_subtitles
        )

        chunks_to_skip = int(
            (settings.split.start_offset_min * 60) / settings.split.max_seconds
        )
        offset_ms = sum(duration for _, duration in video_splits[:chunks_to_skip])

        complete = True
        max_retries_exceeded = False

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            # Load the job result from the temporary JSON file.
            # We look for the final subtitle output
            job_path = (
                settings.dir.tmp
                / f"{video_path.stem}.subtitles.{sanitized_subtitles_model}.json"
            )
            job = SubtitleJob.load(job_path)
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

            # Add the duration of the current segment to the offset for the next one.
            offset_ms += video_duration_ms

            # Sort out max retries exceeded
            # The latest job file will have the highest retry count.
            # We can check in reverse order of stages.
            final_job_retry_count = 0
            if job:  # subtitle job
                final_job_retry_count = job.total_num_retries
            else:
                lyrics_job_path = (
                    settings.dir.tmp
                    / f"{video_path.stem}.lyrics.{sanitized_lyrics_model}.json"
                )
                lyrics_job = LyricsSceneJob.load(lyrics_job_path)
                if lyrics_job:
                    final_job_retry_count = lyrics_job.total_num_retries

            if final_job_retry_count >= settings.retry.max:
                max_retries_exceeded = True

        # Insert version and config, as a single SSAEvent at the beginning (0-1ms)
        # JSON curly braces {} are treated as formatting codes in SRT, so replace them.
        # Also exclude sensitive fields from being displayed
        settings_dict = settings.model_dump(
            mode="json",
            exclude={
                "input_video_file": True,
                "dir": True,
                "ai": {"google": {"key": True, "base_url": True}},
            },
        )
        state_info = {
            "ai_sub_version": version("ai-sub"),
            "lyrics_prompt_version": LYRICS_PROMPT_VERSION,
            "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
            "complete": complete,
            "max_retries_exceeded": max_retries_exceeded,
            "settings": settings_dict,
        }
        info_text = json.dumps(state_info, indent=2).replace("{", "(").replace("}", ")")
        all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info_text))

        # Make sure that the info_text don't overlap with the first actual subtitle
        if len(all_subtitles) > 1 and all_subtitles[1].start < 1:
            all_subtitles[1].start = 1

        all_subtitles.save(
            str(
                settings.dir.out
                / f"{settings.input_video_file.stem}.{sanitized_subtitles_model}.srt"
            )
        )

        if max_retries_exceeded:
            return AiSubResult.MAX_RETRIES_EXHAUSTED
        elif not complete:
            return AiSubResult.INCOMPLETE
        return AiSubResult.COMPLETE


async def ai_sub(settings: Settings, configure_logging: bool = True) -> AiSubResult:
    """
    Orchestrates the subtitle generation pipeline.

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
        # Configure Logfire for observability. This setup includes a console logger
        # and another configuration to instrument libraries like Pydantic AI and HTTPX
        # without sending their logs to the console.
        logfire.configure(
            console=logfire.ConsoleOptions(
                min_log_level=settings.log.level,
                include_timestamps=settings.log.timestamps,
            ),
            service_name=socket.gethostname(),
            service_version=version("ai-sub"),
            send_to_logfire="if-token-present",
            # Logfire scrubs by default (None). We pass False to disable it if configured.
            scrubbing=None if settings.log.scrub else False,
        )
        no_console_logfire = logfire.configure(
            local=True,
            console=False,
            send_to_logfire="if-token-present",
            # Logfire scrubs by default (None). We pass False to disable it if configured.
            scrubbing=None if settings.log.scrub else False,
        )
        no_console_logfire.instrument_pydantic_ai()
        no_console_logfire.instrument_httpx(capture_all=True)

    if settings.split.re_encode.enabled and not settings.split.re_encode.encoder:
        with logfire.span("Detecting hardware encoder"):
            settings.split.re_encode.encoder = get_working_encoder()
            logfire.info(f"Using encoder: {settings.split.re_encode.encoder}")

    # Initialize the AI Agent.
    # A custom wrapper is used to make handling rate limits and differences in models more cleanly
    agent_subtitles = RateLimitedAgentWrapper(settings, settings.ai.model_subtitles)
    use_lyrics = settings.thread.lyrics > 0
    agent_scene = (
        RateLimitedAgentWrapper(settings, settings.ai.model_lyrics, use_web_search=True)
        if use_lyrics
        else None
    )

    sanitized_lyrics_model = settings.ai.get_sanitized_model_name(
        settings.ai.model_lyrics
    )
    sanitized_subtitles_model = settings.ai.get_sanitized_model_name(
        settings.ai.model_subtitles
    )

    # Start the main application logic within a Logfire span for better tracing.
    with logfire.span(f"Generating subtitles for {settings.input_video_file.name}"):

        # Step 1: Split the input video into smaller segments.
        video_splits_paths = split_video(
            settings.input_video_file,
            settings.dir.tmp,
            settings.split.max_seconds,
            output_pattern="part_%03d",
            duration_tolerance_ms=settings.split.re_encode.duration_tolerance_ms,
        )
        video_splits: list[tuple[Path, int]] = [
            (path, get_video_duration_ms(path)) for path in video_splits_paths
        ]

        chunks_to_skip = int(
            (settings.split.start_offset_min * 60) / settings.split.max_seconds
        )
        splits_to_process = video_splits
        if chunks_to_skip > 0:
            skipped_splits = video_splits[:chunks_to_skip]
            initial_offset_ms = sum(duration for _, duration in skipped_splits)
            splits_to_process = video_splits[chunks_to_skip:]
            logfire.info(
                f"Skipping first {chunks_to_skip} chunks ({len(skipped_splits)} segments, "
                f"{initial_offset_ms}ms) due to start_offset_min={settings.split.start_offset_min}"
            )

        # Step 2: Configure the job processing pipeline.

        # Initialize data structures for concurrent processing.
        # Deques are used as thread-safe queues for managing jobs.
        reencode_jobs_queue: asyncio.Queue[SegmentJobs] = asyncio.Queue()
        gemini_upload_jobs_queue: asyncio.Queue[SegmentJobs] = asyncio.Queue()
        scene_detection_jobs_queue: asyncio.Queue[SegmentJobs] = asyncio.Queue()
        subtitle_jobs_queue: asyncio.Queue[SegmentJobs] = asyncio.Queue()

        use_reencode = settings.split.re_encode.enabled
        is_google_sub = agent_subtitles.is_google()
        is_google_scene = agent_scene.is_google() if agent_scene else False
        use_upload = (
            is_google_sub or is_google_scene
        ) and settings.ai.google.use_files_api

        reencode_runner: ReEncodeJobRunner | None = None
        upload_runner: UploadJobRunner | None = None
        scene_detection_runner: LyricsSceneJobRunner | None = None
        subtitle_runner: SubtitleJobRunner | None = None

        # Define callbacks
        # These functions handle the transition between pipeline stages.
        # When a job completes, the next required job is created and queued.
        def on_stage_complete(stage: str, job: SegmentJobs, result: Any) -> None:
            """Handles the transition between pipeline stages."""
            file_handle: Any = result
            duration_ms: int = 0
            name: str = ""

            # Extract data based on the completed stage
            if stage == "reencode":
                assert job.reencode is not None
                name = job.reencode.name
                file_handle = job.reencode.output_file
                duration_ms = get_video_duration_ms(file_handle)
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
                file_handle = job.lyrics.file
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

            # Queue next job
            if next_stage == "upload":
                job.upload = UploadFileJob(
                    name=name,
                    python_file=file_handle,
                    video_duration_ms=duration_ms,
                )
                gemini_upload_jobs_queue.put_nowait(job)

            elif next_stage == "lyrics":
                existing_lyrics = job.lyrics
                new_lyrics_job = LyricsSceneJob(
                    name=name, file=file_handle, video_duration_ms=duration_ms
                )
                if existing_lyrics:
                    new_lyrics_job.total_num_retries = existing_lyrics.total_num_retries
                    if existing_lyrics.response:
                        new_lyrics_job.response = existing_lyrics.response
                        new_lyrics_job.lyrics_prompt_version = (
                            existing_lyrics.lyrics_prompt_version
                        )

                job.lyrics = new_lyrics_job
                scene_detection_jobs_queue.put_nowait(job)

            elif next_stage == "subtitles":
                existing_subs = job.subtitles
                new_subs_job = SubtitleJob(
                    name=name, file=file_handle, video_duration_ms=duration_ms
                )
                if existing_subs:
                    new_subs_job.total_num_retries = existing_subs.total_num_retries
                    if existing_subs.response:
                        new_subs_job.response = existing_subs.response
                        new_subs_job.subtitles_prompt_version = (
                            existing_subs.subtitles_prompt_version
                        )

                job.subtitles = new_subs_job
                subtitle_jobs_queue.put_nowait(job)

        # Setup reencode_runner
        if use_reencode:
            reencode_runner = ReEncodeJobRunner(
                reencode_jobs_queue,
                settings,
                settings.thread.re_encode,
                on_complete=partial(on_stage_complete, "reencode"),
            )

        # Setup upload_runner
        if use_upload:
            upload_runner = UploadJobRunner(
                gemini_upload_jobs_queue,
                settings,
                settings.thread.uploads,
                uploader=GeminiFileUploader(settings),
                on_complete=partial(on_stage_complete, "upload"),
            )

        # Setup scene_detection_runner
        if use_lyrics:
            assert agent_scene is not None
            scene_detection_runner = LyricsSceneJobRunner(
                scene_detection_jobs_queue,
                settings,
                settings.thread.lyrics,
                agent_scene,
                on_complete=partial(on_stage_complete, "lyrics"),
            )

        # Setup subtitle_runner
        subtitle_runner = SubtitleJobRunner(
            subtitle_jobs_queue,
            settings,
            settings.thread.subtitles,
            agent_subtitles,
            on_complete=None,
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
            lyrics_job_path = (
                settings.dir.tmp / f"{split.stem}.lyrics.{sanitized_lyrics_model}.json"
            )
            lyrics_job = LyricsSceneJob.load(lyrics_job_path)
            if lyrics_job:
                job_state.lyrics = lyrics_job

            subtitle_job_path = (
                settings.dir.tmp
                / f"{split.stem}.subtitles.{sanitized_subtitles_model}.json"
            )
            subtitle_job = SubtitleJob.load(subtitle_job_path)
            if subtitle_job:
                job_state.subtitles = subtitle_job

            # Check if the final stage (Subtitles) is already complete.
            if subtitle_job and subtitle_job.response:
                continue

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
                    should_reencode = (
                        file_size_mb >= settings.split.re_encode.threshold_mb
                    )

            if should_reencode:
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
                reencode_jobs_queue.put_nowait(job_state)
                continue

            # 2. Check Upload
            if use_upload:
                job_state.upload = UploadFileJob(
                    name=split.stem,
                    python_file=split,
                    video_duration_ms=duration,
                )
                gemini_upload_jobs_queue.put_nowait(job_state)
                continue

            # 3. Check Lyrics
            if use_lyrics:
                if not job_state.lyrics:
                    # Fall-through: If execution reaches here, it means previous stages (Re-encode/Upload)
                    # were either disabled or skipped (e.g. file size < threshold).
                    # Therefore, we use the original local split file as the input.
                    job_state.lyrics = LyricsSceneJob(
                        name=split.stem,
                        file=split,
                        video_duration_ms=duration,
                    )
                    scene_detection_jobs_queue.put_nowait(job_state)
                    continue
                elif job_state.lyrics.file is None:
                    # Resumption: 'file' is excluded from JSON save. Restore local split path.
                    job_state.lyrics.file = split
                    if not job_state.lyrics.response:
                        scene_detection_jobs_queue.put_nowait(job_state)
                        continue

            # 4. Subtitles
            if not job_state.subtitles:
                # Fall-through: If execution reaches here, it means previous stages (Re-encode/Upload)
                # were either disabled or skipped (e.g. file size < threshold).
                # Therefore, we use the original local split file as the input.
                job_state.subtitles = SubtitleJob(
                    name=split.stem,
                    file=split,
                    video_duration_ms=duration,
                )
                subtitle_jobs_queue.put_nowait(job_state)
            elif job_state.subtitles.file is None:
                # Resumption: 'file' is excluded from JSON save. Restore local split path.
                job_state.subtitles.file = split
                subtitle_jobs_queue.put_nowait(job_state)

        # Step 4: Start all runners and wait for them to complete
        # Start runners
        if reencode_runner:
            reencode_runner.start()
        if upload_runner:
            upload_runner.start()
        if scene_detection_runner:
            scene_detection_runner.start()
        subtitle_runner.start()

        # Wait for runners to complete and signal as needed
        if reencode_runner:
            await reencode_jobs_queue.join()
            await reencode_runner.shutdown()

        if upload_runner:
            await gemini_upload_jobs_queue.join()
            await upload_runner.shutdown()

        if scene_detection_runner:
            await scene_detection_jobs_queue.join()
            await scene_detection_runner.shutdown()

        await subtitle_jobs_queue.join()
        await subtitle_runner.shutdown()

        # Step 5: Assemble the final subtitle file.
        result = stitch_subtitles(video_splits, settings)

        logfire.info(f"Done - {result.name}")
        return result


def main() -> None:
    """
    Parses CLI arguments and runs the main `ai_sub` function.

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
