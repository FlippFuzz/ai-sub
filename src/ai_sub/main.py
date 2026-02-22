import socket
import sys
from collections import deque
from importlib.metadata import version
from pathlib import Path
from threading import Event
from typing import Any, Callable

import logfire
from pydantic_settings import CliApp
from pysubs2 import SSAEvent, SSAFile

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import Settings
from ai_sub.data_models import (
    AiSubResult,
    ReEncodingJob,
    SubtitleGenerationState,
    SubtitleJob,
    UploadFileJob,
)
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.job_runner import JobRunner
from ai_sub.prompt import SUBTITLES_PROMPT, SUBTITLES_PROMPT_VERSION
from ai_sub.video import (
    get_video_duration_ms,
    get_working_encoder,
    reencode_video,
    split_video,
)


class ReEncodeJobRunner(JobRunner[ReEncodingJob]):
    """
    Worker that re-encodes video segments to a lower quality/different format.

    This is typically done to reduce file size before uploading to an API,
    saving bandwidth and potentially processing time.
    """

    def __init__(
        self,
        queue: deque[ReEncodingJob],
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[ReEncodingJob, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "ReEncode",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)

    def process(self, job: ReEncodingJob) -> None:
        """
        Re-encodes the video file specified in the job.

        After re-encoding, it returns None, triggering the on_complete callback in the runner.
        """
        with logfire.span(f"Re-encoding {job.input_file.name}"):
            reencode_video(
                job.input_file,
                job.output_file,
                job.fps,
                job.height,
                job.bitrate_kb,
                self.settings.split.re_encode.encoder or "libx264",
            )

            logfire.info(f"{job.input_file.name} re-encoded to {job.output_file.name}")


class UploadJobRunner(JobRunner[UploadFileJob]):
    """
    Worker that uploads video files to the Gemini Files API.

    This runner is used when the AI model requires the file to be hosted
    on Google's servers (e.g., for Gemini models).
    """

    def __init__(
        self,
        queue: deque[UploadFileJob],
        settings: Settings,
        max_workers: int,
        uploader: GeminiFileUploader,
        on_complete: Callable[[UploadFileJob, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "Upload",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.uploader = uploader

    def process(self, job: UploadFileJob) -> Any:
        """
        Uploads the specified file using the `GeminiFileUploader`.

        Upon successful upload, it returns the file object, triggering the on_complete callback.
        """
        with logfire.span(f"Uploading {job.python_file.name}"):
            # Perform the file upload. This is a blocking operation.
            file = self.uploader.upload_file(job.python_file)
            logfire.info(f"{job.python_file.name} uploaded")
            return file


class SubtitleJobRunner(JobRunner[SubtitleJob]):
    """
    Worker that executes the AI agent to generate subtitles for a video segment.
    """

    def __init__(
        self,
        queue: deque[SubtitleJob],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        stop_events: list[Event] | None = None,
        name: str = "Subtitle",
    ):
        super().__init__(queue, settings, max_workers, None, stop_events, name)
        self.agent = agent

    def process(self, job: SubtitleJob) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        with logfire.span(f"Subtitling {job.name}"):
            job.response = self.agent.run(
                SUBTITLES_PROMPT, job.file, job.video_duration_ms
            )
            logfire.info(f"{job.name} subtitled")

    def post_process(self, job: SubtitleJob) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        # Save the completed job state to a JSON file for persistence.
        sanitized_model = self.settings.ai.get_sanitized_model_name()
        job.save(self.settings.dir.tmp / f"{job.name}.{sanitized_model}.json")
        if job.response is not None:
            # Also generate a subtitle file for this job for the user to view.
            job.response.get_ssafile().save(
                str(self.settings.dir.tmp / f"{job.name}.{sanitized_model}.srt")
            )


def stitch_subtitles(
    video_splits: list[tuple[Path, int]], settings: Settings
) -> SubtitleGenerationState:
    """
    Assembles the final subtitle file from processed segments.

    It iterates through the expected video segments, loads the corresponding
    processed subtitle jobs, and concatenates them. Timestamps are shifted
    appropriately to match the original video's timeline.

    Args:
        video_splits: A list of tuples containing the path and duration of each video segment.
        settings: The application's configuration settings.

    Returns:
        SubtitleGenerationState: The final state of the subtitle generation process.
    """
    with logfire.span("Producing final SRT file"):
        all_subtitles = SSAFile()
        offset_ms = 0

        state = SubtitleGenerationState(
            ai_sub_version=version("ai-sub"),
            subtitles_prompt_version=SUBTITLES_PROMPT_VERSION,
            settings=settings,
        )

        sanitized_model = settings.ai.get_sanitized_model_name()

        for video_path, video_duration_ms in video_splits:
            # Load the job result from the temporary JSON file.
            job = SubtitleJob.load_or_return_new(
                settings.dir.tmp / f"{video_path.stem}.{sanitized_model}.json",
                video_path.stem,
                video_path,
                video_duration_ms,
            )
            if job.response is not None:
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
                state.complete = False

            # Add the duration of the current segment to the offset for the next one.
            offset_ms += video_duration_ms

            # Sort out max retries exceeded
            if job.total_num_retries >= settings.retry.max:
                state.max_retries_exceeded = True

        # Insert version and config, as a single SSAEvent at the beginning (0-1ms)
        # JSON curly braces {} are treated as formatting codes in SRT, so replace them.
        # Also exclude sensitive fields from being displayed
        info_text = (
            state.model_dump_json(
                indent=2,
                exclude={
                    "settings": {
                        "input_video_file": True,
                        "dir": True,
                        "ai": {"google": {"key": True, "base_url": True}},
                    },
                },
            )
            .replace("{", "(")
            .replace("}", ")")
        )
        all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info_text))

        # Make sure that the info_text don't overlap with the first actual subtitle
        if len(all_subtitles) > 1 and all_subtitles[1].start < 1:
            all_subtitles[1].start = 1

        all_subtitles.save(
            str(
                settings.dir.out
                / f"{settings.input_video_file.stem}.{sanitized_model}.srt"
            )
        )
        return state


def ai_sub(settings: Settings, configure_logging: bool = True) -> AiSubResult:
    """
    Runs the subtitle generation process.

    The core workflow is as follows:
    1.  The input video is split into smaller, manageable segments.
    2.  Jobs are created for each segment that hasn't been processed previously.
    3.  If using a Gemini model, video segments are uploaded concurrently.
    4.  Subtitle generation is performed concurrently for all segments.
    5.  The application waits for all processing to complete.
    6.  Finally, it stitches together the subtitles from all segments, adjusting
        timestamps to create a single, synchronized subtitle file for the original video.

    Returns:
        AiSubResult: The result code indicating the success or failure state of the operation.
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
    agent = RateLimitedAgentWrapper(settings)

    # Start the main application logic within a Logfire span for better tracing.
    with logfire.span(f"Generating subtitles for {settings.input_video_file.name}"):

        # Step 1: Split the input video into smaller segments.
        video_splits_paths = split_video(
            settings.input_video_file,
            settings.dir.tmp,
            settings.split.max_seconds,
            output_pattern="part_%03d",
        )
        video_splits: list[tuple[Path, int]] = [
            (path, get_video_duration_ms(path)) for path in video_splits_paths
        ]

        # Step 2: Filter out segments that have already been processed.
        # This allows the process to be resumed. It checks for the existence of a
        # .json file which indicates a completed (or failed) job.
        videos_to_work_on: list[tuple[Path, int]] = []
        sanitized_model = settings.ai.get_sanitized_model_name()
        for split, video_duration_ms in video_splits:
            possibleJob = SubtitleJob.load_or_return_new(
                settings.dir.tmp / f"{split.stem}.{sanitized_model}.json",
                split.stem,
                split,
                video_duration_ms,
            )
            if possibleJob.response is None:
                videos_to_work_on.append((split, video_duration_ms))

        # Step 3: Initialize data structures for concurrent processing.
        # Deques are used as thread-safe queues for managing jobs.
        reencode_jobs_queue: deque[ReEncodingJob] = deque()
        gemini_upload_jobs_queue: deque[UploadFileJob] = deque()
        subtitle_jobs_queue: deque[SubtitleJob] = deque()

        reencode_complete_event = Event()
        gemini_upload_complete_event = Event()

        reencode_runner: ReEncodeJobRunner | None = None
        upload_runner: UploadJobRunner | None = None
        subtitle_runner: SubtitleJobRunner | None = None

        use_reencode = settings.split.re_encode.enabled
        use_upload = agent.is_google() and settings.ai.google.use_files_api

        # Define callbacks
        def on_reencode_complete(job: ReEncodingJob, _: Any) -> None:
            duration = get_video_duration_ms(job.output_file)
            if use_upload:
                gemini_upload_jobs_queue.append(
                    UploadFileJob(
                        python_file=job.output_file, video_duration_ms=duration
                    )
                )
            else:
                subtitle_jobs_queue.append(
                    SubtitleJob(
                        name=job.output_file.stem,
                        file=job.output_file,
                        video_duration_ms=duration,
                    )
                )

        def on_upload_complete(job: UploadFileJob, file: Any) -> None:
            subtitle_jobs_queue.append(
                SubtitleJob(
                    name=job.python_file.stem,
                    file=file,
                    video_duration_ms=job.video_duration_ms,
                )
            )

        # Setup reencode_runner
        if use_reencode:
            reencode_runner = ReEncodeJobRunner(
                reencode_jobs_queue,
                settings,
                settings.thread.re_encode,
                on_complete=on_reencode_complete,
            )

        # Setup upload_runner
        if use_upload:
            stop_events = [reencode_complete_event] if use_reencode else []
            upload_runner = UploadJobRunner(
                gemini_upload_jobs_queue,
                settings,
                settings.thread.uploads,
                uploader=GeminiFileUploader(settings),
                on_complete=on_upload_complete,
                stop_events=stop_events,
            )

        # Setup subtitle_runner
        subtitle_stop_events = []
        if use_upload:
            subtitle_stop_events.append(gemini_upload_complete_event)
        elif use_reencode:
            subtitle_stop_events.append(reencode_complete_event)

        subtitle_runner = SubtitleJobRunner(
            subtitle_jobs_queue,
            settings,
            settings.thread.subtitles,
            agent,
            stop_events=subtitle_stop_events,
        )

        # Step 4: Populate the initial job queues.
        if use_reencode:
            # Create a directory for re-encoded files to avoid name collisions
            # and preserve the file stem for stitching.
            reencode_dir = settings.dir.tmp / "reencoded"
            reencode_dir.mkdir(exist_ok=True)

            for input_file, _ in videos_to_work_on:
                # We want to keep the same stem (e.g. "part_000") so that the
                # SubtitleJob is named correctly for stitching later.
                output_file = reencode_dir / input_file.with_suffix(".mov").name

                reencode_jobs_queue.append(
                    ReEncodingJob(
                        input_file=input_file,
                        output_file=output_file,
                        fps=settings.split.re_encode.fps,
                        height=settings.split.re_encode.height,
                        bitrate_kb=settings.split.re_encode.bitrate_kb,
                    )
                )
        elif use_upload:
            # We start with the gemini file upload tasks
            gemini_upload_jobs_queue.extend(
                UploadFileJob(python_file=path, video_duration_ms=duration)
                for path, duration in videos_to_work_on
            )
        else:
            # We start with the subtitle generation tasks
            subtitle_jobs_queue.extend(
                SubtitleJob(name=path.stem, file=path, video_duration_ms=duration)
                for path, duration in videos_to_work_on
            )

        # Step 5: Start all runners and wait for them to complete
        # Start runners
        if reencode_runner:
            reencode_runner.start()
        if upload_runner:
            upload_runner.start()
        subtitle_runner.start()

        # Wait for runners to complete and signal as needed
        if reencode_runner:
            reencode_runner.wait()
            reencode_complete_event.set()

        if upload_runner:
            upload_runner.wait()
            gemini_upload_complete_event.set()

        subtitle_runner.wait()

        # Shutdown runners when all done
        if reencode_runner:
            reencode_runner.shutdown()
        if upload_runner:
            upload_runner.shutdown()
        subtitle_runner.shutdown()

        # Step 6: Assemble the final subtitle file.
        # Recalculate durations as they might have changed or were unknown during re-encoding
        state = stitch_subtitles(video_splits, settings)

        # Return the final result
        result = AiSubResult.COMPLETE
        if state.max_retries_exceeded:
            result = AiSubResult.MAX_RETRIES_EXHAUSTED
        elif not state.complete:
            result = AiSubResult.INCOMPLETE

        logfire.info(f"Done - {result.name}")
        return result


if __name__ == "__main__":
    # Parse settings from CLI arguments, environment variables, and .env file.
    settings = CliApp.run(Settings)

    sys.exit(ai_sub(settings).value)
