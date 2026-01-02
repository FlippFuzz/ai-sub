import socket
from collections import deque
from importlib.metadata import version
from pathlib import Path
from threading import Event
from typing import Any, Callable

import logfire
from pydantic import ValidationError
from pysubs2 import SSAEvent, SSAFile

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import Settings
from ai_sub.data_models import ReEncodingJob, SubtitleJob, UploadFileJob
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.job_runner import JobRunner
from ai_sub.prompt import PROMPT
from ai_sub.video import get_video_duration_ms, reencode_video, split_video


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
        next_queue: deque[Any],
        next_job_factory: Callable[[Path, int], Any],
        stop_events: list[Event] = [],
        name: str = "ReEncode",
    ):
        super().__init__(queue, settings, max_workers, stop_events, name)
        self.next_queue = next_queue
        self.next_job_factory = next_job_factory

    def process(self, job: ReEncodingJob) -> None:
        """
        Re-encodes the video file specified in the job.

        After re-encoding, it calculates the new duration, creates the next job
        (either an UploadJob or a SubtitleJob) using the factory, and adds it
        to the next queue.
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

            duration = get_video_duration_ms(job.output_file)
            next_job = self.next_job_factory(job.output_file, duration)
            self.next_queue.append(next_job)
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
        jobs_queue: deque[SubtitleJob],
        stop_events: list[Event] = [],
        name: str = "Upload",
    ):
        super().__init__(queue, settings, max_workers, stop_events, name)
        self.uploader = uploader
        self.jobs_queue = jobs_queue

    def process(self, job: UploadFileJob) -> None:
        """
        Uploads the specified file using the `GeminiFileUploader`.

        Upon successful upload, it creates a `SubtitleJob` with the uploaded file handle
        and adds it to the subtitle jobs queue.
        """
        with logfire.span(f"Uploading {job.python_file.name}"):
            # Perform the file upload. This is a blocking operation.
            file = self.uploader.upload_file(job.python_file)
            # On success, create a SubtitleJob and add it to the next queue.
            self.jobs_queue.append(
                SubtitleJob(
                    name=job.python_file.stem,
                    file=file,
                    video_duration_ms=job.video_duration_ms,
                )
            )
            logfire.info(f"{job.python_file.name} uploaded")


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
        stop_events: list[Event] = [],
        name: str = "Subtitle",
    ):
        super().__init__(queue, settings, max_workers, stop_events, name)
        self.agent = agent

    def process(self, job: SubtitleJob) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        with logfire.span(f"Subtitling {job.name}"):
            job.response = self.agent.run(PROMPT, job.file, job.video_duration_ms)
            logfire.info(f"{job.name} subtitled")

    def post_process(self, job: SubtitleJob) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        # Save the completed job state to a JSON file for persistence.
        job.save(self.settings.dir.tmp / f"{job.name}.json")
        if job.response is not None:
            # Also generate a subtitle file for this job for the user to view.
            job.response.get_ssafile().save(
                str(self.settings.dir.tmp / f"{job.name}.srt")
            )


def stitch_subtitles(video_splits: list[tuple[Path, int]], settings: Settings) -> None:
    """
    Assembles the final subtitle file from processed segments.

    It iterates through the expected video segments, loads the corresponding
    processed subtitle jobs, and concatenates them. Timestamps are shifted
    appropriately to match the original video's timeline.

    Args:
        video_splits: A list of tuples containing the path and duration of each video segment.
        settings: The application's configuration settings.
    """
    with logfire.span("Producing final SRT file"):
        all_subtitles = SSAFile()
        offset_ms = 0
        complete = True
        max_retries_exceeded = False

        for video_path, video_duration_ms in video_splits:
            # Load the job result from the temporary JSON file.
            job = SubtitleJob.load_or_return_new(
                settings.dir.tmp / f"{video_path.stem}.json",
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
                complete = False

            # Add the duration of the current segment to the offset for the next one.
            offset_ms += video_duration_ms

            # Sort out max retries exceeded
            max_retries_exceeded = (
                max_retries_exceeded or job.total_num_retries >= settings.retry.max
            )

        # Insert version and config, as a single SSAEvent at the beginning (0-1ms)
        # JSON curly braces {} are treated as formatting codes in SRT, so replace them.
        # Also exclude sensitive fields from being displayed
        config_json = (
            settings.model_dump_json(
                indent=2,
                exclude={
                    "input_video_file": True,
                    "dir": True,
                    "ai": {"google": {"key": True, "base_url": True}},
                },
            )
            .replace("{", "(")
            .replace("}", ")")
        )
        info_text = f"Generated by ai-sub version: {version('ai-sub')}\nComplete: {complete}\nMax Retries Exceeded: {max_retries_exceeded}\nConfig: {config_json}"
        all_subtitles.insert(0, SSAEvent(start=0, end=1, text=info_text))

        # Make sure that the info_text don't overlap with the first actual subtitle
        if len(all_subtitles) > 1 and all_subtitles[1].start < 1:
            all_subtitles[1].start = 1

        all_subtitles.save(
            str(settings.dir.out / f"{settings.input_video_file.stem}.srt")
        )


def main():
    """
    Main function to generate subtitles for a video file.

    This function orchestrates the entire subtitling process. It starts by parsing
    command-line arguments and environment variables to configure the application settings.
    It then sets up logging and initializes the appropriate AI agent based on the
    selected model.

    The core workflow is as follows:
    1.  The input video is split into smaller, manageable segments.
    2.  Jobs are created for each segment that hasn't been processed previously.
    3.  If using a Gemini model, video segments are uploaded concurrently.
    4.  Subtitle generation is performed concurrently for all segments.
    5.  The application waits for all processing to complete.
    6.  Finally, it stitches together the subtitles from all segments, adjusting
        timestamps to create a single, synchronized subtitle file for the original video.
    """
    # Parse settings from CLI arguments, environment variables, and .env file.
    try:
        settings = Settings()  # pyright: ignore
    except ValidationError as ve:
        print(ve)
        exit(-1)

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
        for split, video_duration_ms in video_splits:
            possibleJob = SubtitleJob.load_or_return_new(
                settings.dir.tmp / f"{split.stem}.json",
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

        # Setup reencode_runner
        if use_reencode:
            if use_upload:
                next_queue = gemini_upload_jobs_queue

                def create_upload_job(p: Path, d: int):
                    return UploadFileJob(python_file=p, video_duration_ms=d)

                next_factory = create_upload_job
            else:
                next_queue = subtitle_jobs_queue

                def create_subtitle_job(p: Path, d: int):
                    return SubtitleJob(name=p.stem, file=p, video_duration_ms=d)

                next_factory = create_subtitle_job

            reencode_runner = ReEncodeJobRunner(
                reencode_jobs_queue,
                settings,
                settings.thread.re_encode,
                next_queue=next_queue,
                next_job_factory=next_factory,
            )

        # Setup upload_runner
        if use_upload:
            stop_events = [reencode_complete_event] if use_reencode else []
            upload_runner = UploadJobRunner(
                gemini_upload_jobs_queue,
                settings,
                settings.thread.uploads,
                uploader=GeminiFileUploader(settings),
                jobs_queue=subtitle_jobs_queue,
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
        stitch_subtitles(video_splits, settings)

        logfire.info("Done")


if __name__ == "__main__":
    main()
