import socket
from collections import deque
from importlib.metadata import version
from pathlib import Path
from threading import Event

import logfire
from pydantic import ValidationError
from pydantic_ai.providers.google import GoogleProvider
from pysubs2 import SSAEvent, SSAFile

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import Settings
from ai_sub.data_models import SubtitleJob, UploadFileJob
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.job_runner import JobRunner
from ai_sub.prompt import PROMPT
from ai_sub.video import get_video_duration_ms, split_video


class UploadJobRunner(JobRunner[UploadFileJob]):
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
            logfire.info(f"{job.name} done")

    def post_process(self, job: SubtitleJob) -> None:
        """
        Saves the result (or partial state) to disk.
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

    Args:
        video_splits: A list of tuples containing the path and duration of each video segment.
        settings: The application's configuration settings.
    """
    with logfire.span("Producing final SRT file"):
        all_subtitles = SSAFile()
        offset_ms = 0
        complete = True

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
        info_text = f"Generated by ai-sub version: {version('ai-sub')}\nComplete: {complete}\nConfig: {config_json}"
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
        # Step 1: Split the input video into smaller segments based on the configured duration.
        video_splits_paths: list[Path] = split_video(
            settings.input_video_file,
            settings.dir.tmp,
            settings.split.max_seconds,
            settings.split.re_encode.enabled,
            settings.split.re_encode.fps,
            settings.split.re_encode.height,
            settings.split.re_encode.bitrate_kb,
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
        gemini_upload_jobs_queue: deque[UploadFileJob] = deque()
        subtitle_jobs_queue: deque[SubtitleJob] = deque()
        gemini_upload_complete_event = Event()
        upload_runner: UploadJobRunner | None = None

        # Step 4: Populate the initial job queues.
        # If using a Google model, video parts must be uploaded first. These jobs
        # are added to a dedicated upload queue.
        if agent.is_google() and settings.ai.google.use_files_api:
            gemini_upload_jobs_queue.extend(
                UploadFileJob(python_file=path, video_duration_ms=duration)
                for path, duration in videos_to_work_on
            )
            googleProvider = GoogleProvider(
                api_key=(
                    settings.ai.google.key.get_secret_value()
                    if settings.ai.google.key
                    else None
                ),
                http_client=None,
                base_url=(
                    str(settings.ai.google.base_url)
                    if settings.ai.google.base_url
                    else None
                ),
            )
            uploader = GeminiFileUploader(
                googleProvider, cache_ttl_seconds=settings.ai.google.file_cache_ttl
            )
            upload_runner = UploadJobRunner(
                gemini_upload_jobs_queue,
                settings,
                settings.thread.uploads,
                uploader=uploader,
                jobs_queue=subtitle_jobs_queue,
            )
            upload_runner.start()
        else:
            # For other models (e.g., OpenAI, custom), files are handled locally or
            # as binary data, so we can directly populate the subtitle jobs queue.
            subtitle_jobs_queue.extend(
                SubtitleJob(name=path.stem, file=path, video_duration_ms=duration)
                for path, duration in videos_to_work_on
            )

        # Step 5: Start the subtitle generation workers.
        # A thread pool is created to process subtitle jobs concurrently.
        max_workers = settings.thread.subtitles
        subtitle_runner = SubtitleJobRunner(
            subtitle_jobs_queue,
            settings,
            max_workers,
            agent,
            stop_events=[gemini_upload_complete_event],
        )
        subtitle_runner.start()

        # Step 6: Wait for all jobs to complete.
        # First, wait for all file upload jobs (if any) to finish.
        if upload_runner:
            upload_runner.wait()
        # Signal that no more upload jobs will be added. This tells the subtitle
        # workers they can exit once their queue is empty.
        gemini_upload_complete_event.set()

        # Then, wait for all the subtitle generation jobs to finish.
        subtitle_runner.wait()

        # Shutdown executors
        if upload_runner:
            upload_runner.shutdown()
        subtitle_runner.shutdown()

        # Step 7: Assemble the final subtitle file.
        stitch_subtitles(video_splits, settings)

        logfire.info("Done")


if __name__ == "__main__":
    main()
