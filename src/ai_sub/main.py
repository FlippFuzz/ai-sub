import concurrent.futures
import socket
from collections import deque
from importlib.metadata import version
from pathlib import Path
from threading import Event
from time import sleep

import logfire
from pydantic import ValidationError
from pydantic_ai.providers.google import GoogleProvider
from pysubs2 import SSAEvent, SSAFile

from ai_sub.agent_wrapper import RateLimitedAgentWrapper
from ai_sub.config import Settings
from ai_sub.data_models import SubtitleJob, UploadFileJob
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.prompt import PROMPT
from ai_sub.video import get_video_duration_ms, split_video


def upload_file_job(
    upload_queue: deque[UploadFileJob],
    uploader: GeminiFileUploader,
    jobs_queue: deque,
    settings: Settings,
) -> None:
    """
    Worker function that processes the upload queue.

    This function runs in a loop, continuously pulling jobs from the `upload_queue`.
    For each job, it uploads the specified file using the `GeminiFileUploader`.
    If the upload is successful, it creates a new `SubtitleJob` and adds it
    to the `jobs_queue` for the next stage of processing. The loop terminates
    when the `upload_queue` is empty.

    Args:
        upload_queue: A deque containing `UploadFileJob` objects to be processed.
        uploader: An instance of `GeminiFileUploader` to handle file uploads.
        jobs_queue: A deque to which new `SubtitleJob` objects will be added.
        settings: The application's configuration settings.
    """
    while True:
        job = None
        try:
            # Attempt to get a job from the left of the queue.
            job = upload_queue.popleft()

            # Increment the retry counter for this specific run.
            job.run_num_retries += 1

            with logfire.span(f"Uploading {job.python_file.name}"):
                # Perform the file upload. This is a blocking operation.
                file = uploader.upload_file(job.python_file)
                # On success, create a SubtitleJob and add it to the next queue.
                jobs_queue.append(
                    SubtitleJob(
                        name=job.python_file.stem,
                        file=file,
                        video_duration_ms=job.video_duration_ms,
                    )
                )
                logfire.info(f"{job.python_file.name} uploaded")
        except IndexError:
            # The queue is empty, so this worker thread can safely terminate.
            break
        except Exception:
            logfire.exception("Exception while uploading file")
            if job is not None:
                if job.run_num_retries < settings.retry.run:
                    # If an error occurred and retries are allowed, put the job back at the front of the queue.
                    sleep(settings.retry.delay)
                    upload_queue.insert(0, job)


def subtitle_job(
    jobs_queue: deque[SubtitleJob],
    agent: RateLimitedAgentWrapper,
    gemini_upload_complete_event: Event,
    settings: Settings,
) -> None:
    """
    Worker function that processes subtitle generation jobs from a queue.

    This function runs in a continuous loop, pulling `SubtitleJob` instances from
    the `jobs_queue`. For each job, it constructs a prompt for the AI agent based
    on the configured model type and the video file associated with the job.
    It then invokes the AI agent to generate subtitles, saves the result, and
    handles retries for failed jobs.

    The worker terminates when the `jobs_queue` is empty and the
    `gemini_upload_complete_event` is set, indicating that no new jobs will be added.

    Args:
        jobs_queue (deque[SubtitleJob]): A queue of subtitle jobs to process.
        agent (Agent): The AI agent responsible for generating subtitles.
        gemini_upload_complete_event (Event): An event that signals when all Gemini
                                              file uploads are complete.
        settings (Settings): The application's configuration settings.
    """
    while True:
        job = None
        try:
            # Get a job from the left of the queue. This will raise an IndexError if empty.
            job = jobs_queue.popleft()

            # Increment retry counts
            job.run_num_retries += 1
            job.total_num_retries += 1

            with logfire.span(f"Working on {job.name}"):
                job.response = agent.run(PROMPT, job.file, job.video_duration_ms)

                # Save the completed job state to a JSON file for persistence.
                job.save(settings.dir.tmp / f"{job.name}.json")
                # Also generate a subtitle file for this job for the user to view.
                job.response.get_ssafile().save(
                    str(settings.dir.tmp / f"{job.name}.srt")
                )
                logfire.info(f"{job.name} done")

        except IndexError:
            # This exception means the jobs_queue is currently empty.
            if gemini_upload_complete_event.is_set():
                # If the upload queue is also finished, no more jobs will be added.
                # The worker can safely terminate.
                break
            else:
                # The queue is empty, but uploads might still be in progress.
                # Wait a moment before checking for new jobs.
                sleep(1)
        except Exception:
            logfire.exception("Exception while running job")
            if job is not None:
                # An error occurred. Re-queue the job if retry limits haven't been exceeded.
                if job.run_num_retries < settings.retry.run:
                    if job.total_num_retries < settings.retry.max:
                        # Insert at the front of the queue for immediate reprocessing.
                        sleep(settings.retry.delay)
                        jobs_queue.insert(0, job)


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
    )
    no_console_logfire = logfire.configure(local=True, console=False)
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
            settings.split.max_bytes,
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
        gemini_upload_jobs_futures: list[concurrent.futures.Future] = []
        subtitle_jobs_futures: list[concurrent.futures.Future] = []
        upload_jobs_executor = None

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
            # Create a thread pool to handle file uploads concurrently.
            max_upload_workers = settings.thread.uploads
            upload_jobs_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_upload_workers
            )
            gemini_upload_jobs_futures = [
                upload_jobs_executor.submit(
                    upload_file_job,
                    gemini_upload_jobs_queue,
                    uploader,
                    subtitle_jobs_queue,
                    settings,
                )
                for _ in range(max_upload_workers)
            ]
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
        jobs_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        subtitle_jobs_futures = [
            jobs_executor.submit(
                subtitle_job,
                subtitle_jobs_queue,
                agent,
                gemini_upload_complete_event,
                settings,
            )
            for _ in range(max_workers)
        ]

        # Step 6: Wait for all jobs to complete.
        # First, wait for all file upload jobs (if any) to finish.
        concurrent.futures.wait(gemini_upload_jobs_futures)
        # Signal that no more upload jobs will be added. This tells the subtitle
        # workers they can exit once their queue is empty.
        gemini_upload_complete_event.set()

        # Then, wait for all the subtitle generation jobs to finish.
        concurrent.futures.wait(subtitle_jobs_futures)

        # Shutdown executors
        if upload_jobs_executor:
            upload_jobs_executor.shutdown()
        jobs_executor.shutdown()

        # Step 7: Assemble the final subtitle file.
        stitch_subtitles(video_splits, settings)

        logfire.info("Done")


if __name__ == "__main__":
    main()
