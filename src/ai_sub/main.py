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
    JobState,
    LyricsSceneJob,
    ReEncodingJob,
    SceneResponse,
    SubtitleGenerationState,
    SubtitlePass1Job,
    SubtitlePass1Response,
    SubtitlePass2Job,
    SubtitlePass2Response,
    UploadFileJob,
)
from ai_sub.gemini_file_uploader import GeminiFileUploader
from ai_sub.job_runner import JobRunner
from ai_sub.prompt import (
    SUBTITLES_PROMPT_VERSION,
    get_lyrics_scenes_prompt,
    get_subtitle_pass1_prompt,
    get_subtitle_pass2_prompt,
)
from ai_sub.video import (
    get_video_duration_ms,
    get_working_encoder,
    reencode_video,
    split_video,
)


def _job_file_exists(
    job: LyricsSceneJob | SubtitlePass1Job | SubtitlePass2Job | None,
) -> bool:
    """
    Checks if the file associated with a job exists on the local filesystem.
    For cloud files, it assumes they exist.
    """
    if not job:
        return False
    # If the job's file is a local path, check if it actually exists.
    if isinstance(job.file, Path):
        if not job.file.exists():
            logfire.warning(
                f"File {job.file} for completed job '{job.name}' is missing. Will re-process."
            )
            return False
    # For google.genai.types.File objects, we assume they exist.
    # The API call will fail later if the URI is invalid, which is handled by the retry logic.
    return True


class ReEncodeJobRunner(JobRunner):
    """
    Worker that re-encodes video segments to a lower quality/different format.

    This is typically done to reduce file size before uploading to an API,
    saving bandwidth and potentially processing time.
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[JobState, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "ReEncode",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)

    def process(self, job: JobState) -> None:
        """
        Re-encodes the video file specified in the job.

        After re-encoding, it returns None, triggering the on_complete callback in the runner.
        """
        reencode_job = job.reencode
        assert reencode_job is not None
        with logfire.span(f"Re-encoding {reencode_job.input_file.name}"):
            reencode_video(
                reencode_job.input_file,
                reencode_job.output_file,
                reencode_job.fps,
                reencode_job.height,
                reencode_job.bitrate_kb,
                self.settings.split.re_encode.encoder or "libx264",
            )

            logfire.info(
                f"{reencode_job.input_file.name} re-encoded to {reencode_job.output_file.name}"
            )


class UploadJobRunner(JobRunner):
    """
    Worker that uploads video files to the Gemini Files API.

    This runner is used when the AI model requires the file to be hosted
    on Google's servers (e.g., for Gemini models).
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        uploader: GeminiFileUploader,
        on_complete: Callable[[JobState, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "Upload",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.uploader = uploader

    def process(self, job: JobState) -> Any:
        """
        Uploads the specified file using the `GeminiFileUploader`.

        Upon successful upload, it returns the file object, triggering the on_complete callback.
        """
        upload_job = job.upload
        assert upload_job is not None
        with logfire.span(f"Uploading {upload_job.python_file.name}"):
            # Perform the file upload. This is a blocking operation.
            file = self.uploader.upload_file(upload_job.python_file)
            logfire.info(f"{upload_job.python_file.name} uploaded")
            return file


class LyricsSceneJobRunner(JobRunner):
    """
    Worker that executes the AI agent to detect lyrics and scenes.
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[JobState, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "LyricsSceneJobRunner",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(
            self.agent.model_name
        )

    def process(self, job: JobState) -> None:
        """
        Invokes the AI agent to detect scenes.
        """
        lyrics_job = job.lyrics[self.sanitized_model_name]
        assert lyrics_job is not None
        with logfire.span(f"Scene Detection {lyrics_job.name}"):
            lyrics_job.response = self.agent.run(
                get_lyrics_scenes_prompt(),
                lyrics_job.file,
                lyrics_job.video_duration_ms,
                response_type=SceneResponse,
            )

    def post_process(self, job: JobState) -> None:
        """Saves the result to disk."""
        lyrics_job = job.lyrics[self.sanitized_model_name]
        assert lyrics_job is not None
        job_state_path = self.settings.dir.tmp / f"{lyrics_job.name}.json"
        job.save(job_state_path)


class SubtitlePass1JobRunner(JobRunner):
    """
    Worker that executes the AI agent to generate the first pass of subtitles.
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[JobState, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "SubtitlePass1JobRunner",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(
            self.agent.model_name
        )

    def process(self, job: JobState) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        pass1_job = job.pass1[self.sanitized_model_name]
        assert pass1_job is not None

        sanitized_lyrics_model = self.settings.ai.get_sanitized_model_name(
            self.settings.ai.lyrics_model
        )
        lyrics_job = job.lyrics.get(sanitized_lyrics_model)
        scene_response = lyrics_job.response if lyrics_job else None

        with logfire.span(f"Subtitling Pass 1 {pass1_job.name}"):
            prompt = get_subtitle_pass1_prompt(scene_response)
            pass1_job.response = self.agent.run(
                prompt,
                pass1_job.file,
                pass1_job.video_duration_ms,
                response_type=SubtitlePass1Response,
            )

    def post_process(self, job: JobState) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        # Save the completed job state to a JSON file for persistence.
        pass1_job = job.pass1[self.sanitized_model_name]
        assert pass1_job is not None
        job_state_path = self.settings.dir.tmp / f"{pass1_job.name}.json"
        job.save(job_state_path)

        # Also generate a subtitle file for this job for the user to view.
        if pass1_job.response is not None:
            sanitized_model = self.settings.ai.get_sanitized_model_name(
                self.settings.ai.pass1_model
            )
            pass1_job.response.get_ssafile().save(
                str(
                    self.settings.dir.tmp
                    / f"{pass1_job.name}.{sanitized_model}-pass1.srt"
                )
            )


class SubtitlePass2JobRunner(JobRunner):
    """
    Worker that executes the AI agent to generate the second pass of subtitles (QA & Refinement).
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[JobState, Any], None] | None = None,
        stop_events: list[Event] | None = None,
        name: str = "SubtitlePass2JobRunner",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.agent = agent
        self.sanitized_model_name = self.settings.ai.get_sanitized_model_name(
            self.agent.model_name
        )

    def process(self, job: JobState) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        pass2_job = job.pass2[self.sanitized_model_name]
        assert pass2_job is not None

        sanitized_lyrics_model = self.settings.ai.get_sanitized_model_name(
            self.settings.ai.lyrics_model
        )
        lyrics_job = job.lyrics.get(sanitized_lyrics_model)
        scene_response = lyrics_job.response if lyrics_job else None

        sanitized_pass1_model = self.settings.ai.get_sanitized_model_name(
            self.settings.ai.pass1_model
        )
        pass1_job = job.pass1.get(sanitized_pass1_model)
        draft = pass1_job.response if pass1_job else None

        if not draft:
            logfire.error(
                f"Could not find draft subtitles for {pass2_job.name}. Skipping pass 2."
            )
            return

        with logfire.span(f"Subtitling Pass 2 {pass2_job.name}"):
            prompt = get_subtitle_pass2_prompt(scene_response, draft)
            pass2_job.response = self.agent.run(
                prompt,
                pass2_job.file,
                pass2_job.video_duration_ms,
                response_type=SubtitlePass2Response,
            )
            logfire.info(f"{pass2_job.name} subtitled")

    def post_process(self, job: JobState) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        # Save the completed job state to a JSON file for persistence.
        pass2_job = job.pass2[self.sanitized_model_name]
        assert pass2_job is not None
        job_state_path = self.settings.dir.tmp / f"{pass2_job.name}.json"
        job.save(job_state_path)

        if pass2_job.response is not None:
            # Also generate a subtitle file for this job for the user to view.
            sanitized_model = self.settings.ai.get_sanitized_model_name(
                self.settings.ai.pass2_model
            )
            pass2_job.response.get_ssafile().save(
                str(
                    self.settings.dir.tmp
                    / f"{pass2_job.name}.{sanitized_model}-pass2.srt"
                )
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
        sanitized_model = settings.ai.get_sanitized_model_name(settings.ai.pass2_model)

        chunks_to_skip = int(
            (settings.split.start_offset_min * 60) / settings.split.max_seconds
        )
        offset_ms = sum(duration for _, duration in video_splits[:chunks_to_skip])

        state = SubtitleGenerationState(
            ai_sub_version=version("ai-sub"),
            subtitles_prompt_version=SUBTITLES_PROMPT_VERSION,
            settings=settings,
        )

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            # Load the job result from the temporary JSON file.
            # We look for the final pass 2 output
            job_state = JobState.load(settings.dir.tmp / f"{video_path.stem}.json")
            job = job_state.pass2.get(sanitized_model) if job_state else None
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
                state.complete = False

            # Add the duration of the current segment to the offset for the next one.
            offset_ms += video_duration_ms

            # Sort out max retries exceeded
            if job_state and job_state.total_num_retries >= settings.retry.max:
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
    Runs the main subtitle generation pipeline.

    This function orchestrates the entire process, from video preparation to
    final subtitle file generation. The workflow is as follows:

    1.  **Video Splitting:** The input video is divided into smaller, manageable segments.
    2.  **Job Queueing:** For each segment, the application checks for cached results from
        previous runs. If a step is not complete, it creates a chain of jobs:
        - (Optional) **Re-encoding:** The video segment is re-encoded to a smaller size.
        - (Optional) **Uploading:** The file is uploaded to a cloud service (e.g., Gemini Files API).
        - **Scene Detection:** The video is analyzed for scene changes and lyrics for any detected songs are researched.
        - **Pass 1 (Drafting):** An initial subtitle draft is generated, using the scene/lyrics data as a reference.
        - **Pass 2 (Refinement):** The draft is refined and quality-checked, also using the scene/lyrics data.
    3.  **Concurrent Processing:** Jobs for each stage are processed concurrently by
        dedicated `JobRunner` instances. Callbacks are used to chain dependent jobs
        (e.g., an upload job triggers a scene detection job, which in turn triggers a
        subtitle job).
    4.  **Stitching:** Once all segments are processed, the individual subtitle results
        are stitched together, with timestamps adjusted to align with the original video.
    5.  **Final Output:** A single `.srt` file is created, containing the final subtitles
        and a state summary.

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
    agent1 = RateLimitedAgentWrapper(settings, settings.ai.pass1_model)
    agent_scene = RateLimitedAgentWrapper(settings, settings.ai.lyrics_model)
    agent2 = RateLimitedAgentWrapper(settings, settings.ai.pass2_model)

    sanitized_lyrics_model = settings.ai.get_sanitized_model_name(
        settings.ai.lyrics_model
    )
    sanitized_pass1_model = settings.ai.get_sanitized_model_name(
        settings.ai.pass1_model
    )
    sanitized_pass2_model = settings.ai.get_sanitized_model_name(
        settings.ai.pass2_model
    )

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

        # Step 2: Filter out segments that have already been processed.
        # This allows the process to be resumed. It checks for the existence of a
        # .json file which indicates a completed (or failed) job.

        # Initialize data structures for concurrent processing.
        # Deques are used as thread-safe queues for managing jobs.
        reencode_jobs_queue: deque[JobState] = deque()
        gemini_upload_jobs_queue: deque[JobState] = deque()
        scene_detection_jobs_queue: deque[JobState] = deque()
        subtitle_pass1_jobs_queue: deque[JobState] = deque()
        subtitle_pass2_jobs_queue: deque[JobState] = deque()

        use_reencode = settings.split.re_encode.enabled
        use_upload = agent1.is_google() and settings.ai.google.use_files_api

        reencode_complete_event = Event()
        gemini_upload_complete_event = Event()
        scene_detection_complete_event = Event()
        subtitle1_complete_event = Event()

        reencode_runner: ReEncodeJobRunner | None = None
        upload_runner: UploadJobRunner | None = None
        scene_detection_runner: LyricsSceneJobRunner | None = None
        subtitle_pass1_runner: SubtitlePass1JobRunner | None = None
        subtitle_pass2_runner: SubtitlePass2JobRunner | None = None

        # Define callbacks
        def on_reencode_complete(job: JobState, _: Any) -> None:
            reencode_job = job.reencode
            assert reencode_job is not None
            duration = get_video_duration_ms(reencode_job.output_file)
            if use_upload:
                job.upload = UploadFileJob(
                    python_file=reencode_job.output_file, video_duration_ms=duration
                )
                gemini_upload_jobs_queue.append(job)
            else:
                job.lyrics[sanitized_lyrics_model] = LyricsSceneJob(
                    name=reencode_job.output_file.stem,
                    file=reencode_job.output_file,
                    video_duration_ms=duration,
                )
                scene_detection_jobs_queue.append(job)

        def on_upload_complete(job: JobState, file: Any) -> None:
            upload_job = job.upload
            assert upload_job is not None
            job.lyrics[sanitized_lyrics_model] = LyricsSceneJob(
                name=upload_job.python_file.stem,
                file=file,
                video_duration_ms=upload_job.video_duration_ms,
            )
            scene_detection_jobs_queue.append(job)

        def on_scene_detection_complete(job: JobState, _: Any) -> None:
            lyrics_job = job.lyrics[sanitized_lyrics_model]
            assert lyrics_job is not None
            if lyrics_job.response:
                job.pass1[sanitized_pass1_model] = SubtitlePass1Job(
                    name=lyrics_job.name,
                    file=lyrics_job.file,
                    video_duration_ms=lyrics_job.video_duration_ms,
                )
                subtitle_pass1_jobs_queue.append(job)

        def on_subtitle1_complete(job: JobState, _: Any) -> None:
            pass1_job = job.pass1[sanitized_pass1_model]
            assert pass1_job is not None
            if pass1_job.response:
                job.pass2[sanitized_pass2_model] = SubtitlePass2Job(
                    name=pass1_job.name,
                    file=pass1_job.file,
                    video_duration_ms=pass1_job.video_duration_ms,
                )
                subtitle_pass2_jobs_queue.append(job)

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

        # Setup scene_detection_runner
        scene_stop_events = []
        if use_upload:
            scene_stop_events.append(gemini_upload_complete_event)
        elif use_reencode:
            scene_stop_events.append(reencode_complete_event)

        scene_detection_runner = LyricsSceneJobRunner(
            scene_detection_jobs_queue,
            settings,
            settings.thread.lyrics,
            agent_scene,
            on_complete=on_scene_detection_complete,
            stop_events=scene_stop_events,
        )

        # Setup subtitle_pass1_runner
        # Pass 1 now depends on Scene Detection
        subtitle1_stop_events = [scene_detection_complete_event]

        subtitle_pass1_runner = SubtitlePass1JobRunner(
            subtitle_pass1_jobs_queue,
            settings,
            settings.thread.subtitles1,
            agent1,
            on_complete=on_subtitle1_complete,
            stop_events=subtitle1_stop_events,
        )

        # Setup subtitle_pass2_runner
        subtitle_pass2_runner = SubtitlePass2JobRunner(
            subtitle_pass2_jobs_queue,
            settings,
            settings.thread.subtitles2,
            agent2,
            on_complete=None,
            stop_events=[subtitle1_complete_event],
        )

        # Step 4: Populate the initial job queues.

        # Create a directory for re-encoded files to avoid name collisions
        # and preserve the file stem for stitching.
        reencode_dir = settings.dir.tmp / "reencoded"
        if use_reencode:
            reencode_dir.mkdir(exist_ok=True)

        for split, duration in splits_to_process:
            job_state_path = settings.dir.tmp / f"{split.stem}.json"
            job_state = JobState.load(job_state_path)

            if not job_state:
                job_state = JobState()

            # 1. Check Pass 2 Done
            # Check if the final stage (Pass 2) has already been completed for the target model.
            # If a response exists, we can skip this entire segment.
            pass2_job = job_state.pass2.get(sanitized_pass2_model)
            if pass2_job and pass2_job.response and _job_file_exists(pass2_job):
                continue

            # 2. Check Pass 1 Done
            # If Pass 2 is not done, check if Pass 1 is complete for its target model.
            # If it is, we can create and queue a job for Pass 2.
            pass1_job = job_state.pass1.get(sanitized_pass1_model)
            if pass1_job and pass1_job.response and _job_file_exists(pass1_job):
                # Create a Pass 2 job if it doesn't already exist for the target model.
                if not job_state.pass2.get(sanitized_pass2_model):
                    job_state.pass2[sanitized_pass2_model] = SubtitlePass2Job(
                        name=pass1_job.name,
                        file=pass1_job.file,
                        video_duration_ms=pass1_job.video_duration_ms,
                    )
                subtitle_pass2_jobs_queue.append(job_state)
                continue

            # 3. Check Scene Detection Done
            # If Pass 1 is not done, check if the lyrics/scene detection is complete.
            # If it is, we can create and queue a job for Pass 1.
            lyrics_job = job_state.lyrics.get(sanitized_lyrics_model)
            if lyrics_job and lyrics_job.response and _job_file_exists(lyrics_job):
                # Create a Pass 1 job if it doesn't already exist for the target model.
                if not job_state.pass1.get(sanitized_pass1_model):
                    job_state.pass1[sanitized_pass1_model] = SubtitlePass1Job(
                        name=lyrics_job.name,
                        file=lyrics_job.file,
                        video_duration_ms=lyrics_job.video_duration_ms,
                    )
                subtitle_pass1_jobs_queue.append(job_state)
                continue

            # 4. Start from scratch (Re-encode/Upload/Scene)
            # If none of the AI-driven stages are complete for the specified models,
            # we start the processing chain from the beginning for this segment.
            # This involves potentially re-encoding or uploading the video segment
            # before queueing the first AI job (lyrics/scene detection).
            input_file = split

            should_reencode = False
            if use_reencode:
                if settings.split.re_encode.threshold_mb == 0:
                    should_reencode = True
                else:
                    file_size_mb = input_file.stat().st_size / (1024 * 1024)
                    if file_size_mb >= settings.split.re_encode.threshold_mb:
                        should_reencode = True
                    else:
                        logfire.info(
                            f"Skipping re-encode for {input_file.name} "
                            f"({file_size_mb:.2f}MB < {settings.split.re_encode.threshold_mb}MB)"
                        )

            if should_reencode:
                output_file = reencode_dir / input_file.with_suffix(".mov").name
                # Always create/update the re-encode job and queue it.
                # The runner is idempotent and will skip if the output file already exists.
                job_state.reencode = ReEncodingJob(
                    input_file=input_file,
                    output_file=output_file,
                    fps=settings.split.re_encode.fps,
                    height=settings.split.re_encode.height,
                    bitrate_kb=settings.split.re_encode.bitrate_kb,
                )
                reencode_jobs_queue.append(job_state)
            elif use_upload:
                # Always create/update the upload job and queue it.
                # The uploader is idempotent and will check for existing files on the server.
                job_state.upload = UploadFileJob(
                    python_file=input_file, video_duration_ms=duration
                )
                gemini_upload_jobs_queue.append(job_state)
            else:
                # If no re-encode or upload, queue the lyrics job directly.
                # Create the job if it doesn't exist for the target model.
                if not job_state.lyrics.get(sanitized_lyrics_model):
                    job_state.lyrics[sanitized_lyrics_model] = LyricsSceneJob(
                        name=input_file.stem,
                        file=input_file,
                        video_duration_ms=duration,
                    )
                scene_detection_jobs_queue.append(job_state)

        # Step 5: Start all runners and wait for them to complete
        # Start runners
        if reencode_runner:
            reencode_runner.start()
        if upload_runner:
            upload_runner.start()
        scene_detection_runner.start()
        subtitle_pass1_runner.start()
        subtitle_pass2_runner.start()

        # Wait for runners to complete and signal as needed
        if reencode_runner:
            reencode_runner.wait()
            reencode_complete_event.set()

        if upload_runner:
            upload_runner.wait()
            gemini_upload_complete_event.set()

        scene_detection_runner.wait()
        scene_detection_complete_event.set()

        subtitle_pass1_runner.wait()
        subtitle1_complete_event.set()

        subtitle_pass2_runner.wait()

        # Shutdown runners when all done
        if reencode_runner:
            reencode_runner.shutdown()
        if upload_runner:
            upload_runner.shutdown()
        scene_detection_runner.shutdown()
        subtitle_pass1_runner.shutdown()
        subtitle_pass2_runner.shutdown()

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

    sys.exit(ai_sub(settings).value)


if __name__ == "__main__":
    main()
