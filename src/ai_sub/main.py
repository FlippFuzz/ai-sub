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
    LyricsSceneJob,
    ReEncodingJob,
    SceneResponse,
    SubtitleGenerationState,
    SubtitlePass1Job,
    SubtitlePass1Response,
    SubtitlePass2Job,
    SubtitlePass2Response,
    UploadFileJob,
    VideoPartState,
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


class LyricsSceneJobRunner(JobRunner[LyricsSceneJob]):
    """
    Worker that executes the AI agent to detect lyrics and scenes.
    """

    def __init__(
        self,
        queue: deque[LyricsSceneJob],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[LyricsSceneJob, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "LyricsSceneJobRunner",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.agent = agent

    def process(self, job: LyricsSceneJob) -> None:
        """
        Invokes the AI agent to detect scenes.
        """
        with logfire.span(f"Scene Detection {job.name}"):
            job.response = self.agent.run(
                get_lyrics_scenes_prompt(),
                job.file,
                job.video_duration_ms,
                response_type=SceneResponse,
            )

    def post_process(self, job: LyricsSceneJob) -> None:
        """Saves the result to disk."""
        state_path = self.settings.dir.tmp / f"{job.name}.json"
        state = VideoPartState.load_or_create(
            state_path, job.name, job.file, job.video_duration_ms
        )
        state.scene_job = job
        state.save(state_path)


class SubtitlePass1JobRunner(JobRunner[SubtitlePass1Job]):
    """
    Worker that executes the AI agent to generate the first pass of subtitles.
    """

    def __init__(
        self,
        queue: deque[SubtitlePass1Job],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        on_complete: Callable[[SubtitlePass1Job, Any], None],
        stop_events: list[Event] | None = None,
        name: str = "SubtitlePass1JobRunner",
    ):
        super().__init__(queue, settings, max_workers, on_complete, stop_events, name)
        self.agent = agent

    def process(self, job: SubtitlePass1Job) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        with logfire.span(f"Subtitling Pass 1 {job.name}"):
            prompt = get_subtitle_pass1_prompt(job.scene_response)
            job.response = self.agent.run(
                prompt,
                job.file,
                job.video_duration_ms,
                response_type=SubtitlePass1Response,
            )

    def post_process(self, job: SubtitlePass1Job) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        state_path = self.settings.dir.tmp / f"{job.name}.json"
        state = VideoPartState.load_or_create(
            state_path, job.name, job.file, job.video_duration_ms
        )
        state.pass1_job = job
        state.save(state_path)

        sanitized_model = self.settings.ai.get_sanitized_model_name(
            self.settings.ai.pass1_model
        )
        # Also generate a subtitle file for this job for the user to view.
        if job.response is not None:
            job.response.get_ssafile().save(
                str(self.settings.dir.tmp / f"{job.name}.{sanitized_model}-pass1.srt")
            )


class SubtitlePass2JobRunner(JobRunner[SubtitlePass2Job]):
    """
    Worker that executes the AI agent to generate the second pass of subtitles (QA & Refinement).
    """

    def __init__(
        self,
        queue: deque[SubtitlePass2Job],
        settings: Settings,
        max_workers: int,
        agent: RateLimitedAgentWrapper,
        stop_events: list[Event] | None = None,
        name: str = "SubtitlePass2JobRunner",
    ):
        super().__init__(queue, settings, max_workers, None, stop_events, name)
        self.agent = agent

    def process(self, job: SubtitlePass2Job) -> None:
        """
        Invokes the AI agent to generate subtitles.
        """
        with logfire.span(f"Subtitling Pass 2 {job.name}"):
            prompt = get_subtitle_pass2_prompt(job.scene_response, job.draft)
            job.response = self.agent.run(
                prompt,
                job.file,
                job.video_duration_ms,
                response_type=SubtitlePass2Response,
            )
            logfire.info(f"{job.name} subtitled")

    def post_process(self, job: SubtitlePass2Job) -> None:
        """
        Saves the result (or partial state) to disk.

        This ensures that if the process is interrupted, completed segments
        don't need to be re-processed.
        """
        state_path = self.settings.dir.tmp / f"{job.name}.json"
        state = VideoPartState.load_or_create(
            state_path, job.name, job.file, job.video_duration_ms
        )
        state.pass2_job = job
        state.save(state_path)

        sanitized_model = self.settings.ai.get_sanitized_model_name(
            self.settings.ai.pass2_model
        )
        if job.response is not None:
            # Also generate a subtitle file for this job for the user to view.
            job.response.get_ssafile().save(
                str(self.settings.dir.tmp / f"{job.name}.{sanitized_model}-pass2.srt")
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

        chunks_to_skip = int(
            (settings.split.start_offset_min * 60) / settings.split.max_seconds
        )
        offset_ms = sum(duration for _, duration in video_splits[:chunks_to_skip])

        state = SubtitleGenerationState(
            ai_sub_version=version("ai-sub"),
            subtitles_prompt_version=SUBTITLES_PROMPT_VERSION,
            settings=settings,
        )

        use_pass2 = settings.thread.subtitles2 > 0

        for video_path, video_duration_ms in video_splits[chunks_to_skip:]:
            # Load the job result from the temporary JSON file.
            state_path = settings.dir.tmp / f"{video_path.stem}.json"
            part_state = VideoPartState.load_or_create(
                state_path, video_path.stem, video_path, video_duration_ms
            )
            job = part_state.pass2_job if use_pass2 else part_state.pass1_job

            if job and job.response is not None:
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
            if job and job.total_num_retries >= settings.retry.max:
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

        sanitized_model_name = (
            settings.ai.pass2_model if use_pass2 else settings.ai.pass1_model
        )
        sanitized_model = settings.ai.get_sanitized_model_name(sanitized_model_name)
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
        reencode_jobs_queue: deque[ReEncodingJob] = deque()
        gemini_upload_jobs_queue: deque[UploadFileJob] = deque()
        scene_detection_jobs_queue: deque[LyricsSceneJob] = deque()
        subtitle_pass1_jobs_queue: deque[SubtitlePass1Job] = deque()
        subtitle_pass2_jobs_queue: deque[SubtitlePass2Job] = deque()

        use_reencode = settings.split.re_encode.enabled
        use_upload = agent1.is_google() and settings.ai.google.use_files_api
        use_pass2 = settings.thread.subtitles2 > 0
        use_scene = settings.thread.lyrics > 0

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
        def on_reencode_complete(job: ReEncodingJob, _: Any) -> None:
            duration = get_video_duration_ms(job.output_file)
            if use_upload:
                gemini_upload_jobs_queue.append(
                    UploadFileJob(
                        python_file=job.output_file, video_duration_ms=duration
                    )
                )
            elif use_scene:
                scene_detection_jobs_queue.append(
                    LyricsSceneJob(
                        name=job.output_file.stem,
                        file=job.output_file,
                        video_duration_ms=duration,
                    )
                )
            else:
                subtitle_pass1_jobs_queue.append(
                    SubtitlePass1Job(
                        name=job.output_file.stem,
                        file=job.output_file,
                        video_duration_ms=duration,
                        scene_response=None,
                    )
                )

        def on_upload_complete(job: UploadFileJob, file: Any) -> None:
            if use_scene:
                scene_detection_jobs_queue.append(
                    LyricsSceneJob(
                        name=job.python_file.stem,
                        file=file,
                        video_duration_ms=job.video_duration_ms,
                    )
                )
            else:
                subtitle_pass1_jobs_queue.append(
                    SubtitlePass1Job(
                        name=job.python_file.stem,
                        file=file,
                        video_duration_ms=job.video_duration_ms,
                        scene_response=None,
                    )
                )

        def on_scene_detection_complete(job: LyricsSceneJob, _: Any) -> None:
            if job.response:
                subtitle_pass1_jobs_queue.append(
                    SubtitlePass1Job(
                        name=job.name,
                        file=job.file,
                        video_duration_ms=job.video_duration_ms,
                        scene_response=job.response,
                    )
                )

        def on_subtitle1_complete(job: SubtitlePass1Job, _: Any) -> None:
            if use_pass2 and job.response:
                subtitle_pass2_jobs_queue.append(
                    SubtitlePass2Job(
                        name=job.name,
                        file=job.file,
                        video_duration_ms=job.video_duration_ms,
                        scene_response=job.scene_response,
                        draft=job.response,
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

        # Setup scene_detection_runner
        if use_scene:
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
        # Pass 1 now depends on Scene Detection (if enabled)
        subtitle1_stop_events = []
        if use_scene:
            subtitle1_stop_events.append(scene_detection_complete_event)
        elif use_upload:
            subtitle1_stop_events.append(gemini_upload_complete_event)
        elif use_reencode:
            subtitle1_stop_events.append(reencode_complete_event)

        subtitle_pass1_runner = SubtitlePass1JobRunner(
            subtitle_pass1_jobs_queue,
            settings,
            settings.thread.subtitles1,
            agent1,
            on_complete=on_subtitle1_complete,
            stop_events=subtitle1_stop_events,
        )

        # Setup subtitle_pass2_runner
        if use_pass2:
            subtitle_pass2_runner = SubtitlePass2JobRunner(
                subtitle_pass2_jobs_queue,
                settings,
                settings.thread.subtitles2,
                agent2,
                stop_events=[subtitle1_complete_event],
            )

        # Step 4: Populate the initial job queues.

        # Create a directory for re-encoded files to avoid name collisions
        # and preserve the file stem for stitching.
        reencode_dir = settings.dir.tmp / "reencoded"
        if use_reencode:
            reencode_dir.mkdir(exist_ok=True)

        for split, duration in splits_to_process:
            state_path = settings.dir.tmp / f"{split.stem}.json"
            state = VideoPartState.load_or_create(
                state_path, split.stem, split, duration
            )

            # 1. Check Pass 2 Done
            if use_pass2:
                pass2_job = state.pass2_job
                if pass2_job and pass2_job.response is not None:
                    continue

            # 2. Check Pass 1 Done
            pass1_job = state.pass1_job
            if pass1_job and pass1_job.response is not None:
                if use_pass2:
                    # We need scene response for Pass 2. If Pass 1 was loaded from disk,
                    # it might have scene_response if it was run with the new version.
                    # If not, we might be missing it.
                    # For now, we assume if Pass 1 is done, we can proceed to Pass 2.
                    # If scene_response is missing, Pass 2 prompt will just have null for scene_data.
                    subtitle_pass2_jobs_queue.append(
                        SubtitlePass2Job(
                            name=pass1_job.name,
                            file=pass1_job.file,
                            video_duration_ms=pass1_job.video_duration_ms,
                            scene_response=pass1_job.scene_response,
                            draft=pass1_job.response,
                        )
                    )
                continue

            # 3. Check Scene Detection Done
            if use_scene:
                scene_job = state.scene_job
                if scene_job and scene_job.response is not None:
                    on_scene_detection_complete(scene_job, None)
                    continue

            # 4. Start from scratch (Re-encode/Upload/Scene)
            input_file = split
            if use_reencode:
                should_reencode = True
                if settings.split.re_encode.threshold_mb > 0:
                    file_size_mb = input_file.stat().st_size / (1024 * 1024)
                    if file_size_mb < settings.split.re_encode.threshold_mb:
                        should_reencode = False
                        logfire.info(
                            f"Skipping re-encode for {input_file.name} "
                            f"({file_size_mb:.2f}MB < {settings.split.re_encode.threshold_mb}MB)"
                        )

                if should_reencode:
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
                else:
                    if use_upload:
                        gemini_upload_jobs_queue.append(
                            UploadFileJob(
                                python_file=input_file, video_duration_ms=duration
                            )
                        )
                    elif use_scene:
                        scene_detection_jobs_queue.append(
                            LyricsSceneJob(
                                name=input_file.stem,
                                file=input_file,
                                video_duration_ms=duration,
                            )
                        )
                    else:
                        subtitle_pass1_jobs_queue.append(
                            SubtitlePass1Job(
                                name=input_file.stem,
                                file=input_file,
                                video_duration_ms=duration,
                                scene_response=None,
                            )
                        )
            elif use_upload:
                gemini_upload_jobs_queue.append(
                    UploadFileJob(python_file=input_file, video_duration_ms=duration)
                )
            elif use_scene:
                scene_detection_jobs_queue.append(
                    LyricsSceneJob(
                        name=input_file.stem,
                        file=input_file,
                        video_duration_ms=duration,
                    )
                )
            else:
                subtitle_pass1_jobs_queue.append(
                    SubtitlePass1Job(
                        name=input_file.stem,
                        file=input_file,
                        video_duration_ms=duration,
                        scene_response=None,
                    )
                )

        # Step 5: Start all runners and wait for them to complete
        # Start runners
        if reencode_runner:
            reencode_runner.start()
        if upload_runner:
            upload_runner.start()
        if scene_detection_runner:
            scene_detection_runner.start()
        subtitle_pass1_runner.start()
        if subtitle_pass2_runner:
            subtitle_pass2_runner.start()

        # Wait for runners to complete and signal as needed
        if reencode_runner:
            reencode_runner.wait()
            reencode_complete_event.set()

        if upload_runner:
            upload_runner.wait()
            gemini_upload_complete_event.set()

        if scene_detection_runner:
            scene_detection_runner.wait()
        scene_detection_complete_event.set()

        subtitle_pass1_runner.wait()
        subtitle1_complete_event.set()

        if subtitle_pass2_runner:
            subtitle_pass2_runner.wait()

        # Shutdown runners when all done
        if reencode_runner:
            reencode_runner.shutdown()
        if upload_runner:
            upload_runner.shutdown()
        if scene_detection_runner:
            scene_detection_runner.shutdown()
        subtitle_pass1_runner.shutdown()
        if subtitle_pass2_runner:
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
                    )
                )

        # Step 5: Start all runners and wait for them to complete
        # Start runners
        if reencode_runner:
            reencode_runner.start()
        if upload_runner:
            upload_runner.start()
        scene_detection_runner.start()
        if scene_detection_runner:
            scene_detection_runner.start()
        subtitle_pass1_runner.start()
        if subtitle_pass2_runner:
            subtitle_pass2_runner.start()

        # Wait for runners to complete and signal as needed
        if reencode_runner:
            reencode_runner.wait()
            reencode_complete_event.set()

        if upload_runner:
            upload_runner.wait()
            gemini_upload_complete_event.set()

        scene_detection_runner.wait()
        if scene_detection_runner:
            scene_detection_runner.wait()
        scene_detection_complete_event.set()

        subtitle_pass1_runner.wait()
        subtitle1_complete_event.set()

        if subtitle_pass2_runner:
            subtitle_pass2_runner.wait()

        # Shutdown runners when all done
        if reencode_runner:
            reencode_runner.shutdown()
        if upload_runner:
            upload_runner.shutdown()
        scene_detection_runner.shutdown()
        if scene_detection_runner:
            scene_detection_runner.shutdown()
        subtitle_pass1_runner.shutdown()
        if subtitle_pass2_runner:
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
