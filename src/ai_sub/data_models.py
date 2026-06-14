"""Data models for the AI subtitle generation pipeline."""

from __future__ import annotations

import re
import string
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

import logfire
from google.genai.types import File
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pyrate_limiter import Limiter
from pysubs2 import SSAEvent, SSAFile

from ai_sub.web_search import WebSearchDeps

# ==============================================================================
# Core Enums & Final Result
# ==============================================================================


class AiSubResult(IntEnum):
    """Defines standardized exit codes for the application.

    These codes are returned by the main `ai_sub` function to indicate the final
    status of the subtitle generation process, allowing for programmatic checks
    in scripts or other tools.
    """

    COMPLETE = 0
    """All segments were processed successfully."""

    INCOMPLETE = -1
    """One or more segments failed to process."""

    MAX_RETRIES_EXHAUSTED = -2
    """One or more segments failed to process and exhausted RetrySettings.max_runs."""


class QuotaExceededError(Exception):
    """Raised when the AI model provider's quota is exhausted."""

    pass


# ==============================================================================
# Utility Functions
# ==============================================================================


def _clean_timestamp_string(ts_str: str) -> str:
    """Extracts a valid timestamp pattern from a potentially noisy LLM string.

    LLMs occasionally suffer from "field leakage" where they include the subsequent
    JSON key or structural markers inside a string value. This function uses
    regex to isolate the actual timecode from such noise.

    Args:
        ts_str (str): The potentially noisy timestamp string.

    Returns:
        str: The extracted timestamp string.

    Example:
        "03:52.000,start:" -> "03:52.000"
        "01:23.456"        -> "01:23.456"
        "start: 00:10"     -> "00:10"

    """
    # Matches MM:SS, MM:SS.mmm, or MM:SS:mmm
    match = re.search(r"(\d{1,2}:\d{2}(?:[:.]\d{1,3})?)", ts_str)
    return match.group(1) if match else ts_str


def _parse_timestamp_string_ms(timestamp_string: str) -> int:
    """Parses a timestamp string into milliseconds.

    Supports "MM:SS.mmm", "MM:SS:mmm", and "MM:SS" formats.

    Args:
        timestamp_string: The timestamp string to parse.

    Returns:
        The parsed timestamp in milliseconds.

    Raises:
        ValueError: If the timestamp string is in an invalid format.

    """
    ts = timestamp_string

    if "." in ts:
        # Handles "MM:SS.mmm"
        split1 = ts.split(".")
        split2 = split1[0].split(":")
        minutes = int(split2[0])
        seconds = int(split2[1])
        milliseconds = int(split1[1].ljust(3, "0")[:3])
        timestamp = minutes * 60000 + seconds * 1000 + milliseconds
    elif ts.count(":") == 2:
        # Handles "MM:SS:mmm"
        split = ts.split(":")
        minutes = int(split[0])
        seconds = int(split[1])
        milliseconds = int(split[2].ljust(3, "0")[:3])
        timestamp = minutes * 60000 + seconds * 1000 + milliseconds
    elif ts.count(":") == 1:
        # Handles "MM:SS"
        split = ts.split(":")
        minutes = int(split[0])
        seconds = int(split[1])
        timestamp = minutes * 60000 + seconds * 1000
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_string}")
    return timestamp


def _format_ms_timestamp(ms: int) -> str:
    """Formats milliseconds into a 'MM:SS.mmm' string.

    Args:
        ms: The duration in milliseconds.

    Returns:
        The formatted timestamp string.
    """
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def _get_default_lyrics_version() -> int:
    """Retrieves the current lyrics prompt version for use as a Pydantic default.

    This uses a local import to avoid a circular dependency between `data_models.py`
    (which defines structures used by prompts) and `prompt.py` (which defines the
    version constants). By importing inside the function, the version is only
    accessed at runtime during object instantiation.

    Returns:
        int: The current LYRICS_PROMPT_VERSION.
    """
    from ai_sub.prompt import LYRICS_PROMPT_VERSION

    return LYRICS_PROMPT_VERSION


def _get_default_subtitles_version() -> int:
    """Retrieves the current subtitles prompt version for use as a Pydantic default.

    This uses a local import to avoid a circular dependency between `data_models.py`
    and `prompt.py`. This ensures that the version number is available as soon
    as a job object is created, even if it hasn't been processed yet.

    Returns:
        int: The current SUBTITLES_PROMPT_VERSION.
    """
    from ai_sub.prompt import SUBTITLES_PROMPT_VERSION

    return SUBTITLES_PROMPT_VERSION


# ==============================================================================
# AI Response Models
# ==============================================================================


class AgentDeps(BaseModel):
    """Container for agent dependencies passed to Pydantic AI's RunContext.

    Provides a centralized, extensible way to store and access multiple
    dependencies (e.g., web search clients, databases, etc.) for use by
    agent tools.  Add new fields here as the pipeline grows.

    Example::

        deps = AgentDeps(ollama_search=OllamaWebSearchDeps(settings))
        # In a tool: ctx.deps.ollama_search
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_limiter: Limiter | None = Field(description="Rate limiter for API requests.", default=None)
    token_limiter: Limiter | None = Field(description="Rate limiter for API tokens.", default=None)
    request_tokens: int = 0

    video_duration_ms: NonNegativeInt = 0
    validation_buffer_ms: NonNegativeInt = 2000

    web_search: WebSearchDeps | None = None
    """Web-search dependency (:class:`WebSearchDeps`), or ``None``."""


class Subtitles(BaseModel):
    """Represents a single subtitle entry with start/end times and text."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: str = Field(
        alias="s",
        description="The start timestamp of the subtitle (e.g., 'MM:SS.mmm').",
    )
    end: str = Field(alias="e", description="The end timestamp of the subtitle (e.g., 'MM:SS.mmm').")
    original: str = Field(alias="og", description="The transcription/text in its original language.")
    english: str = Field(alias="en", description="The English translation of the text.")

    @model_validator(mode="before")
    @classmethod
    def clean_leakage(cls, data: Any) -> Any:
        """Strips LLM noise from timestamps before field assignment.

        Args:
            data (Any): The raw input data.

        Returns:
            Any: The cleaned data.
        """
        if isinstance(data, dict):
            # Check both field names and aliases
            for key in ("start", "s", "end", "e"):
                if key in data and isinstance(data[key], str):
                    data[key] = _clean_timestamp_string(data[key])
        return data

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Subtitles":
        """Validates the timestamps for a subtitle.

        Returns:
            Subtitles: The validated subtitle instance.

        Raises:
            ValueError: If the timestamp format is invalid or if the start time
                is not strictly before the end time.
        """
        try:
            start_ms = _parse_timestamp_string_ms(self.start)
            end_ms = _parse_timestamp_string_ms(self.end)
            if start_ms >= end_ms:
                raise ValueError(f"Start time ({self.start}) must be strictly before end time ({self.end})")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp: {e}") from e
        return self


class SubtitleAiResponse(BaseModel):
    """Represents the structured JSON response from the AI model for subtitle generation.

    This model is the expected output from the AI after it has processed a video
    segment for transcription and translation. It includes a high-level analysis
    from the model and a list of individual subtitle entries.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    global_analysis: str = Field(description="A high-level analysis or summary from the AI about its process.")
    subtitles: list[Subtitles] = Field(alias="subs", description="A list of individual subtitle entries.")

    def get_ssafile(self) -> SSAFile:
        """Converts the response's subtitles into an SSAFile object.

        Handles timestamp parsing and combines English and Original text.

        Returns:
            SSAFile: An SSAFile object containing the parsed subtitles.

        """
        subtitles = SSAFile()

        translator = str.maketrans("", "", string.punctuation)

        for subtitle in self.subtitles:
            start = _parse_timestamp_string_ms(subtitle.start)
            end = _parse_timestamp_string_ms(subtitle.end)
            english_text = subtitle.english.strip()
            original_text = subtitle.original.strip()

            english_norm = english_text.casefold().translate(translator)
            original_norm = original_text.casefold().translate(translator)

            # If Gemini returns similar text for both En and Original, just use the Original
            if english_norm == original_norm:
                text = original_text
            else:
                text = f"{original_text}\n{english_text}"

            subtitles.append(SSAEvent(start=start, end=end, text=text))

        return subtitles

    @model_validator(mode="after")
    def _validate_against_duration_validator(self, info: ValidationInfo) -> "SubtitleAiResponse":
        """Internal validator to trigger duration checks via Pydantic AI context.

        Args:
            info (ValidationInfo): The validation context.

        Returns:
            SubtitleAiResponse: The validated instance.
        """
        context = info.context or {}
        duration_ms = context.get("video_duration_ms")
        if duration_ms is not None:
            buffer_ms = context.get("validation_buffer_ms", 2000)
            self.validate_against_duration(duration_ms, buffer_ms)
        return self

    def validate_against_duration(self, duration_ms: int, buffer_ms: int = 2000) -> "SubtitleAiResponse":
        """Validates that no subtitle end timestamp exceeds the video duration + buffer.

        Args:
            duration_ms (int): The duration of the video in milliseconds.
            buffer_ms (int): The allowed buffer in milliseconds.

        Returns:
            SubtitleAiResponse: The validated instance.

        Raises:
            ValueError: If a subtitle end timestamp is too far beyond the duration.
        """
        limit_ms = duration_ms + buffer_ms
        for i, sub in enumerate(self.subtitles):
            end_ms = _parse_timestamp_string_ms(sub.end)
            if end_ms > limit_ms:
                raise ValueError(
                    f"Subtitle {i} end time ({sub.end}) exceeds video duration "
                    f"({_format_ms_timestamp(duration_ms)}) by more than {buffer_ms}ms."
                )
        return self


class Scene(BaseModel):
    """Represents a detected scene in the video."""

    start: str = Field(description="The start timestamp of the scene (e.g., 'MM:SS.mmm').")
    end: str = Field(description="The end timestamp of the scene (e.g., 'MM:SS.mmm').")
    description: str = Field(description="A brief description of the visual and audio elements in the scene.")
    contains_vocal_music: bool = Field(description="True if the scene contains music with vocals.")
    song_title: Optional[str] = Field(default=None, description="The title of the song detected in the scene.")
    original_artist: Optional[str] = Field(default=None, description="The original artist or composer of the song.")
    performer_in_video: Optional[str] = Field(
        default=None,
        description="The performer singing in the video, if different from the original artist.",
    )
    original_language: Optional[str] = Field(
        default=None,
        description="The language the song is primarily sung in.",
    )
    reference_lyrics_og: Optional[str] = Field(
        default=None,
        description="The full lyrics of the song in the original language, found via web search.",
    )
    reference_lyrics_en: Optional[str] = Field(
        default=None,
        description="The English translation of the lyrics, found via web search.",
    )

    @model_validator(mode="before")
    @classmethod
    def clean_leakage(cls, data: Any) -> Any:
        """Strips LLM noise from timestamps before field assignment.

        Args:
            data (Any): The raw input data.

        Returns:
            Any: The cleaned data.
        """
        if isinstance(data, dict):
            for key in ("start", "end"):
                if key in data and isinstance(data[key], str):
                    data[key] = _clean_timestamp_string(data[key])
        return data

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Scene":
        """Validates the timestamps for a scene.

        Returns:
            Scene: The validated scene instance.

        Raises:
            ValueError: If the timestamp format is invalid or if the start time
                is not strictly before the end time.
        """
        try:
            start_ms = _parse_timestamp_string_ms(self.start)
            end_ms = _parse_timestamp_string_ms(self.end)
            if start_ms >= end_ms:
                raise ValueError(f"Start time ({self.start}) must be strictly before end time ({self.end})")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp: {e}") from e
        return self


class LyricsSceneAiResponse(BaseModel):
    """Represents the structured JSON response from the AI for the lyrics/scene detection pass.

    This model captures the AI's analysis of a video segment, including a breakdown
    of scenes, identification of music, and any lyrics found through web searches.
    This data is then used as context for the main subtitle generation pass.
    """

    step_by_step_log: str
    global_summary: str
    scenes: list[Scene] = Field(description="A list of chronological scenes detected in the video segment.")

    @model_validator(mode="after")
    def _validate_against_duration_validator(self, info: ValidationInfo) -> "LyricsSceneAiResponse":
        """Internal validator to trigger duration checks via Pydantic AI context.

        Args:
            info (ValidationInfo): The validation context.

        Returns:
            LyricsSceneAiResponse: The validated instance.
        """
        context = info.context or {}
        duration_ms = context.get("video_duration_ms")
        if duration_ms is not None:
            buffer_ms = context.get("validation_buffer_ms", 2000)
            self.validate_against_duration(duration_ms, buffer_ms)
        return self

    def validate_against_duration(self, duration_ms: int, buffer_ms: int = 2000) -> "LyricsSceneAiResponse":
        """Validates that no scene end timestamp exceeds the video duration + buffer.

        Args:
            duration_ms (int): The duration of the video in milliseconds.
            buffer_ms (int): The allowed buffer in milliseconds.

        Returns:
            LyricsSceneAiResponse: The validated instance.

        Raises:
            ValueError: If a scene end timestamp is too far beyond the duration.
        """
        limit_ms = duration_ms + buffer_ms
        for i, scene in enumerate(self.scenes):
            end_ms = _parse_timestamp_string_ms(scene.end)
            if end_ms > limit_ms:
                raise ValueError(
                    f"Scene {i} end time ({scene.end}) exceeds video duration "
                    f"({_format_ms_timestamp(duration_ms)}) by more than {buffer_ms}ms."
                )
        return self


# ==============================================================================
# Job & Pipeline Models
# ==============================================================================


class Job(BaseModel):
    """Base class for all job types in the processing pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="The unique name of the job, typically derived from the video segment filename (e.g., 'part_001')."
    )
    total_attempts: NonNegativeInt = Field(
        default=0,
        description="The total number of attempts for this specific job stage across all executions, "
        "loaded from the saved stage-specific state file. "
        "Attempts are tracked independently per stage (e.g., lyrics failures do not affect subtitle attempt counts).",
    )

    def save(self, filename: Path):
        """Saves the current object to a JSON file.

        Args:
            filename (Path): The path to the file where the object should be saved.

        """
        json_str = self.model_dump_json(indent=2)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(json_str)


class ReEncodingJob(Job):
    """Represents a job to re-encode a video file."""

    input_file: Path = Field(description="Path to the original video segment.")
    output_file: Path = Field(description="Path where the re-encoded video will be saved.")
    fps: PositiveInt = Field(description="Target frames per second for the re-encoded video.")
    height: PositiveInt = Field(description="Target height (resolution) for the re-encoded video.")
    bitrate_kb: PositiveInt = Field(description="Target bitrate in kilobytes per second.")
    duration_tolerance_ms: NonNegativeInt = Field(
        description="Allowed duration difference in milliseconds to consider an existing re-encoded file valid."
    )


class UploadFileJob(Job):
    """Represents a job to upload a file to the AI provider."""

    python_file: Path = Field(description="Path to the local file to be uploaded.")
    video_duration_ms: PositiveInt = Field(description="Duration of the video file in milliseconds.")


class LyricsSceneJob(Job):
    """Represents a job to detect lyrics and scenes in a video segment."""

    file: Optional[File | Path] = Field(default=None, exclude=True)
    video_duration_ms: PositiveInt
    response: Optional[LyricsSceneAiResponse] = Field(
        default=None,
        description="The structured AI response after successful processing.",
    )
    lyrics_prompt_version: Optional[int] = Field(
        default_factory=_get_default_lyrics_version,
        description="The version of the prompt used to generate the response, for cache validation.",
    )

    @model_validator(mode="after")
    def validate_response_timestamps(self, info: ValidationInfo) -> "LyricsSceneJob":
        """Ensures that the response timestamps are within the video duration.

        Args:
            info (ValidationInfo): The validation context containing the buffer setting.

        Returns:
            LyricsSceneJob: The validated job instance.
        """
        if self.response:
            buffer_ms = info.context.get("validation_buffer_ms", 2000) if info.context else 2000
            self.response.validate_against_duration(self.video_duration_ms, buffer_ms)
        return self

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int = 1000) -> Optional["LyricsSceneJob"]:
        """Loads the job from a JSON file, checking for prompt version mismatch.

        Args:
            save_path (Path): The path to the saved job file.
            validation_buffer_ms (int): The allowed buffer in milliseconds.

        Returns:
            Optional[LyricsSceneJob]: The loaded job, or None if validation fails
                or if there is a version mismatch.
        """
        # Local import to avoid circular dependency
        from ai_sub.prompt import LYRICS_PROMPT_VERSION

        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    job = cls.model_validate_json(
                        f.read(),
                        context={"validation_buffer_ms": validation_buffer_ms},
                    )
                except ValidationError as e:
                    logfire.warning(f"Validation failed for {save_path.name}, ignoring cache. Error: {e}")
                    return None

            if job.lyrics_prompt_version != LYRICS_PROMPT_VERSION:
                logfire.info(
                    f"Lyrics prompt version mismatch for {job.name} "
                    f"(file: {job.lyrics_prompt_version}, current: {LYRICS_PROMPT_VERSION}). Re-processing."
                )
                return None
            return job
        return None


class SubtitleJob(Job):
    """Represents a job to generate subtitles (Transcription).

    Uses scene/lyrics data from a `LyricsSceneAiResponse` as a reference.
    """

    file: Optional[File | Path] = Field(default=None, exclude=True)
    video_duration_ms: PositiveInt
    response: Optional[SubtitleAiResponse] = Field(
        default=None,
        description="The structured AI response after successful processing.",
    )
    subtitles_prompt_version: Optional[int] = Field(
        default_factory=_get_default_subtitles_version,
        description="The version of the prompt used to generate the response, for cache validation.",
    )

    @model_validator(mode="after")
    def validate_response_timestamps(self, info: ValidationInfo) -> "SubtitleJob":
        """Ensures that the response timestamps are within the video duration.

        Args:
            info (ValidationInfo): The validation context containing the buffer setting.

        Returns:
            SubtitleJob: The validated job instance.
        """
        if self.response:
            buffer_ms = info.context.get("validation_buffer_ms", 2000) if info.context else 2000
            self.response.validate_against_duration(self.video_duration_ms, buffer_ms)
        return self

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int = 1000) -> Optional["SubtitleJob"]:
        """Loads the job from a JSON file, checking for prompt version mismatch.

        Args:
            save_path (Path): The path to the saved job file.
            validation_buffer_ms (int): The allowed buffer in milliseconds.

        Returns:
            Optional[SubtitleJob]: The loaded job, or None if validation fails
                or if there is a version mismatch.
        """
        # Local import to avoid circular dependency
        from ai_sub.prompt import SUBTITLES_PROMPT_VERSION

        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    job = cls.model_validate_json(
                        f.read(),
                        context={"validation_buffer_ms": validation_buffer_ms},
                    )
                except ValidationError as e:
                    logfire.warning(f"Validation failed for {save_path.name}, ignoring cache. Error: {e}")
                    return None

            if job.subtitles_prompt_version != SUBTITLES_PROMPT_VERSION:
                logfire.info(
                    f"Subtitles prompt version mismatch for {job.name} "
                    f"(file: {job.subtitles_prompt_version}, current: {SUBTITLES_PROMPT_VERSION}). Re-processing."
                )
                return None
            return job
        return None


class SegmentJobs(BaseModel):
    """A container for all potential jobs related to a single video segment.

    This model acts as a state-passing object that moves through the pipeline.
    As a segment completes one stage (e.g., re-encoding), the result is stored,
    and the object is passed to the next stage's queue (e.g., upload). This
    allows each runner to have access to the state and outputs of previous steps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reencode: Optional[ReEncodingJob] = Field(default=None, description="The re-encoding job for the segment.")
    upload: Optional[UploadFileJob] = Field(default=None, description="The file upload job for the segment.")
    lyrics: Optional[LyricsSceneJob] = Field(
        default=None, description="The lyrics/scene detection job for the segment."
    )
    subtitles: Optional[SubtitleJob] = Field(
        default=None, description="The final subtitle generation job for the segment."
    )
