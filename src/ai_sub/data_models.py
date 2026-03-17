import string
from enum import IntEnum
from pathlib import Path
from typing import Optional

import logfire
from google.genai.types import File
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    field_validator,
)
from pysubs2 import SSAEvent, SSAFile

# ==============================================================================
# Core Enums & Final Result
# ==============================================================================


class AiSubResult(IntEnum):
    """
    Defines standardized exit codes for the application.

    These codes are returned by the main `ai_sub` function to indicate the final
    status of the subtitle generation process, allowing for programmatic checks
    in scripts or other tools.
    """

    COMPLETE = 0
    """All segments were processed successfully."""

    INCOMPLETE = -1
    """One or more segments failed to process and exhausted RetrySettings.run."""

    MAX_RETRIES_EXHAUSTED = -2
    """One or more segments failed to process and exhausted RetrySettings.max."""


# ==============================================================================
# AI Response Models
# ==============================================================================


class Subtitles(BaseModel):
    """Represents a single subtitle entry with start/end times and text."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: str = Field(
        alias="s",
        description="The start timestamp of the subtitle (e.g., 'MM:SS.mmm').",
    )
    end: str = Field(
        alias="e", description="The end timestamp of the subtitle (e.g., 'MM:SS.mmm')."
    )
    original: str = Field(
        alias="og", description="The transcription/text in its original language."
    )
    english: str = Field(alias="en", description="The English translation of the text.")


class SubtitleAiResponse(BaseModel):
    """
    Represents the structured JSON response from the AI model for subtitle generation.

    This model is the expected output from the AI after it has processed a video
    segment for transcription and translation. It includes a high-level analysis
    from the model and a list of individual subtitle entries.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    global_analysis: str = Field(
        description="A high-level analysis or summary from the AI about its process."
    )
    subtitles: list[Subtitles] = Field(
        alias="subs", description="A list of individual subtitle entries."
    )

    @field_validator("subtitles")
    @classmethod
    def validate_timestamps(cls, v: list[Subtitles]) -> list[Subtitles]:
        """
        Validates the timestamps for all subtitles.

        Checks:
        1. Format: Timestamps must be parseable (e.g., "MM:SS.mmm").
        2. Logic: Start time must be strictly before end time.
        """
        for subtitle in v:
            try:
                start_ms = cls._parse_timestamp_string_ms(subtitle.start)
                end_ms = cls._parse_timestamp_string_ms(subtitle.end)

                if start_ms >= end_ms:
                    raise ValueError(
                        f"Start time ({subtitle.start}) must be strictly before end time ({subtitle.end})"
                    )
            except ValueError as e:
                raise ValueError(f"Invalid timestamp in subtitle: {subtitle}. {e}")
        return v

    @staticmethod
    def _parse_timestamp_string_ms(timestamp_string: str) -> int:
        """Parses a timestamp string into milliseconds.

        Supports "MM:SS.mmm", "MM:SS:mmm", and "MM:SS" formats.

        Args:
            timestamp_string (str): The timestamp string to parse.

        Returns:
            int: The parsed timestamp in milliseconds.

        Raises:
            ValueError: If the timestamp string is None or in an invalid format.
        """
        if "." in timestamp_string:
            # Handles "MM:SS.mmm"
            split1 = timestamp_string.split(".")
            split2 = split1[0].split(":")
            minutes = int(split2[0])
            seconds = int(split2[1])
            milliseconds = int(split1[1])
            timestamp = minutes * 60000 + seconds * 1000 + milliseconds
        elif timestamp_string.count(":") == 2:
            # Handles "MM:SS:mmm"
            split = timestamp_string.split(":")
            minutes = int(split[0])
            seconds = int(split[1])
            milliseconds = int(split[2])
            timestamp = minutes * 60000 + seconds * 1000 + milliseconds
        elif timestamp_string.count(":") == 1:
            # Handles "MM:SS"
            split = timestamp_string.split(":")
            minutes = int(split[0])
            seconds = int(split[1])
            timestamp = minutes * 60000 + seconds * 1000
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_string}")
        return timestamp

    def get_ssafile(self) -> SSAFile:
        """
        Converts the response's subtitles into an SSAFile object.
        Handles timestamp parsing and combines English and Original text.

        Returns:
            SSAFile: An SSAFile object containing the parsed subtitles.
        """
        subtitles = SSAFile()

        translator = str.maketrans("", "", string.punctuation)

        for subtitle in self.subtitles:
            start = self._parse_timestamp_string_ms(subtitle.start)
            end = self._parse_timestamp_string_ms(subtitle.end)
            english_text = subtitle.english.strip()
            original_text = subtitle.original.strip()

            english_norm = english_text.casefold().translate(translator)
            original_norm = original_text.casefold().translate(translator)

            # If Gemini returns the similar text for En and Original, just use the Original
            if english_norm == original_norm:
                text = original_text
            else:
                text = f"{original_text}\n{english_text}"

            subtitles.append(SSAEvent(start=start, end=end, text=text))

        return subtitles


class Scene(BaseModel):
    """Represents a detected scene in the video."""

    start: str = Field(
        description="The start timestamp of the scene (e.g., 'MM:SS.mmm')."
    )
    end: str = Field(description="The end timestamp of the scene (e.g., 'MM:SS.mmm').")
    description: str = Field(
        description="A brief description of the visual and audio elements in the scene."
    )
    contains_vocal_music: bool = Field(
        description="True if the scene contains music with vocals."
    )
    song_title: Optional[str] = Field(
        default=None, description="The title of the song detected in the scene."
    )
    original_artist: Optional[str] = Field(
        default=None, description="The original artist or composer of the song."
    )
    performer_in_video: Optional[str] = Field(
        default=None,
        description="The performer singing in the video, if different from the original artist.",
    )
    reference_lyrics_og: Optional[str] = Field(
        default=None,
        description="The full lyrics of the song in the original language, found via web search.",
    )
    reference_lyrics_en: Optional[str] = Field(
        default=None,
        description="The English translation of the lyrics, found via web search.",
    )


class LyricsSceneAiResponse(BaseModel):
    """
    Represents the structured JSON response from the AI for the lyrics/scene detection pass.

    This model captures the AI's analysis of a video segment, including a breakdown
    of scenes, identification of music, and any lyrics found through web searches.
    This data is then used as context for the main subtitle generation pass.
    """

    step_by_step_log: str
    global_summary: str
    scenes: list[Scene] = Field(
        description="A list of chronological scenes detected in the video segment."
    )


# ==============================================================================
# Job & Pipeline Models
# ==============================================================================


class Job(BaseModel):
    """Base class for all job types in the processing pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="The unique name of the job, typically derived from the video segment filename (e.g., 'part_001')."
    )
    run_num_retries: NonNegativeInt = Field(
        default=0,
        description="The number of retries for this job within the current execution of the application.",
    )
    total_num_retries: NonNegativeInt = Field(
        default=0,
        description="The total number of retries for this job across all executions, loaded from the saved state.",
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
    output_file: Path = Field(
        description="Path where the re-encoded video will be saved."
    )
    fps: PositiveInt = Field(
        description="Target frames per second for the re-encoded video."
    )
    height: PositiveInt = Field(
        description="Target height (resolution) for the re-encoded video."
    )
    bitrate_kb: PositiveInt = Field(
        description="Target bitrate in kilobytes per second."
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        description="Allowed duration difference in milliseconds to consider an existing re-encoded file valid."
    )


class UploadFileJob(Job):
    """Represents a job to upload a file to the AI provider."""

    python_file: Path = Field(description="Path to the local file to be uploaded.")
    video_duration_ms: PositiveInt = Field(
        description="Duration of the video file in milliseconds."
    )


class LyricsSceneJob(Job):
    """Represents a job to detect lyrics and scenes in a video segment."""

    file: Optional[File | Path] = Field(default=None, exclude=True)
    video_duration_ms: PositiveInt
    response: Optional[LyricsSceneAiResponse] = Field(
        default=None,
        description="The structured AI response after successful processing.",
    )
    lyrics_prompt_version: Optional[int] = Field(
        default=None,
        description="The version of the prompt used to generate the response, for cache validation.",
    )

    @classmethod
    def load(cls, save_path: Path) -> Optional["LyricsSceneJob"]:
        """Loads the job from a JSON file, checking for prompt version mismatch."""
        # Local import to avoid circular dependency
        from ai_sub.prompt import LYRICS_PROMPT_VERSION

        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    job = cls.model_validate_json(f.read())
                except ValidationError as e:
                    logfire.warning(
                        f"Validation failed for {save_path.name}, ignoring cache. Error: {e}"
                    )
                    return None

            if job.lyrics_prompt_version != LYRICS_PROMPT_VERSION:
                logfire.info(
                    f"Lyrics prompt version mismatch for {job.name} (file: {job.lyrics_prompt_version}, current: {LYRICS_PROMPT_VERSION}). Re-processing."
                )
                return None
            return job
        return None


class SubtitleJob(Job):
    """
    Represents a job to generate subtitles (Transcription),
    using scene/lyrics data from a `SceneResponse` as a reference.
    """

    file: Optional[File | Path] = Field(default=None, exclude=True)
    video_duration_ms: PositiveInt
    response: Optional[SubtitleAiResponse] = Field(
        default=None,
        description="The structured AI response after successful processing.",
    )
    subtitles_prompt_version: Optional[int] = Field(
        default=None,
        description="The version of the prompt used to generate the response, for cache validation.",
    )

    @classmethod
    def load(cls, save_path: Path) -> Optional["SubtitleJob"]:
        """Loads the job from a JSON file, checking for prompt version mismatch."""
        # Local import to avoid circular dependency
        from ai_sub.prompt import SUBTITLES_PROMPT_VERSION

        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    job = cls.model_validate_json(f.read())
                except ValidationError as e:
                    logfire.warning(
                        f"Validation failed for {save_path.name}, ignoring cache. Error: {e}"
                    )
                    return None

            if job.subtitles_prompt_version != SUBTITLES_PROMPT_VERSION:
                logfire.info(
                    f"Subtitles prompt version mismatch for {job.name} (file: {job.subtitles_prompt_version}, current: {SUBTITLES_PROMPT_VERSION}). Re-processing."
                )
                return None
            return job
        return None


class SegmentJobs(BaseModel):
    """
    A container for all potential jobs related to a single video segment.

    This model acts as a state-passing object that moves through the pipeline.
    As a segment completes one stage (e.g., re-encoding), the result is stored,
    and the object is passed to the next stage's queue (e.g., upload). This
    allows each runner to have access to the state and outputs of previous steps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reencode: Optional[ReEncodingJob] = Field(
        default=None, description="The re-encoding job for the segment."
    )
    upload: Optional[UploadFileJob] = Field(
        default=None, description="The file upload job for the segment."
    )
    lyrics: Optional[LyricsSceneJob] = Field(
        default=None, description="The lyrics/scene detection job for the segment."
    )
    subtitles: Optional[SubtitleJob] = Field(
        default=None, description="The final subtitle generation job for the segment."
    )
