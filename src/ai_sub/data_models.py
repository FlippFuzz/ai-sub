import string
from enum import IntEnum
from pathlib import Path
from typing import Optional

from google.genai.types import File
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)
from pysubs2 import SSAEvent, SSAFile

from ai_sub.config import Settings


class AiSubResult(IntEnum):
    """Result codes for the AI subtitle generation process."""

    COMPLETE = 0
    INCOMPLETE = -1
    MAX_RETRIES_EXHAUSTED = -2


class Subtitles(BaseModel):
    """Represents a single subtitle entry with start/end times and text."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: str = Field(alias="s")
    end: str = Field(alias="e")
    original: str = Field(alias="og")
    english: str = Field(alias="en")


class SubtitleResponse(BaseModel):
    """Base class for the structured response from the AI model containing a list of subtitles."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    subtitles: list[Subtitles] = Field(alias="subs")
    model_name: Optional[str] = None

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
            start = SubtitleResponse._parse_timestamp_string_ms(subtitle.start)
            end = SubtitleResponse._parse_timestamp_string_ms(subtitle.end)
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


class AiResponse(SubtitleResponse):
    """Represents the structured response from the AI model containing a list of subtitles."""

    qa_analysis: Optional[str] = None
    global_analysis: Optional[str] = None


class SubtitlePass1Response(SubtitleResponse):
    """Response model for the first pass of subtitle generation."""

    global_analysis: str


class SubtitlePass2Response(SubtitleResponse):
    """Response model for the second pass of subtitle generation."""

    qa_analysis: str


class SubtitleGenerationState(BaseModel):
    """Represents the state of the subtitle generation process."""

    ai_sub_version: str
    subtitles_prompt_version: int
    complete: bool = True
    max_retries_exceeded: bool = False
    settings: Settings


class Scene(BaseModel):
    """Represents a detected scene in the video."""

    start: str
    end: str
    description: str
    contains_vocal_music: bool
    song_title: Optional[str] = None
    reference_lyrics_og: Optional[str] = None
    reference_lyrics_en: Optional[str] = None


class SceneResponse(BaseModel):
    """Represents the structured response from the Scene Detection pass."""

    global_summary: str
    scenes: list[Scene]


class Job(BaseModel):
    """Base class for all job types in the processing pipeline."""

    run_num_retries: NonNegativeInt = 0
    total_num_retries: NonNegativeInt = 0


class ReEncodingJob(Job):
    """Represents a job to re-encode a video file."""

    input_file: Path
    output_file: Path
    fps: PositiveInt
    height: PositiveInt
    bitrate_kb: PositiveInt


class UploadFileJob(Job):
    """Represents a job to upload a file to the AI provider."""

    python_file: Path
    video_duration_ms: PositiveInt


class LyricsSceneJob(Job):
    """Represents a job to detect lyrics and scenes in a video segment."""

    name: str
    file: File | Path
    video_duration_ms: PositiveInt
    response: Optional[SceneResponse] = None


class SubtitlePass1Job(Job):
    """
    Represents a job to generate the first pass of subtitles (Transcription),
    using scene/lyrics data from a `SceneResponse` as a reference.
    """

    name: str
    file: File | Path
    video_duration_ms: PositiveInt
    scene_response: Optional[SceneResponse] = None
    response: Optional[SubtitlePass1Response] = None


class SubtitlePass2Job(Job):
    """
    Represents a job to generate the second pass of subtitles (QA & Refinement),
    using the Pass 1 `draft` and scene/lyrics data from a `SceneResponse` as a reference.
    """

    name: str
    file: File | Path
    video_duration_ms: PositiveInt
    scene_response: Optional[SceneResponse] = None
    draft: SubtitlePass1Response
    response: Optional[SubtitlePass2Response] = None


class JobState(Job):
    """Represents the overall state of all jobs in the pipeline."""

    reencode: Optional[ReEncodingJob] = None
    upload: Optional[UploadFileJob] = None
    lyrics: dict[str, LyricsSceneJob] = Field(default_factory=dict)
    pass1: dict[str, SubtitlePass1Job] = Field(default_factory=dict)
    pass2: dict[str, SubtitlePass2Job] = Field(default_factory=dict)

    def save(self, filename: Path):
        """Saves the current object to a JSON file.

        Args:
            filename (Path): The path to the file where the object should be saved.
        """
        json_str = self.model_dump_json(indent=2)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(json_str)

    @staticmethod
    def load(save_path: Path) -> Optional["JobState"]:
        """Loads the object from a JSON file if it exists.

        Args:
            save_path (Path): The path to the JSON file from which to load the state.

        Returns:
            Optional["JobState"]: The loaded object, or None if the file was not found.
        """
        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                return JobState.model_validate_json(f.read())
        return None
