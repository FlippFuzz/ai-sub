from pathlib import Path
from typing import Optional

from google.genai.types import File
from pydantic import BaseModel, NonNegativeInt, PositiveInt, field_validator
from pysubs2 import SSAEvent, SSAFile


class Subtitles(BaseModel):
    """Represents a single subtitle entry with start/end times and text."""

    start: str
    end: str
    english: str
    japanese: str
    alignment_source: str
    type: str


class AiResponse(BaseModel):
    """Represents the structured response from the AI model containing a list of subtitles."""

    subtitles: list[Subtitles]
    model_name: Optional[str] = None

    @field_validator("subtitles")
    @classmethod
    def validate_timestamps(cls, v: list[Subtitles]) -> list[Subtitles]:
        for subtitle in v:
            try:
                cls._parse_timestamp_string_ms(subtitle.start)
                cls._parse_timestamp_string_ms(subtitle.end)
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
        Converts the AiResponse's subtitles into an SSAFile object.
        Handles timestamp parsing and combines English and Japanese text.

        Returns:
            SSAFile: An SSAFile object containing the parsed subtitles.
        """
        subtitles = SSAFile()

        for subtitle in self.subtitles:
            start = AiResponse._parse_timestamp_string_ms(subtitle.start)
            end = AiResponse._parse_timestamp_string_ms(subtitle.end)
            english_text = subtitle.english.strip()
            japanese_text = subtitle.japanese.strip()

            # If Gemini returns the same text for En and Jp, just use the Jp
            if english_text.lower() == japanese_text.lower():
                text = japanese_text
            else:
                text = f"{japanese_text}\n{english_text}"

            subtitles.append(SSAEvent(start=start, end=end, text=text))

        return subtitles


class Job(BaseModel):
    run_num_retries: NonNegativeInt = 0
    total_num_retries: NonNegativeInt = 0


class UploadFileJob(Job):
    """Represents a job to upload a file to the AI provider."""

    python_file: Path
    video_duration_ms: PositiveInt


class SubtitleJob(Job):
    """Represents a job to generate subtitles for a specific file."""

    name: str
    file: File | Path
    video_duration_ms: PositiveInt
    response: Optional[AiResponse] = None

    def save(self, filename: Path):
        """Saves the current object to a JSON file.

        Args:
            filename (Path): The path to the file where the object should be saved.
        """
        json_str = self.model_dump_json(indent=2)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(json_str)

    @staticmethod
    def load_or_return_new(
        save_path: Path, name: str, file: File | Path, video_duration_ms: int
    ):
        """Loads the object from a JSON file, or returns a new object if the file doesn't exist.

        Args:
            save_path (Path): The path to the JSON file from which to load the state.

        Returns:
            State: The loaded object, or a new object if the file was not found.
        """
        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as f:
                return SubtitleJob.model_validate_json(f.read())
        else:
            return SubtitleJob(
                name=name, file=file, video_duration_ms=video_duration_ms
            )
