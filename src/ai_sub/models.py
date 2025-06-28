import logging
from pathlib import Path

import google.genai.types as genai_types
import json_repair
import pysubs2
from pydantic import BaseModel
from pysubs2 import SSAFile

logger = logging.getLogger(__name__)


class GeminiResponse(BaseModel):
    response: genai_types.GenerateContentResponse
    system_instruction: str | None = None
    user_text: str | None = None
    time: float = 0
    time_with_ratelimit: float = 0


class GenerateSubtitleResponse(GeminiResponse):
    @staticmethod
    def parse_timestamp_string_ms(timestamp_string: str | None) -> int:
        """Parses a timestamp string into milliseconds.

        Supports "MM:SS.mmm", "MM:SS:mmm", and "MM:SS" formats.

        Args:
            timestamp_string (str | None): The timestamp string to parse.

        Returns:
            int: The parsed timestamp in milliseconds.

        Raises:
            ValueError: If the timestamp string is None or in an invalid format.
        """
        if timestamp_string is None:
            raise ValueError("Timestamp string cannot be None")
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

    @staticmethod
    def json_to_ssa(json_dict: dict) -> SSAFile:
        """Converts a dictionary of subtitle data into an SSAFile object.

        Args:
            json_dict (dict): A dictionary containing subtitle entries,
                              each with 'start', 'end', 'english', and 'japanese' keys.

        Returns:
            SSAFile: An SSAFile object containing the converted subtitles.
        """
        subtitles = SSAFile()

        for subtitle in json_dict:
            # Figure out the start and end timestamps
            try:
                start = GenerateSubtitleResponse.parse_timestamp_string_ms(
                    subtitle.get("start")
                )
                end = GenerateSubtitleResponse.parse_timestamp_string_ms(
                    subtitle.get("end")
                )
            except ValueError as e:
                logger.error(
                    f"Error parsing timestamp for subtitle {subtitle}: {e}. Skipping."
                )
                continue
            english_text = subtitle.get("english", "").strip()
            japanese_text = subtitle.get("japanese", "").strip()
            text = f"{japanese_text}\n{english_text}"

            # If Gemini returns the same text for En and Jp, just use the Jp
            if english_text.lower() == japanese_text.lower():
                text = japanese_text
            subtitles.append(pysubs2.SSAEvent(start=start, end=end, text=text))

        return subtitles

    def get_ssafile(self) -> SSAFile:
        """Extracts and converts the generated subtitle response into an SSAFile object.

        This method takes the raw text response from Gemini, repairs it if necessary
        (using json_repair), parses the JSON, and then converts the subtitle data
        into an SSAFile object.

        Returns:
            SSAFile: An SSAFile object containing the subtitles.

        Raises:
            ValueError: If the response text is None or the parsed JSON is not a dictionary.
        """
        if self.response.text is None:
            raise ValueError("response.text is None")

        parsed_json = json_repair.loads(self.response.text)
        if not isinstance(parsed_json, dict):
            raise ValueError("parsed_json is not a dictionary")

        raw_result = parsed_json.get("subtitles", [])
        return GenerateSubtitleResponse.json_to_ssa(raw_result)


class State(BaseModel):
    generateSubtitleResponse: GenerateSubtitleResponse | None = None

    def save(self, filename: Path):
        """Saves the current state object to a JSON file.

        Args:
            filename (Path): The path to the file where the state should be saved.
        """
        json_str = self.model_dump_json(indent=2)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(json_str)

    @staticmethod
    def load_or_return_new(save_path: Path):
        """Loads a State object from a JSON file, or returns a new empty State if the file doesn't exist.

        Args:
            save_path (Path): The path to the JSON file from which to load the state.

        Returns:
            State: The loaded State object, or a new State object if the file was not found.
        """
        if Path(save_path).is_file():
            with open(save_path, "r", encoding="utf-8") as file:
                return State.model_validate_json(file.read())
        else:
            return State()
