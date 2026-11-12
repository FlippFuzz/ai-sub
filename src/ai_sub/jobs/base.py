"""Base data models, YAML serialization utilities, and timestamp helpers for pipeline jobs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self

import logfire
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationError,
)

if TYPE_CHECKING:
    from ai_sub.gemini_executor import GeminiExecutor
    from ai_sub.search.base import BaseSearchProvider


class _YamlDumper(yaml.SafeDumper):
    """Custom YAML dumper that formats multiline strings as literal block scalars."""


def _str_presenter(dumper: _YamlDumper, data: str) -> yaml.Node:
    r"""Formats multiline strings using the literal block scalar style '|'.

    Normalizes escaped line endings (\r, \n, \r\n) to standard newlines.

    Args:
        dumper: The active YAML dumper instance.
        data: The string to represent.

    Returns:
        yaml.Node: Represented scalar node.
    """
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_YamlDumper.add_representer(str, _str_presenter)


def clean_timestamp_string(ts_str: str) -> str:
    """Extracts a valid timestamp pattern from a potentially noisy LLM string.

    Supports both period and comma decimal delimiters (e.g. '01:23.450' and '01:23,450').

    Args:
        ts_str: Potentially noisy timestamp string.

    Returns:
        str: Isolated timecode string normalized to period decimals.
    """
    if not isinstance(ts_str, str):
        return ts_str
    match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}(?!\d)(?:[.,]\d{1,3})?|\d{1,2}:\d{2}(?:[.,]\d{1,3}|:\d{3}(?!\d))|\d{1,2}:\d{2}(?!\d))",
        ts_str,
    )
    return match.group(1).replace(",", ".") if match else ts_str


CleanTimestamp = Annotated[str, BeforeValidator(clean_timestamp_string)]


def parse_timestamp_string_ms(timestamp_string: str) -> int:
    """Parses standard timestamp formats into total milliseconds.

    Supports 'MM:SS.mmm', 'HH:MM:SS.mmm', 'HH:MM:SS', 'MM:SS:mmm', 'MM:SS',
    and comma millisecond delimiters (e.g., '01:23,450').

    Args:
        timestamp_string: Timestamp string to parse.

    Returns:
        int: Duration in milliseconds.

    Raises:
        ValueError: If the timestamp cannot be parsed into milliseconds.
    """
    ts = timestamp_string.strip().replace(",", ".")

    if "." in ts:
        split1 = ts.split(".")
        if len(split1) != 2:
            raise ValueError(f"Invalid timestamp format: {timestamp_string}")

        milliseconds = int(split1[1].ljust(3, "0")[:3])
        split2 = split1[0].split(":")

        if len(split2) == 3:
            hours = int(split2[0])
            minutes = int(split2[1])
            seconds = int(split2[2])
            return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds
        elif len(split2) == 2:
            minutes = int(split2[0])
            seconds = int(split2[1])
            return minutes * 60000 + seconds * 1000 + milliseconds
        elif len(split2) == 1:
            seconds = int(split2[0])
            return seconds * 1000 + milliseconds
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_string}")

    colon_count = ts.count(":")
    if colon_count == 2:
        split = ts.split(":")
        if len(split[2]) == 3 or int(split[2]) >= 60:
            minutes = int(split[0])
            seconds = int(split[1])
            milliseconds = int(split[2].ljust(3, "0")[:3])
            return minutes * 60000 + seconds * 1000 + milliseconds
        else:
            hours = int(split[0])
            minutes = int(split[1])
            seconds = int(split[2])
            return hours * 3600000 + minutes * 60000 + seconds * 1000

    elif colon_count == 1:
        split = ts.split(":")
        minutes = int(split[0])
        seconds = int(split[1])
        return minutes * 60000 + seconds * 1000

    elif colon_count == 3:
        split = ts.split(":")
        hours = int(split[0])
        minutes = int(split[1])
        seconds = int(split[2])
        milliseconds = int(split[3].ljust(3, "0")[:3])
        return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds

    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_string}")


def format_ms_timestamp(ms: int) -> str:
    """Formats a duration in milliseconds to an 'MM:SS.mmm' string.

    Args:
        ms: Milliseconds to format.

    Returns:
        str: Formatted timestamp string.
    """
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def validate_duration_bound(
    items: list[Any],
    duration_ms: int,
    buffer_ms: int,
    item_label: str = "Item",
) -> None:
    """Ensures no item's end timestamp exceeds the segment duration plus buffer.

    Args:
        items: List of objects with an 'end' timestamp attribute.
        duration_ms: Total duration of the video segment in milliseconds.
        buffer_ms: Allowed tolerance buffer in milliseconds.
        item_label: Label for error messages (e.g., 'Scene', 'Subtitle').

    Raises:
        ValueError: If an item's end timestamp exceeds duration + buffer.
    """
    limit_ms = duration_ms + buffer_ms
    for i, item in enumerate(items):
        end_ms = parse_timestamp_string_ms(item.end)
        if end_ms > limit_ms:
            raise ValueError(
                f"{item_label} {i} end time ({item.end}) exceeds duration "
                f"({format_ms_timestamp(duration_ms)}) by more than {buffer_ms}ms."
            )


def _clean_context_item(data: Any) -> Any:
    """Recursively converts Pydantic models, lists, and dicts into JSON-serializable primitives.

    Args:
        data: Input object to clean.

    Returns:
        Any: JSON-safe cleaned representation.
    """
    if hasattr(data, "model_dump"):
        return data.model_dump(
            mode="json",
            exclude={"thoughts": True, "user_context": True},
            exclude_none=True,
            by_alias=True,
        )
    if isinstance(data, dict):
        return {
            k: _clean_context_item(v)
            for k, v in data.items()
            if k not in ("thoughts", "user_context") and v is not None
        }
    if isinstance(data, (list, tuple)):
        return [_clean_context_item(item) for item in data if item is not None]
    return data


def serialize_prompt_context(data: Any) -> str:
    """Serializes models, lists of models, or dictionaries into clean, compact JSON for prompt injection.

    Systematically excludes:
    - `thoughts`: Raw internal model reasoning traces.
    - `user_context`: Prevents duplicate context injection across stages.
    - `None` values: Compacts prompt size and conserves tokens.

    Args:
        data: A Pydantic model instance, list of models, dictionary, or primitive to serialize.

    Returns:
        str: Clean JSON string ready for prompt template encapsulation.
    """
    cleaned = _clean_context_item(data)
    return json.dumps(cleaned, ensure_ascii=False)


class Prompt(BaseModel):
    """Container for system and user prompt strings."""

    system: str = Field(description="System instructions for the AI model.")
    user: str = Field(description="User prompt instructions and input context.")


@dataclass
class WorkerContext:
    """Runtime execution dependencies provided by the worker node or local runner."""

    executor: GeminiExecutor
    media_path: Path
    search_providers: list[BaseSearchProvider] = field(default_factory=list)


class BaseJob(BaseModel):
    """Base class for all pipeline execution tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="Unique name identifier for this job (e.g., 'part_001' or 'summary').",
    )
    total_attempts: NonNegativeInt = Field(
        default=0,
        description="Total processing attempts recorded across all executions.",
    )


class MediaJob(BaseJob):
    """Base class for media manipulation tasks (such as video transcoding)."""

    user_context: str | None = Field(
        default=None,
        description="Resolved user-supplied context string applied during execution.",
    )


class AiJob(BaseJob):
    """Base class for AI-executed jobs with YAML persistence and prompt version validation."""

    user_context: str | None = Field(
        default=None,
        description="Resolved user-supplied context string applied during execution.",
    )
    media_path: Path | None = Field(
        default=None,
        description="Local media path bound to this AI job.",
    )
    segment_index: int | None = Field(
        default=None,
        description="Segment index if applicable.",
    )

    prompt_version_field: ClassVar[str] = ""
    prompt_version: ClassVar[int] = 0

    def save(self, filename: Path) -> None:
        """Serializes the job state to a human-readable YAML file.

        Args:
            filename: Destination file path.
        """
        data: dict[str, Any] = self.model_dump(mode="json")
        with open(filename, "w", encoding="utf-8") as file:
            yaml.dump(data, file, Dumper=_YamlDumper, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int = 0) -> Self | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            Self | None: Loaded job instance, or None if invalid or version mismatch.
        """
        if not save_path.is_file():
            return None

        with open(save_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    logfire.warning(f"Corrupt YAML in {save_path.name}. Re-processing.")
                    return None
                job = cls.model_validate(
                    data,
                    context={
                        "video_duration_ms": data.get("video_duration_ms"),
                        "validation_buffer_ms": validation_buffer_ms,
                    },
                )
            except (yaml.YAMLError, ValidationError) as e:
                logfire.warning(f"Validation failed for {save_path.name}: {e}. Re-processing.")
                return None

        if cls.prompt_version_field and cls.prompt_version:
            file_version = getattr(job, cls.prompt_version_field, None)
            if file_version != cls.prompt_version:
                logfire.info(
                    f"Prompt version mismatch for {job.name} "
                    f"(file: v{file_version}, current: v{cls.prompt_version}). Re-processing."
                )
                return None

        return job
