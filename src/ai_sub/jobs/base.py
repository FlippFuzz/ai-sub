"""Base data models, YAML serialization utilities, and timestamp helpers for pipeline jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


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

    Args:
        ts_str: Potentially noisy timestamp string.

    Returns:
        str: Isolated timecode string.
    """
    match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}(?!\d)(?:\.\d{1,3})?|\d{1,2}:\d{2}(?:[.]\d{1,3}|:\d{3}(?!\d))|\d{1,2}:\d{2}(?!\d))",
        ts_str,
    )
    return match.group(1) if match else ts_str


def parse_timestamp_string_ms(timestamp_string: str) -> int:
    """Parses standard timestamp formats into total milliseconds.

    Supports 'MM:SS.mmm', 'HH:MM:SS.mmm', 'HH:MM:SS', 'MM:SS:mmm', and 'MM:SS'.

    Args:
        timestamp_string: Timestamp string to parse.

    Returns:
        int: Duration in milliseconds.

    Raises:
        ValueError: If the timestamp cannot be parsed into milliseconds.
    """
    ts = timestamp_string.strip()

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


def serialize_prompt_context(data: Any) -> str:
    """Serializes models or dictionaries into clean, indented JSON for prompt injection.

    Systematically excludes:
    - `thoughts`: Raw internal model reasoning traces.
    - `user_context`: Prevents duplicate context injection across stages.
    - `None` values: Compacts prompt size and conserves tokens.

    Args:
        data: A Pydantic model instance, dictionary, or primitive to serialize.

    Returns:
        str: Clean JSON string ready for prompt template encapsulation.
    """
    if hasattr(data, "model_dump"):
        cleaned = data.model_dump(
            mode="json",
            exclude={"thoughts": True, "user_context": True},
            exclude_none=True,
        )
    elif isinstance(data, dict):
        cleaned = {k: v for k, v in data.items() if k not in ("thoughts", "user_context") and v is not None}
    else:
        cleaned = data

    return json.dumps(cleaned, ensure_ascii=False)


class Prompt(BaseModel):
    """Container for system and user prompt strings."""

    system: str = Field(description="System instructions for the AI model.")
    user: str = Field(description="User prompt instructions and input context.")


class BaseJob(BaseModel):
    """Base class for all segment execution jobs with YAML state persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="Unique name identifier for this segment job (e.g., 'part_001').",
    )
    total_attempts: NonNegativeInt = Field(
        default=0,
        description="Total processing attempts recorded across all executions.",
    )
    user_context: str | None = Field(
        default=None,
        description="Resolved user-supplied context string applied during execution.",
    )

    def save(self, filename: Path) -> None:
        """Serializes the job state to a human-readable YAML file.

        Args:
            filename: Destination file path.
        """
        data: dict[str, Any] = self.model_dump(mode="json")
        with open(filename, "w", encoding="utf-8") as file:
            yaml.dump(data, file, Dumper=_YamlDumper, sort_keys=False, allow_unicode=True)
