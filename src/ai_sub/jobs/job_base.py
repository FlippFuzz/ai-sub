"""Base data models, YAML serialization utilities, and timestamp helpers for pipeline jobs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, TypeVar

import logfire
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
)

from ai_sub.config import RetrySettings, ServiceTier, atomic_write_text

if TYPE_CHECKING:
    from ai_sub.gemini_executor import GeminiExecutor
    from ai_sub.search.search_base import BaseSearchProvider

T = TypeVar("T", bound=BaseModel)


class TaskType(str, Enum):
    """Types of distributed AI tasks executed locally or by remote workers."""

    SUMMARY = "summary"
    SUBTITLES = "subtitles"
    LYRICS = "lyrics"


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

    Supports both period and comma decimal delimiters (e.g. '01:23.450' and '01:23,450')
    as well as 'HH:MM:SS:mmm', 'MM:SS:mmm', 3-digit minute stamps (e.g. '105:30'),
    and standalone 'SS.mmm' stamps.

    Args:
        ts_str: Potentially noisy timestamp string.

    Returns:
        str: Isolated timecode string normalized to period decimals.
    """
    if not isinstance(ts_str, str):
        return ts_str
    match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3}|:\d{3}(?!\d)|(?!\d))"
        r"|\d{1,3}:\d{2}(?:[.,]\d{1,3}|:\d{3}(?!\d)|(?!\d)))",
        ts_str,
    )
    if match is None:
        match = re.search(r"(\d+[.,]\d{1,3})", ts_str)
    return match.group(1).replace(",", ".") if match else ts_str


CleanTimestamp = Annotated[str, BeforeValidator(clean_timestamp_string)]


def parse_timestamp_string_ms(timestamp_string: str) -> int:
    """Parses standard timestamp formats into total milliseconds.

    Supports 'MM:SS.mmm', 'HH:MM:SS.mmm', 'HH:MM:SS', 'MM:SS:mmm', 'HH:MM:SS:mmm',
    'MM:SS', 'SS.mmm', and comma millisecond delimiters (e.g., '01:23,450').

    Args:
        timestamp_string: Timestamp string to parse.

    Returns:
        int: Duration in milliseconds.

    Raises:
        ValueError: If the timestamp cannot be parsed into milliseconds.
    """
    ts = timestamp_string.strip().replace(",", ".")
    has_decimal = "." in ts
    ms = 0

    if has_decimal:
        head, *rest = ts.split(".")
        if len(rest) != 1:
            raise ValueError(f"Invalid timestamp format: {timestamp_string}")
        ms = int(rest[0].ljust(3, "0")[:3])
    else:
        head = ts

    raw_parts = head.split(":")
    parts = [int(p) for p in raw_parts]

    if not has_decimal and len(parts) == 3 and (len(raw_parts[2]) == 3 or parts[2] >= 60):
        hours, minutes, seconds = 0, parts[0], parts[1]
        ms = int(raw_parts[2].ljust(3, "0")[:3])
    elif not has_decimal and len(parts) == 4:
        hours, minutes, seconds = parts[0], parts[1], parts[2]
        ms = int(raw_parts[3].ljust(3, "0")[:3])
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    elif has_decimal and len(parts) == 1:
        hours, minutes, seconds = 0, 0, parts[0]
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_string}")

    return hours * 3600000 + minutes * 60000 + seconds * 1000 + ms


def format_ms_timestamp(ms: int, include_hours: bool = False) -> str:
    """Formats a duration in milliseconds to 'MM:SS.mmm' or 'HH:MM:SS.mmm'.

    Args:
        ms: Milliseconds to format.
        include_hours: When True and duration is at least 1 hour, formats as 'HH:MM:SS.mmm'.
            Defaults to False.

    Returns:
        str: Formatted timestamp string.
    """
    if include_hours and ms >= 3_600_000:
        hours, rem = divmod(ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    minutes, rem = divmod(ms, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


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
        item_label: Label for error messages (e.g., 'Subtitle').

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


def append_xml_section(parts: list[str], tag: str, *lines: str) -> None:
    """Appends an XML-tagged section with 2-space indented lines to prompt parts.

    Args:
        parts: Accumulator list of prompt lines.
        tag: XML tag name wrapping the content.
        *lines: Content lines to include inside the tag.
    """
    parts.append(f"<{tag}>")
    parts.extend(f"  {line}" for line in lines)
    parts.append(f"</{tag}>")


def build_user_prompt(
    task_lines: Sequence[str],
    user_context: str | None = None,
    global_summary: Any | None = None,
    segment_offset_ms: int = 0,
    segment_duration_ms: int = 0,
    include_local_timecode_note: bool = False,
    extra_sections: Sequence[tuple[str, Any]] = (),
) -> str:
    """Assembles an XML-sectioned user prompt from common pipeline inputs.

    Args:
        task_lines: Lines describing the primary task.
        user_context: Optional background text or directives supplied by the user. Defaults to None.
        global_summary: Optional Phase 1 global video summary object. Defaults to None.
        segment_offset_ms: Absolute start offset of the segment in milliseconds. Defaults to 0.
        segment_duration_ms: Segment duration in milliseconds. Defaults to 0.
        include_local_timecode_note: Whether to include local-timecode instructions inside the window tag.
            Defaults to False.
        extra_sections: Additional (tag, content_or_model) pairs appended after global summary. Defaults to ().

    Returns:
        str: Formatted user prompt string.
    """
    parts: list[str] = []
    append_xml_section(parts, "task", *task_lines)

    if segment_duration_ms > 0:
        start_tc = format_ms_timestamp(segment_offset_ms, include_hours=True)
        end_tc = format_ms_timestamp(segment_offset_ms + segment_duration_ms, include_hours=True)
        window_lines = [f"This video segment covers {start_tc} to {end_tc} of the full video."]
        if include_local_timecode_note:
            window_lines.extend(
                [
                    "The <global_video_summary> references absolute full-video timestamps.",
                    "CRITICAL: All output timestamps (start, end, s, e) MUST be local timecodes relative to this file",
                    "starting at 00:00.000. Never use full-video absolute timestamps in your output.",
                ]
            )
        append_xml_section(parts, "segment_time_window", *window_lines)

    if user_context and user_context.strip():
        append_xml_section(parts, "user_provided_context", user_context.strip())

    if global_summary:
        append_xml_section(parts, "global_video_summary", serialize_prompt_context(global_summary))

    for tag, content in extra_sections:
        if content is None:
            continue
        if isinstance(content, str):
            if text := content.strip():
                append_xml_section(parts, tag, text)
        elif content:
            append_xml_section(parts, tag, serialize_prompt_context(content))

    return "\n".join(parts)


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
    retry_settings: RetrySettings | None = None
    validation_buffer_ms: int = 2000


class BaseJob(BaseModel):
    """Base class for all pipeline execution tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="Unique name identifier for this job (e.g., 'part_001' or 'summary').",
    )


class AiJob(BaseJob):
    """Base class for AI-executed jobs with YAML persistence and prompt version validation."""

    video_duration_ms: PositiveInt = Field(description="Media duration in milliseconds.")
    model_name: str = Field(default="gemini-3.8-flash", description="Gemini model identifier.")
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier override.",
    )
    agentic_video: bool = Field(
        default=False,
        description="Whether to enable agentic video exploration.",
    )
    user_context: str | None = Field(
        default=None,
        description="Resolved user-supplied context string applied during execution.",
    )
    segment_index: int | None = Field(
        default=None,
        description="Segment index if applicable.",
    )
    total_attempts: NonNegativeInt = Field(
        default=0,
        description="Cumulative processing attempts recorded across all executions and FPS tiers.",
    )
    fallback_tier: NonNegativeInt = Field(
        default=0,
        description="Fallback re-encoding tier index (0 = initial run, 1+ = fallback FPS tier).",
    )
    tier_attempts: NonNegativeInt = Field(
        default=0,
        description="Non-retryable failure attempts recorded at the current FPS tier.",
    )
    target_fps: PositiveFloat | Literal["keyframes"] | None = Field(
        default=None,
        description="Active FPS or 'keyframes' target for this job, or None if using raw media.",
    )
    exhausted: bool = Field(
        default=False,
        description="Whether this job has permanently exhausted all retries and fallback FPS tiers.",
    )

    task_type: ClassVar[TaskType]
    prompt_version_field: ClassVar[str] = ""
    prompt_version: ClassVar[int] = 0

    async def _run_structured(
        self,
        ctx: WorkerContext,
        prompt: Prompt,
        response_type: type[T],
        task_label: str | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any] | BaseSearchProvider] | None = None,
        validation_context: dict[str, Any] | None = None,
    ) -> T:
        """Executes structured Gemini inference using this job's execution settings and runtime context.

        Args:
            ctx: Worker runtime context containing executor, media path, retry policy, and search providers.
            prompt: System and user prompt pair.
            response_type: Expected Pydantic response schema.
            task_label: Optional telemetry span label (defaults to task_type value). Defaults to None.
            tools: Optional tool callables or search providers. Defaults to None.
            validation_context: Optional extra context passed to Pydantic validation. Defaults to None.

        Returns:
            T: Parsed and validated response instance.
        """
        return await ctx.executor.run_structured(
            prompt=prompt,
            media_path=ctx.media_path,
            video_duration_ms=self.video_duration_ms,
            response_type=response_type,
            model_name=self.model_name,
            service_tier=self.service_tier,
            tools=tools,
            agentic_video=self.agentic_video,
            task_label=task_label or self.task_type.value,
            retry_settings=ctx.retry_settings,
            validation_buffer_ms=ctx.validation_buffer_ms,
            validation_context=validation_context,
        )

    def save(self, filename: Path) -> None:
        """Serializes the job state atomically to a human-readable YAML file.

        Args:
            filename: Destination file path.
        """
        data: dict[str, Any] = self.model_dump(mode="json")
        content = yaml.dump(data, Dumper=_YamlDumper, sort_keys=False, allow_unicode=True)
        atomic_write_text(filename, content)

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int = 0) -> Self | None:
        """Loads and validates a saved YAML state file.

        Checks prompt version before Pydantic validation so schema evolutions across prompt
        bumps log a clean version mismatch instead of spurious validation errors.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            Self | None: Loaded job instance, or None if invalid or version mismatch.
        """
        if not save_path.is_file():
            return None

        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logfire.warning(f"Validation failed for {save_path.name}: {e}. Re-processing.")
            return None

        if not isinstance(data, dict):
            logfire.warning(f"Corrupt YAML in {save_path.name}. Re-processing.")
            return None

        if cls.prompt_version_field and cls.prompt_version:
            file_version = data.get(cls.prompt_version_field)
            if isinstance(file_version, bool) or file_version != cls.prompt_version:
                job_name = data.get("name") or save_path.stem
                logfire.info(
                    f"Prompt version mismatch for {job_name} "
                    f"(file: v{file_version}, current: v{cls.prompt_version}). Re-processing."
                )
                return None

        try:
            return cls.model_validate(
                data,
                context={
                    "video_duration_ms": data.get("video_duration_ms"),
                    "validation_buffer_ms": validation_buffer_ms,
                },
            )
        except ValidationError as e:
            logfire.warning(f"Validation failed for {save_path.name}: {e}. Re-processing.")
            return None
