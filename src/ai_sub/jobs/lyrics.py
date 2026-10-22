"""Job definition, prompt templates, and schemas for musical lyrics subtitle refinement."""

from __future__ import annotations

import string
from pathlib import Path
from textwrap import dedent
from typing import Any, Self

import logfire
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from pysubs2 import SSAEvent, SSAFile

from ai_sub.jobs.base import (
    BaseJob,
    Prompt,
    format_ms_timestamp,
    parse_timestamp_string_ms,
    serialize_prompt_context,
    validate_duration_bound,
)
from ai_sub.jobs.subtitles import Subtitles

LYRICS_PROMPT_VERSION = 10

_LYRICS_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an expert AI subtitle editor and musical lyric specialist. You refine draft subtitles for musical segments using official song lyrics and agentic video inspection.
    </role>

    <core_mandates>
      <mandate name="FROZEN TIMESTAMPS (IMMUTABLE TIMING)">
        You MUST preserve all start ('s') and end ('e') timestamps exactly as provided in the draft subtitles.
        - Output the EXACT SAME number of subtitle blocks in identical chronological order.
        - DO NOT add, delete, shift, or stretch timestamps. Your job is to refine text content ('og' and 'en'), NOT to re-align timing.
      </mandate>

      <mandate name="LYRIC & KANJI REFINEMENT (og)">
        Review draft lines that correspond to musical performances against the provided lyrics:
        - Correct phonetic mis-transcriptions, puns, and homophones using official song lyrics (e.g. replace misheard words with official Kanji).
        - Keep live spoken callouts, stage banter, and crowd greetings exactly as heard in the draft.
      </mandate>

      <mandate name="TRANSLATION POLISH (en)">
        Update English translations to reflect the corrected lyric meaning:
        - Inspect the video frames to verify visual context, speaker identity, and staging.
        - Maintain semantic continuity across sentence breaks (use ellipses "..." where thoughts carry over).
        - Retain identical visual tags if present (e.g. `[Sign: ...]`, `[Title: ...]`).
      </mandate>

      <mandate name="SEARCH TOOLS AS FALLBACK">
        Initial search results and lyrics are pre-fetched in `<initial_search_results>`.
        - If the pre-fetched lyrics match the song performed, use them immediately.
        - If the pre-fetched lyrics are missing, incomplete, or for the wrong song, use your search tools to look up the correct lyrics.
      </mandate>
    </core_mandates>

    <formatting_rules>
      - Return valid JSON matching the response schema.
      - Strictly maintain the MM:SS.mmm timecode format on all output blocks.
      - Do NOT output sound descriptions or tags like [applause], (sighs), or ♪.
    </formatting_rules>
    """  # noqa: E501
).strip()


def get_lyrics_prompt(
    draft_subtitles: list[Subtitles],
    detected_performances: list[Any],
    initial_search_results: str | None = None,
    user_context: str | None = None,
    global_summary: Any | None = None,
    segment_offset_ms: int = 0,
    segment_duration_ms: int = 0,
) -> Prompt:
    """Generates the prompt for musical subtitle refinement.

    Args:
        draft_subtitles: List of initial subtitles.
        detected_performances: List of detected musical performances.
        initial_search_results: Optional pre-fetched lyrics and search text from Python.
        user_context: Optional background text or directives supplied by the user.
        global_summary: Optional global summary schema or dictionary.
        segment_offset_ms: Absolute start time offset of this segment in milliseconds.
        segment_duration_ms: Duration of this segment in milliseconds.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    parts = [
        "<task>",
        "  Refine the provided draft subtitles for the musical performance. Correct Japanese lyrics and polish English translations while keeping all timestamps strictly frozen.",  # noqa: E501
        "</task>",
    ]
    if segment_duration_ms > 0:
        start_tc = format_ms_timestamp(segment_offset_ms)
        end_tc = format_ms_timestamp(segment_offset_ms + segment_duration_ms)
        parts.extend(
            [
                "<segment_time_window>",
                f"  This video segment covers {start_tc} to {end_tc} of the full video.",
                "</segment_time_window>",
            ]
        )
    if user_context and user_context.strip():
        parts.extend(
            [
                "<user_provided_context>",
                f"  {user_context.strip()}",
                "</user_provided_context>",
            ]
        )
    if global_summary:
        summary_json = serialize_prompt_context(global_summary)
        parts.extend(
            [
                "<global_video_summary>",
                f"  {summary_json}",
                "</global_video_summary>",
            ]
        )
    if detected_performances:
        perf_json = serialize_prompt_context(detected_performances)
        parts.extend(
            [
                "<detected_musical_performances>",
                f"  {perf_json}",
                "</detected_musical_performances>",
            ]
        )
    if initial_search_results and initial_search_results.strip():
        parts.extend(
            [
                "<initial_search_results>",
                f"  {initial_search_results.strip()}",
                "</initial_search_results>",
            ]
        )

    draft_json = serialize_prompt_context(draft_subtitles)
    parts.extend(
        [
            "<draft_subtitles_to_refine>",
            f"  {draft_json}",
            "</draft_subtitles_to_refine>",
        ]
    )

    return Prompt(system=_LYRICS_SYSTEM_TEMPLATE, user="\n".join(parts))


class LyricsAiResponse(BaseModel):
    """Structured response schema returned by the model for musical subtitle refinement."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    subtitles: list[Subtitles] = Field(
        alias="subs",
        description="Refined chronological synchronized subtitle events with frozen timestamps.",
    )
    thoughts: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Model reasoning traces per turn, preserved on disk but omitted from LLM output schemas.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_thoughts(cls, data: Any) -> Any:
        """Ensures thoughts loaded from state files are converted to a list.

        Args:
            data: Raw input dictionary.

        Returns:
            Any: Cleaned dictionary.
        """
        if isinstance(data, dict):
            raw_thoughts = data.get("thoughts")
            if isinstance(raw_thoughts, str):
                data["thoughts"] = [raw_thoughts]
        return data

    def get_ssafile(self) -> SSAFile:
        """Converts refined subtitles into an SSAFile object.

        Returns:
            SSAFile: Parsed subtitle file container.
        """
        subs = SSAFile()
        translator = str.maketrans("", "", string.punctuation)

        for item in self.subtitles:
            start = parse_timestamp_string_ms(item.start)
            end = parse_timestamp_string_ms(item.end)
            en_text = item.english.strip()
            og_text = item.original.strip()

            en_norm = en_text.casefold().translate(translator)
            og_norm = og_text.casefold().translate(translator)

            text = og_text if en_norm == og_norm else f"{og_text}\n{en_text}"
            subs.append(SSAEvent(start=start, end=end, text=text))

        return subs

    @model_validator(mode="after")
    def validate_duration(self, info: ValidationInfo) -> Self:
        """Validates subtitle bounds against segment duration if context is provided.

        Args:
            info: Pydantic validation context.

        Returns:
            Self: Validated instance.
        """
        context = info.context or {}
        duration_ms = context.get("video_duration_ms")
        buffer_ms = context.get("validation_buffer_ms")
        if duration_ms is not None and buffer_ms is not None:
            validate_duration_bound(self.subtitles, duration_ms, buffer_ms, item_label="Subtitle")
        return self


class LyricsJob(BaseJob):
    """Job container holding refined subtitles for a segment with musical performance."""

    video_duration_ms: PositiveInt = Field(description="Segment duration in milliseconds.")
    lyrics_prompt_version: int = Field(
        default=LYRICS_PROMPT_VERSION,
        description="Prompt version employed for this refinement response.",
    )
    response: LyricsAiResponse | None = Field(
        default=None,
        description="Structured AI output once refined.",
    )

    def save_srt(self, srt_path: Path) -> None:
        """Saves current refined subtitle response as an SRT file.

        Args:
            srt_path: Destination file path.
        """
        if self.response:
            self.response.get_ssafile().save(str(srt_path))

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int) -> LyricsJob | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            LyricsJob | None: Loaded job instance, or None if invalid or version mismatch.
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
                    context={"validation_buffer_ms": validation_buffer_ms},
                )
            except (yaml.YAMLError, ValidationError) as e:
                logfire.warning(f"Validation failed for {save_path.name}: {e}. Re-processing.")
                return None

        if job.lyrics_prompt_version != LYRICS_PROMPT_VERSION:
            logfire.info(
                f"Lyrics prompt version mismatch for {job.name} "
                f"(file: v{job.lyrics_prompt_version}, current: v{LYRICS_PROMPT_VERSION}). Re-processing."
            )
            return None

        return job
