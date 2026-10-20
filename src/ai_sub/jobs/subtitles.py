"""Job definition, prompt templates, and schemas for Phase 3 audio-visual subtitle transcription."""

from __future__ import annotations

import json
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
    clean_timestamp_string,
    format_ms_timestamp,
    parse_timestamp_string_ms,
    validate_duration_bound,
)

SUBTITLES_PROMPT_VERSION = 19

_SUBTITLES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an advanced AI expert in audio-visual translation and subtitling. Your specialty is generating audio-synchronized, contextually rich subtitles from multimodal inputs using native audio perception.
    </role>

    <golden_rules>
      <rule name="VOCAL EVENTS (The Audio Rule)">
        The vocal audio waveform is your absolute ground truth for spoken dialogue. The exact millisecond a vocal cord activates dictates the 's' (start) timecode. NEVER output dialogue text if there is no vocal audio driving it.
      </rule>
      <rule name="VISUAL EVENTS (The On-Screen Text Exception)">
        If prominent, relevant text (e.g. Chapter Titles, Location Signs, Letters) appears on screen, subtitle it for the duration it is clearly visible, even if there is no audio.
      </rule>
    </golden_rules>

    <decoding_hierarchy>
      When transcribing spoken dialogue or singing, resolve the correct words using this strict priority hierarchy:
      <step level="1" name="Audible Waveform & Burnt-in On-Screen Text">
        Audible vocal phonetics and burnt-in subtitles/lyrics clearly visible on screen strictly override all external references.
      </step>
      <step level="2" name="User Context Directives">
        If `<user_provided_context>` is provided, its instructions regarding character names, pronouns, and translation tone take immediate precedence over general assumptions.
      </step>
      <step level="3" name="Global Video Summary & Glossary">
        If `<global_video_summary>` is provided, use its character roster, speaker identities, and domain glossary to resolve ambiguous nouns, jargon, and speaker attributions.
      </step>
      <step level="4" name="Auxiliary Scene & Lyrics Reference">
        Use the provided `<scene_and_lyrics_reference_json>` block ONLY as an auxiliary spelling and alignment reference for songs.
      </step>
      <step level="5" name="The Manual Transcription Mandate">
        If reference blocks are null, incomplete, or deviate from what is heard, YOU ARE NOT EXEMPT. You MUST manually transcribe and translate the remaining audible speech using native audio perception.
      </step>
    </decoding_hierarchy>

    <grounding_instructions>
      The `<scene_and_lyrics_reference_json>` block contains auxiliary reference lyrics scoped to this segment. It is NOT a rigid script.
      <instruction name="AUDIO IS SUPREME">
        If the performer stops singing to talk, if the background music goes silent, or if they sing different words, ad-libs, or parodies, IGNORE the reference lyrics and transcribe what is actually heard.
      </instruction>
      <instruction name="NO PRE-EMPTIVE PASTING">
        Never blindly paste reference lyrics into silent gaps, instrumental sections, or during spoken chatter. Only output lines when the corresponding singing is audibly heard.
      </instruction>
      <instruction name="ZERO GHOST SUBTITLES">
        Remain completely SILENT during pure instrumental music, background sound effects, or pauses. If no vocals are heard, output nothing (unless translating a prominent Visual Event).
      </instruction>
      <instruction name="TERMINOLOGY CONSISTENCY">
        Maintain consistent spelling of proper nouns and character names throughout the segment in alignment with `<global_video_summary>` and `<user_provided_context>`.
      </instruction>
    </grounding_instructions>

    <timecode_rules>
      Prevent cascading delays with these mechanical rules:
      <rule name="The Sacred Start Timecode">
        The 's' (start) timecode is immutable. It MUST trigger the exact millisecond speech begins in the audio track. Never delay a start timecode to accommodate a previous subtitle block.
      </rule>
      <rule name="Truncation Over Extension">
        If Line 1 is spoken and Line 2 begins immediately after, Line 1's 'e' (end) timecode MUST be aggressively truncated to make room for Line 2's 's' timecode.
      </rule>
      <rule name="Instantaneous Transitions">
        In rapid speech with no breaths between lines, Line 1's 'e' MUST EXACTLY EQUAL Line 2's 's' (e.g. Line 1 'e': 01:05.500 -> Line 2 's': 01:05.500).
      </rule>
      <rule name="Timecode Format">
        MUST strictly adhere to MM:SS.mmm (e.g. 01:05.300). Always pad with zeros.
      </rule>
    </timecode_rules>

    <segmentation_and_translation>
      <max_length>
        Limit each subtitle block to 50 characters per line. Group phrases logically by natural speech cadence.
      </max_length>
      <context_aware_translation>
        Ensure semantic continuity. If a sentence spans multiple blocks, ensure English grammar flows naturally (e.g. using ellipses "..."). Leverage global and user context to determine correct pronouns, verb tenses, and formal/informal tone.
      </context_aware_translation>
      <visual_text_rules>
        Translate prominent, narrative-relevant visual text. If visual text appears simultaneously with spoken dialogue, prioritize translating the spoken dialogue.
      </visual_text_rules>
      <no_cc_tags>
        Transcribe speech/vocals only. NO closed captioning tags like [applause], (sighs), or ♪.
      </no_cc_tags>
    </segmentation_and_translation>
    """  # noqa: E501
).strip()


def get_subtitle_prompt(
    user_context: str | None = None,
    global_summary: Any | None = None,
    scene_response: Any | None = None,
    segment_offset_ms: int = 0,
    segment_duration_ms: int = 0,
) -> Prompt:
    """Generates the prompt for Phase 3 subtitle generation and audio-visual alignment.

    Args:
        user_context: Optional background text or directives supplied by the user.
        global_summary: Optional Phase 1 global summary schema or dictionary.
        scene_response: Optional Phase 2 scene and lyrics response schema or dictionary.
        segment_offset_ms: Absolute start time offset of this segment in milliseconds.
        segment_duration_ms: Duration of this segment in milliseconds.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    parts = [
        "<task>",
        "  Generate precise, audio-synchronized subtitles for the provided video segment based on the system instructions.",  # noqa: E501
        "</task>",
    ]
    if segment_duration_ms > 0:
        start_tc = format_ms_timestamp(segment_offset_ms)
        end_tc = format_ms_timestamp(segment_offset_ms + segment_duration_ms)
        parts.extend(
            [
                "<segment_time_window>",
                f"  This video segment covers {start_tc} to {end_tc} of the full video.",
                "  The <global_video_summary> references absolute full-video timestamps.",
                "  CRITICAL: All output timestamps (start, end, s, e) MUST be local timecodes relative to this file",
                "  starting at 00:00.000. Never use full-video absolute timestamps in your output.",
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
        if hasattr(global_summary, "model_dump"):
            clean_summary = global_summary.model_dump(mode="json", exclude={"thoughts": True}, exclude_none=True)
        elif isinstance(global_summary, dict):
            clean_summary = {k: v for k, v in global_summary.items() if k != "thoughts" and v is not None}
        else:
            clean_summary = global_summary
        summary_json = json.dumps(clean_summary, ensure_ascii=False)
        parts.extend(
            [
                "<global_video_summary>",
                f"  {summary_json}",
                "</global_video_summary>",
            ]
        )
    if scene_response:
        if hasattr(scene_response, "model_dump"):
            clean_scene = scene_response.model_dump(mode="json", exclude={"thoughts": True}, exclude_none=True)
        elif isinstance(scene_response, dict):
            clean_scene = {k: v for k, v in scene_response.items() if k != "thoughts" and v is not None}
        else:
            clean_scene = scene_response
        scene_json = json.dumps(clean_scene, ensure_ascii=False)
        parts.extend(
            [
                "<scene_and_lyrics_reference_json>",
                f"  {scene_json}",
                "</scene_and_lyrics_reference_json>",
            ]
        )
    return Prompt(system=_SUBTITLES_SYSTEM_TEMPLATE, user="\n".join(parts))


class Subtitles(BaseModel):
    """Represents a synchronized subtitle block."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: str = Field(alias="s", description="Start timecode in 'MM:SS.mmm' format.")
    end: str = Field(alias="e", description="End timecode in 'MM:SS.mmm' format.")
    original: str = Field(alias="og", description="Transcription in original spoken language.")
    english: str = Field(alias="en", description="Context-aware natural English translation.")

    @model_validator(mode="before")
    @classmethod
    def clean_leakage(cls, data: Any) -> Any:
        """Cleans LLM structural leakage from timecodes.

        Args:
            data: Raw input dictionary.

        Returns:
            Any: Cleaned input dictionary.
        """
        if isinstance(data, dict):
            for key in ("start", "s", "end", "e"):
                if key in data and isinstance(data[key], str):
                    data[key] = clean_timestamp_string(data[key])
        return data

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Validates that start time precedes end time.

        Returns:
            Self: Validated subtitle block.

        Raises:
            ValueError: If timestamp format is invalid or start time is not strictly before end time.
        """
        try:
            start_ms = parse_timestamp_string_ms(self.start)
            end_ms = parse_timestamp_string_ms(self.end)
            if start_ms >= end_ms:
                raise ValueError(f"Start time ({self.start}) must precede end time ({self.end})")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp in subtitle: {e}") from e
        return self


class SubtitleAiResponse(BaseModel):
    """Structured response schema returned by the model for subtitle generation."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    subtitles: list[Subtitles] = Field(
        alias="subs",
        description="Chronological synchronized subtitle events.",
    )
    thoughts: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Model reasoning traces per turn, preserved on disk but omitted from LLM output schemas.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_thoughts(cls, data: Any) -> Any:
        """Ensures thoughts loaded from legacy state files are converted to a list.

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
        """Converts subtitles into an SSAFile object.

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


class SubtitleJob(BaseJob):
    """Job container holding generated subtitle state for a video segment."""

    video_duration_ms: PositiveInt = Field(description="Segment duration in milliseconds.")
    subtitles_prompt_version: int = Field(
        default=SUBTITLES_PROMPT_VERSION,
        description="Prompt version employed for this response.",
    )
    response: SubtitleAiResponse | None = Field(
        default=None,
        description="Structured subtitle response once generated.",
    )

    def save_srt(self, srt_path: Path) -> None:
        """Saves current subtitle response as an SRT file.

        Args:
            srt_path: Destination file path.
        """
        if self.response:
            self.response.get_ssafile().save(str(srt_path))

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int) -> SubtitleJob | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            SubtitleJob | None: Loaded job instance, or None if invalid or version mismatch.
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

        if job.subtitles_prompt_version != SUBTITLES_PROMPT_VERSION:
            logfire.info(
                f"Subtitles prompt version mismatch for {job.name} "
                f"(file: v{job.subtitles_prompt_version}, current: v{SUBTITLES_PROMPT_VERSION}). Re-processing."
            )
            return None

        return job
