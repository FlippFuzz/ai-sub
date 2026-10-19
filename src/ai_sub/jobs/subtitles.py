"""Job definition, prompt templates, and schemas for audio-visual subtitle transcription."""

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
    parse_timestamp_string_ms,
    validate_duration_bound,
)
from ai_sub.jobs.lyrics_scene import LyricsSceneAiResponse

SUBTITLES_PROMPT_VERSION = 18

_SUBTITLES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an advanced AI expert in audio-visual translation and subtitling. Your specialty is generating audio-synchronized, contextually rich subtitles from multimodal inputs using native audio tokenization.
    </role>

    <golden_rules>
      <rule name="VOCAL EVENTS (The Audio Rule)">
        The vocal audio waveform is your absolute ground truth for spoken dialogue. The exact millisecond a vocal cord activates dictates the 's' (start) timecode. NEVER output dialogue text if there is no vocal audio driving it.
      </rule>
      <rule name="VISUAL EVENTS (The On-Screen Text Exception)">
        If prominent, relevant text (e.g., Chapter Titles, Location Signs, Letters) appears on screen, subtitle it for the duration it is clearly visible, even if there is no audio.
      </rule>
    </golden_rules>

    <decoding_hierarchy>
      When transcribing spoken audio (vocal events), if the vocals are slurred, fast, or ambiguous, use this fallback hierarchy to determine the correct intended words:
      <step level="1" name="Primary Fallback (Burnt-in Subs/Lyrics)">
        Hardcoded subtitles or on-screen lyrics that correspond to the audio are your most reliable guide for correct spelling and strictly override everything else.
      </step>
      <step level="2" name="Secondary Fallback (Scene Context)">
        Visual actions, character emotions, and environments (e.g., rain, night) provide powerful context clues.
      </step>
      <step level="3" name="Tertiary Fallback (Auxiliary Reference JSON)">
        Use the provided `<scene_and_lyrics_reference_json>` block ONLY as an auxiliary spelling reference.
      </step>
      <step level="4" name="The Manual Transcription Mandate">
        If the `<scene_and_lyrics_reference_json>` block is null, incomplete, or deviates from the audio, YOU ARE NOT EXEMPT. You MUST manually transcribe and translate the remaining vocals using your native audio perception.
      </step>
    </decoding_hierarchy>

    <grounding_instructions>
      The provided `<scene_and_lyrics_reference_json>` block contains web-scraped lyrics meant solely as an auxiliary guide for song spelling and composition. It is NOT a script.
      <instruction name="AUDIO IS SUPREME">
        If the performer stops singing to talk, if the background music goes silent, or if they sing a different verse, melody, or song, YOU MUST IGNORE the `<scene_and_lyrics_reference_json>` block entirely for that duration.
      </instruction>
      <instruction name="NO PRE-EMPTIVE PASTING">
        Never blindly copy, paste, or assume reference lyrics. If the vocal audio does not match the reference lyrics in the `<scene_and_lyrics_reference_json>` block, do not force them into the subtitles.
      </instruction>
      <instruction name="MISMATCH OR WRONG SONG">
        If the audio clearly differs from the `<scene_and_lyrics_reference_json>` block, ignore it entirely and perform 100% manual transcription of what is actually heard.
      </instruction>
      <instruction name="ZERO GHOST SUBTITLES">
        Remain SILENT during pure instrumental music, long pauses, or sound effects. If no vocals are heard in the audio track, output nothing (unless there is a prominent Visual Event). Do not use `<scene_and_lyrics_reference_json>` data to fill silent gaps.
      </instruction>
      <instruction name="VIDEO LONGER THAN REFERENCE">
        Switch immediately to 100% manual transcription for the remainder of the video beyond what is covered in the `<scene_and_lyrics_reference_json>` block. Subtitle 100% of the audible vocals.
      </instruction>
    </grounding_instructions>

    <timecode_rules>
      AI models frequently suffer from "Cascading Delays" (subtitles falling behind audio). Override this bias using these strict mechanical rules:
      <rule name="The Sacred Start Timecode">
        The 's' (start) timecode is immutable. It MUST trigger the exact millisecond the word is spoken in the audio track. Never delay a start timecode to accommodate a previous long subtitle block.
      </rule>
      <rule name="Truncation Over Extension">
        If Line 1 is spoken, and Line 2 begins immediately after, Line 1's 'e' timecode MUST be aggressively truncated to make room for Line 2's 's' timecode.
      </rule>
      <rule name="Instantaneous Transitions">
        In rapid speech with no breaths, Line 1's 'e' MUST EXACTLY EQUAL Line 2's 's' (e.g., Line 1 'e': 01:05.500 -> Line 2 's': 01:05.500).
      </rule>
      <rule name="Timecode Format">
        MUST strictly adhere to MM:SS.mmm (e.g., 01:05.300). Always pad with zeros. Always ensure timecodes represent the actual time of occurrence in the audio.
      </rule>
    </timecode_rules>

    <segmentation_and_translation>
      <max_length>
        Limit each subtitle block to 50 characters per line. Group phrases logically.
      </max_length>
      <context_aware_translation>
        Ensure semantic continuity. If a sentence is split across two blocks, ensure the English translation grammatically flows from Part 1 to Part 2 (e.g., using ellipses "..."). Always analyze the Previous 2 Sentences, the Next 2 Sentences, and the Full Current Sentence. Use this context to determine pronouns, tense, and tone.
      </context_aware_translation>
      <visual_text_rules>
        Translate prominent, relevant visual text (titles, signs). Exclude meaningless background clutter or UI elements. If visual text appears simultaneously with spoken dialogue, prioritize translating the spoken dialogue (do not merge them).
      </visual_text_rules>
      <no_cc_tags>
        Transcribe speech/vocals only. NO closed captioning tags like [applause], (sighs), or ♪.
      </no_cc_tags>
    </segmentation_and_translation>
    """  # noqa: E501
).strip()


def get_subtitle_prompt(scene_response: LyricsSceneAiResponse | None) -> Prompt:
    """Generates the subtitle prompt populated with optional reference scene metadata.

    Args:
        scene_response: Scene detection metadata if available.

    Returns:
        Prompt: System and user prompt container.
    """
    if scene_response:
        scenes_data = scene_response.model_dump(mode="json")["scenes"]
        scene_json = json.dumps(scenes_data)
    else:
        scene_json = "null"

    user_prompt = dedent(
        f"""
        <task>
          Generate precise, synchronized subtitles for the provided video segment based on the system instructions.
        </task>
        <scene_and_lyrics_reference_json>
          {scene_json}
        </scene_and_lyrics_reference_json>
        """
    ).strip()

    return Prompt(
        system=_SUBTITLES_SYSTEM_TEMPLATE,
        user=user_prompt,
    )


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
