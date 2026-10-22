"""Job definition, prompt templates, and schemas for audio-visual subtitle transcription."""

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
    clean_timestamp_string,
    format_ms_timestamp,
    parse_timestamp_string_ms,
    serialize_prompt_context,
    validate_duration_bound,
)

SUBTITLES_PROMPT_VERSION = 19

_SUBTITLES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You generate audio-synchronized subtitles with context-aware English translations from multimodal video and audio inputs.
    </role>

    <input_priority_and_modality_separation>
      1. Primary Media:
         - Audio Waveform (16kHz): The SOLE authority for vocal start ('s') and end ('e') timestamps. Video frames are sampled coarsely (1 FPS) and CANNOT determine millisecond audio timing.
         - Video Frames: Used strictly for visual translation context (speaker identities, pronouns, physical actions) and determining the duration of on-screen text.
      2. User Directives: Follow `<user_provided_context>` for participant names, pronouns, and translation style.
      3. Background Context: Use `<global_video_summary>` to understand overarching stream lore, participants, and chapter progression.
    </input_priority_and_modality_separation>

    <subtitle_generation_pipeline>
      Process the video chronologically. For each subtitle block, execute these three steps:

      STEP 1: DETECT AND TIME (AUDIO TIMELINE)
      Identify the exact start ('s') and end ('e') timecodes:
      - Spoken & Sung Dialogue:
        * The audio waveform is your absolute ground truth for dialogue timing. Set 's' to the exact millisecond human vocal cords begin producing sound in the audio track.
        * Set 'e' to when that vocal phrase audibly finishes.
        * Instrumental Gaps: Music fanfares, intros, drum grooves, and guitar solos contain no speech or singing. Never default 's' to the previous line's 'e'; advance the timeline to when the next vocal phrase actually begins in the audio.
      - On-Screen Text Events:
        * If prominent narrative text appears on screen (signs, title cards, letters, chat messages) and is NOT read aloud, set 's' and 'e' to the visual duration it is displayed on screen.
        * If a speaker reads on-screen text aloud, subtitle the spoken audio only.

      STEP 2: TRANSCRIBE ORIGINAL ('og')
      Transcribe the original language text for that interval:
      - Dialogue & Ad-libs: Transcribe all spoken words, live stage banter, and crowd interactions as heard.
      - Sung Vocals: Transcribe the phonetics and words actually sung by the performers.
      - Tagged Visual Text: Wrap unread on-screen text in matching tags: `[Sign: <text>]`, `[Title: <text>]`, or `[Text: <text>]`.

      STEP 3: TRANSLATE TO ENGLISH ('en')
      Translate the transcribed block into natural, context-aware English:
      - Use visual context from video frames (who is on screen, who they address, visual expressions) and `<global_video_summary>` to resolve ambiguous pronouns and formal/informal tone.
      - Maintain narrative continuity across line breaks, using ellipses "..." where a thought carries over.
      - For tagged visual text, retain the identical tag in English (e.g. og: `[Sign: 立ち入り禁止]`, en: `[Sign: Keep Out]`).
    </subtitle_generation_pipeline>

    <musical_performance_detection>
      In the `detected_musical_performances` field, list all vocal musical performances or singing detected in this segment:
      - Record the song title and artist if displayed on screen, announced, or recognizable; otherwise set them to null.
      - Record approximate start and end timestamps.
      - If NO singing or vocal music occurs in this segment (e.g. pure gaming, talk, or background BGM only), leave `detected_musical_performances` empty ([]).
    </musical_performance_detection>

    <timing_and_formatting>
      <rule name="CONCURRENCY">
        Overlapping timecodes are permitted when multiple participants speak at once, when vocals harmonize, or when visual text appears while someone is speaking.
        For an individual speaker, timecodes must advance forward without self-overlapping.
      </rule>

      <rule name="NO AMBIENT SOUND TAGS">
        Subtitle spoken dialogue, sung lyrics, and tagged visual text only. Do not add sound-effect descriptions or musical symbols (such as [applause], (laughs), or ♪).
      </rule>

      <rule name="TIMECODE FORMAT">
        Strictly MM:SS.mmm format padded with zeros (e.g. 01:23.450).
      </rule>
    </timing_and_formatting>
    """  # noqa: E501
).strip()


def get_subtitles_prompt(
    user_context: str | None = None,
    global_summary: Any | None = None,
    segment_offset_ms: int = 0,
    segment_duration_ms: int = 0,
) -> Prompt:
    """Generates the prompt for audio-visual subtitle generation.

    Args:
        user_context: Optional background text or directives supplied by the user.
        global_summary: Optional global summary schema or dictionary.
        segment_offset_ms: Absolute start time offset of this segment in milliseconds.
        segment_duration_ms: Duration of this segment in milliseconds.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    parts = [
        "<task>",
        "  Generate precise, audio-synchronized subtitles and identify any musical performances for the provided video segment.",  # noqa: E501
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
        summary_json = serialize_prompt_context(global_summary)
        parts.extend(
            [
                "<global_video_summary>",
                f"  {summary_json}",
                "</global_video_summary>",
            ]
        )
    return Prompt(system=_SUBTITLES_SYSTEM_TEMPLATE, user="\n".join(parts))


class DetectedMusicalPerformance(BaseModel):
    """Represents an identified singing or musical performance detected in the segment."""

    title: str | None = Field(
        default=None,
        description="Song title if identified on screen, announced, or recognized; otherwise null.",
    )
    artist: str | None = Field(
        default=None,
        description="Original artist or composer if known; otherwise null.",
    )
    start: str = Field(description="Approximate start timecode within this segment (e.g. '00:15').")
    end: str = Field(description="Approximate end timecode within this segment (e.g. '03:45').")

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
            for key in ("start", "end"):
                if key in data and isinstance(data[key], str):
                    data[key] = clean_timestamp_string(data[key])
        return data


class Subtitles(BaseModel):
    """Represents a synchronized subtitle block."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: str = Field(alias="s", description="Start timecode in 'MM:SS.mmm' format.")
    end: str = Field(alias="e", description="End timecode in 'MM:SS.mmm' format.")
    original: str = Field(alias="og", description="Transcription in original spoken language or tagged visual text.")
    english: str = Field(alias="en", description="Context-aware natural English translation with matching visual tag.")

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
        """Validates that start timecode precedes end timecode.

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


class SubtitlesAiResponse(BaseModel):
    """Structured response schema returned by the model for subtitle generation."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    detected_musical_performances: list[DetectedMusicalPerformance] = Field(
        default_factory=list,
        description="Chronological list of detected vocal musical performances or songs. Empty if none.",
    )
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


class SubtitlesJob(BaseJob):
    """Job container holding generated subtitle state for a video segment."""

    video_duration_ms: PositiveInt = Field(description="Segment duration in milliseconds.")
    subtitles_prompt_version: int = Field(
        default=SUBTITLES_PROMPT_VERSION,
        description="Prompt version employed for this response.",
    )
    response: SubtitlesAiResponse | None = Field(
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
    def load(cls, save_path: Path, validation_buffer_ms: int) -> SubtitlesJob | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            SubtitlesJob | None: Loaded job instance, or None if invalid or version mismatch.
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
