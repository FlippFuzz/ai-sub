"""Job definition, prompt templates, and schemas for audio-visual subtitle transcription."""

from __future__ import annotations

import string
from textwrap import dedent
from typing import Any, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from pysubs2 import SSAEvent, SSAFile

from ai_sub.jobs.job_base import (
    AiJob,
    CleanTimestamp,
    Prompt,
    TaskType,
    WorkerContext,
    build_user_prompt,
    parse_timestamp_string_ms,
    validate_duration_bound,
)
from ai_sub.jobs.summary import SummaryAiResponse

SUBTITLES_PROMPT_VERSION = 19

_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)

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
      3. Background Context: Use `<global_video_summary>` to understand overarching stream lore, participants, and chapter progression. Treat macroscopic chapter song titles as unverified hints; ground musical performances strictly in what is heard and seen in this segment.
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
      - Record the official song title and artist in their original native script (Kanji/Kana/Hangul) if displayed on screen, announced, or recognizable; otherwise set them to null. NEVER translate or romanize Japanese/Korean titles unless the official title uses Latin script.
      - Title and artist are independent: record whichever is known, and leave the unknown field as null.
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
    audio_only: bool = False,
) -> Prompt:
    """Generates the prompt for audio-visual subtitle generation.

    Args:
        user_context: Optional background text or directives supplied by the user.
        global_summary: Optional global summary schema or dictionary.
        segment_offset_ms: Absolute start time offset of this segment in milliseconds.
        segment_duration_ms: Duration of this segment in milliseconds.
        audio_only: Whether the media input is audio-only without video frames. Defaults to False.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    user_prompt = build_user_prompt(
        task_lines=[
            "Generate precise, audio-synchronized subtitles and identify any musical performances for the provided video segment.",  # noqa: E501
        ],
        user_context=user_context,
        global_summary=global_summary,
        segment_offset_ms=segment_offset_ms,
        segment_duration_ms=segment_duration_ms,
        include_local_timecode_note=True,
        audio_only=audio_only,
    )
    return Prompt(system=_SUBTITLES_SYSTEM_TEMPLATE, user=user_prompt)


def _validate_time_range(start: str, end: str, label: str) -> None:
    """Validates that start and end parse as timecodes with start strictly preceding end.

    Args:
        start: Start timecode string.
        end: End timecode string.
        label: Item label for error messages.

    Raises:
        ValueError: If timestamp parsing fails or start is not strictly before end.
    """
    try:
        start_ms = parse_timestamp_string_ms(start)
        end_ms = parse_timestamp_string_ms(end)
        if start_ms >= end_ms:
            raise ValueError(f"Start time ({start}) must precede end time ({end})")
    except ValueError as e:
        raise ValueError(f"Invalid timestamp in {label}: {e}") from e


class DetectedMusicalPerformance(BaseModel):
    """Represents an identified singing or musical performance detected in the segment."""

    title: str | None = Field(
        default=None,
        description="Official song title in original native script (Kanji/Kana/Hangul) if identified; otherwise null. Do not translate or romanize.",  # noqa: E501
    )
    artist: str | None = Field(
        default=None,
        description="Original artist, group, or composer in native script if known; otherwise null. Independent of title.",  # noqa: E501
    )
    start: CleanTimestamp = Field(description="Approximate start timecode within this segment (e.g. '00:15').")
    end: CleanTimestamp = Field(description="Approximate end timecode within this segment (e.g. '03:45').")

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Validates that start timecode precedes end timecode.

        Returns:
            Self: Validated musical performance block.
        """
        _validate_time_range(self.start, self.end, "detected musical performance")
        return self


class Subtitles(BaseModel):
    """Represents a synchronized subtitle block."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    start: CleanTimestamp = Field(alias="s", description="Start timecode in 'MM:SS.mmm' format.")
    end: CleanTimestamp = Field(alias="e", description="End timecode in 'MM:SS.mmm' format.")
    original: str = Field(alias="og", description="Transcription in original spoken language or tagged visual text.")
    english: str = Field(alias="en", description="Context-aware natural English translation with matching visual tag.")

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Validates that start timecode precedes end timecode.

        Returns:
            Self: Validated subtitle block.
        """
        _validate_time_range(self.start, self.end, "subtitle")
        return self


class BaseSubtitleAiResponse(BaseModel):
    """Base structured response schema for subtitle outputs with timing and SSAFile conversion."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    subtitles: list[Subtitles] = Field(
        alias="subs",
        description="Chronological synchronized subtitle events.",
    )
    thoughts: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Model reasoning traces per turn, preserved on disk but omitted from LLM output schemas.",
    )

    def get_ssafile(self) -> SSAFile:
        """Converts subtitles into an SSAFile object.

        Returns:
            SSAFile: Parsed subtitle file container.
        """
        subs = SSAFile()

        for item in self.subtitles:
            start = parse_timestamp_string_ms(item.start)
            end = parse_timestamp_string_ms(item.end)
            en_text = item.english.strip()
            og_text = item.original.strip()

            if not og_text and not en_text:
                continue

            if not og_text:
                text = en_text
            elif not en_text:
                text = og_text
            else:
                en_norm = " ".join(en_text.casefold().translate(_PUNCT_TRANSLATOR).split())
                og_norm = " ".join(og_text.casefold().translate(_PUNCT_TRANSLATOR).split())
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


class SubtitlesAiResponse(BaseSubtitleAiResponse):
    """Structured response schema returned by the model for subtitle generation."""

    detected_musical_performances: list[DetectedMusicalPerformance] = Field(
        default_factory=list,
        description="Chronological list of detected vocal musical performances or songs. Empty if none.",
    )

    @model_validator(mode="after")
    def validate_performance_durations(self, info: ValidationInfo) -> Self:
        """Validates detected musical performance bounds against segment duration if context is provided.

        Args:
            info: Pydantic validation context.

        Returns:
            Self: Validated instance.
        """
        context = info.context or {}
        duration_ms = context.get("video_duration_ms")
        buffer_ms = context.get("validation_buffer_ms")
        if duration_ms is not None and buffer_ms is not None and self.detected_musical_performances:
            validate_duration_bound(
                self.detected_musical_performances,
                duration_ms,
                buffer_ms,
                item_label="Musical performance",
            )
        return self


class BaseSubtitleJob(AiJob):
    """Base job container holding common segment subtitle state."""

    segment_offset_ms: NonNegativeInt = Field(
        default=0,
        description="Absolute start offset of this segment in milliseconds.",
    )
    global_summary: SummaryAiResponse | None = Field(
        default=None,
        description="Optional global summary context from phase 1.",
    )

    @field_validator("global_summary", mode="wrap")
    @classmethod
    def _validate_global_summary(
        cls,
        value: Any,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> SummaryAiResponse | None:
        """Validates global_summary without applying segment-level video_duration_ms bounds.

        Args:
            value: Raw global_summary input value.
            handler: Default Pydantic field validator handler.
            info: Outer Pydantic validation context.

        Returns:
            SummaryAiResponse | None: Validated global summary or None.
        """
        if value is None:
            return None
        context = info.context
        if isinstance(context, dict) and "video_duration_ms" in context:
            # Strip the segment's video_duration_ms so full-video chapters survive segment checkpoint loads.
            summary_context = {k: v for k, v in context.items() if k != "video_duration_ms"} or None
            return SummaryAiResponse.model_validate(value, context=summary_context)
        return handler(value)


class SubtitlesJob(BaseSubtitleJob):
    """Job container holding generated subtitle state for a video segment."""

    task_type: ClassVar[TaskType] = TaskType.SUBTITLES
    prompt_version_field: ClassVar[str] = "subtitles_prompt_version"
    prompt_version: ClassVar[int] = SUBTITLES_PROMPT_VERSION

    subtitles_prompt_version: int = Field(
        default=SUBTITLES_PROMPT_VERSION,
        description="Prompt version employed for this response.",
    )
    response: SubtitlesAiResponse | None = Field(
        default=None,
        description="Structured subtitle response once generated.",
    )

    async def execute(self, ctx: WorkerContext) -> SubtitlesAiResponse:
        """Generates synchronized bilingual subtitles for this segment.

        Args:
            ctx: Worker runtime context containing executor.

        Returns:
            SubtitlesAiResponse: Parsed AI subtitles response.
        """
        prompt = get_subtitles_prompt(
            user_context=self.user_context,
            global_summary=self.global_summary,
            segment_offset_ms=self.segment_offset_ms,
            segment_duration_ms=self.video_duration_ms,
            audio_only=self.encoding == "audio",
        )
        return await self._run_structured(
            ctx=ctx,
            prompt=prompt,
            response_type=SubtitlesAiResponse,
        )
