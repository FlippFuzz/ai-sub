"""Job definition, prompt templates, and schemas for Phase 2 scene mapping and lyrics retrieval."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Self

import logfire
import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from ai_sub.jobs.base import (
    BaseJob,
    Prompt,
    clean_timestamp_string,
    format_ms_timestamp,
    parse_timestamp_string_ms,
    serialize_prompt_context,
    validate_duration_bound,
)

LYRICS_PROMPT_VERSION = 10

_LYRICS_SCENES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an AI Music and Audio Scene Analyzer. Your job is to analyze a video segment, break it down into chronological scenes, detect ALL vocal music, identify whether lyrics are present on screen, and retrieve official lyrics using on-screen text, the LRCLIB lyrics database, or web search with an audio validation loop.
    </role>

    <critical_constraints>
      <constraint name="NO MANUAL TRANSCRIPTION OF SUNG LYRICS">
        DO NOT attempt to manually transcribe the audio track or translate sung vocals on your own. You are NOT generating subtitles. Reference lyrics must come SOLELY from on-screen text or search tool results.
      </constraint>
      <constraint name="SEARCH FAILURE FALLBACK">
        If official lyrics cannot be confirmed via on-screen text or web search, set the reference lyrics field to null. Never guess or approximate lyrics.
      </constraint>
      <constraint name="SEGMENT-MATCHED LYRICS ONLY (NO FULL-SONG DUMPING)">
        When providing reference lyrics, provide ONLY the specific lines, stanzas, or verses that are ACTUALLY SUNG within this specific video segment. Absolutely NEVER dump the entire song's lyrics if only a portion is performed in this segment.
      </constraint>
      <constraint name="NO PLACEHOLDERS">
        Within the sung portion, write every word found. Never use placeholders like "(Lyrics continue)", "...", or "[chorus]".
      </constraint>
      <constraint name="LOCAL GROUND TRUTH SUPREMACY">
        If a `<global_video_summary>` block is provided, use it as an auxiliary guide for song titles and cast. However, local audio and visuals in this specific video segment strictly override the global summary if there is any contradiction.
      </constraint>
    </critical_constraints>

    <onscreen_lyrics_and_search_rules>
      <rule name="ON-SCREEN LYRICS DETECTION (has_onscreen_lyrics)">
        For every scene containing vocal singing, you MUST determine and record `has_onscreen_lyrics` (true/false).
        - If clear, burnt-in lyrics are displayed on screen in the singing language (e.g. music video captions, karaoke subtitles):
          * Set `has_onscreen_lyrics` to true.
          * Set `reference_lyrics` to null (the subtitle generator will read the text directly from the screen with high visual fidelity).
        - If vocal singing is heard and NO complete lyrics appear on screen:
          * Set `has_onscreen_lyrics` to false.
          * You MUST use your search tools to find the official lyrics.
      </rule>
      <rule name="AUDIO VALIDATION & RETRY LOOP">
        Web searches often return wrong songs with identical titles. When search results return candidate lyrics:
        - Step 1 (Validation): Cross-reference candidate lyrics against audible phonetic hooks, chorus lines, or refrain phrases heard in the audio track.
        - Step 2 (Match): If the text matches what is sung, extract ONLY the lines sung in this segment.
        - Step 3 (Mismatch / Retry): If candidate lyrics contradict what is heard (e.g. wrong song with same title, parody lyrics), reject them. Formulate a refined query in your next turn using the composer or a distinct heard lyric phrase.
        - Step 4 (Exhaustion): If after retry the lyrics still do not match what is heard, set reference_lyrics to null.
      </rule>
    </onscreen_lyrics_and_search_rules>

    <search_rules>
      <rule name="CONCURRENT INITIAL TURN">
        Execute queries concurrently in parallel batches to minimize turns. In your initial turn, call `search_lrclib_lyrics_database` AND your general web search tool simultaneously to find original-language lyrics. Do NOT wait for the lyrics database before starting your web search.
      </rule>
      <rule name="EXACT TITLES FOR LYRICS DATABASE">
        When searching the lyrics database, use the exact song title without adding keywords like 'lyrics', 'full', 'MV', or '歌詞'.
      </rule>
      <rule name="NO TRANSLATIONS">
        Only retrieve verbatim lyrics in the original singing language. NEVER use search turns to look up English translations. The downstream subtitle phase translates natively from the original lyrics.
      </rule>
      <rule name="USE ORIGINAL COMPOSER, NOT PERFORMER">
        Search using the Original Artist/Composer's name (credited on screen or noted in the global summary). Do not search using cover performers.
      </rule>
      <rule name="DO NOT TRANSLATE TITLES FOR SEARCHES">
        If a song title is in Japanese, Korean, or Chinese, search using the original script or Romaji. Never translate titles into English for queries.
      </rule>
    </search_rules>
    """  # noqa: E501
).strip()


def get_lyrics_scenes_prompt(
    user_context: str | None = None,
    global_summary: Any | None = None,
    segment_offset_ms: int = 0,
    segment_duration_ms: int = 0,
) -> Prompt:
    """Generates the prompt for Phase 2 segment scene mapping and lyrics retrieval.

    Args:
        user_context: Optional background text or directives supplied by the user.
        global_summary: Optional Phase 1 global summary schema or dictionary.
        segment_offset_ms: Absolute start time offset of this segment in milliseconds.
        segment_duration_ms: Duration of this segment in milliseconds.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    parts = [
        "<task>",
        "  Analyze the provided video segment to map scenes and detect vocal music.",
        "  Indicate whether on-screen lyrics are present (has_onscreen_lyrics).",
        "  If vocal music lacks on-screen lyrics, find official lyrics via web search, validate them against the audio,",  # noqa: E501
        "  and return ONLY the lines sung in this segment.",
        "  CRITICAL: Do NOT transcribe audio into lyrics yourself if search yields nothing; leave lyrics as null.",
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
                "  CRITICAL: All output timestamps (start, end) MUST be local timecodes relative to this file",
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
    return Prompt(system=_LYRICS_SCENES_SYSTEM_TEMPLATE, user="\n".join(parts))


class Scene(BaseModel):
    """Represents a mapped scene and detected music metadata."""

    start: str = Field(description="Start timestamp in 'MM:SS.mmm' format.")
    end: str = Field(description="End timestamp in 'MM:SS.mmm' format.")
    description: str = Field(description="Detailed visual and auditory scene context.")
    contains_vocal_music: bool = Field(description="True if active sung vocals are present.")
    has_onscreen_lyrics: bool = Field(
        default=False,
        description="True if lyrics are visibly displayed/burnt into the video during this scene.",
    )
    song_title: str | None = Field(default=None, description="Title of the detected song or null.")
    original_artist: str | None = Field(default=None, description="Composer or original artist name.")
    performer_in_video: str | None = Field(default=None, description="Cover performer seen in video.")
    original_language: str | None = Field(default=None, description="Primary singing language.")
    reference_lyrics: str | None = Field(
        default=None,
        description="Official lyrics in original language for lines sung in this segment. Null if on screen or not found.",  # noqa: E501
    )

    @model_validator(mode="before")
    @classmethod
    def clean_leakage(cls, data: Any) -> Any:
        """Cleans LLM structural leakage and escaped newlines from fields.

        Args:
            data: Raw input data.

        Returns:
            Any: Cleaned input dictionary.
        """
        if isinstance(data, dict):
            for key in ("start", "end"):
                if key in data and isinstance(data[key], str):
                    data[key] = clean_timestamp_string(data[key])
            for key in ("reference_lyrics",):
                if key in data and isinstance(data[key], str):
                    data[key] = (
                        data[key].replace("\\r", "\r").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
                    )
        return data

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Validates that start timecode precedes end timecode.

        Returns:
            Self: Validated scene instance.

        Raises:
            ValueError: If timestamp format is invalid or start time is not strictly before end time.
        """
        try:
            start_ms = parse_timestamp_string_ms(self.start)
            end_ms = parse_timestamp_string_ms(self.end)
            if start_ms >= end_ms:
                raise ValueError(f"Start time ({self.start}) must precede end time ({self.end})")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp in scene: {e}") from e
        return self


class LyricsSceneAiResponse(BaseModel):
    """Structured response schema returned by the model for scene analysis."""

    global_summary: str = Field(description="Overview of the video segment flow and musical items.")
    scenes: list[Scene] = Field(description="Chronological, non-overlapping scene items.")
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

    @model_validator(mode="after")
    def validate_duration(self, info: ValidationInfo) -> Self:
        """Validates scene bounds against video duration if context is provided.

        Args:
            info: Pydantic validation context.

        Returns:
            Self: Validated instance.
        """
        context = info.context or {}
        duration_ms = context.get("video_duration_ms")
        buffer_ms = context.get("validation_buffer_ms")
        if duration_ms is not None and buffer_ms is not None:
            validate_duration_bound(self.scenes, duration_ms, buffer_ms, item_label="Scene")
        return self


class LyricsSceneJob(BaseJob):
    """Job container holding scene and lyrics data for a video segment."""

    video_duration_ms: PositiveInt = Field(description="Segment duration in milliseconds.")
    lyrics_prompt_version: int = Field(
        default=LYRICS_PROMPT_VERSION,
        description="Prompt version employed for this response.",
    )
    response: LyricsSceneAiResponse | None = Field(
        default=None,
        description="Structured AI output once processed.",
    )

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int) -> LyricsSceneJob | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Duration tolerance in milliseconds.

        Returns:
            LyricsSceneJob | None: Loaded job instance, or None if invalid or version mismatch.
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

        if job.lyrics_prompt_version != LYRICS_PROMPT_VERSION:
            logfire.info(
                f"Lyrics prompt version mismatch for {job.name} "
                f"(file: v{job.lyrics_prompt_version}, current: v{LYRICS_PROMPT_VERSION}). Re-processing."
            )
            return None

        return job
