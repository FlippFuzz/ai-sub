"""Job definition, prompt templates, and schemas for scene mapping and lyrics search."""

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
    parse_timestamp_string_ms,
    validate_duration_bound,
)

LYRICS_PROMPT_VERSION = 9

_LYRICS_SCENES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an AI Music and Audio Scene Analyzer. Your job is to analyze a video, break it down into chronological scenes, detect ALL vocal songs playing, and use your available search tool to find the official lyrics for every single song.
    </role>

    <critical_constraints>
      <constraint name="NO MANUAL TRANSCRIPTION OR TRANSLATION">
        DO NOT attempt to manually transcribe the video's audio track or translate any vocals on your own. You are NOT responsible for generating subtitles. Your ONLY source of lyrics must be the web search tool.
      </constraint>
      <constraint name="SEARCH FAILURE FALLBACK">
        If the official lyrics or translation for a detected song cannot be retrieved using the search tool, you MUST set the respective reference lyrics fields to null. Do not attempt to guess, listen-and-write, or approximate the lyrics.
      </constraint>
      <constraint name="NO LYRIC TRUNCATION OR PLACEHOLDERS">
        You MUST retrieve and provide the FULL, UNTRUNCATED lyrics of the songs from the web page. Absolutely never use placeholders like "(Lyrics continue)", "...", "[rest of lyrics]", or summarized lyric snippets. If you find the lyrics, write every single word; if you cannot find them, set the field to null.
      </constraint>
    </critical_constraints>

    <execution_pipeline>
      <step name="Scene Mapping & Recognition">
        Watch and listen to the entire video. Map scene boundaries, detect vocal music, and identify song metadata using either on-screen text or auditory recognition. Do not transcribe or translate the spoken words.
      </step>
      <step name="Song Metadata Resolution">
        For every vocal segment, resolve: Song Title, Original Artist (simplified), Performer, and Original Language.
      </step>
      <step name="Parallel Lyrics Lookup">
        Look up the official lyrics using your available search tool. To minimize expensive sequential LLM turns, you MUST group and execute as many search queries as possible in parallel in each turn. You are permitted 2 to 3 turns to refine your search using fallback queries if initial parallel searches fail, but those follow-up queries must also be executed in parallel batches rather than sequentially one-by-one.
      </step>
    </execution_pipeline>

    <search_rules>
      <rule name="USE ORIGINAL COMPOSER, NOT PERFORMER">
        Search using the Original Artist/Composer's name (often credited on-screen as "Music:", "Original:", or "Composer:"). Do NOT include the names of the video performers (e.g., VTubers singing a cover) in your search query.
      </rule>
      <rule name="DO NOT TRANSLATE TITLES FOR SEARCHES">
        If a song title appears on screen in Japanese (Kanji/Kana), search using the ORIGINAL Japanese characters or Romaji. NEVER translate a Japanese title into English for a search query.
      </rule>
      <rule name="NEVER USE QUOTATION MARKS">
        Do NOT use quotes ("") anywhere in your search query. Quotes force exact matches and break the search tool. Use plain, space-separated keywords.
      </rule>
      <rule name="PARALLEL SEARCH STRATEGY (MAX 3 SEARCH TURNS)">
        To minimize expensive LLM roundtrip turns and reduce costs, you are strictly limited to a MAXIMUM of 3 search turns (roundtrips). Individual search queries are extremely cheap, so feel free to execute multiple queries in parallel, but LLM calls are highly expensive. Group and execute your searches concurrently in parallel batches within each turn.

        Follow this multi-turn parallel strategy:
        - Turn 1 (Initial Parallel Batch): Group and run both "Initial" queries (Targeted Original + Targeted English Translation) for all detected songs at the same time in parallel. (Do not execute broad Fallback queries yet).
        - Turn 2 (Refinement Parallel Batch, if needed): Only if an Initial query failed to yield results in Turn 1, execute the corresponding broad "Fallback" query for that specific failed song in parallel.

        Determine whether to pass a single list of queries or execute multiple parallel calls natively based on the parameters specified in your search tool's JSON schema.

        Strategic Query Formulations:
        - Initial - Targeted Original: [Song Title in OG language] [Original Artist/Composer] [lyrics keyword in OG language (e.g., '歌詞' for Japanese, '가사' for Korean, '歌词' for Chinese, or 'lyrics' for English)].
        - Initial - Targeted English Translation: [Romaji or English Title] [Original Artist/Composer] lyrics English.
        - Fallback - Broad Original: [Song Title in OG language] [lyrics keyword in OG language] (omits the artist to broaden search results if targeted search fails).
        - Fallback - Broad English Translation: [Romaji or English Title] lyrics English (omits the artist to broaden translation results if targeted translation fails).
      </rule>
      <rule name="SEARCH-ONLY SOURCE OF TRUTH">
        The lyrics MUST come entirely from your search tool's results. If a search yields no results, do not attempt to listen to the audio to construct the lyrics.
      </rule>
    </search_rules>
    """  # noqa: E501
).strip()

_LYRICS_SCENES_USER_TEMPLATE = dedent(
    """
    <task>
      Analyze the provided video to map scenes and detect songs. Look up the official lyrics using the web search tool.
      CRITICAL: Do NOT transcribe the audio or translate any lyrics yourself. If you cannot find the official lyrics via web search, leave the lyrics fields as null.
    </task>
    """  # noqa: E501
).strip()


def get_lyrics_scenes_prompt() -> Prompt:
    """Generates the system and user instructions for lyrics and scene detection.

    Returns:
        Prompt: Prompt container.
    """
    return Prompt(
        system=_LYRICS_SCENES_SYSTEM_TEMPLATE,
        user=_LYRICS_SCENES_USER_TEMPLATE,
    )


class Scene(BaseModel):
    """Represents a mapped scene and detected music metadata."""

    start: str = Field(description="Start timestamp in 'MM:SS.mmm' format.")
    end: str = Field(description="End timestamp in 'MM:SS.mmm' format.")
    description: str = Field(description="Detailed visual and auditory scene context.")
    contains_vocal_music: bool = Field(description="True if active sung vocals are present.")
    song_title: str | None = Field(default=None, description="Title of the detected song or null.")
    original_artist: str | None = Field(default=None, description="Composer or original artist name.")
    performer_in_video: str | None = Field(default=None, description="Cover performer seen in video.")
    original_language: str | None = Field(default=None, description="Primary singing language.")
    reference_lyrics_og: str | None = Field(
        default=None,
        description="Official lyrics in original language from search tool. Null if not found.",
    )
    reference_lyrics_en: str | None = Field(
        default=None,
        description="Official English translation of lyrics from search tool. Null if not found.",
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
            for key in ("reference_lyrics_og", "reference_lyrics_en"):
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
