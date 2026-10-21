"""Job definition, prompt templates, and schemas for Phase 1 whole-video summary and knowledge indexing."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import logfire
import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from ai_sub.jobs.base import BaseJob, Prompt, clean_timestamp_string

SUMMARY_PROMPT_VERSION = 1

_SUMMARY_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an AI Video and Media Knowledge Analyzer. Analyze the entire video using both audio and visual streams to produce a macroscopic knowledge index and a chronological chapter timeline.
    </role>

    <multimodal_instructions>
      <instruction name="AUDIO-VISUAL EXTRACTION">
        Inspect both visual frames and audio: read on-screen title cards, stream overlays, game UI, visual scene transitions, speaker introductions, and audio dialogue.
      </instruction>
      <instruction name="USER CONTEXT SUPREMACY">
        If `<user_provided_context>` is provided, its definitions of names, terminology, and relationships take absolute precedence.
      </instruction>
    </multimodal_instructions>

    <extraction_targets>
      <target name="overall_summary">
        A concise 1-2 paragraph overview summarizing the main purpose, tone, and narrative progression of the entire video.
      </target>
      <target name="chapters">
        A chronological list of chapters dividing the video naturally wherever major activity, gameplay, or topic transitions occur.
        - Note approximate start/end timestamps and a concise title and description of each activity.
        - If a chapter represents a musical performance and the song title/artist is recognizable, announced, or displayed on screen, record `song_title` and `song_artist`. For non-musical chapters or unknown songs, leave them as null. NEVER invent song titles.
      </target>
      <target name="entities">
        All key participants, speakers, avatars, or characters. Include standardized English names, native script (Kanji/Kana/Hangul), visual cues (appearance, clothing, avatar layout), and affiliations.
      </target>
      <target name="domain_glossary">
        Domain terminology, game mechanics, character/item names, recurring jokes, slang, or catchphrases mapped to their meanings.
      </target>
    </extraction_targets>

    <search_rules>
      <rule name="PARALLEL BATCH SEARCHING">
        Execute all search queries concurrently in a single turn.
        - Emit multiple function calls simultaneously in your initial turn rather than searching sequentially across multiple turns.
        - If verifying multiple entities, game terms, or lore items, bundle all queries into one parallel batch.
      </rule>
      <rule name="SEARCH SCOPE">
        - Use web search ONLY for global stream context, main participant identities, or unfamiliar game/franchise terminology.
        - DO NOT execute individual search queries for every song in a long tracklist or karaoke setlist; rely on on-screen titles and audio announcements.
        - Search non-English terms using their original script or official Romaji.
      </rule>
    </search_rules>
    """  # noqa: E501
).strip()


def get_summary_prompt(user_context: str | None = None) -> Prompt:
    """Generates the prompt for Phase 1 whole-video summary and knowledge indexing.

    Args:
        user_context: Optional background text or directives supplied by the user.

    Returns:
        Prompt: Configured system and user prompt pair.
    """
    parts = [
        "<task>",
        "  Analyze the provided video using your agentic video exploration capabilities.",
        "  Build the global summary, divide the video into natural chapters, identify participants and domain terms,",
        "  and verify spellings and terminology using your web search tool in parallel batches.",
        "</task>",
    ]
    if user_context and user_context.strip():
        parts.extend(
            [
                "<user_provided_context>",
                f"  {user_context.strip()}",
                "</user_provided_context>",
            ]
        )
    return Prompt(system=_SUMMARY_SYSTEM_TEMPLATE, user="\n".join(parts))


class ChapterItem(BaseModel):
    """Represents a macroscopic chapter, section, or musical milestone in the video."""

    start: str = Field(description="Approximate start timestamp in 'MM:SS' or 'HH:MM:SS' format.")
    end: str = Field(description="Approximate end timestamp in 'MM:SS' or 'HH:MM:SS' format.")
    title: str = Field(description="Concise chapter title summarizing the section.")
    description: str = Field(description="Overview of topics, activities, gameplay events, or performance.")
    song_title: str | None = Field(
        default=None,
        description="Song title if this chapter is a singing performance and title is known; otherwise null.",
    )
    song_artist: str | None = Field(
        default=None,
        description="Original artist, group, or composer if known; otherwise null.",
    )

    @model_validator(mode="before")
    @classmethod
    def clean_leakage(cls, data: Any) -> Any:
        """Cleans LLM structural leakage from timestamp strings.

        Args:
            data: Raw input dictionary.

        Returns:
            Any: Cleaned dictionary.
        """
        if isinstance(data, dict):
            for key in ("start", "end"):
                if key in data and isinstance(data[key], str):
                    data[key] = clean_timestamp_string(data[key])
        return data


class EntityItem(BaseModel):
    """Represents an identified person, character, host, or participant in the video."""

    name: str = Field(description="Standardized name in English or common Romaji.")
    original_name: str | None = Field(
        default=None,
        description="Name in native script (e.g. Japanese Kanji/Kana, Korean Hangul) if applicable.",
    )
    role_or_affiliation: str | None = Field(
        default=None,
        description="Role, channel affiliation, agency, or fictional character identity.",
    )
    visual_cues: str | None = Field(
        default=None,
        description="Key visual identifiers (e.g. hair color, costume, avatar appearance, screen position).",
    )


class GlossaryItem(BaseModel):
    """Represents a domain-specific term, jargon, slang, or proper noun."""

    term: str = Field(description="The proper noun, nickname, jargon, catchphrase, or slang.")
    meaning: str = Field(description="Contextual definition, explanation, or translation.")


class SummaryAiResponse(BaseModel):
    """Structured AI output schema for Phase 1 whole-video summary and knowledge extraction."""

    overall_summary: str = Field(
        description="Concise overview summarizing the main purpose, narrative, and tone of the entire video.",
    )
    chapters: list[ChapterItem] = Field(
        default_factory=list,
        description="Chronological chapters dividing the video at natural activity or topic transitions.",
    )
    entities: list[EntityItem] = Field(
        default_factory=list,
        description="Roster of identified speakers, participants, and characters.",
    )
    domain_glossary: list[GlossaryItem] = Field(
        default_factory=list,
        description="Domain terminology, game/franchise jargon, slang, or proper nouns mapped to their meanings.",
    )
    thoughts: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Model reasoning traces per turn, preserved on disk but omitted from LLM output schemas.",
    )

    @property
    def glossary_dict(self) -> dict[str, str]:
        """The glossary is returned as a key-value dictionary mapping term to meaning.

        Returns:
            dict[str, str]: Mapping of terms to their contextual meanings.
        """
        return {item.term: item.meaning for item in self.domain_glossary}

    @model_validator(mode="before")
    @classmethod
    def normalize_glossary(cls, data: Any) -> Any:
        """Normalizes legacy dictionary format to a list of GlossaryItem objects.

        Args:
            data: Raw input dictionary.

        Returns:
            Any: Cleaned input dictionary.
        """
        if isinstance(data, dict):
            raw = data.get("domain_glossary")
            if isinstance(raw, dict):
                data["domain_glossary"] = [{"term": k, "meaning": str(v)} for k, v in raw.items()]
        return data

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


class SummaryJob(BaseJob):
    """Job container holding global summary state for an entire video."""

    video_duration_ms: PositiveInt = Field(description="Total video duration in milliseconds.")
    summary_prompt_version: int = Field(
        default=SUMMARY_PROMPT_VERSION,
        description="Prompt version employed for this summary response.",
    )
    response: SummaryAiResponse | None = Field(
        default=None,
        description="Structured AI output once processed.",
    )

    @classmethod
    def load(cls, save_path: Path, validation_buffer_ms: int = 0) -> SummaryJob | None:
        """Loads and validates a saved YAML state file.

        Args:
            save_path: State file path.
            validation_buffer_ms: Optional duration tolerance buffer in milliseconds.

        Returns:
            SummaryJob | None: Loaded job instance, or None if invalid or version mismatch.
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

        if job.summary_prompt_version != SUMMARY_PROMPT_VERSION:
            logfire.info(
                f"Summary prompt version mismatch for {job.name} "
                f"(file: v{job.summary_prompt_version}, current: v{SUMMARY_PROMPT_VERSION}). Re-processing."
            )
            return None

        return job
