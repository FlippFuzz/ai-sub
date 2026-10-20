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
      You are an AI Video Intelligence and Media Knowledge Analyzer. Your job is to explore the entire video timeline, build a comprehensive global index of people, topics, and terminology, detect all musical performances, and use your search tool to verify background metadata.
    </role>

    <agentic_exploration_guidelines>
      <instruction name="GLOBAL COVERAGE">
        Explore the video timeline to identify the overarching narrative, topic transitions, visual title cards, speaker introductions, and musical performances.
      </instruction>
      <instruction name="HIGH PRECISION OVER SPECULATION">
        Extract factual, verifiable entities. If a name, speaker identity, or song is ambiguous and cannot be confirmed via audio self-introductions, on-screen text, or web search, mark it as ambiguous or omit it. Never invent names, titles, or lore.
      </instruction>
      <instruction name="USER CONTEXT SUPREMACY">
        If a `<user_provided_context>` block is provided in your input, its definitions of names, spelling, relationships, and context are absolute ground truth.
      </instruction>
    </agentic_exploration_guidelines>

    <extraction_targets>
      <target name="synopsis">
        A concise chronological overview of the entire video's narrative flow, topics, and major events.
      </target>
      <target name="entities">
        All key speakers, hosts, content creators, guests, actors, or characters present. Include visual identifiers (clothing, hair, screen layout), standard English spelling, and original language names (Kanji/Kana/Hangul) if applicable.
      </target>
      <target name="domain_glossary">
        Proper nouns, media titles, game franchises, specialized jargon, recurring catchphrases, or slang used in the video with their intended meanings.
      </target>
      <target name="tracklist">
        Chronological list of all vocal songs or prominent music detected across the video, noting approximate start/end timestamps, song title, original composer/artist, performer, and singing language.
      </target>
    </extraction_targets>

    <search_rules>
      <rule name="METADATA RESOLUTION">
        Use your search tool to verify the correct romanization or kanji of names, verify song titles and original composers, or clarify unfamiliar terminology found on screen or in speech.
      </rule>
      <rule name="USE ORIGINAL COMPOSER, NOT PERFORMER">
        When searching for song metadata, search using the original artist/composer's name or title keywords. Do not include cover performers in song metadata searches.
      </rule>
      <rule name="DO NOT TRANSLATE TITLES FOR SEARCHES">
        If a title or name appears in Japanese, Korean, or Chinese, search using the original script or official Romaji. Never translate non-English titles into English keywords.
      </rule>
      <rule name="PARALLEL BATCHING">
        Execute verification queries concurrently in parallel batches to minimize roundtrips within your allowed search turns.
      </rule>
    </search_rules>
    """
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
        "  Build the global synopsis, identify all participants and domain terms, index all musical tracks,",
        "  and verify spellings and composers using your web search tool.",
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


class TrackItem(BaseModel):
    """Represents a song or musical item detected across the video timeline."""

    title: str = Field(description="Title of the detected song.")
    original_artist_or_composer: str | None = Field(
        default=None,
        description="Original artist, composer, or group who created the song.",
    )
    performer: str | None = Field(
        default=None,
        description="Performer singing or covering the song in this video.",
    )
    original_language: str | None = Field(
        default=None,
        description="Primary language of the song lyrics (e.g., 'Japanese', 'English').",
    )
    approximate_start: str | None = Field(
        default=None,
        description="Approximate start timestamp in 'MM:SS' or 'HH:MM:SS' format.",
    )
    approximate_end: str | None = Field(
        default=None,
        description="Approximate end timestamp in 'MM:SS' or 'HH:MM:SS' format.",
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
            for key in ("approximate_start", "approximate_end"):
                if key in data and isinstance(data[key], str):
                    data[key] = clean_timestamp_string(data[key])
        return data


class SummaryAiResponse(BaseModel):
    """Structured AI output schema for Phase 1 whole-video summary and knowledge extraction."""

    overall_synopsis: str = Field(
        description="Chronological synopsis of the entire video narrative flow, topics, and events.",
    )
    entities: list[EntityItem] = Field(
        default_factory=list,
        description="Roster of identified speakers, participants, and characters.",
    )
    domain_glossary: list[GlossaryItem] = Field(
        default_factory=list,
        description="Domain terminology, game/franchise jargon, slang, or proper nouns mapped to their meanings.",
    )
    tracklist: list[TrackItem] = Field(
        default_factory=list,
        description="Chronological tracklist of all musical numbers or songs performed in the video.",
    )
    thoughts: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Model reasoning traces per turn, preserved on disk but omitted from LLM output schemas.",
    )

    @property
    def glossary_dict(self) -> dict[str, str]:
        """Returns the glossary as a key-value dictionary mapping term to meaning.

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
