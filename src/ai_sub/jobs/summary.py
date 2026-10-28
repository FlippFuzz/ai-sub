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

from ai_sub.jobs.base import BaseJob, Prompt, WorkerContext, clean_timestamp_string

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
        - Musical Performances:
          * If a chapter represents a musical performance and the song title or artist is recognizable from memory, announced, or displayed on screen, record `song_title` and/or `song_artist` in official native script (e.g. Japanese Kanji/Kana, Korean Hangul).
          * NEVER translate or romanize Japanese/Korean song titles unless the official commercial release uses Latin script.
          * `song_title` and `song_artist` are independent: record whichever is known, and leave the unknown one as null.
          * If unknown or non-musical, leave them as null. NEVER invent song titles.
      </target>
      <target name="entities">
        All key participants, speakers, avatars, or characters. Include standardized English names, native script (Kanji/Kana/Hangul), visual cues (appearance, clothing, avatar layout), affiliations, persona/character lore (species, backstory, occupation), and verbal speech ticks or catchphrases (語尾 like -のじゃ, -ぺこ).
      </target>
      <target name="domain_glossary">
        Domain terminology, game mechanics, UI terms, character/item names, recurring jokes, community slang, or catchphrases mapped to their meanings.
      </target>
    </extraction_targets>

    <search_rules>
      <rule name="LORE AND ENTITY DISCOVERY">
        Use web search to uncover backgrounds, official agency profiles, character personas, lore, and verbal ticks (語尾) for newly debuted or unfamiliar participants.
        For gameplay or specialized content, search for game mechanics, franchise-specific terminology, skills/items, and streamer memes to ensure downstream subtitles translate them accurately.
      </rule>
      <rule name="PARALLEL BATCH SEARCHING">
        Execute all search queries concurrently using your multi-query search tool.
        - Bundle all uncertain entities, lore questions, game terms, or participant backgrounds into multi-query batches rather than searching one by one across multiple turns.
      </rule>
      <rule name="NO LYRIC SNIPPET SEARCHING">
        DO NOT use web search with lyric fragments or sung audio snippets to reverse-search song titles during Summary.
        - Rely solely on on-screen text, spoken announcements, or memory for song titles.
        - If a song title is unknown, record it as null.
        - Reserve your search turns exclusively for participant lore, stream context, and domain glossary.
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
        "  research participant lore, species/roles, and verbal ticks using web search in parallel batches,",
        "  and compile game mechanics and glossary terminology for accurate downstream subtitle translation.",
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
        description="Official song title in original native script (Kanji/Kana/Hangul) if known; otherwise null. Do not translate or romanize.",  # noqa: E501
    )
    song_artist: str | None = Field(
        default=None,
        description="Original artist, group, or composer in native script if known; otherwise null. Independent of song_title.",  # noqa: E501
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
    persona_lore: str | None = Field(
        default=None,
        description="Character backstory, species/race (e.g. 'salamander firefighter'), archetype, occupation, or channel lore.",  # noqa: E501
    )
    speech_patterns_or_ticks: str | None = Field(
        default=None,
        description="Verbal ticks (e.g. 語尾 like -のじゃ, -ぺこ), signature catchphrases, sentence endings, or speech registers.",  # noqa: E501
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

    async def execute(self, ctx: WorkerContext) -> SummaryAiResponse:
        """Executes whole-video summary analysis using agentic or static video inspection.

        Args:
            ctx: Worker runtime context containing executor and search providers.

        Returns:
            SummaryAiResponse: Parsed AI summary response.
        """
        prompt = get_summary_prompt(user_context=self.user_context)
        res = await ctx.executor.run_structured(
            prompt=prompt,
            media_path=ctx.media_path,
            video_duration_ms=self.video_duration_ms,
            response_type=SummaryAiResponse,
            model_name=ctx.executor.ai_settings.model_summary,
            service_tier=ctx.executor.ai_settings.service_tier_summary,
            tools=ctx.search_providers,
            agentic_video=ctx.executor.ai_settings.agentic_video_summary,
            task_label="summary",
        )
        self.response = res
        self.total_attempts += 1
        return res

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
