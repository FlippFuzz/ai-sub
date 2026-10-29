"""Job definition, prompt templates, and schemas for musical lyrics subtitle refinement."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import logfire
import yaml
from pydantic import (
    Field,
    ValidationError,
)

from ai_sub.jobs.base import (
    Prompt,
    WorkerContext,
    format_ms_timestamp,
    serialize_prompt_context,
)
from ai_sub.jobs.subtitles import (
    BaseSubtitleAiResponse,
    BaseSubtitleJob,
    DetectedMusicalPerformance,
    Subtitles,
)
from ai_sub.search.base import BaseSearchProvider
from ai_sub.search.lrclib import LrclibSearchProvider

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

      <mandate name="PHONETIC COMPATIBILITY CHECK & DRAFT PRESERVATION">
        Before applying any lyrics or search results, verify that the lyrics actually match the vocal phonetics heard in the segment.
        - If pre-fetched lyrics do not match the song actually performed, DISCARD THEM IMMEDIATELY.
        - If matching official lyrics cannot be located via search, DO NOT force-fit mismatched lyrics or alter dialogue to fit the wrong song. PRESERVE the draft subtitles ('og' and 'en') untouched as transcribed.
      </mandate>

      <mandate name="LYRIC & KANJI REFINEMENT (og)">
        Review draft lines that correspond to musical performances against verified official lyrics:
        - Correct phonetic mis-transcriptions, puns, and homophones using official song lyrics (e.g. replace misheard words with official Kanji).
        - Keep live spoken callouts, stage banter, and crowd greetings exactly as heard in the draft.
      </mandate>

      <mandate name="TRANSLATION POLISH (en)">
        Update English translations to reflect the corrected Kanji meaning:
        - Inspect video frames to verify visual context, speaker identity, and staging.
        - Polish the English line ONLY when the corrected original Kanji alters the semantic meaning of that line. Maintain narrative continuity and ellipses "..." across line breaks.
        - Retain identical visual tags if present (e.g. `[Sign: ...]`, `[Title: ...]`).
      </mandate>

      <mandate name="SEARCH TOOLS AS FALLBACK">
        Initial search results and lyrics are pre-fetched in `<initial_search_results>`.
        - If the pre-fetched lyrics match the song performed, use them immediately.
        - If the pre-fetched lyrics are missing, incomplete, or for the wrong song, use your search tools:
          * Use `search_lrclib_lyrics_database` first for exact song titles and original-language lyrics.
          * Use general web search tools if the song is not in the lyrics database or if you need English translation references.
      </mandate>
    </core_mandates>

    <formatting_rules>
      - Return valid JSON matching the response schema.
      - Strictly maintain the MM:SS.mmm timecode format on all output blocks.
      - Do NOT output sound descriptions or tags like [applause], (sighs), or ♪.
    </formatting_rules>
    """  # noqa: E501
).strip()


async def prefetch_lyrics(
    performances: list[DetectedMusicalPerformance],
    search_providers: list[BaseSearchProvider],
) -> str:
    """Pre-fetches lyrics from LRCLIB and web search providers before the first LLM turn.

    Args:
        performances: List of detected musical performances.
        search_providers: Available search providers on the worker.

    Returns:
        str: Formatted initial search results text block.
    """
    results_parts: list[str] = []

    lrclib_provider: LrclibSearchProvider | None = None
    web_provider: BaseSearchProvider | None = None

    for p in search_providers:
        if isinstance(p, LrclibSearchProvider):
            lrclib_provider = p
        elif p.provider_name not in ("builtin", "lrclib") and web_provider is None:
            web_provider = p

    for perf in performances:
        if not perf.title or not perf.title.strip():
            continue

        clean_title = perf.title.strip()
        clean_artist = perf.artist.strip() if perf.artist else None
        logfire.info(f"Pre-fetching lyrics for: '{clean_title}' by '{clean_artist or 'Unknown'}'")

        perf_found = False
        if lrclib_provider is not None:
            try:
                lrclib_results = await lrclib_provider.search_lrclib_lyrics_database(
                    track_title=clean_title,
                    artist_name=clean_artist,
                )
                if lrclib_results:
                    best = lrclib_results[0]
                    results_parts.append(
                        f"Official Lyrics for '{best.get('track_name')}' by {best.get('artist_name')}:\n"
                        f"{best.get('plain_lyrics')}"
                    )
                    perf_found = True
            except Exception as e:
                logfire.warning(f"LRCLIB pre-fetch error for '{clean_title}': {e}")

        if not perf_found and web_provider is not None:
            try:
                query = f"{clean_title} {clean_artist or ''} lyrics 歌詞"
                search_res = await web_provider.search_single(query.strip())
                if search_res:
                    snippets = "\n---\n".join(f"[{r.title}]: {r.content}" for r in search_res[:2])
                    results_parts.append(f"Web Search Results for '{clean_title}':\n{snippets}")
            except Exception as e:
                logfire.debug(f"Web search pre-fetch error for '{clean_title}': {e}")

    return "\n\n".join(results_parts)


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


class LyricsAiResponse(BaseSubtitleAiResponse):
    """Structured response schema returned by the model for musical subtitle refinement."""

    subtitles: list[Subtitles] = Field(
        alias="subs",
        description="Refined chronological synchronized subtitle events with frozen timestamps.",
    )


class LyricsJob(BaseSubtitleJob):
    """Job container holding refined subtitles for a segment with musical performance."""

    lyrics_prompt_version: int = Field(
        default=LYRICS_PROMPT_VERSION,
        description="Prompt version employed for this refinement response.",
    )
    draft_subtitles: list[Subtitles] = Field(
        default_factory=list,
        description="Draft subtitles to refine.",
    )
    detected_performances: list[DetectedMusicalPerformance] = Field(
        default_factory=list,
        description="Detected singing performances in this segment.",
    )
    response: LyricsAiResponse | None = Field(
        default=None,
        description="Structured AI output once refined.",
    )

    async def execute(self, ctx: WorkerContext) -> LyricsAiResponse:
        """Refines draft subtitles for detected singing performances using lyrics search.

        Args:
            ctx: Worker runtime context containing executor and search providers.

        Returns:
            LyricsAiResponse: Refined subtitles response.
        """
        prefetched = await prefetch_lyrics(
            performances=self.detected_performances,
            search_providers=ctx.search_providers,
        )

        prompt = get_lyrics_prompt(
            draft_subtitles=self.draft_subtitles,
            detected_performances=self.detected_performances,
            initial_search_results=prefetched,
            user_context=self.user_context,
            global_summary=self.global_summary,
            segment_offset_ms=self.segment_offset_ms,
            segment_duration_ms=self.video_duration_ms,
        )

        res = await ctx.executor.run_structured(
            prompt=prompt,
            media_path=ctx.media_path,
            video_duration_ms=self.video_duration_ms,
            response_type=LyricsAiResponse,
            model_name=ctx.executor.ai_settings.model_lyrics,
            service_tier=ctx.executor.ai_settings.service_tier_lyrics,
            tools=ctx.search_providers,
            agentic_video=ctx.executor.ai_settings.agentic_video_lyrics,
            task_label="lyrics",
        )
        self.response = res
        self.total_attempts += 1
        return res

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
