"""Modular job types and pipeline schemas across all processing phases."""

from ai_sub.jobs.base import (
    BaseJob,
    Prompt,
    clean_timestamp_string,
    format_ms_timestamp,
    parse_timestamp_string_ms,
    serialize_prompt_context,
    validate_duration_bound,
)
from ai_sub.jobs.music_refine import (
    MUSIC_REFINE_PROMPT_VERSION,
    MusicRefineAiResponse,
    MusicRefineJob,
    get_music_refine_prompt,
)
from ai_sub.jobs.reencode import ReencodeJob
from ai_sub.jobs.subtitle import (
    SUBTITLES_PROMPT_VERSION,
    DetectedMusicalPerformance,
    SubtitleAiResponse,
    SubtitleJob,
    Subtitles,
    get_subtitle_prompt,
)
from ai_sub.jobs.summary import (
    SUMMARY_PROMPT_VERSION,
    ChapterItem,
    EntityItem,
    GlossaryItem,
    SummaryAiResponse,
    SummaryJob,
    get_summary_prompt,
)

__all__ = [
    "MUSIC_REFINE_PROMPT_VERSION",
    "SUBTITLES_PROMPT_VERSION",
    "SUMMARY_PROMPT_VERSION",
    "BaseJob",
    "ChapterItem",
    "DetectedMusicalPerformance",
    "EntityItem",
    "GlossaryItem",
    "MusicRefineAiResponse",
    "MusicRefineJob",
    "Prompt",
    "ReencodeJob",
    "SubtitleAiResponse",
    "SubtitleJob",
    "Subtitles",
    "SummaryAiResponse",
    "SummaryJob",
    "clean_timestamp_string",
    "format_ms_timestamp",
    "get_music_refine_prompt",
    "get_subtitle_prompt",
    "get_summary_prompt",
    "parse_timestamp_string_ms",
    "serialize_prompt_context",
    "validate_duration_bound",
]
