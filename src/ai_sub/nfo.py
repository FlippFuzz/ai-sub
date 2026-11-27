"""Jellyfin-compatible Movie NFO metadata and Markdown plot formatting utilities."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import logfire

from ai_sub.config import atomic_write_text
from ai_sub.jobs.job_base import EncodingMode, parse_timestamp_string_ms
from ai_sub.jobs.summary import SummaryAiResponse
from ai_sub.shortcode import parse_subtitle_track

if TYPE_CHECKING:
    from ai_sub.pipeline import Pipeline

_MD_ESCAPE_RE = re.compile(r"([\\*_`\[\]<>~])")
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")


def _sanitize_xml(text: str) -> str:
    """Strips control and non-character codepoints forbidden in XML 1.0 documents.

    Args:
        text: Input string for an XML element or attribute.

    Returns:
        str: XML-1.0-safe string.
    """
    return _XML_ILLEGAL_RE.sub("", text)


def _collapse_ws(text: str) -> str:
    """Collapses internal whitespace and line breaks into single spaces.

    Args:
        text: Input string to normalize.

    Returns:
        str: Trimmed single-line text.
    """
    return " ".join(text.split())


def _escape_md(text: str) -> str:
    """Escapes inline Markdown and HTML-triggering characters for Jellyfin overview rendering.

    Args:
        text: Raw text string.

    Returns:
        str: Markdown-escaped string.
    """
    return _MD_ESCAPE_RE.sub(r"\\\1", text)


def _format_number(value: float, decimals: int = 6) -> str:
    """Formats a float with bounded decimal precision and stripped trailing zeros.

    Args:
        value: Numeric value to format.
        decimals: Maximum decimal places before stripping zeros. Defaults to 6.

    Returns:
        str: Clean numeric string.
    """
    rounded = round(value, decimals)
    if rounded == 0:
        return "0"
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _format_fps_value(fps: float) -> str:
    """Formats an FPS value concisely while keeping small fallback fractions exact.

    Args:
        fps: Framerate in frames per second.

    Returns:
        str: Formatted numeric FPS string.
    """
    return _format_number(fps, decimals=2 if fps >= 1.0 else 6)


def _format_fps_and_spf(fps: float) -> str:
    """Formats a positive framerate alongside its seconds-per-frame interval for NFO output.

    Args:
        fps: Positive framerate in frames per second.

    Returns:
        str: Formatted string such as '0.25 fps (4s/frame)'.
    """
    fps_str = _format_fps_value(fps)
    spf_str = _format_number(1.0 / fps, decimals=2)
    return f"{fps_str} fps ({spf_str}s/frame)"


def _format_time_range(start_ms: int, end_ms: int) -> str:
    """Formats a millisecond interval as 'MM:SS–MM:SS' or 'HH:MM:SS–HH:MM:SS'.

    Args:
        start_ms: Start offset in milliseconds.
        end_ms: End offset in milliseconds.

    Returns:
        str: Formatted time window string.
    """
    start_s = max(0, round(start_ms / 1000))
    end_s = max(start_s, round(end_ms / 1000))
    use_hours = end_s >= 3600

    def _fmt(sec: int) -> str:
        if use_hours:
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"

    return f"{_fmt(start_s)}–{_fmt(end_s)}"


def _format_encoding_label(encoding: EncodingMode, fps: float | None, fallback_tier: int = 0) -> str:
    """Formats a stage's encoding mode, FPS, and fallback status for NFO metadata.

    Args:
        encoding: Media encoding mode ('raw', 'keyframes', 'reencoded', or 'audio').
        fps: Active numeric FPS, or None if unavailable.
        fallback_tier: Fallback tier index (0 = initial, 1+ = fallback). Defaults to 0.

    Returns:
        str: Human-readable encoding description.
    """
    suffix = " [fallback]" if fallback_tier > 0 else ""
    if encoding == "raw":
        return f"raw ({_format_fps_value(fps)} fps)" if fps is not None and fps > 0 else "raw"
    if encoding == "keyframes":
        if fps is not None and fps > 0:
            fps_str = _format_fps_value(fps)
            spf_str = _format_number(1.0 / fps, decimals=2)
            return f"keyframes ({fps_str} fps, {spf_str}s/frame)"
        return "keyframes"
    if encoding == "audio" or fps == 0:
        return f"0 fps (audio-only){suffix}"
    if fps is not None and fps > 0:
        return f"{_format_fps_and_spf(fps)}{suffix}"
    return f"reencoded{suffix}"


def _build_nfo_plot(summary: SummaryAiResponse, processing_details: str | None = None) -> str:
    """Formats all global summary sections and optional processing details into a Markdown Jellyfin `<plot>`.

    Args:
        summary: Phase 1 global video summary containing overview, chapters, entities, and glossary.
        processing_details: Optional pre-formatted Markdown section appended at the end. Defaults to None.

    Returns:
        str: Formatted plot text, or empty string if all sections are empty.
    """
    sections: list[str] = []

    raw_summary = summary.overall_summary.replace("\r\n", "\n").replace("\r", "\n").strip()
    if raw_summary:
        # Preserve paragraph breaks while escaping inline syntax and accidental ATX headings.
        lines = [line.strip() for line in raw_summary.splitlines()]
        normalized = re.sub(r"\n{2,}", "\n\n", "\n".join(lines))
        clean_summary = re.sub(r"(?m)^(#{1,6})(?=\s|$)", r"\\\1", _escape_md(normalized))
        sections.append(clean_summary)

    chapter_lines: list[str] = []
    for ch in summary.chapters:
        ch_title = _escape_md(_collapse_ws(ch.title))
        if not ch_title:
            continue
        start = _escape_md(_collapse_ws(ch.start))
        end = _escape_md(_collapse_ws(ch.end))
        header = f"**{start}–{end} — {ch_title}**" if (start and end) else f"**{ch_title}**"

        song_title = _escape_md(_collapse_ws(ch.song_title)) if ch.song_title else ""
        song_artist = _escape_md(_collapse_ws(ch.song_artist)) if ch.song_artist else ""
        if song_title and song_artist:
            song_part = f" [🎵 {song_title} — {song_artist}]"
        elif song_title or song_artist:
            song_part = f" [🎵 {song_title or song_artist}]"
        else:
            song_part = ""

        desc = _escape_md(_collapse_ws(ch.description))
        chapter_lines.append(f"- {header}{song_part}: {desc}" if desc else f"- {header}{song_part}")

    if chapter_lines:
        sections.append("\n".join([f"### Chapters ({len(chapter_lines)})", *chapter_lines]))

    entity_lines: list[str] = []
    entity_count = 0
    for ent in summary.entities:
        name = _escape_md(_collapse_ws(ent.name))
        if not name:
            continue
        entity_count += 1
        head = f"- **{name}**"
        if ent.original_name and (orig := _escape_md(_collapse_ws(ent.original_name))):
            head += f" ({orig})"
        if ent.role_or_affiliation and (role := _escape_md(_collapse_ws(ent.role_or_affiliation))):
            head += f" — {role}"
        entity_lines.append(head)

        for label, raw_val in (
            ("Lore", ent.persona_lore),
            ("Speech", ent.speech_patterns_or_ticks),
            ("Visuals", ent.visual_cues),
        ):
            if raw_val and (val := _escape_md(_collapse_ws(raw_val))):
                entity_lines.append(f"  - {label}: {val}")

    if entity_count > 0:
        sections.append("\n".join([f"### Entities ({entity_count})", *entity_lines]))

    glossary_lines: list[str] = []
    for item in summary.domain_glossary:
        term = _escape_md(_collapse_ws(item.term))
        meaning = _escape_md(_collapse_ws(item.meaning))
        if term and meaning:
            glossary_lines.append(f"- **{term}**: {meaning}")
        elif term or meaning:
            glossary_lines.append(f"- {term or meaning}")

    if glossary_lines:
        sections.append("\n".join([f"### Domain Glossary ({len(glossary_lines)})", *glossary_lines]))

    if processing_details and processing_details.strip():
        sections.append(processing_details.strip())

    return "\n\n".join(sections)


def _build_nfo_xml(
    summary: SummaryAiResponse,
    title: str | None = None,
    processing_details: str | None = None,
) -> str:
    """Builds a Jellyfin-compatible Movie NFO XML string from a global summary response.

    Args:
        summary: Phase 1 global video summary containing plot and chapter timeline.
        title: Optional video title to expose in `<title>`. Omitted when None.
        processing_details: Optional Markdown processing details appended to `<plot>`. Defaults to None.

    Returns:
        str: Formatted UTF-8 XML document string.
    """
    root = ET.Element("movie")

    if title and (clean_title := _sanitize_xml(title).strip()):
        title_el = ET.SubElement(root, "title")
        title_el.text = clean_title

    if plot_text := _build_nfo_plot(summary, processing_details=processing_details):
        if clean_plot := _sanitize_xml(plot_text).strip():
            plot_el = ET.SubElement(root, "plot")
            plot_el.text = clean_plot

    chapter_entries: list[tuple[str, int]] = []
    prev_start_s: int | None = None
    for ch in summary.chapters:
        ch_title = _sanitize_xml(ch.title).strip()
        if not ch_title:
            continue
        try:
            start_ms = parse_timestamp_string_ms(ch.start)
        except ValueError:
            continue

        raw_s = start_ms // 1000
        if prev_start_s is not None and raw_s <= prev_start_s:
            start_s = max(prev_start_s + 1, math.ceil(start_ms / 1000))
        else:
            start_s = raw_s

        prev_start_s = start_s
        chapter_entries.append((ch_title, start_s))

    if chapter_entries:
        chapters_el = ET.SubElement(root, "chapters")
        for ch_title, start_s in chapter_entries:
            ET.SubElement(chapters_el, "chapter", name=ch_title, start=str(start_s))

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    return f'<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n{body}\n'


def _build_nfo_processing_details(pipeline: Pipeline) -> str:
    """Formats lineage, encoding, failures, and in-progress segments into a Markdown NFO section.

    Args:
        pipeline: Pipeline instance holding active stage and segment states.

    Returns:
        str: Formatted '### Processing Details' Markdown section.
    """
    track_info = parse_subtitle_track(pipeline.full_shortcode)
    lineage = (
        f"{track_info.models_shortcode} ({track_info.prompt_version_str})"
        if track_info.prompt_version_str
        else track_info.models_shortcode
    )
    lines: list[str] = [
        "### Processing Details",
        f"- **Lineage**: {_escape_md(lineage)}",
        "- **Encoding**",
    ]

    if pipeline.summary_enabled:
        sum_window = _format_time_range(0, pipeline.duration_ms)
        sum_desc = _format_encoding_label(
            pipeline.summary.encoding,
            pipeline.summary.fps,
            pipeline.summary.fallback_tier,
        )
        lines.append(f"  - summary ({sum_window}): {_escape_md(sum_desc)}")

    other_counts: dict[str, int] = {}
    for seg in pipeline.segments:
        if seg.fallback_tier > 0:
            stem = seg.raw_path.stem.removesuffix(f"_{pipeline.media_tag}")
            window = _format_time_range(seg.offset_ms, seg.offset_ms + seg.duration_ms)
            desc = _format_encoding_label(seg.encoding, seg.fps, seg.fallback_tier)
            lines.append(f"  - {_escape_md(stem)} ({window}): {_escape_md(desc)}")
        else:
            label = "raw" if seg.encoding == "raw" else _format_encoding_label(seg.encoding, seg.fps)
            other_counts[label] = other_counts.get(label, 0) + 1

    if other_counts:
        if len(other_counts) == 1:
            other_summary = next(iter(other_counts))
        else:
            parts = [
                f"{count} raw" if label == "raw" else f"{count} at {label}" for label, count in other_counts.items()
            ]
            other_summary = ", ".join(parts)
        lines.append(f"  - Other segments: {_escape_md(other_summary)}")

    failure_lines: list[str] = []
    in_progress_lines: list[str] = []
    for seg in pipeline.segments:
        stem = seg.raw_path.stem.removesuffix(f"_{pipeline.media_tag}")
        window = _format_time_range(seg.offset_ms, seg.offset_ms + seg.duration_ms)
        prefix = f"  - {_escape_md(stem)} ({window})"

        if seg.subtitles_exhausted:
            failure_lines.append(f"{prefix}: Subtitles failed")
        elif seg.lyrics_exhausted:
            failure_lines.append(f"{prefix}: Lyrics failed (draft preserved)")

        sub_concluded = pipeline.segment_subtitles_concluded(seg)
        lyr_concluded = pipeline.segment_lyrics_concluded(seg)
        if not (sub_concluded and lyr_concluded):
            stage_label = "Subtitles pending" if not sub_concluded else "Lyrics pending"
            in_progress_lines.append(f"{prefix}: {stage_label}")

    if failure_lines:
        lines.append("- **Failures**")
        lines.extend(failure_lines)
    else:
        lines.append("- **Failures**: None")

    if in_progress_lines:
        lines.append("- **In Progress**")
        lines.extend(in_progress_lines)

    return "\n".join(lines)


def write_nfo(
    pipeline: Pipeline,
    output_dir: Path | None = None,
    filename_stem: str | None = None,
) -> Path | None:
    """Writes a Jellyfin-compatible .nfo metadata file from the Phase 1 global summary.

    Args:
        pipeline: Pipeline instance holding summary and segment processing states.
        output_dir: Destination directory for the .nfo file. Defaults to pipeline.job_dir.
        filename_stem: Base filename stem for the .nfo output. Defaults to pipeline.filename_stem.

    Returns:
        Path | None: Path to the written .nfo file, or None if summary data is unavailable.
    """
    if pipeline.summary.response is None:
        logfire.warning(f"Skipping .nfo generation for '{pipeline.video_id}': global summary is unavailable.")
        return None

    target_dir = output_dir if output_dir is not None else pipeline.job_dir
    stem = filename_stem if filename_stem is not None else pipeline.filename_stem
    nfo_path = target_dir / f"{stem}.nfo"

    xml_content = _build_nfo_xml(
        summary=pipeline.summary.response,
        title=pipeline.title if pipeline.source_type == "youtube" else None,
        processing_details=_build_nfo_processing_details(pipeline),
    )
    atomic_write_text(nfo_path, xml_content)
    return nfo_path
