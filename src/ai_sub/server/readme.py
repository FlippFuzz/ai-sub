"""Channel README.md index generation for archived subtitle repositories."""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import logfire

from ai_sub.config import atomic_write_text
from ai_sub.shortcode import parse_subtitle_track

JST = timezone(timedelta(hours=9))


def update_channel_readme(uploader_dir: Path, player_base_url: str) -> None:
    """Regenerates the channel README.md index by scanning all video_meta.json files in v/.

    Args:
        uploader_dir: Channel or uploader directory containing the 'v' video archive subdirectory.
        player_base_url: Base URL of the hosted subtitle player web app.
    """
    v_base_dir = uploader_dir / "v"
    if not v_base_dir.is_dir():
        return

    uploader = uploader_dir.name
    entries: list[dict[str, Any]] = []

    for v_dir in sorted(v_base_dir.iterdir(), key=lambda p: p.name):
        if not v_dir.is_dir() or v_dir.name.startswith("."):
            continue

        meta_file = v_dir / "video_meta.json"
        meta_data: dict[str, Any] = {}
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        meta_data = loaded
            except Exception as e:
                logfire.warning(f"Failed to read {meta_file}: {e}")

        vid = str(meta_data.get("video_id") or v_dir.name)
        title = str(meta_data.get("title") or vid)
        source_type = str(meta_data.get("source_type") or "youtube")
        ch_name = str(meta_data["channel_name"]).strip() if meta_data.get("channel_name") else None

        try:
            timestamp = int(meta_data.get("timestamp") or 0)
        except (ValueError, TypeError):
            timestamp = 0

        date_str = "Unknown"
        if timestamp > 0:
            try:
                dt = datetime.fromtimestamp(timestamp, tz=JST)
                # Non-breaking hyphens prevent the date from awkwardly splitting across lines
                date_str = dt.strftime("%Y-%m-%d").replace("-", "&#8209;")
            except Exception:
                date_str = "Unknown"

        best_model_label = "-"
        best_version_str = ""
        subtitles_list = meta_data.get("subtitles")
        if isinstance(subtitles_list, list) and subtitles_list:
            first_track = subtitles_list[0]
            if isinstance(first_track, dict):
                track_path = first_track.get("path")
                if track_path:
                    track_info = parse_subtitle_track(track_path)
                    best_model_label = track_info.display_label
                    best_version_str = track_info.prompt_version_str

        entries.append(
            {
                "video_id": vid,
                "title": title,
                "channel_name": ch_name,
                "source_type": source_type,
                "timestamp": timestamp,
                "date_str": date_str,
                "best_model_label": best_model_label,
                "best_version_str": best_version_str,
            }
        )

    readme_path = uploader_dir / "README.md"
    if not entries:
        if readme_path.is_file():
            try:
                readme_path.unlink()
            except OSError:
                pass
        return

    entries.sort(key=lambda x: (x["timestamp"], x["video_id"]), reverse=True)
    channel_display_name = next((e["channel_name"] for e in entries if e["channel_name"]), uploader)

    lines: list[str] = [
        f"# {channel_display_name}",
        "",
        "| Date (JST) | Thumbnail | Title | Best Model |",
        "| :---: | :---: | :--- | :---: |",
    ]

    clean_player_url = player_base_url.rstrip("/")
    encoded_uploader = urllib.parse.quote(uploader, safe="")

    for item in entries:
        clean_title = (
            (item["title"] or item["video_id"])
            .replace("\\", "\\\\")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("|", "\\|")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        vid = item["video_id"]
        encoded_vid = urllib.parse.quote(vid, safe="")
        model_label = item["best_model_label"]
        version_str = item["best_version_str"]

        if item["source_type"] == "youtube":
            yt_url = f"https://youtu.be/{encoded_vid}"
            p_url = f"{clean_player_url}/?uploader={encoded_uploader}&v={encoded_vid}"
            thumb_cell = (
                f'<a href="{yt_url}"><img src="https://i.ytimg.com/vi/{encoded_vid}/mqdefault.jpg" width="100" /></a>'
            )
            title_cell = f"[{clean_title}]({p_url})"
        else:
            thumb_cell = "-"
            title_cell = clean_title

        if model_label != "-":
            # Use non-breaking hyphens so lineage shortcodes stay on a single line
            safe_label = model_label.replace("-", "&#8209;")
            cell_content = f"{safe_label}<br>{version_str}" if version_str else safe_label
            model_cell = f"[{cell_content}](./v/{encoded_vid})"
        else:
            model_cell = "-"

        lines.append(f"| {item['date_str']} | {thumb_cell} | {title_cell} | {model_cell} |")

    lines.append("")
    atomic_write_text(readme_path, "\n".join(lines))
    logfire.info(f"Updated channel README at {readme_path.name}")
