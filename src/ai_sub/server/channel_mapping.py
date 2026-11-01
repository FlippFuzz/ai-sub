"""YouTube channel and uploader ID mapping manager."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import logfire
from pydantic import BaseModel, RootModel


def _extract_handle(entry: Mapping[str, Any]) -> str | None:
    """Extracts YouTube handle from uploader ID or channel URLs.

    Args:
        entry: Video or channel metadata dictionary.

    Returns:
        str | None: Isolated handle string (e.g. '@Handle') or None.
    """
    if uploader_id := entry.get("uploader_id"):
        return str(uploader_id)
    for key in ("uploader_url", "channel_url"):
        if url_val := entry.get(key):
            if match := re.search(r"/(@[\w.-]+)", str(url_val)):
                return match.group(1)
    return None


def _parse_entry_timestamp(entry: Mapping[str, Any]) -> int:
    """Extracts integer timestamp from metadata entry.

    Args:
        entry: Video metadata dictionary.

    Returns:
        int: Unix timestamp in seconds.
    """
    ts = entry.get("timestamp") or entry.get("release_timestamp") or entry.get("available_at") or 0
    try:
        return int(ts)
    except (ValueError, TypeError):
        return 0


class ChannelInfo(BaseModel):
    """Metadata for a mapped YouTube channel."""

    uploader_id: str
    channel_name: str | None = None
    timestamp: int


class ChannelMap(RootModel[dict[str, ChannelInfo]]):
    """Container mapping channel IDs to channel metadata."""

    root: dict[str, ChannelInfo]


class ChannelMapper:
    """Manages persistent mapping of YouTube channel IDs to handle IDs and channel display names."""

    mapping_file: Path
    mapping: ChannelMap

    def __init__(self, mapping_file: Path | str = "channels.json") -> None:
        """Initializes the channel mapper.

        Args:
            mapping_file: Path to the JSON mapping file. Defaults to 'channels.json'.
        """
        self.mapping_file = Path(mapping_file)
        self.mapping = ChannelMap(root={})
        self.load()

    def load(self) -> None:
        """Loads existing channel mappings from disk if present."""
        if not self.mapping_file.is_file():
            return
        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.mapping = ChannelMap.model_validate(data)
            logfire.debug(f"Loaded {len(self.mapping.root)} channel mappings from {self.mapping_file.name}")
        except Exception as e:
            logfire.warning(f"Could not load channel mapping file {self.mapping_file.name}: {e}")

    def save(self) -> None:
        """Saves current channel mappings atomically to disk in human-readable UTF-8 JSON format."""
        temp_file = self.mapping_file.with_name(f"{self.mapping_file.name}.tmp.{os.getpid()}")
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.mapping.model_dump(), f, indent=2, sort_keys=True, ensure_ascii=False)
            os.replace(temp_file, self.mapping_file)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logfire.warning(f"Failed to save channel mapping file {self.mapping_file.name}: {e}")

    def update_from_entries(self, entries: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Updates internal mappings from yt-dlp entry metadata.

        Args:
            entries: Single entry dictionary or sequence of entry dictionaries.
        """
        entries_list = [entries] if isinstance(entries, Mapping) else entries
        updated = False

        for entry in entries_list:
            if not entry:
                continue
            channel_id = entry.get("channel_id")
            uploader_id = _extract_handle(entry)
            channel_name = entry.get("channel") or entry.get("uploader")
            ts_int = _parse_entry_timestamp(entry)

            if channel_id and uploader_id:
                c_str = str(channel_id)
                u_str = str(uploader_id)
                name_str = str(channel_name) if channel_name else None
                existing = self.mapping.root.get(c_str)

                if not existing or ts_int > existing.timestamp:
                    self.mapping.root[c_str] = ChannelInfo(
                        uploader_id=u_str,
                        channel_name=name_str or (existing.channel_name if existing else None),
                        timestamp=ts_int,
                    )
                    updated = True
                elif name_str and not existing.channel_name:
                    existing.channel_name = name_str
                    updated = True

        if updated:
            self.save()

    def get_uploader_id(self, entry: Mapping[str, Any]) -> str | None:
        """Resolves uploader ID for a metadata entry, updating cache if new.

        Args:
            entry: yt-dlp metadata entry mapping.

        Returns:
            str | None: Resolved uploader identifier or None.
        """
        uploader_id = _extract_handle(entry)
        channel_id = entry.get("channel_id")
        channel_name = entry.get("channel") or entry.get("uploader")
        ts_int = _parse_entry_timestamp(entry)

        if uploader_id:
            u_str = str(uploader_id)
            if channel_id:
                c_str = str(channel_id)
                name_str = str(channel_name) if channel_name else None
                existing = self.mapping.root.get(c_str)
                if not existing or ts_int > existing.timestamp:
                    self.mapping.root[c_str] = ChannelInfo(
                        uploader_id=u_str,
                        channel_name=name_str or (existing.channel_name if existing else None),
                        timestamp=ts_int,
                    )
                    self.save()
            return u_str

        if channel_id and isinstance(channel_id, str):
            info = self.mapping.root.get(channel_id)
            if info:
                return info.uploader_id

        return None

    def get_channel_name(self, entry: Mapping[str, Any]) -> str | None:
        """Resolves human-readable channel display name for an entry.

        Args:
            entry: yt-dlp metadata entry mapping.

        Returns:
            str | None: Resolved channel display name or None.
        """
        if ch := entry.get("channel"):
            return str(ch)
        if up := entry.get("uploader"):
            return str(up)
        if c_id := entry.get("channel_id"):
            info = self.mapping.root.get(str(c_id))
            if info and info.channel_name:
                return info.channel_name
        return None

    def resolve(
        self,
        entry: Mapping[str, Any] | None = None,
        parent: Mapping[str, Any] | None = None,
        url_context: str | None = None,
    ) -> tuple[str, str]:
        """Resolves sanitized uploader ID and channel display name using fallback hierarchy.

        Args:
            entry: Optional video entry metadata from yt-dlp.
            parent: Optional parent playlist/channel tab metadata from yt-dlp.
            url_context: Optional channel handle or ID extracted from configured URL.

        Returns:
            tuple[str, str]: Resolved (uploader_id, channel_name) pair.
        """
        uploader: str | None = None
        if entry:
            uploader = self.get_uploader_id(entry)
        if not uploader and parent:
            uploader = self.get_uploader_id(parent)
        if not uploader and url_context:
            uploader = url_context
        if not uploader:
            uploader = "youtube"

        uploader_clean = str(uploader).strip("@")[:20]

        ch_name: str | None = None
        if entry:
            ch_name = self.get_channel_name(entry)
        if not ch_name and parent:
            ch_name = self.get_channel_name(parent)
        if not ch_name and url_context:
            ch_name = url_context.lstrip("@")
        if not ch_name:
            ch_name = uploader_clean

        return uploader_clean, str(ch_name)
