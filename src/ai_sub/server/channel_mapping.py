"""YouTube channel and uploader ID mapping manager."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import logfire
from pydantic import BaseModel, RootModel


class ChannelInfo(BaseModel):
    """Metadata for a mapped YouTube channel."""

    uploader_id: str
    channel_name: str | None = None
    timestamp: int


class ChannelMap(RootModel[dict[str, ChannelInfo]]):
    """Container mapping channel IDs to channel metadata."""

    root: dict[str, ChannelInfo]


class ChannelMapper:
    """Manages persistent mapping of YouTube channel IDs to handle/uploader IDs and channel display names."""

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
            uploader_id = entry.get("uploader_id")
            channel_name = entry.get("channel") or entry.get("uploader")
            ts = entry.get("timestamp") or entry.get("release_timestamp") or entry.get("available_at") or 0
            try:
                ts_int = int(ts)
            except (ValueError, TypeError):
                ts_int = 0

            # Fall back to uploader_url or channel_url if uploader_id is missing from entry
            if not uploader_id:
                for url_key in ("uploader_url", "channel_url"):
                    if u_url := entry.get(url_key):
                        if m := re.search(r"/(@[\w.-]+)", str(u_url)):
                            uploader_id = m.group(1)
                            break

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
        uploader_id = entry.get("uploader_id")
        channel_id = entry.get("channel_id")
        channel_name = entry.get("channel") or entry.get("uploader")
        ts = entry.get("timestamp") or entry.get("release_timestamp") or entry.get("available_at") or 0
        try:
            ts_int = int(ts)
        except (ValueError, TypeError):
            ts_int = 0

        # Extract handle from URLs if uploader_id is absent
        if not uploader_id:
            for url_key in ("uploader_url", "channel_url"):
                if u_url := entry.get(url_key):
                    if m := re.search(r"/(@[\w.-]+)", str(u_url)):
                        uploader_id = m.group(1)
                        break

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

        Uploader ID strictly resolves to an ID/handle (never a display name):
        1. entry uploader_id (or cached ID via entry channel_id)
        2. parent tab uploader_id (or cached ID via parent channel_id)
        3. url_context (e.g. '@ChannelHandle')
        4. Fallback 'youtube'

        Args:
            entry: Optional video entry metadata from yt-dlp.
            parent: Optional parent playlist/channel tab metadata from yt-dlp.
            url_context: Optional channel handle or ID extracted from configured URL.

        Returns:
            tuple[str, str]: Resolved (uploader_id, channel_name) pair.
        """
        uploader: str | None = None

        if entry:
            uploader = self.get_uploader_id(entry) or entry.get("uploader_id")
        if not uploader and parent:
            uploader = self.get_uploader_id(parent) or parent.get("uploader_id")
        if not uploader and url_context:
            uploader = url_context
        if not uploader:
            uploader = "youtube"

        uploader_clean = str(uploader).strip("@")[:20]

        ch_name: str | None = None
        if entry:
            ch_name = self.get_channel_name(entry)
        if not ch_name and parent:
            ch_name = self.get_channel_name(parent) or parent.get("channel") or parent.get("uploader")
        if not ch_name and url_context:
            ch_name = url_context.lstrip("@")
        if not ch_name:
            ch_name = uploader_clean

        return uploader_clean, str(ch_name)
