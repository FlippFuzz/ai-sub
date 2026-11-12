"""Centralized YouTube client wrapping yt-dlp, PO token server, channel mapping, and media downloads."""

from __future__ import annotations

import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import sleep as time_sleep
from typing import Any, cast

import logfire
from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from pydantic import BaseModel, RootModel
from yt_dlp import YoutubeDL

from ai_sub.server.config import YoutubeSourceConfig
from ai_sub.video import ensure_ffmpeg

_VALID_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}


class YouTubeError(Exception):
    """Base exception for YouTube operations."""


class YouTubeBotDetectionError(YouTubeError):
    """Raised when YouTube returns a bot check, reCAPTCHA, or HTTP 429 response."""


class YouTubeVideoUnavailableError(YouTubeError):
    """Raised when a video is permanently inaccessible (e.g. geo-blocked, private, deleted)."""

    def __init__(self, message: str, reason: str) -> None:
        """Initializes the unavailable video error.

        Args:
            message: Raw error message.
            reason: Normalized failure classification.
        """
        super().__init__(message)
        self.reason = reason


class DiscoveredVideo(BaseModel):
    """Metadata extracted during a flat YouTube feed scan."""

    video_id: str
    uploader: str
    channel_name: str
    title: str
    source_ref: str
    timestamp: int


class DownloadedVideo(BaseModel):
    """Result of a video stream download with verified ground-truth metadata."""

    video_id: str
    uploader: str
    channel_name: str
    file_path: Path


class ChannelInfo(BaseModel):
    """Metadata for a mapped YouTube channel."""

    uploader_id: str
    channel_name: str | None = None
    timestamp: int


class ChannelMap(RootModel[dict[str, ChannelInfo]]):
    """Container mapping channel IDs to channel metadata."""

    root: dict[str, ChannelInfo]


def _extract_handle(entry: Mapping[str, Any]) -> str | None:
    """Extracts YouTube handle from uploader ID or channel URLs.

    Prioritizes '@handle' patterns found in channel/uploader URLs over raw channel IDs.

    Args:
        entry: Video or channel metadata dictionary.

    Returns:
        str | None: Isolated handle string (e.g. '@Handle') or None.
    """
    for key in ("uploader_url", "channel_url"):
        if url_val := entry.get(key):
            if match := re.search(r"/(@[\w.-]+)", str(url_val)):
                return match.group(1)

    if uploader_id := entry.get("uploader_id"):
        u_str = str(uploader_id).strip()
        if u_str.startswith("@") or not re.match(r"^UC[\w-]{22}$", u_str):
            return u_str

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


def _is_bot_detection_error(err_msg: str) -> bool:
    """Detects whether an error message indicates anti-bot throttling or block.

    Args:
        err_msg: Raw error message string.

    Returns:
        bool: True if bot block or rate-limiting is detected, False otherwise.
    """
    lower = err_msg.lower()
    return (
        "not a bot" in lower
        or "confirm you’re not a bot" in lower
        or "confirm you're not a bot" in lower
        or "confirm you are not a bot" in lower
        or "http error 429" in lower
        or "too many requests" in lower
        or "recaptcha" in lower
    )


def _check_permanent_video_error(err_msg: str) -> str | None:
    """Detects whether an error message indicates an unrecoverable video-level error.

    Args:
        err_msg: Raw error message string.

    Returns:
        str | None: Normalized error description or None.
    """
    lower = err_msg.lower()
    if "georestricted" in lower or "not available in your country" in lower:
        return "Region locked"
    if "members-only" in lower or "members only" in lower or "join this channel" in lower:
        return "Members only"
    if "private video" in lower or "this video is private" in lower or "granted access" in lower:
        return "Private video"
    if "removed by the uploader" in lower or "video has been removed" in lower or "terms of service" in lower:
        return "Deleted / Removed"
    if "sign in to confirm your age" in lower:
        return "Age restricted"
    return None


def _has_pot_provider_artifacts(server_dir: Path) -> bool:
    """Verifies that bgutil-ytdlp-pot-provider has compiled scripts and installed dependencies.

    Args:
        server_dir: Path to the provider's server directory.

    Returns:
        bool: True if required node_modules and compiled output scripts exist, False otherwise.
    """
    has_node_modules = (server_dir / "node_modules").is_dir()
    b_dir = server_dir / "build"
    d_dir = server_dir / "dist"
    has_script = (
        (b_dir / "generate_once.js").is_file()
        or (d_dir / "generate_once.js").is_file()
        or (b_dir / "main.js").is_file()
        or (d_dir / "main.js").is_file()
    )
    return has_node_modules and has_script


class YouTubeClient:
    """Encapsulates all YouTube operations: yt-dlp configuration, PO tokens, channel caching, and downloads."""

    config: YoutubeSourceConfig
    base_work_dir: Path
    mapping_file: Path
    _channel_map: ChannelMap
    _pot_server_dir: Path | None
    _base_ydl_opts: dict[str, Any]

    def __init__(self, config: YoutubeSourceConfig, base_work_dir: Path) -> None:
        """Initializes the YouTube client, compiling PO tokens and loading channel mappings.

        Args:
            config: YouTube source configuration settings.
            base_work_dir: Server root working directory.
        """
        self.config = config
        self.base_work_dir = base_work_dir

        channels_path = config.channels_file
        if not channels_path.is_absolute():
            channels_path = base_work_dir / channels_path
        self.mapping_file = channels_path
        self._channel_map = ChannelMap(root={})
        self._load_channel_map()

        pot_path = config.pot_dir
        if not pot_path.is_absolute():
            pot_path = base_work_dir / pot_path
        self._pot_server_dir = self._ensure_pot_provider(pot_path) if config.enable_pot_provider else None

        self._base_ydl_opts = self._build_base_ydl_opts()
        self._log_startup_diagnostics()

    def _is_node_runtime_available(self) -> bool:
        """Checks if Node.js, npm, and npx runtimes are available in PATH.

        Returns:
            bool: True if all required binaries are found, False otherwise.
        """
        return bool(shutil.which("node") and shutil.which("npm") and shutil.which("npx"))

    def _ensure_pot_provider(self, pot_dir: Path) -> Path | None:
        """Clones, syncs tags, and compiles bgutil-ytdlp-pot-provider once at startup.

        Args:
            pot_dir: Directory where the PO token provider repository is located.

        Returns:
            Path | None: Server home directory if built and valid, None otherwise.
        """
        npm_bin = shutil.which("npm")
        npx_bin = shutil.which("npx")
        if not (npm_bin and npx_bin):
            logfire.warning("Node.js, npm, or npx not found in PATH. Skipping PO token provider.")
            return None

        try:
            target_version = importlib.metadata.version("bgutil-ytdlp-pot-provider")
        except Exception:
            target_version = None

        repo_url = "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
        server_dir = pot_dir / "server"
        needs_build = False

        try:
            repo: Repo | None = None
            if pot_dir.exists():
                try:
                    repo = Repo(pot_dir)
                except (InvalidGitRepositoryError, NoSuchPathError):
                    logfire.warning(f"Existing directory at {pot_dir} is not a valid git repository. Re-cloning...")
                    shutil.rmtree(pot_dir, ignore_errors=True)
                    repo = None

            if repo is None:
                logfire.info(f"Cloning bgutil-ytdlp-pot-provider ({target_version or 'default'}) into {pot_dir}")
                pot_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_version:
                    repo = Repo.clone_from(repo_url, pot_dir, branch=target_version)
                else:
                    repo = Repo.clone_from(repo_url, pot_dir)
                needs_build = True
            else:
                try:
                    current_tag = repo.git.describe("--tags", "--exact-match").strip()
                except Exception:
                    current_tag = None

                if target_version and current_tag != target_version:
                    logfire.info(f"Syncing bgutil tag: {current_tag} -> {target_version}...")
                    repo.remotes.origin.fetch(tags=True)
                    repo.git.checkout(target_version, force=True)
                    needs_build = True

            build_dir = server_dir / "build"
            dist_dir = server_dir / "dist"
            if not _has_pot_provider_artifacts(server_dir):
                needs_build = True

            if needs_build:
                logfire.info(f"Building bgutil-ytdlp-pot-provider server ({target_version or 'latest'})...")
                shutil.rmtree(build_dir, ignore_errors=True)
                shutil.rmtree(dist_dir, ignore_errors=True)
                is_win = os.name == "nt"
                subprocess.run([npm_bin, "ci"], cwd=server_dir, check=True, capture_output=True, shell=is_win)
                subprocess.run([npx_bin, "tsc"], cwd=server_dir, check=True, capture_output=True, shell=is_win)
                logfire.info("bgutil-ytdlp-pot-provider built successfully.")

            if _has_pot_provider_artifacts(server_dir):
                return server_dir

            logfire.warning(f"bgutil-ytdlp-pot-provider build artifacts missing in {server_dir}.")
        except Exception as e:
            logfire.warning(f"Failed to setup bgutil-ytdlp-pot-provider: {e}")

        return None

    def _build_base_ydl_opts(self) -> dict[str, Any]:
        """Constructs base yt-dlp parameters including PO token server and FFmpeg path.

        Returns:
            dict[str, Any]: Base yt-dlp configuration options dictionary.
        """
        extractor_args: dict[str, Any] = {"youtubetab": {"approximate_date": [""]}}
        if self._pot_server_dir and self._pot_server_dir.exists():
            extractor_args["youtubepot-bgutilscript"] = {"server_home": [str(self._pot_server_dir.resolve())]}

        ffmpeg_bin, _ = ensure_ffmpeg()
        node_bin = shutil.which("node")
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "ignoreerrors": False,
            "windowsfilenames": True,
            "js_runtimes": {"node": {"path": node_bin}} if node_bin else {"node": {}},
            "ffmpeg_location": ffmpeg_bin,
            "extractor_args": extractor_args,
        }
        if self.config.proxy:
            opts["proxy"] = str(self.config.proxy).rstrip("/")
        if self.config.cookie_file and self.config.cookie_file.is_file():
            opts["cookiefile"] = str(self.config.cookie_file.resolve())

        return opts

    def _log_startup_diagnostics(self) -> None:
        """Logs binaries, cookies, proxy, and PO token setup once upon startup."""
        node_bin = shutil.which("node")
        npm_bin = shutil.which("npm")
        npx_bin = shutil.which("npx")
        ffmpeg_bin, _ = ensure_ffmpeg()

        cookie_p = self.config.cookie_file
        cookie_resolved = cookie_p.resolve() if cookie_p else None
        cookie_exists = bool(cookie_resolved and cookie_resolved.is_file())

        proxy_str = str(self.config.proxy) if self.config.proxy else "None"
        masked_proxy = re.sub(r"://([^:]+):([^@]+)@", "://***:***@", proxy_str)

        try:
            pot_plugin_ver = importlib.metadata.version("bgutil-ytdlp-pot-provider")
        except Exception:
            pot_plugin_ver = "NOT_INSTALLED"

        with logfire.span(
            "YouTube Ingestion Startup Diagnostics",
            node_path=node_bin or "NOT_FOUND",
            npm_path=npm_bin or "NOT_FOUND",
            npx_path=npx_bin or "NOT_FOUND",
            ffmpeg_path=ffmpeg_bin,
            cookie_file=str(cookie_p) if cookie_p else "None",
            cookie_exists=cookie_exists,
            proxy=masked_proxy,
            pot_enabled=self.config.enable_pot_provider,
            pot_plugin_version=pot_plugin_ver,
            pot_server_dir=str(self._pot_server_dir) if self._pot_server_dir else "None",
        ):
            if not cookie_exists and cookie_p:
                logfire.error(f"Cookie file '{cookie_p}' DOES NOT exist. yt-dlp will make requests unauthenticated!")
            elif cookie_exists and cookie_resolved:
                logfire.info(f"Loaded YouTube cookie file from {cookie_resolved}.")

    def _load_channel_map(self) -> None:
        """Loads cached channel mappings from disk if present."""
        if not self.mapping_file.is_file():
            return
        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._channel_map = ChannelMap.model_validate(data)
            logfire.debug(f"Loaded {len(self._channel_map.root)} channel mappings from {self.mapping_file.name}")
        except Exception as e:
            logfire.warning(f"Could not load channel mapping file {self.mapping_file.name}: {e}")

    def _save_channel_map(self) -> None:
        """Saves current channel mappings atomically to disk in UTF-8 JSON format."""
        temp_file = self.mapping_file.with_name(f"{self.mapping_file.name}.tmp.{os.getpid()}")
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._channel_map.model_dump(), f, indent=2, sort_keys=True, ensure_ascii=False)
            os.replace(temp_file, self.mapping_file)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logfire.warning(f"Failed to save channel mapping file {self.mapping_file.name}: {e}")

    def _record_channel_mapping(
        self,
        channel_id: str | None,
        uploader_id: str | None,
        channel_name: str | None,
        timestamp: int = 0,
    ) -> None:
        """Records a new or updated channel mapping and saves to disk if changed.

        Args:
            channel_id: Unique YouTube channel ID (e.g. 'UC...').
            uploader_id: Channel handle or user ID.
            channel_name: Human-readable channel display name.
            timestamp: Latest video timestamp in seconds.
        """
        if not channel_id or not uploader_id or str(uploader_id).lower() == "youtube":
            return

        c_str = str(channel_id)
        u_str = str(uploader_id).strip("@")[:20]
        existing = self._channel_map.root.get(c_str)

        if not existing or timestamp > existing.timestamp or u_str != existing.uploader_id:
            self._channel_map.root[c_str] = ChannelInfo(
                uploader_id=u_str,
                channel_name=str(channel_name) if channel_name else (existing.channel_name if existing else None),
                timestamp=timestamp,
            )
            self._save_channel_map()

    def _lookup_youtube_handle(self, video_id: str | None, channel_id: str | None = None) -> str | None:
        """Performs targeted yt-dlp metadata extraction to resolve an unknown channel handle.

        Args:
            video_id: Specific video ID to query.
            channel_id: Optional channel ID if known.

        Returns:
            str | None: Sanitized uploader handle if resolved, None otherwise.

        Raises:
            YouTubeBotDetectionError: If bot detection is triggered during lookup.
        """
        if not video_id:
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with logfire.span("Looking up YouTube channel handle: {video_id}", video_id=video_id, _level="debug"):
                with YoutubeDL(cast(Any, self._base_ydl_opts)) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        return None
                    handle = _extract_handle(info)
                    name = info.get("channel") or info.get("uploader")
                    c_id = info.get("channel_id") or channel_id
                    if handle and str(handle).lower() != "youtube":
                        clean_handle = str(handle).strip("@")[:20]
                        self._record_channel_mapping(
                            channel_id=str(c_id) if c_id else None,
                            uploader_id=clean_handle,
                            channel_name=str(name) if name else None,
                            timestamp=_parse_entry_timestamp(info),
                        )
                        return clean_handle
        except Exception as e:
            err_msg = str(e)
            logfire.debug(f"Targeted YouTube lookup failed for video {video_id}: {err_msg}")
            if _is_bot_detection_error(err_msg):
                raise YouTubeBotDetectionError(f"Bot detection during channel lookup for video {video_id}") from e

        return None

    def get_uploader_id(self, entry: Mapping[str, Any], url_context: str | None = None) -> str:
        """Resolves the sanitized uploader ID (handle) from a yt-dlp metadata entry.

        Follows a strict hierarchy:
        1. Explicit handle on the entry (e.g. '@handle' in URLs or uploader_id).
        2. Cached handle in channels.json mapped to the entry's channel_id.
        3. Source handle from url_context when channel_id is present (recorded to channels.json).
        4. Targeted YouTube lookup for the video when url_context is unavailable.
        5. Context handle fallback without channel_id.
        6. Generic fallback string 'youtube'.

        Args:
            entry: Video or channel metadata dictionary from yt-dlp.
            url_context: Optional channel handle extracted from the configured source URL.

        Returns:
            str: Resolved and sanitized uploader ID (max 20 characters).
        """
        uploader_id = _extract_handle(entry)
        channel_id = entry.get("channel_id")
        channel_name = entry.get("channel") or entry.get("uploader")
        ts_int = _parse_entry_timestamp(entry)

        # 1. Direct explicit handle on entry
        if uploader_id and str(uploader_id).lower() != "youtube":
            clean_handle = str(uploader_id).strip("@")[:20]
            if channel_id:
                self._record_channel_mapping(
                    channel_id=str(channel_id),
                    uploader_id=clean_handle,
                    channel_name=str(channel_name) if channel_name else None,
                    timestamp=ts_int,
                )
            return clean_handle

        # 2. Check channels.json cache by channel_id
        if channel_id and isinstance(channel_id, str):
            c_str = str(channel_id)
            info = self._channel_map.root.get(c_str)
            if info and info.uploader_id and info.uploader_id.lower() != "youtube":
                return info.uploader_id

            # 3. Associate channel_id with source URL handle when available
            if url_context and str(url_context).lower() != "youtube":
                clean_handle = str(url_context).strip("@")[:20]
                self._record_channel_mapping(
                    channel_id=c_str,
                    uploader_id=clean_handle,
                    channel_name=str(channel_name) if channel_name else None,
                    timestamp=ts_int,
                )
                return clean_handle

            # 4. Targeted YouTube lookup only when source handle is unavailable
            video_id = str(entry.get("id")) if entry.get("id") else None
            looked_up = self._lookup_youtube_handle(video_id=video_id, channel_id=c_str)
            if looked_up:
                return looked_up

        # 5. Fallback to URL context (for channel tabs without channel_id)
        if url_context:
            return str(url_context).strip("@")[:20]

        # 6. Generic fallback
        return "youtube"

    def scan_sources(self, deep_scan: bool = False) -> list[DiscoveredVideo]:
        """Fetches metadata for configured YouTube sources using flat extraction with early date cutoff.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.

        Returns:
            list[DiscoveredVideo]: List of discovered videos across all feeds.

        Raises:
            YouTubeBotDetectionError: If a bot check or HTTP 429 block is triggered.
        """
        if not self.config.urls:
            return []

        ydl_opts = dict(self._base_ydl_opts)
        ydl_opts["extract_flat"] = True

        lookback_seconds = int((2 * self.config.rescan_interval_minutes + self.config.rescan_jitter_minutes) * 60)
        cutoff_ts = int(datetime.now(timezone.utc).timestamp()) - lookback_seconds if not deep_scan else 0

        target_items: list[tuple[str, str | None]] = []
        for raw_url in self.config.urls:
            clean_url = raw_url.rstrip("/")
            if clean_url.endswith("/shorts"):
                continue

            url_handle: str | None = None
            if match := re.search(r"/(@[\w.-]+)", clean_url):
                url_handle = match.group(1)
            elif match := re.search(r"/(channel/UC[\w-]+)", clean_url):
                url_handle = match.group(1).split("/")[-1]

            if re.match(r"^https?://(www\.)?youtube\.com/(@[\w.-]+|channel/UC[\w-]+)$", clean_url):
                target_items.extend([(f"{clean_url}/videos", url_handle), (f"{clean_url}/streams", url_handle)])
            else:
                target_items.append((clean_url, url_handle))

        discovered: list[DiscoveredVideo] = []

        with logfire.span("Scanning YouTube Sources", deep_scan=deep_scan, total_sources=len(target_items)):
            with YoutubeDL(cast(Any, ydl_opts)) as ydl:
                for idx, (url, url_context_handle) in enumerate(target_items):
                    if idx > 0 and self.config.channel_delay_seconds > 0:
                        delay = random.uniform(
                            0.5 * self.config.channel_delay_seconds,
                            1.5 * self.config.channel_delay_seconds,
                        )
                        time_sleep(delay)

                    with logfire.span("Scanning {url}", url=url, _level="debug") as url_span:
                        try:
                            result = ydl.extract_info(url, download=False)
                            if not result:
                                url_span.set_attribute("result", "empty")
                                continue

                            entries_raw = result.get("entries")
                            entries_iterable = [result] if entries_raw is None else cast(Any, entries_raw)
                            url_discovered_count = 0

                            for entry in entries_iterable:
                                if not entry:
                                    continue
                                vid = entry.get("id")
                                if not vid or len(vid) != 11:
                                    continue
                                title = str(entry.get("title") or "Unknown")
                                if title in ["[Deleted video]", "[Private video]", "[Unavailable video]"]:
                                    continue
                                if entry.get("live_status") in ("is_upcoming", "is_live", "post_live"):
                                    continue
                                if "/shorts/" in (entry.get("url") or entry.get("webpage_url") or ""):
                                    continue

                                raw_ts = entry.get("timestamp") or entry.get("release_timestamp") or 0
                                try:
                                    ts = int(raw_ts)
                                except (ValueError, TypeError):
                                    ts = 0

                                if not ts and (ud := entry.get("upload_date")):
                                    try:
                                        dt = datetime.strptime(str(ud), "%Y%m%d").replace(tzinfo=timezone.utc)
                                        ts = int(dt.timestamp())
                                    except ValueError:
                                        ts = 0

                                if not deep_scan and ts > 0 and ts < cutoff_ts:
                                    break

                                uploader = self.get_uploader_id(entry=entry, url_context=url_context_handle)
                                ch_name = str(entry.get("channel") or entry.get("uploader") or uploader)

                                discovered.append(
                                    DiscoveredVideo(
                                        video_id=vid,
                                        uploader=uploader,
                                        channel_name=ch_name,
                                        title=title,
                                        source_ref=f"https://www.youtube.com/watch?v={vid}",
                                        timestamp=ts,
                                    )
                                )
                                url_discovered_count += 1

                            url_span.set_attribute("discovered_count", url_discovered_count)
                        except YouTubeBotDetectionError:
                            url_span.set_attribute("error", "bot_detection")
                            raise
                        except Exception as e:
                            err_msg = str(e)
                            logfire.warning(f"Error scanning YouTube source {url}: {err_msg}")
                            url_span.set_attribute("error", err_msg)
                            if _is_bot_detection_error(err_msg):
                                raise YouTubeBotDetectionError(f"Bot detection scanning source {url}") from e

        return discovered

    def download_video(self, video_id: str, source_ref: str, destination_dir: Path) -> DownloadedVideo:
        """Downloads a video stream and extracts authoritative metadata.

        Args:
            video_id: Unique 11-character video ID.
            source_ref: Video URL.
            destination_dir: Directory where media should be downloaded.

        Returns:
            DownloadedVideo: Verified downloaded media file and ground-truth channel metadata.

        Raises:
            YouTubeBotDetectionError: If a bot check is detected during download.
            YouTubeVideoUnavailableError: If the video is deleted, private, or age/geo-restricted.
            RuntimeError: If download fails or output file cannot be located.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts = dict(self._base_ydl_opts)
        ydl_opts["format"] = "bestvideo[height<=360]+bestaudio/best[height<=360]"
        ydl_opts["outtmpl"] = str(destination_dir / f"%(title).{self.config.max_title_bytes}B [%(id)s].%(ext)s")

        try:
            with YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(source_ref, download=True)
                if not info:
                    raise RuntimeError(f"Could not extract download info for {source_ref}")

                uploader_handle = self.get_uploader_id(info)
                channel_name = str(info.get("channel") or info.get("uploader") or uploader_handle)

                downloaded_path = Path(ydl.prepare_filename(info))
                if not downloaded_path.is_file():
                    matches = [
                        p
                        for p in destination_dir.glob(f"*{video_id}*.*")
                        if p.is_file()
                        and not p.name.startswith("part_")
                        and p.suffix.lower() in _VALID_VIDEO_EXTENSIONS
                    ]
                    if matches:
                        # Prioritize the largest file (e.g. merged stream over unmerged track)
                        matches.sort(key=lambda p: p.stat().st_size, reverse=True)
                        downloaded_path = matches[0]
                    else:
                        raise RuntimeError(f"Downloaded file not found for {video_id}")

                return DownloadedVideo(
                    video_id=video_id,
                    uploader=uploader_handle,
                    channel_name=channel_name,
                    file_path=downloaded_path,
                )
        except Exception as e:
            err_msg = str(e)
            perm_reason = _check_permanent_video_error(err_msg)
            if perm_reason:
                raise YouTubeVideoUnavailableError(err_msg, reason=perm_reason) from e
            if _is_bot_detection_error(err_msg):
                raise YouTubeBotDetectionError(f"Bot detection downloading video {video_id}") from e
            raise
