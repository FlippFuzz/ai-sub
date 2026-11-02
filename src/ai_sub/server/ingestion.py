"""Source ingestion, PO token setup, ChannelMapper, and direct-to-combo video feeding."""

from __future__ import annotations

import asyncio
import importlib.metadata
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import sleep as time_sleep
from time import time
from typing import Any, Literal, cast

import logfire
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL

from ai_sub.server.channel_mapping import ChannelMapper
from ai_sub.server.config import SourcesConfig
from ai_sub.server.coordinator import PipelineCoordinator
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.pipeline import ComboPipeline
from ai_sub.server.pot_provider import ensure_pot_provider
from ai_sub.video import ensure_ffmpeg


def _is_bot_detection_error(err_msg: str) -> bool:
    """Detects whether an error message from yt-dlp indicates anti-bot throttling or block.

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


def _is_permanent_video_error(err_msg: str) -> str | None:
    """Detects whether an error message from yt-dlp indicates an unrecoverable video-level error.

    Args:
        err_msg: Raw error message string.

    Returns:
        str | None: Clean error description (e.g. 'Region locked', 'Members only') or None.
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


class QueuedVideo(BaseModel):
    """Metadata and operational queue state for a video discovered from local files or YouTube feeds."""

    video_id: str = Field(description="Unique video identifier.")
    uploader: str = Field(description="Channel or uploader handle identifier.")
    channel_name: str | None = Field(default=None, description="Human-readable channel display name.")
    title: str = Field(description="Video title.")
    source_type: Literal["local", "youtube"] = Field(description="Ingestion source type.")
    source_ref: str = Field(description="Filesystem path or YouTube URL.")
    timestamp: int = Field(default=0, description="Upload or file modification Unix timestamp.")
    local_path: Path | None = Field(default=None, description="Path to local file if available.")

    status: Literal["ready", "failed", "unplayable"] = Field(
        default="ready",
        description="Current operational queue status.",
    )
    error_reason: str | None = Field(
        default=None,
        description="Reason for failure or unplayable status (e.g. 'Region locked', 'Download timeout').",
    )
    retry_after: float | None = Field(
        default=None,
        description="Unix timestamp when this video becomes eligible to be retried.",
    )
    failure_count: int = Field(
        default=0,
        description="Total failed download attempts for this video.",
    )


class IngestionManager:
    """Orchestrates video intake from watched directories and YouTube feeds."""

    config: SourcesConfig
    coordinator: PipelineCoordinator
    git_sync: GitSyncManager
    channel_mapper: ChannelMapper
    pot_dir: Path
    pot_server_dir: Path | None
    base_work_dir: Path
    local_videos: dict[str, QueuedVideo]
    youtube_videos: dict[str, QueuedVideo]
    _youtube_cooldown_until: float
    _youtube_cooldown_reason: str | None
    _loop: asyncio.AbstractEventLoop | None
    _cooldown_wake_task: asyncio.Task[None] | None
    _files_rescan_task: asyncio.Task[None] | None
    _youtube_rescan_task: asyncio.Task[None] | None
    _scan_lock: asyncio.Lock
    _feed_lock: asyncio.Lock

    def __init__(
        self,
        config: SourcesConfig,
        coordinator: PipelineCoordinator,
        git_sync: GitSyncManager,
        base_work_dir: Path,
    ) -> None:
        """Initializes the ingestion manager and performs one-time environment diagnostics.

        Args:
            config: Intake sources configuration containing files and youtube options.
            coordinator: Pipeline coordinator managing active combinations.
            git_sync: Git synchronization manager.
            base_work_dir: Root working directory for downloads and pipelines.
        """
        self.config = config
        self.coordinator = coordinator
        self.git_sync = git_sync
        self.base_work_dir = base_work_dir

        channels_path = config.youtube.channels_file
        if not channels_path.is_absolute():
            channels_path = base_work_dir / channels_path
        self.channel_mapper = ChannelMapper(channels_path)

        pot_dir = config.youtube.pot_dir
        if not pot_dir.is_absolute():
            pot_dir = base_work_dir / pot_dir
        self.pot_dir = pot_dir

        self.pot_server_dir = ensure_pot_provider(pot_dir) if config.youtube.enable_pot_provider else None

        self.local_videos = {}
        self.youtube_videos = {}
        self._youtube_cooldown_until = 0.0
        self._youtube_cooldown_reason = None
        self._loop = None
        self._cooldown_wake_task = None
        self._files_rescan_task = None
        self._youtube_rescan_task = None
        self._scan_lock = asyncio.Lock()
        self._feed_lock = asyncio.Lock()

        self._log_startup_diagnostics()

    def _log_startup_diagnostics(self) -> None:
        """Logs binaries, cookies, proxy, and PO token setup once upon startup."""
        node_bin = shutil.which("node")
        npm_bin = shutil.which("npm")
        npx_bin = shutil.which("npx")
        ffmpeg_bin, _ = ensure_ffmpeg()

        cookie_p = self.config.youtube.cookie_file
        cookie_resolved = cookie_p.resolve() if cookie_p else None
        cookie_exists = bool(cookie_resolved and cookie_resolved.is_file())

        proxy_str = str(self.config.youtube.proxy) if self.config.youtube.proxy else "None"
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
            pot_enabled=self.config.youtube.enable_pot_provider,
            pot_plugin_version=pot_plugin_ver,
            pot_server_dir=str(self.pot_server_dir) if self.pot_server_dir else "None",
        ):
            if not cookie_exists and cookie_p:
                logfire.error(f"Cookie file '{cookie_p}' DOES NOT exist. yt-dlp will make requests unauthenticated!")
            elif cookie_exists and cookie_resolved:
                logfire.info(f"Loaded YouTube cookie file from {cookie_resolved}.")

    def _get_base_ytdlp_opts(self) -> dict[str, Any]:
        """Constructs base yt-dlp parameters including PO token server and FFmpeg path.

        Returns:
            dict[str, Any]: Base yt-dlp configuration options dictionary.
        """
        extractor_args: dict[str, Any] = {"youtubetab": {"approximate_date": [""]}}
        if self.pot_server_dir and self.pot_server_dir.exists():
            extractor_args["youtubepot-bgutilscript"] = {"server_home": str(self.pot_server_dir)}

        ffmpeg_bin, _ = ensure_ffmpeg()
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "ignoreerrors": False,
            "windowsfilenames": True,
            "js_runtimes": {"node": {}},
            "ffmpeg_location": ffmpeg_bin,
            "extractor_args": extractor_args,
        }
        if self.config.youtube.proxy:
            opts["proxy"] = str(self.config.youtube.proxy)
        if self.config.youtube.cookie_file and self.config.youtube.cookie_file.is_file():
            opts["cookiefile"] = str(self.config.youtube.cookie_file.resolve())

        return opts

    def is_youtube_in_cooldown(self) -> bool:
        """Checks if YouTube operations are temporarily suspended due to bot detection or failure limit.

        Returns:
            bool: True if YouTube operations are within a cooldown window, False otherwise.
        """
        return time() < self._youtube_cooldown_until

    def get_youtube_cooldown_info(self) -> dict[str, Any]:
        """Returns details on the current YouTube intake cooldown status.

        Returns:
            dict[str, Any]: Cooldown state dictionary including remaining seconds and reason.
        """
        in_cooldown = self.is_youtube_in_cooldown()
        remaining = max(0.0, self._youtube_cooldown_until - time()) if in_cooldown else 0.0
        return {
            "in_cooldown": in_cooldown,
            "cooldown_until": self._youtube_cooldown_until if in_cooldown else 0.0,
            "reason": self._youtube_cooldown_reason if in_cooldown else None,
            "remaining_seconds": round(remaining, 1),
        }

    def _trigger_youtube_cooldown(self, reason: str = "") -> None:
        """Sets a cooldown timer on all YouTube operations and schedules an automatic wake-up.

        Args:
            reason: Context description for the log warning. Defaults to "".
        """
        cooldown_sec = self.config.youtube.cooldown_minutes * 60.0
        self._youtube_cooldown_until = time() + cooldown_sec
        self._youtube_cooldown_reason = reason or "Download failure threshold reached"
        logfire.warning(
            f"YouTube intake cooldown triggered ({self._youtube_cooldown_reason}). "
            f"Pausing YouTube downloads for {self.config.youtube.cooldown_minutes} minutes."
        )

        async def _wake_on_cooldown_expiry() -> None:
            try:
                await asyncio.sleep(cooldown_sec)
                logfire.info("YouTube intake cooldown expired. Resuming YouTube job feeding...")
                await self.feed_idle_pipelines()
            except asyncio.CancelledError:
                pass

        def _schedule() -> None:
            if self._cooldown_wake_task and not self._cooldown_wake_task.done():
                self._cooldown_wake_task.cancel()
            self._cooldown_wake_task = asyncio.create_task(_wake_on_cooldown_expiry())

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(_schedule)

    def scan_watch_folders(self) -> list[QueuedVideo]:
        """Scans watched directories recursively for video media files.

        Returns:
            list[QueuedVideo]: Newly discovered or updated local video items.
        """
        valid_extensions = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts"}
        discovered_ids: set[str] = set()

        for folder in self.config.files.folders:
            if not folder.is_dir():
                continue
            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    vid = file_path.stem
                    discovered_ids.add(vid)
                    mtime = int(file_path.stat().st_mtime)
                    uploader = file_path.parent.name or "local"
                    uploader = uploader.strip("@")[:20]

                    if vid not in self.local_videos:
                        self.local_videos[vid] = QueuedVideo(
                            video_id=vid,
                            uploader=uploader,
                            channel_name=uploader,
                            title=vid,
                            source_type="local",
                            source_ref=str(file_path.resolve()),
                            timestamp=mtime,
                            local_path=file_path.resolve(),
                        )
                    else:
                        existing = self.local_videos[vid]
                        existing.local_path = file_path.resolve()
                        existing.timestamp = mtime

        # Prune local files that were removed from the watch directory
        for vid in set(self.local_videos.keys()) - discovered_ids:
            self.local_videos.pop(vid, None)

        return list(self.local_videos.values())

    def scan_youtube_sources(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Fetches metadata for YouTube sources using flat single-request extraction.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.
            Defaults to False.

        Returns:
            list[QueuedVideo]: Discovered YouTube video items.
        """
        if not self.config.youtube.urls or self.is_youtube_in_cooldown():
            return list(self.youtube_videos.values())

        ydl_opts = self._get_base_ytdlp_opts()
        ydl_opts["extract_flat"] = True

        lookback_seconds = int(
            (2 * self.config.youtube.rescan_interval_minutes + self.config.youtube.rescan_jitter_minutes) * 60
        )
        cutoff_ts = int(time()) - lookback_seconds if not deep_scan else 0

        target_items: list[tuple[str, str | None]] = []
        for raw_url in self.config.youtube.urls:
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

        discovered_new = 0

        with logfire.span(
            "Scanning YouTube Sources",
            deep_scan=deep_scan,
            total_sources=len(target_items),
        ) as scan_span:
            with YoutubeDL(cast(Any, ydl_opts)) as ydl:
                for idx, (url, url_context_handle) in enumerate(target_items):
                    if idx > 0 and self.config.youtube.channel_delay_seconds > 0:
                        delay = random.uniform(
                            0.5 * self.config.youtube.channel_delay_seconds,
                            1.5 * self.config.youtube.channel_delay_seconds,
                        )
                        time_sleep(delay)

                    with logfire.span("Scanning {url}", url=url, _level="debug") as url_span:
                        try:
                            result = ydl.extract_info(url, download=False)
                            if not result:
                                url_span.set_attribute("result", "empty")
                                continue

                            self.channel_mapper.update_from_entries([result])
                            entries_raw = result.get("entries")
                            entries_iterator = [result] if entries_raw is None else entries_raw
                            entries_processed: list[Any] = []

                            for entry in entries_iterator:
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
                                    continue

                                entries_processed.append(entry)
                                uploader, ch_name = self.channel_mapper.resolve(
                                    entry=entry,
                                    parent=result,
                                    url_context=url_context_handle,
                                )

                                if vid not in self.youtube_videos:
                                    self.youtube_videos[vid] = QueuedVideo(
                                        video_id=vid,
                                        uploader=uploader,
                                        channel_name=ch_name,
                                        title=title,
                                        source_type="youtube",
                                        source_ref=f"https://www.youtube.com/watch?v={vid}",
                                        timestamp=ts,
                                    )
                                    discovered_new += 1
                                else:
                                    existing = self.youtube_videos[vid]
                                    if title and existing.title != title:
                                        existing.title = title
                                    if ts > 0:
                                        existing.timestamp = ts

                            self.channel_mapper.update_from_entries(entries_processed)
                            url_span.set_attribute("discovered_count", len(entries_processed))
                        except Exception as e:
                            err_msg = str(e)
                            logfire.warning(f"Error scanning YouTube source {url}: {err_msg}")
                            url_span.set_attribute("error", err_msg)
                            if _is_bot_detection_error(err_msg):
                                self._trigger_youtube_cooldown(reason=f"scanning source {url}")
                                break

            scan_span.set_attribute("total_youtube_videos", len(self.youtube_videos))
            logfire.info(
                f"YouTube scan complete: {discovered_new} new video(s) added "
                f"({len(self.youtube_videos)} total tracked in memory)."
            )
            return list(self.youtube_videos.values())

    async def scan_watch_folders_async(self) -> list[QueuedVideo]:
        """Scans local watch directories off-thread and updates in-memory cache.

        Returns:
            list[QueuedVideo]: Discovered local video items.
        """
        async with self._scan_lock:
            return await asyncio.to_thread(self.scan_watch_folders)

    async def scan_youtube_sources_async(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Scans YouTube sources off-thread and updates in-memory cache.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.
            Defaults to False.

        Returns:
            list[QueuedVideo]: Discovered YouTube video items.
        """
        if self.is_youtube_in_cooldown():
            remaining_m = (self._youtube_cooldown_until - time()) / 60.0
            logfire.info(f"Skipping YouTube scan: bot cooldown active ({remaining_m:.1f}m remaining).")
            return list(self.youtube_videos.values())

        async with self._scan_lock:
            return await asyncio.to_thread(self.scan_youtube_sources, deep_scan)

    async def scan_all_sources(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Executes rescan of both local watch folders and YouTube sources off-thread.

        Args:
            deep_scan: If True, crawls full channel history on YouTube. Defaults to False.

        Returns:
            list[QueuedVideo]: List of all discovered videos sorted newest-first.
        """
        await self.scan_watch_folders_async()
        await self.scan_youtube_sources_async(deep_scan=deep_scan)
        all_videos = list(self.local_videos.values()) + list(self.youtube_videos.values())
        all_videos.sort(key=lambda v: (-v.timestamp, v.title.lower()))
        return all_videos

    async def download_youtube_to_combo_dir(self, video: QueuedVideo, combo_work_dir: Path) -> Path:
        """Downloads a video stream into the combination directory.

        Args:
            video: Discovered video metadata.
            combo_work_dir: Working directory of the target pipeline.

        Returns:
            Path: Filesystem path to the downloaded media file.

        Raises:
            RuntimeError: If download fails or output file cannot be located.
        """
        job_dir = combo_work_dir / video.video_id
        job_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = self._get_base_ytdlp_opts()
        ydl_opts["format"] = "bestvideo[height<=360]+bestaudio/best[height<=360]/best"
        ydl_opts["outtmpl"] = str(job_dir / f"%(title).{self.config.youtube.max_title_bytes}B [%(id)s].%(ext)s")

        def _do_download() -> Path:
            with YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(video.source_ref, download=True)
                if not info:
                    raise RuntimeError(f"Could not extract download info for {video.source_ref}")

                self.channel_mapper.update_from_entries([info])
                if video.uploader == "youtube":
                    video.uploader, video.channel_name = self.channel_mapper.resolve(info)

                return Path(ydl.prepare_filename(info))

        with logfire.span("Downloading YouTube Video to {job_dir}: {title}", job_dir=job_dir.name, title=video.title):
            downloaded = await asyncio.to_thread(_do_download)
            if not downloaded.is_file():
                matches = [
                    p
                    for p in job_dir.glob(f"*{video.video_id}*.*")
                    if p.is_file() and not p.name.startswith("part_") and not p.name.endswith(".json")
                ]
                if matches:
                    return matches[0]
                raise RuntimeError(f"Downloaded file not found for {video.title} (expected: {downloaded.name})")
            return downloaded

    def get_pipeline_queue(self, pipeline: ComboPipeline) -> list[dict[str, Any]]:
        """Gathers outstanding queued videos for a given pipeline from memory cache with operational details.

        Args:
            pipeline: Combination pipeline instance.

        Returns:
            list[dict[str, Any]]: List of queued video items pending for this pipeline.
        """
        candidates = list(self.local_videos.values()) + list(self.youtube_videos.values())
        outstanding = []
        now = time()

        for v in candidates:
            if pipeline.active_video and pipeline.active_video.video_id == v.video_id:
                continue

            if not self.git_sync.is_video_completed(v.uploader, v.video_id, pipeline.full_shortcode):
                retry_in_s = max(0, int(v.retry_after - now)) if v.retry_after and v.retry_after > now else None
                outstanding.append(
                    {
                        "video_id": v.video_id,
                        "title": v.title,
                        "uploader": v.uploader,
                        "channel_name": v.channel_name,
                        "source_type": v.source_type,
                        "timestamp": v.timestamp,
                        "status": v.status,
                        "error_reason": v.error_reason,
                        "retry_after": v.retry_after,
                        "retry_in_s": retry_in_s,
                        "failure_count": v.failure_count,
                    }
                )

        outstanding.sort(key=lambda x: (-x["timestamp"], x["title"].lower()))
        return outstanding[:50]

    def get_all_pipeline_queues(self) -> dict[str, list[dict[str, Any]]]:
        """Computes outstanding queues for all active pipelines without network calls.

        Returns:
            dict[str, list[dict[str, Any]]]: Mapping of pipeline names to their outstanding videos.
        """
        return {name: self.get_pipeline_queue(pipe) for name, pipe in self.coordinator.pipelines.items()}

    async def feed_idle_pipelines(self) -> None:
        """Assigns the newest available video to each idle pipeline."""
        async with self._feed_lock:
            idle_pipelines = [p for p in self.coordinator.pipelines.values() if p.is_idle()]
            if not idle_pipelines:
                return

            if not self.local_videos and not self.youtube_videos:
                await self.scan_all_sources(deep_scan=False)

            now = time()

            for pipe in idle_pipelines:
                # 1. Gather local candidates
                candidates: list[QueuedVideo] = list(self.local_videos.values())

                # 2. Include YouTube candidates only if not in cooldown
                if not self.is_youtube_in_cooldown():
                    candidates.extend(self.youtube_videos.values())

                # 3. Filter candidates: not completed in Git, and not in active retry cooldown
                eligible = [
                    v
                    for v in candidates
                    if not self.git_sync.is_video_completed(v.uploader, v.video_id, pipe.full_shortcode)
                    and (v.retry_after is None or now >= v.retry_after)
                ]

                # 4. Sort newest-first with alphabetical title tiebreaker
                eligible.sort(key=lambda v: (-v.timestamp, v.title.lower()))

                if not eligible:
                    continue

                consecutive_download_failures = 0
                max_failures = self.config.youtube.max_download_failures

                for video in eligible:
                    if video.source_type == "local":
                        if not video.local_path or not video.local_path.is_file():
                            video.status = "failed"
                            video.error_reason = "File missing from disk"
                            video.retry_after = now + 120  # Retry in 2 min
                            continue

                        logfire.info(
                            f"Feeding '{pipe.config.name}' with local file '{video.title}' ({video.video_id})."
                        )
                        await pipe.prepare_and_set_video(
                            video_id=video.video_id,
                            uploader=video.uploader,
                            title=video.title,
                            raw_video_path=video.local_path,
                            channel_name=video.channel_name,
                            source_type="local",
                            timestamp=video.timestamp,
                        )
                        break

                    # YouTube candidate - attempt direct download
                    try:
                        media_path = await self.download_youtube_to_combo_dir(video, pipe.work_dir)
                        video.status = "ready"
                        video.error_reason = None
                        video.retry_after = None

                        logfire.info(f"Feeding '{pipe.config.name}' with '{video.title}' ({video.video_id}).")
                        await pipe.prepare_and_set_video(
                            video_id=video.video_id,
                            uploader=video.uploader,
                            title=video.title,
                            raw_video_path=media_path,
                            channel_name=video.channel_name,
                            source_type="youtube",
                            timestamp=video.timestamp,
                        )
                        break

                    except Exception as e:
                        err_msg = str(e)
                        shutil.rmtree(pipe.work_dir / video.video_id, ignore_errors=True)

                        # Check permanent video errors first so age gates/geo-blocks do not trigger bot cooldown
                        perm_reason = _is_permanent_video_error(err_msg)
                        if perm_reason:
                            video.status = "unplayable"
                            video.error_reason = perm_reason
                            video.retry_after = now + (24 * 3600)  # Retry in 24 hours
                            video.failure_count += 1
                            logfire.warning(f"Video '{video.title}' is unplayable ({perm_reason}). Retrying in 24h.")
                            continue

                        if _is_bot_detection_error(err_msg):
                            self._trigger_youtube_cooldown(reason="Bot check detected")
                            break

                        # Transient download failure
                        video.status = "failed"
                        video.error_reason = "Download failed"
                        video.retry_after = now + (30 * 60)  # Retry in 30 minutes
                        video.failure_count += 1
                        consecutive_download_failures += 1
                        logfire.warning(
                            f"Download failed for '{video.title}' "
                            f"({consecutive_download_failures}/{max_failures}): {err_msg}"
                        )

                        if consecutive_download_failures >= max_failures:
                            self._trigger_youtube_cooldown(
                                reason=f"{consecutive_download_failures} consecutive download failures"
                            )
                            break

    async def start(self) -> None:
        """Starts the ingestion manager with an initial scan, followed by periodic delta intake loops."""
        self._loop = asyncio.get_running_loop()
        await self.scan_watch_folders_async()
        await self.scan_youtube_sources_async(deep_scan=True)
        await self.feed_idle_pipelines()

        self._files_rescan_task = asyncio.create_task(self._periodic_files_loop())
        self._youtube_rescan_task = asyncio.create_task(self._periodic_youtube_loop())

    async def stop(self) -> None:
        """Stops the periodic intake loops gracefully."""
        if self._cooldown_wake_task:
            self._cooldown_wake_task.cancel()
            try:
                await self._cooldown_wake_task
            except asyncio.CancelledError:
                pass
            self._cooldown_wake_task = None

        for task in (self._files_rescan_task, self._youtube_rescan_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._files_rescan_task = None
        self._youtube_rescan_task = None

    async def _periodic_files_loop(self) -> None:
        """Periodically scans local watch folders."""
        interval_sec = max(5, self.config.files.rescan_interval_minutes * 60)
        while True:
            await asyncio.sleep(interval_sec)
            try:
                logfire.debug("Running periodic watch folder scan...")
                await self.scan_watch_folders_async()
                await self.coordinator.check_and_finalize_completed_pipelines()
                await self.feed_idle_pipelines()
            except Exception as e:
                logfire.warning(f"Error during periodic watch folder scan: {e}")

    async def _periodic_youtube_loop(self) -> None:
        """Periodically scans YouTube sources using symmetric jittered intervals."""
        while True:
            base_sec = self.config.youtube.rescan_interval_minutes * 60.0
            jitter_sec = max(0.0, self.config.youtube.rescan_jitter_minutes * 60.0)
            delay_sec = max(5.0, base_sec + random.uniform(-jitter_sec, jitter_sec))

            logfire.debug(
                f"Next YouTube delta rescan scheduled in {delay_sec / 60:.1f} minutes "
                f"(base: {self.config.youtube.rescan_interval_minutes}m, "
                f"jitter: ±{self.config.youtube.rescan_jitter_minutes}m)."
            )
            await asyncio.sleep(delay_sec)

            try:
                logfire.debug("Running periodic YouTube delta scan...")
                await self.scan_youtube_sources_async(deep_scan=False)
                await self.coordinator.check_and_finalize_completed_pipelines()
                await self.feed_idle_pipelines()
            except Exception as e:
                logfire.warning(f"Error during periodic YouTube rescan: {e}")
