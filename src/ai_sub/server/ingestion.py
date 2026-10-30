"""Source ingestion, PO token provider setup, ChannelMapper, and direct-to-combo video feeding."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import random
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import sleep as time_sleep
from time import time
from typing import Any, Literal, cast

import logfire
from git import Repo
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL

from ai_sub.server.channel_mapping import ChannelMapper
from ai_sub.server.config import SourcesConfig
from ai_sub.server.coordinator import PipelineCoordinator
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.pipeline import ComboPipeline, PipelineStatus
from ai_sub.video import ensure_ffmpeg


def ensure_pot_provider(pot_dir: Path, check_updates: bool = False) -> Path | None:
    """Clones or updates bgutil-ytdlp-pot-provider to match the installed plugin version.

    Args:
        pot_dir: Directory where the PO token provider repository is located.
        check_updates: Whether to check GitHub for updates if no plugin version is pinned.

    Returns:
        Path | None: Server home directory if built and valid; None otherwise.
    """
    node_bin = shutil.which("node")
    npm_bin = shutil.which("npm")
    npx_bin = shutil.which("npx")

    if not (node_bin and npm_bin and npx_bin):
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
        if not pot_dir.exists():
            logfire.info(f"Cloning bgutil-ytdlp-pot-provider ({target_version or 'default'}) into {pot_dir}")
            pot_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_version:
                Repo.clone_from(repo_url, pot_dir, branch=target_version)
            else:
                Repo.clone_from(repo_url, pot_dir)
            needs_build = True
        else:
            repo = Repo(pot_dir)
            try:
                current_tag = repo.git.describe("--tags", "--exact-match").strip()
            except Exception:
                current_tag = None

            if target_version and current_tag != target_version:
                logfire.info(f"Syncing bgutil tag: {current_tag} -> {target_version}...")
                repo.remotes.origin.fetch(tags=True)
                repo.git.checkout(target_version, force=True)
                needs_build = True
            elif check_updates and not target_version:
                old_hash = repo.head.commit.hexsha
                repo.remotes.origin.fetch()
                try:
                    repo.git.reset("--hard", "@{u}")
                except Exception:
                    repo.git.reset("--hard", "origin/master")
                if old_hash != repo.head.commit.hexsha:
                    needs_build = True

        build_dir = server_dir / "build"
        dist_dir = server_dir / "dist"
        if not (build_dir.exists() or dist_dir.exists()):
            needs_build = True

        if needs_build:
            logfire.info(f"Building bgutil-ytdlp-pot-provider server ({target_version or 'latest'})...")
            shutil.rmtree(build_dir, ignore_errors=True)
            shutil.rmtree(dist_dir, ignore_errors=True)
            is_win = os.name == "nt"
            subprocess.run([npm_bin, "ci"], cwd=server_dir, check=True, capture_output=True, shell=is_win)
            subprocess.run([npx_bin, "tsc"], cwd=server_dir, check=True, capture_output=True, shell=is_win)
            logfire.info("bgutil-ytdlp-pot-provider built successfully.")

        if server_dir.exists():
            return server_dir
    except Exception as e:
        logfire.warning(f"Failed to setup bgutil-ytdlp-pot-provider: {e}")

    return None


class DiscoveredVideo(BaseModel):
    """Metadata container for a video discovered from local files or YouTube feeds."""

    video_id: str = Field(description="Unique video identifier.")
    uploader: str = Field(description="Channel or uploader handle identifier.")
    channel_name: str | None = Field(default=None, description="Human-readable channel display name.")
    title: str = Field(description="Video title.")
    source_type: Literal["local", "youtube"] = Field(description="Ingestion source type.")
    source_ref: str = Field(description="Filesystem path or YouTube URL.")
    timestamp: int = Field(default=0, description="Upload or file modification Unix timestamp.")
    local_path: Path | None = Field(default=None, description="Path to local file if available.")


class IngestionManager:
    """Orchestrates video intake from watched directories and YouTube feeds."""

    config: SourcesConfig
    coordinator: PipelineCoordinator
    git_sync: GitSyncManager
    channel_mapper: ChannelMapper
    pot_dir: Path
    pot_server_dir: Path | None
    base_work_dir: Path
    test_download_dir: Path
    consecutive_failures: int
    failed_video_counts: dict[str, int]
    cached_local_videos: dict[str, DiscoveredVideo]
    cached_youtube_videos: dict[str, DiscoveredVideo]
    cached_videos: list[DiscoveredVideo]
    _last_pot_update: float
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

        self.pot_server_dir = (
            ensure_pot_provider(pot_dir, check_updates=False) if config.youtube.enable_pot_provider else None
        )

        self.test_download_dir = base_work_dir / "tmp_test"
        self.test_download_dir.mkdir(parents=True, exist_ok=True)

        self.consecutive_failures = 0
        self.failed_video_counts = {}
        self.cached_local_videos = {}
        self.cached_youtube_videos = {}
        self.cached_videos = []
        self._last_pot_update = time()
        self._files_rescan_task = None
        self._youtube_rescan_task = None
        self._scan_lock = asyncio.Lock()
        self._feed_lock = asyncio.Lock()

        # Emit environment diagnostic metrics once on startup
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
        """Constructs base yt-dlp parameters including PO token server, node runtime, and FFmpeg path.

        Returns:
            dict[str, Any]: Base yt-dlp configuration options dictionary.
        """
        # approximate_date enables relative date parsing in flat playlist extraction ("2 days ago" -> Unix timestamp)
        extractor_args: dict[str, Any] = {
            "youtubetab": {"approximate_date": [""]},
        }
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

    def _update_combined_cache(self) -> None:
        """Merges discovered local and YouTube videos into the master cache, sorted newest-first."""
        all_vids = list(self.cached_local_videos.values()) + list(self.cached_youtube_videos.values())
        deduped: dict[str, DiscoveredVideo] = {v.video_id: v for v in all_vids}
        self.cached_videos = sorted(deduped.values(), key=lambda v: v.timestamp, reverse=True)

    def scan_watch_folders(self) -> list[DiscoveredVideo]:
        """Scans watched directories recursively for video media files.

        Returns:
            list[DiscoveredVideo]: Discovered local video items.
        """
        valid_extensions = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts"}
        discovered: list[DiscoveredVideo] = []

        for folder in self.config.files.folders:
            if not folder.is_dir():
                continue
            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    mtime = int(file_path.stat().st_mtime)
                    uploader = file_path.parent.name or "local"
                    uploader = uploader.strip("@")[:20]

                    discovered.append(
                        DiscoveredVideo(
                            video_id=file_path.stem,
                            uploader=uploader,
                            channel_name=uploader,
                            title=file_path.stem,
                            source_type="local",
                            source_ref=str(file_path.resolve()),
                            timestamp=mtime,
                            local_path=file_path.resolve(),
                        )
                    )
        return discovered

    def scan_youtube_sources(self, deep_scan: bool = False) -> list[DiscoveredVideo]:
        """Fetches metadata for YouTube sources using flat single-request extraction.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.

        Returns:
            list[DiscoveredVideo]: Discovered YouTube video items.
        """
        if not self.config.youtube.urls:
            return []

        ydl_opts = self._get_base_ytdlp_opts()
        ydl_opts["extract_flat"] = True

        lookback_seconds = int(
            (2 * self.config.youtube.rescan_interval_minutes + self.config.youtube.rescan_jitter_minutes) * 60
        )
        cutoff_ts = int(time()) - lookback_seconds if not deep_scan else 0

        # Expand bare channel URLs into both /videos (Uploads) and /streams (Live VODs), excluding /shorts
        target_urls: list[str] = []
        for raw_url in self.config.youtube.urls:
            clean_url = raw_url.rstrip("/")
            if clean_url.endswith("/shorts"):
                continue
            if re.match(r"^https?://(www\.)?youtube\.com/(@[\w.-]+|channel/UC[\w-]+)$", clean_url):
                target_urls.extend([f"{clean_url}/videos", f"{clean_url}/streams"])
            else:
                target_urls.append(clean_url)

        discovered: list[DiscoveredVideo] = []

        with YoutubeDL(cast(Any, ydl_opts)) as ydl:
            for idx, url in enumerate(target_urls):
                if idx > 0 and self.config.youtube.channel_delay_seconds > 0:
                    delay = random.uniform(
                        0.5 * self.config.youtube.channel_delay_seconds,
                        1.5 * self.config.youtube.channel_delay_seconds,
                    )
                    time_sleep(delay)

                try:
                    result = ydl.extract_info(url, download=False)
                    if not result:
                        continue

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

                        # Skip upcoming premieres, active live streams, and shorts
                        if entry.get("live_status") in ("is_upcoming", "is_live", "post_live"):
                            continue
                        url_val = entry.get("url") or entry.get("webpage_url") or ""
                        if "/shorts/" in url_val:
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

                        # Delta early-exit condition: if past the lookback window, stop paginating
                        if not deep_scan and ts > 0 and ts < cutoff_ts:
                            break

                        entries_processed.append(entry)

                        uploader = self.channel_mapper.get_uploader_id(entry)
                        if not uploader:
                            uploader = str(entry.get("uploader") or "youtube")
                        uploader = uploader.strip("@")[:20]

                        ch_name = self.channel_mapper.get_channel_name(entry)
                        web_url = f"https://www.youtube.com/watch?v={vid}"

                        discovered.append(
                            DiscoveredVideo(
                                video_id=vid,
                                uploader=uploader,
                                channel_name=ch_name,
                                title=title,
                                source_type="youtube",
                                source_ref=web_url,
                                timestamp=ts,
                            )
                        )

                    self.channel_mapper.update_from_entries(entries_processed)

                except Exception as e:
                    err_msg = str(e)
                    logfire.warning(f"Error scanning YouTube source {url}: {err_msg}")
                    if "Sign in to confirm you’re not a bot" in err_msg or "HTTP Error 429" in err_msg:
                        logfire.warning("Bot block encountered. Halting current YouTube scan cycle.")
                        break

        logfire.info(f"Discovered {len(discovered)} video(s) across YouTube sources.")
        return discovered

    async def scan_watch_folders_async(self) -> list[DiscoveredVideo]:
        """Scans local watch directories off-thread and updates the in-memory cache.

        Returns:
            list[DiscoveredVideo]: Discovered local video items.
        """
        async with self._scan_lock:
            loop = asyncio.get_running_loop()
            discovered = await loop.run_in_executor(None, self.scan_watch_folders)
            self.cached_local_videos = {v.video_id: v for v in discovered}
            self._update_combined_cache()
            return discovered

    async def scan_youtube_sources_async(self, deep_scan: bool = False) -> list[DiscoveredVideo]:
        """Scans YouTube sources off-thread and updates the in-memory cache.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.

        Returns:
            list[DiscoveredVideo]: Discovered YouTube video items.
        """
        async with self._scan_lock:
            loop = asyncio.get_running_loop()
            discovered = await loop.run_in_executor(None, self.scan_youtube_sources, deep_scan)
            self.cached_youtube_videos.update({v.video_id: v for v in discovered})
            self._update_combined_cache()
            return discovered

    async def scan_all_sources(self, deep_scan: bool = False) -> list[DiscoveredVideo]:
        """Executes rescan of both local watch folders and YouTube sources off-thread.

        Args:
            deep_scan: If True, crawls full channel history on YouTube. Defaults to False.

        Returns:
            list[DiscoveredVideo]: List of all discovered videos sorted newest-first.
        """
        await self.scan_watch_folders_async()
        await self.scan_youtube_sources_async(deep_scan=deep_scan)
        return self.cached_videos

    def check_video_availability(self, video: DiscoveredVideo) -> tuple[bool, bool]:
        """Tests video availability by downloading a brief snippet.

        Args:
            video: Discovered video metadata.

        Returns:
            tuple[bool, bool]: (is_available, is_bot_blocked).
        """
        if video.source_type == "local":
            return (video.local_path is not None and video.local_path.is_file()), False

        test_opts = self._get_base_ytdlp_opts()
        duration_s = self.config.youtube.download_test_duration_s
        outtmpl = str(self.test_download_dir / f"test_{video.video_id}.%(ext)s")

        test_opts.update(
            {
                "format": "bestaudio/best",
                "download_ranges": lambda info, _: [{"start_time": 0, "end_time": duration_s}],
                "force_keyframes_at_cuts": False,
                "outtmpl": outtmpl,
            }
        )

        try:
            with YoutubeDL(cast(Any, test_opts)) as ydl:
                info = ydl.extract_info(video.source_ref, download=True)
                test_files = list(self.test_download_dir.glob(f"test_{video.video_id}.*"))
                has_file = bool(test_files and test_files[0].is_file())

                for p in test_files:
                    try:
                        p.unlink()
                    except OSError:
                        pass

                return (bool(info and has_file), False)
        except Exception as e:
            for p in self.test_download_dir.glob(f"test_{video.video_id}.*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            err_msg = str(e)
            is_bot = "Sign in to confirm you’re not a bot" in err_msg or "HTTP Error 429" in err_msg
            logfire.warning(f"Availability check failed for '{video.title}' ({video.video_id}): {err_msg}")
            return False, is_bot

    async def download_youtube_to_combo_dir(self, video: DiscoveredVideo, combo_work_dir: Path) -> Path:
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

        loop = asyncio.get_running_loop()

        def _do_download() -> Path:
            with YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(video.source_ref, download=True)
                if not info:
                    raise RuntimeError(f"Could not extract download info for {video.source_ref}")
                fn = ydl.prepare_filename(info)
                return Path(fn)

        with logfire.span("Downloading YouTube Video to {job_dir}: {title}", job_dir=job_dir.name, title=video.title):
            downloaded = await loop.run_in_executor(None, _do_download)
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
        """Gathers outstanding queued videos for a given pipeline from memory cache.

        Args:
            pipeline: Combination pipeline instance.

        Returns:
            list[dict[str, Any]]: List of queued video items pending for this pipeline.
        """
        outstanding = []
        for v in self.cached_videos:
            if pipeline.active_video and pipeline.active_video.video_id == v.video_id:
                continue

            if not self.git_sync.is_video_completed(v.uploader, v.video_id, pipeline.full_shortcode):
                outstanding.append(
                    {
                        "video_id": v.video_id,
                        "title": v.title,
                        "uploader": v.uploader,
                        "channel_name": v.channel_name,
                        "source_type": v.source_type,
                        "timestamp": v.timestamp,
                    }
                )
        return outstanding[:30]

    def get_all_pipeline_queues(self) -> dict[str, list[dict[str, Any]]]:
        """Computes outstanding queues for all active pipelines without network calls.

        Returns:
            dict[str, list[dict[str, Any]]]: Mapping of pipeline names to their outstanding videos.
        """
        return {name: self.get_pipeline_queue(pipe) for name, pipe in self.coordinator.pipelines.items()}

    async def feed_idle_pipelines(self) -> None:
        """Assigns the newest available video to each idle pipeline."""
        async with self._feed_lock:
            if self.consecutive_failures >= self.config.youtube.max_consecutive_failures:
                logfire.warning("Halting feed: max consecutive download failures reached. Will retry on next scan.")
                return

            idle_pipelines = [p for p in self.coordinator.pipelines.values() if p.is_idle()]
            if not idle_pipelines:
                return

            if not self.cached_videos:
                await self.scan_all_sources(deep_scan=False)

            loop = asyncio.get_running_loop()

            for pipe in idle_pipelines:
                for candidate in self.cached_videos:
                    if self.failed_video_counts.get(candidate.video_id, 0) >= 3:
                        continue
                    if self.git_sync.is_video_completed(candidate.uploader, candidate.video_id, pipe.full_shortcode):
                        continue

                    pipe.status = PipelineStatus.PREPARING

                    is_available, is_bot = await loop.run_in_executor(None, self.check_video_availability, candidate)
                    if not is_available:
                        pipe.status = PipelineStatus.IDLE
                        if is_bot:
                            logfire.warning(
                                f"Bot detection triggered on '{candidate.title}' ({candidate.video_id}). "
                                "Halting intake feed for this cycle."
                            )
                            return

                        self.consecutive_failures += 1
                        self.failed_video_counts[candidate.video_id] = (
                            self.failed_video_counts.get(candidate.video_id, 0) + 1
                        )
                        logfire.warning(
                            f"Video '{candidate.title}' ({candidate.video_id}) unavailable "
                            f"(attempt {self.failed_video_counts[candidate.video_id]}/3)."
                        )
                        if self.consecutive_failures >= self.config.youtube.max_consecutive_failures:
                            return
                        continue

                    try:
                        if candidate.source_type == "local" and candidate.local_path:
                            media_path = candidate.local_path
                        else:
                            media_path = await self.download_youtube_to_combo_dir(candidate, pipe.work_dir)

                        self.consecutive_failures = 0
                        logfire.info(f"Feeding '{pipe.config.name}' with '{candidate.title}' ({candidate.video_id}).")
                        await pipe.prepare_and_set_video(
                            video_id=candidate.video_id,
                            uploader=candidate.uploader,
                            title=candidate.title,
                            raw_video_path=media_path,
                            channel_name=candidate.channel_name,
                            source_type=candidate.source_type,
                        )
                        break
                    except Exception as e:
                        pipe.status = PipelineStatus.IDLE
                        pipe.active_video = None
                        shutil.rmtree(pipe.work_dir / candidate.video_id, ignore_errors=True)

                        err_str = str(e)
                        if "Sign in to confirm you’re not a bot" in err_str or "HTTP Error 429" in err_str:
                            logfire.warning(f"Bot detection on download of '{candidate.title}'. Halting feed.")
                            return

                        self.consecutive_failures += 1
                        self.failed_video_counts[candidate.video_id] = (
                            self.failed_video_counts.get(candidate.video_id, 0) + 1
                        )
                        logfire.error(f"Failed to prepare '{candidate.title}' for {pipe.config.name}: {e}")
                        if self.consecutive_failures >= self.config.youtube.max_consecutive_failures:
                            return
                        continue

    async def start(self) -> None:
        """Starts the ingestion manager with a deep scan on boot, followed by periodic delta intake loops."""
        await self.scan_watch_folders_async()
        # Full deep crawl on boot to populate the complete historical video catalog in RAM
        await self.scan_youtube_sources_async(deep_scan=True)
        await self.feed_idle_pipelines()

        self._files_rescan_task = asyncio.create_task(self._periodic_files_loop())
        self._youtube_rescan_task = asyncio.create_task(self._periodic_youtube_loop())

    async def stop(self) -> None:
        """Stops the periodic intake loops gracefully."""
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
        """Periodically scans local watch folders without bot detection throttling."""
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
        """Periodically scans YouTube sources using time-based delta lookback and anti-bot pacing."""
        while True:
            base_sec = self.config.youtube.rescan_interval_minutes * 60
            jitter_sec = max(0.0, self.config.youtube.rescan_jitter_minutes * 60.0)
            delay_sec = base_sec + random.uniform(0.0, jitter_sec)

            logfire.debug(
                f"Next YouTube delta rescan scheduled in {delay_sec / 60:.1f} minutes "
                f"(base: {self.config.youtube.rescan_interval_minutes}m, "
                f"jitter: +{(delay_sec - base_sec) / 60:.1f}m)."
            )
            await asyncio.sleep(delay_sec)

            self.consecutive_failures = 0
            try:
                if self.config.youtube.enable_pot_provider:
                    now = time()
                    interval_sec = self.config.youtube.pot_update_interval_hours * 3600
                    if (now - self._last_pot_update) >= interval_sec:
                        self._last_pot_update = now
                        logfire.info("Checking for bgutil-ytdlp-pot-provider upstream updates...")
                        loop = asyncio.get_running_loop()
                        self.pot_server_dir = await loop.run_in_executor(None, ensure_pot_provider, self.pot_dir, True)

                logfire.debug("Running periodic YouTube delta scan...")
                await self.scan_youtube_sources_async(deep_scan=False)
                await self.coordinator.check_and_finalize_completed_pipelines()
                await self.feed_idle_pipelines()
            except Exception as e:
                logfire.warning(f"Error during periodic YouTube rescan: {e}")
