"""Source ingestion and direct-to-combo video feeding."""

from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import logfire
from pydantic import BaseModel, Field

from ai_sub.server.combo_runner import ComboRunner
from ai_sub.server.coordinator import ComboRunnersCoordinator, is_valid_video_id
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.server_config import SourcesConfig
from ai_sub.server.youtube import (
    VALID_VIDEO_EXTENSIONS,
    DiscoveredVideo,
    DownloadedVideo,
    ScanResult,
    YouTubeBotDetectionError,
    YouTubeClient,
    YouTubeVideoUnavailableError,
)
from ai_sub.video import sanitize_uploader


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
        description="Monotonic timestamp when this video becomes eligible to be retried.",
    )
    failure_count: int = Field(
        default=0,
        description="Total failed download attempts for this video.",
    )


def _cleanup_failed_job_dir(job_dir: Path) -> None:
    """Cleans temporary download artifacts while preserving resumable YAML checkpoints and media.

    Args:
        job_dir: Directory containing job files.
    """
    if not job_dir.is_dir():
        return
    yaml_files = list(job_dir.glob("*.yaml"))
    if yaml_files:
        for part_file in job_dir.glob("*.part"):
            try:
                part_file.unlink()
            except OSError:
                pass
        for ytdl_file in job_dir.glob("*.ytdl"):
            try:
                ytdl_file.unlink()
            except OSError:
                pass
    else:
        shutil.rmtree(job_dir, ignore_errors=True)


class IngestionManager:
    """Orchestrates video intake from watched directories and YouTube feeds."""

    config: SourcesConfig
    coordinator: ComboRunnersCoordinator
    git_sync: GitSyncManager
    base_work_dir: Path
    youtube_client: YouTubeClient
    local_videos: dict[str, QueuedVideo]
    youtube_videos: dict[str, QueuedVideo]
    _ignored_youtube_ids: set[str]
    _aborted_videos: dict[tuple[str, str], str]
    _youtube_cooldown_until: float
    _youtube_cooldown_reason: str | None
    _last_idle_summary: dict[str, str]
    _loop: asyncio.AbstractEventLoop | None
    _cooldown_wake_task: asyncio.Task[None] | None
    _files_rescan_task: asyncio.Task[None] | None
    _youtube_rescan_task: asyncio.Task[None] | None
    _scan_lock: asyncio.Lock
    _feed_lock: asyncio.Lock
    _started: bool
    _stopped: bool

    def __init__(
        self,
        config: SourcesConfig,
        coordinator: ComboRunnersCoordinator,
        git_sync: GitSyncManager,
        base_work_dir: Path,
    ) -> None:
        """Initializes the ingestion manager and configures the YouTube client.

        Args:
            config: Intake sources configuration containing files and youtube options.
            coordinator: Combination runners coordinator managing active runners.
            git_sync: Git synchronization manager.
            base_work_dir: Root working directory for downloads and runners.
        """
        self.config = config
        self.coordinator = coordinator
        self.git_sync = git_sync
        self.base_work_dir = base_work_dir
        self.coordinator.on_runners_finalized = self.feed_idle_runners
        self.coordinator.on_video_aborted = self.mark_video_aborted

        self.youtube_client = YouTubeClient(config.youtube, base_work_dir)
        self.local_videos = {}
        self.youtube_videos = {}
        self._ignored_youtube_ids = set()
        self._aborted_videos = {}
        self._youtube_cooldown_until = 0.0
        self._youtube_cooldown_reason = None
        self._last_idle_summary = {}
        self._loop = None
        self._cooldown_wake_task = None
        self._files_rescan_task = None
        self._youtube_rescan_task = None
        self._scan_lock = asyncio.Lock()
        self._feed_lock = asyncio.Lock()
        self._started = False
        self._stopped = False

    def mark_video_aborted(self, runner_name: str, video_id: str, reason: str) -> None:
        """Records an aborted video for a specific runner so it is held out from automatic retry.

        Args:
            runner_name: Target combination runner name.
            video_id: Video identifier that aborted.
            reason: Human-readable abort reason displayed on the dashboard.
        """
        self._aborted_videos[(runner_name, video_id)] = reason

    def _is_youtube_in_cooldown(self) -> bool:
        """Checks if YouTube operations are temporarily suspended due to bot detection or failure limit.

        Returns:
            bool: True if YouTube operations are within a cooldown window, False otherwise.
        """
        return monotonic() < self._youtube_cooldown_until

    def get_youtube_cooldown_info(self) -> dict[str, Any]:
        """Returns details on the current YouTube intake cooldown status.

        Returns:
            dict[str, Any]: Cooldown state dictionary including remaining seconds and reason.
        """
        in_cooldown = self._is_youtube_in_cooldown()
        remaining = max(0.0, self._youtube_cooldown_until - monotonic()) if in_cooldown else 0.0
        return {
            "in_cooldown": in_cooldown,
            "reason": self._youtube_cooldown_reason if in_cooldown else None,
            "remaining_seconds": round(remaining, 1),
        }

    def _trigger_youtube_cooldown(self, reason: str = "") -> None:
        """Sets a cooldown timer on all YouTube operations and schedules an automatic wake-up.

        Args:
            reason: Context description for the log warning. Defaults to "".
        """
        if self._stopped:
            return

        cooldown_sec = self.config.youtube.cooldown.total_seconds()
        self._youtube_cooldown_until = monotonic() + cooldown_sec
        self._youtube_cooldown_reason = reason or "Download failure threshold reached"
        logfire.warning(
            f"YouTube intake cooldown triggered ({self._youtube_cooldown_reason}). "
            f"Pausing YouTube downloads for {cooldown_sec / 60:.1f} minute(s)."
        )

        async def _wake_on_cooldown_expiry() -> None:
            try:
                await asyncio.sleep(cooldown_sec)
                if self._stopped:
                    return
                self._youtube_cooldown_until = 0.0
                self._youtube_cooldown_reason = None
                logfire.info("YouTube intake cooldown expired. Resuming YouTube job feeding...")
                await self._scan_youtube_sources_async(deep_scan=False)
                await self.feed_idle_runners()
            except asyncio.CancelledError:
                pass

        def _schedule() -> None:
            if self._stopped:
                return
            if self._cooldown_wake_task and not self._cooldown_wake_task.done():
                self._cooldown_wake_task.cancel()
            self._cooldown_wake_task = asyncio.create_task(_wake_on_cooldown_expiry())

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(_schedule)

    def _scan_watch_folders(self) -> dict[str, DiscoveredVideo]:
        """Scans watched directories recursively for video media files off-thread without mutating shared state.

        Returns:
            dict[str, DiscoveredVideo]: Map of video ID to discovered file metadata.
        """
        discovered: dict[str, DiscoveredVideo] = {}

        for folder in self.config.files.folders:
            if not folder.is_dir():
                continue
            for file_path in sorted(folder.rglob("*")):
                if not file_path.is_file() or file_path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
                    continue

                vid = file_path.stem
                if not is_valid_video_id(vid):
                    logfire.warning(f"Skipping local file with invalid filename stem {vid!r}: {file_path}")
                    continue

                resolved_path = file_path.resolve()
                if vid in discovered:
                    if discovered[vid].local_path != resolved_path:
                        logfire.warning(
                            f"Skipping duplicate local video ID '{vid}' at {resolved_path} "
                            f"(already mapped to {discovered[vid].local_path})."
                        )
                    continue

                mtime = int(file_path.stat().st_mtime)
                uploader = sanitize_uploader(file_path.parent.name or "local", fallback="local")

                discovered[vid] = DiscoveredVideo(
                    video_id=vid,
                    uploader=uploader,
                    channel_name=uploader,
                    title=vid,
                    source_ref=str(resolved_path),
                    timestamp=mtime,
                    local_path=resolved_path,
                )

        return discovered

    def _scan_youtube_sources(self, deep_scan: bool, ignored_ids: frozenset[str]) -> ScanResult:
        """Fetches metadata for YouTube sources off-thread without mutating shared state.

        Args:
            deep_scan: If True, crawls full channel history; if False, uses per-URL deep-scan state.
            ignored_ids: Snapshot of ignored YouTube video IDs to skip.

        Returns:
            ScanResult: Discovered YouTube videos and any newly spotted members-only IDs.
        """
        if not self.config.youtube.urls or self._is_youtube_in_cooldown():
            return ScanResult()
        return self.youtube_client.scan_sources(deep_scan=deep_scan, ignored_ids=ignored_ids)

    async def _scan_watch_folders_async(self) -> list[QueuedVideo]:
        """Scans local watch directories off-thread and updates in-memory cache safely on the event loop.

        Returns:
            list[QueuedVideo]: Discovered local video items.
        """
        async with self._scan_lock:
            discovered = await asyncio.to_thread(self._scan_watch_folders)

            for vid, item in discovered.items():
                if vid not in self.local_videos:
                    self.local_videos[vid] = QueuedVideo(
                        video_id=item.video_id,
                        uploader=item.uploader,
                        channel_name=item.channel_name,
                        title=item.title,
                        source_type="local",
                        source_ref=item.source_ref,
                        timestamp=item.timestamp,
                        local_path=item.local_path,
                    )
                else:
                    existing = self.local_videos[vid]
                    if existing.timestamp != item.timestamp or existing.local_path != item.local_path:
                        for key in [k for k in self._aborted_videos if k[1] == vid]:
                            self._aborted_videos.pop(key, None)
                        existing.status = "ready"
                        existing.error_reason = None
                        existing.retry_after = None
                    existing.local_path = item.local_path
                    existing.timestamp = item.timestamp
                    existing.uploader = item.uploader
                    existing.channel_name = item.channel_name

            for vid in set(self.local_videos.keys()) - set(discovered.keys()):
                self.local_videos.pop(vid, None)
                if vid not in self.youtube_videos:
                    for key in [k for k in self._aborted_videos if k[1] == vid]:
                        self._aborted_videos.pop(key, None)

            if self.local_videos:
                logfire.debug(f"Watch folders scan complete: {len(self.local_videos)} local file(s) tracked.")

            return list(self.local_videos.values())

    async def _scan_youtube_sources_async(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Scans YouTube sources off-thread and updates in-memory cache safely on the event loop.

        Args:
            deep_scan: If True, forces a full channel history crawl across all sources.

        Returns:
            list[QueuedVideo]: Discovered YouTube video items.
        """
        if not self.config.youtube.urls:
            return []

        if self._is_youtube_in_cooldown():
            remaining_m = (self._youtube_cooldown_until - monotonic()) / 60.0
            logfire.info(f"Skipping YouTube scan: bot cooldown active ({remaining_m:.1f}m remaining).")
            return list(self.youtube_videos.values())

        async with self._scan_lock:
            ignored_snapshot = frozenset(self._ignored_youtube_ids)
            try:
                scan_result = await asyncio.to_thread(self._scan_youtube_sources, deep_scan, ignored_snapshot)
            except YouTubeBotDetectionError as e:
                self._trigger_youtube_cooldown(reason=str(e))
                return list(self.youtube_videos.values())
            except Exception as e:
                logfire.warning(f"Unexpected error during YouTube source scan: {e}")
                return list(self.youtube_videos.values())

            if scan_result.members_only_ids:
                self._ignored_youtube_ids.update(scan_result.members_only_ids)
                for ignored_vid in scan_result.members_only_ids:
                    self.youtube_videos.pop(ignored_vid, None)

            discovered_new = 0
            for item in scan_result.videos:
                if item.video_id in self._ignored_youtube_ids:
                    continue
                if item.video_id not in self.youtube_videos:
                    self.youtube_videos[item.video_id] = QueuedVideo(
                        video_id=item.video_id,
                        uploader=item.uploader,
                        channel_name=item.channel_name,
                        title=item.title,
                        source_type="youtube",
                        source_ref=item.source_ref,
                        timestamp=item.timestamp,
                    )
                    discovered_new += 1
                else:
                    existing = self.youtube_videos[item.video_id]
                    if item.title and existing.title != item.title:
                        existing.title = item.title
                    if item.timestamp > 0:
                        existing.timestamp = item.timestamp
                    if item.uploader != "youtube" and existing.uploader == "youtube":
                        existing.uploader = item.uploader
                    if item.channel_name and existing.channel_name != item.channel_name:
                        existing.channel_name = item.channel_name

            logfire.info(
                f"YouTube scan complete: {len(self.youtube_videos)} metadata entry(s) tracked in memory "
                f"({discovered_new} newly discovered)."
            )
            return list(self.youtube_videos.values())

    async def _download_youtube_to_combo_dir(self, video: QueuedVideo, combo_work_dir: Path) -> Path:
        """Downloads a video stream into the combination directory and updates authoritative metadata.

        Args:
            video: Discovered video metadata.
            combo_work_dir: Working directory of the target combination runner.

        Returns:
            Path: Filesystem path to the downloaded media file.
        """
        job_dir = combo_work_dir / video.video_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def _do_download() -> DownloadedVideo:
            return self.youtube_client.download_video(
                video_id=video.video_id,
                source_ref=video.source_ref,
                destination_dir=job_dir,
            )

        with logfire.span("Downloading YouTube Video to {job_dir}: {title}", job_dir=job_dir.name, title=video.title):
            downloaded = await asyncio.to_thread(_do_download)
            if downloaded.uploader and downloaded.uploader.lower() != "youtube":
                video.uploader = downloaded.uploader
            if downloaded.channel_name:
                video.channel_name = downloaded.channel_name
            if downloaded.title:
                video.title = downloaded.title
            if downloaded.timestamp > 0:
                video.timestamp = downloaded.timestamp
            return downloaded.file_path

    def _is_completed_for_runner(self, video: QueuedVideo, full_shortcode: str) -> bool:
        """Checks completion in Git using global video ID for YouTube and uploader+ID for local files.

        Args:
            video: Candidate queued video.
            full_shortcode: Pipeline output shortcode.

        Returns:
            bool: True if subtitles already exist in Git, False otherwise.
        """
        if video.source_type == "youtube":
            if video.uploader.lower() == "youtube":
                if known := self.git_sync.get_completed_uploader(video.video_id, full_shortcode):
                    video.uploader = known
            return self.git_sync.is_video_completed(video.video_id, full_shortcode, uploader=None)
        return self.git_sync.is_video_completed(video.video_id, full_shortcode, uploader=video.uploader)

    def _get_runner_queue(self, runner: ComboRunner) -> dict[str, Any]:
        """Gathers outstanding queued videos for a given runner with total count and top 50 items.

        Args:
            runner: Combination runner instance.

        Returns:
            dict[str, Any]: Dictionary containing 'total_count' and top 50 'items'.
        """
        candidates = list(self.local_videos.values()) + list(self.youtube_videos.values())
        outstanding = []
        now = monotonic()

        for v in candidates:
            if runner.pipeline and runner.pipeline.video_id == v.video_id:
                continue

            if runner.preparing_video_info and runner.preparing_video_info.get("video_id") == v.video_id:
                continue

            if not self._is_completed_for_runner(v, runner.full_shortcode):
                abort_reason = self._aborted_videos.get((runner.config.name, v.video_id))
                if abort_reason is not None:
                    status_val = "failed"
                    error_reason = abort_reason
                    retry_after = None
                    retry_in_s = None
                else:
                    status_val = v.status
                    error_reason = v.error_reason
                    retry_after = v.retry_after
                    retry_in_s = max(0, int(v.retry_after - now)) if v.retry_after and v.retry_after > now else None

                outstanding.append(
                    {
                        "video_id": v.video_id,
                        "title": v.title,
                        "uploader": v.uploader,
                        "channel_name": v.channel_name,
                        "source_type": v.source_type,
                        "timestamp": v.timestamp,
                        "status": status_val,
                        "error_reason": error_reason,
                        "retry_after": retry_after,
                        "retry_in_s": retry_in_s,
                        "failure_count": v.failure_count,
                    }
                )

        outstanding.sort(key=lambda x: (-x["timestamp"], x["title"].lower()))
        return {
            "total_count": len(outstanding),
            "items": outstanding[:50],
        }

    def get_all_runner_queues(self) -> dict[str, dict[str, Any]]:
        """Computes outstanding queues for all active runners without network calls.

        Returns:
            dict[str, dict[str, Any]]: Mapping of runner names to their queue dictionaries.
        """
        return {name: self._get_runner_queue(runner) for name, runner in self.coordinator.runners.items()}

    async def _try_feed_runner(
        self,
        runner: ComboRunner,
        video: QueuedVideo,
        now: float,
    ) -> tuple[bool, bool, bool]:
        """Attempts to acquire an idle runner, fetch/verify media, and prepare its pipeline.

        Args:
            runner: Target combination runner.
            video: Candidate queued video.
            now: Current monotonic timestamp.

        Returns:
            tuple[bool, bool, bool]: (fed_successfully, stop_loop_for_runner, is_download_failure).
        """
        if not is_valid_video_id(video.video_id):
            logfire.warning(f"Skipping queued video with invalid video ID {video.video_id!r}.")
            self._aborted_videos[(runner.config.name, video.video_id)] = "Invalid video ID"
            return False, False, False

        if video.source_type == "local" and (not video.local_path or not video.local_path.is_file()):
            video.status = "failed"
            video.error_reason = "File missing from disk"
            video.retry_after = now + 120
            return False, False, False

        acquired = await self.coordinator.acquire_idle_runner(
            runner_name=runner.config.name,
            video_id=video.video_id,
            title=video.title,
            uploader=video.uploader,
            channel_name=video.channel_name,
            source_type=video.source_type,
        )
        if not acquired:
            return False, True, False

        fed = False
        try:
            if video.source_type == "local":
                assert video.local_path is not None
                raw_path = video.local_path
            else:
                try:
                    raw_path = await self._download_youtube_to_combo_dir(video, runner.work_dir)
                except YouTubeVideoUnavailableError as e:
                    _cleanup_failed_job_dir(runner.work_dir / video.video_id)
                    if e.reason == "Members only":
                        self._ignored_youtube_ids.add(video.video_id)
                        self.youtube_videos.pop(video.video_id, None)
                        logfire.info(f"Ignoring members-only video '{video.title}' ({video.video_id}).")
                    else:
                        video.status = "unplayable"
                        video.error_reason = e.reason
                        video.retry_after = now + (24 * 3600)
                        video.failure_count += 1
                        logfire.warning(f"Video '{video.title}' is unplayable ({e.reason}). Retrying in 24h.")
                    return False, False, False
                except YouTubeBotDetectionError as e:
                    _cleanup_failed_job_dir(runner.work_dir / video.video_id)
                    self._trigger_youtube_cooldown(reason=str(e))
                    return False, True, False
                except Exception as e:
                    _cleanup_failed_job_dir(runner.work_dir / video.video_id)
                    video.status = "failed"
                    video.error_reason = "Download failed"
                    video.retry_after = now + (30 * 60)
                    video.failure_count += 1
                    logfire.warning(f"Download failed for '{video.title}': {e}")
                    return False, False, True

            try:
                logfire.info(f"Feeding '{runner.config.name}' with '{video.title}' ({video.video_id}).")
                await self.coordinator.assign_prepared_video(
                    runner_name=runner.config.name,
                    video_id=video.video_id,
                    uploader=video.uploader,
                    title=video.title,
                    raw_video_path=raw_path,
                    channel_name=video.channel_name,
                    source_type=video.source_type,
                    timestamp=video.timestamp,
                )
                video.status = "ready"
                video.error_reason = None
                video.retry_after = None
                fed = True
                return True, True, False
            except Exception as e:
                _cleanup_failed_job_dir(runner.work_dir / video.video_id)
                logfire.warning(
                    f"Failed to prepare {video.source_type} video '{video.title}' for '{runner.config.name}': {e}"
                )
                video.status = "failed"
                video.error_reason = f"Preparation error: {e}"
                video.retry_after = now + 120
                return False, False, False
        finally:
            if not fed:
                await self.coordinator.release_preparing_runner(runner.config.name)

    async def feed_idle_runners(self) -> None:
        """Assigns the newest available video to each idle runner."""
        if not self._started or self._stopped:
            return

        async with self._feed_lock:
            idle_runners = [r for r in self.coordinator.runners.values() if r.is_idle()]
            if not idle_runners:
                return

            now = monotonic()

            for runner in idle_runners:
                candidates: list[QueuedVideo] = list(self.local_videos.values())
                if not self._is_youtube_in_cooldown():
                    candidates.extend(self.youtube_videos.values())

                completed_in_git = 0
                in_cooldown = 0
                aborted_count = 0
                eligible: list[QueuedVideo] = []

                for v in candidates:
                    if self._is_completed_for_runner(v, runner.full_shortcode):
                        completed_in_git += 1
                    elif (runner.config.name, v.video_id) in self._aborted_videos:
                        aborted_count += 1
                    elif v.retry_after is not None and now < v.retry_after:
                        in_cooldown += 1
                    else:
                        eligible.append(v)

                def _feed_sort_key(v: QueuedVideo) -> tuple[int, int, str]:
                    has_dir = 1 if (runner.work_dir / v.video_id).is_dir() else 0
                    return (has_dir, v.timestamp, v.title.lower())

                eligible.sort(key=_feed_sort_key, reverse=True)

                if not eligible:
                    if candidates:
                        reason_parts: list[str] = []
                        if completed_in_git:
                            reason_parts.append(f"{completed_in_git} completed in Git")
                        if aborted_count:
                            reason_parts.append(f"{aborted_count} aborted")
                        if in_cooldown:
                            reason_parts.append(f"{in_cooldown} in cooldown")
                        reason_str = ", ".join(reason_parts) if reason_parts else "none available"
                        summary_key = f"{len(candidates)}:{reason_str}"

                        msg = (
                            f"Runner '{runner.config.name}' is idle: 0 pending videos "
                            f"({len(candidates)} tracked: {reason_str})."
                        )
                        if self._last_idle_summary.get(runner.config.name) != summary_key:
                            self._last_idle_summary[runner.config.name] = summary_key
                            logfire.info(msg)
                        else:
                            logfire.debug(msg)
                    continue

                self._last_idle_summary.pop(runner.config.name, None)

                consecutive_download_failures = 0
                max_failures = self.config.youtube.maxfailures

                for video in eligible:
                    if self._stopped:
                        return

                    fed, stop_runner, is_dl_failure = await self._try_feed_runner(runner, video, now)
                    if is_dl_failure:
                        consecutive_download_failures += 1
                        if consecutive_download_failures >= max_failures:
                            self._trigger_youtube_cooldown(
                                reason=f"{consecutive_download_failures} consecutive download failures"
                            )
                            break
                    if fed or stop_runner:
                        break

    async def start(self) -> None:
        """Starts the ingestion manager with an initial scan, followed by periodic delta intake loops."""
        self._stopped = False
        self._loop = asyncio.get_running_loop()
        await self._scan_watch_folders_async()
        await self._scan_youtube_sources_async(deep_scan=True)
        self._started = True
        await self.feed_idle_runners()

        self._files_rescan_task = asyncio.create_task(self._periodic_files_loop())
        self._youtube_rescan_task = asyncio.create_task(self._periodic_youtube_loop())

    async def stop(self) -> None:
        """Stops the periodic intake loops gracefully."""
        self._stopped = True

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
        interval_sec = max(5.0, self.config.files.interval.total_seconds())
        while not self._stopped:
            await asyncio.sleep(interval_sec)
            if self._stopped:
                break
            try:
                logfire.debug("Running periodic watch folder scan...")
                await self._scan_watch_folders_async()
                await self.coordinator.check_and_finalize_completed_runners()
                await self.feed_idle_runners()
            except Exception as e:
                logfire.warning(f"Error during periodic watch folder scan: {e}")

    async def _periodic_youtube_loop(self) -> None:
        """Periodically scans YouTube sources using symmetric jittered intervals."""
        while not self._stopped:
            base_sec = self.config.youtube.interval.total_seconds()
            jitter_sec = max(0.0, self.config.youtube.jitter.total_seconds())
            delay_sec = max(5.0, base_sec + random.uniform(-jitter_sec, jitter_sec))

            logfire.debug(f"Next YouTube delta rescan scheduled in {delay_sec / 60:.1f} minutes")
            await asyncio.sleep(delay_sec)
            if self._stopped:
                break

            try:
                logfire.debug("Running periodic YouTube delta scan...")
                await self._scan_youtube_sources_async(deep_scan=False)
                await self.coordinator.check_and_finalize_completed_runners()
                await self.feed_idle_runners()
            except Exception as e:
                logfire.warning(f"Error during periodic YouTube rescan: {e}")
