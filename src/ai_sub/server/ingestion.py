"""Source ingestion and direct-to-combo video feeding."""

from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path
from time import time
from typing import Any, Literal

import logfire
from pydantic import BaseModel, Field

from ai_sub.server.config import SourcesConfig
from ai_sub.server.coordinator import PipelineCoordinator
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.pipeline import ComboPipeline
from ai_sub.server.youtube import (
    YouTubeBotDetectionError,
    YouTubeClient,
    YouTubeVideoUnavailableError,
)


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
    base_work_dir: Path
    youtube_client: YouTubeClient
    local_videos: dict[str, QueuedVideo]
    youtube_videos: dict[str, QueuedVideo]
    _youtube_cooldown_until: float
    _youtube_cooldown_reason: str | None
    _last_idle_summary: dict[str, str]
    _loop: asyncio.AbstractEventLoop | None
    _cooldown_wake_task: asyncio.Task[None] | None
    _files_rescan_task: asyncio.Task[None] | None
    _youtube_rescan_task: asyncio.Task[None] | None
    _scan_lock: asyncio.Lock
    _feed_lock: asyncio.Lock
    _stopped: bool

    def __init__(
        self,
        config: SourcesConfig,
        coordinator: PipelineCoordinator,
        git_sync: GitSyncManager,
        base_work_dir: Path,
    ) -> None:
        """Initializes the ingestion manager and configures the YouTube client.

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
        self.coordinator.on_pipelines_finalized = self.feed_idle_pipelines

        self.youtube_client = YouTubeClient(config.youtube, base_work_dir)
        self.local_videos = {}
        self.youtube_videos = {}
        self._youtube_cooldown_until = 0.0
        self._youtube_cooldown_reason = None
        self._last_idle_summary = {}
        self._loop = None
        self._cooldown_wake_task = None
        self._files_rescan_task = None
        self._youtube_rescan_task = None
        self._scan_lock = asyncio.Lock()
        self._feed_lock = asyncio.Lock()
        self._stopped = False

    def _is_youtube_in_cooldown(self) -> bool:
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
        in_cooldown = self._is_youtube_in_cooldown()
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
        if self._stopped:
            return

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
                if self._stopped:
                    return
                logfire.info("YouTube intake cooldown expired. Resuming YouTube job feeding...")
                await self.feed_idle_pipelines()
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

    def _scan_watch_folders(self) -> list[QueuedVideo]:
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

        for vid in set(self.local_videos.keys()) - discovered_ids:
            self.local_videos.pop(vid, None)

        if self.local_videos:
            logfire.debug(f"Watch folders scan complete: {len(self.local_videos)} local file(s) tracked.")

        return list(self.local_videos.values())

    def _scan_youtube_sources(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Fetches metadata for YouTube sources using flat single-request extraction.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.

        Returns:
            list[QueuedVideo]: Discovered YouTube video items.
        """
        if not self.config.youtube.urls or self._is_youtube_in_cooldown():
            return list(self.youtube_videos.values())

        try:
            discovered = self.youtube_client.scan_sources(deep_scan=deep_scan)
            discovered_new = 0

            for item in discovered:
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
        except YouTubeBotDetectionError as e:
            self._trigger_youtube_cooldown(reason=str(e))

        return list(self.youtube_videos.values())

    async def _scan_watch_folders_async(self) -> list[QueuedVideo]:
        """Scans local watch directories off-thread and updates in-memory cache.

        Returns:
            list[QueuedVideo]: Discovered local video items.
        """
        async with self._scan_lock:
            return await asyncio.to_thread(self._scan_watch_folders)

    async def _scan_youtube_sources_async(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Scans YouTube sources off-thread and updates in-memory cache.

        Args:
            deep_scan: If True, crawls full channel history; if False, applies lookback window cutoff.

        Returns:
            list[QueuedVideo]: Discovered YouTube video items.
        """
        if self._is_youtube_in_cooldown():
            remaining_m = (self._youtube_cooldown_until - time()) / 60.0
            logfire.info(f"Skipping YouTube scan: bot cooldown active ({remaining_m:.1f}m remaining).")
            return list(self.youtube_videos.values())

        async with self._scan_lock:
            return await asyncio.to_thread(self._scan_youtube_sources, deep_scan)

    async def _scan_all_sources(self, deep_scan: bool = False) -> list[QueuedVideo]:
        """Executes rescan of both local watch folders and YouTube sources off-thread.

        Args:
            deep_scan: If True, crawls full channel history on YouTube. Defaults to False.

        Returns:
            list[QueuedVideo]: List of all discovered videos sorted newest-first.
        """
        await self._scan_watch_folders_async()
        await self._scan_youtube_sources_async(deep_scan=deep_scan)
        all_videos = list(self.local_videos.values()) + list(self.youtube_videos.values())
        all_videos.sort(key=lambda v: (-v.timestamp, v.title.lower()))
        return all_videos

    async def _download_youtube_to_combo_dir(self, video: QueuedVideo, combo_work_dir: Path) -> Path:
        """Downloads a video stream into the combination directory and updates authoritative metadata.

        Args:
            video: Discovered video metadata.
            combo_work_dir: Working directory of the target pipeline.

        Returns:
            Path: Filesystem path to the downloaded media file.
        """
        job_dir = combo_work_dir / video.video_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def _do_download() -> Path:
            downloaded = self.youtube_client.download_video(
                video_id=video.video_id,
                source_ref=video.source_ref,
                destination_dir=job_dir,
            )
            if downloaded.uploader and downloaded.uploader.lower() != "youtube":
                video.uploader = downloaded.uploader
            if downloaded.channel_name:
                video.channel_name = downloaded.channel_name
            return downloaded.file_path

        with logfire.span("Downloading YouTube Video to {job_dir}: {title}", job_dir=job_dir.name, title=video.title):
            return await asyncio.to_thread(_do_download)

    def _get_pipeline_queue(self, pipeline: ComboPipeline) -> dict[str, Any]:
        """Gathers outstanding queued videos for a given pipeline with total count and top 50 items.

        Args:
            pipeline: Combination pipeline instance.

        Returns:
            dict[str, Any]: Dictionary containing 'total_count' and top 50 'items'.
        """
        candidates = list(self.local_videos.values()) + list(self.youtube_videos.values())
        outstanding = []
        now = time()

        for v in candidates:
            if pipeline.active_video and pipeline.active_video.video_id == v.video_id:
                continue

            if pipeline.preparing_video_info and pipeline.preparing_video_info.get("video_id") == v.video_id:
                continue

            if v.uploader.lower() == "youtube":
                if known := self.git_sync.get_completed_uploader(v.video_id, pipeline.full_shortcode):
                    v.uploader = known

            if not self.git_sync.is_video_completed(v.video_id, pipeline.full_shortcode):
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
        return {
            "total_count": len(outstanding),
            "items": outstanding[:50],
        }

    def get_all_pipeline_queues(self) -> dict[str, dict[str, Any]]:
        """Computes outstanding queues for all active pipelines without network calls.

        Returns:
            dict[str, dict[str, Any]]: Mapping of pipeline names to their queue dictionaries.
        """
        return {name: self._get_pipeline_queue(pipe) for name, pipe in self.coordinator.pipelines.items()}

    async def feed_idle_pipelines(self) -> None:
        """Assigns the newest available video to each idle pipeline."""
        if self._stopped:
            return

        async with self._feed_lock:
            idle_pipelines = [p for p in self.coordinator.pipelines.values() if p.is_idle()]
            if not idle_pipelines:
                return

            if not self.local_videos and not self.youtube_videos:
                await self._scan_all_sources(deep_scan=False)

            now = time()

            for pipe in idle_pipelines:
                candidates: list[QueuedVideo] = list(self.local_videos.values())
                if not self._is_youtube_in_cooldown():
                    candidates.extend(self.youtube_videos.values())

                completed_in_git = 0
                in_cooldown = 0
                eligible: list[QueuedVideo] = []

                for v in candidates:
                    if v.uploader.lower() == "youtube":
                        if known := self.git_sync.get_completed_uploader(v.video_id, pipe.full_shortcode):
                            v.uploader = known

                    if self.git_sync.is_video_completed(v.video_id, pipe.full_shortcode):
                        completed_in_git += 1
                    elif v.retry_after is not None and now < v.retry_after:
                        in_cooldown += 1
                    else:
                        eligible.append(v)

                eligible.sort(key=lambda v: (-v.timestamp, v.title.lower()))

                if not eligible:
                    if candidates:
                        reason_parts: list[str] = []
                        if completed_in_git:
                            reason_parts.append(f"{completed_in_git} completed in Git")
                        if in_cooldown:
                            reason_parts.append(f"{in_cooldown} in cooldown")
                        reason_str = ", ".join(reason_parts) if reason_parts else "none available"
                        summary_key = f"{len(candidates)}:{reason_str}"

                        msg = (
                            f"Pipeline '{pipe.config.name}' is idle: 0 pending videos "
                            f"({len(candidates)} tracked: {reason_str})."
                        )
                        if self._last_idle_summary.get(pipe.config.name) != summary_key:
                            self._last_idle_summary[pipe.config.name] = summary_key
                            logfire.info(msg)
                        else:
                            logfire.debug(msg)
                    continue

                self._last_idle_summary.pop(pipe.config.name, None)

                consecutive_download_failures = 0
                max_failures = self.config.youtube.max_download_failures
                fed = False

                for video in eligible:
                    if self._stopped:
                        return

                    if video.source_type == "local":
                        if not video.local_path or not video.local_path.is_file():
                            video.status = "failed"
                            video.error_reason = "File missing from disk"
                            video.retry_after = now + 120
                            continue

                        pipe.set_preparing(
                            video_id=video.video_id,
                            title=video.title,
                            uploader=video.uploader,
                            channel_name=video.channel_name,
                            source_type="local",
                        )
                        try:
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
                            fed = True
                            break
                        except Exception as e:
                            pipe.clear_preparing()
                            shutil.rmtree(pipe.work_dir / video.video_id, ignore_errors=True)
                            logfire.warning(
                                f"Failed to prepare local video '{video.title}' for '{pipe.config.name}': {e}"
                            )
                            video.status = "failed"
                            video.error_reason = f"Preparation error: {e}"
                            video.retry_after = now + 120
                            continue

                    pipe.set_preparing(
                        video_id=video.video_id,
                        title=video.title,
                        uploader=video.uploader,
                        channel_name=video.channel_name,
                        source_type="youtube",
                    )

                    try:
                        media_path = await self._download_youtube_to_combo_dir(video, pipe.work_dir)
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
                        fed = True
                        break

                    except YouTubeVideoUnavailableError as e:
                        pipe.clear_preparing()
                        shutil.rmtree(pipe.work_dir / video.video_id, ignore_errors=True)
                        video.status = "unplayable"
                        video.error_reason = e.reason
                        video.retry_after = now + (24 * 3600)
                        video.failure_count += 1
                        logfire.warning(f"Video '{video.title}' is unplayable ({e.reason}). Retrying in 24h.")
                        continue

                    except YouTubeBotDetectionError as e:
                        pipe.clear_preparing()
                        shutil.rmtree(pipe.work_dir / video.video_id, ignore_errors=True)
                        self._trigger_youtube_cooldown(reason=str(e))
                        break

                    except Exception as e:
                        pipe.clear_preparing()
                        err_msg = str(e)
                        shutil.rmtree(pipe.work_dir / video.video_id, ignore_errors=True)
                        video.status = "failed"
                        video.error_reason = "Download failed"
                        video.retry_after = now + (30 * 60)
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

                if not fed:
                    pipe.clear_preparing()

    async def start(self) -> None:
        """Starts the ingestion manager with an initial scan, followed by periodic delta intake loops."""
        self._stopped = False
        self._loop = asyncio.get_running_loop()
        await self._scan_watch_folders_async()
        await self._scan_youtube_sources_async(deep_scan=True)
        await self.feed_idle_pipelines()

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
        interval_sec = max(5, self.config.files.rescan_interval_minutes * 60)
        while not self._stopped:
            await asyncio.sleep(interval_sec)
            if self._stopped:
                break
            try:
                logfire.debug("Running periodic watch folder scan...")
                await self._scan_watch_folders_async()
                await self.coordinator.check_and_finalize_completed_pipelines()
                await self.feed_idle_pipelines()
            except Exception as e:
                logfire.warning(f"Error during periodic watch folder scan: {e}")

    async def _periodic_youtube_loop(self) -> None:
        """Periodically scans YouTube sources using symmetric jittered intervals."""
        while not self._stopped:
            base_sec = self.config.youtube.rescan_interval_minutes * 60.0
            jitter_sec = max(0.0, self.config.youtube.rescan_jitter_minutes * 60.0)
            delay_sec = max(5.0, base_sec + random.uniform(-jitter_sec, jitter_sec))

            logfire.debug(f"Next YouTube delta rescan scheduled in {delay_sec / 60:.1f} minutes ")
            await asyncio.sleep(delay_sec)
            if self._stopped:
                break

            try:
                logfire.debug("Running periodic YouTube delta scan...")
                await self._scan_youtube_sources_async(deep_scan=False)
                await self.coordinator.check_and_finalize_completed_pipelines()
                await self.feed_idle_pipelines()
            except Exception as e:
                logfire.warning(f"Error during periodic YouTube rescan: {e}")
