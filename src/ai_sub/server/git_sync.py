"""Git repository synchronization, directory layout, and channel README indexing for ai-sub-server."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import stat
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import logfire
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from ai_sub.server.config import GitConfig

os.environ["GIT_TERMINAL_PROMPT"] = "0"
JST = timezone(timedelta(hours=9))


def _handle_readonly_rmtree(func: Any, path: str, exc_info: Any) -> None:
    """Helper to remove read-only files during directory cleanup.

    Args:
        func: The function that failed (e.g. os.unlink).
        path: Path to the read-only file.
        exc_info: Exception information tuple.
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise


class GitSyncManager:
    """Manages cloning, updating, and pushing completed subtitles to Git with mutex concurrency control."""

    config: GitConfig | None
    local_repo_dir: Path
    _lock: asyncio.Lock

    def __init__(self, config: GitConfig | None, base_work_dir: Path) -> None:
        """Initializes the Git synchronization manager.

        Args:
            config: Git configuration model or None.
            base_work_dir: Base directory where the local repository will be cloned.
        """
        self.config = config
        self._lock = asyncio.Lock()
        repo_name = "subtitles-archive"
        if config:
            url_str = str(config.repo_url).rstrip("/")
            raw_name = url_str.split("/")[-1]
            repo_name = raw_name[:-4] if raw_name.endswith(".git") else raw_name

        self.local_repo_dir = base_work_dir / "git_repo" / repo_name

    def _get_authenticated_url(self) -> str:
        """Constructs a Git remote URL with embedded authentication tokens.

        Returns:
            str: Authenticated Git URL string.
        """
        if not self.config:
            return ""
        raw_url = str(self.config.repo_url)
        pat = self.config.get_pat_value()
        if not pat:
            return raw_url

        auth_part = pat if ":" in pat else f"oauth2:{pat}"
        if raw_url.startswith("https://"):
            auth_url = raw_url.replace("https://", f"https://{auth_part}@", 1)
            if auth_url.endswith(".git"):
                auth_url += "/"
            return auth_url
        return raw_url

    def _ensure_repo_sync(self) -> None:
        """Synchronous implementation of cloning or updating the local repository."""
        if not self.config:
            return

        auth_url = self._get_authenticated_url()
        branch = self.config.branch

        if self.local_repo_dir.exists():
            try:
                logfire.info(f"Updating existing git repo at {self.local_repo_dir}")
                repo = Repo(self.local_repo_dir)
                repo.remotes.origin.set_url(auth_url)
                repo.remotes.origin.fetch()
                repo.git.checkout(branch)
                repo.git.reset("--hard", f"origin/{branch}")
                repo.git.clean("-fd")
                logfire.info(f"Git repo successfully updated to origin/{branch}.")
                return
            except (InvalidGitRepositoryError, GitCommandError, AssertionError) as e:
                logfire.warning(f"Error updating repo at {self.local_repo_dir}: {e}. Re-cloning...")
                shutil.rmtree(self.local_repo_dir, onerror=_handle_readonly_rmtree)

        self.local_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        logfire.info(f"Cloning {self.config.repo_url} to {self.local_repo_dir}")
        repo = Repo.clone_from(auth_url, self.local_repo_dir)
        repo.git.checkout(branch)
        logfire.info("Git repository successfully cloned.")

    async def ensure_repo(self) -> None:
        """Clones or updates the local Git repository checkout under mutex protection."""
        if not self.config:
            return
        async with self._lock:
            await asyncio.to_thread(self._ensure_repo_sync)

    def is_video_completed(self, uploader: str, video_id: str, full_shortcode: str) -> bool:
        """Checks if completed subtitles for a video and shortcode exist in the repository.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model shortcode (e.g. 'g38f-011019').

        Returns:
            bool: True if completed subtitles exist; False otherwise.
        """
        video_dir = self.local_repo_dir / uploader / "v" / video_id
        if not video_dir.exists():
            return False

        for srt_path in video_dir.glob(f"*.{full_shortcode}.srt"):
            if not srt_path.name.startswith("part_") and srt_path.stat().st_size > 0:
                return True
        return False

    def _update_channel_readme_sync(self, uploader: str) -> None:
        """Regenerates the channel README.md index by scanning all video_meta.json files in v/.

        Args:
            uploader: Channel or uploader identifier.
        """
        uploader_dir = self.local_repo_dir / uploader
        v_base_dir = uploader_dir / "v"
        if not v_base_dir.is_dir():
            return

        entries: list[dict[str, Any]] = []
        channel_display_name = uploader

        for v_dir in sorted(v_base_dir.iterdir()):
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
            ch_name = meta_data.get("channel_name")

            try:
                timestamp = int(meta_data.get("timestamp") or 0)
            except (ValueError, TypeError):
                timestamp = 0

            if ch_name and channel_display_name == uploader:
                channel_display_name = str(ch_name)

            date_str = "Unknown"
            if timestamp > 0:
                try:
                    dt = datetime.fromtimestamp(timestamp, tz=JST)
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = "Unknown"

            entries.append(
                {
                    "video_id": vid,
                    "title": title,
                    "source_type": source_type,
                    "timestamp": timestamp,
                    "date_str": date_str,
                }
            )

        if not entries:
            return

        # Sort descending by JST timestamp, with video_id as tiebreaker
        entries.sort(key=lambda x: (x["timestamp"], x["video_id"]), reverse=True)

        lines: list[str] = [
            f"# {channel_display_name}",
            "",
            "| Date (JST) | Title | Subtitles |",
            "| :--- | :--- | :--- |",
        ]

        for item in entries:
            clean_title = item["title"].replace("|", "\\|").replace("\n", " ").strip()
            vid = item["video_id"]
            if item["source_type"] == "youtube":
                title_cell = f"[{clean_title}](https://youtu.be/{vid})"
            else:
                title_cell = clean_title
            sub_cell = f"[Browse](./v/{vid})"
            lines.append(f"| {item['date_str']} | {title_cell} | {sub_cell} |")

        lines.append("")
        readme_path = uploader_dir / "README.md"
        readme_path.write_text("\n".join(lines), encoding="utf-8")
        logfire.info(f"Updated channel README at {readme_path.name}")

    def _finalize_and_push_sync(
        self,
        uploader: str,
        video_id: str,
        full_shortcode: str,
        srt_files: Sequence[Path] | Path,
        yaml_files: list[Path],
        meta_file: Path | None = None,
    ) -> None:
        """Synchronous implementation of committing and pushing finalized subtitles."""
        uploader_dir = self.local_repo_dir / uploader
        video_dir = uploader_dir / "v" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        int_dir = video_dir / ".int" / full_shortcode
        int_dir.mkdir(parents=True, exist_ok=True)

        files_list = [srt_files] if isinstance(srt_files, Path) else srt_files
        for sf in files_list:
            target_srt = int_dir / sf.name if ".raw-" in sf.name else video_dir / sf.name
            shutil.copy2(sf, target_srt)
            logfire.info(f"Copied finalized subtitles to {target_srt}")

        for yf in yaml_files:
            shutil.copy2(yf, int_dir / yf.name)

        # Sanitize metadata: extract only safe public fields to prevent leaking local paths
        clean_meta: dict[str, Any] = {
            "video_id": video_id,
            "uploader": uploader,
            "channel_name": None,
            "title": video_id,
            "source_type": "youtube",
            "timestamp": 0,
        }
        if meta_file and meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    clean_meta["channel_name"] = raw.get("channel_name")
                    clean_meta["title"] = str(raw.get("title") or video_id).strip()
                    clean_meta["source_type"] = str(raw.get("source_type") or "youtube")
                    clean_meta["timestamp"] = int(raw.get("timestamp") or 0)
            except Exception as e:
                logfire.warning(f"Failed to sanitize metadata from {meta_file}: {e}")

        target_meta = video_dir / "video_meta.json"
        with open(target_meta, "w", encoding="utf-8") as f:
            json.dump(clean_meta, f, indent=2, ensure_ascii=False)
        logfire.info(f"Saved clean public metadata to {target_meta.name}")

        self._update_channel_readme_sync(uploader)

        if not self.config:
            return

        repo = Repo(self.local_repo_dir)
        with repo.config_writer() as config:
            config.set_value("user", "name", "ai-sub-server")
            config.set_value("user", "email", "server@ai-sub")

        auth_url = self._get_authenticated_url()
        repo.remotes.origin.set_url(auth_url)
        repo.git.add(all=True)

        try:
            has_head = repo.head.is_valid()
        except Exception:
            has_head = False

        if not has_head or repo.index.diff("HEAD"):
            commit_msg = f"{video_id}.{full_shortcode} subtitled"
            repo.index.commit(commit_msg)
            logfire.info(f"Committed changes: '{commit_msg}'")
        else:
            logfire.info("No Git changes to commit.")
            return

        for attempt in range(5):
            try:
                repo.git.push()
                logfire.info("Git push succeeded.")
                break
            except GitCommandError as e:
                if attempt < 4:
                    backoff = random.uniform(1.0, 10.0)
                    logfire.warning(f"Git push failed ({e}). Retrying in {backoff:.1f}s with pull --rebase...")
                    time.sleep(backoff)
                    try:
                        repo.git.pull("origin", self.config.branch, "--rebase")
                    except Exception:
                        try:
                            repo.git.rebase("--abort")
                        except Exception:
                            pass
                else:
                    logfire.error(f"Git push exhausted retries: {e}")
                    raise

    async def finalize_and_push(
        self,
        uploader: str,
        video_id: str,
        full_shortcode: str,
        srt_files: Sequence[Path] | Path,
        yaml_files: list[Path],
        meta_file: Path | None = None,
    ) -> None:
        """Moves subtitle files and YAML traces into the repository, then commits and pushes under lock.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model shortcode.
            srt_files: One or more generated subtitle files to commit.
            yaml_files: List of intermediate YAML state files.
            meta_file: Optional video metadata JSON file. Defaults to None.
        """
        async with self._lock:
            await asyncio.to_thread(
                self._finalize_and_push_sync,
                uploader,
                video_id,
                full_shortcode,
                srt_files,
                yaml_files,
                meta_file,
            )
