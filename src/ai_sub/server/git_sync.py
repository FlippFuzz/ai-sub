"""Git repository synchronization and subtitle finalization for ai-sub-server."""

from __future__ import annotations

import os
import random
import shutil
import stat
import time
from pathlib import Path
from typing import Any

import logfire
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from ai_sub.server.config import GitConfig

# Prevent interactive prompts in headless server environments
os.environ["GIT_TERMINAL_PROMPT"] = "0"


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
    """Manages cloning, updating, and pushing completed subtitles to Git."""

    config: GitConfig | None
    local_repo_dir: Path

    def __init__(self, config: GitConfig | None, base_work_dir: Path) -> None:
        """Initializes the Git synchronization manager.

        Args:
            config: Git configuration model or None.
            base_work_dir: Base directory where the local repository will be cloned.
        """
        self.config = config
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

    def ensure_repo(self) -> None:
        """Clones or updates the local Git repository checkout."""
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

    def is_video_completed(self, uploader: str, video_id: str, full_shortcode: str) -> bool:
        """Checks if completed subtitles for a video and shortcode exist in the repository.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model shortcode (e.g. 'g38f-011019').

        Returns:
            bool: True if completed subtitles exist; False otherwise.
        """
        video_dir = self.local_repo_dir / uploader / video_id
        if not video_dir.exists():
            return False

        for srt_path in video_dir.glob(f"*.{full_shortcode}.srt"):
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    header = f.read(4096)
                    if '"complete": true' in header:
                        return True
            except Exception:
                pass
        return False

    def finalize_and_push(
        self,
        uploader: str,
        video_id: str,
        full_shortcode: str,
        srt_file: Path,
        yaml_files: list[Path],
    ) -> None:
        """Moves subtitle files and YAML traces into the repository, then commits and pushes.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model shortcode.
            srt_file: Path to the generated SRT file.
            yaml_files: List of intermediate YAML state files.
        """
        video_dir = self.local_repo_dir / uploader / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        target_srt = video_dir / srt_file.name
        shutil.copy2(srt_file, target_srt)
        logfire.info(f"Copied finalized subtitles to {target_srt}")

        int_dir = video_dir / ".int" / full_shortcode
        int_dir.mkdir(parents=True, exist_ok=True)
        for yf in yaml_files:
            shutil.copy2(yf, int_dir / yf.name)

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
                    logfire.warning(f"Git push failed ({e}). Retrying with pull --rebase...")
                    time.sleep(random.randint(1, 15))
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
