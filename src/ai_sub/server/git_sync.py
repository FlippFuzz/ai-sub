"""Git repository synchronization, directory layout, and channel README indexing for ai-sub-server."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.parse
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

import logfire
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from ai_sub.config import atomic_write_text
from ai_sub.server.server_config import GitConfig
from ai_sub.shortcode import parse_subtitle_track, subtitle_track_sort_key

os.environ["GIT_TERMINAL_PROMPT"] = "0"
JST = timezone(timedelta(hours=9))
_MAX_REBASE_STEPS = 50

_T = TypeVar("_T")
_CompletedCache = dict[tuple[str, str], dict[str, bool]]
_CacheUpdate = _CompletedCache | tuple[str, str, str, bool] | None


def redact_url_credentials(url: str) -> str:
    """Masks userinfo credentials embedded in an HTTP/HTTPS URL while preserving host and path.

    Args:
        url: Raw URL string.

    Returns:
        str: URL string without embedded username or password.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return re.sub(r"(https?://)[^/@\s]+@", r"\1***:***@", url)
    if parsed.username is not None or parsed.password is not None:
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    return url


def _decode_git_stream(stream: Any) -> str:
    """Decodes a GitPython stdout/stderr attribute into a clean single string.

    Args:
        stream: Raw bytes, string, or None from a GitCommandError.

    Returns:
        str: Trimmed text representation or empty string.
    """
    if not stream:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace").strip()
    return str(stream).strip()


class GitSyncManager:
    """Manages cloning, updating, and pushing completed subtitles to Git with mutex concurrency control."""

    config: GitConfig | None
    local_repo_dir: Path
    player_base_url: str
    _lock: asyncio.Lock
    _completed: _CompletedCache
    _pending_cache_rescan: bool

    def __init__(self, config: GitConfig | None, base_work_dir: Path) -> None:
        """Initializes the Git synchronization manager.

        Args:
            config: Git configuration model or None.
            base_work_dir: Base directory where the local repository will be cloned.
        """
        self.config = config
        self._lock = asyncio.Lock()
        self._completed = {}
        self._pending_cache_rescan = False
        repo_name = "subtitles-archive"

        if config and config.repourl is not None:
            url_str = str(config.repourl).rstrip("/")
            raw_name = url_str.split("/")[-1]
            repo_name = raw_name[:-4] if raw_name.endswith(".git") else raw_name

        if config:
            self.player_base_url = str(config.playerurl).rstrip("/")
        else:
            default_url = GitConfig.model_fields["playerurl"].default
            self.player_base_url = str(default_url).rstrip("/")

        self.local_repo_dir = base_work_dir / "git_repo" / repo_name

    def _sanitize_text(self, text: str) -> str:
        """Masks any configured Personal Access Token and URL userinfo inside log or error text.

        Args:
            text: Raw text string.

        Returns:
            str: Sanitized string safe for logs and exceptions.
        """
        if self.config:
            pat = self.config.get_pat_value().strip()
            if pat:
                text = text.replace(pat, "***")
        return re.sub(r"(https?://)[^/@\s]+@", r"\1***:***@", text)

    def _sanitize_error(self, err: Exception) -> str:
        """Formats an exception including sanitized Git command stdout/stderr details.

        Args:
            err: Raised exception instance.

        Returns:
            str: Sanitized diagnostic error string.
        """
        if isinstance(err, GitCommandError):
            parts: list[str] = [f"status={err.status}"]
            if stderr := _decode_git_stream(err.stderr):
                parts.append(f"stderr={stderr}")
            if stdout := _decode_git_stream(err.stdout):
                parts.append(f"stdout={stdout}")
            if len(parts) == 1:
                parts.append(str(err))
            return self._sanitize_text(" | ".join(parts))
        return self._sanitize_text(str(err))

    def _get_authenticated_url(self) -> str:
        """Constructs a Git remote URL with embedded authentication tokens.

        Returns:
            str: Authenticated Git URL string, or raw URL if PAT is empty.
        """
        if not self.config or self.config.repourl is None:
            return ""

        raw_url = str(self.config.repourl)
        pat = self.config.get_pat_value().strip()

        # Preserve the raw URL if no PAT is configured to allow anonymous clones
        if not pat:
            return raw_url

        auth_part = pat if ":" in pat else f"oauth2:{pat}"
        if raw_url.startswith("https://"):
            return raw_url.replace("https://", f"https://{auth_part}@", 1)

        return raw_url

    def _configure_repo(self, repo: Repo) -> None:
        """Ensures git author identity and authenticated origin URL are configured.

        Args:
            repo: Active GitPython Repo instance.
        """
        with repo.config_writer() as config:
            config.set_value("user", "name", "ai-sub-server")
            config.set_value("user", "email", "server@ai-sub")

        auth_url = self._get_authenticated_url()
        if auth_url:
            if "origin" in [r.name for r in repo.remotes]:
                repo.remotes.origin.set_url(auth_url)
            else:
                repo.create_remote("origin", auth_url)

    def _has_head(self, repo: Repo) -> bool:
        """Checks whether HEAD resolves to a valid commit.

        Args:
            repo: Active GitPython Repo instance.

        Returns:
            bool: True if at least one commit exists on HEAD; False on an unborn repository.
        """
        try:
            return bool(repo.head.is_valid())
        except Exception:
            return False

    def _remote_branch_exists(self, repo: Repo, branch: str) -> bool:
        """Checks whether origin/<branch> exists in the local remote-tracking refs.

        Args:
            repo: Active GitPython Repo instance.
            branch: Branch name to verify.

        Returns:
            bool: True if refs/remotes/origin/<branch> exists; False otherwise.
        """
        try:
            repo.git.rev_parse("--verify", f"refs/remotes/origin/{branch}")
            return True
        except GitCommandError:
            return False

    def _local_branch_exists(self, repo: Repo, branch: str) -> bool:
        """Checks whether refs/heads/<branch> already exists locally.

        Args:
            repo: Active GitPython Repo instance.
            branch: Branch name to verify.

        Returns:
            bool: True if the local branch exists; False otherwise.
        """
        try:
            repo.git.rev_parse("--verify", f"refs/heads/{branch}")
            return True
        except GitCommandError:
            return False

    def _ensure_head_on_branch(self, repo: Repo, branch: str) -> None:
        """Points an unborn HEAD at <branch>, checks out an existing <branch>, or creates it if absent.

        Never overwrites an existing branch tip with 'checkout -B'.

        Args:
            repo: Active GitPython Repo instance.
            branch: Target branch name.
        """
        if not self._has_head(repo):
            repo.git.symbolic_ref("HEAD", f"refs/heads/{branch}")
            return
        try:
            current = repo.active_branch.name
        except TypeError:
            current = ""
        if current != branch:
            if self._local_branch_exists(repo, branch):
                repo.git.checkout(branch)
            else:
                repo.git.checkout("-b", branch)

    def _has_dirty_worktree(self, repo: Repo) -> bool:
        """Checks whether any staged, modified, or untracked files exist in the working tree.

        Args:
            repo: Active GitPython Repo instance.

        Returns:
            bool: True if git status reports changes; False if working tree is clean.
        """
        return bool(repo.git.status("--porcelain").strip())

    def _reset_dirty_worktree(self, repo: Repo, warning_message: str) -> None:
        """Aborts any interrupted rebase and discards uncommitted working tree changes.

        Args:
            repo: Active GitPython Repo instance.
            warning_message: Diagnostic warning logged when dirty state is found.
        """
        git_dir = Path(repo.git_dir)
        if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
            logfire.warning(warning_message)
            try:
                repo.git.rebase("--abort")
            except GitCommandError:
                pass
        if self._has_dirty_worktree(repo):
            logfire.warning(warning_message)
            if self._has_head(repo):
                repo.git.reset("--hard", "HEAD")
            repo.git.clean("-fd")

    def _has_unpushed_commits(self, repo: Repo, branch: str) -> tuple[bool, int]:
        """Checks whether local HEAD contains commits not yet present on origin/<branch>.

        Args:
            repo: Active GitPython Repo instance.
            branch: Target remote branch name.

        Returns:
            tuple[bool, int]: (has_unpushed_commits, count).
        """
        if not self._has_head(repo):
            return False, 0
        if not self._remote_branch_exists(repo, branch):
            count_str = repo.git.rev_list("HEAD", count=True).strip()
            count = int(count_str) if count_str else 1
            return count > 0, count
        count_str = repo.git.rev_list(f"origin/{branch}..HEAD", count=True).strip()
        count = int(count_str) if count_str else 0
        return count > 0, count

    def _pull_rebase_with_readme_resolution(self, repo: Repo, branch: str) -> bool:
        """Pulls and rebases onto origin/<branch>, auto-resolving channel README.md conflicts.

        Args:
            repo: Active GitPython Repo instance.
            branch: Target remote branch name.

        Returns:
            bool: True if pull/rebase completed cleanly.

        Raises:
            GitCommandError: If pulling fails without starting a rebase.
            RuntimeError: If rebase fails to advance, exceeds max steps, or hits a non-README conflict.
        """
        editor_cmd = f'"{sys.executable}" -c "import sys; sys.exit(0)"'
        rebase_merge_dir = Path(repo.git_dir) / "rebase-merge"
        rebase_apply_dir = Path(repo.git_dir) / "rebase-apply"

        def _rebase_marker() -> str | None:
            for marker_file in (rebase_merge_dir / "msgnum", rebase_apply_dir / "next"):
                if marker_file.is_file():
                    try:
                        return f"{marker_file.parent.name}:{marker_file.read_text(encoding='utf-8').strip()}"
                    except OSError:
                        pass
            return None

        try:
            repo.git.pull("origin", branch, "--rebase")
            return True
        except GitCommandError as pull_err:
            rebase_in_progress = rebase_merge_dir.exists() or rebase_apply_dir.exists()
            if not rebase_in_progress:
                logfire.warning(f"Git pull --rebase failed without active rebase: {self._sanitize_error(pull_err)}")
                raise

            try:
                steps = 0
                while rebase_merge_dir.exists() or rebase_apply_dir.exists():
                    steps += 1
                    if steps > _MAX_REBASE_STEPS:
                        repo.git.rebase("--abort")
                        raise RuntimeError(
                            f"Git rebase aborted after exceeding {_MAX_REBASE_STEPS} conflict resolution steps."
                        ) from pull_err

                    unmerged_output = repo.git.diff("-z", "--diff-filter=U", "--name-only")
                    unmerged_files = [f for f in unmerged_output.split("\0") if f]

                    # Resolve conflicts dynamically if the single conflicted file is an uploader README.md
                    if (
                        len(unmerged_files) == 1
                        and unmerged_files[0].endswith("/README.md")
                        and unmerged_files[0].count("/") == 1
                    ):
                        conflicted_readme = unmerged_files[0]
                        conflicted_uploader = conflicted_readme.split("/")[0]
                        self._update_channel_readme_sync(conflicted_uploader)
                        repo.git.add(conflicted_readme)
                        prev_marker = _rebase_marker()
                        try:
                            repo.git.rebase("--continue", env={**os.environ, "GIT_EDITOR": editor_cmd})
                            logfire.info(f"Continued rebase after resolving {conflicted_readme} conflict.")
                        except GitCommandError as cont_err:
                            if rebase_merge_dir.exists() or rebase_apply_dir.exists():
                                new_marker = _rebase_marker()
                                if new_marker is not None and new_marker != prev_marker:
                                    continue
                            raise RuntimeError(
                                f"Git rebase --continue failed: {self._sanitize_error(cont_err)}"
                            ) from cont_err
                    else:
                        logfire.warning(
                            f"Unresolvable non-README conflict encountered ({unmerged_files}). Aborting rebase."
                        )
                        repo.git.rebase("--abort")
                        raise RuntimeError(
                            f"Git rebase aborted due to unresolvable conflict in files: {unmerged_files}"
                        ) from pull_err

                logfire.info("Resolved all channel README conflicts and completed rebase.")
                return True
            except Exception as rebase_err:
                if rebase_merge_dir.exists() or rebase_apply_dir.exists():
                    try:
                        repo.git.rebase("--abort")
                    except Exception:
                        pass
                logfire.error(f"Failed to resolve git rebase: {self._sanitize_error(rebase_err)}")
                raise

    def _apply_cache_update(self, update: _CacheUpdate) -> None:
        """Applies a thread-produced completed-cache update strictly on the event loop.

        Args:
            update: Full scanned cache dictionary, single entry tuple, or None if scan failed.
        """
        if update is None:
            self._pending_cache_rescan = True
            return
        if isinstance(update, dict):
            self._completed = update
            self._pending_cache_rescan = False
        else:
            video_id, full_shortcode, uploader, is_yt = update
            self._completed.setdefault((video_id, full_shortcode), {})[uploader] = is_yt

    async def _run_locked_thread(
        self,
        fn: Callable[..., _T],
        *args: Any,
        on_result: Callable[[_T], None] | None = None,
    ) -> _T:
        """Executes a blocking Git operation on a thread while holding _lock until the thread finishes.

        Args:
            fn: Synchronous callable to run in a worker thread.
            *args: Positional arguments forwarded to fn.
            on_result: Optional event-loop callback invoked when fn completes successfully.

        Returns:
            _T: Return value of fn.

        Raises:
            asyncio.CancelledError: If the caller was cancelled while awaiting thread completion.
        """
        async with self._lock:
            fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))
            cancelled = False
            while not fut.done():
                try:
                    await asyncio.shield(fut)
                except asyncio.CancelledError:
                    cancelled = True
            result = fut.result()
            if on_result is not None:
                on_result(result)
            if cancelled:
                raise asyncio.CancelledError
            return result

    def _scan_completed_cache_sync(self) -> _CompletedCache | None:
        """Scans the local repository directory and returns a fresh completed subtitles mapping.

        Returns:
            _CompletedCache | None: Completed cache mapping, or None if a scan-level error prevented
            completing the full tree walk.
        """
        completed: _CompletedCache = {}
        if not self.local_repo_dir.is_dir():
            return completed

        try:
            for uploader_dir in sorted(self.local_repo_dir.iterdir(), key=lambda p: p.name):
                if not uploader_dir.is_dir() or uploader_dir.name.startswith("."):
                    continue
                v_dir = uploader_dir / "v"
                if not v_dir.is_dir():
                    continue
                for video_dir in sorted(v_dir.iterdir(), key=lambda p: p.name):
                    if not video_dir.is_dir() or video_dir.name.startswith("."):
                        continue
                    uploader = uploader_dir.name
                    video_id = video_dir.name

                    meta_file = video_dir / "video_meta.json"
                    is_yt = True
                    if meta_file.is_file():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta_data = json.load(f)
                            if isinstance(meta_data, dict) and "source_type" in meta_data:
                                is_yt = meta_data.get("source_type") == "youtube"
                        except Exception:
                            pass

                    for srt in sorted(video_dir.glob("*.srt"), key=lambda p: p.name):
                        try:
                            if not srt.is_file() or srt.stat().st_size <= 0:
                                continue
                        except OSError:
                            continue
                        track_info = parse_subtitle_track(srt)
                        if track_info.shortcode and track_info.is_refined:
                            completed.setdefault((video_id, track_info.shortcode), {})[uploader] = is_yt
        except Exception as e:
            logfire.warning(f"Error rebuilding completed subtitles cache (retaining previous index): {e}")
            return None

        total_entries = sum(len(uploaders) for uploaders in completed.values())
        logfire.debug(f"Rebuilt completed subtitles cache: {total_entries} entry(s).")
        return completed

    def _ensure_repo_sync(self) -> _CompletedCache | None:
        """Synchronous implementation of cloning or updating the local repository.

        Returns:
            _CompletedCache | None: Freshly scanned completed subtitles cache, or None on scan failure.

        Raises:
            RuntimeError: If local_repo_dir is a non-empty non-Git directory or cloning fails.
        """
        if not self.config or self.config.repourl is None:
            return self._scan_completed_cache_sync()

        auth_url = self._get_authenticated_url()
        safe_url = redact_url_credentials(str(self.config.repourl))
        branch = self.config.branch

        if self.local_repo_dir.exists():
            try:
                repo = Repo(self.local_repo_dir)
            except (InvalidGitRepositoryError, NoSuchPathError) as e:
                if any(self.local_repo_dir.iterdir()):
                    msg = (
                        f"Existing directory at {self.local_repo_dir} is non-empty and not a valid git repository "
                        f"({self._sanitize_error(e)}). Refusing to delete existing archive; "
                        "please migrate or remove it manually."
                    )
                    logfire.error(msg)
                    raise RuntimeError(msg) from e
                logfire.info(f"Removing empty non-git directory at {self.local_repo_dir} before cloning.")
                self.local_repo_dir.rmdir()
            else:
                logfire.info(f"Updating existing git repo at {self.local_repo_dir}")
                self._configure_repo(repo)
                try:
                    repo.git.fetch("origin", "--prune")
                except GitCommandError as e:
                    logfire.error(
                        f"Git fetch failed at {self.local_repo_dir}: {self._sanitize_error(e)}. "
                        "Preserving existing local repository."
                    )
                    return self._scan_completed_cache_sync()

                if self._remote_branch_exists(repo, branch):
                    self._reset_dirty_worktree(
                        repo,
                        "Cleaning leftover uncommitted changes in local git repo before syncing.",
                    )
                    self._ensure_head_on_branch(repo, branch)
                    has_unpushed, unpushed_count = self._has_unpushed_commits(repo, branch)
                    if has_unpushed:
                        logfire.warning(
                            f"Preserving {unpushed_count} unpushed local commit(s) on '{branch}'; "
                            f"rebasing onto origin/{branch}..."
                        )
                        try:
                            self._pull_rebase_with_readme_resolution(repo, branch)
                        except Exception as e:
                            logfire.error(f"Startup rebase onto origin/{branch} failed: {self._sanitize_error(e)}")
                    else:
                        repo.git.checkout("-B", branch, f"origin/{branch}")
                        repo.git.reset("--hard", f"origin/{branch}")
                        repo.git.clean("-fd")
                        logfire.info(f"Git repo successfully updated to origin/{branch}.")
                else:
                    self._ensure_head_on_branch(repo, branch)
                    has_unpushed, unpushed_count = self._has_unpushed_commits(repo, branch)
                    if has_unpushed:
                        logfire.info(
                            f"Remote branch origin/{branch} does not exist yet; "
                            f"preserving {unpushed_count} local commit(s)."
                        )
                    else:
                        logfire.info(f"Remote repository is empty; initialized unborn branch '{branch}'.")

                return self._scan_completed_cache_sync()

        self.local_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        logfire.info(f"Cloning {safe_url} to {self.local_repo_dir}")
        try:
            repo = Repo.clone_from(auth_url, self.local_repo_dir)
            self._configure_repo(repo)
            if self._remote_branch_exists(repo, branch):
                repo.git.checkout("-B", branch, f"origin/{branch}")
                logfire.info(f"Git repository successfully cloned on branch '{branch}'.")
            else:
                self._ensure_head_on_branch(repo, branch)
                logfire.info(f"Cloned empty Git repository; initialized unborn branch '{branch}'.")
        except GitCommandError as e:
            sanitized = self._sanitize_error(e)
            logfire.error(f"Failed to clone Git repository '{safe_url}': {sanitized}")
            raise RuntimeError(f"Git clone failed: {sanitized}") from e

        return self._scan_completed_cache_sync()

    async def ensure_repo(self) -> None:
        """Clones or updates the local Git repository checkout under mutex protection."""
        if not self.config or self.config.repourl is None:
            await self._run_locked_thread(self._scan_completed_cache_sync, on_result=self._apply_cache_update)
            return
        await self._run_locked_thread(self._ensure_repo_sync, on_result=self._apply_cache_update)

    def is_video_completed(self, video_id: str, full_shortcode: str, uploader: str | None = None) -> bool:
        """Checks if completed subtitles for a video exist in the repository cache.

        Args:
            video_id: Video identifier.
            full_shortcode: Model lineage shortcode (e.g. 'g35l-g4h-g35l-011910').
            uploader: Optional channel or folder uploader handle. Defaults to None.

        Returns:
            bool: True if completed subtitles exist; False otherwise.
        """
        uploaders = self._completed.get((video_id, full_shortcode))
        if not uploaders:
            return False
        if uploader == "local":
            return uploaders.get(uploader) is False
        if uploader and uploader.lower() != "youtube":
            return uploader in uploaders
        return any(uploaders.values())

    def get_completed_uploader(self, video_id: str, full_shortcode: str) -> str | None:
        """Retrieves the uploader handle for a completed YouTube video ID if known in the repository cache.

        Args:
            video_id: Video identifier.
            full_shortcode: Model lineage shortcode.

        Returns:
            str | None: Uploader handle if present in repository, None otherwise.
        """
        uploaders = self._completed.get((video_id, full_shortcode))
        if not uploaders:
            return None
        yt_uploaders = [u for u, is_yt in uploaders.items() if is_yt]
        return min(yt_uploaders) if yt_uploaders else None

    def _discover_video_subtitles(self, video_dir: Path) -> list[dict[str, str]]:
        """Discovers and ranks all finalized and intermediate SRT files relative to the video directory.

        Tracks are pre-sorted descending by quality (refined > raw, subtitles model, summary model,
        lyrics model, and prompt versions).

        Args:
            video_dir: Target video directory (uploader/v/<video_id>).

        Returns:
            list[dict[str, str]]: Ordered list of dictionaries containing 'name' and relative 'path'.
        """
        discovered_paths: list[Path] = []

        # 1. Primary/Finalized SRTs directly in v/<video_id>/
        for srt in video_dir.glob("*.srt"):
            if srt.is_file() and srt.stat().st_size > 0:
                discovered_paths.append(srt)

        # 2. Raw / intermediate SRTs in .int/
        int_base_dir = video_dir / ".int"
        if int_base_dir.is_dir():
            for srt in int_base_dir.glob("**/*.srt"):
                if srt.is_file() and srt.stat().st_size > 0:
                    discovered_paths.append(srt)

        # Sort descending from highest to lowest quality
        sorted_paths = sorted(discovered_paths, key=subtitle_track_sort_key, reverse=True)

        subtitles: list[dict[str, str]] = []
        for srt_path in sorted_paths:
            track_info = parse_subtitle_track(srt_path)
            subtitles.append(
                {
                    "name": track_info.track_name,
                    "path": srt_path.relative_to(video_dir).as_posix(),
                }
            )

        return subtitles

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

        if not entries:
            readme_path = uploader_dir / "README.md"
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

        clean_player_url = self.player_base_url.rstrip("/")
        encoded_uploader = urllib.parse.quote(uploader, safe="")

        for item in entries:
            clean_title = (
                (item["title"] or item["video_id"])
                .replace("\\", "\\\\")
                .replace("[", "\\[")
                .replace("]", "\\]")
                .replace("|", "\\|")
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
                    f'<a href="{yt_url}">'
                    f'<img src="https://i.ytimg.com/vi/{encoded_vid}/mqdefault.jpg" width="100" /></a>'
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
        readme_path = uploader_dir / "README.md"
        atomic_write_text(readme_path, "\n".join(lines))
        logfire.info(f"Updated channel README at {readme_path.name}")

    def _sync_remote_before_write(self, repo: Repo, branch: str) -> bool:
        """Pulls any upstream remote updates (e.g. WebUI deletions) before copying new artifacts.

        Args:
            repo: Active GitPython Repo instance.
            branch: Configured target branch.

        Returns:
            bool: True if remote commits were pulled into local history, False otherwise.
        """
        self._reset_dirty_worktree(
            repo,
            "Found leftover uncommitted changes in local git repo before finalization; resetting.",
        )

        try:
            old_head = repo.head.commit.hexsha if self._has_head(repo) else None
            repo.git.fetch("origin", "--prune")
            if self._remote_branch_exists(repo, branch):
                self._ensure_head_on_branch(repo, branch)
                self._pull_rebase_with_readme_resolution(repo, branch)
                new_head = repo.head.commit.hexsha if self._has_head(repo) else None
                return new_head != old_head
        except Exception as e:
            logfire.warning(
                f"Could not sync from origin/{branch} before commit ({self._sanitize_error(e)}). "
                "Proceeding with local commit."
            )
        return False

    def _finalize_and_push_sync(
        self,
        uploader: str,
        video_id: str,
        full_shortcode: str,
        srt_files: list[Path],
        yaml_files: list[Path],
        meta_file: Path | None = None,
    ) -> _CacheUpdate:
        """Synchronous implementation of committing and pushing finalized subtitles.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model lineage shortcode.
            srt_files: List of generated subtitle files to commit.
            yaml_files: List of intermediate YAML state files.
            meta_file: Optional video metadata JSON file. Defaults to None.

        Returns:
            _CacheUpdate: Completed cache delta, full scan dictionary, or None if rescan failed.

        Raises:
            RuntimeError: If git staging, commit, rebase, or push fails after exhausting retries.
        """
        git_enabled = bool(self.config and self.config.repourl is not None)
        repo: Repo | None = None

        if git_enabled:
            assert self.config is not None
            branch = self.config.branch
            if not (self.local_repo_dir / ".git").exists():
                initial_cache = self._ensure_repo_sync()
                if initial_cache is None:
                    self._pending_cache_rescan = True
            repo = Repo(self.local_repo_dir)
            self._configure_repo(repo)
            if self._sync_remote_before_write(repo, branch):
                self._pending_cache_rescan = True

        uploader_dir = self.local_repo_dir / uploader
        video_dir = uploader_dir / "v" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        int_dir = video_dir / ".int" / full_shortcode
        int_dir.mkdir(parents=True, exist_ok=True)

        raw_suffix = f".raw-{full_shortcode}.srt"
        for sf in srt_files:
            target_srt = int_dir / sf.name if sf.name.endswith(raw_suffix) else video_dir / sf.name
            shutil.copy2(sf, target_srt)
            logfire.info(f"Copied finalized subtitles to {target_srt}")

        for yf in yaml_files:
            shutil.copy2(yf, int_dir / yf.name)

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
                logfire.warning(f"Failed to read metadata from {meta_file}: {e}")

        clean_meta["subtitles"] = self._discover_video_subtitles(video_dir)

        target_meta = video_dir / "video_meta.json"
        atomic_write_text(target_meta, json.dumps(clean_meta, indent=2, ensure_ascii=False))
        logfire.info(
            f"Saved clean public metadata with {len(clean_meta['subtitles'])} subtitle track(s) to {target_meta.name}"
        )

        self._update_channel_readme_sync(uploader)
        is_yt = clean_meta.get("source_type") == "youtube"
        single_entry: _CacheUpdate = (video_id, full_shortcode, uploader, is_yt)

        if not git_enabled or repo is None or self.config is None:
            return self._scan_completed_cache_sync() if self._pending_cache_rescan else single_entry

        branch = self.config.branch
        try:
            self._ensure_head_on_branch(repo, branch)
            repo.git.add(all=True)

            if self._has_dirty_worktree(repo):
                commit_msg = f"{video_id}.{full_shortcode} subtitled"
                repo.git.commit("-m", commit_msg)
                commit_sha = repo.head.commit.hexsha[:8]
                logfire.info(f"Committed changes ({commit_sha}): '{commit_msg}'")
            else:
                has_unpushed, unpushed_count = self._has_unpushed_commits(repo, branch)
                if has_unpushed:
                    logfire.info(
                        f"No new working tree changes to commit, but found {unpushed_count} "
                        "unpublished local commit(s). Proceeding to push."
                    )
                else:
                    logfire.warning(
                        f"No Git changes detected after staging for {video_id}.{full_shortcode}; skipping push."
                    )
                    return self._scan_completed_cache_sync() if self._pending_cache_rescan else single_entry
        except Exception as e:
            sanitized = self._sanitize_error(e)
            logfire.error(f"Git staging/commit failed for {video_id}.{full_shortcode}: {sanitized}")
            raise RuntimeError(f"Git commit failed: {sanitized}") from e

        for attempt in range(5):
            try:
                repo.git.push("-u", "origin", branch)
                logfire.info(f"Git push succeeded to origin/{branch}.")
                return self._scan_completed_cache_sync() if self._pending_cache_rescan else single_entry
            except GitCommandError as e:
                sanitized = self._sanitize_error(e)
                if attempt >= 4:
                    logfire.error(f"Git push exhausted retries: {sanitized}")
                    raise RuntimeError(f"Git push failed after 5 attempts: {sanitized}") from e

                backoff = random.uniform(1.0, 10.0)
                logfire.warning(
                    f"Git push failed (attempt {attempt + 1}/5: {sanitized}). Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)

                try:
                    repo.git.fetch("origin", "--prune")
                    if self._remote_branch_exists(repo, branch):
                        old_head = repo.head.commit.hexsha if self._has_head(repo) else None
                        self._pull_rebase_with_readme_resolution(repo, branch)
                        new_head = repo.head.commit.hexsha if self._has_head(repo) else None
                        if new_head != old_head:
                            self._pending_cache_rescan = True
                except Exception as pull_err:
                    logfire.warning(f"Git fetch/rebase before push retry failed: {self._sanitize_error(pull_err)}")

        return single_entry

    async def finalize_and_push(
        self,
        uploader: str,
        video_id: str,
        full_shortcode: str,
        srt_files: list[Path],
        yaml_files: list[Path],
        meta_file: Path | None = None,
    ) -> None:
        """Moves subtitle files and YAML traces into the repository, then commits and pushes under lock.

        Args:
            uploader: Channel or uploader identifier.
            video_id: Video identifier.
            full_shortcode: Model lineage shortcode.
            srt_files: List of generated subtitle files to commit.
            yaml_files: List of intermediate YAML state files.
            meta_file: Optional video metadata JSON file. Defaults to None.
        """
        await self._run_locked_thread(
            self._finalize_and_push_sync,
            uploader,
            video_id,
            full_shortcode,
            srt_files,
            yaml_files,
            meta_file,
            on_result=self._apply_cache_update,
        )
