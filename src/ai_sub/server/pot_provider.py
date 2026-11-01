"""Background YouTube PO token provider management and compilation."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
from pathlib import Path

import logfire
from git import Repo


def is_node_runtime_available() -> bool:
    """Checks if Node.js, npm, and npx runtimes are available in PATH.

    Returns:
        bool: True if all required binaries are found, False otherwise.
    """
    return bool(shutil.which("node") and shutil.which("npm") and shutil.which("npx"))


def ensure_pot_provider(pot_dir: Path, check_updates: bool = False) -> Path | None:
    """Clones, syncs tags, and compiles bgutil-ytdlp-pot-provider.

    Args:
        pot_dir: Directory where the PO token provider repository is located.
        check_updates: Whether to check GitHub for updates if unpinned. Defaults to False.

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
