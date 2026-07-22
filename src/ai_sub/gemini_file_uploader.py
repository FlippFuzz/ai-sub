"""Module for handling file uploads to the Google Gemini Files API."""

import asyncio
import base64
import hashlib
import os
import re
from pathlib import Path
from time import time
from typing import Optional

import logfire
from google import genai
from google.genai.types import (
    File,
    FileState,
    HttpOptions,
    ListFilesConfig,
    UploadFileConfig,
)

from ai_sub.config import Settings

# Pattern to match strictly lowercase managed segment files (e.g. part_001.mp4)
PART_PATTERN = re.compile(r"^part_\d+")


def calculate_sha256_hex(filename: Path) -> str:
    """Calculates the SHA256 hex digest of a file, reading it in chunks.

    Args:
        filename (Path): The path to the file.

    Returns:
        str: The SHA256 hex digest string.
    """
    h = hashlib.sha256()
    with logfire.span(f"Calculating sha256 of {filename.name}", _level="debug"):
        with open(filename, "rb") as file:
            for block in iter(lambda: file.read(65536), b""):
                h.update(block)

    return h.hexdigest()


def _hashes_match(remote_hash: str | None, local_hex: str) -> bool:
    """Checks if a remote file's SHA256 hash matches a local file's hex digest.

    Handles Base64-encoded hex strings (Gemini API standard), raw hex strings,
    and Base64-encoded binary digests.

    Args:
        remote_hash (str | None): The SHA256 hash string from Gemini Files API.
        local_hex (str): The calculated local file SHA256 hex digest.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    if not remote_hash:
        return False

    # Direct hex match
    if remote_hash.lower() == local_hex.lower():
        return True

    # Gemini API format: Base64-encoded ASCII hex string
    b64_hex = base64.b64encode(local_hex.encode()).decode()
    if remote_hash == b64_hex:
        return True

    # Base64-encoded binary digest format fallback
    try:
        b64_digest = base64.b64encode(bytes.fromhex(local_hex)).decode()
        if remote_hash == b64_digest:
            return True
    except ValueError:
        pass

    # Decoded remote base64 match against local hex
    try:
        decoded_remote = base64.b64decode(remote_hash).decode("utf-8", errors="ignore")
        if decoded_remote.lower() == local_hex.lower():
            return True
    except Exception:
        pass

    return False


def _is_managed_file(display_name: str | None) -> bool:
    """Checks if a file display name matches the managed segment pattern (e.g., 'part_XXX').

    Args:
        display_name (str | None): The display name of the remote file.

    Returns:
        bool: True if the file matches the managed pattern, False otherwise.
    """
    if not display_name:
        return False
    filename = Path(display_name).name
    return bool(PART_PATTERN.match(filename))


def _get_unique_display_name(file_path: Path) -> str:
    """Constructs a unique display name using the workspace folder and filename.

    Args:
        file_path (Path): Path to the local file.

    Returns:
        str: Unique display name formatted as 'workspace_folder/part_XXX.ext'.
    """
    folder_name = file_path.parent.parent.name if file_path.parent.name == "reencoded" else file_path.parent.name
    return f"{folder_name}/{file_path.name}"


class GeminiFileUploader:
    """Handles uploading files to the Google Gemini Files API.

    Includes caching file lists to avoid redundant API calls, checking for existing
    files across multiple video workspaces using size/hash, and freeing storage only
    for managed 'part_XXX' files when approaching the configured max storage limit.
    """

    _client: genai.Client
    _state: dict[str, File]
    _last_update_time: float = 0
    _list_cache_ttl_seconds: int
    _max_storage_bytes: int
    _lock: asyncio.Lock

    def __init__(self, settings: Settings) -> None:
        """Initializes the GeminiFileUploader instance.

        Args:
            settings (Settings): The application configuration settings.
        """
        http_options: Optional[HttpOptions] = None
        if settings.ai.google.base_url:
            http_options = HttpOptions(base_url=str(settings.ai.google.base_url))

        self._client = genai.Client(
            api_key=(settings.ai.google.key.get_secret_value() if settings.ai.google.key else None),
            http_options=http_options,
        )
        self._list_cache_ttl_seconds = settings.ai.google.file_cache_ttl
        self._max_storage_bytes = settings.ai.google.max_storage_bytes
        self._state = {}
        self._lock = asyncio.Lock()

    async def _update_file_list(self) -> None:
        """Updates the local file list cache from the server if the cache is stale."""
        now = time()
        async with self._lock:
            if (now - self._last_update_time) > self._list_cache_ttl_seconds:
                new_state: dict[str, File] = {}
                async for file in await self._client.aio.files.list(config=ListFilesConfig(page_size=100)):
                    if file.name:
                        new_state[file.name] = file
                self._state = new_state
                self._last_update_time = now

    async def _cleanup_storage_if_needed(self, required_bytes: int = 0) -> None:
        """Deletes oldest uploaded managed files if total remote storage exceeds safety limit.

        Only files with display names matching the managed pattern ('part_XXX') are
        considered candidates for deletion.

        Args:
            required_bytes (int): The size in bytes of the incoming file to be uploaded.
        """
        await self._update_file_list()
        total_bytes = sum(f.size_bytes or 0 for f in self._state.values())

        if total_bytes + required_bytes <= self._max_storage_bytes:
            return

        logfire.info(
            f"Gemini storage threshold reached ({total_bytes / (1024**3):.2f} GB used). "
            "Cleaning up oldest managed files..."
        )

        # Filter candidates to ONLY files matching our managed segment pattern (e.g. part_XXX)
        managed_files = [f for f in self._state.values() if _is_managed_file(f.display_name)]

        # Sort managed files by expiration or creation timestamp (oldest first)
        sorted_files = sorted(
            managed_files,
            key=lambda f: f.expiration_time or f.create_time or "",
        )

        async with self._lock:
            for file in sorted_files:
                if total_bytes + required_bytes <= self._max_storage_bytes:
                    break
                if file.name:
                    try:
                        logfire.debug(f"Deleting old managed file: {file.display_name} ({file.name})")
                        await self._client.aio.files.delete(name=file.name)
                        total_bytes -= file.size_bytes or 0
                        self._state.pop(file.name, None)
                    except Exception as e:
                        logfire.warning(f"Failed to delete remote file {file.name}: {e}")

            self._last_update_time = 0

    async def _find_existing_file(self, file_path: Path, file_size: int, local_hex: str) -> Optional[File]:
        """Finds an existing remote file matching display name, size, and SHA256 hash.

        Args:
            file_path (Path): Path to the local file.
            file_size (int): Size of the local file in bytes.
            local_hex (str): Calculated SHA256 hex digest of the local file.

        Returns:
            Optional[File]: The matching File object if found, otherwise None.
        """
        await self._update_file_list()

        target_display_names = {
            file_path.name,
            _get_unique_display_name(file_path),
        }

        async with self._lock:
            for file in self._state.values():
                if (
                    file.display_name in target_display_names
                    and file.size_bytes == file_size
                    and _hashes_match(file.sha256_hash, local_hex)
                ):
                    if file.state == FileState.FAILED:
                        if file.name:
                            logfire.debug(f"Deleting failed remote file: {file.display_name}")
                            try:
                                await self._client.aio.files.delete(name=file.name)
                            except Exception:
                                pass
                            self._state.pop(file.name, None)
                        continue

                    return file

        return None

    async def upload_file(self, file_path: Path) -> File:
        """Uploads a file to the Gemini Files API.

        If an active remote file matches the size, SHA256 hash, and display name,
        it reuses that file without re-uploading or deleting non-matching files.
        Older managed ('part_XXX') files are deleted only when total remote storage
        approaches the configured limit.

        Args:
            file_path (Path): The path to the local file to be uploaded.

        Returns:
            File: The uploaded or existing file metadata object.

        Raises:
            RuntimeError: If the file fails to process or cannot be retrieved.
        """
        display_name = _get_unique_display_name(file_path)
        file_size = os.path.getsize(file_path)

        with logfire.span("Check if the file is already uploaded", _level="debug"):
            local_hex = await asyncio.to_thread(calculate_sha256_hex, file_path)
            file = await self._find_existing_file(file_path, file_size, local_hex)

        if file is None:
            await self._cleanup_storage_if_needed(file_size)

            with logfire.span("Uploading File", _level="debug"):
                file = await self._client.aio.files.upload(
                    file=file_path,
                    config=UploadFileConfig(display_name=display_name),
                )
                async with self._lock:
                    self._last_update_time = 0

                await asyncio.sleep(self._list_cache_ttl_seconds + 1)

        with logfire.span("Wait for the file to be ready", _level="debug"):
            while file.state != FileState.ACTIVE:
                if file.state == FileState.FAILED:
                    raise RuntimeError(f"File {file_path.name} failed to process on the server.")

                await asyncio.sleep(1)
                if file.name:
                    file = await self._client.aio.files.get(name=file.name)
                else:
                    file = await self._find_existing_file(file_path, file_size, local_hex)

                if not file:
                    raise RuntimeError(f"Could not retrieve file '{file_path.name}' after upload.")

        return file
