import base64
import hashlib
import os
from pathlib import Path
from threading import Lock
from time import sleep, time
from typing import Optional

import logfire
from google.genai.types import (
    File,
    FileState,
    ListFilesConfig,
    UploadFileConfig,
)
from pydantic_ai.providers.google import GoogleProvider


def calculate_sha256(filename: Path):
    """
    Calculates the SHA256 hash of a file, reading it in chunks.

    Args:
        filename (Path): The path to the file.

    Returns:
        str: The hexadecimal representation of the SHA256 hash.
    """
    # Create a sha256 hash object
    h = hashlib.sha256()

    with logfire.span(f"Calculating sha256 of {filename.name}", _level="debug"):
        # Open the file in binary mode ('rb')
        with open(filename, "rb") as file:
            # Read the file in chunks to avoid memory issues with large files
            # 65536 bytes (64 KB) is a common, efficient block size
            for block in iter(lambda: file.read(65536), b""):
                h.update(block)

    # Return the hexadecimal representation of the digest
    return h.hexdigest()


class GeminiFileUploader:
    """
    Handles uploading files to the Google Gemini Files API, including caching
    file lists to avoid redundant API calls and checking for existing files.
    """

    _provider: GoogleProvider
    _state: dict[str, File]
    _last_update_time: float = 0
    _list_cache_ttl_seconds: int
    _lock: Lock

    def __init__(self, provider: GoogleProvider, cache_ttl_seconds: int = 1) -> None:
        """
        Initializes the GeminiFileUploader.

        Args:
            provider (GoogleProvider): The GoogleProvider instance for API access.
            cache_ttl_seconds (int): The time-to-live in seconds for the file list cache.
        """
        self._provider = provider
        self._state = {}
        self._list_cache_ttl_seconds = cache_ttl_seconds
        self._lock = Lock()

    def _update_file_list(self) -> None:
        """
        Updates the local file list cache from the server, but only if the
        cache is older than the TTL and the rate limit allows it.
        """
        now = time()
        with self._lock:
            if (now - self._last_update_time) > self._list_cache_ttl_seconds:
                # The cache is stale, refresh it.
                self._state = {}
                for file in self._provider.client.files.list(
                    config=ListFilesConfig(page_size=100)
                ):
                    if file.display_name:
                        display_name = str(file.display_name)
                        self._state[display_name] = file
                self._last_update_time = now

    def _get_file(self, display_name: str) -> Optional[File]:
        """
        Retrieves a file from the cached list by its display name.
        Updates the cache if it's stale.

        Args:
            display_name (str): The display name of the file to retrieve.

        Returns:
            Optional[File]: The File object if found, otherwise None.
        """
        self._update_file_list()
        return self._state.get(display_name, None)

    def upload_file(self, file_path: Path) -> File:
        """
        Uploads a file to Gemini Files API. If a file with the same display name
        already exists, it checks for differences (size, SHA256 hash) and
        re-uploads if necessary, deleting the old version.
        """
        display_name = file_path.name

        with logfire.span("Check if the file is already uploaded", _level="debug"):
            file = self._get_file(display_name)
            if file is not None:
                needs_delete = False
                # Compare file sizes - If filesizes are different, it's not the same file.
                file_size = os.path.getsize(file_path)
                if file.size_bytes != file_size:
                    needs_delete = True

                # Compare sha256 hashes to be sure
                if not needs_delete:
                    sha256_hex_string = calculate_sha256(file_path)
                    base64_sha256 = base64.b64encode(
                        sha256_hex_string.encode()
                    ).decode()
                    if base64_sha256 != str(file.sha256_hash):
                        needs_delete = True

                if needs_delete and file.name is not None:
                    logfire.debug("Deleting old file with same name")
                    self._provider.client.files.delete(name=file.name)
                    file = None  # Reset file to trigger upload

        # Upload the file
        if file is None:
            with logfire.span("Uploading File", _level="debug"):
                file = self._provider.client.files.upload(
                    file=file_path,
                    config=UploadFileConfig(display_name=file_path.name),
                )
                # We need to sleep for at least _list_cache_ttl_seconds to make sure that
                # the cache contains the file that we just uploaded
                sleep(self._list_cache_ttl_seconds + 1)

        with logfire.span("Wait for the file to be ready", _level="debug"):
            while file.state != FileState.ACTIVE:
                sleep(1)
                file = self._get_file(display_name)
                if not file:
                    # This shouldn't happen. We literally just uploaded the file above
                    raise RuntimeError(
                        f"Could not retrieve file '{display_name}' after upload."
                    )

        return file
