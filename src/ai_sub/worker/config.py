"""Configuration settings for ai-sub-worker loaded from YAML with environment overrides."""

from __future__ import annotations

import os
import socket
from pathlib import Path

from pydantic import (
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
)
from pydantic_settings import SettingsConfigDict

from ai_sub.config import (
    EnvPrioritySettings,
    SearchSettings,
    StringList,
    YamlFileSettings,
)


class WorkerSearchSettings(EnvPrioritySettings, SearchSettings):
    """Worker web search configuration with AISUB_WORKER_SEARCH_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_WORKER_SEARCH_",
        extra="ignore",
    )


class WorkerSettings(YamlFileSettings):
    """Central configuration for an ephemeral ai-sub-worker node."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_WORKER_",
        extra="ignore",
    )

    server_url: HttpUrl = Field(
        default=HttpUrl("http://127.0.0.1:8000"),
        description="Base URL of the central ai-sub-server.",
    )
    server_auth: str | None = Field(
        default=None,
        description="Basic Auth credentials in 'username:password' format for Nginx reverse proxy.",
    )
    worker_id: str = Field(
        default_factory=socket.gethostname,
        description="Unique worker node identifier.",
    )
    concurrency: PositiveInt = Field(
        default=2,
        description="Number of concurrent worker task execution slots.",
    )
    models: StringList = Field(
        default_factory=lambda: [
            "google-gla:gemini-3.8-flash",
            "google-gla:gemini-3.6-flash",
            "google-gla:gemini-3.5-flash-lite",
        ],
        description="List of Gemini models offered by this worker.",
    )
    gemini_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key.",
    )
    search: WorkerSearchSettings = Field(
        default_factory=WorkerSearchSettings,
        description="Web search configuration and provider API keys.",
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        description="Interval in seconds between task heartbeat pings.",
    )
    claim_poll_interval_minutes: PositiveFloat = Field(
        default=3.0,
        description="Interval in minutes to wait before polling the server again when active slots are busy.",
    )
    max_duration_minutes: NonNegativeFloat = Field(
        default=0.0,
        description="Maximum execution time in minutes before stopping new claims and draining (0 disables).",
    )
    max_tasks: NonNegativeInt = Field(
        default=0,
        description="Maximum tasks to execute before exiting (0 runs until queue is empty or quotas exhausted).",
    )
    temp_dir: Path = Field(
        default=Path("/tmp/ai_sub_worker"),
        description="Local directory for temporary media chunk downloads.",
    )

    def get_basic_auth_tuple(self) -> tuple[str, str] | None:
        """Parses server_auth string into a username and password tuple.

        Returns:
            tuple[str, str] | None: (username, password) if configured, or None.
        """
        if not self.server_auth or ":" not in self.server_auth:
            return None
        parts = self.server_auth.split(":", 1)
        return parts[0], parts[1]

    def get_gemini_api_key(self) -> SecretStr | None:
        """Resolves the Gemini API key from settings or ambient environment variables.

        Returns:
            SecretStr | None: Secret key instance if configured, or None.
        """
        if self.gemini_key:
            return self.gemini_key
        if env_k := (
            os.getenv("AISUB_WORKER_GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        ):
            return SecretStr(env_k.strip().strip('"').strip("'"))
        return None
