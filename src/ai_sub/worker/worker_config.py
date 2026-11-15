"""Configuration settings for ai-sub-worker loaded from YAML with environment overrides."""

from __future__ import annotations

import socket
from pathlib import Path

from pydantic import (
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from pydantic_settings import SettingsConfigDict

from ai_sub.config import (
    AiSettings,
    LoggingSettings,
    StringList,
    YamlFileSettings,
)


class WorkerSettings(YamlFileSettings):
    """Central configuration for an ephemeral ai-sub-worker node."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_WORKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
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
        description="Number of concurrent execution slots per configured model.",
    )
    models: StringList = Field(
        default_factory=lambda: [
            "google-gla:gemini-3.8-flash",
            "google-gla:gemini-3.6-flash",
            "google-gla:gemini-3.5-flash-lite",
        ],
        description="List of Gemini models offered by this worker.",
    )
    ai: AiSettings = Field(
        default_factory=AiSettings,
        description="AI models, quotas, storage, and search settings.",
    )
    log: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Telemetry and logging controls.",
    )
    heartbeat_interval_seconds: PositiveFloat = Field(
        default=30.0,
        description="Interval in seconds between task heartbeat pings.",
    )
    claim_poll_interval_minutes: PositiveFloat = Field(
        default=3.0,
        description="Interval in minutes between poll attempts when a slot has no tasks.",
    )
    idle_timeout_minutes: NonNegativeFloat = Field(
        default=15.0,
        description="Minutes all slots must remain idle before worker exits (0 exits immediately when idle).",
    )
    max_duration_minutes: NonNegativeFloat = Field(
        default=0.0,
        description="Maximum execution time in minutes before stopping new claims (0 disables).",
    )
    max_tasks: NonNegativeInt = Field(
        default=0,
        description="Maximum tasks to execute per configured model before exiting (0 runs until idle timeout).",
    )
    work_dir: Path = Field(
        default=Path("/tmp/ai_sub_worker"),
        description="Local working directory for temporary media chunk downloads and processing.",
    )

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Parses server_auth string into a username and password tuple.

        Returns:
            tuple[str, str] | None: (username, password) if configured, or None.
        """
        if not self.server_auth or ":" not in self.server_auth:
            return None
        parts = self.server_auth.split(":", 1)
        return parts[0], parts[1]
