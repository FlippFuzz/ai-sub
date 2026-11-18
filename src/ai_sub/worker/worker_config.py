"""Configuration settings for ai-sub-worker loaded from YAML with environment overrides."""

from __future__ import annotations

import hashlib
import socket
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Self

from annotated_types import Ge, Gt
from pydantic import Field, HttpUrl, NonNegativeInt, PositiveInt, model_validator
from pydantic_settings import SettingsConfigDict

from ai_sub.config import (
    AiSettings,
    LoggingSettings,
    StringList,
    YamlFileSettings,
)


def _default_worker_id() -> str:
    """Computes a SHA-256 hash of the machine hostname to avoid leaking hostnames.

    Returns:
        str: Hexadecimal SHA-256 digest of the hostname.
    """
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()


class WorkerSettings(YamlFileSettings):
    """Central configuration for an ephemeral ai-sub-worker node."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUBWORKER_",
        env_nested_delimiter="_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    serverurl: HttpUrl = Field(
        default=HttpUrl("http://127.0.0.1:8000"),
        description="Base URL of the central ai-sub-server.",
    )
    serverauth: str | None = Field(
        default=None,
        description="Basic Auth credentials in 'username:password' format for Nginx reverse proxy.",
    )
    id: str = Field(
        default_factory=_default_worker_id,
        description="Unique worker node identifier (defaults to SHA-256 of hostname).",
    )
    concurrency: PositiveInt = Field(
        default=2,
        description="Number of concurrent execution slots per configured model.",
    )
    models: StringList = Field(
        default_factory=lambda: [
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
        ],
        description="List of Gemini models offered by this worker.",
    )
    ai: AiSettings = Field(
        default_factory=AiSettings,
        description="AI engine quotas, storage, and search settings.",
    )
    log: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Telemetry and logging controls.",
    )
    heartbeatinterval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=30),
        description="Interval between task heartbeat pings (seconds if numeric).",
    )
    pollinterval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(minutes=3),
        description="Interval between poll attempts when a slot has no tasks (seconds if numeric).",
    )
    idletimeout: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(minutes=15),
        description="Duration all slots must remain idle before exiting (0 exits immediately; seconds if numeric).",
    )
    maxduration: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(0),
        description="Maximum execution duration before stopping new claims (0 disables; seconds if numeric).",
    )
    maxtasks: NonNegativeInt = Field(
        default=0,
        description="Maximum tasks to execute per configured model before exiting (0 runs until idle timeout).",
    )
    workdir: Path = Field(
        default=Path("/tmp/ai_sub_worker"),
        description="Local working directory for temporary media chunk downloads and processing.",
    )

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Parses serverauth string into a username and password tuple.

        Returns:
            tuple[str, str] | None: (username, password) if configured, or None.
        """
        if not self.serverauth or ":" not in self.serverauth:
            return None
        parts = self.serverauth.split(":", 1)
        return parts[0], parts[1]

    @model_validator(mode="after")
    def _validate_api_keys(self) -> Self:
        """Ensures required Gemini and search API keys are configured.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If the Gemini API key or a required search API key is missing.
        """
        if self.ai.key is None or not self.ai.key.get_secret_value().strip():
            raise ValueError(
                "A Google Gemini API key must be provided via 'ai.key', "
                "'AISUBWORKER_AI_KEY', or the GEMINI_API_KEY / GOOGLE_API_KEY environment variables."
            )
        self.ai.search.validate_required_keys(env_prefix="AISUBWORKER")
        return self
