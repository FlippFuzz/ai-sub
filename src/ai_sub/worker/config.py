"""Configuration settings for ai-sub-worker strictly loaded from YAML via pydantic-settings."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Self

from pydantic import Field, HttpUrl, PositiveInt, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from ai_sub.config import SearchSettings


class WorkerSettings(BaseSettings):
    """Central configuration for an ephemeral ai-sub-worker node."""

    model_config = SettingsConfigDict(
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
    models: list[str] = Field(
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
    search: SearchSettings = Field(
        default_factory=SearchSettings,
        description="Web search configuration and provider API keys.",
    )
    quota_file: Path = Field(
        default=Path(".ai-sub-quota.json"),
        description="Path to local quota state JSON tracking Pacific Time resets.",
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        description="Interval in seconds between task heartbeat pings.",
    )
    max_tasks: int = Field(
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
        """Resolves the Gemini API key directly from settings.

        Returns:
            SecretStr | None: Secret key instance if configured, or None.
        """
        return self.gemini_key

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customizes settings source loading order, ignoring environment variables in favor of pure YAML.

        Args:
            settings_cls: BaseSettings class type.
            init_settings: Injected initial settings kwargs.
            env_settings: Environment variable settings source.
            dotenv_settings: Dotenv settings source.
            file_secret_settings: File secrets settings source.

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: Ordered settings sources.
        """
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        """Loads worker settings strictly from a YAML file path via pydantic-settings.

        Args:
            yaml_path: Filesystem path to the worker configuration YAML file.

        Returns:
            Self: Instantiated and validated WorkerSettings object.

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Worker configuration file not found: {path}")

        yaml_source = YamlConfigSettingsSource(cls, yaml_file=path)
        return cls.model_validate(yaml_source())
