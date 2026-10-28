"""Configuration settings for ai-sub-worker loaded from YAML with environment overrides."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self, cast

from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from ai_sub.config import SearchSettings


def _parse_models_list(val: Any) -> list[str]:
    """Parses candidate model list from Python list, JSON array, or comma-separated string.

    Args:
        val: Raw input value from environment variable or YAML.

    Returns:
        list[str]: Normalized list of model identifier strings.
    """
    if isinstance(val, str):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (list, tuple)):
                    return [str(m).strip() for m in parsed if str(m).strip()]
            except Exception:
                pass
        return [m.strip() for m in v.split(",") if m.strip()]
    if isinstance(val, (list, tuple)):
        out: list[str] = []
        for m in val:
            if isinstance(m, str) and "," in m:
                out.extend(sub.strip() for sub in m.split(",") if sub.strip())
            elif isinstance(m, str) and m.strip():
                out.append(m.strip())
        return out
    return val


ModelsList = Annotated[list[str], NoDecode, BeforeValidator(_parse_models_list)]


class _BaseWorkerSettings(BaseSettings):
    """Base settings class for worker sub-components where ENV overrides YAML kwargs."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Ensures environment variables take precedence over kwargs passed from parent YAML.

        Args:
            settings_cls: BaseSettings class type.
            init_settings: Injected initial settings kwargs.
            env_settings: Environment variable settings source.
            dotenv_settings: Dotenv settings source (ignored).
            file_secret_settings: File secrets settings source (ignored).

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: Custom source loading sequence.
        """
        return (env_settings, init_settings)


class WorkerSearchSettings(SearchSettings, _BaseWorkerSettings):
    """Worker web search configuration with AISUB_WORKER_SEARCH_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_WORKER_SEARCH_",
        extra="ignore",
    )


class WorkerSettings(BaseSettings):
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
    models: ModelsList = Field(
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
    quota_file: Path = Field(
        default=Path(".ai-sub-quota.json"),
        description="Path to local quota state JSON tracking Pacific Time resets.",
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customizes settings source loading order, allowing strictly ENV and YAML sources.

        Args:
            settings_cls: BaseSettings class type.
            init_settings: Injected initial settings kwargs.
            env_settings: Environment variable settings source.
            dotenv_settings: Dotenv settings source (ignored).
            file_secret_settings: File secrets settings source (ignored).

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: Ordered settings sources.
        """
        yaml_file = getattr(settings_cls, "_yaml_path", None)
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
            if yaml_file
            else YamlConfigSettingsSource(settings_cls)
        )
        return (
            init_settings,
            env_settings,
            yaml_source,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        """Loads worker settings from a YAML file path with environment variable overrides.

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

        class _LoadedWorkerSettings(cls):
            _yaml_path: ClassVar[Path | None] = path

            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls: type[BaseSettings],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ) -> tuple[PydanticBaseSettingsSource, ...]:
                """Customizes settings source loading order for loaded worker settings.

                Args:
                    settings_cls: BaseSettings class type.
                    init_settings: Injected initial settings kwargs.
                    env_settings: Environment variable settings source.
                    dotenv_settings: Dotenv settings source (ignored).
                    file_secret_settings: File secrets settings source (ignored).

                Returns:
                    tuple[PydanticBaseSettingsSource, ...]: Ordered settings sources.
                """
                return (
                    init_settings,
                    env_settings,
                    YamlConfigSettingsSource(settings_cls, yaml_file=path),
                )

        return cast(Self, _LoadedWorkerSettings())
