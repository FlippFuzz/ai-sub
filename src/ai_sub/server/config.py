"""Configuration settings for ai-sub-server loaded from YAML with environment overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from ai_sub.config import (
    ReEncodeSettings,
    RetrySettings,
    SearchSettings,
    SplittingSettings,
)


def _parse_string_list(val: Any) -> list[str]:
    """Parses list of strings from Python list, JSON array, or comma-separated string.

    Args:
        val: Raw input value from environment variable or YAML.

    Returns:
        list[str]: Normalized list of strings.
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


def _parse_path_list(val: Any) -> list[Path]:
    """Parses list of Paths from Python list, JSON array, or comma-separated string.

    Args:
        val: Raw input value from environment variable or YAML.

    Returns:
        list[Path]: Normalized list of Path objects.
    """
    raw_list = _parse_string_list(val)
    return [Path(p) for p in raw_list]


StringList = Annotated[list[str], NoDecode, BeforeValidator(_parse_string_list)]
PathList = Annotated[list[Path], NoDecode, BeforeValidator(_parse_path_list)]


class _BaseServerSettings(BaseSettings):
    """Base settings class for server components where ENV overrides YAML kwargs."""

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


class PipelineConfig(BaseModel):
    """Configuration for an independent model processing pipeline."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Unique identifier for the pipeline (e.g. 'combo_38f').")
    summary_model: str = Field(
        default="google-gla:gemini-3.5-flash-lite",
        description="Model to use for whole-video summary.",
    )
    subtitles_model: str = Field(
        default="google-gla:gemini-3.8-flash",
        description="Model to use for subtitle transcription.",
    )
    lyrics_model: str = Field(
        default="google-gla:gemini-3.5-flash-lite",
        description="Model to use for lyrics refinement.",
    )
    search_tools: list[str] | None = Field(
        default=None,
        description="Optional pipeline-specific search tools override. Inherits from global search.tools if null.",
    )


class GitConfig(_BaseServerSettings):
    """Git repository synchronization settings."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_GIT_",
        extra="ignore",
    )

    repo_url: HttpUrl = Field(description="Remote git repository URL for pushing completed subtitles.")
    branch: str = Field(default="main", description="Git branch to checkout and push to.")
    pat: SecretStr = Field(description="GitHub Personal Access Token.")

    def get_pat_value(self) -> str:
        """Resolves the Personal Access Token string.

        Returns:
            str: Clean PAT string.
        """
        return self.pat.get_secret_value()


class FilesSourceConfig(_BaseServerSettings):
    """Local filesystem intake configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SOURCES_FILES_",
        extra="ignore",
    )

    folders: PathList = Field(
        default_factory=list,
        description="Local filesystem directories scanned recursively for video media files.",
    )
    rescan_interval_minutes: int = Field(
        default=2,
        description="Background polling interval in minutes to discover newly added local media files.",
    )


class YoutubeSourceConfig(_BaseServerSettings):
    """YouTube intake, anti-bot pacing, and download configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SOURCES_YOUTUBE_",
        extra="ignore",
    )

    urls: StringList = Field(
        default_factory=list,
        description="YouTube playlist, channel, or video URLs to ingest.",
    )
    rescan_interval_minutes: int = Field(
        default=60,
        description="Base interval in minutes between full YouTube channel and playlist intake scans.",
    )
    rescan_jitter_minutes: int = Field(
        default=15,
        description="Maximum additive random delay in minutes added to the YouTube rescan interval (e.g. 60-75m).",
    )
    channel_delay_seconds: float = Field(
        default=5.0,
        description="Target delay in seconds between scraping individual YouTube channels or playlists.",
    )
    proxy: HttpUrl | None = Field(
        default=None,
        description="Proxy URL for yt-dlp metadata and downloads.",
    )
    cookie_file: Path | None = Field(
        default=None,
        description="Path to cookie file for yt-dlp authentication.",
    )
    channels_file: Path = Field(
        default=Path("channels.json"),
        description="Path to the YouTube channel ID to uploader handle mapping JSON cache.",
    )
    max_title_bytes: int = Field(
        default=100,
        description="Maximum byte length of the video title prefix in downloaded filenames.",
    )
    enable_pot_provider: bool = Field(
        default=True,
        description="Whether to automatically clone, update, and build bgutil-ytdlp-pot-provider.",
    )
    pot_dir: Path = Field(
        default=Path("bgutil-ytdlp-pot-provider"),
        description="Directory path for bgutil-ytdlp-pot-provider repository.",
    )
    pot_update_interval_hours: int = Field(
        default=24,
        description="Minimum interval in hours between bgutil-ytdlp-pot-provider git update checks.",
    )
    max_consecutive_failures: int = Field(
        default=3,
        description="Maximum consecutive video download failures triggering a circuit-breaker pause.",
    )
    download_test_duration_s: int = Field(
        default=10,
        description="Length in seconds of the audio stream pre-flight availability download test.",
    )


class SourcesConfig(_BaseServerSettings):
    """Container for distinct intake source categories (local filesystem and YouTube)."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SOURCES_",
        extra="ignore",
    )

    files: FilesSourceConfig = Field(default_factory=FilesSourceConfig)
    youtube: YoutubeSourceConfig = Field(default_factory=YoutubeSourceConfig)


class ServerOptions(_BaseServerSettings):
    """HTTP server and filesystem orchestration options."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SERVER_",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Bind IP address for the FastAPI server.")
    port: int = Field(default=8000, description="Bind port for the FastAPI server.")
    work_dir: Path = Field(default=Path("./work"), description="Base working directory for pipeline state.")
    heartbeat_timeout_seconds: int = Field(
        default=300,
        description="Seconds before an unresponsive worker lease expires (10 missed 30s heartbeats = 300s).",
    )
    validation_buffer_ms: int = Field(
        default=2000,
        description="Tolerance buffer in milliseconds for timestamp bounds checks.",
    )


class ServerReEncodeSettings(ReEncodeSettings, _BaseServerSettings):
    """Server video re-encoding configuration with AISUB_SERVER_REENCODE_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_REENCODE_",
        extra="ignore",
    )


class ServerSearchSettings(SearchSettings, _BaseServerSettings):
    """Server search provider configuration with AISUB_SERVER_SEARCH_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SEARCH_",
        extra="ignore",
    )


class ServerSplittingSettings(SplittingSettings, _BaseServerSettings):
    """Server video splitting configuration with AISUB_SERVER_SPLIT_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_SPLIT_",
        extra="ignore",
    )


class ServerRetrySettings(RetrySettings, _BaseServerSettings):
    """Server task retry policy configuration with AISUB_SERVER_RETRY_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_RETRY_",
        extra="ignore",
    )


def _get_default_pipelines() -> list[PipelineConfig]:
    """Generates the three standard combinations requested.

    Returns:
        list[PipelineConfig]: Standard default pipelines.
    """
    return [
        PipelineConfig(
            name="combo_38f",
            summary_model="google-gla:gemini-3.5-flash-lite",
            subtitles_model="google-gla:gemini-3.8-flash",
            lyrics_model="google-gla:gemini-3.5-flash-lite",
        ),
        PipelineConfig(
            name="combo_36f",
            summary_model="google-gla:gemini-3.5-flash-lite",
            subtitles_model="google-gla:gemini-3.6-flash",
            lyrics_model="google-gla:gemini-3.5-flash-lite",
        ),
        PipelineConfig(
            name="combo_35l",
            summary_model="google-gla:gemini-3.5-flash-lite",
            subtitles_model="google-gla:gemini-3.5-flash-lite",
            lyrics_model="google-gla:gemini-3.5-flash-lite",
        ),
    ]


class ServerSettings(BaseSettings):
    """Central configuration settings model for ai-sub-server loaded from YAML with ENV overrides."""

    model_config = SettingsConfigDict(
        env_prefix="AISUB_SERVER_",
        extra="ignore",
    )

    server: ServerOptions = Field(default_factory=ServerOptions)
    git: GitConfig | None = Field(default=None)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    reencode: ServerReEncodeSettings = Field(default_factory=ServerReEncodeSettings)
    search: ServerSearchSettings = Field(default_factory=ServerSearchSettings)
    split: ServerSplittingSettings = Field(default_factory=ServerSplittingSettings)
    retry: ServerRetrySettings = Field(default_factory=ServerRetrySettings)
    pipelines: list[PipelineConfig] = Field(default_factory=_get_default_pipelines)

    @model_validator(mode="before")
    @classmethod
    def _check_git_env(cls, values: Any) -> Any:
        """Instantiates GitConfig if git environment variables are present but git is omitted in YAML.

        Args:
            values: Raw input data dictionary.

        Returns:
            Any: Updated input data dictionary.
        """
        if isinstance(values, dict):
            if values.get("git") is None and os.getenv("AISUB_SERVER_GIT_REPO_URL"):
                values["git"] = {}
        return values

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
        """Loads server settings from a YAML file path with environment variable overrides.

        Args:
            yaml_path: Filesystem path to the server.yaml file.

        Returns:
            Self: Instantiated and validated ServerSettings object.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Server configuration file not found: {path}")

        class _LoadedServerSettings(cls):
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
                """Customizes settings source loading order for loaded server settings.

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

        return cast(Self, _LoadedServerSettings())
