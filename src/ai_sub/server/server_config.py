"""Configuration settings for ai-sub-server loaded from YAML with environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import SettingsConfigDict

from ai_sub.config import (
    LoggingSettings,
    PathList,
    ReEncodeSettings,
    RetrySettings,
    ServiceTier,
    SplittingSettings,
    StringList,
    YamlFileSettings,
)


class PipelineConfig(BaseModel):
    """Configuration for an independent model processing pipeline."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

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
    service_tier_summary: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier for whole-video summary.",
    )
    service_tier_subtitles: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier for subtitle generation.",
    )
    service_tier_lyrics: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier for lyrics refinement.",
    )
    search_tools: list[str] | None = Field(
        default=None,
        description="Optional pipeline-specific search tools override.",
    )


class GitConfig(YamlFileSettings):
    """Git repository synchronization settings."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_GIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    repo_url: HttpUrl | None = Field(
        default=None,
        description="Remote git repository URL for pushing completed subtitles.",
    )
    branch: str = Field(
        default="main",
        description="Git branch to checkout and push to.",
    )
    pat: SecretStr | None = Field(
        default=None,
        description="GitHub Personal Access Token.",
    )
    player_base_url: HttpUrl = Field(
        default=HttpUrl("https://flippfuzz.github.io/ai-sub-player"),
        description="Base URL of the GitHub Pages subtitle player.",
    )

    @model_validator(mode="after")
    def _validate_when_active(self) -> Self:
        """Ensures PAT is provided when repo URL is configured.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If repo_url is configured without pat.
        """
        if self.repo_url is not None and not self.pat:
            raise ValueError("Git pat must be provided via AISUB_SERVER_GIT_PAT when repo_url is configured.")
        return self

    def get_pat_value(self) -> str:
        """Resolves the Personal Access Token string.

        Returns:
            str: Clean PAT string.
        """
        return self.pat.get_secret_value() if self.pat else ""


class FilesSourceConfig(YamlFileSettings):
    """Local filesystem intake configuration."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_SOURCES_FILES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    folders: PathList = Field(
        default_factory=list,
        description="Local filesystem directories scanned recursively for video media files.",
    )
    rescan_interval_minutes: PositiveInt = Field(
        default=2,
        description="Background polling interval in minutes to discover newly added local media files.",
    )


class YoutubeSourceConfig(YamlFileSettings):
    """YouTube intake, anti-bot pacing, and download configuration."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_SOURCES_YOUTUBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    urls: StringList = Field(
        default_factory=list,
        description="YouTube playlist, channel, or video URLs to ingest.",
    )
    rescan_interval_minutes: PositiveInt = Field(
        default=60,
        description="Base interval in minutes between full YouTube channel and playlist intake scans.",
    )
    rescan_jitter_minutes: NonNegativeInt = Field(
        default=15,
        description="Symmetric random delay variation in minutes added or subtracted from rescan interval.",
    )
    channel_delay_seconds: NonNegativeFloat = Field(
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
        description="Path to the YouTube channel mapping JSON cache.",
    )
    max_title_bytes: PositiveInt = Field(
        default=100,
        description="Maximum byte length of the video title prefix in downloaded filenames.",
    )
    enable_pot_provider: bool = Field(
        default=True,
        description="Whether to automatically build and run bgutil-ytdlp-pot-provider.",
    )
    pot_dir: Path = Field(
        default=Path("bgutil-ytdlp-pot-provider"),
        description="Directory path for bgutil-ytdlp-pot-provider repository.",
    )
    max_download_failures: PositiveInt = Field(
        default=3,
        description="Maximum consecutive video download failures in a cycle before entering cooldown.",
    )
    cooldown_minutes: PositiveInt = Field(
        default=60,
        description="Cooldown duration in minutes when bot detection or download failure limit is reached.",
    )


class SourcesConfig(YamlFileSettings):
    """Container for distinct intake source categories."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_SOURCES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    files: FilesSourceConfig = Field(default_factory=FilesSourceConfig)
    youtube: YoutubeSourceConfig = Field(default_factory=YoutubeSourceConfig)


class ServerOptions(YamlFileSettings):
    """HTTP server and filesystem orchestration options."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = Field(
        default="127.0.0.1",
        description="Bind IP address for the FastAPI server.",
    )
    port: PositiveInt = Field(
        default=8000,
        description="Bind port for the FastAPI server.",
    )
    work_dir: Path = Field(
        default=Path("./work"),
        description="Base working directory for pipeline state.",
    )
    heartbeat_timeout_seconds: PositiveInt = Field(
        default=300,
        description="Seconds before an unresponsive worker lease expires.",
    )
    validation_buffer_ms: NonNegativeInt = Field(
        default=2000,
        description="Tolerance buffer in milliseconds for timestamp bounds checks.",
    )
    activity_log_maxlen: PositiveInt = Field(
        default=50,
        description="Maximum number of recent worker activity events preserved per combination pipeline.",
    )
    reencode_concurrency: PositiveInt = Field(
        default=1,
        description="Maximum concurrent FFmpeg re-encoding processes across all combination runners.",
    )


def _get_default_pipelines() -> list[PipelineConfig]:
    """Generates the three standard default combination pipelines.

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


class ServerSettings(YamlFileSettings):
    """Central configuration settings model for ai-sub-server loaded from YAML with ENV overrides."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUB_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    server: ServerOptions = Field(default_factory=ServerOptions)
    git: GitConfig | None = Field(default=None)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    reencode: ReEncodeSettings = Field(default_factory=ReEncodeSettings)
    default_search_tools: StringList = Field(
        default_factory=lambda: ["tinyfish", "lrclib"],
        description="Default search tools required for pipelines unless overridden per pipeline.",
    )
    split: SplittingSettings = Field(default_factory=SplittingSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    log: LoggingSettings = Field(default_factory=LoggingSettings)
    pipelines: list[PipelineConfig] = Field(default_factory=_get_default_pipelines)

    @model_validator(mode="before")
    @classmethod
    def _check_git_env(cls, values: Any) -> Any:
        """Instantiates git configuration if git environment variables are present but git is omitted in YAML.

        Args:
            values: Raw input data dictionary.

        Returns:
            Any: Updated input data dictionary.
        """
        if isinstance(values, dict) and values.get("git") is None:
            git_cfg = GitConfig()
            if git_cfg.repo_url is not None:
                values["git"] = git_cfg
        return values

    @model_validator(mode="after")
    def _normalize_git_config(self) -> Self:
        """Normalizes an inactive GitConfig (missing repo_url) to None.

        Returns:
            Self: Validated instance.
        """
        if self.git is not None and self.git.repo_url is None:
            self.git = None
        return self
