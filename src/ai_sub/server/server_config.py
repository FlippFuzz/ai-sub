"""Configuration settings for ai-sub-server loaded from YAML with environment overrides."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Self

from annotated_types import Ge, Gt
from pydantic import Field, HttpUrl, PositiveInt, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

from ai_sub.config import (
    BaseConfigModel,
    LoggingSettings,
    PathList,
    ReencodeSettings,
    RetrySettings,
    SearchTool,
    SearchToolList,
    SplittingSettings,
    StageExecutionConfig,
    StringList,
    YamlFileSettings,
)


def _default_pipeline_search_tools() -> list[SearchTool]:
    """Returns default search tools required by server pipelines.

    Returns:
        list[SearchTool]: Default search tools list.
    """
    return ["tinyfish", "lrclib"]


class HttpConfig(BaseConfigModel):
    """HTTP server binding options."""

    host: str = Field(
        default="127.0.0.1",
        description="Bind IP address for the FastAPI server.",
    )
    port: PositiveInt = Field(
        default=8000,
        description="Bind port for the FastAPI server.",
    )


class CoordinatorConfig(BaseConfigModel):
    """Coordinator orchestration and task lifecycle options."""

    workdir: Path = Field(
        default=Path("./work"),
        description="Base working directory for pipeline state.",
    )
    heartbeattimeout: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=300),
        description="Duration before an unresponsive worker lease expires (seconds if numeric).",
    )
    validationbuffer: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(seconds=2),
        description="Tolerance buffer for timestamp bounds checks (seconds if numeric).",
    )
    activitymaxlen: PositiveInt = Field(
        default=50,
        description="Maximum number of recent worker activity events preserved per combination pipeline.",
    )


class ServerReencodeSettings(ReencodeSettings):
    """Server re-encoding settings with global FFmpeg concurrency limit."""

    concurrency: PositiveInt = Field(
        default=1,
        description="Maximum concurrent FFmpeg re-encoding processes across all combination runners.",
    )


class PipelineConfig(BaseConfigModel):
    """Configuration for an independent model processing pipeline."""

    name: str = Field(description="Unique identifier for the pipeline (e.g. 'combo_38f').")
    summary: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        description="Model, tier, and agentic settings for whole-video summary.",
    )
    subtitles: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-3.8-flash", agentic=False),
        description="Model, tier, and agentic settings for subtitle transcription.",
    )
    lyrics: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        description="Model, tier, and agentic settings for lyrics refinement.",
    )
    searchtools: SearchToolList = Field(
        default_factory=_default_pipeline_search_tools,
        description="Search tools required for this pipeline.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_stage_agentic_defaults(cls, val: Any) -> Any:
        """Defaults summary and lyrics agentic to True when omitted in pipeline YAML entries.

        Args:
            val: Raw pipeline dictionary before validation.

        Returns:
            Any: Normalized dictionary with stage-appropriate agentic defaults.
        """
        if not isinstance(val, dict):
            return val
        updated = dict(val)
        for stage in ("summary", "lyrics"):
            stage_val = updated.get(stage)
            if isinstance(stage_val, dict) and "agentic" not in stage_val:
                updated[stage] = {**stage_val, "agentic": True}
        return updated


class GitConfig(BaseConfigModel):
    """Git repository synchronization settings."""

    repourl: HttpUrl | None = Field(
        default=None,
        description="Remote git repository URL for pushing completed subtitles (None disables Git sync).",
    )
    branch: str = Field(
        default="main",
        description="Git branch to checkout and push to.",
    )
    pat: SecretStr | None = Field(
        default=None,
        description="GitHub Personal Access Token.",
    )
    playerurl: HttpUrl = Field(
        default=HttpUrl("https://flippfuzz.github.io/ai-sub-player"),
        description="Base URL of the GitHub Pages subtitle player.",
    )

    @model_validator(mode="after")
    def _validate_when_active(self) -> Self:
        """Ensures PAT is provided when repo URL is configured.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If repourl is configured without a non-empty pat.
        """
        if self.repourl is not None and not self.get_pat_value().strip():
            raise ValueError("Git pat must be provided via AISUBSERVER_GIT_PAT when git.repourl is configured.")
        return self

    def get_pat_value(self) -> str:
        """Resolves the Personal Access Token string.

        Returns:
            str: Clean PAT string.
        """
        return self.pat.get_secret_value() if self.pat else ""


class FilesSourceConfig(BaseConfigModel):
    """Local filesystem intake configuration."""

    folders: PathList = Field(
        default_factory=list,
        description="Local filesystem directories scanned recursively for video media files.",
    )
    interval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(minutes=2),
        description="Background polling interval to discover newly added local media files (seconds if numeric).",
    )
    maxstembytes: PositiveInt = Field(
        default=100,
        description="Maximum UTF-8 byte length for local video filename stems in output folders and SRT filenames.",
    )


class YoutubeSourceConfig(BaseConfigModel):
    """YouTube intake, anti-bot pacing, and download configuration."""

    urls: StringList = Field(
        default_factory=list,
        description="YouTube playlist, channel, or video URLs to ingest.",
    )
    interval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(minutes=60),
        description="Base interval between full YouTube channel and playlist intake scans (seconds if numeric).",
    )
    lookback: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(hours=72),
        description="Minimum age window for delta YouTube scans before cutoff (0 disables; seconds if numeric).",
    )
    jitter: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(minutes=15),
        description="Symmetric random delay variation added or subtracted from scan interval (0 disables).",
    )
    channeldelay: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(seconds=5),
        description="Target delay between scraping individual YouTube channels or playlists (0 disables).",
    )
    proxy: HttpUrl | None = Field(
        default=None,
        description="Proxy URL for yt-dlp metadata and downloads.",
    )
    cookiefile: Path | None = Field(
        default=None,
        description="Path to cookie file for yt-dlp authentication.",
    )
    channelsfile: Path = Field(
        default=Path("channels.json"),
        description="Path to the YouTube channel mapping JSON cache.",
    )
    potdir: Path | None = Field(
        default=Path("bgutil-ytdlp-pot-provider"),
        description="Directory path for bgutil-ytdlp-pot-provider repository (None disables PO token provider).",
    )
    maxfailures: PositiveInt = Field(
        default=3,
        description="Maximum consecutive video download failures in a cycle before entering cooldown.",
    )
    cooldown: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(minutes=60),
        description="Cooldown duration when bot detection or failure limit is reached (seconds if numeric).",
    )


class SourcesConfig(BaseConfigModel):
    """Container for distinct intake source categories."""

    files: FilesSourceConfig = Field(default_factory=FilesSourceConfig)
    youtube: YoutubeSourceConfig = Field(default_factory=YoutubeSourceConfig)


def _get_default_pipelines() -> list[PipelineConfig]:
    """Generates the three standard default combination pipelines.

    Returns:
        list[PipelineConfig]: Standard default pipelines.
    """
    return [
        PipelineConfig(
            name="combo_38f",
            summary=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
            subtitles=StageExecutionConfig(model="gemini-3.8-flash", agentic=False),
            lyrics=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        ),
        PipelineConfig(
            name="combo_36f",
            summary=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
            subtitles=StageExecutionConfig(model="gemini-3.6-flash", agentic=False),
            lyrics=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        ),
        PipelineConfig(
            name="combo_35l",
            summary=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
            subtitles=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=False),
            lyrics=StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        ),
    ]


class ServerSettings(YamlFileSettings):
    """Central configuration settings model for ai-sub-server loaded from YAML with ENV overrides."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        env_prefix="AISUBSERVER_",
        env_nested_delimiter="_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    http: HttpConfig = Field(default_factory=HttpConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    reencode: ServerReencodeSettings = Field(default_factory=ServerReencodeSettings)
    split: SplittingSettings = Field(default_factory=SplittingSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    log: LoggingSettings = Field(default_factory=LoggingSettings)
    pipelines: list[PipelineConfig] = Field(default_factory=_get_default_pipelines)
