"""Configuration settings for ai-sub-server loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

from ai_sub.config import ReEncodeSettings, SearchSettings, SplittingSettings


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


class GitConfig(BaseModel):
    """Git repository synchronization settings."""

    model_config = ConfigDict(extra="ignore")

    repo_url: HttpUrl = Field(description="Remote git repository URL for pushing completed subtitles.")
    branch: str = Field(default="main", description="Git branch to checkout and push to.")
    pat: SecretStr = Field(description="GitHub Personal Access Token.")

    def get_pat_value(self) -> str:
        """Resolves the Personal Access Token string.

        Returns:
            str: Clean PAT string.
        """
        return self.pat.get_secret_value()


class SourcesConfig(BaseModel):
    """Intake sources configuration (watched folders, YouTube feeds, and PO token setup)."""

    model_config = ConfigDict(extra="ignore")

    watch_folders: list[Path] = Field(
        default_factory=list,
        description="Local filesystem directories scanned recursively for video files.",
    )
    youtube: list[str] = Field(
        default_factory=list,
        description="YouTube playlist, channel, or video URLs to ingest.",
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


class ServerOptions(BaseModel):
    """HTTP server and filesystem orchestration options."""

    model_config = ConfigDict(extra="ignore")

    host: str = Field(default="127.0.0.1", description="Bind IP address for the FastAPI server.")
    port: int = Field(default=8000, description="Bind port for the FastAPI server.")
    work_dir: Path = Field(default=Path("./work"), description="Base working directory for pipeline state.")
    heartbeat_timeout_seconds: int = Field(
        default=300,
        description="Seconds before an unresponsive worker lease expires (10 missed 30s heartbeats = 300s).",
    )
    rescan_interval_minutes: int = Field(
        default=15,
        description="Background polling interval in minutes to discover new files/uploads when pipelines are idle.",
    )
    validation_buffer_ms: int = Field(
        default=2000,
        description="Tolerance buffer in milliseconds for timestamp bounds checks.",
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


class ServerSettings(BaseModel):
    """Central configuration settings model for ai-sub-server loaded from YAML."""

    model_config = ConfigDict(extra="ignore")

    server: ServerOptions = Field(default_factory=ServerOptions)
    git: GitConfig | None = Field(default=None)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    reencode: ReEncodeSettings = Field(default_factory=ReEncodeSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    split: SplittingSettings = Field(default_factory=SplittingSettings)
    pipelines: list[PipelineConfig] = Field(default_factory=_get_default_pipelines)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        """Loads and parses settings from a specified YAML file path.

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

        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls.model_validate(data)
