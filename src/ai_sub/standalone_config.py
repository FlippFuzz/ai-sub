"""Configuration settings for standalone ai-sub execution."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Self, cast

from annotated_types import Ge
from pydantic import Field, FilePath, NonNegativeInt, PositiveInt, model_validator
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from ai_sub.config import (
    AiSettings,
    BaseConfigModel,
    LoggingSettings,
    ReencodeSettings,
    RetrySettings,
    SplittingSettings,
    StageModelsMixin,
)


class StandaloneLoggingSettings(LoggingSettings):
    """Telemetry, console logging, and terminal progress bar configuration."""

    progress: bool = Field(
        default=True,
        description="Whether to display live Rich progress bars.",
    )


class StandaloneAiSettings(AiSettings, StageModelsMixin):
    """AI engine settings extended with validation buffer and per-stage model, tier, and agentic options."""

    buffer: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(seconds=2),
        description="Allowed tolerance buffer for subtitle timestamps exceeding segment duration (seconds if numeric).",
    )


class ConcurrencySettings(BaseConfigModel):
    """Concurrency settings controlling parallel worker tasks in standalone mode."""

    reencode: PositiveInt = Field(
        default=2,
        description="Number of concurrent FFmpeg compression processes.",
    )
    summary: NonNegativeInt = Field(
        default=1,
        description="Number of concurrent summary tasks (0 disables whole-video summary).",
    )
    subtitles: PositiveInt = Field(
        default=4,
        description="Number of concurrent subtitle generation tasks.",
    )
    lyrics: NonNegativeInt = Field(
        default=4,
        description="Number of concurrent lyrics refinement tasks (0 disables lyrics lookup).",
    )


class DirectorySettings(BaseConfigModel):
    """Configuration for temporary and output working directories."""

    tmp: Path | None = Field(
        default=None,
        description="Temporary directory for intermediate segments and state files (defaults to out/tmp_<stem>).",
    )
    out: Path | None = Field(
        default=None,
        description="Output directory for generated subtitle files (defaults to input video directory).",
    )


class StandaloneSettings(BaseSettings):
    """Root standalone application configuration model."""

    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        cli_avoid_json=True,
        cli_kebab_case=True,
        env_prefix="AISUB_",
        env_nested_delimiter="_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    ai: StandaloneAiSettings = Field(
        default_factory=StandaloneAiSettings,
        description="AI models, quotas, storage, and search settings.",
    )
    reencode: ReencodeSettings = Field(
        default_factory=ReencodeSettings,
        description="Video re-encoding and compression parameters.",
    )
    split: SplittingSettings = Field(
        default_factory=SplittingSettings,
        description="Video segmentation parameters.",
    )
    concurrency: ConcurrencySettings = Field(
        default_factory=ConcurrencySettings,
        description="Local concurrency limits.",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Retry policy parameters.",
    )
    dir: DirectorySettings = Field(
        default_factory=DirectorySettings,
        description="Input and output directory paths.",
    )
    log: StandaloneLoggingSettings = Field(
        default_factory=StandaloneLoggingSettings,
        description="Telemetry, logging, and progress bar controls.",
    )
    context: str | None = Field(
        default=None,
        description="Inline text providing background context, speaker names, or translation preferences.",
    )
    contextfile: FilePath | None = Field(
        default=None,
        description="Path to a text file containing user background context.",
    )
    video: CliPositionalArg[FilePath] = Field(
        description="Path to the video file to transcribe and subtitle.",
    )

    def get_resolved_user_context(self) -> str | None:
        """Reads and merges context from contextfile and inline context if provided.

        Returns:
            str | None: Combined user context string or None.

        Raises:
            ValueError: If contextfile cannot be read.
        """
        parts: list[str] = []
        if self.contextfile is not None:
            try:
                content = self.contextfile.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except (OSError, UnicodeDecodeError) as e:
                raise ValueError(f"Could not read context file '{self.contextfile}': {e}") from e

        if self.context is not None and self.context.strip():
            parts.append(self.context.strip())

        return "\n\n".join(parts) if parts else None

    @model_validator(mode="after")
    def _validate_api_keys(self) -> Self:
        """Ensures required API keys are configured before pipeline initialization.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If a required API key is missing.
        """
        if self.ai.key is None or not self.ai.key.get_secret_value().strip():
            raise ValueError(
                "A Google Gemini API key must be provided via '--ai.key', "
                "'AISUB_AI_KEY', or the GEMINI_API_KEY / GOOGLE_API_KEY environment variables."
            )

        if self.concurrency.summary > 0 or self.concurrency.lyrics > 0:
            self.ai.search.validate_required_keys(env_prefix="AISUB")

        return self

    @model_validator(mode="after")
    def _setup_file_locations(self) -> Self:
        """Resolves input, output, and temporary paths to absolute directory locations.

        Returns:
            Self: The validated instance.
        """
        video_path = cast(Path, self.video).resolve()
        self.video = cast(Any, video_path)

        out_dir = (self.dir.out if self.dir.out is not None else video_path.parent).resolve()
        tmp_dir = (self.dir.tmp if self.dir.tmp is not None else out_dir / f"tmp_{video_path.stem}").resolve()

        self.dir.out = out_dir
        self.dir.tmp = tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)

        return self
