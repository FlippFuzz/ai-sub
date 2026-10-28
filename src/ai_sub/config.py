"""Configuration settings for the AI Sub subtitle generation pipeline."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Literal, Self, cast

from logfire import LevelName
from pydantic import (
    Field,
    FilePath,
    HttpUrl,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

_BASE_CONFIG: SettingsConfigDict = {
    "env_file": ".env",
    "env_file_encoding": "utf-8",
    "extra": "ignore",
}

ServiceTier = Literal["auto", "default", "flex", "priority"]


def find_config_file(kind: Literal["server", "worker"], cli_arg: str | Path | None = None) -> Path:
    """Resolves configuration YAML file path following strict discovery rules.

    Resolution order:
    1. Explicit positional CLI argument supplied by user.
    2. <hostname>.<kind>.yaml (and .yml)
    3. <short_hostname>.<kind>.yaml (and .yml) if hostname contains domain dots
    4. <kind>.yaml (and .yml)

    Args:
        kind: Configuration type ('server' or 'worker').
        cli_arg: Optional explicit path passed as CLI argument.

    Returns:
        Path: Resolved, existing configuration file path.

    Raises:
        FileNotFoundError: If no matching configuration file exists on disk.
    """
    if cli_arg:
        path = Path(cli_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Specified {kind} configuration file not found: {path}")
        return path

    hostname = socket.gethostname()
    candidates: list[Path] = [
        Path(f"{hostname}.{kind}.yaml"),
        Path(f"{hostname}.{kind}.yml"),
    ]
    if "." in hostname:
        short_host = hostname.split(".")[0]
        candidates.extend(
            [
                Path(f"{short_host}.{kind}.yaml"),
                Path(f"{short_host}.{kind}.yml"),
            ]
        )
    candidates.extend(
        [
            Path(f"{kind}.yaml"),
            Path(f"{kind}.yml"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked_str = ", ".join(repr(str(c)) for c in candidates)
    raise FileNotFoundError(
        f"No {kind} configuration file found. Checked: {checked_str}. "
        f"Please create '{kind}.yaml' or pass a configuration file path."
    )


class StorageSettings(BaseSettings):
    """Configuration for Gemini Files API storage lifecycle and cleanup."""

    model_config = {
        "env_prefix": "AISUB_AI_STORAGE_",
        **_BASE_CONFIG,
    }

    max_gb: PositiveFloat = Field(
        description="High-water mark in Gigabytes (GB) triggering managed file cleanup.",
        default=15.0,
    )
    target_gb: PositiveFloat = Field(
        description="Low-water mark in Gigabytes (GB) to delete down to during cleanup.",
        default=10.0,
    )
    cleanup_interval_minutes: PositiveInt = Field(
        description="Minimum cooldown period in minutes between storage cleanup checks.",
        default=30,
    )
    polling_interval_seconds: PositiveFloat = Field(
        description="Interval in seconds to poll the Gemini Files API while processing an uploaded file.",
        default=10.0,
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        """Validates that low-water mark target_gb is strictly less than max_gb.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If target_gb is greater than or equal to max_gb.
        """
        if self.target_gb >= self.max_gb:
            raise ValueError(
                f"Storage low-water mark 'target_gb' ({self.target_gb} GB) must be strictly "
                f"less than high-water mark 'max_gb' ({self.max_gb} GB)."
            )
        return self


class SearchSettings(BaseSettings):
    """Configuration for auxiliary web search tools used by the AI model."""

    model_config = {
        "env_prefix": "AISUB_AI_SEARCH_",
        "enable_decoding": False,
        **_BASE_CONFIG,
    }

    tools: list[str] = Field(
        description="List of search providers to enable (e.g. ['duckduckgo'], ['tinyfish', 'lrclib']). "
        "Supported: 'builtin', 'duckduckgo', 'ollama', 'langsearch', 'tinyfish', 'lrclib'.",
        default_factory=lambda: ["duckduckgo"],
    )
    key: SecretStr | None = Field(
        description="Fallback API key for search services (Ollama, Langsearch, or Tinyfish).",
        default=None,
    )
    tinyfish_key: SecretStr | None = Field(
        description="Dedicated API key for Tinyfish search.",
        default=None,
    )
    langsearch_key: SecretStr | None = Field(
        description="Dedicated API key for Langsearch.",
        default=None,
    )
    ollama_key: SecretStr | None = Field(
        description="Dedicated API key for Ollama search.",
        default=None,
    )
    max_turns: PositiveInt = Field(
        description="Maximum function-calling search turns permitted per segment.",
        default=3,
    )
    qps: PositiveFloat = Field(
        description="Maximum queries per second for web search requests.",
        default=0.3,
    )
    max_length: PositiveInt = Field(
        description="Maximum characters permitted in a single search result snippet or fetched page.",
        default=4096,
    )
    timeout: PositiveFloat = Field(
        description="Timeout in seconds for external search HTTP requests.",
        default=60.0,
    )
    retries: NonNegativeInt = Field(
        description="Maximum retries for search requests on transient errors.",
        default=3,
    )
    min_wait_seconds: PositiveFloat = Field(
        description="Minimum wait time in seconds for search exponential backoff.",
        default=5.0,
    )
    max_wait_seconds: PositiveFloat = Field(
        description="Maximum wait time in seconds for search exponential backoff.",
        default=60.0,
    )
    multiplier: PositiveFloat = Field(
        description="Exponential backoff multiplier for search retries.",
        default=2.0,
    )

    def get_key_for_provider(self, provider_name: str) -> SecretStr | None:
        """Resolves the API key for a specific search provider.

        Args:
            provider_name: The provider identifier ('tinyfish', 'langsearch', 'ollama').

        Returns:
            SecretStr | None: The resolved API key or None if not set.
        """
        p = provider_name.strip().lower()
        if p == "tinyfish":
            return self.tinyfish_key or self.key
        if p == "langsearch":
            return self.langsearch_key or self.key
        if p == "ollama":
            return self.ollama_key or self.key
        return self.key

    @model_validator(mode="before")
    @classmethod
    def load_api_keys_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Normalizes tools list and loads provider-specific search keys from environment.

        Args:
            values: Raw input dictionary before validation.

        Returns:
            dict[str, Any]: Updated input dictionary.
        """
        if "tool" in values and values.get("tool"):
            raw_tool = values.pop("tool")
            if "tools" not in values:
                values["tools"] = raw_tool

        raw_tools = values.get("tools")
        if raw_tools is None:
            env_tools = os.getenv("AISUB_AI_SEARCH_TOOLS") or os.getenv("AISUB_AI_SEARCH_TOOL")
            if env_tools:
                raw_tools = env_tools
            else:
                raw_tools = ["duckduckgo"]

        if isinstance(raw_tools, str):
            raw_tools_str = raw_tools.strip()
            if raw_tools_str.startswith("[") and raw_tools_str.endswith("]"):
                try:
                    parsed = json.loads(raw_tools_str)
                    if isinstance(parsed, (list, tuple)):
                        raw_tools = parsed
                except Exception:
                    pass

        if isinstance(raw_tools, str):
            values["tools"] = [t.strip() for t in raw_tools.split(",") if t.strip()]
        elif isinstance(raw_tools, (list, tuple)):
            flattened: list[str] = []
            for item in raw_tools:
                if isinstance(item, str) and "," in item:
                    flattened.extend(t.strip() for t in item.split(",") if t.strip())
                elif isinstance(item, str) and item.strip():
                    flattened.append(item.strip())
            values["tools"] = flattened

        if values.get("tinyfish_key") is None:
            if env_val := os.getenv("TINYFISH_API_KEY"):
                values["tinyfish_key"] = env_val.strip().strip('"').strip("'")

        if values.get("langsearch_key") is None:
            if env_val := os.getenv("LANGSEARCH_API_KEY"):
                values["langsearch_key"] = env_val.strip().strip('"').strip("'")

        if values.get("ollama_key") is None:
            if env_val := os.getenv("OLLAMA_API_KEY"):
                values["ollama_key"] = env_val.strip().strip('"').strip("'")

        if values.get("key") is None:
            general_key = os.getenv("AISUB_AI_SEARCH_KEY")
            if general_key:
                values["key"] = general_key.strip().strip('"').strip("'")

        return values

    @model_validator(mode="after")
    def validate_tools_list(self) -> Self:
        """Validates that configured search tools are supported.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If an unknown search tool is specified.
        """
        valid_tools = {"builtin", "duckduckgo", "ollama", "langsearch", "tinyfish", "lrclib"}
        for t in self.tools:
            if t.lower() not in valid_tools:
                raise ValueError(f"Unknown search provider '{t}'. Supported providers: {sorted(valid_tools)}")
        return self


class AiSettings(BaseSettings):
    """Configuration for Google Gemini models and rate limiting."""

    model_config = {
        "env_prefix": "AISUB_AI_",
        **_BASE_CONFIG,
    }

    key: SecretStr | None = Field(
        description="Google Gemini API key. Falls back to GEMINI_API_KEY or GOOGLE_API_KEY.",
        default=None,
    )
    base_url: HttpUrl | None = Field(
        description="Optional custom base URL or proxy for Google Gemini API calls.",
        default=None,
    )
    model: str | None = Field(
        default=None,
        description="Shorthand setting model_subtitles, model_lyrics, and model_summary to the same identifier.",
    )
    model_summary: str = Field(
        description="AI model for whole-video summary, entity extraction, and chapter indexing.",
        default="google-gla:gemini-3.5-flash-lite",
    )
    model_subtitles: str = Field(
        description="AI model for audio-visual subtitle transcription and alignment.",
        default="google-gla:gemini-3.8-flash",
    )
    model_lyrics: str = Field(
        description="AI model for musical lyrics search and refinement with agentic video inspection.",
        default="google-gla:gemini-3.5-flash-lite",
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Shorthand setting service tiers across all pipeline phases.",
    )
    service_tier_summary: ServiceTier | None = Field(
        default=None,
        description="Google Gemini service tier for whole-video summary.",
    )
    service_tier_subtitles: ServiceTier | None = Field(
        default=None,
        description="Google Gemini service tier for subtitle generation.",
    )
    service_tier_lyrics: ServiceTier | None = Field(
        default=None,
        description="Google Gemini service tier for lyrics refinement.",
    )
    agentic_video: bool | None = Field(
        default=None,
        description="Shorthand setting agentic video understanding across all pipeline phases.",
    )
    agentic_video_summary: bool = Field(
        default=True,
        description="Whether to use agentic video understanding for whole-video summary.",
    )
    agentic_video_subtitles: bool = Field(
        default=False,
        description="Whether to use agentic video understanding for subtitle generation.",
    )
    agentic_video_lyrics: bool = Field(
        default=True,
        description="Whether to use agentic video understanding for lyrics refinement.",
    )
    rpm: PositiveInt = Field(
        description="Maximum Requests Per Minute (RPM) allowed for the model.",
        default=4,
    )
    tpm: PositiveInt = Field(
        description="Maximum Tokens Per Minute (TPM) allowed for the model.",
        default=250000,
    )
    validation_buffer_ms: NonNegativeInt = Field(
        description="Allowed tolerance buffer in milliseconds for timestamps exceeding segment duration.",
        default=2000,
    )
    storage: StorageSettings = Field(
        description="Files API storage thresholds and cleanup rules.",
        default_factory=StorageSettings,
    )
    search: SearchSettings = Field(
        description="Web search configuration for lyrics and metadata retrieval.",
        default_factory=SearchSettings,
    )

    @model_validator(mode="before")
    @classmethod
    def load_api_key_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Loads Google Gemini key from ambient environment variables.

        Args:
            values: Raw input dictionary before validation.

        Returns:
            dict[str, Any]: Updated input dictionary.
        """
        if values.get("key") is None:
            if env_key := (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                values["key"] = env_key.strip().strip('"').strip("'")
        return values

    @model_validator(mode="after")
    def validate_models_and_tiers(self) -> Self:
        """Synchronizes shorthands and validates required provider prefixes.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If a model identifier does not start with 'google-gla:'.
        """
        if self.model:
            self.model_summary = self.model
            self.model_lyrics = self.model
            self.model_subtitles = self.model

        for field_name, model_val in (
            ("model_summary", self.model_summary),
            ("model_lyrics", self.model_lyrics),
            ("model_subtitles", self.model_subtitles),
        ):
            if not model_val.lower().startswith("google-gla:"):
                raise ValueError(
                    f"Invalid model '{model_val}' for '{field_name}'. "
                    "Model identifier must start with 'google-gla:' (e.g., 'google-gla:gemini-3.8-flash')."
                )

        if self.service_tier is not None:
            self.service_tier_summary = self.service_tier
            self.service_tier_lyrics = self.service_tier
            self.service_tier_subtitles = self.service_tier

        if self.agentic_video is not None:
            self.agentic_video_summary = self.agentic_video
            self.agentic_video_subtitles = self.agentic_video
            self.agentic_video_lyrics = self.agentic_video

        return self


class SummaryReEncodeSettings(BaseSettings):
    """Re-encoding configuration for the whole-video summary proxy."""

    model_config = {
        "env_prefix": "AISUB_REENCODE_SUMMARY_",
        **_BASE_CONFIG,
    }

    enabled: bool = Field(
        default=True,
        description="Whether whole-video proxy re-encoding is enabled.",
    )
    threshold_mb: NonNegativeInt = Field(
        default=300,
        description="File size threshold in MB above which the full video is compressed. 0 compresses unconditionally.",
    )
    keyframe_only: bool = Field(
        default=True,
        description="Whether to decode only keyframes from the input video for faster summary proxy re-encoding.",
    )
    fps: PositiveFloat = Field(
        default=1.0,
        description="Target framerate (FPS) for the summary proxy video.",
    )
    height: PositiveInt = Field(
        default=360,
        description="Target resolution height in pixels (aspect ratio is preserved).",
    )
    bitrate_kb: PositiveInt = Field(
        default=35,
        description="Target video bitrate in Kilobytes per second (KB/s).",
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        default=100,
        description="Allowed duration divergence in milliseconds for summary proxy output.",
    )


class SegmentReEncodeSettings(BaseSettings):
    """Re-encoding configuration for individual video chunk segments."""

    model_config = {
        "env_prefix": "AISUB_REENCODE_SEGMENT_",
        **_BASE_CONFIG,
    }

    enabled: bool = Field(
        default=False,
        description="Whether segment re-encoding is enabled.",
    )
    threshold_mb: NonNegativeInt = Field(
        default=20,
        description="File size threshold in MB above which chunks are compressed. 0 compresses unconditionally.",
    )
    keyframe_only: bool = Field(
        default=False,
        description="Whether to decode only keyframes from the input video for segment chunk re-encoding.",
    )
    fps: PositiveFloat = Field(
        default=1.0,
        description="Target framerate (FPS) for compressed segment chunks.",
    )
    height: PositiveInt = Field(
        default=360,
        description="Target resolution height in pixels (aspect ratio is preserved).",
    )
    bitrate_kb: PositiveInt = Field(
        default=35,
        description="Target video bitrate in Kilobytes per second (KB/s).",
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        default=100,
        description="Allowed duration divergence in milliseconds for re-encoded chunk output.",
    )


class ReEncodeSettings(BaseSettings):
    """Configuration for FFmpeg video re-encoding across pipeline stages."""

    model_config = {
        "env_prefix": "AISUB_REENCODE_",
        **_BASE_CONFIG,
    }

    encoder: str | None = Field(
        default=None,
        description="Specific H.264 encoder to use (e.g., 'h264_nvenc', 'libx264'). Auto-detected if null.",
    )
    summary: SummaryReEncodeSettings = Field(
        description="Re-encoding rules for the whole-video summary proxy.",
        default_factory=SummaryReEncodeSettings,
    )
    segment: SegmentReEncodeSettings = Field(
        description="Re-encoding rules for individual video chunk segments.",
        default_factory=SegmentReEncodeSettings,
    )


class SplittingSettings(BaseSettings):
    """Configuration for splitting input videos into uniform segments."""

    model_config = {
        "env_prefix": "AISUB_SPLIT_",
        **_BASE_CONFIG,
    }

    max_seconds: PositiveInt = Field(
        description="Maximum duration in seconds for each video chunk.",
        default=300,
    )
    start_offset_min: NonNegativeInt = Field(
        description="Duration in minutes to skip from the beginning of the video.",
        default=0,
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        description="Maximum allowed duration variance in milliseconds per segment during split validation.",
        default=100,
    )


class ConcurrencySettings(BaseSettings):
    """Concurrency settings controlling parallel worker tasks in standalone mode."""

    model_config = {
        "env_prefix": "AISUB_CONCURRENCY_",
        **_BASE_CONFIG,
    }

    re_encode: PositiveInt = Field(
        description="Number of concurrent FFmpeg compression processes.",
        default=2,
    )
    summary: NonNegativeInt = Field(
        description="Number of concurrent summary tasks. Set to 0 to skip whole-video summary.",
        default=1,
    )
    subtitles: PositiveInt = Field(
        description="Number of concurrent subtitle generation tasks.",
        default=4,
    )
    lyrics: NonNegativeInt = Field(
        description="Number of concurrent lyrics refinement tasks. Set to 0 to skip lyrics lookup.",
        default=4,
    )


class RetrySettings(BaseSettings):
    """Retry parameters for transient API failures and cross-run persistence."""

    model_config = {
        "env_prefix": "AISUB_RETRY_",
        **_BASE_CONFIG,
    }

    per_run: NonNegativeInt = Field(
        description="Maximum internal retries by the executor on transient network or schema failures.",
        default=5,
    )
    max_runs: NonNegativeInt = Field(
        description="Maximum total attempt limit persisted across application executions per segment.",
        default=3,
    )
    multiplier: PositiveFloat = Field(
        description="Exponential backoff multiplier between retry attempts.",
        default=2.0,
    )
    min_wait_seconds: PositiveFloat = Field(
        description="Minimum backoff wait time in seconds.",
        default=30.0,
    )
    max_wait_seconds: PositiveInt = Field(
        description="Maximum backoff wait time in seconds.",
        default=300,
    )


class DirectorySettings(BaseSettings):
    """Configuration for working directories."""

    model_config = {
        "env_prefix": "AISUB_DIR_",
        **_BASE_CONFIG,
    }

    tmp: Path = Field(
        description="Temporary directory for intermediate segments and state files.",
        default=Path("tmp_input_video_file"),
    )
    out: Path = Field(
        description="Output directory for generated subtitle files.",
        default=Path("directory_of_input_file"),
    )


class LoggingSettings(BaseSettings):
    """Telemetry and console logging configuration."""

    model_config = {
        "env_prefix": "AISUB_LOG_",
        **_BASE_CONFIG,
    }

    level: LevelName = Field(
        description="Minimum log level to output.",
        default="info",
    )
    timestamps: bool = Field(
        description="Whether to prefix console log lines with timestamps.",
        default=False,
    )
    scrub: bool = Field(
        description="Whether to scrub sensitive API keys and secrets from logs.",
        default=True,
    )
    progress: bool = Field(
        description="Whether to display live Rich progress bars.",
        default=True,
    )


class Settings(BaseSettings):
    """Root application configuration model."""

    model_config = {
        "nested_model_default_partial_update": True,
        "cli_avoid_json": True,
        "cli_kebab_case": True,
        "env_prefix": "AISUB_",
        **_BASE_CONFIG,
    }

    ai: AiSettings = Field(
        description="AI models, quotas, storage, and search settings.",
        default_factory=AiSettings,
    )
    reencode: ReEncodeSettings = Field(
        description="Video re-encoding and compression parameters.",
        default_factory=ReEncodeSettings,
    )
    split: SplittingSettings = Field(
        description="Video segmentation parameters.",
        default_factory=SplittingSettings,
    )
    concurrency: ConcurrencySettings = Field(
        description="Local concurrency limits.",
        default_factory=ConcurrencySettings,
    )
    retry: RetrySettings = Field(
        description="Retry policy parameters.",
        default_factory=RetrySettings,
    )
    dir: DirectorySettings = Field(
        description="Input and output directory paths.",
        default_factory=DirectorySettings,
    )
    log: LoggingSettings = Field(
        description="Telemetry and logging controls.",
        default_factory=LoggingSettings,
    )
    context: str | None = Field(
        default=None,
        description="Inline text providing background context, speaker names, or translation preferences.",
    )
    context_file: FilePath | None = Field(
        default=None,
        description="Path to a text file containing user background context.",
    )

    input_video_file: CliPositionalArg[FilePath] = Field(
        description="Path to the video file to transcribe and subtitle.",
    )

    def get_resolved_user_context(self) -> str | None:
        """Reads and merges context from context_file and inline context if provided.

        Returns:
            str | None: Combined, non-empty user context string or None if neither provided.

        Raises:
            ValueError: If context_file cannot be read due to file errors or encoding issues.
        """
        parts: list[str] = []
        if self.context_file is not None:
            try:
                content = self.context_file.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except (OSError, UnicodeDecodeError) as e:
                raise ValueError(f"Could not read context file '{self.context_file}': {e}") from e

        if self.context is not None and self.context.strip():
            parts.append(self.context.strip())

        return "\n\n".join(parts) if parts else None

    @model_validator(mode="after")
    def validate_api_keys(self) -> Self:
        """Ensures required API keys are configured before pipeline initialization.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If a required API key is missing.
        """
        if self.ai.key is None:
            raise ValueError(
                "A Google Gemini API key must be provided via '--ai.key', "
                "or the GEMINI_API_KEY / GOOGLE_API_KEY environment variables."
            )

        if self.concurrency.summary > 0 or self.concurrency.lyrics > 0:
            for tool in self.ai.search.tools:
                clean_tool = tool.strip().lower()
                if clean_tool in ("ollama", "langsearch", "tinyfish"):
                    if self.ai.search.get_key_for_provider(clean_tool) is None:
                        raise ValueError(
                            f"A search API key must be provided for '{clean_tool}' via '--ai.search.{clean_tool}-key', "
                            f"'--ai.search.key', or the {clean_tool.upper()}_API_KEY environment variable."
                        )

        return self

    @model_validator(mode="after")
    def setup_file_locations(self) -> Self:
        """Resolves input, output, and temporary paths to absolute directory locations.

        Returns:
            Self: The validated instance.
        """
        input_video_path = cast(Path, self.input_video_file).resolve()
        self.input_video_file = cast(Any, input_video_path)

        if self.dir.out == Path("directory_of_input_file"):
            self.dir.out = input_video_path.parent

        if self.dir.tmp == Path("tmp_input_video_file"):
            self.dir.tmp = self.dir.out / f"tmp_{input_video_path.stem}"

        self.dir.out = self.dir.out.resolve()
        self.dir.tmp = self.dir.tmp.resolve()
        self.dir.tmp.mkdir(parents=True, exist_ok=True)

        return self
