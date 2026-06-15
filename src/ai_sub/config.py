"""Configuration settings for the AI Sub subtitle generation pipeline."""

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional, cast

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


class GoogleAiSettings(BaseSettings):
    """Configuration for Google Generative AI (GLA) integration."""

    model_config = {
        "env_prefix": "AISUB_AI_GOOGLE_",
        **_BASE_CONFIG,
    }

    file_cache_ttl: PositiveInt = Field(
        description="The time-to-live (TTL) in seconds for the Gemini file list cache. "
        "This cache helps avoid frequent API calls to list uploaded files.",
        default=10,
    )
    key: Optional[SecretStr] = Field(
        description="The API key for Google's generative language models. "
        "Falls back to the GOOGLE_API_KEY or GEMINI_API_KEY environment variables if not set.",
        # We handle default loading from env in the validator
        default=None,
    )
    use_files_api: bool = Field(
        description="Enable the Gemini Files API for cloud-based multimodal processing.",
        default=True,
    )
    base_url: Optional[HttpUrl] = Field(
        description="The base URL for the Google AI API. This can be used to override the default endpoint, "
        "for instance, to use a proxy. If not provided, Google's default URL will be used.",
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def load_api_key_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Loads the API key from environment variables if it's not provided directly.

        Pydantic-settings handles the prefixed env var (AISUB_AI_GOOGLE_KEY),
        but we also want to check for GOOGLE_API_KEY and GEMINI_API_KEY.

        Args:
            values: The raw input values to validate.

        Returns:
            The updated values dictionary including the API key if found.
        """
        # If 'key' is not provided directly, try to load it from standard env vars.
        if (key := values.get("key")) is None:
            if env_key := (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                values["key"] = key = env_key

        # Sanitize the key to remove accidental surrounding quotes or whitespace
        if isinstance(key, str):
            values["key"] = values["key"].strip().strip('"').strip("'")

        return values


class WebSearchSettings(BaseSettings):
    """Unified configuration for web search operations."""

    model_config = {
        "env_prefix": "AISUB_AI_SEARCH_",
        **_BASE_CONFIG,
    }

    key: Optional[SecretStr] = Field(
        description="The API key for the web search API (Ollama or Langsearch). "
        "Falls back to the OLLAMA_API_KEY or LANGSEARCH_API_KEY environment variables.",
        default=None,
    )
    qps: PositiveFloat = Field(
        description="Maximum queries per second for the web search API.",
        default=0.3,
    )
    max_length: PositiveInt = Field(
        description="Discard search responses that are longer than this number of characters.",
        default=4096,
    )
    web_search_tool: Literal["builtin", "duckduckgo", "ollama", "langsearch"] = Field(
        description="The web search tool to use. "
        "Options are 'builtin' (The provider's native search tool, e.g., Google Search for Gemini), "
        "'duckduckgo', 'ollama', or 'langsearch'. "
        "DuckDuckGo is the default because Gemini's built-in search does not have a free tier.",
        default="duckduckgo",
    )
    timeout: PositiveFloat = Field(
        description="The timeout in seconds for web search HTTP requests. "
        "Search queries can occasionally be slow depending on the provider.",
        default=60.0,
    )

    @model_validator(mode="before")
    @classmethod
    def load_api_key_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Loads the API key from environment variables if it's not provided directly.

        Args:
            values: The raw input values to validate.

        Returns:
            The updated values dictionary including the API key if found.
        """
        if (key := values.get("key")) is None:
            # Determine which tool is selected to avoid loading the wrong key.
            # We check both the provided values and the environment directly.
            tool = values.get("web_search_tool") or os.getenv("AISUB_AI_SEARCH_WEB_SEARCH_TOOL") or "duckduckgo"

            env_key = None
            if tool == "ollama":
                env_key = os.getenv("OLLAMA_API_KEY")
            elif tool == "langsearch":
                env_key = os.getenv("LANGSEARCH_API_KEY")

            if env_key:
                values["key"] = key = env_key

        if isinstance(key, str):
            values["key"] = values["key"].strip().strip('"').strip("'")

        return values


class AiSettings(BaseSettings):
    """Global AI model configuration and rate limits."""

    model_config = {
        "env_prefix": "AISUB_AI_",
        **_BASE_CONFIG,
    }

    model: Optional[str] = Field(
        default=None,
        description="A shorthand to set both model_subtitles and model_lyrics to the same value. "
        "If provided, this will override the other two settings.",
    )
    model_subtitles: str = Field(
        description="The AI model for subtitle generation. Use 'google-gla:<model>' for Google models, "
        "'openai:<model>' for OpenAI, or 'custom:<url>' for a custom endpoint.",
        default="google-gla:gemini-3.5-flash",
    )
    model_lyrics: str = Field(
        description="The AI model for lyrics research and scene detection.",
        default="google-gla:gemini-3.1-flash-lite",
    )
    rpm: PositiveInt = Field(description="Maximum Requests Per Minute (RPM) for the AI model provider.", default=4)
    tpm: PositiveInt = Field(description="Maximum Tokens Per Minute (TPM) for the AI model provider.", default=250000)
    timeout: PositiveFloat = Field(
        description="The timeout in seconds for AI model HTTP requests. "
        "This must be at least 10s for Google Gemini models to avoid 'deadline too short' errors.",
        default=300.0,
    )
    google: GoogleAiSettings = Field(
        description="Settings that only apply to the Google AI model.",
        default_factory=GoogleAiSettings,
    )
    search: WebSearchSettings = Field(
        description="Settings for web search operations.",
        default_factory=WebSearchSettings,
    )
    validation_buffer_ms: NonNegativeInt = Field(
        description="The allowed buffer in milliseconds for AI-generated timestamps to exceed the video duration.",
        default=2000,
    )

    @model_validator(mode="after")
    def validate_models(self) -> "AiSettings":
        """Overrides model-specific settings if the global shorthand is set.

        Returns:
            The updated settings instance.
        """
        if self.model:
            self.model_subtitles = self.model
            self.model_lyrics = self.model
        return self

    def get_sanitized_model_name(self, model_name: str) -> str:
        """Sanitizes the model name to be safe for filenames.

        Strips the provider prefix (if any) and replaces non-alphanumeric
        characters with hyphens.

        Args:
            model_name (str): The full model string (e.g. "google-gla:gemini-1.5-pro")

        Returns:
            str: The sanitized model name.
        """
        model_name = model_name.split(":", 1)[-1]
        return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-")


class ReEncodeSettings(BaseSettings):
    """Configuration for video re-encoding parameters."""

    model_config = {
        "env_prefix": "AISUB_SPLIT_RE_ENCODE_",
        **_BASE_CONFIG,
    }

    enabled: bool = Field(
        description="Re-encode the video chunks to save bandwidth.",
        default=False,
    )
    fps: PositiveInt = Field(
        description="The target framerate (FPS) to re-encode the video segments to.",
        default=1,
    )
    height: PositiveInt = Field(
        description="The target height (resolution) to re-encode to. Aspect ratio is preserved.",
        default=360,
    )
    bitrate_kb: PositiveInt = Field(
        description="The target bitrate in Kilobytes per second (KB/s) for the re-encoded video.",
        default=35,
    )
    threshold_mb: NonNegativeInt = Field(
        description="The threshold in MB for re-encoding. Files smaller than this will not be re-encoded. "
        "Set to 0 to re-encode everything.",
        default=20,
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        description="Maximum allowed duration difference (ms) between original and re-encoded segments "
        "to consider the output valid.",
        default=100,
    )
    encoder: Optional[str] = Field(
        description="The specific encoder to use (e.g., 'h264_nvenc', 'libx264'). "
        "If not provided, it will be automatically detected.",
        default=None,
    )


class SplittingSettings(BaseSettings):
    """Parameters for segmenting input video into chunks."""

    model_config = {
        "env_prefix": "AISUB_SPLIT_",
        **_BASE_CONFIG,
    }

    max_seconds: PositiveInt = Field(
        description="The maximum duration in seconds for each video chunk. "
        "The input video will be split into these smaller segments for processing.",
        default=60 * 5,
    )
    re_encode: ReEncodeSettings = Field(
        description="Settings for re-encoding video chunks.",
        default_factory=ReEncodeSettings,
    )
    start_offset_min: NonNegativeInt = Field(
        description="The number of minutes to skip from the beginning of the video.",
        default=0,
    )


class DirectorySettings(BaseSettings):
    """Configuration for input/output and temporary file paths."""

    model_config = {
        "env_prefix": "AISUB_DIR_",
        **_BASE_CONFIG,
    }

    tmp: Path = Field(
        description="Temporary directory for intermediate files (e.g., video segments). "
        "Defaults to a 'tmp_<video_name>' folder in the output directory.",
        default=Path("tmp_input_video_file"),
    )
    out: Path = Field(
        description="Output directory for the final subtitle files. "
        "Defaults to the same directory as the input video file.",
        default=Path("directory_of_input_file"),
    )


class ThreadSettings(BaseSettings):
    """Concurrency limits for various pipeline stages."""

    model_config = {
        "env_prefix": "AISUB_THREAD_",
        **_BASE_CONFIG,
    }

    uploads: PositiveInt = Field(
        description="Number of concurrent uploads to the Gemini Files API.",
        default=4,
    )
    re_encode: PositiveInt = Field(
        description="Number of concurrent FFmpeg re-encoding processes.",
        default=2,
    )
    lyrics: NonNegativeInt = Field(
        description="Number of concurrent AI tasks for lyrics and scene detection. Set to 0 to skip this stage.",
        default=4,
    )
    subtitles: PositiveInt = Field(
        description="Number of concurrent AI tasks for subtitle generation (transcription and translation).",
        default=4,
    )


class RetrySettings(BaseSettings):
    """Job retry logic configuration."""

    model_config = {
        "env_prefix": "AISUB_RETRY_",
        **_BASE_CONFIG,
    }

    per_run: NonNegativeInt = Field(
        description="Maximum internal retries by the AI agent per request. "
        "This handles transient API errors and validation failures within a single job execution.",
        default=5,
    )
    max_runs: NonNegativeInt = Field(
        description="Total attempt limit for a single segment stage (e.g. lyrics detection or subtitle generation) "
        "across all application runs. "
        "This value is persisted in stage-specific state files to prevent infinite retries on problematic segments. "
        "Note that attempts are tracked independently per stage; "
        "failures in one stage do not count against the attempt limit of subsequent stages.",
        default=3,
    )
    multiplier: PositiveFloat = Field(
        description="The multiplier for exponential backoff between retries.",
        default=2.0,
    )
    max_wait_seconds: PositiveInt = Field(
        description="The maximum wait time in seconds (upper bound) for a single retry attempt.",
        default=300,
    )


class LoggingSettings(BaseSettings):
    """Telemetry and console logging configuration."""

    model_config = {
        "env_prefix": "AISUB_LOG_",
        **_BASE_CONFIG,
    }

    level: LevelName = Field(description="The minimum log level to display.", default="info")
    timestamps: bool = Field(
        description="Whether to include timestamps in the console output.",
        default=False,
    )
    scrub: bool = Field(
        description="Whether to scrub sensitive data from logs.",
        default=True,
    )
    progress_bar_width: PositiveInt = Field(
        description="Fixed width for progress bars (in characters).",
        default=80,
    )
    progress_bar_refresh_seconds: PositiveFloat = Field(
        description="Interval in seconds to refresh progress bars to handle terminal resizing.",
        default=1.0,
    )


class Settings(BaseSettings):
    """The root application configuration model."""

    model_config = {
        "nested_model_default_partial_update": True,
        "cli_avoid_json": True,
        "cli_kebab_case": True,
        "env_prefix": "AISUB_",
        **_BASE_CONFIG,
    }

    ai: AiSettings = Field(description="Settings related to the AI model.", default_factory=AiSettings)
    split: SplittingSettings = Field(
        description="Settings for splitting the input video into chunks.",
        default_factory=SplittingSettings,
    )
    dir: DirectorySettings = Field(
        description="Settings for temporary and output directories.",
        default_factory=DirectorySettings,
    )
    thread: ThreadSettings = Field(
        description="Settings for controlling concurrency.",
        default_factory=ThreadSettings,
    )
    retry: RetrySettings = Field(description="Settings for retrying failed jobs.", default_factory=RetrySettings)
    log: LoggingSettings = Field(description="Settings related to logging.", default_factory=LoggingSettings)

    # Position Argument - input file is always the last
    input_video_file: CliPositionalArg[FilePath] = Field(
        description="The path to the video file for which to generate subtitles."
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validates that required API keys are provided for the selected models and tools.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If a required API key is missing.
        """
        is_google_subtitles = self.ai.model_subtitles.lower().startswith("google-gla")
        # Only check lyrics model if lyrics threads are > 0 (i.e. enabled)
        is_google_scene = self.thread.lyrics > 0 and self.ai.model_lyrics.lower().startswith("google-gla")

        if (is_google_subtitles or is_google_scene) and self.ai.google.key is None:
            raise ValueError(
                "A Google AI API key must be provided either via the 'key' field, "
                "GOOGLE_API_KEY, GEMINI_API_KEY or AISUB_AI_GOOGLE_KEY environment variables."
            )

        if self.ai.search.web_search_tool == "ollama" and self.thread.lyrics > 0 and self.ai.search.key is None:
            raise ValueError(
                "An Ollama web search API key must be provided either via the 'key' field, "
                "OLLAMA_API_KEY or AISUB_AI_SEARCH_KEY environment variables "
                "when using 'ollama' as the web search tool. "
                "Register a free key at: https://ollama.com/settings/keys"
            )

        if self.ai.search.web_search_tool == "langsearch" and self.thread.lyrics > 0 and self.ai.search.key is None:
            raise ValueError(
                "A Langsearch web search API key must be provided either via the 'key' field, "
                "LANGSEARCH_API_KEY or AISUB_AI_SEARCH_KEY environment variables "
                "when using 'langsearch' as the web search tool."
            )

        return self

    @model_validator(mode="after")
    def setup_file_locations(self) -> "Settings":
        """Sets up default file locations for output and temporary directories.

        Calculates absolute paths for the input file, output directory, and
        temporary workspace if they are not explicitly provided. It also creates
        the temporary directory.

        Returns:
            The validated settings instance.
        """
        # Resolve input video file to an absolute path first
        input_video_path = cast(Path, self.input_video_file).resolve()
        self.input_video_file = cast(Any, input_video_path)

        # If the user didn't set out_dir, set it automatically
        if self.dir.out == Path("directory_of_input_file"):
            self.dir.out = input_video_path.parent

        # If the user didn't set tmp_dir, set it automatically
        if self.dir.tmp == Path("tmp_input_video_file"):
            self.dir.tmp = self.dir.out / f"tmp_{input_video_path.stem}"

        # Now resolve the directory paths to be absolute
        self.dir.out = self.dir.out.resolve()
        self.dir.tmp = self.dir.tmp.resolve()

        # Create the tmp directory (works for both user-provided and default paths)
        self.dir.tmp.mkdir(parents=True, exist_ok=True)

        return self
