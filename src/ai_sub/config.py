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


class GeminiCliSettings(BaseSettings):
    """Settings for the Gemini CLI tool integration."""

    model_config = {
        "env_prefix": "AISUB_AI_GEMINI_CLI_",
        **_BASE_CONFIG,
    }

    timeout: PositiveInt = Field(description="The timeout in seconds for Gemini CLI operations.", default=600)
    overwrite_system_prompt: bool = Field(
        description="Whether to overwrite the system prompt using GEMINI_SYSTEM_MD. "
        "If False, the prompt is passed as a regular file argument.",
        default=False,
    )


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
        "If not provided, it will fall back to the GOOGLE_API_KEY or GEMINI_API_KEY environment variables.",
        # We handle default loading from env in the validator
        default=None,
    )
    use_files_api: bool = Field(description="Whether to use the Gemini Files API.", default=True)
    base_url: Optional[HttpUrl] = Field(
        description="The base URL for the Google AI API. This can be used to override the default endpoint, "
        "for instance, to use a proxy. If not provided, Google's default URL will be used.",
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def load_api_key_from_env(cls, values):
        """Loads the API key from environment variables if it's not provided directly.

        Pydantic-settings handles the prefixed env var (AISUB_AI_GOOGLE_KEY),
        but we also want to check for GOOGLE_API_KEY and GEMINI_API_KEY.

        Args:
            values (dict): The raw input values to validate.

        Returns:
            dict: The updated values dictionary including the API key if found.
        """
        # If 'key' is not provided directly, try to load it from standard env vars.
        if values.get("key") is None:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                values["key"] = key

        # Sanitize the key to remove accidental surrounding quotes or whitespace
        if values.get("key") and isinstance(values["key"], str):
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
        default="google-gla:gemini-3-flash-preview",
    )
    model_lyrics: str = Field(
        description="The AI model for lyrics research and scene detection.",
        default="google-gla:gemini-3-flash-preview",
    )
    rpm: PositiveInt = Field(description="Maximum requests per minute for the AI model.", default=4)
    tpm: PositiveInt = Field(description="Maximum tokens per minute for the AI model.", default=250000)
    web_search_tool: Literal["builtin", "genius"] = Field(
        description="The web search tool to use. "
        "Options are 'builtin' (The provider's native search tool, e.g., Google Search for Gemini) or 'genius'. "
        "Genius is the default because it provides accurate lyric data from the Genius database.",
        default="genius",
    )
    google: GoogleAiSettings = Field(
        description="Settings that only apply to the Google AI model.",
        default_factory=GoogleAiSettings,
    )
    gemini_cli: GeminiCliSettings = Field(
        description="Settings that only apply to the Gemini CLI.",
        default_factory=GeminiCliSettings,
    )
    validation_buffer_ms: NonNegativeInt = Field(
        description="The allowed buffer in milliseconds for AI-generated timestamps to exceed the video duration.",
        default=1000,
    )

    @model_validator(mode="after")
    def validate_models(self):
        """Overrides model-specific settings if the global shorthand is set.

        Returns:
            AiSettings: The updated settings instance.
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
        description="The framerate to re-encode the video to.",
        default=1,
    )
    height: PositiveInt = Field(
        description="The height (resolution) to re-encode the video to. Width is scaled automatically.",
        default=360,
    )
    bitrate_kb: PositiveInt = Field(
        description="The bitrate in KB/s (Kilobytes per second) to re-encode the video to.",
        default=35,
    )
    threshold_mb: NonNegativeInt = Field(
        description="The threshold in MB for re-encoding. Files smaller than this will not be re-encoded. "
        "Set to 0 to re-encode everything.",
        default=20,
    )
    duration_tolerance_ms: NonNegativeInt = Field(
        description="The allowed difference in milliseconds between input and output video duration to "
        "consider a re-encode valid.",
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
        description="The number of concurrent threads for uploading video segments. "
        "This is only used for Gemini (google-gla) models.",
        default=4,
    )
    re_encode: PositiveInt = Field(
        description="The number of concurrent threads for re-encoding video chunks.",
        default=2,
    )
    lyrics: NonNegativeInt = Field(
        description="The number of concurrent threads to use for Lyrics/Scene Detection. "
        "Set to 0 to disable Lyrics/Scene detection.",
        default=4,
    )
    subtitles: PositiveInt = Field(
        description="The number of concurrent threads to use for Subtitle Generation (Transcription).",
        default=4,
    )


class RetrySettings(BaseSettings):
    """Job retry logic configuration."""

    model_config = {
        "env_prefix": "AISUB_RETRY_",
        **_BASE_CONFIG,
    }

    run: NonNegativeInt = Field(
        description="The maximum number of times to retry a failed job in this run of the program.",
        default=3,
    )
    max: NonNegativeInt = Field(
        description="The absolute maximum number of times a job can be retried in total.",
        default=9,
    )
    delay: NonNegativeInt = Field(
        description="The number of seconds to wait between retries.",
        default=30,
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
    def validate_api_keys(self):
        """Validates that a Google AI API key is provided if a Google model is selected.

        Returns:
            Settings: The validated settings instance.

        Raises:
            ValueError: If a Google model is used without an API key.
        """
        is_google_subtitles = self.ai.model_subtitles.lower().startswith("google-gla")
        # Only check lyrics model if lyrics threads are > 0 (i.e. enabled)
        is_google_scene = self.thread.lyrics > 0 and self.ai.model_lyrics.lower().startswith("google-gla")

        if (is_google_subtitles or is_google_scene) and self.ai.google.key is None:
            raise ValueError(
                "A Google AI API key must be provided either via the 'key' field, "
                "GOOGLE_API_KEY, GEMINI_API_KEY or AISUB_AI_GOOGLE_KEY environment variables."
            )
        return self

    @model_validator(mode="after")
    def setup_file_locations(self):
        """Sets up default file locations for output and temporary directories.

        Calculates absolute paths for the input file, output directory, and
        temporary workspace if they are not explicitly provided. It also creates
        the temporary directory.

        Returns:
            Settings: The validated settings instance.
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
