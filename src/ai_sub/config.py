import os
from pathlib import Path
from typing import Optional

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


class GeminiCliSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_AI_GEMINI_CLI_",
    )

    timeout: PositiveInt = Field(
        description="The timeout in seconds for Gemini CLI operations.", default=600
    )


class GoogleAiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_AI_GOOGLE_",
    )

    file_cache_ttl: PositiveInt = Field(
        description="The time-to-live (TTL) in seconds for the Gemini file list cache. This cache helps avoid frequent API calls to list uploaded files.",
        default=10,
    )
    key: Optional[SecretStr] = Field(
        description="The API key for Google's generative language models. If not provided, it will fall back to the GOOGLE_API_KEY or GEMINI_API_KEY environment variables.",
        # We handle default loading from env in the validator
        default=None,
    )
    use_files_api: bool = Field(
        description="Whether to use the Gemini Files API.", default=True
    )
    base_url: Optional[HttpUrl] = Field(
        description="The base URL for the Google AI API. This can be used to override the default endpoint, for instance, to use a proxy. If not provided, Google's default URL will be used.",
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def load_api_key_from_env(cls, values):
        """
        Load the API key from environment variables if it's not provided directly.
        Pydantic-settings handles the prefixed env var (AISUB_AI_GOOGLE_KEY),
        but we also want to check for GOOGLE_API_KEY and GEMINI_API_KEY.
        """
        # If 'key' is not provided directly, try to load it from standard env vars.
        if values.get("key") is None:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                values["key"] = key

        return values


class AiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_AI_",
    )

    model: str = Field(
        description="The AI model for subtitle generation. Use 'google-gla:<model>' for Google models, 'openai:<model>' for OpenAI, or 'custom:<url>' for a custom endpoint.",
        default="google-gla:gemini-3-flash-preview",
    )
    rpm: PositiveInt = Field(
        description="Maximum requests per minute for the AI model.", default=4
    )
    tpm: PositiveInt = Field(
        description="Maximum tokens per minute for the AI model.", default=250000
    )
    google: GoogleAiSettings = Field(
        description="Settings that only apply to the Google AI model.",
        default_factory=GoogleAiSettings,
    )
    gemini_cli: GeminiCliSettings = Field(
        description="Settings that only apply to the Gemini CLI.",
        default_factory=GeminiCliSettings,
    )

    @model_validator(mode="after")
    def validate_google_key(self):
        """
        Validates that a Google AI API key is provided if a Google model is selected.
        """
        if self.model.lower().startswith("google-gla") and self.google.key is None:
            raise ValueError(
                "A Google AI API key must be provided either via the 'key' field, GOOGLE_API_KEY, GEMINI_API_KEY or AISUB_AI_GOOGLE_KEY environment variables."
            )
        return self


class ReEncodeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_SPLIT_RE_ENCODE_",
    )

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
    encoder: Optional[str] = Field(
        description="The specific encoder to use (e.g., 'h264_nvenc', 'libx264'). If not provided, it will be automatically detected.",
        default=None,
    )

    @model_validator(mode="after")
    def detect_encoder(self):
        if self.enabled and not self.encoder:
            from ai_sub.video import get_working_encoder

            self.encoder = get_working_encoder()
        return self


class SplittingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_SPLIT_",
    )

    max_seconds: PositiveInt = Field(
        description="The maximum duration in seconds for each video chunk. The input video will be split into these smaller segments for processing.",
        default=60 * 5,
    )
    re_encode: ReEncodeSettings = Field(
        description="Settings for re-encoding video chunks.",
        default_factory=ReEncodeSettings,
    )


class DirectorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_DIR_",
    )

    tmp: Path = Field(
        description="Temporary directory for intermediate files (e.g., video segments). Defaults to a 'tmp_<video_name>' folder in the output directory.",
        default=Path("tmp_input_video_file"),
    )
    out: Path = Field(
        description="Output directory for the final subtitle files. Defaults to the same directory as the input video file.",
        default=Path("directory_of_input_file"),
    )


class ThreadSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_THREAD_",
    )

    uploads: PositiveInt = Field(
        description="The number of concurrent threads for uploading video segments. This is only used for Gemini (google-gla) models.",
        default=4,
    )
    re_encode: PositiveInt = Field(
        description="The number of concurrent threads for re-encoding video chunks.",
        default=2,
    )
    subtitles: PositiveInt = Field(
        description="The number of concurrent threads to use for generating subtitles.",
        default=4,
    )


class RetrySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AISUB_RETRY_",
    )

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
    model_config = SettingsConfigDict(
        env_prefix="AISUB_LOG_",
    )

    level: LevelName = Field(
        description="The minimum log level to display.", default="info"
    )
    timestamps: bool = Field(
        description="Whether to include timestamps in the console output.",
        default=False,
    )
    scrub: bool = Field(
        description="Whether to scrub sensitive data from logs.",
        default=True,
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        cli_avoid_json=True,
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AISUB_",
    )

    ai: AiSettings = Field(
        description="Settings related to the AI model.", default_factory=AiSettings
    )
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
    retry: RetrySettings = Field(
        description="Settings for retrying failed jobs.", default_factory=RetrySettings
    )
    log: LoggingSettings = Field(
        description="Settings related to logging.", default_factory=LoggingSettings
    )

    # Position Argument - input file is always the last
    input_video_file: CliPositionalArg[FilePath] = Field(
        description="The path to the video file for which to generate subtitles."
    )

    @model_validator(mode="after")
    def setup_file_locations(self):
        """
        Validator that sets up default file locations for output and temporary directories
        if they are not explicitly provided by the user. It also creates the temporary directory.
        """

        # If the user didn't set out_dir, set it automatically
        if self.dir.out == Path("directory_of_input_file"):
            self.dir.out = self.input_video_file.parent

        # If the user didn't set tmp_dir, set it automatically
        if self.dir.tmp == Path("tmp_input_video_file"):
            self.dir.tmp = self.dir.out / f"tmp_{self.input_video_file.stem}"

        # Create the tmp directory (works for both user-provided and default paths)
        self.dir.tmp.mkdir(parents=True, exist_ok=True)

        return self
