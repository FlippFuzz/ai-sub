"""Shared configuration models, types, and YAML loading utilities for AI Sub."""

from __future__ import annotations

import json
import os
import random
import socket
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from annotated_types import Ge, Gt
from dotenv import dotenv_values
from logfire import LevelName
from pydantic import (
    BaseModel,
    BeforeValidator,
    ByteSize,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

ServiceTier = Literal["standard", "flex", "priority"]
SearchTool = Literal["builtin", "duckduckgo", "ollama", "langsearch", "tinyfish", "lrclib"]

_YAML_PATH: ContextVar[Path | None] = ContextVar("_yaml_path", default=None)

_SEARCH_PROVIDER_KEY_FIELDS: dict[str, tuple[str, str]] = {
    "tinyfish": ("tinyfishkey", "TINYFISH_API_KEY"),
    "langsearch": ("langsearchkey", "LANGSEARCH_API_KEY"),
    "ollama": ("ollamakey", "OLLAMA_API_KEY"),
}


def _get_env_value(*names: str) -> str | None:
    """Reads the first non-empty value from OS environment variables or a local .env file.

    Args:
        *names: Environment variable names in priority order.

    Returns:
        str | None: First matching non-empty variable value, or None.
    """
    for name in names:
        if (val := os.getenv(name)) and (stripped := val.strip()):
            return stripped

    env_path = Path(".env")
    if env_path.is_file():
        dotenv_map = dotenv_values(env_path)
        for name in names:
            if (val := dotenv_map.get(name)) and (stripped := val.strip()):
                return stripped
    return None


def _parse_string_list(val: Any) -> list[str] | Any:
    """Parses a list of strings from None, a Python list, JSON array, or comma-separated string.

    Args:
        val: Raw input value from environment variable, CLI, or YAML.

    Returns:
        list[str] | Any: Normalized list of strings or unparsed raw input.
    """
    if val is None:
        return []
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
            else:
                s = str(m).strip()
                if s:
                    out.append(s)
        return out
    return val


def _parse_list_items(val: Any, transform: Callable[[str], Any]) -> list[Any] | Any:
    """Parses a string list and applies an item-level transformation when parsing succeeds.

    Args:
        val: Raw input value from environment variable, CLI, or YAML.
        transform: Callable converting each parsed string item.

    Returns:
        list[Any] | Any: Transformed list of items or unparsed raw input.
    """
    parsed = _parse_string_list(val)
    return [transform(item) for item in parsed] if isinstance(parsed, list) else parsed


StringList = Annotated[list[str], NoDecode, BeforeValidator(_parse_string_list)]
SearchToolList = Annotated[list[SearchTool], NoDecode, BeforeValidator(lambda val: _parse_list_items(val, str.lower))]
PathList = Annotated[list[Path], NoDecode, BeforeValidator(lambda val: _parse_list_items(val, Path))]


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merges an override dict onto base defaults, ignoring None override entries.

    Args:
        base: Parent default dictionary.
        override: Child override dictionary.

    Returns:
        dict[str, Any]: Combined dictionary preserving parent defaults for omitted or None keys.
    """
    merged = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def merge_section_defaults(val: Any, section_defaults: dict[str, dict[str, Any]]) -> Any:
    """Strips None section entries and deep-merges partial dicts with section-specific defaults.

    Args:
        val: Raw input mapping before Pydantic validation.
        section_defaults: Mapping of field names to section-specific default dictionaries.

    Returns:
        Any: Normalized dictionary with section-specific defaults preserved, or original input.
    """
    if not isinstance(val, dict):
        return val
    updated = dict(val)
    for field_name, defaults in section_defaults.items():
        if field_name in updated:
            raw_section = updated[field_name]
            if raw_section is None:
                updated.pop(field_name)
            elif isinstance(raw_section, dict):
                updated[field_name] = _deep_merge_dicts(defaults, raw_section)
    return updated


def _merge_model_overrides(val: Any, inherited_keys: tuple[str, ...]) -> Any:
    """Normalizes per-model mapping keys and inherits parent defaults into each model entry.

    Args:
        val: Raw input dictionary before Pydantic validation.
        inherited_keys: Parent field names propagated as defaults into each entry under 'models'.

    Returns:
        Any: Normalized dictionary with merged per-model entries, or original input.

    Raises:
        ValueError: If any model key in 'models' is blank or duplicates another after normalization.
    """
    if not isinstance(val, dict) or "models" not in val:
        return val

    raw_models = val["models"]
    if raw_models is None:
        return {**val, "models": {}}
    if not isinstance(raw_models, dict):
        return val

    base_defaults = {k: val[k] for k in inherited_keys if k in val and val[k] is not None}
    normalized_models: dict[str, Any] = {}

    for raw_key, raw_entry in raw_models.items():
        clean_key = str(raw_key).strip().lower()
        if not clean_key:
            raise ValueError("Model override key in 'models' must not be empty.")
        if clean_key in normalized_models:
            raise ValueError(f"Duplicate model key '{clean_key}' in 'models'.")

        if raw_entry is None:
            normalized_models[clean_key] = _deep_merge_dicts(base_defaults, {})
        elif isinstance(raw_entry, dict):
            normalized_models[clean_key] = _deep_merge_dicts(base_defaults, raw_entry)
        else:
            normalized_models[clean_key] = raw_entry

    return {**val, "models": normalized_models}


class BaseConfigModel(BaseModel):
    """Shared base model for nested configuration sections."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_empty_dict(cls, val: Any) -> Any:
        """Normalizes YAML null into an empty dictionary so nested defaults apply.

        Args:
            val: Raw input value before validation.

        Returns:
            Any: Empty dictionary if None, otherwise original value.
        """
        return {} if val is None else val


class YamlFileSettings(BaseSettings):
    """Base settings model supporting standard YAML loading overridden by ENV and .env."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Configures settings source priority with optional runtime YAML path.

        Args:
            settings_cls: Target settings class.
            init_settings: Initial settings kwargs.
            env_settings: Environment variable settings source.
            dotenv_settings: Dotenv settings source.
            file_secret_settings: File secret settings source.

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: Source priority sequence.
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if (yaml_path := _YAML_PATH.get()) is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        """Loads and validates settings from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Self: Validated settings instance.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        token = _YAML_PATH.set(path)
        try:
            return cls()
        finally:
            _YAML_PATH.reset(token)


def find_config_file(kind: Literal["server", "worker"], cli_arg: str | Path | None = None) -> Path:
    """Resolves configuration YAML file path following strict discovery rules.

    Args:
        kind: Configuration type ('server' or 'worker').
        cli_arg: Optional explicit path passed as CLI argument. Defaults to None.

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
        candidates.extend([Path(f"{short_host}.{kind}.yaml"), Path(f"{short_host}.{kind}.yml")])
    candidates.extend([Path(f"{kind}.yaml"), Path(f"{kind}.yml")])

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked_str = ", ".join(repr(str(c)) for c in candidates)
    raise FileNotFoundError(f"No {kind} configuration file found. Checked: {checked_str}.")


def atomic_write_text(path: Path, content: str) -> None:
    """Writes UTF-8 text atomically via a temporary file and os.replace.

    Args:
        path: Destination file path.
        content: UTF-8 string content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    replaced = False
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
        replaced = True
    finally:
        if not replaced and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


class LoggingSettings(BaseConfigModel):
    """Telemetry and console logging configuration."""

    level: LevelName = Field(
        default="info",
        description="Minimum log level to output.",
    )
    timestamps: bool = Field(
        default=False,
        description="Whether to prefix console log lines with timestamps.",
    )
    scrub: bool = Field(
        default=True,
        description="Whether to scrub sensitive API keys and secrets from logs.",
    )
    token: SecretStr | None = Field(
        default=None,
        description="Logfire write token.",
    )

    @model_validator(mode="after")
    def _apply_env_fallbacks(self) -> Self:
        """Populates unprefixed Logfire write token from environment variables or .env when unset.

        Returns:
            Self: Validated instance.
        """
        if self.token is None and (val := _get_env_value("LOGFIRE_TOKEN")):
            self.token = SecretStr(val)
        return self


class StorageSettings(BaseConfigModel):
    """Configuration for Gemini Files API storage lifecycle and cleanup."""

    max: Annotated[ByteSize, Gt(0)] = Field(
        default=ByteSize(15_000_000_000),
        description="High-water storage mark triggering managed file cleanup (e.g. '15GB').",
    )
    target: Annotated[ByteSize, Gt(0)] = Field(
        default=ByteSize(10_000_000_000),
        description="Low-water storage target to delete down to during cleanup (e.g. '10GB').",
    )
    cleanupinterval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(minutes=30),
        description="Minimum cooldown interval between storage cleanup checks (seconds if numeric).",
    )
    pollinterval: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=10),
        description="Polling interval while waiting for Gemini Files API processing (seconds if numeric).",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        """Validates that low-water mark target is strictly less than max.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If target is greater than or equal to max.
        """
        if self.target >= self.max:
            raise ValueError(
                f"Storage low-water mark 'target' ({self.target}) must be strictly "
                f"less than high-water mark 'max' ({self.max})."
            )
        return self


class BackoffSettings(BaseConfigModel):
    """Exponential backoff timing parameters shared across retryable operations."""

    multiplier: PositiveFloat = Field(
        default=2.0,
        description="Exponential backoff multiplier between retry attempts.",
    )
    minwait: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=30),
        description="Minimum backoff wait duration (seconds if numeric).",
    )
    maxwait: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=300),
        description="Maximum backoff wait duration (seconds if numeric).",
    )

    @model_validator(mode="after")
    def _validate_wait_bounds(self) -> Self:
        """Ensures minimum backoff wait does not exceed maximum backoff wait.

        Returns:
            Self: Validated instance.

        Raises:
            ValueError: If minwait exceeds maxwait.
        """
        if self.minwait > self.maxwait:
            raise ValueError(
                f"Backoff 'minwait' ({self.minwait}) must be less than or equal to 'maxwait' ({self.maxwait})."
            )
        return self

    def compute_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Computes jittered exponential backoff delay in seconds for a retry attempt.

        Args:
            attempt: Zero-based retry attempt index.
            retry_after: Optional server-suggested delay in seconds (e.g. from Retry-After header).

        Returns:
            float: Delay duration in seconds with [0, 1) second random jitter.
        """
        exp_backoff = self.minwait.total_seconds() * (float(self.multiplier) ** attempt)
        effective = max(exp_backoff, retry_after or 0.0)
        return min(self.maxwait.total_seconds(), effective) + random.uniform(0, 1)


class RetryPolicy(BaseConfigModel):
    """Resolved retry parameters for API turns, output validation, and cross-run limits."""

    perturn: NonNegativeInt = Field(
        default=5,
        description="Maximum retries per API turn on retryable network, rate-limit, or 5xx errors.",
    )
    validation: NonNegativeInt = Field(
        default=3,
        description="Maximum full-session retries per run when AI output fails schema validation.",
    )
    maxruns: PositiveInt = Field(
        default=3,
        description="Maximum non-retryable failed runs per FPS tier before fallback or exhaustion.",
    )
    backoff: BackoffSettings = Field(
        default_factory=BackoffSettings,
        description="Exponential backoff settings for API and validation retries.",
    )


class RetrySettings(RetryPolicy):
    """Top-level retry policy with optional per-model overrides inheriting defaults."""

    models: dict[str, RetryPolicy] = Field(
        default_factory=dict,
        description="Optional per-model retry overrides inheriting unspecified fields from top-level defaults.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_model_retries(cls, val: Any) -> Any:
        """Normalizes model keys and inherits top-level retry defaults into per-model policies.

        Args:
            val: Raw input dictionary before validation.

        Returns:
            Any: Normalized dictionary with per-model retry defaults merged.
        """
        return _merge_model_overrides(val, ("perturn", "validation", "maxruns", "backoff"))

    def resolve_for_model(self, model_name: str) -> RetryPolicy:
        """Resolves the effective retry policy for a Gemini model, falling back to top-level defaults.

        Args:
            model_name: Gemini model identifier.

        Returns:
            RetryPolicy: Resolved retry policy for the target model.
        """
        clean = model_name.strip().lower()
        if clean in self.models:
            return self.models[clean]
        return RetryPolicy(
            perturn=self.perturn,
            validation=self.validation,
            maxruns=self.maxruns,
            backoff=self.backoff.model_copy(),
        )


def _default_search_tools() -> list[SearchTool]:
    """Returns default search tools list with explicit literal typing.

    Returns:
        list[SearchTool]: Default search tools list.
    """
    return ["duckduckgo", "lrclib"]


class SearchSettings(BaseConfigModel):
    """Configuration for auxiliary web search tools used by the AI model."""

    tools: SearchToolList = Field(
        default_factory=_default_search_tools,
        description="List of search providers to enable.",
    )
    tinyfishkey: SecretStr | None = Field(
        default=None,
        description="Dedicated API key for Tinyfish search.",
    )
    langsearchkey: SecretStr | None = Field(
        default=None,
        description="Dedicated API key for Langsearch.",
    )
    ollamakey: SecretStr | None = Field(
        default=None,
        description="Dedicated API key for Ollama search.",
    )
    maxturns: PositiveInt = Field(
        default=3,
        description="Maximum function-calling search turns permitted per segment.",
    )
    qps: PositiveFloat = Field(
        default=0.3,
        description="Maximum queries per second for web search requests.",
    )
    maxlength: PositiveInt = Field(
        default=4096,
        description="Maximum characters permitted in a single search result snippet or fetched page.",
    )
    timeout: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=60),
        description="Timeout for external search HTTP requests (seconds if numeric).",
    )
    retries: NonNegativeInt = Field(
        default=3,
        description="Maximum retries for search requests on retryable errors.",
    )
    backoff: BackoffSettings = Field(
        default_factory=lambda: BackoffSettings(
            minwait=timedelta(seconds=5),
            maxwait=timedelta(seconds=60),
        ),
        description="Exponential backoff settings for search retries.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_backoff_defaults(cls, val: Any) -> Any:
        """Preserves search-specific backoff defaults when backoff is null or partial.

        Args:
            val: Raw input dictionary before validation.

        Returns:
            Any: Input dictionary with search backoff defaults merged.
        """
        return merge_section_defaults(
            val,
            {
                "backoff": {
                    "minwait": timedelta(seconds=5),
                    "maxwait": timedelta(seconds=60),
                }
            },
        )

    @model_validator(mode="after")
    def _apply_env_fallbacks(self) -> Self:
        """Populates unprefixed standard search API keys from environment variables or .env when unset.

        Returns:
            Self: Validated instance.
        """
        for field_name, env_name in _SEARCH_PROVIDER_KEY_FIELDS.values():
            if getattr(self, field_name) is None and (val := _get_env_value(env_name)):
                setattr(self, field_name, SecretStr(val))
        return self

    def get_key_for_provider(self, provider_name: str) -> SecretStr | None:
        """Resolves the API key for a specific search provider.

        Args:
            provider_name: The provider identifier ('tinyfish', 'langsearch', 'ollama').

        Returns:
            SecretStr | None: The resolved API key or None if not set.
        """
        entry = _SEARCH_PROVIDER_KEY_FIELDS.get(provider_name.strip().lower())
        return getattr(self, entry[0]) if entry is not None else None

    def validate_required_keys(self, env_prefix: str = "AISUB") -> None:
        """Ensures API keys are present for configured search providers that require authentication.

        Args:
            env_prefix: Environment variable prefix for error messages (e.g. 'AISUB' or 'AISUBWORKER').

        Raises:
            ValueError: If a required search API key is missing or blank.
        """
        for tool in self.tools:
            clean = tool.strip().lower()
            if clean in _SEARCH_PROVIDER_KEY_FIELDS:
                key = self.get_key_for_provider(clean)
                if key is None or not key.get_secret_value().strip():
                    field_name, env_name = _SEARCH_PROVIDER_KEY_FIELDS[clean]
                    raise ValueError(
                        f"A search API key must be provided for '{clean}' via 'ai.search.{field_name}', "
                        f"'{env_prefix}_AI_SEARCH_{field_name.upper()}', or the "
                        f"{env_name} environment variable."
                    )


class FallbackSettings(BaseConfigModel):
    """Fallback FPS step-down rules triggered after repeated non-retryable failures."""

    enabled: bool = Field(
        default=True,
        description="Whether fallback FPS re-encoding is enabled.",
    )
    minfps: PositiveFloat = Field(
        default=0.01,
        description="Minimum video FPS floor; below this FPS, fallback switches to 0 fps (audio-only).",
    )
    divisor: Annotated[float, Gt(1.0)] = Field(
        default=4.0,
        description="Divisor applied per tier (base_fps / divisor ** tier).",
    )


class StageReencodeSettings(BaseConfigModel):
    """Re-encoding configuration for a specific pipeline stage (summary or segment)."""

    threshold: Annotated[ByteSize, Ge(0)] = Field(
        default=ByteSize(0),
        description="Minimum file size triggering initial re-encoding (0 disables initial re-encoding).",
    )
    fps: NonNegativeFloat | Literal["keyframes"] = Field(
        default=1.0,
        description="Target framerate in FPS (0 for audio-only), or 'keyframes' to extract native keyframes only.",
    )
    height: PositiveInt = Field(
        default=360,
        description="Target resolution height in pixels (aspect ratio is preserved).",
    )
    bitrate: Annotated[ByteSize, Gt(0)] = Field(
        default=ByteSize(35_000),
        description="Target video bitrate in bytes per second (e.g. '35KB' = 280 kbps).",
    )
    tolerance: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(milliseconds=100),
        description="Allowed duration divergence for re-encoded output (seconds if numeric).",
    )
    fallback: FallbackSettings = Field(
        default_factory=FallbackSettings,
        description="Fallback FPS step-down configuration.",
    )


class ReencodeSettings(BaseConfigModel):
    """Configuration for FFmpeg video re-encoding across pipeline stages."""

    encoder: str | None = Field(
        default=None,
        description="Specific H.264 encoder to use (e.g., 'h264_nvenc', 'libx264'). Auto-detected if null.",
    )
    summary: StageReencodeSettings = Field(
        default_factory=lambda: StageReencodeSettings(
            threshold=ByteSize(300_000_000),
            fps="keyframes",
            fallback=FallbackSettings(minfps=0.005),
        ),
        description="Re-encoding rules for the whole-video summary proxy.",
    )
    segment: StageReencodeSettings = Field(
        default_factory=lambda: StageReencodeSettings(
            threshold=ByteSize(0),
            fps=1.0,
            fallback=FallbackSettings(minfps=0.01),
        ),
        description="Re-encoding rules for individual video chunk segments.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_stage_defaults(cls, val: Any) -> Any:
        """Preserves stage-specific re-encode defaults when summary or segment is null or partial.

        Args:
            val: Raw input dictionary before validation.

        Returns:
            Any: Input dictionary with stage-specific re-encode defaults merged.
        """
        return merge_section_defaults(
            val,
            {
                "summary": {
                    "threshold": ByteSize(300_000_000),
                    "fps": "keyframes",
                    "fallback": {"minfps": 0.005},
                },
                "segment": {
                    "threshold": ByteSize(0),
                    "fps": 1.0,
                    "fallback": {"minfps": 0.01},
                },
            },
        )


class SplittingSettings(BaseConfigModel):
    """Configuration for splitting input videos into uniform segments."""

    duration: Annotated[timedelta, Gt(timedelta(0))] = Field(
        default=timedelta(seconds=300),
        description="Maximum duration for each video chunk (seconds if numeric).",
    )
    offset: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(0),
        description="Duration to skip from the beginning of the video (0 disables; seconds if numeric).",
    )
    tolerance: Annotated[timedelta, Ge(timedelta(0))] = Field(
        default=timedelta(milliseconds=100),
        description="Maximum allowed duration variance per segment during split validation (seconds if numeric).",
    )


class StageExecutionConfig(BaseConfigModel):
    """Per-stage model, tier, and agentic execution configuration."""

    model: str = Field(description="Gemini model identifier (e.g. 'gemini-4-hydrogen').")
    tier: ServiceTier | None = Field(
        default=None,
        description="Optional Google Gemini service tier.",
    )
    agentic: bool = Field(
        default=False,
        description="Whether to enable agentic video understanding for this stage.",
    )


class StageModelsMixin(BaseConfigModel):
    """Shared per-stage model defaults and agentic fallback validator."""

    summary: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        description="Model, tier, and agentic settings for whole-video summary.",
    )
    subtitles: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-4-hydrogen", agentic=False),
        description="Model, tier, and agentic settings for subtitle transcription.",
    )
    lyrics: StageExecutionConfig = Field(
        default_factory=lambda: StageExecutionConfig(model="gemini-3.5-flash-lite", agentic=True),
        description="Model, tier, and agentic settings for lyrics refinement.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_stage_agentic_defaults(cls, val: Any) -> Any:
        """Strips null stage entries and populates default models and agentic flags when omitted.

        Args:
            val: Raw stage model settings dictionary before validation.

        Returns:
            Any: Normalized dictionary with null stages stripped and stage-appropriate defaults.
        """
        return merge_section_defaults(
            val,
            {
                "summary": {"model": "gemini-3.5-flash-lite", "agentic": True},
                "subtitles": {"model": "gemini-4-hydrogen"},
                "lyrics": {"model": "gemini-3.5-flash-lite", "agentic": True},
            },
        )


class RateLimitPolicy(BaseConfigModel):
    """Per-minute request and token rate limit configuration."""

    rpm: PositiveInt = Field(
        default=4,
        description="Maximum Requests Per Minute (RPM) allowed per model.",
    )
    tpm: PositiveInt = Field(
        default=250000,
        description="Maximum Tokens Per Minute (TPM) allowed per model.",
    )


class AiSettings(RateLimitPolicy):
    """Configuration for the Google Gemini execution engine, rate limits, storage, and search."""

    key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key.",
    )
    baseurl: HttpUrl | None = Field(
        default=None,
        description="Optional custom base URL or proxy for Google Gemini API calls.",
    )
    models: dict[str, RateLimitPolicy] = Field(
        default_factory=dict,
        description="Optional per-model RPM/TPM rate limit overrides inheriting defaults from 'ai'.",
    )
    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description="Files API storage thresholds and cleanup rules.",
    )
    search: SearchSettings = Field(
        default_factory=SearchSettings,
        description="Web search configuration for lyrics and metadata retrieval.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_model_limits(cls, val: Any) -> Any:
        """Normalizes model keys and inherits top-level rpm/tpm defaults into per-model rate limits.

        Args:
            val: Raw input dictionary before validation.

        Returns:
            Any: Normalized dictionary with per-model rate limit defaults merged.
        """
        return _merge_model_overrides(val, ("rpm", "tpm"))

    @model_validator(mode="after")
    def _apply_env_fallbacks(self) -> Self:
        """Populates unprefixed Gemini API key and base URL from environment variables or .env when unset.

        Returns:
            Self: Validated instance.
        """
        if self.key is None and (val := _get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")):
            self.key = SecretStr(val)
        if self.baseurl is None and (val := _get_env_value("GEMINI_BASE_URL")):
            self.baseurl = HttpUrl(val)
        return self

    def resolve_rate_limits(self, model_name: str) -> tuple[int, int]:
        """Resolves effective (rpm, tpm) limits for a Gemini model, falling back to top-level defaults.

        Args:
            model_name: Gemini model identifier.

        Returns:
            tuple[int, int]: Effective (rpm, tpm) limits for the target model.
        """
        clean = model_name.strip().lower()
        if clean in self.models:
            entry = self.models[clean]
            return entry.rpm, entry.tpm
        return self.rpm, self.tpm
