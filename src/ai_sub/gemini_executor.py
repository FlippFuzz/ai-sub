"""Unified execution engine for Google Gemini models and Files API management using the Interactions API."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import mimetypes
import os
import re
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, TypeVar

import json_repair
import logfire
from google import genai
from google.genai import types
from httpx2 import TimeoutException, TransportError
from pydantic import BaseModel, ValidationError
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import AiSettings, RetrySettings, ServiceTier
from ai_sub.jobs.job_base import Prompt
from ai_sub.search.search_base import BaseSearchProvider

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

# Matches managed segment chunks, re-encoded outputs, and summary proxies
MANAGED_PATTERN = re.compile(r"^(?:reencoded_)?(?:part_\d{3,}|summary)_[A-Za-z0-9_-]+\.[^./]+$")

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Transient/server Interactions API error codes (plus standard gRPC aliases); all other codes are non-retryable
_RETRYABLE_INTERACTION_CODES = frozenset(
    {
        "api_error",
        "internal",
        "internal_server_error",
        "service_unavailable",
        "unavailable",
        "deadline_exceeded",
        "gateway_timeout",
        "request_timeout",
        "bad_gateway",
        "rate_limit_exceeded",
        "too_many_requests",
        "resource_exhausted",
        "aborted",
    }
)


class QuotaExceededError(Exception):
    """Raised when an AI model provider's daily quota limit is exhausted."""


class InteractionFailedError(Exception):
    """Raised when a Gemini interaction terminates in a failed or cancelled state."""

    retryable: bool

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        """Initializes the interaction failure exception.

        Args:
            message: Diagnostic failure description.
            retryable: Whether the failure is transient/server-side and eligible for retries.
        """
        super().__init__(message)
        self.retryable = retryable


def _extract_status_code(e: Exception) -> int | None:
    """Extracts an HTTP status code from an exception if available.

    Args:
        e: Caught exception.

    Returns:
        int | None: HTTP status code integer or None.
    """
    for candidate in (
        getattr(e, "status_code", None),
        getattr(e, "code", None),
        getattr(getattr(e, "response", None), "status_code", None),
        getattr(getattr(e, "raw_response", None), "status_code", None),
        getattr(getattr(e, "http_res", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    if not isinstance(e, (ValidationError, ValueError, TypeError, KeyError, InteractionFailedError)):
        if match := re.search(r"\b(408|429|500|502|503|504)\b", str(e)):
            return int(match.group(1))
    return None


def _classify_api_outcome(err: Exception, quota_err: bool, status_code: int | None) -> str:
    """Classifies a failed Gemini API attempt into a canonical outcome tag.

    Args:
        err: Raised exception.
        quota_err: Whether daily project quota was exhausted.
        status_code: Extracted HTTP status code if available.

    Returns:
        str: Canonical outcome string for telemetry grouping.
    """
    if quota_err:
        return "quota_exceeded"
    if isinstance(err, InteractionFailedError):
        return "interaction_failed"
    if isinstance(err, (TimeoutException, asyncio.TimeoutError, TimeoutError)) or status_code == 408:
        return "timeout"
    if status_code is not None:
        return "http_error"
    return "error"


def is_retryable_error(e: Exception) -> bool:
    """Determines if an error is a retryable network, server, or interaction failure.

    Args:
        e: Caught exception.

    Returns:
        bool: True if error can be retried, False otherwise.
    """
    # Deterministic validation or client programming errors are never retryable
    if isinstance(e, (ValidationError, ValueError, TypeError, KeyError)):
        return False

    if _is_daily_quota_exceeded(e):
        return False

    if isinstance(e, InteractionFailedError):
        return e.retryable

    if isinstance(e, (TransportError, TimeoutException, ConnectionError, asyncio.TimeoutError, TimeoutError)):
        return True

    status_code = _extract_status_code(e)
    if status_code is not None:
        return status_code in _RETRYABLE_STATUS_CODES

    err_str = str(e).lower()
    return any(term in err_str for term in ("connection", "timeout", "temporarily unavailable"))


def _get_unique_display_name(file_path: Path) -> str:
    """Constructs a unique display name using workspace folder and filename without dereferencing symlinks.

    Args:
        file_path: Path to the local file.

    Returns:
        str: Display name formatted as 'workspace/filename.ext'.
    """
    # Use abspath instead of resolve() so symlinked summary_<tag> proxies retain their managed name.
    abs_path = Path(os.path.abspath(file_path))
    safe_name = urllib.parse.quote(abs_path.name, safe=".-_")
    safe_parent = urllib.parse.quote(abs_path.parent.name, safe=".-_")
    return f"{safe_parent}/{safe_name}"


def _get_file_timestamp(remote_file: types.File) -> float:
    """Extracts a numeric timestamp in seconds for ordering files chronologically.

    Args:
        remote_file: Remote File object from Gemini API.

    Returns:
        float: Unix timestamp in seconds.
    """
    ts = remote_file.expiration_time or remote_file.create_time
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return 0.0


def _calculate_tokens(text_prompt: Prompt, video_duration_ms: int) -> int:
    """Estimates tokens consumed by text prompts and video duration for rate limiting.

    Args:
        text_prompt: Combined system and user prompt.
        video_duration_ms: Video segment duration in milliseconds.

    Returns:
        int: Estimated token count.
    """
    return len(text_prompt.system) + len(text_prompt.user) + int((video_duration_ms / 1000) * 300)


def _is_daily_quota_exceeded(e: Exception) -> bool:
    """Checks whether an API error represents Google Free Tier daily request limit exhaustion.

    Args:
        e: Caught exception.

    Returns:
        bool: True if daily project quota is exhausted, False otherwise.
    """
    err_str = str(e).lower()
    if "requests per day on free tier" in err_str:
        return True
    if e.__cause__ and "requests per day on free tier" in str(e.__cause__).lower():
        return True
    return False


def _extract_thought_text(step: Any) -> str | None:
    """Extracts reasoning text from a thought step in an interaction response.

    Args:
        step: Step object from interaction.steps.

    Returns:
        str | None: Extracted thought text or None.
    """
    summary = getattr(step, "summary", None)
    if summary:
        if isinstance(summary, list):
            parts_clean: list[str] = []
            for b in summary:
                text: Any = None
                if isinstance(b, str):
                    text = b
                elif isinstance(b, dict):
                    text = b.get("text")
                elif hasattr(b, "text"):
                    text = getattr(b, "text", None)

                if isinstance(text, str) and text.strip():
                    parts_clean.append(text.strip())

            if parts_clean:
                return "\n".join(parts_clean)
        elif isinstance(summary, str) and summary.strip():
            return summary.strip()

    if getattr(step, "text", None) and isinstance(step.text, str) and step.text.strip():
        return step.text.strip()

    return None


def _inspect_interaction_failure(interaction: Any, status: str | None) -> tuple[str, bool]:
    """Extracts error descriptions and retryability from a failed or cancelled interaction.

    Args:
        interaction: Interaction response object.
        status: Terminal interaction status string (e.g. 'failed' or 'cancelled').

    Returns:
        tuple[str, bool]: (formatted_error_details, is_retryable).
    """
    errors = getattr(interaction, "errors", None) or getattr(interaction, "error", None)
    if not errors:
        return "No error details provided.", status != "cancelled"

    items = errors if isinstance(errors, list) else [errors]
    parts: list[str] = []
    saw_code = False
    all_codes_retryable = True

    for item in items:
        if isinstance(item, dict):
            code, msg = item.get("code"), item.get("message")
        else:
            code, msg = getattr(item, "code", None), getattr(item, "message", None)

        if isinstance(code, int) and not isinstance(code, bool):
            saw_code = True
            if code not in _RETRYABLE_STATUS_CODES:
                all_codes_retryable = False
        elif code is not None:
            saw_code = True
            norm_code = str(code).strip().rsplit("/", 1)[-1].lower().replace("-", "_")
            if norm_code not in _RETRYABLE_INTERACTION_CODES:
                all_codes_retryable = False

        text = f"{code}: {msg}" if code and msg else (msg or code or str(item))
        if text:
            parts.append(str(text))

    details = "; ".join(parts) or str(errors)
    retryable = all_codes_retryable if saw_code else (status != "cancelled")
    return details, retryable


def calculate_sha256_hex(file_path: Path) -> str:
    """Calculates the SHA256 hex digest of a file using Python 3.11+ C file_digest.

    Args:
        file_path: Path to the target file.

    Returns:
        str: Lowercase 64-character SHA256 hex digest string.
    """
    with open(file_path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _hashes_match(remote_hash: str | None, local_hex: str) -> bool:
    """Checks if a remote file's SHA256 hash matches a local file's hex digest.

    Args:
        remote_hash: The SHA256 hash string from Gemini Files API.
        local_hex: The calculated local file SHA256 hex digest.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    if not remote_hash:
        return False
    if remote_hash.lower() == local_hex.lower():
        return True
    try:
        if remote_hash == base64.b64encode(bytes.fromhex(local_hex)).decode():
            return True
        return remote_hash == base64.b64encode(local_hex.encode()).decode()
    except Exception:
        return False


class GeminiExecutor:
    """Unified client orchestrating Gemini Files API uploads, quota tracking, and Interactions generation."""

    ai_settings: AiSettings
    default_retry_settings: RetrySettings
    _client: genai.Client
    _quota_exhausted: dict[str, bool]
    _request_limiters: dict[str, Limiter]
    _token_limiters: dict[str, Limiter]
    _last_cleanup_time: float
    _cleanup_lock: asyncio.Lock

    def __init__(self, ai_settings: AiSettings, retry_settings: RetrySettings | None = None) -> None:
        """Initializes the Gemini executor with disabled SDK retries for custom orchestration.

        Args:
            ai_settings: AI engine configuration settings.
            retry_settings: Optional default retry policy. Defaults to RetrySettings().

        Raises:
            ValueError: If the Google Gemini API key is missing.
        """
        self.ai_settings = ai_settings
        self.default_retry_settings = retry_settings or RetrySettings()

        if self.ai_settings.key is None:
            raise ValueError("Google Gemini API key is missing.")

        self._quota_exhausted = {}
        self._request_limiters = {}
        self._token_limiters = {}
        self._last_cleanup_time = 0.0
        self._cleanup_lock = asyncio.Lock()

        api_key = self.ai_settings.key.get_secret_value()
        retry_options = types.HttpRetryOptions(attempts=1)
        http_kwargs: dict[str, Any] = {"retry_options": retry_options}
        if self.ai_settings.baseurl:
            http_kwargs["base_url"] = str(self.ai_settings.baseurl)

        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http_kwargs))

    async def aclose(self) -> None:
        """Closes underlying Gemini API HTTP transports and rate limiters."""
        for limiter in (*self._request_limiters.values(), *self._token_limiters.values()):
            limiter.close()
        self._request_limiters.clear()
        self._token_limiters.clear()
        try:
            await self._client.aio.aclose()
        finally:
            self._client.close()

    def _get_limiters(self, model_id: str) -> tuple[Limiter, Limiter]:
        """Retrieves or creates rate limiters for a specific model ID.

        Args:
            model_id: Model identifier string.

        Returns:
            tuple[Limiter, Limiter]: Request rate limiter and token rate limiter.
        """
        if model_id not in self._request_limiters:
            self._request_limiters[model_id] = limiter_factory.create_inmemory_limiter(
                rate_per_duration=self.ai_settings.rpm, duration=Duration.MINUTE
            )
            self._token_limiters[model_id] = limiter_factory.create_inmemory_limiter(
                rate_per_duration=self.ai_settings.tpm, duration=Duration.MINUTE
            )
        return self._request_limiters[model_id], self._token_limiters[model_id]

    async def _retry_operation(
        self,
        operation: Callable[[int, int], Awaitable[R]],
        retry: RetrySettings,
        op_label: str,
        model_id: str | None = None,
    ) -> R:
        """Executes an API operation with exponential backoff retries on retryable errors.

        Args:
            operation: Async callable receiving (attempt_number, max_attempts) and executing the request.
            retry: Effective retry policy.
            op_label: Descriptive operation label for log messages.
            model_id: Optional model ID to mark quota-exhausted on daily quota errors. Defaults to None.

        Returns:
            R: Operation return value.

        Raises:
            QuotaExceededError: If daily project quota is exhausted.
            RuntimeError: If retries are unexpectedly exhausted without returning or raising.
        """
        max_attempts = retry.perturn + 1
        for attempt in range(max_attempts):
            try:
                return await operation(attempt + 1, max_attempts)
            except QuotaExceededError:
                raise
            except Exception as err:
                if _is_daily_quota_exceeded(err):
                    if model_id:
                        self._quota_exhausted[model_id] = True
                        logfire.warning(f"Daily quota exhausted for {model_id}: {err}")
                    raise QuotaExceededError(str(err)) from err

                if is_retryable_error(err) and attempt < retry.perturn:
                    backoff = retry.backoff.compute_delay(attempt)
                    logfire.warning(
                        f"Retryable error during {op_label} "
                        f"(attempt {attempt + 1}/{max_attempts}): {err}. Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                logfire.error(f"{op_label} failed permanently: {err}")
                raise

        raise RuntimeError(f"{op_label} failed after {max_attempts} attempts.")

    async def _cleanup_storage_if_needed(self) -> None:
        """Deletes oldest managed segment files if total storage exceeds high-water mark."""
        now = time()
        cooldown_seconds = self.ai_settings.storage.cleanupinterval.total_seconds()
        if (now - self._last_cleanup_time) < cooldown_seconds:
            return

        async with self._cleanup_lock:
            if (now - self._last_cleanup_time) < cooldown_seconds:
                return

            with logfire.span("Storage cleanup check", _level="debug"):
                max_bytes = int(self.ai_settings.storage.max)
                target_bytes = int(self.ai_settings.storage.target)

                try:
                    files = [
                        f
                        async for f in await self._client.aio.files.list(config=types.ListFilesConfig(page_size=100))
                        if f.name
                    ]
                    total_bytes = sum(f.size_bytes or 0 for f in files)
                    if total_bytes >= max_bytes:
                        logfire.info(
                            f"Files API storage threshold reached ({total_bytes / 1_000_000_000:.2f} GB used). "
                            f"Cleaning down to {target_bytes / 1_000_000_000:.2f} GB..."
                        )
                        managed = sorted(
                            [f for f in files if f.display_name and MANAGED_PATTERN.match(Path(f.display_name).name)],
                            key=_get_file_timestamp,
                        )
                        for f in managed:
                            if total_bytes <= target_bytes:
                                break
                            if f.name:
                                try:
                                    logfire.debug(f"Deleting remote segment: {f.display_name} ({f.name})")
                                    await self._client.aio.files.delete(name=f.name)
                                    total_bytes -= f.size_bytes or 0
                                except Exception as e:
                                    logfire.debug(f"File delete failed for {f.name}: {e}")

                except Exception as e:
                    logfire.warning(f"Storage cleanup check failed: {e}")
                finally:
                    self._last_cleanup_time = time()

    async def ensure_file_uploaded(
        self,
        file_path: Path,
        retry_settings: RetrySettings | None = None,
    ) -> types.File:
        """Verifies remote file presence, performing upload and status polling as needed.

        Args:
            file_path: Local path to the segment file.
            retry_settings: Optional retry policy for retryable Files API errors. Defaults to None.

        Returns:
            types.File: Active remote file handle from the Gemini Files API.

        Raises:
            RuntimeError: If file processing fails on the server or status cannot be retrieved.
        """
        retry = retry_settings or self.default_retry_settings

        with logfire.span("Gemini Files API: {filename}", filename=file_path.name, _level="debug"):
            await self._cleanup_storage_if_needed()

            display_name = _get_unique_display_name(file_path)
            local_size = file_path.stat().st_size
            local_hex_cache: str | None = None

            async def _get_local_hex() -> str:
                nonlocal local_hex_cache
                if local_hex_cache is None:
                    local_hex_cache = await asyncio.to_thread(calculate_sha256_hex, file_path)
                return local_hex_cache

            existing_active: types.File | None = None
            existing_processing: types.File | None = None

            try:
                with logfire.span("Check existing remote file", _level="debug"):
                    async for remote_file in await self._client.aio.files.list(
                        config=types.ListFilesConfig(page_size=100)
                    ):
                        if remote_file.display_name != display_name:
                            continue

                        if remote_file.state == types.FileState.FAILED and remote_file.name:
                            try:
                                await self._client.aio.files.delete(name=remote_file.name)
                            except Exception:
                                pass
                            continue

                        if remote_file.state in (types.FileState.ACTIVE, types.FileState.PROCESSING):
                            if remote_file.size_bytes == local_size and _hashes_match(
                                remote_file.sha256_hash, await _get_local_hex()
                            ):
                                if remote_file.state == types.FileState.ACTIVE:
                                    existing_active = remote_file
                                    break
                                if existing_processing is None:
                                    existing_processing = remote_file
            except Exception as e:
                logfire.debug(f"Remote file search error for {file_path.name}: {e}")

            if existing_active:
                logfire.info(f"Reusing existing remote file: {file_path.name} ({existing_active.name})")
                return existing_active

            poll_interval = self.ai_settings.storage.pollinterval.total_seconds()
            if existing_processing:
                uploaded = existing_processing
            else:
                upload_path = file_path
                temp_link: Path | None = None

                if not file_path.name.isascii():
                    temp_link = file_path.parent / f"_upload_temp_{uuid.uuid4().hex}{file_path.suffix}"
                    try:
                        os.link(file_path, temp_link)
                        upload_path = temp_link
                    except OSError:
                        try:
                            os.symlink(file_path, temp_link)
                            upload_path = temp_link
                        except OSError:
                            pass

                content_type, _ = mimetypes.guess_type(str(file_path))
                resolved_mime = content_type or "video/mp4"

                try:
                    with logfire.span("Uploading {filename} to Gemini Files API", filename=file_path.name):
                        uploaded = await self._retry_operation(
                            operation=lambda _a, _m: self._client.aio.files.upload(
                                file=upload_path,
                                config=types.UploadFileConfig(display_name=display_name, mime_type=resolved_mime),
                            ),
                            retry=retry,
                            op_label=f"upload '{file_path.name}'",
                        )
                finally:
                    if temp_link and temp_link.exists():
                        try:
                            temp_link.unlink()
                        except OSError:
                            pass

            if uploaded.state != types.FileState.ACTIVE:
                if uploaded.state == types.FileState.FAILED:
                    raise RuntimeError(f"File {file_path.name} failed to process on Gemini server.")
                with logfire.span("Waiting for Gemini to process {filename}", filename=file_path.name):
                    while uploaded.state != types.FileState.ACTIVE:
                        if uploaded.state == types.FileState.FAILED:
                            raise RuntimeError(f"File {file_path.name} failed to process on Gemini server.")
                        remote_name = uploaded.name
                        if not remote_name:
                            raise RuntimeError(f"Could not retrieve file '{file_path.name}' status.")
                        await asyncio.sleep(poll_interval)
                        uploaded = await self._retry_operation(
                            operation=lambda _a, _m, name=remote_name: self._client.aio.files.get(name=name),
                            retry=retry,
                            op_label=f"poll file '{file_path.name}'",
                        )

            return uploaded

    def _setup_tools(
        self,
        tools: Sequence[Callable[..., Any] | dict[str, Any] | BaseSearchProvider] | None,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Callable[..., Any]]]:
        """Builds Interactions API tool configurations and local dispatch mapping.

        Args:
            tools: Sequence of callable tools, Interactions tool dicts, or BaseSearchProvider instances.

        Returns:
            tuple[list[dict[str, Any]] | None, dict[str, Callable[..., Any]]]:
                Registered Interactions tools list and execution dispatch dictionary.
        """
        if not tools:
            return None, {}

        dispatch_map: dict[str, Callable[..., Any]] = {}
        interaction_tools: list[dict[str, Any]] = []
        # FunctionDeclaration.from_callable is typed against BaseApiClient and is needed for use_json_schema=True.
        api_client = self._client._api_client

        items: list[Callable[..., Any] | dict[str, Any]] = []
        for item in tools:
            if isinstance(item, BaseSearchProvider):
                items.extend(item.get_tools())
            else:
                items.append(item)

        for item in items:
            if isinstance(item, dict):
                interaction_tools.append(item)
            elif callable(item):
                fn_name = getattr(item, "__name__", str(item))
                decl = types.FunctionDeclaration.from_callable(
                    client=api_client,
                    callable=item,
                    use_json_schema=True,
                )
                tool_def: dict[str, Any] = {
                    "type": "function",
                    "name": decl.name or fn_name,
                    "description": decl.description or "",
                }
                if decl.parameters_json_schema:
                    tool_def["parameters"] = decl.parameters_json_schema
                interaction_tools.append(tool_def)
                dispatch_map[decl.name or fn_name] = item

        return interaction_tools or None, dispatch_map

    async def _create_interaction(
        self,
        model_id: str,
        input_payload: Any,
        previous_interaction_id: str | None,
        system_instruction: str,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        generation_config: dict[str, Any],
        service_tier: ServiceTier | None,
        token_weight: int,
        task_label: str = "Generation",
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> Any:
        """Executes a single search turn call to Gemini Interactions API with rate limiter acquisition.

        Args:
            model_id: Target Gemini model identifier.
            input_payload: Current search turn input.
            previous_interaction_id: Previous interaction ID for stateful turns.
            system_instruction: System instruction prompt.
            tools: Registered Interactions API tools for this turn.
            response_format: Top-level response format schema.
            generation_config: Generation configuration settings.
            service_tier: Optional Gemini service tier.
            token_weight: Token weight for rate limiter acquisition.
            task_label: Task category label for telemetry attribution. Defaults to 'Generation'.
            attempt: 1-based retry attempt index. Defaults to 1.
            max_attempts: Maximum attempts configured for this turn. Defaults to 1.

        Returns:
            Any: Interaction response object returned by the Gemini API.

        Raises:
            QuotaExceededError: If daily project quota is exhausted.
            InteractionFailedError: If the interaction status is 'failed' or 'cancelled'.
        """
        if self._quota_exhausted.get(model_id, False):
            raise QuotaExceededError(f"Daily quota previously exhausted for model '{model_id}'.")

        req_limiter, tok_limiter = self._get_limiters(model_id)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "input": input_payload,
            "system_instruction": system_instruction,
            "generation_config": generation_config,
            "store": True,
        }
        if previous_interaction_id is not None:
            kwargs["previous_interaction_id"] = previous_interaction_id
        if tools is not None:
            kwargs["tools"] = tools
        elif previous_interaction_id is not None:
            # Explicitly disallow tool calling on subsequent turns when tools are disabled
            kwargs["tools"] = []
        if response_format is not None:
            kwargs["response_format"] = response_format
        if service_tier is not None:
            kwargs["service_tier"] = service_tier

        # Acquire token budget before request slot so a long TPM wait does not hold and expire an RPM slot
        # early before the HTTP request actually fires. Full video tokens are conservatively charged on
        # every API turn (including continuations and retries) to protect daily free-tier request quotas.
        await tok_limiter.try_acquire_async("tpm", weight=token_weight)
        await req_limiter.try_acquire_async("rpm")

        raised_err: Exception | None = None
        with logfire.span(
            "Gemini API: {model_id} ({attempt}/{max_attempts})",
            model_id=model_id,
            attempt=attempt,
            max_attempts=max_attempts,
            task=task_label,
            service_tier=service_tier or "default",
            _level="debug",
        ) as api_span:
            try:
                interaction = await self._client.aio.interactions.create(**kwargs)
                status = getattr(interaction, "status", None)
                if status in ("failed", "cancelled"):
                    details, retryable = _inspect_interaction_failure(interaction, status)
                    raise InteractionFailedError(
                        f"Interaction {status} on {model_id}: {details}",
                        retryable=retryable,
                    )
                api_span.set_attribute("outcome", "success")
                api_span.set_attribute("status_code", 200)
                api_span.set_attribute("is_retryable", False)
                return interaction
            except Exception as err:
                status_code = _extract_status_code(err)
                quota_err = _is_daily_quota_exceeded(err)
                retryable = False if quota_err else is_retryable_error(err)
                api_span.set_attribute("outcome", _classify_api_outcome(err, quota_err, status_code))
                if status_code is not None:
                    api_span.set_attribute("status_code", status_code)
                api_span.set_attribute("error_type", type(err).__name__)
                api_span.set_attribute("is_retryable", retryable)
                api_span.set_level("warning" if (retryable or quota_err) else "error")
                api_span.record_exception(err)
                raised_err = err

        assert raised_err is not None
        raise raised_err

    async def _execute_function_calls(
        self,
        steps: list[Any],
        dispatch_map: dict[str, Callable[..., Any]],
    ) -> list[dict[str, Any]]:
        """Dispatches model-requested tool calls concurrently and formats function_result blocks.

        Args:
            steps: List of step objects from the interaction response.
            dispatch_map: Map of tool names to their registered callables.

        Returns:
            list[dict[str, Any]]: List of serialized function_result dictionaries for the next turn.
        """

        async def _exec(step: Any) -> dict[str, Any]:
            cname = getattr(step, "name", "")
            call_id = getattr(step, "id", "")
            fn = dispatch_map[cname]
            args = getattr(step, "arguments", {}) or {}
            try:
                cres = await fn(**args) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, **args)
                res_str = json.dumps(cres, ensure_ascii=False) if not isinstance(cres, str) else cres
                return {
                    "type": "function_result",
                    "name": cname,
                    "call_id": call_id,
                    "result": [{"type": "text", "text": res_str}],
                }
            except Exception as tool_err:
                logfire.warning(f"Tool '{cname}' failed: {tool_err}")
                err_str = json.dumps({"error": f"Tool execution failed: {tool_err}"})
                return {
                    "type": "function_result",
                    "name": cname,
                    "call_id": call_id,
                    "result": [{"type": "text", "text": err_str}],
                }

        return await asyncio.gather(*(_exec(s) for s in steps))

    def _extract_interaction_text(self, interaction: Any) -> str:
        """Extracts candidate output text from an interaction response or its constituent steps.

        Args:
            interaction: Interaction response object.

        Returns:
            str: Normalized output text.
        """
        if text_content := getattr(interaction, "output_text", None):
            return str(text_content).strip()

        parts: list[str] = []
        for step in getattr(interaction, "steps", []) or []:
            if getattr(step, "type", None) == "model_output":
                for block in getattr(step, "content", []) or []:
                    if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                        parts.append(str(block.text))

        return "".join(parts).strip()

    def _parse_and_validate_structured_output(
        self,
        text_content: str,
        response_type: type[T],
        video_duration_ms: int,
        validation_buffer_ms: int,
        accumulated_thoughts: list[str],
        validation_context: dict[str, Any] | None = None,
    ) -> T:
        """Parses output text into the target Pydantic schema, applying json_repair only on JSON decode errors.

        Args:
            text_content: Raw text returned by the model.
            response_type: Target Pydantic model class.
            video_duration_ms: Duration in milliseconds for timestamp context validation.
            validation_buffer_ms: Allowed subtitle end timestamp overflow buffer in milliseconds.
            accumulated_thoughts: Collected reasoning traces to attach to the output.
            validation_context: Optional extra context merged into Pydantic validation. Defaults to None.

        Returns:
            T: Validated Pydantic model instance.
        """
        merged_context: dict[str, Any] = {
            **(validation_context or {}),
            "video_duration_ms": video_duration_ms,
            "validation_buffer_ms": validation_buffer_ms,
        }

        try:
            parsed = json.loads(text_content)
        except json.JSONDecodeError:
            parsed = json_repair.loads(text_content)

        output = response_type.model_validate(parsed, context=merged_context)

        if hasattr(output, "thoughts") and accumulated_thoughts:
            setattr(output, "thoughts", accumulated_thoughts)

        return output

    async def _run_interaction_session(
        self,
        model_id: str,
        initial_input: Any,
        prompt: Prompt,
        response_type: type[T],
        gemini_tools: list[dict[str, Any]] | None,
        dispatch_map: dict[str, Callable[..., Any]],
        response_format: dict[str, Any],
        generation_config: dict[str, Any],
        service_tier: ServiceTier | None,
        token_weight: int,
        label: str,
        media_name: str,
        video_duration_ms: int,
        validation_buffer_ms: int,
        retry: RetrySettings,
        validation_context: dict[str, Any] | None = None,
    ) -> T:
        """Executes a single end-to-end multi-turn interaction session until final model output is validated.

        Args:
            model_id: Gemini model identifier.
            initial_input: Initial multimodal payload.
            prompt: Prompt container with system and user instructions.
            response_type: Target Pydantic model response schema.
            gemini_tools: Tool declarations registered with Gemini.
            dispatch_map: Mapping of function names to callables.
            response_format: Expected JSON response schema format.
            generation_config: Thinking and temperature generation settings.
            service_tier: Optional Gemini service tier.
            token_weight: Token weight for rate limiter acquisition.
            label: Task category label for telemetry spans.
            media_name: Filename of the media chunk.
            video_duration_ms: Segment duration in milliseconds.
            validation_buffer_ms: Allowed subtitle end timestamp overflow buffer in milliseconds.
            retry: Effective retry settings for turn execution.
            validation_context: Optional extra context passed to Pydantic validation. Defaults to None.

        Returns:
            T: Parsed and validated Pydantic model instance.

        Raises:
            ValueError: If the interaction is incomplete, emits unknown/disallowed tool calls,
                or produces empty text output.
        """
        interaction_input: Any = initial_input
        previous_interaction_id: str | None = None
        accumulated_thoughts: list[str] = []
        search_turns = 0
        max_search_turns = self.ai_settings.search.maxturns

        while True:
            span_desc = (
                f"{label}: {media_name} {model_id}"
                if search_turns == 0
                else f"{label} (Search Turn {search_turns}/{max_search_turns}): {media_name} {model_id}"
            )

            with logfire.span(
                "{span_desc}",
                span_desc=span_desc,
                task=label,
                search_turn=search_turns,
                max_search_turns=max_search_turns,
                model_id=model_id,
                filename=media_name,
            ):
                current_tools = None if search_turns >= max_search_turns else gemini_tools

                interaction = await self._retry_operation(
                    operation=lambda attempt_num, max_attempts: self._create_interaction(
                        model_id=model_id,
                        input_payload=interaction_input,
                        previous_interaction_id=previous_interaction_id,
                        system_instruction=prompt.system,
                        tools=current_tools,
                        response_format=response_format,
                        generation_config=generation_config,
                        service_tier=service_tier,
                        token_weight=token_weight,
                        task_label=label,
                        attempt=attempt_num,
                        max_attempts=max_attempts,
                    ),
                    retry=retry,
                    op_label=f"{model_id} search turn {search_turns}",
                    model_id=model_id,
                )

                if getattr(interaction, "status", None) == "incomplete":
                    raise ValueError(
                        f"Model '{model_id}' returned incomplete interaction on search turn {search_turns}."
                    )

                steps = getattr(interaction, "steps", []) or []
                for step in steps:
                    if getattr(step, "type", None) == "thought":
                        if thought_text := _extract_thought_text(step):
                            accumulated_thoughts.append(thought_text)

                all_function_calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
                unknown_tools = [
                    str(getattr(s, "name", None) or "<unnamed>")
                    for s in all_function_calls
                    if getattr(s, "name", None) not in dispatch_map
                ]
                if unknown_tools:
                    raise ValueError(
                        f"Model '{model_id}' called unknown or disallowed tool(s) "
                        f"{unknown_tools} on search turn {search_turns}."
                    )

                if all_function_calls and search_turns < max_search_turns and dispatch_map:
                    search_turns += 1
                    interaction_input = await self._execute_function_calls(all_function_calls, dispatch_map)
                    raw_id = getattr(interaction, "id", None)
                    previous_interaction_id = raw_id if isinstance(raw_id, str) else None
                    continue

                text_content = self._extract_interaction_text(interaction)
                if not text_content:
                    if all_function_calls:
                        called = [str(getattr(s, "name", None) or "<unnamed>") for s in all_function_calls]
                        raise ValueError(
                            f"Model '{model_id}' attempted tool call(s) {called} after reaching "
                            f"max search turns ({max_search_turns})."
                        )
                    raise ValueError(f"Model '{model_id}' returned empty output on search turn {search_turns}.")

                return self._parse_and_validate_structured_output(
                    text_content=text_content,
                    response_type=response_type,
                    video_duration_ms=video_duration_ms,
                    validation_buffer_ms=validation_buffer_ms,
                    accumulated_thoughts=accumulated_thoughts,
                    validation_context=validation_context,
                )

    async def run_structured(
        self,
        prompt: Prompt,
        media_path: Path,
        video_duration_ms: int,
        response_type: type[T],
        model_name: str,
        service_tier: ServiceTier | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any] | BaseSearchProvider] | None = None,
        agentic_video: bool = False,
        task_label: str | None = None,
        retry_settings: RetrySettings | None = None,
        validation_buffer_ms: int = 2000,
        validation_context: dict[str, Any] | None = None,
    ) -> T:
        """Uploads media and generates validated structured outputs using the Gemini Interactions API.

        Args:
            prompt: Instructions containing system and user prompts.
            media_path: Path to the local video segment.
            video_duration_ms: Duration in milliseconds for token rate calculations.
            response_type: Expected Pydantic model response schema.
            model_name: Gemini model identifier (e.g., 'gemini-3.8-flash').
            service_tier: Optional Gemini service tier.
            tools: Sequence of callable tools, tool dicts, or BaseSearchProvider instances.
            agentic_video: Whether to enable Agentic Video Understanding.
            task_label: Optional label for display in turn spans.
            retry_settings: Optional task-specific retry policy overriding default settings.
            validation_buffer_ms: Allowed subtitle timestamp overflow tolerance in milliseconds. Defaults to 2000.
            validation_context: Optional extra context passed to Pydantic validation. Defaults to None.

        Returns:
            T: Parsed and validated Pydantic model instance.

        Raises:
            QuotaExceededError: If daily quota is exhausted for the model.
            RuntimeError: If media fails to process or retries are exhausted.
            ValidationError: If model output validation fails and validation retries are exhausted.
            ValueError: If input arguments or model identifiers are invalid.
        """
        model_id = model_name.strip()
        if not model_id:
            raise ValueError("Model identifier must not be empty.")
        if self._quota_exhausted.get(model_id, False):
            raise QuotaExceededError(f"Daily quota previously exhausted for model '{model_id}'.")

        effective_retry = retry_settings or self.default_retry_settings
        remote_file = await self.ensure_file_uploaded(media_path, retry_settings=effective_retry)

        tokens = _calculate_tokens(prompt, video_duration_ms)
        token_weight = max(1, min(tokens, self.ai_settings.tpm))

        media_input: dict[str, Any] = {"type": "video", "uri": remote_file.uri, "mime_type": remote_file.mime_type}
        if agentic_video:
            media_input["processing"] = "agentic"

        initial_input = [media_input, {"type": "text", "text": prompt.user}]
        gemini_tools, dispatch_map = self._setup_tools(tools)
        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": response_type.model_json_schema(),
        }
        generation_config = {"thinking_level": "high", "thinking_summaries": "auto"}
        label = task_label.capitalize() if task_label else "Generation"
        max_outer_attempts = effective_retry.validation + 1

        for outer_attempt in range(max_outer_attempts):
            try:
                return await self._run_interaction_session(
                    model_id=model_id,
                    initial_input=initial_input,
                    prompt=prompt,
                    response_type=response_type,
                    gemini_tools=gemini_tools,
                    dispatch_map=dispatch_map,
                    response_format=response_format,
                    generation_config=generation_config,
                    service_tier=service_tier,
                    token_weight=token_weight,
                    label=label,
                    media_name=media_path.name,
                    video_duration_ms=video_duration_ms,
                    validation_buffer_ms=validation_buffer_ms,
                    retry=effective_retry,
                    validation_context=validation_context,
                )
            except QuotaExceededError:
                raise
            except (ValidationError, ValueError) as val_err:
                if outer_attempt < effective_retry.validation:
                    backoff = effective_retry.backoff.compute_delay(outer_attempt)
                    logfire.warning(
                        f"Schema validation failure on {model_id} (attempt {outer_attempt + 1}/"
                        f"{max_outer_attempts}): {val_err}. Restarting in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                logfire.error(f"Schema validation exhausted all {max_outer_attempts} attempts on {model_id}: {val_err}")
                raise

        raise RuntimeError(f"Agent execution failed after {effective_retry.validation} validation retries.")
