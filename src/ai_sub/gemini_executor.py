"""Unified execution engine for Google Gemini models and Files API management."""

from __future__ import annotations

import asyncio
import inspect
import random
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable, TypeVar

import httpx
import json_repair
import logfire
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, ValidationError
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import RetrySettings, ServiceTier, Settings
from ai_sub.jobs.base import Prompt

T = TypeVar("T", bound=BaseModel)

# Strictly matches managed video segments (e.g., 'part_001.mp4', 'part_002.mov')
PART_PATTERN = re.compile(r"^part_\d{3}\.[^./]+$")


class QuotaExceededError(Exception):
    """Raised when an AI model provider's daily quota limit is exhausted."""


def _get_unique_display_name(file_path: Path) -> str:
    """Constructs a unique display name using workspace folder and filename.

    Args:
        file_path: Path to the local file.

    Returns:
        str: Display name formatted as 'workspace/filename.ext' or 'workspace/reencoded/filename.ext'.
    """
    resolved = file_path.resolve()
    if resolved.parent.name == "reencoded":
        return f"{resolved.parent.parent.name}/reencoded/{resolved.name}"
    return f"{resolved.parent.name}/{resolved.name}"


def _get_file_timestamp(remote_file: types.File) -> float:
    """Extracts a numeric timestamp in seconds for ordering files chronologically.

    Args:
        remote_file: Remote File object from Gemini API.

    Returns:
        float: Unix timestamp in seconds.
    """
    ts = remote_file.expiration_time or remote_file.create_time
    if ts is None:
        return 0.0
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _calculate_tokens(text_prompt: Prompt, video_duration_ms: int) -> int:
    """Estimates tokens consumed by text prompts and video duration for rate limiting.

    Args:
        text_prompt: Combined system and user prompt.
        video_duration_ms: Video segment duration in milliseconds.

    Returns:
        int: Estimated token count.
    """
    return int(len(text_prompt.system) + len(text_prompt.user) + (video_duration_ms / 1000) * 300)


def _is_daily_quota_exceeded(e: Exception) -> bool:
    """Checks whether an API error represents Google free tier daily quota exhaustion.

    Args:
        e: Caught exception.

    Returns:
        bool: True if daily project quota is exhausted.
    """
    err_str = str(e).lower()
    return "429" in err_str and "generaterequestsperdayperprojectpermodel-freetier" in err_str


class GeminiExecutor:
    """Unified client orchestrating Gemini Files API uploads, quota tracking, and generation."""

    settings: Settings
    _client: genai.Client
    _quota_exhausted: dict[str, bool]
    _request_limiters: dict[str, Limiter]
    _token_limiters: dict[str, Limiter]
    _last_cleanup_time: float
    _cleanup_lock: asyncio.Lock

    def __init__(self, settings: Settings) -> None:
        """Initializes the Gemini executor.

        Args:
            settings: Root application configuration settings.

        Raises:
            ValueError: If the Google API key is missing.
        """
        if settings.ai.key is None:
            raise ValueError("Google Gemini API key is missing.")

        api_key = settings.ai.key.get_secret_value()

        self.settings = settings
        self._quota_exhausted = {}
        self._request_limiters = {}
        self._token_limiters = {}
        self._last_cleanup_time = 0.0
        self._cleanup_lock = asyncio.Lock()

        http_options: types.HttpOptions | None = None
        if self.settings.ai.base_url:
            http_options = types.HttpOptions(base_url=str(self.settings.ai.base_url))

        self._client = genai.Client(
            api_key=api_key,
            http_options=http_options,
        )

    def _get_limiters(self, model_id: str) -> tuple[Limiter, Limiter]:
        """Retrieves or creates rate limiters for a specific model ID.

        Args:
            model_id: Model identifier string.

        Returns:
            tuple[Limiter, Limiter]: Request rate limiter and token rate limiter.
        """
        if model_id not in self._request_limiters:
            self._request_limiters[model_id] = limiter_factory.create_inmemory_limiter(
                rate_per_duration=self.settings.ai.rpm, duration=Duration.MINUTE
            )
            self._token_limiters[model_id] = limiter_factory.create_inmemory_limiter(
                rate_per_duration=self.settings.ai.tpm, duration=Duration.MINUTE
            )
        return self._request_limiters[model_id], self._token_limiters[model_id]

    async def _cleanup_storage_if_needed(self) -> None:
        """Deletes oldest managed segment files if total storage exceeds high-water mark."""
        now = time()
        cooldown_seconds = self.settings.ai.storage.cleanup_interval_minutes * 60
        if (now - self._last_cleanup_time) < cooldown_seconds:
            return

        async with self._cleanup_lock:
            if (now - self._last_cleanup_time) < cooldown_seconds:
                return

            with logfire.span("Storage cleanup check", _level="debug"):
                max_bytes = int(self.settings.ai.storage.max_gb * 1024 * 1024 * 1024)
                target_bytes = int(self.settings.ai.storage.target_gb * 1024 * 1024 * 1024)

                try:
                    files: list[types.File] = []
                    async for remote_file in await self._client.aio.files.list(
                        config=types.ListFilesConfig(page_size=100)
                    ):
                        if remote_file.name:
                            files.append(remote_file)

                    total_bytes = sum(f.size_bytes or 0 for f in files)
                    if total_bytes >= max_bytes:
                        logfire.info(
                            f"Files API storage threshold reached ({total_bytes / (1024**3):.2f} GB used). "
                            f"Cleaning down to {self.settings.ai.storage.target_gb} GB..."
                        )
                        managed = [f for f in files if f.display_name and PART_PATTERN.match(Path(f.display_name).name)]
                        managed.sort(key=_get_file_timestamp)

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

    async def ensure_file_uploaded(self, file_path: Path) -> types.File:
        """Verifies remote file presence, performing upload and status polling as needed.

        Args:
            file_path: Local path to the segment file.

        Returns:
            types.File: Active remote file handle from the Gemini Files API.

        Raises:
            RuntimeError: If file processing fails on the server.
        """
        with logfire.span("Gemini Files API: {filename}", filename=file_path.name, _level="debug"):
            await self._cleanup_storage_if_needed()

            display_name = _get_unique_display_name(file_path)

            try:
                with logfire.span("Check existing remote file", _level="debug"):
                    async for remote_file in await self._client.aio.files.list(
                        config=types.ListFilesConfig(page_size=100)
                    ):
                        if remote_file.display_name == display_name:
                            if remote_file.state == types.FileState.ACTIVE:
                                logfire.debug(
                                    "Reusing existing remote file: {name} ({file_id})",
                                    name=file_path.name,
                                    file_id=remote_file.name,
                                )
                                return remote_file
                            elif remote_file.state == types.FileState.FAILED:
                                if remote_file.name:
                                    try:
                                        await self._client.aio.files.delete(name=remote_file.name)
                                    except Exception:
                                        pass
            except Exception as e:
                logfire.debug(f"Remote file search error for {file_path.name}: {e}")

            with logfire.span("Uploading file to Files API", _level="debug"):
                logfire.debug(f"Uploading {file_path.name} to Gemini Files API...")
                uploaded = await self._client.aio.files.upload(
                    file=file_path,
                    config=types.UploadFileConfig(display_name=display_name),
                )

            poll_interval = self.settings.ai.storage.polling_interval_seconds
            await asyncio.sleep(poll_interval)

            with logfire.span("Polling file state until ACTIVE", _level="debug"):
                while uploaded.state != types.FileState.ACTIVE:
                    if uploaded.state == types.FileState.FAILED:
                        raise RuntimeError(f"File {file_path.name} failed to process on Gemini server.")

                    await asyncio.sleep(poll_interval)
                    if uploaded.name:
                        uploaded = await self._client.aio.files.get(name=uploaded.name)
                    else:
                        raise RuntimeError(f"Could not retrieve file '{file_path.name}' status.")

            return uploaded

    def _get_thinking_config(self, model_id: str) -> types.ThinkingConfig:
        """Builds thinking reasoning budget based on model generation.

        Args:
            model_id: Model identifier.

        Returns:
            types.ThinkingConfig: Thinking configuration for Google Gen AI.
        """
        model_lower = model_id.lower()
        if model_lower.startswith("gemini-3"):
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=types.ThinkingLevel.HIGH,
            )
        if model_lower.startswith("gemini-2.5-pro"):
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=32768,
            )
        return types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,
        )

    def _get_safety_settings(self) -> list[types.SafetySetting]:
        """Builds permissive safety settings for multimedia transcription.

        Returns:
            list[types.SafetySetting]: Configured safety settings.
        """
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def _setup_tools(
        self,
        tools: Sequence[Callable[..., Any] | types.Tool] | None,
    ) -> tuple[list[types.Tool] | None, dict[str, Callable[..., Any]]]:
        """Builds Gemini tool configurations and local dispatch mapping from tools.

        Args:
            tools: Sequence of callables (bound methods/functions) or types.Tool instances.

        Returns:
            tuple[list[types.Tool] | None, dict[str, Callable[..., Any]]]:
                Registered Gemini tools list and execution dispatch dictionary.
        """
        if not tools:
            return None, {}

        declarations: list[types.FunctionDeclaration] = []
        dispatch_map: dict[str, Callable[..., Any]] = {}
        gemini_tools: list[types.Tool] = []
        api_client = self._client._api_client

        for item in tools:
            if isinstance(item, types.Tool):
                gemini_tools.append(item)
            elif callable(item):
                fn_name = getattr(item, "__name__", str(item))
                decl = types.FunctionDeclaration.from_callable(client=api_client, callable=item)
                declarations.append(decl)
                dispatch_map[fn_name] = item

        if declarations:
            gemini_tools.append(types.Tool(function_declarations=declarations))

        return gemini_tools or None, dispatch_map

    async def run_structured(
        self,
        prompt: Prompt,
        media_path: Path,
        video_duration_ms: int,
        response_type: type[T],
        model_name: str,
        service_tier: ServiceTier | None = None,
        tools: Sequence[Callable[..., Any] | types.Tool] | None = None,
        retry_policy: RetrySettings | None = None,
    ) -> T:
        """Uploads media and generates validated structured outputs with tool turns.

        Args:
            prompt: Instructions containing system and user prompts.
            media_path: Path to the local video segment.
            video_duration_ms: Duration in milliseconds for token rate calculations.
            response_type: Expected Pydantic model response schema.
            model_name: Full model identifier (must begin with 'google-gla:').
            service_tier: Optional Gemini service tier.
            tools: Sequence of callable tools (bound provider methods) or types.Tool instances.
            retry_policy: Retry configuration policy (defaults to settings.retry if omitted).

        Returns:
            T: Parsed and validated Pydantic model instance.

        Raises:
            QuotaExceededError: If daily quota is exhausted for the model.
            RuntimeError: If media fails to process or retries are exhausted.
            APIError: If an unretryable API error occurs or retries are exhausted.
            ValidationError: If model output validation fails and retries are exhausted.
            ValueError: If input arguments or model identifiers are invalid.
            ConnectionError: If network connectivity fails and retries are exhausted.
            TimeoutError: If asynchronous operations time out and retries are exhausted.
            TransportError: If transport connectivity fails and retries are exhausted.
        """
        if not model_name.lower().startswith("google-gla:"):
            raise ValueError(f"Model identifier '{model_name}' must start with 'google-gla:'.")

        model_id = model_name.split(":", 1)[-1]
        if self._quota_exhausted.get(model_id, False):
            raise QuotaExceededError(f"Daily quota previously exhausted for model '{model_id}'.")

        policy = retry_policy or self.settings.retry

        with logfire.span("Gemini {model_id}: {filename}", model_id=model_id, filename=media_path.name):
            remote_file = await self.ensure_file_uploaded(media_path)
            req_limiter, tok_limiter = self._get_limiters(model_id)
            tokens = _calculate_tokens(prompt, video_duration_ms)

            gemini_tools, dispatch_map = self._setup_tools(tools)
            config_kwargs: dict[str, Any] = {
                "system_instruction": prompt.system,
                "thinking_config": self._get_thinking_config(model_id),
                "safety_settings": self._get_safety_settings(),
                "response_mime_type": "application/json",
                "response_schema": response_type,
            }
            if service_tier is not None:
                config_kwargs["service_tier"] = service_tier
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools

            config = types.GenerateContentConfig(**config_kwargs)

            config_final = config
            if gemini_tools and dispatch_map:
                final_kwargs = dict(config_kwargs)
                final_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE,
                    )
                )
                config_final = types.GenerateContentConfig(**final_kwargs)

            max_turns = self.settings.ai.search.max_turns

            for attempt in range(policy.per_run + 1):
                contents: list[Any] = [remote_file, prompt.user]
                search_turns = 0
                accumulated_thoughts: list[str] = []

                try:
                    while True:
                        with logfire.span(
                            "Turn {turn}: {model_id}",
                            turn=search_turns,
                            model_id=model_id,
                        ):
                            await req_limiter.try_acquire_async("rpm")
                            await tok_limiter.try_acquire_async("tpm", weight=tokens)

                            active_config = config_final if search_turns >= max_turns else config
                            response = await self._client.aio.models.generate_content(
                                model=model_id,
                                contents=contents,
                                config=active_config,
                            )

                            candidate = response.candidates[0] if response.candidates else None
                            if candidate is None:
                                raise RuntimeError(f"No response candidate returned by {model_id}.")

                            # Accumulate reasoning traces from this turn
                            if candidate.content and candidate.content.parts:
                                turn_thinking = [
                                    t
                                    for part in candidate.content.parts
                                    if getattr(part, "thought", False) and (t := part.text) is not None
                                ]
                                if turn_thinking:
                                    accumulated_thoughts.append("\n".join(turn_thinking))

                            function_calls = response.function_calls
                            if function_calls and search_turns < max_turns and dispatch_map:
                                search_turns += 1

                                async def _exec(fc: types.FunctionCall) -> tuple[str, Any]:
                                    cname = fc.name or ""
                                    if cname not in dispatch_map:
                                        logfire.warning(f"Unknown tool requested by model: {cname}")
                                        return cname, {"error": f"Unknown tool: {cname}"}
                                    fn = dispatch_map[cname]
                                    try:
                                        args = fc.args or {}
                                        if inspect.iscoroutinefunction(fn):
                                            cres = await fn(**args)
                                        else:
                                            cres = await asyncio.to_thread(fn, **args)
                                        return cname, {"result": cres}
                                    except Exception as tool_err:
                                        logfire.warning(f"Tool '{cname}' failed: {tool_err}")
                                        return cname, {"error": f"Tool execution failed: {tool_err}"}

                                call_results = await asyncio.gather(*(_exec(fc) for fc in function_calls))
                                response_parts = [
                                    types.Part.from_function_response(name=cname, response=resp_data)
                                    for cname, resp_data in call_results
                                ]

                                contents.append(candidate.content)
                                contents.append(types.Content(role="user", parts=response_parts))
                                continue

                            # Extract text directly from non-thought/non-call parts to avoid SDK warnings
                            text_parts: list[str] = []
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if getattr(part, "thought", False):
                                        continue
                                    if getattr(part, "function_call", None) is not None:
                                        continue
                                    if part.text:
                                        text_parts.append(part.text)

                            text_content = "".join(text_parts).strip()
                            if not text_content and response.text:
                                text_content = response.text.strip()

                            try:
                                output = response_type.model_validate_json(
                                    text_content,
                                    context={
                                        "video_duration_ms": video_duration_ms,
                                        "validation_buffer_ms": self.settings.ai.validation_buffer_ms,
                                    },
                                )
                            except Exception:
                                repaired = json_repair.loads(text_content)
                                output = response_type.model_validate(
                                    repaired,
                                    context={
                                        "video_duration_ms": video_duration_ms,
                                        "validation_buffer_ms": self.settings.ai.validation_buffer_ms,
                                    },
                                )

                            if hasattr(output, "thoughts") and accumulated_thoughts:
                                setattr(output, "thoughts", accumulated_thoughts)

                            return output

                except APIError as e:
                    if _is_daily_quota_exceeded(e):
                        self._quota_exhausted[model_id] = True
                        logfire.warning(f"Daily quota exhausted for {model_id}.")
                        raise QuotaExceededError(str(e)) from e

                    is_retryable = e.code in (429, 500, 502, 503, 504) if e.code else True
                    if is_retryable and attempt < policy.per_run:
                        backoff = min(
                            float(policy.max_wait_seconds),
                            (policy.min_wait_seconds * (policy.multiplier**attempt)) + random.uniform(0, 1),
                        )
                        logfire.info(
                            f"API error in {model_id} (attempt {attempt + 1}). Retrying in {backoff:.2f}s: {e}"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise

                except (ValidationError, ValueError) as e:
                    if attempt < policy.per_run:
                        backoff = min(
                            float(policy.max_wait_seconds),
                            (policy.min_wait_seconds * (policy.multiplier**attempt)) + random.uniform(0, 1),
                        )
                        logfire.warning(
                            f"Validation error in {model_id} (attempt {attempt + 1}). Retrying in {backoff:.2f}s: {e}"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise

                except (ConnectionError, asyncio.TimeoutError, httpx.TransportError) as e:
                    if attempt < policy.per_run:
                        backoff = min(
                            float(policy.max_wait_seconds),
                            (policy.min_wait_seconds * (policy.multiplier**attempt)) + random.uniform(0, 1),
                        )
                        logfire.info(
                            f"Network error in {model_id} (attempt {attempt + 1}). Retrying in {backoff:.2f}s: {e}"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise

        raise RuntimeError(f"Agent execution failed after {policy.per_run} retries.")
