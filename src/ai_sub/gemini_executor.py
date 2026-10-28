"""Unified execution engine for Google Gemini models and Files API management using the Interactions API."""

from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import random
import re
import urllib.parse
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable, TypeVar

import json_repair
import logfire
from google import genai
from google.genai import types
from httpx import TimeoutException, TransportError
from pydantic import BaseModel, ValidationError
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import AiSettings, RetrySettings, ServiceTier, Settings
from ai_sub.jobs.base import Prompt
from ai_sub.search.base import BaseSearchProvider

T = TypeVar("T", bound=BaseModel)

# Matches managed segment chunks and global summary proxies
MANAGED_PATTERN = re.compile(r"^(part_\d{3}|summary_proxy)\.[^./]+$")


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
    safe_name = urllib.parse.quote(resolved.name, safe=".-_")
    safe_parent = urllib.parse.quote(resolved.parent.name, safe=".-_")
    if resolved.parent.name == "reencoded":
        safe_grandparent = urllib.parse.quote(resolved.parent.parent.name, safe=".-_")
        return f"{safe_grandparent}/reencoded/{safe_name}"
    return f"{safe_parent}/{safe_name}"


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
    prompt_tokens = len(text_prompt.system) + len(text_prompt.user)
    video_tokens = int((video_duration_ms / 1000) * 300)
    return prompt_tokens + video_tokens


def _is_daily_quota_exceeded(e: Exception) -> bool:
    """Checks whether an API error represents Google Free Tier daily request limit exhaustion.

    Args:
        e: Caught exception.

    Returns:
        bool: True if daily project quota is exhausted.
    """
    err_str = str(e).lower()
    return "429" in err_str and "generaterequestsperdayperprojectpermodel-freetier" in err_str


def _is_retryable_error(e: Exception) -> bool:
    """Determines if an error is a transient network or server error eligible for search turn retry.

    Args:
        e: Caught exception.

    Returns:
        bool: True if error can be retried on the same search turn.
    """
    if _is_daily_quota_exceeded(e):
        return False

    # Network transport or timeout failures
    if isinstance(e, (TransportError, TimeoutException, ConnectionError, asyncio.TimeoutError, TimeoutError)):
        return True

    # Google API status codes
    status_code = getattr(e, "code", None)
    if isinstance(status_code, int) and status_code in (408, 429, 500, 502, 503, 504):
        return True

    err_str = str(e).lower()
    for code in ("503", "502", "504", "500", "408"):
        if code in err_str:
            return True
    if "connection" in err_str or "timeout" in err_str or "temporarily unavailable" in err_str:
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
        parts: list[str] = []
        if isinstance(summary, list):
            for block in summary:
                if isinstance(block, str):
                    if block.strip():
                        parts.append(block.strip())
                elif hasattr(block, "text") and getattr(block, "text", None):
                    parts.append(str(block.text).strip())
                elif isinstance(block, dict) and block.get("text"):
                    parts.append(str(block["text"]).strip())
        elif isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
        if parts:
            return "\n".join(parts)

    if getattr(step, "text", None):
        return str(step.text).strip()

    return None


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

    def __init__(
        self,
        settings: Settings | AiSettings,
    ) -> None:
        """Initializes the Gemini executor with disabled SDK retries for custom orchestration.

        Args:
            settings: Root application configuration or dedicated AiSettings.

        Raises:
            ValueError: If the Google API key is missing.
        """
        if isinstance(settings, Settings):
            self.ai_settings = settings.ai
            self.default_retry_settings = settings.retry
        else:
            self.ai_settings = settings
            self.default_retry_settings = RetrySettings()

        if self.ai_settings.key is None:
            raise ValueError("Google Gemini API key is missing.")

        self._quota_exhausted = {}
        self._request_limiters = {}
        self._token_limiters = {}
        self._last_cleanup_time = 0.0
        self._cleanup_lock = asyncio.Lock()

        self._init_client()

    def _init_client(self) -> None:
        """Initializes genai.Client with attempts=1 to delegate retry handling to our custom logic."""
        api_key = self.ai_settings.key.get_secret_value() if self.ai_settings.key else ""

        # Setting attempts=1 disables Google SDK's internal retries so all retries are orchestrated explicitly
        retry_options = types.HttpRetryOptions(attempts=1)

        http_options_kwargs: dict[str, Any] = {
            "retry_options": retry_options,
        }
        if self.ai_settings.base_url:
            http_options_kwargs["base_url"] = str(self.ai_settings.base_url)

        http_options = types.HttpOptions(**http_options_kwargs)

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
                rate_per_duration=self.ai_settings.rpm, duration=Duration.MINUTE
            )
            self._token_limiters[model_id] = limiter_factory.create_inmemory_limiter(
                rate_per_duration=self.ai_settings.tpm, duration=Duration.MINUTE
            )
        return self._request_limiters[model_id], self._token_limiters[model_id]

    async def _cleanup_storage_if_needed(self) -> None:
        """Deletes oldest managed segment files if total storage exceeds high-water mark."""
        now = time()
        cooldown_seconds = self.ai_settings.storage.cleanup_interval_minutes * 60
        if (now - self._last_cleanup_time) < cooldown_seconds:
            return

        async with self._cleanup_lock:
            if (now - self._last_cleanup_time) < cooldown_seconds:
                return

            with logfire.span("Storage cleanup check", _level="debug"):
                max_bytes = int(self.ai_settings.storage.max_gb * 1024 * 1024 * 1024)
                target_bytes = int(self.ai_settings.storage.target_gb * 1024 * 1024 * 1024)

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
                            f"Cleaning down to {self.ai_settings.storage.target_gb} GB..."
                        )
                        managed = [
                            f for f in files if f.display_name and MANAGED_PATTERN.match(Path(f.display_name).name)
                        ]
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
            RuntimeError: If file processing fails on the server or status cannot be retrieved.
        """
        with logfire.span("Gemini Files API: {filename}", filename=file_path.name, _level="debug"):
            await self._cleanup_storage_if_needed()

            display_name = _get_unique_display_name(file_path)

            existing_active: types.File | None = None
            existing_processing: types.File | None = None

            try:
                with logfire.span("Check existing remote file", _level="debug"):
                    async for remote_file in await self._client.aio.files.list(
                        config=types.ListFilesConfig(page_size=100)
                    ):
                        if remote_file.display_name == display_name:
                            if remote_file.state == types.FileState.ACTIVE:
                                existing_active = remote_file
                                break
                            if remote_file.state == types.FileState.PROCESSING and existing_processing is None:
                                existing_processing = remote_file
                            elif remote_file.state == types.FileState.FAILED and remote_file.name:
                                try:
                                    await self._client.aio.files.delete(name=remote_file.name)
                                except Exception:
                                    pass
            except Exception as e:
                logfire.debug(f"Remote file search error for {file_path.name}: {e}")

            if existing_active:
                logfire.info(f"Reusing existing remote file: {file_path.name} ({existing_active.name})")
                return existing_active

            poll_interval = self.ai_settings.storage.polling_interval_seconds

            if existing_processing:
                uploaded = existing_processing
            else:
                upload_path = file_path
                temp_link: Path | None = None

                if not file_path.name.isascii():
                    temp_link = file_path.parent / f"_upload_temp_{uuid.uuid4().hex}{file_path.suffix}"
                    if temp_link.exists():
                        try:
                            temp_link.unlink()
                        except OSError:
                            pass
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
                        uploaded = await self._client.aio.files.upload(
                            file=upload_path,
                            config=types.UploadFileConfig(
                                display_name=display_name,
                                mime_type=resolved_mime,
                            ),
                        )
                finally:
                    if temp_link and temp_link.exists():
                        try:
                            temp_link.unlink()
                        except OSError:
                            pass

                await asyncio.sleep(poll_interval)

            if uploaded.state != types.FileState.ACTIVE:
                with logfire.span("Waiting for Gemini to process {filename}", filename=file_path.name):
                    while uploaded.state != types.FileState.ACTIVE:
                        if uploaded.state == types.FileState.FAILED:
                            raise RuntimeError(f"File {file_path.name} failed to process on Gemini server.")

                        await asyncio.sleep(poll_interval)
                        if uploaded.name:
                            uploaded = await self._client.aio.files.get(name=uploaded.name)
                        else:
                            raise RuntimeError(f"Could not retrieve file '{file_path.name}' status.")

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
        api_client = self._client._api_client

        unpacked_items: list[Callable[..., Any] | dict[str, Any]] = []
        for item in tools:
            if isinstance(item, BaseSearchProvider):
                unpacked_items.extend(item.get_tools())
            else:
                unpacked_items.append(item)

        for item in unpacked_items:
            if isinstance(item, dict):
                interaction_tools.append(item)
            elif callable(item):
                fn_name = getattr(item, "__name__", str(item))
                decl = types.FunctionDeclaration.from_callable(client=api_client, callable=item)
                params: dict[str, Any] | None = None
                if decl.parameters is not None:
                    if hasattr(decl.parameters, "model_dump"):
                        params = decl.parameters.model_dump(mode="json", exclude_none=True)
                    elif isinstance(decl.parameters, dict):
                        params = decl.parameters

                tool_def: dict[str, Any] = {
                    "type": "function",
                    "name": decl.name or fn_name,
                    "description": decl.description or "",
                }
                if params is not None:
                    tool_def["parameters"] = params

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
    ) -> Any:
        """Executes a single search turn call to Gemini Interactions API with rate limiter acquisition.

        Args:
            model_id: Target Gemini model identifier.
            input_payload: Current search turn input (user prompt, media items, or function_result blocks).
            previous_interaction_id: Previous interaction ID for stateful turns.
            system_instruction: System instruction prompt.
            tools: Registered Interactions API tools for this turn.
            response_format: Top-level response format schema.
            generation_config: Generation configuration settings (e.g. thinking_level).
            service_tier: Optional Gemini service tier.
            token_weight: Token weight for rate limiter acquisition.

        Returns:
            Any: Interaction response object returned by the Gemini API.

        Raises:
            QuotaExceededError: If daily project quota is exhausted.
        """
        if self._quota_exhausted.get(model_id, False):
            raise QuotaExceededError(f"Daily quota previously exhausted for model '{model_id}'.")

        req_limiter, tok_limiter = self._get_limiters(model_id)

        create_kwargs: dict[str, Any] = {
            "model": model_id,
            "input": input_payload,
            "system_instruction": system_instruction,
            "generation_config": generation_config,
            "store": True,
        }
        if previous_interaction_id is not None:
            create_kwargs["previous_interaction_id"] = previous_interaction_id
        if tools is not None:
            create_kwargs["tools"] = tools
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        if service_tier is not None:
            create_kwargs["service_tier"] = service_tier

        await req_limiter.try_acquire_async("rpm")
        await tok_limiter.try_acquire_async("tpm", weight=token_weight)

        try:
            return await self._client.aio.interactions.create(**create_kwargs)
        except Exception as e:
            if _is_daily_quota_exceeded(e):
                self._quota_exhausted[model_id] = True
                logfire.warning(f"Daily quota exhausted for {model_id}.")
                raise QuotaExceededError(str(e)) from e
            raise

    async def _execute_turn_with_retry(
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
        search_turns: int,
        retry: RetrySettings,
    ) -> Any:
        """Executes a single search turn with exponential backoff on transient network or 5xx errors.

        Args:
            model_id: Target Gemini model identifier.
            input_payload: Search turn input payload.
            previous_interaction_id: Previous interaction ID for stateful turns.
            system_instruction: System instruction prompt.
            tools: Registered Interactions API tools for this turn.
            response_format: Target response format schema.
            generation_config: Thinking and temperature generation configuration.
            service_tier: Optional Gemini service tier.
            token_weight: Token weight for rate limiter acquisition.
            search_turns: Current search turn index.
            retry: Effective retry settings for backoff intervals.

        Returns:
            Any: Successful interaction response object.

        Raises:
            QuotaExceededError: If daily project quota is reached.
            RuntimeError: If turn retries are exhausted.
        """
        turn_max_attempts = retry.per_run + 1

        for search_turn_attempt in range(turn_max_attempts):
            try:
                return await self._create_interaction(
                    model_id=model_id,
                    input_payload=input_payload,
                    previous_interaction_id=previous_interaction_id,
                    system_instruction=system_instruction,
                    tools=tools,
                    response_format=response_format,
                    generation_config=generation_config,
                    service_tier=service_tier,
                    token_weight=token_weight,
                )
            except QuotaExceededError:
                raise
            except Exception as err:
                if _is_daily_quota_exceeded(err):
                    self._quota_exhausted[model_id] = True
                    logfire.warning(f"Daily quota exhausted for {model_id}: {err}")
                    raise QuotaExceededError(str(err)) from err

                is_retryable = _is_retryable_error(err)
                if is_retryable and search_turn_attempt < retry.per_run:
                    backoff = min(
                        float(retry.max_wait_seconds),
                        float(retry.min_wait_seconds) * (float(retry.multiplier) ** search_turn_attempt),
                    ) + random.uniform(0, 1)

                    logfire.warning(
                        f"Transient error on {model_id} search turn {search_turns} "
                        f"(attempt {search_turn_attempt + 1}/{turn_max_attempts}): {err}. "
                        f"Retrying search turn in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

                logfire.error(f"Search turn {search_turns} failed permanently on {model_id}: {err}")
                raise

        raise RuntimeError(f"Turn execution failed on {model_id} after {turn_max_attempts} attempts.")

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
                if inspect.iscoroutinefunction(fn):
                    cres = await fn(**args)
                else:
                    cres = await asyncio.to_thread(fn, **args)
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

        text_parts: list[str] = []
        for step in getattr(interaction, "steps", []) or []:
            if getattr(step, "type", None) == "model_output":
                for block in getattr(step, "content", []) or []:
                    if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                        text_parts.append(str(block.text))

        return "".join(text_parts).strip()

    def _parse_and_validate_structured_output(
        self,
        text_content: str,
        response_type: type[T],
        video_duration_ms: int,
        accumulated_thoughts: list[str],
    ) -> T:
        """Parses output text into the target Pydantic schema, applying json_repair fallback if needed.

        Args:
            text_content: Raw text returned by the model.
            response_type: Target Pydantic model class.
            video_duration_ms: Duration in milliseconds for timestamp context validation.
            accumulated_thoughts: Collected reasoning traces to attach to the output.

        Returns:
            T: Validated Pydantic model instance.
        """
        validation_context = {
            "video_duration_ms": video_duration_ms,
            "validation_buffer_ms": self.ai_settings.validation_buffer_ms,
        }

        try:
            output = response_type.model_validate_json(text_content, context=validation_context)
        except Exception:
            repaired = json_repair.loads(text_content)
            output = response_type.model_validate(repaired, context=validation_context)

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
        retry: RetrySettings,
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
            retry: Effective retry settings for turn execution.

        Returns:
            T: Parsed and validated Pydantic model instance.
        """
        interaction_input: Any = initial_input
        previous_interaction_id: str | None = None
        accumulated_thoughts: list[str] = []
        search_turns = 0
        max_search_turns = self.ai_settings.search.max_turns

        while True:
            if search_turns == 0:
                span_desc = f"{label}: {media_name} {model_id}"
            else:
                span_desc = f"{label} (Search Turn {search_turns}/{max_search_turns}): {media_name} {model_id}"

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

                interaction = await self._execute_turn_with_retry(
                    model_id=model_id,
                    input_payload=interaction_input,
                    previous_interaction_id=previous_interaction_id,
                    system_instruction=prompt.system,
                    tools=current_tools,
                    response_format=response_format,
                    generation_config=generation_config,
                    service_tier=service_tier,
                    token_weight=token_weight,
                    search_turns=search_turns,
                    retry=retry,
                )

                for step in getattr(interaction, "steps", []) or []:
                    if getattr(step, "type", None) == "thought":
                        if thought_text := _extract_thought_text(step):
                            accumulated_thoughts.append(thought_text)

                function_call_steps = [
                    s
                    for s in getattr(interaction, "steps", []) or []
                    if getattr(s, "type", None) == "function_call" and getattr(s, "name", None) in dispatch_map
                ]

                # Advance to next turn if tools were called and turns remain
                if function_call_steps and search_turns < max_search_turns and dispatch_map:
                    search_turns += 1
                    interaction_input = await self._execute_function_calls(function_call_steps, dispatch_map)
                    raw_id = getattr(interaction, "id", None)
                    previous_interaction_id = raw_id if isinstance(raw_id, str) else None
                    continue

                text_content = self._extract_interaction_text(interaction)
                return self._parse_and_validate_structured_output(
                    text_content=text_content,
                    response_type=response_type,
                    video_duration_ms=video_duration_ms,
                    accumulated_thoughts=accumulated_thoughts,
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
    ) -> T:
        """Uploads media and generates validated structured outputs using the Gemini Interactions API.

        Implements a two-tier retry hierarchy:
        1. Inner Loop (Search Turn Level): Retries transient network, server (5xx/408), and transient 429
           errors on the CURRENT search turn without repeating earlier search turns or re-running tool calls.
        2. Outer Loop (Schema Validation Level): If the final model output fails Pydantic schema validation,
           restarts the interaction from Turn 1 with a clean slate (bounded by effective_retry.per_run).

        Args:
            prompt: Instructions containing system and user prompts.
            media_path: Path to the local video segment.
            video_duration_ms: Duration in milliseconds for token rate calculations.
            response_type: Expected Pydantic model response schema.
            model_name: Full model identifier (must begin with 'google-gla:').
            service_tier: Optional Gemini service tier.
            tools: Sequence of callable tools, tool dicts, or BaseSearchProvider instances.
            agentic_video: Whether to enable Agentic Video Understanding.
            task_label: Optional label for display in turn spans (e.g. 'summary', 'subtitles', 'lyrics').
            retry_settings: Optional task-specific retry policy overriding default settings.

        Returns:
            T: Parsed and validated Pydantic model instance.

        Raises:
            QuotaExceededError: If daily quota is exhausted for the model.
            RuntimeError: If media fails to process or retries are exhausted.
            ValidationError: If model output validation fails and outer retries are exhausted.
            ValueError: If input arguments or model identifiers are invalid.
        """
        if not model_name.lower().startswith("google-gla:"):
            raise ValueError(f"Model identifier '{model_name}' must start with 'google-gla:'.")

        model_id = model_name.split(":", 1)[-1]

        if self._quota_exhausted.get(model_id, False):
            raise QuotaExceededError(f"Daily quota previously exhausted for model '{model_id}'.")

        effective_retry = retry_settings or getattr(self, "default_retry_settings", None) or RetrySettings()

        remote_file = await self.ensure_file_uploaded(media_path)

        tokens = _calculate_tokens(prompt, video_duration_ms)
        token_weight = min(tokens, self.ai_settings.tpm)

        media_input: dict[str, Any] = {
            "type": "video",
            "uri": remote_file.uri,
            "mime_type": remote_file.mime_type,
        }
        if agentic_video:
            media_input["processing"] = "agentic"

        initial_input = [media_input, {"type": "text", "text": prompt.user}]
        gemini_tools, dispatch_map = self._setup_tools(tools)
        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": response_type.model_json_schema(),
        }
        generation_config = {
            "thinking_level": "high",
            "thinking_summaries": "auto",
        }
        label = task_label.capitalize() if task_label else "Generation"
        max_outer_attempts = effective_retry.per_run + 1

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
                    retry=effective_retry,
                )
            except QuotaExceededError:
                raise
            except (ValidationError, ValueError) as val_err:
                if outer_attempt < effective_retry.per_run:
                    backoff = min(
                        float(effective_retry.max_wait_seconds),
                        float(effective_retry.min_wait_seconds) * (float(effective_retry.multiplier) ** outer_attempt),
                    ) + random.uniform(0, 1)
                    logfire.warning(
                        f"Schema validation failure on {model_id} "
                        f"(outer attempt {outer_attempt + 1}/{max_outer_attempts}): {val_err}. "
                        f"Restarting interaction from scratch in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                logfire.error(
                    f"Schema validation exhausted all {max_outer_attempts} outer attempts on {model_id}: {val_err}"
                )
                raise

        raise RuntimeError(f"Agent execution failed after {effective_retry.per_run} retries.")
