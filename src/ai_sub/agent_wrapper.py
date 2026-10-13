"""AI Agent wrapper for the subtitle generation pipeline using Google Gen AI SDK."""

from __future__ import annotations

import asyncio
import random
from typing import Any, TypeVar

import json_repair
import logfire
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel
from pyrate_limiter import Duration, Limiter, limiter_factory

from ai_sub.config import ServiceTier, Settings
from ai_sub.data_models import AgentDeps, QuotaExceededError
from ai_sub.prompt import Prompt
from ai_sub.web_search_duckduckgo import duckduckgo_search, web_search_duckduckgo_multi
from ai_sub.web_search_langsearch import web_search_langsearch_multi
from ai_sub.web_search_ollama import web_search_ollama_multi

T = TypeVar("T", bound=BaseModel)

MAX_SEARCH_TURNS = 3


def _calculate_tokens(text_prompt: Prompt, video_duration_ms: int) -> int:
    """Estimates the number of tokens for a given text prompt and video duration.

    This estimation is used for rate limiting purposes.

    Args:
        text_prompt: The text prompt containing system and user instructions.
        video_duration_ms: The duration of the video segment in milliseconds.

    Returns:
        int: The estimated number of tokens.
    """
    return int(len(text_prompt.system_prompt) + len(text_prompt.user_prompt) + (video_duration_ms / 1000) * 300)


def _is_free_tier_quota_exceeded(e: Exception) -> bool:
    """Checks if an exception indicates a Google Free Tier daily quota exhaustion.

    Args:
        e: The exception to check.

    Returns:
        bool: True if it is a daily (hard) quota exhaustion error, False otherwise.
    """
    err_str = str(e).lower()
    return "429" in err_str and "generaterequestsperdayperprojectpermodel-freetier" in err_str


class RateLimitedAgentWrapper:
    """A rate-limited wrapper around Google Gen AI client for subtitle and lyrics processing."""

    settings: Settings
    model_name: str
    model_id: str
    service_tier: ServiceTier | None
    use_web_search: bool
    deps: AgentDeps
    request_limiter: Limiter
    token_limiter: Limiter
    _client: genai.Client
    _quota_exceeded: bool

    def __init__(
        self,
        settings: Settings,
        model_name: str,
        deps: AgentDeps | None = None,
        use_web_search: bool = False,
        service_tier: ServiceTier | None = None,
    ) -> None:
        """Initializes the agent wrapper with settings and Google Gen AI client.

        Args:
            settings: The application configuration settings.
            model_name: The name of the model (must start with 'google-gla:').
            deps: Optional dependencies for the agent.
            use_web_search: Whether to enable web search tools.
            service_tier: The service tier for the model.

        Raises:
            ValueError: If the model name does not start with 'google-gla:' or the API key is missing.
        """
        if not model_name.lower().startswith("google-gla:"):
            raise ValueError(
                f"Model '{model_name}' is invalid. Model name must start with 'google-gla:' "
                "(e.g., 'google-gla:gemini-3.7-flash')."
            )

        if settings.ai.google.key is None:
            raise ValueError("Google AI API key is missing.")

        api_key = settings.ai.google.key.get_secret_value()

        self.settings = settings
        self.model_name = model_name
        self.model_id = model_name.split(":", 1)[-1]
        self.use_web_search = use_web_search
        self.service_tier = service_tier
        self._quota_exceeded = False
        self.deps = deps or AgentDeps(validation_buffer_ms=settings.ai.validation_buffer_ms)

        http_options: types.HttpOptions | None = None
        if self.settings.ai.google.base_url:
            http_options = types.HttpOptions(base_url=str(self.settings.ai.google.base_url))

        self._client = genai.Client(
            api_key=api_key,
            http_options=http_options,
        )

        self.request_limiter = limiter_factory.create_inmemory_limiter(
            rate_per_duration=self.settings.ai.rpm, duration=Duration.MINUTE
        )
        self.token_limiter = limiter_factory.create_inmemory_limiter(
            rate_per_duration=self.settings.ai.tpm, duration=Duration.MINUTE
        )

    def _get_thinking_config(self) -> types.ThinkingConfig:
        """Constructs the appropriate thinking configuration based on the model version.

        Returns:
            types.ThinkingConfig: The thinking configuration for Google Gen AI.
        """
        model_lower = self.model_id.lower()
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
        """Builds default permissive safety settings for multimedia transcription.

        Returns:
            list[types.SafetySetting]: The configured safety settings.
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

    def _get_tools(self) -> list[types.Tool] | None:
        """Constructs tool declarations for web search if enabled.

        Returns:
            list[types.Tool] | None: The tool configuration, or None if web search is disabled.
        """
        if not self.use_web_search:
            return None

        tool_type = self.settings.ai.search.web_search_tool
        if tool_type == "builtin":
            return [types.Tool(google_search=types.GoogleSearch())]

        declarations: list[types.FunctionDeclaration] = []
        if tool_type == "langsearch":
            declarations.append(
                types.FunctionDeclaration(
                    name="web_search_langsearch_multi",
                    description="Perform multiple web search queries in parallel using Langsearch.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of search query strings.",
                            },
                        },
                        "required": ["queries"],
                    },
                )
            )
        elif tool_type == "ollama":
            declarations.append(
                types.FunctionDeclaration(
                    name="web_search_ollama_multi",
                    description="Perform multiple web search queries in parallel using Ollama.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of search query strings.",
                            },
                        },
                        "required": ["queries"],
                    },
                )
            )
        elif tool_type == "duckduckgo":
            declarations.extend(
                [
                    types.FunctionDeclaration(
                        name="duckduckgo_search",
                        description="Perform a single web search query using DuckDuckGo.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string.",
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="web_search_duckduckgo_multi",
                        description="Perform multiple web search queries in parallel using DuckDuckGo.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "A list of search query strings.",
                                },
                            },
                            "required": ["queries"],
                        },
                    ),
                ]
            )

        return [types.Tool(function_declarations=declarations)] if declarations else None

    async def _execute_tool_call(self, name: str, args: dict[str, Any]) -> Any:
        """Executes a single tool call requested by the model.

        Args:
            name: The name of the function to execute.
            args: The function arguments provided by the model.

        Returns:
            Any: The JSON-serializable tool output.
        """
        search_deps = self.deps.web_search
        assert search_deps is not None, "WebSearchDeps must be initialized when web search is enabled."

        if name == "web_search_langsearch_multi":
            raw_queries = args.get("queries") or [args.get("query")]
            queries = [str(q) for q in raw_queries if q]
            results = await web_search_langsearch_multi(search_deps, queries)
            return [r.model_dump(mode="json") for r in results]

        if name == "web_search_ollama_multi":
            raw_queries = args.get("queries") or [args.get("query")]
            queries = [str(q) for q in raw_queries if q]
            results = await web_search_ollama_multi(search_deps, queries)
            return [r.model_dump(mode="json") for r in results]

        if name == "duckduckgo_search":
            query = str(args.get("query") or "")
            results = await duckduckgo_search(search_deps, query)
            return [r.model_dump(mode="json") for r in results]

        if name == "web_search_duckduckgo_multi":
            raw_queries = args.get("queries") or [args.get("query")]
            queries = [str(q) for q in raw_queries if q]
            results = await web_search_duckduckgo_multi(search_deps, queries)
            return [r.model_dump(mode="json") for r in results]

        return {"error": f"Unknown tool: {name}"}

    async def run(
        self,
        prompt: Prompt,
        video: types.File,
        video_duration_ms: int,
        response_type: type[T],
    ) -> T:
        """Runs the AI agent to generate structured responses from the uploaded video.

        Args:
            prompt: The Prompt object containing system and user prompt strings.
            video: The uploaded video file handle in the Gemini Files API.
            video_duration_ms: The duration of the video in milliseconds (used for rate limiting).
            response_type: The expected Pydantic model for the structured response.

        Returns:
            T: The parsed and validated structured response.

        Raises:
            QuotaExceededError: If the model's quota has been exhausted.
            APIError: If an unretryable API error occurs or retries are exhausted.
            ConnectionError: If network connectivity fails and retries are exhausted.
            TimeoutError: If request timeouts occur and retries are exhausted.
            RuntimeError: If retries are exhausted or generation fails.
        """
        if self._quota_exceeded:
            raise QuotaExceededError(f"Quota previously exceeded for model {self.model_name}")

        tokens = _calculate_tokens(prompt, video_duration_ms)

        config_kwargs: dict[str, Any] = {
            "system_instruction": prompt.system_prompt,
            "thinking_config": self._get_thinking_config(),
            "safety_settings": self._get_safety_settings(),
            "response_mime_type": "application/json",
            "response_schema": response_type,
        }
        if self.service_tier is not None:
            config_kwargs["service_tier"] = self.service_tier

        tools = self._get_tools()
        if tools:
            config_kwargs["tools"] = tools

        config = types.GenerateContentConfig(**config_kwargs)

        config_final = config
        if tools:
            config_final_kwargs = dict(config_kwargs)
            if self.settings.ai.search.web_search_tool != "builtin":
                config_final_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE,
                    )
                )
            else:
                config_final_kwargs.pop("tools", None)
            config_final = types.GenerateContentConfig(**config_final_kwargs)

        for attempt in range(self.settings.retry.per_run + 1):
            contents: list[Any] = [video, prompt.user_prompt]
            search_turns = 0

            try:
                while True:
                    await self.request_limiter.try_acquire_async("rpm")
                    await self.token_limiter.try_acquire_async("tpm", weight=tokens)

                    current_config = config_final if search_turns >= MAX_SEARCH_TURNS else config
                    response = await self._client.aio.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=current_config,
                    )

                    candidate = response.candidates[0] if response.candidates else None
                    if candidate is None:
                        raise RuntimeError(f"No response candidate returned by {self.model_name}")

                    # Check for function calls
                    function_calls = response.function_calls
                    if function_calls and search_turns < MAX_SEARCH_TURNS:
                        search_turns += 1

                        async def _run_call(call: types.FunctionCall) -> tuple[str, Any]:
                            call_name = call.name or ""
                            result = await self._execute_tool_call(call_name, call.args or {})
                            return call_name, result

                        call_results = await asyncio.gather(*[_run_call(fc) for fc in function_calls])

                        response_parts = [
                            types.Part.from_function_response(
                                name=call_name,
                                response={"result": res},
                            )
                            for call_name, res in call_results
                        ]

                        contents.append(candidate.content)
                        contents.append(types.Content(role="user", parts=response_parts))
                        continue

                    # Final response parsing
                    text_content = response.text or ""
                    try:
                        output = response_type.model_validate_json(
                            text_content,
                            context={
                                "video_duration_ms": video_duration_ms,
                                "validation_buffer_ms": self.settings.ai.validation_buffer_ms,
                            },
                        )
                    except Exception:
                        repaired_data = json_repair.loads(text_content)
                        output = response_type.model_validate(
                            repaired_data,
                            context={
                                "video_duration_ms": video_duration_ms,
                                "validation_buffer_ms": self.settings.ai.validation_buffer_ms,
                            },
                        )

                    # Extract thoughts if present
                    if hasattr(output, "thoughts") and candidate.content and candidate.content.parts:
                        thinking_parts: list[str] = [
                            text
                            for part in candidate.content.parts
                            if getattr(part, "thought", False) and (text := part.text) is not None
                        ]
                        if thinking_parts:
                            setattr(output, "thoughts", "\n".join(thinking_parts))

                    return output

            except APIError as e:
                if _is_free_tier_quota_exceeded(e):
                    self._quota_exceeded = True
                    logfire.warning(f"Free tier daily quota exceeded for {self.model_name}. Halting pipeline.")
                    raise QuotaExceededError(str(e)) from e

                is_retryable = e.code in (429, 500, 502, 503, 504) if e.code else True
                if is_retryable and attempt < self.settings.retry.per_run:
                    backoff = min(
                        float(self.settings.retry.max_wait_seconds),
                        (self.settings.retry.min_wait_seconds * (self.settings.retry.multiplier**attempt))
                        + random.uniform(0, 1),
                    )
                    logfire.info(
                        f"Transient API error in {self.model_name} (attempt {attempt + 1}). "
                        f"Retrying in {backoff:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

            except (ConnectionError, asyncio.TimeoutError) as e:
                if attempt < self.settings.retry.per_run:
                    backoff = min(
                        float(self.settings.retry.max_wait_seconds),
                        (self.settings.retry.min_wait_seconds * (self.settings.retry.multiplier**attempt))
                        + random.uniform(0, 1),
                    )
                    logfire.info(
                        f"Network error in {self.model_name} (attempt {attempt + 1}). "
                        f"Retrying in {backoff:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise RuntimeError(f"Agent failed to run: retry limit ({self.settings.retry.per_run}) reached.")
