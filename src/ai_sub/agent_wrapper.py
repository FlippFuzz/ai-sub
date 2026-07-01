"""AI Agent wrapper for the subtitle generation pipeline."""

from __future__ import annotations as _annotations

import asyncio
import random
from pathlib import Path
from typing import Sequence, TypeVar, cast

import logfire
from google import genai as genai
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    ThinkingConfigDict,
)
from httpx import HTTPStatusError, TransportError
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, ModelRequestContext, RunContext, WebSearchTool
from pydantic_ai.capabilities import AbstractCapability, Hooks, NativeTool
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import DocumentUrl
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pyrate_limiter import Duration, limiter_factory

from ai_sub.config import Settings
from ai_sub.data_models import AgentDeps, QuotaExceededError
from ai_sub.web_search_langsearch import web_search_langsearch_multi
from ai_sub.web_search_ollama import web_search_ollama_multi

T = TypeVar("T", bound=BaseModel)


def _calculate_tokens(text: str, video_duration_ms: int) -> int:
    """Estimates the number of tokens for a given text and video duration.

    This is a rough estimation used for rate limiting purposes. The actual
    token count may vary depending on the model and tokenizer.

    The estimation is based on:
    - Text: A simple character count.
    - Video: A fixed rate of tokens per second of video.

    Args:
        text (str): The text prompt.
        video_duration_ms (int): The duration of the video in milliseconds.

    Returns:
        int: The estimated number of tokens.

    """
    # TODO: Make this more accurate. This is just a rough estimation
    return int(len(text) + (video_duration_ms / 1000) * 300)


def _is_free_tier_quota_exceeded(e: Exception) -> bool:
    """Checks if an exception indicates a Google Free Tier daily quota exhaustion.

    This logic supports a two-stage check:
    1. During retries: It attempts to inspect HTTPStatusError, though this usually
       fails because bodies aren't read yet in async streams.
    2. After exhaustion: It inspects ModelHTTPError, which contains the fully
       buffered error message from the provider.

    Args:
        e: The exception to check.

    Returns:
        bool: True if it is a daily (hard) quota exhaustion error.
    """
    if isinstance(e, HTTPStatusError):
        try:
            # At the tenacity transport level, the response body is typically unread.
            # Accessing .text might raise ResponseNotRead. We catch it to allow
            # the retry logic to treat this as a generic (retryable) 429.
            body = e.response.text.lower()
        except Exception:
            body = ""

        return e.response.status_code == 429 and "GenerateRequestsPerDayPerProjectPerModel-FreeTier".lower() in body

    if isinstance(e, ModelHTTPError):
        # When pydantic-ai raises ModelHTTPError, it has already read the body.
        # This is the reliable place to detect the daily free tier limit.
        return e.status_code == 429 and "GenerateRequestsPerDayPerProjectPerModel-FreeTier".lower() in str(e).lower()
    return False


async def _rate_limit(ctx: RunContext[AgentDeps], request_context: ModelRequestContext) -> ModelRequestContext:
    """Hook for enforcing rate limits before making a model request.

    Args:
        ctx: The run context containing agent dependencies.
        request_context: The context for the current model request.

    Returns:
        The unmodified request context.
    """
    deps = ctx.deps

    request_limiter = deps.request_limiter
    token_limiter = deps.token_limiter

    if request_limiter:
        await request_limiter.try_acquire_async("rpm")

    if token_limiter:
        await token_limiter.try_acquire_async("tpm", weight=deps.request_tokens)

    return request_context


class RateLimitedAgentWrapper:
    """A wrapper around the Pydantic AI Agent.

    Handles rate limiting, token usage tracking, and model-specific configurations
    (e.g., Google).
    """

    settings: Settings
    model_name: str

    def is_google(self) -> bool:
        """Checks if the model is a Google model.

        Returns:
            bool: True if the model is a Google model, False otherwise.

        """
        return self.model_name.lower().startswith("google-gla")

    def __init__(
        self,
        settings: Settings,
        model_name: str,
        deps: AgentDeps | None = None,
        use_web_search: bool = False,
    ):
        """Initializes the agent wrapper with settings.

        Args:
            settings (Settings): The application configuration settings.
            model_name (str): The name of the model to use.
            deps: Optional dependencies to pass to the agent. Defaults to a new ``AgentDeps`` instance.
            use_web_search (bool): Whether to enable the web search tool.

        """
        self.settings = settings
        self.model_name = model_name
        self.use_web_search = use_web_search
        self._quota_exceeded = False
        self.deps = deps or AgentDeps(validation_buffer_ms=settings.ai.validation_buffer_ms)

        self.request_limiter = limiter_factory.create_inmemory_limiter(
            rate_per_duration=self.settings.ai.rpm, duration=Duration.MINUTE
        )
        self.token_limiter = limiter_factory.create_inmemory_limiter(
            rate_per_duration=self.settings.ai.tpm, duration=Duration.MINUTE
        )
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[AgentDeps]:
        """Creates and configures the Pydantic AI Agent based on the model type.

        Returns:
            Agent: The configured Pydantic AI Agent.

        Raises:
            ValueError: If the Google AI API key is missing.
        """
        function_tools = []

        agent: Agent[AgentDeps]
        capabilities: Sequence[AbstractCapability[AgentDeps]] = []
        capabilities.append(Hooks(before_model_request=_rate_limit))

        if self.use_web_search:
            if self.settings.ai.search.web_search_tool == "ollama":
                function_tools.append(web_search_ollama_multi)
            elif self.settings.ai.search.web_search_tool == "langsearch":
                function_tools.append(web_search_langsearch_multi)
            elif self.settings.ai.search.web_search_tool == "duckduckgo":
                from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

                function_tools.append(duckduckgo_search_tool())
            else:
                capabilities.append(NativeTool(WebSearchTool()))

        if self.is_google():
            model_str = self.model_name.split(":", 1)[-1]
            # Configure thinking levels for Google models
            # https://ai.google.dev/gemini-api/docs/thinking
            thinking_config: ThinkingConfigDict
            if model_str.lower().startswith("gemini-3"):
                thinking_config = {
                    "include_thoughts": True,
                    "thinking_level": genai.types.ThinkingLevel.HIGH,
                }
            elif model_str.lower().startswith("gemini-2.5-pro"):
                thinking_config = {
                    "include_thoughts": True,
                    "thinking_budget": 32768,
                }
            else:
                thinking_config = {
                    "include_thoughts": True,
                    "thinking_budget": 24576,
                }

            if self.settings.ai.google.key is None:
                raise ValueError("Google AI API key is missing.")

            model = GoogleModel(
                model_str,
                provider=GoogleProvider(
                    api_key=self.settings.ai.google.key.get_secret_value(),
                    http_client=None,
                    base_url=(str(self.settings.ai.google.base_url) if self.settings.ai.google.base_url else None),
                ),
            )
            google_model_settings = GoogleModelSettings(
                google_thinking_config=thinking_config,
                google_safety_settings=[
                    {
                        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                ],
            )

            if function_tools:
                agent = Agent(
                    model=model,
                    model_settings=google_model_settings,
                    deps_type=AgentDeps,
                    tools=function_tools,
                    capabilities=capabilities,
                    validation_context=lambda ctx: {
                        "video_duration_ms": ctx.deps.video_duration_ms,
                        "validation_buffer_ms": ctx.deps.validation_buffer_ms,
                    },
                    retries=self.settings.retry.per_run,
                )
            else:
                agent = Agent(
                    model,
                    model_settings=google_model_settings,
                    capabilities=capabilities,
                    deps_type=AgentDeps,
                    validation_context=lambda ctx: {
                        "video_duration_ms": ctx.deps.video_duration_ms,
                        "validation_buffer_ms": ctx.deps.validation_buffer_ms,
                    },
                    retries=self.settings.retry.per_run,
                )

            return agent
        else:
            # TODO: Do we need to enable thinking, etc for other models?
            # For now, this is only tested to work against Google
            if function_tools:
                return Agent(
                    model=self.model_name,
                    tools=function_tools,
                    capabilities=capabilities,
                    deps_type=AgentDeps,
                    validation_context=lambda ctx: {
                        "video_duration_ms": ctx.deps.video_duration_ms,
                        "validation_buffer_ms": ctx.deps.validation_buffer_ms,
                    },
                    retries=self.settings.retry.per_run,
                )
            else:
                return Agent(
                    model=self.model_name,
                    capabilities=capabilities,
                    deps_type=AgentDeps,
                    validation_context=lambda ctx: {
                        "video_duration_ms": ctx.deps.video_duration_ms,
                        "validation_buffer_ms": ctx.deps.validation_buffer_ms,
                    },
                    retries=self.settings.retry.per_run,
                )

    async def run(
        self,
        prompt: str,
        video: genai.types.File | Path,
        video_duration_ms: int,
        response_type: type[T],
    ) -> T:
        """Runs the AI agent to generate subtitles for the given video.

        Args:
            prompt (str): The system prompt to guide the AI.
            video (genai.types.File | Path): The video file (either a Google File object or a local Path).
            video_duration_ms (int): The duration of the video in milliseconds (used for token estimation).
            response_type (type[T]): The expected Pydantic model for the response.

        Returns:
            T: The structured response containing subtitles or scene data.

        Raises:
            QuotaExceededError: If the model's quota has been exhausted.
            ModelHTTPError: If the AI provider returns an HTTP error other than quota exhaustion.
            HTTPStatusError: If the HTTP request fails with a status error.
            TransportError: If a transport error occurs during the HTTP request.
            ConnectionError: If a network connection error occurs.
            asyncio.TimeoutError: If the request times out.
            RuntimeError: If the retry limit is reached or the agent fails to execute the loop.

        """
        if self._quota_exceeded:
            raise QuotaExceededError(f"Quota previously exceeded for model {self.model_name}")

        # Prepare the prompt
        # Each model provider requires a different input format for video.
        if self.is_google():
            if isinstance(video, genai.types.File):
                file = cast(genai.types.File, video)
                uri = str(file.uri)
                user_prompt = [
                    DocumentUrl(url=uri, media_type=file.mime_type),
                    prompt,
                ]
            else:
                python_file = cast(Path, video)
                data = await asyncio.to_thread(python_file.read_bytes)
                user_prompt = [
                    BinaryContent(data=data, media_type=f"video/{python_file.suffix[1:]}"),
                    prompt,
                ]
        else:
            # For other models (e.g., OpenAI), we read the file into memory
            # and send it as binary content.
            # TODO: This is not tested. Only tested against Google's models
            python_file = cast(Path, video)
            data = await asyncio.to_thread(python_file.read_bytes)
            user_prompt = [
                BinaryContent(data=data, media_type=f"video/{python_file.suffix[1:]}"),
                prompt,
            ]

        # Create sort out dependencies for handling rate limiting
        tokens = _calculate_tokens(prompt, video_duration_ms)
        deps = AgentDeps(
            request_limiter=self.request_limiter,
            token_limiter=self.token_limiter,
            request_tokens=tokens,
            video_duration_ms=video_duration_ms,
            validation_buffer_ms=self.settings.ai.validation_buffer_ms,
            web_search=self.deps.web_search,
        )

        # Execute the AI agent to generate subtitles and get a structured response.
        for attempt in range(self.settings.retry.per_run + 1):
            try:
                result = await self.agent.run(user_prompt=user_prompt, output_type=response_type, deps=deps)
                return result.output
            except (
                ModelHTTPError,
                HTTPStatusError,
                TransportError,
                ConnectionError,
                asyncio.TimeoutError,
            ) as e:
                # 1. Check for hard daily quota (terminal)
                if _is_free_tier_quota_exceeded(e):
                    self._quota_exceeded = True
                    logfire.warning(f"Free tier daily quota exceeded for {self.model_name}. Halting pipeline.")
                    raise QuotaExceededError(str(e)) from e

                # 2. Determine if the error is retryable (transient)
                is_retryable = False
                if isinstance(e, (ModelHTTPError, HTTPStatusError)):
                    status_code = e.status_code if isinstance(e, ModelHTTPError) else e.response.status_code
                    # Transient status codes
                    is_retryable = status_code in (429, 502, 503, 504)
                else:
                    # Network level errors (Connection, Timeout)
                    is_retryable = True

                if is_retryable and attempt < self.settings.retry.per_run:
                    backoff = min(
                        self.settings.retry.max_wait_seconds,
                        (self.settings.retry.min_wait_seconds * (self.settings.retry.multiplier**attempt))
                        + random.uniform(0, 1),
                    )
                    logfire.info(
                        f"Transient error in {self.model_name} (attempt {attempt + 1}). "
                        f"Retrying in {backoff:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                # Final failure after exhausting retries or non-retryable error
                raise

        # Safeguard for type safety: if the loop completes (e.g. per_run is 0)
        raise RuntimeError(f"Agent failed to run: retry limit ({self.settings.retry.per_run}) reached or invalid.")
