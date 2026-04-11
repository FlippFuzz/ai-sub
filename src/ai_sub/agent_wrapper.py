"""AI Agent wrapper for the subtitle generation pipeline."""

from __future__ import annotations as _annotations

import asyncio
from pathlib import Path
from typing import Sequence, TypeVar, cast

import logfire
from google import genai as genai
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    ThinkingConfigDict,
)
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, ModelRequestContext, RunContext, WebSearchTool
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import DocumentUrl
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pyrate_limiter import Duration, limiter_factory

from ai_sub.config import Settings
from ai_sub.data_models import AgentDeps
from ai_sub.gemini_cli_model import GeminiCliModel
from ai_sub.ollama_web_search import ollama_web_search_multi

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


async def _rate_limit(ctx: RunContext[AgentDeps], request_context: ModelRequestContext) -> ModelRequestContext:
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
    (e.g., Google vs CLI).
    """

    settings: Settings
    model_name: str

    def is_google(self) -> bool:
        """Checks if the model is a Google model.

        Returns:
            bool: True if the model is a Google model, False otherwise.

        """
        return self.model_name.lower().startswith("google-gla")

    def is_gemini_cli(self) -> bool:
        """Checks if the model is a Gemini CLI model.

        Returns:
            bool: True if the model is a Gemini CLI model, False otherwise.

        """
        return self.model_name.lower().startswith("gemini-cli")

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
        self.deps = deps or AgentDeps()

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
        """
        builtin_tools = []
        function_tools = []

        agent: Agent[AgentDeps]
        hooks: Sequence[AbstractCapability[AgentDeps]] = [Hooks(before_model_request=_rate_limit)]

        if self.use_web_search:
            if self.settings.ai.web_search_tool == "ollama":
                function_tools.append(ollama_web_search_multi)
            elif self.settings.ai.web_search_tool == "duckduckgo":
                from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

                function_tools.append(duckduckgo_search_tool())
            else:
                builtin_tools.append(WebSearchTool())

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

            model = GoogleModel(
                model_str,
                provider=GoogleProvider(
                    api_key=(self.settings.ai.google.key.get_secret_value() if self.settings.ai.google.key else None),
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

            if builtin_tools or function_tools:
                agent = Agent(
                    model=model,
                    model_settings=google_model_settings,
                    deps_type=AgentDeps,
                    builtin_tools=builtin_tools,
                    tools=function_tools,
                    capabilities=hooks,
                )
            else:
                agent = Agent(
                    model,
                    model_settings=google_model_settings,
                    capabilities=hooks,
                    deps_type=AgentDeps,
                )

            return agent
        elif self.is_gemini_cli():
            # Gemini CLI model does not support external tools / builtin_tools.
            if self.use_web_search:
                logfire.warn(
                    f"Web search is enabled (settings.ai.web_search_tool='{self.settings.ai.web_search_tool}'), "
                    f"but the Gemini CLI model ({self.model_name!r}) does not support external tools. "
                    f"The WebSearchTool/duckduckgo_search_tool will be skipped. "
                    f"Agent(model=model) will be created without web search capabilities."
                )
            model_str = self.model_name.split(":", 1)[-1]
            model = GeminiCliModel(model_str, self.settings.ai.gemini_cli)
            return Agent(model=model)
        else:
            # TODO: Do we need to enable thinking, etc for other models?
            # For now, this is only tested to work against Google
            if builtin_tools or function_tools:
                return Agent(
                    model=self.model_name,
                    builtin_tools=builtin_tools,
                    tools=function_tools,
                )
            else:
                return Agent(model=self.model_name)

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
            ValueError: If Gemini CLI is used but a local file path is not provided.

        """
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
        elif self.is_gemini_cli():
            if isinstance(video, Path):
                python_file = cast(Path, video)
                # Pass file path as a DocumentUrl with file:// scheme
                user_prompt = [
                    DocumentUrl(url=python_file.as_uri()),
                    prompt,
                ]
            else:
                raise ValueError("Gemini CLI requires a local file path.")

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
            ollama_search=self.deps.ollama_search,
        )

        # Execute the AI agent to generate subtitles and get a structured response.
        result = await self.agent.run(user_prompt=user_prompt, output_type=response_type, deps=deps)

        return result.output
