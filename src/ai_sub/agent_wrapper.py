from __future__ import annotations as _annotations

from pathlib import Path
from typing import cast

from google import genai as genai
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    ThinkingConfigDict,
)
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import DocumentUrl
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pyrate_limiter import Duration, Limiter, Rate

from ai_sub.config import Settings
from ai_sub.data_models import AiResponse
from ai_sub.gemini_cli_wrapper import GeminiCliWrapper


class RateLimitedAgentWrapper:
    """
    A wrapper around the Pydantic AI Agent that handles rate limiting,
    token usage tracking, and model-specific configurations (e.g., Google vs CLI).
    """

    rpm: int
    tpm: int
    agent: Agent
    cli_wrapper: GeminiCliWrapper | None = None
    settings: Settings

    def is_google(self) -> bool:
        """Checks if the model is a Google model.

        Returns:
            bool: True if the model is a Google model, False otherwise.
        """
        return self.settings.ai.model.lower().startswith("google-gla")

    def is_gemini_cli(self) -> bool:
        """Checks if the model is a Gemini CLI model.

        Returns:
            bool: True if the model is a Gemini CLI model, False otherwise.
        """
        return self.settings.ai.model.lower().startswith("gemini-cli")

    def __init__(self, settings: Settings):
        """
        Initializes the agent wrapper with settings.

        Args:
            settings (Settings): The application configuration settings.
        """
        self.settings = settings

        self.request_limiter = Limiter(Rate(self.settings.ai.rpm, Duration.MINUTE))
        self.token_limiter = Limiter(Rate(self.settings.ai.tpm, Duration.MINUTE))

        if self.is_gemini_cli():
            model_str = settings.ai.model.split(":", 1)[1]
            self.cli_wrapper = GeminiCliWrapper(
                model_str, settings.ai.gemini_cli.timeout
            )
        elif self.is_google():
            model_str = settings.ai.model.split(":", 1)[1]

            # Configure Max thinking possible
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
                    api_key=(
                        settings.ai.google.key.get_secret_value()
                        if settings.ai.google.key
                        else None
                    ),
                    http_client=None,
                    base_url=(
                        str(settings.ai.google.base_url)
                        if settings.ai.google.base_url
                        else None
                    ),
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
            self.agent = Agent(model, model_settings=google_model_settings)
        else:
            # TODO: Do we need to enable thinking, etc for other models?
            # For now, this is only tested to work against Google
            self.agent = Agent(model=settings.ai.model)

    def run(
        self, prompt: str, video: genai.types.File | Path, video_duration_ms: int
    ) -> AiResponse:
        """
        Runs the AI agent to generate subtitles for the given video.

        Args:
            prompt (str): The system prompt to guide the AI.
            video (genai.types.File | Path): The video file (either a Google File object or a local Path).
            video_duration_ms (int): The duration of the video in milliseconds (used for token estimation).

        Returns:
            AiResponse: The structured response containing subtitles.
        """
        # Handle Rate limits
        self.request_limiter.try_acquire("rpm")

        tokens = self._calculate_tokens(prompt, video_duration_ms)
        self.token_limiter.try_acquire("tpm", weight=tokens)

        if self.is_gemini_cli():
            if isinstance(video, Path):
                assert self.cli_wrapper
                result = self.cli_wrapper.run_sync(prompt, video)
                if result:
                    return result
                raise RuntimeError("Gemini CLI failed to generate response")
            else:
                raise ValueError("Gemini CLI requires a local file path.")

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
                data = python_file.read_bytes()
                user_prompt = [
                    BinaryContent(
                        data=data, media_type=f"video/{python_file.suffix[1:]}"
                    ),
                    prompt,
                ]

        else:
            # For other models (e.g., OpenAI), we read the file into memory
            # and send it as binary content.
            # TODO: This is not tested. Only tested against Google's models
            python_file = cast(Path, video)
            data = python_file.read_bytes()
            user_prompt = [
                BinaryContent(data=data, media_type=f"video/{python_file.suffix[1:]}"),
                prompt,
            ]

        # Execute the AI agent to generate subtitles and get a structured response.
        result = self.agent.run_sync(user_prompt=user_prompt, output_type=AiResponse)
        result.output.model_name = result.response.model_name
        return result.output

    def _calculate_tokens(self, text: str, video_duration_ms: int) -> int:
        """
        Roughly calculates the number of tokens.
        - Text: 1 token per character.
        - Video: 300 tokens per second.
        """
        # TODO: Make this more accurate. This is just a rough estimation
        return int(len(text) + (video_duration_ms / 1000) * 300)
