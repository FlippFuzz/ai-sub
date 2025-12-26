import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import logfire
from json_repair import repair_json
from pydantic import BaseModel

from ai_sub.data_models import AiResponse
from ai_sub.prompt import PROMPT


class GeminiCliResponseModelStats(BaseModel):
    api: Dict[str, Any]
    tokens: Dict[str, int]


class GeminiCliResponseToolsStats(BaseModel):
    totalCalls: int
    totalSuccess: int
    totalFail: int
    totalDurationMs: int
    totalDecisions: Dict[str, int]
    byName: Dict[str, Any]


class GeminiCliResponseFilesStats(BaseModel):
    totalLinesAdded: int
    totalLinesRemoved: int


class GeminiCliResponseStats(BaseModel):
    models: Dict[str, GeminiCliResponseModelStats]
    tools: GeminiCliResponseToolsStats
    files: GeminiCliResponseFilesStats


class GeminiCliResponseError(BaseModel):
    type: str
    message: str
    code: Optional[int] = None


class GeminiCliResponse(BaseModel):
    response: Optional[str] = None
    stats: Optional[GeminiCliResponseStats] = None
    error: Optional[GeminiCliResponseError] = None


class GeminiCliWrapper:
    """
    A wrapper for the Gemini CLI tool to execute prompts against local files.
    """

    model_name: str
    timeout: int

    def __init__(self, model_name: str, timeout: int = 600):
        self.model_name = model_name
        self.timeout = timeout

    def run_sync(self, prompt: str, video: Path) -> AiResponse | None:
        """
        Runs the Gemini CLI synchronously.

        Args:
            prompt (str): The prompt text.
            video (Path): The path to the video file.

        Returns:
            AiResponse | None: The parsed response or None if execution failed.
        """
        video_directory = video.parent

        with logfire.span("Using Gemini CLI to generate subtitles", _level="debug"):
            # Write the prompt to a .md file in the video directory
            prompt_file = video_directory / "prompt.md"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)

            # Run gemini-cli via subprocess.run and parse it's response
            try:
                raw_result = subprocess.run(
                    [
                        "gemini",
                        "-p",
                        f"@{video.name}",
                        "--model",
                        self.model_name,
                        "--output-format",
                        "json",
                    ],
                    cwd=video_directory,
                    env=os.environ | {"GEMINI_SYSTEM_MD": "prompt.md"},
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    shell=True,
                    timeout=self.timeout,
                    check=True,
                )
            except subprocess.TimeoutExpired as e:
                logfire.error(
                    f"Gemini CLI timed out.\nStdout: {e.stdout}\nStderr: {e.stderr}"
                )
                raise
            except subprocess.CalledProcessError as e:
                logfire.error(
                    f"Gemini CLI failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
                )
                raise

            cli_response = GeminiCliResponse.model_validate_json(raw_result.stdout)
            logfire.debug(f"GeminiCliResponse: {cli_response}")
            if cli_response.response is not None:
                # There is usually leading and trailing ''' characters.
                # repair_json will take care of it
                ai_response = AiResponse.model_validate_json(
                    repair_json(cli_response.response)
                )
                ai_response.model_name = self.model_name
                # logfire.debug(f"AiResponse: {result}")
                return ai_response
            else:
                return None


# TODO: Delete later - Just for testing
if __name__ == "__main__":
    logfire.configure()

    wrapper = GeminiCliWrapper("gemini-3-flash-preview")
    result = wrapper.run_sync(
        PROMPT,
        Path(
            "C:\\Tools\\tmp_【MV】ジェヘナ (Gehenna) 【IRyS x Mumei Cover】 [9qkkPOD85YE]\\part_000.webm"
        ),
    )
    print(result)
