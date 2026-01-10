import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import logfire
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from ai_sub.data_models import AiResponse
from ai_sub.prompt import SUBTITLES_PROMPT


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

            # Run gemini-cli via subprocess.Popen and parse its response.
            # We use Popen instead of run because on Windows with shell=True, the timeout
            # only kills the shell, leaving the child process running and holding pipes open.
            # As a result, this Python program will just hang waiting forever for the pipes to close.
            # This allows us to manually kill the process tree on timeout.
            cmd = [
                "gemini",
                "-p",
                f"@{video.name}",
                "--model",
                self.model_name,
                "--output-format",
                "json",
            ]
            try:
                with subprocess.Popen(
                    cmd,
                    cwd=video_directory,
                    env=os.environ | {"GEMINI_SYSTEM_MD": "prompt.md"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    # On Windows, shell=True is required to execute batch files/scripts (like npm binaries).
                    # On Linux, shell=True with a list of args causes the args to be passed to the shell itself,
                    # effectively stripping them from the command, so we must use shell=False.
                    shell=sys.platform == "win32",
                ) as process:
                    try:
                        stdout, stderr = process.communicate(timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        if sys.platform == "win32":
                            # Kill the entire process tree
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                capture_output=True,
                            )
                        process.kill()
                        process.communicate()
                        raise

                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(
                            process.returncode, cmd, output=stdout, stderr=stderr
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

            try:
                cli_response = GeminiCliResponse.model_validate_json(stdout)
            except ValidationError:
                logfire.error(f"Failed to validate Gemini CLI output: {stdout}")
                raise
            if cli_response.response is not None:
                try:
                    # There is usually leading and trailing ''' characters.
                    # repair_json will take care of it
                    json_str = repair_json(cli_response.response)
                    ai_response = AiResponse.model_validate_json(json_str)
                except ValidationError:
                    logfire.debug(f"GeminiCliResponse: {cli_response}")
                    logfire.error(f"Failed to validate JSON: {json_str}")
                    raise
                ai_response.model_name = self.model_name
                logfire.debug(
                    f"GeminiCliResponse: {cli_response}\nAiResponse: {ai_response}"
                )
                return ai_response
            else:
                return None


# TODO: Delete later - Just for testing
if __name__ == "__main__":
    logfire.configure()

    wrapper = GeminiCliWrapper("gemini-3-flash-preview")
    result = wrapper.run_sync(
        SUBTITLES_PROMPT,
        Path(
            "C:\\Tools\\tmp_【MV】ジェヘナ (Gehenna) 【IRyS x Mumei Cover】 [9qkkPOD85YE]\\part_000.webm"
        ),
    )
    print(result)
