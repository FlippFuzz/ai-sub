import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from threading import Lock
from time import sleep, time

from google.genai import Client
from google.genai.types import (
    File,
    FileState,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    ListFilesConfig,
    SafetySetting,
    Schema,
    ThinkingConfig,
    Type,
    UploadFileConfig,
)
from retrying import retry

from ai_sub.models import (
    GeminiResponse,
    GenerateSubtitleResponse,
)

PROMPT_GENERATE_SUBTITLE_TEMPLATE = dedent(
    """
    You are an expert subtitling and transcription AI. Your primary task is to generate precise, contextually rich English and Japanese subtitles for the provided video content.

    **Key Principles & Requirements:**

    1.  **TIMING ACCURACY (ABSOLUTE HIGHEST PRIORITY):**
        *   **Ensure the `start` and `end` timestamps for each subtitle entry are meticulously accurate.** Both must align perfectly with the *actual beginning and end of the spoken words* for that specific segment.
        *   **CRITICAL: The `end` timestamp is frequently inaccurate and requires special attention.** It must precisely mark the *exact conclusion* of the spoken content for that subtitle entry.
        *   Timestamps must always be in `mm:ss` (minutes:seconds).

    2.  **CRITICAL: Holistic Translation, Contextual Understanding & Comprehensive Analysis:**
        *   **You ABSOLUTELY MUST watch the entire video thoroughly** to understand the visual context, speaker actions, and overall narrative.
        *   **When a single logical sentence or phrase is split across multiple subtitle entries (e.g., due to timing or character limits), you MUST mentally combine these parts to understand the full, complete meaning before translating.**
        *   **NEVER translate individual subtitle entries in isolation if they are part of a larger, continuous thought or sentence.** The translation for each entry must be a coherent and accurate part of the complete, overarching sentence.
        *   Analyze *all* spoken dialogue in the video to produce highly accurate and complete transcriptions.
        *   For songs or singing sections, prioritize on-screen lyrics in the video for subtitles over transcribing the singing.
        *   If the audio is noisy or unclear, leverage *all* contextual cues (visuals from the video, surrounding dialogue) to infer and generate the most probable dialogue.
        *   Utilize the comprehensive understanding gained from *watching the entire video* to enhance the accuracy of transcriptions, translations, and overall subtitle quality. This includes understanding nuances, resolving ambiguities, and ensuring the subtitle content aligns perfectly with what is happening visually.
        *   **Example:**
            *   Original Japanese: "なんでも打ち明けられるママにも" (Subtitle 1) + "言えないことも全部" (Subtitle 2)
            *   Incorrect Translation (isolated): "You're like a mom I can confide anything in." + "Even things I can't say to anyone else."
            *   Correct Translation (holistic): "I can confide everything," (Subtitle 1) + "everything I can't even tell my mum." (Subtitle 2)

    3.  **Dual Language Provision & Translation Quality:**
        *   Provide both English and Japanese subtitles for *all* spoken dialogue.
        *   If the original dialogue is in English, transcribe it and provide an accurate, natural-sounding Japanese translation.
        *   If the original dialogue is in Japanese, transcribe it and provide an accurate, natural-sounding English translation.
        *   Ensure translations are culturally appropriate and maintain the original meaning, tone, and intent, taking into account the full context of the video and any split sentences.

    4.  **Readability & Formatting:**
        * Long sentences MUST be split into multiple, shorter subtitle entries, each with its own start and end time, for better readability on screen. Each subtitle entry (english and japanese) MUST be a maximum of {max_subtitle_chars} characters. Do not use newline characters within a single subtitle entry.

    ** Output Format **
    * Your entire response must be a single JSON object.
    * Timestamps must always be in `mm:ss` (minutes:seconds).

    ** Example Output **

    ```json
    {{
      "subtitles": [
        {{
          "start": "00:00",
          "end": "00:04",
          "english": "Hello everyone, and welcome to our presentation.",
          "japanese": "皆様、プレゼンテーションへようこそ。"
        }},
        {{
          "start": "00:05",
          "end": "00:06",
          "english": "Today, we'll be discussing the future of AI",
          "japanese": "本日は、AIの未来について"
        }},
        {{
          "start": "00:06",
          "end": "00:08",
          "english": "and its impact on various industries worldwide.",
          "japanese": "その影響について議論します。"
        }},
        {{
          "start": "00:09",
          "end": "00:11",
          "english": "It's a very important topic for all of us.",
          "japanese": "これは私たち全員にとって非常に重要なテーマです。"
        }},
        {{
          "start": "00:11",
          "end": "00:14",
          "english": "We sincerely appreciate your attendance today.",
          "japanese": "本日はご出席いただき誠にありがとうございます。"
        }}
      ]
    }}
    ```
    """
).strip()


logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        """Initializes the RateLimiter with requests per minute (rpm) and tokens per minute (tpm).

        Args:
            rpm (int): The maximum number of requests allowed per minute.
            tpm (int): The maximum number of tokens allowed per minute.
        """
        self.rpm = rpm
        self.tpm = tpm
        self.requests_lock = Lock()
        self.tokens_lock = Lock()
        self.last_request_time = 0.0
        self.request_count = 0
        self.last_token_reset_time = 0.0
        self.token_count = 0

    def _replenish_requests(self):
        """Resets the request count if a minute has passed since the last reset."""
        current_time = time()
        elapsed_time = current_time - self.last_request_time
        if elapsed_time >= 60:  # 1 minute
            self.request_count = 0
            self.last_request_time = current_time

    def _replenish_tokens(self):
        """Resets the token count if a minute has passed since the last reset."""
        current_time = time()
        elapsed_time = current_time - self.last_token_reset_time
        if elapsed_time >= 60:  # 1 minute
            self.token_count = 0
            self.last_token_reset_time = current_time

    def wait_for_request(self):
        """Waits until a new request can be made without exceeding the RPM limit.

        If the request limit is reached, it calculates the time to wait until the
        next minute window opens and pauses execution.
        """
        with self.requests_lock:
            self._replenish_requests()
            while self.request_count >= self.rpm:
                current_time = time()
                time_to_wait = (self.last_request_time + 60) - current_time
                logger.info(
                    f"Rate limit exceeded for requests. Current: {self.request_count} RPM: {self.rpm}. Waiting for {time_to_wait:.2f} seconds..."
                )
                if time_to_wait > 0:
                    sleep(time_to_wait)
                self._replenish_requests()
            self.request_count += 1

    def wait_for_tokens(self, tokens_needed: int):
        """Waits until the required number of tokens can be sent without exceeding the TPM limit.

        If adding `tokens_needed` would exceed the TPM limit, it calculates the time
        to wait until the next minute window opens and pauses execution.

        Args:
            tokens_needed (int): The number of tokens required for the upcoming request.
        """
        with self.tokens_lock:
            self._replenish_tokens()
            while self.token_count + tokens_needed > self.tpm:
                current_time = time()
                time_to_wait = (self.last_token_reset_time + 60) - current_time
                logger.info(
                    f"Rate limit exceeded for tokens. Current: {self.token_count}, Needed: {tokens_needed}, TPM: {self.tpm}. Waiting for {time_to_wait:.2f} seconds..."
                )
                if time_to_wait > 0:
                    sleep(time_to_wait)
                self._replenish_tokens()
            self.token_count += tokens_needed


class Gemini:
    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_budget: int,
        rpm: int,
        tpm: int,
        max_subtitle_chars: int,
        num_upload_threads: int,
    ):
        """Initializes the Gemini client with API key and configuration.

        Args:
            api_key (str): Your Google Gemini API key.
            model (str): The Gemini model to use (e.g., "gemini-2.5-flash").
            thinking_budget (int): The thinking budget for Gemini API calls.
            rpm (int): Requests per minute for the Gemini API.
            tpm (int): Tokens per minute for the Gemini API.
            max_subtitle_chars (int): Maximum character length for each subtitle entry.
            num_upload_threads (int): Number of threads to use for parallel file uploads.
        """
        self.client = Client(api_key=api_key)
        self.model = model
        self.thinking_budget = thinking_budget
        self.rate_limiter = RateLimiter(rpm, tpm)
        self.max_subtitle_chars = max_subtitle_chars
        self.num_upload_threads = num_upload_threads

    def send_request(
        self,
        text: str | None = None,
        video: File | None = None,
        config: GenerateContentConfig | None = None,
    ) -> GeminiResponse:
        """Sends a content generation request to the Gemini API.

        This method handles rate limiting, constructs the message parts (text and/or video),
        estimates token usage, and sends the request to the Gemini model. It also
        measures the time taken for the request, including rate limit waits.

        Args:
            text (str | None): Optional text content for the message.
            video (File | None): Optional video file object for the message.
            config (GenerateContentConfig | None): Optional configuration for content generation.

        Returns:
            GeminiResponse: An object encapsulating the Gemini API response and metadata.

        Raises:
            ValueError: If neither 'text' nor 'video' is provided.
        """
        start_time_with_rate_limits = datetime.now()
        self.rate_limiter.wait_for_request()

        chat = self.client.chats.create(model=self.model)
        message_parts = []
        if video:
            message_parts.append(video)
        if text:
            message_parts.append(text)

        if not message_parts:
            raise ValueError("At least one of 'text' or 'video' must be provided.")

        # Estimate tokens before sending the request and wait
        estimated_tokens = 0
        if config and config.system_instruction:
            content_for_counting = [config.system_instruction]
        else:
            content_for_counting = []
        content_for_counting.extend(message_parts)

        try:
            token_count_response = self.client.models.count_tokens(
                model=self.model, contents=content_for_counting
            )
            estimated_tokens = token_count_response.total_tokens or 0
            self.rate_limiter.wait_for_tokens(estimated_tokens)
        except Exception as e:
            logger.warning(
                f"Could not estimate tokens: {e}. Proceeding without token pre-check."
            )

        start_time = datetime.now()
        response = chat.send_message(message=message_parts, config=config)
        end_time = datetime.now()

        end_time_with_rate_limits = datetime.now()

        final_response = GeminiResponse(
            response=response,
            system_instruction=(
                str(config.system_instruction)
                if config and config.system_instruction
                else None
            ),
            user_text=text,
            time=(end_time - start_time).total_seconds(),
            time_with_ratelimit=(
                end_time_with_rate_limits - start_time_with_rate_limits
            ).total_seconds(),
        )

        return final_response

    @retry(wait_fixed=20000, stop_max_attempt_number=3)
    def get_uploaded_files(self) -> dict[str, File]:
        """Retrieves a dictionary of currently uploaded files from the Gemini API.

        This method lists all files associated with the Gemini account and returns
        them in a dictionary where keys are display names and values are File objects.
        It includes a retry mechanism for robustness.

        Returns:
            dict[str, File]: A dictionary mapping file display names to File objects.
        """
        result: dict[str, File] = dict()
        for file in self.client.files.list(config=ListFilesConfig(page_size=100)):
            result[str(file.display_name)] = file
        return result

    def upload_files(self, files_to_upload: list[Path]) -> list[File]:
        """Uploads a list of local video files to the Gemini API.

        This method first checks for existing files with the same display name and
        size to avoid re-uploading. It then uploads new or updated files in parallel
        using a thread pool. Finally, it waits for all uploaded files to become
        active on the Gemini service before returning.

        Args:
            files_to_upload (list[Path]): A list of pathlib.Path objects representing
                                          the local video files to be uploaded.

        Returns:
            list[File]: A list of google.genai.types.File objects representing the
                        successfully uploaded and active files on Gemini.
        """
        existing_files = self.get_uploaded_files()
        files_to_process: list[Path] = []
        uploaded_file_objects: list[File] = []

        # 1. Verify which files need to be re-uploaded.
        for file_path in files_to_upload:
            base_name = file_path.name
            file_size = os.path.getsize(file_path)

            if base_name in existing_files:
                existing_file = existing_files[base_name]
                if existing_file.size_bytes == file_size:
                    logger.info(
                        f"  {file_path} was already uploaded and is up-to-date. Skipping."
                    )
                    uploaded_file_objects.append(existing_file)
                    continue
                else:
                    logger.info(
                        f"  There is already a file with the same display name: {base_name} but different size. Deleting the old one."
                    )
                    file_name_to_delete = existing_file.name
                    if file_name_to_delete:
                        self.client.files.delete(name=file_name_to_delete)
                    else:
                        logger.warning(
                            f"  Could not delete file with display name {base_name} because its name attribute was None."
                        )

            files_to_process.append(file_path)

        # 2. Reupload in parallel, with max-threads = self.num_upload_threads
        logger.info(
            f"Uploading {len(files_to_process)} files in parallel (threads={self.num_upload_threads})..."
        )
        newly_uploaded_files: list[File] = []
        with ThreadPoolExecutor(max_workers=self.num_upload_threads) as executor:
            futures = []
            for file_path in files_to_process:
                base_name = file_path.name
                logger.info(f"  Initiating upload for {base_name}")
                futures.append(
                    executor.submit(
                        self.client.files.upload,
                        file=file_path,
                        config=UploadFileConfig(display_name=base_name),
                    )
                )

            for future in futures:
                try:
                    file = future.result()
                    newly_uploaded_files.append(file)
                    logger.info(f"  Finished uploading {file.display_name}")
                except Exception as exc:
                    logger.error(f"  File upload generated an exception: {exc}")
                    logger.exception("  Error during file upload")

        uploaded_file_objects.extend(newly_uploaded_files)

        # 3. Wait all files to be ready.
        logger.info("Waiting on Google for all files to be ready...")
        while True:
            current_uploaded_files_status = self.get_uploaded_files()

            all_ready = True
            for file_obj in uploaded_file_objects:
                if file_obj.display_name not in current_uploaded_files_status:
                    logger.info(
                        f"  {file_obj.display_name} not found in uploaded files yet. Waiting."
                    )
                    all_ready = False
                    break

                current_state = current_uploaded_files_status[
                    file_obj.display_name
                ].state
                if current_state != FileState.ACTIVE:
                    logger.info(
                        f"  {file_obj.display_name} is not yet active. Waiting. Current State: {current_state}"
                    )
                    all_ready = False
                    break

            if all_ready:
                break
            else:
                sleep(5)

        logger.info("  All files ready.")
        return uploaded_file_objects

    @retry(wait_fixed=20000, stop_max_attempt_number=3)
    def generate_subtitles(self, video_file: File) -> GenerateSubtitleResponse:
        """Generates subtitles for a given video file using the Gemini model.

        This method sends a request to the Gemini API with the video file and a
        pre-defined prompt for subtitle generation. It configures the request
        with thinking budget, response MIME type, and a schema for the expected
        JSON output. It also sets safety settings to BLOCK_NONE for all categories.
        Includes a retry mechanism for transient errors.

        Args:
            video_file (File): The google.genai.types.File object representing the
                               video for which subtitles are to be generated.

        Returns:
            GenerateSubtitleResponse: An object containing the generated subtitles
                                      and other response metadata.

        Raises:
            Exception: If the Gemini API response text is None or if any other
                       unhandled exception occurs during the process.
        """
        try:
            gemini_response = self.send_request(
                video=video_file,
                config=GenerateContentConfig(
                    system_instruction=PROMPT_GENERATE_SUBTITLE_TEMPLATE.format(
                        max_subtitle_chars=self.max_subtitle_chars
                    ),
                    thinking_config=ThinkingConfig(
                        thinking_budget=self.thinking_budget, include_thoughts=True
                    ),
                    response_mime_type="application/json",
                    response_schema=Schema(
                        type=Type.OBJECT,
                        required=["subtitles"],
                        properties={
                            "subtitles": Schema(
                                type=Type.ARRAY,
                                items=Schema(
                                    type=Type.OBJECT,
                                    required=["start", "end", "english", "japanese"],
                                    properties={
                                        "start": Schema(
                                            type=Type.STRING,
                                        ),
                                        "end": Schema(
                                            type=Type.STRING,
                                        ),
                                        "english": Schema(
                                            type=Type.STRING,
                                        ),
                                        "japanese": Schema(
                                            type=Type.STRING,
                                        ),
                                    },
                                ),
                            ),
                        },
                    ),
                    safety_settings=[
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_NONE,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_NONE,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ],
                ),
            )
            logger.info(f"    Duration: {gemini_response.time_with_ratelimit} seconds")
            response = gemini_response.response

            # Validate responses
            # If there is an error the @retry decorator will catch it and retry the request.

            # Response Check 01 - Response text is None
            if response.text is None:
                logger.error(gemini_response.model_dump_json())
                raise ValueError(
                    "Response Check 01 - Response text is None. Retrying..."
                )

            # Response Check 02 - Invalid JSON returned
            try:
                # This will trigger the json.loads() call in GenerateSubtitleResponse.get_ssafile()
                # and raise JSONDecodeError if the JSON is invalid.
                subtitles_response = GenerateSubtitleResponse(
                    **gemini_response.model_dump()
                )
            except json.JSONDecodeError:
                logger.error(gemini_response.model_dump_json())
                raise ValueError(
                    "Response Check 02 - Invalid JSON returned. Retrying..."
                )

            # Response Check 03 - Invalid chronological order of timestamps
            ssa_file = subtitles_response.get_ssafile()

            last_end_time_ms = 0
            for event in ssa_file.events:
                if event.start > event.end:
                    logger.error(gemini_response.model_dump_json())
                    raise ValueError(
                        "Response Check 03 - Invalid chronological order of timestamps. Retrying..."
                        f"Start time ({event.start}ms) is after end time ({event.end}ms)."
                    )

                if event.start < last_end_time_ms:
                    logger.error(gemini_response.model_dump_json())
                    raise ValueError(
                        "Response Check 03 - Invalid chronological order of timestamps. Retrying..."
                        f"Current start time ({event.start}ms) is before previous end time ({last_end_time_ms}ms)."
                    )

                last_end_time_ms = event.end

            return subtitles_response
        except Exception:
            logger.exception("  Error generating subtitles")
            raise
