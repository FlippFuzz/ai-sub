# AI Sub Release Notes

## v1.10.1

This release fixes a critical issue preventing the application from starting when installed via pip.

**Bug Fixes:**

- **Script Entry Point:** Encapsulated the application startup logic within a `main()` function in `ai_sub.main`. This resolves an `ImportError` where the `ai-sub` command (defined in `pyproject.toml` as `ai_sub.main:main`) failed to find the `main` function because the logic was previously contained solely within an `if __name__ == "__main__":` block.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.10.0...v1.10.1

---

## v1.10.0

This release introduces a re-encoding threshold to optimize processing time and includes significant refactoring of the job runner architecture.

**New Features & Improvements:**

- **Re-encoding threshold:** Introduced a `threshold_mb` setting (default: 20MB) to `ReEncodeSettings`. Video segments smaller than this threshold will now skip the re-encoding step, as re-encoding small files often provides minimal size reduction while incurring processing overhead. To re-encode all segments regardless of size, set this threshold to `0`.

**Code Improvements:**

- **JobRunner Refactor:** Refactored `JobRunner` and its subclasses to use `on_complete` callbacks. This decouples job execution from the pipeline flow, moving the logic for creating subsequent jobs into the main application logic.
- **Mutable Defaults:** Fixed a potential issue with mutable default arguments in `JobRunner` constructors. `stop_events` now defaults to `None` instead of `[]`, preventing the sharing of list instances across multiple class instances.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.9.2...v1.10.0

---

## v1.9.2

This release fixes an issue where video segmentation failed if the output directory path contained a percent sign (`%`).

**Bug Fixes:**

- **FFmpeg Path Escaping:** Fixed a bug where FFmpeg's segment muxer would fail or incorrectly interpret paths containing `%` characters (e.g., "Video 100%"). The application now properly escapes these characters in the output directory path, ensuring correct file generation.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.9.1...v1.9.2

---

## v1.9.1

This release fixes a compatibility issue with `pyrate-limiter` v4+.

**Bug Fixes:**

- **Pyrate Limiter Compatibility:** Removed the `raise_when_fail` argument from the `Limiter` constructor in `RateLimitedAgentWrapper`. This argument was removed in `pyrate-limiter` version 4.0.0.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.9.0...v1.9.1

---

## v1.9.0

This release refines the subtitle generation prompt to improve Japanese transcription quality, noise handling, and JSON formatting.

**New Features & Improvements:**

- **Prompt Refinement:** Updated `SUBTITLES_PROMPT` to improve noise handling by explicitly excluding non-speech sounds like applause and laughter. Additionally, on-screen text logic is now more generous, ensuring important context is captured.
- **JSON Formatting:** Added explicit instructions to escape double quotes within JSON strings to ensure valid parsing.

**Bug Fixes:**

- **Japanese Transcription:** Updated the prompt to strictly enforce native script (Kanji/Kana) for the `og` field when transcribing Japanese audio, explicitly forbidding Romaji.
- **Prompt Version:** Incremented `SUBTITLES_PROMPT_VERSION` to 2.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.8.0...v1.9.0

---

## v1.8.0

This release improves the readability of output filenames by refining how AI model names are sanitized.

**New Features & Improvements:**

- **Model Name Sanitization:** Updated `get_sanitized_model_name` to replace non-alphanumeric characters with hyphens (`-`) instead of removing them entirely. This results in more readable filenames (e.g., `gemini-3-pro-preview` instead of `gemini3propreview`). Leading or trailing hyphens are now stripped from the resulting string.

---

## v1.7.0

This release implements standardized return codes for the application and refactors the main entry point.

**New Features & Improvements:**

- **Exit Codes:** The application now returns specific exit codes to indicate the result of the operation: `0` (COMPLETE), `-1` (INCOMPLETE), and `-2` (MAX_RETRIES_EXHAUSTED).

**Code Improvements:**

- **Type Definitions:** Added `AiSubResult` IntEnum in `data_models.py` to formally define execution states.
- **Exports:** Exposed `AiSubResult` in `__init__.py` for easier import.
- **Cleanup:** Removed the `main()` wrapper function in `main.py` and moved execution logic to the `__main__` block.
- **Documentation:** Added missing docstrings to `AiSubResult` and `Job` classes.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.6.2...v1.7.0

---

## v1.6.2

This release fixes an issue where sub-settings classes were not exported from the package root.

**Bug Fixes:**

- **Export Settings Classes:** Updated `__init__.py` to explicitly export all settings models defined in `config.py` (such as `GeminiCliSettings` and `GoogleAiSettings`). Previously, only the main `Settings` class was exported, requiring users to import from the internal `config` module for specific sub-settings.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.6.1...v1.6.2

---

## v1.6.1

This release fixes an issue where the provider prefix was included in the sanitized model name used for filenames.

**Bug Fixes:**

- **Model Name Sanitization:** Updated `get_sanitized_model_name` to strip the provider prefix (e.g., `gemini-cli:`) from the model string, ensuring cleaner filenames (e.g., `gemini3propreview` instead of `geminicligemini3propreview`).

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.6.0...v1.6.1

---

## v1.6.0

This release adds support for disabling default logging configuration and includes the model name in output filenames to prevent collisions.

**New Features & Improvements:**

- **Flexible Logging:** Added an optional `configure_logging` parameter to `ai_sub` (defaulting to `True`). This allows external applications to use `ai_sub` without conflicting with their own logging or `logfire` configuration.
- **Model-Specific Output Files:** Intermediate and final output filenames now include the sanitized AI model name. This prevents file collisions when testing different models on the same video.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.5.0...v1.6.0

---

## v1.5.0

This release introduces a Python API for programmatic usage and adds versioning for subtitle prompts.

**New Features & Improvements:**

- **Python API:** Users can now invoke ai-sub directly within Python scripts, enabling seamless integration into custom workflows.
- **Prompt Versioning:** Added `SUBTITLES_PROMPT_VERSION` to track and manage changes to the subtitle generation prompt.

**Bug Fixes:**

- **Logfire Configuration:** Fixed a `LogfireNotConfiguredWarning` by moving hardware encoder detection logic to run after Logfire initialization.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.4.0...v1.5.0

---

## v1.4.0

This release optimizes token usage, expands language support, and improves subtitle quality through Chain of Thought processing.

**New Features & Improvements:**

- **Universal Language Support:** Updated the prompt to support all input languages, removing the restriction to Japanese.
- **Chain of Thought:** The prompt now requests scene descriptions to trigger Chain of Thought processing, improving subtitle context and accuracy.
- **Token Optimization:** Shortened JSON keys (e.g., `start` to `s`, `end` to `e`) to save output tokens.
- **Smart Display:** Logic added to display only the original subtitle when the English translation is substantially similar.
- **Timestamp Validation:** Enhanced validation logic for subtitle timestamps.
- **Encoder Option:** Added an encoder option to `ReEncodeSettings`.
- **Error Status:** Added "Max Retries Exceeded" status to generated subtitles.

**Bug Fixes:**

- **Pydantic Deprecation:** Switched to `validate_by_name` and `validate_by_alias` to avoid `populate_by_name` deprecation warnings in Pydantic v3.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.3.1...v1.4.0

---

## v1.3.1

This release fixes an issue where videos containing non-UTF-8 data could cause the application to crash.

**Bug Fixes:**

- **UTF-8 Handling:** Improved handling of video files containing non-UTF-8 data to prevent processing failures.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.3.0...v1.3.1

---

## v1.3.0

This release introduces parallel encoding jobs to improve performance and updates the AI prompt to better handle overlapping speech.

**New Features & Improvements:**

- **Parallel Encoding:** Implemented encoding jobs to parallelize video encoding alongside subtitle generation, significantly reducing overall processing time.
- **Prompt Update:** Updated the system prompt to better handle overlapping speech and improve general subtitle quality.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.2.2...v1.3.0

---

## v1.2.2

This release focuses on improving video processing accuracy, fixing Windows-specific issues with the Gemini CLI, and refining the AI prompt.

**New Features & Improvements:**

- **Prompt Refinement:** Updated the system prompt to improve subtitle generation quality.
- **Enhanced Logging:** Added debug logging for Gemini CLI responses and parsed AI responses to aid in troubleshooting.
- **Dependency Cleanup:** Removed `pymediainfo` from the project dependencies.

**Bug Fixes:**

- **Video Splitting Accuracy:** Fixed timing inaccuracies by splitting the video before re-encoding, preventing drift.
- **Windows Timeout Fix:** Fixed an issue where the Gemini CLI timeout was not functioning correctly on Windows systems.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.2.1...v1.2.2

---

## v1.2.1

This release addresses a compatibility issue with the Gemini CLI wrapper on Linux systems.

**Bug Fixes:**

- **Linux Compatibility:** Fixed an issue where the Gemini CLI wrapper failed to execute correctly on Linux environments.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.2.0...v1.2.1

---

## v1.2.0

This release introduces a new logging configuration to control data scrubbing and improves debugging for Gemini CLI integration.

**New Features & Improvements:**

- **Log Scrubbing Control:** Added a new flag `--log.scrub` to enable or disable the scrubbing of sensitive data from logs.

**Bug Fixes:**

- **Gemini CLI Debugging:** Improved error handling to log the original response output when the Gemini CLI output cannot be successfully parsed.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.1.0...v1.2.0

---

## v1.1.0

This release adds a configurable timeout for Gemini CLI operations.

**New Features & Improvements:**

- **Gemini CLI Timeout:** Added a new configuration option `--ai.gemini-cli.timeout` to specify the timeout in seconds for Gemini CLI operations.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.0.2...v1.1.0

---

## v1.0.2

This release introduces hardware acceleration for video encoding and fixes issues with video splitting.

**New Features & Improvements:**

- **Hardware Acceleration:** Added support for hardware acceleration during video encoding to improve processing speed.

**Bug Fixes:**

- **Video Splitting:** Fixed an issue where videos were not correctly split at 20MB intervals. The `--split.max_bytes` argument has been removed.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.0.1...v1.0.2

---

## v1.0.1

This release includes bug fixes for logging configuration and job state persistence, as well as improvements to data validation.

**Bug Fixes:**

- **Logfire Configuration:** Fixed an issue where Logfire was not optional.
- **Job State Persistence:** Fixed a bug where the job state was not saved after a failure.

**Code Improvements:**

- **Data Validation:** Refactored data models to use Pydantic's `NonNegativeInt` and `PositiveInt` for better validation.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.0.0...v1.0.1

---

## v1.0.0

This major release marks a complete re-write of the application architecture to improve flexibility and extensibility.

**New Features & Improvements:**

- **Architecture Overhaul:** The codebase has been completely rewritten to support future growth and maintainability.
- **Multi-Model Support via Pydantic-AI:** Integrated `pydantic-ai` to facilitate seamless integration with various AI models, paving the way for broader model support.
- **Gemini CLI Backend:** Added support for using `gemini-cli` as a backend for processing, offering an alternative to direct API key usage.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.8...v1.0.0

---

## v0.0.8

This release improves subtitle transcription guidance and updates the project's default settings in response to changes in Google's Gemini free tier.

**New Features & Improvements:**

- **Gemini API limits:** Reduced default API settings to match Google's lowered free-tier rate limits and prevent rate-limit errors for users on the free tier.
- **Improved transcription guidance:** Enhanced subtitle-generation guidance to prioritize audio transcription (when available) over static on-screen text, improving accuracy for spoken content.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.7...v0.0.8

---

## v0.0.7

This release introduces a new offset feature for video processing and significant enhancements to subtitle generation prompt instructions.

**New Features & Improvements:**

- **Video Processing Offset:** Added a new `--start_offset_min` argument to allow users to skip a specified number of minutes from the beginning of a video, enabling more flexible processing of long videos.
- **Enhanced Subtitle Prompt Instructions:** The prompt template for subtitle generation has been significantly updated to improve timing precision, enforce strict chronological order, enhance translation accuracy and nuance, and ensure better readability and formatting.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.6...v0.0.7

---

## v0.0.6

This release focuses on minor bug fixes and improvements to the subtitle generation process and project showcase.

**New Features & Improvements:**

- **Prompt Template Update:** The prompt template used for subtitle generation has been updated for improved accuracy.
- **Showcase Updates:** The showcase directory now includes version numbers for SRT files, and a new v0.0.6 SRT file has been added for `42h4ydJS3zk`.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.5...v0.0.6

---

## v0.0.5

This release includes updates to the Gemini API configuration, improved error handling, and minor showcase additions.

**New Features & Improvements:**

- **Gemini API Configuration:** Updated the default thinking budget for the Gemini API to 32768, aligning with the maximum for Gemini 2.5 Pro.
- **Error Handling:** List the video segments that cannot be processed at the end of execution.
- **Showcase Additions:** Added new subtitles to the project showcase.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.4...v0.0.5

---

## v0.0.4

This release focuses on ensuring chronological timestamps in subtitle generation, enhancing error messages, and expanding the project showcase.

**New Features & Improvements:**

- **Chronological Timestamps:** Implemented a mechanism to ensure that timestamps returned by `generate_subtitles` are always chronological, improving subtitle accuracy.
- **Enhanced Error Messages:** Error messages for response checks have been improved to clearly indicate when retries are occurring, providing better debugging information.
- **Showcase Expansion:** The project showcase has been updated with new video entries and corresponding SRT files.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.3...v0.0.4

---

## v0.0.3

This release focuses on improving the robustness of subtitle generation by enhancing error handling for invalid AI model responses.

**New Features & Improvements:**

- **Enhanced Error Logging:** Detailed error messages are now logged for invalid JSON responses received from the Gemini model, providing better insights into issues during subtitle generation.
- **Robust JSON Handling:** The `json-repair` dependency has been removed. The system will now retry when Gemini returns an invalid response, ensuring data integrity and preventing the use of potentially incomplete or corrupted subtitle data.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.2...v0.0.3

---

## v0.0.2

This release includes changes to the default AI model, improvements in subtitle generation, and an expanded showcase.

**New Features & Improvements:**

- **Default Model Update:** The default AI model has been updated to `gemini-2.5-pro`. This change was made possible by Google's updated free tier limits, which now allow up to 100 requests per day against `gemini-2.5-pro`, enabling enhanced performance and accuracy in subtitle generation.
- **AI Prompt Update:** Clarifications have been added regarding subtitle timing accuracy requirements, with an emphasis on the importance of the end timestamp in the prompt template to improve overall subtitle quality.
- **Showcase Expansion:** The project showcase has been updated with new video entries and corresponding SRT files.
- **Improved Subtitle Error Handling:** The subtitle error message and duration handling in [`src/ai_sub/main.py`](src/ai_sub/main.py) have been updated for better clarity and accuracy.
- **Project Description Refinement:** The project description in [`pyproject.toml`](pyproject.toml) has been updated for improved clarity.

**Other Changes:**

- The `RELEASE_NOTES.md` file has been added to the project for future release documentation.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.1...v0.0.2

---

## v0.0.1

This is the initial release of AI Sub.

**Features:**

- **AI-Powered Subtitle Generation:** Leverages Google Gemini for generating English and Japanese subtitles with translation capabilities.
- **Video Segmentation:** Automatically segments input videos into configurable durations for processing.
- **Concurrent Processing:** Supports parallel processing of video segments for efficient subtitle generation.
- **Subtitle Compilation:** Combines all generated subtitle parts into a single, final subtitle file.
