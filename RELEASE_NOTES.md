# AI Sub Release Notes

### v3.2.0

This release introduces a major structural overhaul of the prompt system, utilizing XML-based tags to improve AI instruction clarity and flexibility. It also adds support for capturing and surfacing the model's internal reasoning ("thoughts").

**New Features:**

- **Structured XML-Tagged Prompts:** Refactored all system and user prompts to utilize a structured XML format. This improves instruction grounding, enhances AI capability in complex workflows, and makes it easier to maintain and update prompt templates.
- **Model Reasoning ("Thoughts") Support:** Added support for capturing and returning the AI model's internal "thinking" or reasoning process. This is now available in [`SubtitleAiResponse`](src/ai_sub/data_models.py) and [`LyricsSceneAiResponse`](src/ai_sub/data_models.py) objects as an optional `thoughts` field.

**Fixes & Improvements:**

- **Enhanced Contextual Generation:** Improved subtitle and verification generation to preserve better context when processing scene references and video inputs.
- **Reasoning Capture:** Output handling has been updated to parse and store model reasoning (`ThinkingPart`) when available.

**Refactoring & Chores:**

- **Prompt Class Implementation:** Introduced a dedicated [`Prompt`](src/ai_sub/prompt.py) data model to cleanly separate and manage dynamic system and user instructions.
- **Prompt Versions:** Bumped [`LYRICS_PROMPT_VERSION`](src/ai_sub/prompt.py) to 8 and [`SUBTITLES_PROMPT_VERSION`](src/ai_sub/prompt.py) to 17.

---

### v3.1.0

This release introduces a robust subtitle gap verification and multi-attempt tracking engine, significantly reducing the occurrence of large, untranscribed gaps in AI-generated subtitles.

**New Features:**

- **Dynamic Subtitle Gap Verification:** Introduced automated gap verification to detect and resolve unacceptably large gaps in AI-generated subtitles.
  - Added [`verification_gap_seconds`](src/ai_sub/config.py) and [`gap_verification_retries`](src/ai_sub/config.py) settings to configure the gap threshold and the number of verification passes.
  - The pipeline now automatically triggers re-generation attempts if gaps exceed the defined threshold.
  - Added [`get_verification_prompt`](src/ai_sub/prompt.py) to guide the model through entire-segment regeneration during verification runs.
- **Multi-Attempt Tracking:** [`SubtitleJob`](src/ai_sub/data_models.py) now stores a chronological list of responses (`responses`) to support sequential verification passes, while maintaining backwards compatibility via a property getter.

**Fixes & Improvements:**

- **Improved Gap Detection:** Refactored `SubtitleAiResponse.has_large_gaps` to track maximum end timestamps, preventing false-positive gap detections on overlapping segments.
- **Robustness:** Added checkpointing after the initial generation and each subsequent verification pass to protect progress.
- **Legacy Migration:** Added a model validator to automatically migrate legacy `response` fields to the new `responses` list format upon loading.
- **Documentation:** Updated [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) with new verification settings.

---

## v3.0.3

This release refines the subtitle generation prompt for strict audio grounding and adds essential API key validation for improved runtime safety.

**Fixes & Improvements:**

- **Strict Audio Grounding:** Bumped [`SUBTITLES_PROMPT_VERSION`](src/ai_sub/prompt.py) to 16. The prompt now enforces strict audio grounding, prioritizing audio as the supreme source of truth, replacing "timestamp" with "timecode" throughout, and strengthening anti-hallucination rules regarding reference JSON mismatches.
- **Google AI API Key Validation:** Added missing validation to ensure the Google AI API key is present before creating the `GoogleModel` provider, improving runtime safety. Updated `_create_agent` docstring to include the raised `ValueError`.

---

## v3.0.2

This release enhances the video re-encoding process with stricter FPS verification and improves configuration flexibility for video processing.

**New Features:**

- **FPS Verification:** Added `get_video_fps` to retrieve video framerate using `ffprobe` and updated `reencode_video` to validate both duration and FPS for existing output files, including tolerance logic for low FPS files.
- **Decimal FPS Support:** Updated `ReEncodeSettings.fps` and `ReEncodingJob.fps` to support `PositiveFloat` values. `reencode_video` was updated to handle float FPS and ensure GOP size calculation remains compatible. Updated `docs/CONFIGURATION.md` to reflect the new float type and updated the default value to `1.0`.

**Fixes & Improvements:**

- **Improved Validation Logging:** Enhanced error logging when existing file validation fails.
- **Documentation:** Added `HTTPStatusError` and `TransportError` to the 'Raises' section of the `RateLimitedAgentWrapper.run` docstring to resolve Ruff linting warnings.

---

## v3.0.1

This patch release improves the flexibility of the logging configuration for developers.

**Fixes & Improvements:**

- **Flexible Logging Configuration:** Updated `setup_logging` to accept either the full `Settings` object or just the `LoggingSettings` sub-model. This makes it easier to configure logging programmatically when not using the full CLI configuration suite.

---

## v3.0.0

This major release introduces a completely overhauled retry engine for improved resilience, interactive progress bars for better visibility into long-running tasks, and enhanced library integration for developers.

**New Features:**

- **Interactive Progress Bars:** Integrated `tqdm` to provide real-time status for re-encoding, file uploads, lyrics research, and subtitle generation. Includes new settings to toggle bars (`--log.progress-bars`) and adjust UI refresh rates.
- **Advanced Resilience & Retry Engine:**
  - **Multi-Layered Logic:** Re-engineered the retry system to differentiate between transient network errors (using exponential backoff with jitter) and terminal errors like Google Free Tier quota exhaustion.
  - **Persisted Attempt Tracking:** Job attempts are now tracked independently per stage and saved to disk. This ensures that retry limits are strictly enforced across application restarts, preventing infinite loops on problematic segments.
- **Enhanced Library Integration:**
  - Exposed `setup_logging` as part of the public API for programmatic configuration of telemetry and console output.
  - Exported `TqdmWriteWrapper` to allow external applications to correctly route logs while progress bars are active.

**Fixes & Improvements:**

- **Improved Stage Independence:** Fixed a bug where a "max retries exceeded" state in the lyrics stage would block subtitle generation even if lyrics processing was disabled.
- **Documentation Alignment:** Synchronized `CONFIGURATION.md` and `README.md` with the latest configuration flags and model naming conventions (e.g., transitioning examples to `gemini-3.5-flash`).

---

## v2.10.0b1

This release removes support for the `gemini-cli` backend due to upcoming service changes by Google and includes a fix for millisecond timestamp parsing to improve timing accuracy.

**Breaking Changes:**

- **Removed `gemini-cli` Support:** Support for the `gemini-cli` backend has been dropped. This is because the service will stop serving requests for Google One and unpaid tiers starting June 18th, 2026. Users should transition to using the standard Google AI API keys.

**Fixes & Improvements:**

- **Timestamp Parsing Correction:** Fixed a bug in the `_parse_timestamp_string_ms` utility where single-digit fractional seconds (e.g., `.4`) were incorrectly interpreted as 4ms instead of 400ms. The parser now correctly pads and truncates millisecond strings.
- **Documentation Updates:** Refreshed the README and configuration documentation to reflect the removal of the CLI backend and focus on the Gemini API integration.

---

## v2.9.1

Urgent patch to fix a critical dependency mismatch that broke new installations from PyPI.

The v2.9.0 release introduced a breaking architectural change by migrating to the new `pydantic-ai` `capabilities` API. While `requirements.txt` was correctly updated to `>= 1.99.0`, the package metadata in `pyproject.toml` was accidentally missed. As a result, users performing a fresh `pip install ai-sub` could receive an older, incompatible version of `pydantic-ai-slim`, leading to immediate runtime crashes during agent initialization. This patch synchronizes the metadata to ensure all installation methods enforce the correct version.

**Fixes & Improvements:**

- **Critical Dependency Alignment:** Enforced `pydantic-ai-slim >= 1.99.0` in `pyproject.toml` to strictly match project requirements and `requirements.txt`.
- **Metadata Synchronization:** Bumped project version to `2.9.1` to ensure PyPI correctly distributes the fixed metadata.

## v2.9.0

This release promotes the Langsearch integration and the Pydantic AI capabilities migration to production, while significantly improving pipeline resilience against API instability and enhancing data hygiene.

**New Features:**

- **Promoted Langsearch Integration:** Langsearch is now fully supported as a production-ready web search provider for lyrics and scene detection.
- **Updated Default AI Models:** Default models have been updated to the latest Gemini versions: `gemini-3.5-flash` for subtitles and `gemini-3.1-flash-lite` for lyrics research.

**Fixes & Improvements:**

- **Improved Pipeline Resilience:** Increased default retry limits (`run` increased to 5, `max` to 15) to allow the pipeline more attempts to recover from transient "server unavailable" responses frequently encountered with the Gemini API.
- **Enhanced Data Hygiene:** Updated the subtitle stitching logic to explicitly exclude web search API keys from the metadata header embedded in final SRT files, ensuring cleaner output.
- **Pydantic AI Compatibility:** Finalized the transition of the agent architecture to the new `capabilities` API, ensuring long-term compatibility with the `pydantic-ai` ecosystem.
- **Dependency Management:** Updated minimum requirements for `pydantic-ai-slim` to `>= 1.99.0` to support the updated agent initialization logic.

## v2.9.0b1

This release introduces support for the Langsearch web search provider and updates the underlying Pydantic AI agent initialization to align with the latest `pydantic-ai` capabilities API. Default AI models have also been updated to the latest Gemini Flash versions.

**New Features:**

- **Langsearch Integration:** Added support for `langsearch` as a web search provider for lyrics and scene detection. This can be configured via the `web_search_tool` setting and requires a `LANGSEARCH_API_KEY`.
- **Pydantic AI Capabilities Update:** Updated agent initialization to use the new `capabilities` API for `pydantic-ai`, replacing the deprecated `builtin_tools` argument. This ensures compatibility with newer versions of `pydantic-ai` and aligns with recommended practices for agent configuration.
- **Updated Default AI Models:** The default models for `model_subtitles` and `model_lyrics` have been updated to `gemini-3.5-flash` and `gemini-3.1-flash-lite` respectively.

**Fixes & Improvements:**

- **Unified Web Search Architecture:** Refactored the web search infrastructure to support multiple providers (Ollama, Langsearch) using a unified configuration (`WebSearchSettings`) and dependency model (`WebSearchDeps`). This centralizes HTTP client management, in-memory caching, and rate limiting.

## v2.8.2

This release ensures compatibility with the latest Gemini CLI tool for automated environments.

**Fixes & Improvements:**

- **Gemini CLI Compatibility:** Added the `--skip-trust` flag to the Gemini CLI execution command. This ensures compatibility with the latest version of the tool, which requires explicit trust acknowledgement for non-interactive execution environments, preventing the pipeline from stalling.

## v2.8.1

This release improves the stability of the subtitle generation pipeline by relaxing the timestamp validation constraints.

**Fixes & Improvements:**

- **Increased Validation Buffer:** Adjusted the default `validation_buffer_ms` from 1000ms to 2000ms in `AiSettings`. This provides additional headroom for AI models that frequently generate timestamps slightly exceeding the actual video segment duration, significantly reducing false-positive validation failures and redundant job retries.

## v2.8.0

This release promotes the Ollama web search integration to production, with significant improvements to prompt quality, dependency management, and validation flexibility.

**New Features:**

- **Ollama Web Search Tool:** Added support for Ollama as a web search provider, expanding the available search options beyond DuckDuckGo and builtin provider tools.
  - Introduced `OllamaWebSearchDeps` and `ollama_web_search_multi` function for web search.
  - Added `OllamaSearchSettings` configuration class with API key validation.
  - Updated `web_search_tool` option to include `'ollama'` alongside `'builtin'` and `'duckduckgo'`.
  - Refactored agent initialization to use `AsyncExitStack` for proper resource cleanup.

**Refactoring & Improvements:**

- **AgentDeps Architecture:** Introduced a new `AgentDeps` data model to centralize dependencies passed to Pydantic AI agents. This replaces the loose `Any` type with a typed, extensible container holding `video_duration_ms` and optional `ollama_search` fields.
  - Updated `RateLimitedAgentWrapper` to use typed `AgentDeps` instead of `Any`.
  - Refactored `ollama_web_search` functions to extract dependencies from `ctx.deps`.
  - Updated `main.py` to create and configure `AgentDeps` with Ollama search dependencies.
- **Rate Limiting Migration:** Implemented hook-based rate limiting.
  - Moved rate limiting logic from the `run()` method to model request hooks via a `_rate_limit` hook function.
  - Added token calculation helper `_calculate_tokens()` for accurate rate limit tracking.
  - Improved error message formatting to show human-readable durations.
- **Agent Constructor Updates:** Refactored both Gemini-cli and default Agent initialization to include capabilities (hooks) and `deps_type` (AgentDeps) to ensure that rate limits are enforced across all execution paths.
- **Deferred Response Validation:** Refactored response handling to store AI responses in a local variable, validate against video duration, and only assign to `job.response` if validation passes. This prevents storing potentially invalid responses on the job object.
- **In-Memory Caching for Ollama Search:** Added a cache to `OllamaWebSearchDeps` that stores search results keyed by a normalized query string (punctuation removed, case-folded). This avoids redundant API calls for equivalent queries like `"Hello, world!"` and `"hello world"`, reducing latency and API costs.

**Prompt Engineering:**

- **Lyrics Detection (v7):** Updated `LYRICS_PROMPT_VERSION` to 7 with refined search rules that prioritize original composer names and native titles. Added concrete search query examples for Japanese covers, English covers, and original songs to improve search accuracy.
- **Ollama Search Prompt (v6):** Updated `LYRICS_PROMPT_VERSION` to 6 with refined instructions to prevent excessive search queries. Added explicit rules against query spamming, restrictive quotation marks, and guidance to use native titles for better search results.

**Fixes & Improvements:**

- **Configuration Validation:** Updated config validation to only enforce an Ollama API key when the `web_search_tool` is set to `"ollama"` AND the thread lyrics flag is enabled (lyrics > 0). This allows for flexible configurations where Ollama search is configured but not actively used.
- **Gemini CLI Web Search Warning:** Previously, the Gemini CLI branch silently discarded `builtin_tools` and `function_tools` when `use_web_search` was `True`, meaning the configured `web_search_tool` was silently ignored. Now a `logfire.warn()` is emitted explaining that the tool will be skipped, and the agent is created without web search.
- **Developer Experience:** Added the Ollama API key registration link (`https://ollama.com/settings/keys`) to validation error messages, making it easier for users to configure their API keys.
- **Documentation:** Improved docstring clarity and type annotations across config, data models, models, job runner, main, ollama web search, prompt, and video modules. Updated to use modern Google-style format without redundant type hints in Args/Returns sections.
- **Documentation:** Fixed README to reference the correct split setting (`max-seconds` vs `max_minutes`).
- **Type Safety:** Added proper type annotations to validator methods and fixed field descriptions for rate limiters.

## v2.8.0b2

This release focuses on refactoring and hardening the Ollama web search integration introduced in v2.8.0b1, with improvements to rate limiting, dependency management, and prompt quality.

**New Features:**

- **In-Memory Caching for Ollama Search:** Added a cache to `OllamaWebSearchDeps` that stores search results keyed by a normalized query string (punctuation removed, case-folded). This avoids redundant API calls for equivalent queries like `"Hello, world!"` and `"hello world"`, reducing latency and API costs.

**Refactoring & Improvements:**

- **AgentDeps Architecture:** Introduced a new `AgentDeps` data model to centralize dependencies passed to Pydantic AI agents. This replaces the loose `Any` type with a typed, extensible container holding `video_duration_ms` and optional `ollama_search` fields.
  - Updated `RateLimitedAgentWrapper` to use typed `AgentDeps` instead of `Any`.
  - Refactored `ollama_web_search` functions to extract dependencies from `ctx.deps`.
  - Updated `main.py` to create and configure `AgentDeps` with Ollama search dependencies.
- **Rate Limiting Migration:** Implemented hook-based rate limiting.
  - Moved rate limiting logic from the `run()` method to model request hooks via a `_rate_limit` hook function.
  - Added token calculation helper `_calculate_tokens()` for accurate rate limit tracking.
  - Improved error message formatting to show human-readable durations.
- **Deferred Response Validation:** Refactored response handling to store AI responses in a local variable, validate against video duration, and only assign to `job.response` if validation passes. This prevents storing potentially invalid responses on the job object.

**Fixes & Improvements:**

- **Ollama Search Prompt:** Updated `LYRICS_PROMPT_VERSION` to 6 with refined instructions to prevent excessive search queries. Added explicit rules against query spamming, restrictive quotation marks, and guidance to use native titles for better search results.
- **Developer Experience:** Added the Ollama API key registration link (`https://ollama.com/settings/keys`) to validation error messages, making it easier for users to configure their API keys.
- **Documentation:** Fixed README to reference the correct split setting (`max-seconds` vs `max_minutes`).

## v2.8.0b1

This release reverts the LyricsGenius integration introduced in v2.7.0 due to persistent bot detection issues, and introduces a new Ollama web search tool as an alternative search provider.

**Reverts:**

- **LyricsGenius Integration Removed:** Completely removed the LyricsGenius library and all associated code.
  - Removed `lyricsgenius_web_search.py` module and all Genius API integration.
  - Reverted `agent_wrapper.py` to use the DuckDuckGo search tool.
  - Updated configuration to support `'duckduckgo'` as the default `web_search_tool`.
  - Removed `lyricsgenius` dependency and restored `duckduckgo` in `pydantic-ai-slim`.
  - **Reason:** The LyricsGenius integration (v2.7.0) proved unreliable due to Genius and DuckDuckGo actively detecting and blocking automated requests, making lyrics retrieval consistently fail.

**New Features:**

- **Ollama Web Search Tool:** Added support for Ollama as a web search provider, expanding the available search options beyond DuckDuckGo and builtin provider tools.
  - Introduced `OllamaWebSearchDeps` and `ollama_web_search_multi` function for web search.
  - Added `OllamaSearchSettings` configuration class with API key validation.
  - Updated `web_search_tool` option to include `'ollama'` alongside `'builtin'` and `'duckduckgo'`.
  - Refactored agent initialization to use `AsyncExitStack` for proper resource cleanup.

**Fixes & Improvements:**

- **Gemini CLI Web Search Warning:** Previously, the Gemini CLI branch silently discarded `builtin_tools` and `function_tools` when `use_web_search` was `True`, meaning the configured `web_search_tool` was silently ignored. Now a `logfire.warn()` is emitted explaining that the tool will be skipped, and the agent is created without web search.

## v2.7.0

This release introduces a significant enhancement to the lyrics detection pipeline by replacing generic web search with a specialized lyrics database integration, improving accuracy and search efficiency.

**New Features:**

- **LyricsGenius Integration:** Replaced the DuckDuckGo web search backend with the LyricsGenius library, providing direct access to the Genius database for more accurate and comprehensive lyrics retrieval.
  - **Specialized Search Tool:** Implemented `lyricsgenius_web_search_tool` as a dedicated search provider optimized for song and lyrics lookups.
  - **Intelligent Query Cleaning:** Added automatic removal of common LLM suffixes (e.g., "lyrics", "歌詞", "訳") that hurt search accuracy, ensuring cleaner and more effective queries.
  - **Structured Results:** Introduced new data models (`SearchResult`, `QueryResults`, `WebSearchResponse`) for better organization and type safety of search results.

**Removals:**

- **DuckDuckGo Search Removed:** Completely removed the DuckDuckGo web search option from the codebase. The `web_search_tool` configuration no longer supports `duckduckgo` as a provider, streamlining the search pipeline to focus exclusively on the superior LyricsGenius integration.

**Prompt Engineering:**

- **Lyrics Detection (v5):** Updated the lyrics prompt to reflect the new search capabilities and provide clearer instructions on query optimization for the LyricsGenius tool.
- **Search Efficiency:** The prompt now explicitly instructs the AI to leverage the specialized search tool's capabilities without appending redundant keywords like "lyrics" to queries.

**Internal Tooling:**

- **Dependency Management:** Added `lyricsgenius` to project dependencies in `pyproject.toml`.
- **Code Quality:** Maintained consistent Google-style docstrings and ruff formatting across all new modules.

## v2.6.1

This release focuses on hardening the subtitle generation pipeline by implementing configurable timestamp validation and improving data hygiene when handling AI responses.

**New Features:**

- **Configurable AI Timestamp Validation:** Added `validation_buffer_ms` to `AiSettings` (defaulting to 1000ms). This allows users to control how strictly generated timestamps are verified against video segment durations.
- **Hallucination Protection:** The system now explicitly validates AI-generated timestamps for both subtitles and scene detection against the video duration. This prevents the AI from generating content that extends beyond the actual media length.

**Refactoring & Improvements:**

- **Automated Timestamp Sanitization:** Introduced Pydantic "before-validators" and a new `_clean_timestamp_string` utility. This allows the system to gracefully handle "field leakage" in LLM responses—such as extracting `03:52.000` from noisy strings like `03:52.000,start:`—before data is assigned to model fields.
- **Improved Data Hygiene:** Re-engineered the `Subtitles` and `Scene` data models to ensure they store only sanitized timecodes, improving internal consistency and reducing downstream parsing errors.
- **Validation Pipeline Hardening:** Updated `Job.load` and internal runners to propagate validation context. This ensures that even cached results are re-validated against current duration limits and buffer settings.
- **Strict Parsing:** Refactored the core timestamp parser to enforce strict formats, delegating noise removal to the pre-validation layer.

**Internal Tooling:**

- Transitioned the project's formatting and linting from `black` to `ruff` for a faster and more integrated developer experience.

## v2.6.1b3

This release focuses on hardening the lyrics detection stage against AI formatting errors and improving the efficiency of the web search pipeline.

**Prompt Engineering:**

- **Lyrics Detection (v4):**
  - **JSON Syntax Guard:** Introduced a critical safety layer to prevent "field leakage," where the AI might accidentally include JSON keys or structural markers inside string values.
  - **High-Efficiency Search:** Re-engineered the execution pipeline to explicitly mandate simultaneous, multi-query web searches. This significantly reduces the number of AI turns required, lowering both latency and API costs.
  - **Structured Output Format:** Refined the prompt's output block to provide a clearer template for the model, ensuring consistent JSON generation.

**Backend & Validation:**

- **Robust Timestamp Parsing:**
  - Refactored `_parse_timestamp_string_ms` to use regular expressions. This allows the system to successfully extract valid timecodes even if the AI response contains "noisy" prefixes or suffixes (e.g., `"01:23.456,start:"`) within the timestamp field.
- **Data Model Enhancements:**
  - **Original Language Tracking:** Added a dedicated `original_language` field to the `Scene` model to better capture and track the primary language of detected songs.
  - **Scene Integrity:** Implemented a Pydantic `model_validator` for the `Scene` class, ensuring that all detected scenes maintain chronological integrity (start time must be strictly before end time) before processing continues.

**Bug Fixes:**

- Fixed an issue where "field leakage" in AI responses could lead to JSON validation failures.

## v2.6.1b2

This release refines the AI prompting strategy to improve formatting reliability and better handle visual-only cues.

**Prompt Engineering:**

- **Lyrics Detection (v3):**
  - **JSON Syntax Guards:** Introduced strict rules for escaping, newlines, and key separation to ensure valid JSON output under all conditions.
  - **Simplified Pipeline:** Streamlined the metadata resolution steps for improved efficiency.
- **Subtitle Generation (v15):**
  - **Dual-Trigger System:** Clarified that subtitles are triggered by both vocal audio and prominent on-screen text.
  - **Visual Events Exception:** Explicitly instructs the AI to subtitle relevant visual text (like chapter titles) even when no audio is present.
  - **Anti-Hallucination Hardening:** Strengthened rules for ignoring incorrect or incomplete reference data and mandated manual transcription when the JSON reference fails.
  - **Standardized Analysis:** Updated examples to use a consistent, structured `global_analysis` format.
  - **New Visual Text Example:** Added a dedicated example for handling silent title cards.

**Backend & Validation:**

- **Timestamp Validation:**
  - Refactored timestamp parsing into an internal utility function `_parse_timestamp_string_ms`.
  - Implemented Pydantic `model_validator` on both `Subtitles` and `Scene` models to verify timestamp format and ensure the start time is strictly before the end time.
  - Improved robustness against malformed AI-generated strings (e.g., truncated JSON noise in timestamp fields).
- **Static Analysis Fixes:**
  - Resolved `CliPositionalArg` to `Path` type mismatch for static analysis.
  - Added explicit type casting in `config.py` and `main.py` to address errors where `CliPositionalArg` was not assignable to `Path` for methods like `.resolve()`, `.stem`, and `.name` in Pyright/mypy.

## v2.6.1b1

This release focuses on hardening the AI's transcription logic and translation accuracy by introducing stricter boundaries for reference data and expanding the contextual window for translations.

**Prompt Engineering:**

- **Subtitle Generation (v14):**
  - **The Manual Transcription Mandate:** Explicitly instructs the AI that it is "not exempt" from subtitling if the reference JSON is null or mismatching; it must rely on native audio perception to transcribe.
  - **Strict Scene Boundaries:** Added rules to prevent "cross-contamination" where lyrics from a later scene in the reference JSON are incorrectly applied to earlier audio.
  - **Enhanced Contextual Window:** Refined the translation logic to mandate analysis of the **Previous 2 Sentences** and **Next 2 Sentences** to improve pronoun resolution and tonal consistency.
  - **Refined Anti-Hallucination:** Updated the "Wrong Song" and "Partial Video" scenarios with specific timestamp-based examples to help the model ignore irrelevant reference data.
  - **Improved Decoding Hierarchy:** Streamlined the fallback logic for resolving ambiguous audio, prioritizing visual OCR and scene context.

## v2.6.0

This release promotes the v2.6.0 beta series to production, focusing on improving subtitle synchronization and timing precision by refining the AI's internal logic for handling rapid speech.

**Prompt Engineering:**

- **Subtitle Generation (v13):**
  - **Anti-Drift Mechanics:** Introduced strict "mechanical rules" to solve the "Cascading Delay" effect where subtitles fall behind the audio.
    - **The Sacred Start Time:** Forces the model to anchor start timestamps strictly to audio onsets, regardless of previous line lengths.
    - **Truncation over Extension:** Mandates aggressive truncation of end timestamps to ensure upcoming lines start on time.
    - **Instantaneous Transitions:** Enforces zero-gap transitions for rapid-fire speech.
  - **New Rapid Speech Example:** Added a dedicated prompt example demonstrating how to handle high-speed vocals without synchronization loss.
  - **Verification Logic:** Updated the required `global_analysis` to force the AI to acknowledge audio pacing and its truncation strategy before generating subtitles.

## v2.6.0b1

This release focuses on improving subtitle synchronization and timing precision by refining the AI's internal logic for handling rapid speech.

**Prompt Engineering:**

- **Subtitle Generation (v13):**
  - **Anti-Drift Mechanics:** Introduced strict "mechanical rules" to solve the "Cascading Delay" effect where subtitles fall behind the audio.
    - **The Sacred Start Time:** Forces the model to anchor start timestamps strictly to audio onsets, regardless of previous line lengths.
    - **Truncation over Extension:** Mandates aggressive truncation of end timestamps to ensure upcoming lines start on time.
    - **Instantaneous Transitions:** Enforces zero-gap transitions for rapid-fire speech.
  - **New Rapid Speech Example:** Added a dedicated prompt example demonstrating how to handle high-speed vocals without synchronization loss.
  - **Verification Logic:** Updated the required `global_analysis` to force the AI to acknowledge audio pacing and its truncation strategy before generating subtitles.

## v2.5.0

This release promotes the v2.5.0 beta series to production, finalizing the migration to a fully asynchronous architecture. It includes significant stability improvements to the job runner, improved resource management for concurrent tasks, and critical fixes for blocking I/O operations.

**Fixes & Improvements:**

- **Async Stability & Performance:**
  - **Job Runner Safety:** Refactored the `JobRunner` loop to ensure `queue.task_done()` is always called, preventing deadlocks if an exception occurs during job setup.
  - **Blocking I/O:**
    - Offloaded synchronous video file reading in `RateLimitedAgentWrapper` to a thread to prevent blocking the event loop.
    - Moved the CPU-intensive `get_ssafile()` call in `main.py` to a worker thread during post-processing.
  - **Windows Process Management:** Fixed an issue where the `taskkill` command used for timeouts in `GeminiCliModel` was not properly awaited.

- **Refactoring:**
  - **JobRunner Encapsulation:** The `JobRunner` class now manages its own `asyncio.Queue`, simplifying the main orchestration logic.
  - **Async Start:** `JobRunner.start()` is now an `async` method, ensuring it executes within the correct event loop context.
  - **Optimization:** `reencode_video` now uses `asyncio.gather` for parallel duration checks of input and output files.

- **Resource Management:**
  - **Concurrency Limits:** Implemented a semaphore (limit: 8) for concurrent `ffprobe` duration checks to prevent "Too many open files" errors.

- **Bug Fixes:**
  - **Gemini File Uploader:** Fixed a runtime error by using `async for` when iterating over file lists.
  - **Data Integrity:** Enforced strict zipping of video split paths and durations in `main.py` to fail immediately if list lengths mismatch.

## v2.5.0b1

This release marks a major architectural shift, migrating the core application from a threaded model to a fully asynchronous model using `asyncio`. This improves concurrency handling, reduces blocking operations, and enhances overall stability.

**Refactoring (Asyncio Migration):**

- **Core Architecture:**
  - Transitioned `JobRunner` and the main orchestration loop from `concurrent.futures` and threading to `asyncio.Queue` and `asyncio.Task`.
  - The application entry point now utilizes `asyncio.run()`.
- **Non-Blocking I/O:**
  - **Video Processing:** Converted FFmpeg calls (`split`, `re-encode`, `duration`) to use `asyncio.create_subprocess_exec`.
  - **Gemini Integration:** `GeminiFileUploader` now uses the asynchronous `google.genai.Client`. CPU-intensive tasks like SHA256 hashing are offloaded to threads.
  - **CLI Execution:** `GeminiCliModel` now uses async subprocesses for model execution and offloads prompt file writing to threads.
  - **File Operations:** Heavy synchronous I/O operations (loading job states, reading video bytes, stitching subtitles) are now offloaded using `asyncio.to_thread` to keep the event loop responsive.
- **Agent Execution:**
  - Refactored `RateLimitedAgentWrapper` to use native async methods, removing the need for `nest_asyncio` and thread-local storage.

**Improvements:**

- **Parallelization:** Video duration checks in the splitting stage are now executed in parallel using `asyncio.gather`.
- **Windows Stability:** Fixed a blocking I/O issue where the taskkill command (used for timeouts) relied on synchronous subprocess calls, which could freeze the event loop.

**Internal Changes:**

- Removed the `nest_asyncio` dependency.
- Fixed type safety for `HttpOptions` in the GenAI client initialization.

## v2.4.1

This release fixes a configuration issue where nested settings were ignoring the `.env` file.

**Fixes:**

- **Configuration (.env loading):**
  - Fixed an issue where `AISUB_AI_GOOGLE_KEY` and other nested variables in `.env` were not being loaded because nested Pydantic models do not inherit configuration from the parent Settings class.
  - Defined `_BASE_CONFIG` with `env_file=".env"` and applied it to all nested `BaseSettings` models (`GoogleAiSettings`, `GeminiCliSettings`, `AiSettings`, etc.) to ensure consistent behavior.

## v2.4.0

This release promotes the v2.4.0 beta series to production, incorporating all stability fixes, prompt engineering enhancements, and configuration improvements introduced in the beta cycle.

**Fixes & Improvements:**

- **Documentation:**
  - Updated the README to clarify the current free request limits for the Gemini CLI.

**Known Issues:**

- **Asyncio Event Loop:**
  - Users may see errors like `RuntimeError: <asyncio.locks.Event object ...> is bound to a different event loop`.
  - This is currently under investigation. The application will automatically retry the affected job, so these errors can be ignored.

## v2.4.0b5

This release enhances configuration robustness, improves prompt engineering for subtitles, and updates documentation for clarity.

**New Features:**

- **Configuration (API Key Sanitization):**
  - The application now automatically strips leading/trailing whitespace and surrounding quotes from the Google AI API key during configuration loading. This prevents common authentication failures caused by copy-paste errors or shell quoting.

**Fixes & Improvements:**

- **Configuration (API Key Validation):**
  - Fixed an issue where a Google API key was required even when lyrics processing was disabled (`--thread.lyrics=0`). The API key is now only validated if the lyrics stage is active, allowing for API-key-free workflows when using `gemini-cli` for subtitles.
- **Gemini CLI (Cost Calculation):**
  - Resolved a `CostCalculationFailedWarning` by ensuring the `model_name` is correctly passed to `pydantic-ai`'s instrumentation layer, enabling accurate cost tracking for CLI-based executions.
- **Upload (Logging):**
  - Added debug logging for the full file object details returned by the Gemini Files API after a successful upload, aiding in troubleshooting.

**Prompt Engineering:**

- **Subtitle Generation (v12):**
  - Incremented `SUBTITLES_PROMPT_VERSION` to 12.
  - The prompt has been significantly enhanced with stricter guidelines to improve accuracy:
    - **The "Golden Rule":** Reinforces that audio dictates timing ("When") while visuals and context dictate content ("What").
    - **Decoding Hierarchy:** Establishes a clear fallback order (On-screen Text > Scene Context > Reference JSON) for resolving ambiguous audio.
    - **Anti-Hallucination Rules:** Added robust logic to handle scenarios where the reference JSON contains lyrics for the wrong song or extra verses not present in the video segment.
    - **Timestamp Alignment:** Introduced stricter rules to prevent "cascading delay" errors by ensuring end timestamps tightly wrap the spoken audio, disabling readability biases that could cause sync issues.
    - **Pause Handling:** Mandates splitting subtitles across audible pauses to maintain precise synchronization.

**Documentation:**

- **README Update:** The main `README.md` has been updated with a clearer overview and revised usage instructions, including the latest recommendations for free-tier models.
- **Configuration File:** The detailed `CONFIGURATION.md` has been moved from the root directory to the `docs/` folder for better project organization.

## v2.4.0b4

This release fixes critical bugs related to concurrent processing and file path handling, significantly improving stability and compatibility, especially when using the Gemini CLI backend.

**Fixes & Improvements:**

- **Async Stability (Per-Thread Agent Caching):**
  - Fixed a `RuntimeError` related to `asyncio` event loops that occurred during concurrent agent execution.
  - The issue was caused by creating new AI Agent instances on every call within a worker thread, leading to event loop conflicts.
  - The fix now caches a single `Agent` instance per thread using `threading.local()`. This ensures stability by aligning the agent's lifecycle with the thread's event loop and improves performance by reusing the agent and its connection pool.
- **File Path Resolution (Gemini CLI):**
  - Resolved a `ValueError: relative path can't be expressed as a file URI` that occurred when using the `gemini-cli` backend.
  - All relevant file paths (`input_video_file`, `dir.out`, `dir.tmp`) are now resolved to absolute paths upon configuration load, ensuring compatibility with tools that require absolute file URIs.

## v2.4.0b3

This release improves agent execution stability and compatibility by refining asynchronous operations.

**Fixes & Improvements:**

- **Async Stability:**
  - **Agent Execution:** Replaced manual `asyncio.run()` with `pydantic-ai`'s built-in `run_sync()` method. This avoids potential `RuntimeError` issues when the agent is invoked from a thread with a running asyncio event loop and simplifies the code.
  - **Event Loop Nesting:** Implemented `nest_asyncio` to patch the asyncio event loop. This allows for nested execution, ensuring compatibility with environments like Jupyter notebooks or web servers where an event loop is already active.

## v2.4.0b2

This release significantly hardens the application architecture, improving state management security, data integrity during processing, and Gemini CLI integration. It also reorganizes the documentation for better accessibility.

**Documentation:**

- **Configuration:** Moved all configuration flags to a dedicated [`CONFIGURATION.md`](CONFIGURATION.md) file to declutter the main README.
- **New Settings:** Added documentation for the `duration-tolerance-ms` setting.

**New Features & Improvements:**

- **Gemini CLI Integration:**
  - Refactored the Gemini CLI wrapper into a formal `pydantic_ai.models.Model`.
  - Added automatic instrumentation for `logfire`, allowing capture of token usage and latency metrics from CLI-based executions.
- **Data Integrity:**
  - **Video Splitting:** `split_video` now calculates the total duration of existing segments to verify integrity before skipping the split operation.
  - **Re-encoding:** `reencode_video` now compares the duration of existing output files against the input (with a configurable `duration_tolerance_ms`) to ensure validity.
- **Optimization:** Job runners now check for existing valid responses in the job object before processing, preventing unnecessary API calls.

**Fixes:**

- **Asyncio Stability:** Fixed a `RuntimeError` by ensuring the AI agent and its underlying HTTP client are initialized within the active `asyncio` event loop.
- **Security & Privacy:**
  - **State Sanitization:** The `file` attribute is no longer saved to the JSON state files, preventing the persistence of absolute file paths (which may contain sensitive usernames).
  - **Resumption Logic:** Updated the pipeline to correctly handle job resumption when file handles are missing from the state.
- **Type Safety:** Added runtime assertions in job runners to satisfy type checkers regarding optional file paths.

**Refactoring:**

- **Data Models:**
  - Renamed `JobState` to `SegmentJobs` to better reflect its purpose.
  - Renamed `SceneResponse` to `LyricsSceneAiResponse` and merged `SubtitleResponse` into `SubtitleAiResponse`.
  - Simplified `SegmentJobs` to use optional fields instead of dictionaries.
- **Pipeline:**
  - Centralized pipeline stage completion callbacks into a single `on_stage_complete` function.
  - Simplified the main orchestration logic in `main.py` to use a linear dependency chain.
- **Code Cleanup:** Removed unused data models (`AiResponse`, `SubtitleGenerationState`) and redundant fields.

## v2.4.0b1

This release introduces significant improvements to prompt engineering for better accuracy, adds support for DuckDuckGo search, and allows for disabling the lyrics detection pass. It also includes architectural refactoring to improve stability and support mixed-model workflows.

**BREAKING CHANGES:**

- **Web Search Configuration:**
  - The option `google` has been renamed to `builtin` to be provider-agnostic.
  - The default search tool is now `duckduckgo` instead of `builtin` (Google).

**New Features:**

- **DuckDuckGo Search:** Added support for DuckDuckGo as a web search provider. This is now the default setting to avoid costs associated with Google's built-in search tool when using the API.
- **Optional Lyrics Detection:** The lyrics and scene detection pass can now be disabled by setting `--thread.lyrics 0`. This allows for a transcription-only workflow, saving time and API costs.

**Prompt Engineering:**

- **Lyrics & Scene Detection (v2):**
  - **Granular Metadata:** The model now explicitly separates the "Original Artist" from the "Video Performer" and includes a step-by-step search log.
  - **Improved Search:** Search queries have been optimized with specific templates to improve hit rates for lyrics.
- **Subtitle Generation (v11):**
  - **The "Golden Rule":** Introduced strict logic where audio strictly dictates _timing_ ("When"), while visuals and context dictate _content_ ("What").
  - **Anti-Hallucination:** Added strict rules to handle cases where the reference lyrics are for the wrong song or contain extra verses not present in the video segment.
  - **Ambiguity Resolution:** Clarified the hierarchy for resolving ambiguous audio (On-screen text > Scene Context > JSON).

**Fixes & Improvements:**

- **Resumption Stability:**
  - Fixed an issue where resuming a job with expired Gemini cloud files (older than 48 hours) would fail. The system now forces a re-check and re-upload if necessary.
  - Fixed a `RuntimeError` related to asyncio event loops by lazily initializing the AI Agent.
- **Mixed Provider Support:**
  - Fixed file path handling when using a Google model for lyrics detection and a non-Google (e.g., local or different API) model for subtitle generation.
  - Ensure `WebSearchTool` is correctly enabled for API-based lyrics detection.
- **Internal Refactoring:**
  - Decoupled `JobRunner` from the `Job` structure for better pipeline flexibility.
  - Centralized job logging and error handling.

## v2.3.1

This release improves exception handling in logging and adds a fail-safe to the file uploader.

**Fixes & Improvements:**

- **Logging:**
  - Replaced `logfire.error` with `logfire.exception` in `gemini_cli_wrapper.py` and `video.py` to ensure stack traces are properly captured within the active span before an exception propagates.
  - Removed redundant exception details from log messages, allowing `logfire.exception` to handle context automatically.
- **File Uploader:**
  - Added a fail-fast check for `FileState.FAILED` in `gemini_file_uploader.py` to prevent an infinite loop while waiting for a file to be processed by the server.

## v2.3.0

This release removes the two-pass subtitle generation system in favor of a simpler, single-pass workflow. The second "QA" pass proved to be of limited value and added unnecessary complexity and cost.

**BREAKING CHANGES:**

- **Configuration:**
  - The `ai.pass1_model` and `ai.pass2_model` settings have been removed and replaced with a single `ai.model_subtitles`.
  - The `ai.lyrics_model` setting has been renamed to `ai.model_lyrics`.
  - The `thread.subtitles1` and `thread.subtitles2` settings have been consolidated into `thread.subtitles`.
  - The `ai.model` shorthand now sets `model_subtitles` and `model_lyrics`.
- **State Persistence:** The format for intermediate job state files has changed.
  - `part_XXX.pass1.MODEL.json` is now `part_XXX.subtitles.MODEL.json`.
  - `part_XXX.pass2.MODEL.json` is no longer created.
  - **You must delete temporary directories (e.g., `tmp_<video_name>`) from previous runs before using this version.**

**Improvements:**

- **Simplified Pipeline:** The subtitle generation process is now a single AI call, reducing complexity, processing time, and potential points of failure.

- **Prompt Versioning & Auto-Reprocessing:**
  - Introduced versioning for both the "Lyrics/Scene Detection" prompt (`LYRICS_PROMPT_VERSION`) and the "Subtitle Generation" prompt (`SUBTITLES_PROMPT_VERSION`).
  - The application now automatically detects when a prompt has been updated between runs. It will invalidate cached results for affected segments and re-process them using the new prompt, ensuring subtitles always reflect the latest logic.
  - Cache loading is now more robust. Corrupted or outdated job files from previous runs are automatically ignored and re-processed, preventing crashes.
- **Prompt Engineering:**
  - Updated `SUBTITLES_PROMPT_VERSION` to 10.
  - The subtitle generation prompt has been significantly improved with a "Decoding Hierarchy" to better handle ambiguous audio by prioritizing on-screen text and reference JSON.
  - Added stricter rules to prevent "cascading delay" timing errors and to enforce intelligent segmentation.

## v2.2.0b1

This release introduces a major refactoring of the job processing pipeline to be more flexible and robust, improves state persistence, and refines the QA prompt to better handle timing errors.

**BREAKING CHANGE:**

- The persistence format for intermediate job states has changed from multiple files per segment to a single `part_XXX.json` file. In-progress jobs from previous versions are not compatible. **You must delete temporary directories (e.g., `tmp_<video_name>`) from previous runs before using this version.**

**New Features & Improvements:**

- **Pipeline Rearchitecture & Multi-Model Reruns:**
  - The pipeline now uses a central `JobState` object that stores results for each stage (`lyrics`, `pass1`, `pass2`) in a dictionary keyed by the model name.
  - This enables re-running a specific stage with a different model (e.g., using a more powerful model for Pass 2) without reprocessing the entire pipeline.
- **Robust State Persistence & Resumption:**
  - **Unified Persistence:** The state for each video segment is now saved to a single `part_XXX.json` file, making state management and job resumption more reliable.
  - **Intelligent Resumption:** The processing loop now intelligently inspects the state file to determine which stage to execute next based on the models configured for the current run.
  - **File Existence Validation:** The resumption logic now verifies that the video files for completed stages still exist, preventing errors if intermediate files are deleted.
- **Prompt Engineering:**
  - Updated `SUBTITLES_PROMPT_VERSION` to 9.
  - The Pass 2 (QA) prompt has been significantly updated to prioritize fixing timestamp desynchronization ("Timestamp Drift"), instructing the AI to re-anchor subtitles to the correct audio position.

**Bug Fixes:**

- **Video Duration Calculation:** Fixed an issue where a failure to get a video's duration would result in a silent error and a value of `0`. This led to corrupted timestamp offsets in the final subtitle file. The function now raises a `RuntimeError` to fail fast.

**Code Improvements:**

- **Data Model Refactoring:** The `SubtitlePass1Job` and `SubtitlePass2Job` models have been simplified. The job runners now retrieve prerequisite data (like scene information and Pass 1 drafts) directly from the central `JobState` object, creating a cleaner and more efficient data flow.

---

## v2.1.0b3

This release improves the robustness of subtitle generation when the lyrics reference is inaccurate or missing.

**Improvements:**

- **Prompt Engineering:**
  - Updated `SUBTITLES_PROMPT_VERSION` to 7.
  - Refined Pass 1 and Pass 2 prompts to handle cases where the lyrics reference is missing, incomplete, or incorrect. The AI is now explicitly instructed to prioritize the audio source of truth over the reference JSON if discrepancies exist, preventing it from forcing audio to fit an incorrect lyric sheet.

---

## v2.1.0b2

This release refines the scene detection pass to reduce false positives for lyrics research by ignoring instrumental background music.

**Improvements:**

- **Scene Detection:**
  - Renamed the `contains_song` field to `contains_vocal_music` to explicitly target songs with vocals.
  - Updated the scene detection prompt to instruct the AI to ignore instrumental-only tracks and background music (BGM). This prevents unnecessary web searches for lyrics when no vocals are present.
- **Prompt Versioning:**
  - Incremented `SUBTITLES_PROMPT_VERSION` to 6.

---

## v2.1.0b1

This release introduces a new "Scene Detection & Lyrics Research" pass to the pipeline, further enhancing contextual accuracy, especially for videos containing music.

**New Features & Improvements:**

- **Scene Detection & Lyrics Research Pass:** A new initial pass has been added to the processing pipeline.
  - It analyzes the video to identify distinct scenes (e.g., dialogue vs. music).
  - For scenes containing music, it uses the AI's web search capabilities to find official original and translated lyrics.
- **Context-Aware Subtitle Generation:** The data from the new scene/lyrics pass is fed into both Pass 1 (Drafting) and Pass 2 (Refinement). This provides the AI with crucial context, improving the accuracy of lyric transcription and translation.
- **New Configuration:**
  - `ai.lyrics_model`: A new setting to specify the model for the scene detection pass.
  - `thread.lyrics`: A new setting to control concurrency for this pass.
- **Updated Pipeline Flow:** The processing pipeline is now: Lyrics and Scene Detection -> Subtitle Pass 1 -> Subtitle Pass 2.

**Code Improvements:**

- **Pipeline Orchestration (`main.py`):** The main pipeline logic has been updated to incorporate the new scene detection job runner and manage the new data flow.
- **Data Models (`data_models.py`):** Introduced `LyricsSceneJob`, `Scene`, and `SceneResponse` models to handle the data from the new pass. `SubtitlePass1Job` and `SubtitlePass2Job` now include an optional `scene_response`.
- **Response Models (`data_models.py`):** Split the generic `AiResponse` into specific `SubtitlePass1Response` and `SubtitlePass2Response` models. This ensures that the output from each pass strictly adheres to its expected schema (e.g., `global_analysis` for Pass 1, `qa_analysis` for Pass 2).
- **Prompt Engineering (`prompt.py`):** Prompts for Pass 1 and Pass 2 have been updated to consume and utilize the new scene and lyrics reference data.

---

## v2.0.0

This major release introduces a two-pass subtitle generation workflow to significantly improve subtitle quality, accuracy, and refinement.

**New Features & Improvements:**

- **Two-Pass Subtitle Generation:** The core subtitle generation process has been re-architected into a two-pass system:
  - **Pass 1 (Draft Generation):** The first pass generates an initial draft of the subtitles. This is handled by the new `SubtitlePass1JobRunner` and configured with `ai.pass1_model` and `thread.subtitles1`.
  - **Pass 2 (QA & Refinement):** The second pass takes the JSON output from Pass 1 as a draft and performs quality assurance, correction, and refinement using a separate, specialized prompt. This is handled by the new `SubtitlePass2JobRunner` and configured with `ai.pass2_model` and `thread.subtitles2`.
- **Improved Modularity:** This new architecture allows for using different AI models for each pass (e.g., a faster model for the initial draft and a more powerful model for refinement).

**Breaking Changes & Refactoring:**

- **Configuration (`config.py`):**
  - The `ai.model` setting has been replaced by `ai.pass1_model` and `ai.pass2_model`. An optional `ai.model` shorthand is available to set both to the same value.
  - The `thread.subtitles` setting has been split into `thread.subtitles1` and `thread.subtitles2` to configure concurrency for each pass independently.
- **Main Pipeline (`main.py`):**
  - The main application logic has been significantly updated to orchestrate the new two-pass workflow, managing separate job queues and runners for each pass.
  - Job resumption logic now checks for the state of both passes to correctly resume interrupted sessions.
- **Data Models (`data_models.py`):**
  - `SubtitleJob` has been renamed to `SubtitlePass1Job`.
  - A new `SubtitlePass2Job` has been introduced to manage the state and data for the refinement pass.
- **Agent Wrapper (`agent_wrapper.py`):**
  - The `RateLimitedAgentWrapper` is now initialized with a specific model name, enabling the use of different models for each generation pass.

---

## v1.12.0

This release adds functionality to skip initial video segments and improves prompt handling for the Gemini CLI.

**New Features & Improvements:**

- **Start Offset:** Added `start_offset_min` to `SplittingSettings`. This allows users to skip the first _X_ minutes of the input video (e.g., to bypass waiting screens) by setting `--split.start_offset_min`.
- **Gemini CLI Configuration:** Added `overwrite_system_prompt` to `GeminiCliSettings`. This setting determines whether to overwrite the system prompt using `GEMINI_SYSTEM_MD` or pass the prompt as a regular file argument.

**Code Improvements:**

- **GeminiCliWrapper Refactor:** Refactored `GeminiCliWrapper` to accept the full `GeminiCliSettings` object and removed unused imports and testing code.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.11.0...v1.12.0

---

## v1.11.0

This release updates the subtitle generation prompt to improve timing accuracy and synchronization.

**New Features & Improvements:**

- **Prompt Update:** Updated `SUBTITLES_PROMPT` to version 3. The guidelines now strictly prioritize native audio alignment, explicitly disabling readability biases. This fixes an issue where the AI model would extend timestamps for readability, resulting in cascading delays in subtitle timings.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v1.10.1...v1.11.0

---

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
