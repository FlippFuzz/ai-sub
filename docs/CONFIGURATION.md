# Configuration Reference

All settings can be configured via command-line arguments (e.g., `--ai.rpm 10`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_RPM=10`).

## Positional Arguments

| Argument             | Description                                        | Default    |
| :------------------- | :------------------------------------------------- | :--------- |
| `<input_video_file>` | Path to the video file to transcribe and subtitle. | _Required_ |

---

## User Context Options

| Argument                | Environment Variable | Description                                                                          | Default |
| :---------------------- | :------------------- | :----------------------------------------------------------------------------------- | :------ |
| `--context <text>`      | `AISUB_CONTEXT`      | Inline text providing background context, speaker names, or translation preferences. | _None_  |
| `--context-file <path>` | `AISUB_CONTEXT_FILE` | Path to a text file containing user background context.                              | _None_  |

---

## AI Settings (`--ai.*`)

| Argument                              | Environment Variable                                 | Description                                                                                | Default                            |
| :------------------------------------ | :--------------------------------------------------- | :----------------------------------------------------------------------------------------- | :--------------------------------- |
| `--ai.key <key>`                      | `AISUB_AI_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini API key.                                                                     | _None_                             |
| `--ai.base-url <url>`                 | `AISUB_AI_BASE_URL`                                  | Optional custom base URL or proxy for Google Gemini API calls.                             | _None_                             |
| `--ai.model <model>`                  | `AISUB_AI_MODEL`                                     | Shorthand setting model_subtitles, model_lyrics, and model_summary to the same identifier. | _None_                             |
| `--ai.model-summary <model>`          | `AISUB_AI_MODEL_SUMMARY`                             | AI model for whole-video summary, entity extraction, and chapter indexing.                 | `google-gla:gemini-3.5-flash-lite` |
| `--ai.model-subtitles <model>`        | `AISUB_AI_MODEL_SUBTITLES`                           | AI model for audio-visual subtitle transcription and alignment.                            | `google-gla:gemini-3.8-flash`      |
| `--ai.model-lyrics <model>`           | `AISUB_AI_MODEL_LYRICS`                              | AI model for musical lyrics search and refinement with agentic video inspection.           | `google-gla:gemini-3.5-flash-lite` |
| `--ai.service-tier <tier>`            | `AISUB_AI_SERVICE_TIER`                              | Shorthand setting service tiers across all pipeline phases.                                | _None_                             |
| `--ai.service-tier-summary <tier>`    | `AISUB_AI_SERVICE_TIER_SUMMARY`                      | Google Gemini service tier for whole-video summary.                                        | _None_                             |
| `--ai.service-tier-subtitles <tier>`  | `AISUB_AI_SERVICE_TIER_SUBTITLES`                    | Google Gemini service tier for subtitle generation.                                        | _None_                             |
| `--ai.service-tier-lyrics <tier>`     | `AISUB_AI_SERVICE_TIER_LYRICS`                       | Google Gemini service tier for lyrics refinement.                                          | _None_                             |
| `--ai.agentic-video <bool>`           | `AISUB_AI_AGENTIC_VIDEO`                             | Shorthand setting agentic video understanding across all pipeline phases.                  | _None_                             |
| `--ai.agentic-video-summary <bool>`   | `AISUB_AI_AGENTIC_VIDEO_SUMMARY`                     | Whether to use agentic video understanding for whole-video summary.                        | `True`                             |
| `--ai.agentic-video-subtitles <bool>` | `AISUB_AI_AGENTIC_VIDEO_SUBTITLES`                   | Whether to use agentic video understanding for subtitle generation.                        | `False`                            |
| `--ai.agentic-video-lyrics <bool>`    | `AISUB_AI_AGENTIC_VIDEO_LYRICS`                      | Whether to use agentic video understanding for lyrics refinement.                          | `True`                             |
| `--ai.rpm <int>`                      | `AISUB_AI_RPM`                                       | Maximum Requests Per Minute (RPM) allowed for the model.                                   | `4`                                |
| `--ai.tpm <int>`                      | `AISUB_AI_TPM`                                       | Maximum Tokens Per Minute (TPM) allowed for the model.                                     | `250000`                           |
| `--ai.validation-buffer-ms <int>`     | `AISUB_AI_VALIDATION_BUFFER_MS`                      | Allowed tolerance buffer in milliseconds for timestamps exceeding segment duration.        | `2000`                             |

### Files API Storage Settings (`--ai.storage.*`)

| Argument                                        | Environment Variable                        | Description                                                                         | Default |
| :---------------------------------------------- | :------------------------------------------ | :---------------------------------------------------------------------------------- | :------ |
| `--ai.storage.max-gb <float>`                   | `AISUB_AI_STORAGE_MAX_GB`                   | High-water mark in Gigabytes (GB) triggering managed file cleanup.                  | `15.0`  |
| `--ai.storage.target-gb <float>`                | `AISUB_AI_STORAGE_TARGET_GB`                | Low-water mark in Gigabytes (GB) to delete down to during cleanup.                  | `10.0`  |
| `--ai.storage.cleanup-interval-minutes <int>`   | `AISUB_AI_STORAGE_CLEANUP_INTERVAL_MINUTES` | Minimum cooldown period in minutes between storage cleanup checks.                  | `30`    |
| `--ai.storage.polling-interval-seconds <float>` | `AISUB_AI_STORAGE_POLLING_INTERVAL_SECONDS` | Interval in seconds to poll the Gemini Files API while processing an uploaded file. | `10.0`  |

### Web Search Settings (`--ai.search.*`)

| Argument                               | Environment Variable                                    | Description                                                                                                                                                                         | Default          |
| :------------------------------------- | :------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------- |
| `--ai.search.tools <list>`             | `AISUB_AI_SEARCH_TOOLS`                                 | List of search providers to enable (e.g. `['duckduckgo']`, `['tinyfish', 'lrclib']`). Supported: `'builtin'`, `'duckduckgo'`, `'ollama'`, `'langsearch'`, `'tinyfish'`, `'lrclib'`. | `["duckduckgo"]` |
| `--ai.search.key <key>`                | `AISUB_AI_SEARCH_KEY`                                   | Fallback API key for search services (Ollama, Langsearch, or Tinyfish).                                                                                                             | _None_           |
| `--ai.search.tinyfish-key <key>`       | `AISUB_AI_SEARCH_TINYFISH_KEY` / `TINYFISH_API_KEY`     | Dedicated API key for Tinyfish search.                                                                                                                                              | _None_           |
| `--ai.search.langsearch-key <key>`     | `AISUB_AI_SEARCH_LANGSEARCH_KEY` / `LANGSEARCH_API_KEY` | Dedicated API key for Langsearch.                                                                                                                                                   | _None_           |
| `--ai.search.ollama-key <key>`         | `AISUB_AI_SEARCH_OLLAMA_KEY` / `OLLAMA_API_KEY`         | Dedicated API key for Ollama search.                                                                                                                                                | _None_           |
| `--ai.search.max-turns <int>`          | `AISUB_AI_SEARCH_MAX_TURNS`                             | Maximum function-calling search turns permitted per segment.                                                                                                                        | `3`              |
| `--ai.search.qps <float>`              | `AISUB_AI_SEARCH_QPS`                                   | Maximum queries per second for web search requests.                                                                                                                                 | `0.3`            |
| `--ai.search.max-length <int>`         | `AISUB_AI_SEARCH_MAX_LENGTH`                            | Maximum characters permitted in a single search result snippet or fetched page.                                                                                                     | `4096`           |
| `--ai.search.timeout <float>`          | `AISUB_AI_SEARCH_TIMEOUT`                               | Timeout in seconds for external search HTTP requests.                                                                                                                               | `60.0`           |
| `--ai.search.retries <int>`            | `AISUB_AI_SEARCH_RETRIES`                               | Maximum retries for search requests on transient errors.                                                                                                                            | `3`              |
| `--ai.search.min-wait-seconds <float>` | `AISUB_AI_SEARCH_MIN_WAIT_SECONDS`                      | Minimum wait time in seconds for search exponential backoff.                                                                                                                        | `5.0`            |
| `--ai.search.max-wait-seconds <float>` | `AISUB_AI_SEARCH_MAX_WAIT_SECONDS`                      | Maximum wait time in seconds for search exponential backoff.                                                                                                                        | `60.0`           |
| `--ai.search.multiplier <float>`       | `AISUB_AI_SEARCH_MULTIPLIER`                            | Exponential backoff multiplier for search retries.                                                                                                                                  | `2.0`            |

---

## Re-Encoding Settings (`--reencode.*`)

| Argument                   | Environment Variable     | Description                                                                               | Default |
| :------------------------- | :----------------------- | :---------------------------------------------------------------------------------------- | :------ |
| `--reencode.encoder <str>` | `AISUB_REENCODE_ENCODER` | Specific H.264 encoder to use (e.g., `'h264_nvenc'`, `'libx264'`). Auto-detected if null. | _None_  |

### Summary Proxy Settings (`--reencode.summary.*`)

| Argument                                         | Environment Variable                           | Description                                                                                       | Default |
| :----------------------------------------------- | :--------------------------------------------- | :------------------------------------------------------------------------------------------------ | :------ |
| `--reencode.summary.enabled <bool>`              | `AISUB_REENCODE_SUMMARY_ENABLED`               | Whether whole-video proxy re-encoding is enabled.                                                 | `True`  |
| `--reencode.summary.threshold-mb <int>`          | `AISUB_REENCODE_SUMMARY_THRESHOLD_MB`          | File size threshold in MB above which the full video is compressed. 0 compresses unconditionally. | `300`   |
| `--reencode.summary.keyframe-only <bool>`        | `AISUB_REENCODE_SUMMARY_KEYFRAME_ONLY`         | Whether to decode only keyframes from the input video for faster summary proxy re-encoding.       | `True`  |
| `--reencode.summary.fps <float>`                 | `AISUB_REENCODE_SUMMARY_FPS`                   | Target framerate (FPS) for the summary proxy video.                                               | `1.0`   |
| `--reencode.summary.height <int>`                | `AISUB_REENCODE_SUMMARY_HEIGHT`                | Target resolution height in pixels (aspect ratio is preserved).                                   | `360`   |
| `--reencode.summary.bitrate-kb <int>`            | `AISUB_REENCODE_SUMMARY_BITRATE_KB`            | Target video bitrate in Kilobytes per second (KB/s).                                              | `35`    |
| `--reencode.summary.duration-tolerance-ms <int>` | `AISUB_REENCODE_SUMMARY_DURATION_TOLERANCE_MS` | Allowed duration divergence in milliseconds for summary proxy output.                             | `100`   |

### Segment Chunk Settings (`--reencode.segment.*`)

| Argument                                         | Environment Variable                           | Description                                                                                | Default |
| :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------------------------------------------- | :------ |
| `--reencode.segment.enabled <bool>`              | `AISUB_REENCODE_SEGMENT_ENABLED`               | Whether segment re-encoding is enabled.                                                    | `False` |
| `--reencode.segment.threshold-mb <int>`          | `AISUB_REENCODE_SEGMENT_THRESHOLD_MB`          | File size threshold in MB above which chunks are compressed. 0 compresses unconditionally. | `20`    |
| `--reencode.segment.keyframe-only <bool>`        | `AISUB_REENCODE_SEGMENT_KEYFRAME_ONLY`         | Whether to decode only keyframes from the input video for segment chunk re-encoding.       | `False` |
| `--reencode.segment.fps <float>`                 | `AISUB_REENCODE_SEGMENT_FPS`                   | Target framerate (FPS) for compressed segment chunks.                                      | `1.0`   |
| `--reencode.segment.height <int>`                | `AISUB_REENCODE_SEGMENT_HEIGHT`                | Target resolution height in pixels (aspect ratio is preserved).                            | `360`   |
| `--reencode.segment.bitrate-kb <int>`            | `AISUB_REENCODE_SEGMENT_BITRATE_KB`            | Target video bitrate in Kilobytes per second (KB/s).                                       | `35`    |
| `--reencode.segment.duration-tolerance-ms <int>` | `AISUB_REENCODE_SEGMENT_DURATION_TOLERANCE_MS` | Allowed duration divergence in milliseconds for re-encoded chunk output.                   | `100`   |

---

## Splitting Settings (`--split.*`)

| Argument                              | Environment Variable                | Description                                                                            | Default |
| :------------------------------------ | :---------------------------------- | :------------------------------------------------------------------------------------- | :------ |
| `--split.max-seconds <int>`           | `AISUB_SPLIT_MAX_SECONDS`           | Maximum duration in seconds for each video chunk.                                      | `300`   |
| `--split.start-offset-min <int>`      | `AISUB_SPLIT_START_OFFSET_MIN`      | Duration in minutes to skip from the beginning of the video.                           | `0`     |
| `--split.duration-tolerance-ms <int>` | `AISUB_SPLIT_DURATION_TOLERANCE_MS` | Maximum allowed duration variance in milliseconds per segment during split validation. | `100`   |

---

## Concurrency Settings (`--concurrency.*`)

| Argument                        | Environment Variable          | Description                                                                   | Default |
| :------------------------------ | :---------------------------- | :---------------------------------------------------------------------------- | :------ |
| `--concurrency.re-encode <int>` | `AISUB_CONCURRENCY_RE_ENCODE` | Number of concurrent FFmpeg compression processes.                            | `2`     |
| `--concurrency.summary <int>`   | `AISUB_CONCURRENCY_SUMMARY`   | Number of concurrent summary tasks. Set to 0 to skip whole-video summary.     | `1`     |
| `--concurrency.subtitles <int>` | `AISUB_CONCURRENCY_SUBTITLES` | Number of concurrent subtitle generation tasks.                               | `4`     |
| `--concurrency.lyrics <int>`    | `AISUB_CONCURRENCY_LYRICS`    | Number of concurrent lyrics refinement tasks. Set to 0 to skip lyrics lookup. | `4`     |

---

## Directory Settings (`--dir.*`)

| Argument           | Environment Variable | Description                                                    | Default               |
| :----------------- | :------------------- | :------------------------------------------------------------- | :-------------------- |
| `--dir.tmp <path>` | `AISUB_DIR_TMP`      | Temporary directory for intermediate segments and state files. | `tmp_<input_stem>`    |
| `--dir.out <path>` | `AISUB_DIR_OUT`      | Output directory for generated subtitle files.                 | Input video directory |

---

## Retry Settings (`--retry.*`)

| Argument                           | Environment Variable           | Description                                                                       | Default |
| :--------------------------------- | :----------------------------- | :-------------------------------------------------------------------------------- | :------ |
| `--retry.per-run <int>`            | `AISUB_RETRY_PER_RUN`          | Maximum internal retries by the executor on transient network or schema failures. | `5`     |
| `--retry.max-runs <int>`           | `AISUB_RETRY_MAX_RUNS`         | Maximum total attempt limit persisted across application executions per segment.  | `3`     |
| `--retry.multiplier <float>`       | `AISUB_RETRY_MULTIPLIER`       | Exponential backoff multiplier between retry attempts.                            | `2.0`   |
| `--retry.min-wait-seconds <float>` | `AISUB_RETRY_MIN_WAIT_SECONDS` | Minimum backoff wait time in seconds.                                             | `30.0`  |
| `--retry.max-wait-seconds <int>`   | `AISUB_RETRY_MAX_WAIT_SECONDS` | Maximum backoff wait time in seconds.                                             | `300`   |

---

## Stage Failure & Pipeline Behavior

To prevent wasted API calls and ensure subtitle integrity, stages adhere to the following execution contracts:

1. **Summary Failure**: If the whole-video summary stage is enabled and exhausts its retries without producing a valid response, pipeline execution **halts immediately**.
2. **Subtitles Failure**: If subtitle generation fails for a particular 5-minute chunk, that chunk **will not continue** to lyrics refinement.
3. **Lyrics Failure**: If a singing performance is detected but lyrics refinement completely fails to execute (e.g. crashes or exhausts retries without producing a response), the stitcher **will not fall back** to the draft subtitles. Instead, it marks that specific segment with an explicit error line (`Error processing subtitles for this segment.`) in the final `.srt` so omissions can be clearly identified. _(Note: If the model completes its execution successfully but verifies that official song lyrics are mismatched or unavailable, it gracefully preserves the draft subtitles without producing an error line)._

---

## Logging Settings (`--log.*`)

| Argument                  | Environment Variable   | Description                                                | Default |
| :------------------------ | :--------------------- | :--------------------------------------------------------- | :------ |
| `--log.level <level>`     | `AISUB_LOG_LEVEL`      | Minimum log level to output.                               | `info`  |
| `--log.timestamps <bool>` | `AISUB_LOG_TIMESTAMPS` | Whether to prefix console log lines with timestamps.       | `False` |
| `--log.scrub <bool>`      | `AISUB_LOG_SCRUB`      | Whether to scrub sensitive API keys and secrets from logs. | `True`  |
| `--log.progress <bool>`   | `AISUB_LOG_PROGRESS`   | Whether to display live Rich progress bars.                | `True`  |
