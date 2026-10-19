# Configuration Reference

All settings can be configured via command-line arguments (e.g., `--ai.rpm 10`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_RPM=10`).

## Positional Arguments

| Argument             | Description                                        | Default    |
| :------------------- | :------------------------------------------------- | :--------- |
| `<input_video_file>` | Path to the video file to transcribe and subtitle. | _Required_ |

---

## AI Settings (`--ai.*`)

| Argument                             | Environment Variable                                 | Description                                                                 | Default                            |
| :----------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------- | :--------------------------------- |
| `--ai.key <key>`                     | `AISUB_AI_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini API key.                                                      | _None_                             |
| `--ai.base-url <url>`                | `AISUB_AI_BASE_URL`                                  | Optional base URL or proxy for Google Gemini API calls.                     | _None_                             |
| `--ai.model <model>`                 | `AISUB_AI_MODEL`                                     | Shorthand setting both `model_subtitles` and `model_lyrics` simultaneously. | _None_                             |
| `--ai.model-subtitles <model>`       | `AISUB_AI_MODEL_SUBTITLES`                           | Gemini model for subtitle generation. Must start with `google-gla:`.        | `google-gla:gemini-3.7-flash`      |
| `--ai.model-lyrics <model>`          | `AISUB_AI_MODEL_LYRICS`                              | Gemini model for scene detection and song lyrics search.                    | `google-gla:gemini-3.5-flash-lite` |
| `--ai.service-tier <tier>`           | `AISUB_AI_SERVICE_TIER`                              | Shorthand service tier (`auto`, `default`, `flex`, `priority`).             | _None_                             |
| `--ai.service-tier-subtitles <tier>` | `AISUB_AI_SERVICE_TIER_SUBTITLES`                    | Service tier for subtitle generation.                                       | _None_                             |
| `--ai.service-tier-lyrics <tier>`    | `AISUB_AI_SERVICE_TIER_LYRICS`                       | Service tier for lyrics and scene detection.                                | _None_                             |
| `--ai.rpm <int>`                     | `AISUB_AI_RPM`                                       | Maximum Requests Per Minute (RPM) for model endpoints.                      | `4`                                |
| `--ai.tpm <int>`                     | `AISUB_AI_TPM`                                       | Maximum Tokens Per Minute (TPM) for model endpoints.                        | `250000`                           |
| `--ai.validation-buffer-ms <int>`    | `AISUB_AI_VALIDATION_BUFFER_MS`                      | Allowed tolerance buffer (ms) for timestamps exceeding segment duration.    | `2000`                             |

### Files API Storage Settings (`--ai.storage.*`)

| Argument                                        | Environment Variable                        | Description                                                        | Default |
| :---------------------------------------------- | :------------------------------------------ | :----------------------------------------------------------------- | :------ |
| `--ai.storage.max-gb <float>`                   | `AISUB_AI_STORAGE_MAX_GB`                   | High-water mark in GB triggering cleanup of old managed segments.  | `15.0`  |
| `--ai.storage.target-gb <float>`                | `AISUB_AI_STORAGE_TARGET_GB`                | Low-water mark in GB to delete down to during storage cleanup.     | `10.0`  |
| `--ai.storage.cleanup-interval-minutes <int>`   | `AISUB_AI_STORAGE_CLEANUP_INTERVAL_MINUTES` | Cooldown period in minutes between storage cleanup checks.         | `30`    |
| `--ai.storage.polling-interval-seconds <float>` | `AISUB_AI_STORAGE_POLLING_INTERVAL_SECONDS` | Interval in seconds between upload status checks while processing. | `10.0`  |

### Web Search Settings (`--ai.search.*`)

| Argument                               | Environment Variable                                                                 | Description                                                                      | Default      |
| :------------------------------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :----------- |
| `--ai.search.tool <tool>`              | `AISUB_AI_SEARCH_TOOL`                                                               | Search provider: `duckduckgo`, `langsearch`, `ollama`, `tinyfish`, or `builtin`. | `duckduckgo` |
| `--ai.search.key <key>`                | `AISUB_AI_SEARCH_KEY` / `LANGSEARCH_API_KEY` / `OLLAMA_API_KEY` / `TINYFISH_API_KEY` | API key for Langsearch, Ollama, or Tinyfish.                                     | _None_       |
| `--ai.search.max-turns <int>`          | `AISUB_AI_SEARCH_MAX_TURNS`                                                          | Maximum function-calling search turns allowed per segment.                       | `3`          |
| `--ai.search.qps <float>`              | `AISUB_AI_SEARCH_QPS`                                                                | Maximum queries per second for web search requests.                              | `0.3`        |
| `--ai.search.max-length <int>`         | `AISUB_AI_SEARCH_MAX_LENGTH`                                                         | Character cutoff limit per search result snippet or fetched page.                | `4096`       |
| `--ai.search.timeout <float>`          | `AISUB_AI_SEARCH_TIMEOUT`                                                            | Timeout in seconds for search HTTP requests.                                     | `60.0`       |
| `--ai.search.retries <int>`            | `AISUB_AI_SEARCH_RETRIES`                                                            | Maximum retries on transient search network errors.                              | `3`          |
| `--ai.search.min-wait-seconds <float>` | `AISUB_AI_SEARCH_MIN_WAIT_SECONDS`                                                   | Minimum backoff wait time in seconds for search retries.                         | `1.0`        |
| `--ai.search.max-wait-seconds <float>` | `AISUB_AI_SEARCH_MAX_WAIT_SECONDS`                                                   | Maximum backoff wait time in seconds for search retries.                         | `10.0`       |
| `--ai.search.multiplier <float>`       | `AISUB_AI_SEARCH_MULTIPLIER`                                                         | Exponential backoff multiplier for search retries.                               | `2.0`        |

---

## Splitting Settings (`--split.*`)

| Argument                              | Environment Variable                | Description                                                   | Default |
| :------------------------------------ | :---------------------------------- | :------------------------------------------------------------ | :------ |
| `--split.max-seconds <int>`           | `AISUB_SPLIT_MAX_SECONDS`           | Maximum duration in seconds for each video chunk.             | `300`   |
| `--split.start-offset-min <int>`      | `AISUB_SPLIT_START_OFFSET_MIN`      | Number of minutes to skip from the beginning of the video.    | `0`     |
| `--split.duration-tolerance-ms <int>` | `AISUB_SPLIT_DURATION_TOLERANCE_MS` | Duration tolerance (ms) per segment to reuse existing splits. | `100`   |

### Re-Encoding Settings (`--split.re-encode.*`)

| Argument                                        | Environment Variable                          | Description                                                              | Default |
| :---------------------------------------------- | :-------------------------------------------- | :----------------------------------------------------------------------- | :------ |
| `--split.re-encode.enabled <bool>`              | `AISUB_SPLIT_RE_ENCODE_ENABLED`               | Re-encode segments before uploading to reduce bandwidth.                 | `False` |
| `--split.re-encode.fps <float>`                 | `AISUB_SPLIT_RE_ENCODE_FPS`                   | Target framerate (FPS) for compressed segments.                          | `1.0`   |
| `--split.re-encode.height <int>`                | `AISUB_SPLIT_RE_ENCODE_HEIGHT`                | Target resolution height in pixels (aspect ratio preserved).             | `360`   |
| `--split.re-encode.bitrate-kb <int>`            | `AISUB_SPLIT_RE_ENCODE_BITRATE_KB`            | Target video bitrate in Kilobytes per second (KB/s).                     | `35`    |
| `--split.re-encode.threshold-mb <int>`          | `AISUB_SPLIT_RE_ENCODE_THRESHOLD_MB`          | File size threshold in MB above which segments are re-encoded.           | `20`    |
| `--split.re-encode.duration-tolerance-ms <int>` | `AISUB_SPLIT_RE_ENCODE_DURATION_TOLERANCE_MS` | Allowed duration difference (ms) for re-encoded outputs.                 | `100`   |
| `--split.re-encode.encoder <str>`               | `AISUB_SPLIT_RE_ENCODE_ENCODER`               | Specific encoder (e.g., `h264_nvenc`, `libx264`). Auto-detected if null. | _None_  |

---

## Concurrency Settings (`--concurrency.*`)

| Argument                        | Environment Variable          | Description                                                              | Default |
| :------------------------------ | :---------------------------- | :----------------------------------------------------------------------- | :------ |
| `--concurrency.re-encode <int>` | `AISUB_CONCURRENCY_RE_ENCODE` | Number of concurrent FFmpeg compression tasks.                           | `2`     |
| `--concurrency.lyrics <int>`    | `AISUB_CONCURRENCY_LYRICS`    | Number of concurrent lyrics/scene analysis tasks. Set to `0` to disable. | `4`     |
| `--concurrency.subtitles <int>` | `AISUB_CONCURRENCY_SUBTITLES` | Number of concurrent subtitle generation tasks.                          | `4`     |

---

## Directory Settings (`--dir.*`)

| Argument           | Environment Variable | Description                                                  | Default               |
| :----------------- | :------------------- | :----------------------------------------------------------- | :-------------------- |
| `--dir.tmp <path>` | `AISUB_DIR_TMP`      | Working directory for intermediate segments and state files. | `tmp_<input_stem>`    |
| `--dir.out <path>` | `AISUB_DIR_OUT`      | Output directory for final combined `.srt` subtitle files.   | Input video directory |

---

## Retry Settings (`--retry.*`)

| Argument                           | Environment Variable           | Description                                                              | Default |
| :--------------------------------- | :----------------------------- | :----------------------------------------------------------------------- | :------ |
| `--retry.per-run <int>`            | `AISUB_RETRY_PER_RUN`          | Internal retries by the AI executor on transient API or schema errors.   | `5`     |
| `--retry.max-runs <int>`           | `AISUB_RETRY_MAX_RUNS`         | Total attempt limit persisted across application executions per segment. | `3`     |
| `--retry.multiplier <float>`       | `AISUB_RETRY_MULTIPLIER`       | Multiplier for exponential backoff between retries.                      | `2.0`   |
| `--retry.min-wait-seconds <float>` | `AISUB_RETRY_MIN_WAIT_SECONDS` | Minimum wait time in seconds for a retry backoff.                        | `30.0`  |
| `--retry.max-wait-seconds <int>`   | `AISUB_RETRY_MAX_WAIT_SECONDS` | Maximum wait time in seconds for a single retry backoff.                 | `300`   |

---

## Logging Settings (`--log.*`)

| Argument                  | Environment Variable   | Description                                                | Default |
| :------------------------ | :--------------------- | :--------------------------------------------------------- | :------ |
| `--log.level <level>`     | `AISUB_LOG_LEVEL`      | Minimum log level (`debug`, `info`, `warning`, `error`).   | `info`  |
| `--log.timestamps <bool>` | `AISUB_LOG_TIMESTAMPS` | Whether to prefix console log lines with timestamps.       | `False` |
| `--log.scrub <bool>`      | `AISUB_LOG_SCRUB`      | Whether to scrub sensitive API keys and secrets from logs. | `True`  |
| `--log.progress <bool>`   | `AISUB_LOG_PROGRESS`   | Whether to render live Rich progress dashboard bars.       | `True`  |
