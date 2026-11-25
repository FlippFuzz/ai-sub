# Configuration Reference

All standalone settings can be configured via command-line arguments (e.g., `--ai.rpm 10`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_RPM=10`).

- **Durations (`timedelta`)** accept numbers in **seconds** (e.g. `300`) or ISO 8601 duration strings (e.g. `PT5M`).
- **Sizes (`ByteSize`)** accept byte counts or unit suffixes such as `15GB`, `300MB`, or `35KB`.
- Features are disabled using `0` or `null`/`None` where indicated.

## Positional Arguments

| Argument  | Description                                        | Default    |
| :-------- | :------------------------------------------------- | :--------- |
| `<video>` | Path to the video file to transcribe and subtitle. | _Required_ |

---

## Context & Output Options

| Argument               | Environment Variable | Description                                                                                                                  | Default |
| :--------------------- | :------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :------ |
| `--context <text>`     | `AISUB_CONTEXT`      | Inline text providing background context, speaker names, or translation preferences (`None` disables).                       | _None_  |
| `--contextfile <path>` | `AISUB_CONTEXTFILE`  | Path to a text file containing user background context (`None` disables).                                                    | _None_  |
| `--nfo <bool>`         | `AISUB_NFO`          | Generate `<input_filename_stem>.nfo` (Jellyfin plot & chapters; requires `concurrency.summary > 0` and NFO Chapters plugin). | `False` |

---

## AI Settings (`--ai.*`)

Engine settings are shared across standalone (`ai-sub`) and workers (`ai-sub-worker` under `ai:` with `AISUBWORKER_AI_*`). Validation buffer (`buffer`) and stage model configs (`summary`, `subtitles`, `lyrics`) apply to standalone mode (workers receive `coordinator.validationbuffer` from the server).

| Argument                 | Environment Variable                                 | Description                                                                                                                                     | Default  |
| :----------------------- | :--------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :------- |
| `--ai.key <key>`         | `AISUB_AI_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini API key.                                                                                                                          | _None_   |
| `--ai.baseurl <url>`     | `AISUB_AI_BASEURL` / `GEMINI_BASE_URL`               | Optional custom base URL or proxy for Google Gemini API calls.                                                                                  | _None_   |
| `--ai.rpm <int>`         | `AISUB_AI_RPM`                                       | Maximum Requests Per Minute (RPM) allowed per model.                                                                                            | `4`      |
| `--ai.tpm <int>`         | `AISUB_AI_TPM`                                       | Maximum Tokens Per Minute (TPM) allowed per model.                                                                                              | `250000` |
| `--ai.models <dict>`     | `AISUB_AI_MODELS`                                    | Optional JSON map of per-model `rpm`/`tpm` overrides inheriting unspecified fields from `ai` (e.g. `'{"gemini-3.5-flash-lite": {"rpm": 15}}'`). | `{}`     |
| `--ai.buffer <duration>` | `AISUB_AI_BUFFER`                                    | Allowed tolerance buffer for subtitle timestamps exceeding segment duration.                                                                    | `2` (2s) |

### Per-Stage Models (`--ai.{summary,subtitles,lyrics}.*`)

| Argument                        | Environment Variable         | Description                                                  | Default                 |
| :------------------------------ | :--------------------------- | :----------------------------------------------------------- | :---------------------- |
| `--ai.summary.model <model>`    | `AISUB_AI_SUMMARY_MODEL`     | Gemini model for whole-video summary and knowledge indexing. | `gemini-3.5-flash-lite` |
| `--ai.summary.tier <tier>`      | `AISUB_AI_SUMMARY_TIER`      | Google Gemini service tier (`standard`, `flex`, `priority`). | _None_                  |
| `--ai.summary.agentic <bool>`   | `AISUB_AI_SUMMARY_AGENTIC`   | Whether to enable agentic video understanding for summary.   | `True`                  |
| `--ai.subtitles.model <model>`  | `AISUB_AI_SUBTITLES_MODEL`   | Gemini model for audio-visual subtitle transcription.        | `gemini-4-hydrogen`     |
| `--ai.subtitles.tier <tier>`    | `AISUB_AI_SUBTITLES_TIER`    | Google Gemini service tier for subtitle generation.          | _None_                  |
| `--ai.subtitles.agentic <bool>` | `AISUB_AI_SUBTITLES_AGENTIC` | Whether to enable agentic video understanding for subtitles. | `False`                 |
| `--ai.lyrics.model <model>`     | `AISUB_AI_LYRICS_MODEL`      | Gemini model for musical lyrics refinement.                  | `gemini-3.5-flash-lite` |
| `--ai.lyrics.tier <tier>`       | `AISUB_AI_LYRICS_TIER`       | Google Gemini service tier for lyrics refinement.            | _None_                  |
| `--ai.lyrics.agentic <bool>`    | `AISUB_AI_LYRICS_AGENTIC`    | Whether to enable agentic video understanding for lyrics.    | `True`                  |

### Files API Storage Settings (`--ai.storage.*`)

| Argument                                  | Environment Variable               | Description                                                          | Default      |
| :---------------------------------------- | :--------------------------------- | :------------------------------------------------------------------- | :----------- |
| `--ai.storage.max <bytesize>`             | `AISUB_AI_STORAGE_MAX`             | High-water storage mark triggering managed file cleanup.             | `15GB`       |
| `--ai.storage.target <bytesize>`          | `AISUB_AI_STORAGE_TARGET`          | Low-water storage target to delete down to during cleanup (`< max`). | `10GB`       |
| `--ai.storage.cleanupinterval <duration>` | `AISUB_AI_STORAGE_CLEANUPINTERVAL` | Minimum cooldown interval between storage cleanup checks.            | `1800` (30m) |
| `--ai.storage.pollinterval <duration>`    | `AISUB_AI_STORAGE_POLLINTERVAL`    | Polling interval while waiting for Gemini Files API processing.      | `10` (10s)   |

### Web Search Settings (`--ai.search.*`)

| Argument                                 | Environment Variable                                   | Description                                                                                               | Default                    |
| :--------------------------------------- | :----------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------- |
| `--ai.search.tools <list>`               | `AISUB_AI_SEARCH_TOOLS`                                | Enabled providers (`builtin`, `duckduckgo`, `ollama`, `langsearch`, `tinyfish`, `lrclib`; `[]` disables). | `["duckduckgo", "lrclib"]` |
| `--ai.search.tinyfishkey <key>`          | `AISUB_AI_SEARCH_TINYFISHKEY` / `TINYFISH_API_KEY`     | Dedicated API key for Tinyfish search.                                                                    | _None_                     |
| `--ai.search.langsearchkey <key>`        | `AISUB_AI_SEARCH_LANGSEARCHKEY` / `LANGSEARCH_API_KEY` | Dedicated API key for Langsearch.                                                                         | _None_                     |
| `--ai.search.ollamakey <key>`            | `AISUB_AI_SEARCH_OLLAMAKEY` / `OLLAMA_API_KEY`         | Dedicated API key for Ollama search.                                                                      | _None_                     |
| `--ai.search.maxturns <int>`             | `AISUB_AI_SEARCH_MAXTURNS`                             | Maximum function-calling search turns permitted per segment.                                              | `3`                        |
| `--ai.search.qps <float>`                | `AISUB_AI_SEARCH_QPS`                                  | Maximum queries per second for web search requests.                                                       | `0.3`                      |
| `--ai.search.maxlength <int>`            | `AISUB_AI_SEARCH_MAXLENGTH`                            | Maximum characters permitted in a single search snippet or fetched page.                                  | `4096`                     |
| `--ai.search.timeout <duration>`         | `AISUB_AI_SEARCH_TIMEOUT`                              | HTTP request timeout for external search providers.                                                       | `60` (60s)                 |
| `--ai.search.retries <int>`              | `AISUB_AI_SEARCH_RETRIES`                              | Maximum retries for search requests on transient errors.                                                  | `3`                        |
| `--ai.search.backoff.multiplier <float>` | `AISUB_AI_SEARCH_BACKOFF_MULTIPLIER`                   | Exponential backoff multiplier for search retries.                                                        | `2.0`                      |
| `--ai.search.backoff.minwait <duration>` | `AISUB_AI_SEARCH_BACKOFF_MINWAIT`                      | Minimum backoff wait duration for search retries.                                                         | `5` (5s)                   |
| `--ai.search.backoff.maxwait <duration>` | `AISUB_AI_SEARCH_BACKOFF_MAXWAIT`                      | Maximum backoff wait duration for search retries.                                                         | `60` (60s)                 |

---

## Re-Encoding Settings (`--reencode.*`)

| Argument                   | Environment Variable     | Description                                                                               | Default |
| :------------------------- | :----------------------- | :---------------------------------------------------------------------------------------- | :------ |
| `--reencode.encoder <str>` | `AISUB_REENCODE_ENCODER` | Specific H.264 encoder to use (e.g., `'h264_nvenc'`, `'libx264'`). Auto-detected if null. | _None_  |

### Stage Re-Encoding (`--reencode.summary.*` and `--reencode.segment.*`)

| Argument                                         | Environment Variable                         | Description                                                                                       | Default                                 |
| :----------------------------------------------- | :------------------------------------------- | :------------------------------------------------------------------------------------------------ | :-------------------------------------- |
| `--reencode.<stage>.threshold <bytesize>`        | `AISUB_REENCODE_<STAGE>_THRESHOLD`           | Minimum file size triggering initial re-encoding (`0` disables initial re-encoding).              | `300MB` (summary) / `0` (segment)       |
| `--reencode.<stage>.fps <float\|keyframes>`      | `AISUB_REENCODE_<STAGE>_FPS`                 | Target framerate in FPS, or `keyframes` to extract native keyframes without frame duplication.    | `keyframes` (summary) / `1.0` (segment) |
| `--reencode.<stage>.height <int>`                | `AISUB_REENCODE_<STAGE>_HEIGHT`              | Target resolution height in pixels (aspect ratio is preserved).                                   | `360`                                   |
| `--reencode.<stage>.bitrate <bytesize>`          | `AISUB_REENCODE_<STAGE>_BITRATE`             | Target video bitrate in bytes per second (e.g., `35KB` = 280 kbps; `b` denotes bytes, not bits).  | `35KB`                                  |
| `--reencode.<stage>.tolerance <duration>`        | `AISUB_REENCODE_<STAGE>_TOLERANCE`           | Allowed duration divergence for re-encoded output.                                                | `0.1` (100ms)                           |
| `--reencode.<stage>.fallback.attempts <int>`     | `AISUB_REENCODE_<STAGE>_FALLBACK_ATTEMPTS`   | Number of fallback FPS re-encoding tiers triggered after `retry.maxruns` failures (`0` disables). | `5`                                     |
| `--reencode.<stage>.fallback.multiplier <float>` | `AISUB_REENCODE_<STAGE>_FALLBACK_MULTIPLIER` | Divisor applied per fallback tier (`base_fps / multiplier ** tier`, must be `> 1.0`).             | `2.0`                                   |

---

## Splitting Settings (`--split.*`)

| Argument                       | Environment Variable    | Description                                                            | Default       |
| :----------------------------- | :---------------------- | :--------------------------------------------------------------------- | :------------ |
| `--split.duration <duration>`  | `AISUB_SPLIT_DURATION`  | Maximum duration for each video chunk.                                 | `300` (5m)    |
| `--split.offset <duration>`    | `AISUB_SPLIT_OFFSET`    | Duration to skip from the beginning of the video (`0` starts at 0).    | `0`           |
| `--split.tolerance <duration>` | `AISUB_SPLIT_TOLERANCE` | Maximum allowed duration variance per segment during split validation. | `0.1` (100ms) |

---

## Concurrency Settings (`--concurrency.*`)

| Argument                        | Environment Variable          | Description                                                            | Default |
| :------------------------------ | :---------------------------- | :--------------------------------------------------------------------- | :------ |
| `--concurrency.reencode <int>`  | `AISUB_CONCURRENCY_REENCODE`  | Number of concurrent FFmpeg compression processes.                     | `2`     |
| `--concurrency.summary <int>`   | `AISUB_CONCURRENCY_SUMMARY`   | Number of concurrent summary tasks (`0` disables whole-video summary). | `1`     |
| `--concurrency.subtitles <int>` | `AISUB_CONCURRENCY_SUBTITLES` | Number of concurrent subtitle generation tasks.                        | `4`     |
| `--concurrency.lyrics <int>`    | `AISUB_CONCURRENCY_LYRICS`    | Number of concurrent lyrics refinement tasks (`0` disables lyrics).    | `4`     |

---

## Directory Settings (`--dir.*`)

| Argument           | Environment Variable | Description                                                    | Default                     |
| :----------------- | :------------------- | :------------------------------------------------------------- | :-------------------------- |
| `--dir.tmp <path>` | `AISUB_DIR_TMP`      | Temporary directory for intermediate segments and state files. | _None_ (`out / tmp_<stem>`) |
| `--dir.out <path>` | `AISUB_DIR_OUT`      | Output directory for generated subtitle files.                 | _None_ (input directory)    |

---

## Retry Settings (`--retry.*`)

| Argument                             | Environment Variable             | Description                                                                                                 | Default    |
| :----------------------------------- | :------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- |
| `--retry.perturn <int>`              | `AISUB_RETRY_PERTURN`            | Maximum retries per API turn on transient network, rate-limit, or 5xx errors.                               | `5`        |
| `--retry.validation <int>`           | `AISUB_RETRY_VALIDATION`         | Maximum full-session retries per run when AI output fails schema validation.                                | `3`        |
| `--retry.maxruns <int>`              | `AISUB_RETRY_MAXRUNS`            | Maximum non-retryable failed runs per FPS tier before fallback or exhaustion.                               | `3`        |
| `--retry.backoff.multiplier <float>` | `AISUB_RETRY_BACKOFF_MULTIPLIER` | Exponential backoff multiplier between retry attempts.                                                      | `2.0`      |
| `--retry.backoff.minwait <duration>` | `AISUB_RETRY_BACKOFF_MINWAIT`    | Minimum backoff wait duration.                                                                              | `30` (30s) |
| `--retry.backoff.maxwait <duration>` | `AISUB_RETRY_BACKOFF_MAXWAIT`    | Maximum backoff wait duration.                                                                              | `300` (5m) |
| `--retry.models <dict>`              | `AISUB_RETRY_MODELS`             | Optional JSON map of per-model retry overrides inheriting unspecified fields from top-level retry defaults. | `{}`       |

---

## Logging Settings (`--log.*`)

| Argument                  | Environment Variable                | Description                                                | Default |
| :------------------------ | :---------------------------------- | :--------------------------------------------------------- | :------ |
| `--log.level <level>`     | `AISUB_LOG_LEVEL`                   | Minimum log level to output.                               | `info`  |
| `--log.timestamps <bool>` | `AISUB_LOG_TIMESTAMPS`              | Whether to prefix console log lines with timestamps.       | `False` |
| `--log.scrub <bool>`      | `AISUB_LOG_SCRUB`                   | Whether to scrub sensitive API keys and secrets from logs. | `True`  |
| `--log.token <token>`     | `AISUB_LOG_TOKEN` / `LOGFIRE_TOKEN` | Logfire write token.                                       | _None_  |
| `--log.progress <bool>`   | `AISUB_LOG_PROGRESS`                | Whether to display live Rich progress bars.                | `True`  |
