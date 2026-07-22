# Configuration

All settings can be configured via command-line arguments (e.g., `--ai.rpm 10`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_RPM=10`).

## Positional Arguments

| Argument             | Description                                                 | Default  |
| -------------------- | ----------------------------------------------------------- | -------- |
| `<input-video-file>` | The path to the video file for which to generate subtitles. | Required |

## AI Settings (`--ai.*`)

| Argument                              | Description                                                                                                                                                                    | Default                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| `--ai.model <model>`                  | A shorthand to set both `model_subtitles` and `model_lyrics` to the same value. If provided, this will override the other two settings.                                        | `None`                             |
| `--ai.model-subtitles <model>`        | The AI model for subtitle generation. Use 'google-gla:<model>' for Google models, 'openai:<model>' for OpenAI, or 'custom:<url>' for a custom endpoint.                        | `google-gla:gemini-3.6-flash`      |
| `--ai.model-lyrics <model>`           | The AI model for lyrics research and scene detection.                                                                                                                          | `google-gla:gemini-3.5-flash-lite` |
| `--ai.rpm <int>`                      | Maximum Requests Per Minute (RPM) for the AI model provider.                                                                                                                   | `4`                                |
| `--ai.tpm <int>`                      | Maximum Tokens Per Minute (TPM) for the AI model provider.                                                                                                                     | `250000`                           |
| `--ai.validation-buffer-ms <int>`     | The allowed buffer in milliseconds for AI-generated timestamps to exceed the video duration.                                                                                   | `2000`                             |
| `--ai.verification-gap-seconds <int>` | The maximum allowed gap duration in seconds between consecutive subtitles, or between the segment boundaries and the first/last subtitle, that triggers a verification re-run. | `60`                               |
| `--ai.gap-verification-retries <int>` | The maximum number of verification passes/attempts to run for segments with large gaps. Set to 0 to disable gap verification entirely.                                         | `1`                                |

### Google AI Settings (`--ai.google.*`)

| Argument                           | Description                                                                                                                                                              | Default                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| `--ai.google.key <key>`            | The API key for Google's generative language models. Falls back to the GOOGLE_API_KEY or GEMINI_API_KEY environment variables if not set.                                | `None` (loads from environment) |
| `--ai.google.file-cache-ttl <int>` | The time-to-live (TTL) in seconds for the Gemini file list cache. This cache helps avoid frequent API calls to list uploaded files.                                      | `10`                            |
| `--ai.google.use-files-api <bool>` | Enable the Gemini Files API for cloud-based multimodal processing.                                                                                                       | `True`                          |
| `--ai.google.base-url <url>`       | The base URL for the Google AI API. This can be used to override the default endpoint, for instance, to use a proxy. If not provided, Google's default URL will be used. | `None`                          |

### Web Search Settings (`--ai.search.*`)

| Argument                             | Description                                                                                                                                                                                                                                           | Default      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `--ai.search.key <key>`              | The API key for the web search API (Ollama or Langsearch). Falls back to the OLLAMA_API_KEY or LANGSEARCH_API_KEY environment variables.                                                                                                              | `None`       |
| `--ai.search.web-search-tool <tool>` | The web search tool to use. Options are 'builtin' (The provider's native search tool, e.g., Google Search for Gemini), 'duckduckgo', 'ollama', or 'langsearch'. DuckDuckGo is the default because Gemini's built-in search does not have a free tier. | `duckduckgo` |
| `--ai.search.qps <float>`            | Maximum queries per second for the web search API.                                                                                                                                                                                                    | `0.3`        |
| `--ai.search.max-length <int>`       | Discard search responses that are longer than this number of characters.                                                                                                                                                                              | `4096`       |
| `--ai.search.timeout <float>`        | The timeout in seconds for web search HTTP requests. Search queries can occasionally be slow depending on the provider.                                                                                                                               | `60.0`       |

## Splitting Settings (`--split.*`)

| Argument                             | Description                                                                                                                     | Default |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `--split.max-seconds <seconds>`      | The maximum duration in seconds for each video chunk. The input video will be split into these smaller segments for processing. | `300`   |
| `--split.start-offset-min <minutes>` | The number of minutes to skip from the beginning of the video.                                                                  | `0`     |

### Re-Encode Settings (`--split.re-encode.*`)

| Argument                                        | Description                                                                                                            | Default                |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `--split.re-encode.enabled <bool>`              | Re-encode the video chunks to save bandwidth.                                                                          | `False`                |
| `--split.re-encode.fps <float>`                 | The target framerate (FPS) to re-encode the video segments to.                                                         | `1.0`                  |
| `--split.re-encode.height <int>`                | The target height (resolution) to re-encode to. Aspect ratio is preserved.                                             | `360`                  |
| `--split.re-encode.bitrate-kb <int>`            | The target bitrate in Kilobytes per second (KB/s) for the re-encoded video.                                            | `35`                   |
| `--split.re-encode.threshold-mb <int>`          | The threshold in MB for re-encoding. Files smaller than this will not be re-encoded. Set to 0 to re-encode everything. | `20`                   |
| `--split.re-encode.duration-tolerance-ms <int>` | Maximum allowed duration difference (ms) between original and re-encoded segments to consider the output valid.        | `100`                  |
| `--split.re-encode.encoder <encoder>`           | The specific encoder to use (e.g., 'h264_nvenc', 'libx264'). If not provided, it will be automatically detected.       | `None` (auto-detected) |

## Directory Settings (`--dir.*`)

| Argument           | Description                                                                                                                          | Default                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| `--dir.tmp <path>` | Temporary directory for intermediate files (e.g., video segments). Defaults to a 'tmp\_<video_name>' folder in the output directory. | `tmp_input_video_file`    |
| `--dir.out <path>` | Output directory for the final subtitle files. Defaults to the same directory as the input video file.                               | `directory_of_input_file` |

## Concurrency Settings (`--thread.*`)

| Argument                   | Description                                                                                | Default |
| -------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| `--thread.uploads <int>`   | Number of concurrent uploads to the Gemini Files API.                                      | `4`     |
| `--thread.re-encode <int>` | Number of concurrent FFmpeg re-encoding processes.                                         | `2`     |
| `--thread.lyrics <int>`    | Number of concurrent AI tasks for lyrics and scene detection. Set to 0 to skip this stage. | `4`     |
| `--thread.subtitles <int>` | Number of concurrent AI tasks for subtitle generation (transcription and translation).     | `4`     |

## Retry Settings (`--retry.*`)

| Argument                           | Description                                                                                                                                                                                                                                                                                                                                                                      | Default |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `--retry.per-run <int>`            | Maximum internal retries by the AI agent per request. This handles transient API errors and validation failures within a single job execution.                                                                                                                                                                                                                                   | `5`     |
| `--retry.multiplier <float>`       | The multiplier for exponential backoff between retries.                                                                                                                                                                                                                                                                                                                          | `2.0`   |
| `--retry.max-runs <int>`           | Total attempt limit for a single segment stage (e.g. lyrics detection or subtitle generation) across all application runs. This value is persisted in stage-specific state files to prevent infinite retries on problematic segments. Note that attempts are tracked independently per stage; failures in one stage do not count against the attempt limit of subsequent stages. | `3`     |
| `--retry.min-wait-seconds <float>` | The minimum wait time in seconds (lower bound) for a retry attempt.                                                                                                                                                                                                                                                                                                              | `30.0`  |
| `--retry.max-wait-seconds <int>`   | The maximum wait time in seconds (upper bound) for a single retry attempt.                                                                                                                                                                                                                                                                                                       | `300`   |

## Logging Settings (`--log.*`)

| Argument                                     | Description                                                               | Default |
| -------------------------------------------- | ------------------------------------------------------------------------- | ------- |
| `--log.level <level>`                        | The minimum log level to display.                                         | `info`  |
| `--log.timestamps <bool>`                    | Whether to include timestamps in the console output.                      | `False` |
| `--log.scrub <bool>`                         | Whether to scrub sensitive data from logs.                                | `True`  |
| `--log.progress-bars <bool>`                 | Whether to display progress bars in the console.                          | `True`  |
| `--log.progress-bar-width <int>`             | Fixed width for progress bars (in characters).                            | `80`    |
| `--log.progress-bar-refresh-seconds <float>` | Interval in seconds to refresh progress bars to handle terminal resizing. | `1.0`   |
