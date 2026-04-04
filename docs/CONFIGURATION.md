# Configuration

All settings can be configured via command-line arguments (e.g., `--ai.rpm 10`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_RPM=10`).

## AI Settings (`--ai.*`)

| Argument                       | Description                                                                                                                                             | Default                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `--ai.model <model>`           | A shorthand to set both `model_subtitles` and `model_lyrics` to the same value.                                                                         | `None`                              |
| `--ai.model-subtitles <model>` | The AI model for subtitle generation. Use 'google-gla:<model>' for Google models, 'openai:<model>' for OpenAI, or 'custom:<url>' for a custom endpoint. | `google-gla:gemini-3-flash-preview` |
| `--ai.model-lyrics <model>`    | The AI model for lyrics research and scene detection.                                                                                                   | `google-gla:gemini-3-flash-preview` |
| `--ai.rpm <int>`               | Maximum requests per minute for the AI model.                                                                                                           | `4`                                 |
| `--ai.tpm <int>`               | Maximum tokens per minute for the AI model.                                                                                                             | `250000`                            |
| `--ai.web-search-tool <tool>`  | The web search tool to use. Options: 'builtin', 'lyricsgenius'.                                                                                         | `lyricsgenius`                      |

### Google AI Settings (`--ai.google.*`)

| Argument                               | Description                                                       | Default                                                  |
| -------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| `--ai.google.key <key>`                | The API key for Google's generative language models.              | `None` (loads from `GOOGLE_API_KEY` or `GEMINI_API_KEY`) |
| `--ai.google.file-cache-ttl <seconds>` | The time-to-live (TTL) in seconds for the Gemini file list cache. | `10`                                                     |
| `--ai.google.use-files-api <bool>`     | Whether to use the Gemini Files API.                              | `True`                                                   |
| `--ai.google.base-url <url>`           | The base URL for the Google AI API.                               | `None`                                                   |

### Gemini CLI Settings (`--ai.gemini-cli.*`)

| Argument                                         | Description                                                      | Default |
| ------------------------------------------------ | ---------------------------------------------------------------- | ------- |
| `--ai.gemini-cli.timeout <seconds>`              | The timeout in seconds for Gemini CLI operations.                | `600`   |
| `--ai.gemini-cli.overwrite-system-prompt <bool>` | Whether to overwrite the system prompt using `GEMINI_SYSTEM_MD`. | `False` |

## Splitting Settings (`--split.*`)

| Argument                             | Description                                                    | Default |
| ------------------------------------ | -------------------------------------------------------------- | ------- |
| `--split.max-seconds <seconds>`      | The maximum duration in seconds for each video chunk.          | `300`   |
| `--split.start-offset-min <minutes>` | The number of minutes to skip from the beginning of the video. | `0`     |

### Re-Encode Settings (`--split.re-encode.*`)

| Argument                                        | Description                                                                                                            | Default                |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `--split.re-encode.enabled <bool>`              | Re-encode the video chunks to save bandwidth.                                                                          | `False`                |
| `--split.re-encode.fps <int>`                   | The framerate to re-encode the video to.                                                                               | `1`                    |
| `--split.re-encode.height <int>`                | The height (resolution) to re-encode the video to.                                                                     | `360`                  |
| `--split.re-encode.bitrate-kb <int>`            | The bitrate in KB/s to re-encode the video to.                                                                         | `35`                   |
| `--split.re-encode.threshold-mb <int>`          | The threshold in MB for re-encoding. Files smaller than this will not be re-encoded. Set to 0 to re-encode everything. | `20`                   |
| `--split.re-encode.duration-tolerance-ms <int>` | The allowed difference in milliseconds between input and output video duration.                                        | `100`                  |
| `--split.re-encode.encoder <encoder>`           | The specific encoder to use (e.g., 'h264_nvenc').                                                                      | `None` (auto-detected) |

## Directory Settings (`--dir.*`)

| Argument           | Description                                    | Default                          |
| ------------------ | ---------------------------------------------- | -------------------------------- |
| `--dir.tmp <path>` | Temporary directory for intermediate files.    | `tmp_<video_name>` in output dir |
| `--dir.out <path>` | Output directory for the final subtitle files. | Same directory as input video    |

## Concurrency Settings (`--thread.*`)

| Argument                   | Description                                                                                                      | Default |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------- |
| `--thread.uploads <int>`   | The number of concurrent threads for uploading video segments. This is only used for Gemini (google-gla) models. | `4`     |
| `--thread.re-encode <int>` | The number of concurrent threads for re-encoding video chunks.                                                   | `2`     |
| `--thread.lyrics <int>`    | The number of concurrent threads to use for Lyrics/Scene Detection. Set to 0 to disable.                         | `4`     |
| `--thread.subtitles <int>` | The number of concurrent threads to use for Subtitle Generation (Transcription).                                 | `4`     |

## Retry Settings (`--retry.*`)

| Argument                  | Description                                                                   | Default |
| ------------------------- | ----------------------------------------------------------------------------- | ------- |
| `--retry.run <int>`       | The maximum number of times to retry a failed job in this run of the program. | `3`     |
| `--retry.max <int>`       | The absolute maximum number of times a job can be retried in total.           | `9`     |
| `--retry.delay <seconds>` | The number of seconds to wait between retries.                                | `30`    |

## Logging Settings (`--log.*`)

| Argument                  | Description                                          | Default |
| ------------------------- | ---------------------------------------------------- | ------- |
| `--log.level <level>`     | The minimum log level to display.                    | `info`  |
| `--log.timestamps <bool>` | Whether to include timestamps in the console output. | `False` |
| `--log.scrub <bool>`      | Whether to scrub sensitive data from logs.           | `True`  |
