# AI Sub: AI-Powered Subtitle Generation with Translation

[![PyPI version](https://img.shields.io/pypi/v/ai-sub)](https://pypi.org/project/ai-sub)
[![Downloads](https://img.shields.io/pypi/dw/ai-sub)](https://pypistats.org/packages/ai-sub)

---

## Table of Contents

- [Overview](#overview)
- [Showcase](#showcase)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Release Notes](#release-notes)
- [Known Limitations](#known-limitations)
- [Advanced: Re-processing Segments](#advanced-re-processing-segments)

## Overview

**AI Sub** is a command-line tool that leverages Google's **Gemini** models to generate high-quality, audio-synchronized subtitles. It is designed to produce dual-language subtitles (Original + English translation) by analyzing both audio and visual cues.

---

## Showcase

Please visit [ai-sub-showcase](https://github.com/FlippFuzz/ai-sub-showcase).

[![Video Screenshot](https://github.com/FlippFuzz/ai-sub/raw/main/showcase/old/42h4ydJS3zk.png)](https://raw.githubusercontent.com/FlippFuzz/ai-sub/refs/heads/main/showcase/old/42h4ydJS3zk.v007.srt)

---

## How It Works

1.  **Segmentation:** Splits the input video into manageable 5-minute segments.
2.  **Re-encoding (Optional):** Compresses segments (e.g., 1fps, 360p) to significantly reduce bandwidth usage and upload times without sacrificing AI analysis accuracy.
3.  **Upload (Optional):** Uploads segments to the Gemini Files API for cloud-based processing. This bypasses local processing constraints and leverages the API's multimodal capabilities.
4.  **Lyrics Search and Scene Detection (Optional):** Detects scenes and performs a web search for official lyrics to improve transcription accuracy for songs.
5.  **Generation:** Generates synchronized subtitles and translations by using the audio as the ground truth for timing and visuals for context.
6.  **Assembly:** Stitches the generated subtitles back together into a final SRT file.

---

## Installation

**Prerequisites:** Python 3.10 or higher.

1.  **Set up a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate.bat`
    ```

2.  **Install AI Sub:**

    ```bash
    pip install --upgrade ai-sub
    ```

---

## Usage

You can use AI Sub a Google AI Studio API Key and an optional langsearch API key for song lyrics web search.

### Option 1 (Recommended): Google API Key + Langsearch API Key

This option provides the best quality for the free tier.

- As of 3 June 2026:
- Gemini API provides 20 free requests/day for gemini-3.5-flash
- Gemini API provides 500 free requests/day for gemini-3.1-flash-lite
- Langsearch provides 1000 free request/day for web searches

The limiting factor is the daily free quota for gemini-3.5-flash.
You should be able to create subtitles for between 1 hour to 1 hour 40 mins of videos per day.

1.  **Obtain your API Keys:**
    - Sign in to [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Click "Create API Key".

    - Sign in to [Langsearch](https://langsearch.com/api-keys).
    - API Keys -> Create API Key

    - Copy and securely store your keys. **Never disclose your API key publicly.**

2.  **Run the application:**

    Linux:

    ```bash
    GOOGLE_KEY="YOUR_GOOGLE_API_KEY"
    LANGSEARCH_KEY="YOUR_LANGSEARCH_API_KEY"
    FILE_NAME="path/to/your/video.mp4"

    ai-sub \
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite \
    --ai.model-subtitles=google-gla:gemini-3.5-flash \
    --ai.search.web-search-tool=langsearch \
    --ai.google.key="${GOOGLE_KEY}" \
    --ai.search.key="${LANGSEARCH_KEY}" \
    "${FILE_NAME}"
    ```

    Windows:

    ```cmd
    SET "GOOGLE_KEY=YOUR_GOOGLE_API_KEY"
    SET "LANGSEARCH_KEY=YOUR_LANGSEARCH_API_KEY"
    SET "FILE_NAME=path/to/your/video.mp4"

    ai-sub ^
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite ^
    --ai.model-subtitles=google-gla:gemini-3.5-flash ^
    --ai.search.web-search-tool=langsearch ^
    --ai.google.key="%GOOGLE_KEY%" ^
    --ai.search.key="%LANGSEARCH_KEY%" ^
    "%FILE_NAME%"
    ```

### Option 2: Using Google API Key Only

This is the easiest option if you don't want to bother with creating a Langsearch API key.
We can use duckduckgo to try to lookup lyrics, but it is not as reliable.

1.  **Obtain your API Key:**
    - Sign in to [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Click "Create API Key".
    - Copy and securely store your key. **Never disclose your API key publicly.**

2.  **Run the application:**

    ```bash
    GOOGLE_KEY="YOUR_GOOGLE_API_KEY"
    FILE_NAME="path/to/your/video.mp4"

    ai-sub \
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite \
    --ai.model-subtitles=google-gla:gemini-3.5-flash \
    --ai.search.web-search-tool=duckduckgo \
    --ai.google.key="${GOOGLE_KEY}" \
    "${FILE_NAME}"
    ```

    Windows:

    ```cmd
    SET "GOOGLE_KEY=YOUR_GOOGLE_API_KEY"
    SET "FILE_NAME=path/to/your/video.mp4"

    ai-sub ^
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite ^
    --ai.model-subtitles=google-gla:gemini-3.5-flash ^
    --ai.search.web-search-tool=duckduckgo ^
    --ai.google.key="%GOOGLE_KEY%" ^
    "%FILE_NAME%"
    ```

---

## Configuration

For a detailed list of all configuration options, including AI models, re-encoding settings, and concurrency controls, please refer to [CONFIGURATION.md](https://github.com/FlippFuzz/ai-sub/blob/main/docs/CONFIGURATION.md).

All settings can be configured via command-line arguments (e.g., `--ai.model=google-gla:gemini-3.5-flash`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_MODEL=google-gla:gemini-3.5-flash`).

---

## Release Notes

For the latest updates and bug fixes, please refer to [RELEASE_NOTES.md](https://github.com/FlippFuzz/ai-sub/blob/main/RELEASE_NOTES.md).

---

## Known Limitations

1.  **Timestamp Accuracy:** Subtitle timestamps may occasionally be inaccurate due to limitations of the Gemini AI model. Shorter video segments generally yield better accuracy. Experiment with the `--split.max-seconds` setting.
2.  **AI Hallucinations:** Like all LLMs, Gemini may occasionally produce "hallucinations" or inaccurate information.

If you encounter issues, consider re-processing specific video segments as detailed below.

---

## Advanced: Re-processing Segments

Intermediate files and job states are stored in a temporary directory (default: `tmp_<input_file_name>`). You can customize this location using the `--dir.tmp` flag.

The application creates separate state files for each processing stage (e.g., lyrics detection, subtitle generation). To re-process a specific segment, you must delete the state file for the stage you want to re-run.

File naming format: `part_XXX.<stage>.<model_name>.json`

**Example: To re-run subtitle generation for the third segment:**

1.  Navigate to the temporary directory.
2.  Identify the model name used for subtitles (e.g., `gemini-3.5-flash`).
3.  Delete the corresponding state file (e.g., `part_002.subtitles.gemini-3-5-flash.json`).
4.  Re-run the script. It will detect the missing subtitle job state and re-process only that segment, using any existing lyrics data.

To re-run the entire pipeline for that segment (including lyrics search), delete both the `lyrics` and `subtitles` JSON files for that part.
