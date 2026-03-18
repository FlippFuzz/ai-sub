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

**AI Sub** is a command-line tool that leverages Google's **Gemini** models to generate high-quality, audio-synchronized subtitles. It is designed to produce precise subtitles in the video's original language along with English translations by analyzing both audio and visual cues.

---

## Showcase

Please visit [ai-sub-showcase](https://github.com/FlippFuzz/ai-sub-showcase).

[![Video Screenshot](https://github.com/FlippFuzz/ai-sub/raw/main/showcase/old/42h4ydJS3zk.png)](https://raw.githubusercontent.com/FlippFuzz/ai-sub/refs/heads/main/showcase/old/42h4ydJS3zk.v007.srt)

---

## How It Works

1.  **Segmentation:** Splits the input video into manageable 5-minute segments.
2.  **Re-encoding (Optional):** Compresses segments to lower quality (e.g., 1fps, 360p) to reduce bandwidth and upload times.
3.  **Upload (Optional):** Uploads segments to the Gemini Files API for cloud-based processing. The re-encoding step helps to ensure the segments are below the API's file size limit.
4.  **Lyrics Search and Scene Detection (Optional):** Detects scenes and performs a web search for official lyrics to improve transcription accuracy for songs.
5.  **Generation:** Generates precise, synchronized subtitles and translations using the AI model.
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

##

## Usage

You can use AI Sub with either a Google AI Studio API Key or the Gemini CLI, or a mix of both.

### Option 1: Google API Key for Lyrics Search + Gemini CLI for Subtitle Generation (Best for free tier users)

This option provides the best quality for the free tier but is slightly more involved.

- As of 18 Mar 2026:
- Gemini API provides 500 free requests/day for gemini-3.1-flash-lite-preview
- Gemini CLI provides 50 free requests/day for gemini-3-pro-preview and 1000 free requests/day for gemini-3-flash-preview

1.  **Obtain your API Key:**
    - Sign in to [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Click "Create API Key".
    - Copy and securely store your key. **Never disclose your API key publicly.**

2.  **Install and Authenticate Gemini CLI:**
    - Install: `npm install -g @google/gemini-cli`
    - **Note:** This requires Node.js and npm to be installed.
    - Authenticate: Follow the instructions at [gemini-cli](https://github.com/google-gemini/gemini-cli?tab=readme-ov-file#-authentication-options).

3.  **Run the application:**

    Linux:

    ```bash
    KEY="YOUR_API_KEY"
    FILE_NAME="path/to/your/video.mp4"

    ai-sub \
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite-preview \
    --ai.google.key="${KEY}" \
    --ai.model-subtitles=gemini-cli:gemini-3-pro-preview \
    --split.re-encode.enabled=True \
    "${FILE_NAME}"
    ```

    Windows:

    ```cmd
    SET "KEY=YOUR_API_KEY"
    SET "FILE_NAME=path/to/your/video.mp4"

    ai-sub ^
    --ai.model-lyrics=google-gla:gemini-3.1-flash-lite-preview ^
    --ai.google.key="%KEY%" ^
    --ai.model-subtitles=gemini-cli:gemini-3-pro-preview ^
    --split.re-encode.enabled=True ^
    "%FILE_NAME%"
    ```

### Option 2: Using Gemini CLI Only

I recommend against using Gemini CLI for web searches, especially on the free tier.
You'll likely encounter "429 RESOURCE_EXHAUSTED" errors due to hidden quota limits on web searches.

1.  **Install and Authenticate Gemini CLI:**
    - Install: `npm install -g @google/gemini-cli`
    - **Note:** This requires Node.js and npm to be installed.
    - Authenticate: Follow instructions at [gemini-cli](https://github.com/google-gemini/gemini-cli?tab=readme-ov-file#-authentication-options).

2.  **Run the application:**

    Linux:

    ```bash
    FILE_NAME="path/to/your/video.mp4"

    ai-sub \
      --ai.model-subtitles=gemini-cli:gemini-3-pro-preview \
      --split.re-encode.enabled=True \
      --thread.lyrics=0 \
      "${FILE_NAME}"
    ```

    Windows:

    ```cmd
    SET "FILE_NAME=path/to/your/video.mp4"

    ai-sub ^
      --ai.model-subtitles=gemini-cli:gemini-3-pro-preview ^
      --split.re-encode.enabled=True ^
      --thread.lyrics=0 ^
      "%FILE_NAME%"
    ```

    **Important Notes for CLI Mode:**
    - No API key is required; the tool uses your authenticated Gemini CLI instance.
    - Additional arguments are required to split and re-encode the video because the Gemini CLI has a 20MB upload limit per chunk. The default re-encoding settings are aggressive and should work for most inputs.
    - **Re-encoding is resource-intensive and will increase processing time.**

### Option 3: Using Google API Key Only

This is the easiest option if you don't want to set up Gemini CLI, but the quality is lower because the only free model is weak.

- As of 18 Mar 2026:
- Gemini API provides 500 free request/day for gemini-3.1-flash-lite-preview
- Note that "flash-lite" is the weakest model. Higher models are not free and you need to setup billing.

1.  **Obtain your API Key:**
    - Sign in to [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Click "Create API Key".
    - Copy and securely store your key. **Never disclose your API key publicly.**

2.  **Run the application:**

    Linux:

    ```bash
    KEY="YOUR_API_KEY"
    FILE_NAME="path/to/your/video.mp4"

    ai-sub \
      --ai.model=google-gla:gemini-3.1-flash-lite-preview \
      --ai.google.key="${KEY}" \
      "${FILE_NAME}"
    ```

    Windows:

    ```cmd
    SET "KEY=YOUR_API_KEY"
    SET "FILE_NAME=path/to/your/video.mp4"

    ai-sub ^
      --ai.model=google-gla:gemini-3.1-flash-lite-preview ^
      --ai.google.key="%KEY%" ^
      "%FILE_NAME%"
    ```

    _Note: Replace `YOUR_API_KEY` with your actual key and `"path/to/your/video.mp4"` with the video file path._

---

## Configuration

For a detailed list of all configuration options, including AI models, re-encoding settings, and concurrency controls, please refer to [CONFIGURATION.md](https://github.com/FlippFuzz/ai-sub/blob/main/docs/CONFIGURATION.md).

All settings can be configured via command-line arguments (e.g., `--ai.model=google-gla:gemini-3.0-flash-preview`) or environment variables with the `AISUB_` prefix (e.g., `AISUB_AI_MODEL=gemini-cli:gemini-3.0-flash-preview`).

---

## Release Notes

For the latest updates and bug fixes, please refer to [RELEASE_NOTES.md](https://github.com/FlippFuzz/ai-sub/blob/main/RELEASE_NOTES.md).

---

## Known Limitations

1.  **Timestamp Accuracy:** Subtitle timestamps may occasionally be inaccurate due to limitations of the Gemini AI model. Shorter video segments generally yield better accuracy. Experiment with the `--split.max_minutes` setting.
2.  **AI Hallucinations:** Like all LLMs, Gemini may occasionally produce "hallucinations" or inaccurate information.

If you encounter issues, consider re-processing specific video segments as detailed below.

---

## Advanced: Re-processing Segments

Intermediate files and job states are stored in a temporary directory (default: `tmp_<input_file_name>`). You can customize this location using the `--dir.tmp` flag.

The application creates separate state files for each processing stage (e.g., lyrics detection, subtitle generation). To re-process a specific segment, you must delete the state file for the stage you want to re-run.

File naming format: `part_XXX.<stage>.<model_name>.json`

**Example: To re-run subtitle generation for the third segment:**

1.  Navigate to the temporary directory.
2.  Identify the model name used for subtitles (e.g., `gemini-3-pro-preview`).
3.  Delete the corresponding state file, for example: `part_002.subtitles.gemini-3-pro-preview.json`.
4.  Re-run the script. It will detect the missing subtitle job state and re-process only that segment, using any existing lyrics data.

To re-run the entire pipeline for that segment (including lyrics search), delete both the `lyrics` and `subtitles` JSON files for that part.
