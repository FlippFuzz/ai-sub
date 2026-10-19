# AI Sub: AI-Powered Subtitle Generation with Translation

[![PyPI version](https://img.shields.io/pypi/v/ai-sub)](https://pypi.org/project/ai-sub)
[![Downloads](https://img.shields.io/pypi/dw/ai-sub)](https://pypistats.org/packages/ai-sub)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Intermediate Files & Re-processing](#intermediate-files--re-processing)

---

## Overview

**AI Sub** is a high-performance command-line tool that leverages Google's **Gemini** multimodal models to generate audio-synchronized, context-aware subtitles with English translations.

It uses native audio perception as the absolute ground truth for dialogue timing while utilizing visual cues (on-screen text, signs, character actions) to disambiguate speech, pronouns, and tone. For musical performances, it integrates automated web search tools to locate official song lyrics and translations.

---

## How It Works

```
[Input Video]
      │
      ▼
1. Segmentation (5-minute chunks via FFmpeg copy)
      │
      ▼
2. Optional Compression (1 FPS, 360p video to save upload bandwidth)
      │
      ▼
3. Gemini Files API Management (automatic deduplication & watermark cleanup)
      │
      ▼
4. Optional Lyrics & Scene Mapping (web-searches official song lyrics)
      │
      ▼
5. Subtitle Generation (audio ground-truth timestamps + context-aware translation)
      │
      ▼
6. Assembly (stitches segment subtitles into a unified SRT file)
```

1. **Segmentation:** Splits the input video into 5-minute chunks using stream copying (no re-encoding delay).
2. **Re-encoding (Optional):** Compresses segments (e.g., 1 FPS, 360p) to minimize cloud upload bandwidth without compromising audio perception.
3. **Storage Management:** Checks Google Files API storage against configured high-water/low-water mark thresholds and uploads segments with automatic deduplication.
4. **Scene & Lyrics Detection (Optional):** Analyzes video scenes and uses external web search (DuckDuckGo, Langsearch, or Ollama) to locate official song lyrics.
5. **Subtitle Generation:** Transcribes speech synchronized to the millisecond, paired with semantic English translations.
6. **Assembly:** Stitches all segment subtitles together with offset compensation into `<video_name>.<model_shortcode>.srt`.

---

## Installation

**Prerequisites:** Python 3.11 or higher.

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install AI Sub:**

   ```bash
   pip install --upgrade ai-sub
   ```

---

## Usage

### Option 1 (Recommended): Google API Key + Langsearch

This configuration offers the highest search accuracy for music and lyrics lookups:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"
export LANGSEARCH_API_KEY="YOUR_LANGSEARCH_KEY"

ai-sub \
  --ai.model-lyrics=google-gla:gemini-3.5-flash-lite \
  --ai.model-subtitles=google-gla:gemini-3.7-flash \
  --ai.search.tool=langsearch \
  "path/to/video.mp4"
```

### Option 2: Google API Key + DuckDuckGo (Free Search)

DuckDuckGo requires no additional API key:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"

ai-sub \
  --ai.model-lyrics=google-gla:gemini-3.5-flash-lite \
  --ai.model-subtitles=google-gla:gemini-3.7-flash \
  --ai.search.tool=duckduckgo \
  "path/to/video.mp4"
```

### Option 3: Transcription Only (No Lyrics/Scene Detection)

Set `--concurrency.lyrics 0` to skip lyrics lookups and run pure dialogue transcription directly:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"

ai-sub \
  --ai.model-subtitles=google-gla:gemini-3.7-flash \
  --concurrency.lyrics=0 \
  "path/to/video.mp4"
```

---

## Configuration

For a complete reference of all command-line arguments and environment variables, refer to [CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Intermediate Files & Re-processing

Intermediate files and job states are stored in a temporary directory (default: `tmp_<video_stem>`).

The application persists state for each stage in human-readable YAML files:

- Lyrics/Scene state: `part_XXX.lyrics.<lyrics_shortcode>.yaml`
- Subtitle state: `part_XXX.subtitles.<subtitles_shortcode>.yaml`

### Re-running a Specific Segment

To re-process a specific segment:

1. Navigate to the `tmp_<video_stem>` directory.
2. Delete the YAML file for the stage you wish to re-run (e.g., delete `part_002.subtitles.g37f-0918.yaml`).
3. Re-run `ai-sub`. It will detect the missing state file and re-process only that segment, reusing any existing lyrics data.
