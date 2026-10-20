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

It operates in **3 distinct phases**:

1. **Global Video Intelligence:** Analyzes the full video using Gemini's **Agentic Video Understanding** to build an overarching character roster, tracklist, and domain glossary.
2. **Segment Music & Scene Analysis:** Detects scenes per chunk, validates web-searched song lyrics against the audio waveform, identifies on-screen lyrics, and extracts only the lines performed in each segment.
3. **Synchronized Subtitle Generation:** Transcribes speech synchronized to the millisecond using audio waveforms as absolute ground truth, translating natural English dialogue guided by the global context and user instructions.

---

## How It Works

```
[Input Video] + [Optional User Context]
       │
       ▼
 1. Phase 1: Whole-Video Summary & Knowledge Base
    (Agentic Video Understanding + Web Search for lore, cast, & tracklist)
       │
       ▼
 2. Segmentation (Uniform 5-minute chunks via FFmpeg copy)
       │
       ▼
 3. Optional Compression (1 FPS, 360p chunks to save upload bandwidth)
       │
       ▼
 4. Phase 2: Lyrics & Scene Detection
    (Identifies on-screen lyrics, searches web for singing without screen text,
     validates candidate lyrics against audio chorus, extracts segment-only lines)
       │
       ▼
 5. Phase 3: Subtitle Generation
    (Audio ground-truth timestamps + context-aware translation via Gemini 3.8 Flash)
       │
       ▼
 6. Assembly (Stitches segment subtitles into a unified SRT file)
```

1. **Phase 1 (Global Summary):** Re-encodes a lightweight whole-video proxy (if ≥ 300 MB) and uses Gemini Agentic Video Understanding to map the entire video without massive token overhead.
2. **Segmentation:** Splits the input video into 5-minute chunks using stream copying (no re-encoding delay).
3. **Segment Re-encoding (Optional):** Compresses segments (e.g., 1 FPS, 360p) if bandwidth conservation is desired.
4. **Phase 2 (Lyrics & Scenes):** Maps scene bounds, flags whether lyrics are burnt into the video, retrieves missing lyrics via web search, checks candidate lyrics against audible phonetics, and returns only the lines performed in that segment.
5. **Phase 3 (Subtitles):** Uses Google Gemini 3.8 Flash to align dialogue to the millisecond, resolving pronoun ambiguity and homophones using the Phase 1 glossary.
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

This configuration offers the highest search accuracy for music, character lore, and lyrics lookups:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"
export LANGSEARCH_API_KEY="YOUR_LANGSEARCH_KEY"

ai-sub \
  --ai.search.tool=langsearch \
  "path/to/video.mp4"
```

### Option 2: DuckDuckGo (Free Search)

DuckDuckGo requires no additional API key:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_KEY"

ai-sub \
  --ai.search.tool=duckduckgo \
  "path/to/video.mp4"
```

### Option 3: Providing User Background Context

You can provide inline background context or point to a context file to guide speaker names, spelling, and translation tone:

```bash
# Inline context
ai-sub \
  --context "Streamers are Fuwawa and Mococo (Fuwamoco). Translate 'bau bau' as untranslated catchphrase." \
  "path/to/video.mp4"

# Context from a text file
ai-sub \
  --context-file "path/to/notes.txt" \
  "path/to/video.mp4"
```

### Option 4: Skipping Summary or Lyrics Phases

To skip Phase 1 (Whole-video summary) or Phase 2 (Lyrics search) for fast pure-dialogue transcription:

```bash
# Skip Phase 1 (Global summary)
ai-sub --concurrency.summary=0 "path/to/video.mp4"

# Skip Phase 2 (Lyrics & scene mapping)
ai-sub --concurrency.lyrics=0 "path/to/video.mp4"

# Dialogue transcription only (Skip both Phase 1 and Phase 2)
ai-sub --concurrency.summary=0 --concurrency.lyrics=0 "path/to/video.mp4"
```

---

## Configuration

For a complete reference of all command-line arguments and environment variables, refer to [CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Intermediate Files & Re-processing

Intermediate files and job states are stored in a temporary directory (default: `tmp_<video_stem>`).

The application persists state for each stage in human-readable YAML files:

- **Phase 1 (Global Summary):** `summary.<summary_shortcode>.yaml` (e.g., `summary.g35l-01.yaml`)
- **Phase 2 (Lyrics & Scenes):** `part_XXX.lyrics.<lyrics_shortcode>.yaml` (e.g., `part_002.lyrics.g35l-10.yaml`)
- **Phase 3 (Subtitles):** `part_XXX.subtitles.<subtitles_shortcode>.yaml` (e.g., `part_002.subtitles.g38f-011019.yaml`)

### Re-running a Specific Stage or Segment

The pipeline is fully resumable. To re-process a specific part:

1. Open the `tmp_<video_stem>` directory.
2. Delete the state YAML file you wish to re-run:
   - To re-run Phase 1 for the whole video: delete `summary.*.yaml`.
   - To re-run Phase 2 for chunk 2: delete `part_002.lyrics.*.yaml`.
   - To re-run Phase 3 subtitles for chunk 2: delete `part_002.subtitles.*.yaml`.
3. Re-run `ai-sub`. It will detect the missing YAML state file and execute only the invalidated tasks while reusing all existing intermediate files.
