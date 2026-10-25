# AI Sub: AI-Powered Subtitle Generation with Translation

[![PyPI version](https://img.shields.io/pypi/v/ai-sub)](https://pypi.org/project/ai-sub)
[![Downloads](https://img.shields.io/pypi/dw/ai-sub)](https://pypistats.org/packages/ai-sub)

**AI Sub** uses Google's Gemini AI (via the Interactions API) to automatically transcribe and translate videos into audio-synchronized, bilingual English subtitles.

It handles fast Japanese speech, livestream banter, video game terms, and live singing performances with high timing accuracy.

---

## Table of Contents

- [How It Works (In Simple Terms)](#how-it-works-in-simple-terms)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Option 1: Recommended Setup (Gemini + Tinyfish)](#option-1-recommended-setup-gemini--tinyfish)
  - [Option 2: Free Search Setup (DuckDuckGo)](#option-2-free-search-setup-duckduckgo)
  - [Option 3: Multi-Tool Search Setup](#option-3-multi-tool-search-setup)
  - [Option 4: Providing Custom Notes & Lore](#option-4-providing-custom-notes--lore)
  - [Option 5: Speed Mode (Skipping Steps)](#option-5-speed-mode-skipping-steps)
- [Intermediate Files & Resuming](#intermediate-files--resuming)
- [Configuration](#configuration)

---

## How It Works (In Simple Terms)

AI Sub breaks down video subtitling into a clean, human-like workflow:

```text
[Your Video File]
       │
       ▼
 1. The Big Picture (Summary)
    The AI takes a quick bird's-eye view of the entire video to learn who is in it,
    what they are doing (chatting, gaming, singing), and any inside jokes or names.
    (If this stage fails, the pipeline halts immediately to save quota).
       │
       ▼
 2. Divide into 5-Minute Chunks
    Long videos are split into 5-minute pieces so processing is fast and manageable.
       │
       ▼
 3. Listen & Transcribe (Subtitles)
    The AI listens directly to the audio track down to the millisecond to catch
    exactly when voices speak or sing. It also watches for on-screen signs or text,
    writing dialogue and natural English translations.
       │
       ▼
 4. Music & Lyrics Polish (Lyrics — Only when someone sings!)
    - If the segment was just talking or gaming: This step is skipped completely!
    - If someone sang a song: The AI looks up official lyrics in LRCLIB and web search
      to ensure tricky lyrics, puns, and Kanji are spelled and translated perfectly.
    - If lyrics lookup fails: An error is explicitly logged for that part in the final
      subtitles so you know it requires attention.
       │
       ▼
 5. Stitched Subtitle File (.srt)
    All pieces are assembled into a single subtitle file ready to drag-and-drop into
    VLC, MPV, or YouTube.
```

---

## Prerequisites

- **Python 3.11** or higher installed on your computer.
- A **Google Gemini API Key** ([Get one here from Google AI Studio](https://aistudio.google.com/)).
- (Recommended) A **Tinyfish API Key** for fast, accurate web search and lyric lookups ([Get one here](https://tinyfish.ai/)).

---

## Installation

### Linux & macOS

1. Open your terminal and create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install AI Sub:
   ```bash
   pip install --upgrade ai-sub
   ```

### Windows

1. Open **Command Prompt** or **PowerShell** and create a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install AI Sub:
   ```powershell
   pip install --upgrade ai-sub
   ```

---

## Usage

### Option 1: Recommended Setup (Gemini + Tinyfish)

This configuration gives the best results for song lyrics, VTuber lore, video game jargon, and character names.

#### On Linux / macOS:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export TINYFISH_API_KEY="your_tinyfish_api_key_here"

ai-sub --ai.search.tools=tinyfish,lrclib "path/to/video.mp4"
```

#### On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:TINYFISH_API_KEY="your_tinyfish_api_key_here"

ai-sub --ai.search.tools=tinyfish,lrclib "path\to\video.mp4"
```

#### On Windows (Command Prompt):

```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
set TINYFISH_API_KEY=your_tinyfish_api_key_here

ai-sub --ai.search.tools=tinyfish,lrclib "path\to\video.mp4"
```

---

### Option 2: Free Search Setup (DuckDuckGo)

If you do not have a Tinyfish key, you can use DuckDuckGo search for free with no extra API key:

#### On Linux / macOS:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"

ai-sub --ai.search.tools=duckduckgo,lrclib "path/to/video.mp4"
```

#### On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"

ai-sub --ai.search.tools=duckduckgo,lrclib "path\to\video.mp4"
```

---

### Option 3: Multi-Tool Search Setup

You can provide multiple search providers simultaneously. The AI agent will select between them:

```bash
ai-sub --ai.search.tools=tinyfish,ollama,lrclib "path/to/video.mp4"
```

---

### Option 4: Providing Custom Notes & Lore

You can feed custom notes to the AI (such as character names, spelling preferences, or specific terms) using `--context` or a text file:

```bash
# Inline notes
ai-sub --context "The streamers are Fuwawa and Mococo. Keep 'Bau bau' untranslated." "video.mp4"

# Notes from a text file
ai-sub --context-file "notes.txt" "video.mp4"
```

---

### Option 5: Speed Mode (Skipping Steps)

If you are subtitling a pure talking/gaming stream with no singing, or if you want to skip the whole-video preview to save time:

```bash
# Skip the whole-video summary preview
ai-sub --concurrency.summary=0 "video.mp4"

# Skip lyric searches and music refinement
ai-sub --concurrency.lyrics=0 "video.mp4"

# Fast dialogue-only mode (skip both preview and lyrics refinement)
ai-sub --concurrency.summary=0 --concurrency.lyrics=0 "video.mp4"
```

---

## Intermediate Files & Resuming

AI Sub automatically saves its work in a temporary folder (`tmp_<video_name>`).

If your internet drops, your computer shuts down, or an API error occurs, **you do not lose your progress**. Simply run the exact same command again:

- AI Sub will automatically pick up where it left off.
- It will reuse all completed chunks and only finish the missing pieces.

### Comparing Intermediate Subtitles in VLC

Inside the temporary directory (`tmp_<video_name>`), AI Sub generates standalone subtitle files for each video part with complete upstream lineage:

- `part_000.subtitles-g35l-01_g38f-19.srt`: Initial dialogue transcription and timing (linked to the summary model).
- `part_000.lyrics-g35l-01_g38f-19_g35l-10.srt`: Song lyrics and translation refinement (linked to both summary and subtitle models).

When opened in VLC or MPV alongside the video chunk, the subtitle tracks are labeled with their full lineage for easy comparison.

Once the entire process finishes, the final stitched subtitle file is saved alongside your video with a clean, concise filename:

```text
my_video.mp4
my_video.g38f-011019.srt  <-- Drag this into your video player!
```

---

## Configuration

For advanced settings (such as custom frame rates, custom models, rate limits, per-stage agentic video toggles, and storage cleanup rules), refer to [CONFIGURATION.md](docs/CONFIGURATION.md).
