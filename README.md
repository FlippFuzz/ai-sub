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
  - [Option 3: Providing Custom Notes & Lore](#option-3-providing-custom-notes--lore)
- [Intermediate Files & Resuming](#intermediate-files--resuming)
- [Distributed Server & Worker Deployment](#distributed-server--worker-deployment)
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
       │
       ▼
 2. Divide into 5-Minute Chunks
    Long videos are split into 5-minute pieces to prevent the AI from losing track due
    to context bloat.
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
       │
       ▼
 5. Stitched Subtitle File (.srt)
    All pieces are assembled into a single subtitle file ready to drag-and-drop into
    VLC, MPV, or Substital.
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

1. Open **PowerShell** and create a virtual environment:

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

### Option 3: Providing Custom Notes & Lore

You can feed custom notes to the AI (such as character names, spelling preferences, or specific terms) using `--context` or a text file:

```bash
# Inline notes
ai-sub --context "The streamers are Fuwawa and Mococo. Keep 'Bau bau' untranslated." "video.mp4"

# Notes from a text file
ai-sub --context-file "notes.txt" "video.mp4"
```

---

## Intermediate Files & Forcing Regeneration

Inside the temporary directory (`tmp_<video_name>`), AI Sub generates standalone state files (`.yaml`) and intermediate subtitles (`.srt`) tracking the progress of each video chunk and global summary:

- `summary.<summary_shortcode>.yaml`: Caches the global video summary and metadata (where `<summary_shortcode>` combines the summary model shortcode and prompt version, e.g., `g35l-01`).
- `<part_name>.subtitles.<subtitles_shortcode>.yaml` & `.srt`: Caches raw segment dialogue transcription and timing (where `<subtitles_shortcode>` incorporates summary lineage and subtitle model shortcode/version, e.g., `g35l-01_g38f-19`).
- `<part_name>.lyrics.<lyrics_shortcode>.yaml` & `.srt`: Caches refined song lyrics and translations when musical performances are detected (where `<lyrics_shortcode>` incorporates subtitle lineage and lyrics model shortcode/version, e.g., `g35l-01_g38f-19_g35l-10`).

Delete the corresponding `.yaml` state files in the temporary directory to force a regeneration of a specific summary, chunk transcription, or lyrics pass.

---

## Distributed Server & Worker Deployment

For 24/7 automated video ingestion (e.g. watch folders or YouTube feeds) and distributed multi-node task processing:

- **Central Server (`ai-sub-server`)**: Coordinates jobs, task leases, ingestion, and Git synchronization.
- **Workers (`ai-sub-worker`)**: Ephemeral worker nodes that lease chunks, execute multimodal Gemini inference, and return results.

For complete setup instructions (systemd services, Nginx reverse proxy, Basic Auth, and Fail2ban security), see [`docs/SERVER_DEPLOYMENT.md`](docs/SERVER_DEPLOYMENT.md:1).

---

## Configuration

For advanced settings (such as custom frame rates, custom models, rate limits, per-stage agentic video toggles, and storage cleanup rules), refer to [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md:1).
