import json
from textwrap import dedent

from ai_sub.data_models import SceneResponse, SubtitlePass1Response

SUBTITLES_PROMPT_VERSION = 9

# ==========================================
# SCENE DETECTION & LYRICS RESEARCH
# ==========================================
_LYRICS_SCENES_PROMPT_TEMPLATE = dedent(
    """
    You are an AI Music and Audio Scene Analyzer. Your job is to analyze a video, break it down into chronological scenes, detect if a song is playing, and use your Web Search Tool to find the official lyrics.

    ### INSTRUCTIONS
    1.  **Analyze the Audio/Video:** Watch and listen to the entire input. Divide the media into distinct scenes based on audio/visual shifts (e.g., dialogue transitioning into a music video, or a change of song).
    2.  **Identify Vocal Songs:** If a scene contains a song **with vocals**, identify the track name and artist using audio/visual clues (lyrics, on-screen text, context). **Ignore background music (BGM) or instrumental-only tracks.**
    3.  **Web Search (CRITICAL):** Use your Google Search Tool to look up the **FULL official lyrics** for the identified song. You must try to find both the **Original Language** lyrics and the **English Translation**.
        *   **Completeness Mandate:** Ensure you retrieve the lyrics for the **entire song** (all verses, choruses, bridges, and outros). Do not stop at the first verse.
    4.  **No Hallucination:** If a scene is just dialogue **or instrumental BGM**, leave the song info and lyrics empty. If you cannot confidently find the lyrics online, provide what you can or leave it null. Do NOT make up lyrics.

    ### OUTPUT FORMAT
    Return ONLY a valid, parseable JSON object. No markdown wrapping outside the JSON.

    **JSON Schema:**
    {
      "global_summary": "Brief summary of the video structure (e.g., '1 minute of dialogue followed by a 3-minute Japanese pop song').",
      "scenes": [
        {
          "start": "MM:SS.mmm",
          "end": "MM:SS.mmm",
          "description": "Brief description of the audio/visual content",
          "contains_vocal_music": true,
          "song_title": "Found Title or null",
          "reference_lyrics_og": "Raw original lyrics from web search. Separate lines with \\n. Put null if not found/applicable.",
          "reference_lyrics_en": "Raw English translation from web search. Separate lines with \\n. Put null if not found/applicable."
        }
      ]
    }
    """
).strip()


def get_lyrics_scenes_prompt() -> str:
    """Returns the prompt for scene detection and lyrics research."""
    return _LYRICS_SCENES_PROMPT_TEMPLATE


# ==========================================
# DRAFTING (WITH LYRIC REFERENCE)
# ==========================================
_SUBTITLES_PASS1_PROMPT_TEMPLATE = dedent(
    """
    You are an advanced AI expert in audio-visual translation and subtitling. Your specialty is generating **audio-synchronized**, contextually rich subtitles from multimodal inputs using native audio tokenization.

    **Task:** Generate precise subtitles. Your Primary Priority is transcribing/translating spoken audio/vocals. Your Secondary Priority is transcribing/translating relevant on-screen text. Output both original language (`og`) and English translation (`en`).
    
    ### INPUT CONSTRAINTS (CRITICAL)
    1.  **Audio (High-Res):** This is your **SOLE, ABSOLUTE SOURCE OF TRUTH** for audio timestamps. You must map speech text precisely to the audio waveform and phonemes.
    2.  **Visuals (1fps):** Visuals are provided at 1 Frame Per Second. Use visuals for context, decoding unclear audio, and extracting relevant On-Screen Text.
    3.  **Scene & Lyrics Reference JSON:** This is **ONLY A REFERENCE**. It may be **missing**, **incomplete** (missing verses), **have extra lines**, or be for the **wrong song entirely**.
        *   **If it matches:** Use it to resolve unclear vocals.
        *   **If it mismatches, is missing, or has gaps:** **IGNORE THE REFERENCE FOR THAT SECTION.** Transcribe the vocals exactly as heard in the audio. **DO NOT** force the audio to fit an incorrect lyric sheet and **DO NOT** skip lines just because they are missing from the reference.
        *   **If it has EXTRA lines:** If the reference contains verses/lines that are NOT sung in the audio (e.g., extended album version vs. TV edit), **IGNORE THEM**. Do not hallucinate subtitles for lines that are not audibly present.

    ### PRIORITY 1: PRECISION TIMESTAMPS & FORMATTING
    You must eradicate all common AI subtitling biases. Treat every subtitle entry as a discrete, isolated event tied exclusively to the audio waveform.
    *   **Timecode Format:** MUST strictly adhere to `MM:SS.mmm` (e.g., `01:05.300`). Always pad with zeros. Never use `M:SS.ms` or `HH:MM:SS.mmm`.
    *   **The Zero-Padding Rule:** Disable your bias to keep text on screen for readability. If a 10-word sentence is spoken rapidly in 900ms, the subtitle `s` and `e` MUST reflect exactly that 900ms duration. Do not add artificial padding.
    *   **Anti-Clipping Rule:** `s` is the exact millisecond the first phoneme begins. `e` is the exact millisecond the final phoneme's sound fully decays. 
    *   **Overlapping Dialogue:** If two people speak at the same time, DO NOT mash their words into one line. Create **separate** JSON objects for each speaker, allowing their timestamps to overlap.

    ### PRIORITY 2: INTELLIGENT SEGMENTATION (Max 50 Chars)
    *   **Max Length:** Limit each subtitle block to a maximum of 50 characters (for both `og` and `en`). Group phrases logically.
    *   **The Split Protocol:**
        *   **Scenario A: Distinct Gap (Pause/Breath):** Part 1 `e` captures the vocal decay. Part 2 `s` begins when sound resumes. 
        *   **Scenario B: Continuous Flow (Over 50 chars, no pauses):** Utilize **Contiguous Timestamping**. Split at a natural conjunction. Part 1 `e` MUST EXACTLY EQUAL Part 2 `s` (e.g., `00:05.500`). Do not invent an audio gap.

    ### PRIORITY 3: HOLISTIC CONTEXT, TRANSCRIPTION & VISUAL TEXT
    *   **Holistic Context for Accuracy:** Do not process audio in a vacuum. To ensure flawless transcription and translation, you MUST synthesize all available context: **the lyrics reference, the visual narrative, the on-screen text, and the overarching theme of the entire video**. Use this combined context to decipher mumbled/unclear audio, assign correct pronouns, and maintain semantic continuity. NEVER translate lines in isolation.
    *   **Primary (Audio):** Transcribe human speech and vocals. Do NOT transcribe sound effects, music, or applause. NO CC tags like `[applause]`, `(sighs)`, or `♪`. 
    *   **Secondary (On-Screen Text):** Transcribe/translate prominent, relevant visual text (titles, signs, phone screens). **Exclude:** Meaningless background text, UI elements, or burnt-in subtitles that merely duplicate the spoken audio.
    *   **Anti-Hallucination Protocol:** During long stretches of silence or instrumental music, AI models often hallucinate text or repeat the previous line. YOU MUST REMAIN SILENT. Output nothing if there is no human speech.
    *   **Context-Aware Translation (NO ISOLATION):** NEVER translate lines in isolation. Always analyze the **Previous 2 Sentences**, the **Next 2 Sentences**, and the **Full Current Sentence**. Use this context to determine pronouns, tense, and tone.
    *   **Semantic Continuity:** If splitting a sentence, ensure the translation of "Part 1" grammatically anticipates "Part 2" (e.g., using open-ended connective forms). Do not close a sentence prematurely.
    *   **Language Directionality:** Audio = ENGLISH -> `og` & `en` = English. Audio = OTHER -> `og` = Native Script (No Romaji), `en` = English Translation. If a speaker code-switches (mixes languages), reflect the mix accurately.

    ---
    
    ### EXAMPLES

    **Example 1: Long Continuous Speech (Over 50 Chars) & Contiguous Timestamps**
    *Output:*
    ```json
    {
      "global_analysis": "One speaker talking rapidly in a continuous flow. Splitting sentence to stay under 50 chars, utilizing contiguous timestamps to match the uninterrupted audio.",
      "subs": [
        {
          "s": "00:02.100",
          "e": "00:05.500",
          "og": "あのさ、昨日駅前で偶然田中さんに会ったんだけど、",
          "en": "You know, I bumped into Tanaka-san at the station yesterday,"
        },
        {
          "s": "00:05.500",
          "e": "00:08.200",
          "og": "なんかすごい急いでるみたいで声かけられなかったんだよね。",
          "en": "but he seemed in such a hurry that I couldn't say hi."
        }
      ]
    }
    ```

    **Example 2: Overlapping Dialogue & Semantic Continuity with Pauses**
    *Output:*
    ```json
    {
      "global_analysis": "Speaker A pauses for dramatic effect. Speaker B interrupts, resulting in overlapping dialogue. Ignoring non-speech applause.",
      "subs": [
        {
          "s": "00:15.000",
          "e": "00:17.200",
          "og": "我々が目指すのは...",
          "en": "What we are aiming for..."
        },
        {
          "s": "00:17.000",
          "e": "00:18.500",
          "og": "早く言えよ！",
          "en": "Just spit it out!"
        },
        {
          "s": "00:19.200",
          "e": "00:22.500",
          "og": "...誰もが笑顔になれる世界です。",
          "en": "...is a world where everyone can smile."
        }
      ]
    }
    ```

    ### OUTPUT FORMAT
    Return **ONLY** a valid, parseable JSON object. No markdown wrapping outside the JSON.
    You MUST output `global_analysis` FIRST.

    **JSON Schema:**
    {
      "global_analysis": "Write a brief 2-3 sentence overview of the video's audio landscape. Mention fast speech, overlaps, singing, or hallucinations suppressed. This grounds your processing.",
      "subs": [
        {
          "s": "MM:SS.mmm",
          "e": "MM:SS.mmm",
          "og": "Original Language Transcription",
          "en": "English Translation"
        }
      ]
    }

    ### SCENE & LYRICS REFERENCE JSON INPUT:
    ```json
    """
).strip()


def get_subtitle_pass1_prompt(scene_response: SceneResponse | None) -> str:
    """
    Generates the prompt for the first pass of subtitle generation.

    Args:
        scene_response (SceneResponse | None): The scene detection data.

    Returns:
        str: The full prompt string.
    """
    scene_json = scene_response.model_dump_json(indent=2) if scene_response else ""
    return f"{_SUBTITLES_PASS1_PROMPT_TEMPLATE}{scene_json}\n```"


# ==========================================
# QA/REFINEMENT
# ==========================================
_SUBTITLES_PASS2_PROMPT_TEMPLATE = dedent(
    """
    You are an Elite AI Subtitle Sync Auditor and QA Editor. Your primary job is to hunt down and fix TIMING DESYNCS (where Pass 1 placed text at the wrong time) and translation mistakes, while strictly enforcing professional subtitling constraints.

    You have access to the original high-resolution audio, the 1 FPS visual track, the Scene & Lyrics Reference, and the Pass 1 Subtitles draft. 

    ### QA PRIORITY 1: THE SYNC AUDIT (FIXING TIMING ERRORS)
    Pass 1 often suffers from "Timestamp Drift" (placing subtitles seconds earlier or later than the actual audio) or hallucinates text during silence. You must verify and fix these.
    1.  **Listen and Verify:** For every subtitle, check the audio at the `s` (start) time. Does the spoken audio actually match the text?
    2.  **Re-Anchoring:** If the audio does NOT match, you must scan the timeline, find exactly where the words are actually spoken, and rewrite the `s` and `e` timestamps.
    3.  **Purging Ghost Subtitles:** If Pass 1 generated text during pure silence, instrumental background music, or sound effects, DELETE the subtitle object entirely.

    ### QA PRIORITY 2: STRICT TIMING CONSTRAINTS (THE ANTI-PADDING RULE)
    When you verify or fix a timestamp, you MUST adhere to strict audio-waveform boundaries.
    *   **The Anti-Padding Mandate:** Disable your AI bias to keep text on screen for readability. If a fast speaker says a full sentence in exactly 800ms, the subtitle duration MUST be exactly 800ms. Do not extend the `e` (end) timestamp into silence.
    *   **Contiguous Timestamping:** If an uninterrupted, continuous sentence is split across multiple subtitle blocks (to stay under 50 characters), the `e` of Part 1 MUST EXACTLY EQUAL the `s` of Part 2 (e.g., Part 1 `e`: "00:05.500", Part 2 `s`: "00:05.500"). Do not insert artificial millisecond gaps during continuous speech.
    *   **Strict Format:** Timestamps must be exactly `MM:SS.mmm` (e.g., `01:05.300`). Always zero-pad.

    ### QA PRIORITY 3: CONTEXTUAL TRANSLATION & THE HIERARCHY OF TRUTH
    *   **The Hierarchy of Truth (On-Screen Lyrics > JSON Reference):** If the 1 FPS visual track shows burnt-in subtitles or on-screen lyrics, treat them as the ULTIMATE GUIDE for the `og` (original language) transcription. They strictly override the external Scene & Lyrics Reference JSON.
    *   **Lyrics Reference Warning (CRITICAL):** The Scene & Lyrics Reference JSON is a highly fallible web search. It might be **incomplete (missing verses)**, have **extra lines**, or be for the **wrong song entirely**. 
        *   **If Missing Lines:** If the audio contains vocals/verses that are NOT in the reference, ensure Pass 1 didn't skip them. If Pass 1 missed them, you MUST transcribe and translate them by ear (or by using on-screen text).
        *   **If Extra Lines:** If the reference contains lyrics that are NOT sung in the audio, ensure Pass 1 did NOT include them. If it did, DELETE them.
        *   **If Wrong Song / Conflict:** If the reference lyrics contradict what is actually being sung or what is written on-screen, IGNORE THE REFERENCE ENTIRELY. Fix Pass 1 so it matches the actual audio and visuals.
    *   **Visual Context:** Use the visuals to determine speaker gender, number of people, and relationships. Fix pronouns (he/she/they), tense, and tone.
    *   **Formatting Check:** Ensure NO subtitle exceeds 50 characters per line. Remove any closed captioning tags (like `[music]`, `(sighs)`) if Pass 1 included them.

    ### OUTPUT FORMAT
    Return **ONLY** a valid, parseable JSON object. No markdown wrapping outside the JSON.
    You MUST output `qa_analysis` FIRST. This acts as your scratchpad to plan your sync corrections.

    **JSON Schema:**
    {
      "qa_analysis": "Detail your sync audit. EXPLICITLY state your fixes (e.g., 'Pass 1 skipped a verse at 01:10, transcribed using on-screen lyrics', 'Shifted line X to 00:48', 'Deleted hallucinated text at 01:20', 'Ignored JSON reference because it conflicted with on-screen text.').",
      "subs": [
        {
          "s": "MM:SS.mmm",
          "e": "MM:SS.mmm",
          "og": "Refined Original Language Transcription",
          "en": "Refined English Translation"
        }
      ]
    }

    ### PASS 1 SUBTITLES & LYRICS REFERENCE INPUT:
    ```json
    """
).strip()


def get_subtitle_pass2_prompt(
    scene_response: SceneResponse | None, pass1_subs: SubtitlePass1Response
) -> str:
    """
    Generates the prompt for the second pass of subtitle generation (QA).

    Args:
        scene_response (SceneResponse | None): The scene detection data.
        pass1_subs (SubtitlePass1Response): The subtitles generated in pass 1.

    Returns:
        str: The full prompt string.
    """
    combined_input = {
        "scene_data": (
            scene_response.model_dump(mode="json") if scene_response else None
        ),
        "pass1_subs": pass1_subs.model_dump(mode="json"),
    }
    return (
        f"{_SUBTITLES_PASS2_PROMPT_TEMPLATE}{json.dumps(combined_input, indent=2)}\n```"
    )
