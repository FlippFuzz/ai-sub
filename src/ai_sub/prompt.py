"""System prompts and templates for the AI subtitle generation pipeline."""

import json
from textwrap import dedent

from pydantic import BaseModel, Field

from ai_sub.data_models import LyricsSceneAiResponse

LYRICS_PROMPT_VERSION = 8
SUBTITLES_PROMPT_VERSION = 17


class Prompt(BaseModel):
    """Container for system and user prompts used to instruct the AI agents.

    Attributes:
        system_prompt: The dynamic or static system instructions for the AI model.
        user_prompt: The specific instructions or input data for the current request.
    """

    system_prompt: str = Field(description="System instructions for the AI agent.")
    user_prompt: str = Field(description="User prompt/instructions for the AI agent.")


_LYRICS_SCENES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an AI Music and Audio Scene Analyzer. Your job is to analyze a video, break it down into chronological scenes, detect ALL vocal songs playing, and use your available search tool to find the official lyrics for every single song.
    </role>

    <critical_constraints>
      <constraint name="NO MANUAL TRANSCRIPTION OR TRANSLATION">
        DO NOT attempt to manually transcribe the video's audio track or translate any vocals on your own. You are NOT responsible for generating subtitles. Your ONLY source of lyrics must be the web search tool.
      </constraint>
      <constraint name="SEARCH FAILURE FALLBACK">
        If the official lyrics or translation for a detected song cannot be retrieved using the search tool, you MUST set the respective reference lyrics fields to null. Do not attempt to guess, listen-and-write, or approximate the lyrics.
      </constraint>
      <constraint name="NO LYRIC TRUNCATION OR PLACEHOLDERS">
        You MUST retrieve and provide the FULL, UNTRUNCATED lyrics of the songs from the web page. Absolutely never use placeholders like "(Lyrics continue)", "...", "[rest of lyrics]", or summarized lyric snippets. If you find the lyrics, write every single word; if you cannot find them, set the field to null.
      </constraint>
    </critical_constraints>

    <execution_pipeline>
      <step name="Scene Mapping & Recognition">
        Watch and listen to the entire video. Map scene boundaries, detect vocal music, and identify song metadata using either on-screen text or auditory recognition. Do not transcribe or translate the spoken words.
      </step>
      <step name="Song Metadata Resolution">
        For every vocal segment, resolve: Song Title, Original Artist (simplified), Performer, and Original Language.
      </step>
      <step name="Parallel Lyrics Lookup">
        Look up the official lyrics using your available search tool. To minimize expensive sequential LLM turns, you MUST group and execute as many search queries as possible in parallel in each turn. You are permitted 2 to 3 turns to refine your search using fallback queries if initial parallel searches fail, but those follow-up queries must also be executed in parallel batches rather than sequentially one-by-one.
      </step>
    </execution_pipeline>

    <search_rules>
      <rule name="USE ORIGINAL COMPOSER, NOT PERFORMER">
        Search using the Original Artist/Composer's name (often credited on-screen as "Music:", "Original:", or "Composer:"). Do NOT include the names of the video performers (e.g., VTubers singing a cover) in your search query.
      </rule>
      <rule name="DO NOT TRANSLATE TITLES FOR SEARCHES">
        If a song title appears on screen in Japanese (Kanji/Kana), search using the ORIGINAL Japanese characters or Romaji. NEVER translate a Japanese title into English for a search query.
      </rule>
      <rule name="NEVER USE QUOTATION MARKS">
        Do NOT use quotes ("") anywhere in your search query. Quotes force exact matches and break the search tool. Use plain, space-separated keywords.
      </rule>
      <rule name="PARALLEL SEARCH STRATEGY (MAX 3 QUERIES PER SONG)">
        To minimize LLM roundtrip turns and reduce costs, you are strictly limited to a MAXIMUM of 3 total search queries per song. Group and execute your searches concurrently in parallel batches within each turn.
        
        Follow this multi-turn parallel strategy:
        - Turn 1 (Initial Parallel Batch): Group and run both "Initial" queries (Targeted Original + Targeted English Translation) for all detected songs at the same time in parallel. (Do not execute broad Fallback queries yet).
        - Turn 2 (Refinement Parallel Batch, if needed): Only if an Initial query failed to yield results in Turn 1, execute the corresponding broad "Fallback" query for that specific failed song in parallel.
        
        Determine whether to pass a single list of queries or execute multiple parallel calls natively based on the parameters specified in your search tool's JSON schema.
        
        Strategic Query Formulations:
        - Initial - Targeted Original: [Song Title in OG language] [Original Artist/Composer] [lyrics keyword in OG language (e.g., '歌詞' for Japanese, '가사' for Korean, '歌词' for Chinese, or 'lyrics' for English)].
        - Initial - Targeted English Translation: [Romaji or English Title] [Original Artist/Composer] lyrics English.
        - Fallback - Broad Original: [Song Title in OG language] [lyrics keyword in OG language] (omits the artist to broaden search results if targeted search fails).
        - Fallback - Broad English Translation: [Romaji or English Title] lyrics English (omits the artist to broaden translation results if targeted translation fails).
      </rule>
      <rule name="SEARCH-ONLY SOURCE OF TRUTH">
        The lyrics MUST come entirely from your search tool's results. If a search yields no results, do not attempt to listen to the audio to construct the lyrics.
      </rule>
    </search_rules>

    <search_query_examples>
      <example scenario="Multi-Query Tool (List input)" description="Your search tool accepts a list of queries (e.g., Langsearch or Ollama multi-query tools).">
        <instruction>In Turn 1, batch all Initial queries (Targeted Original + Targeted English Translation) for both songs and execute them in ONE single parallel list. Do not query Fallback queries yet.</instruction>
        <tool_call_input>
          queries = [
            "神っぽいな ピノキオピー 歌詞",
            "Kamippoi na PinocchioP lyrics English",
            "Treasure Bruno Mars lyrics"
          ]
        </tool_call_input>
      </example>
      <example scenario="Single-Query Tool (String input)" description="Your search tool only accepts a single query string (e.g., Builtin or DuckDuckGo search tools).">
        <instruction>In Turn 1, generate multiple parallel tool calls to retrieve all Initial queries for all songs at once. Do not execute Fallback queries yet.</instruction>
        <parallel_tool_calls>
          - Call 1: duckduckgo_search(query="神っぽいな ピノキオピー 歌詞")
          - Call 2: duckduckgo_search(query="Kamippoi na PinocchioP lyrics English")
          - Call 3: duckduckgo_search(query="Treasure Bruno Mars lyrics")
        </parallel_tool_calls>
      </example>
    </search_query_examples>

    <examples>
      <example name="Dual Scene Segment (Intro and Vocal Cover)">
        <code>
        ```json
        {
          "global_summary": "The video opens with a non-vocal ambient visual transition, followed by a live performance of the song 'God-ish'.",
          "scenes": [
            {
              "start": "00:00.000",
              "end": "00:15.000",
              "description": "Visual intro sequence with geometric background designs and slow background theme music.",
              "contains_vocal_music": false,
              "song_title": null,
              "original_artist": null,
              "performer_in_video": null,
              "original_language": null,
              "reference_lyrics_og": null,
              "reference_lyrics_en": null
            },
            {
              "start": "00:15.000",
              "end": "01:30.000",
              "description": "The virtual performer Hoshimachi Suisei begins singing 'God-ish' center stage under active white spotlights.",
              "contains_vocal_music": true,
              "song_title": "神っぽいな",
              "original_artist": "ピノキオピー",
              "performer_in_video": "Hoshimachi Suisei",
              "original_language": "Japanese",
              "reference_lyrics_og": "愛の謳を歌おうぜ\\nそれは神っぽいな\\n（...Full Japanese lyrics here. Never truncate, omit, or write placeholder markers...）",
              "reference_lyrics_en": "Let's sing a song of love\\nThat's very god-like\\n（...Full English translation here. Never truncate, omit, or write placeholder markers...）"
            }
          ]
        }
        ```
        </code>
      </example>
    </examples>
    """  # noqa: E501
).strip()

_LYRICS_SCENES_USER_TEMPLATE = dedent(
    """
    <task>
      Analyze the provided video to map scenes and detect songs. Look up the official lyrics using the web search tool. 
      CRITICAL: Do NOT transcribe the audio or translate any lyrics yourself. If you cannot find the official lyrics via web search, leave the lyrics fields as null.
    </task>
    """  # noqa: E501
).strip()


def get_lyrics_scenes_prompt() -> Prompt:
    """Returns the Prompt object for scene detection and lyrics research.

    Returns:
        Prompt: The full Prompt object containing system and user prompts.
    """
    return Prompt(
        system_prompt=_LYRICS_SCENES_SYSTEM_TEMPLATE,
        user_prompt=_LYRICS_SCENES_USER_TEMPLATE,
    )


_SUBTITLES_SYSTEM_TEMPLATE = dedent(
    """
    <role>
      You are an advanced AI expert in audio-visual translation and subtitling. Your specialty is generating audio-synchronized, contextually rich subtitles from multimodal inputs using native audio tokenization.
    </role>

    <golden_rules>
      <rule name="VOCAL EVENTS (The Audio Rule)">
        The vocal audio waveform is your absolute ground truth for spoken dialogue. The exact millisecond a vocal cord activates dictates the 's' (start) timecode. NEVER output dialogue text if there is no vocal audio driving it.
      </rule>
      <rule name="VISUAL EVENTS (The On-Screen Text Exception)">
        If prominent, relevant text (e.g., Chapter Titles, Location Signs, Letters) appears on screen, subtitle it for the duration it is clearly visible, even if there is no audio.
      </rule>
    </golden_rules>

    <decoding_hierarchy>
      When transcribing spoken audio (vocal events), if the vocals are slurred, fast, or ambiguous, use this fallback hierarchy to determine the correct intended words:
      <step level="1" name="Primary Fallback (Burnt-in Subs/Lyrics)">
        Hardcoded subtitles or on-screen lyrics that correspond to the audio are your most reliable guide for correct spelling and strictly override everything else.
      </step>
      <step level="2" name="Secondary Fallback (Scene Context)">
        Visual actions, character emotions, and environments (e.g., rain, night) provide powerful context clues.
      </step>
      <step level="3" name="Tertiary Fallback (Auxiliary Reference JSON)">
        Use the provided Reference JSON ONLY as an auxiliary spelling reference.
      </step>
      <step level="4" name="The Manual Transcription Mandate">
        If the Auxiliary Reference is null, incomplete, or deviates from the audio, YOU ARE NOT EXEMPT. You MUST manually transcribe and translate the remaining vocals using your native audio perception.
      </step>
    </decoding_hierarchy>

    <grounding_instructions>
      The provided Reference JSON contains web-scraped lyrics meant solely as an auxiliary guide for song spelling and composition. It is NOT a script.
      <instruction name="AUDIO IS SUPREME">
        If the performer stops singing to talk, if the background music goes silent, or if they sing a different verse, melody, or song, YOU MUST IGNORE the reference JSON entirely for that duration.
      </instruction>
      <instruction name="NO PRE-EMPTIVE PASTING">
        Never blindly copy, paste, or assume reference lyrics. If the vocal audio does not match the reference lyrics, do not force them into the subtitles.
      </instruction>
      <instruction name="MISMATCH OR WRONG SONG">
        If the audio clearly differs from the reference JSON, ignore the JSON entirely and perform 100% manual transcription of what is actually heard.
      </instruction>
      <instruction name="ZERO GHOST SUBTITLES">
        Remain SILENT during pure instrumental music, long pauses, or sound effects. If no vocals are heard in the audio track, output nothing (unless there is a prominent Visual Event). Do not use reference lyrics to fill silent gaps.
      </instruction>
      <instruction name="VIDEO LONGER THAN REFERENCE">
        Switch immediately to 100% manual transcription for the remainder of the video. Subtitle 100% of the audible vocals.
      </instruction>
    </grounding_instructions>

    <timecode_rules>
      AI models frequently suffer from "Cascading Delays" (subtitles falling behind audio). Override this bias using these strict mechanical rules:
      <rule name="The Sacred Start Timecode">
        The 's' (start) timecode is immutable. It MUST trigger the exact millisecond the word is spoken in the audio track. Never delay a start timecode to accommodate a previous long subtitle block.
      </rule>
      <rule name="Truncation Over Extension">
        If Line 1 is spoken, and Line 2 begins immediately after, Line 1's 'e' timecode MUST be aggressively truncated to make room for Line 2's 's' timecode.
      </rule>
      <rule name="Instantaneous Transitions">
        In rapid speech with no breaths, Line 1's 'e' MUST EXACTLY EQUAL Line 2's 's' (e.g., Line 1 'e': 01:05.500 -> Line 2 's': 01:05.500).
      </rule>
      <rule name="Timecode Format">
        MUST strictly adhere to MM:SS.mmm (e.g., 01:05.300). Always pad with zeros. Always ensure timecodes represent the actual time of occurrence in the audio.
      </rule>
    </timecode_rules>

    <segmentation_and_translation>
      <max_length>
        Limit each subtitle block to 50 characters per line. Group phrases logically.
      </max_length>
      <context_aware_translation>
        Ensure semantic continuity. If a sentence is split across two blocks, ensure the English translation grammatically flows from Part 1 to Part 2 (e.g., using ellipses "..."). Always analyze the Previous 2 Sentences, the Next 2 Sentences, and the Full Current Sentence. Use this context to determine pronouns, tense, and tone.
      </context_aware_translation>
      <visual_text_rules>
        Translate prominent, relevant visual text (titles, signs). Exclude meaningless background clutter or UI elements. If visual text appears simultaneously with spoken dialogue, prioritize translating the spoken dialogue (do not merge them).
      </visual_text_rules>
      <no_cc_tags>
        Transcribe speech/vocals only. NO closed captioning tags like [applause], (sighs), or ♪.
      </no_cc_tags>
    </segmentation_and_translation>

    <examples>
      <example name="Resolving Ambiguous Audio with Text/JSON">
        <code>
        ```json
        {
          "subs": [
            {
              "s": "00:10.000",
              "e": "00:11.500",
              "og": "パレード",
              "en": "Parade"
            }
          ]
        }
        ```
        </code>
      </example>
      <example name="Partial Video Scenario & Fast-Paced Song">
        <code>
        ```json
        {
          "subs": [
            {
              "s": "01:12.100",
              "e": "01:12.800",
              "og": "息つく暇もないほど",
              "en": "With no time to even catch my breath,"
            },
            {
              "s": "01:12.800",
              "e": "01:13.500",
              "og": "走り抜けてきた",
              "en": "I've been running through."
            }
          ]
        }
        ```
        </code>
      </example>
      <example name="Handling Pauses Mid-Sentence (Preventing Early Cutoff)">
        <code>
        ```json
        {
          "subs": [
            {
              "s": "01:07.800",
              "e": "01:13.200",
              "og": "人の世に...",
              "en": "From the world of men..."
            },
            {
              "s": "01:14.200",
              "e": "01:18.500",
              "og": "生まれし悪を 闇にへと 葬れよ",
              "en": "exorcise born evil into the darkness."
            }
          ]
        }
        ```
        </code>
      </example>
      <example name="Preventing Cascading Delays in Rapid Speech">
        <code>
        ```json
        {
          "subs": [
            {
              "s": "00:45.200",
              "e": "00:46.000",
              "og": "考える隙すらなく",
              "en": "With no time to even think,"
            },
            {
              "s": "00:46.000",
              "e": "00:46.900",
              "og": "直感だけで動いた",
              "en": "I moved on pure instinct."
            }
          ]
        }
        ```
        </code>
      </example>
      <example name="Handling Silent On-Screen Text Before Dialogue">
        <code>
        ```json
        {
          "subs": [
            {
              "s": "00:02.000",
              "e": "00:05.000",
              "og": "第1章",
              "en": "[Title: Chapter 1]"
            },
            {
              "s": "00:06.500",
              "e": "00:08.000",
              "og": "さあ、始めようか",
              "en": "Now then, shall we begin?"
            }
          ]
        }
        ```
        </code>
      </example>
    </examples>
    """  # noqa: E501
).strip()


def get_subtitle_prompt(scene_response: LyricsSceneAiResponse | None) -> Prompt:
    """Generates the Prompt object for subtitle generation.

    Args:
        scene_response: The scene detection data.
            Can be None if lyrics/scene detection is disabled.

    Returns:
        Prompt: The full Prompt object containing system and user prompts.
    """
    if scene_response:
        scenes_data = scene_response.model_dump(mode="json")["scenes"]
        scene_json = json.dumps(scenes_data)
    else:
        scene_json = "null"

    user_prompt = dedent(
        f"""
        <task>
          Generate precise, synchronized subtitles for the provided video segment based on the system instructions.
        </task>
        <scene_and_lyrics_reference_json>
          {scene_json}
        </scene_and_lyrics_reference_json>
        """
    ).strip()

    return Prompt(
        system_prompt=_SUBTITLES_SYSTEM_TEMPLATE,
        user_prompt=user_prompt,
    )


def get_verification_prompt(base_prompt: Prompt, video_duration_ms: int) -> Prompt:
    """Wraps the base subtitle prompt with strict verification instructions.

    Args:
        base_prompt: The original base Prompt object.
        video_duration_ms: The duration of the video segment in milliseconds.

    Returns:
        Prompt: The updated Prompt object with verification instructions in the user prompt.
    """
    # Note: We deliberately do not provide the "bad" (gapped) subtitle run from the previous
    # attempt to the AI model here. If we do, the model has a strong tendency to "double down"
    # and insist that its previous output was complete, rather than performing a fresh, thorough
    # transcription pass to catch the missed segments.

    duration_s = video_duration_ms / 1000.0

    verification_instruction = dedent(
        f"""
        <critical_requirement>
          In a previous attempt, your output contained unacceptably large gaps where audio was not transcribed.
          The quality of the previous generation is suspect.
          You MUST completely regenerate the subtitles for the ENTIRE video segment (total duration: {duration_s:.1f} seconds).
          Do not be lazy. Do not skip sections. Ensure every single vocal event from the very beginning to the very end is accurately transcribed.
        </critical_requirement>
        """  # noqa: E501
    ).strip()

    return Prompt(
        system_prompt=base_prompt.system_prompt,
        user_prompt=f"{base_prompt.user_prompt}\n\n{verification_instruction}",
    )
