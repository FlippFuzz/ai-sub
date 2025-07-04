# AI Sub Release Notes

## v0.0.6

This release focuses on minor bug fixes and improvements to the subtitle generation process and project showcase.

**New Features & Improvements:**

*   **Prompt Template Update:** The prompt template used for subtitle generation has been updated for improved accuracy.
*   **Showcase Updates:** The showcase directory now includes version numbers for SRT files, and a new v0.0.6 SRT file has been added for `42h4ydJS3zk`.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.5...v0.0.6

---

## v0.0.5

This release includes updates to the Gemini API configuration, improved error handling, and minor showcase additions.

**New Features & Improvements:**

*   **Gemini API Configuration:** Updated the default thinking budget for the Gemini API to 32768, aligning with the maximum for Gemini 2.5 Pro.
*   **Error Handling:** List the video segments that cannot be processed at the end of execution.
*   **Showcase Additions:** Added new subtitles to the project showcase.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.4...v0.0.5

---

## v0.0.4

This release focuses on ensuring chronological timestamps in subtitle generation, enhancing error messages, and expanding the project showcase.

**New Features & Improvements:**

*   **Chronological Timestamps:** Implemented a mechanism to ensure that timestamps returned by `generate_subtitles` are always chronological, improving subtitle accuracy.
*   **Enhanced Error Messages:** Error messages for response checks have been improved to clearly indicate when retries are occurring, providing better debugging information.
*   **Showcase Expansion:** The project showcase has been updated with new video entries and corresponding SRT files.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.3...v0.0.4

---

## v0.0.3

This release focuses on improving the robustness of subtitle generation by enhancing error handling for invalid AI model responses.

**New Features & Improvements:**

*   **Enhanced Error Logging:** Detailed error messages are now logged for invalid JSON responses received from the Gemini model, providing better insights into issues during subtitle generation.
*   **Robust JSON Handling:** The `json-repair` dependency has been removed. The system will now retry when Gemini returns an invalid response, ensuring data integrity and preventing the use of potentially incomplete or corrupted subtitle data.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.2...v0.0.3

---

## v0.0.2

This release includes changes to the default AI model, improvements in subtitle generation, and an expanded showcase.

**New Features & Improvements:**

*   **Default Model Update:** The default AI model has been updated to `gemini-2.5-pro`. This change was made possible by Google's updated free tier limits, which now allow up to 100 requests per day against `gemini-2.5-pro`, enabling enhanced performance and accuracy in subtitle generation.
*   **AI Prompt Update:** Clarifications have been added regarding subtitle timing accuracy requirements, with an emphasis on the importance of the end timestamp in the prompt template to improve overall subtitle quality.
*   **Showcase Expansion:** The project showcase has been updated with new video entries and corresponding SRT files.
*   **Improved Subtitle Error Handling:** The subtitle error message and duration handling in [`src/ai_sub/main.py`](src/ai_sub/main.py) have been updated for better clarity and accuracy.
*   **Project Description Refinement:** The project description in [`pyproject.toml`](pyproject.toml) has been updated for improved clarity.

**Other Changes:**

*   The `RELEASE_NOTES.md` file has been added to the project for future release documentation.

**Full Changelog**: https://github.com/FlippFuzz/ai-sub/compare/v0.0.1...v0.0.2

---

## v0.0.1

This is the initial release of AI Sub.

**Features:**

*   **AI-Powered Subtitle Generation:** Leverages Google Gemini for generating English and Japanese subtitles with translation capabilities.
*   **Video Segmentation:** Automatically segments input videos into configurable durations for processing.
*   **Concurrent Processing:** Supports parallel processing of video segments for efficient subtitle generation.
*   **Subtitle Compilation:** Combines all generated subtitle parts into a single, final subtitle file.