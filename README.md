# AI Sub: Research Findings on Bot Detection

## Development on this branch is paused.

### Problem

Web scraping for lyrics search is fundamentally broken due to aggressive bot detection.

#### Affected Services

- **DuckDuckGo:** Detects automated requests and returns 0 results.
- **Genius:** Detects automated requests and withholds lyrics content.

#### Attempted Solutions

- **`duckduckgo-search` (now renamed to `ddgs`):** Same problem — returns empty responses when rate-limited or detected as a bot.
