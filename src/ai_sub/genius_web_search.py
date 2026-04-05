"""Web search tool using the Genius API via Scrapling for song and lyrics lookups."""

import asyncio
import logging
import re
import urllib.parse

import logfire
from bs4 import BeautifulSoup, Tag
from install_playwright import install
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, RootModel
from scrapling.fetchers import AsyncStealthySession

logger = logging.getLogger(__name__)

# Words to strip from song titles before searching
_CLEANUP_PATTERNS = [
    # English
    r"\blyrics?\b",
    r"\btranslation?s?\b",
    r"\btranslated?\b",
    r"\beng sub\b",
    r"\beng subs\b",
    r"\bsubtitles?\b",
    r"\bromanized?\b",
    r"\bromaji\b",
    # Japanese
    r"\b歌詞\b",
    r"\b訳\b",
    r"\b翻訳\b",
    r"\b日本語訳\b",
    r"\b和訳\b",
    r"\b英訳\b",
    r"\b罗马字\b",
    r"\bローマ字\b",
]
_CLEANUP_RE = re.compile("|".join(_CLEANUP_PATTERNS), re.IGNORECASE)


def _clean_song_title(title: str) -> str:
    """Removes common filler words like 'lyrics', 'translation', etc. from a song title.

    Handles both English and Japanese terms.

    Args:
        title: The raw song title string, potentially containing filler words.

    Returns:
        The cleaned title with filler words removed.
    """
    cleaned = _CLEANUP_RE.sub("", title).strip()
    # Remove leftover brackets/parentheses that may have surrounded the removed word
    cleaned = re.sub(r"[\[\]（）()]", "", cleaned).strip()
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


class LogfireHandler(logging.Handler):
    """Routes standard library log records to logfire."""

    # Messages to downgrade to DEBUG level to reduce noise
    _DOWNGRADE_MESSAGES = {"no cloudflare challenge found"}

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record by routing it to logfire.

        Args:
            record: The logging record to process and send to logfire.
        """
        level = record.levelname.lower()
        message = self.format(record)

        # Downgrade specific noisy messages to DEBUG
        if message.lower() in self._DOWNGRADE_MESSAGES:
            level = "debug"

        logfire_attr = getattr(logfire, level, logfire.info)
        logfire_attr(message, logger=record.name)


def _route_scrapling_logs_to_logfire() -> None:
    """Attach a logfire handler to Scrapling's loggers and suppress console output."""
    handler = LogfireHandler()
    log = logging.getLogger("scrapling")
    log.setLevel(logging.INFO)
    log.handlers.clear()  # Remove scrapling's built-in handlers to avoid duplicate console output
    log.addHandler(handler)
    log.propagate = False


_route_scrapling_logs_to_logfire()


class SearchResult(BaseModel):
    """Represents a single search result from the Genius API.

    Attributes:
        title: The title of the song or web page returned by the search engine.
        lyrics: The lyrics found for the song.
    """

    title: str = Field(description="The headline or title of the web page returned by the search engine.")
    lyrics: str = Field(description="The lyrics found.")


class QueryResultsList(RootModel):
    """A collection of search outcomes for a single query.

    Attributes:
        root: The underlying list of search results.
    """

    root: list[SearchResult] = Field(
        default_factory=list,
        description="A list containing the search results for the query.",
    )


@logfire.instrument("Starting Genius lyrics search")
async def genius_web_search_tool(song_title: str, language: str) -> QueryResultsList:
    """Searches for a song and retrieves full lyrics using the Genius database.

    When to use:
    - You need accurate, full lyrics for a known song title
    - You have a specific song identified and want its complete lyrics

    When NOT to use:
    - Searching by partial or unknown lyrics (this tool searches by title only)
    - Looking up artist discographies (provide specific song titles instead)
    - Finding songs by genre, mood, or topic (use a general web search instead)

    Args:
        song_title: The exact or approximate title of the song to search for.
        language: The language of the lyrics (e.g., "Japanese", "English", "Korean").
            This helps narrow down results when songs share titles across languages.

    Returns:
        A QueryResultsList containing up to 5 SearchResult items with the song title
        and full lyrics text.

    Example:
        >>> # Get lyrics for a Japanese song
        >>> await genius_web_search_tool("ジェヘナ", "Japanese")
        QueryResultsList(root=[
            SearchResult(title='...', lyrics='...'),
            ...
        ])

    """
    # Clean the song title by removing filler words
    cleaned_title = _clean_song_title(song_title)
    logfire.debug("Cleaned song title", original=song_title, cleaned=cleaned_title)

    # Install playwright
    with logfire.span("Install Playwright"):
        async with async_playwright() as p:
            install([p.firefox])

    # Build search query from title and language
    clean_query = f"{cleaned_title} {language}"
    search_url = f"https://genius.com/api/search?{urllib.parse.urlencode({'q': clean_query, 'per_page': 5})}"
    logfire.debug("Search query and URL", query=clean_query, url=search_url)

    async with AsyncStealthySession(headless=True, solve_cloudflare=True, max_pages=5) as session:
        # Phase 1: Fetch search API results
        with logfire.span("Phase 1: Fetch search API results"):
            response = await session.fetch(search_url)
            logfire.debug("Fetched search response", status=response.status)

        # Parse search results and collect song URLs
        data = response.json().get("response", {}) if response.status == 200 else {}
        songs = []
        for hit in data.get("hits", []):
            song = hit.get("result", {})
            if song.get("url"):
                songs.append(song)

        if not songs:
            logfire.warn("No songs found", query=clean_query)
            return QueryResultsList(root=[])

        # Phase 2: Fetch all lyrics pages concurrently
        with logfire.span("Phase 2: Fetch lyrics pages"):
            lyrics_responses = await asyncio.gather(*(session.fetch(song["url"]) for song in songs))
            logfire.debug("Fetched lyrics pages", count=len(lyrics_responses))

        # Phase 3: Parse lyrics and build results
        with logfire.span("Phase 3: Parse lyrics and build results"):
            results: list[SearchResult] = []
            for song, lyrics_response in zip(songs, lyrics_responses):
                if lyrics_response.status != 200:
                    logfire.warn(
                        "Failed to fetch lyrics",
                        url=song.get("url"),
                        status=lyrics_response.status,
                    )
                    continue

                soup = BeautifulSoup(lyrics_response.html_content, "html.parser")

                # Remove LyricsHeader divs from the DOM
                for header in soup.find_all("div", class_=re.compile("LyricsHeader")):
                    header.decompose()

                # Find all lyrics containers
                containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
                if not containers:
                    logfire.warn("No lyrics containers found", url=song.get("url"))
                    continue

                lyrics_text = ""
                for container in containers:
                    if not container.contents:
                        lyrics_text += "\n"
                        continue
                    for element in container.contents:
                        if isinstance(element, Tag) and element.name == "br":
                            lyrics_text += "\n"
                        elif not isinstance(element, Tag):  # NavigableString
                            lyrics_text += str(element)
                        elif element.get("data-exclude-from-selection") != "true":
                            lyrics_text += element.get_text(separator="\n")

                # Remove [Verse], [Bridge], etc.
                lyrics_text = re.sub(r"(\[.*?\])*", "", lyrics_text)
                lyrics_text = re.sub(r"\n{2}", "\n", lyrics_text)
                lyrics = lyrics_text.strip("\n") if lyrics_text else None

                if lyrics:
                    results.append(
                        SearchResult(
                            title=song.get("full_title", "Unknown"),
                            lyrics=lyrics,
                        )
                    )

    response = QueryResultsList(root=results)
    logfire.debug("Search complete", response=response)
    return response
