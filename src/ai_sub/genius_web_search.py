"""Web search tool using the Genius API via Scrapling for song and lyrics lookups."""

import asyncio
import logging
import re
import time
import urllib.parse
from asyncio import TimerHandle

import logfire
from bs4 import BeautifulSoup, Tag
from install_playwright import install
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, RootModel
from scrapling.fetchers import AsyncStealthySession

logger = logging.getLogger(__name__)


class LogfireHandler(logging.Handler):
    """Routes standard library log records to logfire."""

    # Messages to downgrade to DEBUG level to reduce noise
    _DOWNGRADE_MESSAGES = {"no cloudflare challenge found."}

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record by routing it to logfire.

        Args:
            record: The logging record to process and send to logfire.
        """
        level = record.levelname.lower()
        message = self.format(record)

        # Downgrade specific noisy messages to DEBUG
        if message.lower().strip() in self._DOWNGRADE_MESSAGES:
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


class _SessionManager:
    """Manages a shared AsyncStealthySession with a serial queue to avoid concurrent Cloudflare solve conflicts."""

    def __init__(self) -> None:
        self._session: AsyncStealthySession | None = None
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[asyncio.Future] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._last_used: float = 0
        self._IDLE_TIMEOUT = 300  # 5 minutes
        self._cleanup_timer: TimerHandle | None = None

    async def _worker(self):
        """Process queued fetch requests one at a time."""
        while True:
            future: asyncio.Future = await self._queue.get()
            if future.cancelled():
                self._queue.task_done()
                continue
            try:
                url, done_event = future.result()
                if self._session and not getattr(self._session, "_closed", False):
                    result = await self._session.fetch(url)
                else:
                    result = None
                done_event.set_result(result)
            except Exception as exc:
                if not done_event.done():
                    done_event.set_exception(exc)
            finally:
                self._queue.task_done()
                self._last_used = time.monotonic()
                self._schedule_cleanup()

    def _close_if_idle(self):
        """Close the session if it hasn't been used recently."""
        if time.monotonic() - self._last_used >= self._IDLE_TIMEOUT:
            if self._session and not getattr(self._session, "_closed", False):
                asyncio.create_task(self._session.__aexit__(None, None, None))
                logfire.debug("Closed idle shared session")
            self._session = None
        self._cleanup_timer = None

    def _schedule_cleanup(self):
        """Schedule session cleanup after idle timeout."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        loop = asyncio.get_event_loop()
        self._cleanup_timer = loop.call_later(self._IDLE_TIMEOUT, self._close_if_idle)

    async def _ensure_session(self):
        """Create the session and worker if needed."""
        async with self._lock:
            if self._session is None or getattr(self._session, "_closed", False):
                async with async_playwright() as p:
                    install([p.firefox])
                self._session = AsyncStealthySession(
                    headless=True, disable_resources=True, solve_cloudflare=True, max_pages=5
                )
                await self._session.__aenter__()
                logfire.debug("Created new shared session")
                # Start the worker (only once per session lifecycle)
                if self._worker_task is None or self._worker_task.done():
                    self._worker_task = asyncio.create_task(self._worker(), name="genius-session-worker")

    async def fetch(self, url: str):
        """Queue a fetch request and wait for the result (processed serially).

        Args:
            url: The URL to fetch.

        Returns:
            The fetch response from the session.
        """
        await self._ensure_session()
        done_event: asyncio.Future = asyncio.get_event_loop().create_future()
        future_result: asyncio.Future = asyncio.get_event_loop().create_future()
        future_result.set_result((url, done_event))
        await self._queue.put(future_result)
        return await done_event

    async def close(self):
        """Shutdown the worker and close the session."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        # Drain the queue
        while not self._queue.empty():
            try:
                f = self._queue.get_nowait()
                if not f.done():
                    f.set_result((None, asyncio.get_event_loop().create_future()))
            except asyncio.QueueEmpty:
                break
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._session and not getattr(self._session, "_closed", False):
            await self._session.__aexit__(None, None, None)
            logfire.debug("Closed session on shutdown")
        self._session = None


# Module-level singleton
_session_manager = _SessionManager()


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

    # Build search query from title and language
    clean_query = f"{cleaned_title} {language}"
    search_url = f"https://genius.com/api/search?{urllib.parse.urlencode({'q': clean_query, 'per_page': 5})}"

    # Phase 1: Fetch search API results (queued serially through shared session)
    with logfire.span("Phase 1: Fetch search API results") as span:
        span.set_attribute("query", clean_query)
        span.set_attribute("url", search_url)
        response = await _session_manager.fetch(search_url)
        span.set_attribute("status", response.status)

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

    # Phase 2: Fetch all lyrics pages (queued serially through shared session)
    with logfire.span("Phase 2: Fetch lyrics pages") as span:
        lyrics_responses = await asyncio.gather(*(_session_manager.fetch(song["url"]) for song in songs))
        span.set_attribute("count", len(lyrics_responses))

    # Phase 3: Parse lyrics and build results
    with logfire.span("Phase 3: Parse lyrics and build results") as span:
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
    with logfire.span("Search complete") as span:
        span.set_attribute("result_count", len(results))
    return response
