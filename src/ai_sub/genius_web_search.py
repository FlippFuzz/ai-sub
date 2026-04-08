"""Web search and lyrics extraction from Genius.

This module provides tools for searching Genius via DuckDuckGo,
fetching song pages, and extracting structured lyrics data. It uses
stealth browser sessions to avoid detection and supports parallel
execution for bulk queries.
"""

import asyncio
import html
import logging
import re
from logging import Handler, LogRecord
from typing import Any, Self
from urllib.parse import urlencode

import logfire
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from scrapling import Selector
from scrapling.fetchers import AsyncStealthySession

from ai_sub.config import GeniusSearchSettings

_NUM_LYRICS_TO_FETCH = 5  # TODO: expose this as a setting in future


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class _DuckDuckGoSearchResult(BaseModel):
    title: str = Field(
        description='The display title of the search result (e.g., "Python Tutorial - W3Schools").',
    )
    href: str = Field(
        description="The destination URL the search result links to "
        '(e.g., HttpUrl("https://www.w3schools.com/python/")).',
    )


class GeniusLyrics(BaseModel):
    """Represents a single song's lyrics data extracted from Genius.

    Attributes:
        title: The headline or title of the web page returned by Genius.
        lyrics: The lyrics found on the page.
    """

    title: str = Field(description="The headline or title of the web page returned by Genius.")
    lyrics: str = Field(description="The lyrics found.")


class QueryResults(BaseModel):
    """Groups multiple search results under the specific query that generated them.

    Attributes:
        query: The search string used to perform the lookup.
        results: The collection of findings associated with the query.
    """

    query: str = Field(description="The original search string or keywords used to generate these results.")
    results: list[GeniusLyrics] = Field(description="The sequence of search results discovered for this query.")


# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------


class _LogfireHandler(Handler):
    """Custom logging handler that routes Python standard library logs to logfire.

    This handler intercepts log records from third-party libraries (specifically
    Scrapling) and forwards them to logfire for centralized observability. The
    original log level is preserved so that warnings and errors remain visible.

    Attributes:
        None — state is fully managed by the parent logging.Handler class.
    """

    def emit(self, record: LogRecord) -> None:
        """Process a log record by routing it to logfire.

        The original log level from Scrapling is preserved and forwarded
        to logfire. The formatted log message is forwarded along with
        the original logger name for traceability.

        Args:
            record: The logging record emitted by a Scrapling logger. Contains
                the message, level, and metadata that gets forwarded to logfire.
        """
        level = record.levelname.lower()
        message = self.format(record)

        # Downgrade known noisy message to debug to avoid flooding logs.
        if message == "No Cloudflare challenge found.":
            level = "debug"

        # Dynamically resolve the logfire function for the target level,
        # falling back to logfire.info if the level name is invalid.
        logfire_attr = getattr(logfire, level, logfire.info)
        logfire_attr(message, logger=record.name)


def redirect_scrapling_logs_to_logfire() -> None:
    """Redirect Scrapling's internal logs to logfire and silence console output.

    This function configures the "scrapling" logger by:
        1. Removing all existing handlers (typically console/stderr handlers).
        2. Attaching a custom _LogfireHandler that forwards logs to logfire.
        3. Setting the log level to INFO to filter out debug noise.
        4. Disabling propagation to the root logger to prevent duplicate output.

    This should be called once during application startup if you want Scrapling
    logs integrated into your logfire observability pipeline.

    """
    # Instantiate the custom handler that bridges to logfire.
    handler = _LogfireHandler()

    # Obtain the top-level Scrapling logger — all Scrapling loggers
    # are children of this logger, so configuring it affects the entire library.
    log = logging.getLogger("scrapling")

    # Set threshold to INFO; debug logs from Scrapling are excessively noisy.
    log.setLevel(logging.INFO)

    # Remove all pre-configured handlers (e.g., StreamHandler writing to stderr)
    # to prevent logs from appearing in both the console and logfire.
    log.handlers.clear()

    # Attach our logfire-bridging handler.
    log.addHandler(handler)

    # Prevent the log record from propagating to the root logger, which would
    # cause duplicate console output from other configured handlers.
    log.propagate = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class AsyncGeniusWebSearchSession:
    """A wrapper around AsyncStealthySession that limits concurrent requests and supports proxy configuration."""

    def __init__(self, config: GeniusSearchSettings):
        """Initialize the session with search configuration.

        Args:
            config: The search settings containing concurrency and proxy configuration.
        """
        self._config = config
        self._session: AsyncStealthySession | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    async def __aenter__(self) -> Self:
        """Initialize the underlying session and enter its context.

        Returns:
            The initialized session instance.
        """
        self._session = AsyncStealthySession(
            max_pages=self._config.max_concurrent,
            headless=True,
            solve_cloudflare=True,
            disable_resources=True,
            proxy=self._config.proxy,
        )
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Close the underlying session when exiting the context."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)

    async def fetch(self, url: str, params: dict[str, str] | None = None):
        """Perform a fetch request with concurrency limiting and URL parameter encoding.

        Args:
            url: The URL to fetch.
            params: Optional query parameters to append to the URL.

        Returns:
            The response object from the stealthy session.

        Raises:
            RuntimeError: If the session is not initialized via context manager.
        """
        if self._session is None:
            raise RuntimeError(
                "GeniusWebSearchSession is not initialized. "
                "Use 'async with' or enter_async_context before calling fetch()."
            )

        with logfire.span("Fetching", _level="debug") as span:
            span.set_attribute("url", url)
            span.set_attribute("params", params)

            if params:
                url = f"{url}?{urlencode(params)}"

            async with self._semaphore:
                response = await self._session.fetch(url)
                span.set_attribute("result", response.html_content)
                return response


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


async def _duckduckgo_search(session: AsyncGeniusWebSearchSession, query: str) -> list[_DuckDuckGoSearchResult]:
    """Perform a DuckDuckGo web search and return structured results.

    Sends a query to DuckDuckGo via a stealth session and parses the HTML
    to extract search results.

    Args:
        session: The stealth browser session for fetching pages.
        query: The search query string (e.g., "python web scraping tutorial").

    Returns:
        A list of search results with titles and URLs. Entries with missing
        titles or invalid URLs are silently skipped.
    """
    params = {
        "ia": "web",
        "origin": "funnel_home_google",
        "t": "h_",
        "q": query,
        "chip-select": "search",
    }
    url = "https://duckduckgo.com"

    # Fetch the search results page using the stealth browser session.
    response = await session.fetch(url, params=params)

    # Parse the HTML response using Scrapling's Selector, which provides
    page = Selector(response.html_content)

    # Container for all successfully parsed search results.
    results: list[_DuckDuckGoSearchResult] = []

    # DuckDuckGo wraps each search result in an <h2> element. The structure
    # is roughly:
    #
    #   <h2 class="...">
    #     <a href="..." class="...">
    #       <span class="...">Result Title Text</span>
    #     </a>
    #   </h2>
    #
    # We iterate over all <h2> elements to extract each result.
    for h2 in page.find_all("h2"):
        # Find all <a> tags with an href attribute inside this <h2>.
        # Typically there is only one — the main result link — but we
        # iterate defensively in case the structure varies.
        a_list = h2.css("a[href]")
        for a in a_list:
            href = ""
            title = ""

            # Extract the href attribute from the <a> tag.
            href = a.attrib.get("href", "")

            # Skip if the href does not have any content.
            if not href:
                continue

            # Search for <span> elements inside the <a> tag to find the title.
            # DuckDuckGo places the visible result title inside a <span>.
            for span in a.css("span"):
                title = span.text

                # Skip if the span exists but contains no text content.
                if not title:
                    continue

            # Both title and validated href are available — add the result.
            if title:
                results.append(_DuckDuckGoSearchResult(title=title, href=href))
                break

    return results


def _genius_lyrics_validate_url(url: str) -> bool:
    """Validate that a URL points to a valid Genius song page.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is a valid Genius song page, False otherwise.
    """
    # Remove query parameters before any path comparisons
    if "?" in url:
        url = url.split("?")[0]

    # Check if URL starts with the Genius base URL
    if not url.startswith("https://genius.com/"):
        return False

    # Remove the base URL to check the path
    path = url[len("https://genius.com/") :]

    # Don't allow the main page (empty path)
    if not path:
        return False

    # Don't allow album pages
    if path.startswith("albums/"):
        return False

    # Don't allow artist pages
    if path.startswith("artists/"):
        return False

    # Don't allow songs list pages (contain song listings, not lyrics)
    if path.startswith("songs/"):
        return False

    return True


async def _genius_lyrics_get(session: AsyncGeniusWebSearchSession, url: str) -> GeniusLyrics | None:
    """Fetch and extract lyrics from a single Genius page.

    Results are cached by URL to avoid redundant HTTP requests.

    Args:
        session: The stealth browser session for fetching pages.
        url: The Genius page URL to extract lyrics from.

    Returns:
        A GeniusLyrics object containing the title and lyrics, or None if
        extraction fails.
    """
    # Check cache first
    if url in _url_cache:
        return _url_cache[url]

    title = ""
    lyrics = ""

    with logfire.span("genius_lyrics", _level="debug") as span:
        # Validate the URL before proceeding with extraction
        if not _genius_lyrics_validate_url(url):
            logfire.warn("Invalid Genius URL", url=url)
            _url_cache[url] = None
            return None

        span.set_attribute("url", url)

        # Fetch the HTML content using the fetch utility
        response = await session.fetch(url)
        if not response.status or response.status >= 400:
            logfire.warn("Bad HTTP response", url=url, status=response.status)
            return None
        page = Selector(response.html_content)

        # Extract the song title from the first h1 element on the page
        # Example:
        # <h1 class="SongHeader-desktop__Title-sc-cb565fd5-9 dHFnIx">
        #   <span class="SongHeader-desktop__HiddenMask-sc-cb565fd5-13 hqVdGN"
        #     >XY&amp;Z</span
        #   >
        # </h1>
        h1 = page.find("h1")
        if h1:
            # Unescape HTML entities (e.g., &amp; -> &) and strip whitespace
            title = html.unescape(h1.get_all_text()).strip()
        else:
            logfire.debug("No <h1> found on page", url=url)

        # Parse lyrics with BeautifulSoup for more detailed element-level control
        # Adapted from https://github.com/johnwmillr/LyricsGenius/blob/cae66181ac2614c5b3faff96f973dec8b18c0416/lyricsgenius/genius.py#L151
        soup = BeautifulSoup(response.html_content, "html.parser")

        # Remove LyricsHeader divs from the DOM to avoid duplicate title/headers
        for header in soup.find_all("div", class_=re.compile("LyricsHeader")):
            header.decompose()

        # Find all lyrics containers marked with data-lyrics-container="true"
        # These are the div elements that contain the actual lyrics text
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        span.set_attribute("lyrics_containers", len(containers))

        if not containers:
            logfire.debug("No lyrics containers found", url=url)

        # Iterate through each container to build the complete lyrics text
        for container in containers:
            if not container.contents:
                # Empty container indicates a paragraph break
                lyrics += "\n"
                continue
            for element in container.contents:
                if isinstance(element, NavigableString):
                    # Plain text nodes are added directly
                    lyrics += str(element)
                elif isinstance(element, Tag):
                    if element.name == "br":
                        # Line breaks become newlines
                        lyrics += "\n"
                    elif element.get("data-exclude-from-selection") != "true":
                        # Skip elements marked for exclusion
                        lyrics += element.get_text(separator="\n")

        # Clean up the lyrics text:
        # 1. Remove section headers like [Verse], [Chorus], [Bridge], etc.
        lyrics = re.sub(r"(\[.*?\])*", "", lyrics)
        # 2. Collapse multiple consecutive newlines into single newlines
        lyrics = re.sub(r"\n{2,}", "\n", lyrics)
        # 3. Strip leading/trailing whitespace
        lyrics = lyrics.strip()

        if not title:
            logfire.warn(f"No title extracted {url}", url=url)
        if not lyrics:
            logfire.warn(f"No lyrics extracted {url}", url=url)

        # Only return a result if both title and lyrics were successfully extracted
        result = None
        if title and lyrics:
            result = GeniusLyrics(title=title, lyrics=lyrics)
        span.set_attribute("result", result)

        # Cache the result
        _url_cache[url] = result

        return result


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# Simple in-memory cache to avoid redundant HTTP requests for the same URL.
# Key: Genius URL string, Value: GeniusLyrics result or None (on failure).
_url_cache: dict[str, GeniusLyrics | None] = {}


# ---------------------------------------------------------------------------
# Public API / Tool Functions
# ---------------------------------------------------------------------------


async def genius_web_single_search_tool(ctx: RunContext[AsyncGeniusWebSearchSession], query: str) -> list[GeniusLyrics]:
    """Search Genius for lyrics matching a single query.

    Searches for song pages on Genius, validates the resulting URLs, and
    fetches full lyrics from multiple pages in parallel.

    Args:
        ctx: The run context containing the stealth browser session in ``ctx.deps``.
        query: The search string (typically a song title, artist name, or partial
            lyric).

    Returns:
        A list of lyrics objects, each containing ``title`` and ``lyrics``
        string fields. Returns an empty list if no valid pages are found or
        all fetches fail.
    """
    session = ctx.deps
    with logfire.span(f"genius_web_single_search_tool {query}", _level="debug") as span:
        # Search scoped to Genius pages
        search_results = await _duckduckgo_search(session, f"{query} site:genius.com")

        # Collect valid Genius song URLs
        to_fetch_lyrics = []
        for search_result in search_results:
            if _genius_lyrics_validate_url(search_result.href):
                to_fetch_lyrics.append(search_result.href)

            if len(to_fetch_lyrics) > _NUM_LYRICS_TO_FETCH:
                break

        span.set_attribute("urls_to_fetch", to_fetch_lyrics)

        # Fetch lyrics from all collected URLs in parallel
        tasks = [_genius_lyrics_get(session, url) for url in to_fetch_lyrics]
        results = await asyncio.gather(*tasks)

        # Filter out failed extractions (None values)
        lyrics_results = [r for r in results if r is not None]
        span.set_attribute("results", lyrics_results)

        return lyrics_results


async def genius_web_search_tool(
    ctx: RunContext[AsyncGeniusWebSearchSession], queries: list[str]
) -> list[QueryResults]:
    """Search Genius for lyrics across multiple queries in parallel.

    This is the primary entry point for bulk lyrics retrieval. Each query is
    processed independently and concurrently, making it suitable for looking up
    multiple songs, language variants, or alternate title formats in a single
    call.

    Designed for use as an AI agent tool. Provide song titles, partial lyrics,
    or combinations with language hints to retrieve structured lyrics data from
    Genius.

    Example inputs:

    - Song title: ``["Bohemian Rhapsody"]``
    - Title + language: ``["ジェヘナ Japanese", "ジェヘナ English"]``
    - Partial lyrics: ``["遠く手を伸ばしても", "君について行っても"]``

    Args:
        ctx: The run context containing the stealth browser session in ``ctx.deps``.
        queries: A list of search strings. Each element is treated as an
            independent search and will yield its own result entry containing
            the query string and a list of lyrics objects.

    Returns:
        A list of results, one for every input query in the same order. Each
        result contains the original query string and a list of lyrics objects,
        where each lyrics object has ``title`` and ``lyrics`` string fields.
    """
    with logfire.span("genius_web_search_tool", queries=queries, _level="debug"):
        # Run all queries in parallel for maximum throughput
        tasks = [genius_web_single_search_tool(ctx, query) for query in queries]
        results = await asyncio.gather(*tasks)

        # Build the grouped response, preserving input order
        return [QueryResults(query=query, results=lyrics) for query, lyrics in zip(queries, results)]
