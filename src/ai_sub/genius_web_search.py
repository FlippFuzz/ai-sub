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


class LogfireHandler(logging.Handler):
    """Routes standard library log records to logfire."""

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record by routing it to logfire.

        Args:
            record: The logging record to process and send to logfire.
        """
        level = record.levelname.lower()
        message = self.format(record)
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

# Common suffix words that LLMs append but hurt Genius search accuracy.
_QUERY_CLEAN_RE = re.compile(r"\b(lyrics?|歌詞|訳歌詞?|翻訳)\b", re.IGNORECASE)


class SearchResult(BaseModel):
    """Represents a single search result from the Genius API.

    Attributes:
        title: The title of the song or web page returned by the search engine.
        lyrics: The lyrics found for the song.
    """

    title: str = Field(description="The headline or title of the web page returned by the search engine.")
    lyrics: str = Field(description="The lyrics found.")


class QueryResults(BaseModel):
    """Groups multiple search results under the specific query that generated them.

    Attributes:
        query: The search string used to perform the lookup.
        results: The collection of findings associated with the query.
    """

    query: str = Field(description="The original search string or keywords used to generate these results.")
    results: list[SearchResult] = Field(description="The sequence of search results discovered for this query.")


class WebSearchResponse(RootModel):
    """A collection of search outcomes for one or more independent web queries.

    Attributes:
        root: The underlying list of query results, organized by search string.
    """

    root: list[QueryResults] = Field(
        default_factory=list,
        description="A list containing the search results grouped by their respective queries.",
    )


@logfire.instrument("Starting Genius lyrics search")
async def genius_web_search_tool(queries: list[str]) -> WebSearchResponse:
    """Searches the web across multiple queries using the Genius database.

    This tool can search by song title or partial lyrics. It retrieves
     full lyrics from the Genius database.

    Example Searches:
    * `["ジェヘナ Japanese", "ジェヘナ English"]`
    * `["Shiny Smily Story Japanese", "聖槍爆裂ボーイ English"]`
    * `["遠く手を伸ばしても", "君について行っても"]`

    Args:
        queries: A list of search strings or keywords to execute. Each query is
            processed independently, and results are grouped by their originating
            query.

    Returns:
        A structured response containing the combined results for all provided
        queries, organized by search string.

    Example:
        >>> lyricsgenius_web_search_tool(["ジェヘナ Japanese", "ジェヘナ English"])
        WebSearchResponse(root=[QueryResults(query='ジェヘナ Japanese', results=[...]), ...])

    """
    # Install playwright
    with logfire.span("Install Playwright"):
        async with async_playwright() as p:
            install([p.firefox])

    clean_queries = [_QUERY_CLEAN_RE.sub("", q).strip() for q in queries]
    search_urls = [
        f"https://genius.com/api/search?{urllib.parse.urlencode({'q': q, 'per_page': 5})}" for q in clean_queries
    ]
    logfire.debug("Clean queries and URLs", clean_queries=clean_queries, urls=search_urls)

    async with AsyncStealthySession(headless=True, solve_cloudflare=True, max_pages=5) as session:
        # Phase 1: Fetch all search API results concurrently
        # Reference: https://github.com/johnwmillr/LyricsGenius/blob/cae66181ac2614c5b3faff96f973dec8b18c0416/lyricsgenius/api/public_methods/search.py#L13
        with logfire.span("Phase 1: Fetch search API results"):
            search_responses = await asyncio.gather(*(session.fetch(url) for url in search_urls))
            logfire.debug("Fetched search responses", search_responses=search_responses)

        # Parse search results and collect song URLs
        all_songs: list[tuple[int, dict]] = []  # (query_index, song_info)
        for query_idx, response in enumerate(search_responses):
            data = response.json().get("response", {}) if response.status == 200 else {}
            for hit in data.get("hits", []):
                song = hit.get("result", {})
                if song.get("url"):
                    all_songs.append((query_idx, song))

        # Phase 2: Fetch all lyrics pages concurrently
        # Reference: https://github.com/johnwmillr/LyricsGenius/blob/cae66181ac2614c5b3faff96f973dec8b18c0416/lyricsgenius/genius.py#L151
        with logfire.span("Phase 2: Fetch lyrics pages"):
            lyrics_responses = await asyncio.gather(*(session.fetch(song["url"]) for _, song in all_songs))
            logfire.debug("Fetched lyrics pages", lyrics_responses=lyrics_responses)

        # Phase 3: Parse lyrics and build results grouped by query
        with logfire.span("Phase 3: Parse lyrics and build results"):
            query_results_map: dict[int, list[SearchResult]] = {i: [] for i in range(len(queries))}
            for (query_idx, song), lyrics_response in zip(all_songs, lyrics_responses):
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

                # # Remove [Verse], [Bridge], etc.
                lyrics_text = re.sub(r"(\[.*?\])*", "", lyrics_text)
                lyrics_text = re.sub(r"\n{2}", "\n", lyrics_text)
                lyrics = lyrics_text.strip("\n") if lyrics_text else None

                if lyrics:
                    query_results_map[query_idx].append(
                        SearchResult(
                            title=song.get("full_title", "Unknown"),
                            lyrics=lyrics,
                        )
                    )

        all_query_results = [
            QueryResults(
                query=query,
                results=query_results_map[i],
            )
            for i, query in enumerate(queries)
        ]

    response = WebSearchResponse(root=all_query_results)
    logfire.debug("Search complete", response=response)
    return response
