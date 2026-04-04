"""Web search tool using the Lyrics Genius API for song and lyrics lookups."""

import re

from lyricsgenius import Genius
from pydantic import BaseModel, Field, RootModel

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


def lyricsgenius_web_search_tool(queries: list[str]) -> WebSearchResponse:
    """Searches the web across multiple queries using the Lyrics Genius library.

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
    genius = Genius(access_token="XXX")  # Hardcoded to XXX. The library requires us to provide a non-empty string

    all_query_results: list[QueryResults] = []

    for query in queries:
        # Strip common suffix words the LLM might append (e.g., "lyrics").
        clean_query = _QUERY_CLEAN_RE.sub("", query).strip()

        search_results = genius.search(clean_query, per_page=5)
        query_results: list[SearchResult] = []

        for search_result in search_results.get("hits", []):
            song = search_result.get("result", {})
            song_id = song.get("id")
            song_url = song.get("url")
            song_title = song.get("full_title", "Unknown")

            if song_id is None and song_url is None:
                continue

            lyrics = genius.lyrics(song_url=song_url, song_id=song_id, remove_section_headers=True)

            if lyrics:
                query_results.append(
                    SearchResult(
                        title=song_title,
                        lyrics=lyrics,
                    )
                )

        all_query_results.append(
            QueryResults(
                query=query,
                results=query_results,
            )
        )

    return WebSearchResponse(root=all_query_results)
