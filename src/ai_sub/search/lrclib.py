"""LRCLIB lyrics database provider."""

from __future__ import annotations

from typing import Any, Callable

import logfire

from ai_sub.config import SearchSettings
from ai_sub.search.base import BaseHttpSearchProvider, normalize_query
from ai_sub.telemetry import get_app_version


class LrclibSearchProvider(BaseHttpSearchProvider):
    """Encapsulates LRCLIB lyrics search operations using the shared HTTP search base."""

    def __init__(self, settings: SearchSettings) -> None:
        """Initializes the LRCLIB provider.

        Args:
            settings: Search configuration settings.
        """
        super().__init__(settings, provider_name="lrclib", requires_api_key=False)

    def _get_headers(self) -> dict[str, str]:
        """Constructs headers for LRCLIB API requests including client identification.

        Returns:
            dict[str, str]: HTTP request headers.
        """
        return {
            "User-Agent": f"ai-sub {get_app_version()} (https://github.com/FlippFuzz/ai-sub)",
        }

    def get_tools(self) -> list[Callable[..., Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any]]: Bound search methods.
        """
        return [self.search_lrclib_lyrics_database]

    async def search_lrclib_lyrics_database(
        self,
        track_title: str,
        artist_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the LRCLIB database for official song lyrics using exact track title and optional artist name.

        Use this tool when a song or vocal performance is detected and you need original-language lyrics.
        This tool returns ONLY original-language lyrics (e.g. Japanese, Korean, English); it NEVER returns
        English translations. If an English translation of non-English lyrics is needed, use the general web
        search tool instead.
        DO NOT include extra words like 'lyrics', 'full', 'MV', 'off vocal', or '歌詞' in track_title.

        Args:
            track_title: Exact title of the song (e.g. 'いいわけバニー', 'アイドル', 'Bohemian Rhapsody').
            artist_name: Optional original artist or composer name (e.g. '兎田ぺこら', 'YOASOBI', 'Queen').

        Returns:
            list[dict[str, Any]]: Matching track entries containing track_name, artist_name, and plain_lyrics.
        """
        norm_title = normalize_query(track_title)
        norm_artist = normalize_query(artist_name or "")
        cache_key = f"lrclib:{norm_title}:{norm_artist}"
        if cache_key in self._cache:
            logfire.debug("Using cached LRCLIB search result for {title=}", title=track_title)
            return self._cache[cache_key]

        params: dict[str, str] = {"track_name": track_title.strip()}
        if artist_name and artist_name.strip():
            params["artist_name"] = artist_name.strip()

        with logfire.span("lrclib_search: {title=}, {artist=}", title=track_title, artist=artist_name or ""):
            response = await self._request("GET", "https://lrclib.net/api/search", params=params)
            raw_data = response.json()

            # If structured track_name search returns no results, fall back to keyword query
            if not raw_data:
                fallback_res = await self._request(
                    "GET",
                    "https://lrclib.net/api/search",
                    params={"q": track_title.strip()},
                )
                raw_data = fallback_res.json()

            results: list[dict[str, Any]] = []
            if isinstance(raw_data, list):
                for item in raw_data:
                    if item.get("instrumental"):
                        continue
                    plain_lyrics = item.get("plainLyrics")
                    if not plain_lyrics or not str(plain_lyrics).strip():
                        continue

                    truncated_lyrics = str(plain_lyrics).strip()[: self.settings.max_length]
                    results.append(
                        {
                            "track_name": item.get("trackName") or item.get("name"),
                            "artist_name": item.get("artistName"),
                            "plain_lyrics": truncated_lyrics,
                        }
                    )
                    if len(results) >= 3:
                        break

            self._cache[cache_key] = results
            return results
