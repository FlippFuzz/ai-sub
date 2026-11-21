"""LRCLIB lyrics database provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_sub.config import SearchSettings
from ai_sub.search.search_base import BaseHttpSearchProvider, normalize_query
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
        return {"User-Agent": f"ai-sub {get_app_version()} (https://github.com/FlippFuzz/ai-sub)"}

    def get_tools(self) -> list[Callable[..., Any] | dict[str, Any]]:
        """Returns tool callables for Gemini model registration.

        Returns:
            list[Callable[..., Any] | dict[str, Any]]: Bound search methods.
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
        clean_title = track_title.strip() if track_title else ""
        if not clean_title:
            return []

        params: dict[str, str] = {"track_name": clean_title}
        if artist_name and artist_name.strip():
            params["artist_name"] = artist_name.strip()

        def _extract(data: Any) -> list[dict[str, Any]]:
            extracted: list[dict[str, Any]] = []
            if isinstance(data, list):
                for item in data:
                    if item.get("instrumental"):
                        continue
                    plain = str(item.get("plainLyrics") or "").strip()
                    if plain:
                        extracted.append(
                            {
                                "track_name": item.get("trackName") or item.get("name"),
                                "artist_name": item.get("artistName"),
                                "plain_lyrics": plain[: self.settings.maxlength],
                            }
                        )
                        if len(extracted) >= 3:
                            break
            return extracted

        async def _fetch() -> list[dict[str, Any]]:
            res = await self._request("GET", "https://lrclib.net/api/search", params=params)
            results = _extract(res.json())
            if not results:
                fallback_res = await self._request(
                    "GET",
                    "https://lrclib.net/api/search",
                    params={"q": clean_title},
                )
                results = _extract(fallback_res.json())
            return results

        cache_key = f"lrclib:{normalize_query(clean_title)}:{normalize_query(artist_name or '')}"
        return await self._cached_search(
            cache_key=cache_key,
            span_template="lrclib_search: {title=}, {artist=}",
            fetcher=_fetch,
            cache_empty=True,
            title=clean_title,
            artist=artist_name or "",
        )
