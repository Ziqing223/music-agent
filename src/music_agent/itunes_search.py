"""P11-T3: credential-free iTunes Search API discovery adapter.

One read-only adapter for discovering songs through Apple's public iTunes Search API. No
Apple Developer Token, user token, or API key exists anywhere on this path; the transport
speaks plain HTTPS GET to ``itunes.apple.com`` and needs nothing from the environment.

The module maps the iTunes JSON shape into the shared :class:`CatalogTrack` domain type
(reusing the MusicKit adapter's error classes), under the distinct ``itunes_store`` external
identity namespace:

- ``trackId`` -> Track external identity (never inferred equal to an Apple Music Catalog ID)
- ``artistId`` -> Artist external identity
- ``collectionId`` -> Album external identity

The iTunes API supplies no ISRC; ``isrc`` is therefore always ``None`` and never fabricated.
``previewUrl`` is carried for the preview slice (T4): candidates carry it as a source fact, and
``lookup_preview_url`` re-resolves it at preview time from the durable ``itunes_store`` binding
(the signed preview URL rotates, so it is deliberately never persisted on the canonical Track).
``trackViewUrl``, Apple's official web page for the track, is likewise deliberately never
persisted (P16-S4): ``lookup_track_view_url`` re-resolves it from the durable binding at open
time -- the model receives only the canonical id, never composes a URL.
"""

from __future__ import annotations

from datetime import date

import json
import urllib.error
import urllib.parse
import urllib.request

from music_agent.apple_music_catalog import (
    CatalogMappingError,
    CatalogTrack,
    CatalogTransportError,
)


ITUNES_SOURCE_SYSTEM = "itunes_store"

ITUNES_SEARCH_URL_TEMPLATE = (
    "https://itunes.apple.com/search?term={term}&media=music&entity=song&limit={limit}"
    "&country={country}"
)

ITUNES_LOOKUP_URL_TEMPLATE = (
    "https://itunes.apple.com/lookup?id={id}&country={country}"
)

DEFAULT_COUNTRY = "us"
DEFAULT_TIMEOUT_SECONDS = 10.0


class iTunesSearchTransport:
    """Credential-free HTTPS transport for Apple's public iTunes Search API."""

    def __init__(
        self,
        *,
        country: str = DEFAULT_COUNTRY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.country = country
        self.timeout_seconds = timeout_seconds

    def search(self, term: str, limit: int) -> str:
        if not isinstance(term, str) or not term.strip():
            raise CatalogMappingError("term must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise CatalogMappingError("limit must be a positive integer")
        return self._get(
            ITUNES_SEARCH_URL_TEMPLATE.format(
                term=urllib.parse.quote(term),
                limit=limit,
                country=urllib.parse.quote(self.country, safe=""),
            )
        )

    def lookup(self, itunes_id: str) -> str:
        if not isinstance(itunes_id, str) or not itunes_id.strip():
            raise CatalogMappingError("itunes_id must be a non-empty string")
        return self._get(
            ITUNES_LOOKUP_URL_TEMPLATE.format(
                id=urllib.parse.quote(itunes_id, safe=""),
                country=urllib.parse.quote(self.country, safe=""),
            )
        )

    def _get(self, url: str) -> str:
        # No Authorization header exists on this path: the public iTunes API needs no credential.
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            raise CatalogTransportError(
                f"iTunes request failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CatalogTransportError(f"iTunes request failed: {error}") from error
        if not isinstance(body, bytes):
            raise CatalogTransportError("iTunes request returned a non-bytes body")
        return body.decode("utf-8")


class iTunesSearchAdapter:
    """Parse iTunes Search payloads into validated ``CatalogTrack`` values.

    Each result must carry a ``trackId``, ``artistId``, ``trackName``, and ``artistName``
    (fail closed); ``collectionId`` / ``collectionName`` are required to resolve an album.
    """

    def __init__(self, transport: iTunesSearchTransport) -> None:
        if not callable(getattr(transport, "search", None)):
            raise CatalogMappingError("transport must provide search(term, limit)")
        self.transport = transport

    def search(self, term: str, limit: int = 25) -> tuple[CatalogTrack, ...]:
        return self.parse_search_results(self.transport.search(term, limit))

    def lookup_preview_url(self, itunes_id: str) -> str | None:
        """Resolve one ``itunes_store`` trackId to its current preview URL (absent -> None).

        The lookup payload shape is the same results array as search; a result only matches
        when its ``trackId`` agrees exactly with the requested ID (never a name or position
        match). Any payload that cannot be interpreted raises; a lookup with no matching
        preview is a real ``absent`` fact and returns None.
        """
        if not isinstance(itunes_id, str) or not itunes_id.strip():
            raise CatalogMappingError("itunes_id must be a non-empty string")
        try:
            payload = json.loads(self.transport.lookup(itunes_id))
        except (TypeError, ValueError) as error:
            raise CatalogMappingError(f"lookup response is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise CatalogMappingError("lookup response must be a JSON object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise CatalogMappingError("lookup response lacks a results array")
        for entry in results:
            if not isinstance(entry, dict):
                continue  # uninterpretable entries are skipped, never guessed
            if str(entry.get("trackId")) != itunes_id:
                continue
            preview_url = entry.get("previewUrl")
            if isinstance(preview_url, str) and preview_url:
                return preview_url
        return None

    def lookup_track_view_url(self, itunes_id: str) -> str | None:
        """Resolve one ``itunes_store`` trackId to Apple's official track page URL.

        P16-S4: the same strict-trackId lookup contract as ``lookup_preview_url`` --
        a result matches only when its ``trackId`` agrees exactly with the requested
        ID (never a name or position match). The returned ``trackViewUrl`` is Apple's
        own link for that exact track identity; an absent URL is a real ``absent``
        fact (None), never fabricated. Uninterpretable payloads raise.
        """
        if not isinstance(itunes_id, str) or not itunes_id.strip():
            raise CatalogMappingError("itunes_id must be a non-empty string")
        try:
            payload = json.loads(self.transport.lookup(itunes_id))
        except (TypeError, ValueError) as error:
            raise CatalogMappingError(f"lookup response is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise CatalogMappingError("lookup response must be a JSON object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise CatalogMappingError("lookup response lacks a results array")
        for entry in results:
            if not isinstance(entry, dict):
                continue  # uninterpretable entries are skipped, never guessed
            if str(entry.get("trackId")) != itunes_id:
                continue
            track_view_url = entry.get("trackViewUrl")
            if isinstance(track_view_url, str) and track_view_url:
                return track_view_url
        return None

    def parse_search_results(self, text: str) -> tuple[CatalogTrack, ...]:
        if not isinstance(text, str):
            raise CatalogMappingError("search response must be text")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as error:
            raise CatalogMappingError(f"search response is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise CatalogMappingError("search response must be a JSON object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise CatalogMappingError("search response lacks a results array")
        return tuple(self._parse_result(entry) for entry in results)

    def _parse_result(self, entry: object) -> CatalogTrack:
        if not isinstance(entry, dict):
            raise CatalogMappingError("each result entry must be an object")
        track_id = self._required_id(entry, "trackId")
        artist_id = self._required_id(entry, "artistId")
        collection_id = self._optional_id(entry, "collectionId")
        name = self._required_text(entry, "trackName", f"result {track_id}")
        artist_name = self._required_text(entry, "artistName", f"result {track_id}")
        album_name = self._optional_text(entry.get("collectionName"))
        genre = self._optional_text(entry.get("primaryGenreName"))
        duration_ms = self._optional_int(entry.get("trackTimeMillis"))
        preview_url = self._optional_text(entry.get("previewUrl"))
        url = self._optional_text(entry.get("trackViewUrl"))
        return CatalogTrack(
            catalog_id=track_id,
            name=name,
            artist_names=(artist_name,),
            album_name=album_name,
            genres=(genre,) if genre is not None else (),
            isrc=None,  # the iTunes Search API supplies no ISRC; never fabricated
            duration_ms=duration_ms,
            release_date=self._release_date(entry.get("releaseDate")),
            url=url,
            artist_catalog_ids=(artist_id,),
            album_catalog_id=collection_id,
            source_system=ITUNES_SOURCE_SYSTEM,
            preview_url=preview_url,
        )

    @staticmethod
    def _required_id(entry: dict[str, object], key: str) -> str:
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CatalogMappingError(f"result entry requires a positive integer {key}")
        return str(value)

    @staticmethod
    def _optional_id(entry: dict[str, object], key: str) -> str | None:
        value = entry.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CatalogMappingError(f"result {key} must be a positive integer when present")
        return str(value)

    @staticmethod
    def _required_text(entry: dict[str, object], key: str, label: str) -> str:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CatalogMappingError(f"{label} requires a non-empty {key}")
        return value

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise CatalogMappingError("expected an optional non-empty string")
        return value

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CatalogMappingError("expected an optional non-negative integer")
        return value

    @staticmethod
    def _release_date(value: object) -> str | None:
        """Normalize the API's ISO timestamp to the canonical day precision (``YYYY-MM-DD``).

        Unparseable values degrade to ``None`` (missing evidence), never a fabricated date.
        """
        if not isinstance(value, str) or not value:
            return None
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            return None
