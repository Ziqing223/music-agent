"""Read-only Apple Music Catalog search boundary (P11.1).

One narrow adapter for discovering songs in the Apple Music Catalog -- songs that are NOT yet in
the user's local Music.app library. The module separates three concerns, each its own type:

``CatalogTransport`` / :class:`MusicKitTransport`
    Transport and authentication. The MusicKit implementation speaks HTTPS to the Apple Music
    Web API and reads its credentials from the environment (never from committed secrets):
    ``MUSIC_AGENT_APPLE_MUSIC_DEVELOPER_TOKEN`` is required, ``MUSIC_AGENT_APPLE_MUSIC_USER_TOKEN``
    is optional (catalog search itself needs only the developer token). A missing developer token
    raises :class:`CatalogCredentialsError` with an actionable message.

``CatalogTrack``
    The catalog domain representation: the Catalog Song ID plus structured metadata later slices
    need (name, artists, album, genres, ISRC, duration, release date, catalog URL).

``AppleMusicCatalogAdapter``
    Response parsing. It validates the MusicKit payload shape and maps it to ``CatalogTrack``
    values, failing closed on unknown shapes.

Catalog read access and private-library mutation have different authorization requirements; this
module only ever reads the public catalog and never carries a mutation path. Preview, playback,
and library-add are separate capabilities (P11.3).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


DEFAULT_DEVELOPER_TOKEN_ENV = "MUSIC_AGENT_APPLE_MUSIC_DEVELOPER_TOKEN"
DEFAULT_USER_TOKEN_ENV = "MUSIC_AGENT_APPLE_MUSIC_USER_TOKEN"
DEFAULT_STOREFRONT = "us"
DEFAULT_TIMEOUT_SECONDS = 10.0

CATALOG_SEARCH_URL_TEMPLATE = (
    "https://api.music.apple.com/v1/catalog/{storefront}/search"
    "?term={term}&types=songs&limit={limit}"
)


class CatalogCredentialsError(RuntimeError):
    """The transport cannot authenticate against the Apple Music Web API."""

    code = "catalog_credentials_missing"


class CatalogTransportError(RuntimeError):
    """The catalog transport failed to produce a usable response."""

    code = "catalog_transport_failed"


class CatalogMappingError(ValueError):
    """The catalog payload could not be mapped to catalog domain values."""

    code = "validation_error"


class CatalogTransport(Protocol):
    """Injected boundary: one raw catalog search returning the response body as text."""

    def search(self, term: str, limit: int) -> str: ...


class MusicKitTransport:
    """HTTPS transport for the Apple Music Web API with environment-injected credentials."""

    def __init__(
        self,
        *,
        developer_token: str | None = None,
        user_token: str | None = None,
        storefront: str = DEFAULT_STOREFRONT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.developer_token = developer_token or os.environ.get(DEFAULT_DEVELOPER_TOKEN_ENV)
        self.user_token = user_token or os.environ.get(DEFAULT_USER_TOKEN_ENV) or None
        self.storefront = storefront
        self.timeout_seconds = timeout_seconds

    def _require_credentials(self) -> None:
        if not self.developer_token:
            raise CatalogCredentialsError(
                f"Apple Music developer token is required; set {DEFAULT_DEVELOPER_TOKEN_ENV} "
                "(a signed MusicKit JWT) or pass developer_token explicitly"
            )

    def search(self, term: str, limit: int) -> str:
        if not isinstance(term, str) or not term.strip():
            raise CatalogMappingError("term must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise CatalogMappingError("limit must be a positive integer")
        self._require_credentials()
        url = CATALOG_SEARCH_URL_TEMPLATE.format(
            storefront=urllib.parse.quote(self.storefront, safe=""),
            term=urllib.parse.quote(term),
            limit=limit,
        )
        headers = {
            "Authorization": f"Bearer {self.developer_token}",
        }
        if self.user_token:
            headers["Music-User-Token"] = self.user_token
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            raise CatalogTransportError(
                f"catalog search failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CatalogTransportError(f"catalog search failed: {error}") from error
        if not isinstance(body, bytes):
            raise CatalogTransportError("catalog search returned a non-bytes body")
        return body.decode("utf-8")


@dataclass(frozen=True, slots=True)
class CatalogTrack:
    """One discovered catalog song with the structured metadata later slices need.

    ``catalog_id`` is the source catalog's song ID; ``source_system`` names which external
    identity namespace that ID lives in (``apple_music_catalog`` for MusicKit results,
    ``itunes_store`` for iTunes Search results -- the two are never inferred equal).
    ``isrc`` is present only when the response reliably supplies it.
    ``artist_catalog_ids`` / ``album_catalog_id`` are the authoritative source-catalog
    Artist / Album identities; they are present only when the response supplies them. Display
    names (``artist_names``, ``album_name``) are presentation-only and never resolve a
    relation. ``preview_url`` is the source's preview audio URL, carried for the preview
    slice (T4); this module never opens it.
    """

    catalog_id: str
    name: str
    artist_names: tuple[str, ...]
    album_name: str | None
    genres: tuple[str, ...]
    isrc: str | None
    duration_ms: int | None
    release_date: str | None
    url: str | None
    artist_catalog_ids: tuple[str, ...] = ()
    album_catalog_id: str | None = None
    source_system: str = "apple_music_catalog"
    preview_url: str | None = None


class AppleMusicCatalogAdapter:
    """Parse MusicKit search payloads into validated ``CatalogTrack`` values."""

    def __init__(self, transport: CatalogTransport) -> None:
        if not callable(getattr(transport, "search", None)):
            raise CatalogMappingError("transport must provide search(term, limit)")
        self.transport = transport

    def search(self, term: str, limit: int = 25) -> tuple[CatalogTrack, ...]:
        return self.parse_search_results(self.transport.search(term, limit))

    def parse_search_results(self, text: str) -> tuple[CatalogTrack, ...]:
        if not isinstance(text, str):
            raise CatalogMappingError("search response must be text")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as error:
            raise CatalogMappingError(f"search response is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise CatalogMappingError("search response must be a JSON object")
        try:
            songs = payload["results"]["songs"]["data"]
        except (KeyError, TypeError) as error:
            raise CatalogMappingError("search response lacks results.songs.data") from error
        if not isinstance(songs, list):
            raise CatalogMappingError("results.songs.data must be an array")
        tracks: list[CatalogTrack] = []
        for entry in songs:
            tracks.append(self._parse_song(entry))
        return tuple(tracks)

    def _parse_song(self, entry: object) -> CatalogTrack:
        if not isinstance(entry, dict):
            raise CatalogMappingError("each song entry must be an object")
        catalog_id = entry.get("id")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise CatalogMappingError("song entry requires a non-empty id")
        attributes = entry.get("attributes")
        if not isinstance(attributes, dict):
            raise CatalogMappingError(f"song {catalog_id} requires an attributes object")
        name = attributes.get("name")
        if not isinstance(name, str) or not name:
            raise CatalogMappingError(f"song {catalog_id} requires a non-empty name")
        artist_names = self._string_list(attributes.get("artistName"))
        if not artist_names:
            raise CatalogMappingError(f"song {catalog_id} requires a non-empty artistName")
        genres = self._string_list(attributes.get("genreNames"))
        isrc = attributes.get("isrc")
        duration_ms = attributes.get("durationInMillis")
        release_date = attributes.get("releaseDate")
        url = attributes.get("url")
        return CatalogTrack(
            catalog_id=catalog_id,
            name=name,
            artist_names=artist_names,
            album_name=self._optional_string(attributes.get("albumName")),
            genres=genres,
            isrc=self._optional_string(isrc),
            duration_ms=self._optional_int(duration_ms),
            release_date=self._optional_string(release_date),
            url=self._optional_string(url),
            artist_catalog_ids=self._relationship_ids(entry, "artists"),
            album_catalog_id=self._single_relationship_id(entry, "albums"),
        )

    def _relationship_ids(self, entry: dict[str, object], name: str) -> tuple[str, ...]:
        """Authoritative relationship IDs for ``name`` (e.g. ``artists``), absent -> ().

        Present-but-malformed relationship data fails closed; a payload without the
        relationship at all simply carries no authoritative identity evidence.
        """
        data = self._relationship_data(entry, name)
        if data is None:
            return ()
        ids: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                raise CatalogMappingError(f"{name} relationship entries must be objects")
            item_type = item.get("type")
            if item_type is not None and item_type != name:
                continue  # e.g. a nested "songs" entry inside artists: not identity evidence
            identity = item.get("id")
            if not isinstance(identity, str) or not identity:
                raise CatalogMappingError(f"{name} relationship entries require a non-empty id")
            ids.append(identity)
        return tuple(ids)

    def _single_relationship_id(self, entry: dict[str, object], name: str) -> str | None:
        ids = self._relationship_ids(entry, name)
        return ids[0] if ids else None

    @staticmethod
    def _relationship_data(entry: dict[str, object], name: str) -> list[object] | None:
        relationships = entry.get("relationships")
        if relationships is None:
            return None
        if not isinstance(relationships, dict):
            raise CatalogMappingError("song relationships must be an object")
        relation = relationships.get(name)
        if relation is None:
            return None
        if not isinstance(relation, dict):
            raise CatalogMappingError(f"relationship {name} must be an object")
        data = relation.get("data")
        if data is None:
            return None
        if not isinstance(data, list):
            raise CatalogMappingError(f"relationship {name} data must be an array")
        return data

    @staticmethod
    def _string_list(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list) and all(isinstance(entry, str) and entry for entry in value):
            return tuple(value)
        raise CatalogMappingError("expected a string or array of strings")

    @staticmethod
    def _optional_string(value: object) -> str | None:
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
