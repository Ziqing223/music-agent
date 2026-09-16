"""P11.3: catalog song -> user library mutation, readback, and identity reconciliation.

Three capabilities stay strictly separate here, mirroring the phase's capability model:

- **preview** (see ``catalog_preview``) -- play the 30-second audio preview, no library state changes;
- **playback** -- the existing P10.12 playback boundary, usable only once a song has a
  Music.app persistent ID binding;
- **library mutation** -- this module: add a Catalog Song to the user's Apple Music library
  through the MusicKit user-scoped endpoint, read the resulting state back, and reconcile the
  library identity to the SAME canonical Track.

Safety discipline (the project's mutation/readback/recovery rules):

- baseline -> mutation -> readback -> reconciliation, never a bare mutation;
- an ambiguous external outcome is never classified as deterministic success: a missing or
  mismatched ``playParams.catalogId``, zero or multiple matching library entries, or a failed
  HTTP call all fail closed into ``AMBIGUOUS`` / ``FAILED``;
- reconciliation binds only through the sealed durable path (one atomic canonical save): the
  Music.app persistent ID and the ISRC land on the SAME canonical Track that already holds the
  Catalog Song ID -- there is never T1 + T2 for one proven song;
- recovery: the add is idempotent (adding an already-added song is a no-op upstream); a failed
  reconciliation rolls the canonical save back, so no partial bindings survive. Deleting from
  the library is destructive and deliberately out of scope.

Credentials: the MusicKit user-scoped mutation needs both the developer token and the
Music User Token; a missing user token raises an actionable error, never a fabricated one.
Catalog read access (search) and private-library mutation have different authorization
requirements, and this module only ever carries the mutation path.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from music_agent.apple_music_catalog import (
    DEFAULT_DEVELOPER_TOKEN_ENV,
    DEFAULT_STOREFRONT,
    DEFAULT_USER_TOKEN_ENV,
    CatalogMappingError,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.library_sync import LIBRARY_TRACKS_SCOPE
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import SourcePresence

LIBRARY_ADD_URL = "https://api.music.apple.com/v1/me/library"
LIBRARY_SEARCH_URL_TEMPLATE = (
    "https://api.music.apple.com/v1/me/library/search"
    "?term={term}&types=library-songs&limit={limit}"
)


class CatalogLibraryCredentialsError(RuntimeError):
    """The transport cannot authenticate a private-library mutation."""

    code = "catalog_library_credentials_missing"


class CatalogLibraryTransportError(RuntimeError):
    """The library transport failed to produce a usable response."""

    code = "catalog_library_transport_failed"


class CatalogLibraryUnavailableError(RuntimeError):
    """The library-mutation boundary is not wired; the capability stays unavailable."""

    code = "catalog_library_unavailable"


class LibraryMutationTransport(Protocol):
    """Injected boundary: one add mutation and one library readback, raw JSON in / out."""

    def add_song(self, catalog_id: str) -> str: ...

    def search_library_songs(self, term: str, limit: int = 25) -> str: ...


class MusicKitLibraryTransport:
    """User-scoped MusicKit transport for private-library mutation and readback."""

    def __init__(
        self,
        *,
        developer_token: str | None = None,
        user_token: str | None = None,
        storefront: str = DEFAULT_STOREFRONT,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.developer_token = developer_token or os.environ.get(DEFAULT_DEVELOPER_TOKEN_ENV)
        self.user_token = user_token or os.environ.get(DEFAULT_USER_TOKEN_ENV) or None
        self.storefront = storefront
        self.timeout_seconds = timeout_seconds

    def _require_credentials(self) -> None:
        if not self.developer_token:
            raise CatalogLibraryCredentialsError(
                f"Apple Music developer token is required; set {DEFAULT_DEVELOPER_TOKEN_ENV} "
                "or pass developer_token explicitly"
            )
        if not self.user_token:
            raise CatalogLibraryCredentialsError(
                f"Music User Token is required for private-library mutation; set "
                f"{DEFAULT_USER_TOKEN_ENV} or pass user_token explicitly"
            )

    def _request(self, url: str, method: str, body: bytes | None = None) -> str:
        self._require_credentials()
        headers = {
            "Authorization": f"Bearer {self.developer_token}",
            "Music-User-Token": self.user_token,
        }
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, headers=headers, method=method, data=body)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise CatalogLibraryTransportError(
                f"library request failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CatalogLibraryTransportError(f"library request failed: {error}") from error
        if not isinstance(raw, bytes):
            raise CatalogLibraryTransportError("library request returned a non-bytes body")
        return raw.decode("utf-8")

    def add_song(self, catalog_id: str) -> str:
        if not isinstance(catalog_id, str) or not catalog_id:
            raise CatalogMappingError("catalog_id must be a non-empty string")
        body = urllib.parse.urlencode({"ids[songs]": catalog_id}).encode("utf-8")
        return self._request(LIBRARY_ADD_URL, "POST", body)

    def search_library_songs(self, term: str, limit: int = 25) -> str:
        if not isinstance(term, str) or not term.strip():
            raise CatalogMappingError("term must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise CatalogMappingError("limit must be a positive integer")
        url = LIBRARY_SEARCH_URL_TEMPLATE.format(
            term=urllib.parse.quote(term),
            limit=limit,
        )
        return self._request(url, "GET")


@dataclass(frozen=True, slots=True)
class LibrarySongEvidence:
    """One readback-verified library representation of a catalog song.

    ``library_song_id`` is the MusicKit library-song id (the Music.app persistent ID the
    library enumerates; the live format agreement is verified at the live gate, never assumed
    here). ``isrc`` is present only when the readback reliably supplied it.
    """

    catalog_id: str
    library_song_id: str
    isrc: str | None


@dataclass(frozen=True, slots=True)
class LibraryAddOutcome:
    """The parsed result of one add mutation, fail-closed on any ambiguity."""

    catalog_id: str
    status: str  # "added" | "ambiguous"
    evidence: LibrarySongEvidence | None = None
    error: str | None = None


def parse_add_response(catalog_id: str, text: str) -> LibraryAddOutcome:
    """Parse a ``POST /v1/me/library`` response, requiring exact catalog-id agreement.

    Zero or multiple data entries, a mismatched ``playParams.catalogId``, or any malformed
    shape is ``AMBIGUOUS`` -- never deterministic success.
    """
    if not isinstance(text, str):
        raise CatalogMappingError("add response must be text")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise CatalogMappingError(f"add response is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CatalogMappingError("add response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1:
        return LibraryAddOutcome(
            catalog_id, "ambiguous", error=f"add response carried {0 if not isinstance(data, list) else len(data)} entries"
        )
    return _evidence_from_library_entry(catalog_id, data[0], "add")


def parse_library_search(catalog_id: str, term: str, text: str) -> LibraryAddOutcome | None:
    """Read the user library back for one catalog song by exact catalog-id relationship.

    Returns ``None`` when the song is not present (a real ``not_in_library`` fact) and an
    ``AMBIGUOUS`` outcome when the payload cannot be interpreted or carries multiple matches.
    """
    if not isinstance(text, str):
        raise CatalogMappingError("search response must be text")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise CatalogMappingError(f"search response is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CatalogMappingError("search response must be a JSON object")
    try:
        entries = payload["results"]["library-songs"]["data"]
    except (KeyError, TypeError) as error:
        raise CatalogMappingError("search response lacks results.library-songs.data") from error
    if not isinstance(entries, list):
        raise CatalogMappingError("results.library-songs.data must be an array")
    matches = [
        entry for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("attributes"), dict)
        and isinstance(entry["attributes"].get("playParams"), dict)
        and entry["attributes"]["playParams"].get("catalogId") == catalog_id
    ]
    if not matches:
        return None
    if len(matches) > 1:
        return LibraryAddOutcome(
            catalog_id, "ambiguous", error=f"library readback found {len(matches)} matches for catalog id"
        )
    return _evidence_from_library_entry(catalog_id, matches[0], "readback")


def _evidence_from_library_entry(
    catalog_id: str, entry: object, label: str
) -> LibraryAddOutcome:
    if not isinstance(entry, dict):
        return LibraryAddOutcome(catalog_id, "ambiguous", error=f"{label} entry must be an object")
    library_song_id = entry.get("id")
    if not isinstance(library_song_id, str) or not library_song_id:
        return LibraryAddOutcome(catalog_id, "ambiguous", error=f"{label} entry requires a non-empty id")
    attributes = entry.get("attributes")
    if not isinstance(attributes, dict):
        return LibraryAddOutcome(catalog_id, "ambiguous", error=f"{label} entry requires attributes")
    play_params = attributes.get("playParams")
    matched_catalog_id = play_params.get("catalogId") if isinstance(play_params, dict) else None
    if matched_catalog_id != catalog_id:
        return LibraryAddOutcome(
            catalog_id, "ambiguous", error=f"{label} catalog id mismatch: {matched_catalog_id!r}"
        )
    isrc = attributes.get("isrc")
    if isrc is not None and (not isinstance(isrc, str) or not isrc):
        return LibraryAddOutcome(catalog_id, "ambiguous", error=f"{label} entry carries an invalid isrc")
    return LibraryAddOutcome(
        catalog_id,
        "added",
        evidence=LibrarySongEvidence(catalog_id, library_song_id, isrc),
    )


@dataclass(frozen=True, slots=True)
class CatalogLibraryReconciliation:
    """The typed outcome of add -> readback -> reconcile for one canonical catalog Track."""

    catalog_id: str
    canonical_id: str
    add_status: str  # "added" | "ambiguous" | "failed"
    readback_status: str  # "matched" | "absent" | "ambiguous" | "failed"
    bound_persistent_id: str | None = None
    bound_isrc: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.add_status == "added" and self.readback_status == "matched"


def add_catalog_song_to_library(
    repository: CanonicalRepository,
    transport: LibraryMutationTransport,
    catalog_id: str,
    canonical_id: str,
    *,
    term: str | None = None,
) -> CatalogLibraryReconciliation:
    """Add one canonical catalog Track to the user library and reconcile identity.

    Baseline -> mutation -> readback -> reconciliation. The readback match is the exact
    catalog-id relationship, never a name match; the search ``term`` only scopes the search
    and defaults to the canonical Track's name. Bindings land atomically through one canonical
    save -- a conflict rolls everything back and reports ``ambiguous``.
    """
    if not isinstance(repository, CanonicalRepository):
        raise CatalogMappingError("repository must be a CanonicalRepository")
    if not callable(getattr(transport, "add_song", None)) or not callable(
        getattr(transport, "search_library_songs", None)
    ):
        raise CatalogMappingError("transport must provide add_song and search_library_songs")

    # Baseline: the canonical Track must already hold the Catalog Song ID.
    catalog_key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, catalog_id)
    if repository.lookup_external_identity(catalog_key) != canonical_id:
        raise CatalogMappingError(
            f"canonical Track {canonical_id} is not bound to catalog id {catalog_id}"
        )
    model = repository.load_model()
    track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
    if track is None:
        raise CatalogMappingError(f"canonical Track {canonical_id} does not exist")
    if term is None:
        term = track["name"]

    # Mutation.
    try:
        add_outcome = parse_add_response(catalog_id, transport.add_song(catalog_id))
    except (CatalogMappingError, CatalogLibraryTransportError, CatalogLibraryCredentialsError) as error:
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, "failed", "failed", error=str(error)
        )
    if add_outcome.status != "added":
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, "ambiguous", "absent",
            error=add_outcome.error,
        )

    # Readback (independent of the add response).
    try:
        readback = parse_library_search(catalog_id, term, transport.search_library_songs(term))
    except (CatalogMappingError, CatalogLibraryTransportError, CatalogLibraryCredentialsError) as error:
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, add_outcome.status, "failed", error=str(error)
        )
    if readback is None:
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, "added", "absent",
            error="library readback found no entry for the catalog id",
        )
    if readback.status != "added" or readback.evidence is None:
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, "added", "ambiguous", error=readback.error
        )

    # Reconciliation: bind the library identity to the SAME canonical Track, atomically.
    # The readback proved library membership, so the library_tracks presence fact lands in the
    # same transaction as the bindings.
    evidence = readback.evidence
    updated = _update_track_external_ids(
        track,
        persistent_id=evidence.library_song_id,
        isrc=evidence.isrc,
    )
    saved_model = dict(model)
    saved_model["tracks"] = [
        updated if item["id"] == canonical_id else item for item in model["tracks"]
    ]
    try:
        repository.save_model_with_source_presence(
            saved_model,
            [
                SourcePresenceRecord(
                    "apple_music",
                    EntityType.TRACK,
                    canonical_id,
                    LIBRARY_TRACKS_SCOPE,
                    SourcePresence.PRESENT,
                )
            ],
        )
    except Exception as error:  # identity conflict / validation: fail closed, no partial bindings
        return CatalogLibraryReconciliation(
            catalog_id, canonical_id, "added", "ambiguous",
            error=f"identity reconciliation failed: {error}",
        )
    return CatalogLibraryReconciliation(
        catalog_id,
        canonical_id,
        "added",
        "matched",
        bound_persistent_id=evidence.library_song_id,
        bound_isrc=evidence.isrc,
    )


def _update_track_external_ids(
    track: dict, *, persistent_id: str, isrc: str | None
) -> dict:
    updated = dict(track)
    external_ids = dict(track["external_ids"])
    external_ids["apple_music_persistent_id"] = persistent_id
    if isrc is not None:
        external_ids["isrc"] = isrc
    updated["external_ids"] = external_ids
    return updated
