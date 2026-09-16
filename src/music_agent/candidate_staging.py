"""Durable SQLite staging for non-canonical ingestion candidates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from music_agent.identity import EntityType, ExternalIdentityKey, IdentityValidationError
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    AlbumRelationState,
    ArtistRelationResolution,
    ArtistRelationState,
    CandidateValidationError,
    IngestionCandidate,
)
from music_agent.repository import _open_store_connection
from music_agent.source_observation import ObservationState, ObservedValue


class CandidateStagingError(ValueError):
    code = "staging_validation_error"


class CandidateStagingRepository:
    """Persist Candidate state separately inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CandidateStagingRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def stage_candidate(
        self, candidate: IngestionCandidate, scope_key: str | None = None
    ) -> None:
        if not isinstance(candidate, IngestionCandidate):
            raise CandidateStagingError("candidate must be an IngestionCandidate")
        if scope_key is not None and (
            not isinstance(scope_key, str) or scope_key == ""
        ):
            raise CandidateStagingError("scope_key must be a non-empty string when supplied")
        identity = candidate.external_identity
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._connection.execute(
                """SELECT payload FROM ingestion_candidates
                WHERE source_system=? AND entity_type=? AND external_id=?""",
                (identity.source_system, identity.entity_type.value, identity.external_id),
            ).fetchone()
            stored_candidate = candidate
            if existing_row is not None:
                existing = _decode_candidate_payload(identity, existing_row[0])
                stored_candidate = IngestionCandidate(
                    identity,
                    candidate.source_facts,
                    existing.artist_relation,
                    existing.album_relation,
                )
            payload = _encode_candidate_payload(stored_candidate)
            self._connection.execute(
                """INSERT INTO ingestion_candidates(
                    source_system, entity_type, external_id, payload
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_system, entity_type, external_id)
                DO UPDATE SET payload=excluded.payload""",
                (
                    identity.source_system,
                    identity.entity_type.value,
                    identity.external_id,
                    payload,
                ),
            )
            if scope_key is not None:
                self._connection.execute(
                    """INSERT INTO ingestion_candidate_scopes(
                        source_system, entity_type, external_id, scope_key
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_system, entity_type, external_id, scope_key)
                    DO NOTHING""",
                    (
                        identity.source_system,
                        identity.entity_type.value,
                        identity.external_id,
                        scope_key,
                    ),
                )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def update_relation_resolution(
        self,
        key: ExternalIdentityKey,
        *,
        artist_resolution: ArtistRelationResolution | None = None,
        album_resolution: AlbumRelationResolution | None = None,
    ) -> None:
        key = _require_key(key)
        if artist_resolution is None and album_resolution is None:
            raise CandidateStagingError("at least one relation resolution is required")
        if artist_resolution is not None and (
            not isinstance(artist_resolution, ArtistRelationResolution)
            or artist_resolution.state is ArtistRelationState.UNRESOLVED
        ):
            raise CandidateStagingError(
                "artist_resolution must be an explicit resolved Artist decision"
            )
        if album_resolution is not None and (
            not isinstance(album_resolution, AlbumRelationResolution)
            or album_resolution.state is AlbumRelationState.UNRESOLVED
        ):
            raise CandidateStagingError(
                "album_resolution must be an explicit resolved or absent Album decision"
            )

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """SELECT payload FROM ingestion_candidates
                WHERE source_system=? AND entity_type=? AND external_id=?""",
                (key.source_system, key.entity_type.value, key.external_id),
            ).fetchone()
            if row is None:
                raise CandidateStagingError("candidate does not exist")
            existing = _decode_candidate_payload(key, row[0])
            updated = IngestionCandidate(
                key,
                existing.source_facts,
                artist_resolution or existing.artist_relation,
                album_resolution or existing.album_relation,
                existing.secondary_identities,
            )
            self._connection.execute(
                """UPDATE ingestion_candidates SET payload=?
                WHERE source_system=? AND entity_type=? AND external_id=?""",
                (
                    _encode_candidate_payload(updated),
                    key.source_system,
                    key.entity_type.value,
                    key.external_id,
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def get_candidate(self, key: ExternalIdentityKey) -> IngestionCandidate | None:
        key = _require_key(key)
        row = self._connection.execute(
            """SELECT payload FROM ingestion_candidates
            WHERE source_system=? AND entity_type=? AND external_id=?""",
            (key.source_system, key.entity_type.value, key.external_id),
        ).fetchone()
        if row is None:
            return None
        return _decode_candidate_payload(key, row[0])

    def list_candidate_scopes(self, key: ExternalIdentityKey) -> tuple[str, ...]:
        key = _require_key(key)
        rows = self._connection.execute(
            """SELECT scope_key FROM ingestion_candidate_scopes
            WHERE source_system=? AND entity_type=? AND external_id=?
            ORDER BY scope_key""",
            (key.source_system, key.entity_type.value, key.external_id),
        )
        return tuple(str(row[0]) for row in rows)


def _encode_candidate_payload(candidate: IngestionCandidate) -> str:
    payload = {
        "source_facts": {
            path: _encode_observed_value(observed)
            for path, observed in candidate.source_facts.items()
        },
        "artist_relation": _encode_artist_relation(candidate.artist_relation),
        "album_relation": _encode_album_relation(candidate.album_relation),
        "secondary_identities": [
            {
                "source_system": identity.source_system,
                "entity_type": identity.entity_type.value,
                "external_id": identity.external_id,
            }
            for identity in candidate.secondary_identities
        ],
    }
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CandidateStagingError(f"candidate payload is not JSON serializable: {error}") from error


def _decode_candidate_payload(key: ExternalIdentityKey, encoded: object) -> IngestionCandidate:
    if not isinstance(encoded, str):
        raise CandidateStagingError("candidate payload must be JSON text")
    try:
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or not {
            "source_facts", "artist_relation", "album_relation"
        } <= set(payload):
            raise CandidateStagingError(
                "candidate payload must contain source_facts, artist_relation, and album_relation"
            )
        facts_payload = payload["source_facts"]
        if not isinstance(facts_payload, dict):
            raise CandidateStagingError("source_facts must be an object")
        facts = {
            path: _decode_observed_value(observed)
            for path, observed in facts_payload.items()
            if _require_string_key(path)
        }
        return IngestionCandidate(
            key,
            facts,
            _decode_artist_relation(payload["artist_relation"]),
            _decode_album_relation(payload["album_relation"]),
            _decode_secondary_identities(payload.get("secondary_identities", [])),
        )
    except CandidateStagingError:
        raise
    except (CandidateValidationError, IdentityValidationError, KeyError, TypeError, ValueError) as error:
        raise CandidateStagingError(f"invalid candidate payload: {error}") from error


def _encode_observed_value(observed: ObservedValue) -> dict[str, Any]:
    if observed.state is ObservationState.VALUE:
        return {"state": observed.state.value, "value": observed.payload}
    return {"state": observed.state.value}


def _decode_observed_value(payload: object) -> ObservedValue:
    if not isinstance(payload, dict):
        raise CandidateStagingError("ObservedValue payload must be an object")
    state_value = payload.get("state")
    try:
        state = ObservationState(state_value)
    except (TypeError, ValueError) as error:
        raise CandidateStagingError(f"unknown ObservedValue state: {state_value!r}") from error
    expected = {"state", "value"} if state is ObservationState.VALUE else {"state"}
    _require_exact_keys(payload, expected, "ObservedValue payload")
    if state is ObservationState.MISSING:
        return ObservedValue.missing()
    if state is ObservationState.NULL:
        return ObservedValue.null()
    return ObservedValue.value(payload["value"])


def _encode_artist_relation(relation: ArtistRelationResolution) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": relation.state.value}
    if relation.state is ArtistRelationState.RESOLVED_TO_ARTISTS:
        payload["canonical_ids"] = list(relation.canonical_ids)
    return payload


def _decode_artist_relation(payload: object) -> ArtistRelationResolution:
    if not isinstance(payload, dict):
        raise CandidateStagingError("artist_relation must be an object")
    state_value = payload.get("state")
    try:
        state = ArtistRelationState(state_value)
    except (TypeError, ValueError) as error:
        raise CandidateStagingError(f"unknown artist relation state: {state_value!r}") from error
    expected = {"state", "canonical_ids"} if state is ArtistRelationState.RESOLVED_TO_ARTISTS else {"state"}
    _require_exact_keys(payload, expected, "artist_relation")
    if state is ArtistRelationState.UNRESOLVED:
        return ArtistRelationResolution.unresolved()
    canonical_ids = payload["canonical_ids"]
    if not isinstance(canonical_ids, list) or any(
        not isinstance(canonical_id, str) for canonical_id in canonical_ids
    ):
        raise CandidateStagingError("artist canonical_ids must be an array of strings")
    return ArtistRelationResolution.resolved_to_artists(canonical_ids)


def _encode_album_relation(relation: AlbumRelationResolution) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": relation.state.value}
    if relation.state is AlbumRelationState.RESOLVED_TO_ALBUM:
        payload["canonical_id"] = relation.canonical_id
    return payload


def _decode_album_relation(payload: object) -> AlbumRelationResolution:
    if not isinstance(payload, dict):
        raise CandidateStagingError("album_relation must be an object")
    state_value = payload.get("state")
    try:
        state = AlbumRelationState(state_value)
    except (TypeError, ValueError) as error:
        raise CandidateStagingError(f"unknown album relation state: {state_value!r}") from error
    expected = {"state", "canonical_id"} if state is AlbumRelationState.RESOLVED_TO_ALBUM else {"state"}
    _require_exact_keys(payload, expected, "album_relation")
    if state is AlbumRelationState.UNRESOLVED:
        return AlbumRelationResolution.unresolved()
    if state is AlbumRelationState.RESOLVED_ABSENT:
        return AlbumRelationResolution.resolved_absent()
    canonical_id = payload["canonical_id"]
    if not isinstance(canonical_id, str):
        raise CandidateStagingError("album canonical_id must be a string")
    return AlbumRelationResolution.resolved_to_album(canonical_id)


def _decode_secondary_identities(payload: object) -> tuple[ExternalIdentityKey, ...]:
    if not isinstance(payload, list):
        raise CandidateStagingError("secondary_identities must be an array")
    identities: list[ExternalIdentityKey] = []
    for entry in payload:
        if not isinstance(entry, dict) or set(entry) != {
            "source_system", "entity_type", "external_id"
        }:
            raise CandidateStagingError(
                "each secondary identity must contain source_system, entity_type, and external_id"
            )
        try:
            identities.append(
                ExternalIdentityKey(
                    str(entry["source_system"]),
                    EntityType(entry["entity_type"]),
                    str(entry["external_id"]),
                )
            )
        except (TypeError, ValueError) as error:
            raise CandidateStagingError(f"invalid secondary identity: {error}") from error
    return tuple(identities)


def _require_key(key: object) -> ExternalIdentityKey:
    if not isinstance(key, ExternalIdentityKey):
        raise CandidateStagingError("key must be an ExternalIdentityKey")
    return key


def _require_exact_keys(payload: object, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise CandidateStagingError(f"{label} must contain exactly {sorted(expected)}")


def _require_string_key(value: object) -> bool:
    if not isinstance(value, str):
        raise CandidateStagingError("source fact paths must be strings")
    return True
