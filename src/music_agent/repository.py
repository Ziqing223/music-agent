"""SQLite persistence for validated canonical music models."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from music_agent.identity import (
    EXTERNAL_ID_FIXTURE_KEYS,
    SCALAR_SOURCE_SYSTEMS,
    EntityType,
    ExternalIdentityKey,
    IdentityConflictError,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.source_observation import SourcePresence
from music_agent.validation import validate_fixture


CURRENT_SCHEMA_VERSION = 19
MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
    (6, "0006_pending_write_intent_requirements.sql"),
    (7, "0007_capability_probes.sql"),
    (8, "0008_capability_probe_recovery_attempts.sql"),
    (9, "0009_capability_verification_evidence.sql"),
    (10, "0010_write_ambiguous_outcome.sql"),
    (11, "0011_preference_persistence.sql"),
    (12, "0012_recommendation_history.sql"),
    (13, "0013_feedback_history.sql"),
    (14, "0014_learning_applications.sql"),
    (15, "0015_agent_requests.sql"),
    (16, "0016_runtime_task_runs.sql"),
    (17, "0017_catalog_identity.sql"),
    (18, "0018_itunes_store.sql"),
    (19, "0019_catalog_track_state.sql"),
)


def _open_store_connection(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(database_path), isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        _apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _apply_migrations(connection: sqlite3.Connection) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    applied = set()
    if exists:
        applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, filename in MIGRATIONS:
        if version in applied:
            continue
        sql = resources.files("music_agent.migrations").joinpath(filename).read_text(encoding="utf-8")
        script = (
            f"BEGIN IMMEDIATE;\n{sql}\n"
            f"INSERT INTO schema_migrations(version) VALUES ({version});\nCOMMIT;"
        )
        try:
            connection.executescript(script)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise


@dataclass(frozen=True, slots=True)
class SourcePresenceRecord:
    source_system: str
    entity_type: EntityType
    canonical_id: str
    scope_key: str
    presence: SourcePresence

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise IdentityValidationError("source_system must be a non-empty string")
        validate_canonical_id(self.entity_type, self.canonical_id)
        if not isinstance(self.scope_key, str) or self.scope_key == "":
            raise IdentityValidationError("scope_key must be a non-empty string")
        if not isinstance(self.presence, SourcePresence) or self.presence not in (
            SourcePresence.PRESENT,
            SourcePresence.CONFIRMED_DELETED,
        ):
            raise IdentityValidationError("only confirmed source presence can be persisted")


class CanonicalRepository:
    """Domain-facing repository backed by one SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CanonicalRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0])

    @property
    def foreign_keys_enabled(self) -> bool:
        return bool(self._connection.execute("PRAGMA foreign_keys").fetchone()[0])

    def counts(self) -> dict[str, int]:
        tables = ("canonical_entities", "external_identity_bindings", "track_artists", "album_artists", "playlist_memberships")
        return {table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

    def save_model(self, model: dict[str, Any]) -> None:
        self.save_model_with_source_presence(model, ())

    def save_model_with_source_presence(
        self,
        model: dict[str, Any],
        presence_updates: tuple[SourcePresenceRecord, ...] | list[SourcePresenceRecord],
    ) -> None:
        validate_fixture(model)
        updates = tuple(presence_updates)
        if any(not isinstance(update, SourcePresenceRecord) for update in updates):
            raise IdentityValidationError("presence updates must be SourcePresenceRecord values")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._save_model_contents(model)
            for update in updates:
                self._save_source_presence(update)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _commit_staged_track_promotion(
        self,
        model: dict[str, Any],
        key: ExternalIdentityKey,
        canonical_id: str,
    ) -> tuple[str, ...]:
        """Atomically persist one promoted Track and retire its staged Candidate."""
        validate_fixture(model)
        if not isinstance(key, ExternalIdentityKey):
            raise IdentityValidationError("key must be an ExternalIdentityKey")
        if key.source_system not in (
            "apple_music", "apple_music_catalog", "itunes_store"
        ) or key.entity_type is not EntityType.TRACK:
            raise IdentityValidationError("promotion supports only Apple Music, Catalog, or iTunes Store Track candidates")
        validate_canonical_id(EntityType.TRACK, canonical_id)
        self._require_promoted_track_match(model, key, canonical_id)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            candidate = self._connection.execute(
                """SELECT 1 FROM ingestion_candidates
                WHERE source_system=? AND entity_type=? AND external_id=?""",
                (key.source_system, key.entity_type.value, key.external_id),
            ).fetchone()
            if candidate is None:
                raise IdentityValidationError("staged Candidate does not exist")
            existing = self.lookup_external_identity(key)
            if existing is not None:
                raise IdentityConflictError(key, existing, canonical_id)
            self._save_model_contents(model)
            scopes = self._commit_promotion_entry_contents(key, canonical_id)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return scopes

    def _require_promoted_track_match(
        self,
        model: dict[str, Any],
        key: ExternalIdentityKey,
        canonical_id: str,
    ) -> None:
        matching_tracks = [track for track in model["tracks"] if track["id"] == canonical_id]
        if len(matching_tracks) != 1:
            raise IdentityValidationError("promoted model must contain exactly one target Track")
        if (
            matching_tracks[0]["external_ids"][EXTERNAL_ID_FIXTURE_KEYS[key.source_system]]
            != key.external_id
        ):
            raise IdentityValidationError("promoted Track external identity does not match Candidate")

    def _commit_promotion_entry_contents(
        self,
        key: ExternalIdentityKey,
        canonical_id: str,
    ) -> tuple[str, ...]:
        """Transfer Candidate scopes to source presence and retire the Candidate.

        Runs inside an open promotion transaction whose model save has already
        happened -- the shared retirement tail of the sealed single-track promotion
        and the batched catalog discovery commit (P20 Performance Fix 01).
        """
        scopes = tuple(
            str(row[0])
            for row in self._connection.execute(
                """SELECT scope_key FROM ingestion_candidate_scopes
                WHERE source_system=? AND entity_type=? AND external_id=?
                ORDER BY scope_key""",
                (key.source_system, key.entity_type.value, key.external_id),
            )
        )
        if self.lookup_external_identity(key) != canonical_id:
            raise IdentityValidationError("promotion did not create the expected binding")
        for scope_key in scopes:
            self._save_source_presence(SourcePresenceRecord(
                key.source_system,
                key.entity_type,
                canonical_id,
                scope_key,
                SourcePresence.PRESENT,
            ))
        self._connection.execute(
            """DELETE FROM ingestion_candidate_scopes
            WHERE source_system=? AND entity_type=? AND external_id=?""",
            (key.source_system, key.entity_type.value, key.external_id),
        )
        retired = self._connection.execute(
            """DELETE FROM ingestion_candidates
            WHERE source_system=? AND entity_type=? AND external_id=?""",
            (key.source_system, key.entity_type.value, key.external_id),
        )
        if retired.rowcount != 1:
            raise IdentityValidationError("staged Candidate retirement failed")
        return scopes

    def commit_catalog_discovery(
        self,
        model: dict[str, Any],
        promotions: tuple[tuple[ExternalIdentityKey, str], ...] | list[tuple[ExternalIdentityKey, str]],
    ) -> None:
        """Atomically persist one batched catalog discovery (P20 Performance Fix 01).

        One transaction serves the whole ingest batch: the canonical model is written
        exactly once, and every promoted Track retires its staged Candidate with scopes
        transferred to source presence. Per-entry durable semantics match
        ``_commit_staged_track_promotion``; all entries are validated before any write
        and any failure rolls the whole batch back, so a malformed entry can never
        leave a half-written store (fail closed).
        """
        entries = tuple(promotions)
        for key, canonical_id in entries:
            if not isinstance(key, ExternalIdentityKey):
                raise IdentityValidationError("key must be an ExternalIdentityKey")
            if key.source_system not in (
                "apple_music", "apple_music_catalog", "itunes_store"
            ) or key.entity_type is not EntityType.TRACK:
                raise IdentityValidationError("promotion supports only Apple Music, Catalog, or iTunes Store Track candidates")
            validate_canonical_id(EntityType.TRACK, canonical_id)
        validate_fixture(model)
        for key, canonical_id in entries:
            self._require_promoted_track_match(model, key, canonical_id)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for key, canonical_id in entries:
                candidate = self._connection.execute(
                    """SELECT 1 FROM ingestion_candidates
                    WHERE source_system=? AND entity_type=? AND external_id=?""",
                    (key.source_system, key.entity_type.value, key.external_id),
                ).fetchone()
                if candidate is None:
                    raise IdentityValidationError("staged Candidate does not exist")
                existing = self.lookup_external_identity(key)
                if existing is not None and existing != canonical_id:
                    raise IdentityConflictError(key, existing, canonical_id)
            self._save_model_contents(model)
            for key, canonical_id in entries:
                self._commit_promotion_entry_contents(key, canonical_id)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def get_source_presence(
        self,
        source_system: str,
        entity_type: EntityType,
        canonical_id: str,
        scope_key: str,
    ) -> SourcePresence | None:
        probe = SourcePresenceRecord(
            source_system, entity_type, canonical_id, scope_key, SourcePresence.PRESENT
        )
        row = self._connection.execute(
            """SELECT presence FROM source_entity_presence
            WHERE source_system=? AND entity_type=? AND canonical_id=? AND scope_key=?""",
            (probe.source_system, probe.entity_type.value, probe.canonical_id, probe.scope_key),
        ).fetchone()
        return None if row is None else SourcePresence(row[0])

    def list_source_presence(
        self, source_system: str, entity_type: EntityType, scope_key: str
    ) -> tuple[SourcePresenceRecord, ...]:
        if not isinstance(source_system, str) or source_system == "":
            raise IdentityValidationError("source_system must be a non-empty string")
        if not isinstance(entity_type, EntityType):
            raise IdentityValidationError("entity_type must be an EntityType")
        if not isinstance(scope_key, str) or scope_key == "":
            raise IdentityValidationError("scope_key must be a non-empty string")
        rows = self._connection.execute(
            """SELECT canonical_id, presence FROM source_entity_presence
            WHERE source_system=? AND entity_type=? AND scope_key=? ORDER BY canonical_id""",
            (source_system, entity_type.value, scope_key),
        )
        return tuple(
            SourcePresenceRecord(source_system, entity_type, row[0], scope_key, SourcePresence(row[1]))
            for row in rows
        )

    def list_external_identity_bindings(
        self, source_system: str, entity_type: EntityType
    ) -> tuple[tuple[ExternalIdentityKey, str], ...]:
        if not isinstance(source_system, str) or source_system == "":
            raise IdentityValidationError("source_system must be a non-empty string")
        if not isinstance(entity_type, EntityType):
            raise IdentityValidationError("entity_type must be an EntityType")
        rows = self._connection.execute(
            """SELECT external_id, canonical_id FROM external_identity_bindings
            WHERE source_system=? AND entity_type=? ORDER BY external_id""",
            (source_system, entity_type.value),
        )
        return tuple(
            (ExternalIdentityKey(source_system, entity_type, row[0]), str(row[1])) for row in rows
        )

    def load_model(self) -> dict[str, Any]:
        model = {
            "tracks": [self._load_track(row) for row in self._rows("tracks")],
            "artists": [self._load_named_entity(row, EntityType.ARTIST) for row in self._rows("artists")],
            "albums": [self._load_album(row) for row in self._rows("albums")],
            "playlists": [self._load_named_entity(row, EntityType.PLAYLIST) for row in self._rows("playlists")],
            "playlist_memberships": [self._load_membership(row) for row in self._rows("playlist_memberships")],
        }
        validate_fixture(model)
        return model

    def bind_external_identity(self, key: ExternalIdentityKey, canonical_id: str) -> str:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            result = self._bind_external_identity(key, canonical_id)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return result

    def lookup_external_identity(self, key: ExternalIdentityKey) -> str | None:
        if not isinstance(key, ExternalIdentityKey):
            raise IdentityValidationError("key must be an ExternalIdentityKey")
        row = self._connection.execute(
            "SELECT canonical_id FROM external_identity_bindings WHERE source_system = ? AND entity_type = ? AND external_id = ?",
            (key.source_system, key.entity_type.value, key.external_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def get_entity_type(self, canonical_id: str) -> EntityType | None:
        """Return the durable entity type reserved for a canonical ID, or ``None`` if unreserved."""
        if not isinstance(canonical_id, str) or canonical_id == "":
            return None
        row = self._connection.execute(
            "SELECT entity_type FROM canonical_entities WHERE id = ?", (canonical_id,)
        ).fetchone()
        return None if row is None else EntityType(row[0])

    def get_bound_external_id(
        self, source_system: str, entity_type: EntityType, canonical_id: str
    ) -> str | None:
        """Return the durable external ID bound to a canonical ID, or ``None`` if unbound."""
        if not isinstance(canonical_id, str) or canonical_id == "":
            return None
        row = self._connection.execute(
            """SELECT external_id FROM external_identity_bindings
            WHERE source_system = ? AND entity_type = ? AND canonical_id = ?""",
            (source_system, entity_type.value, canonical_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def _save_model_contents(self, model: dict[str, Any]) -> None:
        self._reserve_model_ids(model)
        self._save_artists(model["artists"])
        self._save_albums(model["albums"])
        self._save_tracks(model["tracks"])
        self._save_playlists(model["playlists"])
        self._save_memberships(model["playlist_memberships"])
        self._sync_model_external_identities(model)

    def _save_source_presence(self, record: SourcePresenceRecord) -> None:
        binding = self._connection.execute(
            """SELECT 1 FROM external_identity_bindings
            WHERE source_system=? AND entity_type=? AND canonical_id=?""",
            (record.source_system, record.entity_type.value, record.canonical_id),
        ).fetchone()
        if binding is None:
            raise IdentityValidationError("source presence requires an existing external binding")
        self._connection.execute(
            """INSERT INTO source_entity_presence(
                source_system, entity_type, canonical_id, scope_key, presence
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_system, entity_type, canonical_id, scope_key)
            DO UPDATE SET presence=excluded.presence""",
            (
                record.source_system,
                record.entity_type.value,
                record.canonical_id,
                record.scope_key,
                record.presence.value,
            ),
        )

    def _reserve_model_ids(self, model: dict[str, Any]) -> None:
        collections = (
            ("tracks", EntityType.TRACK), ("artists", EntityType.ARTIST), ("albums", EntityType.ALBUM),
            ("playlists", EntityType.PLAYLIST), ("playlist_memberships", EntityType.PLAYLIST_MEMBERSHIP),
        )
        for collection, entity_type in collections:
            for entity in model[collection]:
                self._reserve_identity(entity["id"], entity_type)

    def _reserve_identity(self, canonical_id: str, entity_type: EntityType) -> None:
        validate_canonical_id(entity_type, canonical_id)
        self._connection.execute(
            "INSERT INTO canonical_entities(id, entity_type) VALUES (?, ?) ON CONFLICT(id) DO NOTHING",
            (canonical_id, entity_type.value),
        )
        stored = self._connection.execute(
            "SELECT entity_type FROM canonical_entities WHERE id = ?", (canonical_id,)
        ).fetchone()[0]
        if stored != entity_type.value:
            raise IdentityValidationError(f"{canonical_id} is already reserved as {stored}")

    def _save_artists(self, entities: list[dict[str, Any]]) -> None:
        for ordinal, entity in enumerate(entities):
            self._connection.execute(
                "INSERT INTO artists(id, ordinal, name) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET ordinal=excluded.ordinal, name=excluded.name",
                (entity["id"], ordinal, entity["name"]),
            )

    def _save_albums(self, entities: list[dict[str, Any]]) -> None:
        for ordinal, entity in enumerate(entities):
            self._connection.execute(
                "INSERT INTO albums(id, ordinal, name, release_date) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET ordinal=excluded.ordinal, name=excluded.name, release_date=excluded.release_date",
                (entity["id"], ordinal, entity["name"], entity["release_date"]),
            )
            self._replace_relations("album_artists", "album_id", entity["id"], entity["artist_ids"])

    def _save_tracks(self, entities: list[dict[str, Any]]) -> None:
        for ordinal, entity in enumerate(entities):
            state = entity["library_state"]
            self._connection.execute(
                """INSERT INTO tracks(
                    id, ordinal, name, album_id, duration_ms, track_number, disc_number, release_date, composer,
                    favorited, disliked, rating, play_count, skip_count, added_to_library_at, last_played_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    ordinal=excluded.ordinal, name=excluded.name, album_id=excluded.album_id,
                    duration_ms=excluded.duration_ms, track_number=excluded.track_number,
                    disc_number=excluded.disc_number, release_date=excluded.release_date, composer=excluded.composer,
                    favorited=excluded.favorited, disliked=excluded.disliked, rating=excluded.rating,
                    play_count=excluded.play_count, skip_count=excluded.skip_count,
                    added_to_library_at=excluded.added_to_library_at, last_played_at=excluded.last_played_at""",
                (entity["id"], ordinal, entity["name"], entity["album_id"], entity["duration_ms"],
                 entity["track_number"], entity["disc_number"], entity["release_date"], entity["composer"],
                 state["favorited"], state["disliked"], state["rating"], state["play_count"], state["skip_count"],
                 state["added_to_library_at"], state["last_played_at"]),
            )
            self._replace_relations("track_artists", "track_id", entity["id"], entity["artist_ids"])
            self._replace_values("track_genres", entity["id"], entity["genres"])
            self._replace_values("track_tags", entity["id"], entity["agent_metadata"]["tags"])

    def _save_playlists(self, entities: list[dict[str, Any]]) -> None:
        for ordinal, entity in enumerate(entities):
            self._connection.execute(
                "INSERT INTO playlists(id, ordinal, name) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET ordinal=excluded.ordinal, name=excluded.name",
                (entity["id"], ordinal, entity["name"]),
            )

    def _save_memberships(self, entities: list[dict[str, Any]]) -> None:
        for ordinal, entity in enumerate(entities):
            self._connection.execute(
                """INSERT INTO playlist_memberships(id, ordinal, playlist_id, track_id, position, added_at)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET ordinal=excluded.ordinal,
                playlist_id=excluded.playlist_id, track_id=excluded.track_id,
                position=excluded.position, added_at=excluded.added_at""",
                (entity["id"], ordinal, entity["playlist_id"], entity["track_id"], entity["position"], entity["added_at"]),
            )

    def _replace_relations(self, table: str, owner_column: str, owner_id: str, references: list[str]) -> None:
        target_column = "artist_id"
        self._connection.execute(f"DELETE FROM {table} WHERE {owner_column} = ?", (owner_id,))
        self._connection.executemany(
            f"INSERT INTO {table}({owner_column}, {target_column}, position) VALUES (?, ?, ?)",
            ((owner_id, reference, position) for position, reference in enumerate(references)),
        )

    def _replace_values(self, table: str, track_id: str, values: list[str]) -> None:
        self._connection.execute(f"DELETE FROM {table} WHERE track_id = ?", (track_id,))
        self._connection.executemany(
            f"INSERT INTO {table}(track_id, position, value) VALUES (?, ?, ?)",
            ((track_id, position, value) for position, value in enumerate(values)),
        )

    def _sync_model_external_identities(self, model: dict[str, Any]) -> None:
        collections = (
            ("tracks", EntityType.TRACK), ("artists", EntityType.ARTIST), ("albums", EntityType.ALBUM),
            ("playlists", EntityType.PLAYLIST),
        )
        for collection, entity_type in collections:
            for entity in model[collection]:
                external_ids = entity["external_ids"]
                for source_system, fixture_key in EXTERNAL_ID_FIXTURE_KEYS.items():
                    if fixture_key not in external_ids:
                        # Key absent: no claim either way; existing bindings are left untouched.
                        continue
                    external_id = external_ids[fixture_key]
                    existing = self._connection.execute(
                        "SELECT external_id FROM external_identity_bindings WHERE source_system=? AND entity_type=? AND canonical_id=?",
                        (source_system, entity_type.value, entity["id"]),
                    ).fetchone()
                    if external_id is None:
                        if existing is not None:
                            raise IdentityValidationError("canonical external ID cannot be cleared by persistence")
                        continue
                    key = ExternalIdentityKey(source_system, entity_type, external_id)
                    self._bind_external_identity(key, entity["id"])

    def _bind_external_identity(self, key: ExternalIdentityKey, canonical_id: str) -> str:
        if not isinstance(key, ExternalIdentityKey):
            raise IdentityValidationError("key must be an ExternalIdentityKey")
        validate_canonical_id(key.entity_type, canonical_id)
        entity = self._connection.execute(
            "SELECT entity_type FROM canonical_entities WHERE id = ?", (canonical_id,)
        ).fetchone()
        if entity is None or entity[0] != key.entity_type.value:
            raise IdentityValidationError("external identity target does not exist with the requested entity type")
        existing = self.lookup_external_identity(key)
        if existing is not None:
            if existing == canonical_id:
                return canonical_id
            raise IdentityConflictError(key, existing, canonical_id)
        scalar = self._connection.execute(
            "SELECT external_id FROM external_identity_bindings WHERE source_system=? AND entity_type=? AND canonical_id=?",
            (key.source_system, key.entity_type.value, canonical_id),
        ).fetchone() if key.source_system in SCALAR_SOURCE_SYSTEMS else None
        if scalar is not None and scalar[0] != key.external_id:
            raise IdentityValidationError(
                f"canonical entity already has a different {key.source_system} external ID"
            )
        self._connection.execute(
            "INSERT INTO external_identity_bindings(source_system, entity_type, external_id, canonical_id) VALUES (?, ?, ?, ?)",
            (key.source_system, key.entity_type.value, key.external_id, canonical_id),
        )
        return canonical_id

    def _rows(self, table: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(f"SELECT * FROM {table} ORDER BY ordinal, id"))

    def _external_id(self, canonical_id: str, entity_type: EntityType, source_system: str = "apple_music") -> str | None:
        row = self._connection.execute(
            "SELECT external_id FROM external_identity_bindings WHERE source_system=? AND entity_type=? AND canonical_id=?",
            (source_system, entity_type.value, canonical_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def _load_external_ids(self, canonical_id: str, entity_type: EntityType) -> dict[str, str | None]:
        """Project all bound external identities into the fixture's ``external_ids`` object.

        ``apple_music_persistent_id`` is always present (the fixture schema requires it); the
        catalog and ISRC keys appear only when bound, so legacy fixtures round-trip unchanged.
        """
        external_ids: dict[str, str | None] = {
            "apple_music_persistent_id": self._external_id(canonical_id, entity_type),
        }
        for source_system, fixture_key in EXTERNAL_ID_FIXTURE_KEYS.items():
            if source_system == "apple_music":
                continue
            bound = self._external_id(canonical_id, entity_type, source_system)
            if bound is not None:
                external_ids[fixture_key] = bound
        return external_ids

    def _load_named_entity(self, row: sqlite3.Row, entity_type: EntityType) -> dict[str, Any]:
        return {"id": row["id"], "external_ids": self._load_external_ids(row["id"], entity_type), "name": row["name"]}

    def _load_album(self, row: sqlite3.Row) -> dict[str, Any]:
        return {**self._load_named_entity(row, EntityType.ALBUM), "artist_ids": self._relation_values("album_artists", "album_id", row["id"], "artist_id"), "release_date": row["release_date"]}

    def _load_track(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "external_ids": self._load_external_ids(row["id"], EntityType.TRACK), "name": row["name"],
            "artist_ids": self._relation_values("track_artists", "track_id", row["id"], "artist_id"), "album_id": row["album_id"],
            "duration_ms": row["duration_ms"], "genres": self._relation_values("track_genres", "track_id", row["id"], "value"),
            "track_number": row["track_number"], "disc_number": row["disc_number"], "release_date": row["release_date"], "composer": row["composer"],
            "library_state": {"favorited": self._to_bool(row["favorited"]), "disliked": self._to_bool(row["disliked"]), "rating": row["rating"], "play_count": row["play_count"], "skip_count": row["skip_count"], "added_to_library_at": row["added_to_library_at"], "last_played_at": row["last_played_at"]},
            "agent_metadata": {"tags": self._relation_values("track_tags", "track_id", row["id"], "value")},
        }

    def _load_membership(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "playlist_id": row["playlist_id"], "track_id": row["track_id"], "position": row["position"], "added_at": row["added_at"]}

    def _relation_values(self, table: str, owner_column: str, owner_id: str, value_column: str) -> list[Any]:
        return [row[0] for row in self._connection.execute(f"SELECT {value_column} FROM {table} WHERE {owner_column}=? ORDER BY position", (owner_id,))]

    @staticmethod
    def _to_bool(value: int | None) -> bool | None:
        return None if value is None else bool(value)
