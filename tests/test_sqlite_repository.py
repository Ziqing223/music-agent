import copy
import json
import sqlite3
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from unittest.mock import patch

from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.identity import EntityType, ExternalIdentityKey, IdentityConflictError, IdentityValidationError
from music_agent.ingestion_candidate import IngestionCandidate
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS, CanonicalRepository
from music_agent.repository import SourcePresenceRecord
from music_agent.source_observation import ObservedValue, SourcePresence
from music_agent.validation import GraphValidationError, validate_fixture


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class SQLiteRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_empty_database_initializes_and_reopens_at_current_version(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertTrue(repository.foreign_keys_enabled)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            migrations = repository._connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
            self.assertEqual(migrations, CURRENT_SCHEMA_VERSION)

    def test_actual_v1_store_upgrades_to_current_schema_without_changing_model_or_bindings(self) -> None:
        fixture = load_fixture()
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        with patch("music_agent.repository.MIGRATIONS", ((1, "0001_canonical_store.sql"),)):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model(fixture)
                self.assertEqual(repository.schema_version, 1)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(key), fixture["tracks"][0]["id"])
            for table_name in ("source_entity_presence", "ingestion_candidates"):
                with self.subTest(table_name=table_name):
                    table = repository._connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table_name,),
                    ).fetchone()
                    self.assertIsNotNone(table)

    def test_second_migration_failure_rolls_back_without_partial_v2_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", ((1, "0001_canonical_store.sql"),)):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 1)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v2(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                CanonicalRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0], 1)
            partial = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v2'"
            ).fetchone()
            self.assertIsNone(partial)

    def test_failed_migration_rolls_back_without_partial_schema(self) -> None:
        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_marker(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                CanonicalRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            marker = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_marker'"
            ).fetchone()
            self.assertIsNone(marker)

    def test_mid_0010_migration_failure_rolls_back_entire_transaction(self) -> None:
        intent_id = "int_00000000-0000-4000-8000-000000000001"
        requirement_canonical_id = "trk_00000000-0000-4000-8000-000000000001"
        attempt_id = "att_00000000-0000-4000-8000-000000000001"

        # Build a real v9 store and seed one intent + requirement + attempt so the rollback has
        # durable rows to preserve.
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:9]):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 9)
                connection = repository._connection
                connection.execute(
                    "INSERT INTO pending_write_intents"
                    "(intent_id, operation, requested_state, requested_value_json, lifecycle_state)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (intent_id, "set_favorited", "value", "true", "pending"),
                )
                connection.execute(
                    "INSERT INTO pending_write_intent_requirements"
                    "(intent_id, role, canonical_id, source_system, entity_type, external_id)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (intent_id, "target", requirement_canonical_id, "apple_music", "track", "EXT-MID0010"),
                )
                connection.execute(
                    "INSERT INTO write_execution_attempts(attempt_id, intent_id, state)"
                    " VALUES (?, ?, ?)",
                    (attempt_id, intent_id, "started"),
                )

        # Reopen at the full migration set, but inject a failure into the *real* 0010 content after
        # its first DDL step (the intent table rename), so the migration fails mid-rebuild rather
        # than before any DDL. The whole migration transaction must roll back.
        real_files = resources.files
        real_0010 = (
            real_files("music_agent.migrations")
            .joinpath("0010_write_ambiguous_outcome.sql")
            .read_text(encoding="utf-8")
        )
        anchor = "ALTER TABLE pending_write_intents RENAME TO pending_write_intents_old;"
        injected_0010 = real_0010.replace(
            anchor,
            anchor + "\nSELECT * FROM no_such_table_0010_mid_rollback;",
            1,
        )

        class Injected0010Files:
            def __init__(self, injected: str) -> None:
                self._injected = injected

            def joinpath(self, name: str):
                if name == "0010_write_ambiguous_outcome.sql":
                    return _StringText(self._injected)
                return real_files("music_agent.migrations").joinpath(name)

        class _StringText:
            def __init__(self, content: str) -> None:
                self._content = content

            def read_text(self, **_: object) -> str:
                return self._content

        with patch(
            "music_agent.repository.resources.files",
            return_value=Injected0010Files(injected_0010),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                CanonicalRepository(self.database_path)

        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                9,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 10"
                ).fetchone()
            )
            for table_name in (
                "pending_write_intents_old",
                "write_execution_attempts_new",
                "pending_write_intent_requirements_new",
            ):
                with self.subTest(table_name=table_name):
                    self.assertIsNone(
                        connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (table_name,),
                        ).fetchone()
                    )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents WHERE intent_id=?",
                    (intent_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intent_requirements WHERE intent_id=?",
                    (intent_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM write_execution_attempts WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_fixture_round_trip_survives_close_and_reopen(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
        validate_fixture(loaded)
        self.assertEqual(loaded, fixture)

    def test_modified_valid_model_round_trips_without_fixture_assumptions(self) -> None:
        model = copy.deepcopy(load_fixture())
        model["artists"][0]["name"] = "Independent Artist Name"
        model["tracks"][0]["name"] = "Independent Track Name"
        model["tracks"][0]["genres"] = ["Ambient", "Electronic"]
        model["tracks"][0]["library_state"]["play_count"] = 42
        validate_fixture(model)

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()

        self.assertEqual(loaded, model)

    def test_scalar_and_array_semantics_round_trip(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())
            loaded = repository.load_model()
        first, third = loaded["tracks"][0], loaded["tracks"][2]
        self.assertIs(first["library_state"]["favorited"], False)
        self.assertEqual(first["library_state"]["play_count"], 0)
        self.assertEqual(first["agent_metadata"]["tags"], [])
        self.assertIsNone(third["album_id"])
        self.assertIsNone(third["duration_ms"])
        self.assertEqual(third["genres"], [])

    def test_relation_order_and_membership_multiplicity_are_preserved(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            loaded = repository.load_model()
        self.assertEqual(loaded["tracks"][0]["artist_ids"], fixture["tracks"][0]["artist_ids"])
        self.assertEqual(loaded["albums"][0]["artist_ids"], fixture["albums"][0]["artist_ids"])
        first, second = loaded["playlist_memberships"][:2]
        self.assertEqual((first["playlist_id"], first["track_id"]), (second["playlist_id"], second["track_id"]))
        self.assertNotEqual(first["id"], second["id"])

    def test_external_binding_survives_restart_and_conflict_is_stable(self) -> None:
        fixture = load_fixture()
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        original_id = fixture["tracks"][0]["id"]
        attempted_id = fixture["tracks"][1]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            self.assertEqual(repository.lookup_external_identity(key), original_id)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.lookup_external_identity(key), original_id)
            self.assertEqual(repository.bind_external_identity(key, original_id), original_id)
            with self.assertRaises(IdentityConflictError):
                repository.bind_external_identity(key, attempted_id)
            self.assertEqual(repository.lookup_external_identity(key), original_id)

    def test_secondary_external_binding_survives_restart(self) -> None:
        fixture = load_fixture()
        key = ExternalIdentityKey("secondary_source", EntityType.TRACK, "opaque-secondary-id")
        canonical_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository.bind_external_identity(key, canonical_id)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.lookup_external_identity(key), canonical_id)

    def test_canonical_type_reservation_survives_restart(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        with CanonicalRepository(self.database_path) as repository:
            row = repository._connection.execute(
                "SELECT entity_type FROM canonical_entities WHERE id=?", (track_id,)
            ).fetchone()
            self.assertEqual(row[0], "track")
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE canonical_entities SET entity_type='artist' WHERE id=?", (track_id,)
                )

    def test_repeated_fixture_save_is_idempotent(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            first_counts = repository.counts()
            repository.save_model(fixture)
            self.assertEqual(repository.counts(), first_counts)
            self.assertEqual(repository.load_model(), fixture)


def _staging_candidate(external_id: str = "STAGING-TRACK-1", source_system: str = "apple_music") -> IngestionCandidate:
    return IngestionCandidate(
        ExternalIdentityKey(source_system, EntityType.TRACK, external_id),
        {
            "name": ObservedValue.value("Staged Candidate"),
            "duration_ms": ObservedValue.value(123000),
            "genres": ObservedValue.value(["Synthetic"]),
            "composer": ObservedValue.null(),
            "library_state.favorited": ObservedValue.value(False),
            "library_state.rating": ObservedValue.value(0),
            "library_state.play_count": ObservedValue.value(0),
        },
    )


class CatalogIdentityMigrationTest(unittest.TestCase):
    """Schema v17: apple_music_catalog / isrc identities (P11.1)."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_v16_store_with_staged_candidate_upgrades_to_v17_preserving_rows_and_bindings(self) -> None:
        fixture = load_fixture()
        library_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:16]):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model(fixture)
                self.assertEqual(repository.schema_version, 16)
            with CandidateStagingRepository(self.database_path) as staging:
                staging.stage_candidate(_staging_candidate(), "library_tracks")
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(library_key), fixture["tracks"][0]["id"])
        # Staged candidate and its scope survive the ingestion_candidates table rebuild.
        staged_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "STAGING-TRACK-1")
        with CandidateStagingRepository(self.database_path) as staging:
            preserved = staging.get_candidate(staged_key)
            self.assertIsNotNone(preserved)
            self.assertEqual(staging.list_candidate_scopes(staged_key), ("library_tracks",))
            # The widened CHECK admits apple_music_catalog candidates.
            catalog_key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1")
            staging.stage_candidate(_staging_candidate("CATALOG-1", "apple_music_catalog"), "catalog")
            self.assertIsNotNone(staging.get_candidate(catalog_key))
        # Both new scalar unique indexes exist.
        with CanonicalRepository(self.database_path) as repository:
            indexes = {
                row[0] for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            self.assertIn("ux_apple_music_catalog_scalar_identity", indexes)
            self.assertIn("ux_isrc_scalar_identity", indexes)

    def test_one_canonical_track_holds_persistent_catalog_and_isrc_bindings(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        catalog_key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1")
        isrc_key = ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001")
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository.bind_external_identity(catalog_key, track_id)
            repository.bind_external_identity(isrc_key, track_id)
            self.assertEqual(repository.lookup_external_identity(catalog_key), track_id)
            self.assertEqual(repository.lookup_external_identity(isrc_key), track_id)
            loaded = repository.load_model()
            self.assertEqual(
                loaded["tracks"][0]["external_ids"]["apple_music_catalog_id"], "CATALOG-1"
            )
            self.assertEqual(loaded["tracks"][0]["external_ids"]["isrc"], "USSYN2400001")
            # The persistent ID binding is untouched by the two new bindings.
            self.assertEqual(
                loaded["tracks"][0]["external_ids"]["apple_music_persistent_id"], "SYNTH-TRACK-001"
            )

    def test_second_catalog_binding_on_same_canonical_fails_scalar_rule(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository.bind_external_identity(
                ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1"), track_id
            )
            with self.assertRaises(IdentityValidationError):
                repository.bind_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-2"), track_id
                )

    def test_same_isrc_cannot_bind_to_two_canonical_tracks(self) -> None:
        fixture = load_fixture()
        first_id = fixture["tracks"][0]["id"]
        second_id = fixture["tracks"][1]["id"]
        isrc_key = ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001")
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository.bind_external_identity(isrc_key, first_id)
            with self.assertRaises(IdentityConflictError):
                repository.bind_external_identity(isrc_key, second_id)

    def test_fixture_with_catalog_and_isrc_keys_round_trips(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["external_ids"]["apple_music_catalog_id"] = "CATALOG-1"
        fixture["tracks"][0]["external_ids"]["isrc"] = "USSYN2400001"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            self.assertEqual(repository.load_model(), fixture)

    def test_absent_fixture_keys_leave_existing_catalog_binding_untouched(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository.bind_external_identity(
                ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1"), track_id
            )
            # Saving the legacy fixture again (no catalog key) must not clear the binding.
            repository.save_model(fixture)
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1")
                ),
                track_id,
            )

    def test_explicit_null_catalog_id_cannot_clear_an_existing_binding(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        fixture["tracks"][0]["external_ids"]["apple_music_catalog_id"] = "CATALOG-1"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            fixture["tracks"][0]["external_ids"]["apple_music_catalog_id"] = None
            with self.assertRaises(IdentityValidationError):
                repository.save_model(fixture)
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-1")
                ),
                track_id,
            )

    def test_same_track_id_updates_current_state_without_new_identity(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            before = repository.counts()["canonical_entities"]
            fixture["tracks"][0]["name"] = "Updated Synthetic Name"
            repository.save_model(fixture)
            loaded = repository.load_model()
            self.assertEqual(repository.counts()["canonical_entities"], before)
        self.assertEqual(loaded["tracks"][0]["id"], track_id)
        self.assertEqual(loaded["tracks"][0]["name"], "Updated Synthetic Name")

    def test_application_validation_failure_causes_no_mutation(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["artist_ids"] = ["art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"]
        with CanonicalRepository(self.database_path) as repository:
            with self.assertRaises(GraphValidationError):
                repository.save_model(fixture)
            self.assertEqual(repository.counts()["canonical_entities"], 0)

    def test_external_identity_conflict_rolls_back_entire_fixture(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][1]["external_ids"]["apple_music_persistent_id"] = "SYNTH-TRACK-001"
        with CanonicalRepository(self.database_path) as repository:
            before = repository.counts()
            with self.assertRaises(IdentityConflictError):
                repository.save_model(fixture)
            self.assertEqual(repository.counts(), before)

    def test_sqlite_foreign_keys_reject_all_dangling_reference_classes(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            connection = repository._connection
            statements = (
                ("UPDATE tracks SET album_id=? WHERE id=?", ("alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd", fixture["tracks"][0]["id"])),
                ("INSERT INTO track_artists(track_id, artist_id, position) VALUES (?, ?, ?)", (fixture["tracks"][0]["id"], "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd", 99)),
                ("INSERT INTO album_artists(album_id, artist_id, position) VALUES (?, ?, ?)", (fixture["albums"][0]["id"], "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd", 99)),
                ("UPDATE playlist_memberships SET playlist_id=? WHERE id=?", ("pl_dddddddd-dddd-4ddd-8ddd-dddddddddddd", fixture["playlist_memberships"][0]["id"])),
                ("UPDATE playlist_memberships SET track_id=? WHERE id=?", ("trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd", fixture["playlist_memberships"][0]["id"])),
                ("INSERT INTO external_identity_bindings VALUES (?, ?, ?, ?)", ("source", "track", "missing", "trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd")),
            )
            for statement, parameters in statements:
                with self.subTest(statement=statement):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(statement, parameters)

    def test_database_constraints_reject_duplicate_external_key_and_type_reuse(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            connection = repository._connection
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO external_identity_bindings VALUES (?, ?, ?, ?)",
                    ("apple_music", "track", "SYNTH-TRACK-001", fixture["tracks"][1]["id"]),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE canonical_entities SET entity_type='artist' WHERE id=?",
                    (fixture["tracks"][0]["id"],),
                )

    def test_external_binding_requires_existing_matching_entity_type(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            artist_key = ExternalIdentityKey("source", EntityType.ARTIST, "external")
            with self.assertRaises(IdentityValidationError):
                repository.bind_external_identity(artist_key, fixture["tracks"][0]["id"])

    def test_repository_has_no_physical_delete_api(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            self.assertFalse(hasattr(repository, "delete"))
            self.assertFalse(hasattr(repository, "delete_entity"))

    def test_model_and_presence_updates_rollback_together(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            repository._connection.execute(
                """CREATE TRIGGER fail_presence BEFORE INSERT ON source_entity_presence
                BEGIN SELECT RAISE(ABORT, 'presence failure'); END"""
            )
            changed = copy.deepcopy(fixture)
            changed["tracks"][0]["name"] = "Must Roll Back"
            update = SourcePresenceRecord(
                "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.save_model_with_source_presence(changed, [update])
            self.assertEqual(repository.load_model(), fixture)
            self.assertIsNone(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                )
            )

    def test_runtime_or_untyped_presence_cannot_enter_durable_api(self) -> None:
        track_id = load_fixture()["tracks"][0]["id"]
        for presence in (SourcePresence.MISSING, SourcePresence.UNKNOWN, "present"):
            with self.subTest(presence=presence):
                with self.assertRaises(IdentityValidationError):
                    SourcePresenceRecord(
                        "apple_music", EntityType.TRACK, track_id, "library_tracks", presence
                    )


if __name__ == "__main__":
    unittest.main()
