import copy
import dataclasses
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository, PendingIntentRepositoryError
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.source_observation import ObservationState, ObservedValue, SourcePresence
from music_agent.write_intent import (
    RELATION_WRITE_VALUE,
    IntentState,
    PendingIntent,
    RequirementRole,
    WriteEvent,
    WriteOperation,
    WriteRequirement,
    WriteTransitionError,
    create_pending_intent,
    create_scalar_pending_intent,
    is_execution_ready,
    resolve_capability,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
PLAYLIST_ID = "pl_55555555-5555-4555-8555-555555555555"
FIXED_INTENT_ID = "int_11111111-1111-4111-8111-111111111111"
V3_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
)
V5_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = "SYNTH-TRACK-001") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


def playlist_requirement(
    playlist_id: str = PLAYLIST_ID, external_id: str = "SYNTH-PLAYLIST-1"
) -> WriteRequirement:
    return WriteRequirement(
        RequirementRole.PLAYLIST,
        playlist_id,
        ExternalIdentityKey("apple_music", EntityType.PLAYLIST, external_id),
    )


def track_requirement(
    track_id: str = TRACK_ID, external_id: str = "SYNTH-TRACK-001"
) -> WriteRequirement:
    return WriteRequirement(
        RequirementRole.TRACK,
        track_id,
        ExternalIdentityKey("apple_music", EntityType.TRACK, external_id),
    )


def relation_intent(
    playlist_ext: str = "SYNTH-PLAYLIST-1",
    track_ext: str = "SYNTH-TRACK-001",
) -> PendingIntent:
    return create_pending_intent(
        WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
        (playlist_requirement(external_id=playlist_ext), track_requirement(external_id=track_ext)),
        RELATION_WRITE_VALUE,
    )


def fixed_scalar_intent() -> PendingIntent:
    return PendingIntent(
        intent_id=FIXED_INTENT_ID,
        operation=WriteOperation.SET_FAVORITED,
        requirements=(WriteRequirement(RequirementRole.TARGET, TRACK_ID, track_key()),),
        requested_value=ObservedValue.value(True),
    )


def favorited_intent(value: object = True) -> PendingIntent:
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.value(value)
    )


class PendingIntentRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_fresh_database_reaches_v6_with_requirement_table(self) -> None:
        with PendingIntentRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("pending_write_intents", tables)
            self.assertIn("pending_write_intent_requirements", tables)
            intent_columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(pending_write_intents)"
                )
            }
            self.assertEqual(
                intent_columns,
                {
                    "intent_id",
                    "operation",
                    "requested_state",
                    "requested_value_json",
                    "lifecycle_state",
                    "created_at",
                    "updated_at",
                },
            )
            requirement_columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(pending_write_intent_requirements)"
                )
            }
            self.assertEqual(
                requirement_columns,
                {"intent_id", "role", "canonical_id", "source_system", "entity_type", "external_id"},
            )

    def test_real_v3_store_upgrades_to_current_without_changing_canonical_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with patch("music_agent.repository.MIGRATIONS", V3_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 3)

        with PendingIntentRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            repository.save_intent(favorited_intent())

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_real_v5_store_upgrades_to_v6_preserving_existing_intent(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V5_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 5)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """INSERT INTO pending_write_intents(
                    intent_id, operation, target_canonical_id,
                    required_source_system, required_entity_type, required_external_id,
                    requested_state, requested_value_json, lifecycle_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    FIXED_INTENT_ID,
                    "set_favorited",
                    TRACK_ID,
                    "apple_music",
                    "track",
                    "SYNTH-TRACK-001",
                    "value",
                    "true",
                    "pending",
                ),
            )
            connection.commit()

        with PendingIntentRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(repository.get_intent(FIXED_INTENT_ID), fixed_scalar_intent())

        # The singular columns are gone and the requirement now lives in the child table.
        with sqlite3.connect(self.database_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(pending_write_intents)")
            }
            self.assertNotIn("target_canonical_id", columns)
            self.assertNotIn("required_external_id", columns)
            child_rows = list(
                connection.execute(
                    "SELECT role, canonical_id, entity_type, external_id "
                    "FROM pending_write_intent_requirements WHERE intent_id=?",
                    (FIXED_INTENT_ID,),
                )
            )
            self.assertEqual(
                child_rows, [("target", TRACK_ID, "track", "SYNTH-TRACK-001")]
            )

    def test_v4_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V3_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 3)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v4(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                PendingIntentRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                3,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v4'"
                ).fetchone()
            )

    def test_v6_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V5_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 5)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v6(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                PendingIntentRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                5,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v6'"
                ).fetchone()
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='pending_write_intent_requirements'"
                ).fetchone()
            )

    def test_save_get_exact_round_trip(self) -> None:
        intent = favorited_intent(False)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            self.assertEqual(repository.get_intent(intent.intent_id), intent)

    def test_relation_intent_exact_round_trip(self) -> None:
        intent = relation_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            self.assertEqual(repository.get_intent(intent.intent_id), intent)

    def test_relation_intent_requirement_order_is_canonical_on_load(self) -> None:
        intent = relation_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            loaded = repository.get_intent(intent.intent_id)
            self.assertEqual(
                [requirement.role for requirement in loaded.requirements],
                [RequirementRole.PLAYLIST, RequirementRole.TRACK],
            )

    def test_false_zero_and_null_requested_values_round_trip(self) -> None:
        intents = (
            favorited_intent(False),
            create_scalar_pending_intent(
                WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.value(0)
            ),
            create_scalar_pending_intent(
                WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.null()
            ),
        )
        with PendingIntentRepository(self.database_path) as repository:
            for intent in intents:
                repository.save_intent(intent)
            loaded = {intent.intent_id: repository.get_intent(intent.intent_id) for intent in intents}
        for intent in intents:
            self.assertEqual(loaded[intent.intent_id], intent)
        self.assertIs(loaded[intents[0].intent_id].requested_value.payload, False)
        self.assertEqual(loaded[intents[1].intent_id].requested_value.payload, 0)
        self.assertIs(loaded[intents[2].intent_id].requested_value.state, ObservationState.NULL)

    def test_pending_survives_restart(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)

    def test_relation_intent_survives_restart(self) -> None:
        intent = relation_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertEqual(repository.get_intent(intent.intent_id), intent)

    def test_awaiting_readback_survives_restart(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK
            )

    def test_confirmed_survives_restart(self) -> None:
        intent = favorited_intent(False)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            repository.update_intent_state(intent.intent_id, WriteEvent.READBACK_MATCHED)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.CONFIRMED)

    def test_execution_failed_survives_restart(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_FAILED)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.EXECUTION_FAILED
            )

    def test_readback_mismatch_survives_restart(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            repository.update_intent_state(intent.intent_id, WriteEvent.READBACK_MISMATCHED)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.READBACK_MISMATCH
            )

    def test_update_intent_state_updates_existing_row_not_new(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                1,
            )
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK
            )

    def test_retry_resave_does_not_create_second_intent(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.save_intent(intent)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(repository.get_intent(intent.intent_id), intent)

    def test_relation_retry_resave_is_idempotent(self) -> None:
        intent = relation_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.save_intent(intent)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intent_requirements"
                ).fetchone()[0],
                2,
            )
            self.assertEqual(repository.get_intent(intent.intent_id), intent)

    def test_save_intent_rejects_intent_id_reuse_with_different_content(self) -> None:
        first = favorited_intent(True)
        conflicting = dataclasses.replace(first, requested_value=ObservedValue.value(False))
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(first)
            with self.assertRaises(PendingIntentRepositoryError):
                repository.save_intent(conflicting)
            self.assertEqual(repository.get_intent(first.intent_id), first)

    def test_relation_same_id_with_changed_requirement_is_conflict(self) -> None:
        first = relation_intent(track_ext="SYNTH-TRACK-001")
        changed = dataclasses.replace(
            first,
            requirements=(
                playlist_requirement(),
                track_requirement(external_id="SYNTH-TRACK-999"),
            ),
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(first)
            with self.assertRaises(PendingIntentRepositoryError):
                repository.save_intent(changed)
            self.assertEqual(repository.get_intent(first.intent_id), first)

    def test_save_intent_requires_pending_state(self) -> None:
        advanced = dataclasses.replace(
            favorited_intent(True), state=IntentState.AWAITING_READBACK
        )
        with PendingIntentRepository(self.database_path) as repository:
            with self.assertRaises(PendingIntentRepositoryError):
                repository.save_intent(advanced)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                0,
            )

    def test_illegal_transition_persists_nothing(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            with self.assertRaises(WriteTransitionError):
                repository.update_intent_state(intent.intent_id, WriteEvent.READBACK_MATCHED)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                1,
            )

    def test_stale_transition_cannot_overwrite_later_durable_state(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            with self.assertRaises(WriteTransitionError):
                repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_FAILED)
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK
            )
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK
            )

    def test_get_intent_missing_returns_none(self) -> None:
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_intent("int_ffffffff-ffff-4fff-8fff-ffffffffffff"))

    def test_list_intents_filters_by_state(self) -> None:
        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            self.assertEqual(repository.list_intents(IntentState.PENDING), ())
            awaiting = repository.list_intents(IntentState.AWAITING_READBACK)
            self.assertEqual([candidate.intent_id for candidate in awaiting], [intent.intent_id])
            self.assertEqual(
                [candidate.intent_id for candidate in repository.list_intents()], [intent.intent_id]
            )

    def test_save_rolls_back_intent_and_requirements_atomically(self) -> None:
        intent = relation_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository._connection.execute(
                """CREATE TRIGGER fail_requirements
                BEFORE INSERT ON pending_write_intent_requirements
                BEGIN SELECT RAISE(ABORT, 'requirement failure'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.save_intent(intent)
            self.assertIsNone(repository.get_intent(intent.intent_id))
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intents"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM pending_write_intent_requirements"
                ).fetchone()[0],
                0,
            )

    def test_requirement_rows_cannot_orphan_without_intent(self) -> None:
        with PendingIntentRepository(self.database_path) as repository:
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    """INSERT INTO pending_write_intent_requirements(
                        intent_id, role, canonical_id, source_system, entity_type, external_id
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        "int_ffffffff-ffff-4fff-8fff-ffffffffffff",
                        "target",
                        TRACK_ID,
                        "apple_music",
                        "track",
                        "SYNTH-TRACK-001",
                    ),
                )

    def test_intent_persistence_does_not_mutate_canonical_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()

        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
            repository.update_intent_state(intent.intent_id, WriteEvent.COMMAND_SUCCEEDED)
            repository.update_intent_state(intent.intent_id, WriteEvent.READBACK_MATCHED)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.CONFIRMED)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_current_execution_ready_set_remains_empty(self) -> None:
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))


if __name__ == "__main__":
    unittest.main()
