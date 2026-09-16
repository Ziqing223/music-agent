import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.capability_probe import (
    CapabilityProbe,
    create_probe,
)
from music_agent.capability_probe_recovery_attempt import (
    RecoveryAttempt,
    RecoveryAttemptEvent,
    RecoveryAttemptState,
    RecoveryAttemptTransitionError,
    RecoveryAttemptValidationError,
    advance_attempt,
    generate_recovery_attempt_id,
    validate_recovery_attempt_id,
)
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
    RecoveryAttemptAlreadyExistsError,
    RecoveryAttemptNotFoundError,
    RecoveryAttemptProbeNotFoundError,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.source_observation import ObservedValue, SourcePresence
from music_agent.write_execution import AttemptState
from music_agent.write_execution_repository import WriteExecutionRepository
from music_agent.write_intent import (
    DomainPermission,
    IntentState,
    WriteOperation,
    create_scalar_pending_intent,
    is_execution_ready,
    resolve_capability,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
V7_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
    (6, "0006_pending_write_intent_requirements.sql"),
    (7, "0007_capability_probes.sql"),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def track_key() -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)


def favorited_intent():
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), value(True)
    )


class CapabilityProbeRecoveryAttemptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def save_probe(self) -> CapabilityProbe:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
        return probe

    def save_canonical_model(self):
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()
        return fixture, track_id, before_counts

    def seed_isolated_store(self) -> dict:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()
            before_model = repository.load_model()
        intent = favorited_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
        probe = create_probe(track_id, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
        return {
            "track_id": track_id,
            "binding_key": binding_key,
            "intent": intent,
            "attempt": attempt,
            "probe": probe,
            "before_counts": before_counts,
            "before_model": before_model,
        }

    # --- identity -----------------------------------------------------------

    def test_generate_recovery_attempt_id_uses_rec_namespace(self) -> None:
        attempt_id = generate_recovery_attempt_id()
        self.assertTrue(attempt_id.startswith("rec_"))
        self.assertFalse(attempt_id.startswith("prb_"))
        self.assertFalse(attempt_id.startswith("att_"))
        self.assertFalse(attempt_id.startswith("int_"))
        self.assertFalse(attempt_id.startswith("trk_"))
        validate_recovery_attempt_id(attempt_id)  # must not raise

    def test_validate_recovery_attempt_id_rejects_foreign_or_malformed(self) -> None:
        bad_values = [
            "att_11111111-1111-4111-8111-111111111111",
            "prb_11111111-1111-4111-8111-111111111111",
            "int_11111111-1111-4111-8111-111111111111",
            "trk_11111111-1111-4111-8111-111111111111",
            "rec_11111111",
            "rec_gggggggg-gggg-4ggg-8ggg-gggggggggggg",
            "rec_11111111-1111-4111-8111-111111111111-extra",
            123,
            None,
        ]
        for bad in bad_values:
            with self.assertRaises(RecoveryAttemptValidationError):
                validate_recovery_attempt_id(bad)

    def test_attempt_id_is_independent_of_probe_id(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
        self.assertTrue(attempt.attempt_id.startswith("rec_"))
        self.assertNotEqual(attempt.attempt_id, probe.probe_id)

    # --- schema / migration -------------------------------------------------

    def test_fresh_database_reaches_v8_with_recovery_attempt_table(self) -> None:
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("capability_probe_recovery_attempts", tables)
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(capability_probe_recovery_attempts)"
                )
            }
            self.assertEqual(
                columns, {"attempt_id", "probe_id", "state", "created_at", "updated_at"}
            )

    def test_real_v7_store_upgrades_to_v8_without_changing_prior_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        probe = create_probe(track_id, TRACK_PID, False, False)
        with patch("music_agent.repository.MIGRATIONS", V7_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 7)
            with CapabilityProbeRepository(self.database_path) as repository:
                repository.save_probe(probe)

        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_v8_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V7_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 7)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v8(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                CapabilityProbeRecoveryAttemptRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                7,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v8'"
                ).fetchone()
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='capability_probe_recovery_attempts'"
                ).fetchone()
            )

    # --- begin / round-trip -------------------------------------------------

    def test_begin_attempt_returns_started_with_exact_round_trip(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            self.assertIs(attempt.state, RecoveryAttemptState.STARTED)
            self.assertEqual(attempt.probe_id, probe.probe_id)
            self.assertEqual(repository.get_attempt(attempt.attempt_id), attempt)
            self.assertEqual(repository.get_for_probe(probe.probe_id), attempt)

    def test_started_attempt_survives_restart(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            self.assertEqual(repository.get_attempt(attempt.attempt_id), attempt)
            self.assertEqual(repository.get_for_probe(probe.probe_id), attempt)
            self.assertIs(repository.get_for_probe(probe.probe_id).state, RecoveryAttemptState.STARTED)

    def test_begin_attempt_requires_existing_probe(self) -> None:
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            with self.assertRaises(RecoveryAttemptProbeNotFoundError):
                repository.begin_attempt("prb_ffffffff-ffff-4fff-8fff-ffffffffffff")

    # --- transitions --------------------------------------------------------

    def test_mark_command_succeeded_advances_started(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            advanced = repository.mark_command_succeeded(attempt.attempt_id)
            self.assertIs(advanced.state, RecoveryAttemptState.COMMAND_SUCCEEDED)
            self.assertIs(
                repository.get_attempt(attempt.attempt_id).state,
                RecoveryAttemptState.COMMAND_SUCCEEDED,
            )

    def test_mark_command_failed_advances_started(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            advanced = repository.mark_command_failed(attempt.attempt_id)
            self.assertIs(advanced.state, RecoveryAttemptState.COMMAND_FAILED)
            self.assertIs(
                repository.get_attempt(attempt.attempt_id).state,
                RecoveryAttemptState.COMMAND_FAILED,
            )

    def test_terminal_transition_is_immutable(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_succeeded(attempt.attempt_id)
            with self.assertRaises(RecoveryAttemptTransitionError):
                repository.mark_command_failed(attempt.attempt_id)
            with self.assertRaises(RecoveryAttemptTransitionError):
                repository.mark_command_succeeded(attempt.attempt_id)
            self.assertIs(
                repository.get_attempt(attempt.attempt_id).state,
                RecoveryAttemptState.COMMAND_SUCCEEDED,
            )

    def test_domain_rejects_terminal_transition_and_restart(self) -> None:
        succeeded = RecoveryAttempt(
            "rec_11111111-1111-4111-8111-111111111111",
            "prb_11111111-1111-4111-8111-111111111111",
            RecoveryAttemptState.COMMAND_SUCCEEDED,
        )
        with self.assertRaises(RecoveryAttemptTransitionError):
            advance_attempt(succeeded, RecoveryAttemptEvent.COMMAND_FAILED)
        with self.assertRaises(RecoveryAttemptTransitionError):
            advance_attempt(succeeded, RecoveryAttemptEvent.COMMAND_SUCCEEDED)

    def test_mark_outcome_on_missing_attempt_fails_closed(self) -> None:
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            with self.assertRaises(RecoveryAttemptNotFoundError):
                repository.mark_command_succeeded("rec_ffffffff-ffff-4fff-8fff-ffffffffffff")

    # --- one attempt per probe ---------------------------------------------

    def test_second_begin_after_started_rejected(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            repository.begin_attempt(probe.probe_id)
            with self.assertRaises(RecoveryAttemptAlreadyExistsError):
                repository.begin_attempt(probe.probe_id)

    def test_second_begin_after_succeeded_rejected(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_succeeded(attempt.attempt_id)
            with self.assertRaises(RecoveryAttemptAlreadyExistsError):
                repository.begin_attempt(probe.probe_id)

    def test_second_begin_after_failed_rejected(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_failed(attempt.attempt_id)
            with self.assertRaises(RecoveryAttemptAlreadyExistsError):
                repository.begin_attempt(probe.probe_id)

    def test_database_rejects_second_attempt_directly(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            repository.begin_attempt(probe.probe_id)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    """INSERT INTO capability_probe_recovery_attempts(attempt_id, probe_id, state)
                    VALUES (?, ?, ?)""",
                    ("rec_ffffffff-ffff-4fff-8fff-ffffffffffff", probe.probe_id, "started"),
                )

    def test_one_attempt_per_probe_even_after_terminal_state(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_succeeded(attempt.attempt_id)
            with self.assertRaises(RecoveryAttemptAlreadyExistsError):
                repository.begin_attempt(probe.probe_id)
            remaining = repository.get_for_probe(probe.probe_id)
            self.assertEqual(remaining.attempt_id, attempt.attempt_id)
            self.assertIs(remaining.state, RecoveryAttemptState.COMMAND_SUCCEEDED)

    # --- concurrency --------------------------------------------------------

    def test_two_workers_cannot_create_two_recovery_attempts(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)

        results = []
        errors = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
            try:
                barrier.wait()
                results.append(repository.begin_attempt(probe.probe_id))
            except Exception as error:
                errors.append(error)
            finally:
                repository.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RecoveryAttemptAlreadyExistsError)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            self.assertEqual(repository.get_for_probe(probe.probe_id), results[0])

    # --- probe / canonical / operational isolation --------------------------

    def test_recovery_attempt_does_not_modify_probe(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_succeeded(attempt.attempt_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_recovery_attempt_does_not_mutate_canonical_or_operational_state(self) -> None:
        seed = self.seed_isolated_store()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(seed["probe"].probe_id)
            repository.mark_command_failed(attempt.attempt_id)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), seed["before_counts"])
            self.assertEqual(repository.load_model(), seed["before_model"])
            self.assertEqual(
                repository.lookup_external_identity(seed["binding_key"]), seed["track_id"]
            )
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, seed["track_id"], "library_tracks"
                ),
                SourcePresence.PRESENT,
            )
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(seed["intent"].intent_id).state, IntentState.PENDING)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_attempt(seed["attempt"].attempt_id).state, AttemptState.STARTED)

    # --- capability isolation -----------------------------------------------

    def test_capability_matrix_unchanged_and_set_favorited_not_execution_ready(self) -> None:
        probe = self.save_probe()
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as repository:
            attempt = repository.begin_attempt(probe.probe_id)
            repository.mark_command_succeeded(attempt.attempt_id)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.adapter_implemented, True)
        self.assertIs(capability.readback_implemented, True)
        self.assertFalse(is_execution_ready(capability))
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))


if __name__ == "__main__":
    unittest.main()
