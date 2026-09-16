import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.source_observation import ObservationState, ObservedValue, SourcePresence
from music_agent.write_execution import (
    AmbiguousCommandOutcomeError,
    AttemptEvent,
    AttemptState,
    AttemptTransitionError,
    DeterministicCommandError,
    ExecutionAttempt,
    advance_attempt,
)
from music_agent.write_execution_repository import (
    AmbiguousAttemptError,
    AttemptNotFoundError,
    ExecutionNotPendingError,
    ReadbackGateError,
    WriteExecutionRepository,
)
from music_agent.write_intent import (
    IntentState,
    WriteEvent,
    WriteOperation,
    WriteTransitionError,
    create_scalar_pending_intent,
    is_execution_ready,
    resolve_capability,
)
from music_agent.write_orchestrator import (
    NotExecutionReadyError,
    WriteOrchestrator,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
V4_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
)

V9_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
    (6, "0006_pending_write_intent_requirements.sql"),
    (7, "0007_capability_probes.sql"),
    (8, "0008_capability_probe_recovery_attempts.sql"),
    (9, "0009_capability_verification_evidence.sql"),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = "SYNTH-TRACK-001") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


def favorited_intent(value: object = True):
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.value(value)
    )


def permissive_policy(intent) -> bool:
    return True


REBUILT_TABLES = (
    "pending_write_intents",
    "write_execution_attempts",
    "pending_write_intent_requirements",
)


def _table_signature(connection: sqlite3.Connection, table: str) -> dict:
    """Capture the durable schema surface of one table, excluding autoindex/PK artifacts.

    Columns, foreign keys, and explicit (non-PK) indexes are the invariants a table rebuild must
    preserve; CHECK constraints are asserted functionally by separate tests.
    """
    columns = [
        (row["name"], row["type"], row["notnull"], row["dflt_value"], row["pk"])
        for row in connection.execute(f"PRAGMA table_info({table})")
    ]
    foreign_keys = [
        (row["from"], row["to"], row["table"], row["on_update"], row["on_delete"])
        for row in connection.execute(f"PRAGMA foreign_key_list({table})")
    ]
    indexes = sorted(
        (
            row["name"],
            row["unique"],
            row["partial"],
            tuple(
                info["name"]
                for info in connection.execute(f"PRAGMA index_info({row['name']})")
            ),
        )
        for row in connection.execute(f"PRAGMA index_list({table})")
        if row["origin"] != "pk"
    )
    return {"columns": columns, "foreign_keys": foreign_keys, "indexes": indexes}


class FakeWriteAdapter:
    """Records command/readback invocations and returns a configurable readback."""

    def __init__(
        self,
        *,
        readback_value: ObservedValue | None = None,
        command_error: Exception | None = None,
        readback_error: Exception | None = None,
    ) -> None:
        self.readback_value = readback_value
        self.command_error = command_error
        self.readback_error = readback_error
        self.command_calls = 0
        self.readback_calls = 0

    def command(self, intent) -> None:
        self.command_calls += 1
        if self.command_error is not None:
            raise self.command_error

    def readback(self, intent) -> ObservedValue:
        self.readback_calls += 1
        if self.readback_error is not None:
            raise self.readback_error
        if self.readback_value is not None:
            return self.readback_value
        return intent.requested_value


class MutatingThenErrorAdapter(FakeWriteAdapter):
    """Apply a side effect to external state, then raise an arbitrary exception.

    Models a command that actually took effect on the external system but then failed with an
    exception that is neither ``AmbiguousCommandOutcomeError`` nor ``DeterministicCommandError`` --
    the exact scenario that must fail closed to ``COMMAND_UNKNOWN``, never ``COMMAND_FAILED``.
    """

    def __init__(self, *, external_state: dict, error: Exception) -> None:
        super().__init__(command_error=error)
        self.external_state = external_state

    def command(self, intent) -> None:
        self.command_calls += 1
        self.external_state["favorited"] = intent.requested_value.payload
        raise self.command_error


class WriteExecutionRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def save_pending_intent(self, value: object = True):
        intent = favorited_intent(value)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        return intent

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

    def orchestrator(self, repository: WriteExecutionRepository, adapter: FakeWriteAdapter):
        return WriteOrchestrator(repository, adapter, policy=permissive_policy)

    def build_populated_v9(self) -> tuple:
        """Create a v9 store holding canonical model + one intent + requirement + STARTED attempt."""
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        intent = favorited_intent(True)
        with patch("music_agent.repository.MIGRATIONS", V9_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
            with PendingIntentRepository(self.database_path) as repository:
                repository.save_intent(intent)
            with WriteExecutionRepository(self.database_path) as repository:
                attempt = repository.begin_execution(intent.intent_id)
        return fixture, track_id, binding_key, intent, attempt

    # --- schema / migration -------------------------------------------------

    def test_fresh_database_reaches_v6_with_execution_attempt_table(self) -> None:
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("write_execution_attempts", tables)
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(write_execution_attempts)"
                )
            }
            self.assertEqual(
                columns, {"attempt_id", "intent_id", "state", "created_at", "updated_at"}
            )

    def test_real_v4_store_upgrades_to_v6_without_changing_canonical_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with patch("music_agent.repository.MIGRATIONS", V4_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 4)

        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("write_execution_attempts", tables)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_v5_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V4_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 4)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v5(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                WriteExecutionRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                4,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v5'"
                ).fetchone()
            )

    def test_v9_store_upgrades_to_v10_preserving_data_and_accepts_ambiguous_outcome(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        intent = favorited_intent(True)
        with patch("music_agent.repository.MIGRATIONS", V9_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 9)
            with PendingIntentRepository(self.database_path) as repository:
                repository.save_intent(intent)
            with WriteExecutionRepository(self.database_path) as repository:
                attempt = repository.begin_execution(intent.intent_id)
                self.assertEqual(repository.schema_version, 9)

        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.STARTED
            )
            advanced = repository.record_command_unknown(intent.intent_id, attempt.attempt_id)
            self.assertIs(advanced.state, AttemptState.COMMAND_UNKNOWN)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)

    def test_v10_migration_preserves_schema_structure_except_widened_checks(self) -> None:
        self.build_populated_v9()

        def signatures() -> dict:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            try:
                return {table: _table_signature(connection, table) for table in REBUILT_TABLES}
            finally:
                connection.close()

        v9 = signatures()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
        v10 = signatures()

        for table in REBUILT_TABLES:
            self.assertEqual(v9[table], v10[table], f"schema drift rebuilding {table}")

    def test_v10_migration_preserves_rows_and_foreign_key_integrity(self) -> None:
        fixture, track_id, binding_key, intent, attempt = self.build_populated_v9()

        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(repository.get_intent(intent.intent_id), intent)
            self.assertEqual(repository.get_attempt(attempt.attempt_id), attempt)
            self.assertEqual(repository.get_latest_attempt(intent.intent_id), attempt)
            # The rebuilt tables re-declare their foreign keys and preserve every referenced row.
            self.assertEqual(
                list(repository._connection.execute("PRAGMA foreign_key_check")), []
            )
            requirement = repository._connection.execute(
                """SELECT role, canonical_id, source_system, entity_type, external_id
                FROM pending_write_intent_requirements WHERE intent_id=?""",
                (intent.intent_id,),
            ).fetchone()
            self.assertIsNotNone(requirement)
            self.assertEqual(requirement["role"], "target")
            self.assertEqual(requirement["canonical_id"], track_id)
            self.assertEqual(requirement["source_system"], "apple_music")
            self.assertEqual(requirement["entity_type"], "track")
            self.assertEqual(requirement["external_id"], "SYNTH-TRACK-001")

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)

    def test_v10_schema_accepts_new_states_and_rejects_illegal_states(self) -> None:
        _, track_id, _, intent, _ = self.build_populated_v9()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            connection = repository._connection
            # The widened CHECKs accept the two new enum values.
            connection.execute(
                """INSERT INTO pending_write_intents(
                    intent_id, operation, requested_state, requested_value_json, lifecycle_state
                ) VALUES (?, 'set_favorited', 'value', 'true', 'outcome_unknown')""",
                ("int_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",),
            )
            connection.execute(
                """INSERT INTO write_execution_attempts(attempt_id, intent_id, state)
                VALUES (?, ?, 'command_unknown')""",
                ("att_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", intent.intent_id),
            )
            # The original illegal states remain rejected.
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO pending_write_intents(
                        intent_id, operation, requested_state, requested_value_json, lifecycle_state
                    ) VALUES (?, 'set_favorited', 'value', 'true', 'bogus')""",
                    ("int_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO write_execution_attempts(attempt_id, intent_id, state)
                    VALUES (?, ?, 'bogus')""",
                    ("att_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", intent.intent_id),
                )
            # The requirements CHECKs are unchanged.
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO pending_write_intent_requirements(
                        intent_id, role, canonical_id, source_system, entity_type, external_id
                    ) VALUES (?, 'bogus', ?, 'apple_music', 'track', 'x')""",
                    ("int_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", track_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO pending_write_intent_requirements(
                        intent_id, role, canonical_id, source_system, entity_type, external_id
                    ) VALUES (?, 'target', ?, 'apple_music', 'bogus', 'x')""",
                    ("int_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", track_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO pending_write_intent_requirements(
                        intent_id, role, canonical_id, source_system, entity_type, external_id
                    ) VALUES (?, 'target', ?, 'apple_music', 'track', '')""",
                    ("int_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", track_id),
                )

    def test_v10_active_attempt_uniqueness_invariant_preserved_after_migration(self) -> None:
        _, _, _, intent, _ = self.build_populated_v9()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            # A second STARTED attempt for the same intent is still rejected by the preserved
            # partial unique index.
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    """INSERT INTO write_execution_attempts(attempt_id, intent_id, state)
                    VALUES (?, ?, 'started')""",
                    ("att_ffffffff-ffff-4fff-8fff-ffffffffffff", intent.intent_id),
                )

    # --- attempt durable fact ----------------------------------------------

    def test_started_attempt_is_durable_with_independent_identity(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            self.assertIs(attempt.state, AttemptState.STARTED)
            self.assertTrue(attempt.attempt_id.startswith("att_"))
            self.assertNotEqual(attempt.attempt_id, intent.intent_id)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.get_attempt(attempt.attempt_id), attempt)
            self.assertEqual(repository.get_latest_attempt(intent.intent_id), attempt)

    # --- command outcome persistence ---------------------------------------

    def test_record_command_success_advances_attempt_and_intent(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            advanced = repository.record_command_success(intent.intent_id, attempt.attempt_id)
            self.assertIs(advanced.state, AttemptState.COMMAND_SUCCEEDED)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK)

    def test_record_command_failure_advances_attempt_and_intent(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            advanced = repository.record_command_failure(intent.intent_id, attempt.attempt_id)
            self.assertIs(advanced.state, AttemptState.COMMAND_FAILED)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.EXECUTION_FAILED)

    def test_record_command_unknown_advances_attempt_and_intent(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            advanced = repository.record_command_unknown(intent.intent_id, attempt.attempt_id)
            self.assertIs(advanced.state, AttemptState.COMMAND_UNKNOWN)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)

    def test_command_outcome_and_intent_transition_are_atomic(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            repository._connection.execute(
                """CREATE TRIGGER fail_attempt_update BEFORE UPDATE ON write_execution_attempts
                BEGIN SELECT RAISE(ABORT, 'attempt update failure'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.record_command_success(intent.intent_id, attempt.attempt_id)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)
            self.assertIs(repository.get_attempt(attempt.attempt_id).state, AttemptState.STARTED)

    # --- readback orchestration --------------------------------------------

    def test_matching_readback_confirms_without_rerunning_command(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state, IntentState.CONFIRMED
            )
        self.assertEqual(adapter.command_calls, 1)
        self.assertEqual(adapter.readback_calls, 1)

    def test_mismatching_readback_lands_on_readback_mismatch(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(readback_value=ObservedValue.value(False))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.READBACK_MISMATCH,
            )

    def test_missing_readback_does_not_confirm(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(readback_value=ObservedValue.missing())
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK)

    def test_readback_exception_does_not_rerun_command(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(readback_error=RuntimeError("readback boom"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
        self.assertEqual(adapter.command_calls, 1)
        self.assertEqual(adapter.readback_calls, 1)

    def test_readback_requires_awaiting_readback(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, FakeWriteAdapter())
            with self.assertRaises(WriteTransitionError):
                orchestrator.resume_readback(intent.intent_id)

    def test_stale_readback_result_never_overwrites_terminal_state(self) -> None:
        # Two callers observe the same AWAITING_READBACK intent but reach different readback
        # results. The first durable transition wins; the second, stale result must fail closed
        # and never overwrite the terminal state.
        confirmed = self.save_pending_intent(value=True)
        mismatched = self.save_pending_intent(value=True)

        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, FakeWriteAdapter())
            orchestrator.execute_pending_intent(confirmed.intent_id)
            self.assertIs(orchestrator.resume_readback(confirmed.intent_id).state, IntentState.CONFIRMED)
            with self.assertRaises(WriteTransitionError):
                repository.record_readback(confirmed.intent_id, WriteEvent.READBACK_MISMATCHED)

            mismatch_orchestrator = self.orchestrator(
                repository, FakeWriteAdapter(readback_value=ObservedValue.value(False))
            )
            mismatch_orchestrator.execute_pending_intent(mismatched.intent_id)
            self.assertIs(
                mismatch_orchestrator.resume_readback(mismatched.intent_id).state,
                IntentState.READBACK_MISMATCH,
            )
            with self.assertRaises(WriteTransitionError):
                repository.record_readback(mismatched.intent_id, WriteEvent.READBACK_MATCHED)

        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(confirmed.intent_id).state, IntentState.CONFIRMED)
            self.assertIs(
                repository.get_intent(mismatched.intent_id).state, IntentState.READBACK_MISMATCH
            )

    # --- restart / crash windows -------------------------------------------

    def test_crash_after_started_does_not_reexecute_command(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            repository.begin_execution(intent.intent_id)

        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            with self.assertRaises(AmbiguousAttemptError):
                orchestrator.execute_pending_intent(intent.intent_id)
        self.assertEqual(adapter.command_calls, 0)

    def test_crash_after_command_success_resumes_readback_only(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.orchestrator(repository, adapter).execute_pending_intent(intent.intent_id)

        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, adapter).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(adapter.command_calls, 1)
        self.assertEqual(adapter.readback_calls, 1)

    def test_resume_fails_closed_on_ambiguous_started(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            repository.begin_execution(intent.intent_id)

        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            with self.assertRaises(AmbiguousAttemptError):
                self.orchestrator(repository, adapter).resume(intent.intent_id)
        self.assertEqual(adapter.command_calls, 0)

    # --- concurrency / duplicate-command prevention ------------------------

    def test_begin_execution_fails_closed_on_existing_started_attempt(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            repository.begin_execution(intent.intent_id)
            with self.assertRaises(AmbiguousAttemptError):
                repository.begin_execution(intent.intent_id)
            self.assertEqual(len(repository.list_attempts(intent.intent_id)), 1)

    def test_begin_execution_requires_pending_intent(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            repository.record_command_failure(intent.intent_id, attempt.attempt_id)
            with self.assertRaises(ExecutionNotPendingError):
                repository.begin_execution(intent.intent_id)

    def test_database_rejects_second_started_attempt_directly(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            repository.begin_execution(intent.intent_id)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    """INSERT INTO write_execution_attempts(attempt_id, intent_id, state)
                    VALUES (?, ?, ?)""",
                    ("att_ffffffff-ffff-4fff-8fff-ffffffffffff", intent.intent_id, "started"),
                )

    def test_two_workers_cannot_create_two_active_executions(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)

        results = []
        errors = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            repository = WriteExecutionRepository(self.database_path)
            try:
                barrier.wait()
                results.append(repository.begin_execution(intent.intent_id))
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
        self.assertIsInstance(errors[0], AmbiguousAttemptError)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_attempts(intent.intent_id)), 1)

    # --- stale caller fail closed ------------------------------------------

    def test_stale_command_outcome_fails_closed(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            repository.record_command_success(intent.intent_id, attempt.attempt_id)
            with self.assertRaises(WriteTransitionError):
                repository.record_command_failure(intent.intent_id, attempt.attempt_id)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.AWAITING_READBACK)
            self.assertIs(repository.get_attempt(attempt.attempt_id).state, AttemptState.COMMAND_SUCCEEDED)

    def test_command_outcome_requires_matching_attempt(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
            other = self.save_pending_intent()
            with self.assertRaises(AttemptNotFoundError):
                repository.record_command_success(other.intent_id, attempt.attempt_id)
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)
            self.assertIs(repository.get_attempt(attempt.attempt_id).state, AttemptState.STARTED)

    def test_attempt_outcome_cannot_flip_after_success(self) -> None:
        attempt = ExecutionAttempt(
            "att_11111111-1111-4111-8111-111111111111",
            "int_11111111-1111-4111-8111-111111111111",
            AttemptState.COMMAND_SUCCEEDED,
        )
        with self.assertRaises(AttemptTransitionError):
            advance_attempt(attempt, AttemptEvent.COMMAND_FAILED)

    # --- command exception --------------------------------------------------

    def test_unknown_command_exception_fails_closed_to_unknown(self) -> None:
        # A generic exception is not positive proof that the command never dispatched, so it must
        # be recorded as COMMAND_UNKNOWN / OUTCOME_UNKNOWN -- never as a deterministic failure --
        # even when nothing observable changed.
        intent = self.save_pending_intent()
        adapter = FakeWriteAdapter(command_error=RuntimeError("command boom"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.COMMAND_UNKNOWN
            )
        self.assertEqual(adapter.command_calls, 1)

    def test_deterministic_command_error_records_failure_not_unknown(self) -> None:
        # DeterministicCommandError is positive proof the command never dispatched, so it is the
        # only exception mapped to a deterministic COMMAND_FAILED / EXECUTION_FAILED.
        intent = self.save_pending_intent()
        adapter = FakeWriteAdapter(command_error=DeterministicCommandError("validation failed"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            with self.assertRaises(DeterministicCommandError):
                orchestrator.execute_pending_intent(intent.intent_id)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.EXECUTION_FAILED)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.COMMAND_FAILED
            )
        self.assertEqual(adapter.command_calls, 1)

    def test_raise_after_mutate_generic_error_fails_closed_to_unknown(self) -> None:
        # The command applied the requested mutation and then raised a plain RuntimeError. The
        # orchestrator cannot prove no side effect occurred, so it must record COMMAND_UNKNOWN /
        # OUTCOME_UNKNOWN, and a restart must reconcile by readback without replaying the command.
        intent = self.save_pending_intent(value=True)
        external_state: dict = {}
        adapter = MutatingThenErrorAdapter(
            external_state=external_state, error=RuntimeError("post-mutation boom")
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.COMMAND_UNKNOWN
            )
        self.assertEqual(external_state, {"favorited": True})
        self.assertEqual(adapter.command_calls, 1)

        fresh_adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, fresh_adapter).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(fresh_adapter.command_calls, 0)
        self.assertEqual(fresh_adapter.readback_calls, 1)

    # --- ambiguous command outcome / reconciliation ------------------------

    def test_ambiguous_command_outcome_records_unknown_not_failure(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(command_error=AmbiguousCommandOutcomeError("timeout"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.COMMAND_UNKNOWN
            )
        self.assertEqual(adapter.command_calls, 1)

    def test_ambiguous_command_outcome_confirms_via_matching_readback_without_reexecution(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(command_error=AmbiguousCommandOutcomeError("timeout"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.CONFIRMED)
        self.assertEqual(adapter.command_calls, 1)
        self.assertEqual(adapter.readback_calls, 1)

    def test_ambiguous_command_outcome_mismatched_readback_stays_unknown(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(
            command_error=AmbiguousCommandOutcomeError("timeout"),
            readback_value=ObservedValue.value(False),
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)
            self.assertIs(
                repository.get_latest_attempt(intent.intent_id).state, AttemptState.COMMAND_UNKNOWN
            )
        self.assertEqual(adapter.command_calls, 1)

    def test_ambiguous_command_outcome_unavailable_readback_stays_unknown(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(
            command_error=AmbiguousCommandOutcomeError("timeout"),
            readback_value=ObservedValue.missing(),
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.OUTCOME_UNKNOWN)

    def test_ambiguous_command_outcome_does_not_reexecute_on_resume(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(command_error=AmbiguousCommandOutcomeError("timeout"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(orchestrator.resume(intent.intent_id).state, IntentState.CONFIRMED)
        self.assertEqual(adapter.command_calls, 1)

    def test_restart_reconciles_unknown_outcome_without_reexecution(self) -> None:
        intent = self.save_pending_intent(value=True)
        with WriteExecutionRepository(self.database_path) as repository:
            self.orchestrator(
                repository, FakeWriteAdapter(command_error=AmbiguousCommandOutcomeError("timeout"))
            ).execute_pending_intent(intent.intent_id)

        fresh_adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, fresh_adapter).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(fresh_adapter.command_calls, 0)
        self.assertEqual(fresh_adapter.readback_calls, 1)

    def test_reconciliation_gate_requires_command_unknown(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            repository.begin_execution(intent.intent_id)
            with self.assertRaises(ReadbackGateError):
                repository.record_reconciliation(intent.intent_id)

    def test_reconcile_requires_outcome_unknown_intent(self) -> None:
        intent = self.save_pending_intent()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, FakeWriteAdapter())
            with self.assertRaises(WriteTransitionError):
                orchestrator.reconcile_unknown_outcome(intent.intent_id)

    # --- readback retry / recovery ----------------------------------------

    def test_known_success_unavailable_readback_then_matched_confirms_once(self) -> None:
        intent = self.save_pending_intent(value=True)
        adapter = FakeWriteAdapter(readback_value=ObservedValue.missing())
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
            # A later readback that now matches confirms without a second command.
            adapter.readback_value = ObservedValue.value(True)
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state, IntentState.CONFIRMED
            )
        self.assertEqual(adapter.command_calls, 1)
        self.assertEqual(adapter.readback_calls, 2)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_attempts(intent.intent_id)), 1)

    def test_known_success_raising_readback_then_restart_matched_confirms_once(self) -> None:
        intent = self.save_pending_intent(value=True)
        failing = FakeWriteAdapter(readback_error=RuntimeError("readback boom"))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, failing)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )
            # A readback exception never turns the known command success into a failure.
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.AWAITING_READBACK,
            )

        fresh = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, fresh).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(failing.command_calls, 1)
        self.assertEqual(failing.readback_calls, 1)
        self.assertEqual(fresh.command_calls, 0)
        self.assertEqual(fresh.readback_calls, 1)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_attempts(intent.intent_id)), 1)

    def test_unknown_outcome_unavailable_reconcile_then_restart_matched_confirms_once(self) -> None:
        intent = self.save_pending_intent(value=True)
        unavailable = FakeWriteAdapter(
            command_error=AmbiguousCommandOutcomeError("timeout"),
            readback_value=ObservedValue.missing(),
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, unavailable)
            self.assertIs(
                orchestrator.execute_pending_intent(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
            # Unavailable reconciliation neither forges success nor failure.
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )

        fresh = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, fresh).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(unavailable.command_calls, 1)
        self.assertEqual(unavailable.readback_calls, 1)
        self.assertEqual(fresh.command_calls, 0)
        self.assertEqual(fresh.readback_calls, 1)

    def test_unknown_outcome_raising_reconcile_then_matched_confirms_once(self) -> None:
        intent = self.save_pending_intent(value=True)
        failing = FakeWriteAdapter(
            command_error=AmbiguousCommandOutcomeError("timeout"),
            readback_error=RuntimeError("readback boom"),
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, failing)
            orchestrator.execute_pending_intent(intent.intent_id)
            # A raising reconciliation readback leaves the outcome unknown and recoverable.
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )
            failing.readback_error = None
            failing.readback_value = ObservedValue.value(True)
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(failing.command_calls, 1)
        self.assertEqual(failing.readback_calls, 2)

    def test_known_success_mismatch_is_terminal_and_does_not_retry(self) -> None:
        intent = self.save_pending_intent(value=True)
        mismatching = FakeWriteAdapter(readback_value=ObservedValue.value(False))
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, mismatching)
            orchestrator.execute_pending_intent(intent.intent_id)
            self.assertIs(
                orchestrator.resume_readback(intent.intent_id).state,
                IntentState.READBACK_MISMATCH,
            )

        fresh = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            # READBACK_MISMATCH is terminal: resume returns it unchanged and never re-reads or
            # re-runs the command.
            self.assertIs(
                self.orchestrator(repository, fresh).resume(intent.intent_id).state,
                IntentState.READBACK_MISMATCH,
            )
        self.assertEqual(mismatching.command_calls, 1)
        self.assertEqual(fresh.command_calls, 0)
        self.assertEqual(fresh.readback_calls, 0)

    def test_unknown_outcome_mismatch_stays_unknown_then_later_matched_confirms(self) -> None:
        intent = self.save_pending_intent(value=True)
        mismatching = FakeWriteAdapter(
            command_error=AmbiguousCommandOutcomeError("timeout"),
            readback_value=ObservedValue.value(False),
        )
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, mismatching)
            orchestrator.execute_pending_intent(intent.intent_id)
            # A mismatching reconciliation readback must not be read as "command failed"; it stays
            # OUTCOME_UNKNOWN and remains recoverable.
            self.assertIs(
                orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )

        fresh = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(
                self.orchestrator(repository, fresh).resume(intent.intent_id).state,
                IntentState.CONFIRMED,
            )
        self.assertEqual(mismatching.command_calls, 1)
        self.assertEqual(fresh.command_calls, 0)
        self.assertEqual(fresh.readback_calls, 1)

    def test_unknown_outcome_is_explicit_in_returned_intent_state(self) -> None:
        # Caller-visible UNKNOWN semantics: an ambiguous or generic command outcome returns an
        # intent whose lifecycle_state is OUTCOME_UNKNOWN -- never a success state -- while a
        # deterministic failure raises instead of returning a silently misreadable intent.
        ambiguous = self.save_pending_intent(value=True)
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(
                repository, FakeWriteAdapter(command_error=AmbiguousCommandOutcomeError("timeout"))
            )
            result = orchestrator.execute_pending_intent(ambiguous.intent_id)
            self.assertIs(result.state, IntentState.OUTCOME_UNKNOWN)
            self.assertIsNot(result.state, IntentState.CONFIRMED)

        generic = self.save_pending_intent(value=True)
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(
                repository, FakeWriteAdapter(command_error=RuntimeError("boom"))
            )
            self.assertIs(
                orchestrator.execute_pending_intent(generic.intent_id).state,
                IntentState.OUTCOME_UNKNOWN,
            )

        deterministic = self.save_pending_intent(value=True)
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(
                repository, FakeWriteAdapter(command_error=DeterministicCommandError("no dispatch"))
            )
            with self.assertRaises(DeterministicCommandError):
                orchestrator.execute_pending_intent(deterministic.intent_id)

    # --- production gate / canonical isolation -----------------------------

    def test_production_policy_rejects_execution(self) -> None:
        intent = self.save_pending_intent()
        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = WriteOrchestrator(repository, adapter)
            with self.assertRaises(NotExecutionReadyError):
                orchestrator.execute_pending_intent(intent.intent_id)
        self.assertEqual(adapter.command_calls, 0)

    def test_current_execution_ready_set_remains_empty(self) -> None:
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))

    def test_confirmed_intent_does_not_mutate_canonical_state(self) -> None:
        fixture, track_id, before_counts = self.save_canonical_model()
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")

        intent = favorited_intent(True)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        adapter = FakeWriteAdapter()
        with WriteExecutionRepository(self.database_path) as repository:
            orchestrator = self.orchestrator(repository, adapter)
            orchestrator.execute_pending_intent(intent.intent_id)
            orchestrator.resume_readback(intent.intent_id)

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


if __name__ == "__main__":
    unittest.main()
