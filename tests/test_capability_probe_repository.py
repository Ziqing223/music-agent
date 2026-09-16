import copy
import dataclasses
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    finalize,
    mark_forward_started,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_repository import (
    CapabilityProbeRepository,
    ProbeConflictError,
    ProbeNotFoundError,
    StaleProbeUpdateError,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.source_observation import ObservationState, ObservedValue, SourcePresence
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
V6_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
    (6, "0006_pending_write_intent_requirements.sql"),
)

STEP_STAGES = (
    "baseline_captured",
    "forward_started",
    "forward_observed",
    "restore_started",
    "restore_observed",
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def missing() -> ObservedValue:
    return ObservedValue.missing()


def track_key() -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)


def favorited_intent() -> object:
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.value(True)
    )


def probe_at(stage: str) -> CapabilityProbe:
    probe = create_probe(TRACK_ID, TRACK_PID, False, False)
    if stage == "baseline_captured":
        return probe
    probe = mark_forward_started(probe)
    if stage == "forward_started":
        return probe
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
    if stage == "forward_observed":
        return probe
    probe = mark_restore_started(probe)
    if stage == "restore_started":
        return probe
    if stage == "restore_observed":
        return observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))
    raise AssertionError(f"unknown stage {stage!r}")


class CapabilityProbeRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    # --- schema / migration -------------------------------------------------

    def test_fresh_database_reaches_v7_with_probe_table(self) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("capability_probes", tables)
            columns = {
                row[1]
                for row in repository._connection.execute("PRAGMA table_info(capability_probes)")
            }
            self.assertEqual(
                columns,
                {
                    "probe_id",
                    "operation",
                    "target_canonical_id",
                    "target_persistent_id",
                    "baseline_favorited",
                    "baseline_disliked",
                    "step_state",
                    "recovery_status",
                    "verification_verdict",
                    "forward_command_outcome",
                    "forward_favorited_state",
                    "forward_favorited_value",
                    "forward_disliked_state",
                    "forward_disliked_value",
                    "restore_command_outcome",
                    "restore_favorited_state",
                    "restore_favorited_value",
                    "restore_disliked_state",
                    "restore_disliked_value",
                    "created_at",
                    "updated_at",
                },
            )

    def test_three_axes_are_independent_columns(self) -> None:
        probe = probe_at("forward_observed")
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            row = repository._connection.execute(
                "SELECT step_state, recovery_status, verification_verdict "
                "FROM capability_probes WHERE probe_id=?",
                (probe.probe_id,),
            ).fetchone()
        self.assertEqual(row["step_state"], "forward_observed")
        self.assertEqual(row["recovery_status"], "baseline_confirmed")
        self.assertEqual(row["verification_verdict"], "pending")

    def test_real_v6_store_upgrades_to_v7_without_changing_canonical_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with patch("music_agent.repository.MIGRATIONS", V6_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 6)

        with CapabilityProbeRepository(self.database_path) as repository:
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

    def test_v7_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V6_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 6)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v7(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                CapabilityProbeRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                6,
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v7'"
                ).fetchone()
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='capability_probes'"
                ).fetchone()
            )

    # --- exact round-trip ---------------------------------------------------

    def test_initial_probe_exact_round_trip(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_false_baseline_round_trip(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            loaded = repository.get_probe(probe.probe_id)
        self.assertIs(loaded.baseline_favorited, False)
        self.assertIs(loaded.baseline_disliked, False)

    def test_true_baseline_round_trip(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, True, True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            loaded = repository.get_probe(probe.probe_id)
        self.assertIs(loaded.baseline_favorited, True)
        self.assertIs(loaded.baseline_disliked, True)

    def test_missing_observation_round_trip(self) -> None:
        probe = mark_forward_started(create_probe(TRACK_ID, TRACK_PID, False, False))
        probe = observe_forward(probe, CommandOutcome.SUCCESS, missing(), value(False))
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            loaded = repository.get_probe(probe.probe_id)
        self.assertIsNotNone(loaded.forward_favorited)
        self.assertIs(loaded.forward_favorited.state, ObservationState.MISSING)
        self.assertNotEqual(loaded.forward_favorited, value(False))
        self.assertIs(loaded.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_clean_path_intermediate_states_round_trip(self) -> None:
        probes = {stage: probe_at(stage) for stage in STEP_STAGES}
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in probes.values():
                repository.save_probe(probe)
            loaded = {stage: repository.get_probe(probe.probe_id) for stage, probe in probes.items()}
        for stage in STEP_STAGES:
            self.assertEqual(loaded[stage], probes[stage])
        self.assertIs(loaded["baseline_captured"].step_state, ProbeStepState.BASELINE_CAPTURED)
        self.assertIs(loaded["forward_started"].step_state, ProbeStepState.FORWARD_STARTED)
        self.assertIs(loaded["forward_observed"].step_state, ProbeStepState.FORWARD_OBSERVED)
        self.assertIs(loaded["restore_started"].step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(loaded["restore_observed"].step_state, ProbeStepState.RESTORE_OBSERVED)

    def test_recovery_status_three_values_round_trip(self) -> None:
        confirmed = create_probe(TRACK_ID, TRACK_PID, False, False)
        restored = finalize(probe_at("restore_observed"))
        manual = mark_recovery_status(
            create_probe(TRACK_ID, TRACK_PID, False, False), RecoveryStatus.NEEDS_MANUAL_CHECK
        )
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in (confirmed, restored, manual):
                repository.save_probe(probe)
            self.assertIs(
                repository.get_probe(confirmed.probe_id).recovery_status,
                RecoveryStatus.BASELINE_CONFIRMED,
            )
            self.assertIs(
                repository.get_probe(restored.probe_id).recovery_status, RecoveryStatus.RESTORED
            )
            self.assertIs(
                repository.get_probe(manual.probe_id).recovery_status,
                RecoveryStatus.NEEDS_MANUAL_CHECK,
            )

    def test_verdict_four_values_round_trip(self) -> None:
        pending = create_probe(TRACK_ID, TRACK_PID, False, False)
        verified = finalize(probe_at("restore_observed"))
        failed = mark_forward_started(create_probe(TRACK_ID, TRACK_PID, False, False))
        failed = observe_forward(failed, CommandOutcome.SUCCESS, value(True), value(True))
        inconclusive = mark_forward_started(create_probe(TRACK_ID, TRACK_PID, False, False))
        inconclusive = observe_forward(
            inconclusive, CommandOutcome.UNKNOWN, value(True), value(False)
        )
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in (pending, verified, failed, inconclusive):
                repository.save_probe(probe)
            self.assertIs(
                repository.get_probe(pending.probe_id).verification_verdict,
                VerificationVerdict.PENDING,
            )
            self.assertIs(
                repository.get_probe(verified.probe_id).verification_verdict,
                VerificationVerdict.VERIFIED,
            )
            self.assertIs(
                repository.get_probe(failed.probe_id).verification_verdict,
                VerificationVerdict.FAILED,
            )
            self.assertIs(
                repository.get_probe(inconclusive.probe_id).verification_verdict,
                VerificationVerdict.INCONCLUSIVE,
            )

    # --- restart / idempotency / conflict ----------------------------------

    def test_probe_survives_restart(self) -> None:
        probe = finalize(probe_at("restore_observed"))
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(probe.probe_id), probe)
            self.assertEqual(
                [candidate.probe_id for candidate in repository.list_probes()], [probe.probe_id]
            )

    def test_resave_exact_content_is_idempotent(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.save_probe(probe)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM capability_probes"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_changed_baseline_conflicts_fail_closed(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        conflicting = dataclasses.replace(probe, baseline_favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            with self.assertRaises(ProbeConflictError):
                repository.save_probe(conflicting)
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_changed_persistent_id_conflicts_fail_closed(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        conflicting = dataclasses.replace(probe, target_persistent_id="SYNTH-TRACK-999")
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            with self.assertRaises(ProbeConflictError):
                repository.save_probe(conflicting)
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    # --- update (durable transition) ---------------------------------------

    def test_update_probe_advances_mutable_state(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        advanced = mark_forward_started(probe)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, advanced)
            self.assertEqual(repository.get_probe(probe.probe_id), advanced)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM capability_probes"
                ).fetchone()[0],
                1,
            )

    def test_update_probe_rejects_immutable_change(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        conflicting = dataclasses.replace(probe, baseline_disliked=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            with self.assertRaises(ProbeConflictError):
                repository.update_probe(probe, conflicting)
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_update_probe_missing_fails_closed(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            with self.assertRaises(ProbeNotFoundError):
                repository.update_probe(probe, mark_forward_started(probe))

    def test_update_probe_same_content_is_noop(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, probe)
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM capability_probes"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    # --- stale-object update guard (optimistic CAS) ------------------------

    def test_stale_forward_started_cannot_overwrite_forward_observed(self) -> None:
        base = create_probe(TRACK_ID, TRACK_PID, False, False)
        forward_started = mark_forward_started(base)
        forward_observed = observe_forward(
            forward_started, CommandOutcome.SUCCESS, value(True), value(False)
        )
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(base)
            repository.update_probe(base, forward_started)
            repository.update_probe(forward_started, forward_observed)
            with self.assertRaises(StaleProbeUpdateError):
                repository.update_probe(base, forward_started)
            self.assertEqual(repository.get_probe(base.probe_id), forward_observed)
            self.assertIs(
                repository.get_probe(base.probe_id).step_state, ProbeStepState.FORWARD_OBSERVED
            )

    def test_stale_object_cannot_overwrite_verified(self) -> None:
        restore_observed = probe_at("restore_observed")
        verified = finalize(restore_observed)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(verified)
            with self.assertRaises(StaleProbeUpdateError):
                repository.update_probe(restore_observed, verified)
            self.assertEqual(repository.get_probe(verified.probe_id), verified)
            self.assertIs(
                repository.get_probe(verified.probe_id).verification_verdict,
                VerificationVerdict.VERIFIED,
            )

    def test_stale_recovery_status_cannot_regress(self) -> None:
        restore_observed = probe_at("restore_observed")
        verified = finalize(restore_observed)
        stale = mark_recovery_status(restore_observed, RecoveryStatus.NEEDS_MANUAL_CHECK)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(verified)
            with self.assertRaises(StaleProbeUpdateError):
                repository.update_probe(restore_observed, stale)
            self.assertIs(
                repository.get_probe(verified.probe_id).recovery_status, RecoveryStatus.RESTORED
            )

    def test_stale_update_cannot_erase_restore_observation(self) -> None:
        base = create_probe(TRACK_ID, TRACK_PID, False, False)
        forward_started = mark_forward_started(base)
        forward_observed = observe_forward(
            forward_started, CommandOutcome.SUCCESS, value(True), value(False)
        )
        restore_started = mark_restore_started(forward_observed)
        restore_observed = observe_restore(
            restore_started, CommandOutcome.SUCCESS, value(False), value(False)
        )
        verified = finalize(restore_observed)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(verified)
            with self.assertRaises(StaleProbeUpdateError):
                repository.update_probe(restore_started, restore_observed)
            loaded = repository.get_probe(verified.probe_id)
            self.assertEqual(loaded, verified)
            self.assertIsNotNone(loaded.restore_command_outcome)
            self.assertIsNotNone(loaded.restore_favorited)
            self.assertIsNotNone(loaded.restore_disliked)

    def test_update_probe_requires_matching_probe_id(self) -> None:
        first = create_probe(TRACK_ID, TRACK_PID, False, False)
        second = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            with self.assertRaises(ProbeConflictError):
                repository.update_probe(first, mark_forward_started(second))

    def test_save_probe_rejects_mutable_difference_as_conflict(self) -> None:
        base = create_probe(TRACK_ID, TRACK_PID, False, False)
        advanced = mark_forward_started(base)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(base)
            with self.assertRaises(ProbeConflictError):
                repository.save_probe(advanced)
            self.assertEqual(repository.get_probe(base.probe_id), base)

    # --- transaction failure rollback --------------------------------------

    def test_save_probe_rolls_back_on_failure(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository._connection.execute(
                """CREATE TRIGGER fail_probe_insert BEFORE INSERT ON capability_probes
                BEGIN SELECT RAISE(ABORT, 'probe insert failure'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.save_probe(probe)
            self.assertIsNone(repository.get_probe(probe.probe_id))
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM capability_probes"
                ).fetchone()[0],
                0,
            )

    def test_update_probe_rolls_back_on_failure(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        advanced = mark_forward_started(probe)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository._connection.execute(
                """CREATE TRIGGER fail_probe_update BEFORE UPDATE ON capability_probes
                BEGIN SELECT RAISE(ABORT, 'probe update failure'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.update_probe(probe, advanced)
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    # --- ordering / decode determinism -------------------------------------

    def test_list_probes_is_deterministic(self) -> None:
        probes = [probe_at(stage) for stage in STEP_STAGES]
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in probes:
                repository.save_probe(probe)
            loaded = repository.list_probes()
        self.assertEqual(
            [candidate.probe_id for candidate in loaded],
            sorted(probe.probe_id for probe in probes),
        )
        self.assertEqual(
            {candidate.probe_id: candidate for candidate in loaded},
            {probe.probe_id: probe for probe in probes},
        )

    def test_get_probe_missing_returns_none(self) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_probe("prb_ffffffff-ffff-4fff-8fff-ffffffffffff"))

    # --- operational isolation ---------------------------------------------

    def test_probe_persistence_does_not_mutate_canonical_or_operational_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()

        intent = favorited_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)

        probe = create_probe(track_id, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_forward_started(probe))

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
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(intent.intent_id).state, IntentState.PENDING)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_attempt(attempt.attempt_id).state, AttemptState.STARTED)

    def test_capability_matrix_unchanged_and_set_favorited_not_execution_ready(self) -> None:
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
