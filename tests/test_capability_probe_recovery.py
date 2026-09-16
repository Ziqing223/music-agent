import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from music_agent.apple_music_write import AppleMusicWriteError
from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_restore_started,
    mark_restore_started,
    observe_forward,
)
from music_agent.capability_probe_orchestrator import (
    CapabilityProbeOrchestrator,
    ProbeNotFoundError,
    ProbeReadback,
    UnrecoverableProbeError,
)
from music_agent.capability_probe_recovery_attempt import RecoveryAttemptState
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import (
    CapabilityProbeRepository,
    StaleProbeUpdateError,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
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


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def missing() -> ObservedValue:
    return ObservedValue.missing()


def track_key() -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)


def crashed_probe(step_state: ProbeStepState) -> CapabilityProbe:
    """A probe durably crashed at ``step_state`` with its outcome still ``PENDING``."""
    probe = create_probe(TRACK_ID, TRACK_PID, False, False)
    if step_state is ProbeStepState.FORWARD_STARTED:
        return mark_forward_started(probe)
    probe = mark_forward_started(probe)
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
    return mark_restore_started(probe)


class FakeRecoveryAdapter:
    """Scripted command/readback boundary for recovery.

    ``command`` echoes the requested favorited value into internal state and records the call;
    ``readback`` reports that state unless an override (including a MISSING readback) is set. The
    only command recovery issues is the restore command, requested at the baseline value.
    """

    def __init__(self, favorited: bool, disliked: bool = False) -> None:
        self.commands: list[tuple[str, bool]] = []
        self.before_command: Callable[[bool], None] | None = None
        self.restore_command_error: Exception | None = None
        self.current_favorited: ObservedValue | None = None
        self.current_disliked: ObservedValue | None = None
        self._favorited = favorited
        self._disliked = disliked

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        if self.before_command is not None:
            self.before_command(requested_favorited)
        self.commands.append((target_persistent_id, requested_favorited))
        if self.restore_command_error is not None:
            raise self.restore_command_error
        self._favorited = requested_favorited

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        favorited = (
            self.current_favorited if self.current_favorited is not None else value(self._favorited)
        )
        disliked = (
            self.current_disliked if self.current_disliked is not None else value(self._disliked)
        )
        return ProbeReadback(favorited, disliked)


class CapabilityProbeRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _orchestrator(self, repository: CapabilityProbeRepository, adapter):
        recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
        self.addCleanup(recovery_repository.close)
        return CapabilityProbeOrchestrator(repository, recovery_repository, adapter)

    def _recover(
        self,
        step_state: ProbeStepState,
        favorited: bool = False,
        *,
        missing_current: bool = False,
        adapter: FakeRecoveryAdapter | None = None,
    ) -> tuple[CapabilityProbe, FakeRecoveryAdapter, CapabilityProbe]:
        probe = crashed_probe(step_state)
        adapter = adapter or FakeRecoveryAdapter(favorited)
        if missing_current:
            adapter.current_favorited = missing()
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        return result, adapter, probe

    def _seed_isolated_store(self) -> dict:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()
            before_model = repository.load_model()
        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, track_id, binding_key, value(True)
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
        return {
            "track_id": track_id,
            "binding_key": binding_key,
            "intent": intent,
            "attempt": attempt,
            "before_counts": before_counts,
            "before_model": before_model,
        }

    # --- FORWARD_STARTED ambiguity -----------------------------------------

    def test_forward_started_current_f0_no_restore(self) -> None:
        result, adapter, _ = self._recover(ProbeStepState.FORWARD_STARTED, favorited=False)
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.BASELINE_CONFIRMED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [])

    def test_forward_started_current_opposite_restores(self) -> None:
        result, adapter, _ = self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])

    def test_forward_started_current_missing_manual_check(self) -> None:
        result, adapter, _ = self._recover(
            ProbeStepState.FORWARD_STARTED, favorited=False, missing_current=True
        )
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [])

    # --- RESTORE_STARTED ambiguity -----------------------------------------

    def test_restore_started_current_f0_no_duplicate_restore(self) -> None:
        result, adapter, _ = self._recover(ProbeStepState.RESTORE_STARTED, favorited=False)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [])

    def test_restore_started_current_opposite_restores(self) -> None:
        result, adapter, _ = self._recover(ProbeStepState.RESTORE_STARTED, favorited=True)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])

    def test_restore_started_current_missing_manual_check(self) -> None:
        result, adapter, _ = self._recover(
            ProbeStepState.RESTORE_STARTED, favorited=False, missing_current=True
        )
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [])

    # --- STARTED-before-side-effect ----------------------------------------

    def test_restore_started_durable_before_restore_command(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        observed: list[ProbeStepState] = []
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)

            def before_command(requested_favorited: bool) -> None:
                observed.append(repository.get_probe(probe.probe_id).step_state)

            adapter.before_command = before_command
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(observed, [ProbeStepState.RESTORE_STARTED])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    # --- CAS / competing callers -------------------------------------------

    def test_cas_failure_before_restore_does_not_invoke_restore(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            real_update = repository.update_probe

            def stale_on_restore_start(expected, updated):
                if updated.step_state is ProbeStepState.RESTORE_STARTED:
                    raise StaleProbeUpdateError("concurrent restore start won")
                return real_update(expected, updated)

            with patch.object(repository, "update_probe", side_effect=stale_on_restore_start):
                orchestrator = self._orchestrator(repository, adapter)
                with self.assertRaises(StaleProbeUpdateError):
                    orchestrator.recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    def test_two_stale_callers_at_most_one_restores(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            orchestrator = self._orchestrator(repository, adapter)
            stale = repository.get_probe(probe.probe_id)
            with patch.object(repository, "get_probe", return_value=stale):
                first = orchestrator.recover_probe(probe.probe_id)
                with self.assertRaises(StaleProbeUpdateError):
                    orchestrator.recover_probe(probe.probe_id)
        self.assertEqual(len(adapter.commands), 1)
        self.assertIs(first.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(first.recovery_status, RecoveryStatus.RESTORED)

    # --- recovery reentrancy after a claim-then-crash -----------------------

    def test_restore_started_claim_then_crash_resumes(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            # Simulate recovery claiming INCONCLUSIVE, then crashing before the restore command.
            repository.update_probe(probe, mark_inconclusive(probe))
        # Restart: durable is RESTORE_STARTED + INCONCLUSIVE.
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])

    def test_forward_started_claim_then_crash_resumes(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])

    def test_forward_started_inconclusive_competing_callers_at_most_one_restores(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
            orchestrator = self._orchestrator(repository, adapter)
            stale = repository.get_probe(probe.probe_id)  # FORWARD_STARTED + INCONCLUSIVE
            with patch.object(repository, "get_probe", return_value=stale):
                first = orchestrator.recover_probe(probe.probe_id)
                with self.assertRaises(StaleProbeUpdateError):
                    orchestrator.recover_probe(probe.probe_id)
        self.assertEqual(len(adapter.commands), 1)
        self.assertIs(first.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(first.recovery_status, RecoveryStatus.RESTORED)

    def test_restart_recovery_success_never_verified(self) -> None:
        for step_state in (ProbeStepState.FORWARD_STARTED, ProbeStepState.RESTORE_STARTED):
            probe = crashed_probe(step_state)
            adapter = FakeRecoveryAdapter(favorited=True)
            with CapabilityProbeRepository(self.database_path) as repository:
                repository.save_probe(probe)
                repository.update_probe(probe, mark_inconclusive(probe))
            with CapabilityProbeRepository(self.database_path) as repository:
                result = self._orchestrator(repository, adapter).recover_probe(
                    probe.probe_id
                )
            self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
            self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
            self.assertIsNot(result.verification_verdict, VerificationVerdict.VERIFIED)

    # --- verdict separation -------------------------------------------------

    def test_successful_recovery_never_creates_verified(self) -> None:
        for step_state in (ProbeStepState.FORWARD_STARTED, ProbeStepState.RESTORE_STARTED):
            result, _, _ = self._recover(step_state, favorited=True)
            self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
            self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
            self.assertIsNot(result.verification_verdict, VerificationVerdict.VERIFIED)

    def test_restore_command_exception_is_inconclusive_needs_manual_check(self) -> None:
        adapter = FakeRecoveryAdapter(favorited=True)
        adapter.restore_command_error = AppleMusicWriteError("restore command failed")
        result, _, _ = self._recover(ProbeStepState.FORWARD_STARTED, favorited=True, adapter=adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)

    # --- restart persistence ------------------------------------------------

    def test_inconclusive_restored_survives_restart(self) -> None:
        result, _, probe = self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        with CapabilityProbeRepository(self.database_path) as repository:
            reloaded = repository.get_probe(probe.probe_id)
        self.assertEqual(reloaded, result)
        self.assertIs(reloaded.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(reloaded.recovery_status, RecoveryStatus.RESTORED)

    # --- fail-closed entry --------------------------------------------------

    def test_non_ambiguous_step_fails_closed(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = FakeRecoveryAdapter(False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            orchestrator = self._orchestrator(repository, adapter)
            with self.assertRaises(UnrecoverableProbeError):
                orchestrator.recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    def test_terminal_verdict_fails_closed(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, observe_forward(probe, CommandOutcome.FAILED, value(True), value(False)))
            orchestrator = self._orchestrator(repository, adapter)
            with self.assertRaises(UnrecoverableProbeError):
                orchestrator.recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    def test_recover_missing_probe_raises(self) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            orchestrator = self._orchestrator(repository, FakeRecoveryAdapter(False))
            with self.assertRaises(ProbeNotFoundError):
                orchestrator.recover_probe("prb_ffffffff-ffff-4fff-8fff-ffffffffffff")

    # --- operational isolation ----------------------------------------------

    def test_recovery_does_not_mutate_canonical_or_operational_state(self) -> None:
        seed = self._seed_isolated_store()
        self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
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

    def test_capability_matrix_unchanged_after_recovery(self) -> None:
        result, _, _ = self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.adapter_implemented, True)
        self.assertIs(capability.readback_implemented, True)
        self.assertIs(capability.readback_verified, False)

    def test_production_set_favorited_still_not_execution_ready(self) -> None:
        self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.readback_verified, False)
        self.assertFalse(is_execution_ready(capability))
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))

    # --- RecoveryAttempt gate ----------------------------------------------

    def _run_competing_recoveries(
        self, probe_id: str, adapter: FakeRecoveryAdapter
    ) -> tuple[list[CapabilityProbe], list[Exception]]:
        """Run two concurrent ``recover_probe`` callers, each with its own repositories."""
        barrier = threading.Barrier(2)
        results: list[CapabilityProbe] = []
        errors: list[Exception] = []

        def worker() -> None:
            repository = CapabilityProbeRepository(self.database_path)
            recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
            try:
                orchestrator = CapabilityProbeOrchestrator(
                    repository, recovery_repository, adapter
                )
                barrier.wait()
                results.append(orchestrator.recover_probe(probe_id))
            except Exception as error:
                errors.append(error)
            finally:
                recovery_repository.close()
                repository.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results, errors

    def test_no_attempt_and_source_already_baseline_creates_no_attempt(self) -> None:
        result, adapter, probe = self._recover(ProbeStepState.FORWARD_STARTED, favorited=False)
        self.assertIs(result.recovery_status, RecoveryStatus.BASELINE_CONFIRMED)
        self.assertEqual(adapter.commands, [])
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIsNone(recovery_repository.get_for_probe(probe.probe_id))

    def test_restore_attempt_started_durable_before_command(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        observed: list[RecoveryAttemptState | None] = []
        with CapabilityProbeRepository(self.database_path) as repository:
            recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
            self.addCleanup(recovery_repository.close)
            repository.save_probe(probe)

            def before_command(requested_favorited: bool) -> None:
                attempt = recovery_repository.get_for_probe(probe.probe_id)
                observed.append(attempt.state if attempt is not None else None)

            adapter.before_command = before_command
            result = CapabilityProbeOrchestrator(
                repository, recovery_repository, adapter
            ).recover_probe(probe.probe_id)
        self.assertEqual(observed, [RecoveryAttemptState.STARTED])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])

    def test_successful_recovery_command_marks_attempt_succeeded(self) -> None:
        result, _, probe = self._recover(ProbeStepState.FORWARD_STARTED, favorited=True)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIs(
                recovery_repository.get_for_probe(probe.probe_id).state,
                RecoveryAttemptState.COMMAND_SUCCEEDED,
            )

    def test_restore_command_exception_leaves_attempt_started(self) -> None:
        adapter = FakeRecoveryAdapter(favorited=True)
        adapter.restore_command_error = AppleMusicWriteError("restore command failed")
        result, _, probe = self._recover(
            ProbeStepState.FORWARD_STARTED, favorited=True, adapter=adapter
        )
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIs(
                recovery_repository.get_for_probe(probe.probe_id).state,
                RecoveryAttemptState.STARTED,
            )

    def test_crash_like_exception_leaves_attempt_started(self) -> None:
        adapter = FakeRecoveryAdapter(favorited=True)
        adapter.restore_command_error = RuntimeError("crash-like ambiguous outcome")
        result, _, probe = self._recover(
            ProbeStepState.FORWARD_STARTED, favorited=True, adapter=adapter
        )
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIs(
                recovery_repository.get_for_probe(probe.probe_id).state,
                RecoveryAttemptState.STARTED,
            )

    def test_started_attempt_restart_never_recommands(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            probe = repository.update_probe(probe, mark_inconclusive(probe))
            probe = repository.update_probe(probe, mark_recovery_restore_started(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            recovery_repository.begin_attempt(probe.probe_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIs(
                recovery_repository.get_for_probe(probe.probe_id).state,
                RecoveryAttemptState.STARTED,
            )

    def test_succeeded_attempt_restart_reconciles_restored(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        adapter = FakeRecoveryAdapter(favorited=False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            attempt = recovery_repository.begin_attempt(probe.probe_id)
            recovery_repository.mark_command_succeeded(attempt.attempt_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)

    def test_succeeded_attempt_restart_source_changed_is_manual(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            attempt = recovery_repository.begin_attempt(probe.probe_id)
            recovery_repository.mark_command_succeeded(attempt.attempt_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)

    def test_failed_attempt_restart_source_safe_is_restored(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        adapter = FakeRecoveryAdapter(favorited=False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            attempt = recovery_repository.begin_attempt(probe.probe_id)
            recovery_repository.mark_command_failed(attempt.attempt_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)

    def test_failed_attempt_restart_source_changed_is_manual(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            attempt = recovery_repository.begin_attempt(probe.probe_id)
            recovery_repository.mark_command_failed(attempt.attempt_id)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertIs(result.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)

    def test_crash_after_restore_started_before_attempt_restart_claims(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            probe = repository.update_probe(probe, mark_inconclusive(probe))
            repository.update_probe(probe, mark_recovery_restore_started(probe))
            # crash before begin_attempt
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            self.assertIs(
                recovery_repository.get_for_probe(probe.probe_id).state,
                RecoveryAttemptState.COMMAND_SUCCEEDED,
            )

    def test_crash_after_command_before_terminal_no_retry(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            probe = repository.update_probe(probe, mark_inconclusive(probe))
            repository.update_probe(probe, mark_recovery_restore_started(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            recovery_repository.begin_attempt(probe.probe_id)
        # The command was sent, then the process crashed before the terminal update.
        adapter.command(TRACK_PID, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            result = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [(TRACK_PID, False)])
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)

    def test_restore_started_competing_callers_at_most_one_command(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        adapter = FakeRecoveryAdapter(favorited=True)
        self._run_competing_recoveries(probe.probe_id, adapter)
        self.assertEqual(len(adapter.commands), 1)

    def test_forward_started_competing_callers_at_most_one_command(self) -> None:
        probe = crashed_probe(ProbeStepState.FORWARD_STARTED)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        adapter = FakeRecoveryAdapter(favorited=True)
        self._run_competing_recoveries(probe.probe_id, adapter)
        self.assertEqual(len(adapter.commands), 1)

    def test_existing_started_attempt_competing_callers_no_command(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as recovery_repository:
            recovery_repository.begin_attempt(probe.probe_id)
        adapter = FakeRecoveryAdapter(favorited=True)
        self._run_competing_recoveries(probe.probe_id, adapter)
        self.assertEqual(len(adapter.commands), 0)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_probe(probe.probe_id).recovery_status,
                RecoveryStatus.NEEDS_MANUAL_CHECK,
            )

    def test_stale_caller_cannot_regress_restored_status(self) -> None:
        probe = crashed_probe(ProbeStepState.RESTORE_STARTED)
        first_adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_inconclusive(probe))
            result = self._orchestrator(repository, first_adapter).recover_probe(probe.probe_id)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        stale = mark_inconclusive(probe)  # RESTORE_STARTED + INCONCLUSIVE + BASELINE_CONFIRMED
        second_adapter = FakeRecoveryAdapter(favorited=True)
        with CapabilityProbeRepository(self.database_path) as repository:
            with patch.object(repository, "get_probe", return_value=stale):
                with self.assertRaises(StaleProbeUpdateError):
                    self._orchestrator(repository, second_adapter).recover_probe(probe.probe_id)
            self.assertIs(
                repository.get_probe(probe.probe_id).recovery_status,
                RecoveryStatus.RESTORED,
            )


if __name__ == "__main__":
    unittest.main()
