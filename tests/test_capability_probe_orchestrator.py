import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from music_agent.apple_music_write import AppleMusicWriteError
from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    ProbeTransitionError,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    mark_forward_started,
)
from music_agent.capability_probe_orchestrator import (
    CapabilityProbeOrchestrator,
    ProbeNotFoundError,
    ProbeReadback,
)
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


def track_key() -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)


class FakeProbeAdapter:
    """Scripted command/readback boundary for the clean probe path.

    ``command`` echoes the requested value into internal state (a real write's desired-state
    effect); ``readback`` echoes that state unless an override is set. The forward command is the
    first call, the restore command the second; the same ordering applies to readbacks.
    """

    def __init__(self, baseline_favorited: bool, baseline_disliked: bool) -> None:
        self.commands: list[tuple[str, bool]] = []
        self.readbacks: list[str] = []
        self.before_command: Callable[[str, bool], None] | None = None
        self.forward_command_error: Exception | None = None
        self.restore_command_error: Exception | None = None
        self.forward_favorited: ObservedValue | None = None
        self.forward_disliked: ObservedValue | None = None
        self.restore_favorited: ObservedValue | None = None
        self.restore_disliked: ObservedValue | None = None
        self._favorited = baseline_favorited
        self._disliked = baseline_disliked

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        phase = "forward" if not self.commands else "restore"
        if self.before_command is not None:
            self.before_command(phase, requested_favorited)
        self.commands.append((target_persistent_id, requested_favorited))
        error = self.forward_command_error if phase == "forward" else self.restore_command_error
        if error is not None:
            raise error
        self._favorited = requested_favorited

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        phase = "forward" if not self.readbacks else "restore"
        self.readbacks.append(target_persistent_id)
        if phase == "forward":
            favorited = (
                self.forward_favorited if self.forward_favorited is not None else value(self._favorited)
            )
            disliked = (
                self.forward_disliked if self.forward_disliked is not None else value(self._disliked)
            )
        else:
            favorited = (
                self.restore_favorited if self.restore_favorited is not None else value(self._favorited)
            )
            disliked = (
                self.restore_disliked if self.restore_disliked is not None else value(self._disliked)
            )
        return ProbeReadback(favorited, disliked)


class AmbiguousForwardThenRestoreAdapter:
    """Forward command applies the write, then raises an ambiguous outcome; restore succeeds."""

    def __init__(self) -> None:
        self.favorited: bool = False
        self.disliked: bool = False
        self.commands: list[tuple[str, bool]] = []
        self.readbacks: list[str] = []
        self._forward_issued = False

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        self.commands.append((target_persistent_id, requested_favorited))
        self.favorited = requested_favorited  # the write applies before the ambiguous outcome
        if not self._forward_issued:
            self._forward_issued = True
            raise RuntimeError("ambiguous forward outcome")

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        self.readbacks.append(target_persistent_id)
        return ProbeReadback(value(self.favorited), value(self.disliked))


class AmbiguousRestoreAdapter:
    """Forward succeeds; restore applies the write back to baseline, then raises ambiguously."""

    def __init__(self) -> None:
        self.favorited: bool = False
        self.disliked: bool = False
        self.commands: list[tuple[str, bool]] = []
        self.readbacks: list[str] = []

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        self.commands.append((target_persistent_id, requested_favorited))
        self.favorited = requested_favorited
        if requested_favorited is False:  # the restore command
            raise RuntimeError("ambiguous restore outcome")

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        self.readbacks.append(target_persistent_id)
        return ProbeReadback(value(self.favorited), value(self.disliked))


class CapabilityProbeOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _orchestrator(self, repository: CapabilityProbeRepository, adapter):
        recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
        self.addCleanup(recovery_repository.close)
        return CapabilityProbeOrchestrator(repository, recovery_repository, adapter)

    def _run(
        self,
        baseline_favorited: bool,
        baseline_disliked: bool,
        adapter: FakeProbeAdapter | None = None,
    ) -> tuple[CapabilityProbe, FakeProbeAdapter, CapabilityProbe]:
        probe = create_probe(TRACK_ID, TRACK_PID, baseline_favorited, baseline_disliked)
        adapter = adapter or FakeProbeAdapter(baseline_favorited, baseline_disliked)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            result = self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        return result, adapter, probe

    def _run_observing_states(
        self, baseline_favorited: bool, baseline_disliked: bool
    ) -> tuple[CapabilityProbe, list[tuple[str, ProbeStepState]]]:
        probe = create_probe(TRACK_ID, TRACK_PID, baseline_favorited, baseline_disliked)
        adapter = FakeProbeAdapter(baseline_favorited, baseline_disliked)
        observed: list[tuple[str, ProbeStepState]] = []
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)

            def before_command(phase: str, requested_favorited: bool) -> None:
                observed.append((phase, repository.get_probe(probe.probe_id).step_state))

            adapter.before_command = before_command
            result = self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        return result, observed

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

    # --- clean path ---------------------------------------------------------

    def test_clean_path_false_baseline_verified(self) -> None:
        result, adapter, _ = self._run(False, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_OBSERVED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])
        self.assertEqual(adapter.readbacks, [TRACK_PID, TRACK_PID])

    def test_clean_path_true_baseline_verified(self) -> None:
        result, adapter, _ = self._run(True, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_OBSERVED)
        self.assertEqual(adapter.commands, [(TRACK_PID, False), (TRACK_PID, True)])

    def test_forward_started_durable_before_forward_command(self) -> None:
        result, observed = self._run_observing_states(False, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertEqual(observed[0], ("forward", ProbeStepState.FORWARD_STARTED))

    def test_restore_started_durable_before_restore_command(self) -> None:
        result, observed = self._run_observing_states(False, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertEqual(observed[1], ("restore", ProbeStepState.RESTORE_STARTED))

    # --- deterministic failures --------------------------------------------

    def test_forward_command_exception_is_inconclusive_not_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.forward_command_error = AppleMusicWriteError("forward command failed")
        result, adapter, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.forward_command_outcome, CommandOutcome.UNKNOWN)
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True)])

    def test_forward_generic_exception_is_inconclusive_not_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.forward_command_error = RuntimeError("crash-like ambiguous outcome")
        result, adapter, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.forward_command_outcome, CommandOutcome.UNKNOWN)
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True)])

    def test_forward_readback_mismatch_is_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.forward_favorited = value(False)  # no-op: readback still shows baseline
        result, _, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.FAILED)
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)

    def test_forward_disliked_side_effect_is_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.forward_disliked = value(True)  # D1 != D0
        result, _, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.FAILED)

    def test_restore_command_exception_is_inconclusive_not_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.restore_command_error = AppleMusicWriteError("restore command failed")
        result, adapter, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.restore_command_outcome, CommandOutcome.UNKNOWN)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])

    def test_restore_generic_exception_is_inconclusive_not_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.restore_command_error = RuntimeError("crash-like ambiguous outcome")
        result, adapter, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.restore_command_outcome, CommandOutcome.UNKNOWN)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])

    def test_forward_ambiguous_exception_with_mutated_source_recovers(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = AmbiguousForwardThenRestoreAdapter()
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            result = self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertIs(result.forward_command_outcome, CommandOutcome.UNKNOWN)
        with CapabilityProbeRepository(self.database_path) as repository:
            recovered = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertIs(recovered.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(recovered.recovery_status, RecoveryStatus.RESTORED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])

    def test_restore_ambiguous_exception_source_safe_reconciles(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = AmbiguousRestoreAdapter()
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            result = self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(result.restore_command_outcome, CommandOutcome.UNKNOWN)
        with CapabilityProbeRepository(self.database_path) as repository:
            recovered = self._orchestrator(repository, adapter).recover_probe(probe.probe_id)
        self.assertIs(recovered.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(recovered.recovery_status, RecoveryStatus.RESTORED)
        # No third command: the source was already back at baseline.
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])

    def test_restore_readback_mismatch_is_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.restore_favorited = value(True)  # not baseline
        result, _, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.FAILED)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_STARTED)

    def test_restore_disliked_side_effect_is_failed(self) -> None:
        adapter = FakeProbeAdapter(False, False)
        adapter.restore_disliked = value(True)  # D2 != D0
        result, _, _ = self._run(False, False, adapter)
        self.assertIs(result.verification_verdict, VerificationVerdict.FAILED)

    # --- stale CAS ordering -------------------------------------------------

    def test_stale_cas_before_forward_command_does_not_invoke_command(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = FakeProbeAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            real_update = repository.update_probe

            def stale_on_forward_start(expected, updated):
                if updated.step_state is ProbeStepState.FORWARD_STARTED:
                    raise StaleProbeUpdateError("concurrent forward start won")
                return real_update(expected, updated)

            with patch.object(repository, "update_probe", side_effect=stale_on_forward_start):
                with self.assertRaises(StaleProbeUpdateError):
                    self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])
        self.assertEqual(adapter.readbacks, [])

    def test_stale_cas_before_restore_command_does_not_invoke_restore(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = FakeProbeAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            real_update = repository.update_probe

            def stale_on_restore_start(expected, updated):
                if updated.step_state is ProbeStepState.RESTORE_STARTED:
                    raise StaleProbeUpdateError("concurrent restore start won")
                return real_update(expected, updated)

            with patch.object(repository, "update_probe", side_effect=stale_on_restore_start):
                with self.assertRaises(StaleProbeUpdateError):
                    self._orchestrator(repository, adapter).run_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [(TRACK_PID, True)])
        self.assertEqual(adapter.readbacks, [TRACK_PID])

    # --- restart persistence ------------------------------------------------

    def test_verified_probe_survives_restart(self) -> None:
        result, _, probe = self._run(False, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        with CapabilityProbeRepository(self.database_path) as repository:
            reloaded = repository.get_probe(probe.probe_id)
        self.assertEqual(reloaded, result)
        self.assertIs(reloaded.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(reloaded.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(reloaded.step_state, ProbeStepState.RESTORE_OBSERVED)

    # --- fail-closed entry --------------------------------------------------

    def test_run_probe_missing_raises(self) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            orchestrator = self._orchestrator(repository, FakeProbeAdapter(False, False))
            with self.assertRaises(ProbeNotFoundError):
                orchestrator.run_probe("prb_ffffffff-ffff-4fff-8fff-ffffffffffff")

    def test_non_baseline_probe_fails_closed_without_command(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = FakeProbeAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            repository.update_probe(probe, mark_forward_started(probe))
            orchestrator = self._orchestrator(repository, adapter)
            with self.assertRaises(ProbeTransitionError):
                orchestrator.run_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    # --- operational isolation ----------------------------------------------

    def test_canonical_model_unchanged(self) -> None:
        seed = self._seed_isolated_store()
        self._run(False, False)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), seed["before_counts"])
            self.assertEqual(repository.load_model(), seed["before_model"])

    def test_external_identity_bindings_unchanged(self) -> None:
        seed = self._seed_isolated_store()
        self._run(False, False)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                repository.lookup_external_identity(seed["binding_key"]), seed["track_id"]
            )

    def test_source_presence_unchanged(self) -> None:
        seed = self._seed_isolated_store()
        self._run(False, False)
        with CanonicalRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, seed["track_id"], "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_pending_intents_and_attempts_unchanged(self) -> None:
        seed = self._seed_isolated_store()
        self._run(False, False)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(seed["intent"].intent_id).state, IntentState.PENDING)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_attempt(seed["attempt"].attempt_id).state, AttemptState.STARTED)

    # --- capability isolation -----------------------------------------------

    def test_capability_matrix_unchanged_after_verified(self) -> None:
        result, _, _ = self._run(False, False)
        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.readback_verified, False)
        self.assertTrue(capability.adapter_implemented)
        self.assertTrue(capability.readback_implemented)

    def test_production_execution_ready_still_false(self) -> None:
        self._run(False, False)
        self.assertFalse(is_execution_ready(resolve_capability(WriteOperation.SET_FAVORITED)))
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))


if __name__ == "__main__":
    unittest.main()
