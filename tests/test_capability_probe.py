import unittest

from music_agent.capability_probe import (
    CapabilityProbe,
    CapabilityProbeValidationError,
    CommandOutcome,
    ProbeStepState,
    ProbeTransitionError,
    RecoveryAction,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    decide_recovery,
    finalize,
    generate_probe_id,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_restore_started,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
    validate_probe_id,
)
from music_agent.identity import IdentityValidationError
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.write_intent import WriteOperation


TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
PLAYLIST_ID = "pl_11111111-1111-4111-8111-111111111111"


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def missing() -> ObservedValue:
    return ObservedValue.missing()


def clean_restore_observed() -> CapabilityProbe:
    probe = create_probe(TRACK_ID, TRACK_PID, False, False)
    probe = mark_forward_started(probe)
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
    probe = mark_restore_started(probe)
    return observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))


class ProbeIdentityTest(unittest.TestCase):
    def test_generate_probe_id_uses_prb_namespace(self) -> None:
        probe_id = generate_probe_id()
        self.assertTrue(probe_id.startswith("prb_"))
        validate_probe_id(probe_id)

    def test_validate_probe_id_rejects_wrong_namespace(self) -> None:
        with self.assertRaises(CapabilityProbeValidationError):
            validate_probe_id("int_11111111-1111-4111-8111-111111111111")

    def test_validate_probe_id_rejects_malformed_suffix(self) -> None:
        with self.assertRaises(CapabilityProbeValidationError):
            validate_probe_id("prb_not-a-uuid")


class ProbeConstructionTest(unittest.TestCase):
    def test_initial_state(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        self.assertEqual(probe.operation, WriteOperation.SET_FAVORITED)
        self.assertEqual(probe.target_canonical_id, TRACK_ID)
        self.assertEqual(probe.target_persistent_id, TRACK_PID)
        self.assertIs(probe.step_state, ProbeStepState.BASELINE_CAPTURED)
        self.assertIs(probe.recovery_status, RecoveryStatus.BASELINE_CONFIRMED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.PENDING)
        self.assertIsNone(probe.forward_command_outcome)
        self.assertIsNone(probe.forward_favorited)
        self.assertIsNone(probe.forward_disliked)
        self.assertIsNone(probe.restore_command_outcome)
        self.assertIsNone(probe.restore_favorited)
        self.assertIsNone(probe.restore_disliked)

    def test_baseline_false_preserved(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        self.assertIs(probe.baseline_favorited, False)
        self.assertIs(probe.baseline_disliked, False)

    def test_baseline_true_preserved(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, True, True)
        self.assertIs(probe.baseline_favorited, True)
        self.assertIs(probe.baseline_disliked, True)

    def test_wrong_operation_fails_closed(self) -> None:
        with self.assertRaises(CapabilityProbeValidationError):
            CapabilityProbe(
                probe_id=generate_probe_id(),
                operation=WriteOperation.SET_RATING,
                target_canonical_id=TRACK_ID,
                target_persistent_id=TRACK_PID,
                baseline_favorited=False,
                baseline_disliked=False,
                step_state=ProbeStepState.BASELINE_CAPTURED,
                recovery_status=RecoveryStatus.BASELINE_CONFIRMED,
                verification_verdict=VerificationVerdict.PENDING,
            )

    def test_wrong_track_identity_fails_closed(self) -> None:
        with self.assertRaises(IdentityValidationError):
            create_probe(PLAYLIST_ID, TRACK_PID, False, False)

    def test_non_bool_baseline_fails_closed(self) -> None:
        with self.assertRaises(CapabilityProbeValidationError):
            create_probe(TRACK_ID, TRACK_PID, 1, False)  # type: ignore[arg-type]

    def test_empty_persistent_id_fails_closed(self) -> None:
        with self.assertRaises(CapabilityProbeValidationError):
            create_probe(TRACK_ID, "", False, False)


class CleanPathTest(unittest.TestCase):
    def test_clean_forward_path(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        self.assertIs(probe.step_state, ProbeStepState.FORWARD_STARTED)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
        self.assertIs(probe.step_state, ProbeStepState.FORWARD_OBSERVED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.PENDING)
        self.assertIs(probe.forward_favorited.payload, True)
        self.assertIs(probe.forward_disliked.payload, False)

    def test_clean_forward_from_true_baseline_uses_false(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, True, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(False), value(False))
        self.assertIs(probe.step_state, ProbeStepState.FORWARD_OBSERVED)
        self.assertIs(probe.forward_favorited.payload, False)
        self.assertIsNot(probe.forward_favorited.state, ObservationState.MISSING)

    def test_clean_restore_path(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
        probe = mark_restore_started(probe)
        self.assertIs(probe.step_state, ProbeStepState.RESTORE_STARTED)
        probe = observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))
        self.assertIs(probe.step_state, ProbeStepState.RESTORE_OBSERVED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.PENDING)

    def test_finalize_verified(self) -> None:
        probe = clean_restore_observed()
        probe = finalize(probe)
        self.assertIs(probe.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(probe.recovery_status, RecoveryStatus.RESTORED)


class SideEffectTest(unittest.TestCase):
    def test_forward_side_effect_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(True))
        self.assertIs(probe.verification_verdict, VerificationVerdict.FAILED)

    def test_restore_side_effect_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
        probe = mark_restore_started(probe)
        probe = observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(True))
        self.assertIs(probe.verification_verdict, VerificationVerdict.FAILED)


class AmbiguousPathTest(unittest.TestCase):
    def test_forward_unknown_inconclusive(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.UNKNOWN, value(True), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_restore_unknown_inconclusive(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
        probe = mark_restore_started(probe)
        probe = observe_restore(probe, CommandOutcome.UNKNOWN, value(False), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_forward_readback_missing_inconclusive(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, missing(), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_command_failed_is_failed(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.FAILED, value(True), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.FAILED)

    def test_noop_write_is_failed(self) -> None:
        # A successful command whose readback still shows the baseline means the write no-op'd.
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(False), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.FAILED)

    def test_crash_marker_inconclusive(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = mark_inconclusive(probe)
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_recovery_restore_started_advances_from_forward_started(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = mark_inconclusive(probe)
        probe = mark_recovery_restore_started(probe)
        self.assertIs(probe.step_state, ProbeStepState.RESTORE_STARTED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_recovery_restore_started_requires_inconclusive(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        with self.assertRaises(ProbeTransitionError):
            mark_recovery_restore_started(probe)

    def test_recovery_restore_started_requires_ambiguous_step(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_inconclusive(probe)
        with self.assertRaises(ProbeTransitionError):
            mark_recovery_restore_started(probe)


class RecoveryTest(unittest.TestCase):
    def _inconclusive_probe(self) -> CapabilityProbe:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        return observe_forward(probe, CommandOutcome.UNKNOWN, value(True), value(False))

    def test_recovery_after_inconclusive_can_become_restored(self) -> None:
        probe = self._inconclusive_probe()
        probe = mark_recovery_status(probe, RecoveryStatus.RESTORED)
        self.assertIs(probe.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)

    def test_recovery_cannot_upgrade_inconclusive_to_verified(self) -> None:
        probe = self._inconclusive_probe()
        probe = mark_recovery_status(probe, RecoveryStatus.RESTORED)
        with self.assertRaises(ProbeTransitionError):
            finalize(probe)

    def test_recovery_cannot_upgrade_failed_to_verified(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.FAILED, value(True), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.FAILED)
        probe = mark_recovery_status(probe, RecoveryStatus.RESTORED)
        with self.assertRaises(ProbeTransitionError):
            finalize(probe)

    def test_decide_recovery_f0_no_restore(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        self.assertIs(decide_recovery(probe, value(False)), RecoveryAction.NO_RESTORE)

    def test_decide_recovery_opposite_restore(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        self.assertIs(decide_recovery(probe, value(True)), RecoveryAction.RESTORE)

    def test_decide_recovery_missing_manual_check(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        self.assertIs(decide_recovery(probe, missing()), RecoveryAction.MANUAL_CHECK)


class TransitionSafetyTest(unittest.TestCase):
    def test_observe_forward_before_started_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with self.assertRaises(ProbeTransitionError):
            observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))

    def test_restore_before_forward_observed_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with self.assertRaises(ProbeTransitionError):
            mark_restore_started(probe)

    def test_observe_restore_before_restore_started_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
        with self.assertRaises(ProbeTransitionError):
            observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))

    def test_finalize_before_restore_observed_fails(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with self.assertRaises(ProbeTransitionError):
            finalize(probe)

    def test_terminal_cannot_reenter_clean_path(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        probe = mark_forward_started(probe)
        probe = observe_forward(probe, CommandOutcome.UNKNOWN, value(True), value(False))
        self.assertIs(probe.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        with self.assertRaises(ProbeTransitionError):
            mark_restore_started(probe)


if __name__ == "__main__":
    unittest.main()
