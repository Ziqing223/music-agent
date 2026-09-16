"""Repository-level tests for the E2-B forward-capture gate (``begin_forward_probe``).

These tests exercise ``CapabilityProbeRepository.begin_forward_probe`` directly — the durable,
``BEGIN IMMEDIATE``-guarded boundary that E2-B capture calls — rather than the pure
``existing_probe_blocks_new_probe`` predicate or the ``CapabilityProbeExecutionService`` wrapper.
They prove that the FULL shared same-target rule is enforced inside one atomic check-then-insert
transaction: every blocking state is refused with no durable row written, every safe state is
captured, a different target never blocks, and two concurrent captures for the same target are
serialized by the ``BEGIN IMMEDIATE`` lock.

Unknown-combination coverage: ``existing_probe_blocks_new_probe`` ends with a fail-closed ``return
True`` for any combination its branches do not recognize. That fallback is unreachable through the
domain/repository APIs, so there is no schema-valid unknown combination to construct here. Both
``VerificationVerdict`` (``pending`` / ``failed`` / ``inconclusive`` / ``verified``) and
``RecoveryStatus`` (``baseline_confirmed`` / ``restored`` / ``needs_manual_check``) are closed
``StrEnum`` types validated in ``CapabilityProbe.__post_init__`` and constrained by the
``capability_probes`` table CHECK constraints, and the predicate branches exhaustively over all
``4 × 3 = 12`` pairs (``PENDING``/``FAILED`` unconditional, ``INCONCLUSIVE``/``VERIFIED``
conditional on recovery status). ``test_all_verdict_recovery_combinations_are_classified`` drives
``begin_forward_probe`` across every one of those 12 pairs to demonstrate that exhaustiveness at the
repository boundary, so the fail-closed fallback is provably defense-in-depth rather than a hidden
allow-list.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from itertools import product
from pathlib import Path

from music_agent.capability_probe import (
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_forward_started_probe,
    create_probe,
    finalize,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import (
    CapabilityProbeRepository,
    ProbeTargetBlockedError,
)
from music_agent.source_observation import ObservedValue

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
OTHER_TRACK_ID = "trk_22222222-2222-4222-8222-222222222222"
OTHER_TRACK_PID = "SYNTH-TRACK-002"


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def new_forward_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    """The incoming probe E2-B wants to capture: born at ``FORWARD_STARTED``."""
    return create_forward_started_probe(track_id, pid, False, False)


def pending_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    return create_probe(track_id, pid, False, False)


def failed_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = mark_forward_started(create_probe(track_id, pid, False, False))
    return observe_forward(probe, CommandOutcome.FAILED, value(True), value(False))


def inconclusive(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    return mark_inconclusive(create_probe(track_id, pid, False, False))


def verified_restored(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = create_probe(track_id, pid, False, False)
    probe = mark_forward_started(probe)
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
    probe = mark_restore_started(probe)
    probe = observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))
    return finalize(probe)


def combination_probe(verdict: VerificationVerdict, status: RecoveryStatus):
    """Build a schema-valid probe in every reachable ``(verdict, recovery_status)`` pair."""
    if verdict is VerificationVerdict.PENDING:
        probe = pending_probe()
    elif verdict is VerificationVerdict.FAILED:
        probe = failed_probe()
    elif verdict is VerificationVerdict.INCONCLUSIVE:
        probe = inconclusive()
    else:  # VERIFIED
        probe = verified_restored()
    if probe.recovery_status is not status:
        probe = mark_recovery_status(probe, status)
    return probe


class CapabilityProbeForwardGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _seed(self, *probes) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in probes:
                repository.save_probe(probe)

    def _seed_started_attempt(self, probe) -> None:
        with CapabilityProbeRecoveryAttemptRepository(self.database_path) as attempts:
            attempts.begin_attempt(probe.probe_id)

    def _capture(self, *, expect_blocked: bool, track_id: str = TRACK_ID, pid: str = TRACK_PID):
        with CapabilityProbeRepository(self.database_path) as repository:
            if expect_blocked:
                with self.assertRaises(ProbeTargetBlockedError):
                    repository.begin_forward_probe(new_forward_probe(track_id, pid))
            else:
                captured = repository.begin_forward_probe(new_forward_probe(track_id, pid))
                self.assertIs(captured.step_state, ProbeStepState.FORWARD_STARTED)
                self.assertEqual(repository.get_probe(captured.probe_id), captured)
            return list(repository.list_probes())

    def _assert_only_seed_remains(self, probe_id: str) -> None:
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(
                [loaded.probe_id for loaded in repository.list_probes()], [probe_id]
            )

    # --- blocking states -----------------------------------------------------

    def test_started_recovery_attempt_blocks(self) -> None:
        safe = mark_recovery_status(inconclusive(), RecoveryStatus.RESTORED)
        self._seed(safe)
        self._seed_started_attempt(safe)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(safe.probe_id)

    def test_pending_blocks(self) -> None:
        blocker = pending_probe()
        self._seed(blocker)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(blocker.probe_id)

    def test_failed_blocks(self) -> None:
        blocker = failed_probe()
        self._seed(blocker)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(blocker.probe_id)

    def test_inconclusive_needs_manual_check_blocks(self) -> None:
        blocker = mark_recovery_status(inconclusive(), RecoveryStatus.NEEDS_MANUAL_CHECK)
        self._seed(blocker)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(blocker.probe_id)

    def test_verified_baseline_confirmed_blocks(self) -> None:
        blocker = mark_recovery_status(verified_restored(), RecoveryStatus.BASELINE_CONFIRMED)
        self._seed(blocker)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(blocker.probe_id)

    def test_verified_needs_manual_check_blocks(self) -> None:
        blocker = mark_recovery_status(verified_restored(), RecoveryStatus.NEEDS_MANUAL_CHECK)
        self._seed(blocker)
        self._capture(expect_blocked=True)
        self._assert_only_seed_remains(blocker.probe_id)

    # --- safe states ---------------------------------------------------------

    def test_inconclusive_restored_allows(self) -> None:
        safe = mark_recovery_status(inconclusive(), RecoveryStatus.RESTORED)
        self._seed(safe)
        self._capture(expect_blocked=False)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_probes()), 2)

    def test_inconclusive_baseline_confirmed_allows(self) -> None:
        safe = inconclusive()
        self._seed(safe)
        self._capture(expect_blocked=False)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_probes()), 2)

    def test_verified_restored_allows(self) -> None:
        safe = verified_restored()
        self._seed(safe)
        self._capture(expect_blocked=False)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_probes()), 2)

    # --- different target ----------------------------------------------------

    def test_different_target_unsafe_probe_does_not_block(self) -> None:
        self._seed(pending_probe(OTHER_TRACK_ID, OTHER_TRACK_PID))
        self._capture(expect_blocked=False)

    def test_different_target_started_attempt_does_not_block(self) -> None:
        other = mark_recovery_status(
            inconclusive(OTHER_TRACK_ID, OTHER_TRACK_PID), RecoveryStatus.RESTORED
        )
        self._seed(other)
        self._seed_started_attempt(other)
        self._capture(expect_blocked=False)

    # --- full (verdict, recovery_status) matrix ------------------------------

    def test_all_verdict_recovery_combinations_are_classified(self) -> None:
        allow = {
            (VerificationVerdict.INCONCLUSIVE, RecoveryStatus.BASELINE_CONFIRMED),
            (VerificationVerdict.INCONCLUSIVE, RecoveryStatus.RESTORED),
            (VerificationVerdict.VERIFIED, RecoveryStatus.RESTORED),
        }
        for verdict, status in product(VerificationVerdict, RecoveryStatus):
            with self.subTest(verdict=verdict.value, status=status.value):
                with tempfile.TemporaryDirectory() as directory:
                    database_path = Path(directory) / "canonical.sqlite3"
                    probe = combination_probe(verdict, status)
                    self.assertEqual(
                        (probe.verification_verdict, probe.recovery_status), (verdict, status)
                    )
                    with CapabilityProbeRepository(database_path) as repository:
                        repository.save_probe(probe)
                    with CapabilityProbeRepository(database_path) as repository:
                        if (verdict, status) in allow:
                            captured = repository.begin_forward_probe(new_forward_probe())
                            self.assertIs(captured.step_state, ProbeStepState.FORWARD_STARTED)
                            self.assertEqual(len(repository.list_probes()), 2)
                        else:
                            with self.assertRaises(ProbeTargetBlockedError):
                                repository.begin_forward_probe(new_forward_probe())
                            self.assertEqual(len(repository.list_probes()), 1)

    # --- BEGIN IMMEDIATE serialization ---------------------------------------

    def test_concurrent_same_target_capture_is_serialized(self) -> None:
        CapabilityProbeRepository(self.database_path).close()
        captured: list[str] = []
        blocked: list[str] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            repository = CapabilityProbeRepository(self.database_path)
            try:
                barrier.wait()
                repository.begin_forward_probe(new_forward_probe())
                captured.append("ok")
            except ProbeTargetBlockedError:
                blocked.append("blocked")
            finally:
                repository.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(blocked), 1)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_probes()), 1)


if __name__ == "__main__":
    unittest.main()
