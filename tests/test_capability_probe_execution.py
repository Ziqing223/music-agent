import json
import tempfile
import unittest
from pathlib import Path

from music_agent.apple_music import AppleMusicReadError, AppleMusicSourceAdapter
from music_agent.capability_probe import (
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    finalize,
    mark_forward_started,
    mark_inconclusive,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_execution import (
    CapabilityProbeExecutionError,
    CapabilityProbeExecutionService,
    CapabilityProbeStartFailure,
)
from music_agent.capability_probe_orchestrator import (
    CapabilityProbeOrchestrator,
    ProbeNotStartedError,
    ProbeReadback,
)
from music_agent.capability_probe_preflight import (
    CapabilityProbePreflightResult,
    CapabilityProbePreflightService,
    PreflightRejectionReason,
)
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.identity import EntityType
from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import ObservedValue, SourcePresence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
SCOPE_KEY = "library_tracks"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
OTHER_TRACK_ID = "trk_22222222-2222-4222-8222-222222222222"

FOUND_FALSE_FALSE = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":false,'
    '"rating":0,"played_count":0}}'
)
FOUND_TRUE_FALSE = (
    '{"status":"found","fields":{"name":"X","favorited":true,"disliked":false}}'
)
FOUND_FAVORITED_NULL = (
    '{"status":"found","fields":{"name":"X","favorited":null,"disliked":false}}'
)
FOUND_DISLIKED_NULL = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":null}}'
)
FOUND_FAVORITED_NON_BOOL = (
    '{"status":"found","fields":{"name":"X","favorited":123,"disliked":false}}'
)
CONFIRMED_NOT_FOUND = '{"status":"confirmed_not_found"}'


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def eligible(
    favorited: bool = False,
    disliked: bool = False,
    pid: str = TRACK_PID,
    track_id: str = TRACK_ID,
) -> CapabilityProbePreflightResult:
    return CapabilityProbePreflightResult(
        eligible=True,
        canonical_track_id=track_id,
        target_persistent_id=pid,
        baseline_favorited=favorited,
        baseline_disliked=disliked,
        rejection_reason=None,
    )


def pending_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    return create_probe(track_id, pid, False, False)


def verified_restored(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = create_probe(track_id, pid, False, False)
    probe = mark_forward_started(probe)
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(True), value(False))
    probe = mark_restore_started(probe)
    probe = observe_restore(probe, CommandOutcome.SUCCESS, value(False), value(False))
    return finalize(probe)


class FakeReadRunner:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        if self.error is not None:
            raise self.error
        if self.output is None:
            raise AppleMusicReadError("no read output configured")
        return self.output


class FakeCommandAdapter:
    """Scripted command/readback boundary: ``command`` echoes requested favorited into state."""

    def __init__(self, favorited: bool = False, disliked: bool = False) -> None:
        self.commands: list[tuple[str, bool]] = []
        self._favorited = favorited
        self._disliked = disliked

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        self.commands.append((target_persistent_id, requested_favorited))
        self._favorited = requested_favorited

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        return ProbeReadback(value(self._favorited), value(self._disliked))


class CapabilityProbeExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _orchestrator(self, repository: CapabilityProbeRepository, adapter):
        recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
        self.addCleanup(recovery_repository.close)
        return CapabilityProbeOrchestrator(repository, recovery_repository, adapter)

    def _start(
        self,
        result: CapabilityProbePreflightResult,
        *,
        read_output: str | None = None,
        read_error: Exception | None = None,
        seed_probes: tuple = (),
    ):
        source_adapter = AppleMusicSourceAdapter(FakeReadRunner(read_output, read_error))
        with CapabilityProbeRepository(self.database_path) as repository:
            for probe in seed_probes:
                repository.save_probe(probe)
            service = CapabilityProbeExecutionService(source_adapter, repository)
            return service.start(result)

    def _seed_canonical(self) -> None:
        fixture = load_fixture()
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, TRACK_ID, SCOPE_KEY, SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])

    # --- capture ------------------------------------------------------------

    def test_start_captures_forward_started_with_fresh_baseline(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_FALSE_FALSE)
        self.assertTrue(result.started)
        self.assertIsNone(result.failure)
        probe = result.probe
        self.assertIs(probe.step_state, ProbeStepState.FORWARD_STARTED)
        self.assertIs(probe.verification_verdict, VerificationVerdict.PENDING)
        self.assertIs(probe.recovery_status, RecoveryStatus.BASELINE_CONFIRMED)
        self.assertIs(probe.baseline_favorited, False)
        self.assertIs(probe.baseline_disliked, False)
        self.assertEqual(probe.target_persistent_id, TRACK_PID)
        self.assertEqual(probe.target_canonical_id, TRACK_ID)
        self.assertIsNone(probe.forward_command_outcome)
        self.assertIsNone(probe.forward_favorited)
        self.assertIsNone(probe.forward_disliked)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    def test_start_persists_true_baseline(self) -> None:
        result = self._start(eligible(True, False), read_output=FOUND_TRUE_FALSE)
        self.assertTrue(result.started)
        self.assertIs(result.probe.baseline_favorited, True)
        self.assertIs(result.probe.baseline_disliked, False)

    def test_start_performs_single_fresh_read_of_persistent_id(self) -> None:
        runner = FakeReadRunner(output=FOUND_FALSE_FALSE)
        with CapabilityProbeRepository(self.database_path) as repository:
            service = CapabilityProbeExecutionService(
                AppleMusicSourceAdapter(runner), repository
            )
            service.start(eligible(False, False))
        self.assertEqual(runner.calls, [TRACK_PID])

    # --- fresh-read divergence / invalidity ---------------------------------

    def test_start_aborts_when_fresh_read_differs_from_preflight(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_TRUE_FALSE)
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.BASELINE_CHANGED)
        self.assertIsNone(result.probe)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.list_probes(), ())

    def test_start_aborts_when_fresh_disliked_differs(self) -> None:
        result = self._start(
            eligible(False, False),
            read_output='{"status":"found","fields":{"name":"X","favorited":false,"disliked":true}}',
        )
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.BASELINE_CHANGED)

    def test_start_aborts_not_found(self) -> None:
        result = self._start(eligible(False, False), read_output=CONFIRMED_NOT_FOUND)
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.FRESH_READ_NOT_FOUND)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.list_probes(), ())

    def test_start_aborts_lookup_failed(self) -> None:
        result = self._start(eligible(False, False), read_error=AppleMusicReadError("boom"))
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.FRESH_READ_LOOKUP_FAILED)

    def test_start_aborts_favorited_missing(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_FAVORITED_NULL)
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.FRESH_READ_FAVORITED_MISSING)

    def test_start_aborts_disliked_missing(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_DISLIKED_NULL)
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.FRESH_READ_DISLIKED_MISSING)

    def test_start_aborts_favorited_not_bool(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_FAVORITED_NON_BOOL)
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.FRESH_READ_FAVORITED_NOT_BOOL)

    def test_start_rejects_ineligible_result(self) -> None:
        ineligible = CapabilityProbePreflightResult(
            eligible=False,
            canonical_track_id=TRACK_ID,
            target_persistent_id=None,
            baseline_favorited=None,
            baseline_disliked=None,
            rejection_reason=PreflightRejectionReason.CANONICAL_ENTITY_NOT_FOUND,
        )
        with CapabilityProbeRepository(self.database_path) as repository:
            service = CapabilityProbeExecutionService(
                AppleMusicSourceAdapter(FakeReadRunner(output=FOUND_FALSE_FALSE)), repository
            )
            with self.assertRaises(CapabilityProbeExecutionError):
                service.start(ineligible)

    # --- same-target gate (shared predicate) --------------------------------

    def test_start_blocks_pending_same_target(self) -> None:
        blocker = pending_probe()
        result = self._start(
            eligible(False, False), read_output=FOUND_FALSE_FALSE, seed_probes=(blocker,)
        )
        self.assertFalse(result.started)
        self.assertIs(result.failure, CapabilityProbeStartFailure.TARGET_BLOCKED)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(
                [probe.probe_id for probe in repository.list_probes()], [blocker.probe_id]
            )

    def test_start_allows_verified_restored_same_target(self) -> None:
        safe = verified_restored()
        result = self._start(
            eligible(False, False), read_output=FOUND_FALSE_FALSE, seed_probes=(safe,)
        )
        self.assertTrue(result.started)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_probes()), 2)

    def test_start_allows_different_target(self) -> None:
        other = pending_probe(track_id=OTHER_TRACK_ID, pid="SYNTH-TRACK-002")
        result = self._start(
            eligible(False, False), read_output=FOUND_FALSE_FALSE, seed_probes=(other,)
        )
        self.assertTrue(result.started)

    # --- lost probe_id: discovery + recovery (contract #13) ------------------

    def test_lost_probe_id_discovered_and_recovered_without_forward_reissue(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_FALSE_FALSE)
        self.assertTrue(result.started)
        self.assertIs(result.probe.step_state, ProbeStepState.FORWARD_STARTED)

        # The caller "loses" result.probe.probe_id; discovery must find the orphan.
        with CapabilityProbeRepository(self.database_path) as repository:
            recoverable = repository.list_recoverable_probes()
        self.assertEqual([probe.probe_id for probe in recoverable], [result.probe.probe_id])

        # Recovery resolves it from source; the forward command is never issued.
        adapter = FakeCommandAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            orchestrator = self._orchestrator(repository, adapter)
            recovered = orchestrator.recover_probe(recoverable[0].probe_id)
        self.assertIs(recovered.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIs(recovered.recovery_status, RecoveryStatus.BASELINE_CONFIRMED)
        self.assertEqual(adapter.commands, [])

    # --- forward dispatch after capture (contract #8) ------------------------

    def test_run_started_probe_dispatches_forward_after_capture(self) -> None:
        result = self._start(eligible(False, False), read_output=FOUND_FALSE_FALSE)
        adapter = FakeCommandAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            orchestrator = self._orchestrator(repository, adapter)
            final = orchestrator.run_started_probe(result.probe.probe_id)
        self.assertIs(final.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(final.recovery_status, RecoveryStatus.RESTORED)
        self.assertEqual(adapter.commands, [(TRACK_PID, True), (TRACK_PID, False)])

    def test_run_started_probe_rejects_baseline_captured(self) -> None:
        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        adapter = FakeCommandAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            orchestrator = self._orchestrator(repository, adapter)
            with self.assertRaises(ProbeNotStartedError):
                orchestrator.run_started_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    def test_run_started_probe_rejects_inconclusive_forward_started(self) -> None:
        probe = mark_inconclusive(mark_forward_started(create_probe(TRACK_ID, TRACK_PID, False, False)))
        adapter = FakeCommandAdapter(False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            orchestrator = self._orchestrator(repository, adapter)
            with self.assertRaises(ProbeNotStartedError):
                orchestrator.run_started_probe(probe.probe_id)
        self.assertEqual(adapter.commands, [])

    # --- end-to-end E2-A -> E2-B --------------------------------------------

    def test_preflight_then_start_captures_bound_persistent_id(self) -> None:
        self._seed_canonical()
        runner = FakeReadRunner(output=FOUND_FALSE_FALSE)
        source_adapter = AppleMusicSourceAdapter(runner)
        with CapabilityProbeRepository(self.database_path) as probes, CapabilityProbeRecoveryAttemptRepository(
            self.database_path
        ) as attempts, CanonicalRepository(self.database_path) as canonical:
            preflight = CapabilityProbePreflightService(
                canonical, probes, attempts, source_adapter, SCOPE_KEY
            )
            eligible_result = preflight.check_target(TRACK_ID)
            self.assertTrue(eligible_result.eligible)
            service = CapabilityProbeExecutionService(source_adapter, probes)
            result = service.start(eligible_result)
        self.assertTrue(result.started)
        self.assertEqual(result.probe.target_persistent_id, TRACK_PID)
        self.assertEqual(result.probe.target_canonical_id, TRACK_ID)
        self.assertIs(result.probe.baseline_favorited, False)
        self.assertIs(result.probe.baseline_disliked, False)
        self.assertIs(result.probe.step_state, ProbeStepState.FORWARD_STARTED)
        # Preflight read + fresh execution read, both keyed solely on the persistent ID.
        self.assertEqual(runner.calls, [TRACK_PID, TRACK_PID])

    # --- schema isolation ---------------------------------------------------

    def test_schema_remains_v8_after_capture(self) -> None:
        self._start(eligible(False, False), read_output=FOUND_FALSE_FALSE)
        self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)


if __name__ == "__main__":
    unittest.main()
