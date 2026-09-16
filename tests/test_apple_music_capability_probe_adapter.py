import json
import tempfile
import unittest
from pathlib import Path

from music_agent.apple_music import AppleMusicReadError, AppleMusicSourceAdapter
from music_agent.apple_music_capability_probe_adapter import AppleMusicCapabilityProbeAdapter
from music_agent.apple_music_write import AppleMusicWriteError, AppleMusicWriteMappingError
from music_agent.capability_probe import (
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
)
from music_agent.capability_probe_orchestrator import CapabilityProbeOrchestrator
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
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

FOUND_TRUE_FALSE = (
    '{"status":"found","fields":{"name":"X","favorited":true,"disliked":false,'
    '"rating":0,"played_count":0}}'
)
FOUND_FALSE_TRUE = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":true,'
    '"rating":0,"played_count":0}}'
)
FOUND_FALSE_FALSE = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":false,'
    '"rating":0,"played_count":0}}'
)
FOUND_NO_FAVORITED = '{"status":"found","fields":{"name":"X","disliked":false,"played_count":0}}'
FOUND_NO_DISLIKED = '{"status":"found","fields":{"name":"X","favorited":false,"played_count":0}}'
CONFIRMED_NOT_FOUND = '{"status":"confirmed_not_found"}'


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeFavoritedCommandRunner:
    def __init__(self, output: str = "set", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, bool]] = []

    def run(self, track_persistent_id: str, favorited: bool) -> str:
        self.calls.append((track_persistent_id, favorited))
        if self.error is not None:
            raise self.error
        return self.output


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


class StatefulFavoritedCommandRunner:
    """Command runner that echoes the requested value into shared source state."""

    def __init__(self, state: dict[str, bool]) -> None:
        self._state = state
        self.calls: list[tuple[str, bool]] = []

    def run(self, track_persistent_id: str, favorited: bool) -> str:
        self.calls.append((track_persistent_id, favorited))
        self._state["favorited"] = favorited
        return "set"


class StatefulFavoritedReadRunner:
    """Read runner that returns the shared source state as a found Track observation."""

    def __init__(self, state: dict[str, bool]) -> None:
        self._state = state
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        return json.dumps(
            {
                "status": "found",
                "fields": {
                    "name": "X",
                    "favorited": self._state["favorited"],
                    "disliked": self._state["disliked"],
                    "rating": 0,
                    "played_count": 0,
                },
            }
        )


def adapter(
    command_runner: FakeFavoritedCommandRunner,
    read_runner_output: str | None = None,
    read_runner: FakeReadRunner | None = None,
) -> AppleMusicCapabilityProbeAdapter:
    runner = read_runner if read_runner is not None else FakeReadRunner(output=read_runner_output)
    return AppleMusicCapabilityProbeAdapter(command_runner, AppleMusicSourceAdapter(runner))


class ProbeAdapterCommandTest(unittest.TestCase):
    def test_command_true_passes_exact_persistent_id_and_bool(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter(runner, FOUND_TRUE_FALSE).command(TRACK_PID, True)
        self.assertEqual(runner.calls, [(TRACK_PID, True)])

    def test_command_false_is_a_real_value_not_missing(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter(runner, FOUND_FALSE_TRUE).command(TRACK_PID, False)
        self.assertEqual(runner.calls, [(TRACK_PID, False)])

    def test_command_success_returns_none(self) -> None:
        probe_adapter = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE)
        self.assertIsNone(probe_adapter.command(TRACK_PID, True))

    def test_command_apple_music_write_error_propagates(self) -> None:
        error = AppleMusicWriteError("Music unavailable")
        probe_adapter = adapter(FakeFavoritedCommandRunner(error=error), FOUND_TRUE_FALSE)
        with self.assertRaises(AppleMusicWriteError):
            probe_adapter.command(TRACK_PID, True)

    def test_command_generic_exception_propagates_not_fabricated_success(self) -> None:
        error = RuntimeError("unexpected crash")
        probe_adapter = adapter(FakeFavoritedCommandRunner(error=error), FOUND_TRUE_FALSE)
        # Any exception is left ambiguous for the orchestrator to map to UNKNOWN; the adapter
        # neither swallows it nor invents a success or a deterministic failure.
        with self.assertRaises(RuntimeError):
            probe_adapter.command(TRACK_PID, True)

    def test_command_rejects_non_bool_requested_favorited(self) -> None:
        runner = FakeFavoritedCommandRunner()
        probe_adapter = adapter(runner, FOUND_TRUE_FALSE)
        with self.assertRaises(AppleMusicWriteMappingError):
            probe_adapter.command(TRACK_PID, 1)  # type: ignore[arg-type]
        self.assertEqual(runner.calls, [])

    def test_command_rejects_empty_persistent_id(self) -> None:
        runner = FakeFavoritedCommandRunner()
        probe_adapter = adapter(runner, FOUND_TRUE_FALSE)
        with self.assertRaises(AppleMusicWriteMappingError):
            probe_adapter.command("", True)
        self.assertEqual(runner.calls, [])


class ProbeAdapterReadbackTest(unittest.TestCase):
    def test_readback_found_favorited_true_disliked_false(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.VALUE)
        self.assertIs(result.favorited.payload, True)
        self.assertIs(result.disliked.state, ObservationState.VALUE)
        self.assertIs(result.disliked.payload, False)

    def test_readback_found_favorited_false_disliked_true(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), FOUND_FALSE_TRUE).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.VALUE)
        self.assertIs(result.favorited.payload, False)
        self.assertIs(result.disliked.state, ObservationState.VALUE)
        self.assertIs(result.disliked.payload, True)

    def test_readback_false_round_trips_exactly(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), FOUND_FALSE_TRUE).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.VALUE)
        self.assertIs(result.favorited.payload, False)
        self.assertIsNot(result.favorited.state, ObservationState.MISSING)

    def test_readback_favorited_missing(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), FOUND_NO_FAVORITED).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.MISSING)
        self.assertIs(result.disliked.state, ObservationState.VALUE)
        self.assertIs(result.disliked.payload, False)

    def test_readback_disliked_missing(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), FOUND_NO_DISLIKED).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.VALUE)
        self.assertIs(result.favorited.payload, False)
        self.assertIs(result.disliked.state, ObservationState.MISSING)

    def test_readback_confirmed_not_found_is_unavailable(self) -> None:
        result = adapter(FakeFavoritedCommandRunner(), CONFIRMED_NOT_FOUND).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.MISSING)
        self.assertIs(result.disliked.state, ObservationState.MISSING)

    def test_readback_lookup_failed_is_unavailable(self) -> None:
        read_runner = FakeReadRunner(error=AppleMusicReadError("boom"))
        result = adapter(
            FakeFavoritedCommandRunner(), read_runner=read_runner
        ).readback(TRACK_PID)
        self.assertIs(result.favorited.state, ObservationState.MISSING)
        self.assertIs(result.disliked.state, ObservationState.MISSING)

    def test_command_and_readback_use_the_same_persistent_id(self) -> None:
        command_runner = FakeFavoritedCommandRunner()
        read_runner = FakeReadRunner(output=FOUND_TRUE_FALSE)
        probe_adapter = AppleMusicCapabilityProbeAdapter(
            command_runner, AppleMusicSourceAdapter(read_runner)
        )
        probe_adapter.command(TRACK_PID, True)
        probe_adapter.readback(TRACK_PID)
        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])
        self.assertEqual(read_runner.calls, [TRACK_PID])

    def test_no_name_or_fuzzy_lookup(self) -> None:
        # The payload carries a name, but identity is keyed solely on the persistent ID: the
        # runners each receive exactly the passed persistent ID, never a name or a fuzzy match.
        command_runner = FakeFavoritedCommandRunner()
        read_runner = FakeReadRunner(output=FOUND_TRUE_FALSE)
        probe_adapter = AppleMusicCapabilityProbeAdapter(
            command_runner, AppleMusicSourceAdapter(read_runner)
        )
        probe_adapter.command(TRACK_PID, True)
        result = probe_adapter.readback(TRACK_PID)
        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])
        self.assertEqual(read_runner.calls, [TRACK_PID])
        self.assertIs(result.favorited.state, ObservationState.VALUE)


class ProbeAdapterIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

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
            WriteOperation.SET_FAVORITED, track_id, binding_key, ObservedValue.value(True)
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

    def test_adapter_holds_no_intent_or_attempt_dependencies(self) -> None:
        probe_adapter = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE)
        # Built only from a command runner and a read adapter; there is no PendingIntent or
        # ExecutionAttempt repository, so the adapter cannot create or mutate those objects.
        self.assertEqual(set(vars(probe_adapter)), {"_command_runner", "_read_adapter"})

    def test_canonical_bindings_presence_and_intents_unchanged(self) -> None:
        seed = self._seed_isolated_store()
        probe_adapter = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE)
        probe_adapter.command(TRACK_PID, True)
        probe_adapter.readback(TRACK_PID)

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


class ProbeAdapterCapabilityTest(unittest.TestCase):
    def test_capability_matrix_unchanged_after_adapter_use(self) -> None:
        probe_adapter = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE)
        probe_adapter.command(TRACK_PID, True)
        probe_adapter.readback(TRACK_PID)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.readback_verified, False)
        self.assertTrue(capability.adapter_implemented)
        self.assertTrue(capability.readback_implemented)

    def test_execution_ready_still_false(self) -> None:
        probe_adapter = adapter(FakeFavoritedCommandRunner(), FOUND_TRUE_FALSE)
        probe_adapter.command(TRACK_PID, True)
        probe_adapter.readback(TRACK_PID)
        self.assertFalse(is_execution_ready(resolve_capability(WriteOperation.SET_FAVORITED)))


class ProbeAdapterIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_clean_probe_with_production_adapter_and_fake_source(self) -> None:
        state = {"favorited": False, "disliked": False}
        command_runner = StatefulFavoritedCommandRunner(state)
        read_runner = StatefulFavoritedReadRunner(state)
        probe_adapter = AppleMusicCapabilityProbeAdapter(
            command_runner, AppleMusicSourceAdapter(read_runner)
        )

        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
            self.addCleanup(recovery_repository.close)
            orchestrator = CapabilityProbeOrchestrator(
                repository, recovery_repository, probe_adapter
            )
            result = orchestrator.run_probe(probe.probe_id)

        self.assertIs(result.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(result.recovery_status, RecoveryStatus.RESTORED)
        self.assertIs(result.step_state, ProbeStepState.RESTORE_OBSERVED)
        self.assertEqual(command_runner.calls, [(TRACK_PID, True), (TRACK_PID, False)])
        self.assertEqual(read_runner.calls, [TRACK_PID, TRACK_PID])

    def test_ambiguous_command_outcome_is_not_failed(self) -> None:
        command_runner = FakeFavoritedCommandRunner(error=RuntimeError("ambiguous outcome"))
        read_runner = FakeReadRunner(output=FOUND_FALSE_FALSE)
        probe_adapter = AppleMusicCapabilityProbeAdapter(
            command_runner, AppleMusicSourceAdapter(read_runner)
        )

        probe = create_probe(TRACK_ID, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
            recovery_repository = CapabilityProbeRecoveryAttemptRepository(self.database_path)
            self.addCleanup(recovery_repository.close)
            orchestrator = CapabilityProbeOrchestrator(
                repository, recovery_repository, probe_adapter
            )
            result = orchestrator.run_probe(probe.probe_id)

        self.assertIs(result.verification_verdict, VerificationVerdict.INCONCLUSIVE)
        self.assertIsNot(result.verification_verdict, VerificationVerdict.FAILED)
        self.assertIs(result.forward_command_outcome, CommandOutcome.UNKNOWN)
        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])


if __name__ == "__main__":
    unittest.main()
