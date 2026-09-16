import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.apple_music import AppleMusicReadError, AppleMusicSourceAdapter
from music_agent.apple_music_favorited_write import (
    FavoritedWriteAdapter,
    OsascriptFavoritedCommandRunner,
)
from music_agent.apple_music_write import (
    AppleMusicWriteError,
    AppleMusicWriteMappingError,
    RepositoryMembershipBindingResolver,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import ObservationState, ObservedValue, SourcePresence
from music_agent.write_execution import (
    AmbiguousCommandOutcomeError,
    AttemptState,
    DeterministicCommandError,
)
from music_agent.write_execution_repository import WriteExecutionRepository
from music_agent.write_intent import (
    DomainPermission,
    IntentState,
    ReadbackDecision,
    WriteOperation,
    create_scalar_pending_intent,
    evaluate_readback,
    is_execution_ready,
    resolve_capability,
)
from music_agent.write_orchestrator import WriteOrchestrator


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"

FOUND_TRUE = '{"status":"found","fields":{"name":"X","favorited":true,"disliked":false,"rating":0,"played_count":0}}'
FOUND_FALSE = '{"status":"found","fields":{"name":"X","favorited":false,"disliked":false,"rating":0,"played_count":0}}'
FOUND_NO_FAVORITED = '{"status":"found","fields":{"name":"X","played_count":0}}'
CONFIRMED_NOT_FOUND = '{"status":"confirmed_not_found"}'


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = TRACK_PID) -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


def favorited_intent(value: object = True):
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.value(value)
    )


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


class MutatingRaisingCommandRunner:
    """Mutate shared external state to the requested value, then raise a generic exception.

    The exception is deliberately neither ``AmbiguousCommandOutcomeError`` nor
    ``DeterministicCommandError``: it models a dispatched command whose outcome is externally
    ambiguous, which the production adapter propagates unchanged and the orchestrator must fail
    closed to OUTCOME_UNKNOWN rather than a deterministic failure.
    """

    def __init__(self, state: dict[str, bool]) -> None:
        self.state = state
        self.calls: list[tuple[str, bool]] = []

    def run(self, track_persistent_id: str, favorited: bool) -> str:
        self.calls.append((track_persistent_id, favorited))
        self.state["favorited"] = favorited
        raise RuntimeError("command dispatched; outcome ambiguous")


class StatefulReadRunner:
    """Reflect the shared external state's ``favorited`` value as a FOUND read result."""

    def __init__(self, state: dict[str, bool]) -> None:
        self.state = state
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        favorited = "true" if self.state["favorited"] else "false"
        return (
            f'{{"status":"found","fields":{{"name":"X","favorited":{favorited},'
            '"disliked":false,"rating":0,"played_count":0}}'
        )


class FakeBindingResolver:
    def __init__(self, bindings: dict[tuple[EntityType, str], str] | None = None) -> None:
        self.bindings = dict(bindings or {})

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None:
        return self.bindings.get((entity_type, canonical_id))


def bound_adapter(
    command_runner: FakeFavoritedCommandRunner,
    read_runner_output: str | None,
    bindings: dict[tuple[EntityType, str], str] | None = None,
    read_runner: FakeReadRunner | None = None,
) -> FavoritedWriteAdapter:
    resolver = FakeBindingResolver(
        bindings if bindings is not None else {(EntityType.TRACK, TRACK_ID): TRACK_PID}
    )
    runner = read_runner if read_runner is not None else FakeReadRunner(output=read_runner_output)
    return FavoritedWriteAdapter(command_runner, AppleMusicSourceAdapter(runner), resolver)


class FavoritedCommandTest(unittest.TestCase):
    def test_command_sets_favorited_true(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter = bound_adapter(runner, FOUND_TRUE)
        adapter.command(favorited_intent(True))
        self.assertEqual(runner.calls, [(TRACK_PID, True)])

    def test_command_sets_favorited_false(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter = bound_adapter(runner, FOUND_TRUE)
        adapter.command(favorited_intent(False))
        self.assertEqual(runner.calls, [(TRACK_PID, False)])

    def test_command_resolves_through_durable_binding_table(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            database_path = Path(temporary.name) / "canonical.sqlite3"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(load_fixture())
            with CanonicalRepository(database_path) as repository:
                resolver = RepositoryMembershipBindingResolver(repository)
                runner = FakeFavoritedCommandRunner()
                adapter = FavoritedWriteAdapter(
                    runner, AppleMusicSourceAdapter(FakeReadRunner(output=FOUND_TRUE)), resolver
                )
                adapter.command(favorited_intent(True))
                self.assertEqual(runner.calls, [(TRACK_PID, True)])
        finally:
            temporary.cleanup()

    def test_missing_track_binding_fails_closed(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter = bound_adapter(runner, FOUND_TRUE, bindings={})
        with self.assertRaises(DeterministicCommandError):
            adapter.command(favorited_intent(True))
        self.assertEqual(runner.calls, [])

    def test_projection_only_identity_fails_closed(self) -> None:
        # The canonical external_ids projection is never consulted; only the physical binding
        # resolver counts. A resolver reporting no binding fails closed.
        runner = FakeFavoritedCommandRunner()
        adapter = bound_adapter(runner, FOUND_TRUE, bindings={})
        with self.assertRaises(DeterministicCommandError):
            adapter.command(favorited_intent(True))
        self.assertEqual(runner.calls, [])

    def test_binding_drift_fails_closed(self) -> None:
        runner = FakeFavoritedCommandRunner()
        # The intent captures TRACK_PID; the resolver now reports a different persistent ID.
        adapter = bound_adapter(
            runner, FOUND_TRUE, bindings={(EntityType.TRACK, TRACK_ID): "SYNTH-TRACK-999"}
        )
        with self.assertRaises(DeterministicCommandError):
            adapter.command(favorited_intent(True))
        self.assertEqual(runner.calls, [])

    def test_non_favorited_intent_fails_closed(self) -> None:
        rating_intent = create_scalar_pending_intent(
            WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.value(0)
        )
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_TRUE)
        with self.assertRaises(DeterministicCommandError):
            adapter.command(rating_intent)
        with self.assertRaises(AppleMusicWriteMappingError):
            adapter.readback(rating_intent)

    def test_non_bool_requested_value_fails_closed(self) -> None:
        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.value(1)
        )
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_TRUE)
        with self.assertRaises(DeterministicCommandError):
            adapter.command(intent)

    def test_null_requested_value_fails_closed(self) -> None:
        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.null()
        )
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_TRUE)
        with self.assertRaises(DeterministicCommandError):
            adapter.command(intent)

    def test_fake_runner_command_success(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_TRUE)
        self.assertIsNone(adapter.command(favorited_intent(True)))

    def test_fake_runner_command_failure_propagates(self) -> None:
        error = AppleMusicWriteError("Music unavailable")
        adapter = bound_adapter(FakeFavoritedCommandRunner(error=error), FOUND_TRUE)
        with self.assertRaises(AppleMusicWriteError):
            adapter.command(favorited_intent(True))

    def test_production_command_path_never_invokes_real_osascript(self) -> None:
        runner = FakeFavoritedCommandRunner()
        adapter = bound_adapter(runner, FOUND_TRUE)
        with patch("music_agent.apple_music_favorited_write.subprocess.run") as run:
            adapter.command(favorited_intent(True))
        run.assert_not_called()
        self.assertEqual(runner.calls, [(TRACK_PID, True)])

    def test_production_runner_passes_persistent_id_and_true_as_argv(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="set\n", stderr="")
        with patch(
            "music_agent.apple_music_favorited_write.subprocess.run", return_value=completed
        ) as run:
            OsascriptFavoritedCommandRunner(timeout_seconds=3).run(TRACK_PID, True)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["osascript", "-e"])
        self.assertEqual(argv[-2], TRACK_PID)
        self.assertEqual(argv[-1], "true")
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 3)

    def test_production_runner_passes_false_flag(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="set\n", stderr="")
        with patch(
            "music_agent.apple_music_favorited_write.subprocess.run", return_value=completed
        ) as run:
            OsascriptFavoritedCommandRunner().run(TRACK_PID, False)
        self.assertEqual(run.call_args.args[0][-1], "false")

    def test_production_runner_nonzero_exit_is_ambiguous_outcome(self) -> None:
        failed = SimpleNamespace(returncode=1, stdout="", stderr="Music unavailable")
        with patch(
            "music_agent.apple_music_favorited_write.subprocess.run", return_value=failed
        ):
            with self.assertRaises(AmbiguousCommandOutcomeError):
                OsascriptFavoritedCommandRunner().run(TRACK_PID, True)

    def test_production_runner_timeout_is_ambiguous_outcome(self) -> None:
        with patch(
            "music_agent.apple_music_favorited_write.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["osascript"], 3),
        ):
            with self.assertRaises(AmbiguousCommandOutcomeError):
                OsascriptFavoritedCommandRunner().run(TRACK_PID, True)

    def test_production_runner_spawn_failure_is_deterministic_failure(self) -> None:
        with patch(
            "music_agent.apple_music_favorited_write.subprocess.run",
            side_effect=OSError("osascript not found"),
        ):
            with self.assertRaises(DeterministicCommandError):
                OsascriptFavoritedCommandRunner().run(TRACK_PID, True)

    def test_production_runner_rejects_non_bool_before_osascript(self) -> None:
        with patch("music_agent.apple_music_favorited_write.subprocess.run") as run:
            with self.assertRaises(AppleMusicWriteMappingError):
                OsascriptFavoritedCommandRunner().run(TRACK_PID, 1)  # type: ignore[arg-type]
        run.assert_not_called()


class FavoritedReadbackTest(unittest.TestCase):
    def test_requested_true_observed_true_matches(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_TRUE)
        intent = favorited_intent(True)
        observed = adapter.readback(intent)
        self.assertIs(observed.state, ObservationState.VALUE)
        self.assertIs(observed.payload, True)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.MATCHED)

    def test_requested_false_observed_false_matches(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_FALSE)
        intent = favorited_intent(False)
        observed = adapter.readback(intent)
        self.assertIs(observed.state, ObservationState.VALUE)
        self.assertIs(observed.payload, False)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.MATCHED)

    def test_opposite_bool_mismatches(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_FALSE)
        intent = favorited_intent(True)
        observed = adapter.readback(intent)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.MISMATCHED)

    def test_false_is_not_mistaken_for_missing(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_FALSE)
        intent = favorited_intent(False)
        observed = adapter.readback(intent)
        self.assertIs(observed.state, ObservationState.VALUE)
        self.assertIs(observed.payload, False)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.MATCHED)

    def test_missing_favorited_is_unavailable(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), FOUND_NO_FAVORITED)
        intent = favorited_intent(True)
        observed = adapter.readback(intent)
        self.assertIs(observed.state, ObservationState.MISSING)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.UNAVAILABLE)

    def test_confirmed_not_found_is_unavailable(self) -> None:
        adapter = bound_adapter(FakeFavoritedCommandRunner(), CONFIRMED_NOT_FOUND)
        intent = favorited_intent(True)
        observed = adapter.readback(intent)
        self.assertIs(observed.state, ObservationState.MISSING)
        self.assertIs(evaluate_readback(intent.requested_value, observed), ReadbackDecision.UNAVAILABLE)

    def test_lookup_failed_raises(self) -> None:
        read_runner = FakeReadRunner(error=AppleMusicReadError("boom"))
        adapter = bound_adapter(FakeFavoritedCommandRunner(), None, read_runner=read_runner)
        with self.assertRaises(AppleMusicWriteError):
            adapter.readback(favorited_intent(True))

    def test_readback_uses_same_track_identity_as_command(self) -> None:
        command_runner = FakeFavoritedCommandRunner()
        read_runner = FakeReadRunner(output=FOUND_TRUE)
        resolver = FakeBindingResolver({(EntityType.TRACK, TRACK_ID): TRACK_PID})
        adapter = FavoritedWriteAdapter(
            command_runner, AppleMusicSourceAdapter(read_runner), resolver
        )
        intent = favorited_intent(True)
        adapter.command(intent)
        adapter.readback(intent)
        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])
        self.assertEqual(read_runner.calls, [TRACK_PID])


class FavoritedCapabilityTest(unittest.TestCase):
    def test_set_favorited_adapter_and_readback_implemented_not_verified(self) -> None:
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertTrue(capability.adapter_implemented)
        self.assertTrue(capability.readback_implemented)
        self.assertFalse(capability.capability_verified)
        self.assertFalse(capability.readback_verified)

    def test_set_favorited_is_not_execution_ready(self) -> None:
        self.assertFalse(is_execution_ready(resolve_capability(WriteOperation.SET_FAVORITED)))

    def test_add_membership_axes_unchanged(self) -> None:
        capability = resolve_capability(WriteOperation.ADD_PLAYLIST_MEMBERSHIP)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertTrue(capability.capability_verified)
        self.assertTrue(capability.adapter_implemented)
        self.assertFalse(capability.readback_implemented)

    def test_disliked_and_rating_not_enabled(self) -> None:
        for operation in (WriteOperation.SET_DISLIKED, WriteOperation.SET_RATING):
            capability = resolve_capability(operation)
            self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
            self.assertFalse(capability.capability_verified)
            self.assertFalse(capability.adapter_implemented)
            self.assertFalse(capability.readback_implemented)
            self.assertFalse(is_execution_ready(capability))

    def test_create_delete_playlist_remain_unspecified(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.DELETE_PLAYLIST):
            self.assertIs(
                resolve_capability(operation).domain_permission, DomainPermission.UNSPECIFIED
            )


class FavoritedOrchestrationIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.fixture = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_path_confirms_with_production_adapter_and_isolation(self) -> None:
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, TRACK_ID, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(self.fixture, [presence])
            before_counts = repository.counts()
            before_model = repository.load_model()

        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, TRACK_ID, binding_key, ObservedValue.value(True)
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)

        command_runner = FakeFavoritedCommandRunner()
        read_runner = FakeReadRunner(output=FOUND_TRUE)

        with CanonicalRepository(self.database_path) as canonical_repository:
            resolver = RepositoryMembershipBindingResolver(canonical_repository)
            with WriteExecutionRepository(self.database_path) as execution_repository:
                adapter = FavoritedWriteAdapter(
                    command_runner, AppleMusicSourceAdapter(read_runner), resolver
                )
                orchestrator = WriteOrchestrator(execution_repository, adapter, policy=lambda _: True)
                self.assertIs(
                    orchestrator.execute_pending_intent(intent.intent_id).state,
                    IntentState.AWAITING_READBACK,
                )
                self.assertIs(
                    orchestrator.resume_readback(intent.intent_id).state,
                    IntentState.CONFIRMED,
                )

        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])
        self.assertEqual(read_runner.calls, [TRACK_PID])

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)
            self.assertEqual(repository.load_model(), before_model)
            self.assertEqual(repository.lookup_external_identity(binding_key), TRACK_ID)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, TRACK_ID, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_ambiguous_outcome_after_mutation_reconciles_without_replay(self) -> None:
        # The external state starts opposite the requested value so the later CONFIRMED readback
        # can only reflect the command's own mutation, not a pre-existing matching value.
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, TRACK_ID, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(self.fixture, [presence])

        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, TRACK_ID, binding_key, ObservedValue.value(True)
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)

        external_state: dict[str, bool] = {"favorited": False}
        self.assertIs(external_state["favorited"], False)
        command_runner = MutatingRaisingCommandRunner(external_state)
        read_runner = StatefulReadRunner(external_state)

        with CanonicalRepository(self.database_path) as canonical_repository:
            resolver = RepositoryMembershipBindingResolver(canonical_repository)
            with WriteExecutionRepository(self.database_path) as execution_repository:
                adapter = FavoritedWriteAdapter(
                    command_runner, AppleMusicSourceAdapter(read_runner), resolver
                )
                orchestrator = WriteOrchestrator(execution_repository, adapter, policy=lambda _: True)

                self.assertIs(
                    orchestrator.execute_pending_intent(intent.intent_id).state,
                    IntentState.OUTCOME_UNKNOWN,
                )
                self.assertIs(
                    execution_repository.get_latest_attempt(intent.intent_id).state,
                    AttemptState.COMMAND_UNKNOWN,
                )
                self.assertIs(
                    orchestrator.reconcile_unknown_outcome(intent.intent_id).state,
                    IntentState.CONFIRMED,
                )

        self.assertEqual(command_runner.calls, [(TRACK_PID, True)])
        self.assertEqual(read_runner.calls, [TRACK_PID])
        self.assertIs(external_state["favorited"], True)


if __name__ == "__main__":
    unittest.main()
