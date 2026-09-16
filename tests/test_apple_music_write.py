import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.apple_music_write import (
    AppleMusicWriteAdapter,
    AppleMusicWriteError,
    AppleMusicWriteMappingError,
    MembershipBindingMissingError,
    OsascriptAddMembershipRunner,
    OsascriptPlaylistMembersRunner,
    RepositoryMembershipBindingResolver,
    evaluate_membership_readback,
)
from music_agent.identity import EntityType
from music_agent.repository import CanonicalRepository
from music_agent.write_intent import (
    DomainPermission,
    ReadbackDecision,
    WriteOperation,
    is_execution_ready,
    resolve_capability,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
PLAYLIST_ID = "pl_55555555-5555-4555-8555-555555555555"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
PLAYLIST_PID = "SYNTH-PLAYLIST-1"
TRACK_PID = "SYNTH-TRACK-001"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeCommandRunner:
    def __init__(self, output: str = "added", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def run(self, playlist_persistent_id: str, track_persistent_id: str) -> str:
        self.calls.append((playlist_persistent_id, track_persistent_id))
        if self.error is not None:
            raise self.error
        return self.output


class FakeMembersRunner:
    def __init__(self, output: str = "[]", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, playlist_persistent_id: str) -> str:
        self.calls.append(playlist_persistent_id)
        if self.error is not None:
            raise self.error
        return self.output


class FakeBindingResolver:
    def __init__(self, bindings: dict[tuple[EntityType, str], str] | None = None) -> None:
        self.bindings = bindings or {}

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None:
        return self.bindings.get((entity_type, canonical_id))


def bound_adapter(command_runner, members_runner, bindings=None) -> AppleMusicWriteAdapter:
    resolver = FakeBindingResolver(bindings if bindings is not None else {
        (EntityType.PLAYLIST, PLAYLIST_ID): PLAYLIST_PID,
        (EntityType.TRACK, TRACK_ID): TRACK_PID,
    })
    return AppleMusicWriteAdapter(command_runner, members_runner, resolver)


class AppleMusicWriteCommandTest(unittest.TestCase):
    def test_command_uses_durable_playlist_and_track_bindings(self) -> None:
        command_runner = FakeCommandRunner()
        adapter = bound_adapter(command_runner, FakeMembersRunner())
        adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)
        self.assertEqual(command_runner.calls, [(PLAYLIST_PID, TRACK_PID)])

    def test_missing_playlist_binding_fails_closed(self) -> None:
        command_runner = FakeCommandRunner()
        adapter = bound_adapter(command_runner, FakeMembersRunner(), {
            (EntityType.TRACK, TRACK_ID): TRACK_PID,
        })
        with self.assertRaises(MembershipBindingMissingError):
            adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)
        self.assertEqual(command_runner.calls, [])

    def test_missing_track_binding_fails_closed(self) -> None:
        command_runner = FakeCommandRunner()
        adapter = bound_adapter(command_runner, FakeMembersRunner(), {
            (EntityType.PLAYLIST, PLAYLIST_ID): PLAYLIST_PID,
        })
        with self.assertRaises(MembershipBindingMissingError):
            adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)
        self.assertEqual(command_runner.calls, [])

    def test_canonical_external_ids_projection_cannot_substitute_for_binding(self) -> None:
        # The fixture carries external_ids.apple_music_persistent_id projections, but the adapter
        # must trust only the binding resolver (physical authority). A resolver reporting no
        # binding fails closed even though a projection exists.
        temporary = tempfile.TemporaryDirectory()
        try:
            database_path = Path(temporary.name) / "canonical.sqlite3"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(load_fixture())
            # The real resolver reads external_identity_bindings and resolves the projection-backed
            # entities, proving the physical table is the authority.
            with CanonicalRepository(database_path) as repository:
                resolver = RepositoryMembershipBindingResolver(repository)
                self.assertEqual(resolver.resolve(EntityType.PLAYLIST, PLAYLIST_ID), PLAYLIST_PID)
                self.assertEqual(resolver.resolve(EntityType.TRACK, TRACK_ID), TRACK_PID)
        finally:
            temporary.cleanup()

        # A resolver that reports no binding fails closed regardless of any projection.
        command_runner = FakeCommandRunner()
        adapter = AppleMusicWriteAdapter(
            command_runner, FakeMembersRunner(), FakeBindingResolver()
        )
        with self.assertRaises(MembershipBindingMissingError):
            adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)
        self.assertEqual(command_runner.calls, [])

    def test_fake_runner_command_success(self) -> None:
        adapter = bound_adapter(FakeCommandRunner(), FakeMembersRunner())
        self.assertIsNone(adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID))

    def test_fake_runner_command_failure_propagates(self) -> None:
        error = AppleMusicWriteError("Music unavailable")
        adapter = bound_adapter(FakeCommandRunner(error=error), FakeMembersRunner())
        with self.assertRaises(AppleMusicWriteError):
            adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)

    def test_production_runner_passes_persistent_ids_as_argv(self) -> None:
        hostile = 'opaque ID; $(touch /tmp/never) "quoted"'
        completed = SimpleNamespace(returncode=0, stdout="added\n", stderr="")
        with patch("music_agent.apple_music_write.subprocess.run", return_value=completed) as run:
            OsascriptAddMembershipRunner(timeout_seconds=3).run(hostile, TRACK_PID)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["osascript", "-e"])
        self.assertEqual(argv[-2], hostile)
        self.assertEqual(argv[-1], TRACK_PID)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 3)

    def test_production_runner_nonzero_exit_is_write_error(self) -> None:
        failed = SimpleNamespace(returncode=1, stdout="", stderr="Music unavailable")
        with patch("music_agent.apple_music_write.subprocess.run", return_value=failed):
            with self.assertRaises(AppleMusicWriteError):
                OsascriptAddMembershipRunner().run(PLAYLIST_PID, TRACK_PID)

    def test_production_command_path_never_invokes_real_osascript(self) -> None:
        # The adapter is driven by fake runners; the real subprocess boundary is untouched.
        command_runner = FakeCommandRunner()
        adapter = bound_adapter(command_runner, FakeMembersRunner())
        with patch("music_agent.apple_music_write.subprocess.run") as run:
            adapter.add_playlist_membership(PLAYLIST_ID, TRACK_ID)
        run.assert_not_called()
        self.assertEqual(command_runner.calls, [(PLAYLIST_PID, TRACK_PID)])


class AppleMusicMembersReadTest(unittest.TestCase):
    def test_read_playlist_members_parses_persistent_ids(self) -> None:
        members_runner = FakeMembersRunner(json.dumps(["A", "B", "A"]))
        adapter = bound_adapter(FakeCommandRunner(), members_runner)
        self.assertEqual(adapter.read_playlist_members(PLAYLIST_ID), ("A", "B", "A"))
        self.assertEqual(members_runner.calls, [PLAYLIST_PID])

    def test_read_requires_playlist_binding(self) -> None:
        members_runner = FakeMembersRunner()
        adapter = AppleMusicWriteAdapter(FakeCommandRunner(), members_runner, FakeBindingResolver())
        with self.assertRaises(MembershipBindingMissingError):
            adapter.read_playlist_members(PLAYLIST_ID)
        self.assertEqual(members_runner.calls, [])

    def test_malformed_members_output_is_mapping_error(self) -> None:
        adapter = bound_adapter(FakeCommandRunner(), FakeMembersRunner("not-json"))
        with self.assertRaises(AppleMusicWriteMappingError):
            adapter.read_playlist_members(PLAYLIST_ID)

    def test_non_array_members_output_is_mapping_error(self) -> None:
        adapter = bound_adapter(FakeCommandRunner(), FakeMembersRunner('{"a": 1}'))
        with self.assertRaises(AppleMusicWriteMappingError):
            adapter.read_playlist_members(PLAYLIST_ID)

    def test_members_runner_passes_playlist_persistent_id_as_argv(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="[]\n", stderr="")
        with patch("music_agent.apple_music_write.subprocess.run", return_value=completed) as run:
            OsascriptPlaylistMembersRunner(timeout_seconds=3).run(PLAYLIST_PID)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["osascript", "-e"])
        self.assertEqual(argv[-1], PLAYLIST_PID)
        self.assertNotIn("shell", run.call_args.kwargs)


class MembershipReadbackEvaluationTest(unittest.TestCase):
    def test_pre_existing_same_track_does_not_false_confirm(self) -> None:
        # The requested Track is already present; presence alone cannot prove this command added
        # a new occurrence, so the decision must stay UNAVAILABLE, never MATCHED.
        decision = evaluate_membership_readback((TRACK_PID,), TRACK_PID)
        self.assertIs(decision, ReadbackDecision.UNAVAILABLE)

    def test_duplicate_track_membership_does_not_false_confirm(self) -> None:
        decision = evaluate_membership_readback((TRACK_PID, TRACK_PID), TRACK_PID)
        self.assertIs(decision, ReadbackDecision.UNAVAILABLE)

    def test_absent_track_is_unavailable_not_mismatch(self) -> None:
        # Absence is also not a safe mismatch (concurrent edits / crash windows), so it stays
        # UNAVAILABLE rather than confirming or mismatching.
        decision = evaluate_membership_readback(("OTHER-PID",), TRACK_PID)
        self.assertIs(decision, ReadbackDecision.UNAVAILABLE)

    def test_empty_members_is_unavailable_not_confirmed(self) -> None:
        decision = evaluate_membership_readback((), TRACK_PID)
        self.assertIs(decision, ReadbackDecision.UNAVAILABLE)

    def test_evaluation_never_returns_matched(self) -> None:
        for members in ((), (TRACK_PID,), (TRACK_PID, TRACK_PID), ("X", TRACK_PID)):
            with self.subTest(members=members):
                self.assertIsNot(
                    evaluate_membership_readback(members, TRACK_PID),
                    ReadbackDecision.MATCHED,
                )

    def test_invalid_readback_evidence_is_rejected(self) -> None:
        with self.assertRaises(AppleMusicWriteMappingError):
            evaluate_membership_readback("not-a-sequence", TRACK_PID)  # type: ignore[arg-type]
        with self.assertRaises(AppleMusicWriteMappingError):
            evaluate_membership_readback((TRACK_PID, ""), TRACK_PID)
        with self.assertRaises(AppleMusicWriteMappingError):
            evaluate_membership_readback((TRACK_PID,), "")


class CapabilityTruthTest(unittest.TestCase):
    def test_add_membership_adapter_and_readback_flags_are_split(self) -> None:
        capability = resolve_capability(WriteOperation.ADD_PLAYLIST_MEMBERSHIP)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertTrue(capability.capability_verified)
        self.assertTrue(capability.adapter_implemented)
        self.assertFalse(capability.readback_implemented)

    def test_add_membership_is_not_execution_ready_without_readback(self) -> None:
        self.assertFalse(is_execution_ready(resolve_capability(WriteOperation.ADD_PLAYLIST_MEMBERSHIP)))

    def test_create_delete_playlist_remain_unspecified_and_unimplemented(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.DELETE_PLAYLIST):
            capability = resolve_capability(operation)
            self.assertIs(capability.domain_permission, DomainPermission.UNSPECIFIED)
            self.assertFalse(capability.adapter_implemented)
            self.assertFalse(capability.readback_implemented)
            self.assertFalse(is_execution_ready(capability))

    def test_remove_membership_is_not_enabled(self) -> None:
        capability = resolve_capability(WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP)
        self.assertFalse(capability.adapter_implemented)
        self.assertFalse(capability.readback_implemented)
        self.assertFalse(is_execution_ready(capability))


if __name__ == "__main__":
    unittest.main()
