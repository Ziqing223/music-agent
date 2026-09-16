import unittest

from music_agent.apple_music_write import AppleMusicWriteAdapter
from music_agent.identity import EntityType
from music_agent.write_intent import RequirementRole
from music_agent.write_intent_formation import (
    BindingDriftError,
    MissingExternalBindingError,
    form_add_playlist_membership_intent,
    resolve_requirements,
)


PLAYLIST_ID = "pl_55555555-5555-4555-8555-555555555555"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
PLAYLIST_PID = "SYNTH-PLAYLIST-1"
TRACK_PID = "SYNTH-TRACK-001"


class FakeBindingResolver:
    """Resolve canonical -> external through an in-memory binding map, returning None on absence."""

    def __init__(self, bindings: dict[tuple[EntityType, str], str] | None = None) -> None:
        self.bindings = dict(bindings or {})

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None:
        return self.bindings.get((entity_type, canonical_id))


class FakeCommandRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, playlist_persistent_id: str, track_persistent_id: str) -> str:
        self.calls.append((playlist_persistent_id, track_persistent_id))
        return "added"


def both_bound_resolver() -> FakeBindingResolver:
    return FakeBindingResolver(
        {
            (EntityType.PLAYLIST, PLAYLIST_ID): PLAYLIST_PID,
            (EntityType.TRACK, TRACK_ID): TRACK_PID,
        }
    )


class RelationIntentFormationTest(unittest.TestCase):
    def test_formation_captures_playlist_and_track_requirements(self) -> None:
        intent = form_add_playlist_membership_intent(
            PLAYLIST_ID, TRACK_ID, both_bound_resolver()
        )
        by_role = {requirement.role: requirement for requirement in intent.requirements}
        self.assertEqual(set(by_role), {RequirementRole.PLAYLIST, RequirementRole.TRACK})
        self.assertEqual(by_role[RequirementRole.PLAYLIST].canonical_id, PLAYLIST_ID)
        self.assertEqual(
            by_role[RequirementRole.PLAYLIST].external_identity.external_id, PLAYLIST_PID
        )
        self.assertEqual(by_role[RequirementRole.TRACK].canonical_id, TRACK_ID)
        self.assertEqual(
            by_role[RequirementRole.TRACK].external_identity.external_id, TRACK_PID
        )

    def test_missing_playlist_binding_forms_no_intent(self) -> None:
        resolver = FakeBindingResolver({(EntityType.TRACK, TRACK_ID): TRACK_PID})
        with self.assertRaises(MissingExternalBindingError):
            form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)

    def test_missing_track_binding_forms_no_intent(self) -> None:
        resolver = FakeBindingResolver({(EntityType.PLAYLIST, PLAYLIST_ID): PLAYLIST_PID})
        with self.assertRaises(MissingExternalBindingError):
            form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)

    def test_projection_only_identity_forms_no_intent(self) -> None:
        # A resolver that reports no durable binding fails closed, even though a canonical
        # external_ids projection could exist elsewhere; formation never consults projections.
        with self.assertRaises(MissingExternalBindingError):
            form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, FakeBindingResolver())


class RequirementResolutionTest(unittest.TestCase):
    def test_resolution_returns_captured_external_ids_by_role(self) -> None:
        resolver = both_bound_resolver()
        intent = form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)
        resolved = resolve_requirements(intent.requirements, resolver)
        self.assertEqual(resolved[RequirementRole.PLAYLIST], PLAYLIST_PID)
        self.assertEqual(resolved[RequirementRole.TRACK], TRACK_PID)

    def test_missing_binding_at_execution_fails_closed(self) -> None:
        resolver = both_bound_resolver()
        intent = form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)
        del resolver.bindings[(EntityType.PLAYLIST, PLAYLIST_ID)]
        with self.assertRaises(MissingExternalBindingError):
            resolve_requirements(intent.requirements, resolver)


class BindingDriftTest(unittest.TestCase):
    def test_playlist_drift_fails_closed_without_command(self) -> None:
        resolver = both_bound_resolver()
        intent = form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)
        resolver.bindings[(EntityType.PLAYLIST, PLAYLIST_ID)] = "SYNTH-PLAYLIST-2"
        runner = FakeCommandRunner()
        adapter = AppleMusicWriteAdapter(runner, None, resolver)
        with self.assertRaises(BindingDriftError):
            adapter.add_playlist_membership_requirements(intent.requirements)
        self.assertEqual(runner.calls, [])

    def test_track_drift_fails_closed_without_command(self) -> None:
        resolver = both_bound_resolver()
        intent = form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)
        resolver.bindings[(EntityType.TRACK, TRACK_ID)] = "SYNTH-TRACK-999"
        runner = FakeCommandRunner()
        adapter = AppleMusicWriteAdapter(runner, None, resolver)
        with self.assertRaises(BindingDriftError):
            adapter.add_playlist_membership_requirements(intent.requirements)
        self.assertEqual(runner.calls, [])

    def test_drift_free_requirements_run_command_with_captured_ids(self) -> None:
        resolver = both_bound_resolver()
        intent = form_add_playlist_membership_intent(PLAYLIST_ID, TRACK_ID, resolver)
        runner = FakeCommandRunner()
        adapter = AppleMusicWriteAdapter(runner, None, resolver)
        adapter.add_playlist_membership_requirements(intent.requirements)
        self.assertEqual(runner.calls, [(PLAYLIST_PID, TRACK_PID)])


if __name__ == "__main__":
    unittest.main()
