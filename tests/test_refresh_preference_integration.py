"""P10: Refresh -> P06 preference ingestion integration tests (sealed semantics only).

Proves the production wiring: the SAME observation that refreshes canonical state is fed
through the sealed ``ingest_track_observation`` path, with the sealed repository semantics
doing the idempotency work (identical values confirm without revisions; changed values
transition; MISSING/NULL stay three-state). No new preference rules are asserted here.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference
from music_agent.refresh import PreferenceRefreshError, refresh_known_track
from music_agent.repository import CanonicalRepository
from music_agent.runtime_refresh import MusicRefreshOrchestrator

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
PERSISTENT_ID = "SYNTH-TRACK-001"


def fixture_model() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeRunner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        return self.output


def signal_head(repository: PreferencePersistenceRepository, signal_path: str):
    return repository.get_head(
        SignalIdentity(
            PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
            "apple_music",
            signal_path,
        )
    )


class RefreshPreferenceIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture_model())

    def _refresh(self, output: str, with_preference: bool = True):
        runner = FakeRunner(output)
        preference = (
            PreferencePersistenceRepository(self.database_path)
            if with_preference
            else None
        )
        try:
            with CanonicalRepository(self.database_path) as repository:
                result = refresh_known_track(
                    repository,
                    AppleMusicSourceAdapter(runner),
                    TRACK_ID,
                    preference_repository=preference,
                )
        finally:
            if preference is not None:
                preference.close()
        return result, runner

    def test_refresh_feeds_sealed_ingestion_and_query_sees_evidence(self) -> None:
        output = json.dumps(
            {"status": "found",
             "fields": {"name": "起风了", "favorited": True, "rating": 80, "played_count": 46}}
        )
        self._refresh(output)
        with PreferencePersistenceRepository(self.database_path) as preference:
            favorited = signal_head(preference, "favorited")
            self.assertIsNotNone(favorited)
            self.assertIs(favorited.current_semantic_value, True)
            rating = signal_head(preference, "rating")
            self.assertEqual(rating.current_semantic_value, 80)
            play_count = signal_head(preference, "play_count")
            self.assertEqual(play_count.current_semantic_value, 46)
            # Exactly one baseline revision per signal (no duplicates).
            for path in ("favorited", "rating", "play_count"):
                identity = SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                    "apple_music",
                    path,
                )
                self.assertEqual(len(preference.list_revisions(identity)), 1)

    def test_repeated_unchanged_refresh_never_inflates_revisions(self) -> None:
        output = json.dumps(
            {"status": "found", "fields": {"favorited": True, "rating": 80, "played_count": 46}}
        )
        for _ in range(3):
            result, _ = self._refresh(output)
            self.assertIn(result.status.value, ("updated", "unchanged"))
        with PreferencePersistenceRepository(self.database_path) as preference:
            for path in ("favorited", "rating", "play_count"):
                identity = SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                    "apple_music",
                    path,
                )
                self.assertEqual(len(preference.list_revisions(identity)), 1)
                self.assertEqual(preference.get_head(identity).current_revision_sequence, 1)

    def test_changed_value_flows_through_revision_semantics(self) -> None:
        first = json.dumps({"status": "found", "fields": {"favorited": True}})
        self._refresh(first)
        changed = json.dumps({"status": "found", "fields": {"favorited": False}})
        self._refresh(changed)
        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                "apple_music",
                "favorited",
            )
            self.assertEqual(len(preference.list_revisions(identity)), 2)
            head = preference.get_head(identity)
            self.assertIs(head.current_semantic_value, False)
            self.assertEqual(head.current_revision_sequence, 2)

    def test_play_count_changes_familiarity_only(self) -> None:
        from music_agent.agent_service import (
            PRODUCTION_FAMILIARITY_POLICY,
            PRODUCTION_MAGNITUDE_POLICY,
            PRODUCTION_RATING_POLICY,
        )
        from music_agent.preference_query import query_track_preference

        self._refresh(json.dumps({"status": "found", "fields": {"favorited": True, "played_count": 5}}))
        with PreferencePersistenceRepository(self.database_path) as preference:
            before = query_track_preference(
                preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                rating_policy=PRODUCTION_RATING_POLICY,
                magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
                familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
            )
        self._refresh(json.dumps({"status": "found", "fields": {"favorited": True, "played_count": 50}}))
        with PreferencePersistenceRepository(self.database_path) as preference:
            after = query_track_preference(
                preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                rating_policy=PRODUCTION_RATING_POLICY,
                magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
                familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
            )
        # Familiarity moved (sealed S4), the directional preference did not change.
        self.assertNotEqual(before.familiarity, after.familiarity)
        self.assertEqual(before.direct_preference.strength.state, after.direct_preference.strength.state)

    def test_missing_and_null_stay_three_state(self) -> None:
        # The sealed read adapter maps absent-or-null source fields to MISSING (it only
        # ever emits VALUE for actual values); MISSING heads record state without
        # revisions -- the three-state semantics stay exactly as sealed.
        self._refresh(json.dumps({"status": "found", "fields": {"rating": None, "played_count": 0}}))
        with PreferencePersistenceRepository(self.database_path) as preference:
            favorited = signal_head(preference, "favorited")
            self.assertEqual(favorited.last_observed_state.value, "missing")
            self.assertEqual(len(preference.list_revisions(favorited.identity)), 0)
            rating = signal_head(preference, "rating")
            self.assertEqual(rating.last_observed_state.value, "missing")
            self.assertEqual(len(preference.list_revisions(rating.identity)), 0)
            play_count = signal_head(preference, "play_count")
            self.assertEqual(play_count.current_semantic_value, 0)

    def test_ingestion_failure_is_isolated_and_canonical_state_stays_saved(self) -> None:
        output = json.dumps({"status": "found", "fields": {"name": "Saved Before Ingestion"}})
        runner = FakeRunner(output)
        with patch(
            "music_agent.preference_ingestion.ingest_track_observation",
            side_effect=RuntimeError("p06 down"),
        ):
            with CanonicalRepository(self.database_path) as repository:
                with PreferencePersistenceRepository(self.database_path) as preference:
                    with self.assertRaises(PreferenceRefreshError):
                        refresh_known_track(
                            repository,
                            AppleMusicSourceAdapter(runner),
                            TRACK_ID,
                            preference_repository=preference,
                        )
        # The canonical save committed BEFORE the ingestion failure: the store kept it.
        with CanonicalRepository(self.database_path) as repository:
            track = next(t for t in repository.load_model()["tracks"] if t["id"] == TRACK_ID)
            self.assertEqual(track["name"], "Saved Before Ingestion")
        # Retry after the failure: ingestion recovers (idempotent, one baseline revision).
        self._refresh(json.dumps({"status": "found", "fields": {"name": "Saved Before Ingestion"}}))
        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID),
                "apple_music",
                "favorited",
            )
            self.assertIsNotNone(preference.get_head(identity))

    def test_sealed_default_unchanged_without_preference_repository(self) -> None:
        output = json.dumps({"status": "found", "fields": {"favorited": True}})
        self._refresh(output, with_preference=False)
        with PreferencePersistenceRepository(self.database_path) as preference:
            self.assertIsNone(signal_head(preference, "favorited"))

    def test_orchestrator_cycle_feeds_p06_for_every_bound_track(self) -> None:
        output = json.dumps({"status": "found", "fields": {"favorited": True}})
        runner = FakeRunner(output)
        with CanonicalRepository(self.database_path) as repository:
            with PreferencePersistenceRepository(self.database_path) as preference:
                report = MusicRefreshOrchestrator(
                    repository, AppleMusicSourceAdapter(runner), preference_repository=preference
                ).run_cycle()
        self.assertTrue(report.succeeded)
        with PreferencePersistenceRepository(self.database_path) as preference:
            for persistent_id in ("SYNTH-TRACK-001", "SYNTH-TRACK-002", "SYNTH-TRACK-004"):
                key = ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
                with CanonicalRepository(self.database_path) as repository:
                    canonical_id = repository.lookup_external_identity(key)
                head = preference.get_head(
                    SignalIdentity(
                        PreferenceTargetReference(PreferenceTargetKind.TRACK, canonical_id),
                        "apple_music",
                        "favorited",
                    )
                )
                self.assertIsNotNone(head)
                self.assertIs(head.current_semantic_value, True)


if __name__ == "__main__":
    unittest.main()
