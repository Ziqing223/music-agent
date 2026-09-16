"""P11.4: the causal learning loop through the catalog candidate path.

Deterministic end-to-end validation using the REAL production surfaces (shared agent
service tools, P06 signal heads, the recommendation-feedback bridge, the affinity
reducers, the catalog candidate generator). Only the human's external action -- recording
feedback -- is simulated through the real ``record_feedback`` + ``apply_learning`` tools.

Scenario:

    A. baseline: two apple_music library preferences (Rock positive, Metal positive)
    B. run A: catalog candidates C_pos (Rock) and C_neg (Metal) BOTH recommended
    C. history: run A persisted (the LIKED feedback below can only resolve against it)
    D. feedback: LIKED on C_pos (via the persisted run's candidate), DISLIKED on C_neg
       (direct target) -- both under the feedback_learning source
    E. the feedback updates the unified P06 state: C_pos POSITIVE, C_neg NEGATIVE
    F. run B: same store, same target ids, source feedback_learning
    G. observable, attributable change: C_neg disappears (its Metal basis now matches a
       NEGATIVE inferred input derived from the feedback); C_pos stays and scores > 0.
       The apple_music control run still recommends C_neg -- the suppression came from
       the learning path, not from any global state wipe. Unrelated library preferences
       survive untouched.

A test that merely observed "two different lists" would not pass this shape: each link
of the causal chain (feedback row -> P06 revision -> affinity input -> candidate
rejection) is asserted individually.
"""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient, AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import (
    PRODUCTION_FAMILIARITY_POLICY,
    PRODUCTION_MAGNITUDE_POLICY,
    PRODUCTION_RATING_POLICY,
    SharedAgentService,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_query import query_track_preference
from music_agent.preference_strength import PreferenceState
from music_agent.recommendation_contract import decode_recommendation_result
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

TRACK_A = "trk_11111111-1111-4111-8111-111111111111"  # baseline Rock positive
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"  # baseline Metal positive
C_POS = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
C_NEG = "trk_c3c3c3c3-c3c3-4c3c-8c3c-c3c3c3c3c3c3"

OBSERVED_AT = "2026-08-17T01:00:00+00:00"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class P11LearningLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = load_fixture()
        artist_id = fixture["artists"][0]["id"]
        for track in fixture["tracks"]:
            if track["id"] == TRACK_A:
                track["genres"] = ["Rock"]
                track["artist_ids"] = [artist_id]
            elif track["id"] == TRACK_C:
                track["genres"] = ["Metal"]
                track["artist_ids"] = [artist_id]
        for track_id, genres, catalog_id in (
            (C_POS, ["Rock"], "CATALOG-POS"),
            (C_NEG, ["Metal"], "CATALOG-NEG"),
        ):
            fixture["tracks"].append(
                {
                    "id": track_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "apple_music_catalog_id": catalog_id,
                    },
                    "name": f"Catalog {catalog_id}",
                    "artist_ids": [artist_id],
                    "album_id": fixture["albums"][0]["id"],
                    "duration_ms": 201000,
                    "genres": genres,
                    "track_number": None,
                    "disc_number": None,
                    "release_date": "2024-01-15",
                    "composer": None,
                    "library_state": {
                        "favorited": None, "disliked": None, "rating": None,
                        "play_count": None, "skip_count": None,
                        "added_to_library_at": None, "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self._seed_signal(TRACK_A, "favorited", True)
        self._seed_signal(TRACK_C, "favorited", True)
        service = SharedAgentService(
            self.database_path, clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL})
        )
        self.addCleanup(service.close)
        self.service = service
        self.client = AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def _seed_signal(self, track_id: str, signal_path: str, value: bool) -> None:
        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                "apple_music",
                signal_path,
            )
            preference.record_observation(identity, ObservedValue.value(value))

    def recommend(self, source_system: str) -> dict:
        # P19-T15: explicit opt-out of recent-run dedup -- the learning-loop
        # causality under test must not be confounded by novelty exclusions.
        result = self.client.call(
            "generate_inferred_recommendation",
            {
                "target_ids": [TRACK_A, TRACK_C],
                "limit": 10,
                "source_system": source_system,
                "avoid_previous_runs": False,
            },
        )
        self.assertEqual(result.outcome.value, "ok")
        return result.payload

    def _recommend_at(self, produced_at: str) -> dict:
        # P15 burn-down Issue 1: produced_at rides the trusted completed_at
        # execution seam (the service's execution instant), never the payload.
        # P19-T15: explicit opt-out of recent-run dedup -- these runs are
        # pinned history fixtures, not novelty probes.
        result = self.client.call(
            "generate_inferred_recommendation",
            {
                "target_ids": [TRACK_A, TRACK_C],
                "limit": 10,
                "source_system": "apple_music",
                "avoid_previous_runs": False,
            },
            completed_at=produced_at,
        )
        self.assertEqual(result.outcome.value, "ok")
        return result.payload

    def record_and_apply(
        self,
        kind: str,
        *,
        target_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
    ) -> str:
        payload = {
            "kind": kind,
            "source_system": "feedback_learning",
            "source_path": "recommendation_ui",
        }
        if target_id is not None:
            payload["target_id"] = target_id
        if run_id is not None:
            payload["run_id"] = run_id
        if candidate_id is not None:
            payload["candidate_id"] = candidate_id
        recorded = self.client.call("record_feedback", payload)
        self.assertEqual(recorded.outcome.value, "ok")
        applied = self.client.call("apply_learning", {"feedback_id": recorded.payload["feedback_id"]})
        self.assertEqual(applied.outcome.value, "ok")
        self.assertTrue(applied.payload["applied"])
        return recorded.payload["feedback_id"]

    def direct_state(self, track_id: str, source_system: str) -> PreferenceState:
        with PreferencePersistenceRepository(self.database_path) as preference:
            state = query_track_preference(
                preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                rating_policy=PRODUCTION_RATING_POLICY,
                magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
                familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
                source_system=source_system,
            )
        return state.direct_preference.strength.state

    def test_catalog_learning_loop_changes_the_next_run_causally(self) -> None:
        # A + B: baseline preferences put BOTH catalog songs into run A.
        run_a = self.recommend("apple_music")
        run_a_items = {item["target_id"]: item for item in run_a["items"]}
        self.assertEqual(run_a_items[C_POS]["label"], "catalog")
        self.assertEqual(run_a_items[C_NEG]["label"], "catalog")
        self.assertGreater(run_a_items[C_POS]["score_total"], 0)
        self.assertGreater(run_a_items[C_NEG]["score_total"], 0)

        # C: run A is persisted history; the LIKED feedback below resolves ONLY against it.
        history = self.client.call("get_recommendation_run", {"run_id": run_a["run_id"]})
        self.assertEqual(history.outcome.value, "ok")
        decoded_a = decode_recommendation_result(history.payload["encoded_result"])
        pos_candidate_id = next(
            item.candidate.candidate_id
            for item in decoded_a.items
            if item.candidate.target.target_id == C_POS
        )

        # D: explicit feedback on previously unknown songs, through the real tools.
        self.record_and_apply(
            "liked", run_id=run_a["run_id"], candidate_id=pos_candidate_id
        )
        self.record_and_apply("disliked", target_id=C_NEG)

        # E: the feedback updated the unified P06 state (same signal heads the
        # recommendations read), scoped to the feedback_learning source.
        self.assertEqual(self.direct_state(C_POS, "feedback_learning"), PreferenceState.POSITIVE)
        self.assertEqual(self.direct_state(C_NEG, "feedback_learning"), PreferenceState.NEGATIVE)
        # Unrelated library preferences are NOT erased.
        self.assertEqual(self.direct_state(TRACK_A, "apple_music"), PreferenceState.POSITIVE)
        self.assertEqual(self.direct_state(TRACK_C, "apple_music"), PreferenceState.POSITIVE)

        # F: run B -- same store, same targets, feedback_learning scope.
        run_b = self.recommend("feedback_learning")
        run_b_items = {item["target_id"]: item for item in run_b["items"]}
        self.assertIn(C_POS, run_b_items)
        self.assertGreater(run_b_items[C_POS]["score_total"], 0)
        self.assertNotIn(C_NEG, run_b_items)  # G: suppressed by the negative feedback

        # The mechanism: the run B context carries a NEGATIVE inferred Metal input that
        # exists ONLY because of the C_neg feedback, and it is that input which vetoes
        # C_neg's candidate (Metal is its basis genre).
        decoded_b = decode_recommendation_result(run_b["encoded_result"])
        metal_inputs = [
            input_
            for input_ in decoded_b.request.context.preference_inputs
            if input_.target == PreferenceTargetReference(PreferenceTargetKind.GENRE, "Metal")
        ]
        self.assertTrue(metal_inputs)
        self.assertTrue(
            all(input_.strength.state is PreferenceState.NEGATIVE for input_ in metal_inputs)
        )

        # Control: the apple_music scope is untouched by the feedback source -- the SAME
        # song is still recommended there. The suppression is attributable to the
        # learning path, not to a global state change.
        control = self.recommend("apple_music")
        control_items = {item["target_id"]: item for item in control["items"]}
        self.assertIn(C_NEG, control_items)
        self.assertIn(C_POS, control_items)

        # The positive feedback similarly carries through the same mechanism.
        rock_inputs = [
            input_
            for input_ in decoded_b.request.context.preference_inputs
            if input_.target == PreferenceTargetReference(PreferenceTargetKind.GENRE, "Rock")
        ]
        self.assertTrue(rock_inputs)
        self.assertTrue(
            all(input_.strength.state is PreferenceState.POSITIVE for input_ in rock_inputs)
        )

    def test_history_resolves_recent_run_for_feedback_reference(self) -> None:
        # P12-C03: "刚才推荐的 Lamp《Yume Utsutsu》" must resolve to run + candidate
        # from list_recommendation_runs alone -- no user-provided run id.
        yume_id = "trk_00f0edb5-f12a-4554-95fd-eea75db45274"
        lamp_id = "art_16e9c847-34b8-4794-bd7d-b9f05bb1f9cb"
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            model["artists"].append(
                {"id": lamp_id, "external_ids": {"apple_music_persistent_id": None}, "name": "Lamp"}
            )
            model["tracks"].append(
                {
                    "id": yume_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "apple_music_catalog_id": "CATALOG-YUME",
                    },
                    "name": "Yume Utsutsu",
                    "artist_ids": [lamp_id],
                    "album_id": None,
                    "duration_ms": 201000,
                    "genres": ["Rock"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": "2024-01-15",
                    "composer": None,
                    "library_state": {
                        "favorited": None, "disliked": None, "rating": None,
                        "play_count": None, "skip_count": None,
                        "added_to_library_at": None, "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
            repository.save_model(model)

        older_at = "2026-08-16T23:00:00+00:00"
        newer_at = "2026-08-17T00:30:00+00:00"
        older = self._recommend_at(older_at)
        newer = self._recommend_at(newer_at)
        # created_at is written by SQLite CURRENT_TIMESTAMP (second resolution) at insert
        # time, so both saves can land in the same second. Pin the two insertion instants
        # deterministically instead of waiting on the wall clock: the immutability trigger
        # blocks UPDATE, so drop it for this test-only backfill and restore it verbatim
        # (migration 0012_recommendation_history.sql).
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        try:
            connection.execute("DROP TRIGGER trg_recommendation_runs_immutable_update")
            connection.execute(
                "UPDATE recommendation_runs SET created_at=? WHERE run_id=?",
                ("2026-08-16 22:00:00", older["run_id"]),
            )
            connection.execute(
                "UPDATE recommendation_runs SET created_at=? WHERE run_id=?",
                ("2026-08-16 23:00:00", newer["run_id"]),
            )
        finally:
            connection.execute(
                """CREATE TRIGGER trg_recommendation_runs_immutable_update
    BEFORE UPDATE ON recommendation_runs
BEGIN
    SELECT RAISE(ABORT, 'recommendation runs are immutable');
END;"""
            )
            connection.close()
        self.assertNotEqual(older["run_id"], newer["run_id"])

        listed = self.client.call("list_recommendation_runs", {})
        self.assertEqual(listed.outcome.value, "ok")
        runs = listed.payload["runs"]
        # Newest-first: the latest run is deterministically runs[0] ("刚才").
        self.assertEqual(runs[0]["run_id"], newer["run_id"])
        self.assertEqual(runs[0]["produced_at"], newer_at)
        self.assertEqual(runs[1]["run_id"], older["run_id"])
        self.assertEqual(runs[1]["produced_at"], older_at)

        # The referenced candidate is identifiable by display names in the latest run.
        yume_item = next(
            item for item in runs[0]["items"] if item["target_id"] == yume_id
        )
        self.assertEqual(yume_item["name"], "Yume Utsutsu")
        self.assertEqual(yume_item["artist_name"], "Lamp")
        self.assertEqual(yume_item["target_kind"], "track")
        self.assertIn("candidate_id", yume_item)
        self.assertIn("score_total", yume_item)

        # get_recommendation_run hydrates the same display names alongside the
        # canonical encoded_result.
        fetched = self.client.call("get_recommendation_run", {"run_id": newer["run_id"]})
        self.assertEqual(fetched.outcome.value, "ok")
        fetched_yume = next(
            item for item in fetched.payload["items"] if item["target_id"] == yume_id
        )
        self.assertEqual(fetched_yume["name"], "Yume Utsutsu")
        self.assertEqual(fetched_yume["artist_name"], "Lamp")
        decode_recommendation_result(fetched.payload["encoded_result"])

        # The resolved run_id + candidate_id write feedback directly.
        self.record_and_apply(
            "liked", run_id=newer["run_id"], candidate_id=yume_item["candidate_id"]
        )


if __name__ == "__main__":
    unittest.main()
