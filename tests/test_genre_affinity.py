"""P10.17b: Genre-affinity reducer + inferred-recommendation tool tests (fakes only)."""

import json
import tempfile
import unittest
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.genre_affinity import (
    GenreAffinityError,
    GenreAffinityPolicy,
    SourcedContribution,
    build_genre_affinities,
    infer_track_affinity,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import (
    ConstraintMode,
    InferredAffinityContribution,
    PropagationKind,
    propagate_track_preference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_22222222-2222-4222-8222-222222222222"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
TRACK_D = "trk_44444444-4444-4444-8444-444444444444"


def contribution(source_system: str, track_id: str, genre: str, direction: str, magnitude: float):
    direct = PreferenceStrength(
        PreferenceState.POSITIVE if direction == "positive" else PreferenceState.NEGATIVE,
        magnitude,
    )
    from music_agent.preference_attribution import DerivedPreference

    propagated = propagate_track_preference(
        DerivedPreference(
            PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id), direct
        ),
        genres=[genre],
    )
    genre_contribution = next(c for c in propagated if c.kind is PropagationKind.GENRE)
    return SourcedContribution(source_system, genre_contribution)


class GenreAffinityReducerTest(unittest.TestCase):
    def test_aggregation_saturation_and_determinism(self) -> None:
        sourced = [
            contribution("apple_music", "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "J-Pop", "positive", 0.9),
            contribution("apple_music", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "J-Pop", "positive", 0.8),
            contribution("apple_music", "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc", "Rock", "positive", 0.5),
        ]
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        by_key = {a.genre_key: a for a in affinities}
        self.assertEqual(by_key["J-Pop"].positive_count, 2)
        self.assertEqual(by_key["J-Pop"].net, 1.7)
        self.assertEqual(by_key["J-Pop"].affinity, 1.0)  # capped at +1
        self.assertEqual(by_key["Rock"].affinity, 0.5)
        self.assertEqual(by_key["J-Pop"].source_track_ids, ("trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"))
        # Deterministic: keys sorted.
        self.assertEqual([a.genre_key for a in affinities], ["J-Pop", "Rock"])

    def test_positive_negative_arithmetic_and_single_dislike_bound(self) -> None:
        sourced = [
            contribution("apple_music", "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "J-Pop", "positive", 0.9),
            contribution("apple_music", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "J-Pop", "positive", 0.8),
            contribution("apple_music", "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc", "J-Pop", "negative", 0.9),
        ]
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        affinity = next(a for a in affinities if a.genre_key == "J-Pop")
        self.assertEqual(affinity.positive_count, 2)
        self.assertEqual(affinity.negative_count, 1)
        self.assertEqual(affinity.net, 0.8)  # 1.7 - 0.9
        self.assertEqual(affinity.affinity, 0.8)
        # A single dislike lowered the aggregate by exactly its attenuated magnitude.
        self.assertIn("正面 2 首 / 负面 1 首", affinity.explanation())

    def test_negative_flip_and_threshold(self) -> None:
        sourced = [
            contribution("apple_music", "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "J-Pop", "positive", 0.3),
            contribution("apple_music", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "J-Pop", "negative", 0.9),
            contribution("apple_music", "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc", "Rock", "positive", 0.02),
        ]
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        by_key = {a.genre_key: a for a in affinities}
        self.assertLess(by_key["J-Pop"].affinity, 0)  # flipped negative
        # Rock's tiny affinity still exists in the aggregate but fails the threshold.
        selected = infer_track_affinity("t", ["Rock"], "apple_music", affinities)
        self.assertIsNone(selected)
        selected = infer_track_affinity("t", ["J-Pop"], "apple_music", affinities)
        self.assertEqual(selected.state, PreferenceState.NEGATIVE)
        self.assertAlmostEqual(selected.magnitude, 0.6)

    def test_source_separation(self) -> None:
        sourced = [
            contribution("apple_music", "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "J-Pop", "positive", 0.9),
            contribution("feedback_learning", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "J-Pop", "negative", 0.9),
        ]
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        apple = next(a for a in affinities if a.source_system == "apple_music")
        feedback = next(a for a in affinities if a.source_system == "feedback_learning")
        self.assertEqual(apple.affinity, 0.9)  # unaffected by the other source's negative
        self.assertEqual(feedback.affinity, -0.9)
        # Track inference for one source never sees the other.
        selected = infer_track_affinity("t", ["J-Pop"], "apple_music", affinities)
        self.assertEqual(selected.state, PreferenceState.POSITIVE)
        selected = infer_track_affinity("t", ["J-Pop"], "feedback_learning", affinities)
        self.assertEqual(selected.state, PreferenceState.NEGATIVE)

    def test_multi_genre_strongest_with_tie_break(self) -> None:
        sourced = [
            contribution("apple_music", "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "J-Pop", "positive", 0.6),
            contribution("apple_music", "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "Rock", "negative", 0.4),
        ]
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        selected = infer_track_affinity("t", ["Rock", "J-Pop"], "apple_music", affinities)
        self.assertEqual(selected.state, PreferenceState.POSITIVE)  # |0.6| > |0.4|

    def test_non_genre_contribution_rejected(self) -> None:
        direct = PreferenceStrength(PreferenceState.POSITIVE, 0.8)
        from music_agent.preference_attribution import DerivedPreference

        album_contribution = next(
            c for c in propagate_track_preference(
                DerivedPreference(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"), direct
                ),
                album_id="alb_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            )
            if c.kind is not PropagationKind.GENRE
        )
        with self.assertRaises(GenreAffinityError):
            build_genre_affinities([SourcedContribution("apple_music", album_contribution)])


class InferredRecommendationToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for track in fixture["tracks"]:
            track["genres"] = ["J-Pop"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def _seed_signal(self, track_id: str, signal_path: str, value: bool) -> None:
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import PreferencePersistenceRepository

        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                "apple_music",
                signal_path,
            )
            preference.record_observation(identity, ObservedValue.value(value))

    def _service(self, policy: AgentClientPolicy) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path, clients=AgentClientRegistry({CLIENT_ID: policy})
        )
        self.addCleanup(service.close)
        return service

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def test_novel_candidate_emerges_from_genre_inference(self) -> None:
        self._seed_signal(TRACK_A, "favorited", True)  # direct positive, genre J-Pop
        self._seed_signal(TRACK_B, "favorited", True)  # direct positive, genre J-Pop
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_B, TRACK_C, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = result.payload["items"]
        labels = {item["target_id"]: item for item in items}
        self.assertEqual(labels[TRACK_A]["label"], "known_positive")
        self.assertEqual(labels[TRACK_B]["label"], "known_positive")
        # TRACK_C (unbound) and TRACK_D carry no direct evidence: genre inference
        # makes them novel through the J-Pop affinity from A and B.
        self.assertEqual(labels[TRACK_D]["label"], "novel")
        self.assertIn("genre: J-Pop", labels[TRACK_D]["explanation"])

    def test_direct_negative_vetoes_inferred_positive(self) -> None:
        self._seed_signal(TRACK_A, "favorited", True)
        self._seed_signal(TRACK_D, "disliked", True)  # sealed: disliked -> direct NEGATIVE
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = {item["target_id"]: item for item in result.payload["items"]}
        self.assertNotIn(TRACK_D, items)  # rejected by the sealed negative veto

    def test_catalog_bound_track_surfaces_as_catalog_item_with_unified_affinity(self) -> None:
        self._seed_signal(TRACK_A, "favorited", True)  # direct positive, genre J-Pop
        self._seed_signal(TRACK_B, "favorited", True)
        catalog_id = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            model["tracks"].append(
                {
                    "id": catalog_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "apple_music_catalog_id": "CATALOG-1",
                    },
                    "name": "Catalog J-Pop Song",
                    "artist_ids": [model["artists"][0]["id"]],
                    "album_id": None,
                    "duration_ms": 201000,
                    "genres": ["J-Pop"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": "2024-01-15",
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
            repository.save_model(model)
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_B, TRACK_C, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        labels = {item["target_id"]: item for item in result.payload["items"]}
        self.assertEqual(labels[catalog_id]["label"], "catalog")
        self.assertGreater(labels[catalog_id]["score_total"], 0)

    def test_recommendation_items_carry_track_and_artist_display_names(self) -> None:
        # P12 real-user failure: a catalog item rendered as 《Yume Utsutsu》 with no
        # artist. The compact item projection must be display-self-contained.
        self._seed_signal(TRACK_A, "favorited", True)
        self._seed_signal(TRACK_B, "favorited", True)
        catalog_id = "trk_00f0edb5-f12a-4554-95fd-eea75db45274"
        artist_id = "art_16e9c847-34b8-4794-bd7d-b9f05bb1f9cb"
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            model["artists"].append(
                {"id": artist_id, "external_ids": {"apple_music_persistent_id": None}, "name": "Lamp"}
            )
            model["tracks"].append(
                {
                    "id": catalog_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "apple_music_catalog_id": "CATALOG-YUME",
                    },
                    "name": "Yume Utsutsu",
                    "artist_ids": [artist_id],
                    "album_id": None,
                    "duration_ms": 201000,
                    "genres": ["J-Pop"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": "2024-01-15",
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
            repository.save_model(model)
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_B, TRACK_C, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = {item["target_id"]: item for item in result.payload["items"]}
        for item in result.payload["items"]:
            self.assertIn("name", item)
            self.assertIn("artist_name", item)
        # Catalog item hydrated through the canonical Track -> Artist relation.
        self.assertEqual(items[catalog_id]["name"], "Yume Utsutsu")
        self.assertEqual(items[catalog_id]["artist_name"], "Lamp")
        # Non-catalog items carry display names too, multi-artist joined.
        self.assertEqual(items[TRACK_A]["name"], "Synthetic Duet")
        self.assertEqual(items[TRACK_A]["artist_name"], "Artist Alpha, Artist Beta")

    def _append_catalog_track(self, catalog_id: str) -> None:
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            model["tracks"].append(
                {
                    "id": catalog_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "apple_music_catalog_id": f"CATALOG-{catalog_id}",
                    },
                    "name": "Catalog J-Pop Song",
                    "artist_ids": [model["artists"][0]["id"]],
                    "album_id": None,
                    "duration_ms": 201000,
                    "genres": ["J-Pop"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": "2024-01-15",
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
            repository.save_model(model)

    def test_persistent_id_only_track_contributes_affinity_for_catalog_target(self) -> None:
        # TRACK_A carries apple_music_persistent_id only (local-library, not
        # catalog-bound) and is NOT a request target: its evidence reaches the
        # affinity reducers only through the widened all-canonical-tracks scan.
        self._seed_signal(TRACK_A, "favorited", True)
        catalog_id = "trk_c2c2c2c2-c2c2-4c2c-8c2c-c2c2c2c2c2c2"
        self._append_catalog_track(catalog_id)
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_B, TRACK_C, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        labels = {item["target_id"]: item for item in result.payload["items"]}
        # The catalog target's only basis is the J-Pop affinity built from the
        # persistent-id-only local track.
        self.assertEqual(labels[catalog_id]["label"], "catalog")
        self.assertGreater(labels[catalog_id]["score_total"], 0)
        self.assertIn("genre: J-Pop", labels[catalog_id]["explanation"])

    def test_local_only_tracks_never_become_catalog_candidates(self) -> None:
        # Widening preference contributors must not widen the candidate set:
        # a persistent-id-only track feeds affinity but is never a candidate,
        # and non-catalog targets may still become novel via that affinity.
        self._seed_signal(TRACK_A, "favorited", True)
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_B, TRACK_C, TRACK_D], "limit": 5},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = {item["target_id"]: item for item in result.payload["items"]}
        self.assertNotIn(TRACK_A, items)  # contributor, never a candidate
        self.assertEqual(items[TRACK_B]["label"], "novel")
        self.assertIn("genre: J-Pop", items[TRACK_B]["explanation"])

    def test_feedback_learning_source_sees_only_its_evidence(self) -> None:
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import PreferencePersistenceRepository

        self._seed_signal(TRACK_A, "favorited", True)  # apple_music positive only
        with PreferencePersistenceRepository(self.database_path) as preference:
            preference.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_A),
                    "feedback_learning",
                    "favorited",
                ),
                ObservedValue.value(False),  # explicit dislike under feedback_learning
            )
            # P14-R2 refuses empty generations, so the source-scoped batch must stay
            # non-empty: give feedback_learning an eligible target of its own
            # (apple_music evidence must not leak in to fill the batch).
            preference.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_B),
                    "feedback_learning",
                    "favorited",
                ),
                ObservedValue.value(True),  # explicit like under feedback_learning
            )
        service = self._service(AgentClientPolicy.FULL)
        result = self._client(service).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_B, TRACK_D], "limit": 5, "source_system": "feedback_learning"},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = {item["target_id"]: item for item in result.payload["items"]}
        # The apple_music-only favorite of TRACK_A must not leak into the
        # feedback_learning-scoped batch: its direct state stays insufficient
        # (a leaked positive would render it known_positive) and it can only
        # surface as a novel inference from feedback_learning's own evidence.
        self.assertEqual(items[TRACK_B]["label"], "known_positive")
        self.assertEqual(items[TRACK_A]["direct_state"], "insufficient")
        self.assertEqual(result.payload["source_system"], "feedback_learning")

    def test_permission_and_replay(self) -> None:
        self._seed_signal(TRACK_A, "favorited", True)
        readonly = self._service(AgentClientPolicy.READ_ONLY)
        denied = self._client(readonly).call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A], "limit": 1},
        )
        self.assertEqual(denied.outcome.value, "permission_denied")

        service = self._service(AgentClientPolicy.FULL)
        client = self._client(service)
        request_id = "req_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
        first = client.call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A], "limit": 1},
            request_id=request_id,
        )
        second = client.call(
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A], "limit": 1},
            request_id=request_id,
        )
        self.assertEqual(first.outcome.value, "ok")
        self.assertEqual(second.outcome.value, "ok")
        self.assertTrue(second.replayed)
        self.assertEqual(first.payload["run_id"], second.payload["run_id"])

    def test_tool_schema_and_registry_coverage(self) -> None:
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

        self.assertIn(
            "generate_inferred_recommendation",
            {schema.name for schema in PROVIDER_TOOL_SCHEMAS},
        )


if __name__ == "__main__":
    unittest.main()
