"""True Similarity V1 service integration over temporary canonical stores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
    generate_request_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.intent_router import (
    TurnPrimarySemantic,
    resolve_turn_plan,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.recommendation_contract import decode_recommendation_result
from music_agent.recommendation_presenter import render_recommendation_for_user
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue
from music_agent.track_similarity import (
    SIMILARITY_SOURCE_PATH,
    SimilarityExecutionContext,
)
from music_agent.validation import validate_fixture


CLIENT = "agt_11111111-1111-4111-8111-111111111111"
SEED = "trk_11111111-1111-4111-8111-111111111111"
HIGH = "trk_22222222-2222-4222-8222-222222222222"
LOW = "trk_33333333-3333-4333-8333-333333333333"
UNRELATED = "trk_44444444-4444-4444-8444-444444444444"
TIE_A = "trk_55555555-5555-4555-8555-555555555555"
TIE_B = "trk_66666666-6666-4666-8666-666666666666"
OTHER_SEED = "trk_77777777-7777-4777-8777-777777777777"
HARUJION_ALBUM = "trk_08000000-0000-4000-8000-000000000001"
MONSTER_ALBUM = "trk_08000000-0000-4000-8000-000000000002"
PINK_BLOOD_ALBUM = "trk_08000000-0000-4000-8000-000000000003"
HARUJION_SINGLE = "trk_88000000-0000-4000-8000-000000000001"
MONSTER_SINGLE = "trk_88000000-0000-4000-8000-000000000002"
PINK_BLOOD_SINGLE = "trk_88000000-0000-4000-8000-000000000003"
ARTIST_A = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ARTIST_B = "art_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
NOW = "2026-09-10T00:00:00+00:00"


def canonical_track(
    track_id: str,
    name: str,
    *,
    artist_ids: list[str],
    genres: list[str],
    composer: str | None = None,
    tags: list[str] | None = None,
    duration_ms: int | None = None,
    release_date: str | None = None,
) -> dict:
    return {
        "id": track_id,
        "external_ids": {
            "apple_music_persistent_id": f"PID-{track_id.removeprefix('trk_')}"
        },
        "name": name,
        "artist_ids": artist_ids,
        "album_id": None,
        "duration_ms": duration_ms,
        "genres": genres,
        "track_number": None,
        "disc_number": None,
        "release_date": release_date,
        "composer": composer,
        "library_state": {
            "favorited": None,
            "disliked": None,
            "rating": None,
            "play_count": None,
            "skip_count": None,
            "added_to_library_at": None,
            "last_played_at": None,
        },
        "agent_metadata": {"tags": tags or []},
    }


def base_model() -> dict:
    tracks = [
        canonical_track(
            SEED,
            "Seed Song",
            artist_ids=[ARTIST_A],
            genres=["Seed Pop"],
            composer="Seed Composer",
            tags=["dreamy"],
            duration_ms=200_000,
            release_date="2020",
        ),
        canonical_track(
            HIGH,
            "High Similarity",
            artist_ids=[ARTIST_A],
            genres=["Seed Pop"],
            duration_ms=210_000,
            release_date="2021",
        ),
        canonical_track(
            LOW,
            "Low Similarity Favorite",
            artist_ids=[ARTIST_A],
            genres=["Different"],
            duration_ms=500_000,
            release_date="2000",
        ),
        canonical_track(
            UNRELATED,
            "Global Favorite But Unrelated",
            artist_ids=[ARTIST_B],
            genres=["Metal"],
            composer="Other Composer",
            tags=["aggressive"],
            duration_ms=200_000,
            release_date="2020",
        ),
        canonical_track(
            TIE_A,
            "Tie A",
            artist_ids=[ARTIST_B],
            genres=["Seed Pop"],
        ),
        canonical_track(
            TIE_B,
            "Tie B Preferred",
            artist_ids=[ARTIST_B],
            genres=["Seed Pop"],
        ),
        canonical_track(
            OTHER_SEED,
            "Other Seed",
            artist_ids=[ARTIST_B],
            genres=["Metal"],
        ),
    ]
    return {
        "tracks": tracks,
        "artists": [
            {
                "id": ARTIST_A,
                "external_ids": {"apple_music_persistent_id": "ART-A"},
                "name": "Artist A",
            },
            {
                "id": ARTIST_B,
                "external_ids": {"apple_music_persistent_id": "ART-B"},
                "name": "Artist B",
            },
        ],
        "albums": [],
        "playlists": [],
        "playlist_memberships": [],
    }


class TrueSimilarityServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "true-similarity.sqlite3"
        self.service = SharedAgentService(
            self.database,
            clients=AgentClientRegistry({CLIENT: AgentClientPolicy.FULL}),
        )
        model = base_model()
        validate_fixture(model)
        with CanonicalRepository(self.database) as repository:
            repository.save_model(model)

    def tearDown(self) -> None:
        self.service.close()
        self.temp.cleanup()

    def request(
        self,
        *,
        target_ids: list[str] | None = None,
        limit: int = 5,
        request_id: str | None = None,
        extra: dict | None = None,
    ) -> AgentRequest:
        payload = {"target_ids": target_ids or [SEED], "limit": limit}
        payload.update(extra or {})
        from datetime import datetime, timezone

        return AgentRequest(
            request_id or generate_request_id(),
            AgentClientIdentity(CLIENT, "tests"),
            "generate_inferred_recommendation",
            payload,
            datetime(2026, 9, 10, tzinfo=timezone.utc),
        )

    def generate(
        self,
        *,
        seed: str = SEED,
        target_ids: list[str] | None = None,
        limit: int = 5,
        request_id: str | None = None,
        extra: dict | None = None,
        completed_at: str = NOW,
    ):
        return self.service.execute(
            self.request(
                target_ids=target_ids,
                limit=limit,
                request_id=request_id,
                extra=extra,
            ),
            completed_at=completed_at,
            similarity_context=SimilarityExecutionContext(seed),
        )

    def prefer(self, track_id: str, *, negative: bool = False) -> None:
        with PreferencePersistenceRepository(self.database) as repository:
            repository.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                    "apple_music",
                    "disliked" if negative else "favorited",
                ),
                ObservedValue.value(True),
                observed_at=NOW,
                provenance="true_similarity_fixture",
            )

    def targets(self, result) -> list[str]:
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_message)
        return [item["target_id"] for item in result.payload["items"]]

    def test_higher_seed_similarity_outranks_stronger_personal_preference(self) -> None:
        self.prefer(LOW)
        targets = self.targets(self.generate(limit=5))
        self.assertLess(targets.index(HIGH), targets.index(LOW))

    def test_equal_similarity_uses_positive_preference_tiebreak(self) -> None:
        self.prefer(TIE_B)
        targets = self.targets(self.generate(limit=5))
        self.assertLess(targets.index(TIE_B), targets.index(TIE_A))

    def test_global_favorite_cannot_masquerade_as_seed_similarity(self) -> None:
        self.prefer(UNRELATED)
        targets = self.targets(self.generate(limit=5))
        self.assertIn(HIGH, targets)
        self.assertNotIn(UNRELATED, targets)

    def test_negative_preference_is_a_hard_veto(self) -> None:
        self.prefer(HIGH, negative=True)
        self.assertNotIn(HIGH, self.targets(self.generate(limit=5)))

    def test_seed_and_provider_proposed_unrelated_target_are_never_candidates(self) -> None:
        self.prefer(UNRELATED)
        targets = self.targets(
            self.generate(target_ids=[UNRELATED, OTHER_SEED], limit=5)
        )
        self.assertNotIn(SEED, targets)
        self.assertNotIn(UNRELATED, targets)
        observed = self.service.execute(
            self.request(target_ids=[UNRELATED, OTHER_SEED], limit=5),
            completed_at="2026-09-10T00:02:00+00:00",
            similarity_context=SimilarityExecutionContext(SEED),
        )
        # Recent-run exhaustion is allowed, but the public target list still
        # cannot retarget the context: a success must name SEED and a refusal
        # must remain the honest empty result, never an unrelated batch.
        if observed.outcome is AgentToolOutcome.OK:
            self.assertEqual(observed.payload["similarity_seed_canonical_id"], SEED)
        else:
            self.assertEqual(observed.error_code, "empty_recommendation")

    def test_recent_run_exclusion_selects_a_different_second_result(self) -> None:
        first = self.targets(self.generate(limit=1, completed_at=NOW))
        second = self.targets(
            self.generate(
                limit=1,
                completed_at="2026-09-10T00:01:00+00:00",
            )
        )
        self.assertFalse(set(first) & set(second))

    def test_similarity_cross_run_filter_keeps_the_recent_five_run_window(
        self,
    ) -> None:
        history = self.service._recommendation_history
        with patch.object(history, "list_runs", wraps=history.list_runs) as observed:
            self.generate(limit=1, completed_at=NOW)
        self.assertGreaterEqual(observed.call_count, 1)
        self.assertTrue(
            all(call.kwargs == {"limit": 5} for call in observed.call_args_list)
        )

    def test_owner_followup_regression_stays_similarity_driven_without_padding(
        self,
    ) -> None:
        turns = ("找类似这首的", "再找一些类似这首的")
        plans = [resolve_turn_plan(text) for text in turns]
        for plan in plans:
            self.assertEqual(plan.primary, TurnPrimarySemantic.RECOMMENDATION)
            self.assertEqual(plan.recommendation.mode, "similarity_seed")
            self.assertEqual(plan.recommendation.seed_source, "current_track")

        first = self.generate(limit=2, completed_at=NOW)
        second = self.generate(
            limit=2,
            completed_at="2026-09-10T00:01:00+00:00",
        )
        batches = [decode_recommendation_result(result.payload["encoded_result"])
                   for result in (first, second)]
        target_sets = [
            {item.candidate.target.target_id for item in batch.items}
            for batch in batches
        ]
        self.assertTrue(target_sets[0])
        self.assertTrue(target_sets[1])
        self.assertFalse(target_sets[0] & target_sets[1])
        for batch, targets in zip(batches, target_sets):
            self.assertNotIn(SEED, targets)
            self.assertNotIn(UNRELATED, targets)
            self.assertLessEqual(len(targets), 2)
            for item in batch.items:
                self.assertEqual(
                    item.candidate.source.source_path,
                    SIMILARITY_SOURCE_PATH,
                )

    def test_owner_release_siblings_do_not_cross_recent_similarity_runs(
        self,
    ) -> None:
        model = base_model()
        for track_id, name in (
            (HARUJION_ALBUM, "Harujion"),
            (MONSTER_ALBUM, "Monster"),
            (PINK_BLOOD_ALBUM, "PINK BLOOD"),
            (HARUJION_SINGLE, "Harujion"),
            (MONSTER_SINGLE, "Monster"),
            (PINK_BLOOD_SINGLE, "PINK BLOOD"),
        ):
            model["tracks"].append(
                canonical_track(
                    track_id,
                    name,
                    artist_ids=[ARTIST_B],
                    genres=["Seed Pop"],
                    duration_ms=200_000,
                )
            )
        validate_fixture(model)
        with CanonicalRepository(self.database) as repository:
            repository.save_model(model)

        first = self.generate(limit=5, completed_at=NOW)
        second = self.generate(
            limit=5,
            completed_at="2026-09-10T00:01:00+00:00",
        )
        first_result = decode_recommendation_result(
            first.payload["encoded_result"]
        )
        second_result = decode_recommendation_result(
            second.payload["encoded_result"]
        )
        first_ids = {
            item.candidate.target.target_id for item in first_result.items
        }
        second_ids = {
            item.candidate.target.target_id for item in second_result.items
        }
        self.assertTrue(
            {HARUJION_ALBUM, MONSTER_ALBUM, PINK_BLOOD_ALBUM}
            <= first_ids
        )
        self.assertFalse(
            {
                HARUJION_SINGLE,
                MONSTER_SINGLE,
                PINK_BLOOD_SINGLE,
            }
            & second_ids
        )
        self.assertFalse(first_ids & second_ids)
        self.assertNotIn(SEED, first_ids | second_ids)
        self.assertNotIn(UNRELATED, first_ids | second_ids)
        self.assertLess(len(second_ids), 5)
        for result in (first_result, second_result):
            for item in result.items:
                self.assertEqual(
                    item.candidate.source.source_path,
                    SIMILARITY_SOURCE_PATH,
                )

    def test_requested_count_never_pads_with_unrelated_tracks(self) -> None:
        targets = self.targets(self.generate(limit=5))
        self.assertLess(len(targets), 5)
        self.assertNotIn(UNRELATED, targets)

    def test_auxiliary_only_pool_returns_honest_empty_without_a_run(self) -> None:
        model = base_model()
        for existing in model["tracks"]:
            if existing["id"] == SEED:
                continue
            existing["artist_ids"] = [ARTIST_B]
            existing["genres"] = ["Metal"]
            existing["composer"] = "Other Composer"
            existing["agent_metadata"] = {"tags": ["aggressive"]}
            existing["duration_ms"] = 200_000
            existing["release_date"] = "2020"
        validate_fixture(model)
        with CanonicalRepository(self.database) as repository:
            repository.save_model(model)
        empty = self.generate(limit=5)
        self.assertEqual(empty.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(empty.error_code, "empty_recommendation")
        runs = self.service.execute(
            AgentRequest(
                generate_request_id(),
                AgentClientIdentity(CLIENT, "tests"),
                "list_recommendation_runs",
                {},
                empty.completed_at,
            ),
            completed_at=NOW,
        )
        self.assertEqual(runs.payload["runs"], [])

    def test_missing_or_nonexistent_strict_seed_fails_closed(self) -> None:
        unknown = "trk_99999999-9999-4999-8999-999999999999"
        result = self.generate(seed=unknown)
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "similarity_seed_unavailable")

    def test_request_replay_is_seed_bound_and_deterministic(self) -> None:
        request_id = generate_request_id()
        first = self.generate(seed=SEED, limit=2, request_id=request_id)
        replay = self.generate(seed=SEED, limit=2, request_id=request_id)
        conflict = self.generate(seed=OTHER_SEED, limit=2, request_id=request_id)
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.payload, first.payload)
        self.assertEqual(conflict.outcome, AgentToolOutcome.REPLAY_CONFLICT)

    def test_run_lifecycle_source_and_seed_basis_are_preserved(self) -> None:
        result = self.generate(limit=2)
        decoded = decode_recommendation_result(result.payload["encoded_result"])
        self.assertEqual(decoded.request.limit, 2)
        self.assertTrue(decoded.items)
        for item in decoded.items:
            self.assertEqual(item.candidate.source.source_path, SIMILARITY_SOURCE_PATH)
            self.assertEqual(item.candidate.basis_targets[0].target_id, SEED)
            self.assertNotEqual(item.candidate.target.target_id, SEED)
        self.assertEqual(self.service._active_context.active_run_id, decoded.run_id)

    def test_similarity_presentation_is_deterministic_and_metadata_truthful(self) -> None:
        result = self.generate(limit=1)
        rendered = render_recommendation_for_user(result.payload)
        self.assertIsNotNone(rendered)
        self.assertIn("与《Seed Song》共享", rendered)
        self.assertIn("Seed Pop", rendered)
        run = self.service.execute(
            AgentRequest(
                generate_request_id(),
                AgentClientIdentity(CLIENT, "tests"),
                "get_recommendation_run",
                {"run_id": result.payload["run_id"]},
                result.completed_at,
            ),
            completed_at=NOW,
        )
        self.assertEqual(
            run.payload["items"][0]["evidence"],
            result.payload["items"][0]["evidence"],
        )

    def test_public_payload_cannot_supply_seed_authority(self) -> None:
        refused = self.service.execute(
            self.request(extra={"similarity_seed_canonical_id": OTHER_SEED}),
            completed_at=NOW,
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)


if __name__ == "__main__":
    unittest.main()
