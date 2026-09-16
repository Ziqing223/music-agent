"""True Similarity V1 pure metadata contract tests."""

from __future__ import annotations

import copy
import math
import unittest

from music_agent.track_similarity import (
    ARTIST_WEIGHT,
    COMPOSER_WEIGHT,
    DURATION_WEIGHT,
    GENRE_WEIGHT,
    RELEASE_YEAR_WEIGHT,
    TAG_WEIGHT,
    SimilarityExecutionContext,
    TrackSimilarityValidationError,
    score_track_similarity,
)


SEED = "trk_11111111-1111-4111-8111-111111111111"
CANDIDATE = "trk_22222222-2222-4222-8222-222222222222"
ARTIST_A = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ARTIST_B = "art_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def track(track_id: str, **overrides: object) -> dict:
    value = {
        "id": track_id,
        "artist_ids": [],
        "genres": [],
        "composer": None,
        "duration_ms": None,
        "release_date": None,
        "agent_metadata": {"tags": []},
    }
    value.update(overrides)
    return value


class TrackSimilarityTest(unittest.TestCase):
    def test_v1_weights_remain_frozen(self) -> None:
        self.assertEqual(
            (
                GENRE_WEIGHT,
                ARTIST_WEIGHT,
                COMPOSER_WEIGHT,
                TAG_WEIGHT,
                DURATION_WEIGHT,
                RELEASE_YEAR_WEIGHT,
            ),
            (0.50, 0.20, 0.10, 0.10, 0.05, 0.05),
        )

    def test_identical_categorical_metadata_scores_high(self) -> None:
        seed = track(
            SEED,
            artist_ids=[ARTIST_A],
            genres=["Pop"],
            composer="Composer",
            duration_ms=200_000,
            release_date="2024-01-01",
            agent_metadata={"tags": ["bright"]},
        )
        result = score_track_similarity(seed, track(CANDIDATE, **{k: v for k, v in seed.items() if k != "id"}))
        self.assertTrue(result.categorical_eligible)
        self.assertTrue(math.isclose(result.total, 1.0))

    def test_each_categorical_signal_alone_establishes_eligibility(self) -> None:
        cases = (
            ({"artist_ids": [ARTIST_A]}, {"artist_ids": [ARTIST_A]}, ARTIST_WEIGHT),
            ({"genres": ["Pop"]}, {"genres": ["Pop"]}, GENRE_WEIGHT),
            ({"composer": "Example Composer"}, {"composer": " example   composer "}, COMPOSER_WEIGHT),
            (
                {"agent_metadata": {"tags": ["Dreamy"]}},
                {"agent_metadata": {"tags": ["dreamy"]}},
                TAG_WEIGHT,
            ),
        )
        for seed_fields, candidate_fields, expected in cases:
            with self.subTest(seed_fields=seed_fields):
                result = score_track_similarity(
                    track(SEED, **seed_fields), track(CANDIDATE, **candidate_fields)
                )
                self.assertTrue(result.categorical_eligible)
                self.assertTrue(math.isclose(result.total, expected))

    def test_auxiliary_signals_never_establish_eligibility(self) -> None:
        seed = track(SEED, duration_ms=180_000, release_date="2024")
        cases = (
            track(CANDIDATE, duration_ms=180_000),
            track(CANDIDATE, release_date="2024"),
            track(CANDIDATE, duration_ms=180_000, release_date="2024"),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate):
                result = score_track_similarity(seed, candidate)
                self.assertFalse(result.categorical_eligible)

    def test_no_shared_evidence_is_ineligible(self) -> None:
        result = score_track_similarity(
            track(SEED, artist_ids=[ARTIST_A], genres=["Pop"]),
            track(CANDIDATE, artist_ids=[ARTIST_B], genres=["Rock"]),
        )
        self.assertFalse(result.categorical_eligible)
        self.assertEqual(result.total, 0.0)

    def test_missing_fields_contribute_zero_without_renormalization(self) -> None:
        result = score_track_similarity(
            {"id": SEED, "genres": ["Pop"]},
            {"id": CANDIDATE, "genres": ["Pop"]},
        )
        self.assertEqual(result.genre_overlap, 1.0)
        self.assertTrue(math.isclose(result.total, GENRE_WEIGHT))
        self.assertEqual(result.artist_overlap, 0.0)
        self.assertEqual(result.duration_proximity, 0.0)

    def test_genre_and_artist_jaccard_are_exact(self) -> None:
        result = score_track_similarity(
            track(SEED, genres=["Pop", "Rock"], artist_ids=[ARTIST_A, ARTIST_B]),
            track(CANDIDATE, genres=["Pop", "Jazz"], artist_ids=[ARTIST_A]),
        )
        self.assertTrue(math.isclose(result.genre_overlap, 1 / 3))
        self.assertTrue(math.isclose(result.artist_overlap, 1 / 2))
        self.assertEqual(result.shared_genres, ("Pop",))
        self.assertEqual(result.shared_artist_ids, (ARTIST_A,))

    def test_composer_is_normalized_exact_not_fuzzy(self) -> None:
        exact = score_track_similarity(
            track(SEED, composer="Example Composer"),
            track(CANDIDATE, composer=" example   COMPOSER "),
        )
        fuzzy = score_track_similarity(
            track(SEED, composer="Example Composer"),
            track(CANDIDATE, composer="Example Composer Jr."),
        )
        self.assertEqual(exact.composer_overlap, 1.0)
        self.assertEqual(fuzzy.composer_overlap, 0.0)

    def test_curated_tags_use_set_overlap_only(self) -> None:
        result = score_track_similarity(
            track(SEED, agent_metadata={"tags": ["Dreamy", "Night"]}),
            track(CANDIDATE, agent_metadata={"tags": ["dreamy", "Live"]}),
        )
        self.assertTrue(math.isclose(result.tag_overlap, 1 / 3))
        self.assertEqual(result.shared_tags, ("dreamy",))

    def test_duration_proximity_boundaries(self) -> None:
        seed = track(SEED, genres=["Pop"], duration_ms=200_000)
        same = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], duration_ms=200_000))
        halfway = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], duration_ms=290_000))
        boundary = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], duration_ms=380_000))
        self.assertEqual(same.duration_proximity, 1.0)
        self.assertEqual(halfway.duration_proximity, 0.5)
        self.assertEqual(boundary.duration_proximity, 0.0)

    def test_release_year_proximity_boundaries(self) -> None:
        seed = track(SEED, genres=["Pop"], release_date="2020-01-01")
        same = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], release_date="2020"))
        halfway = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], release_date="2025-02"))
        boundary = score_track_similarity(seed, track(CANDIDATE, genres=["Pop"], release_date="2030"))
        self.assertEqual(same.release_year_proximity, 1.0)
        self.assertEqual(halfway.release_year_proximity, 0.5)
        self.assertEqual(boundary.release_year_proximity, 0.0)

    def test_component_sum_matches_total(self) -> None:
        result = score_track_similarity(
            track(SEED, genres=["Pop"], duration_ms=200_000, release_date="2024"),
            track(CANDIDATE, genres=["Pop"], duration_ms=245_000, release_date="2027"),
        )
        component_sum = sum(component.value for component in result.score_breakdown.components)
        self.assertTrue(math.isclose(component_sum, result.total))
        self.assertTrue(math.isclose(result.total, GENRE_WEIGHT + DURATION_WEIGHT * 0.75 + RELEASE_YEAR_WEIGHT * 0.7))

    def test_scorer_is_deterministic_order_independent_and_non_mutating(self) -> None:
        seed = track(SEED, genres=["Rock", "Pop"], artist_ids=[ARTIST_B, ARTIST_A])
        candidate = track(CANDIDATE, genres=["Pop", "Rock"], artist_ids=[ARTIST_A, ARTIST_B])
        before_seed = copy.deepcopy(seed)
        before_candidate = copy.deepcopy(candidate)
        first = score_track_similarity(seed, candidate)
        second = score_track_similarity(seed, candidate)
        self.assertEqual(first, second)
        self.assertEqual(seed, before_seed)
        self.assertEqual(candidate, before_candidate)

    def test_execution_context_is_immutable_and_validates_track_identity(self) -> None:
        context = SimilarityExecutionContext(SEED)
        self.assertEqual(context.seed_canonical_id, SEED)
        with self.assertRaises(AttributeError):
            context.seed_canonical_id = CANDIDATE  # type: ignore[misc]
        with self.assertRaises(TrackSimilarityValidationError):
            SimilarityExecutionContext("not-a-track")


if __name__ == "__main__":
    unittest.main()
