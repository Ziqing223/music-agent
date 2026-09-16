"""P11.2: catalog candidate generation against unified preference inputs."""

import unittest
from datetime import datetime, timezone

from music_agent.preference_attribution import (
    InferredAffinity,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
)

from music_agent.catalog_candidate_generation import (
    CATALOG_CANDIDATE_SOURCE_PATH,
    CATALOG_CANDIDATE_SOURCE_SYSTEM,
    REJECTION_NEGATIVE_PREFERENCE,
    CatalogCandidateGenerationValidationError,
    generate_catalog_candidates,
)

NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)

ARTIST_A = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ARTIST_B = "art_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_C1 = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
TRACK_C2 = "trk_c2c2c2c2-c2c2-4c2c-8c2c-c2c2c2c2c2c2"


def inferred_input(kind: PreferenceTargetKind, target_id: str, state: PreferenceState, magnitude: float = 0.8) -> PreferenceInput:
    return PreferenceInput.from_inferred(
        InferredAffinity(
            PreferenceTargetReference(kind, target_id),
            PreferenceStrength(state, magnitude),
        )
    )


def track(track_id: str, *, genres=(), artist_ids=()) -> dict:
    return {"id": track_id, "genres": list(genres), "artist_ids": list(artist_ids)}


class CatalogCandidateGenerationTest(unittest.TestCase):
    def test_positive_artist_affinity_yields_eligible_candidate_with_artist_basis(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.ARTIST, ARTIST_A, PreferenceState.POSITIVE),)
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A])]
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidate.target.target_id, TRACK_C1)
        self.assertEqual(candidate.source.source_system, CATALOG_CANDIDATE_SOURCE_SYSTEM)
        self.assertEqual(candidate.source.source_path, CATALOG_CANDIDATE_SOURCE_PATH)
        self.assertEqual(
            candidate.basis_targets,
            (PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_A),),
        )

    def test_positive_genre_affinity_yields_eligible_candidate_with_genre_basis(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),)
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock",), artist_ids=[])]
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(
            candidates[0].basis_targets[0],
            PreferenceTargetReference(PreferenceTargetKind.GENRE, "Rock"),
        )

    def test_negative_genre_vetoes_candidate(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.NEGATIVE),)
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock",), artist_ids=[])]
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)
        self.assertEqual(candidates[0].rejection.reason, REJECTION_NEGATIVE_PREFERENCE)

    def test_negative_artist_vetoes_candidate(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.ARTIST, ARTIST_A, PreferenceState.NEGATIVE),)
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, artist_ids=[ARTIST_A])]
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)

    def test_any_negative_basis_target_vetoes_even_with_positive_others(self) -> None:
        context = RecommendationContext(
            NOW,
            (
                inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),
                inferred_input(PreferenceTargetKind.GENRE, "Jazz", PreferenceState.NEGATIVE),
            ),
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock", "Jazz"), artist_ids=[])]
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)

    def test_direct_negative_overrides_inferred_positive_on_same_genre(self) -> None:
        context = RecommendationContext(
            NOW,
            (
                PreferenceInput(
                    PreferenceTargetReference(PreferenceTargetKind.GENRE, "Rock"),
                    PreferenceProvenance.DIRECT,
                    PreferenceStrength(PreferenceState.NEGATIVE, 0.7),
                ),
                inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),
            ),
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock",), artist_ids=[])]
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)

    def test_non_directional_inputs_produce_no_candidate(self) -> None:
        context = RecommendationContext(
            NOW,
            (inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.NEUTRAL, 0.0),),
        )
        self.assertEqual(
            generate_catalog_candidates(context, [track(TRACK_C1, genres=("Rock",))]), ()
        )

    def test_no_matching_inputs_produces_no_candidate(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.GENRE, "Jazz", PreferenceState.POSITIVE),)
        )
        self.assertEqual(
            generate_catalog_candidates(context, [track(TRACK_C1, genres=("Rock",))]), ()
        )

    def test_basis_includes_all_matched_targets_in_deterministic_order(self) -> None:
        context = RecommendationContext(
            NOW,
            (
                inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),
                inferred_input(PreferenceTargetKind.ARTIST, ARTIST_A, PreferenceState.POSITIVE),
            ),
        )
        candidates = generate_catalog_candidates(
            context, [track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A])]
        )
        self.assertEqual(
            candidates[0].basis_targets,
            (
                PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_A),
                PreferenceTargetReference(PreferenceTargetKind.GENRE, "Rock"),
            ),
        )

    def test_candidates_emit_in_canonical_target_order(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),)
        )
        candidates = generate_catalog_candidates(
            context,
            [
                track(TRACK_C2, genres=("Rock",)),
                track(TRACK_C1, genres=("Rock",)),
            ],
        )
        self.assertEqual(
            [candidate.target.target_id for candidate in candidates], [TRACK_C1, TRACK_C2]
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        context = RecommendationContext(NOW, ())
        with self.assertRaises(CatalogCandidateGenerationValidationError):
            generate_catalog_candidates(context, [{"id": "not-a-canonical-id"}])
        with self.assertRaises(CatalogCandidateGenerationValidationError):
            generate_catalog_candidates(context, [track(TRACK_C1), track(TRACK_C1)])
        with self.assertRaises(CatalogCandidateGenerationValidationError):
            generate_catalog_candidates("nope", [track(TRACK_C1)])
        with self.assertRaises(CatalogCandidateGenerationValidationError):
            generate_catalog_candidates(
                context, [track(TRACK_C1, artist_ids=["not-canonical"])]
            )

    def test_candidate_identity_is_never_derived_from_target(self) -> None:
        context = RecommendationContext(
            NOW, (inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),)
        )
        first = generate_catalog_candidates(context, [track(TRACK_C1, genres=("Rock",))])
        second = generate_catalog_candidates(context, [track(TRACK_C1, genres=("Rock",))])
        self.assertEqual(first[0].target, second[0].target)
        self.assertNotEqual(first[0].candidate_id, second[0].candidate_id)


if __name__ == "__main__":
    unittest.main()
