"""P15-S3-S3E: fresh-driven candidate generation (the explicit Fresh-intent channel).

The module contract under test: only authoritative Fresh IDs are considered,
the pool is read-only, an eligible candidate carries an EMPTY basis (no
preference claim), any NEGATIVE operative conclusion on the track's resolvable
surface (Track / Artist / Genre) fails closed to REJECTED
(``negative_preference``), every non-NEGATIVE state (including UNKNOWN,
NEUTRAL, CONFLICT, and total absence of evidence) stays eligible, and wrong
argument shapes fail closed with validation errors. Determinism: candidates
emit in canonical-target order.
"""

import unittest
from datetime import datetime, timezone

from music_agent.fresh_candidate_generation import (
    FRESH_CANDIDATE_SOURCE_PATH,
    FRESH_CANDIDATE_SOURCE_SYSTEM,
    REJECTION_NEGATIVE_PREFERENCE,
    FreshCandidateGenerationValidationError,
    generate_fresh_candidates,
)
from music_agent.preference_attribution import (
    InferredAffinity,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Eligibility,
    PreferenceInput,
    RecommendationContext,
)

NOW = datetime(2026, 8, 19, 0, 0, 0, tzinfo=timezone.utc)

ARTIST_Z = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
ARTIST_OTHER = "art_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
TRACK_Z1 = "trk_ffffffff-ffff-4fff-8fff-000000000004"
TRACK_Z2 = "trk_ffffffff-ffff-4fff-8fff-000000000005"


def direct_input(kind: PreferenceTargetKind, target_id: str, state: PreferenceState, magnitude: float | None = 0.8) -> PreferenceInput:
    return PreferenceInput(
        PreferenceTargetReference(kind, target_id),
        PreferenceProvenance.DIRECT,
        PreferenceStrength(state, magnitude),
    )


def inferred_input(kind: PreferenceTargetKind, target_id: str, state: PreferenceState, magnitude: float = 0.8) -> PreferenceInput:
    return PreferenceInput.from_inferred(
        InferredAffinity(
            PreferenceTargetReference(kind, target_id),
            PreferenceStrength(state, magnitude),
        )
    )


def track(track_id: str, *, genres=(), artist_ids=()) -> dict:
    return {"id": track_id, "genres": list(genres), "artist_ids": list(artist_ids)}


class FreshCandidateGenerationTest(unittest.TestCase):
    def _candidates(self, fresh_ids, pool, inputs=()):
        return generate_fresh_candidates(
            RecommendationContext(NOW, tuple(inputs)), fresh_ids, pool
        )

    # --- eligibility: the zero-affinity contract (items 7-10) -----------------

    def test_zero_affinity_context_yields_eligible_empty_basis_candidate(self) -> None:
        """The exact live-failure shape: no preference evidence anywhere -- the
        candidate is ELIGIBLE with an EMPTY basis (zero preference claim), not
        omitted like the frozen catalog layer would omit it."""
        candidates = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidate.target.target_id, TRACK_Z1)
        self.assertEqual(
            candidate.target.kind, PreferenceTargetKind.TRACK
        )
        self.assertEqual(candidate.source.source_system, FRESH_CANDIDATE_SOURCE_SYSTEM)
        self.assertEqual(candidate.source.source_path, FRESH_CANDIDATE_SOURCE_PATH)
        self.assertEqual(candidate.basis_targets, ())

    def test_positive_evidence_elsewhere_never_becomes_a_basis_claim(self) -> None:
        """Positives on OTHER targets must not leak into the fresh candidate --
        zero fabrication: the basis stays empty no matter the context."""
        candidates = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
            inputs=(
                inferred_input(PreferenceTargetKind.GENRE, "Rock", PreferenceState.POSITIVE),
                inferred_input(PreferenceTargetKind.ARTIST, ARTIST_OTHER, PreferenceState.POSITIVE),
            ),
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidates[0].basis_targets, ())

    def test_non_negative_states_do_not_veto(self) -> None:
        """Freshness exploration exists precisely because evidence is absent:
        UNKNOWN / NEUTRAL / CONFLICT (and absence) never veto."""
        magnitudes = {
            PreferenceState.UNKNOWN: None,
            PreferenceState.NEUTRAL: 0.0,
            PreferenceState.CONFLICT: None,
        }
        for state in (PreferenceState.UNKNOWN, PreferenceState.NEUTRAL, PreferenceState.CONFLICT):
            with self.subTest(state=state):
                candidates = self._candidates(
                    [TRACK_Z1],
                    [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
                    inputs=(
                        direct_input(
                            PreferenceTargetKind.GENRE,
                            "Dark Ambient",
                            state,
                            magnitudes[state],
                        ),
                    ),
                )
                self.assertEqual(candidates[0].eligibility, Eligibility.ELIGIBLE)

    # --- negative veto: fail closed on every surface (items 11-14) ------------

    def test_negative_track_evidence_vetoes(self) -> None:
        candidates = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
            inputs=(direct_input(PreferenceTargetKind.TRACK, TRACK_Z1, PreferenceState.NEGATIVE),),
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)
        self.assertEqual(candidates[0].rejection.reason, REJECTION_NEGATIVE_PREFERENCE)
        self.assertEqual(candidates[0].basis_targets, ())

    def test_negative_artist_evidence_vetoes(self) -> None:
        candidates = self._candidates(
            [TRACK_Z2],
            [track(TRACK_Z2, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
            inputs=(direct_input(PreferenceTargetKind.ARTIST, ARTIST_Z, PreferenceState.NEGATIVE),),
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)

    def test_negative_genre_evidence_vetoes(self) -> None:
        candidates = self._candidates(
            [TRACK_Z2],
            [track(TRACK_Z2, genres=("Dark Ambient",), artist_ids=[])],
            inputs=(inferred_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.NEGATIVE),),
        )
        self.assertEqual(candidates[0].eligibility, Eligibility.REJECTED)

    def test_operative_fallback_rules_are_the_frozen_p06_shape(self) -> None:
        """The veto scan reuses the operative rule: a direct UNKNOWN/
        INSUFFICIENT gap falls back to the inferred conclusion; a formed direct
        conclusion governs; non-negatives never veto."""
        # UNKNOWN direct + NEGATIVE inferred -> inferred fills the gap: veto.
        vetoed = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[])],
            inputs=(
                direct_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.UNKNOWN, None),
                inferred_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.NEGATIVE),
            ),
        )
        self.assertEqual(vetoed[0].eligibility, Eligibility.REJECTED)
        # NEGATIVE direct + POSITIVE inferred -> the direct conclusion governs.
        governed = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[])],
            inputs=(
                direct_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.NEGATIVE),
                inferred_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.POSITIVE),
            ),
        )
        self.assertEqual(governed[0].eligibility, Eligibility.REJECTED)
        # UNKNOWN direct + POSITIVE inferred -> positive, never a veto.
        eligible = self._candidates(
            [TRACK_Z1],
            [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[])],
            inputs=(
                direct_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.UNKNOWN, None),
                inferred_input(PreferenceTargetKind.GENRE, "Dark Ambient", PreferenceState.POSITIVE),
            ),
        )
        self.assertEqual(eligible[0].eligibility, Eligibility.ELIGIBLE)

    # --- identity gate + no fabrication (items 15-16) -------------------------

    def test_not_in_pool_fresh_id_contributes_nothing(self) -> None:
        """A fresh ID absent from the direction-filtered pool is skipped
        silently -- the direction filter stays a hard filter for fresh too."""
        self.assertEqual(
            self._candidates(
                [TRACK_Z1],
                [track(TRACK_Z2, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
            ),
            (),
        )

    def test_pool_tracks_outside_the_fresh_set_are_never_generated(self) -> None:
        """No fabrication: only the supplied authoritative set is iterated; a
        pool track whose id is not fresh contributes nothing on its own."""
        candidates = self._candidates(
            [TRACK_Z1],
            [
                track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z]),
                track(TRACK_Z2, genres=("Dark Ambient",), artist_ids=[ARTIST_Z]),
            ],
        )
        self.assertEqual(
            [candidate.target.target_id for candidate in candidates], [TRACK_Z1]
        )

    def test_candidates_emit_in_canonical_target_order(self) -> None:
        candidates = self._candidates(
            [TRACK_Z2, TRACK_Z1],
            [
                track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z]),
                track(TRACK_Z2, genres=("Dark Ambient",), artist_ids=[ARTIST_Z]),
            ],
        )
        self.assertEqual(
            [candidate.target.target_id for candidate in candidates],
            [TRACK_Z1, TRACK_Z2],
        )

    def test_empty_fresh_set_yields_no_candidates(self) -> None:
        self.assertEqual(
            self._candidates(
                [],
                [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])],
            ),
            (),
        )

    def test_candidate_identity_is_never_derived_from_target(self) -> None:
        first = self._candidates(
            [TRACK_Z1], [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])]
        )
        second = self._candidates(
            [TRACK_Z1], [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])]
        )
        self.assertEqual(first[0].target, second[0].target)
        self.assertNotEqual(first[0].candidate_id, second[0].candidate_id)

    # --- validation fail-closed (item 17) -------------------------------------

    def test_invalid_inputs_fail_closed(self) -> None:
        context = RecommendationContext(NOW, ())
        pool = [track(TRACK_Z1, genres=("Dark Ambient",), artist_ids=[ARTIST_Z])]
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates("nope", [TRACK_Z1], pool)
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(context, "trk-not-an-iterable", pool)
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(context, ["not-canonical"], pool)
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(context, [""], pool)
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(context, [TRACK_Z1], [{"genres": []}])
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(context, [TRACK_Z1], [TRACK_Z1])  # not mappings
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(
                context, [TRACK_Z1], [pool[0], dict(pool[0])]  # duplicate pool id
            )
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(
                context, [TRACK_Z1], [track(TRACK_Z1, genres=(1,))]  # non-str genre
            )
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(
                context, [TRACK_Z1], [track(TRACK_Z1, artist_ids="not-a-sequence")]
            )
        with self.assertRaises(FreshCandidateGenerationValidationError):
            generate_fresh_candidates(
                context, [TRACK_Z1], [track(TRACK_Z1, artist_ids=["not-canonical"])]
            )


if __name__ == "__main__":
    unittest.main()