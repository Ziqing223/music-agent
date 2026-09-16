"""P07.3: the scoring model -- eligible candidate to ScoreBreakdown (Track B).

These tests prove the pure domain scoring slice only: the fixed component vocabulary, the
documented component-to-total mapping, the direct+inferred combination rule, multi-basis
aggregation, the empty-basis rule, determinism, and the fail-closed behavior. Expected values
are hand-computed from the documented mapping, never re-derived from production logic. No
ranking, no RecommendationItem/RecommendationResult assembly, and nothing touches SQLite.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    Rejection,
)
from music_agent.recommendation_scoring import (
    SCORE_COMPONENT_NAMES,
    RecommendationScoringError,
    RecommendationScoringValidationError,
    score_candidate,
)

TRACK_1 = "trk_11111111-1111-4111-8111-111111111111"
TRACK_2 = "trk_22222222-2222-4222-8222-222222222222"
TRACK_3 = "trk_33333333-3333-4333-8333-333333333333"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def track(target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id)


def preference_input(
    target: PreferenceTargetReference,
    state: PreferenceState,
    magnitude: float | None = None,
    provenance: PreferenceProvenance = PreferenceProvenance.DIRECT,
) -> PreferenceInput:
    strength = PreferenceStrength(state, magnitude)
    if provenance is PreferenceProvenance.DIRECT:
        return PreferenceInput.from_direct(DerivedPreference(target, strength))
    return PreferenceInput.from_inferred(InferredAffinity(target, strength))


def context(*inputs: PreferenceInput) -> RecommendationContext:
    return RecommendationContext(NOW, inputs)


def candidate(*basis: PreferenceTargetReference) -> Candidate:
    return Candidate(
        CANDIDATE_ID,
        track(TRACK_1),
        CandidateSourceReference("candidate_gen", "preference_match"),
        basis,
    )


def assert_breakdown(
    test: unittest.TestCase,
    breakdown,
    *,
    total: float,
    support: float,
    strength: float,
    negative: float,
    directness: float,
) -> None:
    """Assert the fixed vocabulary, the documented mapping, and hand-computed values."""
    test.assertAlmostEqual(breakdown.total, total, places=12)
    test.assertEqual(tuple(component.name for component in breakdown.components), SCORE_COMPONENT_NAMES)
    by_name = {component.name: component.value for component in breakdown.components}
    test.assertAlmostEqual(by_name["basis_support"], support, places=12)
    test.assertAlmostEqual(by_name["basis_strength"], strength, places=12)
    test.assertAlmostEqual(by_name["negative_evidence"], negative, places=12)
    test.assertAlmostEqual(by_name["provenance_directness"], directness, places=12)
    # Documented mapping: total = basis_support * basis_strength.
    test.assertAlmostEqual(breakdown.total, support * strength, places=12)


class ScoreCandidateFailClosedTest(unittest.TestCase):
    def test_rejects_non_candidate(self) -> None:
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate("not-a-candidate", context())  # type: ignore[arg-type]
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(None, context())  # type: ignore[arg-type]

    def test_rejects_non_context(self) -> None:
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(candidate(track(TRACK_1)), "not-a-context")  # type: ignore[arg-type]
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(candidate(track(TRACK_1)), None)  # type: ignore[arg-type]

    def test_rejects_rejected_candidate(self) -> None:
        rejected = Candidate(
            CANDIDATE_ID,
            track(TRACK_1),
            CandidateSourceReference("candidate_gen", "preference_match"),
            (),
            Eligibility.REJECTED,
            Rejection("already_in_library"),
        )
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(rejected, context())

    def test_rejects_missing_basis_input_in_empty_context(self) -> None:
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(candidate(track(TRACK_1)), context())

    def test_rejects_missing_basis_input_when_other_targets_are_present(self) -> None:
        ctx = context(preference_input(track(TRACK_2), PreferenceState.POSITIVE, 0.9))
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(candidate(track(TRACK_1)), ctx)

    def test_rejects_missing_input_for_one_of_several_basis_targets(self) -> None:
        ctx = context(preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9))
        with self.assertRaises(RecommendationScoringValidationError):
            score_candidate(candidate(track(TRACK_1), track(TRACK_2)), ctx)

    def test_errors_are_module_errors_with_documented_codes(self) -> None:
        with self.assertRaises(RecommendationScoringError) as caught:
            score_candidate("not-a-candidate", context())  # type: ignore[arg-type]
        self.assertEqual(caught.exception.code, "validation_error")
        self.assertEqual(RecommendationScoringValidationError.code, "validation_error")
        self.assertEqual(RecommendationScoringError.code, "recommendation_scoring_error")


class SingleBasisTargetTest(unittest.TestCase):
    def test_direct_positive(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9)),
        )
        assert_breakdown(self, breakdown, total=0.9, support=1.0, strength=0.9, negative=0.0, directness=1.0)

    def test_inferred_positive(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.7, PreferenceProvenance.INFERRED
                )
            ),
        )
        assert_breakdown(self, breakdown, total=0.7, support=1.0, strength=0.7, negative=0.0, directness=0.0)

    def test_negative(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.NEGATIVE, 1.0)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=1.0, directness=1.0)

    def test_neutral(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.NEUTRAL, 0.0)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)

    def test_unknown(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.UNKNOWN)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)

    def test_insufficient(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.INSUFFICIENT)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)

    def test_conflict(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(preference_input(track(TRACK_1), PreferenceState.CONFLICT)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)


class DirectInferredCombinationTest(unittest.TestCase):
    def test_direct_governs_a_formed_conclusion(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.5, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.9, support=1.0, strength=0.9, negative=0.0, directness=1.0)

    def test_inferred_fills_gap_after_unknown(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.UNKNOWN),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.7, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.7, support=1.0, strength=0.7, negative=0.0, directness=0.0)

    def test_inferred_fills_gap_after_insufficient(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.INSUFFICIENT),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.6, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.6, support=1.0, strength=0.6, negative=0.0, directness=0.0)

    def test_formed_neutral_conclusion_blocks_inferred_override(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.NEUTRAL, 0.0),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.9, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)

    def test_formed_negative_conclusion_blocks_inferred_override(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.NEGATIVE, 1.0),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.9, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=1.0, directness=1.0)

    def test_formed_conflict_conclusion_blocks_inferred_override(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1)),
            context(
                preference_input(track(TRACK_1), PreferenceState.CONFLICT),
                preference_input(
                    track(TRACK_1), PreferenceState.POSITIVE, 0.9, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=1.0)

    def test_combination_is_invariant_to_input_order(self) -> None:
        inputs = [
            preference_input(track(TRACK_1), PreferenceState.UNKNOWN),
            preference_input(
                track(TRACK_1), PreferenceState.POSITIVE, 0.7, PreferenceProvenance.INFERRED
            ),
        ]
        forward = score_candidate(candidate(track(TRACK_1)), context(*inputs))
        reversed_ = score_candidate(candidate(track(TRACK_1)), context(*reversed(inputs)))
        self.assertEqual(forward, reversed_)


class MultiBasisAggregationTest(unittest.TestCase):
    def test_two_positive_targets_average_magnitudes(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 1.0),
                preference_input(track(TRACK_2), PreferenceState.POSITIVE, 0.5),
            ),
        )
        assert_breakdown(self, breakdown, total=0.75, support=1.0, strength=0.75, negative=0.0, directness=1.0)

    def test_positive_and_negative_split_support(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 1.0),
                preference_input(track(TRACK_2), PreferenceState.NEGATIVE, 1.0),
            ),
        )
        assert_breakdown(self, breakdown, total=0.5, support=0.5, strength=1.0, negative=0.5, directness=1.0)

    def test_mixed_provenance_halves_directness(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.8),
                preference_input(
                    track(TRACK_2), PreferenceState.POSITIVE, 0.4, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.6, support=1.0, strength=0.6, negative=0.0, directness=0.5)

    def test_unknown_target_lowers_support(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 1.0),
                preference_input(track(TRACK_2), PreferenceState.UNKNOWN),
            ),
        )
        assert_breakdown(self, breakdown, total=0.5, support=0.5, strength=1.0, negative=0.0, directness=1.0)

    def test_three_targets_hand_computed(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2), track(TRACK_3)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
                preference_input(track(TRACK_2), PreferenceState.POSITIVE, 0.6),
                preference_input(track(TRACK_3), PreferenceState.NEGATIVE, 0.8),
            ),
        )
        # support = 2/3, strength = (0.9 + 0.6) / 2 = 0.75, total = 0.5,
        # negative = 1/3, directness = 1.0.
        assert_breakdown(
            self,
            breakdown,
            total=0.5,
            support=2 / 3,
            strength=0.75,
            negative=1 / 3,
            directness=1.0,
        )

    def test_fallback_operative_input_is_aggregated_as_inferred(self) -> None:
        breakdown = score_candidate(
            candidate(track(TRACK_1), track(TRACK_2)),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.6),
                preference_input(track(TRACK_2), PreferenceState.UNKNOWN),
                preference_input(
                    track(TRACK_2), PreferenceState.POSITIVE, 0.3, PreferenceProvenance.INFERRED
                ),
            ),
        )
        assert_breakdown(self, breakdown, total=0.45, support=1.0, strength=0.45, negative=0.0, directness=0.5)

    def test_basis_order_does_not_change_the_breakdown(self) -> None:
        inputs = (
            preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
            preference_input(track(TRACK_2), PreferenceState.NEGATIVE, 1.0),
        )
        forward = score_candidate(candidate(track(TRACK_1), track(TRACK_2)), context(*inputs))
        reversed_ = score_candidate(candidate(track(TRACK_2), track(TRACK_1)), context(*inputs))
        self.assertEqual(forward, reversed_)

    def test_unrelated_context_inputs_do_not_affect_the_breakdown(self) -> None:
        basis = (track(TRACK_1), track(TRACK_2))
        inputs = (
            preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
            preference_input(track(TRACK_2), PreferenceState.POSITIVE, 0.5),
        )
        bare = score_candidate(candidate(*basis), context(*inputs))
        with_extra = score_candidate(
            candidate(*basis),
            context(*inputs, preference_input(track(TRACK_3), PreferenceState.POSITIVE, 0.2)),
        )
        self.assertEqual(bare, with_extra)


class EmptyBasisTest(unittest.TestCase):
    def test_empty_basis_scores_zero_with_fixed_vocabulary(self) -> None:
        breakdown = score_candidate(candidate(), context())
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=0.0)

    def test_empty_basis_ignores_context_inputs(self) -> None:
        breakdown = score_candidate(
            candidate(),
            context(preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9)),
        )
        assert_breakdown(self, breakdown, total=0.0, support=0.0, strength=0.0, negative=0.0, directness=0.0)


class VocabularyAndBoundsTest(unittest.TestCase):
    def test_fixed_vocabulary_is_non_empty_name_unique(self) -> None:
        self.assertGreaterEqual(len(SCORE_COMPONENT_NAMES), 1)
        self.assertEqual(len(set(SCORE_COMPONENT_NAMES)), len(SCORE_COMPONENT_NAMES))
        for name in SCORE_COMPONENT_NAMES:
            self.assertIsInstance(name, str)
            self.assertNotEqual(name, "")

    def test_every_state_and_provenance_yields_bounded_values(self) -> None:
        for state in PreferenceState:
            for provenance in PreferenceProvenance:
                magnitude = None
                if state is PreferenceState.POSITIVE or state is PreferenceState.NEGATIVE:
                    magnitude = 0.8
                elif state is PreferenceState.NEUTRAL:
                    magnitude = 0.0
                breakdown = score_candidate(
                    candidate(track(TRACK_1)),
                    context(preference_input(track(TRACK_1), state, magnitude, provenance)),
                )
                self._assert_bounded(breakdown)

    def test_multi_target_combinations_yield_bounded_values(self) -> None:
        basis = (track(TRACK_1), track(TRACK_2), track(TRACK_3))
        contexts = (
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 1.0),
                preference_input(track(TRACK_2), PreferenceState.POSITIVE, 0.5),
                preference_input(
                    track(TRACK_3), PreferenceState.POSITIVE, 0.25, PreferenceProvenance.INFERRED
                ),
            ),
            context(
                preference_input(track(TRACK_1), PreferenceState.NEGATIVE, 1.0),
                preference_input(track(TRACK_2), PreferenceState.UNKNOWN),
                preference_input(track(TRACK_3), PreferenceState.NEUTRAL, 0.0),
            ),
            context(
                preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
                preference_input(
                    track(TRACK_2), PreferenceState.POSITIVE, 0.6, PreferenceProvenance.INFERRED
                ),
                preference_input(track(TRACK_3), PreferenceState.INSUFFICIENT),
            ),
        )
        for ctx in contexts:
            self._assert_bounded(score_candidate(candidate(*basis), ctx))

    def _assert_bounded(self, breakdown) -> None:
        self.assertTrue(math.isfinite(breakdown.total))
        self.assertGreaterEqual(breakdown.total, 0.0)
        self.assertLessEqual(breakdown.total, 1.0)
        self.assertGreaterEqual(len(breakdown.components), 1)
        names = [component.name for component in breakdown.components]
        self.assertEqual(len(names), len(set(names)))
        for component in breakdown.components:
            self.assertTrue(math.isfinite(component.value))
            self.assertGreaterEqual(component.value, 0.0)
            self.assertLessEqual(component.value, 1.0)


class DeterminismTest(unittest.TestCase):
    def test_same_inputs_produce_identical_breakdown(self) -> None:
        cand = candidate(track(TRACK_1), track(TRACK_2))
        ctx = context(
            preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
            preference_input(
                track(TRACK_2), PreferenceState.POSITIVE, 0.5, PreferenceProvenance.INFERRED
            ),
        )
        self.assertEqual(score_candidate(cand, ctx), score_candidate(cand, ctx))

    def test_context_input_order_does_not_change_the_breakdown(self) -> None:
        inputs = [
            preference_input(track(TRACK_1), PreferenceState.POSITIVE, 0.9),
            preference_input(
                track(TRACK_2), PreferenceState.POSITIVE, 0.5, PreferenceProvenance.INFERRED
            ),
        ]
        cand = candidate(track(TRACK_1), track(TRACK_2))
        forward = score_candidate(cand, context(*inputs))
        reversed_ = score_candidate(cand, context(*reversed(inputs)))
        self.assertEqual(forward, reversed_)


if __name__ == "__main__":
    unittest.main()
