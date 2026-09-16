"""P07.4: production ranking and recommendation chain completion.

These tests prove the ranking slice: the strict total order (score descending, candidate_id
tie-break), request.limit truncation, kind enforcement, contract assembly through the single
boundary, injected run_id/produced_at determinism, the end-to-end chain built by
build_recommendation (candidates -> scores -> rank -> result), and the optional P07.6 quality
policy hook. Nothing touches SQLite or the clock.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    RECOMMENDATION_CONTRACT_VERSION,
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
)
from music_agent.recommendation_quality import (
    CONTROL_REPEAT_RECOMMENDATION,
    REASON_PREVIOUSLY_RECOMMENDED,
    QualityEvidence,
)
from music_agent.recommendation_ranking import (
    RankingOutcome,
    RecommendationRankingError,
    RecommendationRankingValidationError,
    build_recommendation,
    rank_items,
    rank_recommendations,
)

TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_22222222-2222-4222-8222-222222222222"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
ARTIST_X = "art_44444444-4444-4444-8444-444444444444"
CANDIDATE_A = "cnd_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CANDIDATE_B = "cnd_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
CANDIDATE_C = "cnd_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
RUN_ID = "rcm_55555555-5555-4555-8555-555555555555"
NOW = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)
PRODUCED_AT = datetime(2026, 8, 16, 10, 5, 0, tzinfo=timezone.utc)


def _target(target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id)


def _item(candidate_id: str, target_id: str, total: float) -> RecommendationItem:
    candidate = Candidate(
        candidate_id,
        _target(target_id),
        CandidateSourceReference("music_agent", "preference_driven"),
        basis_targets=(_target(target_id),),
        eligibility=Eligibility.ELIGIBLE,
    )
    return RecommendationItem(
        candidate,
        ScoreBreakdown(total, (ScoreComponent("basis_support", total),)),
    )


def _request(*, limit: int = 3, kind: RecommendedItemKind = RecommendedItemKind.TRACK) -> RecommendationRequest:
    return RecommendationRequest(RecommendationContext(NOW, ()), kind, limit)


def _context_with_preferences() -> RecommendationContext:
    return RecommendationContext(
        NOW,
        (
            PreferenceInput.from_direct(
                DerivedPreference(_target(TRACK_A), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
            ),
            PreferenceInput.from_direct(
                DerivedPreference(_target(TRACK_B), PreferenceStrength(PreferenceState.POSITIVE, 0.6))
            ),
            PreferenceInput.from_direct(
                DerivedPreference(_target(TRACK_C), PreferenceStrength(PreferenceState.NEGATIVE, 0.8))
            ),
            PreferenceInput.from_inferred(
                InferredAffinity(
                    PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_X),
                    PreferenceStrength(PreferenceState.POSITIVE, 0.5),
                )
            ),
        ),
    )


class RankItemsTest(unittest.TestCase):
    def test_orders_by_total_descending(self) -> None:
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.9)]
        )
        self.assertEqual([item.candidate.candidate_id for item in ranked], [CANDIDATE_A, CANDIDATE_B])

    def test_equal_totals_break_ties_by_target_id_ascending(self) -> None:
        # P10 reproducibility amendment: the tie-break is the canonical target id,
        # never the per-run candidate id.
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)]
        )
        self.assertEqual([item.candidate.target.target_id for item in ranked], [TRACK_A, TRACK_B])

    # --- P19-T15: recommendation-count novelty tie-break --------------------

    def test_equal_totals_break_ties_by_recommendation_count_ascending_then_target_id(self) -> None:
        # Score ties order by persisted recommendation_count ascending before the
        # target-id fallback: the never-recommended B (count 0) surfaces ahead of
        # the heavily repeated A (count 3) even though A's target id is smaller.
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)],
            recommendation_counts={TRACK_A: 3, TRACK_B: 0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked], [TRACK_B, TRACK_A]
        )

    def test_equal_counts_fall_back_to_target_id_ascending(self) -> None:
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)],
            recommendation_counts={TRACK_A: 2, TRACK_B: 2},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked], [TRACK_A, TRACK_B]
        )

    def test_higher_score_always_beats_lower_score_regardless_of_rec_count(self) -> None:
        # Relevance is primary: a score-0.9 item with the heaviest possible
        # recommendation history outranks a never-recommended score-0.4 item.
        ranked = rank_items(
            [_item(CANDIDATE_A, TRACK_A, 0.9), _item(CANDIDATE_B, TRACK_B, 0.4)],
            recommendation_counts={TRACK_A: 999, TRACK_B: 0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked], [TRACK_A, TRACK_B]
        )

    def test_unknown_target_counts_as_never_recommended(self) -> None:
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)],
            recommendation_counts={TRACK_A: 10},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked], [TRACK_B, TRACK_A]
        )

    def test_counts_preserve_determinism_on_identical_inputs(self) -> None:
        counts = {TRACK_A: 4, TRACK_B: 1}
        first = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)],
            recommendation_counts=counts,
        )
        second = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)],
            recommendation_counts=dict(counts),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [item.candidate.target.target_id for item in first], [TRACK_B, TRACK_A]
        )

    def test_omitting_counts_keeps_the_pre_t15_order(self) -> None:
        ranked = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)]
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked], [TRACK_A, TRACK_B]
        )

    def test_invalid_recommendation_counts_fail_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items([], recommendation_counts=["not-a-mapping"])  # type: ignore[arg-type]
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items([], recommendation_counts={TRACK_A: "three"})  # type: ignore[dict-item]
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items([], recommendation_counts={123: 1})  # type: ignore[dict-item]

    def test_preference_tiebreak_applies_only_after_equal_primary_score(self) -> None:
        ranked = rank_items(
            [_item(CANDIDATE_A, TRACK_A, 0.8), _item(CANDIDATE_B, TRACK_B, 0.7)],
            preference_tiebreaks={TRACK_A: 0.0, TRACK_B: 1.0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in ranked],
            [TRACK_A, TRACK_B],
        )
        tied = rank_items(
            [_item(CANDIDATE_A, TRACK_A, 0.8), _item(CANDIDATE_B, TRACK_B, 0.8)],
            preference_tiebreaks={TRACK_A: 0.0, TRACK_B: 1.0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in tied],
            [TRACK_B, TRACK_A],
        )

    def test_invalid_preference_tiebreaks_fail_closed(self) -> None:
        for invalid in (
            ["not-a-mapping"],
            {TRACK_A: -0.1},
            {TRACK_A: 1.1},
            {TRACK_A: True},
            {123: 0.5},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RecommendationRankingValidationError):
                    rank_items([], preference_tiebreaks=invalid)  # type: ignore[arg-type]

    def test_candidate_id_never_influences_ranking(self) -> None:
        # Identical inputs with DIFFERENT candidate ids must produce the same target order.
        first = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.5), _item(CANDIDATE_A, TRACK_A, 0.5)]
        )
        other_a = "cnd_ffffffff-ffff-4fff-8fff-ffffffffffff"
        other_b = "cnd_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
        second = rank_items(
            [_item(other_a, TRACK_A, 0.5), _item(other_b, TRACK_B, 0.5)]
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in first],
            [item.candidate.target.target_id for item in second],
        )
        # The candidate ids remain intact as record identities.
        self.assertEqual(
            {item.candidate.candidate_id for item in first}, {CANDIDATE_A, CANDIDATE_B}
        )
        self.assertEqual(
            {item.candidate.candidate_id for item in second}, {other_a, other_b}
        )

    def test_input_order_never_matters(self) -> None:
        forward = rank_items(
            [_item(CANDIDATE_A, TRACK_A, 0.4), _item(CANDIDATE_B, TRACK_B, 0.7)]
        )
        reversed_ = rank_items(
            [_item(CANDIDATE_B, TRACK_B, 0.7), _item(CANDIDATE_A, TRACK_A, 0.4)]
        )
        self.assertEqual(forward, reversed_)

    def test_empty_input_ranks_empty(self) -> None:
        self.assertEqual(rank_items(()), ())

    def test_single_item_passes_through(self) -> None:
        only = _item(CANDIDATE_A, TRACK_A, 0.3)
        self.assertEqual(rank_items([only]), (only,))

    def test_non_iterable_fails_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items(None)
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items("not-an-iterable")

    def test_non_item_entries_fail_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items([_item(CANDIDATE_A, TRACK_A, 0.5), "not-an-item"])

    def test_duplicate_candidate_ids_fail_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_items(
                [_item(CANDIDATE_A, TRACK_A, 0.5), _item(CANDIDATE_A, TRACK_B, 0.5)]
            )


class RankRecommendationsTest(unittest.TestCase):
    def test_assembles_contract_valid_result_with_limit(self) -> None:
        request = _request(limit=2)
        outcome = rank_recommendations(
            request,
            [_item(CANDIDATE_A, TRACK_A, 0.4), _item(CANDIDATE_B, TRACK_B, 0.9), _item(CANDIDATE_C, TRACK_C, 0.7)],
            run_id=RUN_ID,
            produced_at=PRODUCED_AT,
        )
        result = outcome.result
        self.assertEqual(result.contract_version, RECOMMENDATION_CONTRACT_VERSION)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(result.produced_at, PRODUCED_AT)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(
            [item.candidate.candidate_id for item in result.items], [CANDIDATE_B, CANDIDATE_C]
        )

    def test_empty_items_produce_valid_empty_result(self) -> None:
        outcome = rank_recommendations(
            _request(), [], run_id=RUN_ID, produced_at=PRODUCED_AT
        )
        self.assertEqual(outcome.result.items, ())

    def test_kind_mismatch_fails_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_recommendations(
                _request(kind=RecommendedItemKind.ALBUM),
                [_item(CANDIDATE_A, TRACK_A, 0.9)],
                run_id=RUN_ID,
                produced_at=PRODUCED_AT,
            )

    def test_non_request_fails_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_recommendations(
                None, [], run_id=RUN_ID, produced_at=PRODUCED_AT
            )

    def test_non_item_entries_fail_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            rank_recommendations(
                _request(), ["not-an-item"], run_id=RUN_ID, produced_at=PRODUCED_AT
            )

    def test_deterministic_for_injected_identity_and_instant(self) -> None:
        items = [_item(CANDIDATE_A, TRACK_A, 0.4), _item(CANDIDATE_B, TRACK_B, 0.9)]
        first = rank_recommendations(_request(), items, run_id=RUN_ID, produced_at=PRODUCED_AT)
        second = rank_recommendations(_request(), items, run_id=RUN_ID, produced_at=PRODUCED_AT)
        self.assertEqual(first, second)

    def test_counts_tie_break_flows_through_rank_recommendations(self) -> None:
        # P19-T15: the injected count lookup reaches the ranking comparator through
        # the rank_recommendations seam (score tie -> count asc before target id).
        outcome = rank_recommendations(
            _request(),
            [_item(CANDIDATE_A, TRACK_A, 0.5), _item(CANDIDATE_B, TRACK_B, 0.5)],
            run_id=RUN_ID,
            produced_at=PRODUCED_AT,
            recommendation_counts={TRACK_A: 3, TRACK_B: 0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in outcome.result.items],
            [TRACK_B, TRACK_A],
        )

    def test_without_evidence_report_is_none(self) -> None:
        outcome = rank_recommendations(
            _request(), [_item(CANDIDATE_A, TRACK_A, 0.9)], run_id=RUN_ID, produced_at=PRODUCED_AT
        )
        self.assertIsNone(outcome.quality_report)

    def test_with_evidence_quality_filters_before_ranking(self) -> None:
        evidence = QualityEvidence(
            previous_targets=frozenset({_target(TRACK_A)}),
        )
        outcome = rank_recommendations(
            _request(limit=5),
            [_item(CANDIDATE_A, TRACK_A, 0.9), _item(CANDIDATE_B, TRACK_B, 0.5)],
            run_id=RUN_ID,
            produced_at=PRODUCED_AT,
            quality_evidence=evidence,
        )
        self.assertIsNotNone(outcome.quality_report)
        self.assertEqual([item.candidate.candidate_id for item in outcome.result.items], [CANDIDATE_B])
        decision = outcome.quality_report.decisions[0]
        self.assertEqual(decision.control, CONTROL_REPEAT_RECOMMENDATION)
        self.assertTrue(decision.applied)
        self.assertEqual(
            [exclusion.reason for exclusion in decision.exclusions],
            [REASON_PREVIOUSLY_RECOMMENDED],
        )

    def test_limit_applies_after_quality_filtering(self) -> None:
        evidence = QualityEvidence(previous_targets=frozenset({_target(TRACK_A)}))
        outcome = rank_recommendations(
            _request(limit=1),
            [_item(CANDIDATE_A, TRACK_A, 0.9), _item(CANDIDATE_B, TRACK_B, 0.5)],
            run_id=RUN_ID,
            produced_at=PRODUCED_AT,
            quality_evidence=evidence,
        )
        # The top-scoring item is quality-excluded; the survivor fills the single slot.
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.result.items], [CANDIDATE_B]
        )


class BuildRecommendationTest(unittest.TestCase):
    def _run(self, *, limit: int = 5):
        request = RecommendationRequest(_context_with_preferences(), RecommendedItemKind.TRACK, limit)
        return request, build_recommendation(
            request, run_id=RUN_ID, produced_at=PRODUCED_AT
        )

    def test_completes_chain_and_drops_rejected(self) -> None:
        request, outcome = self._run()
        result = outcome.result
        self.assertEqual(result.contract_version, RECOMMENDATION_CONTRACT_VERSION)
        self.assertIs(result.request, request)
        target_ids = [item.candidate.target.target_id for item in result.items]
        self.assertIn(TRACK_A, target_ids)
        self.assertIn(TRACK_B, target_ids)
        self.assertNotIn(TRACK_C, target_ids)  # REJECTED at the candidate layer
        self.assertNotIn(ARTIST_X, target_ids)  # other kind never becomes a TRACK candidate
        totals = [item.score.total for item in result.items]
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_limit_applies_end_to_end(self) -> None:
        _, outcome = self._run(limit=1)
        self.assertEqual(len(outcome.result.items), 1)
        self.assertEqual(outcome.result.items[0].candidate.target.target_id, TRACK_A)

    def test_deterministic_shape_and_order_for_injected_identity_and_instant(self) -> None:
        # Candidate IDs are minted per run by contract design, so identity is the only
        # nondeterminism; rank order, targets, scores, and the injected identity must be stable.
        _, first = self._run()
        _, second = self._run()
        self.assertEqual(
            [(i.candidate.target.target_id, i.score.total) for i in first.result.items],
            [(i.candidate.target.target_id, i.score.total) for i in second.result.items],
        )
        self.assertEqual(first.result.run_id, second.result.run_id)
        self.assertEqual(first.result.produced_at, second.result.produced_at)
        self.assertEqual(first.result.contract_version, second.result.contract_version)

    def test_non_request_fails_closed(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            build_recommendation(None, run_id=RUN_ID, produced_at=PRODUCED_AT)

    def test_counts_never_outrank_relevance_end_to_end(self) -> None:
        # P19-T15: the heaviest recommendation history must never lift a
        # lower-scoring track above a higher-scoring one, end to end.
        request = RecommendationRequest(
            _context_with_preferences(), RecommendedItemKind.TRACK, 5
        )
        outcome = build_recommendation(
            request,
            run_id=RUN_ID,
            produced_at=PRODUCED_AT,
            recommendation_counts={TRACK_A: 999, TRACK_B: 0},
        )
        self.assertEqual(
            [item.candidate.target.target_id for item in outcome.result.items],
            [TRACK_A, TRACK_B],
        )

    def test_invalid_counts_fail_closed_end_to_end(self) -> None:
        request = RecommendationRequest(
            _context_with_preferences(), RecommendedItemKind.TRACK, 5
        )
        with self.assertRaises(RecommendationRankingValidationError):
            build_recommendation(
                request,
                run_id=RUN_ID,
                produced_at=PRODUCED_AT,
                recommendation_counts="not-a-mapping",  # type: ignore[arg-type]
            )

    def test_ranking_outcome_validates_its_fields(self) -> None:
        with self.assertRaises(RecommendationRankingValidationError):
            RankingOutcome(result="not-a-result")


class RankingErrorHierarchyTest(unittest.TestCase):
    def test_validation_error_is_a_ranking_error(self) -> None:
        self.assertTrue(
            issubclass(RecommendationRankingValidationError, RecommendationRankingError)
        )


if __name__ == "__main__":
    unittest.main()
