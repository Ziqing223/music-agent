"""P07 integration: the cross-track recommendation pipeline assembled at the contract boundary.

P07.1 froze the shared contract; P07.2 (candidate generation), P07.3 (scoring), P07.5
(persistence/history), P07.4 (production ranking), and P07.6 (quality controls) each landed as
an independent track. This lead-owned integration test proves the tracks compose: preference
inputs become candidates, eligible candidates become scored items, the items assemble into a
contract-valid ``RecommendationResult``, the result round-trips through the durable history
repository, the production chain (``build_recommendation``) applies the quality policy with an
explainable report, and repeat-recommendation control is driven by real persisted history.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.candidate_generation import (
    REJECTION_NEGATIVE_PREFERENCE,
    generate_candidates,
)
from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    RECOMMENDATION_CONTRACT_VERSION,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    assemble_recommendation_result,
    decode_recommendation_result,
    encode_recommendation_result,
    generate_run_id,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.recommendation_quality import (
    CONTROL_REPEAT_RECOMMENDATION,
    REASON_PREVIOUSLY_RECOMMENDED,
    QualityEvidence,
    collect_previous_targets,
)
from music_agent.recommendation_ranking import build_recommendation, rank_items
from music_agent.recommendation_scoring import score_candidate

TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_22222222-2222-4222-8222-222222222222"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
ARTIST_X = "art_44444444-4444-4444-8444-444444444444"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
PRODUCED_AT = datetime(2026, 8, 16, 12, 30, 0, tzinfo=timezone.utc)


def _target(kind: PreferenceTargetKind, target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(kind, target_id)


def _direct(target_id: str, state: PreferenceState, magnitude: float | None) -> PreferenceInput:
    return PreferenceInput.from_direct(
        DerivedPreference(_target(PreferenceTargetKind.TRACK, target_id), PreferenceStrength(state, magnitude))
    )


def _inferred(target_id: str, state: PreferenceState, magnitude: float | None) -> PreferenceInput:
    return PreferenceInput.from_inferred(
        InferredAffinity(_target(PreferenceTargetKind.TRACK, target_id), PreferenceStrength(state, magnitude))
    )


def _pipeline_context() -> RecommendationContext:
    """Two positive tracks, one negative track, one direct+inferred pair, one other-kind input."""
    return RecommendationContext(
        NOW,
        (
            _direct(TRACK_A, PreferenceState.POSITIVE, 0.9),
            _inferred(TRACK_B, PreferenceState.POSITIVE, 0.6),
            _direct(TRACK_C, PreferenceState.NEGATIVE, 0.8),
            _inferred(TRACK_A, PreferenceState.POSITIVE, 0.4),  # merged with the direct claim
            PreferenceInput.from_inferred(
                InferredAffinity(_target(PreferenceTargetKind.ARTIST, ARTIST_X), PreferenceStrength(PreferenceState.POSITIVE, 0.5))
            ),
        ),
    )


class RecommendationPipelineIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "pipeline.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run_pipeline(self) -> tuple[RecommendationContext, object]:
        context = _pipeline_context()
        request = RecommendationRequest(context, RecommendedItemKind.TRACK, limit=5)
        candidates = generate_candidates(context, request.recommended_kind)
        rejected = [c for c in candidates if c.eligibility is Eligibility.REJECTED]
        eligible = [c for c in candidates if c.eligibility is Eligibility.ELIGIBLE]
        items = [
            RecommendationItem(candidate, score_candidate(candidate, context))
            for candidate in eligible
        ]
        result = assemble_recommendation_result(
            request,
            rank_items(items),
            run_id=generate_run_id(),
            produced_at=PRODUCED_AT,
        )
        return context, result

    def test_pipeline_assembles_a_contract_valid_result(self) -> None:
        _, result = self._run_pipeline()
        self.assertEqual(result.contract_version, RECOMMENDATION_CONTRACT_VERSION)
        self.assertEqual(result.request.recommended_kind, RecommendedItemKind.TRACK)
        # Tuple order is rank: totals descend, and the rejected track never becomes an item.
        totals = [item.score.total for item in result.items]
        self.assertEqual(totals, sorted(totals, reverse=True))
        recommended_targets = {item.candidate.target.target_id for item in result.items}
        self.assertNotIn(TRACK_C, recommended_targets)

    def test_pipeline_excludes_rejected_and_skips_other_kinds(self) -> None:
        context, result = self._run_pipeline()
        candidates = generate_candidates(context, RecommendedItemKind.TRACK)
        rejected = [c for c in candidates if c.eligibility is Eligibility.REJECTED]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].target.target_id, TRACK_C)
        self.assertEqual(rejected[0].rejection.reason, REJECTION_NEGATIVE_PREFERENCE)
        # The artist input never becomes a TRACK candidate.
        candidate_targets = {c.target.target_id for c in candidates}
        self.assertNotIn(ARTIST_X, candidate_targets)
        # Two positive targets merge per target; direct+inferred on TRACK_A stays one candidate.
        self.assertEqual(len(candidates), 3)
        item_ids = {item.candidate.candidate_id for item in result.items}
        self.assertEqual(len(item_ids), 2)

    def test_direct_claim_governs_merged_target_in_scoring(self) -> None:
        context, result = self._run_pipeline()
        item_a = next(
            item for item in result.items if item.candidate.target.target_id == TRACK_A
        )
        # Operative conclusion is the formed DIRECT POSITIVE (0.9); the inferred input must not
        # override it, so the breakdown reflects the direct claim only.
        components = {component.name: component.value for component in item_a.score.components}
        self.assertEqual(components["basis_support"], 1.0)
        self.assertEqual(components["basis_strength"], 0.9)
        self.assertEqual(components["provenance_directness"], 1.0)
        self.assertEqual(components["negative_evidence"], 0.0)
        self.assertEqual(item_a.score.total, 0.9)

    def test_result_round_trips_through_history_repository(self) -> None:
        _, result = self._run_pipeline()
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            repository.save_result(result)
            self.assertEqual(repository.get_result(result.run_id), result)
            self.assertEqual(repository.list_runs(), (result,))
            self.assertEqual(
                decode_recommendation_result(encode_recommendation_result(result)), result
            )

    def test_empty_outcome_is_a_valid_persisted_run(self) -> None:
        empty_context = RecommendationContext(NOW, ())
        request = RecommendationRequest(empty_context, RecommendedItemKind.ALBUM, limit=3)
        self.assertEqual(generate_candidates(empty_context, request.recommended_kind), ())
        result = assemble_recommendation_result(
            request, (), run_id=generate_run_id(), produced_at=PRODUCED_AT
        )
        self.assertEqual(result.items, ())
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(result)
            self.assertEqual(repository.get_result(result.run_id), result)

    def test_context_now_stays_distinct_from_produced_at(self) -> None:
        context, result = self._run_pipeline()
        self.assertEqual(context.now, NOW)
        self.assertEqual(result.produced_at, PRODUCED_AT)
        self.assertNotEqual(context.now, result.produced_at)

    def test_pipeline_never_touches_preference_tables(self) -> None:
        _, result = self._run_pipeline()
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(result)
            for table in (
                "preference_signal_heads",
                "preference_evidence_revisions",
                "canonical_entities",
            ):
                count = repository._connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                self.assertEqual(count, 0, f"{table} must stay untouched by P07")


class ProductionChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chain.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _request(self) -> RecommendationRequest:
        return RecommendationRequest(_pipeline_context(), RecommendedItemKind.TRACK, limit=5)

    def test_build_recommendation_produces_contract_valid_result(self) -> None:
        request = self._request()
        outcome = build_recommendation(
            request, run_id=generate_run_id(), produced_at=PRODUCED_AT
        )
        result = outcome.result
        self.assertEqual(result.contract_version, RECOMMENDATION_CONTRACT_VERSION)
        totals = [item.score.total for item in result.items]
        self.assertEqual(totals, sorted(totals, reverse=True))
        target_ids = [item.candidate.target.target_id for item in result.items]
        self.assertNotIn(TRACK_C, target_ids)
        self.assertNotIn(ARTIST_X, target_ids)
        self.assertIsNone(outcome.quality_report)

    def test_quality_evidence_shapes_result_with_explainable_report(self) -> None:
        request = self._request()
        evidence = QualityEvidence(
            previous_targets=frozenset({_target(PreferenceTargetKind.TRACK, TRACK_A)})
        )
        outcome = build_recommendation(
            request,
            run_id=generate_run_id(),
            produced_at=PRODUCED_AT,
            quality_evidence=evidence,
        )
        target_ids = [item.candidate.target.target_id for item in outcome.result.items]
        self.assertNotIn(TRACK_A, target_ids)
        self.assertIn(TRACK_B, target_ids)
        decision = next(
            d for d in outcome.quality_report.decisions
            if d.control == CONTROL_REPEAT_RECOMMENDATION
        )
        self.assertTrue(decision.applied)
        self.assertEqual(
            [exclusion.reason for exclusion in decision.exclusions],
            [REASON_PREVIOUSLY_RECOMMENDED],
        )

    def test_history_backed_repeat_control_prevents_reruns(self) -> None:
        request = self._request()
        first = build_recommendation(
            request, run_id=generate_run_id(), produced_at=PRODUCED_AT
        )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(first.result)
            previous = collect_previous_targets(repository.list_runs())
        self.assertEqual(
            previous,
            frozenset({_target(PreferenceTargetKind.TRACK, t) for t in (TRACK_A, TRACK_B)}),
        )
        second = build_recommendation(
            request,
            run_id=generate_run_id(),
            produced_at=PRODUCED_AT,
            quality_evidence=QualityEvidence(previous_targets=previous),
        )
        self.assertEqual(second.result.items, ())
        decision = next(
            d for d in second.quality_report.decisions
            if d.control == CONTROL_REPEAT_RECOMMENDATION
        )
        self.assertTrue(decision.applied)
        self.assertEqual(len(decision.exclusions), 2)


if __name__ == "__main__":
    unittest.main()
