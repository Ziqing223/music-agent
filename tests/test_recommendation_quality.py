"""P07.6: bounded first-version recommendation quality controls.

These tests prove the quality slice: deterministic scenario derivation, QualityEvidence
validation (orphan caps fail closed), each of the four controls applied independently and in
the fixed order (repeat -> scenario -> diversity -> familiar/new), keep-best decisions using
the canonical order, explainable exclusion reasons, skipped-control reporting, and the pure
collect_previous_targets history bridge. Nothing touches SQLite or the clock.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
    generate_run_id,
)
from music_agent.recommendation_quality import (
    CONTROL_DIVERSITY,
    CONTROL_FAMILIAR_NEW_BALANCE,
    CONTROL_REPEAT_RECOMMENDATION,
    CONTROL_SCENARIO_RELEVANCE,
    REASON_DIVERSITY_GROUP_CAP,
    REASON_PREVIOUSLY_RECOMMENDED,
    REASON_SCENARIO_UNSUITABLE,
    REASON_UNFAMILIAR_CAP,
    QualityControlDecision,
    QualityControlReport,
    QualityEvidence,
    QualityExclusion,
    RecommendationQualityError,
    RecommendationQualityValidationError,
    Scenario,
    apply_quality_policy,
    collect_previous_targets,
    derive_scenario,
)

TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_22222222-2222-4222-8222-222222222222"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
TRACK_D = "trk_44444444-4444-4444-8444-444444444444"
CANDIDATE_A = "cnd_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CANDIDATE_B = "cnd_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
CANDIDATE_C = "cnd_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
CANDIDATE_D = "cnd_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
NOW_MORNING = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)


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


def _item_tuple(*specs) -> tuple[RecommendationItem, ...]:
    return tuple(_item(candidate_id, target_id, total) for candidate_id, target_id, total in specs)


def _decision(report: QualityControlReport, control: str) -> QualityControlDecision:
    return next(decision for decision in report.decisions if decision.control == control)


class DeriveScenarioTest(unittest.TestCase):
    def test_daypart_boundaries(self) -> None:
        cases = (
            (datetime(2026, 8, 16, 5, 59, 0, tzinfo=timezone.utc), Scenario.NIGHT),
            (datetime(2026, 8, 16, 6, 0, 0, tzinfo=timezone.utc), Scenario.MORNING),
            (datetime(2026, 8, 16, 11, 59, 0, tzinfo=timezone.utc), Scenario.MORNING),
            (datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc), Scenario.AFTERNOON),
            (datetime(2026, 8, 16, 17, 59, 0, tzinfo=timezone.utc), Scenario.AFTERNOON),
            (datetime(2026, 8, 16, 18, 0, 0, tzinfo=timezone.utc), Scenario.EVENING),
            (datetime(2026, 8, 16, 23, 59, 0, tzinfo=timezone.utc), Scenario.EVENING),
            (datetime(2026, 8, 17, 0, 0, 0, tzinfo=timezone.utc), Scenario.NIGHT),
        )
        for now, expected in cases:
            with self.subTest(now=now):
                self.assertIs(derive_scenario(now), expected)

    def test_non_aware_datetime_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            derive_scenario(datetime(2026, 8, 16, 10, 0, 0))
        with self.assertRaises(RecommendationQualityValidationError):
            derive_scenario("not-a-datetime")


class QualityEvidenceValidationTest(unittest.TestCase):
    def test_orphan_max_unfamiliar_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(max_unfamiliar=2)

    def test_missing_max_unfamiliar_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(unfamiliar_targets=frozenset({_target(TRACK_A)}))

    def test_orphan_max_per_group_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(max_per_group=2)

    def test_missing_max_per_group_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(diversity_groups={"g": frozenset({_target(TRACK_A)})})

    def test_non_positive_and_bool_caps_fail_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(unfamiliar_targets=frozenset(), max_unfamiliar=True)
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(unfamiliar_targets=frozenset(), max_unfamiliar=-1)
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(diversity_groups={"g": frozenset()}, max_per_group=0)

    def test_wrong_evidence_types_fail_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(previous_targets=frozenset({TRACK_A}))
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(diversity_groups={"" : frozenset()}, max_per_group=1)
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(
                scenario_suitability={_target(TRACK_A): frozenset({"morning"})}
            )

    def test_unhashable_evidence_entries_fail_closed_with_quality_error(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(previous_targets=[[_target(TRACK_A)]])
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(
                diversity_groups={"g": [[_target(TRACK_A)]]}, max_per_group=1
            )
        with self.assertRaises(RecommendationQualityValidationError):
            QualityEvidence(
                scenario_suitability={_target(TRACK_A): [[Scenario.MORNING]]}
            )

    def test_exclusion_vocabulary_is_fixed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            QualityExclusion(CANDIDATE_A, _target(TRACK_A), "made_up_reason")
        with self.assertRaises(RecommendationQualityValidationError):
            QualityExclusion("not-a-cnd-id", _target(TRACK_A), REASON_UNFAMILIAR_CAP)


class ApplyQualityPolicySkippedControlsTest(unittest.TestCase):
    def test_none_evidence_skips_every_control(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        outcome = apply_quality_policy(items, NOW_MORNING, None)
        self.assertEqual(outcome.selected_items, items)
        self.assertIs(outcome.report.scenario, Scenario.MORNING)
        self.assertEqual(len(outcome.report.decisions), 4)
        self.assertTrue(all(not decision.applied for decision in outcome.report.decisions))

    def test_partial_evidence_skips_only_missing_controls(self) -> None:
        outcome = apply_quality_policy(
            _item_tuple((CANDIDATE_A, TRACK_A, 0.9)),
            NOW_MORNING,
            QualityEvidence(previous_targets=frozenset()),
        )
        self.assertTrue(_decision(outcome.report, CONTROL_REPEAT_RECOMMENDATION).applied)
        for control in (CONTROL_SCENARIO_RELEVANCE, CONTROL_DIVERSITY, CONTROL_FAMILIAR_NEW_BALANCE):
            self.assertFalse(_decision(outcome.report, control).applied)

    def test_non_item_entries_fail_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            apply_quality_policy(["not-an-item"], NOW_MORNING, None)

    def test_bad_evidence_type_fails_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            apply_quality_policy((), NOW_MORNING, "not-evidence")


class RepeatRecommendationControlTest(unittest.TestCase):
    def test_excludes_previously_recommended_targets(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        outcome = apply_quality_policy(
            items,
            NOW_MORNING,
            QualityEvidence(previous_targets=frozenset({_target(TRACK_A)})),
        )
        self.assertEqual(outcome.selected_items, (items[1],))
        decision = _decision(outcome.report, CONTROL_REPEAT_RECOMMENDATION)
        self.assertTrue(decision.applied)
        exclusion = decision.exclusions[0]
        self.assertEqual(exclusion.candidate_id, CANDIDATE_A)
        self.assertEqual(exclusion.reason, REASON_PREVIOUSLY_RECOMMENDED)

    def test_empty_previous_targets_excludes_nothing(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9))
        outcome = apply_quality_policy(
            items, NOW_MORNING, QualityEvidence(previous_targets=frozenset())
        )
        self.assertEqual(outcome.selected_items, items)
        self.assertEqual(_decision(outcome.report, CONTROL_REPEAT_RECOMMENDATION).exclusions, ())


class ScenarioRelevanceControlTest(unittest.TestCase):
    def test_unsuitable_target_excluded_with_reason(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            scenario_suitability={_target(TRACK_A): frozenset({Scenario.EVENING})}
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(outcome.selected_items, (items[1],))
        decision = _decision(outcome.report, CONTROL_SCENARIO_RELEVANCE)
        self.assertEqual(
            [(e.candidate_id, e.reason) for e in decision.exclusions],
            [(CANDIDATE_A, REASON_SCENARIO_UNSUITABLE)],
        )

    def test_suitable_and_unlisted_targets_survive(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            scenario_suitability={_target(TRACK_A): frozenset({Scenario.MORNING})}
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        # TRACK_A listed as morning-suitable; TRACK_B absent from evidence -> unrestricted.
        self.assertEqual(outcome.selected_items, items)


class DiversityControlTest(unittest.TestCase):
    def test_group_cap_keeps_best_by_canonical_order(self) -> None:
        items = _item_tuple(
            (CANDIDATE_A, TRACK_A, 0.4), (CANDIDATE_B, TRACK_B, 0.9), (CANDIDATE_C, TRACK_C, 0.7)
        )
        evidence = QualityEvidence(
            diversity_groups={"g": frozenset({_target(t) for t in (TRACK_A, TRACK_B, TRACK_C)})},
            max_per_group=2,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items],
            [CANDIDATE_B, CANDIDATE_C],
        )
        decision = _decision(outcome.report, CONTROL_DIVERSITY)
        self.assertEqual(
            [(e.candidate_id, e.reason) for e in decision.exclusions],
            [(CANDIDATE_A, REASON_DIVERSITY_GROUP_CAP)],
        )

    def test_target_in_no_group_is_unrestricted(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            diversity_groups={"g": frozenset({_target(TRACK_A)})}, max_per_group=1
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(outcome.selected_items, items)

    def test_target_in_multiple_groups_fails_closed(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9))
        evidence = QualityEvidence(
            diversity_groups={
                "g1": frozenset({_target(TRACK_A)}),
                "g2": frozenset({_target(TRACK_A)}),
            },
            max_per_group=1,
        )
        with self.assertRaises(RecommendationQualityValidationError):
            apply_quality_policy(items, NOW_MORNING, evidence)

    def test_group_cap_tie_breaks_by_candidate_id(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.5), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            diversity_groups={"g": frozenset({_target(TRACK_A), _target(TRACK_B)})},
            max_per_group=1,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items], [CANDIDATE_A]
        )
        decision = _decision(outcome.report, CONTROL_DIVERSITY)
        self.assertEqual(
            [exclusion.candidate_id for exclusion in decision.exclusions], [CANDIDATE_B]
        )

    def test_survivors_keep_relative_input_order(self) -> None:
        items = _item_tuple(
            (CANDIDATE_C, TRACK_C, 0.7), (CANDIDATE_B, TRACK_B, 0.9), (CANDIDATE_A, TRACK_A, 0.4)
        )
        evidence = QualityEvidence(
            diversity_groups={
                "g": frozenset({_target(TRACK_A), _target(TRACK_B)}),
            },
            max_per_group=1,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        # Group keeps its best member (B, score 0.9); ungrouped C survives; input order C, B kept.
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items],
            [CANDIDATE_C, CANDIDATE_B],
        )


class FamiliarNewBalanceControlTest(unittest.TestCase):
    def test_under_cap_keeps_everything(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            unfamiliar_targets=frozenset({_target(TRACK_A)}), max_unfamiliar=1
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(outcome.selected_items, items)
        self.assertEqual(_decision(outcome.report, CONTROL_FAMILIAR_NEW_BALANCE).exclusions, ())

    def test_over_cap_keeps_highest_scoring_unfamiliar(self) -> None:
        items = _item_tuple(
            (CANDIDATE_A, TRACK_A, 0.4), (CANDIDATE_B, TRACK_B, 0.9), (CANDIDATE_C, TRACK_C, 0.7)
        )
        evidence = QualityEvidence(
            unfamiliar_targets=frozenset({_target(t) for t in (TRACK_A, TRACK_B, TRACK_C)}),
            max_unfamiliar=2,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items],
            [CANDIDATE_B, CANDIDATE_C],
        )
        decision = _decision(outcome.report, CONTROL_FAMILIAR_NEW_BALANCE)
        self.assertEqual(
            [(e.candidate_id, e.reason) for e in decision.exclusions],
            [(CANDIDATE_A, REASON_UNFAMILIAR_CAP)],
        )

    def test_zero_cap_excludes_all_unfamiliar(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            unfamiliar_targets=frozenset({_target(TRACK_A)}), max_unfamiliar=0
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(outcome.selected_items, (items[1],))

    def test_equal_scores_break_ties_by_candidate_id(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.5), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            unfamiliar_targets=frozenset({_target(TRACK_A), _target(TRACK_B)}),
            max_unfamiliar=1,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items], [CANDIDATE_A]
        )


class CombinedPolicyTest(unittest.TestCase):
    def test_controls_run_in_fixed_order_with_one_decision_each(self) -> None:
        items = _item_tuple(
            (CANDIDATE_A, TRACK_A, 0.9),
            (CANDIDATE_B, TRACK_B, 0.5),
            (CANDIDATE_C, TRACK_C, 0.7),
            (CANDIDATE_D, TRACK_D, 0.3),
        )
        evidence = QualityEvidence(
            previous_targets=frozenset({_target(TRACK_A)}),
            scenario_suitability={_target(TRACK_B): frozenset({Scenario.EVENING})},
            diversity_groups={"g": frozenset({_target(t) for t in (TRACK_C, TRACK_D)})},
            max_per_group=1,
            unfamiliar_targets=frozenset({_target(TRACK_C)}),
            max_unfamiliar=1,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        # A dropped by repeat; B unsuitable in morning; C survives diversity (best in group);
        # D dropped by group cap; unfamiliar cap keeps C (only one unfamiliar).
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items], [CANDIDATE_C]
        )
        controls = [decision.control for decision in outcome.report.decisions]
        self.assertEqual(
            controls,
            [
                CONTROL_REPEAT_RECOMMENDATION,
                CONTROL_SCENARIO_RELEVANCE,
                CONTROL_DIVERSITY,
                CONTROL_FAMILIAR_NEW_BALANCE,
            ],
        )
        self.assertTrue(all(decision.applied for decision in outcome.report.decisions))
        self.assertEqual(len(_decision(outcome.report, CONTROL_DIVERSITY).exclusions), 1)
        self.assertEqual(
            _decision(outcome.report, CONTROL_DIVERSITY).exclusions[0].candidate_id, CANDIDATE_D
        )

    def test_each_control_sees_only_the_survivors_of_earlier_controls(self) -> None:
        items = _item_tuple((CANDIDATE_A, TRACK_A, 0.9), (CANDIDATE_B, TRACK_B, 0.5))
        evidence = QualityEvidence(
            previous_targets=frozenset({_target(TRACK_A)}),
            unfamiliar_targets=frozenset({_target(TRACK_A), _target(TRACK_B)}),
            max_unfamiliar=1,
        )
        outcome = apply_quality_policy(items, NOW_MORNING, evidence)
        # Repeat control removes A before the unfamiliar cap runs, so the cap sees only B:
        # one unfamiliar survivor within cap -> no familiar/new exclusion.
        self.assertEqual(
            [item.candidate.candidate_id for item in outcome.selected_items], [CANDIDATE_B]
        )
        familiar_decision = _decision(outcome.report, CONTROL_FAMILIAR_NEW_BALANCE)
        self.assertTrue(familiar_decision.applied)
        self.assertEqual(familiar_decision.exclusions, ())


class CollectPreviousTargetsTest(unittest.TestCase):
    def _result(self, target_ids: tuple[str, ...]) -> RecommendationResult:
        items = _item_tuple(
            *((f"cnd_{i}0000000-0000-4000-8000-00000000000{i}", t, 0.5) for i, t in enumerate(target_ids))
        )
        request = RecommendationRequest(
            RecommendationContext(NOW_MORNING, ()), RecommendedItemKind.TRACK, limit=5
        )
        return assemble_recommendation_result(
            request, items, run_id=generate_run_id(), produced_at=NOW_MORNING
        )

    def test_collects_targets_across_results(self) -> None:
        targets = collect_previous_targets(
            [self._result((TRACK_A, TRACK_B)), self._result((TRACK_C,))]
        )
        self.assertEqual(
            targets, frozenset({_target(t) for t in (TRACK_A, TRACK_B, TRACK_C)})
        )

    def test_empty_history_collects_empty(self) -> None:
        self.assertEqual(collect_previous_targets([]), frozenset())

    def test_wrong_entry_types_fail_closed(self) -> None:
        with self.assertRaises(RecommendationQualityValidationError):
            collect_previous_targets(["not-a-result"])


class ErrorHierarchyTest(unittest.TestCase):
    def test_validation_error_is_a_quality_error(self) -> None:
        self.assertTrue(
            issubclass(RecommendationQualityValidationError, RecommendationQualityError)
        )


if __name__ == "__main__":
    unittest.main()
