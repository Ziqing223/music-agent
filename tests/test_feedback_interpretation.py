"""P08.3: derived feedback interpretation over immutable observation history.

These tests prove the pure interpretation layer only: the conservative policy-v1 per-kind
semantics (explicit statements carry their stated direction; favoriting/replay are direction-only
implicit claims; skip and completion stay ambiguous; corrections and single plays carry no
preference claim), the fail-closed ``NONE`` outcome, attribution and provenance preservation,
explicit-vs-implicit distinguishability, the versioned policy/derivation boundary, and the
absence of any weight, magnitude, confidence, or persistence surface. No observation history is
rewritten, no preference state is touched, and nothing touches SQLite.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone

from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackDirection,
    FeedbackExplicitness,
    FeedbackKind,
    FeedbackObservation,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
)
from music_agent.feedback_interpretation import (
    INTERPRETATION_CONTRACT_VERSION,
    FeedbackInterpretation,
    FeedbackInterpretationError,
    FeedbackInterpretationValidationError,
    InterpretationPolicy,
    InterpretationReason,
    interpret_observation,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

POLICY_V1 = InterpretationPolicy(1)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def source() -> FeedbackSourceReference:
    return FeedbackSourceReference("recommendation_ui", "card_actions")


def recommendation_ref() -> FeedbackRecommendationReference:
    return FeedbackRecommendationReference(RUN_ID, CANDIDATE_ID)


def observation(
    kind: FeedbackKind,
    *,
    attribution: FeedbackAttribution | None = None,
) -> FeedbackObservation:
    return assemble_feedback_observation(
        feedback_id="fbk_44444444-4444-4444-8444-444444444444",
        kind=kind,
        source=source(),
        observed_at=NOW,
        target=track_target(),
        attribution=attribution,
    )


class InterpretationSemanticsTest(unittest.TestCase):
    """The frozen policy-v1 mapping: conservative, kind-by-kind."""

    def test_explicit_statements_carry_their_stated_direction(self) -> None:
        expected = {
            FeedbackKind.LIKED: FeedbackDirection.POSITIVE,
            FeedbackKind.DISLIKED: FeedbackDirection.NEGATIVE,
            FeedbackKind.DIRECTION_GOOD: FeedbackDirection.POSITIVE,
        }
        for kind, direction in expected.items():
            with self.subTest(kind=kind):
                interpretation = interpret_observation(observation(kind), POLICY_V1)
                self.assertEqual(interpretation.direction, direction)
                self.assertEqual(interpretation.reason, InterpretationReason.EXPLICIT_STATEMENT)
                self.assertEqual(interpretation.explicitness, FeedbackExplicitness.EXPLICIT)

    def test_corrections_carry_no_preference_claim(self) -> None:
        for kind in (FeedbackKind.CORRECTED, FeedbackKind.ATTRIBUTION_CORRECTION):
            with self.subTest(kind=kind):
                interpretation = interpret_observation(
                    observation(
                        kind,
                        attribution=FeedbackAttribution(
                            artist_target(), AttributionRelation.EXCLUDED
                        ),
                    ),
                    POLICY_V1,
                )
                self.assertEqual(interpretation.direction, FeedbackDirection.NONE)
                self.assertEqual(
                    interpretation.reason, InterpretationReason.NO_PREFERENCE_CLAIM
                )

    def test_favoriting_is_a_positive_implicit_claim_without_strength(self) -> None:
        interpretation = interpret_observation(
            observation(FeedbackKind.FAVORITED), POLICY_V1
        )
        self.assertEqual(interpretation.direction, FeedbackDirection.POSITIVE)
        self.assertEqual(interpretation.reason, InterpretationReason.IMPLICIT_BEHAVIOR)
        self.assertEqual(interpretation.explicitness, FeedbackExplicitness.IMPLICIT)

    def test_replay_is_a_positive_implicit_claim_without_strength(self) -> None:
        interpretation = interpret_observation(
            observation(FeedbackKind.REPLAYED), POLICY_V1
        )
        self.assertEqual(interpretation.direction, FeedbackDirection.POSITIVE)
        self.assertEqual(interpretation.reason, InterpretationReason.IMPLICIT_BEHAVIOR)
        self.assertEqual(interpretation.explicitness, FeedbackExplicitness.IMPLICIT)

    def test_skip_is_not_dislike(self) -> None:
        interpretation = interpret_observation(
            observation(FeedbackKind.SKIPPED), POLICY_V1
        )
        self.assertEqual(interpretation.direction, FeedbackDirection.NONE)
        self.assertEqual(interpretation.reason, InterpretationReason.AMBIGUOUS_BEHAVIOR)

    def test_completed_playback_is_not_a_like(self) -> None:
        interpretation = interpret_observation(
            observation(FeedbackKind.COMPLETED), POLICY_V1
        )
        self.assertEqual(interpretation.direction, FeedbackDirection.NONE)
        self.assertEqual(interpretation.reason, InterpretationReason.AMBIGUOUS_BEHAVIOR)

    def test_single_play_carries_no_claim_because_meaning_requires_aggregation(self) -> None:
        interpretation = interpret_observation(
            observation(FeedbackKind.PLAYED), POLICY_V1
        )
        self.assertEqual(interpretation.direction, FeedbackDirection.NONE)
        self.assertEqual(
            interpretation.reason, InterpretationReason.REQUIRES_AGGREGATION
        )

    def test_every_kind_has_a_frozen_interpretation(self) -> None:
        for kind in FeedbackKind:
            with self.subTest(kind=kind):
                attribution = (
                    FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED)
                    if kind is FeedbackKind.ATTRIBUTION_CORRECTION
                    else None
                )
                interpretation = interpret_observation(
                    observation(kind, attribution=attribution), POLICY_V1
                )
                self.assertIn(interpretation.direction, FeedbackDirection)
                self.assertIn(interpretation.reason, InterpretationReason)


class InterpretationDerivationBoundaryTest(unittest.TestCase):
    def test_interpretation_is_deterministic_for_observation_and_policy(self) -> None:
        first = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        second = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        self.assertEqual(first, second)

    def test_interpretation_stamps_current_contract_version(self) -> None:
        interpretation = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        self.assertEqual(interpretation.contract_version, INTERPRETATION_CONTRACT_VERSION)

    def test_interpretation_records_the_policy_version_that_produced_it(self) -> None:
        interpretation = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        self.assertEqual(interpretation.policy_version, 1)

    def test_unknown_policy_version_fails_closed(self) -> None:
        with self.assertRaises(FeedbackInterpretationValidationError):
            interpret_observation(observation(FeedbackKind.LIKED), InterpretationPolicy(2))

    def test_rejects_non_observation(self) -> None:
        for bad in ("not-an-observation", None, {"feedback_id": "fbk_x"}, 42):
            with self.subTest(bad=bad):
                with self.assertRaises(FeedbackInterpretationValidationError):
                    interpret_observation(bad, POLICY_V1)  # type: ignore[arg-type]

    def test_rejects_non_policy(self) -> None:
        for bad in ("v1", None, 1, {"version": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(FeedbackInterpretationValidationError):
                    interpret_observation(observation(FeedbackKind.LIKED), bad)  # type: ignore[arg-type]


class InterpretationProvenanceAndAttributionTest(unittest.TestCase):
    def test_interpretation_carries_the_full_observation_for_traceability(self) -> None:
        obs = observation(FeedbackKind.SKIPPED)
        interpretation = interpret_observation(obs, POLICY_V1)
        self.assertIs(interpretation.observation, obs)
        self.assertEqual(interpretation.feedback_id, obs.feedback_id)

    def test_attribution_is_preserved_exactly(self) -> None:
        attribution = FeedbackAttribution(artist_target(), AttributionRelation.ATTRIBUTED)
        obs = observation(FeedbackKind.LIKED, attribution=attribution)
        interpretation = interpret_observation(obs, POLICY_V1)
        self.assertEqual(interpretation.attribution, attribution)
        self.assertEqual(interpretation.attribution.relation, AttributionRelation.ATTRIBUTED)

    def test_excluded_attribution_survives_interpretation(self) -> None:
        attribution = FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED)
        obs = observation(FeedbackKind.ATTRIBUTION_CORRECTION, attribution=attribution)
        interpretation = interpret_observation(obs, POLICY_V1)
        self.assertEqual(interpretation.direction, FeedbackDirection.NONE)
        self.assertEqual(
            interpretation.attribution.relation, AttributionRelation.EXCLUDED
        )

    def test_absent_attribution_stays_absent(self) -> None:
        interpretation = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        self.assertIsNone(interpretation.attribution)

    def test_constructor_rejects_attribution_that_disagrees_with_the_observation(self) -> None:
        obs = observation(FeedbackKind.LIKED)
        with self.assertRaises(FeedbackInterpretationValidationError):
            FeedbackInterpretation(
                observation=obs,
                direction=FeedbackDirection.POSITIVE,
                reason=InterpretationReason.EXPLICIT_STATEMENT,
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.ATTRIBUTED
                ),
                policy_version=1,
                contract_version=1,
            )

    def test_observation_is_not_rewritten_by_interpretation(self) -> None:
        obs = observation(FeedbackKind.SKIPPED)
        interpret_observation(obs, POLICY_V1)
        # The observation's own direction remains NONE for implicit kinds: the interpretation
        # never writes anything back onto the evidence.
        self.assertEqual(obs.direction, FeedbackDirection.NONE)
        self.assertEqual(obs.explicitness, FeedbackExplicitness.IMPLICIT)


class InterpretationShapeTest(unittest.TestCase):
    def test_interpretation_carries_no_weight_magnitude_or_confidence(self) -> None:
        names = {field.name for field in fields(FeedbackInterpretation)}
        self.assertEqual(
            names,
            {
                "observation",
                "direction",
                "reason",
                "attribution",
                "policy_version",
                "contract_version",
            },
        )
        interpretation = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        for absent in ("weight", "magnitude", "confidence", "strength", "decay"):
            self.assertFalse(hasattr(interpretation, absent), absent)

    def test_interpretation_is_immutable_and_hashable(self) -> None:
        interpretation = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        with self.assertRaises(FrozenInstanceError):
            interpretation.direction = FeedbackDirection.NONE
        self.assertEqual(hash(interpretation), hash(interpretation))

    def test_policy_is_a_frozen_value(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            POLICY_V1.version = 2

    def test_policy_version_validation(self) -> None:
        with self.assertRaises(FeedbackInterpretationValidationError):
            InterpretationPolicy(0)
        with self.assertRaises(FeedbackInterpretationValidationError):
            InterpretationPolicy(True)

    def test_implicit_claims_are_distinguishable_from_explicit_claims(self) -> None:
        explicit = interpret_observation(observation(FeedbackKind.LIKED), POLICY_V1)
        implicit = interpret_observation(observation(FeedbackKind.FAVORITED), POLICY_V1)
        self.assertEqual(explicit.direction, implicit.direction)
        self.assertEqual(explicit.explicitness, FeedbackExplicitness.EXPLICIT)
        self.assertEqual(implicit.explicitness, FeedbackExplicitness.IMPLICIT)
        self.assertNotEqual(explicit.reason, implicit.reason)


if __name__ == "__main__":
    unittest.main()
