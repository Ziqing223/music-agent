"""P08.4: the learning-effect contract between interpretation and preference learning.

These tests prove the pure derived contract only: the frozen policy-v1 mapping from a
:class:`FeedbackInterpretation` to a categorical permitted action (``POSITIVE_EVIDENCE`` /
``NEGATIVE_EVIDENCE`` / ``ATTRIBUTION_EXCLUSION`` / ``NO_EFFECT``), fail-closed ``NO_EFFECT``
outcomes for no-claim interpretations and target-less observations, full traceability to
feedback_id / policy versions / target / attribution, preserved attribution exclusion, the
explicit-vs-implicit origin, and the absence of any numeric weight, delta, or P06 mutation
surface. No preference state is touched and nothing touches SQLite.
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
    FeedbackInterpretation,
    InterpretationPolicy,
    interpret_observation,
)
from music_agent.learning_effect import (
    LEARNING_EFFECT_CONTRACT_VERSION,
    LearningEffect,
    LearningEffectKind,
    LearningEffectPolicy,
    LearningEffectReason,
    LearningEffectValidationError,
    derive_learning_effect,
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

INTERPRETATION_POLICY = InterpretationPolicy(1)
EFFECT_POLICY = LearningEffectPolicy(1)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def genre_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.GENRE, "jazz")


def source() -> FeedbackSourceReference:
    return FeedbackSourceReference("recommendation_ui", "card_actions")


def recommendation_ref() -> FeedbackRecommendationReference:
    return FeedbackRecommendationReference(RUN_ID, CANDIDATE_ID)


def observation(
    kind: FeedbackKind,
    *,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
) -> FeedbackObservation:
    return assemble_feedback_observation(
        feedback_id="fbk_44444444-4444-4444-8444-444444444444",
        kind=kind,
        source=source(),
        observed_at=NOW,
        target=target,
        recommendation=recommendation,
        attribution=attribution,
    )


def interpretation(
    kind: FeedbackKind,
    *,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
) -> FeedbackInterpretation:
    return interpret_observation(
        observation(
            kind,
            target=target,
            recommendation=recommendation,
            attribution=attribution,
        ),
        INTERPRETATION_POLICY,
    )


class LearningEffectMappingTest(unittest.TestCase):
    """The frozen policy-v1 mapping: interpretation -> permitted categorical action."""

    def test_liked_statement_permits_positive_evidence_for_the_track(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(effect.kind, LearningEffectKind.POSITIVE_EVIDENCE)
        self.assertEqual(effect.target, track_target())
        self.assertEqual(effect.reason, LearningEffectReason.EXPLICIT_EVIDENCE)
        self.assertEqual(effect.direction, FeedbackDirection.POSITIVE)
        self.assertEqual(effect.explicitness, FeedbackExplicitness.EXPLICIT)

    def test_disliked_statement_permits_negative_evidence_for_the_track(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.DISLIKED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(effect.kind, LearningEffectKind.NEGATIVE_EVIDENCE)
        self.assertEqual(effect.target, track_target())
        self.assertEqual(effect.reason, LearningEffectReason.EXPLICIT_EVIDENCE)
        self.assertEqual(effect.direction, FeedbackDirection.NEGATIVE)

    def test_direction_good_permits_positive_evidence_for_a_genre_direction(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.DIRECTION_GOOD, target=genre_target()),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.POSITIVE_EVIDENCE)
        self.assertEqual(effect.target, genre_target())

    def test_favoriting_permits_positive_evidence_with_implicit_origin(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.FAVORITED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(effect.kind, LearningEffectKind.POSITIVE_EVIDENCE)
        self.assertEqual(effect.reason, LearningEffectReason.IMPLICIT_EVIDENCE)
        self.assertEqual(effect.explicitness, FeedbackExplicitness.IMPLICIT)

    def test_replay_permits_positive_evidence_with_implicit_origin(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.REPLAYED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(effect.kind, LearningEffectKind.POSITIVE_EVIDENCE)
        self.assertEqual(effect.reason, LearningEffectReason.IMPLICIT_EVIDENCE)

    def test_implicit_and_explicit_positive_evidence_stay_distinguishable(self) -> None:
        explicit = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        implicit = derive_learning_effect(
            interpretation(FeedbackKind.FAVORITED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(explicit.kind, implicit.kind)
        self.assertEqual(explicit.explicitness, FeedbackExplicitness.EXPLICIT)
        self.assertEqual(implicit.explicitness, FeedbackExplicitness.IMPLICIT)
        self.assertNotEqual(explicit.reason, implicit.reason)

    def test_no_claim_interpretations_never_become_learning_effects(self) -> None:
        for kind in (
            FeedbackKind.SKIPPED,
            FeedbackKind.COMPLETED,
            FeedbackKind.PLAYED,
            FeedbackKind.CORRECTED,
        ):
            with self.subTest(kind=kind):
                effect = derive_learning_effect(
                    interpretation(kind, target=track_target()), EFFECT_POLICY
                )
                self.assertEqual(effect.kind, LearningEffectKind.NO_EFFECT)
                self.assertEqual(effect.reason, LearningEffectReason.NO_DIRECTIONAL_CLAIM)
                self.assertEqual(effect.direction, FeedbackDirection.NONE)

    def test_corrected_with_attributed_relation_carries_no_effect(self) -> None:
        effect = derive_learning_effect(
            interpretation(
                FeedbackKind.CORRECTED,
                target=track_target(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.ATTRIBUTED
                ),
            ),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.NO_EFFECT)
        self.assertEqual(effect.reason, LearningEffectReason.NO_DIRECTIONAL_CLAIM)

    def test_attribution_correction_permits_attribution_exclusion(self) -> None:
        effect = derive_learning_effect(
            interpretation(
                FeedbackKind.ATTRIBUTION_CORRECTION,
                target=track_target(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            ),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.ATTRIBUTION_EXCLUSION)
        self.assertEqual(effect.target, track_target())
        self.assertEqual(effect.reason, LearningEffectReason.ATTRIBUTION_EXCLUDED)
        self.assertEqual(effect.attribution.aspect, artist_target())
        self.assertEqual(effect.attribution.relation, AttributionRelation.EXCLUDED)

    def test_liked_with_excluded_attribution_keeps_evidence_and_exclusion(self) -> None:
        effect = derive_learning_effect(
            interpretation(
                FeedbackKind.LIKED,
                target=track_target(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            ),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.POSITIVE_EVIDENCE)
        self.assertEqual(effect.attribution.relation, AttributionRelation.EXCLUDED)


class LearningEffectTargetTest(unittest.TestCase):
    def test_recommendation_only_observation_permits_no_effect(self) -> None:
        effect = derive_learning_effect(
            interpretation(
                FeedbackKind.DIRECTION_GOOD, recommendation=recommendation_ref()
            ),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.NO_EFFECT)
        self.assertEqual(effect.reason, LearningEffectReason.NO_PREFERENCE_TARGET)
        self.assertIsNone(effect.target)

    def test_target_less_attribution_exclusion_permits_no_effect(self) -> None:
        effect = derive_learning_effect(
            interpretation(
                FeedbackKind.ATTRIBUTION_CORRECTION,
                recommendation=recommendation_ref(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            ),
            EFFECT_POLICY,
        )
        self.assertEqual(effect.kind, LearningEffectKind.NO_EFFECT)
        self.assertEqual(effect.reason, LearningEffectReason.NO_PREFERENCE_TARGET)

    def test_non_no_effect_kinds_require_a_target(self) -> None:
        with self.assertRaises(LearningEffectValidationError):
            LearningEffect(
                interpretation=interpretation(
                    FeedbackKind.LIKED, target=track_target()
                ),
                kind=LearningEffectKind.POSITIVE_EVIDENCE,
                target=None,
                reason=LearningEffectReason.EXPLICIT_EVIDENCE,
                policy_version=1,
                contract_version=1,
            )

    def test_no_effect_may_carry_no_target(self) -> None:
        effect = LearningEffect(
            interpretation=interpretation(FeedbackKind.LIKED, target=track_target()),
            kind=LearningEffectKind.NO_EFFECT,
            target=None,
            reason=LearningEffectReason.NO_DIRECTIONAL_CLAIM,
            policy_version=1,
            contract_version=1,
        )
        self.assertIsNone(effect.target)


class LearningEffectTraceabilityTest(unittest.TestCase):
    def test_effect_carries_the_full_interpretation(self) -> None:
        interp = interpretation(FeedbackKind.SKIPPED, target=track_target())
        effect = derive_learning_effect(interp, EFFECT_POLICY)
        self.assertIs(effect.interpretation, interp)
        self.assertIs(effect.observation, interp.observation)

    def test_effect_is_traceable_to_feedback_id(self) -> None:
        obs = observation(FeedbackKind.LIKED, target=track_target())
        effect = derive_learning_effect(
            interpret_observation(obs, INTERPRETATION_POLICY), EFFECT_POLICY
        )
        self.assertEqual(effect.feedback_id, obs.feedback_id)

    def test_effect_records_policy_and_contract_versions(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(effect.policy_version, 1)
        self.assertEqual(effect.contract_version, LEARNING_EFFECT_CONTRACT_VERSION)
        self.assertEqual(effect.interpretation.policy_version, 1)

    def test_effect_is_deterministic_for_interpretation_and_policy(self) -> None:
        first = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        second = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        self.assertEqual(first, second)

    def test_unknown_policy_version_fails_closed(self) -> None:
        with self.assertRaises(LearningEffectValidationError):
            derive_learning_effect(
                interpretation(FeedbackKind.LIKED, target=track_target()),
                LearningEffectPolicy(2),
            )

    def test_rejects_non_interpretation_input(self) -> None:
        for bad in ("not-an-interpretation", None, {"direction": "positive"}, 42):
            with self.subTest(bad=bad):
                with self.assertRaises(LearningEffectValidationError):
                    derive_learning_effect(bad, EFFECT_POLICY)  # type: ignore[arg-type]

    def test_rejects_non_policy_input(self) -> None:
        for bad in ("v1", None, 1, {"version": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(LearningEffectValidationError):
                    derive_learning_effect(
                        interpretation(FeedbackKind.LIKED, target=track_target()),
                        bad,  # type: ignore[arg-type]
                    )


class LearningEffectShapeTest(unittest.TestCase):
    def test_effect_carries_no_numeric_learning_surface(self) -> None:
        names = {field.name for field in fields(LearningEffect)}
        self.assertEqual(
            names,
            {
                "interpretation",
                "kind",
                "target",
                "reason",
                "policy_version",
                "contract_version",
            },
        )
        effect = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        for absent in (
            "weight",
            "magnitude",
            "delta",
            "confidence",
            "strength",
            "decay",
            "multiplier",
        ):
            self.assertFalse(hasattr(effect, absent), absent)

    def test_effect_is_immutable_and_hashable(self) -> None:
        effect = derive_learning_effect(
            interpretation(FeedbackKind.LIKED, target=track_target()), EFFECT_POLICY
        )
        with self.assertRaises(FrozenInstanceError):
            effect.kind = LearningEffectKind.NO_EFFECT
        self.assertEqual(hash(effect), hash(effect))

    def test_effect_policy_is_a_frozen_value(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            EFFECT_POLICY.version = 2

    def test_effect_policy_version_validation(self) -> None:
        with self.assertRaises(LearningEffectValidationError):
            LearningEffectPolicy(0)
        with self.assertRaises(LearningEffectValidationError):
            LearningEffectPolicy(True)

    def test_direction_is_fixed_by_kind(self) -> None:
        self.assertEqual(
            LearningEffect(
                interpretation=interpretation(
                    FeedbackKind.LIKED, target=track_target()
                ),
                kind=LearningEffectKind.ATTRIBUTION_EXCLUSION,
                target=track_target(),
                reason=LearningEffectReason.ATTRIBUTION_EXCLUDED,
                policy_version=1,
                contract_version=1,
            ).direction,
            FeedbackDirection.NONE,
        )
        self.assertEqual(
            derive_learning_effect(
                interpretation(FeedbackKind.DISLIKED, target=track_target()),
                EFFECT_POLICY,
            ).direction,
            FeedbackDirection.NEGATIVE,
        )


if __name__ == "__main__":
    unittest.main()
