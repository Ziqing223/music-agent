"""P08.5: the conservative learning policy proposing bounded preference updates (policy v2).

These tests prove the pure policy layer only: the corrected policy-v2 mapping from a categorical
:class:`LearningEffect` to a bounded :class:`ProposedPreferenceUpdate` (or ``None`` for
``NO_EFFECT``), the evidence-based proposal shape grounded in P06's ``SignalIdentity`` /
``ObservedValue`` invariants, the direction-carrying P06-frozen signal paths under the
``feedback_learning`` source system, the evidence-class provenance encoding that survives P06
application durably (explicit vs implicit), the attribution-exclusion directive that can never
become ordinary (negative) evidence, full traceability, fail-closed behavior for superseded/
unknown versions and unmapped kinds, and the complete absence of numeric magnitude proposals and
P06 writes. No persistence is touched and nothing mutates preference state.
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
    InterpretationReason,
    interpret_observation,
)
from music_agent.learning_effect import (
    LearningEffect,
    LearningEffectKind,
    LearningEffectReason,
    LearningEffectPolicy,
    derive_learning_effect,
)
from music_agent.learning_policy import (
    EXPLICIT_FEEDBACK_PROVENANCE,
    FEEDBACK_LEARNING_SOURCE_SYSTEM,
    IMPLICIT_FEEDBACK_PROVENANCE,
    LEARNING_POLICY_CONTRACT_VERSION,
    LearningPolicy,
    LearningPolicyValidationError,
    ProposedPreferenceUpdate,
    ProposedUpdateKind,
    propose_preference_update,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.source_observation import ObservedValue, ObservationState

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

INTERPRETATION_POLICY = InterpretationPolicy(1)
EFFECT_POLICY = LearningEffectPolicy(1)
POLICY = LearningPolicy(2)


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
    event_at: datetime | None = None,
) -> FeedbackObservation:
    return assemble_feedback_observation(
        feedback_id="fbk_44444444-4444-4444-8444-444444444444",
        kind=kind,
        source=source(),
        observed_at=NOW,
        target=target,
        recommendation=recommendation,
        attribution=attribution,
        event_at=event_at,
    )


def effect(
    kind: FeedbackKind,
    *,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
    event_at: datetime | None = None,
) -> LearningEffect:
    return derive_learning_effect(
        interpret_observation(
            observation(
                kind,
                target=target,
                recommendation=recommendation,
                attribution=attribution,
                event_at=event_at,
            ),
            INTERPRETATION_POLICY,
        ),
        EFFECT_POLICY,
    )


class ProposedUpdateMappingTest(unittest.TestCase):
    def test_liked_proposes_positive_evidence_on_the_favorited_path(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        self.assertEqual(proposal.kind, ProposedUpdateKind.EVIDENCE_OBSERVATION)
        self.assertEqual(proposal.signal_identity.target, track_target())
        self.assertEqual(
            proposal.signal_identity.source_system, FEEDBACK_LEARNING_SOURCE_SYSTEM
        )
        self.assertEqual(proposal.signal_identity.signal_path, "favorited")
        self.assertIs(proposal.proposed_value.state, ObservationState.VALUE)
        self.assertIs(proposal.proposed_value.payload, True)
        self.assertEqual(proposal.provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        self.assertEqual(proposal.observed_at, NOW.isoformat())
        self.assertIsNone(proposal.event_at)

    def test_disliked_proposes_negative_evidence_on_the_disliked_path(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.DISLIKED, target=track_target()), POLICY
        )
        self.assertEqual(proposal.kind, ProposedUpdateKind.EVIDENCE_OBSERVATION)
        self.assertEqual(proposal.signal_identity.signal_path, "disliked")
        self.assertEqual(proposal.provenance, EXPLICIT_FEEDBACK_PROVENANCE)

    def test_direction_good_proposes_positive_evidence_for_the_direction_target(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.DIRECTION_GOOD, target=genre_target()), POLICY
        )
        self.assertEqual(proposal.signal_identity.target, genre_target())
        self.assertEqual(proposal.signal_identity.signal_path, "favorited")
        self.assertEqual(proposal.provenance, EXPLICIT_FEEDBACK_PROVENANCE)

    def test_favorited_proposes_positive_evidence_with_implicit_provenance(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.FAVORITED, target=track_target()), POLICY
        )
        self.assertEqual(proposal.signal_identity.signal_path, "favorited")
        self.assertEqual(proposal.provenance, IMPLICIT_FEEDBACK_PROVENANCE)
        self.assertEqual(proposal.explicitness, FeedbackExplicitness.IMPLICIT)

    def test_replayed_proposes_positive_evidence_with_implicit_provenance(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.REPLAYED, target=track_target()), POLICY
        )
        self.assertEqual(proposal.signal_identity.signal_path, "favorited")
        self.assertEqual(proposal.provenance, IMPLICIT_FEEDBACK_PROVENANCE)

    def test_explicit_and_implicit_evidence_stay_distinguishable(self) -> None:
        explicit = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        implicit = propose_preference_update(
            effect(FeedbackKind.FAVORITED, target=track_target()), POLICY
        )
        self.assertEqual(explicit.signal_identity, implicit.signal_identity)
        self.assertNotEqual(explicit.provenance, implicit.provenance)
        self.assertEqual(explicit.explicitness, FeedbackExplicitness.EXPLICIT)
        self.assertEqual(implicit.explicitness, FeedbackExplicitness.IMPLICIT)

    def test_event_times_are_carried_for_the_future_observation(self) -> None:
        event_time = datetime(2026, 8, 15, 23, 0, 0, tzinfo=timezone.utc)
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target(), event_at=event_time),
            POLICY,
        )
        self.assertEqual(proposal.event_at, event_time.isoformat())


class NoEffectAndExclusionTest(unittest.TestCase):
    def test_no_effect_produces_no_proposed_mutation(self) -> None:
        for kind in (
            FeedbackKind.SKIPPED,
            FeedbackKind.COMPLETED,
            FeedbackKind.PLAYED,
            FeedbackKind.CORRECTED,
        ):
            with self.subTest(kind=kind):
                self.assertIsNone(
                    propose_preference_update(
                        effect(kind, target=track_target()), POLICY
                    )
                )

    def test_target_less_effect_produces_no_proposed_mutation(self) -> None:
        self.assertIsNone(
            propose_preference_update(
                effect(FeedbackKind.DIRECTION_GOOD, recommendation=recommendation_ref()),
                POLICY,
            )
        )

    def test_attribution_exclusion_is_never_ordinary_negative_evidence(self) -> None:
        proposal = propose_preference_update(
            effect(
                FeedbackKind.ATTRIBUTION_CORRECTION,
                target=track_target(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            ),
            POLICY,
        )
        self.assertEqual(proposal.kind, ProposedUpdateKind.ATTRIBUTION_EXCLUSION)
        self.assertIsNone(proposal.signal_identity)
        self.assertIsNone(proposal.proposed_value)
        self.assertIsNone(proposal.provenance)
        self.assertIsNone(proposal.observed_at)
        self.assertIsNone(proposal.event_at)
        self.assertEqual(proposal.attribution.aspect, artist_target())
        self.assertEqual(proposal.attribution.relation, AttributionRelation.EXCLUDED)
        self.assertEqual(proposal.target, track_target())

    def test_exclusion_survives_on_positive_evidence_proposals(self) -> None:
        proposal = propose_preference_update(
            effect(
                FeedbackKind.LIKED,
                target=track_target(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            ),
            POLICY,
        )
        self.assertEqual(proposal.kind, ProposedUpdateKind.EVIDENCE_OBSERVATION)
        self.assertEqual(proposal.attribution.relation, AttributionRelation.EXCLUDED)


class ProposedUpdateTraceabilityTest(unittest.TestCase):
    def test_proposal_carries_the_full_effect(self) -> None:
        e = effect(FeedbackKind.LIKED, target=track_target())
        proposal = propose_preference_update(e, POLICY)
        self.assertIs(proposal.effect, e)

    def test_proposal_is_traceable_to_feedback_id(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        self.assertEqual(
            proposal.feedback_id, "fbk_44444444-4444-4444-8444-444444444444"
        )

    def test_proposal_records_all_upstream_policy_versions(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        self.assertEqual(proposal.policy_version, 2)
        self.assertEqual(proposal.contract_version, LEARNING_POLICY_CONTRACT_VERSION)
        self.assertEqual(LEARNING_POLICY_CONTRACT_VERSION, 2)
        self.assertEqual(proposal.effect.policy_version, 1)
        self.assertEqual(proposal.effect.interpretation.policy_version, 1)

    def test_proposal_is_deterministic(self) -> None:
        first = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        second = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        self.assertEqual(first, second)


class ProposedUpdateFailClosedTest(unittest.TestCase):
    def test_superseded_policy_v1_fails_closed(self) -> None:
        with self.assertRaises(LearningPolicyValidationError):
            propose_preference_update(
                effect(FeedbackKind.LIKED, target=track_target()),
                LearningPolicy(1),
            )

    def test_unknown_policy_version_fails_closed(self) -> None:
        with self.assertRaises(LearningPolicyValidationError):
            propose_preference_update(
                effect(FeedbackKind.LIKED, target=track_target()),
                LearningPolicy(3),
            )

    def test_rejects_non_effect_input(self) -> None:
        for bad in ("not-an-effect", None, {"kind": "positive_evidence"}, 42):
            with self.subTest(bad=bad):
                with self.assertRaises(LearningPolicyValidationError):
                    propose_preference_update(bad, POLICY)  # type: ignore[arg-type]

    def test_rejects_non_policy_input(self) -> None:
        for bad in ("v2", None, 2, {"version": 2}):
            with self.subTest(bad=bad):
                with self.assertRaises(LearningPolicyValidationError):
                    propose_preference_update(
                        effect(FeedbackKind.LIKED, target=track_target()),
                        bad,  # type: ignore[arg-type]
                    )

    def test_observation_kind_outside_the_frozen_mapping_fails_closed(self) -> None:
        # A contract-violating upstream combination (an evidence effect derived from a kind
        # that has no evidence mapping) must raise, never propose a guessed signal path.
        exclusion = FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED)
        crafted_interpretation = FeedbackInterpretation(
            observation=observation(
                FeedbackKind.ATTRIBUTION_CORRECTION,
                target=track_target(),
                attribution=exclusion,
            ),
            direction=FeedbackDirection.POSITIVE,
            reason=InterpretationReason.EXPLICIT_STATEMENT,
            attribution=exclusion,
            policy_version=1,
            contract_version=1,
        )
        crafted_effect = LearningEffect(
            interpretation=crafted_interpretation,
            kind=LearningEffectKind.POSITIVE_EVIDENCE,
            target=track_target(),
            reason=LearningEffectReason.EXPLICIT_EVIDENCE,
            policy_version=1,
            contract_version=1,
        )
        with self.assertRaises(LearningPolicyValidationError):
            propose_preference_update(crafted_effect, POLICY)


class ProposedUpdateShapeTest(unittest.TestCase):
    def test_proposal_carries_no_magnitude_or_delta_surface(self) -> None:
        names = {field.name for field in fields(ProposedPreferenceUpdate)}
        self.assertEqual(
            names,
            {
                "effect",
                "kind",
                "signal_identity",
                "proposed_value",
                "provenance",
                "observed_at",
                "event_at",
                "policy_version",
                "contract_version",
            },
        )
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        for absent in (
            "proposed_strength",
            "magnitude",
            "delta",
            "weight",
            "confidence",
            "decay",
            "multiplier",
        ):
            self.assertFalse(hasattr(proposal, absent), absent)

    def test_proposal_is_immutable_and_hashable(self) -> None:
        proposal = propose_preference_update(
            effect(FeedbackKind.LIKED, target=track_target()), POLICY
        )
        with self.assertRaises(FrozenInstanceError):
            proposal.proposed_value = None
        self.assertEqual(hash(proposal), hash(proposal))

    def test_evidence_proposal_requires_identity_value_and_provenance(self) -> None:
        e = effect(FeedbackKind.LIKED, target=track_target())
        with self.assertRaises(LearningPolicyValidationError):
            ProposedPreferenceUpdate(
                effect=e, kind=ProposedUpdateKind.EVIDENCE_OBSERVATION
            )
        with self.assertRaises(LearningPolicyValidationError):
            ProposedPreferenceUpdate(
                effect=e,
                kind=ProposedUpdateKind.EVIDENCE_OBSERVATION,
                signal_identity=propose_preference_update(e, POLICY).signal_identity,
                proposed_value=ObservedValue.value(True),
                provenance="",
                observed_at=NOW.isoformat(),
            )

    def test_exclusion_proposal_rejects_evidence_parts(self) -> None:
        e = effect(
            FeedbackKind.ATTRIBUTION_CORRECTION,
            target=track_target(),
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED),
        )
        with self.assertRaises(LearningPolicyValidationError):
            ProposedPreferenceUpdate(
                effect=e,
                kind=ProposedUpdateKind.ATTRIBUTION_EXCLUSION,
                provenance=EXPLICIT_FEEDBACK_PROVENANCE,
            )

    def test_policy_is_a_frozen_value(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            POLICY.version = 3


if __name__ == "__main__":
    unittest.main()
