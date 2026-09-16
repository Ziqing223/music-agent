import dataclasses
import inspect
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from music_agent.confidence import (
    ClaimScope,
    ConfidenceClaim,
    ConfidenceComponents,
    Contradiction,
)
from music_agent.preference_attribution import (
    AttributionConstraint,
    ConstraintMode,
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_explainability import (
    ConflictKind,
    ContributingSignal,
    DerivedPreferenceExplanation,
    DerivationKind,
    ExplainabilityValidationError,
    PreferenceConflict,
)
from music_agent.preference_propagation import (
    AttributionConstraintBinding,
    InferredAffinityContribution,
    PropagationKind,
)
from music_agent.preference_signal import (
    PreferenceSignal,
    SignalContribution,
    SignalDirection,
    SignalExplicitness,
    SignalReason,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.source_observation import ObservedValue
from music_agent.temporal_evolution import (
    PreferenceEvidence,
    TemporalInterpretation,
    TemporalReason,
    TemporalRelation,
)

TRACK_ID = "trk_00000000-0000-4000-8000-000000000000"
TRACK_2 = "trk_00000000-0000-4000-8000-000000000001"
ARTIST_A = "art_00000000-0000-4000-8000-000000000001"
ALBUM_ID = "alb_00000000-0000-4000-8000-000000000000"

UTC = timezone.utc


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def other_track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_2)


def ref(kind: PreferenceTargetKind, target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(kind, target_id)


def artist_target() -> PreferenceTargetReference:
    return ref(PreferenceTargetKind.ARTIST, ARTIST_A)


def album_target() -> PreferenceTargetReference:
    return ref(PreferenceTargetKind.ALBUM, ALBUM_ID)


def genre_target(key: str = "rock") -> PreferenceTargetReference:
    return ref(PreferenceTargetKind.GENRE, key)


def strength(state: PreferenceState, magnitude: float | None = None) -> PreferenceStrength:
    return PreferenceStrength(state, magnitude)


def direct(state: PreferenceState, magnitude: float | None = None) -> DerivedPreference:
    return DerivedPreference(track_target(), strength(state, magnitude))


def contribution(
    *,
    source_track: PreferenceTargetReference | None = None,
    target: PreferenceTargetReference | None = None,
    direction: SignalDirection = SignalDirection.POSITIVE,
    input_magnitude: float = 0.9,
    derived_magnitude: float = 0.9,
    kind: PropagationKind = PropagationKind.ARTIST,
    constraint: ConstraintMode | None = None,
    split_count: int = 1,
) -> InferredAffinityContribution:
    return InferredAffinityContribution(
        source_track=source_track if source_track is not None else track_target(),
        target=target if target is not None else artist_target(),
        direction=direction,
        input_magnitude=input_magnitude,
        derived_magnitude=derived_magnitude,
        kind=kind,
        constraint=constraint,
        split_count=split_count,
    )


def components(**overrides) -> ConfidenceComponents:
    base = dict(
        quality=1.0,
        quantity=3,
        freshness=0.5,
        consistency=1.0,
        contradiction=Contradiction.NONE,
        source_reliability=0.9,
        inference_distance=0.0,
    )
    base.update(overrides)
    return ConfidenceComponents(**base)


def claim(scope: ClaimScope) -> ConfidenceClaim:
    return ConfidenceClaim(scope, components())


def observed_signal(
    signal: PreferenceSignal,
    direction: SignalDirection,
    *,
    explicitness: SignalExplicitness = SignalExplicitness.EXPLICIT,
    reason: SignalReason = SignalReason.EXPLICIT_SIGNAL,
) -> SignalContribution:
    return SignalContribution(signal, direction, ObservedValue.value(True), explicitness, reason)


def contributing_signal(
    signal: PreferenceSignal,
    direction: SignalDirection,
    source_target: PreferenceTargetReference | None = None,
) -> ContributingSignal:
    return ContributingSignal(
        observed_signal(signal, direction),
        source_target if source_target is not None else track_target(),
    )


def aware(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def fact(direction: SignalDirection, event_at: datetime | None) -> PreferenceEvidence:
    return PreferenceEvidence(TRACK_ID, direction, aware(2024, 1, 1), event_at)


class DerivationKindTest(unittest.TestCase):
    def test_kinds_are_stable_and_distinct(self) -> None:
        expected = {
            DerivationKind.DIRECT: "direct",
            DerivationKind.INFERRED: "inferred",
            DerivationKind.EFFECTIVE: "effective",
        }
        self.assertEqual(list(DerivationKind), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)
        self.assertEqual(len(set(DerivationKind)), 3)


class ConflictKindTest(unittest.TestCase):
    def test_kinds_are_stable_and_distinct(self) -> None:
        expected = {
            ConflictKind.DIRECT_CATEGORICAL: "direct_categorical",
            ConflictKind.DIRECT_VS_INFERRED: "direct_vs_inferred",
            ConflictKind.TEMPORAL_EVOLUTION: "temporal_evolution",
            ConflictKind.INDETERMINATE_TEMPORAL: "indeterminate_temporal",
        }
        self.assertEqual(list(ConflictKind), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)
        self.assertEqual(len(set(ConflictKind)), 4)


class ContributingSignalTest(unittest.TestCase):
    def test_binds_signal_to_source_track(self) -> None:
        signal = contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
        self.assertIs(signal.contribution.signal, PreferenceSignal.FAVORITED)
        self.assertIs(signal.contribution.direction, SignalDirection.POSITIVE)
        self.assertEqual(signal.source_target, track_target())

    def test_non_track_source_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            ContributingSignal(
                observed_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE),
                artist_target(),
            )

    def test_non_contribution_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            ContributingSignal("positive", track_target())  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        signal = contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
        with self.assertRaises(FrozenInstanceError):
            signal.source_target = other_track_target()  # type: ignore[misc]


class PreferenceConflictTest(unittest.TestCase):
    def test_opposite_directions_round_trip(self) -> None:
        conflict = PreferenceConflict(
            ConflictKind.DIRECT_CATEGORICAL,
            SignalDirection.POSITIVE,
            SignalDirection.NEGATIVE,
        )
        self.assertIs(conflict.kind, ConflictKind.DIRECT_CATEGORICAL)
        self.assertEqual(
            {conflict.first_direction, conflict.second_direction},
            {SignalDirection.POSITIVE, SignalDirection.NEGATIVE},
        )

    def test_same_direction_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            PreferenceConflict(
                ConflictKind.DIRECT_CATEGORICAL,
                SignalDirection.POSITIVE,
                SignalDirection.POSITIVE,
            )

    def test_no_claim_direction_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            PreferenceConflict(
                ConflictKind.DIRECT_CATEGORICAL,
                SignalDirection.POSITIVE,
                SignalDirection.NO_CLAIM,
            )

    def test_optional_provenance_round_trips(self) -> None:
        conflict = PreferenceConflict(
            ConflictKind.DIRECT_VS_INFERRED,
            SignalDirection.NEGATIVE,
            SignalDirection.POSITIVE,
            first_source=track_target(),
            second_source=artist_target(),
        )
        self.assertEqual(conflict.first_source, track_target())
        self.assertEqual(conflict.second_source, artist_target())

    def test_frozen(self) -> None:
        conflict = PreferenceConflict(
            ConflictKind.DIRECT_CATEGORICAL,
            SignalDirection.POSITIVE,
            SignalDirection.NEGATIVE,
        )
        with self.assertRaises(FrozenInstanceError):
            conflict.kind = ConflictKind.TEMPORAL_EVOLUTION  # type: ignore[misc]


class DirectExplanationTest(unittest.TestCase):
    def test_direct_positive_explanation(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.POSITIVE, 0.9),
        )
        self.assertIs(explanation.derivation, DerivationKind.DIRECT)
        self.assertIs(explanation.result.state, PreferenceState.POSITIVE)
        self.assertEqual(explanation.result.magnitude, 0.9)
        self.assertEqual(explanation.direct_preference.target, track_target())
        self.assertEqual(explanation.direct_preference.target, explanation.target)
        self.assertEqual(explanation.inferred_contributions, ())

    def test_direct_negative_explanation(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.NEGATIVE, 0.8),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.NEGATIVE, 0.8),
        )
        self.assertIs(explanation.result.state, PreferenceState.NEGATIVE)
        self.assertEqual(explanation.result.magnitude, 0.8)
        self.assertEqual(explanation.direct_preference.provenance.value, "direct")

    def test_direct_requires_direct_preference(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
            )

    def test_direct_result_must_match_direct_strength(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.5),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            )

    def test_direct_cannot_carry_inferred_contributions(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                inferred_contributions=[contribution()],
            )


class UnknownInsufficientTest(unittest.TestCase):
    def test_unknown_has_no_confidence_and_no_score(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.UNKNOWN),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.UNKNOWN),
        )
        self.assertIs(explanation.result.state, PreferenceState.UNKNOWN)
        self.assertIsNone(explanation.result.magnitude)
        self.assertEqual(explanation.confidence_claims, ())
        for field in ("score", "confidence", "overall", "total"):
            self.assertFalse(hasattr(explanation, field), field)

    def test_insufficient_has_no_confidence_and_no_score(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.INSUFFICIENT),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.INSUFFICIENT),
        )
        self.assertEqual(explanation.confidence_claims, ())
        self.assertFalse(hasattr(explanation, "score"))


class ConflictExplanationTest(unittest.TestCase):
    def test_conflict_preserves_both_sides(self) -> None:
        favorited = contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
        disliked = contributing_signal(PreferenceSignal.DISLIKED, SignalDirection.NEGATIVE)
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.CONFLICT),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.CONFLICT),
            signals=[favorited, disliked],
            conflicts=[
                PreferenceConflict(
                    ConflictKind.DIRECT_CATEGORICAL,
                    SignalDirection.POSITIVE,
                    SignalDirection.NEGATIVE,
                )
            ],
        )
        self.assertIs(explanation.result.state, PreferenceState.CONFLICT)
        directions = {signal.contribution.direction for signal in explanation.signals}
        self.assertEqual(
            directions, {SignalDirection.POSITIVE, SignalDirection.NEGATIVE}
        )
        self.assertIs(explanation.conflicts[0].kind, ConflictKind.DIRECT_CATEGORICAL)


class InferredExplanationTest(unittest.TestCase):
    def test_inferred_artist_from_track_contribution(self) -> None:
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[contribution(kind=PropagationKind.ARTIST)],
        )
        self.assertIs(explanation.derivation, DerivationKind.INFERRED)
        self.assertIsNone(explanation.direct_preference)
        self.assertEqual(len(explanation.inferred_contributions), 1)
        self.assertIs(
            explanation.inferred_contributions[0].kind, PropagationKind.ARTIST
        )
        self.assertEqual(
            explanation.inferred_contributions[0].target, artist_target()
        )

    def test_inferred_album(self) -> None:
        explanation = DerivedPreferenceExplanation(
            album_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.INFERRED,
            inferred_contributions=[
                contribution(target=album_target(), kind=PropagationKind.ALBUM)
            ],
        )
        self.assertIs(
            explanation.inferred_contributions[0].kind, PropagationKind.ALBUM
        )
        self.assertEqual(explanation.inferred_contributions[0].target, album_target())

    def test_inferred_genre(self) -> None:
        explanation = DerivedPreferenceExplanation(
            genre_target(),
            strength(PreferenceState.NEGATIVE, 0.3),
            DerivationKind.INFERRED,
            inferred_contributions=[
                contribution(
                    target=genre_target(),
                    direction=SignalDirection.NEGATIVE,
                    kind=PropagationKind.GENRE,
                )
            ],
        )
        self.assertIs(
            explanation.inferred_contributions[0].kind, PropagationKind.GENRE
        )
        self.assertIs(
            explanation.inferred_contributions[0].direction, SignalDirection.NEGATIVE
        )

    def test_inferred_requires_a_contribution(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.INFERRED,
            )

    def test_inferred_cannot_carry_direct_preference(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.INFERRED,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                inferred_contributions=[contribution()],
            )

    def test_inferred_preserves_source_track_and_magnitudes(self) -> None:
        contribution_ = contribution(
            source_track=other_track_target(),
            input_magnitude=0.9,
            derived_magnitude=0.45,
            kind=PropagationKind.ARTIST,
        )
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[contribution_],
        )
        stored = explanation.inferred_contributions[0]
        self.assertEqual(stored.source_track, other_track_target())
        self.assertEqual(stored.input_magnitude, 0.9)
        self.assertEqual(stored.derived_magnitude, 0.45)
        self.assertEqual(stored.split_count, 1)


class AttributionVisibilityTest(unittest.TestCase):
    def test_discount_is_visible_on_contribution_and_binding(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), artist_target(), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[
                contribution(constraint=ConstraintMode.DISCOUNT, derived_magnitude=0.45)
            ],
            attribution_constraints=[binding],
        )
        self.assertIs(
            explanation.inferred_contributions[0].constraint, ConstraintMode.DISCOUNT
        )
        self.assertIs(
            explanation.attribution_constraints[0].constraint.mode, ConstraintMode.DISCOUNT
        )

    def test_block_is_visible_even_when_it_removed_a_path(self) -> None:
        # Track 2 contributes positively; Track 1's path is BLOCKed, so it has no
        # contribution of its own, but the BLOCK constraint remains visible.
        blocked = AttributionConstraintBinding(
            track_target(), artist_target(), AttributionConstraint(ConstraintMode.BLOCK)
        )
        surviving = contribution(source_track=other_track_target(), kind=PropagationKind.ARTIST)
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.INFERRED,
            inferred_contributions=[surviving],
            attribution_constraints=[blocked],
        )
        self.assertEqual([c.source_track for c in explanation.inferred_contributions], [other_track_target()])
        self.assertIs(
            explanation.attribution_constraints[0].constraint.mode, ConstraintMode.BLOCK
        )

    def test_direct_cannot_carry_attribution_constraints(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), artist_target(), AttributionConstraint(ConstraintMode.BLOCK)
        )
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                attribution_constraints=[binding],
            )


class TemporalExplanationTest(unittest.TestCase):
    def test_temporal_evolution_remains_distinct_from_conflict(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        current = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        interpretation = TemporalInterpretation(
            TemporalRelation.TEMPORAL_EVOLUTION,
            TemporalReason.TEMPORALLY_ORDERED_OPPOSITE,
            historical,
            current,
            aware(2024, 7, 1),
        )
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.NEGATIVE, 0.8),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.NEGATIVE, 0.8),
            temporal_interpretation=interpretation,
            conflicts=[
                PreferenceConflict(
                    ConflictKind.TEMPORAL_EVOLUTION,
                    SignalDirection.POSITIVE,
                    SignalDirection.NEGATIVE,
                )
            ],
        )
        self.assertIs(
            explanation.temporal_interpretation.relation,
            TemporalRelation.TEMPORAL_EVOLUTION,
        )
        self.assertIsNot(
            explanation.temporal_interpretation.relation,
            TemporalRelation.CONTEMPORANEOUS_CONFLICT,
        )
        self.assertIs(explanation.conflicts[0].kind, ConflictKind.TEMPORAL_EVOLUTION)
        self.assertIsNot(explanation.conflicts[0].kind, ConflictKind.DIRECT_CATEGORICAL)

    def test_unknown_event_time_remains_indeterminate(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        unknown = fact(SignalDirection.NEGATIVE, None)
        interpretation = TemporalInterpretation(
            TemporalRelation.INDETERMINATE,
            TemporalReason.UNKNOWN_EVENT_TIME,
            historical,
            unknown,
            aware(2024, 7, 1),
        )
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.NEGATIVE, 0.8),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.NEGATIVE, 0.8),
            temporal_interpretation=interpretation,
        )
        self.assertIs(
            explanation.temporal_interpretation.relation, TemporalRelation.INDETERMINATE
        )
        self.assertIs(
            explanation.temporal_interpretation.reason, TemporalReason.UNKNOWN_EVENT_TIME
        )


class ConfidenceVisibilityTest(unittest.TestCase):
    def test_components_visible_without_aggregated_score(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            confidence_claims=[claim(ClaimScope.CURRENT_PREFERENCE)],
        )
        stored = explanation.confidence_claims[0]
        self.assertIs(stored.scope, ClaimScope.CURRENT_PREFERENCE)
        self.assertIsInstance(stored.components, ConfidenceComponents)
        self.assertEqual(stored.components.quality, 1.0)
        self.assertEqual(stored.components.quantity, 3)
        for field in ("score", "confidence", "aggregate", "overall", "value"):
            self.assertFalse(hasattr(stored, field), field)
            self.assertFalse(hasattr(explanation, field), field)


class DirectInferredSeparationTest(unittest.TestCase):
    def test_direct_and_inferred_are_never_flattened(self) -> None:
        direct_explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.POSITIVE, 0.9),
        )
        inferred_explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[contribution()],
        )
        self.assertIsNotNone(direct_explanation.direct_preference)
        self.assertEqual(direct_explanation.inferred_contributions, ())
        self.assertIsNone(inferred_explanation.direct_preference)
        self.assertEqual(len(inferred_explanation.inferred_contributions), 1)
        self.assertIsInstance(direct_explanation.direct_preference, DerivedPreference)
        self.assertIsInstance(
            inferred_explanation.inferred_contributions[0], InferredAffinityContribution
        )

    def test_fields_are_exactly_declared(self) -> None:
        expected = [
            "target",
            "result",
            "derivation",
            "direct_preference",
            "inferred_contributions",
            "confidence_claims",
            "temporal_interpretation",
            "conflicts",
            "attribution_constraints",
            "signals",
        ]
        self.assertEqual(
            [f.name for f in dataclasses.fields(DerivedPreferenceExplanation)], expected
        )


class InputImmutabilityTest(unittest.TestCase):
    def test_collection_inputs_are_copied_not_aliased(self) -> None:
        contributions = [contribution()]
        signals = [contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)]
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=contributions,
            signals=signals,
        )
        self.assertIsInstance(explanation.inferred_contributions, tuple)
        self.assertIsInstance(explanation.signals, tuple)
        contributions.clear()
        signals.clear()
        self.assertEqual(len(explanation.inferred_contributions), 1)
        self.assertEqual(len(explanation.signals), 1)

    def test_result_and_preference_inputs_are_not_mutated(self) -> None:
        result = strength(PreferenceState.POSITIVE, 0.9)
        preference = direct(PreferenceState.POSITIVE, 0.9)
        DerivedPreferenceExplanation(
            track_target(),
            result,
            DerivationKind.DIRECT,
            direct_preference=preference,
        )
        self.assertEqual(result, strength(PreferenceState.POSITIVE, 0.9))
        self.assertEqual(preference, direct(PreferenceState.POSITIVE, 0.9))

    def test_explanation_is_frozen(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.POSITIVE, 0.9),
        )
        with self.assertRaises(FrozenInstanceError):
            explanation.result = strength(PreferenceState.NEGATIVE, 0.9)  # type: ignore[misc]


class DeterminismAndEqualityTest(unittest.TestCase):
    def _build(self) -> DerivedPreferenceExplanation:
        return DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[
                contribution(source_track=track_target()),
                contribution(source_track=other_track_target()),
            ],
        )

    def test_equality_and_hash_are_deterministic(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_repr_is_deterministic(self) -> None:
        self.assertEqual(repr(self._build()), repr(self._build()))

    def test_ordering_is_preserved(self) -> None:
        explanation = self._build()
        self.assertEqual(
            [c.source_track for c in explanation.inferred_contributions],
            [track_target(), other_track_target()],
        )


class ValidationTest(unittest.TestCase):
    def test_direct_target_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                other_track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            )

    def test_inferred_target_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                album_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.INFERRED,
                inferred_contributions=[contribution()],  # targets ARTIST_A
            )

    def test_duplicate_inferred_contribution_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.INFERRED,
                inferred_contributions=[contribution(), contribution()],
            )

    def test_incompatible_confidence_scope_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                confidence_claims=[claim(ClaimScope.INFERRED_AFFINITY)],
            )

    def test_inferred_rejects_current_preference_scope(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.INFERRED,
                inferred_contributions=[contribution()],
                confidence_claims=[claim(ClaimScope.CURRENT_PREFERENCE)],
            )

    def test_unrelated_attribution_constraint_fails_closed(self) -> None:
        unrelated = AttributionConstraintBinding(
            track_target(), album_target(), AttributionConstraint(ConstraintMode.BLOCK)
        )
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.INFERRED,
                inferred_contributions=[contribution()],
                attribution_constraints=[unrelated],
            )

    def test_duplicate_contributing_signal_fails_closed(self) -> None:
        first = contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
        second = contributing_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                signals=[first, second],
            )

    def test_non_iterable_collection_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                signals=5,  # type: ignore[arg-type]
            )

    def test_string_is_not_a_collection(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                signals="positive",  # type: ignore[arg-type]
            )

    def test_wrong_element_type_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
                signals=[observed_signal(PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)],
            )

    def test_malformed_result_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                "positive",  # type: ignore[arg-type]
                DerivationKind.DIRECT,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            )

    def test_malformed_derivation_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                "direct",  # type: ignore[arg-type]
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            )

    def test_malformed_direct_preference_fails_closed(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                track_target(),
                strength(PreferenceState.POSITIVE, 0.9),
                DerivationKind.DIRECT,
                direct_preference="positive",  # type: ignore[arg-type]
            )

    def test_effective_requires_both_direct_and_inferred(self) -> None:
        with self.assertRaises(ExplainabilityValidationError):
            DerivedPreferenceExplanation(
                artist_target(),
                strength(PreferenceState.POSITIVE, 0.45),
                DerivationKind.EFFECTIVE,
                direct_preference=direct(PreferenceState.POSITIVE, 0.9),
            )

    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(ExplainabilityValidationError, ValueError))

    def test_error_code_is_stable(self) -> None:
        self.assertEqual(ExplainabilityValidationError.code, "validation_error")


class NoNaturalLanguageTest(unittest.TestCase):
    def test_module_has_no_prose_generator(self) -> None:
        import music_agent.preference_explainability as module

        for name in ("explain", "render", "describe", "to_text", "format", "summarize", "narrate"):
            self.assertNotIn(name, vars(module), name)

    def test_no_public_module_level_functions(self) -> None:
        import music_agent.preference_explainability as module

        locally_defined_public_functions = [
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and inspect.isfunction(value)
            and getattr(value, "__module__", None) == "music_agent.preference_explainability"
        ]
        self.assertEqual(locally_defined_public_functions, [])

    def test_explanation_exposes_no_text_accessor(self) -> None:
        explanation = DerivedPreferenceExplanation(
            track_target(),
            strength(PreferenceState.POSITIVE, 0.9),
            DerivationKind.DIRECT,
            direct_preference=direct(PreferenceState.POSITIVE, 0.9),
        )
        for name in ("explain", "render", "describe", "to_text", "text", "sentence"):
            self.assertFalse(hasattr(explanation, name), name)


class PurityTest(unittest.TestCase):
    def test_no_io_clock_random_or_repository_imports(self) -> None:
        import music_agent.preference_explainability as module

        forbidden = {
            "random",
            "time",
            "datetime",
            "os",
            "io",
            "sys",
            "pathlib",
            "json",
            "subprocess",
            "uuid",
            "sqlite3",
            "repository",
        }
        for name in forbidden:
            self.assertNotIn(name, vars(module), name)

    def test_module_reads_no_clock(self) -> None:
        import music_agent.preference_explainability as module

        source = inspect.getsource(module)
        for banned in (
            "datetime.now(",
            "datetime.utcnow(",
            "datetime.today(",
            "time.time(",
            "time.monotonic(",
        ):
            self.assertNotIn(banned, source, banned)

    def test_module_performs_no_persistence(self) -> None:
        import music_agent.preference_explainability as module

        source = inspect.getsource(module)
        for banned in (
            "open(",
            "sqlite",
            "connect(",
            "write(",
            "insert(",
            "music_agent.repository",
            "from music_agent.repository",
            "import music_agent.repository",
        ):
            self.assertNotIn(banned, source, banned)

    def test_explanation_has_no_aggregation_accessor(self) -> None:
        explanation = DerivedPreferenceExplanation(
            artist_target(),
            strength(PreferenceState.POSITIVE, 0.45),
            DerivationKind.INFERRED,
            inferred_contributions=[contribution()],
        )
        for name in ("aggregate", "combine", "reduce", "weight", "propagate", "classify"):
            self.assertFalse(hasattr(explanation, name), name)


if __name__ == "__main__":
    unittest.main()
