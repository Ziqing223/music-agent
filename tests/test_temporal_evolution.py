import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from music_agent.preference_signal import SignalDirection
from music_agent.temporal_evolution import (
    EvidenceInfluencePolicy,
    PreferenceEvidence,
    TemporalEvolutionValidationError,
    TemporalInterpretation,
    TemporalReason,
    TemporalRelation,
    TemporalScopePolicy,
    classify_temporal_relation,
)

UTC = timezone.utc


def aware(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def fact(
    direction: SignalDirection,
    event_at: datetime | None,
    *,
    observed_at: datetime | None = None,
    target: str = "trk_1",
) -> PreferenceEvidence:
    return PreferenceEvidence(
        target=target,
        direction=direction,
        observed_at=observed_at if observed_at is not None else aware(2024, 6, 1),
        event_at=event_at,
    )


class _FixedGapScopePolicy:
    """Contemporaneous iff two event times are within a fixed gap of each other.

    Ignores ``now`` on purpose: it exercises the pure chronological conflict/evolution split.
    """

    def __init__(self, gap: timedelta) -> None:
        self._gap = gap

    def contemporaneous(self, first: datetime, second: datetime, *, now: datetime) -> bool:
        return abs((first - second).total_seconds()) < self._gap.total_seconds()


class _NowAnchoredScopePolicy:
    """Contemporaneous iff both event times fall within ``window`` before ``now``."""

    def __init__(self, window: timedelta) -> None:
        self._window = window

    def contemporaneous(self, first: datetime, second: datetime, *, now: datetime) -> bool:
        first_recent = now - first <= self._window
        second_recent = now - second <= self._window
        return first_recent and second_recent


class _NonBoolScopePolicy:
    def contemporaneous(self, first: datetime, second: datetime, *, now: datetime) -> object:
        return "yes"  # type: ignore[return-value]


def fixed_gap(days: float = 7.0) -> _FixedGapScopePolicy:
    return _FixedGapScopePolicy(timedelta(days=days))


class ContemporaneousConflictTest(unittest.TestCase):
    def test_opposite_direction_within_scope_is_conflict(self) -> None:
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1, 12, 0, 0))
        negative = fact(SignalDirection.NEGATIVE, aware(2020, 1, 1, 13, 0, 0))
        result = classify_temporal_relation(
            positive, negative, now=aware(2020, 1, 2), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)
        self.assertIs(result.reason, TemporalReason.CONTEMPORANEOUS_OPPOSITE)

    def test_opposite_direction_any_argument_order_is_conflict(self) -> None:
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1, 12, 0, 0))
        negative = fact(SignalDirection.NEGATIVE, aware(2020, 1, 1, 13, 0, 0))
        forward = classify_temporal_relation(
            positive, negative, now=aware(2020, 1, 2), scope_policy=fixed_gap()
        )
        backward = classify_temporal_relation(
            negative, positive, now=aware(2020, 1, 2), scope_policy=fixed_gap()
        )
        self.assertIs(forward.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)
        self.assertIs(backward.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)


class TemporalEvolutionTest(unittest.TestCase):
    def test_historical_positive_then_later_negative_is_evolution(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        later = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.TEMPORAL_EVOLUTION)
        self.assertIs(result.reason, TemporalReason.TEMPORALLY_ORDERED_OPPOSITE)

    def test_historical_negative_then_later_positive_is_evolution(self) -> None:
        historical = fact(SignalDirection.NEGATIVE, aware(2020, 1, 1))
        later = fact(SignalDirection.POSITIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.TEMPORAL_EVOLUTION)

    def test_evolution_is_never_a_hard_conflict(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        later = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIsNot(result.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)
        self.assertIsNot(result.reason, TemporalReason.CONTEMPORANEOUS_OPPOSITE)


class NoTemporalContradictionTest(unittest.TestCase):
    def test_same_positive_direction_is_no_contradiction(self) -> None:
        first = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        second = fact(SignalDirection.POSITIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            first, second, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.NO_TEMPORAL_CONTRADICTION)
        self.assertIs(result.reason, TemporalReason.SAME_DIRECTION)

    def test_same_negative_direction_is_no_contradiction(self) -> None:
        first = fact(SignalDirection.NEGATIVE, aware(2020, 1, 1))
        second = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            first, second, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.NO_TEMPORAL_CONTRADICTION)

    def test_no_claim_fact_is_no_contradiction(self) -> None:
        no_claim = fact(SignalDirection.NO_CLAIM, aware(2024, 6, 1))
        positive = fact(SignalDirection.POSITIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            no_claim, positive, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.NO_TEMPORAL_CONTRADICTION)
        self.assertIs(result.reason, TemporalReason.NO_DIRECTIONAL_CLAIM)


class ObservedAtEventAtDistinctionTest(unittest.TestCase):
    def test_event_at_not_observed_at_decides_evolution(self) -> None:
        # Events are far apart (evolution), but observed_at values are close together.
        positive = fact(
            SignalDirection.POSITIVE,
            aware(2020, 1, 1),
            observed_at=aware(2024, 6, 1, 12, 0, 0),
        )
        negative = fact(
            SignalDirection.NEGATIVE,
            aware(2024, 6, 1),
            observed_at=aware(2024, 6, 1, 13, 0, 0),
        )
        result = classify_temporal_relation(
            positive, negative, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.TEMPORAL_EVOLUTION)

    def test_event_at_not_observed_at_decides_conflict(self) -> None:
        # Events are close together (conflict), but observed_at values are far apart.
        positive = fact(
            SignalDirection.POSITIVE,
            aware(2024, 6, 1, 12, 0, 0),
            observed_at=aware(2020, 1, 1),
        )
        negative = fact(
            SignalDirection.NEGATIVE,
            aware(2024, 6, 1, 13, 0, 0),
            observed_at=aware(2024, 6, 2),
        )
        result = classify_temporal_relation(
            positive, negative, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)

    def test_observed_at_is_preserved_distinct_from_event_at(self) -> None:
        positive = fact(
            SignalDirection.POSITIVE,
            aware(2020, 1, 1),
            observed_at=aware(2020, 1, 5),
        )
        self.assertEqual(positive.observed_at, aware(2020, 1, 5))
        self.assertEqual(positive.event_at, aware(2020, 1, 1))
        self.assertNotEqual(positive.observed_at, positive.event_at)


class UnknownEventAtTest(unittest.TestCase):
    def test_unknown_event_at_on_one_side_is_indeterminate(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        later_unknown = fact(SignalDirection.NEGATIVE, None, observed_at=aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later_unknown, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.INDETERMINATE)
        self.assertIs(result.reason, TemporalReason.UNKNOWN_EVENT_TIME)

    def test_both_unknown_event_at_is_indeterminate(self) -> None:
        first = fact(SignalDirection.POSITIVE, None)
        second = fact(SignalDirection.NEGATIVE, None)
        result = classify_temporal_relation(
            first, second, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.INDETERMINATE)

    def test_unknown_event_at_is_never_misclassified_as_conflict_or_evolution(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        later_unknown = fact(SignalDirection.NEGATIVE, None, observed_at=aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later_unknown, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIsNot(result.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)
        self.assertIsNot(result.relation, TemporalRelation.TEMPORAL_EVOLUTION)

    def test_observed_at_ordering_is_not_substituted_for_unknown_event_at(self) -> None:
        # observed_at proves the negative was observed later, but event_at is unknown, so the
        # temporal ordering of the events themselves cannot be established.
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1), observed_at=aware(2020, 1, 5))
        later_unknown = fact(SignalDirection.NEGATIVE, None, observed_at=aware(2024, 6, 1))
        result = classify_temporal_relation(
            historical, later_unknown, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.relation, TemporalRelation.INDETERMINATE)


class InjectedNowDeterminismTest(unittest.TestCase):
    def test_same_inputs_and_now_are_deterministic(self) -> None:
        historical = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        later = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        first = classify_temporal_relation(
            historical, later, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        second = classify_temporal_relation(
            historical, later, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertEqual(first, second)

    def test_result_records_the_injected_now(self) -> None:
        injected_now = aware(2024, 7, 1, 3, 4, 5)
        result = classify_temporal_relation(
            fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
            fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
            now=injected_now,
            scope_policy=fixed_gap(),
        )
        self.assertEqual(result.now, injected_now)

    def test_result_depends_on_injected_now_via_policy(self) -> None:
        policy = _NowAnchoredScopePolicy(timedelta(days=30))
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        negative = fact(SignalDirection.NEGATIVE, aware(2020, 1, 2))

        near = classify_temporal_relation(
            positive, negative, now=aware(2020, 1, 3), scope_policy=policy
        )
        far = classify_temporal_relation(
            positive, negative, now=aware(2024, 6, 1), scope_policy=policy
        )
        self.assertIs(near.relation, TemporalRelation.CONTEMPORANEOUS_CONFLICT)
        self.assertIs(far.relation, TemporalRelation.TEMPORAL_EVOLUTION)

    def test_now_is_required_and_injected(self) -> None:
        with self.assertRaises(TypeError):
            classify_temporal_relation(
                fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
                fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
                scope_policy=fixed_gap(),
            )

    def test_classification_never_reads_the_system_clock(self) -> None:
        import inspect

        import music_agent.temporal_evolution as te

        source = inspect.getsource(te)
        for banned in (
            "datetime.now(",
            "datetime.utcnow(",
            "datetime.today(",
            "time.time(",
            "time.monotonic(",
        ):
            self.assertNotIn(banned, source, banned)


class ImmutableInputTest(unittest.TestCase):
    def test_fact_is_frozen(self) -> None:
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        with self.assertRaises(FrozenInstanceError):
            positive.event_at = aware(2024, 6, 1)  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            positive.direction = SignalDirection.NEGATIVE  # type: ignore[misc]

    def test_classification_does_not_mutate_inputs(self) -> None:
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        negative = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        positive_before = PreferenceEvidence(
            positive.target, positive.direction, positive.observed_at, positive.event_at
        )
        negative_before = PreferenceEvidence(
            negative.target, negative.direction, negative.observed_at, negative.event_at
        )
        classify_temporal_relation(
            positive, negative, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertEqual(positive, positive_before)
        self.assertEqual(negative, negative_before)

    def test_interpretation_carries_the_original_fact_objects(self) -> None:
        positive = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        negative = fact(SignalDirection.NEGATIVE, aware(2024, 6, 1))
        result = classify_temporal_relation(
            positive, negative, now=aware(2024, 7, 1), scope_policy=fixed_gap()
        )
        self.assertIs(result.first, positive)
        self.assertIs(result.second, negative)

    def test_interpretation_is_frozen(self) -> None:
        result = classify_temporal_relation(
            fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
            fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
            now=aware(2024, 7, 1),
            scope_policy=fixed_gap(),
        )
        with self.assertRaises(FrozenInstanceError):
            result.relation = TemporalRelation.INDETERMINATE  # type: ignore[misc]


class NoFrozenDecayTest(unittest.TestCase):
    def test_module_ships_no_decay_or_window_constants_or_operator(self) -> None:
        import music_agent.temporal_evolution as te

        for name in vars(te):
            lowered = name.lower()
            for banned in (
                "decay",
                "half_life",
                "halflife",
                "half-life",
                "exponential",
                "window",
                "lambda",
            ):
                self.assertNotIn(banned, lowered, name)

    def test_interpretation_is_categorical_not_decayed(self) -> None:
        result = classify_temporal_relation(
            fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
            fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
            now=aware(2024, 7, 1),
            scope_policy=fixed_gap(),
        )
        self.assertIsInstance(result.relation, TemporalRelation)
        for field in ("influence", "weight", "decay", "score", "recency"):
            self.assertFalse(hasattr(result, field), field)

    def test_policies_are_abstract_seams_with_no_concrete_implementation(self) -> None:
        for protocol in (TemporalScopePolicy, EvidenceInfluencePolicy):
            with self.subTest(protocol=protocol.__name__):
                with self.assertRaises(TypeError):
                    protocol()  # type: ignore[misc]


class ValidationTest(unittest.TestCase):
    def test_different_target_fails_closed(self) -> None:
        with self.assertRaises(TemporalEvolutionValidationError):
            classify_temporal_relation(
                fact(SignalDirection.POSITIVE, aware(2020, 1, 1), target="trk_1"),
                fact(SignalDirection.NEGATIVE, aware(2024, 6, 1), target="trk_2"),
                now=aware(2024, 7, 1),
                scope_policy=fixed_gap(),
            )

    def test_non_evidence_fails_closed(self) -> None:
        valid = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        for bad in (None, "fact", 5, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    classify_temporal_relation(
                        bad, valid, now=aware(2024, 7, 1), scope_policy=fixed_gap()
                    )
                with self.assertRaises(TemporalEvolutionValidationError):
                    classify_temporal_relation(
                        valid, bad, now=aware(2024, 7, 1), scope_policy=fixed_gap()
                    )

    def test_invalid_scope_policy_fails_closed(self) -> None:
        for bad in (None, "policy", 0.9, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    classify_temporal_relation(
                        fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
                        fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
                        now=aware(2024, 7, 1),
                        scope_policy=bad,  # type: ignore[arg-type]
                    )

    def test_non_bool_scope_policy_result_fails_closed(self) -> None:
        with self.assertRaises(TemporalEvolutionValidationError):
            classify_temporal_relation(
                fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
                fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
                now=aware(2024, 7, 1),
                scope_policy=_NonBoolScopePolicy(),
            )

    def test_naive_datetime_fails_closed(self) -> None:
        naive = datetime(2020, 1, 1)
        with self.assertRaises(TemporalEvolutionValidationError):
            PreferenceEvidence(
                "trk_1", SignalDirection.POSITIVE, naive, naive
            )
        with self.assertRaises(TemporalEvolutionValidationError):
            PreferenceEvidence(
                "trk_1", SignalDirection.POSITIVE, aware(2020, 1, 1), naive
            )
        with self.assertRaises(TemporalEvolutionValidationError):
            classify_temporal_relation(
                fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
                fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
                now=naive,
                scope_policy=fixed_gap(),
            )

    def test_non_datetime_observed_at_fails_closed(self) -> None:
        for bad in ("2020-01-01", 5, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    PreferenceEvidence("trk_1", SignalDirection.POSITIVE, bad)  # type: ignore[arg-type]

    def test_non_datetime_event_at_fails_closed(self) -> None:
        for bad in ("2020-01-01", 5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    PreferenceEvidence(
                        "trk_1", SignalDirection.POSITIVE, aware(2020, 1, 1), bad  # type: ignore[arg-type]
                    )

    def test_invalid_target_fails_closed(self) -> None:
        for bad in ("", 5, None):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    PreferenceEvidence(
                        bad, SignalDirection.POSITIVE, aware(2020, 1, 1)  # type: ignore[arg-type]
                    )

    def test_invalid_direction_fails_closed(self) -> None:
        for bad in ("positive", 1, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TemporalEvolutionValidationError):
                    PreferenceEvidence(
                        "trk_1", bad, aware(2020, 1, 1)  # type: ignore[arg-type]
                    )

    def test_invalid_interpretation_fields_fail_closed(self) -> None:
        valid = fact(SignalDirection.POSITIVE, aware(2020, 1, 1))
        with self.assertRaises(TemporalEvolutionValidationError):
            TemporalInterpretation("evolution", TemporalReason.SAME_DIRECTION, valid, valid, aware(2024, 7, 1))  # type: ignore[arg-type]
        with self.assertRaises(TemporalEvolutionValidationError):
            TemporalInterpretation(TemporalRelation.TEMPORAL_EVOLUTION, "same", valid, valid, aware(2024, 7, 1))  # type: ignore[arg-type]

    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(TemporalEvolutionValidationError, ValueError))

    def test_equality_and_hash(self) -> None:
        def interpret() -> TemporalInterpretation:
            return classify_temporal_relation(
                fact(SignalDirection.POSITIVE, aware(2020, 1, 1)),
                fact(SignalDirection.NEGATIVE, aware(2024, 6, 1)),
                now=aware(2024, 7, 1),
                scope_policy=fixed_gap(),
            )

        self.assertEqual(interpret(), interpret())
        self.assertEqual(hash(interpret()), hash(interpret()))


if __name__ == "__main__":
    unittest.main()
