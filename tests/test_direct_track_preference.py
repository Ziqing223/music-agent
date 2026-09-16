import unittest

from music_agent.direct_track_preference import (
    DirectPreferenceMagnitudePolicy,
    DirectPreferenceResolutionError,
    resolve_direct_track_preference,
)
from music_agent.preference_signal import (
    PreferenceSignal,
    RatingBandPolicy,
    SignalContribution,
    SignalDirection,
    SignalExplicitness,
    SignalReason,
    normalize_preference_signal,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.source_observation import ObservedValue


def policy(positive: float = 0.9, negative: float = 0.8) -> DirectPreferenceMagnitudePolicy:
    return DirectPreferenceMagnitudePolicy(positive_magnitude=positive, negative_magnitude=negative)


def favorited(value: bool) -> SignalContribution:
    return normalize_preference_signal(PreferenceSignal.FAVORITED, ObservedValue.value(value))


def disliked(value: bool) -> SignalContribution:
    return normalize_preference_signal(PreferenceSignal.DISLIKED, ObservedValue.value(value))


def rating(value: int) -> SignalContribution:
    return normalize_preference_signal(
        PreferenceSignal.RATING, ObservedValue.value(value), RatingBandPolicy(70, 30)
    )


def missing(signal: PreferenceSignal) -> SignalContribution:
    return normalize_preference_signal(signal, ObservedValue.missing())


class DirectionTest(unittest.TestCase):
    def test_favorited_true_is_positive(self) -> None:
        result = resolve_direct_track_preference([favorited(True)], policy())
        self.assertIs(result.state, PreferenceState.POSITIVE)
        self.assertEqual(result.magnitude, 0.9)

    def test_disliked_true_is_negative(self) -> None:
        result = resolve_direct_track_preference([disliked(True)], policy())
        self.assertIs(result.state, PreferenceState.NEGATIVE)
        self.assertEqual(result.magnitude, 0.8)

    def test_positive_rating_is_positive(self) -> None:
        result = resolve_direct_track_preference([rating(85)], policy())
        self.assertIs(result.state, PreferenceState.POSITIVE)
        self.assertEqual(result.magnitude, 0.9)

    def test_negative_rating_is_negative(self) -> None:
        result = resolve_direct_track_preference([rating(10)], policy())
        self.assertIs(result.state, PreferenceState.NEGATIVE)
        self.assertEqual(result.magnitude, 0.8)


class ConflictTest(unittest.TestCase):
    def test_favorited_and_disliked_true_is_conflict(self) -> None:
        result = resolve_direct_track_preference(
            [favorited(True), disliked(True)], policy()
        )
        self.assertIs(result.state, PreferenceState.CONFLICT)
        self.assertIsNone(result.magnitude)


class PriorityTest(unittest.TestCase):
    def test_favorited_wins_over_agreeing_rating_without_double_count(self) -> None:
        result = resolve_direct_track_preference(
            [favorited(True), rating(85)], policy()
        )
        self.assertIs(result.state, PreferenceState.POSITIVE)
        self.assertEqual(result.magnitude, 0.9)

    def test_favorited_wins_over_disagreeing_rating(self) -> None:
        result = resolve_direct_track_preference(
            [favorited(True), rating(10)], policy()
        )
        self.assertIs(result.state, PreferenceState.POSITIVE)
        self.assertEqual(result.magnitude, 0.9)

    def test_disliked_wins_over_disagreeing_rating(self) -> None:
        result = resolve_direct_track_preference(
            [disliked(True), rating(85)], policy()
        )
        self.assertIs(result.state, PreferenceState.NEGATIVE)
        self.assertEqual(result.magnitude, 0.8)

    def test_disliked_wins_over_agreeing_rating(self) -> None:
        result = resolve_direct_track_preference(
            [disliked(True), rating(10)], policy()
        )
        self.assertIs(result.state, PreferenceState.NEGATIVE)
        self.assertEqual(result.magnitude, 0.8)


class AbsenceClassificationTest(unittest.TestCase):
    def test_all_missing_is_unknown(self) -> None:
        result = resolve_direct_track_preference(
            [missing(PreferenceSignal.FAVORITED), missing(PreferenceSignal.DISLIKED), missing(PreferenceSignal.RATING)],
            policy(),
        )
        self.assertIs(result.state, PreferenceState.UNKNOWN)

    def test_empty_contributions_is_unknown(self) -> None:
        result = resolve_direct_track_preference([], policy())
        self.assertIs(result.state, PreferenceState.UNKNOWN)

    def test_rating_zero_only_is_unknown(self) -> None:
        result = resolve_direct_track_preference([rating(0)], policy())
        self.assertIs(result.state, PreferenceState.UNKNOWN)

    def test_null_observation_is_unknown(self) -> None:
        null = normalize_preference_signal(PreferenceSignal.FAVORITED, ObservedValue.null())
        result = resolve_direct_track_preference([null], policy())
        self.assertIs(result.state, PreferenceState.UNKNOWN)

    def test_favorited_false_and_disliked_false_is_insufficient(self) -> None:
        result = resolve_direct_track_preference(
            [favorited(False), disliked(False)], policy()
        )
        self.assertIs(result.state, PreferenceState.INSUFFICIENT)

    def test_favorited_false_only_is_insufficient(self) -> None:
        result = resolve_direct_track_preference([favorited(False)], policy())
        self.assertIs(result.state, PreferenceState.INSUFFICIENT)

    def test_middle_band_rating_is_insufficient(self) -> None:
        result = resolve_direct_track_preference([rating(50)], policy())
        self.assertIs(result.state, PreferenceState.INSUFFICIENT)


class NeutralTest(unittest.TestCase):
    def test_v1_never_produces_neutral(self) -> None:
        batteries = [
            [],
            [missing(PreferenceSignal.FAVORITED), missing(PreferenceSignal.DISLIKED), missing(PreferenceSignal.RATING)],
            [rating(0)],
            [favorited(False), disliked(False)],
            [favorited(False)],
            [rating(50)],
            [favorited(False), disliked(False), rating(0)],
        ]
        for contributions in batteries:
            with self.subTest(contributions=contributions):
                result = resolve_direct_track_preference(contributions, policy())
                self.assertIsNot(result.state, PreferenceState.NEUTRAL)


class CalibrationSeamTest(unittest.TestCase):
    def test_direction_stable_magnitude_varies_by_policy(self) -> None:
        high = policy(positive=0.9, negative=0.8)
        low = policy(positive=0.4, negative=0.3)
        high_result = resolve_direct_track_preference([favorited(True)], high)
        low_result = resolve_direct_track_preference([favorited(True)], low)
        self.assertIs(high_result.state, PreferenceState.POSITIVE)
        self.assertIs(low_result.state, PreferenceState.POSITIVE)
        self.assertEqual(high_result.magnitude, 0.9)
        self.assertEqual(low_result.magnitude, 0.4)
        self.assertNotEqual(high_result.magnitude, low_result.magnitude)

    def test_result_is_typed_preference_strength(self) -> None:
        result = resolve_direct_track_preference([favorited(True)], policy())
        self.assertIsInstance(result, PreferenceStrength)

    def test_magnitude_for_rejects_non_directional_state(self) -> None:
        for state in (PreferenceState.UNKNOWN, PreferenceState.NEUTRAL, PreferenceState.CONFLICT):
            with self.subTest(state=state):
                with self.assertRaises(DirectPreferenceResolutionError):
                    policy().magnitude_for(state)


class DeterminismTest(unittest.TestCase):
    def test_contribution_order_does_not_matter(self) -> None:
        baseline = [favorited(True), rating(85), disliked(False)]
        permutations = [
            [disliked(False), rating(85), favorited(True)],
            [rating(85), favorited(True), disliked(False)],
        ]
        expected = resolve_direct_track_preference(baseline, policy())
        for perm in permutations:
            with self.subTest(perm=perm):
                self.assertEqual(expected, resolve_direct_track_preference(perm, policy()))

    def test_repeated_call_is_equal(self) -> None:
        contributions = [favorited(True), rating(10), disliked(False)]
        first = resolve_direct_track_preference(contributions, policy())
        second = resolve_direct_track_preference(contributions, policy())
        self.assertEqual(first, second)


class _BrokenMagnitudePolicy(DirectPreferenceMagnitudePolicy):
    def magnitude_for(self, state: PreferenceState) -> float:
        return 2.0


class ValidationTest(unittest.TestCase):
    def test_duplicate_signal_fails_closed(self) -> None:
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([favorited(True), favorited(True)], policy())
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([favorited(True), favorited(False)], policy())

    def test_malformed_contribution_fails_closed(self) -> None:
        bad_favorited = SignalContribution(
            PreferenceSignal.FAVORITED,
            SignalDirection.NEGATIVE,
            ObservedValue.value(True),
            SignalExplicitness.EXPLICIT,
            SignalReason.EXPLICIT_SIGNAL,
        )
        bad_disliked = SignalContribution(
            PreferenceSignal.DISLIKED,
            SignalDirection.POSITIVE,
            ObservedValue.value(True),
            SignalExplicitness.EXPLICIT,
            SignalReason.EXPLICIT_SIGNAL,
        )
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([bad_favorited], policy())
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([bad_disliked], policy())

    def test_non_contribution_element_fails_closed(self) -> None:
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([favorited(True), "not a contribution"], policy())

    def test_non_iterable_contributions_fails_closed(self) -> None:
        for contributions in (None, favorited(True), 5):
            with self.subTest(contributions=contributions):
                with self.assertRaises(DirectPreferenceResolutionError):
                    resolve_direct_track_preference(contributions, policy())

    def test_invalid_magnitude_policy_type_fails_closed(self) -> None:
        for bad in (None, "policy", 0.9, RatingBandPolicy(70, 30)):
            with self.subTest(bad=bad):
                with self.assertRaises(DirectPreferenceResolutionError):
                    resolve_direct_track_preference([favorited(True)], bad)

    def test_invalid_policy_construction_fails_closed(self) -> None:
        invalid = [
            (1.5, 0.8),   # positive above 1
            (0.9, 0.0),   # negative at 0
            (0.9, -0.5),  # negative below 0
            (float("nan"), 0.8),
            (0.9, float("inf")),
            (True, 0.8),
            (0.9, "high"),
            (None, 0.8),
        ]
        for positive, negative in invalid:
            with self.subTest(positive=positive, negative=negative):
                with self.assertRaises(DirectPreferenceResolutionError):
                    DirectPreferenceMagnitudePolicy(positive_magnitude=positive, negative_magnitude=negative)

    def test_invalid_magnitude_policy_result_fails_closed(self) -> None:
        broken = _BrokenMagnitudePolicy(0.9, 0.8)
        with self.assertRaises(DirectPreferenceResolutionError):
            resolve_direct_track_preference([favorited(True)], broken)


if __name__ == "__main__":
    unittest.main()
