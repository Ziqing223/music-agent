import unittest

from music_agent.preference_signal import (
    PreferenceSignal,
    PreferenceSignalValidationError,
    RatingBandPolicy,
    SignalDirection,
    SignalExplicitness,
    SignalReason,
    normalize_preference_signal,
)
from music_agent.source_observation import ObservedValue, ObservationState


def policy(positive: int = 70, negative: int = 30) -> RatingBandPolicy:
    return RatingBandPolicy(positive_threshold=positive, negative_threshold=negative)


class PreferenceSignalTest(unittest.TestCase):
    # --- favorited ----------------------------------------------------------

    def test_favorited_true_is_positive_explicit(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.FAVORITED, ObservedValue.value(True)
        )
        self.assertIs(contribution.direction, SignalDirection.POSITIVE)
        self.assertIs(contribution.explicitness, SignalExplicitness.EXPLICIT)
        self.assertIs(contribution.reason, SignalReason.EXPLICIT_SIGNAL)
        self.assertEqual(contribution.raw, ObservedValue.value(True))

    def test_favorited_false_is_no_claim_not_negative(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.FAVORITED, ObservedValue.value(False)
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.NO_DIRECTIONAL_CLAIM)
        self.assertEqual(contribution.raw, ObservedValue.value(False))

    def test_favorited_missing_is_no_claim(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.FAVORITED, ObservedValue.missing()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.MISSING)

    def test_favorited_null_is_distinct_non_directional(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.FAVORITED, ObservedValue.null()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.NULL_VALUE)

    # --- disliked -----------------------------------------------------------

    def test_disliked_true_is_negative_explicit(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.DISLIKED, ObservedValue.value(True)
        )
        self.assertIs(contribution.direction, SignalDirection.NEGATIVE)
        self.assertIs(contribution.explicitness, SignalExplicitness.EXPLICIT)
        self.assertIs(contribution.reason, SignalReason.EXPLICIT_SIGNAL)

    def test_disliked_false_is_no_claim_not_positive(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.DISLIKED, ObservedValue.value(False)
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.NO_DIRECTIONAL_CLAIM)

    def test_disliked_missing_is_no_claim(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.DISLIKED, ObservedValue.missing()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.MISSING)

    def test_disliked_null_is_distinct_non_directional(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.DISLIKED, ObservedValue.null()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.NULL_VALUE)

    # --- rating -------------------------------------------------------------

    def test_rating_zero_is_ambiguous_source_value(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(0), policy()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.AMBIGUOUS_SOURCE_VALUE)
        self.assertIs(contribution.explicitness, SignalExplicitness.NONE)
        self.assertEqual(contribution.raw.payload, 0)

    def test_rating_zero_is_intercepted_before_policy(self) -> None:
        # A raw zero must not require a policy: it is intercepted first.
        contribution = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(0)
        )
        self.assertIs(contribution.reason, SignalReason.AMBIGUOUS_SOURCE_VALUE)

    def test_rating_positive_band_is_positive(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(85), policy()
        )
        self.assertIs(contribution.direction, SignalDirection.POSITIVE)
        self.assertIs(contribution.explicitness, SignalExplicitness.DERIVED)
        self.assertIs(contribution.reason, SignalReason.RATING_BAND)

    def test_rating_negative_band_is_negative(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(10), policy()
        )
        self.assertIs(contribution.direction, SignalDirection.NEGATIVE)
        self.assertIs(contribution.explicitness, SignalExplicitness.DERIVED)
        self.assertIs(contribution.reason, SignalReason.RATING_BAND)

    def test_rating_middle_band_is_no_claim(self) -> None:
        contribution = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(50), policy()
        )
        self.assertIs(contribution.direction, SignalDirection.NO_CLAIM)
        self.assertIs(contribution.reason, SignalReason.NO_DIRECTIONAL_CLAIM)

    def test_rating_band_thresholds_are_inclusive(self) -> None:
        self.assertIs(
            normalize_preference_signal(
                PreferenceSignal.RATING, ObservedValue.value(70), policy()
            ).direction,
            SignalDirection.POSITIVE,
        )
        self.assertIs(
            normalize_preference_signal(
                PreferenceSignal.RATING, ObservedValue.value(30), policy()
            ).direction,
            SignalDirection.NEGATIVE,
        )

    def test_rating_missing_and_null_are_no_claim(self) -> None:
        missing = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.missing(), policy()
        )
        null = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.null(), policy()
        )
        self.assertIs(missing.direction, SignalDirection.NO_CLAIM)
        self.assertIs(missing.reason, SignalReason.MISSING)
        self.assertIs(null.direction, SignalDirection.NO_CLAIM)
        self.assertIs(null.reason, SignalReason.NULL_VALUE)

    def test_rating_zero_null_missing_are_not_folded(self) -> None:
        zero = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.value(0), policy()
        )
        null = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.null(), policy()
        )
        missing = normalize_preference_signal(
            PreferenceSignal.RATING, ObservedValue.missing(), policy()
        )
        self.assertIs(zero.reason, SignalReason.AMBIGUOUS_SOURCE_VALUE)
        self.assertIs(null.reason, SignalReason.NULL_VALUE)
        self.assertIs(missing.reason, SignalReason.MISSING)
        self.assertIs(zero.raw.state, ObservationState.VALUE)
        self.assertIs(null.raw.state, ObservationState.NULL)
        self.assertIs(missing.raw.state, ObservationState.MISSING)
        self.assertNotEqual(zero.reason, null.reason)
        self.assertNotEqual(zero.reason, missing.reason)
        self.assertNotEqual(null.reason, missing.reason)

    # --- invalid data -------------------------------------------------------

    def test_rating_out_of_range_fails_closed(self) -> None:
        for rating in (101, -1):
            with self.subTest(rating=rating):
                with self.assertRaises(PreferenceSignalValidationError):
                    normalize_preference_signal(
                        PreferenceSignal.RATING, ObservedValue.value(rating), policy()
                    )

    def test_rating_non_integer_payload_fails_closed(self) -> None:
        for payload in ("high", 1.5, True):
            with self.subTest(payload=payload):
                with self.assertRaises(PreferenceSignalValidationError):
                    normalize_preference_signal(
                        PreferenceSignal.RATING, ObservedValue.value(payload), policy()
                    )

    def test_boolean_signal_non_bool_payload_fails_closed(self) -> None:
        for payload in (1, 0, "true"):
            with self.subTest(payload=payload):
                with self.assertRaises(PreferenceSignalValidationError):
                    normalize_preference_signal(
                        PreferenceSignal.FAVORITED, ObservedValue.value(payload)
                    )

    def test_invalid_policy_boundaries_fail_closed(self) -> None:
        invalid = (
            (70, 70),   # equal thresholds leave no bands
            (30, 70),   # negative above positive
            (101, 30),  # positive out of range
            (0, 30),    # positive below range
            (70, 0),    # negative below range
            (70, 101),  # negative out of range
            (True, 30), # bool positive
            (70, True), # bool negative
        )
        for positive, negative in invalid:
            with self.subTest(positive=positive, negative=negative):
                with self.assertRaises(PreferenceSignalValidationError):
                    RatingBandPolicy(positive_threshold=positive, negative_threshold=negative)

    def test_rating_without_policy_fails_closed(self) -> None:
        with self.assertRaises(PreferenceSignalValidationError):
            normalize_preference_signal(PreferenceSignal.RATING, ObservedValue.value(50))

    def test_policy_classify_rejects_ambiguous_zero(self) -> None:
        with self.assertRaises(PreferenceSignalValidationError):
            policy().classify(0)
        with self.assertRaises(PreferenceSignalValidationError):
            policy().classify(101)

    def test_invalid_signal_and_observed_fail_closed(self) -> None:
        with self.assertRaises(PreferenceSignalValidationError):
            normalize_preference_signal("favorited", ObservedValue.value(True))  # type: ignore[arg-type]
        with self.assertRaises(PreferenceSignalValidationError):
            normalize_preference_signal(PreferenceSignal.FAVORITED, True)  # type: ignore[arg-type]
        with self.assertRaises(PreferenceSignalValidationError):
            normalize_preference_signal(
                PreferenceSignal.FAVORITED, ObservedValue.value(True), rating_policy="high"  # type: ignore[arg-type]
            )

    # --- purity -------------------------------------------------------------

    def test_normalization_is_pure_and_deterministic(self) -> None:
        observed = ObservedValue.value(50)
        first = normalize_preference_signal(PreferenceSignal.RATING, observed, policy())
        second = normalize_preference_signal(PreferenceSignal.RATING, observed, policy())
        self.assertEqual(first, second)
        self.assertEqual(observed, ObservedValue.value(50))


if __name__ == "__main__":
    unittest.main()
