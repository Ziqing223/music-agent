import math
import types
import unittest
from dataclasses import FrozenInstanceError

from music_agent.familiarity import (
    Familiarity,
    FamiliarityLevel,
    FamiliarityNormalizationPolicy,
    FamiliarityReason,
    FamiliarityValidationError,
    derive_track_familiarity,
)
from music_agent.source_observation import ObservedValue, ObservationState


class _CappedPolicy:
    """Deterministic linear normalization with an explicit saturation cap."""

    def __init__(self, divisor: float, cap: float = 1.0) -> None:
        self._divisor = divisor
        self._cap = cap

    def normalize(self, play_count: int) -> float:
        return min(self._cap, play_count / self._divisor)


def capped_policy(divisor: float = 10.0) -> FamiliarityNormalizationPolicy:
    return _CappedPolicy(divisor)


class _BrokenPolicy:
    """A policy with the right shape that returns a fixed, possibly invalid, magnitude."""

    def __init__(self, value: object) -> None:
        self._value = value

    def normalize(self, play_count: int) -> float:
        return self._value  # type: ignore[return-value]


class ObservationSemanticsTest(unittest.TestCase):
    def test_missing_is_unknown(self) -> None:
        result = derive_track_familiarity(ObservedValue.missing(), capped_policy())
        self.assertIs(result.level, FamiliarityLevel.UNKNOWN)
        self.assertIs(result.reason, FamiliarityReason.MISSING)
        self.assertIsNone(result.magnitude)
        self.assertIs(result.raw.state, ObservationState.MISSING)

    def test_null_is_unknown_like_state(self) -> None:
        result = derive_track_familiarity(ObservedValue.null(), capped_policy())
        self.assertIs(result.level, FamiliarityLevel.UNKNOWN)
        self.assertIs(result.reason, FamiliarityReason.NULL_VALUE)
        self.assertIsNone(result.magnitude)
        self.assertIs(result.raw.state, ObservationState.NULL)

    def test_value_zero_is_known_zero_familiarity(self) -> None:
        result = derive_track_familiarity(ObservedValue.value(0), capped_policy())
        self.assertIs(result.level, FamiliarityLevel.KNOWN)
        self.assertIs(result.reason, FamiliarityReason.OBSERVED_EXPOSURE)
        self.assertEqual(result.magnitude, 0.0)
        self.assertEqual(result.raw.payload, 0)

    def test_value_positive_is_known_positive_familiarity(self) -> None:
        result = derive_track_familiarity(ObservedValue.value(1), capped_policy())
        self.assertIs(result.level, FamiliarityLevel.KNOWN)
        self.assertIs(result.reason, FamiliarityReason.OBSERVED_EXPOSURE)
        self.assertGreater(result.magnitude, 0.0)
        self.assertLessEqual(result.magnitude, 1.0)
        self.assertEqual(result.raw.payload, 1)

    def test_missing_null_value_zero_are_not_folded(self) -> None:
        missing = derive_track_familiarity(ObservedValue.missing(), capped_policy())
        null = derive_track_familiarity(ObservedValue.null(), capped_policy())
        zero = derive_track_familiarity(ObservedValue.value(0), capped_policy())

        self.assertIs(missing.level, FamiliarityLevel.UNKNOWN)
        self.assertIs(null.level, FamiliarityLevel.UNKNOWN)
        self.assertIs(zero.level, FamiliarityLevel.KNOWN)

        self.assertNotEqual(missing.reason, null.reason)
        self.assertNotEqual(missing.reason, zero.reason)
        self.assertNotEqual(null.reason, zero.reason)

        self.assertIs(missing.raw.state, ObservationState.MISSING)
        self.assertIs(null.raw.state, ObservationState.NULL)
        self.assertIs(zero.raw.state, ObservationState.VALUE)

        self.assertNotEqual(missing, null)
        self.assertNotEqual(missing, zero)
        self.assertNotEqual(null, zero)

    def test_raw_provenance_is_preserved(self) -> None:
        observed = ObservedValue.value(42)
        result = derive_track_familiarity(observed, capped_policy())
        self.assertEqual(result.raw, observed)
        self.assertEqual(result.raw.payload, 42)
        self.assertIs(result.raw.state, ObservationState.VALUE)


class CalibrationSeamTest(unittest.TestCase):
    def test_same_play_count_different_magnitude_by_policy(self) -> None:
        steep = capped_policy(divisor=10.0)
        shallow = capped_policy(divisor=1000.0)
        steep_result = derive_track_familiarity(ObservedValue.value(10), steep)
        shallow_result = derive_track_familiarity(ObservedValue.value(10), shallow)

        self.assertIs(steep_result.level, FamiliarityLevel.KNOWN)
        self.assertIs(shallow_result.level, FamiliarityLevel.KNOWN)
        self.assertIs(steep_result.reason, shallow_result.reason)
        self.assertEqual(steep_result.magnitude, 1.0)
        self.assertEqual(shallow_result.magnitude, 0.01)
        self.assertNotEqual(steep_result.magnitude, shallow_result.magnitude)
        self.assertEqual(steep_result.raw.payload, shallow_result.raw.payload)

    def test_state_semantics_unchanged_by_policy(self) -> None:
        for observed in (
            ObservedValue.missing(),
            ObservedValue.null(),
            ObservedValue.value(0),
        ):
            with self.subTest(observed=observed):
                steep = derive_track_familiarity(observed, capped_policy(divisor=10.0))
                shallow = derive_track_familiarity(observed, capped_policy(divisor=1000.0))
                self.assertIs(steep.level, shallow.level)
                self.assertIs(steep.reason, shallow.reason)
                self.assertEqual(steep.magnitude, shallow.magnitude)


class SaturationTest(unittest.TestCase):
    def test_high_count_saturates_at_policy_cap(self) -> None:
        result = derive_track_familiarity(ObservedValue.value(10**12), capped_policy())
        self.assertIs(result.level, FamiliarityLevel.KNOWN)
        self.assertEqual(result.magnitude, 1.0)
        self.assertLessEqual(result.magnitude, 1.0)

    def test_high_count_never_exceeds_one(self) -> None:
        for count in (100, 10**6, 10**12):
            with self.subTest(count=count):
                result = derive_track_familiarity(ObservedValue.value(count), capped_policy())
                self.assertLessEqual(result.magnitude, 1.0)
                self.assertGreaterEqual(result.magnitude, 0.0)


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_output(self) -> None:
        observed = ObservedValue.value(37)
        first = derive_track_familiarity(observed, capped_policy())
        second = derive_track_familiarity(observed, capped_policy())
        self.assertEqual(first, second)

    def test_input_observation_is_not_mutated(self) -> None:
        observed = ObservedValue.value(37)
        derive_track_familiarity(observed, capped_policy())
        self.assertEqual(observed, ObservedValue.value(37))

    def test_result_is_frozen_immutable(self) -> None:
        result = derive_track_familiarity(ObservedValue.value(3), capped_policy())
        with self.assertRaises(FrozenInstanceError):
            result.magnitude = 0.9  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.level = FamiliarityLevel.UNKNOWN  # type: ignore[misc]


class SeparationFromPreferenceTest(unittest.TestCase):
    def test_module_namespace_has_no_preference_imports(self) -> None:
        import music_agent.familiarity as familiarity

        for name in vars(familiarity):
            self.assertNotIn("preference", name.lower())
        for value in vars(familiarity).values():
            self.assertFalse(
                isinstance(value, types.ModuleType)
                and value.__name__.startswith("music_agent.preference")
            )

    def test_high_play_count_is_familiarity_not_preference(self) -> None:
        result = derive_track_familiarity(ObservedValue.value(10**9), capped_policy())
        self.assertIsInstance(result, Familiarity)
        self.assertIs(result.level, FamiliarityLevel.KNOWN)
        self.assertFalse(hasattr(result, "direction"))
        self.assertFalse(hasattr(result, "state"))


class FamiliarityConstructionTest(unittest.TestCase):
    def test_known_accepts_bounded_magnitudes(self) -> None:
        for magnitude in (0, 0.0, 0.5, 1, 1.0):
            with self.subTest(magnitude=magnitude):
                result = Familiarity(
                    FamiliarityLevel.KNOWN,
                    ObservedValue.value(3),
                    FamiliarityReason.OBSERVED_EXPOSURE,
                    magnitude,
                )
                self.assertEqual(result.magnitude, magnitude)

    def test_unknown_accepts_none_magnitude(self) -> None:
        explicit = Familiarity(
            FamiliarityLevel.UNKNOWN,
            ObservedValue.missing(),
            FamiliarityReason.MISSING,
            None,
        )
        defaulted = Familiarity(
            FamiliarityLevel.UNKNOWN, ObservedValue.missing(), FamiliarityReason.MISSING
        )
        self.assertEqual(explicit, defaulted)

    def test_unknown_rejects_magnitude(self) -> None:
        for magnitude in (0, 0.5, 1):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(FamiliarityValidationError):
                    Familiarity(
                        FamiliarityLevel.UNKNOWN,
                        ObservedValue.missing(),
                        FamiliarityReason.MISSING,
                        magnitude,
                    )

    def test_known_rejects_none_magnitude(self) -> None:
        with self.assertRaises(FamiliarityValidationError):
            Familiarity(
                FamiliarityLevel.KNOWN,
                ObservedValue.value(3),
                FamiliarityReason.OBSERVED_EXPOSURE,
            )

    def test_known_rejects_out_of_range_magnitude(self) -> None:
        for magnitude in (-0.5, 1.000001, 2, float("nan"), float("inf"), -float("inf")):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(FamiliarityValidationError):
                    Familiarity(
                        FamiliarityLevel.KNOWN,
                        ObservedValue.value(3),
                        FamiliarityReason.OBSERVED_EXPOSURE,
                        magnitude,
                    )

    def test_known_rejects_bool_and_non_numeric_magnitude(self) -> None:
        for magnitude in (True, False, "0.5", None):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(FamiliarityValidationError):
                    Familiarity(
                        FamiliarityLevel.KNOWN,
                        ObservedValue.value(3),
                        FamiliarityReason.OBSERVED_EXPOSURE,
                        magnitude,
                    )

    def test_invalid_level_reason_raw_fail_closed(self) -> None:
        with self.assertRaises(FamiliarityValidationError):
            Familiarity("known", ObservedValue.value(3), FamiliarityReason.OBSERVED_EXPOSURE, 0.5)  # type: ignore[arg-type]
        with self.assertRaises(FamiliarityValidationError):
            Familiarity(FamiliarityLevel.KNOWN, "raw", FamiliarityReason.OBSERVED_EXPOSURE, 0.5)  # type: ignore[arg-type]
        with self.assertRaises(FamiliarityValidationError):
            Familiarity(FamiliarityLevel.KNOWN, ObservedValue.value(3), "observed", 0.5)  # type: ignore[arg-type]

    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(FamiliarityValidationError, ValueError))

    def test_equality_and_hash(self) -> None:
        a = derive_track_familiarity(ObservedValue.value(5), capped_policy())
        b = derive_track_familiarity(ObservedValue.value(5), capped_policy())
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(
            a, derive_track_familiarity(ObservedValue.value(6), capped_policy())
        )


class ValidationTest(unittest.TestCase):
    def test_negative_play_count_fails_closed(self) -> None:
        for count in (-1, -10):
            with self.subTest(count=count):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(count), capped_policy())

    def test_bool_play_count_fails_closed(self) -> None:
        for count in (True, False):
            with self.subTest(count=count):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(count), capped_policy())

    def test_float_and_string_payload_fails_closed(self) -> None:
        for payload in (1.5, "10", "high"):
            with self.subTest(payload=payload):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(payload), capped_policy())

    def test_malformed_observed_value_fails_closed(self) -> None:
        for bad in (None, 5, "missing", True, []):
            with self.subTest(bad=bad):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(bad, capped_policy())  # type: ignore[arg-type]

    def test_invalid_policy_fails_closed(self) -> None:
        for bad in (None, "policy", 0.9, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(5), bad)  # type: ignore[arg-type]

    def test_policy_output_out_of_range_fails_closed(self) -> None:
        for value in (2.0, -0.5, 1.000001):
            with self.subTest(value=value):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(5), _BrokenPolicy(value))

    def test_policy_output_non_finite_fails_closed(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(5), _BrokenPolicy(value))

    def test_policy_output_non_numeric_fails_closed(self) -> None:
        for value in ("0.5", None, True, []):
            with self.subTest(value=value):
                with self.assertRaises(FamiliarityValidationError):
                    derive_track_familiarity(ObservedValue.value(5), _BrokenPolicy(value))


if __name__ == "__main__":
    unittest.main()
