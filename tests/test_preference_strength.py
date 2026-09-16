import math
import unittest
from dataclasses import FrozenInstanceError

from music_agent.preference_strength import (
    PreferenceState,
    PreferenceStrength,
    PreferenceStrengthValidationError,
)


class PreferenceStateTest(unittest.TestCase):
    def test_enum_values_are_stable(self) -> None:
        expected = {
            PreferenceState.POSITIVE: "positive",
            PreferenceState.NEGATIVE: "negative",
            PreferenceState.NEUTRAL: "neutral",
            PreferenceState.UNKNOWN: "unknown",
            PreferenceState.INSUFFICIENT: "insufficient",
            PreferenceState.CONFLICT: "conflict",
        }
        self.assertEqual(list(PreferenceState), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)

    def test_special_states_are_distinct(self) -> None:
        self.assertNotEqual(PreferenceState.UNKNOWN, PreferenceState.INSUFFICIENT)
        self.assertNotEqual(PreferenceState.UNKNOWN, PreferenceState.NEUTRAL)
        self.assertNotEqual(PreferenceState.UNKNOWN, PreferenceState.CONFLICT)
        self.assertNotEqual(PreferenceState.INSUFFICIENT, PreferenceState.NEUTRAL)
        self.assertNotEqual(PreferenceState.INSUFFICIENT, PreferenceState.CONFLICT)
        self.assertNotEqual(PreferenceState.NEUTRAL, PreferenceState.CONFLICT)


class PreferenceStrengthValidTest(unittest.TestCase):
    def test_positive_magnitude_is_valid(self) -> None:
        for magnitude in (0.000001, 0.5, 1, 1.0):
            with self.subTest(magnitude=magnitude):
                strength = PreferenceStrength(PreferenceState.POSITIVE, magnitude)
                self.assertIs(strength.state, PreferenceState.POSITIVE)
                self.assertEqual(strength.magnitude, magnitude)

    def test_negative_magnitude_is_valid(self) -> None:
        for magnitude in (0.000001, 0.5, 1, 1.0):
            with self.subTest(magnitude=magnitude):
                strength = PreferenceStrength(PreferenceState.NEGATIVE, magnitude)
                self.assertIs(strength.state, PreferenceState.NEGATIVE)
                self.assertEqual(strength.magnitude, magnitude)

    def test_neutral_zero_is_valid_int_and_float(self) -> None:
        int_zero = PreferenceStrength(PreferenceState.NEUTRAL, 0)
        float_zero = PreferenceStrength(PreferenceState.NEUTRAL, 0.0)
        self.assertEqual(int_zero.magnitude, 0)
        self.assertEqual(float_zero.magnitude, 0.0)

    def test_special_states_are_valid_with_none(self) -> None:
        for state in (PreferenceState.UNKNOWN, PreferenceState.INSUFFICIENT, PreferenceState.CONFLICT):
            with self.subTest(state=state):
                explicit = PreferenceStrength(state, None)
                defaulted = PreferenceStrength(state)
                self.assertIsNone(explicit.magnitude)
                self.assertIsNone(defaulted.magnitude)
                self.assertEqual(explicit, defaulted)


class PreferenceStrengthInvalidTest(unittest.TestCase):
    def test_directional_requires_magnitude(self) -> None:
        for state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            with self.subTest(state=state):
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(state, None)
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(state)

    def test_directional_rejects_zero(self) -> None:
        for state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            for magnitude in (0, 0.0):
                with self.subTest(state=state, magnitude=magnitude):
                    with self.assertRaises(PreferenceStrengthValidationError):
                        PreferenceStrength(state, magnitude)

    def test_directional_rejects_negative(self) -> None:
        for state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            for magnitude in (-0.000001, -0.5, -1):
                with self.subTest(state=state, magnitude=magnitude):
                    with self.assertRaises(PreferenceStrengthValidationError):
                        PreferenceStrength(state, magnitude)

    def test_directional_rejects_above_one(self) -> None:
        for state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            for magnitude in (1.000001, 1.5, 100):
                with self.subTest(state=state, magnitude=magnitude):
                    with self.assertRaises(PreferenceStrengthValidationError):
                        PreferenceStrength(state, magnitude)

    def test_neutral_rejects_nonzero(self) -> None:
        for magnitude in (0.000001, 0.5, 1, -0.5):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(PreferenceState.NEUTRAL, magnitude)

    def test_neutral_rejects_none(self) -> None:
        with self.assertRaises(PreferenceStrengthValidationError):
            PreferenceStrength(PreferenceState.NEUTRAL, None)
        with self.assertRaises(PreferenceStrengthValidationError):
            PreferenceStrength(PreferenceState.NEUTRAL)

    def test_special_states_reject_numbers(self) -> None:
        for state in (PreferenceState.UNKNOWN, PreferenceState.INSUFFICIENT, PreferenceState.CONFLICT):
            for magnitude in (0, 0.0, 0.5, 1, -0.5):
                with self.subTest(state=state, magnitude=magnitude):
                    with self.assertRaises(PreferenceStrengthValidationError):
                        PreferenceStrength(state, magnitude)

    def test_non_finite_magnitude_fails_closed(self) -> None:
        for magnitude in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(PreferenceState.POSITIVE, magnitude)
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(PreferenceState.NEGATIVE, magnitude)

    def test_bool_magnitude_fails_closed(self) -> None:
        for state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE, PreferenceState.NEUTRAL):
            for magnitude in (True, False):
                with self.subTest(state=state, magnitude=magnitude):
                    with self.assertRaises(PreferenceStrengthValidationError):
                        PreferenceStrength(state, magnitude)

    def test_non_numeric_magnitude_fails_closed(self) -> None:
        for magnitude in ("0.5", "high", [], {}):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(PreferenceState.POSITIVE, magnitude)

    def test_invalid_state_fails_closed(self) -> None:
        for state in ("positive", "POSITIVE", None, 1, PreferenceState.POSITIVE.value):
            with self.subTest(state=state):
                with self.assertRaises(PreferenceStrengthValidationError):
                    PreferenceStrength(state, None)


class PreferenceStrengthBehaviorTest(unittest.TestCase):
    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(PreferenceStrengthValidationError, ValueError))

    def test_frozen_immutable(self) -> None:
        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.5)
        with self.assertRaises(FrozenInstanceError):
            strength.magnitude = 0.6
        with self.assertRaises(FrozenInstanceError):
            strength.state = PreferenceState.NEUTRAL

    def test_equality_and_hash(self) -> None:
        a = PreferenceStrength(PreferenceState.POSITIVE, 0.5)
        b = PreferenceStrength(PreferenceState.POSITIVE, 0.5)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, PreferenceStrength(PreferenceState.POSITIVE, 0.6))
        self.assertNotEqual(a, PreferenceStrength(PreferenceState.NEGATIVE, 0.5))

    def test_deterministic_repr(self) -> None:
        a = PreferenceStrength(PreferenceState.CONFLICT)
        b = PreferenceStrength(PreferenceState.CONFLICT)
        self.assertEqual(repr(a), repr(b))

    def test_special_states_not_folded(self) -> None:
        unknown = PreferenceStrength(PreferenceState.UNKNOWN)
        insufficient = PreferenceStrength(PreferenceState.INSUFFICIENT)
        neutral = PreferenceStrength(PreferenceState.NEUTRAL, 0)
        conflict = PreferenceStrength(PreferenceState.CONFLICT)
        self.assertEqual(len({unknown, insufficient, neutral, conflict}), 4)


if __name__ == "__main__":
    unittest.main()
