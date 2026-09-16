import inspect
import unittest
from dataclasses import FrozenInstanceError, fields

from music_agent.preference_attribution import (
    AttributionConstraint,
    ConstraintMode,
    DerivedPreference,
    InferredAffinity,
    PreferenceAttributionError,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
    is_inferred_fallback_eligible,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength


_CANONICAL_IDS: dict[PreferenceTargetKind, str] = {
    PreferenceTargetKind.TRACK: "trk_00000000-0000-4000-8000-000000000000",
    PreferenceTargetKind.ARTIST: "art_00000000-0000-4000-8000-000000000000",
    PreferenceTargetKind.ALBUM: "alb_00000000-0000-4000-8000-000000000000",
    PreferenceTargetKind.GENRE: "indie-rock",
}


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(
        PreferenceTargetKind.TRACK, _CANONICAL_IDS[PreferenceTargetKind.TRACK]
    )


def target(kind: PreferenceTargetKind, target_id: str | None = None) -> PreferenceTargetReference:
    return PreferenceTargetReference(
        kind, target_id if target_id is not None else _CANONICAL_IDS[kind]
    )


def direct(state: PreferenceState, magnitude: float | None = None) -> DerivedPreference:
    return DerivedPreference(track_target(), PreferenceStrength(state, magnitude))


def inferred(state: PreferenceState, magnitude: float | None = None) -> InferredAffinity:
    return InferredAffinity(track_target(), PreferenceStrength(state, magnitude))


class PreferenceTargetKindTest(unittest.TestCase):
    def test_enum_values_are_stable(self) -> None:
        expected = {
            PreferenceTargetKind.TRACK: "track",
            PreferenceTargetKind.ARTIST: "artist",
            PreferenceTargetKind.ALBUM: "album",
            PreferenceTargetKind.GENRE: "genre",
        }
        self.assertEqual(list(PreferenceTargetKind), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)

    def test_four_kinds_are_distinct(self) -> None:
        kinds = [
            PreferenceTargetKind.TRACK,
            PreferenceTargetKind.ARTIST,
            PreferenceTargetKind.ALBUM,
            PreferenceTargetKind.GENRE,
        ]
        self.assertEqual(len(set(kinds)), 4)


class PreferenceTargetReferenceValidTest(unittest.TestCase):
    def test_valid_for_all_four_kinds(self) -> None:
        for kind in PreferenceTargetKind:
            with self.subTest(kind=kind):
                ref = PreferenceTargetReference(kind, _CANONICAL_IDS[kind])
                self.assertIs(ref.kind, kind)
                self.assertEqual(ref.target_id, _CANONICAL_IDS[kind])

    def test_empty_is_valid_payload_shape(self) -> None:
        ref = PreferenceTargetReference(PreferenceTargetKind.GENRE, "g")
        self.assertEqual(ref.target_id, "g")


class PreferenceTargetReferenceInvalidTest(unittest.TestCase):
    def test_invalid_kind_fails_closed(self) -> None:
        for bad in ("track", "TRACK", None, 1, PreferenceTargetKind.TRACK.value, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    PreferenceTargetReference(bad, "ref-1")

    def test_invalid_target_id_fails_closed(self) -> None:
        for bad in ("", None, 1, True, ["a"], object()):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, bad)


class PreferenceTargetReferenceCanonicalIdTest(unittest.TestCase):
    def test_canonical_kinds_accept_valid_canonical_id(self) -> None:
        for kind in (
            PreferenceTargetKind.TRACK,
            PreferenceTargetKind.ARTIST,
            PreferenceTargetKind.ALBUM,
        ):
            with self.subTest(kind=kind):
                ref = PreferenceTargetReference(kind, _CANONICAL_IDS[kind])
                self.assertEqual(ref.target_id, _CANONICAL_IDS[kind])

    def test_canonical_kinds_reject_non_canonical_id(self) -> None:
        for kind in (
            PreferenceTargetKind.TRACK,
            PreferenceTargetKind.ARTIST,
            PreferenceTargetKind.ALBUM,
        ):
            with self.subTest(kind=kind):
                with self.assertRaises(PreferenceAttributionError):
                    PreferenceTargetReference(kind, "ref-1")

    def test_canonical_kinds_reject_wrong_namespace(self) -> None:
        track_id = _CANONICAL_IDS[PreferenceTargetKind.TRACK]
        with self.assertRaises(PreferenceAttributionError):
            PreferenceTargetReference(PreferenceTargetKind.ARTIST, track_id)

    def test_canonical_kinds_reject_bad_suffix(self) -> None:
        with self.assertRaises(PreferenceAttributionError):
            PreferenceTargetReference(PreferenceTargetKind.TRACK, "trk_not-a-uuid")

    def test_genre_accepts_any_non_empty_key(self) -> None:
        for key in ("rock", "indie-rock", "g"):
            with self.subTest(key=key):
                ref = PreferenceTargetReference(PreferenceTargetKind.GENRE, key)
                self.assertEqual(ref.target_id, key)

    def test_genre_rejects_empty_key(self) -> None:
        with self.assertRaises(PreferenceAttributionError):
            PreferenceTargetReference(PreferenceTargetKind.GENRE, "")


class DerivedPreferenceTest(unittest.TestCase):
    def test_construction_and_fields(self) -> None:
        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
        derived = DerivedPreference(track_target(), strength)
        self.assertEqual(derived.target, track_target())
        self.assertIs(derived.strength, strength)

    def test_provenance_is_fixed_direct(self) -> None:
        derived = direct(PreferenceState.POSITIVE, 0.9)
        self.assertIs(derived.provenance, PreferenceProvenance.DIRECT)

    def test_invalid_target_fails_closed(self) -> None:
        for bad in (None, "track", 1, PreferenceStrength(PreferenceState.UNKNOWN)):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    DerivedPreference(bad, PreferenceStrength(PreferenceState.UNKNOWN))

    def test_invalid_strength_fails_closed(self) -> None:
        for bad in (None, "positive", 1, PreferenceState.POSITIVE):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    DerivedPreference(track_target(), bad)


class InferredAffinityTest(unittest.TestCase):
    def test_construction_and_fields(self) -> None:
        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
        affinity = InferredAffinity(track_target(), strength)
        self.assertEqual(affinity.target, track_target())
        self.assertIs(affinity.strength, strength)

    def test_provenance_is_fixed_inferred(self) -> None:
        affinity = inferred(PreferenceState.POSITIVE, 0.9)
        self.assertIs(affinity.provenance, PreferenceProvenance.INFERRED)

    def test_invalid_target_fails_closed(self) -> None:
        for bad in (None, "track", 1):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    InferredAffinity(bad, PreferenceStrength(PreferenceState.UNKNOWN))

    def test_invalid_strength_fails_closed(self) -> None:
        for bad in (None, "positive", 1, PreferenceState.POSITIVE):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    InferredAffinity(track_target(), bad)


class ProvenanceSeparationTest(unittest.TestCase):
    def test_direct_and_inferred_are_distinct_types(self) -> None:
        self.assertNotEqual(type(direct(PreferenceState.POSITIVE, 0.9)),
                            type(inferred(PreferenceState.POSITIVE, 0.9)))

    def test_same_target_and_strength_are_not_equal(self) -> None:
        a = DerivedPreference(track_target(), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
        b = InferredAffinity(track_target(), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
        self.assertNotEqual(a, b)
        # The fields coincide, so the two may hash alike; that is harmless because equality
        # distinguishes them. The real guarantee is that a direct preference and an inferred
        # affinity remain distinct keys even in a mixed collection.
        self.assertEqual(len({a, b}), 2)

    def test_provenance_values_are_distinct(self) -> None:
        self.assertNotEqual(PreferenceProvenance.DIRECT, PreferenceProvenance.INFERRED)

    def test_inferred_construction_does_not_mutate_direct(self) -> None:
        strength = PreferenceStrength(PreferenceState.UNKNOWN)
        original = DerivedPreference(track_target(), strength)
        InferredAffinity(track_target(), PreferenceStrength(PreferenceState.POSITIVE, 0.5))
        self.assertEqual(original.strength, strength)
        self.assertIs(original.provenance, PreferenceProvenance.DIRECT)


class FallbackEligibilityTest(unittest.TestCase):
    def test_ineligible_direct_states(self) -> None:
        ineligible = {
            PreferenceState.POSITIVE,
            PreferenceState.NEGATIVE,
            PreferenceState.NEUTRAL,
            PreferenceState.CONFLICT,
        }
        for state in ineligible:
            with self.subTest(state=state):
                self.assertFalse(is_inferred_fallback_eligible(state))

    def test_eligible_direct_states(self) -> None:
        eligible = {PreferenceState.UNKNOWN, PreferenceState.INSUFFICIENT}
        for state in eligible:
            with self.subTest(state=state):
                self.assertTrue(is_inferred_fallback_eligible(state))

    def test_all_six_states_are_classified(self) -> None:
        classified = {state: is_inferred_fallback_eligible(state) for state in PreferenceState}
        self.assertEqual(len(classified), 6)
        self.assertEqual(set(PreferenceState), set(classified))

    def test_eligibility_does_not_touch_preference_objects(self) -> None:
        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
        derived = DerivedPreference(track_target(), strength)
        self.assertFalse(is_inferred_fallback_eligible(derived.strength.state))
        self.assertEqual(derived.strength, strength)


class FallbackEligibilityInvalidTest(unittest.TestCase):
    def test_non_preference_state_fails_closed(self) -> None:
        for bad in ("unknown", "UNKNOWN", None, 1, True, PreferenceState.UNKNOWN.value, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    is_inferred_fallback_eligible(bad)


class AttributionConstraintTest(unittest.TestCase):
    def test_block_and_discount_are_valid(self) -> None:
        self.assertIs(AttributionConstraint(ConstraintMode.BLOCK).mode, ConstraintMode.BLOCK)
        self.assertIs(AttributionConstraint(ConstraintMode.DISCOUNT).mode, ConstraintMode.DISCOUNT)

    def test_constraint_mode_values_are_stable(self) -> None:
        self.assertEqual(ConstraintMode.BLOCK.value, "block")
        self.assertEqual(ConstraintMode.DISCOUNT.value, "discount")
        self.assertEqual(list(ConstraintMode), [ConstraintMode.BLOCK, ConstraintMode.DISCOUNT])

    def test_invalid_mode_fails_closed(self) -> None:
        for bad in ("block", "BLOCK", None, 1, True, ConstraintMode.BLOCK.value, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(PreferenceAttributionError):
                    AttributionConstraint(bad)

    def test_no_numeric_discount_field(self) -> None:
        self.assertEqual([f.name for f in fields(AttributionConstraint)], ["mode"])

    def test_no_hardcoded_discount_constant(self) -> None:
        module = inspect.getmodule(AttributionConstraint)
        for name, value in vars(module).items():
            if name.isupper():
                self.assertNotIsInstance(value, (int, float), f"numeric constant {name}")

    def test_constraint_has_no_effect_on_track_preference(self) -> None:
        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
        derived = DerivedPreference(track_target(), strength)
        AttributionConstraint(ConstraintMode.BLOCK)
        AttributionConstraint(ConstraintMode.DISCOUNT)
        self.assertEqual(derived.strength, strength)
        self.assertIs(derived.provenance, PreferenceProvenance.DIRECT)


class ImmutabilityEqualityTest(unittest.TestCase):
    def test_target_reference_is_frozen(self) -> None:
        ref = track_target()
        with self.assertRaises(FrozenInstanceError):
            ref.target_id = "other"
        with self.assertRaises(FrozenInstanceError):
            ref.kind = PreferenceTargetKind.ARTIST

    def test_derived_preference_is_frozen(self) -> None:
        derived = direct(PreferenceState.POSITIVE, 0.9)
        with self.assertRaises(FrozenInstanceError):
            derived.strength = PreferenceStrength(PreferenceState.NEGATIVE, 0.8)
        with self.assertRaises(FrozenInstanceError):
            derived.target = track_target()

    def test_inferred_affinity_is_frozen(self) -> None:
        affinity = inferred(PreferenceState.UNKNOWN)
        with self.assertRaises(FrozenInstanceError):
            affinity.strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)

    def test_attribution_constraint_is_frozen(self) -> None:
        constraint = AttributionConstraint(ConstraintMode.BLOCK)
        with self.assertRaises(FrozenInstanceError):
            constraint.mode = ConstraintMode.DISCOUNT

    def test_equality_and_hash(self) -> None:
        a = DerivedPreference(track_target(), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
        b = DerivedPreference(track_target(), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, direct(PreferenceState.POSITIVE, 0.8))
        self.assertNotEqual(
            a, DerivedPreference(target(PreferenceTargetKind.ARTIST), a.strength)
        )

    def test_inferred_equality_and_hash(self) -> None:
        a = InferredAffinity(track_target(), PreferenceStrength(PreferenceState.NEGATIVE, 0.7))
        b = InferredAffinity(track_target(), PreferenceStrength(PreferenceState.NEGATIVE, 0.7))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_attribution_constraint_equality_and_hash(self) -> None:
        self.assertEqual(
            AttributionConstraint(ConstraintMode.BLOCK),
            AttributionConstraint(ConstraintMode.BLOCK),
        )
        self.assertNotEqual(
            AttributionConstraint(ConstraintMode.BLOCK),
            AttributionConstraint(ConstraintMode.DISCOUNT),
        )

    def test_deterministic_repr(self) -> None:
        a = DerivedPreference(track_target(), PreferenceStrength(PreferenceState.CONFLICT))
        b = DerivedPreference(track_target(), PreferenceStrength(PreferenceState.CONFLICT))
        self.assertEqual(repr(a), repr(b))


class NoPropagationAlgorithmTest(unittest.TestCase):
    def test_module_exposes_no_propagation_function(self) -> None:
        import music_agent.preference_attribution as module

        for name, value in vars(module).items():
            if callable(value) and not inspect.isclass(value):
                self.assertNotIn("propagat", name, f"propagation helper {name} must not exist")
                self.assertNotIn("aggregat", name, f"aggregation helper {name} must not exist")

    def test_no_fallback_selection_function(self) -> None:
        import music_agent.preference_attribution as module

        for absent in ("select_inferred", "choose_inferred", "best_inferred", "combine",
                       "merge_preference", "fallback"):
            self.assertFalse(hasattr(module, absent), f"{absent} must not exist")

    def test_only_public_callable_is_eligibility_predicate(self) -> None:
        import music_agent.preference_attribution as module

        public_callables = [
            name
            for name, value in vars(module).items()
            if callable(value)
            and not name.startswith("_")
            and not inspect.isclass(value)
            and getattr(value, "__module__", None) == module.__name__
        ]
        self.assertEqual(public_callables, ["is_inferred_fallback_eligible"])


class ErrorHierarchyTest(unittest.TestCase):
    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(PreferenceAttributionError, ValueError))

    def test_error_code_is_stable(self) -> None:
        self.assertEqual(PreferenceAttributionError.code, "validation_error")


if __name__ == "__main__":
    unittest.main()
