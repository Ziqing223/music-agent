import dataclasses
import inspect
import types
import unittest
from dataclasses import FrozenInstanceError

from music_agent.confidence import (
    ClaimScope,
    ConfidenceAggregationPolicy,
    ConfidenceClaim,
    ConfidenceComponent,
    ConfidenceComponents,
    ConfidenceValidationError,
    Contradiction,
)

SCALAR_FIELDS = ("quality", "freshness", "consistency", "source_reliability", "inference_distance")


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


class ClaimScopeTest(unittest.TestCase):
    def test_scopes_are_stable_and_distinct(self) -> None:
        expected = {
            ClaimScope.CURRENT_PREFERENCE: "current_preference",
            ClaimScope.HISTORICAL_PREFERENCE: "historical_preference",
            ClaimScope.INFERRED_AFFINITY: "inferred_affinity",
        }
        self.assertEqual(list(ClaimScope), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)
        self.assertEqual(len(set(ClaimScope)), 3)


class ConfidenceComponentTest(unittest.TestCase):
    def test_seven_components_are_stable(self) -> None:
        expected = {
            ConfidenceComponent.QUALITY: "quality",
            ConfidenceComponent.QUANTITY: "quantity",
            ConfidenceComponent.FRESHNESS: "freshness",
            ConfidenceComponent.CONSISTENCY: "consistency",
            ConfidenceComponent.CONTRADICTION: "contradiction",
            ConfidenceComponent.SOURCE_RELIABILITY: "source_reliability",
            ConfidenceComponent.INFERENCE_DISTANCE: "inference_distance",
        }
        self.assertEqual(list(ConfidenceComponent), list(expected))
        self.assertEqual(len(set(ConfidenceComponent)), 7)

    def test_component_enum_matches_container_fields(self) -> None:
        field_names = {field.name for field in dataclasses.fields(ConfidenceComponents)}
        enum_values = {component.value for component in ConfidenceComponent}
        self.assertEqual(field_names, enum_values)


class ConfidenceComponentsValidTest(unittest.TestCase):
    def test_scalar_components_accept_unit_interval(self) -> None:
        for field in SCALAR_FIELDS:
            for good in (0, 0.0, 0.5, 1, 1.0):
                with self.subTest(field=field, value=good):
                    self.assertEqual(getattr(components(**{field: good}), field), good)

    def test_quantity_accepts_non_negative_distinct_count(self) -> None:
        for n in (0, 1, 3, 1000):
            with self.subTest(n=n):
                self.assertEqual(components(quantity=n).quantity, n)

    def test_contradiction_accepts_both_states(self) -> None:
        self.assertIs(
            components(contradiction=Contradiction.NONE).contradiction, Contradiction.NONE
        )
        self.assertIs(
            components(contradiction=Contradiction.PRESENT).contradiction,
            Contradiction.PRESENT,
        )


class ConfidenceComponentsInvalidTest(unittest.TestCase):
    def test_scalar_out_of_range_fails_closed(self) -> None:
        for field in SCALAR_FIELDS:
            for bad in (-0.0001, -1, 1.0001, 2):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ConfidenceValidationError):
                        components(**{field: bad})

    def test_scalar_non_finite_fails_closed(self) -> None:
        for field in SCALAR_FIELDS:
            for bad in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ConfidenceValidationError):
                        components(**{field: bad})

    def test_scalar_bool_and_non_numeric_fails_closed(self) -> None:
        for field in SCALAR_FIELDS:
            for bad in (True, False, "0.5", None, [], {}):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ConfidenceValidationError):
                        components(**{field: bad})

    def test_quantity_invalid_fails_closed(self) -> None:
        for bad in (-1, -5, True, False, 1.5, "3", None, [1, 2], (1, 2)):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    components(quantity=bad)

    def test_contradiction_invalid_fails_closed(self) -> None:
        for bad in ("present", "none", True, False, 1, 0, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    components(contradiction=bad)

    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(ConfidenceValidationError, ValueError))


class ConfidenceClaimTest(unittest.TestCase):
    def test_valid_claim_round_trips(self) -> None:
        claim = ConfidenceClaim(ClaimScope.CURRENT_PREFERENCE, components())
        self.assertIs(claim.scope, ClaimScope.CURRENT_PREFERENCE)
        self.assertIsInstance(claim.components, ConfidenceComponents)

    def test_scope_distinguishes_otherwise_identical_claims(self) -> None:
        comps = components()
        current = ConfidenceClaim(ClaimScope.CURRENT_PREFERENCE, comps)
        historical = ConfidenceClaim(ClaimScope.HISTORICAL_PREFERENCE, comps)
        inferred = ConfidenceClaim(ClaimScope.INFERRED_AFFINITY, comps)
        self.assertEqual(current.components, historical.components)
        self.assertNotEqual(current, historical)
        self.assertNotEqual(current, inferred)
        self.assertNotEqual(historical, inferred)
        self.assertEqual(len({current, historical, inferred}), 3)

    def test_malformed_scope_fails_closed(self) -> None:
        for bad in ("current_preference", "CURRENT_PREFERENCE", None, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    ConfidenceClaim(bad, components())  # type: ignore[arg-type]

    def test_malformed_components_fails_closed(self) -> None:
        for bad in (None, "components", components().quality):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    ConfidenceClaim(ClaimScope.CURRENT_PREFERENCE, bad)  # type: ignore[arg-type]


class ContradictionTest(unittest.TestCase):
    def test_states_are_distinct(self) -> None:
        self.assertNotEqual(Contradiction.NONE, Contradiction.PRESENT)

    def test_contradiction_is_explicit_not_scalar(self) -> None:
        # A conflict is a named state, not a float, so it cannot be zero-masked.
        conflicted = components(contradiction=Contradiction.PRESENT)
        self.assertIs(conflicted.contradiction, Contradiction.PRESENT)
        self.assertNotIsInstance(conflicted.contradiction, (int, float))

    def test_present_is_preserved_verbatim(self) -> None:
        conflicted = components(contradiction=Contradiction.PRESENT)
        self.assertEqual(conflicted, components(contradiction=Contradiction.PRESENT))
        self.assertNotEqual(conflicted, components(contradiction=Contradiction.NONE))


class QuantitySemanticsTest(unittest.TestCase):
    def test_quantity_round_trips_distinct_count_verbatim(self) -> None:
        for n in (0, 1, 3, 1000):
            with self.subTest(n=n):
                self.assertEqual(components(quantity=n).quantity, n)

    def test_quantity_rejects_row_collections(self) -> None:
        # quantity is a pre-deduplicated count, never a list/tuple/mapping of rows.
        for bad in ([1, 2, 3], (1, 2), {"row": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    components(quantity=bad)

    def test_quantity_has_no_dedup_or_counting_api(self) -> None:
        import music_agent.confidence as confidence

        for name in ("count", "dedup", "deduplicate", "accumulate", "increment"):
            self.assertNotIn(name, vars(confidence))


class UnknownInsufficientTest(unittest.TestCase):
    def test_module_has_no_unknown_or_insufficient_state(self) -> None:
        import music_agent.confidence as confidence

        for name in ("UNKNOWN", "INSUFFICIENT", "NO_CLAIM", "EMPTY"):
            self.assertNotIn(name, vars(confidence))

    def test_components_require_every_field(self) -> None:
        for field in dataclasses.fields(ConfidenceComponents):
            self.assertIs(field.default, dataclasses.MISSING, field.name)
            self.assertIs(field.default_factory, dataclasses.MISSING, field.name)

    def test_partial_components_cannot_be_constructed(self) -> None:
        with self.assertRaises(TypeError):
            ConfidenceComponents(quality=1.0)  # type: ignore[call-arg]

    def test_no_empty_or_none_claim_constructor(self) -> None:
        for name in ("none", "empty", "unknown", "insufficient"):
            self.assertFalse(hasattr(ConfidenceClaim, name))
            self.assertFalse(hasattr(ConfidenceComponents, name))


class AggregationSeamTest(unittest.TestCase):
    def test_seam_is_a_protocol_not_an_implementation(self) -> None:
        self.assertTrue(getattr(ConfidenceAggregationPolicy, "_is_protocol", False))

    def test_seam_exposes_aggregate_method(self) -> None:
        self.assertTrue(hasattr(ConfidenceAggregationPolicy, "aggregate"))

    def test_no_module_level_aggregation_function(self) -> None:
        import music_agent.confidence as confidence

        for name in (
            "aggregate",
            "combine",
            "compute_confidence",
            "score",
            "overall_confidence",
            "total",
        ):
            self.assertNotIn(name, vars(confidence))

    def test_components_and_claim_have_no_combined_score_accessor(self) -> None:
        for obj in (ConfidenceComponents, ConfidenceClaim):
            for name in ("score", "aggregate", "combine", "total", "overall", "value"):
                self.assertFalse(hasattr(obj, name), f"{obj.__name__}.{name}")

    def test_no_public_derivation_function_defined_in_module(self) -> None:
        import music_agent.confidence as confidence

        locally_defined_public_functions = [
            name
            for name, value in vars(confidence).items()
            if not name.startswith("_")
            and inspect.isfunction(value)
            and getattr(value, "__module__", None) == "music_agent.confidence"
        ]
        self.assertEqual(locally_defined_public_functions, [])


class DeterminismAndPurityTest(unittest.TestCase):
    def test_equality_and_hash(self) -> None:
        a = components()
        b = components()
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, components(quantity=4))

    def test_deterministic_repr(self) -> None:
        self.assertEqual(repr(components()), repr(components()))

    def test_frozen_immutability(self) -> None:
        comps = components()
        with self.assertRaises(FrozenInstanceError):
            comps.quantity = 4  # type: ignore[misc]
        claim = ConfidenceClaim(ClaimScope.CURRENT_PREFERENCE, comps)
        with self.assertRaises(FrozenInstanceError):
            claim.scope = ClaimScope.INFERRED_AFFINITY  # type: ignore[misc]

    def test_no_io_clock_or_random_imports(self) -> None:
        import music_agent.confidence as confidence

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
        }
        for name in forbidden:
            self.assertNotIn(name, vars(confidence), name)

    def test_no_preference_module_dependency(self) -> None:
        import music_agent.confidence as confidence

        for name in vars(confidence):
            self.assertNotIn("preference", name.lower())
        for value in vars(confidence).values():
            self.assertFalse(
                isinstance(value, types.ModuleType)
                and value.__name__.startswith("music_agent.preference")
            )


if __name__ == "__main__":
    unittest.main()
