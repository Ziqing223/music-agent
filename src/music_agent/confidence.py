"""Claim-scoped confidence component model (P06 S5).

This module is the fifth slice of preference modeling. It establishes the typed, validated,
explainable *components* of confidence and the aggregation seam, and nothing more. It does not
compute a final confidence score and does not freeze any aggregation operator (no product, no
weighted sum, no calibrated curve).

A confidence conclusion decomposes into seven frozen components:

``quality``
    How trustworthy the evidence itself is, in ``[0, 1]``.
``quantity``
    How much *distinct* evidence exists -- a non-negative integer count. It is already
    deduplicated by the caller: it is never a raw row count, and repeated polling of the same
    evidence must not increase it.
``freshness``
    How recent the evidence is, in ``[0, 1]``.
``consistency``
    How much the evidence agrees, in ``[0, 1]``.
``contradiction``
    An explicit state (``NONE`` / ``PRESENT``), never a scalar. A conflict is evidence, not
    absence of evidence, so it must never be zero-masked onto a ``0.0`` scalar.
``source_reliability``
    How reliable the source of the evidence is, in ``[0, 1]``.
``inference_distance``
    How far the claim is from a direct observation, in ``[0, 1]`` (``0`` = direct, ``1`` =
    maximally inferred).

Frozen semantics
----------------

Claim scope is explicit and never implicit. A :class:`ConfidenceClaim` binds exactly one
:class:`ClaimScope` -- ``CURRENT_PREFERENCE``, ``HISTORICAL_PREFERENCE``, or
``INFERRED_AFFINITY`` -- to one :class:`ConfidenceComponents` value. Two claims with different
scopes are never equal, so confidence about current preference is never silently treated as
confidence about historical preference or inferred affinity.

``UNKNOWN`` / ``INSUFFICIENT`` have **no confidence claim**. This module represents components
only and does not invent confidence for those states: there is no ``UNKNOWN`` / ``INSUFFICIENT``
confidence state, and every component field is required, so an empty or partial component set
cannot be constructed. A future aggregation policy maps the *absence* of a claim to ``None``.

``CONFLICT`` is not "no evidence". It is represented explicitly by
``contradiction=Contradiction.PRESENT``, which is kept visible rather than folded into a zero.

Bounds and validation
---------------------

- The five scalar components (``quality``, ``freshness``, ``consistency``,
  ``source_reliability``, ``inference_distance``) are bounded to ``[0, 1]``.
- ``quantity`` is bounded to ``[0, +infinity)`` as a distinct-evidence integer; a negative
  count, a boolean, a float, or any collection of rows is rejected.
- ``contradiction`` is one of the two :class:`Contradiction` states.

``float('nan')``, infinities, booleans masquerading as numbers, strings, and out-of-range
values are rejected rather than coerced. Every object is frozen and immutable.

Aggregation seam
----------------

:class:`ConfidenceAggregationPolicy` is a ``Protocol`` exposing an ``aggregate(claim)``
entry point for a future aggregation rule. This slice deliberately ships no production
implementation and no module-level derivation function; the operator and the final score type
are deferred to a later slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ConfidenceValidationError(ValueError):
    code = "validation_error"


class ClaimScope(StrEnum):
    """The claim a confidence conclusion is about."""

    CURRENT_PREFERENCE = "current_preference"
    HISTORICAL_PREFERENCE = "historical_preference"
    INFERRED_AFFINITY = "inferred_affinity"


class ConfidenceComponent(StrEnum):
    """The seven frozen confidence component kinds."""

    QUALITY = "quality"
    QUANTITY = "quantity"
    FRESHNESS = "freshness"
    CONSISTENCY = "consistency"
    CONTRADICTION = "contradiction"
    SOURCE_RELIABILITY = "source_reliability"
    INFERENCE_DISTANCE = "inference_distance"


class Contradiction(StrEnum):
    """Explicit contradiction state; a conflict is never folded into a zero scalar."""

    NONE = "none"
    PRESENT = "present"


@dataclass(frozen=True, slots=True)
class ConfidenceComponents:
    """Seven named, bounded, independently inspectable confidence components.

    Every field is required: a partial or empty component set cannot be constructed, so there is
    no way to build a confidence claim out of nothing. ``quantity`` is a *distinct* evidence
    count already deduplicated by the caller, never a raw row count. ``contradiction`` is an
    explicit state so a conflict is never zero-masked.
    """

    quality: float
    quantity: int
    freshness: float
    consistency: float
    contradiction: Contradiction
    source_reliability: float
    inference_distance: float

    def __post_init__(self) -> None:
        _require_unit_interval(self.quality, label="quality")
        _require_unit_interval(self.freshness, label="freshness")
        _require_unit_interval(self.consistency, label="consistency")
        _require_unit_interval(self.source_reliability, label="source_reliability")
        _require_unit_interval(self.inference_distance, label="inference_distance")
        _require_quantity(self.quantity)
        if not isinstance(self.contradiction, Contradiction):
            raise ConfidenceValidationError("contradiction must be a Contradiction")


@dataclass(frozen=True, slots=True)
class ConfidenceClaim:
    """An explicit binding of a :class:`ClaimScope` to its :class:`ConfidenceComponents`.

    A claim is scoped to exactly one of current preference, historical preference, or inferred
    affinity. The scope is immutable and two claims with different scopes are never equal.
    """

    scope: ClaimScope
    components: ConfidenceComponents

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ClaimScope):
            raise ConfidenceValidationError("scope must be a ClaimScope")
        if not isinstance(self.components, ConfidenceComponents):
            raise ConfidenceValidationError("components must be ConfidenceComponents")


@runtime_checkable
class ConfidenceAggregationPolicy(Protocol):
    """Seam for a future confidence aggregation rule.

    ``aggregate`` receives a scoped claim and returns a future-defined confidence score. The
    aggregation operator (product, weighted sum, calibrated curve) and the score type are
    deliberately **not** frozen in this slice. A ``UNKNOWN`` / ``INSUFFICIENT`` preference state
    has no :class:`ConfidenceClaim` at all, so a future policy maps the absence of a claim to
    ``None``. There is no production implementation here.
    """

    def aggregate(self, claim: ConfidenceClaim) -> object:
        """Return a future-defined confidence score for ``claim``."""
        ...


def _require_unit_interval(value: object, *, label: str) -> int | float:
    """Return ``value`` as a finite real in ``[0, 1]``, failing closed otherwise.

    Booleans are rejected explicitly (``bool`` is an ``int`` subclass), as are ``str``, ``None``,
    and any other non-real type. ``float('nan')`` and infinities are rejected as non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfidenceValidationError(
            f"{label} must be an int or float, not {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ConfidenceValidationError(f"{label} must be finite")
    if not 0 <= value <= 1:
        raise ConfidenceValidationError(f"{label} must be within [0, 1]")
    return value


def _require_quantity(value: object) -> int:
    """Return ``value`` as a non-negative distinct-evidence integer, failing closed otherwise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfidenceValidationError(
            f"quantity must be an integer, not {type(value).__name__}"
        )
    if value < 0:
        raise ConfidenceValidationError("quantity must be >= 0")
    return value
