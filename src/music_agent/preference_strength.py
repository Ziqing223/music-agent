"""Typed, immutable representation of an already-inferred preference conclusion.

This module is the second slice of preference modeling (P06 S2). It defines the pure domain
result type that a resolver will eventually produce, and nothing more. It does not resolve
signal contributions, does not read ``favorited`` / ``disliked`` / ``rating`` / ``play_count``,
and does not persist anything. A :class:`PreferenceStrength` is a conclusion, never an
observation or a derivation step.

Frozen semantics
----------------

A preference conclusion has exactly one of six states:

``POSITIVE``
    A valid positive preference claim exists.
``NEGATIVE``
    A valid negative preference claim exists.
``NEUTRAL``
    There is explicit evidence that no directional preference exists.
``UNKNOWN``
    There are not enough observations or evidence to form any judgment at all.
``INSUFFICIENT``
    There is some observation/evidence, but not enough to form a directional claim.
``CONFLICT``
    There is simultaneously valid, mutually contradictory directional evidence that cannot be
    resolved under the current semantics.

These states are never folded together. In particular, ``UNKNOWN``, ``INSUFFICIENT``,
``NEUTRAL``, and ``CONFLICT`` are four distinct conclusions: "no evidence at all", "some evidence
but not enough", "explicitly no direction", and "contradictory evidence". A numeric score cannot
express any of these, so none of them may be collapsed onto a scalar.

Magnitude invariant
-------------------

``magnitude`` qualifies the *strength* of a directional claim and is only meaningful for the two
directional states:

``POSITIVE`` / ``NEGATIVE``
    ``magnitude`` is required and must satisfy ``0 < magnitude <= 1``.
``NEUTRAL``
    ``magnitude`` must be exactly ``0``.
``UNKNOWN`` / ``INSUFFICIENT`` / ``CONFLICT``
    ``magnitude`` must be ``None``.

A direction is therefore never represented by a signed scalar: the sign lives in ``state`` and
the magnitude is a non-negative magnitude. A ``float('nan')``, an infinity, a boolean, and any
non-numeric value are rejected rather than coerced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class PreferenceStrengthValidationError(ValueError):
    code = "validation_error"


class PreferenceState(StrEnum):
    """The frozen set of possible preference conclusions."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class PreferenceStrength:
    """An immutable, validated preference conclusion.

    ``state`` selects the conclusion kind. ``magnitude`` defaults to ``None``, which is the only
    legal value for the non-directional special states and is rejected for every directional or
    neutral state.
    """

    state: PreferenceState
    magnitude: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, PreferenceState):
            raise PreferenceStrengthValidationError("state must be a PreferenceState")

        if self.state is PreferenceState.POSITIVE or self.state is PreferenceState.NEGATIVE:
            magnitude = _require_finite_number(self.magnitude)
            if not 0 < magnitude <= 1:
                raise PreferenceStrengthValidationError(
                    f"{self.state.value} requires a magnitude within (0, 1]"
                )
        elif self.state is PreferenceState.NEUTRAL:
            magnitude = _require_finite_number(self.magnitude)
            if magnitude != 0:
                raise PreferenceStrengthValidationError(
                    "NEUTRAL requires magnitude 0"
                )
        else:
            if self.magnitude is not None:
                raise PreferenceStrengthValidationError(
                    f"{self.state.value} requires magnitude None"
                )


def _require_finite_number(magnitude: object) -> int | float:
    """Return ``magnitude`` as a finite real number, failing closed otherwise.

    Booleans are rejected explicitly (``bool`` is an ``int`` subclass), as are ``str``, ``None``,
    and any other non-real type. ``float('nan')`` and infinities are rejected as non-finite.
    """
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise PreferenceStrengthValidationError(
            "magnitude must be an int or float, not bool/None/str"
        )
    if not math.isfinite(magnitude):
        raise PreferenceStrengthValidationError("magnitude must be finite")
    return magnitude
