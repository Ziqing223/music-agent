"""Familiarity axis derived from observed playback exposure (P06 S4).

This module is the fourth slice of preference modeling. It derives a bounded
:class:`Familiarity` conclusion from a single observed ``play_count`` and nothing else.
Familiarity is a *separate* axis from Preference: a high play count means "well-known track",
never "liked track". This slice therefore never changes ``PreferenceState``, never produces a
preference direction, and never calls or imports the Direct Track Preference resolver.

Frozen semantics
----------------

``play_count`` is live-readable and means *historical exposure / familiarity*. It is **not**
positive preference: a high ``play_count`` only supports "the user is familiar with this
track", never "the user likes it".

The three observation states are never folded together:

``MISSING``
    Not observed at all. Resolves to ``FamiliarityLevel.UNKNOWN`` with reason
    ``MISSING`` -- there is no observation to base a familiarity claim on.

``NULL``
    Observed with no value. Resolves to ``FamiliarityLevel.UNKNOWN`` with reason
    ``NULL_VALUE``. This is a *distinct domain state* from ``MISSING``: the source reported a
    field that carried no usable value, rather than reporting nothing. Both are
    ``UNKNOWN`` (no reliable familiarity claim), but their reasons and raw states differ.

``VALUE(n)``
    An observed, non-negative integer count. Resolves to ``FamiliarityLevel.KNOWN`` with reason
    ``OBSERVED_EXPOSURE``. ``VALUE(0)`` is *observed zero exposure* and is intercepted before
    the normalization policy with magnitude ``0``; ``VALUE(n > 0)`` has its magnitude produced
    by the injected :class:`FamiliarityNormalizationPolicy`.

The raw :class:`~music_agent.source_observation.ObservedValue` is preserved unchanged on every
result, so ``MISSING``, ``NULL``, and ``VALUE(0)`` remain distinguishable by provenance even
when their magnitudes coincide.

Magnitude invariant
-------------------

``magnitude`` qualifies *how familiar* a ``KNOWN`` track is and is only meaningful for
``KNOWN``:

``KNOWN``
    ``magnitude`` is required and must satisfy ``0 <= magnitude <= 1``.
``UNKNOWN``
    ``magnitude`` must be ``None``.

A ``float('nan')``, an infinity, a boolean, and any non-numeric value are rejected rather than
coerced.

Calibration seam
----------------

``FamiliarityNormalizationPolicy`` is a pure, injected interface with a single
``normalize(play_count) -> float`` method. No concrete curve, threshold, or saturation constant
is frozen in this module: saturation behavior is the policy's responsibility. This slice only
owns observation semantics and bounded-magnitude validation.

Deferred
--------

Recent engagement, recency, temporal decay, ``last_played_at``, and ``added_to_library_at`` are
*not* implemented here; they belong to later slices (S8 / live wiring). This slice expresses
historical familiarity from ``play_count`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from music_agent.source_observation import ObservedValue, ObservationState


class FamiliarityValidationError(ValueError):
    code = "validation_error"


class FamiliarityLevel(StrEnum):
    """Whether a reliable familiarity claim exists."""

    UNKNOWN = "unknown"
    KNOWN = "known"


class FamiliarityReason(StrEnum):
    """Why a familiarity conclusion is what it is."""

    MISSING = "missing"
    NULL_VALUE = "null_value"
    OBSERVED_EXPOSURE = "observed_exposure"


@runtime_checkable
class FamiliarityNormalizationPolicy(Protocol):
    """Injected, deterministic play-count normalization.

    Maps a non-negative integer ``play_count`` to a bounded familiarity magnitude in ``[0, 1]``.
    The policy is a pure interface with no concrete curve frozen here; callers inject the
    calibration they want and tests inject explicit curves. An implementation must be
    deterministic and free of global mutable state, must not perform I/O or read the clock, and
    must return a finite real number within ``[0, 1]``. Saturation behavior (for example, at
    what count magnitude reaches ``1``) is the policy's responsibility and is intentionally not
    frozen in this slice.
    """

    def normalize(self, play_count: int) -> float:
        """Map a non-negative ``play_count`` to a familiarity magnitude in ``[0, 1]``."""
        ...


@dataclass(frozen=True, slots=True)
class Familiarity:
    """An immutable, validated familiarity conclusion for one observed play count.

    ``level`` selects the conclusion kind. ``raw`` preserves the originating
    :class:`~music_agent.source_observation.ObservedValue` unchanged, so ``MISSING``, ``NULL``,
    and ``VALUE(0)`` are never folded together even when their magnitudes coincide. ``reason``
    names the familiarity-domain meaning of that observation. ``magnitude`` defaults to
    ``None``, the only legal value for ``UNKNOWN``, and must be a finite ``[0, 1]`` number for
    ``KNOWN``.
    """

    level: FamiliarityLevel
    raw: ObservedValue
    reason: FamiliarityReason
    magnitude: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, FamiliarityLevel):
            raise FamiliarityValidationError("level must be a FamiliarityLevel")
        if not isinstance(self.raw, ObservedValue):
            raise FamiliarityValidationError("raw must be an ObservedValue")
        if not isinstance(self.reason, FamiliarityReason):
            raise FamiliarityValidationError("reason must be a FamiliarityReason")

        if self.level is FamiliarityLevel.KNOWN:
            _require_known_magnitude(self.magnitude, label="magnitude")
        elif self.magnitude is not None:
            raise FamiliarityValidationError("UNKNOWN requires magnitude None")


def derive_track_familiarity(
    observed_play_count: ObservedValue,
    policy: FamiliarityNormalizationPolicy,
) -> Familiarity:
    """Derive a :class:`Familiarity` conclusion from one observed ``play_count``.

    ``policy`` is the injected :class:`FamiliarityNormalizationPolicy` consulted for any
    ``VALUE(n > 0)``; ``MISSING``, ``NULL``, and ``VALUE(0)`` are resolved directly without
    consulting the policy, but a valid policy must still be supplied because it is part of the
    API contract. The result is deterministic, side-effect free, and preserves the raw
    observation unchanged.
    """
    if not isinstance(observed_play_count, ObservedValue):
        raise FamiliarityValidationError("observed_play_count must be an ObservedValue")
    if not isinstance(policy, FamiliarityNormalizationPolicy):
        raise FamiliarityValidationError("policy must be a FamiliarityNormalizationPolicy")

    if observed_play_count.state is ObservationState.MISSING:
        return Familiarity(
            FamiliarityLevel.UNKNOWN, observed_play_count, FamiliarityReason.MISSING
        )
    if observed_play_count.state is ObservationState.NULL:
        return Familiarity(
            FamiliarityLevel.UNKNOWN, observed_play_count, FamiliarityReason.NULL_VALUE
        )

    play_count = observed_play_count.payload
    if isinstance(play_count, bool) or not isinstance(play_count, int):
        raise FamiliarityValidationError(
            f"play_count must be an integer, not {type(play_count).__name__}"
        )
    if play_count < 0:
        raise FamiliarityValidationError("play_count must be >= 0")
    if play_count == 0:
        return Familiarity(
            FamiliarityLevel.KNOWN,
            observed_play_count,
            FamiliarityReason.OBSERVED_EXPOSURE,
            0.0,
        )

    magnitude = _require_known_magnitude(
        policy.normalize(play_count), label="familiarity policy result"
    )
    return Familiarity(
        FamiliarityLevel.KNOWN,
        observed_play_count,
        FamiliarityReason.OBSERVED_EXPOSURE,
        magnitude,
    )


def _require_known_magnitude(value: object, *, label: str) -> int | float:
    """Return ``value`` as a finite real magnitude in ``[0, 1]``, failing closed otherwise.

    Booleans are rejected explicitly (``bool`` is an ``int`` subclass), as are ``str``, ``None``,
    and any other non-real type. ``float('nan')`` and infinities are rejected as non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FamiliarityValidationError(
            f"{label} must be an int or float, not bool/None/str"
        )
    if not math.isfinite(value):
        raise FamiliarityValidationError(f"{label} must be finite")
    if not 0 <= value <= 1:
        raise FamiliarityValidationError(f"{label} must be within [0, 1]")
    return value
