"""Pure resolver from normalized direct-track signals to a preference conclusion.

This module is the third slice of preference modeling (P06 S3). It turns an unordered
collection of :class:`~music_agent.preference_signal.SignalContribution` values (already
normalized by S1) into a single :class:`~music_agent.preference_strength.PreferenceStrength`
conclusion. It is a pure domain function: deterministic, side-effect free, order-independent,
and independent of SQLite rows, external source payloads, the clock, and global state. It never
reads ``favorited`` / ``disliked`` / ``rating`` / ``play_count`` directly, and never persists.

The resolver does not compute Confidence, Familiarity, Artist/Album/Genre propagation, temporal
evolution, or explainability, and it never fuses a categorical claim with a scalar claim into a
weighted sum. Magnitude is supplied by an injected :class:`DirectPreferenceMagnitudePolicy`;
this module owns direction and state logic only and freezes no calibration constants.

Frozen resolution rules
-----------------------

1. **Hard conflict first.** A favorited claim (``favorited=true``, normalized to ``FAVORITED`` +
   ``POSITIVE``) together with a disliked claim (``disliked=true``, ``DISLIKED`` + ``NEGATIVE``)
   is ``CONFLICT``. The conflict is detected and preserved, not explained or re-read.

2. **Categorical explicit claim.** If there is no conflict, a favorited claim resolves to
   ``POSITIVE`` and a disliked claim resolves to ``NEGATIVE``.

3. **Scalar rating claim.** With no categorical claim, a rating's normalized
   ``SignalDirection`` (``POSITIVE`` / ``NEGATIVE``) resolves the direction. ``rating=0`` is
   already normalized by S1 to ``NO_CLAIM`` / ``AMBIGUOUS_SOURCE_VALUE`` and therefore never
   reaches this step as a directional claim.

4. **Otherwise.** Classify ``UNKNOWN`` or ``INSUFFICIENT`` (see below).

Categorical claims take priority over rating: ``favorited=true`` + a negative rating is still
``POSITIVE``, and ``disliked=true`` + a positive rating is still ``NEGATIVE``. A rating
disagreement is not a hard ``CONFLICT``; it is a consistency issue that is deferred to a later
confidence slice. A rating that *agrees* with a categorical claim does not double-count
magnitude -- the direction is decided once and magnitude is looked up once.

``UNKNOWN`` vs ``INSUFFICIENT``
-------------------------------

The boundary is typed on S1's :class:`~music_agent.preference_signal.SignalReason`:

- ``MISSING``, ``NULL_VALUE``, and ``AMBIGUOUS_SOURCE_VALUE`` carry no reliable directional
  information. When every contribution is one of these (or there are none at all), the result is
  ``UNKNOWN``. This includes "all signals MISSING" and "rating=0 only".
- ``NO_DIRECTIONAL_CLAIM`` is a definite, observed non-directional value (``favorited=false``,
  ``disliked=false``, or a middle-band rating). When at least one such contribution exists and
  no directional claim or conflict is present, the result is ``INSUFFICIENT``: there is source
  state, but it is not enough to form a directional claim.

``UNKNOWN``, ``INSUFFICIENT``, and ``NEUTRAL`` are never folded together. This slice has no path
to ``NEUTRAL``: S1 has no signal that expresses an explicit "no directional preference", so S3
v1 must not invent one. ``NEUTRAL`` is reserved for future evidence that can express it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from music_agent.preference_signal import (
    PreferenceSignal,
    SignalContribution,
    SignalDirection,
    SignalReason,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength


class DirectPreferenceResolutionError(ValueError):
    code = "validation_error"


@dataclass(frozen=True, slots=True)
class DirectPreferenceMagnitudePolicy:
    """Injected, deterministic magnitude calibration for directional conclusions.

    Holds one magnitude per directional state. The resolver decides *direction and state*; this
    policy only supplies the magnitude, so no calibration constant is frozen into the resolver.
    Each magnitude must satisfy the S2 directional invariant ``0 < magnitude <= 1``. The policy
    is a plain immutable value with no global mutable config; callers inject the calibration they
    want, and tests inject explicit magnitudes.
    """

    positive_magnitude: float
    negative_magnitude: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positive_magnitude",
            _require_magnitude(self.positive_magnitude, label="positive_magnitude"),
        )
        object.__setattr__(
            self,
            "negative_magnitude",
            _require_magnitude(self.negative_magnitude, label="negative_magnitude"),
        )

    def magnitude_for(self, state: PreferenceState) -> float:
        """Return the calibrated magnitude for a directional state."""
        if state is PreferenceState.POSITIVE:
            return self.positive_magnitude
        if state is PreferenceState.NEGATIVE:
            return self.negative_magnitude
        raise DirectPreferenceResolutionError(
            f"magnitude is only defined for POSITIVE and NEGATIVE, not {state}"
        )


def resolve_direct_track_preference(
    contributions: Iterable[SignalContribution],
    magnitude_policy: DirectPreferenceMagnitudePolicy,
) -> PreferenceStrength:
    """Resolve normalized direct-track contributions into a single preference conclusion.

    ``contributions`` is an unordered collection of ``SignalContribution`` values. At most one
    contribution per ``PreferenceSignal`` is allowed; a duplicate (contradictory or not) fails
    closed rather than silently picking one. ``magnitude_policy`` supplies the magnitude for any
    directional conclusion. The result is order-independent and equal across repeated calls for
    the same input.
    """
    if not isinstance(magnitude_policy, DirectPreferenceMagnitudePolicy):
        raise DirectPreferenceResolutionError(
            "magnitude_policy must be a DirectPreferenceMagnitudePolicy"
        )

    normalized = _normalize_contributions(contributions)

    favorited_positive = _has_claim(normalized, PreferenceSignal.FAVORITED, SignalDirection.POSITIVE)
    disliked_negative = _has_claim(normalized, PreferenceSignal.DISLIKED, SignalDirection.NEGATIVE)
    rating_positive = _has_claim(normalized, PreferenceSignal.RATING, SignalDirection.POSITIVE)
    rating_negative = _has_claim(normalized, PreferenceSignal.RATING, SignalDirection.NEGATIVE)

    if favorited_positive and disliked_negative:
        return PreferenceStrength(PreferenceState.CONFLICT)
    if favorited_positive:
        return PreferenceStrength(
            PreferenceState.POSITIVE,
            _policy_magnitude(magnitude_policy, PreferenceState.POSITIVE),
        )
    if disliked_negative:
        return PreferenceStrength(
            PreferenceState.NEGATIVE,
            _policy_magnitude(magnitude_policy, PreferenceState.NEGATIVE),
        )
    if rating_positive:
        return PreferenceStrength(
            PreferenceState.POSITIVE,
            _policy_magnitude(magnitude_policy, PreferenceState.POSITIVE),
        )
    if rating_negative:
        return PreferenceStrength(
            PreferenceState.NEGATIVE,
            _policy_magnitude(magnitude_policy, PreferenceState.NEGATIVE),
        )
    if any(contribution.reason is SignalReason.NO_DIRECTIONAL_CLAIM for contribution in normalized):
        return PreferenceStrength(PreferenceState.INSUFFICIENT)
    return PreferenceStrength(PreferenceState.UNKNOWN)


def _has_claim(
    contributions: list[SignalContribution],
    signal: PreferenceSignal,
    direction: SignalDirection,
) -> bool:
    return any(
        contribution.signal is signal and contribution.direction is direction
        for contribution in contributions
    )


def _normalize_contributions(
    contributions: Iterable[SignalContribution],
) -> list[SignalContribution]:
    try:
        iterator = iter(contributions)
    except TypeError:
        raise DirectPreferenceResolutionError(
            "contributions must be an iterable of SignalContribution"
        ) from None

    normalized: list[SignalContribution] = []
    seen_signals: set[PreferenceSignal] = set()
    for contribution in iterator:
        if not isinstance(contribution, SignalContribution):
            raise DirectPreferenceResolutionError(
                "each contribution must be a SignalContribution"
            )
        _require_supported_contribution(contribution)
        if contribution.signal in seen_signals:
            raise DirectPreferenceResolutionError(
                f"duplicate contribution for signal {contribution.signal.value}"
            )
        seen_signals.add(contribution.signal)
        normalized.append(contribution)
    return normalized


def _require_supported_contribution(contribution: SignalContribution) -> None:
    """Fail closed on a contribution whose direction its signal can never produce."""
    if contribution.signal is PreferenceSignal.FAVORITED:
        if contribution.direction is SignalDirection.NEGATIVE:
            raise DirectPreferenceResolutionError(
                "favorited cannot produce a negative direction"
            )
        return
    if contribution.signal is PreferenceSignal.DISLIKED:
        if contribution.direction is SignalDirection.POSITIVE:
            raise DirectPreferenceResolutionError(
                "disliked cannot produce a positive direction"
            )
        return
    if contribution.signal is PreferenceSignal.RATING:
        return
    raise DirectPreferenceResolutionError(
        f"unsupported preference signal: {contribution.signal}"
    )


def _policy_magnitude(
    magnitude_policy: DirectPreferenceMagnitudePolicy,
    state: PreferenceState,
) -> int | float:
    """Validate and return the policy's magnitude for a directional state.

    A genuine ``DirectPreferenceMagnitudePolicy`` validates its own values at construction, but
    the resolver re-checks the returned magnitude so an overridden or otherwise inconsistent
    policy still fails closed before a ``PreferenceStrength`` is built.
    """
    magnitude = magnitude_policy.magnitude_for(state)
    return _require_magnitude(magnitude, label="magnitude policy result")


def _require_magnitude(value: object, *, label: str) -> int | float:
    """Return ``value`` as a finite real magnitude in ``(0, 1]``, failing closed otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectPreferenceResolutionError(
            f"{label} must be an int or float, not {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise DirectPreferenceResolutionError(f"{label} must be finite")
    if not 0 < value <= 1:
        raise DirectPreferenceResolutionError(f"{label} must be within (0, 1]")
    return value
