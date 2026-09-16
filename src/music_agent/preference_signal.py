"""Signal normalization primitive for explicit preference evidence.

This module is the first slice of preference modeling (P06 S1). It normalizes a single
observed source signal into a typed ``SignalContribution`` and nothing more. It is a pure
domain layer: deterministic, side-effect free, and independent of SQLite rows and external
source payloads. It never computes preference strength, confidence, decay, or inferred
affinity, and never persists anything.

Frozen semantics
----------------

- Preference is not Familiarity.
- Preference strength is not Confidence.
- A direct preference is not an inferred affinity.

Concretely, for the three supported signals:

``favorited``
    ``VALUE(true)`` is a positive explicit contribution. ``VALUE(false)`` is *not* a negative
    preference -- it is ``NO_CLAIM``.

``disliked``
    ``VALUE(true)`` is a negative explicit contribution. ``VALUE(false)`` is ``NO_CLAIM``.

``rating``
    ``VALUE(0)`` is ``NO_CLAIM`` with reason ``AMBIGUOUS_SOURCE_VALUE``: a raw zero carries no
    directional claim and must not be read as positive, negative, neutral, corroborating, or
    contradicting. It is intercepted *before* any band policy runs, and its raw value is
    preserved for provenance. A ``rating > 0`` is converted to a direction only through an
    injected :class:`RatingBandPolicy`; this slice never hard-codes calibration thresholds.

``MISSING`` / ``NULL`` / ``VALUE``
    are three distinct states and are never folded together. ``MISSING`` means not observed,
    ``NULL`` means observed with no value, and ``VALUE`` means an observed payload. All three
    resolve to a non-directional contribution, but their reasons and raw states remain distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.source_observation import ObservedValue, ObservationState


class PreferenceSignalValidationError(ValueError):
    code = "validation_error"


class PreferenceSignal(StrEnum):
    """The kind of observed preference signal being normalized."""

    FAVORITED = "favorited"
    DISLIKED = "disliked"
    RATING = "rating"


class SignalDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NO_CLAIM = "no_claim"


class SignalExplicitness(StrEnum):
    """Whether a directional claim is stated directly or derived from a scalar signal.

    ``EXPLICIT`` marks a direct boolean statement (``favorited``/``disliked`` true).
    ``DERIVED`` marks a direction converted from a ``rating`` value through a band policy.
    ``NONE`` marks a contribution that makes no directional claim, so explicitness does not
    apply.
    """

    EXPLICIT = "explicit"
    DERIVED = "derived"
    NONE = "none"


class SignalReason(StrEnum):
    EXPLICIT_SIGNAL = "explicit_signal"
    RATING_BAND = "rating_band"
    AMBIGUOUS_SOURCE_VALUE = "ambiguous_source_value"
    MISSING = "missing"
    NULL_VALUE = "null_value"
    NO_DIRECTIONAL_CLAIM = "no_directional_claim"


@dataclass(frozen=True, slots=True)
class RatingBandPolicy:
    """Injected, deterministic rating band boundaries.

    A rating of ``1..100`` is classified against two inclusive thresholds:

    - ``rating >= positive_threshold`` -> ``SignalDirection.POSITIVE``
    - ``rating <= negative_threshold`` -> ``SignalDirection.NEGATIVE``
    - otherwise -> ``SignalDirection.NO_CLAIM``

    The policy is a plain immutable value with no global mutable config; callers inject the
    calibration they want, and tests inject explicit thresholds. ``rating=0`` is *not* a valid
    band input and must be intercepted by the normalizer as ``AMBIGUOUS_SOURCE_VALUE`` before
    this policy is consulted.
    """

    positive_threshold: int
    negative_threshold: int

    def __post_init__(self) -> None:
        if isinstance(self.positive_threshold, bool) or not isinstance(
            self.positive_threshold, int
        ):
            raise PreferenceSignalValidationError("positive_threshold must be an integer")
        if isinstance(self.negative_threshold, bool) or not isinstance(
            self.negative_threshold, int
        ):
            raise PreferenceSignalValidationError("negative_threshold must be an integer")
        if not 1 <= self.positive_threshold <= 100:
            raise PreferenceSignalValidationError("positive_threshold must be within 1..100")
        if not 1 <= self.negative_threshold <= 100:
            raise PreferenceSignalValidationError("negative_threshold must be within 1..100")
        if not self.negative_threshold < self.positive_threshold:
            raise PreferenceSignalValidationError(
                "negative_threshold must be less than positive_threshold"
            )

    def classify(self, rating: int) -> SignalDirection:
        """Map a ``1..100`` rating to a direction.

        ``rating=0`` and out-of-range values are rejected rather than silently folded into a
        band, because a raw zero is ``AMBIGUOUS_SOURCE_VALUE``, not a no-claim band member.
        """
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 100:
            raise PreferenceSignalValidationError("rating must be an integer within 1..100")
        if rating >= self.positive_threshold:
            return SignalDirection.POSITIVE
        if rating <= self.negative_threshold:
            return SignalDirection.NEGATIVE
        return SignalDirection.NO_CLAIM


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """A normalized directional claim for a single observed preference signal."""

    signal: PreferenceSignal
    direction: SignalDirection
    raw: ObservedValue
    explicitness: SignalExplicitness
    reason: SignalReason

    def __post_init__(self) -> None:
        if not isinstance(self.signal, PreferenceSignal):
            raise PreferenceSignalValidationError("signal must be a PreferenceSignal")
        if not isinstance(self.direction, SignalDirection):
            raise PreferenceSignalValidationError("direction must be a SignalDirection")
        if not isinstance(self.raw, ObservedValue):
            raise PreferenceSignalValidationError("raw must be an ObservedValue")
        if not isinstance(self.explicitness, SignalExplicitness):
            raise PreferenceSignalValidationError("explicitness must be a SignalExplicitness")
        if not isinstance(self.reason, SignalReason):
            raise PreferenceSignalValidationError("reason must be a SignalReason")


def normalize_preference_signal(
    signal: PreferenceSignal,
    observed: ObservedValue,
    rating_policy: RatingBandPolicy | None = None,
) -> SignalContribution:
    """Normalize one observed preference signal into a :class:`SignalContribution`.

    ``rating_policy`` is required only when ``signal`` is ``PreferenceSignal.RATING`` and the
    observed value is a non-zero rating; a ``rating=0`` is intercepted before the policy is
    consulted and therefore does not require one.
    """
    if not isinstance(signal, PreferenceSignal):
        raise PreferenceSignalValidationError("signal must be a PreferenceSignal")
    if not isinstance(observed, ObservedValue):
        raise PreferenceSignalValidationError("observed must be an ObservedValue")
    if rating_policy is not None and not isinstance(rating_policy, RatingBandPolicy):
        raise PreferenceSignalValidationError("rating_policy must be a RatingBandPolicy")

    if signal is PreferenceSignal.FAVORITED:
        return _normalize_boolean(signal, observed, SignalDirection.POSITIVE)
    if signal is PreferenceSignal.DISLIKED:
        return _normalize_boolean(signal, observed, SignalDirection.NEGATIVE)
    return _normalize_rating(observed, rating_policy)


def _normalize_boolean(
    signal: PreferenceSignal,
    observed: ObservedValue,
    direction_when_true: SignalDirection,
) -> SignalContribution:
    if observed.state is ObservationState.MISSING:
        return SignalContribution(
            signal, SignalDirection.NO_CLAIM, observed, SignalExplicitness.NONE, SignalReason.MISSING
        )
    if observed.state is ObservationState.NULL:
        return SignalContribution(
            signal, SignalDirection.NO_CLAIM, observed, SignalExplicitness.NONE, SignalReason.NULL_VALUE
        )
    value = observed.payload
    if not isinstance(value, bool):
        raise PreferenceSignalValidationError(
            f"{signal.value} must be a boolean, not {type(value).__name__}"
        )
    if value is True:
        return SignalContribution(
            signal, direction_when_true, observed, SignalExplicitness.EXPLICIT, SignalReason.EXPLICIT_SIGNAL
        )
    return SignalContribution(
        signal, SignalDirection.NO_CLAIM, observed, SignalExplicitness.NONE, SignalReason.NO_DIRECTIONAL_CLAIM
    )


def _normalize_rating(
    observed: ObservedValue, rating_policy: RatingBandPolicy | None
) -> SignalContribution:
    if observed.state is ObservationState.MISSING:
        return SignalContribution(
            PreferenceSignal.RATING,
            SignalDirection.NO_CLAIM,
            observed,
            SignalExplicitness.NONE,
            SignalReason.MISSING,
        )
    if observed.state is ObservationState.NULL:
        return SignalContribution(
            PreferenceSignal.RATING,
            SignalDirection.NO_CLAIM,
            observed,
            SignalExplicitness.NONE,
            SignalReason.NULL_VALUE,
        )
    rating = observed.payload
    if isinstance(rating, bool) or not isinstance(rating, int):
        raise PreferenceSignalValidationError(
            f"rating must be an integer, not {type(rating).__name__}"
        )
    if not 0 <= rating <= 100:
        raise PreferenceSignalValidationError("rating must be within 0..100")
    if rating == 0:
        return SignalContribution(
            PreferenceSignal.RATING,
            SignalDirection.NO_CLAIM,
            observed,
            SignalExplicitness.NONE,
            SignalReason.AMBIGUOUS_SOURCE_VALUE,
        )
    if rating_policy is None:
        raise PreferenceSignalValidationError(
            "rating normalization requires a RatingBandPolicy for non-zero ratings"
        )
    direction = rating_policy.classify(rating)
    if direction is SignalDirection.NO_CLAIM:
        return SignalContribution(
            PreferenceSignal.RATING,
            direction,
            observed,
            SignalExplicitness.NONE,
            SignalReason.NO_DIRECTIONAL_CLAIM,
        )
    return SignalContribution(
        PreferenceSignal.RATING,
        direction,
        observed,
        SignalExplicitness.DERIVED,
        SignalReason.RATING_BAND,
    )
