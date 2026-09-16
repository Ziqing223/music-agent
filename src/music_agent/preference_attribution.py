"""Direct vs inferred preference separation and the attribution constraint (P06 S6).

This module is the sixth slice of preference modeling. It establishes the *structural* split
between a direct preference and an inferred affinity, a typed target reference, and the
cross-entity propagation constraint, and nothing more. It does not propagate a Track preference
to Artist/Album/Genre, does not aggregate affinity, does not split multi-artist/multi-genre
signals, does not aggregate confidence, and does not persist anything. It is a pure domain
layer: deterministic, side-effect free, independent of SQLite rows, external source payloads,
the clock, and global state.

Frozen separation
-----------------

Direct and inferred conclusions are **structurally distinct types**, not two flavours of one
value:

``DerivedPreference``
    A *direct* preference conclusion for a target. Its ``provenance`` is ``DIRECT`` and is fixed
    by the type -- it can never be anything else, and it is not a constructor argument.

``InferredAffinity``
    An *inferred* affinity conclusion for a target. Its ``provenance`` is ``INFERRED`` and is
    likewise fixed by the type.

Because provenance is a fixed property of the type rather than a settable field, the two are
never equal, never interchangeable, and can never be mistaken for one another even when they
reference the same target with the same :class:`~music_agent.preference_strength.PreferenceStrength`.
An inferred affinity can never *become* a direct preference, and there is no operation in this
slice that replaces or overwrites a direct preference with an inferred one.

Fallback eligibility
--------------------

Inferred fallback is gated on the direct preference state. :func:`is_inferred_fallback_eligible`
encodes the frozen rule that an inferred affinity may only ever fill a gap left by a direct
preference, and never override one:

- ``POSITIVE``, ``NEGATIVE``, ``NEUTRAL``, ``CONFLICT`` -> **ineligible**. A direct preference
  has already formed a conclusion; inferred affinity must not override or replace it.
- ``UNKNOWN``, ``INSUFFICIENT`` -> **eligible**. There is no usable direct conclusion, so a
  future slice may fall back to an inferred affinity.

This is a pure predicate. It does not select a "best" inferred affinity and does not combine
direct and inferred values; the future fallback/selection slice remains deferred.

Attribution constraint
----------------------

:class:`AttributionConstraint` governs *future* cross-entity propagation only. It never changes
a Track's own direct preference. Its :class:`ConstraintMode` is exactly one of:

``BLOCK``
    Propagation is forbidden.
``DISCOUNT``
    Propagation is permitted but requires a future attenuation rule.

No numeric discount factor is frozen in this slice: the mode carries the *kind* of constraint and
nothing else, so a discount magnitude can be chosen by a later slice without invalidating this
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.identity import (
    EntityType,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength


class PreferenceAttributionError(ValueError):
    code = "validation_error"


class PreferenceTargetKind(StrEnum):
    """The kind of entity a preference target references."""

    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"
    GENRE = "genre"


class PreferenceProvenance(StrEnum):
    """Whether a preference conclusion is observed directly or inferred."""

    DIRECT = "direct"
    INFERRED = "inferred"


class ConstraintMode(StrEnum):
    """The frozen set of cross-entity propagation constraint modes."""

    BLOCK = "block"
    DISCOUNT = "discount"


_CANONICAL_ENTITY_TYPES: dict[PreferenceTargetKind, EntityType] = {
    PreferenceTargetKind.TRACK: EntityType.TRACK,
    PreferenceTargetKind.ARTIST: EntityType.ARTIST,
    PreferenceTargetKind.ALBUM: EntityType.ALBUM,
}


@dataclass(frozen=True, slots=True)
class PreferenceTargetReference:
    """A reference to one preference target with kind-specific identity validation.

    ``kind`` selects the target kind (``TRACK`` / ``ARTIST`` / ``ALBUM`` / ``GENRE``).
    ``target_id`` names the target:

    - ``TRACK`` / ``ARTIST`` / ``ALBUM`` must be a valid canonical entity ID, validated with
      :func:`music_agent.identity.validate_canonical_id` against the kind's entity namespace.
    - ``GENRE`` is an independent non-empty string key: genre has no canonical entity namespace,
      so it is not subject to canonical-ID validation.

    Canonical entity identity safety is not relaxed to share one target API: the kind decides
    the validation rule, and the canonical kinds never accept an arbitrary non-empty string.
    """

    kind: PreferenceTargetKind
    target_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PreferenceTargetKind):
            raise PreferenceAttributionError("kind must be a PreferenceTargetKind")
        if not isinstance(self.target_id, str) or self.target_id == "":
            raise PreferenceAttributionError("target_id must be a non-empty string")
        entity_type = _CANONICAL_ENTITY_TYPES.get(self.kind)
        if entity_type is not None:
            try:
                validate_canonical_id(entity_type, self.target_id)
            except IdentityValidationError as error:
                raise PreferenceAttributionError(
                    f"target_id for {self.kind.value} must be a valid canonical id"
                ) from error


@dataclass(frozen=True, slots=True)
class DerivedPreference:
    """A direct preference conclusion for a target.

    ``target`` names the referenced target and ``strength`` is the direct
    :class:`~music_agent.preference_strength.PreferenceStrength` conclusion. ``provenance`` is
    fixed to ``DIRECT`` by the type and is not a constructor argument, so a direct preference can
    never be constructed with inferred provenance.
    """

    target: PreferenceTargetReference
    strength: PreferenceStrength

    def __post_init__(self) -> None:
        if not isinstance(self.target, PreferenceTargetReference):
            raise PreferenceAttributionError("target must be a PreferenceTargetReference")
        if not isinstance(self.strength, PreferenceStrength):
            raise PreferenceAttributionError("strength must be a PreferenceStrength")

    @property
    def provenance(self) -> PreferenceProvenance:
        return PreferenceProvenance.DIRECT


@dataclass(frozen=True, slots=True)
class InferredAffinity:
    """An inferred affinity conclusion for a target.

    ``target`` names the referenced target and ``strength`` is the inferred
    :class:`~music_agent.preference_strength.PreferenceStrength` conclusion. ``provenance`` is
    fixed to ``INFERRED`` by the type, so an inferred affinity can never be constructed with
    direct provenance and can never be mistaken for a :class:`DerivedPreference`.
    """

    target: PreferenceTargetReference
    strength: PreferenceStrength

    def __post_init__(self) -> None:
        if not isinstance(self.target, PreferenceTargetReference):
            raise PreferenceAttributionError("target must be a PreferenceTargetReference")
        if not isinstance(self.strength, PreferenceStrength):
            raise PreferenceAttributionError("strength must be a PreferenceStrength")

    @property
    def provenance(self) -> PreferenceProvenance:
        return PreferenceProvenance.INFERRED


@dataclass(frozen=True, slots=True)
class AttributionConstraint:
    """A cross-entity propagation constraint.

    ``mode`` is one of :class:`ConstraintMode`: ``BLOCK`` forbids propagation and ``DISCOUNT``
    permits it subject to a future attenuation rule. The constraint governs *future* propagation
    only and never changes a Track's own direct preference. No numeric discount factor is frozen
    here -- the mode is the whole value, so a later slice can attach a discount magnitude without
    changing this type.
    """

    mode: ConstraintMode

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ConstraintMode):
            raise PreferenceAttributionError("mode must be a ConstraintMode")


def is_inferred_fallback_eligible(direct_state: PreferenceState) -> bool:
    """Return whether a direct preference state permits an inferred fallback.

    ``POSITIVE``, ``NEGATIVE``, ``NEUTRAL``, and ``CONFLICT`` are ineligible: a direct preference
    has already formed a conclusion, so inferred affinity must not override it. ``UNKNOWN`` and
    ``INSUFFICIENT`` are eligible: there is no usable direct conclusion, so a future slice may
    fall back to an inferred affinity. A non-:class:`PreferenceState` input fails closed rather
    than being coerced.
    """
    if not isinstance(direct_state, PreferenceState):
        raise PreferenceAttributionError("direct_state must be a PreferenceState")
    return direct_state in (PreferenceState.UNKNOWN, PreferenceState.INSUFFICIENT)
