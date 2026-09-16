"""Per-track hierarchical inferred propagation (P06 S7).

This module is the seventh slice of preference modeling. It turns one *direct* Track
preference into a bounded list of *inferred* Artist / Album / Genre contributions, and nothing
more. It is a pure domain layer: deterministic, side-effect free, and independent of SQLite
rows, external source payloads, the system clock, and global state. It never persists, never
computes a final confidence score, and never aggregates across tracks.

Frozen semantics
----------------

Direct preference and inferred affinity remain **structurally separate**. A direct Track
:class:`~music_agent.preference_attribution.DerivedPreference` is never mutated by propagation,
and a propagated result is a list of :class:`InferredAffinityContribution` values whose
``provenance`` is fixed to ``INFERRED`` -- never a direct preference and never an overwrite of
one.

Only an established directional Track preference may propagate. ``POSITIVE`` and ``NEGATIVE``
produce contributions; ``UNKNOWN``, ``INSUFFICIENT``, ``NEUTRAL``, and ``CONFLICT`` produce an
empty result (they carry no directional claim to propagate).

Propagation is **bounded** (every ``derived_magnitude`` satisfies ``0 < derived <= input <= 1``,
so attenuation never amplifies), **saturating-ready** (each contribution is a bounded
``(0, 1]`` magnitude a future reducer may saturate), **provenance-preserving** (the source track
and the fixed ``INFERRED`` provenance travel with every contribution), and **attribution-aware**
(an applicable :class:`~music_agent.preference_attribution.AttributionConstraint` is resolved per
source-track + target and carried on the contribution).

No production calibration constant is frozen. Every attenuation -- artist split, album
attenuation, genre split, and ``DISCOUNT`` attenuation -- flows through an injected policy seam;
this module validates the policy's output and owns only the structural rules.

Aggregation boundary (deferred)
-------------------------------

This slice produces *per-track* contributions only. It does **not** reduce contributions across
tracks into a final Artist / Album / Genre affinity state. In particular, the frozen rule that
"one Track must not be able to create a strong Artist affinity by itself" is an *aggregate*
claim: a single contribution may carry a large ``derived_magnitude``, but it is not yet an
affinity conclusion. Enforcing a "strong" bound requires a saturation threshold, which is an
unfrozen calibration constant, so that rule is deferred to the future reducer. No reducer seam or
aggregation operator is implemented here.

Temporal boundary
-----------------

S8 owns temporal evolution. This module performs no decay, reads no clock, and takes no ``now``;
it never mutates or re-interprets evidence over time.

Confidence boundary
-------------------

S5 owns confidence. This module never aggregates confidence and never ranks by confidence. It
carries only provenance / magnitude metadata, which a future explainability slice (S9) may reuse.

Artist rules
------------

A Track may have multiple ``artist_ids``. Propagation never duplicates the full contribution to
every artist: a multi-artist track splits its magnitude across its artists through the injected
:class:`ArtistSplitPolicy`; a single-artist track passes its magnitude through unsplit. An
``AttributionConstraint`` of ``BLOCK`` removes that artist's contribution entirely, and
``DISCOUNT`` reduces it further through the injected :class:`DiscountAttenuationPolicy`.

Album rules
-----------

``album_id`` may be ``None``, which yields no album contribution. A valid album id yields one
inferred album contribution. Album affinity is neither stronger nor weaker than artist affinity
by construction: any attenuation is supplied by the optional
:class:`AlbumAttenuationPolicy` seam, and its absence means pass-through (no attenuation).

Genre rules
-----------

Genre remains a string key. Keys are canonicalized with the frozen v1 rule: trim, collapse
internal whitespace, Unicode NFC, preserve case, preserve punctuation, and no alias merge.
Canonicalization is *not* semantic alias resolution: ``"Hip Hop"``, ``"Hip-Hop"``, and
``"hip-hop"`` are three distinct keys. There is no first-class Genre entity and no alias table. A
Track with N genres must not become N full-strength independent signals: a multi-genre track
splits its magnitude through the injected :class:`GenreSplitPolicy`; a single-genre track passes
through unsplit.

AttributionConstraint matching
------------------------------

Constraints are scoped to source Track + target via :class:`AttributionConstraintBinding`.

- ``BLOCK`` removes that propagation path entirely (no contribution for that target).
- ``DISCOUNT`` keeps the path, marks the contribution ``constraint=DISCOUNT``, and reduces the
  magnitude through the injected :class:`DiscountAttenuationPolicy`.
- Multiple constraints for the same source Track + target fail closed: no deterministic
  resolution rule exists, so the input is rejected rather than silently picking one.
- A constraint never changes the Track's own direct preference.

Genre constraint targets must use canonical genre keys; the matcher compares the canonicalized
track genre key against the binding's target key exactly.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from music_agent.preference_attribution import (
    AttributionConstraint,
    ConstraintMode,
    DerivedPreference,
    PreferenceAttributionError,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_signal import SignalDirection
from music_agent.preference_strength import PreferenceState


class PropagationValidationError(ValueError):
    code = "validation_error"


class PropagationKind(StrEnum):
    """The kind of hierarchical relationship a contribution propagates along."""

    ARTIST = "artist"
    ALBUM = "album"
    GENRE = "genre"


_KIND_TO_TARGET_KIND: dict[PropagationKind, PreferenceTargetKind] = {
    PropagationKind.ARTIST: PreferenceTargetKind.ARTIST,
    PropagationKind.ALBUM: PreferenceTargetKind.ALBUM,
    PropagationKind.GENRE: PreferenceTargetKind.GENRE,
}

_PROPAGATABLE_STATES = {PreferenceState.POSITIVE, PreferenceState.NEGATIVE}

_STATE_TO_DIRECTION: dict[PreferenceState, SignalDirection] = {
    PreferenceState.POSITIVE: SignalDirection.POSITIVE,
    PreferenceState.NEGATIVE: SignalDirection.NEGATIVE,
}


def canonicalize_genre_key(genre: str) -> str:
    """Canonicalize one genre key with the frozen v1 rule.

    The frozen canonicalization is: trim, collapse internal whitespace, Unicode NFC, preserve
    case, preserve punctuation, and no alias merge. It is a pure string transformation and is
    *not* semantic alias resolution: ``"Hip Hop"``, ``"Hip-Hop"``, and ``"hip-hop"`` remain three
    distinct keys. A non-string input fails closed rather than being coerced; an empty or
    whitespace-only string canonicalizes to ``""``, which propagation rejects as an invalid key.
    """
    if not isinstance(genre, str):
        raise PropagationValidationError(
            f"genre must be a string, not {type(genre).__name__}"
        )
    normalized = unicodedata.normalize("NFC", genre)
    return " ".join(normalized.split())


@runtime_checkable
class ArtistSplitPolicy(Protocol):
    """Injected, deterministic split of a Track's magnitude across its artists.

    Maps a directional magnitude and the number of artists to the per-artist share. This is the
    seam for the "a Track with multiple artists must not duplicate its full contribution to every
    artist" rule. No split constant is frozen here; callers inject the calibration and tests
    inject explicit splits. An implementation must be deterministic, free of global mutable
    state, and must return a finite real magnitude in ``(0, magnitude]``.
    """

    def split_artist_magnitude(self, magnitude: float, artist_count: int) -> float:
        """Return the per-artist share of ``magnitude`` across ``artist_count`` artists."""
        ...


@runtime_checkable
class AlbumAttenuationPolicy(Protocol):
    """Injected, deterministic attenuation of a Track's magnitude for its album.

    Optional by design: album affinity is neither stronger nor weaker than artist affinity by
    construction, so a caller supplies this seam only when album attenuation is wanted. When no
    policy is supplied, the album contribution passes the magnitude through unchanged. No
    attenuation constant is frozen here.
    """

    def attenuate_album(self, magnitude: float) -> float:
        """Return the attenuated album magnitude for ``magnitude``."""
        ...


@runtime_checkable
class GenreSplitPolicy(Protocol):
    """Injected, deterministic split of a Track's magnitude across its genres.

    Maps a directional magnitude and the number of genres to the per-genre share. This is the
    seam for the "a Track with N genres must not become N full-strength signals" rule. No split
    constant is frozen here.
    """

    def split_genre_magnitude(self, magnitude: float, genre_count: int) -> float:
        """Return the per-genre share of ``magnitude`` across ``genre_count`` genres."""
        ...


@runtime_checkable
class DiscountAttenuationPolicy(Protocol):
    """Injected, deterministic attenuation for a ``DISCOUNT`` attribution constraint.

    Reduces an already-split magnitude when a ``DISCOUNT`` constraint applies. No discount factor
    is frozen here; callers inject the calibration.
    """

    def attenuate_discount(self, magnitude: float) -> float:
        """Return the discounted magnitude for ``magnitude``."""
        ...


@dataclass(frozen=True, slots=True)
class InferredAffinityContribution:
    """One bounded inferred Artist / Album / Genre contribution from one Track.

    ``source_track`` is the direct Track preference's target (kind ``TRACK``). ``target`` is the
    inferred entity (kind ``ARTIST`` / ``ALBUM`` / ``GENRE``). ``direction`` is the established
    directional claim (``POSITIVE`` / ``NEGATIVE``). ``input_magnitude`` is the direct
    preference's magnitude; ``derived_magnitude`` is that magnitude after the injected split /
    attenuation / discount, and always satisfies ``0 < derived <= input <= 1`` so propagation
    never amplifies. ``kind`` names the relationship the contribution propagates along.
    ``constraint`` records the applicable attribution mode (``None`` when unconstrained).
    ``split_count`` records how many sibling targets shared the Track's signal for this kind (1
    for album and single-artist/single-genre), which S9 explainability may reuse.

    ``provenance`` is fixed to ``INFERRED`` by the type, so a contribution can never be mistaken
    for a direct preference.
    """

    source_track: PreferenceTargetReference
    target: PreferenceTargetReference
    direction: SignalDirection
    input_magnitude: float
    derived_magnitude: float
    kind: PropagationKind
    constraint: ConstraintMode | None
    split_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_track, PreferenceTargetReference):
            raise PropagationValidationError("source_track must be a PreferenceTargetReference")
        if self.source_track.kind is not PreferenceTargetKind.TRACK:
            raise PropagationValidationError("source_track must reference a TRACK")
        if not isinstance(self.target, PreferenceTargetReference):
            raise PropagationValidationError("target must be a PreferenceTargetReference")
        if not isinstance(self.kind, PropagationKind):
            raise PropagationValidationError("kind must be a PropagationKind")
        expected = _KIND_TO_TARGET_KIND[self.kind]
        if self.target.kind is not expected:
            raise PropagationValidationError(
                f"kind {self.kind.value} requires target kind {expected.value}"
            )
        if self.direction not in (
            SignalDirection.POSITIVE,
            SignalDirection.NEGATIVE,
        ):
            raise PropagationValidationError(
                "direction must be POSITIVE or NEGATIVE"
            )
        _require_attenuated(self.input_magnitude, 1.0, label="input_magnitude")
        _require_attenuated(
            self.derived_magnitude,
            self.input_magnitude,
            label="derived_magnitude",
        )
        if self.constraint is not None and not isinstance(self.constraint, ConstraintMode):
            raise PropagationValidationError("constraint must be a ConstraintMode or None")
        _require_positive_int(self.split_count, label="split_count")

    @property
    def provenance(self) -> PreferenceProvenance:
        return PreferenceProvenance.INFERRED


@dataclass(frozen=True, slots=True)
class AttributionConstraintBinding:
    """An attribution constraint scoped to one source Track + one target.

    ``source_track`` must reference a ``TRACK`` and ``target`` must reference an ``ARTIST`` /
    ``ALBUM`` / ``GENRE``. ``constraint`` carries the mode. The binding is immutable and has no
    effect on the source Track's own direct preference.
    """

    source_track: PreferenceTargetReference
    target: PreferenceTargetReference
    constraint: AttributionConstraint

    def __post_init__(self) -> None:
        if not isinstance(self.source_track, PreferenceTargetReference):
            raise PropagationValidationError("source_track must be a PreferenceTargetReference")
        if self.source_track.kind is not PreferenceTargetKind.TRACK:
            raise PropagationValidationError("source_track must reference a TRACK")
        if not isinstance(self.target, PreferenceTargetReference):
            raise PropagationValidationError("target must be a PreferenceTargetReference")
        if self.target.kind is PreferenceTargetKind.TRACK:
            raise PropagationValidationError("target must not reference a TRACK")
        if not isinstance(self.constraint, AttributionConstraint):
            raise PropagationValidationError("constraint must be an AttributionConstraint")


def propagate_track_preference(
    track_preference: DerivedPreference,
    *,
    artist_ids: Iterable[str] = (),
    album_id: str | None = None,
    genres: Iterable[str] = (),
    constraints: Iterable[AttributionConstraintBinding] = (),
    artist_split: ArtistSplitPolicy | None = None,
    album_attenuation: AlbumAttenuationPolicy | None = None,
    genre_split: GenreSplitPolicy | None = None,
    discount_attenuation: DiscountAttenuationPolicy | None = None,
) -> list[InferredAffinityContribution]:
    """Propagate one direct Track preference into inferred Artist / Album / Genre contributions.

    ``track_preference`` is the direct Track preference to propagate; its target must be a
    ``TRACK``. ``artist_ids``, ``album_id``, and ``genres`` are the Track's canonical
    relationships (``artist_ids`` is an ordered collection of canonical artist ids, ``album_id``
    is an optional canonical album id, ``genres`` is an ordered collection of genre keys).
    ``constraints`` are :class:`AttributionConstraintBinding` values scoped to this Track.

    ``artist_split`` and ``genre_split`` are required only for multi-artist / multi-genre tracks
    (failing closed if omitted), because a split cannot be decided without an injected policy.
    ``album_attenuation`` is optional; its absence means the album passes through unsplit.
    ``discount_attenuation`` is required only when a ``DISCOUNT`` constraint applies.

    The result is deterministic and ordered: artists (in ``artist_ids`` order), then the album,
    then genres (in ``genres`` order). A non-directional direct state yields an empty list. The
    input ``track_preference`` is never mutated.
    """
    _require_track_preference(track_preference)
    _require_optional_policy(artist_split, ArtistSplitPolicy, "artist_split")
    _require_optional_policy(album_attenuation, AlbumAttenuationPolicy, "album_attenuation")
    _require_optional_policy(genre_split, GenreSplitPolicy, "genre_split")
    _require_optional_policy(discount_attenuation, DiscountAttenuationPolicy, "discount_attenuation")

    state = track_preference.strength.state
    if state not in _PROPAGATABLE_STATES:
        return []

    direction = _STATE_TO_DIRECTION[state]
    input_magnitude = _require_attenuated(
        track_preference.strength.magnitude, 1.0, label="direct magnitude"
    )
    source_track = track_preference.target

    artist_targets = _normalize_artist_targets(artist_ids)
    album_target = _normalize_album_target(album_id)
    genre_targets = _normalize_genre_targets(genres)
    constraint_map = _normalize_constraints(constraints, source_track)

    contributions: list[InferredAffinityContribution] = []

    artist_count = len(artist_targets)
    if artist_count > 1 and artist_split is None:
        raise PropagationValidationError(
            "artist_split policy is required for a multi-artist track"
        )
    for target in artist_targets:
        mode = constraint_map.get(target)
        if mode is ConstraintMode.BLOCK:
            continue
        base = (
            input_magnitude
            if artist_count == 1
            else _require_attenuated(
                artist_split.split_artist_magnitude(input_magnitude, artist_count),
                input_magnitude,
                label="artist split result",
            )
        )
        derived, mode = _apply_discount(base, mode, discount_attenuation)
        contributions.append(
            InferredAffinityContribution(
                source_track=source_track,
                target=target,
                direction=direction,
                input_magnitude=input_magnitude,
                derived_magnitude=derived,
                kind=PropagationKind.ARTIST,
                constraint=mode,
                split_count=artist_count,
            )
        )

    if album_target is not None:
        mode = constraint_map.get(album_target)
        if mode is not ConstraintMode.BLOCK:
            base = (
                input_magnitude
                if album_attenuation is None
                else _require_attenuated(
                    album_attenuation.attenuate_album(input_magnitude),
                    input_magnitude,
                    label="album attenuation result",
                )
            )
            derived, mode = _apply_discount(base, mode, discount_attenuation)
            contributions.append(
                InferredAffinityContribution(
                    source_track=source_track,
                    target=album_target,
                    direction=direction,
                    input_magnitude=input_magnitude,
                    derived_magnitude=derived,
                    kind=PropagationKind.ALBUM,
                    constraint=mode,
                    split_count=1,
                )
            )

    genre_count = len(genre_targets)
    if genre_count > 1 and genre_split is None:
        raise PropagationValidationError(
            "genre_split policy is required for a multi-genre track"
        )
    for target in genre_targets:
        mode = constraint_map.get(target)
        if mode is ConstraintMode.BLOCK:
            continue
        base = (
            input_magnitude
            if genre_count == 1
            else _require_attenuated(
                genre_split.split_genre_magnitude(input_magnitude, genre_count),
                input_magnitude,
                label="genre split result",
            )
        )
        derived, mode = _apply_discount(base, mode, discount_attenuation)
        contributions.append(
            InferredAffinityContribution(
                source_track=source_track,
                target=target,
                direction=direction,
                input_magnitude=input_magnitude,
                derived_magnitude=derived,
                kind=PropagationKind.GENRE,
                constraint=mode,
                split_count=genre_count,
            )
        )

    return contributions


def _require_track_preference(track_preference: object) -> DerivedPreference:
    if not isinstance(track_preference, DerivedPreference):
        raise PropagationValidationError("track_preference must be a DerivedPreference")
    if track_preference.target.kind is not PreferenceTargetKind.TRACK:
        raise PropagationValidationError("track_preference target must be a TRACK")
    return track_preference


def _require_optional_policy(
    policy: object, protocol: type, label: str
) -> None:
    if policy is not None and not isinstance(policy, protocol):
        raise PropagationValidationError(f"{label} must be a {protocol.__name__}")


def _normalize_artist_targets(artist_ids: Iterable[str]) -> list[PreferenceTargetReference]:
    targets: list[PreferenceTargetReference] = []
    seen: set[PreferenceTargetReference] = set()
    for artist_id in _iter_relationships(artist_ids, "artist_ids"):
        target = _canonical_target(PreferenceTargetKind.ARTIST, artist_id, "artist id")
        if target in seen:
            raise PropagationValidationError(f"duplicate artist id {artist_id!r}")
        seen.add(target)
        targets.append(target)
    return targets


def _normalize_album_target(album_id: str | None) -> PreferenceTargetReference | None:
    if album_id is None:
        return None
    if not isinstance(album_id, str):
        raise PropagationValidationError(
            f"album_id must be a string or None, not {type(album_id).__name__}"
        )
    return _canonical_target(PreferenceTargetKind.ALBUM, album_id, "album id")


def _normalize_genre_targets(genres: Iterable[str]) -> list[PreferenceTargetReference]:
    targets: list[PreferenceTargetReference] = []
    seen: set[PreferenceTargetReference] = set()
    for genre in _iter_relationships(genres, "genres"):
        key = canonicalize_genre_key(genre)
        if key == "":
            raise PropagationValidationError("genre must be non-empty after canonicalization")
        target = PreferenceTargetReference(PreferenceTargetKind.GENRE, key)
        if target in seen:
            raise PropagationValidationError(f"duplicate genre {key!r}")
        seen.add(target)
        targets.append(target)
    return targets


def _canonical_target(
    kind: PreferenceTargetKind, target_id: object, label: str
) -> PreferenceTargetReference:
    try:
        return PreferenceTargetReference(kind, target_id)
    except PreferenceAttributionError as error:
        raise PropagationValidationError(
            f"{label} must be a valid canonical {kind.value} id"
        ) from error


def _normalize_constraints(
    constraints: Iterable[AttributionConstraintBinding],
    source_track: PreferenceTargetReference,
) -> dict[PreferenceTargetReference, ConstraintMode]:
    mapping: dict[PreferenceTargetReference, ConstraintMode] = {}
    for binding in _iter_relationships(constraints, "constraints"):
        if not isinstance(binding, AttributionConstraintBinding):
            raise PropagationValidationError(
                "each constraint must be an AttributionConstraintBinding"
            )
        if binding.source_track != source_track:
            raise PropagationValidationError(
                "constraint source_track must match the propagated track"
            )
        if binding.target in mapping:
            raise PropagationValidationError(
                f"conflicting constraints for target {binding.target}"
            )
        mapping[binding.target] = binding.constraint.mode
    return mapping


def _apply_discount(
    base: float,
    mode: ConstraintMode | None,
    discount_attenuation: DiscountAttenuationPolicy | None,
) -> tuple[float, ConstraintMode | None]:
    if mode is not ConstraintMode.DISCOUNT:
        return base, mode
    if discount_attenuation is None:
        raise PropagationValidationError(
            "discount_attenuation policy is required when a DISCOUNT constraint applies"
        )
    discounted = _require_attenuated(
        discount_attenuation.attenuate_discount(base),
        base,
        label="discount result",
    )
    return discounted, mode


def _iter_relationships(iterable: object, label: str):
    if isinstance(iterable, (str, bytes)) or not hasattr(iterable, "__iter__"):
        raise PropagationValidationError(
            f"{label} must be an iterable, not {type(iterable).__name__}"
        )
    return iter(iterable)


def _require_attenuated(value: object, cap: float, *, label: str) -> float:
    """Return ``value`` as a finite real magnitude in ``(0, cap]``, failing closed otherwise.

    Booleans, non-numeric types, ``nan``, infinities, and values outside the half-open bound are
    rejected rather than coerced. The ``cap`` enforces that attenuation never amplifies: a split
    or discount result can never exceed the magnitude it was derived from.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PropagationValidationError(
            f"{label} must be an int or float, not {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise PropagationValidationError(f"{label} must be finite")
    if not 0 < value <= cap:
        raise PropagationValidationError(f"{label} must be within (0, {cap}]")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PropagationValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise PropagationValidationError(f"{label} must be >= 1")
    return value
