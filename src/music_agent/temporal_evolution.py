"""Temporal-evolution primitives for preference evidence (P06 S8).

This module is the eighth slice of preference modeling. It classifies the *temporal
relationship* between two immutable preference evidence facts and nothing more. It is a pure
domain layer: deterministic, side-effect free, and independent of SQLite rows, external source
payloads, the system clock, and global state. It never mutates an input fact, never persists,
and never computes a decayed influence, a recency weight, or a confidence score.

Frozen semantics
----------------

Evidence Fact never decays.
    A :class:`PreferenceEvidence` is an immutable fact: it records a target, a directional
    claim, when it was observed (``observed_at``), and when the underlying event happened
    (``event_at``, which may be unknown). Passing a fact through this module never changes it.

Evidence Influence may decay.
    The *influence* a fact exerts on a current interpretation may decay over time, but this
    slice does not implement that. It only declares the :class:`EvidenceInfluencePolicy` seam;
    no concrete decay rule (exponential decay, half-life, fixed window, calibration constant)
    is frozen here.

Conflict != Temporal Evolution.
    Contemporaneous opposite-direction evidence is a hard
    :attr:`TemporalRelation.CONTEMPORANEOUS_CONFLICT`. Temporally ordered opposite-direction
    evidence is :attr:`TemporalRelation.TEMPORAL_EVOLUTION`: a preference that changed over
    time. The two are never folded together.

Historical preference is preserved.
    Both facts are carried on the derived :class:`TemporalInterpretation` unchanged. Nothing is
    erased or overwritten.

Long-term preference does not disappear merely because time passes.
    A historical claim is preserved as a fact regardless of how much time has passed; this
    module never drops a fact for being old.

Recent/current interpretation may change with time.
    Classification is relative to an injected ``now`` and an injected
    :class:`TemporalScopePolicy`, so the same facts may interpret differently at different
    ``now`` instants without any fact changing.

Observed-at vs event-at
-----------------------

``observed_at`` is when the evidence was *observed* (read from a source). It is always known.

``event_at`` is when the event the evidence is *about* happened (for example, when a track was
favorited). It may be unknown (``None``): a source often reports a current boolean with no
timestamp. The two are never folded together: temporal ordering is decided on ``event_at``
only, never on ``observed_at``. When an opposite-direction pair has an unknown ``event_at`` on
either side, the ordering cannot be established and the result is
:attr:`TemporalRelation.INDETERMINATE` rather than a guessed conflict or evolution.

Classification rules
--------------------

Given two facts for the *same* target (a different target fails closed):

1. If either fact makes no directional claim (``NO_CLAIM``), the result is
   ``NO_TEMPORAL_CONTRADICTION`` with reason ``NO_DIRECTIONAL_CLAIM``.
2. If both make the *same* directional claim, the result is ``NO_TEMPORAL_CONTRADICTION`` with
   reason ``SAME_DIRECTION``.
3. If the directions are opposite and either ``event_at`` is unknown, the result is
   ``INDETERMINATE`` with reason ``UNKNOWN_EVENT_TIME``.
4. If the directions are opposite and both ``event_at`` are known, the injected
   :class:`TemporalScopePolicy` decides whether they are contemporaneous (same temporal scope)
   or temporally ordered (different scopes). Same scope -> ``CONTEMPORANEOUS_CONFLICT`` with
   reason ``CONTEMPORANEOUS_OPPOSITE``; different scope -> ``TEMPORAL_EVOLUTION`` with reason
   ``TEMPORALLY_ORDERED_OPPOSITE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from music_agent.preference_signal import SignalDirection


class TemporalEvolutionValidationError(ValueError):
    code = "validation_error"


class TemporalRelation(StrEnum):
    """The frozen set of temporal relationships between two evidence facts."""

    CONTEMPORANEOUS_CONFLICT = "contemporaneous_conflict"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    NO_TEMPORAL_CONTRADICTION = "no_temporal_contradiction"
    INDETERMINATE = "indeterminate"


class TemporalReason(StrEnum):
    """Why a temporal relationship is what it is."""

    SAME_DIRECTION = "same_direction"
    NO_DIRECTIONAL_CLAIM = "no_directional_claim"
    CONTEMPORANEOUS_OPPOSITE = "contemporaneous_opposite"
    TEMPORALLY_ORDERED_OPPOSITE = "temporally_ordered_opposite"
    UNKNOWN_EVENT_TIME = "unknown_event_time"


@dataclass(frozen=True, slots=True)
class PreferenceEvidence:
    """An immutable, validated preference evidence fact with temporal metadata.

    ``target`` is an opaque, non-empty target key (a canonical id in practice). ``direction``
    reuses the S1 ``SignalDirection`` vocabulary. ``observed_at`` is the time the evidence was
    observed and is always known. ``event_at`` is the time of the underlying event and may be
    ``None`` when the source does not expose it.

    Both timestamps must be timezone-aware. A fact is immutable: it never decays and is never
    modified by temporal logic.
    """

    target: str
    direction: SignalDirection
    observed_at: datetime
    event_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or self.target == "":
            raise TemporalEvolutionValidationError("target must be a non-empty string")
        if not isinstance(self.direction, SignalDirection):
            raise TemporalEvolutionValidationError("direction must be a SignalDirection")
        _require_aware_datetime(self.observed_at, label="observed_at")
        if self.event_at is not None:
            _require_aware_datetime(self.event_at, label="event_at")


@runtime_checkable
class TemporalScopePolicy(Protocol):
    """Injected, deterministic contemporaneity decision.

    Decides whether two ``event_at`` instants fall within the same temporal scope at ``now``.
    ``True`` means "contemporaneous" (same scope); ``False`` means "temporally ordered"
    (different scopes). This is the seam for a future fixed window, half-life, or recency rule;
    no concrete window or calibration constant is frozen in this module.
    """

    def contemporaneous(self, first: datetime, second: datetime, *, now: datetime) -> bool:
        """Return ``True`` when two event times share a temporal scope at ``now``."""
        ...


@runtime_checkable
class EvidenceInfluencePolicy(Protocol):
    """Future seam for recency-weighted evidence influence.

    An evidence *fact* never decays; the *influence* a fact exerts on a current interpretation
    may. A future implementation may weight influence by exponential decay, half-life, or a
    fixed window. This module ships no concrete rule, constant, or operator, and this seam has
    no production implementation or call site.
    """

    def influence(self, fact: PreferenceEvidence, *, now: datetime) -> object:
        """Return a future-defined influence weight for ``fact`` as of ``now``."""
        ...


@dataclass(frozen=True, slots=True)
class TemporalInterpretation:
    """A derived temporal interpretation of the relationship between two evidence facts.

    This is the *only* thing temporal logic produces: a classification, never a mutation. The
    input facts are carried unchanged (``first`` and ``second`` in argument order), and ``now``
    records the reference instant the interpretation was computed against.
    """

    relation: TemporalRelation
    reason: TemporalReason
    first: PreferenceEvidence
    second: PreferenceEvidence
    now: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.relation, TemporalRelation):
            raise TemporalEvolutionValidationError("relation must be a TemporalRelation")
        if not isinstance(self.reason, TemporalReason):
            raise TemporalEvolutionValidationError("reason must be a TemporalReason")
        if not isinstance(self.first, PreferenceEvidence):
            raise TemporalEvolutionValidationError("first must be a PreferenceEvidence")
        if not isinstance(self.second, PreferenceEvidence):
            raise TemporalEvolutionValidationError("second must be a PreferenceEvidence")
        _require_aware_datetime(self.now, label="now")


def classify_temporal_relation(
    first: PreferenceEvidence,
    second: PreferenceEvidence,
    *,
    now: datetime,
    scope_policy: TemporalScopePolicy,
) -> TemporalInterpretation:
    """Classify the temporal relation between two preference evidence facts.

    ``first`` and ``second`` must share the same ``target`` (anything else fails closed).
    ``now`` is the injected reference instant (never the system clock) and ``scope_policy`` is
    the injected contemporaneity decision. The result is deterministic, side-effect free, and
    never mutates either input fact.
    """
    if not isinstance(first, PreferenceEvidence):
        raise TemporalEvolutionValidationError("first must be a PreferenceEvidence")
    if not isinstance(second, PreferenceEvidence):
        raise TemporalEvolutionValidationError("second must be a PreferenceEvidence")
    _require_aware_datetime(now, label="now")
    if not isinstance(scope_policy, TemporalScopePolicy):
        raise TemporalEvolutionValidationError("scope_policy must be a TemporalScopePolicy")

    if first.target != second.target:
        raise TemporalEvolutionValidationError(
            "first and second must share the same target"
        )

    if (
        first.direction is SignalDirection.NO_CLAIM
        or second.direction is SignalDirection.NO_CLAIM
    ):
        return TemporalInterpretation(
            TemporalRelation.NO_TEMPORAL_CONTRADICTION,
            TemporalReason.NO_DIRECTIONAL_CLAIM,
            first,
            second,
            now,
        )

    if first.direction is second.direction:
        return TemporalInterpretation(
            TemporalRelation.NO_TEMPORAL_CONTRADICTION,
            TemporalReason.SAME_DIRECTION,
            first,
            second,
            now,
        )

    if first.event_at is None or second.event_at is None:
        return TemporalInterpretation(
            TemporalRelation.INDETERMINATE,
            TemporalReason.UNKNOWN_EVENT_TIME,
            first,
            second,
            now,
        )

    contemporaneous = scope_policy.contemporaneous(first.event_at, second.event_at, now=now)
    if not isinstance(contemporaneous, bool):
        raise TemporalEvolutionValidationError(
            "scope_policy.contemporaneous must return a bool"
        )
    if contemporaneous:
        return TemporalInterpretation(
            TemporalRelation.CONTEMPORANEOUS_CONFLICT,
            TemporalReason.CONTEMPORANEOUS_OPPOSITE,
            first,
            second,
            now,
        )
    return TemporalInterpretation(
        TemporalRelation.TEMPORAL_EVOLUTION,
        TemporalReason.TEMPORALLY_ORDERED_OPPOSITE,
        first,
        second,
        now,
    )


def _require_aware_datetime(value: object, *, label: str) -> datetime:
    """Return ``value`` as a timezone-aware ``datetime``, failing closed otherwise.

    A naive datetime, a date, a string, and any non-datetime are rejected rather than coerced.
    """
    if not isinstance(value, datetime):
        raise TemporalEvolutionValidationError(
            f"{label} must be a datetime, not {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise TemporalEvolutionValidationError(f"{label} must be timezone-aware")
    return value
