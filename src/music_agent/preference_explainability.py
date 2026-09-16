"""Structured explainability contract for derived preferences (P06 S9).

This module is the ninth slice of preference modeling. It defines the frozen, typed
representation that answers, in structured data, "why does the system think this target has this
preference?" and nothing more. It is a pure domain layer: deterministic, side-effect free, and
independent of SQLite rows, external source payloads, the system clock, and global state.

A :class:`DerivedPreferenceExplanation` is a **view** over already-derived facts, never a
derivation itself. It does not recompute a preference direction, does not aggregate confidence,
does not reduce multi-track affinity into one number, does not invent propagation weights, does
not re-run temporal classification, does not mutate evidence, does not query a repository, and
does not generate natural-language prose. Every fact it carries was produced by an earlier slice
(S1--S8); S9 only binds those facts into one validated, immutable structure, leaving any value
that does not yet exist as absent rather than deriving it here.

Frozen separation
-----------------

Direct and inferred evidence remain **structurally separate**. An explanation never flattens the
two into one anonymous evidence list: ``direct_preference`` holds the single
:class:`~music_agent.preference_attribution.DerivedPreference` (if any) and
``inferred_contributions`` holds the per-track
:class:`~music_agent.preference_propagation.InferredAffinityContribution` list, each preserving
its source Track, destination target, propagation kind, input magnitude, derived magnitude, and
constraint metadata. The two are distinct fields, never merged.

Derivation kind
---------------

``derivation`` is one of :class:`DerivationKind`:

``DIRECT``
    The conclusion is a direct observation. ``direct_preference`` is required and
    ``inferred_contributions`` must be empty.
``INFERRED``
    The conclusion is inferred affinity. ``inferred_contributions`` is non-empty and
    ``direct_preference`` must be ``None``.
``EFFECTIVE``
    A future combined view reconciling direct and inferred facts. It requires *both*
    ``direct_preference`` and ``inferred_contributions``. S9 defines the shape but ships no
    producer and performs no reconciliation.

Confidence boundary
-------------------

``UNKNOWN`` / ``INSUFFICIENT`` explanations carry **no** confidence claims and no fabricated
score. Confidence is exposed only as the S5 :class:`~music_agent.confidence.ConfidenceClaim` /
:class:`~music_agent.confidence.ConfidenceComponents` values that already exist, and S9
introduces no aggregation operator and never produces a final numeric confidence score. Claim
scope must be compatible with the derivation kind: ``DIRECT`` allows ``CURRENT_PREFERENCE`` and
``HISTORICAL_PREFERENCE``, ``INFERRED`` allows ``INFERRED_AFFINITY``, and ``EFFECTIVE`` allows
all three.

Temporal and conflict boundary
------------------------------

Temporal state is exposed through the S8 :class:`~music_agent.temporal_evolution.TemporalInterpretation`
and is never collapsed into conflict, and conflict is never collapsed into temporal evolution.
The S8 relations ``CONTEMPORANEOUS_CONFLICT``, ``TEMPORAL_EVOLUTION``,
``NO_TEMPORAL_CONTRADICTION``, and ``INDETERMINATE`` remain visible. Structured conflicts are
carried as :class:`PreferenceConflict` values whose :class:`ConflictKind` distinguishes a direct
categorical conflict, a direct-vs-inferred disagreement, temporal evolution, and indeterminate
temporal ordering -- never diagnosing a cause (stale read, race) unless the domain evidence
already proves it.

Contributing signals
--------------------

S3 resolves to a single :class:`~music_agent.preference_strength.PreferenceStrength` and drops the
per-signal contribution detail. S9 does not rewrite S3; instead callers provide the contributing
normalized signals explicitly as :class:`ContributingSignal` values, each binding one S1
:class:`~music_agent.preference_signal.SignalContribution` (signal identity, normalized direction,
reason, and raw observation) to the source Track it was observed about.

Validation
----------

Construction fails closed on: an explanation target that does not match a contained direct or
inferred target; a malformed preference result; a malformed inferred contribution; an
incompatible confidence scope; an attribution constraint whose target is unrelated to the
explanation target; a duplicate logically-identical inferred contribution or contributing signal
(where duplication would misrepresent evidence); and a mutable collection alias (every
collection is defensively copied into an immutable ``tuple``). ``DIRECT`` explanations may not
carry attribution constraints, which govern cross-entity propagation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.confidence import ClaimScope, ConfidenceClaim
from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import (
    AttributionConstraintBinding,
    InferredAffinityContribution,
)
from music_agent.preference_signal import SignalContribution, SignalDirection
from music_agent.preference_strength import PreferenceStrength
from music_agent.temporal_evolution import TemporalInterpretation


class ExplainabilityValidationError(ValueError):
    code = "validation_error"


class DerivationKind(StrEnum):
    """Whether a preference conclusion is observed directly, inferred, or a combined effective view."""

    DIRECT = "direct"
    INFERRED = "inferred"
    EFFECTIVE = "effective"


class ConflictKind(StrEnum):
    """The frozen set of conflict kinds an explanation can distinguish."""

    DIRECT_CATEGORICAL = "direct_categorical"
    DIRECT_VS_INFERRED = "direct_vs_inferred"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    INDETERMINATE_TEMPORAL = "indeterminate_temporal"


@dataclass(frozen=True, slots=True)
class ContributingSignal:
    """One normalized contributing signal bound to the Track it was observed about.

    ``contribution`` is the S1 :class:`~music_agent.preference_signal.SignalContribution`,
    carrying the signal identity, normalized direction, reason, explicitness, and raw
    observation. ``source_target`` is the ``TRACK`` the signal was observed about. This is the
    S9 additive path for the per-signal detail that S3's single
    :class:`~music_agent.preference_strength.PreferenceStrength` result drops.
    """

    contribution: SignalContribution
    source_target: PreferenceTargetReference

    def __post_init__(self) -> None:
        if not isinstance(self.contribution, SignalContribution):
            raise ExplainabilityValidationError("contribution must be a SignalContribution")
        if not isinstance(self.source_target, PreferenceTargetReference):
            raise ExplainabilityValidationError("source_target must be a PreferenceTargetReference")
        if self.source_target.kind is not PreferenceTargetKind.TRACK:
            raise ExplainabilityValidationError("source_target must reference a TRACK")


@dataclass(frozen=True, slots=True)
class PreferenceConflict:
    """One structured conflict, preserving both opposing directions and optional provenance.

    ``kind`` distinguishes the conflict. ``first_direction`` and ``second_direction`` are the two
    opposing directional claims (always ``POSITIVE`` vs ``NEGATIVE`` in some order). ``first_source``
    and ``second_source`` optionally name where each side came from. This type distinguishes a
    conflict; it never diagnoses its cause.
    """

    kind: ConflictKind
    first_direction: SignalDirection
    second_direction: SignalDirection
    first_source: PreferenceTargetReference | None = None
    second_source: PreferenceTargetReference | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConflictKind):
            raise ExplainabilityValidationError("kind must be a ConflictKind")
        for direction, label in (
            (self.first_direction, "first_direction"),
            (self.second_direction, "second_direction"),
        ):
            if direction not in (SignalDirection.POSITIVE, SignalDirection.NEGATIVE):
                raise ExplainabilityValidationError(f"{label} must be POSITIVE or NEGATIVE")
        if self.first_direction is self.second_direction:
            raise ExplainabilityValidationError("a conflict requires opposite directions")
        if self.first_source is not None and not isinstance(
            self.first_source, PreferenceTargetReference
        ):
            raise ExplainabilityValidationError("first_source must be a PreferenceTargetReference or None")
        if self.second_source is not None and not isinstance(
            self.second_source, PreferenceTargetReference
        ):
            raise ExplainabilityValidationError("second_source must be a PreferenceTargetReference or None")


_ALLOWED_CLAIM_SCOPES: dict[DerivationKind, frozenset[ClaimScope]] = {
    DerivationKind.DIRECT: frozenset(
        {ClaimScope.CURRENT_PREFERENCE, ClaimScope.HISTORICAL_PREFERENCE}
    ),
    DerivationKind.INFERRED: frozenset({ClaimScope.INFERRED_AFFINITY}),
    DerivationKind.EFFECTIVE: frozenset(
        {
            ClaimScope.CURRENT_PREFERENCE,
            ClaimScope.HISTORICAL_PREFERENCE,
            ClaimScope.INFERRED_AFFINITY,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class DerivedPreferenceExplanation:
    """An immutable, validated structured explanation of one preference conclusion.

    ``target`` names the explained target; ``result`` is the already-derived preference
    conclusion being explained; ``derivation`` selects the conclusion kind. ``direct_preference``
    holds the direct conclusion (for ``DIRECT`` / ``EFFECTIVE``), ``inferred_contributions`` the
    per-track inferred contributions (for ``INFERRED`` / ``EFFECTIVE``), ``confidence_claims`` the
    already-derived scoped confidence components, ``temporal_interpretation`` the S8 temporal
    classification, ``conflicts`` the structured conflict records, ``attribution_constraints`` the
    applicable propagation constraints, and ``signals`` the contributing normalized signals.

    Every collection is stored as an immutable ``tuple`` and validated against ``target`` and
    ``derivation``. Nothing is recomputed, aggregated, or invented here.
    """

    target: PreferenceTargetReference
    result: PreferenceStrength
    derivation: DerivationKind
    direct_preference: DerivedPreference | None = None
    inferred_contributions: tuple[InferredAffinityContribution, ...] = ()
    confidence_claims: tuple[ConfidenceClaim, ...] = ()
    temporal_interpretation: TemporalInterpretation | None = None
    conflicts: tuple[PreferenceConflict, ...] = ()
    attribution_constraints: tuple[AttributionConstraintBinding, ...] = ()
    signals: tuple[ContributingSignal, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, PreferenceTargetReference):
            raise ExplainabilityValidationError("target must be a PreferenceTargetReference")
        if not isinstance(self.result, PreferenceStrength):
            raise ExplainabilityValidationError("result must be a PreferenceStrength")
        if not isinstance(self.derivation, DerivationKind):
            raise ExplainabilityValidationError("derivation must be a DerivationKind")

        inferred = _coerce(
            self.inferred_contributions,
            "inferred_contributions",
            InferredAffinityContribution,
        )
        claims = _coerce(self.confidence_claims, "confidence_claims", ConfidenceClaim)
        conflicts = _coerce(self.conflicts, "conflicts", PreferenceConflict)
        constraints = _coerce(
            self.attribution_constraints,
            "attribution_constraints",
            AttributionConstraintBinding,
        )
        signals = _coerce(self.signals, "signals", ContributingSignal)

        _validate_direct(self.target, self.result, self.derivation, self.direct_preference)
        _validate_inferred(self.target, self.derivation, self.direct_preference, inferred)
        _validate_claims(self.derivation, claims)
        _validate_temporal(self.temporal_interpretation)
        _validate_constraints(self.target, self.derivation, constraints)
        _validate_signals(signals)

        object.__setattr__(self, "inferred_contributions", inferred)
        object.__setattr__(self, "confidence_claims", claims)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "attribution_constraints", constraints)
        object.__setattr__(self, "signals", signals)


def _validate_direct(
    target: PreferenceTargetReference,
    result: PreferenceStrength,
    derivation: DerivationKind,
    direct_preference: DerivedPreference | None,
) -> None:
    if direct_preference is None:
        if derivation is DerivationKind.DIRECT:
            raise ExplainabilityValidationError("DIRECT explanation requires direct_preference")
        return
    if not isinstance(direct_preference, DerivedPreference):
        raise ExplainabilityValidationError("direct_preference must be a DerivedPreference")
    if direct_preference.target != target:
        raise ExplainabilityValidationError("direct_preference target must match explanation target")
    if derivation is DerivationKind.DIRECT and direct_preference.strength != result:
        raise ExplainabilityValidationError("result must equal direct_preference.strength for DIRECT")
    if derivation is DerivationKind.INFERRED:
        raise ExplainabilityValidationError("INFERRED explanation must not carry a direct preference")


def _validate_inferred(
    target: PreferenceTargetReference,
    derivation: DerivationKind,
    direct_preference: DerivedPreference | None,
    inferred: tuple[InferredAffinityContribution, ...],
) -> None:
    seen: set[tuple[PreferenceTargetReference, PreferenceTargetReference, object]] = set()
    for contribution in inferred:
        if contribution.target != target:
            raise ExplainabilityValidationError(
                "inferred contribution target must match explanation target"
            )
        key = (contribution.source_track, contribution.target, contribution.kind)
        if key in seen:
            raise ExplainabilityValidationError("duplicate inferred contribution")
        seen.add(key)

    if derivation is DerivationKind.DIRECT and inferred:
        raise ExplainabilityValidationError("DIRECT explanation must not carry inferred contributions")
    if derivation is DerivationKind.INFERRED and not inferred:
        raise ExplainabilityValidationError("INFERRED explanation requires an inferred contribution")
    if (
        derivation is DerivationKind.EFFECTIVE
        and (direct_preference is None or not inferred)
    ):
        raise ExplainabilityValidationError(
            "EFFECTIVE explanation requires both direct_preference and inferred contributions"
        )


def _validate_claims(
    derivation: DerivationKind, claims: tuple[ConfidenceClaim, ...]
) -> None:
    allowed = _ALLOWED_CLAIM_SCOPES[derivation]
    for claim in claims:
        if claim.scope not in allowed:
            raise ExplainabilityValidationError(
                f"confidence scope {claim.scope.value} is incompatible with {derivation.value}"
            )


def _validate_temporal(temporal_interpretation: TemporalInterpretation | None) -> None:
    if temporal_interpretation is not None and not isinstance(
        temporal_interpretation, TemporalInterpretation
    ):
        raise ExplainabilityValidationError(
            "temporal_interpretation must be a TemporalInterpretation or None"
        )


def _validate_constraints(
    target: PreferenceTargetReference,
    derivation: DerivationKind,
    constraints: tuple[AttributionConstraintBinding, ...],
) -> None:
    if derivation is DerivationKind.DIRECT and constraints:
        raise ExplainabilityValidationError(
            "DIRECT explanation must not carry attribution constraints"
        )
    for constraint in constraints:
        if constraint.target != target:
            raise ExplainabilityValidationError(
                "attribution constraint target must match explanation target"
            )


def _validate_signals(signals: tuple[ContributingSignal, ...]) -> None:
    seen: set[tuple[object, PreferenceTargetReference]] = set()
    for signal in signals:
        key = (signal.contribution.signal, signal.source_target)
        if key in seen:
            raise ExplainabilityValidationError(
                "duplicate contributing signal for the same signal and source target"
            )
        seen.add(key)


def _coerce(value: object, label: str, expected_type: type) -> tuple:
    """Return ``value`` as an immutable tuple of ``expected_type``, failing closed otherwise.

    A string or bytes is rejected (it is iterable but not a collection of records), as is any
    non-iterable. Every element must be an instance of ``expected_type``. The returned tuple is a
    defensive copy, so a caller's mutable list is never aliased into the frozen explanation.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ExplainabilityValidationError(
            f"{label} must be an iterable, not {type(value).__name__}"
        )
    items = tuple(value)
    for item in items:
        if not isinstance(item, expected_type):
            raise ExplainabilityValidationError(
                f"each {label} entry must be a {expected_type.__name__}"
            )
    return items
