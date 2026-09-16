"""Read-side confidence derivation for directional track preferences (P13-C02).

This module turns the S5 :class:`~music_agent.confidence.ConfidenceComponents` and the S5
aggregation seam into a working read model, computed exclusively from durable evidence -- the
heads and immutable revisions owned by :mod:`~music_agent.preference_persistence_repository`.
It adds reliability on top of the existing preference semantics and changes nothing under
them: it never persists, never mutates, never introduces a new signal, never reads the wall
clock on its own (``now`` is injected), and never implements temporal decay. Playback does not
enter: ``play_count`` takes the familiarity axis only, exactly as before.

Two collaborators sit side by side:

``ConfidenceDerivationPolicy``
    Maps durable evidence metadata onto the seven frozen components. The components are
    *derived from* evidence -- evidence count, evidence provenance, direction agreement, and
    descriptive recency -- and are never stored.
``ConservativeConfidencePolicy``
    The production :class:`~music_agent.confidence.ConfidenceAggregationPolicy` implementation.
    It folds a scoped claim into a single bounded reliability score in ``[0, 1]`` using
    multiplicative geometry: any zeroed component floors the score, and directional conflict
    reduces but does not erase it. It maps an absent claim to ``None``.

Component mapping (all inputs already durable)
----------------------------------------------

``quantity``
    One per *distinct* directional evidence event: a baseline / transition revision whose
    normalized contribution is POSITIVE or NEGATIVE. Same-value confirmations never create a
    revision, so the count is already deduplicated and repeated polling can never inflate it.
``quality``
    The arithmetic mean of the per-revision evidence quality looked up from
    ``quality_by_provenance`` by each revision's provenance label; unknown labels fall back to
    ``default_quality``.
``freshness``
    *Descriptive* recency: how near the newest ``last_observed_at`` of the three directional
    signal heads is to the injected ``now``, within ``freshness_window``. It is ``1.0`` when the
    window is disabled (``None``) or ``now`` is absent. Description only: the production
    aggregation assigns ``freshness_weight`` zero, so freshness can never decay a reliability
    score -- no temporal decay is implemented anywhere here.
``consistency``
    The share of directional evidence agreeing with the majority direction
    (``max(positives, negatives) / total``).
``contradiction``
    ``PRESENT`` iff directional evidence exists in both directions. A conflict is evidence, not
    absence of evidence, so it stays an explicit state and is never zero-masked.
``source_reliability``
    The observing ``source_system`` looked up from ``source_reliability_by_system``; unknown
    systems fall back to ``default_source_reliability``.
``inference_distance``
    ``0.0``: this module derives direct-track claims only. Inferred-affinity confidence is a
    later slice.

Claim policy (frozen S5 contract)
---------------------------------

``UNKNOWN`` / ``INSUFFICIENT`` conclusions carry no claim (there is nothing to be reliable
about) and NEUTRAL conclusions carry none either (a directionless conclusion has no directional
reliability). Only ``POSITIVE`` / ``NEGATIVE`` / ``CONFLICT`` produce a
:class:`~music_agent.confidence.ConfidenceClaim` scoped ``CURRENT_PREFERENCE``; ``CONFLICT`` is
kept visible through ``contradiction=PRESENT`` rather than dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Mapping

from music_agent.confidence import (
    ClaimScope,
    ConfidenceAggregationPolicy,
    ConfidenceClaim,
    ConfidenceComponents,
    ConfidenceValidationError,
    Contradiction,
)
from music_agent.learning_policy import (
    EXPLICIT_FEEDBACK_PROVENANCE,
    FEEDBACK_LEARNING_SOURCE_SYSTEM,
    IMPLICIT_FEEDBACK_PROVENANCE,
)
from music_agent.preference_attribution import PreferenceTargetReference
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    SignalIdentity,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_signal import (
    PreferenceSignal,
    RatingBandPolicy,
    SignalDirection,
    normalize_preference_signal,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.source_observation import ObservedValue


class ConfidenceDerivationError(ValueError):
    code = "confidence_derivation_error"


# The P06-frozen directional preference signal set, mirrored locally to avoid a
# query -> derivation -> query import cycle (preference_query owns the same frozen set).
_PREFERENCE_SIGNAL_PATHS: tuple[str, ...] = ("favorited", "disliked", "rating")

_SIGNAL_PATH_TO_PREFERENCE_SIGNAL: dict[str, PreferenceSignal] = {
    "favorited": PreferenceSignal.FAVORITED,
    "disliked": PreferenceSignal.DISLIKED,
    "rating": PreferenceSignal.RATING,
}


@dataclass(frozen=True, slots=True)
class ConfidenceDerivationPolicy:
    """Calibration mapping durable evidence metadata onto the seven S5 components.

    ``quality_by_provenance`` maps revision provenance labels to per-evidence quality in
    ``[0, 1]``; ``default_quality`` covers labels the table does not know. (The production
    default treats direct source observations and explicit feedback as fully trustworthy and
    implicit feedback as less so.) ``source_reliability_by_system`` maps the observing
    ``source_system`` to reliability in ``[0, 1]`` with ``default_source_reliability`` as the
    unknown-system fallback. ``freshness_window`` enables descriptive recency: ``None`` (the
    production default) keeps ``freshness`` at a neutral ``1.0``; a positive window computes
    ``1 - elapsed / window`` clamped to ``[0, 1]`` against the injected ``now``.

    The policy is an immutable value with no global mutable config. Both mapping attributes
    are stored read-only.
    """

    quality_by_provenance: Mapping[str, float] = field(
        default_factory=lambda: {
            DIRECT_OBSERVATION_PROVENANCE: 1.0,
            EXPLICIT_FEEDBACK_PROVENANCE: 1.0,
            IMPLICIT_FEEDBACK_PROVENANCE: 0.7,
        }
    )
    default_quality: float = 0.5
    source_reliability_by_system: Mapping[str, float] = field(
        default_factory=lambda: {
            "apple_music": 1.0,
            FEEDBACK_LEARNING_SOURCE_SYSTEM: 0.8,
        }
    )
    default_source_reliability: float = 0.5
    freshness_window: timedelta | None = None

    def __post_init__(self) -> None:
        quality = _freeze_unit_table(self.quality_by_provenance, "quality_by_provenance")
        reliability = _freeze_unit_table(
            self.source_reliability_by_system, "source_reliability_by_system"
        )
        object.__setattr__(self, "quality_by_provenance", quality)
        object.__setattr__(self, "source_reliability_by_system", reliability)
        _require_unit_interval(self.default_quality, label="default_quality")
        _require_unit_interval(self.default_source_reliability, label="default_source_reliability")
        if self.freshness_window is not None:
            if not isinstance(self.freshness_window, timedelta):
                raise ConfidenceValidationError("freshness_window must be a timedelta or None")
            if self.freshness_window.total_seconds() <= 0:
                raise ConfidenceValidationError("freshness_window must be positive")


@dataclass(frozen=True, slots=True)
class ConservativeConfidencePolicy:
    """Fail-closed production aggregation of a confidence claim into a ``[0, 1]`` score.

    The operator is multiplicative geometry -- every factor contributes and any zeroed factor
    floors the score -- which keeps the score conservative and independently inspectable:

    - ``quantity_factor = min(1, quantity / quantity_full)`` -- reliability grows with distinct
      evidence events only up to ``quantity_full``; more evidence is inspected, not rewarded.
    - ``conflict_factor`` -- ``CONFLICT`` evidence (``contradiction=PRESENT``) halves the score
      by default rather than erasing it: conflicting evidence *reduces* reliability.
    - ``freshness_factor = 1 - freshness_weight * (1 - freshness)`` -- the production
      ``freshness_weight`` is ``0``, which keeps the descriptive freshness from ever decaying
      the score. No temporal decay is implemented.

    An absent claim (``UNKNOWN`` / ``INSUFFICIENT`` / ``NEUTRAL``, or ``None`` input) maps to
    ``None``: the policy never invents a reliability number where no claim exists.
    """

    quantity_full: float = 2.0
    conflict_penalty: float = 0.5
    freshness_weight: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.quantity_full, bool)
            or not isinstance(self.quantity_full, (int, float))
            or not math.isfinite(self.quantity_full)
            or self.quantity_full <= 0
        ):
            raise ConfidenceValidationError("quantity_full must be a positive finite number")
        _require_unit_interval(self.conflict_penalty, label="conflict_penalty")
        _require_unit_interval(self.freshness_weight, label="freshness_weight")

    def aggregate(self, claim: ConfidenceClaim | None) -> float | None:
        """Fold ``claim`` into a bounded reliability score, or ``None`` when no claim exists."""
        if claim is None:
            return None
        components = claim.components
        quantity_factor = min(1.0, components.quantity / self.quantity_full)
        freshness_factor = 1.0 - self.freshness_weight * (1.0 - components.freshness)
        conflict_factor = (
            self.conflict_penalty
            if components.contradiction is Contradiction.PRESENT
            else 1.0
        )
        score = (
            components.quality
            * components.source_reliability
            * components.consistency
            * quantity_factor
            * freshness_factor
            * conflict_factor
        )
        return min(1.0, max(0.0, score))


def derive_confidence_claim(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    strength: PreferenceStrength,
    *,
    rating_policy: RatingBandPolicy,
    policy: ConfidenceDerivationPolicy,
    source_system: str,
    now: datetime | None = None,
) -> ConfidenceClaim | None:
    """Derive the current-preference confidence claim for one Track, or ``None``.

    ``strength`` is the already-resolved S3 conclusion for the three directional signals (the
    caller resolves it once and passes it in; the derivation never re-resolves direction).
    ``rating_policy`` re-normalizes each durable revision the same way the query module does, so
    the counted evidence agrees with the conclusion being explained. ``policy`` supplies every
    calibration constant; this module owns none. ``source_system`` selects the heads and
    revisions being judged. ``now`` is the injected clock for the descriptive ``freshness``
    component (``None`` keeps freshness neutral at ``1.0``).

    ``UNKNOWN`` / ``INSUFFICIENT`` / ``NEUTRAL`` conclusions produce ``None`` -- no claim is
    invented where no directional conclusion exists. The result is deterministic, side-effect
    free, and read-only over the repository.
    """
    if not isinstance(repository, PreferencePersistenceRepository):
        raise ConfidenceDerivationError("repository must be a PreferencePersistenceRepository")
    if not isinstance(target, PreferenceTargetReference):
        raise ConfidenceDerivationError("target must be a PreferenceTargetReference")
    if not isinstance(strength, PreferenceStrength):
        raise ConfidenceDerivationError("strength must be a PreferenceStrength")
    if not isinstance(rating_policy, RatingBandPolicy):
        raise ConfidenceDerivationError("rating_policy must be a RatingBandPolicy")
    if not isinstance(policy, ConfidenceDerivationPolicy):
        raise ConfidenceDerivationError("policy must be a ConfidenceDerivationPolicy")
    _require_non_empty_string(source_system, "source_system")
    if now is not None:
        _require_aware_datetime(now)

    if strength.state not in (
        PreferenceState.POSITIVE,
        PreferenceState.NEGATIVE,
        PreferenceState.CONFLICT,
    ):
        return None

    facts = _directional_facts(repository, target, source_system, rating_policy, policy)
    quantity = len(facts)
    positives = sum(1 for direction, _ in facts if direction is SignalDirection.POSITIVE)
    negatives = quantity - positives

    contradiction = (
        Contradiction.PRESENT if positives > 0 and negatives > 0 else Contradiction.NONE
    )
    consistency = max(positives, negatives) / quantity if quantity else 0.0
    quality = (
        sum(evidence_quality for _, evidence_quality in facts) / quantity if quantity else 0.0
    )
    freshness = _derive_freshness(repository, target, source_system, policy, now)

    return ConfidenceClaim(
        ClaimScope.CURRENT_PREFERENCE,
        ConfidenceComponents(
            quality=quality,
            quantity=quantity,
            freshness=freshness,
            consistency=consistency,
            contradiction=contradiction,
            source_reliability=policy.source_reliability_by_system.get(
                source_system, policy.default_source_reliability
            ),
            inference_distance=0.0,
        ),
    )


def _directional_facts(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    source_system: str,
    rating_policy: RatingBandPolicy,
    policy: ConfidenceDerivationPolicy,
) -> list[tuple[SignalDirection, float]]:
    """Collect each durable revision whose normalized contribution is directional.

    Each entry is ``(direction, evidence_quality)`` where ``evidence_quality`` is the
    provenance-derived quality of that revision. Revisions normalizing to ``NO_CLAIM`` /
    ``NEUTRAL`` evidence are not directional and contribute nothing; same-value confirmations
    never create a revision, so every collected revision is a distinct evidence event.
    """
    facts: list[tuple[SignalDirection, float]] = []
    for signal_path, signal in _SIGNAL_PATH_TO_PREFERENCE_SIGNAL.items():
        identity = SignalIdentity(target, source_system, signal_path)
        for revision in repository.list_revisions(identity):
            contribution = normalize_preference_signal(
                signal, ObservedValue.value(revision.semantic_value), rating_policy
            )
            if contribution.direction not in (
                SignalDirection.POSITIVE,
                SignalDirection.NEGATIVE,
            ):
                continue
            facts.append(
                (
                    contribution.direction,
                    policy.quality_by_provenance.get(
                        revision.provenance, policy.default_quality
                    ),
                )
            )
    return facts


def _derive_freshness(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    source_system: str,
    policy: ConfidenceDerivationPolicy,
    now: datetime | None,
) -> float:
    """Describe how recent the newest directional-head observation is, in ``[0, 1]``.

    ``1.0`` (neutral) when the descriptive window is disabled, ``now`` is absent, no head has
    been observed, or no head carries a parseable aware timestamp. This component is purely
    descriptive: the production aggregation weights it at zero, so it never decays a score.
    """
    if policy.freshness_window is None or now is None:
        return 1.0
    latest: datetime | None = None
    for signal_path in _PREFERENCE_SIGNAL_PATHS:
        head = repository.get_head(SignalIdentity(target, source_system, signal_path))
        if head is None:
            continue
        observed_at = _parse_aware_datetime(head.last_observed_at)
        if observed_at is not None and (latest is None or observed_at > latest):
            latest = observed_at
    if latest is None:
        return 1.0
    elapsed_seconds = (now - latest).total_seconds()
    if elapsed_seconds <= 0:
        return 1.0
    fraction = min(1.0, elapsed_seconds / policy.freshness_window.total_seconds())
    return 1.0 - fraction


def _freeze_unit_table(table: object, label: str) -> Mapping[str, float]:
    """Validate and freeze a provenance / system -> ``[0, 1]`` calibration table."""
    if not isinstance(table, Mapping) or isinstance(table, (str, bytes)):
        raise ConfidenceValidationError(f"{label} must be a string -> unit-interval mapping")
    frozen: dict[str, float] = {}
    for key, value in table.items():
        if not isinstance(key, str) or key == "":
            raise ConfidenceValidationError(f"{label} keys must be non-empty strings")
        _require_unit_interval(value, label=f"{label}[{key!r}]")
        frozen[key] = value
    return MappingProxyType(frozen)


def _parse_aware_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp to a timezone-aware ``datetime``, or ``None`` when unusable."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _require_unit_interval(value: object, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfidenceValidationError(
            f"{label} must be an int or float, not {type(value).__name__}"
        )
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ConfidenceValidationError(f"{label} must be finite and within [0, 1]")
    return value


def _require_non_empty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or value == "":
        raise ConfidenceValidationError(f"{label} must be a non-empty string")


def _require_aware_datetime(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, datetime):
        raise ConfidenceValidationError("now must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConfidenceValidationError("now must be timezone-aware")