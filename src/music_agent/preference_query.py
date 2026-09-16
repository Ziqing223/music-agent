"""Query-time read model: reconstruct Track-level P06 state from durable evidence.

This module is the *read* half of the P06 end-to-end integration. It reconstructs the current
Track-level preference state from the durable heads and evidence revisions owned by
:mod:`~music_agent.preference_persistence_repository`, and nothing else. It is a query service, not
a derivation layer: it reuses the frozen S1/S3/S4/S8/S9 functions verbatim, never persists any
query-time result, never re-implements inference, and never reads the system clock or mutates state.

A single call produces a :class:`TrackPreferenceState` carrying:

``direct_preference``
    The S1-normalized + S3-resolved :class:`~music_agent.preference_attribution.DerivedPreference`
    for the Track, reconstructed from the current heads of ``favorited`` / ``disliked`` / ``rating``.

``familiarity``
    The S4 :class:`~music_agent.familiarity.Familiarity` derived from the current ``play_count`` head.

``temporal_interpretation``
    The S8 :class:`~music_agent.temporal_evolution.TemporalInterpretation` between the earliest and
    latest opposite-direction evidence facts that carry a parseable ``event_at`` -- or ``None`` when
    no such pair exists or no ``scope_policy`` / ``now`` was supplied.

``explanation``
    The S9 :class:`~music_agent.preference_explainability.DerivedPreferenceExplanation` binding the
    direct preference, its contributing normalized signals, the temporal interpretation, and any
    structured conflict.

``confidence_claim``
    The S5 :class:`~music_agent.confidence.ConfidenceClaim` for the current direct preference,
    derived read-only from the durable heads and revisions when ``confidence_policy`` is
    injected, and ``None`` otherwise (a directional conclusion with no directional evidence also
    yields ``None``). Optional and additive: it never changes any other field.

MISSING / NULL / VALUE are reconstructed faithfully from each head's ``current_semantic_value`` and
``last_observed_state``: a head that has ever seen a ``VALUE`` reproduces that value (a later
``MISSING`` / ``NULL`` does not erase it), while a head with no value reproduces its last observed
state (``NULL`` vs ``MISSING``). All three states therefore remain distinguishable through the
derived result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from music_agent.confidence import ConfidenceClaim
from music_agent.confidence_derivation import (
    ConfidenceDerivationPolicy,
    derive_confidence_claim,
)
from music_agent.direct_track_preference import (
    DirectPreferenceMagnitudePolicy,
    resolve_direct_track_preference,
)
from music_agent.familiarity import (
    Familiarity,
    FamiliarityNormalizationPolicy,
    derive_track_familiarity,
)
from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_explainability import (
    ConflictKind,
    ContributingSignal,
    DerivedPreferenceExplanation,
    DerivationKind,
    PreferenceConflict,
)
from music_agent.preference_persistence import SignalIdentity, SignalHead
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_signal import (
    PreferenceSignal,
    RatingBandPolicy,
    SignalContribution,
    SignalDirection,
    normalize_preference_signal,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.temporal_evolution import (
    PreferenceEvidence,
    TemporalInterpretation,
    TemporalScopePolicy,
    classify_temporal_relation,
)


class PreferenceQueryError(ValueError):
    code = "preference_query_error"


# The three directional preference signals reconstructed through S1 + S3, keyed by their durable
# ``signal_path``. ``play_count`` is handled separately through S4 (familiarity).
_PREFERENCE_SIGNAL_PATHS: tuple[str, ...] = ("favorited", "disliked", "rating")
_PLAY_COUNT_PATH = "play_count"

_SIGNAL_PATH_TO_PREFERENCE_SIGNAL: dict[str, PreferenceSignal] = {
    "favorited": PreferenceSignal.FAVORITED,
    "disliked": PreferenceSignal.DISLIKED,
    "rating": PreferenceSignal.RATING,
}


@dataclass(frozen=True, slots=True)
class TrackPreferenceState:
    """The reconstructed current P06 state for one Track, derived from durable evidence."""

    target: PreferenceTargetReference
    direct_preference: DerivedPreference
    familiarity: Familiarity
    temporal_interpretation: TemporalInterpretation | None
    explanation: DerivedPreferenceExplanation
    confidence_claim: ConfidenceClaim | None = None


def query_track_preference(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    *,
    rating_policy: RatingBandPolicy,
    magnitude_policy: DirectPreferenceMagnitudePolicy,
    familiarity_policy: FamiliarityNormalizationPolicy,
    source_system: str = "apple_music",
    scope_policy: TemporalScopePolicy | None = None,
    now: datetime | None = None,
    confidence_policy: ConfidenceDerivationPolicy | None = None,
) -> TrackPreferenceState:
    """Reconstruct current Track-level P06 state from durable heads and revisions.

    ``target`` must reference a ``TRACK``. ``rating_policy``, ``magnitude_policy``, and
    ``familiarity_policy`` are the injected S1 / S3 / S4 calibration policies; this module owns no
    calibration constants. ``source_system`` selects the observation source the heads were written
    under. ``scope_policy`` and ``now`` are the injected S8 inputs; when either is omitted, no
    temporal interpretation is produced (it is ``None``), which is also the result when the durable
    evidence carries no parseable event times. ``confidence_policy`` optionally enables the S5
    read-side confidence derivation; omitted, the result carries ``confidence_claim=None``.
    """
    if not isinstance(repository, PreferencePersistenceRepository):
        raise PreferenceQueryError("repository must be a PreferencePersistenceRepository")
    _require_track_target(target)
    _require_rating_policy(rating_policy)
    _require_magnitude_policy(magnitude_policy)
    _require_familiarity_policy(familiarity_policy)
    _require_source_system(source_system)
    if scope_policy is not None and not isinstance(scope_policy, TemporalScopePolicy):
        raise PreferenceQueryError("scope_policy must be a TemporalScopePolicy or None")
    if now is not None:
        _require_aware_datetime(now)
    if confidence_policy is not None and not isinstance(
        confidence_policy, ConfidenceDerivationPolicy
    ):
        raise PreferenceQueryError(
            "confidence_policy must be a ConfidenceDerivationPolicy or None"
        )

    contributions: list[SignalContribution] = []
    for signal_path in _PREFERENCE_SIGNAL_PATHS:
        identity = SignalIdentity(target, source_system, signal_path)
        observed = _head_observed(repository.get_head(identity))
        contributions.append(
            normalize_preference_signal(
                _SIGNAL_PATH_TO_PREFERENCE_SIGNAL[signal_path], observed, rating_policy
            )
        )

    strength = resolve_direct_track_preference(contributions, magnitude_policy)
    direct_preference = DerivedPreference(target, strength)

    confidence_claim: ConfidenceClaim | None = None
    if confidence_policy is not None:
        confidence_claim = derive_confidence_claim(
            repository,
            target,
            strength,
            rating_policy=rating_policy,
            policy=confidence_policy,
            source_system=source_system,
            now=now,
        )

    play_count_identity = SignalIdentity(target, source_system, _PLAY_COUNT_PATH)
    familiarity = derive_track_familiarity(
        _head_observed(repository.get_head(play_count_identity)), familiarity_policy
    )

    temporal_interpretation = _derive_temporal_interpretation(
        repository, target, source_system, rating_policy, scope_policy, now
    )

    conflicts = _direct_conflicts(strength)
    explanation = DerivedPreferenceExplanation(
        target,
        strength,
        DerivationKind.DIRECT,
        direct_preference=direct_preference,
        confidence_claims=(
            (confidence_claim,) if confidence_claim is not None else ()
        ),
        temporal_interpretation=temporal_interpretation,
        conflicts=conflicts,
        signals=tuple(
            ContributingSignal(contribution, target) for contribution in contributions
        ),
    )

    return TrackPreferenceState(
        target, direct_preference, familiarity, temporal_interpretation, explanation,
        confidence_claim,
    )


def _head_observed(head: SignalHead | None) -> ObservedValue:
    """Reconstruct the three-state ``ObservedValue`` a head currently represents.

    ``current_semantic_value`` is the last ``VALUE`` ever observed and is preserved unchanged by
    later ``MISSING`` / ``NULL`` observations, so it takes precedence: a head that has ever seen a
    value reproduces that value. Only a head with no value falls back to its ``last_observed_state``
    (``NULL`` vs ``MISSING``), so the three states remain distinguishable and a ``MISSING`` / ``NULL``
    observation never erases a prior semantic value.
    """
    if head is None:
        return ObservedValue.missing()
    if head.current_semantic_value is not None:
        return ObservedValue.value(head.current_semantic_value)
    if head.last_observed_state is ObservationState.NULL:
        return ObservedValue.null()
    return ObservedValue.missing()


def _derive_temporal_interpretation(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    source_system: str,
    rating_policy: RatingBandPolicy,
    scope_policy: TemporalScopePolicy | None,
    now: datetime | None,
) -> TemporalInterpretation | None:
    """Classify the earliest vs latest opposite-direction evidence with a parseable event time."""
    if scope_policy is None or now is None:
        return None

    facts = _directional_facts(repository, target, source_system, rating_policy)
    if len(facts) < 2:
        return None
    first, last = facts[0], facts[-1]
    if first.direction is last.direction:
        return None
    return classify_temporal_relation(first, last, now=now, scope_policy=scope_policy)


def _directional_facts(
    repository: PreferencePersistenceRepository,
    target: PreferenceTargetReference,
    source_system: str,
    rating_policy: RatingBandPolicy,
) -> list[PreferenceEvidence]:
    """Collect directional evidence facts that carry a parseable ``event_at``, chronologically."""
    facts: list[tuple[datetime, datetime, PreferenceEvidence]] = []
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
            observed_at = _parse_aware_datetime(revision.observed_at)
            event_at = _parse_aware_datetime(revision.event_at)
            if observed_at is None or event_at is None:
                continue
            facts.append(
                (
                    event_at,
                    observed_at,
                    PreferenceEvidence(
                        target.target_id, contribution.direction, observed_at, event_at
                    ),
                )
            )
    facts.sort(key=lambda item: (item[0], item[1]))
    return [fact for _, _, fact in facts]


def _direct_conflicts(strength: PreferenceStrength) -> tuple[PreferenceConflict, ...]:
    """Surface the direct categorical conflict when S3 resolved a hard ``CONFLICT``."""
    if strength.state is PreferenceState.CONFLICT:
        return (
            PreferenceConflict(
                ConflictKind.DIRECT_CATEGORICAL,
                SignalDirection.POSITIVE,
                SignalDirection.NEGATIVE,
            ),
        )
    return ()


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


def _require_track_target(target: object) -> PreferenceTargetReference:
    if not isinstance(target, PreferenceTargetReference):
        raise PreferenceQueryError("target must be a PreferenceTargetReference")
    if target.kind is not PreferenceTargetKind.TRACK:
        raise PreferenceQueryError("target must reference a TRACK")
    return target


def _require_rating_policy(policy: object) -> None:
    if not isinstance(policy, RatingBandPolicy):
        raise PreferenceQueryError("rating_policy must be a RatingBandPolicy")


def _require_magnitude_policy(policy: object) -> None:
    if not isinstance(policy, DirectPreferenceMagnitudePolicy):
        raise PreferenceQueryError("magnitude_policy must be a DirectPreferenceMagnitudePolicy")


def _require_familiarity_policy(policy: object) -> None:
    if not isinstance(policy, FamiliarityNormalizationPolicy):
        raise PreferenceQueryError("familiarity_policy must be a FamiliarityNormalizationPolicy")


def _require_source_system(source_system: object) -> None:
    if not isinstance(source_system, str) or source_system == "":
        raise PreferenceQueryError("source_system must be a non-empty string")


def _require_aware_datetime(value: object) -> None:
    if not isinstance(value, datetime):
        raise PreferenceQueryError("now must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise PreferenceQueryError("now must be timezone-aware")
