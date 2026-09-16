"""Playlist / PlaylistMembership reconciliation evidence and identity contract.

This module classifies Playlist and membership-occurrence evidence and produces a typed
decision. It is a pure domain layer: it never touches SQLite, never mutates the canonical
model, never creates a Playlist or PlaylistMembership, and never generates a canonical ID.

It reuses the Artist/Album ``ReconciliationOutcome`` for Playlist identity decisions and the
caller-completed ``BoundExternalIdentityEvidence`` binding carrier. Membership occurrence
decisions use a dedicated ``PlaylistReconciliationOutcome`` whose positive state is
``IDENTIFIED`` (not ``RESOLVED``), because an occurrence identity is not a canonical
``PlaylistMembership`` identity and never produces a ``pm_`` resolution.

Authority boundary
------------------

Same as ``music_agent.reconciliation``: external identity binding authority lives in durable
persistence (``external_identity_bindings``). The evaluator receives an already-resolved
``BoundExternalIdentityEvidence`` for a found binding, or an ``unbound_identity`` key for a
negative lookup result. It never scans canonical ``external_ids`` projections to re-derive a
binding.

Membership occurrence identity
------------------------------

A PlaylistMembership has an independent ``pm_`` canonical identity. ``(playlist_id, track_id)``
is NOT a membership identity: the same Track may appear multiple times in the same Playlist.

``position`` and ``added_at`` are observations, not identity. A source that provides only an
ordered Track list has no per-occurrence identity, so no stable membership can be created or
matched. The evaluator fails closed to ``UNRESOLVED`` (``AMBIGUOUS_OCCURRENCE_IDENTITY``)
rather than fabricating identity from position or ``(playlist, track)``.

Canonical membership authority
------------------------------

A membership occurrence result whose outcome is ``IDENTIFIED`` is NOT a canonical
``PlaylistMembership`` resolution. Source occurrence identity is not a ``pm_`` canonical
identity: ``IDENTIFIED`` carries no ``pm_`` ID. It means only that the source occurrence is
distinguishable and its Playlist and Track references resolve to canonical entities.

Resolving an occurrence to an existing canonical ``PlaylistMembership`` requires, separately
and at a later stage, either:

1. a caller-completed durable external binding from occurrence identity to an existing
   ``pm_`` canonical ID, or
2. an explicit canonical membership decision whose target exists with the correct entity
   type.

Neither path is implemented in this slice. The occurrence evaluator must never treat a
non-empty/unique occurrence identity as authority to produce a canonical membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from music_agent.identity import (
    EntityType,
    ExternalIdentityKey,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.reconciliation import (
    BoundExternalIdentityEvidence,
    ReconciliationOutcome,
    ReconciliationValidationError,
)
from music_agent.source_observation import ObservedValue


class PlaylistReconciliationOutcome(StrEnum):
    """Membership-occurrence decision outcomes.

    ``IDENTIFIED`` means the source occurrence is distinguishable and its Playlist and Track
    references resolve to canonical entities. It is NOT a canonical ``PlaylistMembership``
    resolution: it carries no ``pm_`` canonical ID, and a non-empty/unique occurrence identity
    is never authority to produce a canonical membership. Resolution to a canonical membership
    is a separate, later stage (durable occurrence binding or explicit canonical decision),
    neither of which this slice implements.
    """

    IDENTIFIED = "identified"
    UNRESOLVED = "unresolved"
    CONFLICT = "conflict"


class PlaylistReconciliationReason(StrEnum):
    EXPLICIT_CANONICAL_DECISION = "explicit_canonical_decision"
    EXACT_EXTERNAL_IDENTITY = "exact_external_identity"
    UNKNOWN_EXTERNAL_IDENTITY = "unknown_external_identity"
    DISPLAY_NAME_INSUFFICIENT = "display_name_insufficient"
    MISSING_IDENTITY_EVIDENCE = "missing_identity_evidence"
    WRONG_ENTITY_TYPE = "wrong_entity_type"
    DANGLING_CANONICAL_REFERENCE = "dangling_canonical_reference"
    CONFLICTING_STRONG_EVIDENCE = "conflicting_strong_evidence"
    MISSING_TRACK_IDENTITY = "missing_track_identity"
    UNKNOWN_TRACK_IDENTITY = "unknown_track_identity"
    AMBIGUOUS_OCCURRENCE_IDENTITY = "ambiguous_occurrence_identity"
    DUPLICATE_OCCURRENCE_IDENTITY = "duplicate_occurrence_identity"


@dataclass(frozen=True, slots=True)
class PlaylistIdentityEvidence:
    """Evidence available to reconcile a Playlist's identity.

    ``bound_identity`` is strong identity evidence: a durable lookup that already resolved
    the key to a canonical ID. ``unbound_identity`` is the negative result of a durable
    lookup. ``explicit_canonical_id`` is explicit reconciliation authority. ``display_name``
    is weak evidence and never resolves an identity on its own.
    """

    bound_identity: BoundExternalIdentityEvidence | None = None
    unbound_identity: ExternalIdentityKey | None = None
    explicit_canonical_id: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        if self.bound_identity is not None and not isinstance(
            self.bound_identity, BoundExternalIdentityEvidence
        ):
            raise ReconciliationValidationError(
                "bound_identity must be a BoundExternalIdentityEvidence"
            )
        if self.unbound_identity is not None and not isinstance(
            self.unbound_identity, ExternalIdentityKey
        ):
            raise ReconciliationValidationError("unbound_identity must be an ExternalIdentityKey")
        if self.bound_identity is not None and self.unbound_identity is not None:
            raise ReconciliationValidationError(
                "bound_identity and unbound_identity are mutually exclusive"
            )
        if self.explicit_canonical_id is not None and (
            not isinstance(self.explicit_canonical_id, str) or self.explicit_canonical_id == ""
        ):
            raise ReconciliationValidationError(
                "explicit_canonical_id must be a non-empty string when supplied"
            )
        if self.display_name is not None and (
            not isinstance(self.display_name, str) or self.display_name == ""
        ):
            raise ReconciliationValidationError(
                "display_name must be a non-empty string when supplied"
            )


@dataclass(frozen=True, slots=True)
class MembershipOccurrenceIdentity:
    """A source-level, occurrence-specific identity for one membership occurrence.

    This is the only evidence that distinguishes two occurrences of the same Track within
    the same Playlist. It is a source occurrence discriminator, not a canonical entity
    binding: the current model has no ``external_ids`` projection for memberships.
    """

    source_system: str
    external_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise ReconciliationValidationError("source_system must be a non-empty string")
        if not isinstance(self.external_id, str) or self.external_id == "":
            raise ReconciliationValidationError("external_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MembershipOccurrenceEvidence:
    """Evidence available for one PlaylistMembership occurrence.

    ``playlist_evidence`` identifies the owning Playlist. ``track_bound`` / ``track_unbound``
    identify the member Track via a completed or negative durable lookup. ``occurrence_identity``
    is the only stable per-occurrence discriminator; when absent the occurrence is ambiguous.
    ``position`` and ``added_at`` are observations and never contribute to identity.
    """

    playlist_evidence: PlaylistIdentityEvidence
    track_bound: BoundExternalIdentityEvidence | None = None
    track_unbound: ExternalIdentityKey | None = None
    occurrence_identity: MembershipOccurrenceIdentity | None = None
    position: ObservedValue | None = None
    added_at: ObservedValue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.playlist_evidence, PlaylistIdentityEvidence):
            raise ReconciliationValidationError(
                "playlist_evidence must be a PlaylistIdentityEvidence"
            )
        if self.track_bound is not None and not isinstance(
            self.track_bound, BoundExternalIdentityEvidence
        ):
            raise ReconciliationValidationError(
                "track_bound must be a BoundExternalIdentityEvidence"
            )
        if self.track_unbound is not None and not isinstance(
            self.track_unbound, ExternalIdentityKey
        ):
            raise ReconciliationValidationError("track_unbound must be an ExternalIdentityKey")
        if self.track_bound is not None and self.track_unbound is not None:
            raise ReconciliationValidationError(
                "track_bound and track_unbound are mutually exclusive"
            )
        if self.occurrence_identity is not None and not isinstance(
            self.occurrence_identity, MembershipOccurrenceIdentity
        ):
            raise ReconciliationValidationError(
                "occurrence_identity must be a MembershipOccurrenceIdentity"
            )
        for name, value in (("position", self.position), ("added_at", self.added_at)):
            if value is not None and not isinstance(value, ObservedValue):
                raise ReconciliationValidationError(f"{name} must be an ObservedValue")


@dataclass(frozen=True, slots=True)
class PlaylistIdentityResult:
    outcome: ReconciliationOutcome
    reason: PlaylistReconciliationReason
    canonical_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ReconciliationOutcome):
            raise ReconciliationValidationError("outcome must be a ReconciliationOutcome")
        if not isinstance(self.reason, PlaylistReconciliationReason):
            raise ReconciliationValidationError("reason must be a PlaylistReconciliationReason")
        if self.outcome is ReconciliationOutcome.RESOLVED:
            if self.canonical_id is None:
                raise ReconciliationValidationError(
                    "a RESOLVED playlist result requires a canonical_id"
                )
        elif self.canonical_id is not None:
            raise ReconciliationValidationError(
                "only RESOLVED playlist results carry a canonical_id"
            )


@dataclass(frozen=True, slots=True)
class MembershipReconciliationResult:
    outcome: PlaylistReconciliationOutcome
    reason: PlaylistReconciliationReason
    playlist_canonical_id: str | None = None
    track_canonical_id: str | None = None
    occurrence_identity: MembershipOccurrenceIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, PlaylistReconciliationOutcome):
            raise ReconciliationValidationError("outcome must be a PlaylistReconciliationOutcome")
        if not isinstance(self.reason, PlaylistReconciliationReason):
            raise ReconciliationValidationError("reason must be a PlaylistReconciliationReason")
        if self.outcome is PlaylistReconciliationOutcome.IDENTIFIED:
            if self.playlist_canonical_id is None or self.track_canonical_id is None:
                raise ReconciliationValidationError(
                    "an IDENTIFIED membership result requires playlist and track canonical IDs"
                )
            if self.occurrence_identity is None:
                raise ReconciliationValidationError(
                    "an IDENTIFIED membership result requires an occurrence identity"
                )
        else:
            if (
                self.playlist_canonical_id is not None
                or self.track_canonical_id is not None
                or self.occurrence_identity is not None
            ):
                raise ReconciliationValidationError(
                    "only IDENTIFIED membership results carry resolution payload"
                )


def evaluate_playlist_identity(
    model: Mapping[str, Any], evidence: PlaylistIdentityEvidence
) -> PlaylistIdentityResult:
    """Classify Playlist identity evidence against the canonical model."""
    if not isinstance(evidence, PlaylistIdentityEvidence):
        raise ReconciliationValidationError("evidence must be a PlaylistIdentityEvidence")
    playlists = _require_collection(model, "playlists")
    known_ids = {playlist["id"] for playlist in playlists}

    explicit_id = evidence.explicit_canonical_id
    bound = evidence.bound_identity
    unbound = evidence.unbound_identity

    explicit_decision_id: str | None = None
    if explicit_id is not None:
        invalid = _invalid_explicit_playlist_reason(explicit_id, known_ids)
        if invalid is not None:
            return PlaylistIdentityResult(ReconciliationOutcome.CONFLICT, invalid)
        explicit_decision_id = explicit_id

    bound_resolved_id: str | None = None
    if bound is not None:
        invalid = _invalid_bound_identity_reason(bound, EntityType.PLAYLIST, known_ids)
        if invalid is not None:
            return PlaylistIdentityResult(ReconciliationOutcome.CONFLICT, invalid)
        bound_resolved_id = bound.canonical_id

    if explicit_decision_id is not None and bound_resolved_id is not None:
        if bound_resolved_id != explicit_decision_id:
            return PlaylistIdentityResult(
                ReconciliationOutcome.CONFLICT,
                PlaylistReconciliationReason.CONFLICTING_STRONG_EVIDENCE,
            )
        return PlaylistIdentityResult(
            ReconciliationOutcome.RESOLVED,
            PlaylistReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            explicit_decision_id,
        )
    if explicit_decision_id is not None:
        return PlaylistIdentityResult(
            ReconciliationOutcome.RESOLVED,
            PlaylistReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            explicit_decision_id,
        )
    if bound_resolved_id is not None:
        return PlaylistIdentityResult(
            ReconciliationOutcome.RESOLVED,
            PlaylistReconciliationReason.EXACT_EXTERNAL_IDENTITY,
            bound_resolved_id,
        )
    if unbound is not None:
        if unbound.entity_type is not EntityType.PLAYLIST:
            return PlaylistIdentityResult(
                ReconciliationOutcome.CONFLICT, PlaylistReconciliationReason.WRONG_ENTITY_TYPE
            )
        return PlaylistIdentityResult(
            ReconciliationOutcome.UNRESOLVED,
            PlaylistReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY,
        )
    if evidence.display_name is not None:
        return PlaylistIdentityResult(
            ReconciliationOutcome.UNRESOLVED,
            PlaylistReconciliationReason.DISPLAY_NAME_INSUFFICIENT,
        )
    return PlaylistIdentityResult(
        ReconciliationOutcome.UNRESOLVED,
        PlaylistReconciliationReason.MISSING_IDENTITY_EVIDENCE,
    )


def evaluate_membership_occurrence(
    model: Mapping[str, Any], evidence: MembershipOccurrenceEvidence
) -> MembershipReconciliationResult:
    """Classify one membership occurrence, failing closed without occurrence identity.

    A positive result is ``IDENTIFIED``, meaning the source occurrence is distinguishable and
    its Playlist and Track references resolve to canonical entities. It is NOT a canonical
    ``PlaylistMembership`` resolution: it carries no ``pm_`` ID.
    """
    if not isinstance(evidence, MembershipOccurrenceEvidence):
        raise ReconciliationValidationError("evidence must be a MembershipOccurrenceEvidence")
    return _evaluate_occurrence(model, evidence)


def evaluate_membership_occurrences(
    model: Mapping[str, Any], occurrences: Sequence[MembershipOccurrenceEvidence]
) -> tuple[MembershipReconciliationResult, ...]:
    """Classify a set of occurrences and fail closed on duplicate occurrence identity.

    Each occurrence is evaluated independently, then any occurrence whose
    ``MembershipOccurrenceIdentity`` collides with another occurrence in the same set is
    forced to ``CONFLICT`` / ``DUPLICATE_OCCURRENCE_IDENTITY``. Two distinct occurrence
    identities pointing at the same ``(playlist, track)`` are identified as two distinct
    *source occurrences*; they are not asserted to be two distinct canonical memberships.
    """
    if isinstance(occurrences, (str, bytes)) or not isinstance(occurrences, Sequence):
        raise ReconciliationValidationError("occurrences must be a sequence")
    items = list(occurrences)
    if any(not isinstance(item, MembershipOccurrenceEvidence) for item in items):
        raise ReconciliationValidationError(
            "occurrences must contain MembershipOccurrenceEvidence values"
        )

    seen: dict[tuple[str, str], int] = {}
    for item in items:
        identity = item.occurrence_identity
        if identity is None:
            continue
        key = (identity.source_system, identity.external_id)
        seen[key] = seen.get(key, 0) + 1

    results: list[MembershipReconciliationResult] = []
    for item in items:
        identity = item.occurrence_identity
        if identity is not None and seen[(identity.source_system, identity.external_id)] > 1:
            results.append(MembershipReconciliationResult(
                PlaylistReconciliationOutcome.CONFLICT,
                PlaylistReconciliationReason.DUPLICATE_OCCURRENCE_IDENTITY,
            ))
            continue
        results.append(_evaluate_occurrence(model, item))
    return tuple(results)


def _evaluate_occurrence(
    model: Mapping[str, Any], evidence: MembershipOccurrenceEvidence
) -> MembershipReconciliationResult:
    playlist_result = evaluate_playlist_identity(model, evidence.playlist_evidence)
    if playlist_result.outcome is not ReconciliationOutcome.RESOLVED:
        outcome = (
            PlaylistReconciliationOutcome.CONFLICT
            if playlist_result.outcome is ReconciliationOutcome.CONFLICT
            else PlaylistReconciliationOutcome.UNRESOLVED
        )
        return MembershipReconciliationResult(outcome, playlist_result.reason)

    track_resolved_id = _resolve_track(model, evidence)
    if isinstance(track_resolved_id, MembershipReconciliationResult):
        return track_resolved_id

    if evidence.occurrence_identity is None:
        return MembershipReconciliationResult(
            PlaylistReconciliationOutcome.UNRESOLVED,
            PlaylistReconciliationReason.AMBIGUOUS_OCCURRENCE_IDENTITY,
        )

    return MembershipReconciliationResult(
        PlaylistReconciliationOutcome.IDENTIFIED,
        PlaylistReconciliationReason.EXACT_EXTERNAL_IDENTITY,
        playlist_canonical_id=playlist_result.canonical_id,
        track_canonical_id=track_resolved_id,
        occurrence_identity=evidence.occurrence_identity,
    )


def _resolve_track(
    model: Mapping[str, Any], evidence: MembershipOccurrenceEvidence
) -> str | MembershipReconciliationResult:
    bound = evidence.track_bound
    unbound = evidence.track_unbound
    if bound is not None:
        invalid = _invalid_bound_identity_reason(
            bound, EntityType.TRACK, {track["id"] for track in _require_collection(model, "tracks")}
        )
        if invalid is not None:
            return MembershipReconciliationResult(
                PlaylistReconciliationOutcome.CONFLICT, invalid
            )
        return bound.canonical_id
    if unbound is not None:
        if unbound.entity_type is not EntityType.TRACK:
            return MembershipReconciliationResult(
                PlaylistReconciliationOutcome.CONFLICT,
                PlaylistReconciliationReason.WRONG_ENTITY_TYPE,
            )
        return MembershipReconciliationResult(
            PlaylistReconciliationOutcome.UNRESOLVED,
            PlaylistReconciliationReason.UNKNOWN_TRACK_IDENTITY,
        )
    return MembershipReconciliationResult(
        PlaylistReconciliationOutcome.UNRESOLVED,
        PlaylistReconciliationReason.MISSING_TRACK_IDENTITY,
    )


def _invalid_bound_identity_reason(
    bound: BoundExternalIdentityEvidence, entity_type: EntityType, known_ids: set[str]
) -> PlaylistReconciliationReason | None:
    if bound.external_identity.entity_type is not entity_type:
        return PlaylistReconciliationReason.WRONG_ENTITY_TYPE
    try:
        validate_canonical_id(entity_type, bound.canonical_id)
    except IdentityValidationError:
        return PlaylistReconciliationReason.WRONG_ENTITY_TYPE
    if bound.canonical_id not in known_ids:
        return PlaylistReconciliationReason.DANGLING_CANONICAL_REFERENCE
    return None


def _invalid_explicit_playlist_reason(
    playlist_id: str, known_ids: set[str]
) -> PlaylistReconciliationReason | None:
    try:
        validate_canonical_id(EntityType.PLAYLIST, playlist_id)
    except IdentityValidationError:
        return PlaylistReconciliationReason.WRONG_ENTITY_TYPE
    if playlist_id not in known_ids:
        return PlaylistReconciliationReason.DANGLING_CANONICAL_REFERENCE
    return None


def _require_collection(model: Mapping[str, Any], name: str) -> Any:
    if not isinstance(model, Mapping):
        raise ReconciliationValidationError("model must be a mapping")
    collection = model.get(name)
    if not isinstance(collection, (list, tuple)):
        raise ReconciliationValidationError(f"model must contain a {name} collection")
    return collection
