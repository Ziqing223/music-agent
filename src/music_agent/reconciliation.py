"""Reconciliation evidence and decision contract for Artist and Album relations.

This module classifies source evidence and produces a typed decision. It is a pure
domain layer: it never touches SQLite, never mutates a Candidate or the canonical model,
never creates a canonical entity, and never generates a canonical ID. Decisions reuse the
existing ``ArtistRelationResolution`` and ``AlbumRelationResolution`` types rather than
introducing a parallel relation-state model.

Authority boundary
------------------

External identity binding authority lives in durable persistence
(``external_identity_bindings``). The canonical model's ``external_ids.apple_music_persistent_id``
is only a projection reconstructed from that binding and is *not* an independent source of
binding truth. This evaluator therefore never scans canonical ``external_ids`` projections to
derive which canonical ID an external key resolves to.

The persistence/caller boundary performs the durable lookup and hands the evaluator an already
resolved result:

- ``BoundExternalIdentityEvidence(external_identity, canonical_id)`` when the lookup found a
  binding, or
- an ``unbound_identity`` key when the lookup found no binding.

The evaluator only validates entity type and canonical-target existence, then decides
resolve / unresolved / conflict. Display names never resolve a relation; they are only ever
insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from music_agent.identity import (
    EntityType,
    ExternalIdentityKey,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    ArtistRelationResolution,
)


class ReconciliationValidationError(ValueError):
    code = "validation_error"


class ReconciliationOutcome(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    CONFLICT = "conflict"


class ReconciliationReason(StrEnum):
    EXACT_EXTERNAL_IDENTITY = "exact_external_identity"
    EXPLICIT_CANONICAL_DECISION = "explicit_canonical_decision"
    EXPLICIT_ABSENCE = "explicit_absence"
    DISPLAY_NAME_INSUFFICIENT = "display_name_insufficient"
    MISSING_RELATION_EVIDENCE = "missing_relation_evidence"
    UNKNOWN_EXTERNAL_IDENTITY = "unknown_external_identity"
    WRONG_ENTITY_TYPE = "wrong_entity_type"
    DANGLING_CANONICAL_REFERENCE = "dangling_canonical_reference"
    DUPLICATE_CANONICAL_ID = "duplicate_canonical_id"
    CONFLICTING_STRONG_EVIDENCE = "conflicting_strong_evidence"


@dataclass(frozen=True, slots=True)
class BoundExternalIdentityEvidence:
    """A completed durable external identity lookup: key resolves to ``canonical_id``.

    The lookup itself happens at the persistence boundary against
    ``external_identity_bindings``. This value carries the result; the evaluator validates
    entity type and target existence but never re-derives the binding from canonical
    ``external_ids`` projections.
    """

    external_identity: ExternalIdentityKey
    canonical_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.external_identity, ExternalIdentityKey):
            raise ReconciliationValidationError("external_identity must be an ExternalIdentityKey")
        if not isinstance(self.canonical_id, str) or self.canonical_id == "":
            raise ReconciliationValidationError("canonical_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ArtistEvidence:
    """Evidence available to reconcile a Track's artist relation.

    ``bound_identity`` is strong identity evidence: a durable lookup that already resolved
    the key to a canonical ID. ``unbound_identity`` is the negative result of a durable
    lookup: the key exists in the source but has no binding. ``explicit_canonical_ids`` is
    explicit reconciliation authority: the caller asserts specific canonical Artist IDs.
    ``display_name`` is weak evidence and never resolves a relation on its own.
    """

    bound_identity: BoundExternalIdentityEvidence | None = None
    unbound_identity: ExternalIdentityKey | None = None
    explicit_canonical_ids: tuple[str, ...] | None = None
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
        if self.explicit_canonical_ids is not None:
            if not isinstance(self.explicit_canonical_ids, (tuple, list)):
                raise ReconciliationValidationError(
                    "explicit_canonical_ids must be a sequence of strings"
                )
            ids = tuple(self.explicit_canonical_ids)
            if any(not isinstance(candidate_id, str) for candidate_id in ids):
                raise ReconciliationValidationError(
                    "explicit_canonical_ids must contain only strings"
                )
            object.__setattr__(self, "explicit_canonical_ids", ids)
        if self.display_name is not None and (
            not isinstance(self.display_name, str) or self.display_name == ""
        ):
            raise ReconciliationValidationError(
                "display_name must be a non-empty string when supplied"
            )


@dataclass(frozen=True, slots=True)
class AlbumEvidence:
    """Evidence available to reconcile a Track's album relation.

    ``explicit_absence`` is the only path to ``resolved_absent``. It must be an explicit,
    semantically reliable absence signal; missing or null source values are not absence
    evidence and do not set this flag.
    """

    bound_identity: BoundExternalIdentityEvidence | None = None
    unbound_identity: ExternalIdentityKey | None = None
    explicit_canonical_id: str | None = None
    explicit_absence: bool = False
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
        if not isinstance(self.explicit_absence, bool):
            raise ReconciliationValidationError("explicit_absence must be a boolean")
        if self.display_name is not None and (
            not isinstance(self.display_name, str) or self.display_name == ""
        ):
            raise ReconciliationValidationError(
                "display_name must be a non-empty string when supplied"
            )


@dataclass(frozen=True, slots=True)
class ArtistReconciliationResult:
    outcome: ReconciliationOutcome
    reason: ReconciliationReason
    resolution: ArtistRelationResolution | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ReconciliationOutcome):
            raise ReconciliationValidationError("outcome must be a ReconciliationOutcome")
        if not isinstance(self.reason, ReconciliationReason):
            raise ReconciliationValidationError("reason must be a ReconciliationReason")
        if self.outcome is ReconciliationOutcome.RESOLVED:
            if not isinstance(self.resolution, ArtistRelationResolution):
                raise ReconciliationValidationError(
                    "a RESOLVED artist outcome requires a resolution"
                )
        elif self.resolution is not None:
            raise ReconciliationValidationError(
                "only RESOLVED artist outcomes carry a resolution"
            )


@dataclass(frozen=True, slots=True)
class AlbumReconciliationResult:
    outcome: ReconciliationOutcome
    reason: ReconciliationReason
    resolution: AlbumRelationResolution | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ReconciliationOutcome):
            raise ReconciliationValidationError("outcome must be a ReconciliationOutcome")
        if not isinstance(self.reason, ReconciliationReason):
            raise ReconciliationValidationError("reason must be a ReconciliationReason")
        if self.outcome is ReconciliationOutcome.RESOLVED:
            if not isinstance(self.resolution, AlbumRelationResolution):
                raise ReconciliationValidationError(
                    "a RESOLVED album outcome requires a resolution"
                )
        elif self.resolution is not None:
            raise ReconciliationValidationError(
                "only RESOLVED album outcomes carry a resolution"
            )


def evaluate_artist_reconciliation(
    model: Mapping[str, Any], evidence: ArtistEvidence
) -> ArtistReconciliationResult:
    """Classify Artist evidence against the canonical model and produce a decision."""
    if not isinstance(evidence, ArtistEvidence):
        raise ReconciliationValidationError("evidence must be an ArtistEvidence")
    artists = _require_collection(model, "artists")
    known_ids = {artist["id"] for artist in artists}

    explicit_ids = evidence.explicit_canonical_ids
    bound = evidence.bound_identity
    unbound = evidence.unbound_identity

    explicit_decision_ids: tuple[str, ...] | None = None
    if explicit_ids is not None:
        invalid = _invalid_explicit_artist_reason(explicit_ids, known_ids)
        if invalid is not None:
            return ArtistReconciliationResult(ReconciliationOutcome.CONFLICT, invalid)
        explicit_decision_ids = explicit_ids

    bound_resolved_id: str | None = None
    if bound is not None:
        invalid = _invalid_bound_identity_reason(bound, EntityType.ARTIST, known_ids)
        if invalid is not None:
            return ArtistReconciliationResult(ReconciliationOutcome.CONFLICT, invalid)
        bound_resolved_id = bound.canonical_id

    if explicit_decision_ids is not None and bound_resolved_id is not None:
        if bound_resolved_id not in explicit_decision_ids:
            return ArtistReconciliationResult(
                ReconciliationOutcome.CONFLICT,
                ReconciliationReason.CONFLICTING_STRONG_EVIDENCE,
            )
        return ArtistReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            ArtistRelationResolution.resolved_to_artists(explicit_decision_ids),
        )
    if explicit_decision_ids is not None:
        return ArtistReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            ArtistRelationResolution.resolved_to_artists(explicit_decision_ids),
        )
    if bound_resolved_id is not None:
        return ArtistReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXACT_EXTERNAL_IDENTITY,
            ArtistRelationResolution.resolved_to_artists((bound_resolved_id,)),
        )
    if unbound is not None:
        if unbound.entity_type is not EntityType.ARTIST:
            return ArtistReconciliationResult(
                ReconciliationOutcome.CONFLICT, ReconciliationReason.WRONG_ENTITY_TYPE
            )
        return ArtistReconciliationResult(
            ReconciliationOutcome.UNRESOLVED, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY
        )
    if evidence.display_name is not None:
        return ArtistReconciliationResult(
            ReconciliationOutcome.UNRESOLVED, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT
        )
    return ArtistReconciliationResult(
        ReconciliationOutcome.UNRESOLVED, ReconciliationReason.MISSING_RELATION_EVIDENCE
    )


def evaluate_album_reconciliation(
    model: Mapping[str, Any], evidence: AlbumEvidence
) -> AlbumReconciliationResult:
    """Classify Album evidence against the canonical model and produce a decision."""
    if not isinstance(evidence, AlbumEvidence):
        raise ReconciliationValidationError("evidence must be an AlbumEvidence")
    albums = _require_collection(model, "albums")
    known_ids = {album["id"] for album in albums}

    explicit_id = evidence.explicit_canonical_id
    bound = evidence.bound_identity
    unbound = evidence.unbound_identity
    absence = evidence.explicit_absence

    explicit_decision_id: str | None = None
    if explicit_id is not None:
        invalid = _invalid_explicit_album_reason(explicit_id, known_ids)
        if invalid is not None:
            return AlbumReconciliationResult(ReconciliationOutcome.CONFLICT, invalid)
        explicit_decision_id = explicit_id

    bound_resolved_id: str | None = None
    if bound is not None:
        invalid = _invalid_bound_identity_reason(bound, EntityType.ALBUM, known_ids)
        if invalid is not None:
            return AlbumReconciliationResult(ReconciliationOutcome.CONFLICT, invalid)
        bound_resolved_id = bound.canonical_id

    if absence and (explicit_decision_id is not None or bound_resolved_id is not None):
        return AlbumReconciliationResult(
            ReconciliationOutcome.CONFLICT,
            ReconciliationReason.CONFLICTING_STRONG_EVIDENCE,
        )

    if explicit_decision_id is not None and bound_resolved_id is not None:
        if bound_resolved_id != explicit_decision_id:
            return AlbumReconciliationResult(
                ReconciliationOutcome.CONFLICT,
                ReconciliationReason.CONFLICTING_STRONG_EVIDENCE,
            )
        return AlbumReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            AlbumRelationResolution.resolved_to_album(explicit_decision_id),
        )
    if explicit_decision_id is not None:
        return AlbumReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXPLICIT_CANONICAL_DECISION,
            AlbumRelationResolution.resolved_to_album(explicit_decision_id),
        )
    if bound_resolved_id is not None:
        return AlbumReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXACT_EXTERNAL_IDENTITY,
            AlbumRelationResolution.resolved_to_album(bound_resolved_id),
        )
    if absence:
        return AlbumReconciliationResult(
            ReconciliationOutcome.RESOLVED,
            ReconciliationReason.EXPLICIT_ABSENCE,
            AlbumRelationResolution.resolved_absent(),
        )
    if unbound is not None:
        if unbound.entity_type is not EntityType.ALBUM:
            return AlbumReconciliationResult(
                ReconciliationOutcome.CONFLICT, ReconciliationReason.WRONG_ENTITY_TYPE
            )
        return AlbumReconciliationResult(
            ReconciliationOutcome.UNRESOLVED, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY
        )
    if evidence.display_name is not None:
        return AlbumReconciliationResult(
            ReconciliationOutcome.UNRESOLVED, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT
        )
    return AlbumReconciliationResult(
        ReconciliationOutcome.UNRESOLVED, ReconciliationReason.MISSING_RELATION_EVIDENCE
    )


def _invalid_bound_identity_reason(
    bound: BoundExternalIdentityEvidence, entity_type: EntityType, known_ids: set[str]
) -> ReconciliationReason | None:
    if bound.external_identity.entity_type is not entity_type:
        return ReconciliationReason.WRONG_ENTITY_TYPE
    try:
        validate_canonical_id(entity_type, bound.canonical_id)
    except IdentityValidationError:
        return ReconciliationReason.WRONG_ENTITY_TYPE
    if bound.canonical_id not in known_ids:
        return ReconciliationReason.DANGLING_CANONICAL_REFERENCE
    return None


def _invalid_explicit_artist_reason(
    ids: tuple[str, ...], known_ids: set[str]
) -> ReconciliationReason | None:
    for artist_id in ids:
        try:
            validate_canonical_id(EntityType.ARTIST, artist_id)
        except IdentityValidationError:
            return ReconciliationReason.WRONG_ENTITY_TYPE
    if len(set(ids)) != len(ids):
        return ReconciliationReason.DUPLICATE_CANONICAL_ID
    if not ids:
        return ReconciliationReason.DANGLING_CANONICAL_REFERENCE
    if any(artist_id not in known_ids for artist_id in ids):
        return ReconciliationReason.DANGLING_CANONICAL_REFERENCE
    return None


def _invalid_explicit_album_reason(
    album_id: str, known_ids: set[str]
) -> ReconciliationReason | None:
    try:
        validate_canonical_id(EntityType.ALBUM, album_id)
    except IdentityValidationError:
        return ReconciliationReason.WRONG_ENTITY_TYPE
    if album_id not in known_ids:
        return ReconciliationReason.DANGLING_CANONICAL_REFERENCE
    return None


def _require_collection(model: Mapping[str, Any], name: str) -> Any:
    if not isinstance(model, Mapping):
        raise ReconciliationValidationError("model must be a mapping")
    collection = model.get(name)
    if not isinstance(collection, (list, tuple)):
        raise ReconciliationValidationError(f"model must contain a {name} collection")
    return collection
