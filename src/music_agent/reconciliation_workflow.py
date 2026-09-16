"""Explicit reconciliation workflow: validated decision -> durable Candidate relation state.

This is the command boundary for explicit Artist/Album reconciliation. The caller supplies
explicit canonical decisions (never source-derived name/identity evidence); the workflow
validates them through the P03.9.1 evidence evaluator and persists them through the existing
``CandidateStagingRepository.update_relation_resolution(...)`` boundary. It never re-derives
external identity bindings, never performs name matching, and never creates canonical entities.

Authority boundary
------------------

- P03.9.1 ``reconciliation`` evaluator = decision validation (pure domain).
- ``update_relation_resolution(...)`` = durable relation-state mutation (single transaction).

The workflow composes them: evaluate every requested decision first, then mutate once. A
requested decision that is not ``RESOLVED`` aborts the whole call before any mutation, so a
partial Artist-only or Album-only write is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.reconciliation import (
    AlbumEvidence,
    AlbumReconciliationResult,
    ArtistEvidence,
    ArtistReconciliationResult,
    ReconciliationOutcome,
    evaluate_album_reconciliation,
    evaluate_artist_reconciliation,
)
from music_agent.repository import CanonicalRepository


class ReconciliationWorkflowError(ValueError):
    code = "reconciliation_workflow_error"


class ReconciliationWorkflowStatus(StrEnum):
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    BLOCKED = "blocked"
    CANDIDATE_NOT_FOUND = "candidate_not_found"


@dataclass(frozen=True, slots=True)
class ReconciliationWorkflowResult:
    status: ReconciliationWorkflowStatus
    candidate_identity: ExternalIdentityKey
    artist_result: ArtistReconciliationResult | None = None
    album_result: AlbumReconciliationResult | None = None


def reconcile_staged_track_relations(
    database_path: str | Path,
    candidate_identity: ExternalIdentityKey,
    *,
    artist_canonical_ids: tuple[str, ...] | list[str] | None = None,
    album_canonical_id: str | None = None,
    album_absent: bool = False,
) -> ReconciliationWorkflowResult:
    """Validate and persist explicit Artist/Album decisions for one staged Track Candidate.

    ``artist_canonical_ids`` is an explicit Artist decision (one or more canonical Artist IDs).
    ``album_canonical_id`` is an explicit resolve-to-Album decision; ``album_absent`` is an
    explicit Album-absence decision. Supplying both album inputs is a conflicting request and
    fails closed. A decision channel left as its default (``None`` / ``False``) is not requested
    and leaves that relation untouched.

    Every requested decision must evaluate to ``RESOLVED`` before any durable mutation occurs.
    """
    candidate_identity = _require_track_key(candidate_identity)
    artist_requested = artist_canonical_ids is not None
    album_requested = album_canonical_id is not None or album_absent
    if not artist_requested and not album_requested:
        raise ReconciliationWorkflowError("at least one explicit relation decision is required")

    with CandidateStagingRepository(database_path) as staging:
        existing = staging.get_candidate(candidate_identity)
    if existing is None:
        return ReconciliationWorkflowResult(
            ReconciliationWorkflowStatus.CANDIDATE_NOT_FOUND, candidate_identity
        )

    with CanonicalRepository(database_path) as repository:
        model = repository.load_model()

    artist_result = None
    if artist_requested:
        artist_result = evaluate_artist_reconciliation(
            model, ArtistEvidence(explicit_canonical_ids=artist_canonical_ids)
        )
    album_result = None
    if album_requested:
        album_result = evaluate_album_reconciliation(
            model,
            AlbumEvidence(
                explicit_canonical_id=album_canonical_id,
                explicit_absence=album_absent,
            ),
        )

    if (artist_result is not None and artist_result.outcome is not ReconciliationOutcome.RESOLVED) or (
        album_result is not None and album_result.outcome is not ReconciliationOutcome.RESOLVED
    ):
        return ReconciliationWorkflowResult(
            ReconciliationWorkflowStatus.BLOCKED, candidate_identity, artist_result, album_result
        )

    artist_resolution = artist_result.resolution if artist_result is not None else None
    album_resolution = album_result.resolution if album_result is not None else None

    changed = (
        (artist_resolution is not None and artist_resolution != existing.artist_relation)
        or (album_resolution is not None and album_resolution != existing.album_relation)
    )
    if not changed:
        return ReconciliationWorkflowResult(
            ReconciliationWorkflowStatus.UNCHANGED, candidate_identity, artist_result, album_result
        )

    with CandidateStagingRepository(database_path) as staging:
        staging.update_relation_resolution(
            candidate_identity,
            artist_resolution=artist_resolution,
            album_resolution=album_resolution,
        )

    return ReconciliationWorkflowResult(
        ReconciliationWorkflowStatus.UPDATED, candidate_identity, artist_result, album_result
    )


def _require_track_key(candidate_identity: object) -> ExternalIdentityKey:
    if not isinstance(candidate_identity, ExternalIdentityKey):
        raise ReconciliationWorkflowError("candidate_identity must be an ExternalIdentityKey")
    if candidate_identity.entity_type is not EntityType.TRACK:
        raise ReconciliationWorkflowError(
            "reconciliation workflow supports Track candidates only"
        )
    return candidate_identity
