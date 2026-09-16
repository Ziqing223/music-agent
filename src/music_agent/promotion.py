"""Atomic promotion orchestration for durable Track ingestion candidates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.identity import EntityType, ExternalIdentityKey, generate_canonical_id
from music_agent.ingestion_candidate import (
    PromotionBlocker,
    build_promotable_track,
    evaluate_track_promotion,
)
from music_agent.repository import CanonicalRepository


class PromotionStatus(StrEnum):
    PROMOTED = "promoted"
    BLOCKED = "blocked"
    ALREADY_BOUND = "already_bound"


class StagedCandidateNotFoundError(ValueError):
    code = "staged_candidate_not_found"


@dataclass(frozen=True, slots=True)
class PromotionResult:
    status: PromotionStatus
    external_identity: ExternalIdentityKey
    canonical_id: str | None = None
    blockers: tuple[PromotionBlocker, ...] = ()
    transferred_scopes: tuple[str, ...] = ()


def promote_staged_track(
    database_path: str | Path,
    key: ExternalIdentityKey,
) -> PromotionResult:
    """Promote one staged Apple Music Track Candidate, or return a typed no-op."""
    if not isinstance(key, ExternalIdentityKey):
        raise TypeError("key must be an ExternalIdentityKey")

    with CanonicalRepository(database_path) as repository:
        existing = repository.lookup_external_identity(key)
        if existing is not None:
            return PromotionResult(PromotionStatus.ALREADY_BOUND, key, existing)

        with CandidateStagingRepository(database_path) as staging:
            candidate = staging.get_candidate(key)
        if candidate is None:
            raise StagedCandidateNotFoundError("staged Candidate does not exist")

        canonical_model = repository.load_model()
        evaluation = evaluate_track_promotion(candidate, canonical_model)
        if not evaluation.is_promotable:
            return PromotionResult(
                PromotionStatus.BLOCKED,
                key,
                blockers=evaluation.blockers,
            )

        canonical_id = generate_canonical_id(EntityType.TRACK)
        track = build_promotable_track(candidate, canonical_id, canonical_model)
        promoted_model = deepcopy(canonical_model)
        promoted_model["tracks"].append(track)
        transferred_scopes = repository._commit_staged_track_promotion(
            promoted_model,
            key,
            canonical_id,
        )
        return PromotionResult(
            PromotionStatus.PROMOTED,
            key,
            canonical_id,
            transferred_scopes=transferred_scopes,
        )
