"""Known-entity, read-only Apple Music refresh orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.apple_music import (
    add_library_relation_values,
    AppleMusicMappingError,
    AppleMusicSourceAdapter,
    materialize_library_track_relations,
    SourceReadStatus,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.merge import MergeValidationError, merge_observations
from music_agent.repository import CanonicalRepository


class RefreshStatus(StrEnum):
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    NO_SOURCE_BINDING = "no_source_binding"
    CANONICAL_NOT_FOUND = "canonical_not_found"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_LOOKUP_FAILED = "source_lookup_failed"
    MERGE_FAILED = "merge_failed"


class PreferenceRefreshError(RuntimeError):
    """P10: preference ingestion failed AFTER the canonical refresh committed.

    The canonical merge/save has already succeeded when this raises; the caller's
    per-track failure isolation records it and retries the whole track next cycle
    (ingestion is idempotent -- identical values confirm without new revisions).
    """

    code = "preference_refresh_error"


@dataclass(frozen=True, slots=True)
class RefreshResult:
    canonical_id: str
    source_system: str
    status: RefreshStatus
    changed_fields: tuple[str, ...] = ()
    source_status: SourceReadStatus | None = None
    error: str | None = None


def refresh_known_track(
    repository: CanonicalRepository,
    adapter: AppleMusicSourceAdapter,
    canonical_id: str,
    *,
    preference_repository=None,
) -> RefreshResult:
    model = repository.load_model()
    track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
    if track is None:
        return RefreshResult(canonical_id, "apple_music", RefreshStatus.CANONICAL_NOT_FOUND)

    persistent_id = track["external_ids"]["apple_music_persistent_id"]
    if persistent_id is None:
        return RefreshResult(canonical_id, "apple_music", RefreshStatus.NO_SOURCE_BINDING)
    key = ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
    if repository.lookup_external_identity(key) != canonical_id:
        return RefreshResult(canonical_id, "apple_music", RefreshStatus.NO_SOURCE_BINDING)

    read_result = adapter.read_track(persistent_id)
    if read_result.status is SourceReadStatus.CONFIRMED_NOT_FOUND:
        return RefreshResult(
            canonical_id,
            "apple_music",
            RefreshStatus.SOURCE_NOT_FOUND,
            source_status=read_result.status,
        )
    if read_result.status is SourceReadStatus.LOOKUP_FAILED:
        return RefreshResult(
            canonical_id,
            "apple_music",
            RefreshStatus.SOURCE_LOOKUP_FAILED,
            source_status=read_result.status,
            error=read_result.error,
        )

    try:
        observation = adapter.build_observation(canonical_id, read_result)
        relation_fields = materialize_library_track_relations(
            model, canonical_id, read_result.record.fields
        )
        observation = add_library_relation_values(observation, relation_fields)
        merge_result = merge_observations(model, [observation])
    except (AppleMusicMappingError, MergeValidationError) as error:
        return RefreshResult(
            canonical_id,
            "apple_music",
            RefreshStatus.MERGE_FAILED,
            source_status=read_result.status,
            error=str(error),
        )
    if not merge_result.changed_fields:
        if preference_repository is not None:
            _ingest_preference(preference_repository, observation)
        return RefreshResult(
            canonical_id,
            "apple_music",
            RefreshStatus.UNCHANGED,
            source_status=read_result.status,
        )
    repository.save_model(merge_result.model)
    if preference_repository is not None:
        _ingest_preference(preference_repository, observation)
    return RefreshResult(
        canonical_id,
        "apple_music",
        RefreshStatus.UPDATED,
        merge_result.changed_fields,
        read_result.status,
    )


def _ingest_preference(preference_repository, observation) -> None:
    """P10: feed the SAME observation that refreshed canonical state through the sealed
    P06 ingestion path. Runs only after the canonical save/merge succeeded; any failure
    raises :class:`PreferenceRefreshError` and never affects the already-committed
    canonical state. Identical re-observation confirms without new revisions (the sealed
    repository semantics), so repeated refresh never inflates evidence.
    """
    from music_agent.preference_ingestion import ingest_track_observation

    try:
        ingest_track_observation(preference_repository, observation)
    except Exception as error:
        raise PreferenceRefreshError(str(error)) from error
