"""Synthetic complete-snapshot evaluation and safe source-presence persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.merge import MergeValidationError, merge_observations
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import ObservedValue, SourceObservation, SourcePresence


class SnapshotValidationError(ValueError):
    code = "validation_error"


class SnapshotCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceSnapshotScope:
    source_system: str
    entity_type: EntityType
    scope_key: str
    completeness: SnapshotCompleteness
    deletion_authoritative: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise SnapshotValidationError("source_system must be a non-empty string")
        if not isinstance(self.entity_type, EntityType):
            raise SnapshotValidationError("entity_type must be an EntityType")
        if not isinstance(self.scope_key, str) or self.scope_key == "":
            raise SnapshotValidationError("scope_key must be a non-empty string")
        if not isinstance(self.completeness, SnapshotCompleteness):
            raise SnapshotValidationError("completeness must be a SnapshotCompleteness")
        if not isinstance(self.deletion_authoritative, bool):
            raise SnapshotValidationError("deletion_authoritative must be a boolean")


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    external_identity: ExternalIdentityKey
    fields: Mapping[str, ObservedValue]

    def __post_init__(self) -> None:
        if not isinstance(self.external_identity, ExternalIdentityKey):
            raise SnapshotValidationError("external_identity must be an ExternalIdentityKey")
        if not isinstance(self.fields, Mapping):
            raise SnapshotValidationError("fields must be a mapping")
        copied: dict[str, ObservedValue] = {}
        for path, value in self.fields.items():
            if not isinstance(path, str) or path == "":
                raise SnapshotValidationError("field paths must be non-empty strings")
            if not isinstance(value, ObservedValue):
                raise SnapshotValidationError(f"{path} must contain an ObservedValue")
            copied[path] = value
        object.__setattr__(self, "fields", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    scope: SourceSnapshotScope
    records: tuple[SnapshotRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, SourceSnapshotScope):
            raise SnapshotValidationError("scope must be a SourceSnapshotScope")
        records = tuple(self.records)
        seen: set[ExternalIdentityKey] = set()
        for record in records:
            if not isinstance(record, SnapshotRecord):
                raise SnapshotValidationError("records must contain SnapshotRecord values")
            key = record.external_identity
            if key.source_system != self.scope.source_system or key.entity_type is not self.scope.entity_type:
                raise SnapshotValidationError("snapshot record identity is outside the declared scope")
            if key in seen:
                raise SnapshotValidationError(f"duplicate source identity: {key.external_id}")
            seen.add(key)
        object.__setattr__(self, "records", records)


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    resulting_model: dict[str, Any]
    changed_fields: tuple[str, ...]
    presence_outcomes: dict[str, SourcePresence]
    durable_presence_updates: tuple[SourcePresenceRecord, ...]
    confirmed_deleted_canonical_ids: tuple[str, ...]
    unresolved_source_records: tuple[ExternalIdentityKey, ...]


def apply_snapshot(repository: CanonicalRepository, snapshot: SourceSnapshot) -> SnapshotResult:
    scope = snapshot.scope
    if scope.source_system != "apple_music":
        raise SnapshotValidationError(f"unsupported source system: {scope.source_system}")
    if scope.entity_type is not EntityType.TRACK:
        raise SnapshotValidationError("P03.7 supports only Track snapshots")

    current_model = repository.load_model()
    bindings = dict(
        repository.list_external_identity_bindings(scope.source_system, scope.entity_type)
    )
    tracked_scope_members = {
        record.canonical_id
        for record in repository.list_source_presence(
            scope.source_system, scope.entity_type, scope.scope_key
        )
    }
    observed_canonical_ids: set[str] = set()
    observations: list[SourceObservation] = []
    unresolved: list[ExternalIdentityKey] = []
    presence_outcomes: dict[str, SourcePresence] = {}
    durable_updates: list[SourcePresenceRecord] = []

    for record in snapshot.records:
        key = record.external_identity
        canonical_id = bindings.get(key)
        if canonical_id is None:
            unresolved.append(key)
            continue
        observed_canonical_ids.add(canonical_id)
        observations.append(SourceObservation(
            scope.entity_type,
            canonical_id,
            scope.source_system,
            record.fields,
            SourcePresence.PRESENT,
        ))
        presence_outcomes[canonical_id] = SourcePresence.PRESENT
        durable_updates.append(SourcePresenceRecord(
            scope.source_system,
            scope.entity_type,
            canonical_id,
            scope.scope_key,
            SourcePresence.PRESENT,
        ))

    authoritative = (
        scope.completeness is SnapshotCompleteness.COMPLETE
        and scope.deletion_authoritative
    )
    confirmed_deleted: list[str] = []
    for canonical_id in sorted(tracked_scope_members):
        if canonical_id in observed_canonical_ids:
            continue
        if authoritative:
            presence = SourcePresence.CONFIRMED_DELETED
            confirmed_deleted.append(canonical_id)
            durable_updates.append(SourcePresenceRecord(
                scope.source_system,
                scope.entity_type,
                canonical_id,
                scope.scope_key,
                presence,
            ))
        elif scope.completeness is SnapshotCompleteness.UNKNOWN:
            presence = SourcePresence.UNKNOWN
        else:
            presence = SourcePresence.MISSING
        presence_outcomes[canonical_id] = presence

    try:
        merge_result = merge_observations(current_model, observations)
    except MergeValidationError as error:
        raise SnapshotValidationError(str(error)) from error
    repository.save_model_with_source_presence(merge_result.model, durable_updates)
    return SnapshotResult(
        merge_result.model,
        merge_result.changed_fields,
        presence_outcomes,
        tuple(durable_updates),
        tuple(confirmed_deleted),
        tuple(unresolved),
    )
