"""Ownership-aware merge of source observations into a canonical candidate model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

from music_agent.identity import EntityType
from music_agent.source_observation import ObservationState, SourceObservation, SourcePresence
from music_agent.validation import GraphValidationError, StructuralValidationError, validate_fixture


class MergeValidationError(ValueError):
    code = "validation_error"


class Authority(StrEnum):
    APPLE_MUSIC = "apple_music"
    SHARED_MODEL = "shared_model"


class FieldDisposition(StrEnum):
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    MISSING_PRESERVED = "missing_preserved"
    NOT_AUTHORITATIVE_PRESERVED = "not_authoritative_preserved"


@dataclass(frozen=True, slots=True)
class FieldOutcome:
    entity_type: EntityType
    canonical_id: str
    field_path: str
    disposition: FieldDisposition


@dataclass(frozen=True, slots=True)
class MergeResult:
    model: dict[str, Any]
    changed_fields: tuple[str, ...]
    field_outcomes: tuple[FieldOutcome, ...]
    source_presence: dict[str, SourcePresence]


OWNERSHIP_POLICY: dict[tuple[EntityType, str], Authority] = {
    (EntityType.TRACK, path): Authority.APPLE_MUSIC
    for path in (
        "name", "artist_ids", "album_id", "duration_ms", "genres", "track_number",
        "disc_number", "release_date", "composer", "library_state.favorited",
        "library_state.disliked", "library_state.rating", "library_state.play_count",
        "library_state.skip_count", "library_state.added_to_library_at",
        "library_state.last_played_at",
    )
} | {
    (EntityType.TRACK, "agent_metadata.tags"): Authority.SHARED_MODEL,
    (EntityType.ARTIST, "name"): Authority.APPLE_MUSIC,
    (EntityType.ALBUM, "name"): Authority.APPLE_MUSIC,
    (EntityType.ALBUM, "artist_ids"): Authority.APPLE_MUSIC,
    (EntityType.ALBUM, "release_date"): Authority.APPLE_MUSIC,
    (EntityType.PLAYLIST, "name"): Authority.APPLE_MUSIC,
    (EntityType.PLAYLIST_MEMBERSHIP, "playlist_id"): Authority.APPLE_MUSIC,
    (EntityType.PLAYLIST_MEMBERSHIP, "track_id"): Authority.APPLE_MUSIC,
    (EntityType.PLAYLIST_MEMBERSHIP, "position"): Authority.APPLE_MUSIC,
    (EntityType.PLAYLIST_MEMBERSHIP, "added_at"): Authority.APPLE_MUSIC,
}


ENTITY_COLLECTIONS: dict[EntityType, str] = {
    EntityType.TRACK: "tracks",
    EntityType.ARTIST: "artists",
    EntityType.ALBUM: "albums",
    EntityType.PLAYLIST: "playlists",
    EntityType.PLAYLIST_MEMBERSHIP: "playlist_memberships",
}


def merge_observations(
    current_model: dict[str, Any], observations: Iterable[SourceObservation]
) -> MergeResult:
    """Return a validated candidate without mutating the current canonical model."""
    try:
        validate_fixture(current_model)
    except (StructuralValidationError, GraphValidationError) as error:
        raise MergeValidationError(f"current model is invalid: {error}") from error

    candidate = deepcopy(current_model)
    outcomes: list[FieldOutcome] = []
    touched_fields: list[tuple[EntityType, str, str]] = []
    presence: dict[str, SourcePresence] = {}

    for observation in observations:
        if not isinstance(observation, SourceObservation):
            raise MergeValidationError("observations must contain SourceObservation values")
        if observation.source_system != Authority.APPLE_MUSIC.value:
            raise MergeValidationError(f"unsupported source system: {observation.source_system}")
        entity = _find_target(candidate, observation.entity_type, observation.canonical_id)
        target_key = _target_key(observation.entity_type, observation.canonical_id)
        presence[target_key] = observation.source_presence
        if observation.source_presence is not SourcePresence.PRESENT:
            continue

        for field_path, observed in observation.fields.items():
            authority = _field_authority(observation.entity_type, field_path)
            if observed.state is ObservationState.MISSING:
                disposition = FieldDisposition.MISSING_PRESERVED
            elif authority is Authority.SHARED_MODEL:
                disposition = FieldDisposition.NOT_AUTHORITATIVE_PRESERVED
            else:
                new_value = None if observed.state is ObservationState.NULL else deepcopy(observed.payload)
                old_value = _get_field(entity, field_path)
                if old_value == new_value:
                    disposition = FieldDisposition.UNCHANGED
                else:
                    _set_field(entity, field_path, new_value)
                    disposition = FieldDisposition.UPDATED
                    touched_fields.append(
                        (observation.entity_type, observation.canonical_id, field_path)
                    )
            outcomes.append(FieldOutcome(
                observation.entity_type, observation.canonical_id, field_path, disposition
            ))

    try:
        validate_fixture(candidate)
    except (StructuralValidationError, GraphValidationError) as error:
        raise MergeValidationError(f"candidate model is invalid: {error}") from error
    changed_fields = _net_changed_fields(current_model, candidate, touched_fields)
    return MergeResult(candidate, changed_fields, tuple(outcomes), presence)


def _find_target(model: dict[str, Any], entity_type: EntityType, canonical_id: str) -> dict[str, Any]:
    for entity in model[ENTITY_COLLECTIONS[entity_type]]:
        if entity["id"] == canonical_id:
            return entity
    raise MergeValidationError(f"unresolved target: {_target_key(entity_type, canonical_id)}")


def _field_authority(entity_type: EntityType, field_path: str) -> Authority:
    if field_path == "id" or field_path == "external_ids" or field_path.startswith("external_ids."):
        raise MergeValidationError(f"identity field is not patchable: {field_path}")
    try:
        return OWNERSHIP_POLICY[(entity_type, field_path)]
    except KeyError as error:
        raise MergeValidationError(
            f"unknown field path for {entity_type.value}: {field_path}"
        ) from error


def _get_field(entity: dict[str, Any], field_path: str) -> Any:
    value: Any = entity
    for part in field_path.split("."):
        value = value[part]
    return value


def _set_field(entity: dict[str, Any], field_path: str, value: Any) -> None:
    parts = field_path.split(".")
    target = entity
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _target_key(entity_type: EntityType, canonical_id: str) -> str:
    return f"{entity_type.value}:{canonical_id}"


def _net_changed_fields(
    current_model: dict[str, Any],
    candidate: dict[str, Any],
    touched_fields: list[tuple[EntityType, str, str]],
) -> tuple[str, ...]:
    changed: list[str] = []
    seen: set[tuple[EntityType, str, str]] = set()
    for entity_type, canonical_id, field_path in touched_fields:
        key = (entity_type, canonical_id, field_path)
        if key in seen:
            continue
        seen.add(key)
        current_entity = _find_target(current_model, entity_type, canonical_id)
        candidate_entity = _find_target(candidate, entity_type, canonical_id)
        if _get_field(current_entity, field_path) != _get_field(candidate_entity, field_path):
            changed.append(f"{_target_key(entity_type, canonical_id)}:{field_path}")
    return tuple(changed)
