"""Typed source observations for already-resolved canonical entities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from music_agent.identity import EntityType, validate_canonical_id


class ObservationValidationError(ValueError):
    code = "validation_error"


class ObservationState(StrEnum):
    MISSING = "missing"
    NULL = "null"
    VALUE = "value"


class SourcePresence(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    CONFIRMED_DELETED = "confirmed_deleted"
    UNKNOWN = "unknown"


_UNSET = object()


@dataclass(frozen=True, slots=True, init=False)
class ObservedValue:
    state: ObservationState
    payload: Any

    def __init__(self, state: ObservationState, payload: Any = _UNSET) -> None:
        if not isinstance(state, ObservationState):
            raise ObservationValidationError("state must be an ObservationState")
        if state is ObservationState.VALUE:
            if payload is _UNSET:
                raise ObservationValidationError("VALUE requires a payload")
            if payload is None:
                raise ObservationValidationError("VALUE(None) is invalid; use NULL")
        elif payload is not _UNSET:
            raise ObservationValidationError(f"{state.value.upper()} cannot carry a payload")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def missing(cls) -> ObservedValue:
        return cls(ObservationState.MISSING)

    @classmethod
    def null(cls) -> ObservedValue:
        return cls(ObservationState.NULL)

    @classmethod
    def value(cls, payload: Any) -> ObservedValue:
        return cls(ObservationState.VALUE, payload)


@dataclass(frozen=True, slots=True)
class SourceObservation:
    entity_type: EntityType
    canonical_id: str
    source_system: str
    fields: Mapping[str, ObservedValue]
    source_presence: SourcePresence = SourcePresence.PRESENT

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise ObservationValidationError("entity_type must be an EntityType")
        validate_canonical_id(self.entity_type, self.canonical_id)
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise ObservationValidationError("source_system must be a non-empty string")
        if not isinstance(self.source_presence, SourcePresence):
            raise ObservationValidationError("source_presence must be a SourcePresence")
        if not isinstance(self.fields, Mapping):
            raise ObservationValidationError("fields must be a mapping")
        copied_fields: dict[str, ObservedValue] = {}
        for path, observed in self.fields.items():
            if not isinstance(path, str) or path == "":
                raise ObservationValidationError("field paths must be non-empty strings")
            if not isinstance(observed, ObservedValue):
                raise ObservationValidationError(f"{path} must contain an ObservedValue")
            copied_fields[path] = observed
        if self.source_presence is not SourcePresence.PRESENT and copied_fields:
            raise ObservationValidationError("non-present observations cannot carry fields")
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))
