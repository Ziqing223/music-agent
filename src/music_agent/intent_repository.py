"""Durable SQLite persistence for PendingIntent operational state.

A ``PendingIntent`` is operational state, not a canonical entity. This module persists it in its
own table (``pending_write_intents``) with an owned child table
(``pending_write_intent_requirements``) that holds its explicit identity requirements. Both are
isolated from ``canonical_entities``, external identity bindings, and source presence, and never
mutate the canonical model.

The repository does not re-derive transition legality. It persists intents that the domain layer
(``write_intent``) has already validated, and ``update_intent_state`` delegates state advancement
to ``advance_intent`` so an illegal transition raises before anything is written. A command
success is never persisted as ``CONFIRMED``; only a matching readback event can reach it.

The parent row and its requirement rows share one transaction, so an intent and its requirements
can never split brain. Idempotency compares the full requirement set: re-saving the same
``intent_id`` with identical requirements is a no-op, and reusing it with any different
requirement (a changed playlist or track) fails closed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.repository import _open_store_connection
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.write_intent import (
    IntentState,
    PendingIntent,
    RequirementRole,
    WriteEvent,
    WriteOperation,
    WriteRequirement,
    advance_intent,
    validate_intent_id,
)


class PendingIntentRepositoryError(ValueError):
    code = "pending_intent_repository_error"


_INTENT_COLUMNS = (
    "intent_id",
    "operation",
    "requested_state",
    "requested_value_json",
    "lifecycle_state",
)

_REQUIREMENT_COLUMNS = (
    "intent_id",
    "role",
    "canonical_id",
    "source_system",
    "entity_type",
    "external_id",
)


class PendingIntentRepository:
    """Persist PendingIntent operational state inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PendingIntentRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def save_intent(self, intent: PendingIntent) -> None:
        """Durably record a PENDING intent and its requirements, never minting a second intent.

        Re-saving the same ``intent_id`` with identical content (including the full requirement
        set) is a no-op (a retry reuses the same identity); reusing an ``intent_id`` with any
        different requirement fails closed. Only PENDING intents may be saved -- state changes go
        through ``update_intent_state``.
        """
        _require_intent(intent)
        if intent.state is not IntentState.PENDING:
            raise PendingIntentRepositoryError(
                "save_intent only persists PENDING intents; use update_intent_state to advance"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = _load_pending_intent(self._connection, intent.intent_id)
            if existing is not None:
                if existing != intent:
                    raise PendingIntentRepositoryError(
                        f"intent_id {intent.intent_id!r} already exists with different content"
                    )
            else:
                self._connection.execute(
                    """INSERT INTO pending_write_intents(
                        intent_id, operation, requested_state, requested_value_json, lifecycle_state
                    ) VALUES (?, ?, ?, ?, ?)""",
                    _encode_intent_values(intent),
                )
                self._connection.executemany(
                    """INSERT INTO pending_write_intent_requirements(
                        intent_id, role, canonical_id, source_system, entity_type, external_id
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    _encode_requirement_rows(intent),
                )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def get_intent(self, intent_id: str) -> PendingIntent | None:
        validate_intent_id(intent_id)
        return _load_pending_intent(self._connection, intent_id)

    def update_intent_state(self, intent_id: str, event: WriteEvent) -> PendingIntent:
        """Advance one intent's lifecycle via a domain event and persist the new state.

        Legality is delegated to ``advance_intent``: an illegal transition raises
        ``WriteTransitionError`` before any write, so it never persists. The update targets the
        existing ``intent_id`` row and does not create a second logical intent.
        """
        validate_intent_id(intent_id)
        if not isinstance(event, WriteEvent):
            raise PendingIntentRepositoryError("event must be a WriteEvent")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            intent = _load_pending_intent(self._connection, intent_id)
            if intent is None:
                raise PendingIntentRepositoryError(f"intent {intent_id!r} does not exist")
            advanced = advance_intent(intent, event)
            self._connection.execute(
                """UPDATE pending_write_intents
                SET lifecycle_state=?, updated_at=CURRENT_TIMESTAMP
                WHERE intent_id=?""",
                (advanced.state.value, intent_id),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return advanced

    def list_intents(self, state: IntentState | None = None) -> tuple[PendingIntent, ...]:
        """Return intents, optionally filtered to one lifecycle state."""
        if state is not None and not isinstance(state, IntentState):
            raise PendingIntentRepositoryError("state must be an IntentState or None")
        if state is None:
            rows = self._connection.execute(
                "SELECT intent_id FROM pending_write_intents ORDER BY intent_id"
            )
        else:
            rows = self._connection.execute(
                """SELECT intent_id FROM pending_write_intents
                WHERE lifecycle_state=? ORDER BY intent_id""",
                (state.value,),
            )
        return tuple(
            intent
            for intent in (
                _load_pending_intent(self._connection, row["intent_id"]) for row in rows
            )
            if intent is not None
        )


def _require_intent(intent: object) -> PendingIntent:
    if not isinstance(intent, PendingIntent):
        raise PendingIntentRepositoryError("intent must be a PendingIntent")
    return intent


def _load_pending_intent(
    connection: sqlite3.Connection, intent_id: str
) -> PendingIntent | None:
    """Load one intent together with its requirement rows, or ``None`` if absent.

    Shared by both the intent repository and the execution repository so a decoded intent always
    carries its full requirement set -- including across state transitions, which re-derive the
    next intent from the latest durable content.
    """
    row = connection.execute(
        "SELECT * FROM pending_write_intents WHERE intent_id=?", (intent_id,)
    ).fetchone()
    if row is None:
        return None
    requirement_rows = connection.execute(
        """SELECT * FROM pending_write_intent_requirements
        WHERE intent_id=? ORDER BY role""",
        (intent_id,),
    ).fetchall()
    return _decode_intent(row, requirement_rows)


def _encode_intent_values(intent: PendingIntent) -> tuple[Any, ...]:
    requested_state, requested_value_json = _encode_requested_value(intent.requested_value)
    return (
        intent.intent_id,
        intent.operation.value,
        requested_state,
        requested_value_json,
        intent.state.value,
    )


def _encode_requirement_rows(intent: PendingIntent) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            intent.intent_id,
            requirement.role.value,
            requirement.canonical_id,
            requirement.external_identity.source_system,
            requirement.external_identity.entity_type.value,
            requirement.external_identity.external_id,
        )
        for requirement in intent.requirements
    )


def _decode_intent(
    row: sqlite3.Row, requirement_rows: list[sqlite3.Row]
) -> PendingIntent:
    try:
        requirements = tuple(_decode_requirement(r) for r in requirement_rows)
        return PendingIntent(
            intent_id=row["intent_id"],
            operation=WriteOperation(row["operation"]),
            requirements=requirements,
            requested_value=_decode_requested_value(
                row["requested_state"], row["requested_value_json"]
            ),
            state=IntentState(row["lifecycle_state"]),
        )
    except PendingIntentRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise PendingIntentRepositoryError(f"corrupt pending intent row: {error}") from error


def _decode_requirement(row: sqlite3.Row) -> WriteRequirement:
    entity_type = EntityType(row["entity_type"])
    return WriteRequirement(
        role=RequirementRole(row["role"]),
        canonical_id=row["canonical_id"],
        external_identity=ExternalIdentityKey(
            row["source_system"], entity_type, row["external_id"]
        ),
    )


def _encode_requested_value(requested: ObservedValue) -> tuple[str, str | None]:
    if requested.state is ObservationState.NULL:
        return requested.state.value, None
    if requested.state is ObservationState.MISSING:
        raise PendingIntentRepositoryError("requested_value cannot be MISSING")
    try:
        encoded = json.dumps(
            requested.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PendingIntentRepositoryError(
            f"requested value is not JSON serializable: {error}"
        ) from error
    return requested.state.value, encoded


def _decode_requested_value(state_value: str, value_json: str | None) -> ObservedValue:
    try:
        state = ObservationState(state_value)
    except ValueError as error:
        raise PendingIntentRepositoryError(f"unknown requested state {state_value!r}") from error
    if state is ObservationState.NULL:
        if value_json is not None:
            raise PendingIntentRepositoryError("NULL requested state cannot carry a value")
        return ObservedValue.null()
    if state is ObservationState.MISSING:
        raise PendingIntentRepositoryError("MISSING requested state cannot be persisted")
    if value_json is None:
        raise PendingIntentRepositoryError("VALUE requested state requires a value")
    try:
        payload = json.loads(value_json)
    except ValueError as error:
        raise PendingIntentRepositoryError(f"invalid requested value JSON: {error}") from error
    return ObservedValue.value(payload)
