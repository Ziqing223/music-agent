"""Durable SQLite persistence for write-execution attempts.

A ``write_execution_attempt`` is an operational fact about one command attempt, stored in its own
table (``write_execution_attempts``) and isolated from ``canonical_entities``,
``external_identity_bindings``, and ``source_entity_presence``. It is NOT the PendingIntent
lifecycle: that stays in ``pending_write_intents`` and is advanced through the domain
``advance_intent``, never re-derived here.

This repository owns durable facts and transactions. It never calls the external system and never
re-derives transition legality: intent transitions delegate to ``advance_intent`` and attempt
transitions to ``advance_attempt``. Command and readback sequencing lives in the orchestrator.

The critical safety invariant is that a STARTED attempt is durably committed BEFORE any external
command is invoked, inside a transaction that atomically re-confirms the intent is still PENDING
and that no conflicting attempt exists. The command outcome is then persisted in the SAME
transaction that advances the intent, so an attempt outcome and an intent state can never split
brain.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from music_agent.intent_repository import _load_pending_intent
from music_agent.repository import _open_store_connection
from music_agent.write_execution import (
    AttemptEvent,
    AttemptState,
    ExecutionAttempt,
    advance_attempt,
    create_attempt,
    validate_attempt_id,
)
from music_agent.write_intent import (
    IntentState,
    PendingIntent,
    WriteEvent,
    advance_intent,
    validate_intent_id,
)


class WriteExecutionRepositoryError(ValueError):
    code = "write_execution_repository_error"


class ExecutionNotPendingError(WriteExecutionRepositoryError):
    code = "execution_not_pending"


class AmbiguousAttemptError(WriteExecutionRepositoryError):
    code = "ambiguous_execution_attempt"


class AttemptNotFoundError(WriteExecutionRepositoryError):
    code = "attempt_not_found"


class ReadbackGateError(WriteExecutionRepositoryError):
    code = "readback_not_allowed"


class WriteExecutionRepository:
    """Persist execution-attempt facts inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> WriteExecutionRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def begin_execution(self, intent_id: str) -> ExecutionAttempt:
        """Atomically commit a STARTED attempt, or fail closed.

        Inside one transaction this re-reads the latest durable intent state and confirms it is
        still PENDING and that no STARTED attempt already exists. A STARTED attempt whose outcome
        is unknown (a crash between STARTED and outcome persistence) blocks a second attempt, so
        the external command is never automatically re-issued.
        """
        validate_intent_id(intent_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            intent = _load_pending_intent(self._connection, intent_id)
            if intent is None:
                raise ExecutionNotPendingError(f"intent {intent_id!r} does not exist")
            if intent.state is not IntentState.PENDING:
                raise ExecutionNotPendingError(
                    f"intent {intent_id!r} is {intent.state.value!r}; "
                    "execution requires PENDING"
                )
            active = self._connection.execute(
                "SELECT 1 FROM write_execution_attempts WHERE intent_id=? AND state=?",
                (intent_id, AttemptState.STARTED.value),
            ).fetchone()
            if active is not None:
                raise AmbiguousAttemptError(
                    f"intent {intent_id!r} already has a STARTED attempt with unknown outcome"
                )
            attempt = create_attempt(intent_id)
            self._connection.execute(
                """INSERT INTO write_execution_attempts(attempt_id, intent_id, state)
                VALUES (?, ?, ?)""",
                (attempt.attempt_id, attempt.intent_id, attempt.state.value),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return attempt

    def record_command_success(self, intent_id: str, attempt_id: str) -> ExecutionAttempt:
        """Persist a known command success atomically with the intent advancement."""
        return self._record_command_outcome(
            intent_id, attempt_id, AttemptEvent.COMMAND_SUCCEEDED, WriteEvent.COMMAND_SUCCEEDED
        )

    def record_command_failure(self, intent_id: str, attempt_id: str) -> ExecutionAttempt:
        """Persist a known command failure atomically with the intent advancement."""
        return self._record_command_outcome(
            intent_id, attempt_id, AttemptEvent.COMMAND_FAILED, WriteEvent.COMMAND_FAILED
        )

    def record_command_unknown(self, intent_id: str, attempt_id: str) -> ExecutionAttempt:
        """Persist an ambiguous command outcome atomically with the intent advancement.

        The command was dispatched and its side effect cannot be proven absent, so the attempt
        lands on ``COMMAND_UNKNOWN`` and the intent on ``OUTCOME_UNKNOWN`` (never
        ``COMMAND_FAILED`` / ``EXECUTION_FAILED``).
        """
        return self._record_command_outcome(
            intent_id, attempt_id, AttemptEvent.COMMAND_UNKNOWN, WriteEvent.COMMAND_UNKNOWN
        )

    def _record_command_outcome(
        self,
        intent_id: str,
        attempt_id: str,
        attempt_event: AttemptEvent,
        intent_event: WriteEvent,
    ) -> ExecutionAttempt:
        validate_intent_id(intent_id)
        validate_attempt_id(attempt_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            intent = _load_pending_intent(self._connection, intent_id)
            if intent is None:
                raise WriteExecutionRepositoryError(f"intent {intent_id!r} does not exist")
            attempt_row = self._load_attempt_row(attempt_id)
            if attempt_row is None or attempt_row["intent_id"] != intent_id:
                raise AttemptNotFoundError(
                    f"attempt {attempt_id!r} does not exist for intent {intent_id!r}"
                )
            advanced_intent = advance_intent(intent, intent_event)
            advanced_attempt = advance_attempt(_decode_attempt(attempt_row), attempt_event)
            self._connection.execute(
                """UPDATE pending_write_intents
                SET lifecycle_state=?, updated_at=CURRENT_TIMESTAMP
                WHERE intent_id=?""",
                (advanced_intent.state.value, intent_id),
            )
            self._connection.execute(
                """UPDATE write_execution_attempts
                SET state=?, updated_at=CURRENT_TIMESTAMP
                WHERE attempt_id=?""",
                (advanced_attempt.state.value, attempt_id),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return advanced_attempt

    def record_readback(self, intent_id: str, event: WriteEvent) -> PendingIntent:
        """Advance an AWAITING_READBACK intent after a durable command success.

        The transition is gated on two durable facts re-read inside one transaction: the intent is
        AWAITING_READBACK and the latest attempt outcome is COMMAND_SUCCEEDED. Legality is
        delegated to ``advance_intent``; the readback gate itself fails closed.
        """
        validate_intent_id(intent_id)
        if event not in (WriteEvent.READBACK_MATCHED, WriteEvent.READBACK_MISMATCHED):
            raise WriteExecutionRepositoryError(
                "readback event must be READBACK_MATCHED or READBACK_MISMATCHED"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            intent = _load_pending_intent(self._connection, intent_id)
            if intent is None:
                raise WriteExecutionRepositoryError(f"intent {intent_id!r} does not exist")
            latest = self._latest_attempt_row(intent_id)
            if latest is None or latest["state"] != AttemptState.COMMAND_SUCCEEDED.value:
                raise ReadbackGateError(
                    f"intent {intent_id!r} has no durable COMMAND_SUCCEEDED attempt; "
                    "readback is not allowed"
                )
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

    def record_reconciliation(self, intent_id: str) -> PendingIntent:
        """Reconcile an OUTCOME_UNKNOWN intent to CONFIRMED after a matching readback.

        This is the readback-based confirmation for an ambiguous command outcome. It is gated on
        two durable facts re-read inside one transaction: the intent is OUTCOME_UNKNOWN and the
        latest attempt outcome is COMMAND_UNKNOWN. Legality is delegated to ``advance_intent``;
        the gate fails closed. It never re-runs the command and never records a mismatch.
        """
        validate_intent_id(intent_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            intent = _load_pending_intent(self._connection, intent_id)
            if intent is None:
                raise WriteExecutionRepositoryError(f"intent {intent_id!r} does not exist")
            if intent.state is not IntentState.OUTCOME_UNKNOWN:
                raise ReadbackGateError(
                    f"intent {intent_id!r} is {intent.state.value!r}; "
                    "reconciliation requires OUTCOME_UNKNOWN"
                )
            latest = self._latest_attempt_row(intent_id)
            if latest is None or latest["state"] != AttemptState.COMMAND_UNKNOWN.value:
                raise ReadbackGateError(
                    f"intent {intent_id!r} has no durable COMMAND_UNKNOWN attempt; "
                    "reconciliation is not allowed"
                )
            advanced = advance_intent(intent, WriteEvent.READBACK_MATCHED)
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

    def get_intent(self, intent_id: str) -> PendingIntent | None:
        validate_intent_id(intent_id)
        return _load_pending_intent(self._connection, intent_id)

    def get_attempt(self, attempt_id: str) -> ExecutionAttempt | None:
        validate_attempt_id(attempt_id)
        row = self._load_attempt_row(attempt_id)
        return None if row is None else _decode_attempt(row)

    def get_latest_attempt(self, intent_id: str) -> ExecutionAttempt | None:
        validate_intent_id(intent_id)
        row = self._latest_attempt_row(intent_id)
        return None if row is None else _decode_attempt(row)

    def list_attempts(self, intent_id: str) -> tuple[ExecutionAttempt, ...]:
        validate_intent_id(intent_id)
        rows = self._connection.execute(
            """SELECT * FROM write_execution_attempts
            WHERE intent_id=? ORDER BY created_at, attempt_id""",
            (intent_id,),
        )
        return tuple(_decode_attempt(row) for row in rows)

    def _load_attempt_row(self, attempt_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM write_execution_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()

    def _latest_attempt_row(self, intent_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            """SELECT * FROM write_execution_attempts
            WHERE intent_id=? ORDER BY created_at DESC, attempt_id DESC LIMIT 1""",
            (intent_id,),
        ).fetchone()


def _decode_attempt(row: sqlite3.Row) -> ExecutionAttempt:
    try:
        return ExecutionAttempt(
            attempt_id=row["attempt_id"],
            intent_id=row["intent_id"],
            state=AttemptState(row["state"]),
        )
    except WriteExecutionRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise WriteExecutionRepositoryError(f"corrupt execution attempt row: {error}") from error
