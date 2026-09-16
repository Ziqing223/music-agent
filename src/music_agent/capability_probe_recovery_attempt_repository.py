"""Durable SQLite persistence for capability-probe recovery attempts.

A ``capability_probe_recovery_attempt`` is an operational fact about one recovery-driven restore
command attempt, stored in its own table (``capability_probe_recovery_attempts``) and isolated from
``canonical_entities``, ``external_identity_bindings``, ``source_entity_presence``,
``pending_write_intents``, ``write_execution_attempts``, and ``capability_probes``. It never mutates
the probe row's ``verification_verdict`` and never mutates the write capability matrix.

This repository owns durable facts and transactions. It never calls the external system and never
re-derives transition legality: attempt transitions delegate to ``advance_attempt``. Command and
readback sequencing lives in the (future) recovery orchestrator, which is out of scope here.

The critical safety invariant is that a STARTED attempt is durably committed BEFORE any external
recovery restore command, inside a transaction that atomically re-confirms (a) the probe still
exists and (b) no recovery attempt of *any* state already exists for that probe. A strict
fail-closed contract then guarantees at most one automatic recovery attempt per probe: the
``UNIQUE(probe_id)`` column constraint backs the in-transaction check at the database level, so a
second attempt is rejected even after a prior attempt reaches a terminal state. A terminal attempt
is immutable: ``advance_attempt`` only accepts ``STARTED -> COMMAND_SUCCEEDED/COMMAND_FAILED`` and
rejects every other transition, so a stale outcome can never overwrite a terminal state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from music_agent.capability_probe import validate_probe_id
from music_agent.capability_probe_recovery_attempt import (
    RecoveryAttempt,
    RecoveryAttemptEvent,
    RecoveryAttemptState,
    advance_attempt,
    create_attempt,
    validate_recovery_attempt_id,
)
from music_agent.capability_probe_repository import _load_probe
from music_agent.repository import _open_store_connection


class CapabilityProbeRecoveryAttemptRepositoryError(ValueError):
    code = "capability_probe_recovery_attempt_repository_error"


class RecoveryAttemptAlreadyExistsError(CapabilityProbeRecoveryAttemptRepositoryError):
    code = "recovery_attempt_already_exists"


class RecoveryAttemptNotFoundError(CapabilityProbeRecoveryAttemptRepositoryError):
    code = "recovery_attempt_not_found"


class RecoveryAttemptProbeNotFoundError(CapabilityProbeRecoveryAttemptRepositoryError):
    code = "recovery_attempt_probe_not_found"


class CapabilityProbeRecoveryAttemptRepository:
    """Persist recovery-attempt facts inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CapabilityProbeRecoveryAttemptRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def begin_attempt(self, probe_id: str) -> RecoveryAttempt:
        """Atomically commit a STARTED recovery attempt, or fail closed.

        Inside one transaction this re-reads the durable probe and confirms it still exists and that
        no recovery attempt of any state already exists for it, then inserts STARTED. The
        ``UNIQUE(probe_id)`` constraint backs the existence check at the database level, so at most
        one automatic recovery attempt can ever exist per probe, regardless of prior terminal state.
        The caller commits STARTED *before* issuing any external restore command.
        """
        validate_probe_id(probe_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if _load_probe(self._connection, probe_id) is None:
                raise RecoveryAttemptProbeNotFoundError(
                    f"probe {probe_id!r} does not exist"
                )
            if self._load_attempt_for_probe(probe_id) is not None:
                raise RecoveryAttemptAlreadyExistsError(
                    f"probe {probe_id!r} already has a recovery attempt"
                )
            attempt = create_attempt(probe_id)
            self._connection.execute(
                """INSERT INTO capability_probe_recovery_attempts(attempt_id, probe_id, state)
                VALUES (?, ?, ?)""",
                (attempt.attempt_id, attempt.probe_id, attempt.state.value),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return attempt

    def get_attempt(self, attempt_id: str) -> RecoveryAttempt | None:
        validate_recovery_attempt_id(attempt_id)
        row = self._load_attempt_row(attempt_id)
        return None if row is None else _decode_attempt(row)

    def get_for_probe(self, probe_id: str) -> RecoveryAttempt | None:
        validate_probe_id(probe_id)
        row = self._load_attempt_for_probe(probe_id)
        return None if row is None else _decode_attempt(row)

    def mark_command_succeeded(self, attempt_id: str) -> RecoveryAttempt:
        """Persist a known recovery-command success, failing closed if the attempt is not STARTED."""
        return self._record_outcome(attempt_id, RecoveryAttemptEvent.COMMAND_SUCCEEDED)

    def mark_command_failed(self, attempt_id: str) -> RecoveryAttempt:
        """Persist a known recovery-command failure, failing closed if the attempt is not STARTED."""
        return self._record_outcome(attempt_id, RecoveryAttemptEvent.COMMAND_FAILED)

    def _record_outcome(
        self, attempt_id: str, event: RecoveryAttemptEvent
    ) -> RecoveryAttempt:
        validate_recovery_attempt_id(attempt_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._load_attempt_row(attempt_id)
            if row is None:
                raise RecoveryAttemptNotFoundError(
                    f"recovery attempt {attempt_id!r} does not exist"
                )
            advanced = advance_attempt(_decode_attempt(row), event)
            self._connection.execute(
                """UPDATE capability_probe_recovery_attempts
                SET state=?, updated_at=CURRENT_TIMESTAMP
                WHERE attempt_id=?""",
                (advanced.state.value, attempt_id),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return advanced

    def _load_attempt_row(self, attempt_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM capability_probe_recovery_attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()

    def _load_attempt_for_probe(self, probe_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM capability_probe_recovery_attempts WHERE probe_id=?",
            (probe_id,),
        ).fetchone()


def _decode_attempt(row: sqlite3.Row) -> RecoveryAttempt:
    try:
        return RecoveryAttempt(
            attempt_id=row["attempt_id"],
            probe_id=row["probe_id"],
            state=RecoveryAttemptState(row["state"]),
        )
    except CapabilityProbeRecoveryAttemptRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise CapabilityProbeRecoveryAttemptRepositoryError(
            f"corrupt recovery attempt row: {error}"
        ) from error
