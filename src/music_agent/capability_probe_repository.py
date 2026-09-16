"""Durable SQLite persistence for capability-probe operational state.

A ``CapabilityProbe`` is operational verification state, not a canonical entity, an intent, or an
execution attempt. This module persists it in its own table (``capability_probes``), isolated from
``canonical_entities``, ``external_identity_bindings``, ``source_entity_presence``,
``pending_write_intents``, and ``write_execution_attempts``, and it never mutates the canonical
model or the write capability matrix.

The repository persists durable facts and transactions; it does not re-derive probe transition
legality. The domain layer (``capability_probe``) advances a probe through its forward/restore
cycle and the repository records the resulting durable state. ``save_probe`` inserts the initial
probe and is idempotent only on an exact re-save; a conflicting identity or baseline fails closed.
``update_probe`` is guarded by an optimistic expected-state contract: the caller declares the exact
durable probe it transitioned *from*, and the transaction re-reads the latest durable probe and
requires it to equal that declaration before writing, so a stale caller can never overwrite a later
durable lifecycle state.

The three orthogonal axes (``step_state``, ``recovery_status``, ``verification_verdict``) are three
independent columns, never collapsed into one state. The six trailing observations record the
forward/restore command outcome and readback observations; an unobserved field is ``NULL`` in
storage and ``None`` in the domain object, and a real ``MISSING`` readback is a distinct stored
state, so ``False`` is never conflated with missing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    existing_probe_blocks_new_probe,
    is_recoverable,
    validate_probe_id,
)
from music_agent.capability_probe_recovery_attempt import RecoveryAttemptState
from music_agent.repository import _open_store_connection
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.write_intent import WriteOperation


class CapabilityProbeRepositoryError(ValueError):
    code = "capability_probe_repository_error"


class ProbeConflictError(CapabilityProbeRepositoryError):
    code = "probe_conflict"


class ProbeNotFoundError(CapabilityProbeRepositoryError):
    code = "probe_not_found"


class StaleProbeUpdateError(CapabilityProbeRepositoryError):
    code = "stale_probe_update"


class ProbeTargetBlockedError(CapabilityProbeRepositoryError):
    code = "probe_target_blocked"


class CapabilityProbeRepository:
    """Persist capability-probe operational state inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CapabilityProbeRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def save_probe(self, probe: CapabilityProbe) -> CapabilityProbe:
        """Durably record a probe, never minting a second probe for the same identity.

        Re-saving the same ``probe_id`` with identical content is a no-op. A probe whose identity
        or baseline contract differs (``operation``, ``target_canonical_id``,
        ``target_persistent_id``, ``baseline_favorited``, ``baseline_disliked``) fails closed, and
        any other content difference is also rejected -- lifecycle transitions go through
        ``update_probe``.
        """
        _require_probe(probe)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = _load_probe(self._connection, probe.probe_id)
            if existing is not None:
                if _immutable_signature(existing) != _immutable_signature(probe):
                    raise ProbeConflictError(
                        f"probe_id {probe.probe_id!r} already exists with a different "
                        "identity or baseline"
                    )
                if existing != probe:
                    raise ProbeConflictError(
                        f"probe_id {probe.probe_id!r} already exists with different content"
                    )
            else:
                self._insert_probe_row(probe)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        return probe

    def begin_forward_probe(self, probe: CapabilityProbe) -> CapabilityProbe:
        """Atomically capture a probe at ``FORWARD_STARTED``, or fail closed if the target blocks.

        Inside one ``BEGIN IMMEDIATE`` transaction this re-reads every same-target probe (with its
        recovery attempt state) and refuses if any blocks a new probe, then inserts the new probe at
        ``FORWARD_STARTED``. No external I/O happens inside the transaction, so the forward command
        is never issued while the capture is still uncommitted. The same-target decision uses the
        shared ``existing_probe_blocks_new_probe`` predicate, never a second copy of the rule.
        """
        _require_probe(probe)
        if probe.step_state is not ProbeStepState.FORWARD_STARTED:
            raise CapabilityProbeRepositoryError(
                "begin_forward_probe requires a FORWARD_STARTED probe"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if _load_probe(self._connection, probe.probe_id) is not None:
                raise ProbeConflictError(f"probe_id {probe.probe_id!r} already exists")
            blocker = self._blocking_probe_for_target(probe.target_canonical_id)
            if blocker is not None:
                raise ProbeTargetBlockedError(
                    f"target {probe.target_canonical_id!r} is blocked by existing probe "
                    f"{blocker.probe_id!r}"
                )
            self._insert_probe_row(probe)
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        return probe

    def get_probe(self, probe_id: str) -> CapabilityProbe | None:
        validate_probe_id(probe_id)
        return _load_probe(self._connection, probe_id)

    def update_probe(
        self, expected_previous: CapabilityProbe, updated_probe: CapabilityProbe
    ) -> CapabilityProbe:
        """Persist a lifecycle transition under an optimistic expected-state guard.

        The caller declares the exact durable probe it transitioned *from*
        (``expected_previous``). Inside the transaction the latest durable probe is re-read and must
        equal ``expected_previous`` exactly; otherwise the update is stale and fails closed with
        ``StaleProbeUpdateError``. This guarantees a stale caller can never overwrite a later durable
        lifecycle state -- ``FORWARD_OBSERVED`` back to ``FORWARD_STARTED``, a terminal ``VERIFIED``
        back to ``PENDING``, a ``RESTORED`` recovery status back to an earlier value, or an existing
        observation back to unobserved. The identity / baseline contract must also match the incoming
        probe exactly. Transition legality itself is the domain layer's responsibility and is not
        re-derived here.
        """
        _require_probe(expected_previous)
        _require_probe(updated_probe)
        if expected_previous.probe_id != updated_probe.probe_id:
            raise ProbeConflictError(
                "expected_previous and updated_probe must share the same probe_id"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            latest = _load_probe(self._connection, updated_probe.probe_id)
            if latest is None:
                raise ProbeNotFoundError(f"probe {updated_probe.probe_id!r} does not exist")
            if latest != expected_previous:
                raise StaleProbeUpdateError(
                    f"probe {updated_probe.probe_id!r} changed since the caller's expected state"
                )
            if _immutable_signature(latest) != _immutable_signature(updated_probe):
                raise ProbeConflictError(
                    f"probe_id {updated_probe.probe_id!r} cannot change its identity or baseline"
                )
            self._connection.execute(
                """UPDATE capability_probes SET
                    step_state=?, recovery_status=?, verification_verdict=?,
                    forward_command_outcome=?,
                    forward_favorited_state=?, forward_favorited_value=?,
                    forward_disliked_state=?, forward_disliked_value=?,
                    restore_command_outcome=?,
                    restore_favorited_state=?, restore_favorited_value=?,
                    restore_disliked_state=?, restore_disliked_value=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE probe_id=?""",
                _encode_mutable_values(updated_probe) + (updated_probe.probe_id,),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        return updated_probe

    def list_probes(self) -> tuple[CapabilityProbe, ...]:
        """Return every durable probe in a deterministic order (by ``probe_id``)."""
        rows = self._connection.execute("SELECT * FROM capability_probes ORDER BY probe_id")
        return tuple(_decode_probe(row) for row in rows)

    def list_recoverable_probes(self) -> tuple[CapabilityProbe, ...]:
        """Return every durable probe in a recoverable ambiguous state, in ``probe_id`` order."""
        return tuple(probe for probe in self.list_probes() if is_recoverable(probe))

    def _insert_probe_row(self, probe: CapabilityProbe) -> None:
        self._connection.execute(
            """INSERT INTO capability_probes(
                probe_id, operation, target_canonical_id, target_persistent_id,
                baseline_favorited, baseline_disliked,
                step_state, recovery_status, verification_verdict,
                forward_command_outcome,
                forward_favorited_state, forward_favorited_value,
                forward_disliked_state, forward_disliked_value,
                restore_command_outcome,
                restore_favorited_state, restore_favorited_value,
                restore_disliked_state, restore_disliked_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _encode_probe_values(probe),
        )

    def _blocking_probe_for_target(self, target_canonical_id: str) -> CapabilityProbe | None:
        rows = self._connection.execute(
            """SELECT p.*, a.state AS recovery_attempt_state
               FROM capability_probes p
               LEFT JOIN capability_probe_recovery_attempts a ON a.probe_id = p.probe_id
               WHERE p.target_canonical_id = ?
               ORDER BY p.probe_id""",
            (target_canonical_id,),
        )
        for row in rows:
            probe = _decode_probe(row)
            started = row["recovery_attempt_state"] == RecoveryAttemptState.STARTED.value
            if existing_probe_blocks_new_probe(probe, started):
                return probe
        return None


def _require_probe(probe: object) -> CapabilityProbe:
    if not isinstance(probe, CapabilityProbe):
        raise CapabilityProbeRepositoryError("probe must be a CapabilityProbe")
    return probe


def _immutable_signature(probe: CapabilityProbe) -> tuple[Any, ...]:
    return (
        probe.probe_id,
        probe.operation,
        probe.target_canonical_id,
        probe.target_persistent_id,
        probe.baseline_favorited,
        probe.baseline_disliked,
    )


def _load_probe(connection: sqlite3.Connection, probe_id: str) -> CapabilityProbe | None:
    row = connection.execute(
        "SELECT * FROM capability_probes WHERE probe_id=?", (probe_id,)
    ).fetchone()
    return None if row is None else _decode_probe(row)


def _encode_probe_values(probe: CapabilityProbe) -> tuple[Any, ...]:
    return (
        probe.probe_id,
        probe.operation.value,
        probe.target_canonical_id,
        probe.target_persistent_id,
        int(probe.baseline_favorited),
        int(probe.baseline_disliked),
        probe.step_state.value,
        probe.recovery_status.value,
        probe.verification_verdict.value,
        _encode_command_outcome(probe.forward_command_outcome),
        *_encode_observation(probe.forward_favorited),
        *_encode_observation(probe.forward_disliked),
        _encode_command_outcome(probe.restore_command_outcome),
        *_encode_observation(probe.restore_favorited),
        *_encode_observation(probe.restore_disliked),
    )


def _encode_mutable_values(probe: CapabilityProbe) -> tuple[Any, ...]:
    return (
        probe.step_state.value,
        probe.recovery_status.value,
        probe.verification_verdict.value,
        _encode_command_outcome(probe.forward_command_outcome),
        *_encode_observation(probe.forward_favorited),
        *_encode_observation(probe.forward_disliked),
        _encode_command_outcome(probe.restore_command_outcome),
        *_encode_observation(probe.restore_favorited),
        *_encode_observation(probe.restore_disliked),
    )


def _encode_command_outcome(value: CommandOutcome | None) -> str | None:
    return None if value is None else value.value


def _encode_observation(value: ObservedValue | None) -> tuple[str | None, int | None]:
    if value is None:
        return (None, None)
    if value.state is ObservationState.MISSING:
        return ("missing", None)
    if value.state is ObservationState.VALUE:
        return ("value", 1 if value.payload else 0)
    raise CapabilityProbeRepositoryError(
        f"cannot persist a NULL observation (state={value.state.value!r})"
    )


def _decode_probe(row: sqlite3.Row) -> CapabilityProbe:
    try:
        return CapabilityProbe(
            probe_id=row["probe_id"],
            operation=WriteOperation(row["operation"]),
            target_canonical_id=row["target_canonical_id"],
            target_persistent_id=row["target_persistent_id"],
            baseline_favorited=bool(row["baseline_favorited"]),
            baseline_disliked=bool(row["baseline_disliked"]),
            step_state=ProbeStepState(row["step_state"]),
            recovery_status=RecoveryStatus(row["recovery_status"]),
            verification_verdict=VerificationVerdict(row["verification_verdict"]),
            forward_command_outcome=_decode_command_outcome(row["forward_command_outcome"]),
            forward_favorited=_decode_observation(
                row["forward_favorited_state"], row["forward_favorited_value"]
            ),
            forward_disliked=_decode_observation(
                row["forward_disliked_state"], row["forward_disliked_value"]
            ),
            restore_command_outcome=_decode_command_outcome(row["restore_command_outcome"]),
            restore_favorited=_decode_observation(
                row["restore_favorited_state"], row["restore_favorited_value"]
            ),
            restore_disliked=_decode_observation(
                row["restore_disliked_state"], row["restore_disliked_value"]
            ),
        )
    except CapabilityProbeRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise CapabilityProbeRepositoryError(f"corrupt capability probe row: {error}") from error


def _decode_command_outcome(value: str | None) -> CommandOutcome | None:
    return None if value is None else CommandOutcome(value)


def _decode_observation(state_value: str | None, int_value: int | None) -> ObservedValue | None:
    if state_value is None:
        if int_value is not None:
            raise CapabilityProbeRepositoryError(
                "unobserved favorited/disliked cannot carry a value"
            )
        return None
    if state_value == "missing":
        if int_value is not None:
            raise CapabilityProbeRepositoryError("MISSING favorited/disliked cannot carry a value")
        return ObservedValue.missing()
    if state_value == "value":
        if int_value is None:
            raise CapabilityProbeRepositoryError("VALUE favorited/disliked requires a value")
        return ObservedValue.value(bool(int_value))
    raise CapabilityProbeRepositoryError(f"unknown observation state {state_value!r}")
