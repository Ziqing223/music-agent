"""Durable capability-probe recovery-attempt domain.

A recovery attempt is an operational fact about one *recovery-driven* restore command attempt. It is
distinct from the probe's clean-path restore (which the probe row already records via its
``restore_command_outcome`` / ``restore_*`` observation fields) and from the probe lifecycle in
``capability_probe``. It answers exactly one question: has the automatic recovery restore command
been claimed / started / resolved to a known outcome?

The contract mirrors the write-execution attempt discipline (ADR 0012): a ``STARTED`` attempt is
committed *before* any external recovery restore side effect, so ``STARTED`` means "the command may
or may not have applied", never "not applied" and never "applied". A crash after ``STARTED`` leaves
an ambiguous outcome, and the strict fail-closed contract never re-issues the restore command on
that ambiguity. Recovery is a safety axis only: a recovery attempt never produces capability
verification evidence and never touches ``verification_verdict``.

The attempt identity is independent of every other identity: ``attempt_id != probe_id != intent ID
!= canonical ID != external ID``, and it is never derived from a ``target + operation`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping
from uuid import UUID, uuid4

from music_agent.capability_probe import validate_probe_id

_RECOVERY_ATTEMPT_ID_PREFIX = "rec_"


class RecoveryAttemptError(ValueError):
    code = "recovery_attempt_error"


class RecoveryAttemptValidationError(RecoveryAttemptError):
    code = "validation_error"


class RecoveryAttemptTransitionError(RecoveryAttemptError):
    code = "invalid_attempt_transition"


class RecoveryAttemptState(StrEnum):
    """One recovery-driven restore command attempt's durable lifecycle.

    ``STARTED`` is committed before any external recovery restore side effect. ``COMMAND_SUCCEEDED``
    / ``COMMAND_FAILED`` record the known outcome. There is no attempt-level readback state: that is
    the recovery orchestrator's concern, not the recovery attempt's.
    """

    STARTED = "started"
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"


class RecoveryAttemptEvent(StrEnum):
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"


def generate_recovery_attempt_id() -> str:
    """Generate a stable recovery-attempt identity.

    The ``rec_`` prefix is outside ``prb_``, ``att_``, ``int_``, every canonical ``ENTITY_ID_PREFIX``,
    and every external ID, so a recovery attempt ID can never be confused with any other identity,
    and it is never derived from a target + operation pair.
    """
    return f"{_RECOVERY_ATTEMPT_ID_PREFIX}{uuid4()}"


def validate_recovery_attempt_id(attempt_id: str) -> None:
    if not isinstance(attempt_id, str) or not attempt_id.startswith(_RECOVERY_ATTEMPT_ID_PREFIX):
        raise RecoveryAttemptValidationError(
            f"attempt_id must use the {_RECOVERY_ATTEMPT_ID_PREFIX} namespace"
        )
    suffix = attempt_id[len(_RECOVERY_ATTEMPT_ID_PREFIX) :]
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise RecoveryAttemptValidationError("attempt_id suffix must be a canonical UUID") from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise RecoveryAttemptValidationError("attempt_id suffix must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class RecoveryAttempt:
    """An operational recovery-attempt fact, not a canonical entity and not a probe lifecycle.

    Timestamps live only in the durable row (``created_at`` / ``updated_at``); the domain object
    carries identity and state only, matching ``ExecutionAttempt``.
    """

    attempt_id: str
    probe_id: str
    state: RecoveryAttemptState

    def __post_init__(self) -> None:
        validate_recovery_attempt_id(self.attempt_id)
        validate_probe_id(self.probe_id)
        if not isinstance(self.state, RecoveryAttemptState):
            raise RecoveryAttemptValidationError("state must be a RecoveryAttemptState")


def create_attempt(probe_id: str) -> RecoveryAttempt:
    """Create a STARTED recovery attempt for ``probe_id`` with a fresh independent identity."""
    validate_probe_id(probe_id)
    return RecoveryAttempt(generate_recovery_attempt_id(), probe_id, RecoveryAttemptState.STARTED)


_RECOVERY_ATTEMPT_TRANSITIONS: Mapping[
    tuple[RecoveryAttemptState, RecoveryAttemptEvent], RecoveryAttemptState
] = MappingProxyType(
    {
        (RecoveryAttemptState.STARTED, RecoveryAttemptEvent.COMMAND_SUCCEEDED):
            RecoveryAttemptState.COMMAND_SUCCEEDED,
        (RecoveryAttemptState.STARTED, RecoveryAttemptEvent.COMMAND_FAILED):
            RecoveryAttemptState.COMMAND_FAILED,
    }
)


def advance_attempt(attempt: RecoveryAttempt, event: RecoveryAttemptEvent) -> RecoveryAttempt:
    """Return the attempt advanced by ``event``, failing closed on illegal transitions.

    Only a STARTED attempt may record an outcome, and an outcome is terminal: an attempt cannot move
    from COMMAND_SUCCEEDED to COMMAND_FAILED or vice versa, and a terminal attempt can never re-enter
    STARTED.
    """
    if not isinstance(attempt, RecoveryAttempt):
        raise RecoveryAttemptValidationError("attempt must be a RecoveryAttempt")
    if not isinstance(event, RecoveryAttemptEvent):
        raise RecoveryAttemptValidationError("event must be a RecoveryAttemptEvent")
    try:
        next_state = _RECOVERY_ATTEMPT_TRANSITIONS[(attempt.state, event)]
    except KeyError as error:
        raise RecoveryAttemptTransitionError(
            f"illegal attempt transition from {attempt.state.value} on {event.value}"
        ) from error
    return replace(attempt, state=next_state)
