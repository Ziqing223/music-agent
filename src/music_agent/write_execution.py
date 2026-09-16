"""Durable write-execution attempt domain.

An execution attempt is an operational fact about a single command attempt against the external
system. It is distinct from the ``PendingIntent`` lifecycle in ``write_intent`` and from canonical
entities: the intent records the logical lifecycle (PENDING -> AWAITING_READBACK / EXECUTION_FAILED
/ OUTCOME_UNKNOWN -> CONFIRMED / READBACK_MISMATCH), while an attempt records only the durable fact
of one command attempt (STARTED -> COMMAND_SUCCEEDED / COMMAND_FAILED / COMMAND_UNKNOWN).

The attempt identity is independent of every other identity: ``attempt_id != intent_id !=
canonical ID != external ID``, and it is never derived from a ``target + operation`` pair.

An attempt records a *command* outcome, which is one of three durable facts: ``COMMAND_SUCCEEDED``
(a known success), ``COMMAND_FAILED`` (a failure that is actually provable -- the command was never
dispatched, so no side effect occurred), and ``COMMAND_UNKNOWN`` (an ambiguous outcome -- the
command was dispatched but a timeout or non-zero exit means it may or may not have applied). An
``AmbiguousCommandOutcomeError`` is the adapter-level signal for the third case, and a
``DeterministicCommandError`` is the only adapter-level signal for the second; the orchestrator
maps an ``AmbiguousCommandOutcomeError`` -- and, critically, any *other* unexpected exception -- to
``COMMAND_UNKNOWN``, never to ``COMMAND_FAILED``, because an exception is not proof of no side
effect. Only a ``DeterministicCommandError`` is positive proof of no side effect.

Nothing here persists anything or calls the external system. Intent transition legality stays in
``write_intent.advance_intent``; this module owns only the attempt's own transition map, and does
not reproduce the PendingIntent lifecycle as a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping
from uuid import UUID, uuid4

from music_agent.write_intent import validate_intent_id


_ATTEMPT_ID_PREFIX = "att_"


class WriteExecutionError(ValueError):
    code = "write_execution_error"


class WriteExecutionValidationError(WriteExecutionError):
    code = "validation_error"


class AttemptTransitionError(WriteExecutionError):
    code = "invalid_attempt_transition"


class AmbiguousCommandOutcomeError(WriteExecutionError):
    """Signal that a command was dispatched but its outcome cannot be determined.

    Raised by a command runner when the external command may or may not have applied (a subprocess
    timeout, or a non-zero exit whose side effect cannot be proven absent). It is distinct from an
    ordinary failure: an ordinary failure proves no side effect, whereas this exception proves
    nothing, so the orchestrator must record a durable ``COMMAND_UNKNOWN`` outcome rather than a
    deterministic ``COMMAND_FAILED``.
    """

    code = "ambiguous_command_outcome"


class DeterministicCommandError(WriteExecutionError):
    """Positive proof that a command was never dispatched and produced no side effect.

    This is the *only* signal the orchestrator maps to a deterministic ``COMMAND_FAILED`` (and
    therefore ``EXECUTION_FAILED``). An adapter raises it solely when it has affirmative evidence
    the external command could not have run: a pre-dispatch argument or intent validation failure,
    an unresolvable binding checked before dispatch, or a subprocess that failed to spawn
    (``OSError``). The absence of an ``AmbiguousCommandOutcomeError`` is *not* evidence of no side
    effect: any other unexpected exception is treated by the orchestrator as an unknown outcome,
    never as a deterministic failure.
    """

    code = "deterministic_command_failure"


class AttemptState(StrEnum):
    """One command attempt's durable lifecycle.

    ``STARTED`` is committed before any external side effect. ``COMMAND_SUCCEEDED`` /
    ``COMMAND_FAILED`` record a known outcome; ``COMMAND_FAILED`` is reachable only when no side
    effect is provable (the command was never dispatched). ``COMMAND_UNKNOWN`` records an ambiguous
    outcome: the command may or may not have applied. There is no attempt-level "readback" state:
    that is the PendingIntent's concern, not the command attempt's.
    """

    STARTED = "started"
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_UNKNOWN = "command_unknown"


class AttemptEvent(StrEnum):
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_UNKNOWN = "command_unknown"


def generate_attempt_id() -> str:
    """Generate a stable execution-attempt identity.

    The ``att_`` prefix is outside ``identity.ENTITY_ID_PREFIX`` and outside the intent ``int_``
    namespace, so an attempt ID can never be confused with a canonical ID or an intent ID, and it
    is never derived from a target + operation pair.
    """
    return f"{_ATTEMPT_ID_PREFIX}{uuid4()}"


def validate_attempt_id(attempt_id: str) -> None:
    if not isinstance(attempt_id, str) or not attempt_id.startswith(_ATTEMPT_ID_PREFIX):
        raise WriteExecutionValidationError(
            f"attempt_id must use the {_ATTEMPT_ID_PREFIX} namespace"
        )
    suffix = attempt_id[len(_ATTEMPT_ID_PREFIX) :]
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise WriteExecutionValidationError("attempt_id suffix must be a canonical UUID") from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise WriteExecutionValidationError("attempt_id suffix must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    """An operational execution-attempt fact, not a canonical entity and not an intent."""

    attempt_id: str
    intent_id: str
    state: AttemptState

    def __post_init__(self) -> None:
        validate_attempt_id(self.attempt_id)
        validate_intent_id(self.intent_id)
        if not isinstance(self.state, AttemptState):
            raise WriteExecutionValidationError("state must be an AttemptState")


def create_attempt(intent_id: str) -> ExecutionAttempt:
    """Create a STARTED attempt for ``intent_id`` with a fresh independent identity."""
    validate_intent_id(intent_id)
    return ExecutionAttempt(generate_attempt_id(), intent_id, AttemptState.STARTED)


_ATTEMPT_TRANSITIONS: Mapping[tuple[AttemptState, AttemptEvent], AttemptState] = MappingProxyType(
    {
        (AttemptState.STARTED, AttemptEvent.COMMAND_SUCCEEDED): AttemptState.COMMAND_SUCCEEDED,
        (AttemptState.STARTED, AttemptEvent.COMMAND_FAILED): AttemptState.COMMAND_FAILED,
        (AttemptState.STARTED, AttemptEvent.COMMAND_UNKNOWN): AttemptState.COMMAND_UNKNOWN,
    }
)


def advance_attempt(attempt: ExecutionAttempt, event: AttemptEvent) -> ExecutionAttempt:
    """Return the attempt advanced by ``event``, failing closed on illegal transitions.

    Only a STARTED attempt may record an outcome, and an outcome is terminal: an attempt cannot
    move from one outcome to another (including from COMMAND_SUCCEEDED / COMMAND_FAILED /
    COMMAND_UNKNOWN back to STARTED).
    """
    if not isinstance(attempt, ExecutionAttempt):
        raise WriteExecutionValidationError("attempt must be an ExecutionAttempt")
    if not isinstance(event, AttemptEvent):
        raise WriteExecutionValidationError("event must be an AttemptEvent")
    try:
        next_state = _ATTEMPT_TRANSITIONS[(attempt.state, event)]
    except KeyError as error:
        raise AttemptTransitionError(
            f"illegal attempt transition from {attempt.state.value} on {event.value}"
        ) from error
    return replace(attempt, state=next_state)
