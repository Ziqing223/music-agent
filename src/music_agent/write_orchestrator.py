"""Write execution and readback orchestration.

The orchestrator owns command + readback sequencing around a durable execution attempt. It does
not own durable facts (the repository does) and it does not own transition legality (the domain
does). It enforces the execution policy and the crash-safety ordering:

1. before any external command, a STARTED attempt is durably committed via the repository's
   ``begin_execution``, and the execution policy is checked;
2. the external command is invoked only after leaving that transaction;
3. the command outcome is persisted in the same transaction that advances the intent. A command
   exception is not automatically a known failure: an ``AmbiguousCommandOutcomeError`` (a timeout,
   or a non-zero exit whose side effect cannot be proven absent) is recorded as COMMAND_UNKNOWN /
   OUTCOME_UNKNOWN, never as COMMAND_FAILED / EXECUTION_FAILED, and any other unexpected exception
   fails closed the same way. Only a ``DeterministicCommandError`` -- positive proof the command
   never dispatched -- is recorded as COMMAND_FAILED / EXECUTION_FAILED;
4. readback runs only for an AWAITING_READBACK intent whose durable command outcome is
   COMMAND_SUCCEEDED, and never re-runs the command;
5. an ambiguous STARTED attempt (a crash between STARTED and outcome persistence) fails closed:
   it is reported and never automatically re-executed;
6. an OUTCOME_UNKNOWN intent is reconciled by readback only: a matching readback confirms
   CONFIRMED, an unavailable or mismatching readback fails closed and leaves it OUTCOME_UNKNOWN.
   Reconciliation never re-runs the command.

The production capability matrix marks ``SET_FAVORITED`` as ``adapter_implemented = True`` /
``readback_implemented = True`` (the one implemented scalar write), but it remains
``capability_verified = False``, so the execution-ready set is still empty and the default policy
rejects every execution. Tests inject a permissive policy and a fake adapter; production capability
truth is not modified.
"""

from __future__ import annotations

from typing import Callable, Protocol

from music_agent.source_observation import ObservedValue
from music_agent.write_execution import (
    AmbiguousCommandOutcomeError,
    AttemptState,
    DeterministicCommandError,
)
from music_agent.write_execution_repository import (
    AmbiguousAttemptError,
    ReadbackGateError,
    WriteExecutionRepository,
)
from music_agent.write_intent import (
    IntentState,
    PendingIntent,
    ReadbackDecision,
    WriteEvent,
    WriteTransitionError,
    evaluate_readback,
    is_execution_ready,
    resolve_capability,
)


class WriteCommandAdapter(Protocol):
    """Minimal external write + readback boundary, replaced by a fake in tests."""

    def command(self, intent: PendingIntent) -> None: ...

    def readback(self, intent: PendingIntent) -> ObservedValue: ...


ExecutionPolicy = Callable[[PendingIntent], bool]


def production_execution_policy(intent: PendingIntent) -> bool:
    """The real gate: production capability matrix stays not execution-ready."""
    return is_execution_ready(resolve_capability(intent.operation))


class WriteOrchestrationError(ValueError):
    code = "write_orchestration_error"


class NotExecutionReadyError(WriteOrchestrationError):
    code = "not_execution_ready"


class IntentNotFoundError(WriteOrchestrationError):
    code = "intent_not_found"


class WriteOrchestrator:
    """Sequence command and readback across durable attempt boundaries."""

    def __init__(
        self,
        repository: WriteExecutionRepository,
        adapter: WriteCommandAdapter,
        *,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.adapter = adapter
        self.policy = production_execution_policy if policy is None else policy

    def execute_pending_intent(self, intent_id: str) -> PendingIntent:
        """Policy-gate, durably START, run the command, and record its outcome.

        Returns the intent in its post-execution state (AWAITING_READBACK on known success, or
        OUTCOME_UNKNOWN on an ambiguous or otherwise unknown outcome). Only a
        ``DeterministicCommandError`` -- positive proof the command never dispatched -- is recorded
        as COMMAND_FAILED / EXECUTION_FAILED, and in that case the original exception is re-raised
        so a command failure is never mistaken for success. An ``AmbiguousCommandOutcomeError`` is
        recorded as OUTCOME_UNKNOWN, never as a deterministic failure, and never re-runs the
        command. Any *other* unexpected exception is likewise fail-closed to OUTCOME_UNKNOWN,
        because an exception that is not explicit proof of no side effect must not be recorded as a
        deterministic failure.
        """
        intent = self._require_intent(intent_id)
        if intent.state is not IntentState.PENDING:
            raise WriteTransitionError(
                f"intent {intent_id!r} is {intent.state.value}; only PENDING intents execute"
            )
        if not self.policy(intent):
            raise NotExecutionReadyError(
                f"operation {intent.operation.value} is not execution-ready"
            )
        attempt = self.repository.begin_execution(intent_id)
        try:
            self.adapter.command(intent)
        except AmbiguousCommandOutcomeError:
            self.repository.record_command_unknown(intent_id, attempt.attempt_id)
        except DeterministicCommandError:
            self.repository.record_command_failure(intent_id, attempt.attempt_id)
            raise
        except Exception:
            self.repository.record_command_unknown(intent_id, attempt.attempt_id)
        else:
            self.repository.record_command_success(intent_id, attempt.attempt_id)
        return self._require_intent(intent_id)

    def resume_readback(self, intent_id: str) -> PendingIntent:
        """Read back an AWAITING_READBACK intent, never re-running the command.

        A matching readback confirms; a mismatch lands on READBACK_MISMATCH. A readback that is
        unavailable or raises leaves the intent AWAITING_READBACK for a later attempt, and never
        triggers another command.
        """
        intent = self._require_intent(intent_id)
        if intent.state is not IntentState.AWAITING_READBACK:
            raise WriteTransitionError(
                f"intent {intent_id!r} is {intent.state.value}; "
                "only AWAITING_READBACK intents can be read back"
            )
        self._require_command_succeeded(intent_id)
        try:
            observed = self.adapter.readback(intent)
        except Exception:
            return intent
        decision = evaluate_readback(intent.requested_value, observed)
        if decision is ReadbackDecision.MATCHED:
            return self.repository.record_readback(intent_id, WriteEvent.READBACK_MATCHED)
        if decision is ReadbackDecision.MISMATCHED:
            return self.repository.record_readback(intent_id, WriteEvent.READBACK_MISMATCHED)
        return intent

    def reconcile_unknown_outcome(self, intent_id: str) -> PendingIntent:
        """Reconcile an OUTCOME_UNKNOWN intent by readback, never re-running the command.

        A matching readback confirms CONFIRMED (desired-state confirmation). An unavailable or
        mismatching readback, or a readback that raises, leaves the intent OUTCOME_UNKNOWN for a
        later attempt or manual recovery: the outcome is neither forged into success nor into
        failure.
        """
        intent = self._require_intent(intent_id)
        if intent.state is not IntentState.OUTCOME_UNKNOWN:
            raise WriteTransitionError(
                f"intent {intent_id!r} is {intent.state.value}; "
                "only OUTCOME_UNKNOWN intents can be reconciled"
            )
        self._require_command_unknown(intent_id)
        try:
            observed = self.adapter.readback(intent)
        except Exception:
            return intent
        decision = evaluate_readback(intent.requested_value, observed)
        if decision is ReadbackDecision.MATCHED:
            return self.repository.record_reconciliation(intent_id)
        return intent

    def resume(self, intent_id: str) -> PendingIntent:
        """Restart-safe dispatch.

        AWAITING_READBACK resumes readback only; OUTCOME_UNKNOWN reconciles by readback only.
        A PENDING intent with a STARTED attempt is ambiguous and fails closed without re-executing.
        Terminal states are returned unchanged. No path re-runs the command after an outcome is
        recorded.
        """
        intent = self._require_intent(intent_id)
        if intent.state is IntentState.AWAITING_READBACK:
            return self.resume_readback(intent_id)
        if intent.state is IntentState.OUTCOME_UNKNOWN:
            return self.reconcile_unknown_outcome(intent_id)
        if intent.state is IntentState.PENDING:
            latest = self.repository.get_latest_attempt(intent_id)
            if latest is not None and latest.state is AttemptState.STARTED:
                raise AmbiguousAttemptError(
                    f"intent {intent_id!r} has a STARTED attempt with unknown outcome; "
                    "refusing to re-execute"
                )
            return self.execute_pending_intent(intent_id)
        return intent

    def _require_intent(self, intent_id: str) -> PendingIntent:
        intent = self.repository.get_intent(intent_id)
        if intent is None:
            raise IntentNotFoundError(f"intent {intent_id!r} does not exist")
        return intent

    def _require_command_succeeded(self, intent_id: str) -> None:
        latest = self.repository.get_latest_attempt(intent_id)
        if latest is None or latest.state is not AttemptState.COMMAND_SUCCEEDED:
            raise ReadbackGateError(
                f"intent {intent_id!r} has no durable COMMAND_SUCCEEDED attempt; "
                "readback is not allowed"
            )

    def _require_command_unknown(self, intent_id: str) -> None:
        latest = self.repository.get_latest_attempt(intent_id)
        if latest is None or latest.state is not AttemptState.COMMAND_UNKNOWN:
            raise ReadbackGateError(
                f"intent {intent_id!r} has no durable COMMAND_UNKNOWN attempt; "
                "reconciliation is not allowed"
            )
