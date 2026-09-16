"""Capability-probe clean-path orchestration against an injected fake boundary.

A capability probe proves, *before* any real Music.app mutation, that a single reversible
``set_favorited`` forward/restore cycle actually mutates the source and can be restored. This module
sequences that cycle across durable ``*_STARTED`` checkpoints:

    BASELINE_CAPTURED
    -> durable FORWARD_STARTED -> forward command -> forward readback -> FORWARD_OBSERVED
    -> durable RESTORE_STARTED -> restore command -> restore readback -> RESTORE_OBSERVED
    -> finalize -> VERIFIED + RESTORED

The orchestrator owns sequencing only. Transition legality stays in ``capability_probe`` (the
domain), durable facts stay in ``capability_probe_repository``, and the external side effect goes
through the injected ``CapabilityProbeAdapter`` -- never Music.app or ``osascript`` directly.

The one hard ordering rule it enforces: every external side effect is preceded by a durable
``*_STARTED`` CAS. The forward command runs only after ``FORWARD_STARTED`` is committed via
``update_probe``, and the restore command runs only after ``RESTORE_STARTED`` is committed. If
either CAS fails as stale, the corresponding command is never invoked.

This block covers the deterministic clean path, its readback-confirmed failures (readback
mismatch, cross-field side effect), command-outcome ambiguity, and crash / ambiguous recovery. A
probe not at
``BASELINE_CAPTURED`` fails closed on the clean path; a probe not at an ambiguous restart-safe state
(``FORWARD_STARTED`` / ``RESTORE_STARTED`` with a ``PENDING`` or ``INCONCLUSIVE`` verdict) fails
closed on recovery. Recovery restores the user's source state but keeps the verdict ``INCONCLUSIVE``:
recovery safety and verification evidence are separate axes, and a recovery can never produce
``VERIFIED``. A ``VERIFIED`` probe's verdict is evidence about that one probe record only; it never
mutates the write capability matrix.

The recovery-driven restore command is guarded by two durable layers committed strictly in order
before the side effect: the probe's ``RESTORE_STARTED`` CAS, then a ``STARTED``
``RecoveryAttempt``. At most one automatic attempt exists per probe (``UNIQUE(probe_id)``), an
existing attempt of any state never re-issues the restore command, and a ``STARTED`` attempt left
by a crash is ambiguous rather than retried. The attempt records command outcome only; the probe's
``recovery_status`` is always resolved from a live source readback, never from the attempt state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryAction,
    RecoveryStatus,
    VerificationVerdict,
    decide_recovery,
    finalize,
    is_recoverable,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_restore_started,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.source_observation import ObservedValue, ObservationState


class CapabilityProbeOrchestrationError(ValueError):
    code = "capability_probe_orchestration_error"


class ProbeNotFoundError(CapabilityProbeOrchestrationError):
    code = "probe_not_found"


class ProbeNotStartedError(CapabilityProbeOrchestrationError):
    code = "probe_not_started"


class UnrecoverableProbeError(CapabilityProbeOrchestrationError):
    code = "unrecoverable_probe"


@dataclass(frozen=True, slots=True)
class ProbeReadback:
    """favorited + disliked readback of the probe's target Track.

    Both fields are non-nullable in the source, so a ``NULL`` observation is rejected. ``MISSING``
    means the readback could not observe the field (unavailable); a ``VALUE`` carries the observed
    boolean.
    """

    favorited: ObservedValue
    disliked: ObservedValue

    def __post_init__(self) -> None:
        _require_bool_observation(self.favorited, "favorited")
        _require_bool_observation(self.disliked, "disliked")


class CapabilityProbeAdapter(Protocol):
    """Injected command + readback boundary for one ``set_favorited`` probe.

    Replaced by a fake in tests; a production implementation must never call Music.app here.
    ``command`` applies the requested favorited value to the target Track and returns ``None`` on a
    clean run or raises on an exception, which the orchestrator maps to an ambiguous ``UNKNOWN``.
    ``readback`` returns the target's current favorited + disliked observations.
    """

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None: ...

    def readback(self, target_persistent_id: str) -> ProbeReadback: ...


class CapabilityProbeOrchestrator:
    """Sequence a clean forward/restore probe cycle across durable CAS boundaries."""

    def __init__(
        self,
        repository: CapabilityProbeRepository,
        recovery_attempt_repository: CapabilityProbeRecoveryAttemptRepository,
        adapter: CapabilityProbeAdapter,
    ) -> None:
        self.repository = repository
        self.recovery_attempt_repository = recovery_attempt_repository
        self.adapter = adapter

    def run_probe(self, probe_id: str) -> CapabilityProbe:
        """Run the clean path for one durably-baselined probe, returning its final durable state.

        The probe must already exist durably at ``BASELINE_CAPTURED``. It is advanced forward,
        read back, restored, read back, and finalized; each lifecycle change is committed via
        ``update_probe`` under its optimistic expected-state guard. A command exception is an
        ambiguous outcome (it does not prove the side effect did not happen) and terminates with
        ``INCONCLUSIVE``, never ``FAILED``; a readback mismatch or cross-field side effect
        terminates with ``FAILED``; an unavailable readback terminates with ``INCONCLUSIVE``. Both
        are durably recorded.
        """
        probe = self._require_probe(probe_id)
        forward = self._run_forward(probe)
        return self._finish_cycle(forward)

    def run_started_probe(self, probe_id: str) -> CapabilityProbe:
        """Run the clean cycle for a probe already captured at ``FORWARD_STARTED``.

        This is the E2-B forward-dispatch continuation: the probe's durable ``FORWARD_STARTED``
        checkpoint is already committed, so the forward command may now be dispatched. The forward
        command, readback, restore, and finalize are identical to ``run_probe``; only the initial
        ``mark_forward_started`` is skipped because the capture already recorded it. The probe must
        still be ``PENDING`` -- a ``FORWARD_STARTED`` probe whose verdict is no longer pending is a
        recovery case and must never re-issue the forward command.

        Coordination boundary (single writer, not fenced): the forward command is external I/O
        issued *before* the end-of-step ``update_probe`` CAS, so that CAS protects the durable
        transition -- a stale caller cannot overwrite a later durable state -- but it cannot fence
        the command itself. A concurrent ``recover_probe`` may claim the probe
        (``PENDING -> INCONCLUSIVE``) and issue a restore command after this method reads the probe
        but before its forward command runs. Concurrent takeover is therefore NOT safe and is not
        claimed to be: recovery may only take over a probe after the original forward execution
        owner is no longer active. See ADR 0019.
        """
        probe = self._require_probe(probe_id)
        if probe.step_state is not ProbeStepState.FORWARD_STARTED:
            raise ProbeNotStartedError(
                f"probe {probe_id!r} is {probe.step_state.value!r}; "
                "run_started_probe requires FORWARD_STARTED"
            )
        if probe.verification_verdict is not VerificationVerdict.PENDING:
            raise ProbeNotStartedError(
                f"probe {probe_id!r} has verdict {probe.verification_verdict.value!r}; "
                "forward dispatch requires PENDING"
            )
        forward = self._run_forward_from_started(probe)
        return self._finish_cycle(forward)

    def _finish_cycle(self, forward: CapabilityProbe) -> CapabilityProbe:
        if forward.step_state is not ProbeStepState.FORWARD_OBSERVED:
            return forward
        restore = self._run_restore(forward)
        if restore.step_state is not ProbeStepState.RESTORE_OBSERVED:
            return restore
        verified = finalize(restore)
        return self.repository.update_probe(restore, verified)

    def recover_probe(self, probe_id: str) -> CapabilityProbe:
        """Recover a durably-crashed probe from an ambiguous restart-safe state.

        ``FORWARD_STARTED`` / ``RESTORE_STARTED`` with a ``PENDING`` or ``INCONCLUSIVE`` verdict are
        restart-safe ambiguity: the corresponding ``*_STARTED`` checkpoint committed, but the
        command outcome was not reliably persisted before the crash, so the source may or may not
        have changed. A ``PENDING`` probe is durably marked ``INCONCLUSIVE`` (the CAS claim that
        serializes competing ``PENDING`` callers); an already-``INCONCLUSIVE`` probe is an
        interrupted prior recovery and is resumed without a verdict change.

        The recovery restore side effect is guarded by a durable ``RecoveryAttempt``: an existing
        attempt of *any* state (``STARTED`` / ``COMMAND_SUCCEEDED`` / ``COMMAND_FAILED``) never
        re-issues the restore command, and at most one automatic attempt can exist per probe. The
        verdict can never leave ``INCONCLUSIVE``: recovery is safety, not verification evidence.
        """
        probe = self._require_probe(probe_id)
        self._require_recoverable(probe)
        if probe.verification_verdict is VerificationVerdict.PENDING:
            # Durable INCONCLUSIVE is the recovery claim; it also prevents two callers from both
            # proceeding past this point on the same stale PENDING view.
            probe = self.repository.update_probe(probe, mark_inconclusive(probe))
        # An existing attempt (STARTED / SUCCEEDED / FAILED) means the restore command may already
        # have run, so it is never re-issued; only the recovery status is reconciled from source.
        if self.recovery_attempt_repository.get_for_probe(probe_id) is not None:
            return self._reconcile_recovery_status(probe)
        current_favorited = self._readback(probe)[0]
        action = decide_recovery(probe, current_favorited)
        if action is RecoveryAction.NO_RESTORE:
            status = (
                RecoveryStatus.RESTORED
                if probe.step_state is ProbeStepState.RESTORE_STARTED
                else RecoveryStatus.BASELINE_CONFIRMED
            )
            return self.repository.update_probe(probe, mark_recovery_status(probe, status))
        if action is RecoveryAction.MANUAL_CHECK:
            return self.repository.update_probe(
                probe, mark_recovery_status(probe, RecoveryStatus.NEEDS_MANUAL_CHECK)
            )
        return self._recover_restore(probe)

    def _run_forward(self, probe: CapabilityProbe) -> CapabilityProbe:
        started = mark_forward_started(probe)
        # Durable FORWARD_STARTED must commit before the forward side effect.
        self.repository.update_probe(probe, started)
        return self._run_forward_from_started(started)

    def _run_forward_from_started(self, probe: CapabilityProbe) -> CapabilityProbe:
        """Dispatch the forward command + readback from an already-durable FORWARD_STARTED probe."""
        outcome = self._run_command(probe, not probe.baseline_favorited)
        favorited, disliked = self._readback(probe)
        observed = observe_forward(probe, outcome, favorited, disliked)
        return self.repository.update_probe(probe, observed)

    def _run_restore(self, probe: CapabilityProbe) -> CapabilityProbe:
        started = mark_restore_started(probe)
        # Durable RESTORE_STARTED must commit before the restore side effect.
        self.repository.update_probe(probe, started)
        outcome = self._run_command(probe, probe.baseline_favorited)
        favorited, disliked = self._readback(probe)
        observed = observe_restore(started, outcome, favorited, disliked)
        return self.repository.update_probe(started, observed)

    def _recover_restore(self, probe: CapabilityProbe) -> CapabilityProbe:
        # ``probe`` is durably INCONCLUSIVE at FORWARD_STARTED or RESTORE_STARTED with no prior
        # attempt. The restore side effect is guarded by two durable layers committed strictly in
        # order before the command: (1) RESTORE_STARTED and (2) a STARTED RecoveryAttempt. From
        # FORWARD_STARTED the first layer is a CAS transition; from RESTORE_STARTED it is already
        # satisfied. A competing caller fails on the CAS or on the attempt's UNIQUE(probe_id)
        # constraint, either way before any command is issued.
        if probe.step_state is not ProbeStepState.RESTORE_STARTED:
            restore_started = mark_recovery_restore_started(probe)
            probe = self.repository.update_probe(probe, restore_started)
        attempt = self.recovery_attempt_repository.begin_attempt(probe.probe_id)
        outcome = self._run_recovery_command(probe, probe.baseline_favorited)
        if outcome is CommandOutcome.SUCCESS:
            self.recovery_attempt_repository.mark_command_succeeded(attempt.attempt_id)
        elif outcome is CommandOutcome.FAILED:
            self.recovery_attempt_repository.mark_command_failed(attempt.attempt_id)
        # A UNKNOWN outcome leaves the attempt STARTED (the external outcome is unprovable); it is
        # never forged into SUCCEEDED or FAILED. Recovery status is always resolved from the source
        # readback, never from the attempt state.
        return self._reconcile_recovery_status(probe)

    def _reconcile_recovery_status(self, probe: CapabilityProbe) -> CapabilityProbe:
        """Read the current source and mark RESTORED (== baseline) or NEEDS_MANUAL_CHECK.

        The source readback is the safety authority: a ``COMMAND_SUCCEEDED`` attempt is never
        trusted to imply the source is restored, and a ``COMMAND_FAILED`` attempt is never trusted
        to imply it is not. Only the live observation resolves ``recovery_status``, and it never
        touches ``verification_verdict``.
        """
        current_favorited = self._readback(probe)[0]
        if (
            current_favorited.state is ObservationState.VALUE
            and current_favorited.payload == probe.baseline_favorited
        ):
            status = RecoveryStatus.RESTORED
        else:
            status = RecoveryStatus.NEEDS_MANUAL_CHECK
        return self.repository.update_probe(probe, mark_recovery_status(probe, status))

    def _run_recovery_command(
        self, probe: CapabilityProbe, requested_favorited: bool
    ) -> CommandOutcome:
        """Run one recovery restore command, treating every exception as an ambiguous outcome.

        No exception proves the restore side effect did not happen: ``AppleMusicWriteError`` is
        also raised for a subprocess timeout, after which the command may or may not have applied.
        Any exception therefore maps to ``UNKNOWN`` and leaves the attempt ``STARTED``, never to a
        forged ``COMMAND_FAILED``. A clean return is ``SUCCESS``.
        """
        try:
            self.adapter.command(probe.target_persistent_id, requested_favorited)
        except Exception:
            return CommandOutcome.UNKNOWN
        return CommandOutcome.SUCCESS

    def _require_recoverable(self, probe: CapabilityProbe) -> None:
        if not is_recoverable(probe):
            raise UnrecoverableProbeError(
                f"probe {probe.probe_id!r} is not a recoverable ambiguous state "
                f"(step_state={probe.step_state.value}, "
                f"verdict={probe.verification_verdict.value})"
            )

    def _run_command(
        self, probe: CapabilityProbe, requested_favorited: bool
    ) -> CommandOutcome:
        """Run one clean-path command, treating every exception as an ambiguous outcome.

        A command exception does not prove the side effect did not happen (a subprocess timeout,
        for example, may leave the write applied), so it maps to ``UNKNOWN`` and the domain records
        ``INCONCLUSIVE`` rather than a deterministic ``FAILED``. Recovery is then the only path
        forward, reconciling the source by readback. A clean return is ``SUCCESS``.
        """
        try:
            self.adapter.command(probe.target_persistent_id, requested_favorited)
        except Exception:
            return CommandOutcome.UNKNOWN
        return CommandOutcome.SUCCESS

    def _readback(self, probe: CapabilityProbe) -> tuple[ObservedValue, ObservedValue]:
        try:
            result = self.adapter.readback(probe.target_persistent_id)
        except Exception:
            # An unavailable readback fails closed to MISSING, which the domain maps to
            # INCONCLUSIVE for a known-success command (and leaves a known failure FAILED).
            return ObservedValue.missing(), ObservedValue.missing()
        return result.favorited, result.disliked

    def _require_probe(self, probe_id: str) -> CapabilityProbe:
        probe = self.repository.get_probe(probe_id)
        if probe is None:
            raise ProbeNotFoundError(f"probe {probe_id!r} does not exist")
        return probe


def _require_bool_observation(value: object, field: str) -> ObservedValue:
    if not isinstance(value, ObservedValue):
        raise CapabilityProbeOrchestrationError(f"{field} must be an ObservedValue")
    if value.state is ObservationState.NULL:
        raise CapabilityProbeOrchestrationError(
            f"{field} cannot be NULL (favorited/disliked are non-nullable)"
        )
    if value.state is ObservationState.VALUE and not isinstance(value.payload, bool):
        raise CapabilityProbeOrchestrationError(f"{field} VALUE must be a strict bool")
    return value
