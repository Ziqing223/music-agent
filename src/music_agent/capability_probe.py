"""Pure domain contract for a ``set_favorited`` live capability probe.

A capability probe is an operational object that records, *before* any real Music.app mutation,
the baseline ``favorited`` / ``disliked`` state of one durable persistent-ID-bound Track and then
drives a minimal reversible write/readback cycle against the real source:

    BASELINE_CAPTURED -> FORWARD_STARTED -> FORWARD_OBSERVED
                      -> RESTORE_STARTED -> RESTORE_OBSERVED -> (finalize -> VERIFIED)

The domain keeps three orthogonal axes strictly separate (P03.12-D-A):

- ``ProbeStepState`` -- durable *position* in the forward/restore cycle. ``*_STARTED`` states are
  committed before the external side effect (mirroring the execution-attempt discipline in
  ``write_execution``), so a STARTED state means "the command may or may not have applied", never
  "not applied" and never "applied".

- ``RecoveryStatus`` -- recovery *safety*: whether the real source has been confirmed back at its
  original baseline. This is orthogonal to evidence and is never used to grant a verification
  verdict.

- ``VerificationVerdict`` -- verification *evidence* quality. ``VERIFIED`` is reachable only
  through a clean, uninterrupted, command-outcome-known, readback-confirmed forward + restore
  cycle. Any crash, timeout, or unknown command outcome makes the probe ``INCONCLUSIVE``, and
  recovery can change ``recovery_status`` but can never upgrade ``INCONCLUSIVE`` / ``FAILED`` to
  ``VERIFIED``.

The two hard invariants of this module:

1. ``VERIFIED`` is reachable only from a clean ``RESTORE_OBSERVED`` step state while the verdict is
   still ``PENDING`` (``finalize``).
2. Recovery is a pure safety axis: ``decide_recovery`` only decides an action and never produces a
   verdict, and ``mark_recovery_status`` only mutates ``recovery_status``, never
   ``verification_verdict``. An ``INCONCLUSIVE`` or ``FAILED`` verdict is terminal and can never
   become ``VERIFIED``.

Nothing here persists to SQLite, calls Music.app or ``osascript``, mutates the canonical model, or
mutates the write capability matrix. It is pure domain logic exercised against in-memory values in
tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import UUID, uuid4

from music_agent.identity import EntityType, validate_canonical_id
from music_agent.source_observation import ObservedValue, ObservationState
from music_agent.write_intent import WriteOperation

_PROBE_ID_PREFIX = "prb_"


class CapabilityProbeError(ValueError):
    code = "capability_probe_error"


class CapabilityProbeValidationError(CapabilityProbeError):
    code = "validation_error"


class ProbeTransitionError(CapabilityProbeError):
    code = "invalid_probe_transition"


class ProbeStepState(StrEnum):
    """Durable position in the forward/restore cycle.

    ``*_STARTED`` states are committed before the external side effect, so they mean "the command
    may or may not have applied", never "not applied" and never "applied". ``VERIFIED`` / ``FAILED``
    / ``INCONCLUSIVE`` are *verdicts*, not step states, and live on ``VerificationVerdict``.
    """

    BASELINE_CAPTURED = "baseline_captured"
    FORWARD_STARTED = "forward_started"
    FORWARD_OBSERVED = "forward_observed"
    RESTORE_STARTED = "restore_started"
    RESTORE_OBSERVED = "restore_observed"


class RecoveryStatus(StrEnum):
    """Recovery safety: has the real source been confirmed back at its original baseline?"""

    BASELINE_CONFIRMED = "baseline_confirmed"
    RESTORED = "restored"
    NEEDS_MANUAL_CHECK = "needs_manual_check"


class VerificationVerdict(StrEnum):
    """Verification evidence quality.

    ``PENDING`` means an active, normally-progressing probe; it is distinct from ``INCONCLUSIVE``,
    which is terminal and means the evidence is ambiguous (crash, timeout, unknown outcome).
    """

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class CommandOutcome(StrEnum):
    """The known outcome of one forward or restore command.

    ``UNKNOWN`` is for a timeout or a crash before the outcome could be recorded; it is ambiguity,
    not a definitive failure.
    """

    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RecoveryAction(StrEnum):
    """What recovery should do next, derived from baseline vs the current real observation."""

    NO_RESTORE = "no_restore"
    RESTORE = "restore"
    MANUAL_CHECK = "manual_check"


def generate_probe_id() -> str:
    """Generate a stable capability-probe identity.

    The ``prb_`` prefix is outside ``identity.ENTITY_ID_PREFIX`` and outside the intent ``int_`` and
    attempt ``att_`` namespaces, so a probe ID can never be confused with a canonical, intent, or
    attempt ID, and it is never derived from a target + operation pair.
    """
    return f"{_PROBE_ID_PREFIX}{uuid4()}"


def validate_probe_id(probe_id: str) -> None:
    if not isinstance(probe_id, str) or not probe_id.startswith(_PROBE_ID_PREFIX):
        raise CapabilityProbeValidationError(
            f"probe_id must use the {_PROBE_ID_PREFIX} namespace"
        )
    suffix = probe_id[len(_PROBE_ID_PREFIX) :]
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise CapabilityProbeValidationError("probe_id suffix must be a canonical UUID") from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise CapabilityProbeValidationError("probe_id suffix must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    """One capability probe's durable domain state, not a canonical entity and not an intent.

    The first eight fields are identity + baseline + the three orthogonal axes. The six trailing
    fields record the forward/restore command outcome and readback observations; each is ``None``
    until that step is observed, and ``None`` means "not yet observed" -- an *unavailable* readback
    is a real ``ObservedValue.missing()``, never ``None``, so false / missing are never conflated.
    """

    probe_id: str
    operation: WriteOperation
    target_canonical_id: str
    target_persistent_id: str
    baseline_favorited: bool
    baseline_disliked: bool
    step_state: ProbeStepState
    recovery_status: RecoveryStatus
    verification_verdict: VerificationVerdict
    forward_command_outcome: CommandOutcome | None = None
    forward_favorited: ObservedValue | None = None
    forward_disliked: ObservedValue | None = None
    restore_command_outcome: CommandOutcome | None = None
    restore_favorited: ObservedValue | None = None
    restore_disliked: ObservedValue | None = None

    def __post_init__(self) -> None:
        validate_probe_id(self.probe_id)
        if self.operation is not WriteOperation.SET_FAVORITED:
            raise CapabilityProbeValidationError(
                f"capability probe supports only set_favorited, got {self.operation.value}"
            )
        validate_canonical_id(EntityType.TRACK, self.target_canonical_id)
        _require_persistent_id(self.target_persistent_id)
        _require_bool(self.baseline_favorited, "baseline_favorited")
        _require_bool(self.baseline_disliked, "baseline_disliked")
        _require_enum(self.step_state, ProbeStepState, "step_state")
        _require_enum(self.recovery_status, RecoveryStatus, "recovery_status")
        _require_enum(self.verification_verdict, VerificationVerdict, "verification_verdict")
        if self.forward_command_outcome is not None:
            _require_enum(self.forward_command_outcome, CommandOutcome, "forward_command_outcome")
        if self.restore_command_outcome is not None:
            _require_enum(self.restore_command_outcome, CommandOutcome, "restore_command_outcome")
        if self.forward_favorited is not None:
            _require_bool_observation(self.forward_favorited, "forward_favorited")
        if self.forward_disliked is not None:
            _require_bool_observation(self.forward_disliked, "forward_disliked")
        if self.restore_favorited is not None:
            _require_bool_observation(self.restore_favorited, "restore_favorited")
        if self.restore_disliked is not None:
            _require_bool_observation(self.restore_disliked, "restore_disliked")


def create_probe(
    target_canonical_id: str,
    target_persistent_id: str,
    baseline_favorited: bool,
    baseline_disliked: bool,
) -> CapabilityProbe:
    """Create a fresh probe capturing the baseline, in its initial active state.

    The target Track must be a valid Track canonical ID and a non-empty captured Apple Music
    persistent ID. ``baseline_favorited`` / ``baseline_disliked`` are strict booleans: ``False`` is
    a real value and is preserved as such.
    """
    return CapabilityProbe(
        probe_id=generate_probe_id(),
        operation=WriteOperation.SET_FAVORITED,
        target_canonical_id=target_canonical_id,
        target_persistent_id=target_persistent_id,
        baseline_favorited=baseline_favorited,
        baseline_disliked=baseline_disliked,
        step_state=ProbeStepState.BASELINE_CAPTURED,
        recovery_status=RecoveryStatus.BASELINE_CONFIRMED,
        verification_verdict=VerificationVerdict.PENDING,
    )


def create_forward_started_probe(
    target_canonical_id: str,
    target_persistent_id: str,
    baseline_favorited: bool,
    baseline_disliked: bool,
) -> CapabilityProbe:
    """Create a probe captured directly at ``FORWARD_STARTED`` for the live execution boundary.

    E2-B must never leave a durable ``BASELINE_CAPTURED``-only window: a stranded
    ``BASELINE_CAPTURED`` probe is unrecoverable and blocks re-probing the same target. This factory
    reuses ``create_probe`` + ``mark_forward_started`` so identity and baseline validation are
    identical, but the probe is *born* at ``FORWARD_STARTED`` + ``PENDING`` + ``BASELINE_CONFIRMED``,
    meaning "the forward command may now be dispatched".
    """
    return mark_forward_started(
        create_probe(
            target_canonical_id,
            target_persistent_id,
            baseline_favorited,
            baseline_disliked,
        )
    )


def mark_forward_started(probe: CapabilityProbe) -> CapabilityProbe:
    """Commit the forward-command STARTED position before the external side effect."""
    probe = _require_active_probe(probe)
    if probe.step_state is not ProbeStepState.BASELINE_CAPTURED:
        raise ProbeTransitionError(
            f"mark_forward_started requires BASELINE_CAPTURED, got {probe.step_state.value}"
        )
    return replace(probe, step_state=ProbeStepState.FORWARD_STARTED)


def observe_forward(
    probe: CapabilityProbe,
    command_outcome: CommandOutcome,
    favorited: ObservedValue,
    disliked: ObservedValue,
) -> CapabilityProbe:
    """Record the forward command outcome and readback, advancing or terminating the probe.

    A clean forward requires a ``SUCCESS`` outcome, a readback of ``favorited == !baseline`` (the
    write actually took effect), and ``disliked == baseline`` (no cross-field side effect). Any
    ``UNKNOWN`` outcome or missing readback is ``INCONCLUSIVE``; a definitive failure, a no-op
    write, or an observed side effect is ``FAILED``.
    """
    probe = _require_active_probe(probe)
    if probe.step_state is not ProbeStepState.FORWARD_STARTED:
        raise ProbeTransitionError(
            f"observe_forward requires FORWARD_STARTED, got {probe.step_state.value}"
        )
    command_outcome = _require_enum(command_outcome, CommandOutcome, "command_outcome")
    favorited = _require_bool_observation(favorited, "forward_favorited")
    disliked = _require_bool_observation(disliked, "forward_disliked")
    probe = replace(
        probe,
        forward_command_outcome=command_outcome,
        forward_favorited=favorited,
        forward_disliked=disliked,
    )
    if command_outcome is CommandOutcome.UNKNOWN:
        return replace(probe, verification_verdict=VerificationVerdict.INCONCLUSIVE)
    if command_outcome is CommandOutcome.FAILED:
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    if favorited.state is ObservationState.MISSING or disliked.state is ObservationState.MISSING:
        return replace(probe, verification_verdict=VerificationVerdict.INCONCLUSIVE)
    if favorited.payload != (not probe.baseline_favorited):
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    if disliked.payload != probe.baseline_disliked:
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    return replace(probe, step_state=ProbeStepState.FORWARD_OBSERVED)


def mark_restore_started(probe: CapabilityProbe) -> CapabilityProbe:
    """Commit the restore-command STARTED position before the external side effect."""
    probe = _require_active_probe(probe)
    if probe.step_state is not ProbeStepState.FORWARD_OBSERVED:
        raise ProbeTransitionError(
            f"mark_restore_started requires FORWARD_OBSERVED, got {probe.step_state.value}"
        )
    return replace(probe, step_state=ProbeStepState.RESTORE_STARTED)


def observe_restore(
    probe: CapabilityProbe,
    command_outcome: CommandOutcome,
    favorited: ObservedValue,
    disliked: ObservedValue,
) -> CapabilityProbe:
    """Record the restore command outcome and readback, advancing or terminating the probe.

    A clean restore requires a ``SUCCESS`` outcome, a readback of ``favorited == baseline`` (the
    original state was restored), and ``disliked == baseline``. The same ambiguity and failure
    rules as ``observe_forward`` apply.
    """
    probe = _require_active_probe(probe)
    if probe.step_state is not ProbeStepState.RESTORE_STARTED:
        raise ProbeTransitionError(
            f"observe_restore requires RESTORE_STARTED, got {probe.step_state.value}"
        )
    command_outcome = _require_enum(command_outcome, CommandOutcome, "command_outcome")
    favorited = _require_bool_observation(favorited, "restore_favorited")
    disliked = _require_bool_observation(disliked, "restore_disliked")
    probe = replace(
        probe,
        restore_command_outcome=command_outcome,
        restore_favorited=favorited,
        restore_disliked=disliked,
    )
    if command_outcome is CommandOutcome.UNKNOWN:
        return replace(probe, verification_verdict=VerificationVerdict.INCONCLUSIVE)
    if command_outcome is CommandOutcome.FAILED:
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    if favorited.state is ObservationState.MISSING or disliked.state is ObservationState.MISSING:
        return replace(probe, verification_verdict=VerificationVerdict.INCONCLUSIVE)
    if favorited.payload != probe.baseline_favorited:
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    if disliked.payload != probe.baseline_disliked:
        return replace(probe, verification_verdict=VerificationVerdict.FAILED)
    return replace(probe, step_state=ProbeStepState.RESTORE_OBSERVED)


def finalize(probe: CapabilityProbe) -> CapabilityProbe:
    """Resolve a clean ``RESTORE_OBSERVED`` probe to ``VERIFIED`` / ``RESTORED``.

    ``VERIFIED`` is reachable only here: from a clean ``RESTORE_OBSERVED`` step state while the
    verdict is still ``PENDING``. A probe that already reached ``INCONCLUSIVE`` or ``FAILED`` can
    never finalize.
    """
    probe = _require_active_probe(probe)
    if probe.step_state is not ProbeStepState.RESTORE_OBSERVED:
        raise ProbeTransitionError(
            f"finalize requires RESTORE_OBSERVED, got {probe.step_state.value}"
        )
    return replace(
        probe,
        verification_verdict=VerificationVerdict.VERIFIED,
        recovery_status=RecoveryStatus.RESTORED,
    )


def mark_inconclusive(probe: CapabilityProbe) -> CapabilityProbe:
    """Mark an active probe ``INCONCLUSIVE`` (a crash or ambiguous interruption).

    Valid from any step state while the verdict is ``PENDING``. This only changes the verdict; the
    recovery status is resolved separately by ``decide_recovery`` / ``mark_recovery_status``.
    """
    probe = _require_active_probe(probe)
    return replace(probe, verification_verdict=VerificationVerdict.INCONCLUSIVE)


def mark_recovery_restore_started(probe: CapabilityProbe) -> CapabilityProbe:
    """Commit the restore-command STARTED checkpoint from an ambiguous recovery state.

    The clean-path route to ``RESTORE_STARTED`` is ``mark_restore_started``, which requires
    ``FORWARD_OBSERVED`` because a clean probe has observed the forward take effect. Recovery
    skips forward observation: a crash at ``FORWARD_STARTED`` (or ``RESTORE_STARTED``) never
    produced ``FORWARD_OBSERVED``, yet the source may still be in the opposite state and need
    restoring. This is the recovery route to the same ``RESTORE_STARTED`` position.

    Valid only while the verdict is ``INCONCLUSIVE``, from ``FORWARD_STARTED`` (advances) or
    ``RESTORE_STARTED`` (already at the checkpoint, a no-op). It never touches the verdict, so it
    can never contribute to verification evidence.
    """
    probe = _require_probe(probe)
    if probe.verification_verdict is not VerificationVerdict.INCONCLUSIVE:
        raise ProbeTransitionError(
            f"mark_recovery_restore_started requires INCONCLUSIVE, got "
            f"{probe.verification_verdict.value}"
        )
    if probe.step_state is ProbeStepState.RESTORE_STARTED:
        return probe
    if probe.step_state is not ProbeStepState.FORWARD_STARTED:
        raise ProbeTransitionError(
            f"mark_recovery_restore_started requires FORWARD_STARTED or RESTORE_STARTED, got "
            f"{probe.step_state.value}"
        )
    return replace(probe, step_state=ProbeStepState.RESTORE_STARTED)


def decide_recovery(probe: CapabilityProbe, current_favorited: ObservedValue) -> RecoveryAction:
    """Decide the recovery action from the durable baseline vs the current real observation.

    This is a pure decision: ``== baseline`` means no restore, ``!= baseline`` means restore, and an
    unavailable observation means a human must check. It never produces a ``VerificationVerdict``.
    """
    probe = _require_probe(probe)
    current_favorited = _require_bool_observation(current_favorited, "current_favorited")
    if current_favorited.state is ObservationState.MISSING:
        return RecoveryAction.MANUAL_CHECK
    if current_favorited.payload == probe.baseline_favorited:
        return RecoveryAction.NO_RESTORE
    return RecoveryAction.RESTORE


def mark_recovery_status(probe: CapabilityProbe, status: RecoveryStatus) -> CapabilityProbe:
    """Set ``recovery_status`` without touching ``verification_verdict``.

    This is the recovery-path setter: it may move an ``INCONCLUSIVE`` / ``FAILED`` probe to
    ``RESTORED`` or ``NEEDS_MANUAL_CHECK``, but it never changes the verdict, so it can never
    upgrade an inconclusive or failed probe to ``VERIFIED``.
    """
    probe = _require_probe(probe)
    _require_enum(status, RecoveryStatus, "status")
    return replace(probe, recovery_status=status)


def is_recoverable(probe: CapabilityProbe) -> bool:
    """True when ``recover_probe`` may reconcile ``probe`` from an ambiguous restart.

    ``FORWARD_STARTED`` / ``RESTORE_STARTED`` with a ``PENDING`` or ``INCONCLUSIVE`` verdict are the
    restart-safe ambiguous states: the corresponding ``*_STARTED`` checkpoint committed, but the
    command outcome may not have been reliably persisted. Every other combination is either not yet
    started (``BASELINE_CAPTURED``) or already terminal (``FORWARD_OBSERVED`` / ``RESTORE_OBSERVED`` /
    ``VERIFIED`` / ``FAILED``).
    """
    return (
        probe.step_state in (ProbeStepState.FORWARD_STARTED, ProbeStepState.RESTORE_STARTED)
        and probe.verification_verdict
        in (VerificationVerdict.PENDING, VerificationVerdict.INCONCLUSIVE)
    )


def existing_probe_blocks_new_probe(
    probe: CapabilityProbe, recovery_attempt_started: bool
) -> bool:
    """True if an existing same-target probe blocks a new probe for that target.

    This is the single shared definition used by both E2-A preflight and E2-B capture; neither may
    drift into its own copy. A probe is safe for a new probe only when its durable state proves the
    source is back at baseline and no ambiguous ``STARTED`` recovery attempt is in flight:

    - a ``STARTED`` recovery attempt always blocks (an in-flight restore);
    - ``PENDING`` always blocks (a live probe may still be mutating the source);
    - ``FAILED`` always blocks (it may have been reached after the command applied);
    - ``INCONCLUSIVE`` blocks unless ``RESTORED`` or ``BASELINE_CONFIRMED``;
    - ``VERIFIED`` blocks unless ``RESTORED``;
    - any unrecognized combination blocks.

    ``recovery_attempt_started`` is derived by the caller from the probe's recovery attempt (its
    ``state is RecoveryAttemptState.STARTED``).
    """
    if recovery_attempt_started:
        return True
    verdict = probe.verification_verdict
    if verdict is VerificationVerdict.PENDING:
        return True
    if verdict is VerificationVerdict.FAILED:
        return True
    if verdict is VerificationVerdict.INCONCLUSIVE:
        return probe.recovery_status not in (
            RecoveryStatus.RESTORED,
            RecoveryStatus.BASELINE_CONFIRMED,
        )
    if verdict is VerificationVerdict.VERIFIED:
        return probe.recovery_status is not RecoveryStatus.RESTORED
    return True


def _require_probe(probe: object) -> CapabilityProbe:
    if not isinstance(probe, CapabilityProbe):
        raise CapabilityProbeValidationError("probe must be a CapabilityProbe")
    return probe


def _require_active_probe(probe: object) -> CapabilityProbe:
    probe = _require_probe(probe)
    if probe.verification_verdict is not VerificationVerdict.PENDING:
        raise ProbeTransitionError(
            f"probe {probe.probe_id!r} has terminal verdict "
            f"{probe.verification_verdict.value}; cannot continue the clean path"
        )
    return probe


def _require_persistent_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise CapabilityProbeValidationError("persistent_id must be a non-empty string")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CapabilityProbeValidationError(f"{field} must be a strict bool")
    return value


def _require_enum(value: object, enum_type: type, field: str):
    if not isinstance(value, enum_type):
        raise CapabilityProbeValidationError(f"{field} must be a {enum_type.__name__}")
    return value


def _require_bool_observation(value: object, field: str) -> ObservedValue:
    if not isinstance(value, ObservedValue):
        raise CapabilityProbeValidationError(f"{field} must be an ObservedValue")
    if value.state is ObservationState.NULL:
        raise CapabilityProbeValidationError(
            f"{field} cannot be NULL (favorited/disliked are non-nullable)"
        )
    if value.state is ObservationState.VALUE and not isinstance(value.payload, bool):
        raise CapabilityProbeValidationError(f"{field} VALUE must be a strict bool")
    return value
