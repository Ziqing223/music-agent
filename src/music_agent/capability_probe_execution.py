"""E2-B: durable capture of an eligible preflight result into a ``FORWARD_STARTED`` probe.

E2-A's ``CapabilityProbePreflightResult`` is an eligibility observation, not an execution
authorization: the source may have changed between the preflight read and the probe. E2-B therefore
re-reads the same persistent ID immediately before any durable capture, and only proceeds when the
fresh read is ``FOUND``, yields strict boolean ``favorited`` / ``disliked``, and matches the E2-A
``F0`` / ``D0`` exactly. If the value changed (or the read is invalid), it aborts *before* any probe
row exists or any Music.app command runs, and returns a typed start failure rather than a durable
probe failure.

A valid fresh baseline is captured as a probe born atomically at ``FORWARD_STARTED`` (``PENDING`` +
``BASELINE_CONFIRMED``) under one ``BEGIN IMMEDIATE`` transaction that also re-asserts the shared
same-target blocking rule. No external I/O happens inside that transaction. The forward command is
dispatched only afterwards, by the orchestrator's ``run_started_probe``, once the ``FORWARD_STARTED``
row is committed. ``FORWARD_STARTED`` means "the command may have executed; durable state must never
assume it did not", and a command exception is never proof of no side effect.

Operational exclusivity invariant (documented, not enforceable in code): no other actor -- human UI,
another process/agent, or sync automation -- may mutate the target Track's ``favorited`` or
``disliked`` fields from the fresh execution-baseline read until source safety is durably resolved
(``BASELINE_CONFIRMED`` or ``RESTORED``). This window includes crashes and recovery, and it ends only
when source safety is durably resolved; ``NEEDS_MANUAL_CHECK`` remains blocked pending operator
resolution. The code cannot enforce this exclusivity, and readback cannot attribute a state
transition to the probe versus an external actor (desired-state confirmation, not causal
attribution).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from music_agent.apple_music import AppleMusicSourceAdapter, SourceReadStatus
from music_agent.capability_probe import CapabilityProbe, create_forward_started_probe
from music_agent.capability_probe_preflight import CapabilityProbePreflightResult
from music_agent.capability_probe_repository import (
    CapabilityProbeRepository,
    ProbeTargetBlockedError,
)


class CapabilityProbeExecutionError(ValueError):
    code = "capability_probe_execution_error"


class CapabilityProbeStartFailure(StrEnum):
    """Why an eligible preflight result could not be durably captured at ``FORWARD_STARTED``."""

    FRESH_READ_NOT_FOUND = "fresh_read_not_found"
    FRESH_READ_LOOKUP_FAILED = "fresh_read_lookup_failed"
    FRESH_READ_FAVORITED_MISSING = "fresh_read_favorited_missing"
    FRESH_READ_DISLIKED_MISSING = "fresh_read_disliked_missing"
    FRESH_READ_FAVORITED_NOT_BOOL = "fresh_read_favorited_not_bool"
    FRESH_READ_DISLIKED_NOT_BOOL = "fresh_read_disliked_not_bool"
    BASELINE_CHANGED = "baseline_changed"
    TARGET_BLOCKED = "target_blocked"


@dataclass(frozen=True, slots=True)
class CapabilityProbeStartResult:
    """Outcome of one E2-B capture attempt.

    ``started=True`` carries the durable ``FORWARD_STARTED`` probe and no failure. ``started=False``
    carries exactly one typed failure and no probe (nothing durable was written and no command ran).
    """

    started: bool
    probe: CapabilityProbe | None
    failure: CapabilityProbeStartFailure | None

    def __post_init__(self) -> None:
        if self.started:
            if self.probe is None:
                raise CapabilityProbeExecutionError("a started result requires a probe")
            if self.failure is not None:
                raise CapabilityProbeExecutionError("a started result cannot carry a failure")
        else:
            if self.probe is not None:
                raise CapabilityProbeExecutionError("a failed result cannot carry a probe")
            if not isinstance(self.failure, CapabilityProbeStartFailure):
                raise CapabilityProbeExecutionError("a failed result requires a failure")


_MISSING = object()
_INVALID = object()


class CapabilityProbeExecutionService:
    """Fresh-read then durably capture an eligible preflight result as a ``FORWARD_STARTED`` probe.

    The service holds only read + capture dependencies: the Apple Music source read adapter (for the
    fresh execution-baseline read) and the probe repository (for the atomic capture). It has no
    command runner, so it can never issue a Music.app command itself; the forward command is
    dispatched later by the orchestrator against the committed ``FORWARD_STARTED`` probe.
    """

    def __init__(
        self,
        source_adapter: AppleMusicSourceAdapter,
        probe_repository: CapabilityProbeRepository,
    ) -> None:
        self._source_adapter = source_adapter
        self._probe_repository = probe_repository

    def start(self, eligible: CapabilityProbePreflightResult) -> CapabilityProbeStartResult:
        """Capture an eligible result, or return a typed start failure with nothing durable.

        Re-reads the same persistent ID the preflight resolved and requires ``FOUND`` + strict
        boolean ``favorited`` / ``disliked`` equal to the E2-A ``F0`` / ``D0``. Any divergence or
        invalid read aborts before persistence and before any command. On success the fresh baseline
        is captured as a probe born at ``FORWARD_STARTED`` under one transaction that re-asserts the
        shared same-target blocking rule.
        """
        if not isinstance(eligible, CapabilityProbePreflightResult) or not eligible.eligible:
            raise CapabilityProbeExecutionError("start requires an eligible preflight result")
        persistent_id = eligible.target_persistent_id
        if persistent_id is None:
            raise CapabilityProbeExecutionError("eligible result must carry a persistent ID")

        read_result = self._source_adapter.read_track(persistent_id)
        if read_result.status is SourceReadStatus.CONFIRMED_NOT_FOUND:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_NOT_FOUND)
        if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_LOOKUP_FAILED)

        favorited = _strict_bool(read_result.record.fields, "favorited")
        if favorited is _MISSING:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_FAVORITED_MISSING)
        if favorited is _INVALID:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_FAVORITED_NOT_BOOL)
        disliked = _strict_bool(read_result.record.fields, "disliked")
        if disliked is _MISSING:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_DISLIKED_MISSING)
        if disliked is _INVALID:
            return self._fail(CapabilityProbeStartFailure.FRESH_READ_DISLIKED_NOT_BOOL)

        if favorited != eligible.baseline_favorited or disliked != eligible.baseline_disliked:
            return self._fail(CapabilityProbeStartFailure.BASELINE_CHANGED)

        probe = create_forward_started_probe(
            eligible.canonical_track_id,
            persistent_id,
            favorited,
            disliked,
        )
        try:
            durable = self._probe_repository.begin_forward_probe(probe)
        except ProbeTargetBlockedError:
            return self._fail(CapabilityProbeStartFailure.TARGET_BLOCKED)
        return CapabilityProbeStartResult(started=True, probe=durable, failure=None)

    def _fail(self, failure: CapabilityProbeStartFailure) -> CapabilityProbeStartResult:
        return CapabilityProbeStartResult(started=False, probe=None, failure=failure)


def _strict_bool(fields: object, key: str) -> object:
    """Return a ``bool``, ``_MISSING`` (absent/null), or ``_INVALID`` (non-bool) for ``fields[key]``."""
    if not isinstance(fields, Mapping) or key not in fields or fields[key] is None:
        return _MISSING
    value = fields[key]
    if not isinstance(value, bool):
        return _INVALID
    return value
