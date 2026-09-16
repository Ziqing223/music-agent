"""Read-only target preflight for a live ``set_favorited`` capability probe.

Before a live probe can run, its target must be proven eligible: a durable Apple Music Track with
an authoritative persistent ID, currently present in the source, with a clean live readback of the
two probe baseline fields. This module answers that question without mutating anything.

``CapabilityProbePreflightService.check_target`` takes one explicit canonical Track ID (the operator
or the later E2-B stage chooses *which* track) and returns a ``CapabilityProbePreflightResult``. It
never selects a track itself, never runs a write command, and never persists a probe, a recovery
attempt, an intent, or an execution attempt.

The eligibility gates, in order, all fail closed:

1. the canonical entity exists and is a Track (``canonical_entities``),
2. a durable Apple Music binding exists and yields a non-empty persistent ID
   (``external_identity_bindings`` is the only identity authority -- no name or fuzzy lookup),
3. ``source_entity_presence`` is ``PRESENT`` for the target source/scope,
4. no existing same-target probe leaves the source in an unprovable state,
5. the live ``read_track`` is ``FOUND``,
6. ``favorited`` and ``disliked`` are both strict boolean ``VALUE`` observations.

A command-level identity detail: the live baseline (``F0`` / ``D0``) comes *only* from the live
source read, never from a historical snapshot, the canonical cached ``library_state``, a name
lookup, or a canonical ``external_ids`` projection.

The existing-probe safety gate treats a terminal verdict as *evidence about that one probe record*,
not as proof the real source is back at its baseline. ``FAILED`` in particular may have been reached
*after* a command applied and the readback diverged, so ``FAILED`` does not by itself prove a safe
baseline. A same-target probe is allowed only when its verdict + recovery status prove the source is
back at baseline and no ambiguous ``STARTED`` recovery attempt is in flight; anything else blocks.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from music_agent.apple_music import AppleMusicSourceAdapter, SourceReadStatus
from music_agent.capability_probe import (
    VerificationVerdict,
    existing_probe_blocks_new_probe,
)
from music_agent.capability_probe_recovery_attempt import RecoveryAttemptState
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.identity import EntityType
from music_agent.source_observation import SourcePresence


class CapabilityProbePreflightError(ValueError):
    code = "capability_probe_preflight_error"


class PreflightRejectionReason(StrEnum):
    """Why a candidate target is not eligible. Every value is a fail-closed refusal."""

    CANONICAL_ENTITY_NOT_FOUND = "canonical_entity_not_found"
    NOT_A_TRACK = "not_a_track"
    APPLE_MUSIC_BINDING_MISSING = "apple_music_binding_missing"
    PERSISTENT_ID_EMPTY = "persistent_id_empty"
    SOURCE_PRESENCE_NOT_PRESENT = "source_presence_not_present"
    LIVE_READ_NOT_FOUND = "live_read_not_found"
    LIVE_LOOKUP_FAILED = "live_lookup_failed"
    FAVORITED_MISSING = "favorited_missing"
    DISLIKED_MISSING = "disliked_missing"
    FAVORITED_NOT_BOOL = "favorited_not_bool"
    DISLIKED_NOT_BOOL = "disliked_not_bool"
    EXISTING_PROBE_PENDING = "existing_probe_pending"
    EXISTING_PROBE_UNRESOLVED_SOURCE = "existing_probe_unresolved_source"
    EXISTING_PROBE_STARTED_ATTEMPT = "existing_probe_started_attempt"


class CanonicalPreflightReadPort(Protocol):
    """The read-only canonical/identity/presence surface preflight depends on.

    ``CanonicalRepository`` satisfies this port; a fake may stand in for edge-case tests.
    """

    def get_entity_type(self, canonical_id: str) -> EntityType | None: ...

    def get_bound_external_id(
        self, source_system: str, entity_type: EntityType, canonical_id: str
    ) -> str | None: ...

    def get_source_presence(
        self, source_system: str, entity_type: EntityType, canonical_id: str, scope_key: str
    ) -> SourcePresence | None: ...


@dataclass(frozen=True, slots=True)
class CapabilityProbePreflightResult:
    """A point-in-time eligibility observation for one candidate probe target.

    ``eligible=True`` carries the authoritative persistent ID and the live baseline ``F0`` / ``D0``.
    ``eligible=False`` carries none of the target data (a later stage must not guess it) and always
    names a ``rejection_reason``.
    """

    eligible: bool
    canonical_track_id: str
    target_persistent_id: str | None
    baseline_favorited: bool | None
    baseline_disliked: bool | None
    rejection_reason: PreflightRejectionReason | None

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_track_id, str):
            raise CapabilityProbePreflightError("canonical_track_id must be a string")
        if self.eligible:
            if not isinstance(self.target_persistent_id, str) or self.target_persistent_id == "":
                raise CapabilityProbePreflightError(
                    "an eligible result requires a non-empty target_persistent_id"
                )
            if not isinstance(self.baseline_favorited, bool):
                raise CapabilityProbePreflightError(
                    "an eligible result requires a strict bool baseline_favorited"
                )
            if not isinstance(self.baseline_disliked, bool):
                raise CapabilityProbePreflightError(
                    "an eligible result requires a strict bool baseline_disliked"
                )
            if self.rejection_reason is not None:
                raise CapabilityProbePreflightError(
                    "an eligible result cannot carry a rejection_reason"
                )
        else:
            if not isinstance(self.rejection_reason, PreflightRejectionReason):
                raise CapabilityProbePreflightError(
                    "an ineligible result requires a rejection_reason"
                )
            if (
                self.target_persistent_id is not None
                or self.baseline_favorited is not None
                or self.baseline_disliked is not None
            ):
                raise CapabilityProbePreflightError(
                    "an ineligible result cannot carry target data"
                )


_MISSING = object()
_INVALID = object()


class CapabilityProbePreflightService:
    """Read-only eligibility check for a candidate live-probe target.

    The service holds only read dependencies: a canonical read port, the probe and recovery-attempt
    repositories (read via ``list_probes`` / ``get_for_probe``), and the Apple Music source read
    adapter. It has no command runner and no write repository, so it cannot mutate any store or the
    source.
    """

    def __init__(
        self,
        canonical_repository: CanonicalPreflightReadPort,
        probe_repository: CapabilityProbeRepository,
        recovery_attempt_repository: CapabilityProbeRecoveryAttemptRepository,
        source_adapter: AppleMusicSourceAdapter,
        scope_key: str,
    ) -> None:
        if not isinstance(scope_key, str) or scope_key == "":
            raise CapabilityProbePreflightError("scope_key must be a non-empty string")
        self._canonical_repository = canonical_repository
        self._probe_repository = probe_repository
        self._recovery_attempt_repository = recovery_attempt_repository
        self._source_adapter = source_adapter
        self._scope_key = scope_key

    def check_target(self, canonical_track_id: str) -> CapabilityProbePreflightResult:
        """Return the eligibility of ``canonical_track_id`` as a live-probe target.

        This is a point-in-time observation. It performs only DB reads and one source read; it never
        writes. The ordering matters: durable identity/presence gates and the existing-probe safety
        gate are checked before the live source read, so an unsafe or already-probed target never
        triggers a source read.
        """
        entity_type = self._canonical_repository.get_entity_type(canonical_track_id)
        if entity_type is None:
            return self._reject(
                canonical_track_id, PreflightRejectionReason.CANONICAL_ENTITY_NOT_FOUND
            )
        if entity_type is not EntityType.TRACK:
            return self._reject(canonical_track_id, PreflightRejectionReason.NOT_A_TRACK)

        persistent_id = self._canonical_repository.get_bound_external_id(
            "apple_music", EntityType.TRACK, canonical_track_id
        )
        if persistent_id is None:
            return self._reject(
                canonical_track_id, PreflightRejectionReason.APPLE_MUSIC_BINDING_MISSING
            )
        if persistent_id == "":
            return self._reject(
                canonical_track_id, PreflightRejectionReason.PERSISTENT_ID_EMPTY
            )

        presence = self._canonical_repository.get_source_presence(
            "apple_music", EntityType.TRACK, canonical_track_id, self._scope_key
        )
        if presence is not SourcePresence.PRESENT:
            return self._reject(
                canonical_track_id, PreflightRejectionReason.SOURCE_PRESENCE_NOT_PRESENT
            )

        blocking = self._existing_probe_block(canonical_track_id)
        if blocking is not None:
            return self._reject(canonical_track_id, blocking)

        read_result = self._source_adapter.read_track(persistent_id)
        if read_result.status is SourceReadStatus.CONFIRMED_NOT_FOUND:
            return self._reject(canonical_track_id, PreflightRejectionReason.LIVE_READ_NOT_FOUND)
        if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
            return self._reject(canonical_track_id, PreflightRejectionReason.LIVE_LOOKUP_FAILED)

        favorited = _read_bool_field(read_result.record.fields, "favorited")
        if favorited is _MISSING:
            return self._reject(canonical_track_id, PreflightRejectionReason.FAVORITED_MISSING)
        if favorited is _INVALID:
            return self._reject(canonical_track_id, PreflightRejectionReason.FAVORITED_NOT_BOOL)
        disliked = _read_bool_field(read_result.record.fields, "disliked")
        if disliked is _MISSING:
            return self._reject(canonical_track_id, PreflightRejectionReason.DISLIKED_MISSING)
        if disliked is _INVALID:
            return self._reject(canonical_track_id, PreflightRejectionReason.DISLIKED_NOT_BOOL)

        return CapabilityProbePreflightResult(
            eligible=True,
            canonical_track_id=canonical_track_id,
            target_persistent_id=persistent_id,
            baseline_favorited=favorited,
            baseline_disliked=disliked,
            rejection_reason=None,
        )

    def _existing_probe_block(self, canonical_track_id: str) -> PreflightRejectionReason | None:
        """Return a blocking reason if any same-target probe blocks a new probe, else ``None``.

        The blocking decision itself is the shared ``existing_probe_blocks_new_probe`` predicate
        (the same definition E2-B capture uses); only the fail-closed reason is derived here.
        """
        for probe in self._probe_repository.list_probes():
            if probe.target_canonical_id != canonical_track_id:
                continue
            attempt = self._recovery_attempt_repository.get_for_probe(probe.probe_id)
            attempt_started = attempt is not None and attempt.state is RecoveryAttemptState.STARTED
            if not existing_probe_blocks_new_probe(probe, attempt_started):
                continue
            if attempt_started:
                return PreflightRejectionReason.EXISTING_PROBE_STARTED_ATTEMPT
            if probe.verification_verdict is VerificationVerdict.PENDING:
                return PreflightRejectionReason.EXISTING_PROBE_PENDING
            return PreflightRejectionReason.EXISTING_PROBE_UNRESOLVED_SOURCE
        return None

    def _reject(
        self, canonical_track_id: str, reason: PreflightRejectionReason
    ) -> CapabilityProbePreflightResult:
        return CapabilityProbePreflightResult(
            eligible=False,
            canonical_track_id=canonical_track_id,
            target_persistent_id=None,
            baseline_favorited=None,
            baseline_disliked=None,
            rejection_reason=reason,
        )


def _read_bool_field(fields: object, key: str) -> object:
    """Return a bool, ``_MISSING`` (absent/null), or ``_INVALID`` (non-bool) for ``fields[key]``."""
    if not isinstance(fields, Mapping) or key not in fields or fields[key] is None:
        return _MISSING
    value = fields[key]
    if not isinstance(value, bool):
        return _INVALID
    return value
