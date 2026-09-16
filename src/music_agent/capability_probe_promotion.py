"""Probe-promotion service: promote one durable, eligible probe into historical evidence.

``CapabilityProbePromotionService.promote_probe(probe_id)`` is the caller-facing promotion boundary.
It re-reads the durable probe by identity, then delegates to
``CapabilityVerificationEvidenceRepository.record_from_probe``, which validates the shared
promotion-eligibility predicate and inserts the evidence idempotently.

The service holds only read/persist dependencies -- a probe repository and an evidence repository --
so it can never run a Music.app command or read, mutate the probe row, or change any capability
flag. It has no command or readback adapter, and it never touches the canonical model, bindings,
presence, intents, or attempts.
"""

from __future__ import annotations

from music_agent.capability_probe import validate_probe_id
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.capability_verification_evidence import CapabilityVerificationEvidence
from music_agent.capability_verification_evidence_repository import (
    CapabilityVerificationEvidenceRepository,
)


class CapabilityProbePromotionError(ValueError):
    code = "capability_probe_promotion_error"


class ProbeNotFoundError(CapabilityProbePromotionError):
    code = "probe_not_found"


class CapabilityProbePromotionService:
    """Promote a durable, promotion-eligible probe into one immutable evidence record."""

    def __init__(
        self,
        probe_repository: CapabilityProbeRepository,
        evidence_repository: CapabilityVerificationEvidenceRepository,
    ) -> None:
        self.probe_repository = probe_repository
        self.evidence_repository = evidence_repository

    def promote_probe(self, probe_id: str) -> CapabilityVerificationEvidence:
        """Re-read the durable probe and promote it to evidence, or fail closed.

        The probe is re-read inside the probe repository's connection; ``record_from_probe`` then
        re-reads it again inside its own ``BEGIN IMMEDIATE`` transaction and fails closed on any
        drift or conflicting evidence. No external I/O occurs and the probe row is never mutated.
        """
        validate_probe_id(probe_id)
        probe = self.probe_repository.get_probe(probe_id)
        if probe is None:
            raise ProbeNotFoundError(f"probe {probe_id!r} does not exist")
        return self.evidence_repository.record_from_probe(probe)
