"""First-class historical verification evidence derived from a capability probe.

A capability probe (``capability_probe``) is an *operational* object: it records the live
forward/restore cycle and the terminal verdict of one ``set_favorited`` attempt. E2-C promotes
exactly one kind of probe into an immutable, append-only ``CapabilityVerificationEvidence`` record:
a clean ``VERIFIED`` + ``RESTORED`` probe whose durable observations prove the complete evidence
shape required by promotion.

A promoted evidence record means *both* of:

1. this probe supplied evidence that the forward write mutated ``favorited`` as expected
   (``forward_favorited == !baseline_favorited``) with no cross-field side effect
   (``forward_disliked == baseline_disliked``);
2. the write/readback loop confirmed it -- the restore command succeeded and the readback loop
   confirmed the source back at baseline (``restore_favorited == baseline_favorited`` and
   ``restore_disliked == baseline_disliked``, both with a ``SUCCESS`` command outcome).

It also records the cross-field coverage class (``F0`` / ``D0``) exercised by that probe, where
``F0 = baseline_favorited`` and ``D0 = baseline_disliked``.

The evidence is *historical only*. It never changes ``is_execution_ready``, never mutates the P01
``capability_verified`` / ``readback_verified`` flags, never performs Music.app I/O, and never
claims global verification, current validity, or that a ``D0=false`` probe proves ``D0=true``
safety. ``D0=true`` remains unsafe to probe live today because there is no verified
``set_disliked`` restore path.

Verification contract version
-----------------------------

``SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION`` scopes the *verification semantics* under which
evidence is produced: the current ``set_favorited`` command semantics, readback semantics, and probe
verification semantics. It does **not** version the macOS / Music.app runtime behavior. Any material
change to those verification semantics must bump the constant, so future coverage aggregation can
distinguish evidence produced under different contracts instead of silently mixing it. The version
makes evidence attributable; it does not make any evidence "currently valid".

The promotion predicate is the single shared definition of the complete evidence shape: it re-derives
the full forward/restore evidence from the probe's durable observations rather than merely trusting
the ``VERIFIED`` + ``RESTORED`` enum pair, so a malformed or incomplete probe (constructible through
the repository's write API, which validates types but not cross-field invariants) can never promote.
"""

from __future__ import annotations

from dataclasses import dataclass

from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    validate_probe_id,
)
from music_agent.identity import EntityType, validate_canonical_id
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.write_intent import WriteOperation

# The current verification contract for ``set_favorited`` probe-derived evidence. It scopes the
# command / readback / probe-verification semantics under which the evidence was produced. Bump it
# on any material change to those semantics (see the module docstring); it does not version the
# macOS / Music.app runtime and does not make any evidence "currently valid".
SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION = 1


class CapabilityVerificationEvidenceError(ValueError):
    code = "capability_verification_evidence_error"


class CapabilityVerificationEvidenceValidationError(CapabilityVerificationEvidenceError):
    code = "validation_error"


class PromotionNotEligibleError(CapabilityVerificationEvidenceError):
    code = "promotion_not_eligible"


@dataclass(frozen=True, slots=True)
class CapabilityVerificationEvidence:
    """One immutable historical verification-evidence record derived from one probe.

    ``probe_id`` is the evidence identity, idempotency, and provenance key: promotion is
    ``probe_id -> one evidence record``, so a separate ``evidence_id`` would carry no meaning the
    probe identity does not already provide. The first seven fields are the probe's immutable
    identity + baseline snapshot; ``verification_contract_version`` scopes the verification
    semantics; ``verified_at`` records when the evidence was durably promoted.
    """

    probe_id: str
    operation: WriteOperation
    target_canonical_id: str
    target_persistent_id: str
    baseline_favorited: bool
    baseline_disliked: bool
    verification_contract_version: int
    verified_at: str

    def __post_init__(self) -> None:
        validate_probe_id(self.probe_id)
        if not isinstance(self.operation, WriteOperation):
            raise CapabilityVerificationEvidenceValidationError(
                "operation must be a WriteOperation"
            )
        if self.operation is not WriteOperation.SET_FAVORITED:
            raise CapabilityVerificationEvidenceValidationError(
                f"verification evidence supports only set_favorited, got {self.operation.value}"
            )
        validate_canonical_id(EntityType.TRACK, self.target_canonical_id)
        _require_non_empty_str(self.target_persistent_id, "target_persistent_id")
        _require_bool(self.baseline_favorited, "baseline_favorited")
        _require_bool(self.baseline_disliked, "baseline_disliked")
        if (
            not isinstance(self.verification_contract_version, int)
            or isinstance(self.verification_contract_version, bool)
            or self.verification_contract_version < 1
        ):
            raise CapabilityVerificationEvidenceValidationError(
                "verification_contract_version must be a positive int"
            )
        _require_non_empty_str(self.verified_at, "verified_at")


def is_promotion_eligible(probe: CapabilityProbe) -> bool:
    """Return True only when ``probe`` proves the complete evidence shape required by promotion.

    This is the single shared promotion predicate: it does not merely trust the ``VERIFIED`` +
    ``RESTORED`` enum pair. A probe whose durable observations are missing or incomplete is
    constructible through the repository's write API (which validates field types, not the
    cross-field ``VERIFIED`` invariant), so promotion re-derives the full evidence here:

    - a clean ``VERIFIED`` verdict reached from ``RESTORE_OBSERVED``;
    - ``RESTORED`` recovery safety;
    - a known ``SUCCESS`` forward command whose readback shows ``favorited`` mutated to
      ``!baseline`` with ``disliked`` unchanged (no cross-field side effect);
    - a known ``SUCCESS`` restore command whose readback shows both fields back at baseline.

    Any missing observation, unknown/failed command outcome, or divergent value fails closed. The
    predicate is coverage-class agnostic: a structurally complete ``D0=true`` probe is eligible and
    records its ``D0=true`` coverage class, but promoting it does not prove anything beyond that one
    probe.
    """
    if not isinstance(probe, CapabilityProbe):
        return False
    if probe.operation is not WriteOperation.SET_FAVORITED:
        return False
    if probe.verification_verdict is not VerificationVerdict.VERIFIED:
        return False
    if probe.recovery_status is not RecoveryStatus.RESTORED:
        return False
    if probe.step_state is not ProbeStepState.RESTORE_OBSERVED:
        return False
    if probe.forward_command_outcome is not CommandOutcome.SUCCESS:
        return False
    if probe.restore_command_outcome is not CommandOutcome.SUCCESS:
        return False
    if not _value_is(probe.forward_favorited, not probe.baseline_favorited):
        return False
    if not _value_is(probe.forward_disliked, probe.baseline_disliked):
        return False
    if not _value_is(probe.restore_favorited, probe.baseline_favorited):
        return False
    if not _value_is(probe.restore_disliked, probe.baseline_disliked):
        return False
    return True


def evidence_from_probe(
    probe: CapabilityProbe, verified_at: str
) -> CapabilityVerificationEvidence:
    """Build the immutable evidence record for an eligible probe under the current contract.

    This fails closed unless ``is_promotion_eligible(probe)`` holds, so it can never be used to mint
    evidence from a malformed, incomplete, ``FAILED``, or ``INCONCLUSIVE`` probe. The record captures
    the probe's immutable identity + baseline (the coverage class ``F0`` / ``D0``) and stamps the
    current ``SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION``.
    """
    if not is_promotion_eligible(probe):
        raise PromotionNotEligibleError(
            f"probe {probe.probe_id!r} is not promotion-eligible: "
            "a VERIFIED + RESTORED set_favorited probe with complete forward/restore evidence "
            "is required"
        )
    return CapabilityVerificationEvidence(
        probe_id=probe.probe_id,
        operation=probe.operation,
        target_canonical_id=probe.target_canonical_id,
        target_persistent_id=probe.target_persistent_id,
        baseline_favorited=probe.baseline_favorited,
        baseline_disliked=probe.baseline_disliked,
        verification_contract_version=SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION,
        verified_at=verified_at,
    )


def _value_is(observed: ObservedValue | None, expected: bool) -> bool:
    return (
        isinstance(observed, ObservedValue)
        and observed.state is ObservationState.VALUE
        and observed.payload is expected
    )


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CapabilityVerificationEvidenceValidationError(f"{field} must be a strict bool")
    return value


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise CapabilityVerificationEvidenceValidationError(
            f"{field} must be a non-empty string"
        )
    return value
