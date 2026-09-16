"""Durable SQLite persistence for probe-derived verification evidence.

A ``CapabilityVerificationEvidence`` is immutable, append-only historical evidence derived from one
durable ``CapabilityProbe``. It lives in its own table (``capability_verification_evidence``),
isolated from ``canonical_entities``, ``external_identity_bindings``, ``source_entity_presence``,
``pending_write_intents``, ``write_execution_attempts``, ``capability_probes``, and
``capability_probe_recovery_attempts``. It never mutates the probe row, the canonical model, or the
write capability matrix, and it performs no external I/O.

``record_from_probe`` is the single promotion boundary. Inside one ``BEGIN IMMEDIATE`` transaction
it re-reads the durable probe, validates the shared ``is_promotion_eligible`` predicate against that
re-read, and inserts the evidence idempotently:

- same probe promoted twice -> the same row is returned (one row, same ``verified_at``);
- a conflicting existing evidence signature -> ``EvidenceConflictError`` (fail closed);
- a ``FAILED`` / ``INCONCLUSIVE`` / non-``RESTORED`` / malformed probe -> ``PromotionNotEligibleError``.

Later ``FAILED`` / ``INCONCLUSIVE`` probes never delete or negate existing evidence: the table has no
update or delete path, and the probe row is never touched by promotion.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from music_agent.capability_probe import CapabilityProbe, validate_probe_id
from music_agent.capability_probe_repository import _load_probe
from music_agent.capability_verification_evidence import (
    CapabilityVerificationEvidence,
    PromotionNotEligibleError,
    evidence_from_probe,
    is_promotion_eligible,
)
from music_agent.repository import _open_store_connection
from music_agent.write_intent import WriteOperation


class CapabilityVerificationEvidenceRepositoryError(ValueError):
    code = "capability_verification_evidence_repository_error"


class ProbeNotFoundError(CapabilityVerificationEvidenceRepositoryError):
    code = "probe_not_found"


class EvidenceConflictError(CapabilityVerificationEvidenceRepositoryError):
    code = "evidence_conflict"


class CapabilityVerificationEvidenceRepository:
    """Persist immutable probe-derived verification evidence inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CapabilityVerificationEvidenceRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def record_from_probe(self, probe: CapabilityProbe) -> CapabilityVerificationEvidence:
        """Promote one durable, eligible probe into an immutable evidence record, idempotently.

        This is the single promotion boundary. It fails closed unless ``probe`` is promotion-eligible
        (``is_promotion_eligible``), re-reads the durable probe inside one ``BEGIN IMMEDIATE``
        transaction, requires that re-read to equal the caller's probe exactly, and then inserts the
        evidence only if no conflicting evidence already exists. Re-promoting the same probe returns
        the existing row unchanged (one row per probe, preserving the original ``verified_at``). A
        pre-existing evidence row with a different immutable signature fails closed. The probe row is
        never mutated and no external I/O occurs.
        """
        _require_probe(probe)
        if not is_promotion_eligible(probe):
            raise PromotionNotEligibleError(
                f"probe {probe.probe_id!r} is not promotion-eligible"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            durable = _load_probe(self._connection, probe.probe_id)
            if durable is None:
                raise ProbeNotFoundError(f"probe {probe.probe_id!r} does not exist")
            if durable != probe:
                raise EvidenceConflictError(
                    f"probe {probe.probe_id!r} diverged from the promotion input"
                )
            # evidence_from_probe re-asserts eligibility against the durable re-read and fails
            # closed rather than trusting the caller's in-memory probe.
            candidate = evidence_from_probe(durable, _current_verified_at())
            existing = self._load_evidence(probe.probe_id)
            if existing is not None:
                if _immutable_signature(existing) != _immutable_signature(candidate):
                    raise EvidenceConflictError(
                        f"evidence for probe {probe.probe_id!r} already exists with a "
                        "conflicting signature"
                    )
                result = existing
            else:
                self._insert_evidence(candidate)
                result = candidate
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        return result

    def get_by_probe_id(self, probe_id: str) -> CapabilityVerificationEvidence | None:
        validate_probe_id(probe_id)
        return self._load_evidence(probe_id)

    def list_evidence(
        self,
        *,
        baseline_favorited: bool | None = None,
        baseline_disliked: bool | None = None,
    ) -> tuple[CapabilityVerificationEvidence, ...]:
        """Return evidence in deterministic ``probe_id`` order, optionally by coverage class.

        ``baseline_favorited`` / ``baseline_disliked`` are the ``F0`` / ``D0`` coverage dimensions;
        filtering by them distinguishes, e.g., the ``D0=false`` class from the ``D0=true`` class.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if baseline_favorited is not None:
            _require_bool(baseline_favorited, "baseline_favorited")
            conditions.append("baseline_favorited = ?")
            params.append(int(baseline_favorited))
        if baseline_disliked is not None:
            _require_bool(baseline_disliked, "baseline_disliked")
            conditions.append("baseline_disliked = ?")
            params.append(int(baseline_disliked))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._connection.execute(
            f"SELECT * FROM capability_verification_evidence{where} ORDER BY probe_id",
            params,
        )
        return tuple(_decode_evidence(row) for row in rows)

    def _insert_evidence(self, evidence: CapabilityVerificationEvidence) -> None:
        self._connection.execute(
            """INSERT INTO capability_verification_evidence(
                probe_id, operation, target_canonical_id, target_persistent_id,
                baseline_favorited, baseline_disliked, verification_contract_version, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            _encode_evidence(evidence),
        )

    def _load_evidence(self, probe_id: str) -> CapabilityVerificationEvidence | None:
        row = self._connection.execute(
            "SELECT * FROM capability_verification_evidence WHERE probe_id=?", (probe_id,)
        ).fetchone()
        return None if row is None else _decode_evidence(row)


def _require_probe(probe: object) -> CapabilityProbe:
    if not isinstance(probe, CapabilityProbe):
        raise CapabilityVerificationEvidenceRepositoryError("probe must be a CapabilityProbe")
    return probe


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CapabilityVerificationEvidenceRepositoryError(f"{field} must be a strict bool")
    return value


def _current_verified_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _immutable_signature(evidence: CapabilityVerificationEvidence) -> tuple[Any, ...]:
    """The probe-derived signature that must never differ for the same ``probe_id``.

    ``verified_at`` is recording metadata, not part of the evidence's meaning, so it is excluded:
    re-promoting the same probe must return the original row rather than fail a conflict check on a
    freshly generated timestamp.
    """
    return (
        evidence.probe_id,
        evidence.operation,
        evidence.target_canonical_id,
        evidence.target_persistent_id,
        evidence.baseline_favorited,
        evidence.baseline_disliked,
        evidence.verification_contract_version,
    )


def _encode_evidence(evidence: CapabilityVerificationEvidence) -> tuple[Any, ...]:
    return (
        evidence.probe_id,
        evidence.operation.value,
        evidence.target_canonical_id,
        evidence.target_persistent_id,
        int(evidence.baseline_favorited),
        int(evidence.baseline_disliked),
        evidence.verification_contract_version,
        evidence.verified_at,
    )


def _decode_evidence(row: sqlite3.Row) -> CapabilityVerificationEvidence:
    try:
        return CapabilityVerificationEvidence(
            probe_id=row["probe_id"],
            operation=WriteOperation(row["operation"]),
            target_canonical_id=row["target_canonical_id"],
            target_persistent_id=row["target_persistent_id"],
            baseline_favorited=bool(row["baseline_favorited"]),
            baseline_disliked=bool(row["baseline_disliked"]),
            verification_contract_version=int(row["verification_contract_version"]),
            verified_at=str(row["verified_at"]),
        )
    except CapabilityVerificationEvidenceRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise CapabilityVerificationEvidenceRepositoryError(
            f"corrupt verification evidence row: {error}"
        ) from error
