"""Durable SQLite persistence for the preference source-of-truth (P06 S10).

A preference signal observation flows into a mutable ``preference_signal_heads`` row (one per
:class:`~music_agent.preference_persistence.SignalIdentity`) and, when it carries a semantic
``VALUE``, into an immutable append-only ``preference_evidence_revisions`` row. These two tables
are the only durable preference state; they are isolated from ``canonical_entities``, the
``artists`` / ``albums`` / ``tracks`` canonical rows, external bindings, presence, intents,
attempts, probes, and evidence.

``record_observation`` is the single write boundary. Inside one ``BEGIN IMMEDIATE`` transaction it
re-reads the current head, applies the optional expected-state (CAS) guard, and then:

- a first ``VALUE`` sighting -> one ``BASELINE`` revision at sequence ``1``;
- a ``VALUE`` that differs from the head's current semantic value -> one ``TRANSITION`` revision;
- a ``VALUE`` equal to the current semantic value -> confirmation only, no new revision;
- a ``MISSING`` / ``NULL`` -> head metadata update only, never a revision.

Because the head update and the revision append happen in the same transaction, guarded by the
revision primary key and the ``BEGIN IMMEDIATE`` write lock, the boundary is retry-safe and
restart-safe: a crash before commit rolls back to the prior head, and a retry after commit either
reproduces the same confirmation (no duplicate revision) or fails the CAS guard closed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    PREFERENCE_EVIDENCE_CONTRACT_VERSION,
    EvidenceRevision,
    RecordObservationOutcome,
    RevisionKind,
    SignalHead,
    SignalIdentity,
    decode_semantic_value,
    encode_semantic_value,
)
from music_agent.repository import _open_store_connection
from music_agent.source_observation import ObservationState, ObservedValue


class PreferencePersistenceRepositoryError(ValueError):
    code = "preference_persistence_repository_error"


class StaleHeadStateError(PreferencePersistenceRepositoryError):
    """A CAS-guarded write saw a head revision sequence different from the caller's expectation."""

    code = "stale_head_state"


class PreferencePersistenceRepository:
    """Persist signal heads and evidence revisions inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PreferencePersistenceRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def get_head(self, identity: SignalIdentity) -> SignalHead | None:
        """Return the durable head for ``identity``, or ``None`` if no observation exists."""
        identity = _require_identity(identity)
        row = self._load_head_row(identity)
        return None if row is None else _decode_head(row)

    def list_revisions(self, identity: SignalIdentity) -> tuple[EvidenceRevision, ...]:
        """Return the identity's revisions in deterministic revision-sequence order."""
        identity = _require_identity(identity)
        rows = self._connection.execute(
            """SELECT * FROM preference_evidence_revisions
            WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?
            ORDER BY revision_sequence""",
            _identity_params(identity),
        )
        return tuple(_decode_revision(row) for row in rows)

    def record_observation(
        self,
        identity: SignalIdentity,
        observed: ObservedValue,
        *,
        observed_at: str | None = None,
        event_at: str | None = None,
        provenance: str = DIRECT_OBSERVATION_PROVENANCE,
        expected_revision_sequence: int | None = None,
    ) -> RecordObservationOutcome:
        """Record one observation against ``identity`` atomically and idempotently.

        ``observed`` is the three-state observation (``MISSING`` / ``NULL`` / ``VALUE``).
        ``observed_at`` defaults to the current instant when omitted. ``event_at`` is the optional
        underlying event time. ``provenance`` labels the observation provenance. When
        ``expected_revision_sequence`` is supplied, the write fails closed with
        :class:`StaleHeadStateError` unless the head's current revision sequence matches it before
        the observation is applied (an optimistic concurrency guard).

        Returns the post-observation head and the newly appended revision, or ``None`` for the
        revision when the observation produced no revision (``MISSING`` / ``NULL`` / confirmation).
        """
        identity = _require_identity(identity)
        observed = _require_observed_value(observed)
        observed_at = _resolve_observed_at(observed_at)
        event_at = _require_optional_string(event_at, "event_at")
        provenance = _require_non_empty_string(provenance, "provenance")
        expected_revision_sequence = _require_expected_sequence(expected_revision_sequence)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            head_row = self._load_head_row(identity)
            current_sequence = 0 if head_row is None else int(head_row["current_revision_sequence"])
            if (
                expected_revision_sequence is not None
                and expected_revision_sequence != current_sequence
            ):
                raise StaleHeadStateError(
                    f"head for {identity!r} is at revision sequence {current_sequence}; "
                    f"expected {expected_revision_sequence}"
                )

            if observed.state is not ObservationState.VALUE:
                head = self._record_observation_metadata(
                    identity, observed.state, observed_at, head_row
                )
                revision: EvidenceRevision | None = None
            else:
                encoded = encode_semantic_value(observed.payload)
                if head_row is None:
                    head = self._record_baseline(
                        identity, observed.payload, observed_at, event_at, provenance
                    )
                    revision = EvidenceRevision(
                        identity,
                        1,
                        RevisionKind.BASELINE,
                        observed.payload,
                        observed_at,
                        event_at,
                        provenance,
                        PREFERENCE_EVIDENCE_CONTRACT_VERSION,
                    )
                elif head_row["current_semantic_value_json"] == encoded:
                    head = self._record_confirmation(identity, observed_at, head_row)
                    revision = None
                else:
                    new_sequence = current_sequence + 1
                    revision_kind = (
                        RevisionKind.BASELINE
                        if new_sequence == 1
                        else RevisionKind.TRANSITION
                    )
                    head = self._record_transition(
                        identity,
                        encoded,
                        observed.payload,
                        new_sequence,
                        revision_kind,
                        observed_at,
                        event_at,
                        provenance,
                        head_row,
                    )
                    revision = EvidenceRevision(
                        identity,
                        new_sequence,
                        revision_kind,
                        observed.payload,
                        observed_at,
                        event_at,
                        provenance,
                        PREFERENCE_EVIDENCE_CONTRACT_VERSION,
                    )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
            return RecordObservationOutcome(head, revision)

    # --- internal write helpers -------------------------------------------

    def _load_head_row(self, identity: SignalIdentity) -> sqlite3.Row | None:
        return self._connection.execute(
            """SELECT * FROM preference_signal_heads
            WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?""",
            _identity_params(identity),
        ).fetchone()

    def _insert_head(self, head: SignalHead) -> None:
        self._connection.execute(
            """INSERT INTO preference_signal_heads(
                target_kind, target_key, source_system, signal_path,
                current_semantic_value_json, last_observed_state, first_observed_at,
                last_observed_at, current_revision_sequence, evidence_contract_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *_identity_params(head.identity),
                None
                if head.current_semantic_value is None
                else encode_semantic_value(head.current_semantic_value),
                head.last_observed_state.value,
                head.first_observed_at,
                head.last_observed_at,
                head.current_revision_sequence,
                head.evidence_contract_version,
            ),
        )

    def _insert_revision(self, revision: EvidenceRevision) -> None:
        self._connection.execute(
            """INSERT INTO preference_evidence_revisions(
                target_kind, target_key, source_system, signal_path, revision_sequence,
                revision_kind, semantic_value_json, observed_at, event_at, provenance,
                evidence_contract_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *_identity_params(revision.identity),
                revision.revision_sequence,
                revision.revision_kind.value,
                encode_semantic_value(revision.semantic_value),
                revision.observed_at,
                revision.event_at,
                revision.provenance,
                revision.evidence_contract_version,
            ),
        )

    def _record_observation_metadata(
        self,
        identity: SignalIdentity,
        state: ObservationState,
        observed_at: str,
        head_row: sqlite3.Row | None,
    ) -> SignalHead:
        if head_row is None:
            head = SignalHead(
                identity,
                None,
                state,
                observed_at,
                observed_at,
                0,
                PREFERENCE_EVIDENCE_CONTRACT_VERSION,
            )
            self._insert_head(head)
            return head
        head = SignalHead(
            identity,
            None
            if head_row["current_semantic_value_json"] is None
            else decode_semantic_value(head_row["current_semantic_value_json"]),
            state,
            head_row["first_observed_at"],
            observed_at,
            int(head_row["current_revision_sequence"]),
            int(head_row["evidence_contract_version"]),
        )
        self._connection.execute(
            """UPDATE preference_signal_heads
            SET last_observed_state=?, last_observed_at=?, updated_at=CURRENT_TIMESTAMP
            WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?""",
            (state.value, observed_at, *_identity_params(identity)),
        )
        return head

    def _record_confirmation(
        self, identity: SignalIdentity, observed_at: str, head_row: sqlite3.Row
    ) -> SignalHead:
        head = SignalHead(
            identity,
            decode_semantic_value(head_row["current_semantic_value_json"]),
            ObservationState.VALUE,
            head_row["first_observed_at"],
            observed_at,
            int(head_row["current_revision_sequence"]),
            int(head_row["evidence_contract_version"]),
        )
        self._connection.execute(
            """UPDATE preference_signal_heads
            SET last_observed_state=?, last_observed_at=?, updated_at=CURRENT_TIMESTAMP
            WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?""",
            (ObservationState.VALUE.value, observed_at, *_identity_params(identity)),
        )
        return head

    def _record_baseline(
        self,
        identity: SignalIdentity,
        value: Any,
        observed_at: str,
        event_at: str | None,
        provenance: str,
    ) -> SignalHead:
        head = SignalHead(
            identity,
            value,
            ObservationState.VALUE,
            observed_at,
            observed_at,
            1,
            PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        self._insert_head(head)
        self._insert_revision(
            EvidenceRevision(
                identity,
                1,
                RevisionKind.BASELINE,
                value,
                observed_at,
                event_at,
                provenance,
                PREFERENCE_EVIDENCE_CONTRACT_VERSION,
            )
        )
        return head

    def _record_transition(
        self,
        identity: SignalIdentity,
        encoded: str,
        value: Any,
        new_sequence: int,
        revision_kind: RevisionKind,
        observed_at: str,
        event_at: str | None,
        provenance: str,
        head_row: sqlite3.Row,
    ) -> SignalHead:
        head = SignalHead(
            identity,
            value,
            ObservationState.VALUE,
            head_row["first_observed_at"],
            observed_at,
            new_sequence,
            PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        self._connection.execute(
            """UPDATE preference_signal_heads
            SET current_semantic_value_json=?, last_observed_state=?, last_observed_at=?,
                current_revision_sequence=?, evidence_contract_version=?, updated_at=CURRENT_TIMESTAMP
            WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?""",
            (
                encoded,
                ObservationState.VALUE.value,
                observed_at,
                new_sequence,
                PREFERENCE_EVIDENCE_CONTRACT_VERSION,
                *_identity_params(identity),
            ),
        )
        self._insert_revision(
            EvidenceRevision(
                identity,
                new_sequence,
                revision_kind,
                value,
                observed_at,
                event_at,
                provenance,
                PREFERENCE_EVIDENCE_CONTRACT_VERSION,
            )
        )
        return head


def _identity_params(identity: SignalIdentity) -> tuple[str, str, str, str]:
    return (
        identity.target.kind.value,
        identity.target.target_id,
        identity.source_system,
        identity.signal_path,
    )


def _require_identity(identity: object) -> SignalIdentity:
    if not isinstance(identity, SignalIdentity):
        raise PreferencePersistenceRepositoryError("identity must be a SignalIdentity")
    return identity


def _require_observed_value(observed: object) -> ObservedValue:
    if not isinstance(observed, ObservedValue):
        raise PreferencePersistenceRepositoryError("observed must be an ObservedValue")
    return observed


def _resolve_observed_at(observed_at: object) -> str:
    if observed_at is None:
        return datetime.now(timezone.utc).isoformat()
    return _require_non_empty_string(observed_at, "observed_at")


def _require_optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field)


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise PreferencePersistenceRepositoryError(f"{field} must be a non-empty string")
    return value


def _require_expected_sequence(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreferencePersistenceRepositoryError(
            "expected_revision_sequence must be a non-negative int or None"
        )
    return value


def _decode_head(row: sqlite3.Row) -> SignalHead:
    try:
        return SignalHead(
            identity=_decode_identity(row),
            current_semantic_value=(
                None
                if row["current_semantic_value_json"] is None
                else decode_semantic_value(row["current_semantic_value_json"])
            ),
            last_observed_state=ObservationState(row["last_observed_state"]),
            first_observed_at=row["first_observed_at"],
            last_observed_at=row["last_observed_at"],
            current_revision_sequence=row["current_revision_sequence"],
            evidence_contract_version=row["evidence_contract_version"],
        )
    except PreferencePersistenceRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise PreferencePersistenceRepositoryError(
            f"corrupt signal head row: {error}"
        ) from error


def _decode_revision(row: sqlite3.Row) -> EvidenceRevision:
    try:
        return EvidenceRevision(
            identity=_decode_identity(row),
            revision_sequence=row["revision_sequence"],
            revision_kind=RevisionKind(row["revision_kind"]),
            semantic_value=decode_semantic_value(row["semantic_value_json"]),
            observed_at=row["observed_at"],
            event_at=row["event_at"],
            provenance=row["provenance"],
            evidence_contract_version=row["evidence_contract_version"],
        )
    except PreferencePersistenceRepositoryError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise PreferencePersistenceRepositoryError(
            f"corrupt evidence revision row: {error}"
        ) from error


def _decode_identity(row: sqlite3.Row) -> SignalIdentity:
    target = PreferenceTargetReference(
        PreferenceTargetKind(row["target_kind"]), row["target_key"]
    )
    return SignalIdentity(target, row["source_system"], row["signal_path"])
