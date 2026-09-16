"""Learning application (P08.6): apply ProposedPreferenceUpdate to durable P06 state, safely.

This module is the *application* layer of the feedback-learning loop -- the first layer allowed
to mutate P06 persisted learning state. It applies a
:class:`~music_agent.learning_policy.ProposedPreferenceUpdate` produced by the corrected policy
v2 to the real P06 preference system through the existing
:class:`~music_agent.preference_persistence_repository.PreferencePersistenceRepository` write
boundary, and journals every application in the append-only ``learning_applications`` table
(migration 0014). It never re-implements P06 inference, never touches source-of-truth heads
(``apple_music`` and other sources), and never invents confidence, weights, or deltas.

Application semantics
---------------------

``EVIDENCE_OBSERVATION``
    One P06 semantic observation is recorded against the proposed
    :class:`~music_agent.preference_persistence.SignalIdentity` (target, ``feedback_learning``,
    frozen path) with the proposed ``VALUE``, the feedback event times, and the evidence-class
    provenance label (``feedback_learning:explicit`` / ``feedback_learning:implicit``). The
    direction of the feedback therefore becomes durable P06 evidence that the existing P06 query
    derivation reads; the explicit-vs-implicit class survives durably as the revision
    provenance. The magnitude that reaches recommendation scoring is P06's own query-time
    injected calibration policy -- never duplicated here.

``ATTRIBUTION_EXCLUSION``
    No P06 observation is recorded. The exclusion is journaled as its own application row
    carrying the excluded aspect (``attribution_kind`` / ``attribution_id`` /
    ``attribution_relation``), so "do not attribute this feedback to this aspect" is durable and
    visible at the learning boundary without ever becoming ordinary preference evidence -- in
    particular it can never become negative evidence.

``NO_EFFECT`` never reaches this layer (the policy produces ``None``); a
:class:`~music_agent.learning_policy.ProposedPreferenceUpdate` that does not match the current
policy and contract versions fails closed.

Replay safety
-------------

``feedback_id`` is the primary key of the application journal: the same feedback observation can
never be applied twice (``DuplicateLearningApplicationError``), and the journal is append-only
with SQLite immutability triggers. Application order is evidence-first, journal-second: if a
crash interrupts between the P06 evidence write and the journal insert, re-applying the same
proposal is safe because P06's ``record_observation`` is idempotent for the same semantic value
(a confirmation produces no new revision), after which the journal row is written. A concurrent
double-apply fails closed on the primary key.

Readback
--------

``get_application`` / ``list_applications`` return immutable :class:`AppliedLearningRecord`
views of the journal; the learned evidence itself is read back through the existing P06
repository and query interfaces. The journal stores the full application provenance
(``feedback_id``, all upstream policy/contract versions, target, signal identity, evidence-class
provenance, attribution), so every durable learned state can be traced back through the whole
pipeline: feedback -> interpretation policy -> learning-effect policy -> learning policy ->
applied evidence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from music_agent.feedback_contract import AttributionRelation, FeedbackAttribution
from music_agent.learning_policy import (
    LEARNING_POLICY_CONTRACT_VERSION,
    ProposedPreferenceUpdate,
    ProposedUpdateKind,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.repository import _open_store_connection


class LearningApplicationRepositoryError(ValueError):
    code = "learning_application_repository_error"


class DuplicateLearningApplicationError(LearningApplicationRepositoryError):
    """The feedback observation was already applied; the journal has no overwrite path."""

    code = "duplicate_learning_application"


@dataclass(frozen=True, slots=True)
class AppliedLearningRecord:
    """One immutable journal view of an applied proposed update."""

    feedback_id: str
    proposal_kind: ProposedUpdateKind
    target: PreferenceTargetReference
    signal_source_system: str | None
    signal_path: str | None
    provenance: str | None
    attribution: FeedbackAttribution | None
    interpretation_policy_version: int
    effect_policy_version: int
    learning_policy_version: int
    learning_policy_contract_version: int
    applied_at: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.feedback_id, "feedback_id")
        if not isinstance(self.proposal_kind, ProposedUpdateKind):
            raise LearningApplicationRepositoryError(
                "proposal_kind must be a ProposedUpdateKind"
            )
        if not isinstance(self.target, PreferenceTargetReference):
            raise LearningApplicationRepositoryError(
                "target must be a PreferenceTargetReference"
            )
        for label, value in (
            ("signal_source_system", self.signal_source_system),
            ("signal_path", self.signal_path),
            ("provenance", self.provenance),
        ):
            if value is not None:
                _require_non_empty_string(value, label)
        if self.attribution is not None and not isinstance(
            self.attribution, FeedbackAttribution
        ):
            raise LearningApplicationRepositoryError(
                "attribution must be a FeedbackAttribution or None"
            )
        for label, value in (
            ("interpretation_policy_version", self.interpretation_policy_version),
            ("effect_policy_version", self.effect_policy_version),
            ("learning_policy_version", self.learning_policy_version),
            ("learning_policy_contract_version", self.learning_policy_contract_version),
        ):
            _require_positive_int(value, label)
        _require_non_empty_string(self.applied_at, "applied_at")


class LearningApplicationRepository:
    """Apply proposed updates to P06 state and journal them in the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)
        self._preference_repository = PreferencePersistenceRepository(self.database_path)

    def close(self) -> None:
        self._preference_repository.close()
        self._connection.close()

    def __enter__(self) -> LearningApplicationRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def apply(
        self, proposal: ProposedPreferenceUpdate, *, applied_at: str | None = None
    ) -> AppliedLearningRecord:
        """Apply one proposed update to durable P06 state and journal it.

        ``proposal`` must be a policy-v2 :class:`ProposedPreferenceUpdate`` stamped with the
        current learning-policy contract version; anything else fails closed. An already-applied
        ``feedback_id`` fails closed with :class:`DuplicateLearningApplicationError`` before any
        write. An ``EVIDENCE_OBSERVATION`` proposal records one P06 semantic observation through
        the existing :class:`~music_agent.preference_persistence_repository.PreferencePersistenceRepository``
        boundary (never a hand-rolled SQL write); an ``ATTRIBUTION_EXCLUSION`` proposal records
        no P06 observation -- the exclusion is journaled only. ``applied_at`` is the application
        instant as an ISO-8601 string and defaults to the current UTC instant (the journal is
        operational metadata, mirroring the P06 observed_at default precedent).
        """
        proposal = _require_proposal(proposal)
        applied_at = _resolve_applied_at(applied_at)

        existing = self._connection.execute(
            "SELECT 1 FROM learning_applications WHERE feedback_id=?",
            (proposal.feedback_id,),
        ).fetchone()
        if existing is not None:
            raise DuplicateLearningApplicationError(
                f"feedback {proposal.feedback_id!r} is already applied"
            )

        if proposal.kind is ProposedUpdateKind.EVIDENCE_OBSERVATION:
            self._preference_repository.record_observation(
                proposal.signal_identity,
                proposal.proposed_value,
                observed_at=proposal.observed_at,
                event_at=proposal.event_at,
                provenance=proposal.provenance,
            )

        attribution = proposal.attribution
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                "SELECT 1 FROM learning_applications WHERE feedback_id=?",
                (proposal.feedback_id,),
            ).fetchone()
            if existing is not None:
                raise DuplicateLearningApplicationError(
                    f"feedback {proposal.feedback_id!r} is already applied"
                )
            self._connection.execute(
                """INSERT INTO learning_applications(
                    feedback_id, proposal_kind, target_kind, target_id,
                    signal_source_system, signal_path, provenance,
                    attribution_kind, attribution_id, attribution_relation,
                    interpretation_policy_version, effect_policy_version,
                    learning_policy_version, learning_policy_contract_version,
                    applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal.feedback_id,
                    proposal.kind.value,
                    proposal.target.kind.value,
                    proposal.target.target_id,
                    (
                        proposal.signal_identity.source_system
                        if proposal.signal_identity is not None
                        else None
                    ),
                    (
                        proposal.signal_identity.signal_path
                        if proposal.signal_identity is not None
                        else None
                    ),
                    proposal.provenance,
                    (
                        None
                        if attribution is None
                        else attribution.aspect.kind.value
                    ),
                    None if attribution is None else attribution.aspect.target_id,
                    None if attribution is None else attribution.relation.value,
                    proposal.effect.interpretation.policy_version,
                    proposal.effect.policy_version,
                    proposal.policy_version,
                    proposal.contract_version,
                    applied_at,
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

        return AppliedLearningRecord(
            feedback_id=proposal.feedback_id,
            proposal_kind=proposal.kind,
            target=proposal.target,
            signal_source_system=(
                proposal.signal_identity.source_system
                if proposal.signal_identity is not None
                else None
            ),
            signal_path=(
                proposal.signal_identity.signal_path
                if proposal.signal_identity is not None
                else None
            ),
            provenance=proposal.provenance,
            attribution=attribution,
            interpretation_policy_version=proposal.effect.interpretation.policy_version,
            effect_policy_version=proposal.effect.policy_version,
            learning_policy_version=proposal.policy_version,
            learning_policy_contract_version=proposal.contract_version,
            applied_at=applied_at,
        )

    def get_application(self, feedback_id: str) -> AppliedLearningRecord | None:
        """Return the application journaled for ``feedback_id``, or ``None`` if absent."""
        feedback_id = _require_feedback_id(feedback_id)
        row = self._connection.execute(
            "SELECT * FROM learning_applications WHERE feedback_id=?", (feedback_id,)
        ).fetchone()
        return None if row is None else _record_from_row(row)

    def list_applications(self) -> tuple[AppliedLearningRecord, ...]:
        """Return every journaled application ordered by ``applied_at``, then ``feedback_id``."""
        rows = self._connection.execute(
            "SELECT * FROM learning_applications ORDER BY applied_at ASC, feedback_id ASC"
        )
        return tuple(_record_from_row(row) for row in rows)


def _record_from_row(row: sqlite3.Row) -> AppliedLearningRecord:
    attribution = None
    if row["attribution_id"] is not None:
        attribution = FeedbackAttribution(
            PreferenceTargetReference(
                PreferenceTargetKind(row["attribution_kind"]), row["attribution_id"]
            ),
            AttributionRelation(row["attribution_relation"]),
        )
    return AppliedLearningRecord(
        feedback_id=row["feedback_id"],
        proposal_kind=ProposedUpdateKind(row["proposal_kind"]),
        target=PreferenceTargetReference(
            PreferenceTargetKind(row["target_kind"]), row["target_id"]
        ),
        signal_source_system=row["signal_source_system"],
        signal_path=row["signal_path"],
        provenance=row["provenance"],
        attribution=attribution,
        interpretation_policy_version=row["interpretation_policy_version"],
        effect_policy_version=row["effect_policy_version"],
        learning_policy_version=row["learning_policy_version"],
        learning_policy_contract_version=row["learning_policy_contract_version"],
        applied_at=row["applied_at"],
    )


def _require_proposal(proposal: object) -> ProposedPreferenceUpdate:
    if not isinstance(proposal, ProposedPreferenceUpdate):
        raise LearningApplicationRepositoryError(
            "proposal must be a ProposedPreferenceUpdate"
        )
    if proposal.contract_version != LEARNING_POLICY_CONTRACT_VERSION:
        raise LearningApplicationRepositoryError(
            f"unsupported learning policy contract version {proposal.contract_version}"
        )
    if proposal.policy_version != 2:
        raise LearningApplicationRepositoryError(
            f"unsupported learning policy version {proposal.policy_version}"
        )
    if proposal.kind is ProposedUpdateKind.ATTRIBUTION_EXCLUSION:
        if (
            proposal.attribution is None
            or proposal.attribution.relation is not AttributionRelation.EXCLUDED
        ):
            raise LearningApplicationRepositoryError(
                "an attribution-exclusion proposal requires an EXCLUDED attribution"
            )
    return proposal


def _resolve_applied_at(applied_at: str | None) -> str:
    if applied_at is None:
        applied_at = datetime.now(timezone.utc).isoformat()
    _require_non_empty_string(applied_at, "applied_at")
    return applied_at


def _require_feedback_id(feedback_id: object) -> str:
    if not isinstance(feedback_id, str) or feedback_id == "":
        raise LearningApplicationRepositoryError(
            "feedback_id must be a non-empty string"
        )
    return feedback_id


def _require_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise LearningApplicationRepositoryError(f"{field} must be a non-empty string")


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LearningApplicationRepositoryError(f"{label} must be a positive int")
    return value
