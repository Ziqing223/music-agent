-- Durable learning-application journal (P08.6): one immutable row per applied
-- ProposedPreferenceUpdate, keyed by the applied feedback observation identity.
-- The row records what was applied where (proposal kind, P06 target, feedback_learning signal
-- identity, evidence-class provenance, attribution columns, and the upstream policy/contract
-- versions), so replay protection and provenance are hard SQLite guarantees. The authoritative
-- learned *evidence* lives in the P06 preference heads/revisions written by the application; the
-- authoritative feedback record lives in feedback_observations (0013). The journal is
-- append-only: the repository exposes no update/delete path and the triggers below make
-- immutability a hard SQLite guarantee.

CREATE TABLE learning_applications (
    feedback_id TEXT PRIMARY KEY CHECK (feedback_id LIKE 'fbk_%'),
    proposal_kind TEXT NOT NULL CHECK (proposal_kind IN ('evidence_observation', 'attribution_exclusion')),
    target_kind TEXT NOT NULL CHECK (target_kind <> ''),
    target_id TEXT NOT NULL CHECK (target_id <> ''),
    signal_source_system TEXT,
    signal_path TEXT,
    provenance TEXT,
    attribution_kind TEXT,
    attribution_id TEXT,
    attribution_relation TEXT,
    interpretation_policy_version INTEGER NOT NULL CHECK (interpretation_policy_version >= 1),
    effect_policy_version INTEGER NOT NULL CHECK (effect_policy_version >= 1),
    learning_policy_version INTEGER NOT NULL CHECK (learning_policy_version >= 1),
    learning_policy_contract_version INTEGER NOT NULL CHECK (learning_policy_contract_version >= 1),
    applied_at TEXT NOT NULL CHECK (applied_at <> ''),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_learning_applications_applied_at ON learning_applications (applied_at);

CREATE TRIGGER trg_learning_applications_immutable_update
    BEFORE UPDATE ON learning_applications
BEGIN
    SELECT RAISE(ABORT, 'learning applications are immutable');
END;

CREATE TRIGGER trg_learning_applications_immutable_delete
    BEFORE DELETE ON learning_applications
BEGIN
    SELECT RAISE(ABORT, 'learning applications are immutable');
END;
