-- Durable preference persistence (P06 S10): a mutable signal head per signal identity and an
-- append-only, immutable evidence-revision history. These tables own the *durable* preference
-- source-of-truth and nothing else: recency, evidence influence, current preference projection,
-- and confidence are query-time derivations and are never materialized here.

CREATE TABLE preference_signal_heads (
    target_kind TEXT NOT NULL CHECK (target_kind IN ('track', 'artist', 'album', 'genre')),
    target_key TEXT NOT NULL CHECK (target_key <> ''),
    source_system TEXT NOT NULL CHECK (source_system <> ''),
    signal_path TEXT NOT NULL CHECK (signal_path <> ''),
    current_semantic_value_json TEXT,
    last_observed_state TEXT NOT NULL CHECK (last_observed_state IN ('missing', 'null', 'value')),
    first_observed_at TEXT NOT NULL CHECK (first_observed_at <> ''),
    last_observed_at TEXT NOT NULL CHECK (last_observed_at <> ''),
    current_revision_sequence INTEGER NOT NULL DEFAULT 0 CHECK (current_revision_sequence >= 0),
    evidence_contract_version INTEGER NOT NULL CHECK (evidence_contract_version >= 1),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (target_kind, target_key, source_system, signal_path),
    CHECK (
        (last_observed_state = 'value' AND current_semantic_value_json IS NOT NULL)
        OR (last_observed_state IN ('missing', 'null'))
    )
);

CREATE TABLE preference_evidence_revisions (
    target_kind TEXT NOT NULL,
    target_key TEXT NOT NULL,
    source_system TEXT NOT NULL,
    signal_path TEXT NOT NULL,
    revision_sequence INTEGER NOT NULL CHECK (revision_sequence >= 1),
    revision_kind TEXT NOT NULL CHECK (revision_kind IN ('baseline', 'transition')),
    semantic_value_json TEXT NOT NULL CHECK (semantic_value_json <> ''),
    observed_at TEXT NOT NULL CHECK (observed_at <> ''),
    event_at TEXT,
    provenance TEXT NOT NULL CHECK (provenance <> ''),
    evidence_contract_version INTEGER NOT NULL CHECK (evidence_contract_version >= 1),
    PRIMARY KEY (target_kind, target_key, source_system, signal_path, revision_sequence),
    FOREIGN KEY (target_kind, target_key, source_system, signal_path)
        REFERENCES preference_signal_heads (target_kind, target_key, source_system, signal_path),
    CHECK (
        (revision_kind = 'baseline' AND revision_sequence = 1)
        OR (revision_kind = 'transition' AND revision_sequence >= 2)
    )
);

CREATE INDEX ix_preference_evidence_revisions_head
    ON preference_evidence_revisions (target_kind, target_key, source_system, signal_path, revision_sequence);

-- Evidence revisions are immutable and append-only. The repository exposes no update/delete
-- path, and these triggers make the invariant a hard SQLite guarantee even against a raw write.
CREATE TRIGGER trg_preference_evidence_revisions_immutable_update
    BEFORE UPDATE ON preference_evidence_revisions
BEGIN
    SELECT RAISE(ABORT, 'preference evidence revisions are immutable');
END;

CREATE TRIGGER trg_preference_evidence_revisions_immutable_delete
    BEFORE DELETE ON preference_evidence_revisions
BEGIN
    SELECT RAISE(ABORT, 'preference evidence revisions are immutable');
END;
