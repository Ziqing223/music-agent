-- Durable feedback observation history (P08.2): one immutable row per FeedbackObservation.
-- The authoritative persisted unit is the canonical encode_feedback_observation JSON text carried
-- in encoded_observation; feedback_id, kind, contract_version, observed_at, and duplicate_key are
-- mirrored as columns so observations can be enumerated, deduplicated, and ordered without
-- decoding. duplicate_key is the canonical TEXT form of feedback_duplicate_key computed by the
-- repository (UTC-canonicalized), and its UNIQUE index makes "one stored row per observed event" a
-- hard SQLite guarantee even across distinct feedback_ids. History is append-only: the repository
-- exposes no update/delete path and the triggers below make immutability a hard SQLite guarantee.

CREATE TABLE feedback_observations (
    feedback_id TEXT PRIMARY KEY CHECK (feedback_id LIKE 'fbk_%'),
    encoded_observation TEXT NOT NULL CHECK (encoded_observation <> ''),
    kind TEXT NOT NULL CHECK (kind <> ''),
    contract_version INTEGER NOT NULL CHECK (contract_version >= 1),
    observed_at TEXT NOT NULL CHECK (observed_at <> ''),
    duplicate_key TEXT NOT NULL CHECK (duplicate_key <> ''),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_feedback_observations_observed_at ON feedback_observations (observed_at);

CREATE UNIQUE INDEX ux_feedback_observations_duplicate_key ON feedback_observations (duplicate_key);

CREATE TRIGGER trg_feedback_observations_immutable_update
    BEFORE UPDATE ON feedback_observations
BEGIN
    SELECT RAISE(ABORT, 'feedback observations are immutable');
END;

CREATE TRIGGER trg_feedback_observations_immutable_delete
    BEFORE DELETE ON feedback_observations
BEGIN
    SELECT RAISE(ABORT, 'feedback observations are immutable');
END;
