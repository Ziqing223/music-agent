-- Durable recommendation run history (P07.5): one immutable row per ranked recommendation run.
-- The authoritative persisted unit is the canonical encode_recommendation_result JSON text carried
-- in encoded_result; run_id, contract_version, and produced_at are mirrored as columns so runs can
-- be enumerated and ordered without decoding. History is append-only: the repository exposes no
-- update/delete path and the triggers below make immutability a hard SQLite guarantee.

CREATE TABLE recommendation_runs (
    run_id TEXT PRIMARY KEY CHECK (run_id LIKE 'rcm_%'),
    encoded_result TEXT NOT NULL CHECK (encoded_result <> ''),
    contract_version INTEGER NOT NULL CHECK (contract_version >= 1),
    produced_at TEXT NOT NULL CHECK (produced_at <> ''),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_recommendation_runs_produced_at ON recommendation_runs (produced_at);

CREATE TRIGGER trg_recommendation_runs_immutable_update
    BEFORE UPDATE ON recommendation_runs
BEGIN
    SELECT RAISE(ABORT, 'recommendation runs are immutable');
END;

CREATE TRIGGER trg_recommendation_runs_immutable_delete
    BEFORE DELETE ON recommendation_runs
BEGIN
    SELECT RAISE(ABORT, 'recommendation runs are immutable');
END;
