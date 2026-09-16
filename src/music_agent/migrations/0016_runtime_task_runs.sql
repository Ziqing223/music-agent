-- Durable automation task-run journal (P10.4): one immutable row per completed or failed
-- automation task run, keyed by the opaque rn_ run identity. Runtime observability only --
-- "last successful refresh", "last failed refresh", per-task status -- no domain user state
-- lives here: canonical music, preference, recommendation, feedback, learning, and agent
-- request state remain in the established shared tables. The table is append-only: the
-- repository exposes no update/delete path and the triggers below make immutability a hard
-- SQLite guarantee. detail_json carries the task-specific JSON-safe outcome shape (e.g. the
-- refresh-cycle counts); a failed run always carries error, a completed run never does.

CREATE TABLE runtime_task_runs (
    run_id TEXT PRIMARY KEY CHECK (run_id LIKE 'rn_%'),
    task_name TEXT NOT NULL CHECK (task_name <> ''),
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    error TEXT,
    detail_json TEXT NOT NULL CHECK (detail_json <> ''),
    started_at TEXT NOT NULL CHECK (started_at <> ''),
    finished_at TEXT NOT NULL CHECK (finished_at <> ''),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((status = 'completed' AND error IS NULL) OR (status = 'failed' AND error IS NOT NULL))
);

CREATE INDEX ix_runtime_task_runs_task_started
    ON runtime_task_runs (task_name, started_at);

CREATE TRIGGER trg_runtime_task_runs_immutable_update
    BEFORE UPDATE ON runtime_task_runs
BEGIN
    SELECT RAISE(ABORT, 'runtime task runs are immutable');
END;

CREATE TRIGGER trg_runtime_task_runs_immutable_delete
    BEFORE DELETE ON runtime_task_runs
BEGIN
    SELECT RAISE(ABORT, 'runtime task runs are immutable');
END;
