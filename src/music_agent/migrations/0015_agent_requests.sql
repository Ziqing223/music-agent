-- Durable agent request journal (P09.4): one immutable row per executed (or refused) agent
-- request, keyed by the opaque req_ request identity. The row records who called what with
-- which payload, and the exact outcome envelope returned -- so a replayed request id can be
-- answered from the journal without re-executing, and mismatched replays fail closed.
-- request_text and result_text are the full canonical agent JSON texts; payload_text is the
-- canonical deterministic payload text, the replay-equality key. No domain user state lives
-- here: canonical music, preference, recommendation, feedback, and learning state remain in the
-- established shared tables. The journal is append-only: the repository exposes no update/delete
-- path and the triggers below make immutability a hard SQLite guarantee.

CREATE TABLE agent_requests (
    request_id TEXT PRIMARY KEY CHECK (request_id LIKE 'req_%'),
    client_id TEXT NOT NULL CHECK (client_id LIKE 'agt_%'),
    model_id TEXT NOT NULL CHECK (model_id <> ''),
    tool_name TEXT NOT NULL CHECK (tool_name <> ''),
    contract_version INTEGER NOT NULL CHECK (contract_version >= 1),
    request_text TEXT NOT NULL CHECK (request_text <> ''),
    payload_text TEXT NOT NULL CHECK (payload_text <> ''),
    outcome TEXT NOT NULL CHECK (outcome IN (
        'ok',
        'invalid_request',
        'replay_conflict',
        'tool_not_supported',
        'unknown_client',
        'permission_denied',
        'not_execution_ready',
        'execution_error'
    )),
    result_text TEXT NOT NULL CHECK (result_text <> ''),
    error_code TEXT,
    error_message TEXT,
    issued_at TEXT NOT NULL CHECK (issued_at <> ''),
    completed_at TEXT NOT NULL CHECK (completed_at <> ''),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_agent_requests_client_id ON agent_requests (client_id);
CREATE INDEX ix_agent_requests_issued_at ON agent_requests (issued_at);

CREATE TRIGGER trg_agent_requests_immutable_update
    BEFORE UPDATE ON agent_requests
BEGIN
    SELECT RAISE(ABORT, 'agent requests are immutable');
END;

CREATE TRIGGER trg_agent_requests_immutable_delete
    BEFORE DELETE ON agent_requests
BEGIN
    SELECT RAISE(ABORT, 'agent requests are immutable');
END;
