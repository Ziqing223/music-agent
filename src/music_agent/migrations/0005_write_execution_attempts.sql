CREATE TABLE write_execution_attempts (
    attempt_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('started', 'command_succeeded', 'command_failed')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (intent_id) REFERENCES pending_write_intents (intent_id)
);

CREATE INDEX ix_write_execution_attempts_intent
    ON write_execution_attempts (intent_id);

CREATE UNIQUE INDEX ux_write_execution_attempts_active
    ON write_execution_attempts (intent_id)
    WHERE state = 'started';
