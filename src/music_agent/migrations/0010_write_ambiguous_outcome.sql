-- Widen the write-execution state machines to admit an ambiguous command outcome.

-- A command that was dispatched but whose side effect cannot be proven absent (a subprocess
-- timeout, or a non-zero exit) is a third durable fact, distinct from a known success and a known
-- failure. ``write_execution_attempts.state`` gains ``command_unknown`` and
-- ``pending_write_intents.lifecycle_state`` gains ``outcome_unknown``. SQLite cannot widen a CHECK
-- constraint in place, so both tables -- plus ``pending_write_intent_requirements``, which
-- references the intents parent -- are rebuilt in place, preserving every row and every foreign
-- key.

ALTER TABLE pending_write_intents RENAME TO pending_write_intents_old;

CREATE TABLE pending_write_intents (
    intent_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    requested_state TEXT NOT NULL CHECK (requested_state IN ('value', 'null')),
    requested_value_json TEXT,
    lifecycle_state TEXT NOT NULL CHECK (
        lifecycle_state IN (
            'pending',
            'execution_failed',
            'awaiting_readback',
            'confirmed',
            'readback_mismatch',
            'outcome_unknown'
        )
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (requested_state = 'value' AND requested_value_json IS NOT NULL)
        OR (requested_state = 'null' AND requested_value_json IS NULL)
    )
);

INSERT INTO pending_write_intents SELECT * FROM pending_write_intents_old;

CREATE TABLE write_execution_attempts_new (
    attempt_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('started', 'command_succeeded', 'command_failed', 'command_unknown')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (intent_id) REFERENCES pending_write_intents (intent_id)
);

INSERT INTO write_execution_attempts_new SELECT * FROM write_execution_attempts;
DROP TABLE write_execution_attempts;
ALTER TABLE write_execution_attempts_new RENAME TO write_execution_attempts;

CREATE INDEX ix_write_execution_attempts_intent
    ON write_execution_attempts (intent_id);

CREATE UNIQUE INDEX ux_write_execution_attempts_active
    ON write_execution_attempts (intent_id)
    WHERE state = 'started';

CREATE TABLE pending_write_intent_requirements_new (
    intent_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('target', 'playlist', 'track')),
    canonical_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (
        entity_type IN ('track', 'artist', 'album', 'playlist', 'playlist_membership')
    ),
    external_id TEXT NOT NULL CHECK (external_id <> ''),
    PRIMARY KEY (intent_id, role),
    FOREIGN KEY (intent_id) REFERENCES pending_write_intents (intent_id) ON DELETE CASCADE
);

INSERT INTO pending_write_intent_requirements_new SELECT * FROM pending_write_intent_requirements;
DROP TABLE pending_write_intent_requirements;
ALTER TABLE pending_write_intent_requirements_new RENAME TO pending_write_intent_requirements;

DROP TABLE pending_write_intents_old;

CREATE INDEX ix_pending_write_intents_state ON pending_write_intents (lifecycle_state);
