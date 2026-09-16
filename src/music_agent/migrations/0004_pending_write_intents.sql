CREATE TABLE pending_write_intents (
    intent_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    target_canonical_id TEXT NOT NULL,
    required_source_system TEXT NOT NULL,
    required_entity_type TEXT NOT NULL CHECK (
        required_entity_type IN ('track', 'artist', 'album', 'playlist', 'playlist_membership')
    ),
    required_external_id TEXT NOT NULL CHECK (required_external_id <> ''),
    requested_state TEXT NOT NULL CHECK (requested_state IN ('value', 'null')),
    requested_value_json TEXT,
    lifecycle_state TEXT NOT NULL CHECK (
        lifecycle_state IN ('pending', 'execution_failed', 'awaiting_readback', 'confirmed', 'readback_mismatch')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (requested_state = 'value' AND requested_value_json IS NOT NULL)
        OR (requested_state = 'null' AND requested_value_json IS NULL)
    )
);

CREATE INDEX ix_pending_write_intents_state ON pending_write_intents (lifecycle_state);
