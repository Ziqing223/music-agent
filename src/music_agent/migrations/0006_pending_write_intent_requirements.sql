CREATE TABLE pending_write_intent_requirements (
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

INSERT INTO pending_write_intent_requirements
    (intent_id, role, canonical_id, source_system, entity_type, external_id)
SELECT
    intent_id,
    'target',
    target_canonical_id,
    required_source_system,
    required_entity_type,
    required_external_id
FROM pending_write_intents;

ALTER TABLE pending_write_intents DROP COLUMN target_canonical_id;
ALTER TABLE pending_write_intents DROP COLUMN required_source_system;
ALTER TABLE pending_write_intents DROP COLUMN required_entity_type;
ALTER TABLE pending_write_intents DROP COLUMN required_external_id;
