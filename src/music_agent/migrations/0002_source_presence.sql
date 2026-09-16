CREATE TABLE source_entity_presence (
    source_system TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    scope_key TEXT NOT NULL CHECK (scope_key <> ''),
    presence TEXT NOT NULL CHECK (presence IN ('present', 'confirmed_deleted')),
    PRIMARY KEY (source_system, entity_type, canonical_id, scope_key),
    FOREIGN KEY (canonical_id, entity_type) REFERENCES canonical_entities (id, entity_type)
);
