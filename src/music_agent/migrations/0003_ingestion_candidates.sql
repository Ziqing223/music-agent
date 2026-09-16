CREATE TABLE ingestion_candidates (
    source_system TEXT NOT NULL CHECK (source_system = 'apple_music'),
    entity_type TEXT NOT NULL CHECK (entity_type = 'track'),
    external_id TEXT NOT NULL CHECK (external_id <> ''),
    payload TEXT NOT NULL,
    PRIMARY KEY (source_system, entity_type, external_id)
);

CREATE TABLE ingestion_candidate_scopes (
    source_system TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    scope_key TEXT NOT NULL CHECK (scope_key <> ''),
    PRIMARY KEY (source_system, entity_type, external_id, scope_key),
    FOREIGN KEY (source_system, entity_type, external_id)
        REFERENCES ingestion_candidates (source_system, entity_type, external_id)
);
