-- P11.1: admit Apple Music Catalog Track identities alongside Music.app persistent IDs.

-- ``ingestion_candidates`` gains ``apple_music_catalog`` as a second source system. SQLite cannot
-- widen a CHECK constraint in place, so the candidates table and the scopes table that references
-- it are rebuilt in place, preserving every row and every foreign key (the same rebuild pattern
-- migration 0010 uses).

ALTER TABLE ingestion_candidates RENAME TO ingestion_candidates_old;

CREATE TABLE ingestion_candidates (
    source_system TEXT NOT NULL CHECK (source_system IN ('apple_music', 'apple_music_catalog')),
    entity_type TEXT NOT NULL CHECK (entity_type = 'track'),
    external_id TEXT NOT NULL CHECK (external_id <> ''),
    payload TEXT NOT NULL,
    PRIMARY KEY (source_system, entity_type, external_id)
);

INSERT INTO ingestion_candidates SELECT * FROM ingestion_candidates_old;

ALTER TABLE ingestion_candidate_scopes RENAME TO ingestion_candidate_scopes_old;

CREATE TABLE ingestion_candidate_scopes (
    source_system TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    scope_key TEXT NOT NULL CHECK (scope_key <> ''),
    PRIMARY KEY (source_system, entity_type, external_id, scope_key),
    FOREIGN KEY (source_system, entity_type, external_id)
        REFERENCES ingestion_candidates (source_system, entity_type, external_id)
);

INSERT INTO ingestion_candidate_scopes SELECT * FROM ingestion_candidate_scopes_old;

DROP TABLE ingestion_candidate_scopes_old;
DROP TABLE ingestion_candidates_old;

-- One canonical entity holds at most one external ID per scalar source system. The existing
-- apple_music rule (migration 0001) now extends to apple_music_catalog and isrc: a canonical
-- Track may carry one Music.app persistent ID, one Catalog Song ID, and one ISRC, never two of
-- any one kind.

CREATE UNIQUE INDEX ux_apple_music_catalog_scalar_identity
    ON external_identity_bindings (canonical_id, entity_type)
    WHERE source_system = 'apple_music_catalog';

CREATE UNIQUE INDEX ux_isrc_scalar_identity
    ON external_identity_bindings (canonical_id, entity_type)
    WHERE source_system = 'isrc';
