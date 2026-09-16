-- P11-T3: admit the credential-free iTunes Store namespace for catalog discovery.

-- ``ingestion_candidates`` gains ``itunes_store`` as a third source system. SQLite cannot
-- widen a CHECK constraint in place, so the candidates table and the scopes table that
-- references it are rebuilt in place, preserving every row and every foreign key (the same
-- rebuild pattern migrations 0010 and 0017 use).

ALTER TABLE ingestion_candidates RENAME TO ingestion_candidates_old;

CREATE TABLE ingestion_candidates (
    source_system TEXT NOT NULL CHECK (source_system IN ('apple_music', 'apple_music_catalog', 'itunes_store')),
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

-- Scalar rule parity: one canonical entity holds at most one iTunes Store ID per entity type,
-- exactly like apple_music, apple_music_catalog, and isrc.

CREATE UNIQUE INDEX ux_itunes_store_scalar_identity
    ON external_identity_bindings (canonical_id, entity_type)
    WHERE source_system = 'itunes_store';
