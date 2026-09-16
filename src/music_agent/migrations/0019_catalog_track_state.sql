-- P15-S3-S1: durable long-term state per canonical Catalog track.
--
-- ``catalog_track_state`` is derived, re-buildable memory -- never a second source of truth for
-- canonical identity (canonical_entities / external_identity_bindings / source_entity_presence)
-- and never a replacement for recommendation history (recommendation_runs). One row per
-- canonical Catalog track carries:
--
--   * discovery memory: first/last discovery instant, occurrence count, and a bounded
--     normalized-term summary. Backfilled rows carry NULL first/last_discovered_at and
--     discovery_count 0 -- explicitly "no durable historical discovery events recorded",
--     never "track was never discovered".
--   * recommendation-history projection: first/last recommended instant and occurrence count
--     derived from persisted successful recommendation items only. Request input targets
--     ($.request.context.preference_inputs), failed/empty runs (refused before persistence),
--     and rejected candidates (never become items) are structurally excluded.
--
-- Backfill is idempotent (INSERT OR IGNORE plus recomputed UPDATE); the script runs once via
-- schema_migrations, and re-running it would repeat the same result, never accumulate.

CREATE TABLE catalog_track_state (
    canonical_id           TEXT PRIMARY KEY CHECK (canonical_id LIKE 'trk_%'),
    entity_type            TEXT NOT NULL DEFAULT 'track' CHECK (entity_type = 'track'),
    source_system          TEXT NOT NULL
        CHECK (source_system IN ('itunes_store', 'apple_music_catalog')),
    first_discovered_at    TEXT,
    last_discovered_at     TEXT,
    discovery_count        INTEGER NOT NULL DEFAULT 0 CHECK (discovery_count >= 0),
    discovery_terms_json   TEXT
        CHECK (discovery_terms_json IS NULL OR json_valid(discovery_terms_json)),
    first_recommended_at   TEXT,
    last_recommended_at    TEXT,
    recommendation_count   INTEGER NOT NULL DEFAULT 0 CHECK (recommendation_count >= 0),
    updated_at             TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (canonical_id, entity_type) REFERENCES canonical_entities (id, entity_type)
);

CREATE INDEX ix_catalog_track_state_first_recommended ON catalog_track_state (first_recommended_at);
CREATE INDEX ix_catalog_track_state_last_discovered ON catalog_track_state (last_discovered_at);

-- A. One row per already-bound canonical Catalog track. Discovery instants and terms are NOT
-- fabricated: first/last_discovered_at stay NULL, discovery_count stays 0, no historical terms.
-- source_system is derived deterministically, preferring the catalog-scope presence row, then
-- the catalog binding, preferring itunes_store in each step.
INSERT OR IGNORE INTO catalog_track_state (canonical_id, source_system, discovery_count, recommendation_count)
SELECT
    b.canonical_id,
    COALESCE(
        (SELECT p.source_system
         FROM source_entity_presence p
         WHERE p.canonical_id = b.canonical_id
           AND p.entity_type = 'track'
           AND p.scope_key = 'catalog'
         ORDER BY CASE p.source_system WHEN 'itunes_store' THEN 0 ELSE 1 END
         LIMIT 1),
        (SELECT b2.source_system
         FROM external_identity_bindings b2
         WHERE b2.canonical_id = b.canonical_id
           AND b2.entity_type = 'track'
           AND b2.source_system IN ('itunes_store', 'apple_music_catalog')
         ORDER BY CASE b2.source_system WHEN 'itunes_store' THEN 0 ELSE 1 END
         LIMIT 1)
    ),
    0,
    0
FROM external_identity_bindings b
WHERE b.entity_type = 'track'
  AND b.source_system IN ('itunes_store', 'apple_music_catalog')
GROUP BY b.canonical_id;

-- B. Recommendation-history projection for rows in this table. Only persisted successful
-- recommendation items count: the $.items[*].candidate.target entries whose kind is 'track',
-- matched against this table's canonical ids. Runs without items, request preference inputs, and
-- runs that were never persisted contribute nothing. first/last store the verbatim authoritative
-- produced_at text of the chronologically earliest/latest matching run -- ordering is by true
-- instant (julianday normalizes the mixed UTC offsets real runs carry), never by lexical text.
-- recommendation_count counts item occurrences (each appearance of the track as a
-- recommendation item), so the projection stays re-buildable from recommendation_runs.
UPDATE catalog_track_state
SET first_recommended_at = (
        SELECT r.produced_at
        FROM recommendation_runs r, json_each(r.encoded_result, '$.items') i
        WHERE json_extract(i.value, '$.candidate.target.kind') = 'track'
          AND json_extract(i.value, '$.candidate.target.target_id') = catalog_track_state.canonical_id
        ORDER BY julianday(r.produced_at) ASC, r.produced_at ASC
        LIMIT 1
    ),
    last_recommended_at = (
        SELECT r.produced_at
        FROM recommendation_runs r, json_each(r.encoded_result, '$.items') i
        WHERE json_extract(i.value, '$.candidate.target.kind') = 'track'
          AND json_extract(i.value, '$.candidate.target.target_id') = catalog_track_state.canonical_id
        ORDER BY julianday(r.produced_at) DESC, r.produced_at DESC
        LIMIT 1
    ),
    recommendation_count = (
        SELECT COUNT(*)
        FROM recommendation_runs r, json_each(r.encoded_result, '$.items') i
        WHERE json_extract(i.value, '$.candidate.target.kind') = 'track'
          AND json_extract(i.value, '$.candidate.target.target_id') = catalog_track_state.canonical_id
    )
WHERE EXISTS (
        SELECT 1
        FROM recommendation_runs r, json_each(r.encoded_result, '$.items') i
        WHERE json_extract(i.value, '$.candidate.target.kind') = 'track'
          AND json_extract(i.value, '$.candidate.target.target_id') = catalog_track_state.canonical_id
    );