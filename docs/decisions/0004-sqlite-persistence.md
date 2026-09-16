# 0004 — SQLite Persistence

- Status: Accepted
- Stage: P03.4
- Date: 2026-08-15

## Context

P03.2 established executable canonical validation and P03.3 established in-process stable identity.
P03.4 persists those verified contracts across closing and reopening one SQLite store without
introducing source synchronization or ownership policy.

## Decisions

1. Persistence uses Python 3.12 standard-library `sqlite3`, with foreign keys explicitly enabled on
   every repository connection. No ORM is introduced.
2. A normalized `canonical_entities` registry reserves every canonical ID and its immutable entity
   type. Concrete entity tables reference that registry; no public physical-delete API exists.
3. Tracks, Artists, Albums, Playlists, and PlaylistMemberships use dedicated tables. Artist
   relations, genres, and tags use ordered child tables; they are not stored as whole-entity JSON.
4. External identity bindings are the physical authority for canonical external IDs. The scalar
   `external_ids.apple_music_persistent_id` field is reconstructed from the Apple Music binding.
   A partial unique index permits at most one such scalar Apple Music binding per canonical entity,
   while bindings from multiple source systems may still target the same entity.
5. A complete canonical model save through `save_model(...)` validates structural and graph
   contracts before mutation, then writes
   identities, entities, relations, memberships, and external bindings in one transaction. Any
   failure rolls back the entire save.
6. `load_model()` reconstructs the complete canonical model. Saving an existing canonical ID
   updates current canonical fields and ordered relations without changing identity. Repeating the
   same model save is idempotent.
7. Migration SQL is a packaged runtime resource. Store-level `schema_migrations` owns schema
   version; records do not repeat a schema version.
8. Within one surviving SQLite store, canonical ID reservation, entity type, external bindings, and
   conflict behavior survive restart. Global uniqueness after database deletion or across separate
   stores is not claimed.

## Deferred work

- Physical deletion, tombstones, archival policy, and recovery from deleted/corrupt stores.
- Source observation, refresh, ownership-aware merge, and Apple Music adapter behavior.
- Artist/Album and PlaylistMembership source reconciliation.
- Pending intents, write/readback, renderers, parsers, and derived lifecycle.
