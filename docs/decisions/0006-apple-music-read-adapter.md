# 0006 — Apple Music Read Adapter

- Status: Accepted
- Stage: P03.6
- Date: 2026-08-15

## Context

P01 verified Music.app plus AppleScript as the local personal-data path, persistent-ID lookup, and
reading Track name, played count/date, date added, favorited, disliked, and rating. P03.6 turns the
supported subset into a production read-only adapter for already-bound canonical Tracks.

## Decisions

1. P03.6 refreshes only known canonical Tracks with an existing durable Apple Music external
   binding. It never creates IDs, ingests new entities, resolves by name, or adds bindings.
2. `OsascriptMusicRunner` is replaceable and invokes a read-only AppleScript with argv through
   `subprocess` without `shell=True`. `AppleMusicSourceAdapter` owns JSON parsing and conversion;
   `refresh_known_track` owns Repository → adapter → merge → Repository orchestration.
3. Raw reads distinguish `found`, `confirmed_not_found`, and `lookup_failed`. A successful direct
   persistent-ID query with zero matches is classified as confirmed not found; process errors,
   timeouts, and malformed output are lookup failures and never deletion evidence.
4. The bundled Music.app path maps name, played count, favorited, disliked, and rating. A structured
   runner may also supply timezone-qualified ISO 8601 `date_added` and `played_date`; naive or
   malformed datetimes fail mapping. The bundled AppleScript leaves dates unavailable because P01
   did not preserve a locale-independent datetime serialization pattern.
5. AppleScript `missing value`, absent properties, and JSON null map conservatively to `MISSING`,
   not canonical `NULL`. False and zero remain explicit `VALUE` values; rating is rejected outside
   `0..100` and counts are never defaulted.
6. Track artist/album source strings are diagnostics only and do not become `artist_ids` or
   `album_id`. Relations, duration, genres, numbering, release date, composer, skip count, and other
   unverified fields are emitted as `MISSING`. Shared-owned tags are never emitted by the adapter.
7. Successful changed candidates are saved by `CanonicalRepository`; unchanged, lookup-failed,
   malformed, invalid, unbound, and unresolved cases do not save. Repeated identical refresh is
   idempotent.
8. All Music.app writes, full Library enumeration, new-entity ingestion, complete-snapshot deletion
   inference, Artist/Album reconciliation, PlaylistMembership reconciliation, command/readback,
   and durable source-presence persistence remain deferred.
