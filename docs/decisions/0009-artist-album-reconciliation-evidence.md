# 0009 — Artist/Album Reconciliation Evidence Contract

- Status: Accepted
- Stage: P03.9.1
- Date: 2026-08-15

## Context

P03.8 established that a Track ingestion Candidate cannot be promoted while its Artist and Album
relations remain unresolved, and it made relation states change only through the explicit
`update_relation_resolution(...)` boundary. The reconciliation algorithm itself was deferred.

This slice defines the reconciliation *evidence and decision contract*: what source evidence is
strong enough to legally produce `RESOLVED_TO_ARTISTS`, `RESOLVED_TO_ALBUM`, or `RESOLVED_ABSENT`,
and what evidence must keep a relation `UNRESOLVED`. It does not implement automatic Artist/Album
matching; that is explicitly out of scope.

## Source evidence inventory

The current Apple Music read adapter (`src/music_agent/apple_music.py`) reads only Track scalar
facts — name, played count, favorited, disliked, and rating. Its `TRACK_OBSERVATION_FIELDS`
includes `artist_ids` and `album_id`, but neither is ever populated: the bundled AppleScript reads
no artist or album identity, and `_map_field` never writes those paths. Every such path remains
`MISSING`.

Concretely:

1. The Apple Music Track read path provides **no** Artist evidence. It does not emit an artist
   name, an artist persistent ID, or any artist relation data.
2. It provides **no** Album evidence. It does not emit an album name, an album persistent ID, or any
   album relation data.
3. There is **no** reliable Artist external ID available from the current source adapter.
4. There is **no** reliable Album external ID available from the current source adapter.
5. `artist_name` / `album_name` do not enter the Candidate at all. The adapter rejects them as
   unverified diagnostic metadata (Decision 0006 decision 6), and they are not in
   `TRACK_SOURCE_FACT_PATHS`.
6. Canonical Artists carry `id`, `name`, and `external_ids.apple_music_persistent_id`; canonical
   Albums carry the same plus `artist_ids` and `release_date`.
7. `external_identity_bindings` already supports every entity type, including Artist and Album.
   The schema does not restrict bindings to Tracks.
8. Relation *missing* and explicit relation *absent* are distinct states. The existing relation
   model already encodes this: `AlbumRelationState.RESOLVED_ABSENT` is explicit absence, while
   `UNRESOLVED` means no decision has been made. A missing source field maps to `MISSING`, never to
   absence.
9. Multi-artist Track source evidence is not present at all, so it cannot be split reliably.

The current production source therefore cannot support automatic Artist/Album reconciliation by
name or by stable related-entity identity. This is a source-capability limitation, not an
implementation failure. The contract is still specified in full so that future source capability
(for example, an Apple Music relation read that emits stable Artist/Album persistent IDs) can be
handled safely without inventing a name matcher.

## Evidence classes

### Strong identity evidence

An exact `ExternalIdentityKey(source_system, entity_type, external_id)` that resolves through a
durable binding to an existing canonical entity of the matching entity type is strong evidence. It
may produce `RESOLVED_TO_ARTISTS` (single resolved Artist) or `RESOLVED_TO_ALBUM`. Rebinding is
never performed and canonical IDs are never derived from external IDs.

The durable lookup happens at the persistence boundary against `external_identity_bindings`, not in
the evaluator. The evaluator receives an already-resolved
`BoundExternalIdentityEvidence(external_identity, canonical_id)` for a found binding, or an
`unbound_identity` key for a negative lookup result. The canonical model's
`external_ids.apple_music_persistent_id` is a projection reconstructed from the durable binding; it
is never scanned to infer which canonical ID an external key resolves to. `external_identity_bindings`
remains the physical authority and `external_ids` remains a canonical projection.

### Explicit canonical decision

A caller-supplied canonical Artist/Album ID is explicit reconciliation authority, not name
matching. It resolves only when the entity type is correct, the canonical entity exists, and
(Artists only) the IDs are non-empty and duplicate-free. This is consumed through the existing
`update_relation_resolution(...)` boundary; no second staging update API is introduced.

### Display name only

A display name alone is insufficient and always leaves the relation `UNRESOLVED`, even when it
matches an existing canonical entity exactly and uniquely. Display names are not stable identity,
and are confounded by same-named entities, compilation/featured/collaboration displays,
localization and formatting, and multi-artist display strings.

### Explicit album absence

Only explicit, semantically reliable absence evidence produces `RESOLVED_ABSENT`. A missing field,
a JSON null, or an empty diagnostic is interpreted as `MISSING`/unknown by the existing contract
and is never promoted to absence. The current adapter provides no explicit absence evidence, so
`RESOLVED_ABSENT` is reachable today only through an explicit caller decision.

### Artist absence

There is no `RESOLVED_ABSENT` Artist state in the existing contract, and none is added. With no
reliable Artist evidence, the Artist relation remains `UNRESOLVED`.

## Conflict policy

When two strong evidence sources disagree — for example, an exact external identity resolving to
one Artist while an explicit canonical decision names another — no winner is guessed. The
evaluator returns a typed conflict and fails closed. Existing reconciliation decisions are never
silently overwritten; source rediscovery still does not reset reconciliation.

## Explicit reconciliation workflow (P03.9.2)

A thin orchestration module `src/music_agent/reconciliation_workflow.py` composes the evaluator
with the existing durable mutation boundary. It is a command boundary, not a source-sync path.

- Explicit reconciliation is a command: the caller supplies explicit canonical decisions
  (canonical Artist IDs, a canonical Album ID, or an explicit Album-absence flag), never
  source-derived name or identity evidence.
- The evaluator validates every requested decision against the current canonical model before any
  mutation. A decision that does not reach `RESOLVED` (for example a dangling target, a duplicate
  Artist ID, or a wrong entity type) blocks the whole call.
- Durable mutation reuses `CandidateStagingRepository.update_relation_resolution(...)`; no second
  relation persistence API exists. The workflow never writes `ingestion_candidates` payloads
  directly and never duplicates the Candidate JSON codec.
- All requested relation decisions validate before mutation. There is no partial Artist-only or
  Album-only write: if any requested decision fails, no relation state changes.
- A requested `UNRESOLVED` decision does not write any resolved state. In this workflow the only
  requested outcomes are `RESOLVED` or a conflict, so "insufficient" is expressed by not requesting
  that channel and leaving the relation unchanged.
- An existing resolution is preserved when its channel is not requested. Supplying new decisions
  can change it, but omitting a decision never clears it back to `UNRESOLVED`.
- Repeating an identical explicit decision is idempotent: the final Candidate relation state is
  unchanged and source facts and staging scopes are untouched.
- Album absence is reachable only through an explicit absence decision, never inferred from
  missing, null, or empty evidence.
- Automatic source reconciliation remains unavailable. Actual name/fuzzy matching and related
  entity ingestion remain deferred.

## Decisions

1. A new pure domain layer `src/music_agent/reconciliation.py` classifies evidence and produces
   typed decisions. It never touches SQLite, never mutates a Candidate or the canonical model,
   never creates a canonical entity, and never generates a canonical ID.
2. Results are typed, not booleans: `ReconciliationOutcome` (`RESOLVED` / `UNRESOLVED` /
   `CONFLICT`) plus a `ReconciliationReason` code and, only when resolved, the existing
   `ArtistRelationResolution` or `AlbumRelationResolution` value.
3. Reason codes cover `exact_external_identity`, `explicit_canonical_decision`,
   `explicit_absence`, `display_name_insufficient`, `missing_relation_evidence`,
   `unknown_external_identity`, `wrong_entity_type`, `dangling_canonical_reference`,
   `duplicate_canonical_id`, and `conflicting_strong_evidence`.
4. External identity lookup is a persistence-boundary responsibility; the evaluator only
   validates the already-resolved result. It never scans canonical `external_ids` projections to
   re-derive a binding.
5. Automatic name/fuzzy reconciliation is not adopted. Exact-name, case-insensitive, normalized,
   fuzzy, token-similarity, Levenshtein, and LLM semantic matching are all out of scope for this
   slice.
6. The evaluator does not create canonical Artists or Albums, and does not turn an unknown
   external identity into new entity ingestion.
7. Persistence integration is deferred. This slice adds no migration, changes no canonical schema,
   and changes no staging schema.

## Deferred work

- Wiring automatic source-derived evidence into the explicit workflow (stable related
  Artist/Album identity from the Apple Music adapter, or an explicit absence signal), so the
  workflow can be driven by source capability rather than caller decisions.
- New Artist/Album ingestion and canonical ID generation for related entities.
- Conflict resolution UX and any reset/invalidation command for existing reconciliation decisions.
- Any name/fuzzy/display-name matching algorithm.
