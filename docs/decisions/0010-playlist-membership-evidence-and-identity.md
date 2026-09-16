# 0010 — Playlist / Membership Source Evidence & Identity Contract

- Status: Accepted
- Stage: P03.10.1
- Date: 2026-08-15

## Context

P03.10 introduces Playlist and PlaylistMembership reconciliation. Before any reconciliation
workflow can be written, the evidence the production Apple Music source can actually observe must
be established from real code, and the identity contract must be defined so that the pre-existing
invariant is not violated: a PlaylistMembership has an independent `pm_` canonical identity, and
`(playlist_id, track_id)` is NOT a membership identity. The same Track may appear multiple times
in the same Playlist.

## Source evidence inventory (from real code)

The production read path is `src/music_agent/apple_music.py`. Its bundled AppleScript queries only
`library playlist 1` for a single Track by persistent ID and returns name, played count,
favorited, disliked, and rating (plus optional timezone datetimes from a structured runner). The
adapter maps those into `SourceObservation` for one Track.

Concretely, for Playlist and PlaylistMembership:

1. **No Playlist identity evidence.** There is no adapter that reads Playlists, no bundled
   AppleScript that enumerates a Playlist's persistent ID or name, and no
   `TRACK_OBSERVATION_FIELDS`-style path for Playlists. The adapter reads a Track only.
2. **No Playlist ordering evidence.** No source path emits a Track's position or order within any
   Playlist.
3. **No membership occurrence identity.** No source path emits any per-occurrence discriminator
   (such as an Apple Music playlist-item persistent ID) that would distinguish two occurrences of
   the same Track inside one Playlist.
4. **No `added_at` membership evidence.** The only `added_at`-like field the adapter maps is a
   Track-level `library_state.added_to_library_at`; it is a Track library fact, not a
   membership-occurrence fact, and it is not mapped by the bundled AppleScript.
5. **No complete/authoritative Playlist snapshot.** `src/music_agent/snapshot.py` explicitly
   rejects any scope whose `entity_type` is not `EntityType.TRACK` ("P03.7 supports only Track
   snapshots"), and rejects any source system other than `apple_music`. There is no snapshot
   machinery for Playlists or PlaylistMemberships, so no completeness or deletion-authority
   guarantee exists for them today.
6. **Track persistent ID is available only as a by-hand lookup.** The Track persistent ID that
   `read_track(...)` consumes is the same stable persistent ID already used for Track external
   identity, but the current adapter does not *discover* it from within a Playlist — a caller must
   already know which Track persistent ID to read.

The canonical model (schema and fixture) already supports Playlists and PlaylistMemberships:
`playlists` carries `id` / `external_ids.apple_music_persistent_id` / `name`; `playlist_memberships`
carries `id` (`pm_` prefix), `playlist_id`, `track_id`, `position`, and `added_at`. The fixture
demonstrates the invariant directly: one Playlist contains the same Track twice, at `position 0`
and `position 1`, with two distinct `pm_` IDs.

## Identity authority

Playlist identity follows the same boundary as Artist/Album reconciliation (Decision 0009).
`external_identity_bindings` is the physical authority for a Playlist external persistent ID, and
the canonical `external_ids.apple_music_persistent_id` is only a projection reconstructed from
that binding. A Playlist identity resolves only through a caller-completed durable binding lookup
(strong identity evidence) or an explicit canonical decision. A display name alone is
insufficient and never resolves a Playlist, even when it is unique. The evaluator never scans
canonical `external_ids` to re-derive a binding.

## Membership identity limitation

A PlaylistMembership has an independent `pm_` identity. `(playlist_id, track_id)` is not a
membership identity. `position` and `added_at` are observations, not identity: `position` is only
an ordering observation and shifts when the Playlist is reordered, and `added_at` can be null and
is not unique per occurrence. Therefore a source that provides only an ordered Track list has no
occurrence identity and cannot safely create or match a PlaylistMembership. The contract fails
closed to `UNRESOLVED` (`AMBIGUOUS_OCCURRENCE_IDENTITY`) rather than fabricating identity from
position or `(playlist, track)`.

## Duplicate Track occurrence semantics

Two occurrences of the same Track in the same Playlist are two distinct memberships only when each
carries its own stable occurrence identity. With two distinct occurrence identities, the same
`(playlist, track)` resolves as two distinct membership results. With no occurrence identity, the
occurrence is ambiguous and stays `UNRESOLVED`. With a *colliding* occurrence identity within one
evaluated set, every colliding occurrence is forced to `CONFLICT`
(`DUPLICATE_OCCURRENCE_IDENTITY`) — the input is invalid and no membership is produced.

## Ordering evidence is an observation

`position` (and any future ordering evidence) is captured as an `ObservedValue` observation only.
It is never part of identity and never participates in matching. Ordering alone cannot re-identify
an occurrence across snapshots.

## Complete snapshot capability

There is no complete/authoritative Playlist snapshot capability today. `apply_snapshot` is
Track-only, so no Playlist omission can become `confirmed_deleted`, and no membership absence can
be inferred. Completeness and deletion authority for Playlists remain future source capability.

## What can resolve, and what must stay unresolved

- **RESOLVED** (Playlist identity): a caller-completed durable binding to an existing canonical
  Playlist, or an explicit canonical Playlist decision. Reason codes `EXACT_EXTERNAL_IDENTITY` /
  `EXPLICIT_CANONICAL_DECISION`.
- **IDENTIFIED** (membership occurrence): a resolved Playlist plus a resolved member Track plus a
  distinct occurrence identity. Reason code `EXACT_EXTERNAL_IDENTITY`. `IDENTIFIED` is NOT a
  canonical `PlaylistMembership` resolution: it carries no `pm_` ID and never asserts that a
  canonical membership exists or matches.
- **UNRESOLVED**: unknown external identity, display-name-only, missing identity evidence,
  missing/unknown Track identity, or an occurrence without occurrence identity
  (`AMBIGUOUS_OCCURRENCE_IDENTITY`).
- **CONFLICT**: wrong entity type, dangling canonical reference, conflicting strong evidence, or a
  duplicate occurrence identity within one evaluated set.

The evaluator never creates a Playlist or PlaylistMembership, never generates a `pm_` ID, and never
mutates the canonical model.

## Canonical membership resolution is a separate stage

Resolving a source occurrence to an existing canonical `PlaylistMembership` is not part of this
slice. It requires, later, either a caller-completed durable external binding from occurrence
identity to an existing `pm_` canonical ID, or an explicit canonical membership decision whose
target exists with the correct entity type. The occurrence evaluator does not perform that
resolution: a non-empty, unique occurrence identity is only proof that the source occurrence is
distinguishable, never authority to produce a canonical membership.

## Current production capability limitation

The current production Apple Music read path provides no Playlist identity, no Playlist ordering,
no membership occurrence identity, no membership `added_at`, and no complete/authoritative Playlist
snapshot. Automatic source-derived PlaylistMembership reconciliation is therefore **Deferred**, not
because the contract is hard but because the source has no occurrence identity to feed it. This is
a source-capability limitation, not an implementation bug, and it must not be papered over with
name matching or `(playlist_id, track_id)` identity.

## Decisions

1. A new pure domain layer `src/music_agent/playlist_reconciliation.py` classifies Playlist
   identity and membership-occurrence evidence and produces typed decisions. It touches no SQLite,
   never mutates the canonical model, never creates a Playlist or PlaylistMembership, and never
   generates a canonical ID.
2. Playlist identity reuses `ReconciliationOutcome` (`RESOLVED` / `UNRESOLVED` / `CONFLICT`).
   Membership occurrence decisions use a dedicated `PlaylistReconciliationOutcome` whose positive
   state is `IDENTIFIED`, not `RESOLVED`, so a source occurrence identity can never be misread as a
   canonical membership resolution. A
   Playlist-specific reason enum adds membership-only codes:
   `MISSING_TRACK_IDENTITY`, `UNKNOWN_TRACK_IDENTITY`, `AMBIGUOUS_OCCURRENCE_IDENTITY`, and
   `DUPLICATE_OCCURRENCE_IDENTITY`, alongside the Playlist identity codes.
3. Playlist identity uses the same caller-completed `BoundExternalIdentityEvidence` carrier as
   Artist/Album; durable lookup is a persistence-boundary responsibility and the evaluator never
   scans canonical `external_ids`.
4. Membership occurrence identity is a source-level `MembershipOccurrenceIdentity
   (source_system, external_id)` discriminator, distinct from any canonical binding. The current
   model has no `external_ids` projection for memberships, so this is not persisted.
5. `position` and `added_at` are carried as `ObservedValue` observations and never contribute to
   identity.
6. A set-evaluation boundary `evaluate_membership_occurrences(...)` detects duplicate occurrence
   identities within one set and fails those occurrences closed to `CONFLICT`.
7. No SQLite, no migration 0004, no canonical schema change, no Playlist/Membership creation, no
   `pm_` generation, and no Apple Music write/read are introduced by this slice.

## Explicit PlaylistMembership workflow (P03.10.2)

A command module `src/music_agent/playlist_membership_workflow.py` provides the explicit,
caller-driven creation boundary for canonical PlaylistMemberships. It is not a source-sync path.

- `create_playlist_membership(database_path, *, playlist_id, track_id, position, added_at=None)`
  validates the Playlist and Track references, then persists one new canonical membership with a
  fresh `pm_` ID generated by the P03.3 `generate_canonical_id(EntityType.PLAYLIST_MEMBERSHIP)`.
- Playlist and Track references must already exist as canonical entities of the correct namespace
  (`pl_` / `trk_`) and type. The workflow never creates a Playlist or a Track and never matches by
  name.
- `position` is a non-negative integer and `added_at` is an optional timezone-qualified ISO 8601
  string. Both are relation data, not identity; neither participates in the `pm_` identity.
- The same Playlist + Track may be used to create multiple memberships; each call produces a
  distinct `pm_` ID, so duplicate same-Track occurrences become independent memberships.
- Persistence reuses `CanonicalRepository.save_model`, which validates the full model (structural
  + graph) and writes atomically in one transaction. A failed request never leaves a partial
  membership: reference validation happens before `save_model`, and `save_model` itself validates
  before mutating.
- Membership creation is not a whole-model *replacement*: `save_model` upserts
  `playlist_memberships` by primary key and never deletes membership rows, so a stale-model save
  cannot drop a concurrently-created membership. Existing memberships are preserved, and two
  callers that persist from stale snapshots both retain their distinct `pm_` memberships.

### Retry / idempotency contract

`create_playlist_membership` is **non-idempotent**. Every invocation generates a fresh `pm_` ID and
inserts one membership. There is no command/request identity in this slice, and the workflow does
not deduplicate on `(playlist_id, track_id)`, position, or `added_at`. A caller that retries the
same logical create produces a second distinct canonical membership — which is correct when the
retry is a legitimate second occurrence, but means the boundary cannot distinguish "retry of the
same create" from "second occurrence of the same Track." Callers that require idempotency must
supply their own request identity and reconcile it above this boundary; that is out of scope.

### What this slice does NOT do

The workflow never consumes source occurrence identity, never turns an `IDENTIFIED` occurrence
into a membership, never derives a `pm_` ID from `(playlist_id, track_id)` or position/added_at,
and never performs source-driven automatic reconciliation. It introduces no migration 0004 and
changes no schema: the existing `playlist_memberships` table already stores memberships.

## Deferred work

- A real Apple Music Playlist read adapter that emits Playlist persistent IDs, member Track
  persistent IDs, per-occurrence identity, ordering, and `added_at`.
- A complete/authoritative Playlist snapshot path (extension of `apply_snapshot` beyond Track).
- Caller-supplied request identity to make explicit membership creation idempotent.
- Automatic source-derived membership reconciliation; it remains blocked until occurrence identity
  exists.
