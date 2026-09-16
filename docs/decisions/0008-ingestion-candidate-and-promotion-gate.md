# 0008 — Ingestion Candidate and Promotion Gate

- Status: Accepted
- Stage: P03.8
- Date: 2026-08-15

## Context

P03.8's initial new-entity ingestion feasibility gate found that P03.6 source records cannot form a
truthful canonical Track while Artist and Album relations remain unresolved. Empty or null
canonical relations cannot be used as placeholders for unresolved source facts.

## Decisions

1. An `IngestionCandidate` is an in-memory domain object outside the canonical model. It is not a
   partial canonical Track and is never passed to `CanonicalRepository.save_model(...)`.
2. Candidate identity is exactly `ExternalIdentityKey(source_system, entity_type, external_id)`.
   P03.8.1 supports Apple Music Track candidates only; no candidate or canonical UUID is generated.
3. Promotion-relevant source facts reuse `ObservedValue` and preserve `MISSING`, `NULL`, and
   `VALUE`. Shared-owned tags are not source facts.
4. Artist relations remain `unresolved` or become `resolved_to_artists` with a non-empty set of
   distinct, existing canonical Artist IDs. P03.8.1 does not invent a no-Artist domain state.
5. Album relations remain `unresolved`, become `resolved_to_album` with an existing canonical
   Album ID, or become explicitly `resolved_absent`. Only the last state becomes canonical null.
6. Missing genres block promotion. `VALUE([])` means known-empty; missing is never converted to an
   empty canonical array.
7. Missing nullable scalar source facts initialize canonical null during construction under the
   established unknown-scalar contract; this is canonical initialization, not a source NULL
   observation. Missing library-state values likewise become null, never false or zero.
8. Shared-owned `agent_metadata.tags` initializes to `[]`, meaning that the Shared Model has not
   assigned tags. It is not an Apple Music observation.
9. Promotion evaluation returns calculated typed blockers. Construction is a separate pure step,
   requires an injected valid Track ID, and validates the resulting Track structurally and within
   a complete candidate graph. Non-promotable candidates never emit partial canonical Tracks.
10. P03.8.2 stores Candidates durably in the same SQLite store as canonical data but in separate
    staging tables. The external identity triple is the Candidate primary key and authority;
    Candidate payloads never repeat identity or contain a canonical Track ID.
11. Staging payloads use deterministic standard-library JSON with explicit tagged codecs for
    `ObservedValue` and relation states. Blockers and promotability remain derived and are never
    cached in storage.
12. Candidate scope evidence is stored separately from canonical source presence and supports
    multiple idempotent scopes per Candidate. Candidate payload and new scope evidence update in
    one transaction.
13. No Candidate delete or lifecycle timestamp exists yet. Actual Artist/Album reconciliation,
    canonical ID generation, Candidate retirement, and atomic entity/binding/presence promotion
    remain deferred.
14. Source facts and relation reconciliation state have separate authority. Rediscovery through
    `stage_candidate(...)` refreshes source facts and accumulates scope evidence but preserves any
    existing reconciliation decision. Relation states change only through the explicit,
    transactional `update_relation_resolution(...)` boundary; the reconciliation algorithm and
    any reset/invalidation command remain deferred.
15. Promotion reloads the durable Candidate and current canonical model, then re-evaluates the
    Promotion Gate at execution time. Promotability is never read from cached staging state.
16. A new Track ID is generated through the stable identity layer only after the execution-time
    gate passes. Track construction continues to use `build_promotable_track(...)`; promotion does
    not duplicate canonical construction or relation reconciliation logic.
17. Canonical model persistence, external identity binding, staging-scope transfer to canonical
    `present` source presence, and Candidate/scope retirement commit in one SQLite transaction.
    Any failure rolls back every durable effect and retains the Candidate with all scope evidence.
18. Every staging scope transfers exactly once to canonical `present` membership. A Candidate with
    no scope creates no presence row; binding alone remains distinct from scope membership.
19. Successful promotion retires the Candidate and its staging scopes. Repeating promotion by the
    same external identity returns the existing canonical Track and never creates a duplicate.
20. A pre-existing binding returns `already_bound` without canonical mutation or Candidate cleanup.
    Stale already-bound Candidate cleanup remains deferred rather than silently discarding staging
    information. Artist/Album reconciliation algorithms and new Artist/Album creation remain
    deferred.
