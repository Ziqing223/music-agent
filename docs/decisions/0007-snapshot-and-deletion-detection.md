# 0007 — Snapshot and Deletion Detection

- Status: Accepted
- Stage: P03.7
- Date: 2026-08-15

## Context

P03.7 implements the sealed P02 rule that source omission becomes deletion evidence only within a
known-complete, deletion-authoritative scope, while canonical identity and source presence remain
separate.

## Decisions

1. A snapshot declares `(source_system, entity_type, scope_key)`, completeness (`complete`,
   `partial`, or `unknown`), and an independent deletion-authority flag. Completeness alone never
   grants deletion authority.
2. P03.7 accepts synthetic Apple Music Track snapshots. Records are identified only by
   `ExternalIdentityKey`; duplicate identities and identities outside the declared scope fail the
   snapshot. Unknown source records remain unresolved and never create entities or bindings.
3. A durable source binding resolves an observed record but does not prove membership in an
   arbitrary snapshot scope. Deletion candidates are canonical Tracks already tracked by a durable
   presence row in the same `(source_system, entity_type, scope_key)`. First observation establishes
   `present`; only a later omission from that same complete, deletion-authoritative scope can become
   `confirmed_deleted`. Partial, unknown, and non-authoritative omission provide no durable update.
4. Observed known records reuse `SourceObservation` and `merge_observations`; no snapshot-specific
   ownership merge exists. Observation confirms durable `present` even when metadata is unchanged.
5. Durable presence stores only confirmed `present` and `confirmed_deleted`. Runtime `missing` and
   `unknown` outcomes cannot overwrite an existing durable confirmed state.
6. Migration v2 adds `source_entity_presence`, keyed by source, entity type, canonical ID, and
   scope. Presence references the canonical entity and is persisted together with a validated
   candidate model in one Repository transaction.
7. Confirmed deletion preserves the canonical entity and external binding. Reappearance upserts
   the same bound canonical ID from `confirmed_deleted` back to `present`.
8. Real Library enumeration, real completeness proof, new-entity ingestion, physical deletion,
   binding deletion, reconciliation, tombstones, and source-presence history remain deferred.
