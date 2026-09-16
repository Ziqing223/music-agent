# 0003 — Stable Identity Layer

- Status: Accepted
- Stage: P03.3
- Date: 2026-08-15

## Context

P02 requires stable, opaque, system-owned canonical identity and conservative external identity
resolution. P03.2 fixed five entity namespaces and structural ID shape but deferred generation and
binding behavior. P03.3 implements those semantics in process without introducing persistence.

## Decisions

1. New canonical IDs use the entity prefix plus a random standard-library UUIDv4 in canonical
   hyphenated lowercase form, for example `trk_2f4c5f85-a95a-4a59-8666-b2e47bbad879`.
2. `EntityType` is the single code-level entity definition, mapped one-to-one to `trk_`, `art_`,
   `alb_`, `pl_`, and `pm_`.
3. External identity keys are immutable `(source_system, entity_type, external_id)` values.
   External IDs are non-empty opaque strings: no case folding, Unicode normalization, metadata
   inference, or canonical-ID derivation occurs.
4. First binding creates a mapping; repeating the same binding is an idempotent no-op. Rebinding an
   existing key to a different canonical ID raises typed `identity_conflict` and preserves the old
   mapping.
5. Multiple external keys may bind to one canonical entity when entity types agree. Canonical
   Artist and Album IDs may exist without any external binding.
6. PlaylistMembership IDs are generated independently from playlist, track, position, and time.
7. The registry is intentionally non-persistent. Permanent non-reuse across restarts, deletion, or
   database rebuild is deferred to durable persistence.

## Deferred work

- Durable identity tables, tombstones, and cross-process non-reuse.
- Real Apple Music identity resolution and source adapter behavior.
- Artist/Album name reconciliation and duplicate-candidate detection.
- PlaylistMembership refresh reconciliation.
- Force rebind, external-ID reassignment reconciliation, merge, and unlink workflows.
