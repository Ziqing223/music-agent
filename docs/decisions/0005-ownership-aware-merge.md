# 0005 — Ownership-aware Merge Core

- Status: Accepted
- Stage: P03.5
- Date: 2026-08-15

## Context

P03.5 makes the sealed P02 refresh, ownership, missing/null, and source-presence rules executable
without introducing a real source adapter or changing canonical persistence.

## Decisions

1. `ObservedValue` is a strict tagged value with distinct `MISSING`, `NULL`, and `VALUE` states.
   Missing and null cannot carry payloads; value requires a non-null payload, so `False`, `0`, and
   `[]` remain explicit values while `VALUE(None)` is rejected.
2. `SourceObservation` targets an already-resolved `(entity_type, canonical_id, source_system)`.
   P03.5 supports only `apple_music` as a refresh source; every other source is rejected before
   field ownership is evaluated and is not implicitly treated as `shared_model`. Merge never
   resolves by metadata, creates identities, or changes `id` and `external_ids`.
3. Source presence is one of `present`, `missing`, `confirmed_deleted`, or `unknown`. Only `present`
   applies fields. Other states preserve the canonical entity; omission never implies deletion.
4. One centralized `(entity_type, field_path) -> Authority` policy controls refresh. Current
   executable source fields are Apple Music-owned except `Track.agent_metadata.tags`, which is
   Shared Model-owned and is preserved with an explicit `not_authoritative_preserved` outcome.
5. Every observed field reports `updated`, `unchanged`, `missing_preserved`, or
   `not_authoritative_preserved`. Actual changes are also returned as stable entity-qualified field
   paths for future consumers.
6. Batch merge deep-copies the current model, applies all observations, then reruns structural and
   graph validation. Any invalid observation or candidate fails the whole merge without mutating
   the input model.
7. Confirmed source deletion does not physically delete canonical entities. Durable source-presence
   persistence, complete-snapshot deletion inference, Derived invalidation, pending intents, and
   command/readback behavior remain deferred.
