# 0002 — Executable Canonical Schema

- Status: Accepted
- Stage: P03.2
- Date: 2026-08-15

## Context

P02 defined the five core entity shapes and their null/reference semantics but did not select an
executable schema technology, strictness policy, concrete entity prefixes, partial release-date
representation, or cross-entity validation mechanism. P03.1 already fixed IDs as entity-prefixed
random opaque UUIDs and fixed Apple Music rating as `integer | null` in `0..100`.

## Decisions

1. JSON Schema Draft 2020-12 is the single executable structural schema. Python validation uses
   `jsonschema`'s `Draft202012Validator` with explicit format checking.
2. Canonical objects have a stable required shape and reject unexpected fields. Unknown scalars use
   `null`; multi-value fields use arrays; optional single references use a value or `null`.
3. Concrete internal ID prefixes are `trk_`, `art_`, `alb_`, `pl_`, and `pm_`, followed by a UUID.
   This validates physical shape only; ID generation remains outside P03.2.
4. A non-null `release_date` preserves known precision directly as `YYYY`, `YYYY-MM`, or
   `YYYY-MM-DD`. All digits and separators use ASCII canonical representation; bare years are in
   `0001..9999`, and month/day values pass calendar validation. No missing month or day is
   fabricated. This is a reversible P03 implementation decision for the precision issue left open
   by P02.
5. Structural validation and fixture graph validation are separate production functions. JSON
   Schema validates local shape; Python graph validation checks canonical ID uniqueness and
   references across the synthetic fixture.
6. Playlist membership uniqueness is based on its own canonical ID. Repeated
   `(playlist_id, track_id)` pairs are valid.

## Carried forward, not newly decided here

- The five entity types, fields, reference directions, null semantics, date/timestamp distinction,
  and standalone PlaylistMembership come from P02.
- Rating `integer | null` with range `0..100` and entity-prefixed random opaque UUIDs were accepted
  in P03.1 Decision 0001.

## Deferred work

- ID generation and lifecycle enforcement.
- Persistence, repositories, SQLite constraints, and migrations.
- Source-observation `MISSING / NULL / VALUE` runtime semantics.
- External identity resolution, merging, adapters, sync, renderers, and exporters.
