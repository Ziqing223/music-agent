# Architecture Decisions

This directory contains a curated subset of Music Agent's early architecture decision records (ADRs) that remain useful for understanding the current implementation.

These documents are **engineering history and design rationale**, not a declaration of the current V1 product surface. The V1 product scope is intentionally narrower than the full set of capabilities explored during development. In particular, durable Apple Music mutation capabilities such as Like/Favorite, Add to Library, Rating, Create Playlist, and Edit Playlist are not presented here as V1 user-facing capabilities.

The included ADRs focus on durable architectural foundations:

- executable canonical schema;
- stable canonical and external identity;
- SQLite persistence;
- ownership-aware merge semantics;
- Apple Music read adaptation;
- snapshot and deletion evidence;
- ingestion candidate promotion;
- Artist / Album reconciliation evidence;
- Playlist / PlaylistMembership identity and reconciliation.

The later write-path and capability-probe ADRs are intentionally omitted from the public V1 snapshot. They document internal platform-feasibility and safety work rather than the public V1 capability contract, and publishing them without the full project-history context could imply support for persistent Apple Music writes that V1 does not claim.

When an ADR uses historical `P02` / `P03` phase identifiers, treat those labels only as provenance for when the decision was made. Current behavior is defined by the repository code, tests, and V1 product/evaluation contracts.
