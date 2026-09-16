"""P11.1/P11-T2: Apple Music Catalog discovery -> durable staging -> automatic promotion.

One coherent path from catalog search results into the canonical Music Agent store:

    catalog search (adapter, read-only)
    -> for each CatalogTrack: identity dedupe / known-library exclusion
    -> durable staging as IngestionCandidate (the track's source system, catalog scope)
    -> authoritative Artist / Album relation resolution (Catalog relationship IDs only)
    -> automatic promotion through the existing sealed promote_staged_track path

Identity safety semantics:

- The Catalog Song ID is a distinct namespace (``apple_music_catalog``); it never shares the
  Music.app persistent ID namespace (``apple_music``), so one canonical Track can hold one ID of
  each kind (scalar rule, migration 0017).
- Repeated discovery converges: an already-bound catalog ID resolves to its existing canonical
  Track and stages nothing.
- A catalog hit whose ISRC already resolves to a canonical Track with confirmed library presence
  (``library_tracks`` scope) is excluded from "new music" discovery: reliable identity evidence
  proves it is already known.
- A catalog hit whose ISRC resolves to a canonical Track without library presence resolves to
  that same canonical Track (no duplicate identity is ever staged).
- Artist/album relations resolve ONLY through authoritative Apple Music Catalog Artist / Album
  IDs carried by the catalog payload. Display names never resolve a relation, and no fuzzy
  matching is authoritative. A canonical Artist / Album is reused when its Catalog identity is
  already bound, otherwise the minimum canonical entity is created with that binding. Missing
  identity evidence fails closed: the candidate stays staged and promotion stays blocked with an
  explicit reason.

This module never mutates Music.app, never plays or adds anything; canonical writes are one
atomic batch commit carrying the promoted Tracks plus the minimum Artist / Album entities
their relations require (P20 Performance Fix 01; per-entry durable semantics match the sealed
single-promotion path exactly).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Protocol

from music_agent.apple_music_catalog import CatalogTrack
from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.catalog_track_state_repository import CatalogTrackStateRepository
from music_agent.identity import (
    EXTERNAL_ID_FIXTURE_KEYS,
    EntityType,
    ExternalIdentityKey,
    IdentityConflictError,
    generate_canonical_id,
)
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    ArtistRelationResolution,
    IngestionCandidate,
    build_promotable_track,
    evaluate_track_promotion,
)
from music_agent.library_sync import LIBRARY_TRACKS_SCOPE
from music_agent.promotion import StagedCandidateNotFoundError
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue, SourcePresence

logger = logging.getLogger("music_agent.catalog_ingestion")

CATALOG_SCOPE = "catalog"


class CatalogIngestionError(ValueError):
    code = "catalog_ingestion_error"


class CatalogIngestStatus(StrEnum):
    STAGED = "staged"
    PROMOTED = "promoted"
    STAGED_BLOCKED = "staged_blocked"
    IDENTITY_CONFLICT = "identity_conflict"
    ALREADY_BOUND = "already_bound"
    LIBRARY_KNOWN = "library_known"
    STAGING_FAILED = "staging_failed"


@dataclass(frozen=True, slots=True)
class CatalogIngestOutcome:
    """One catalog hit's durable-ingestion outcome."""

    catalog_id: str
    status: CatalogIngestStatus
    canonical_id: str | None = None
    error: str | None = None
    blocker_codes: tuple[str, ...] = ()


@dataclass
class _DiscoveryBatchState:
    """In-memory working state for one ``ingest`` batch (P20 Performance Fix 01).

    The canonical model is loaded at most once per batch and persisted through one
    atomic batch commit when anything changed. Identity lookups stay SQL-first with a
    batch-local binding overlay, so in-batch creations (tracks, their ISRC secondaries,
    artists, albums) resolve exactly like a sequentially persisted batch would.
    """

    model: dict[str, Any] | None = None
    batch_bindings: dict[tuple[str, EntityType, str], str] = field(default_factory=dict)
    model_changed: bool = False
    promotions: list[tuple[ExternalIdentityKey, str]] = field(default_factory=list)


class CatalogSearchSource(Protocol):
    def search(self, term: str, limit: int) -> tuple[CatalogTrack, ...]: ...


def default_catalog_search_source() -> CatalogSearchSource:
    """The runtime catalog discovery source: credential-free iTunes Search by default.

    ``MUSIC_AGENT_CATALOG_PROVIDER=music_kit`` selects the MusicKit adapter (which needs a
    developer token); anything else falls back to the public iTunes Search API. Both return
    the same ``CatalogTrack`` domain values, so discovery behaves identically downstream.
    """
    if os.environ.get("MUSIC_AGENT_CATALOG_PROVIDER") == "music_kit":
        from music_agent.apple_music_catalog import AppleMusicCatalogAdapter, MusicKitTransport

        return AppleMusicCatalogAdapter(MusicKitTransport())
    from music_agent.itunes_search import iTunesSearchAdapter, iTunesSearchTransport

    return iTunesSearchAdapter(iTunesSearchTransport())


def build_catalog_candidate(track: CatalogTrack) -> IngestionCandidate:
    """Map one CatalogTrack to a durable IngestionCandidate under its own source system.

    Catalog metadata becomes source facts; the ISRC (when the source reliably supplied it)
    rides as a secondary identity and is bound together with the source ID at promotion.
    ``preview_url`` rides as a fact for the preview slice (T4); promotion ignores it.
    Library-state facts are deliberately absent: a catalog song has no library state yet.
    """
    facts: dict[str, ObservedValue] = {
        "name": ObservedValue.value(track.name),
        "genres": ObservedValue.value(list(track.genres)),
    }
    if track.duration_ms is not None:
        facts["duration_ms"] = ObservedValue.value(track.duration_ms)
    if track.release_date is not None:
        facts["release_date"] = ObservedValue.value(track.release_date)
    if track.preview_url is not None:
        facts["preview_url"] = ObservedValue.value(track.preview_url)
    secondaries: tuple[ExternalIdentityKey, ...] = ()
    if track.isrc is not None:
        secondaries = (ExternalIdentityKey("isrc", EntityType.TRACK, track.isrc),)
    return IngestionCandidate(
        ExternalIdentityKey(track.source_system, EntityType.TRACK, track.catalog_id),
        facts,
        secondary_identities=secondaries,
    )


class CatalogIngestionOrchestrator:
    """Ingest parsed catalog results into durable staging with identity dedupe and exclusion."""

    def __init__(
        self,
        repository: CanonicalRepository,
        staging: CandidateStagingRepository,
        catalog_source: CatalogSearchSource | None = None,
        *,
        track_state: CatalogTrackStateRepository,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(repository, CanonicalRepository):
            raise CatalogIngestionError("repository must be a CanonicalRepository")
        if not isinstance(staging, CandidateStagingRepository):
            raise CatalogIngestionError("staging must be a CandidateStagingRepository")
        if not isinstance(track_state, CatalogTrackStateRepository):
            raise CatalogIngestionError("track_state must be a CatalogTrackStateRepository")
        if catalog_source is not None and not callable(getattr(catalog_source, "search", None)):
            raise CatalogIngestionError("catalog_source must provide search(term, limit)")
        if now_fn is not None and not callable(now_fn):
            raise CatalogIngestionError("now_fn must be a zero-argument callable")
        self._repository = repository
        self._staging = staging
        self._catalog_source = catalog_source
        self._track_state = track_state
        self._now_fn = now_fn

    def discover_and_ingest(self, term: str, limit: int = 25) -> tuple[CatalogIngestOutcome, ...]:
        """Search the catalog and ingest the results (adapter must be injected)."""
        if self._catalog_source is None:
            raise CatalogIngestionError("no catalog source injected")
        tracks = self._catalog_source.search(term, limit)
        return self.ingest(tracks, term=term)

    def ingest(self, tracks: object, *, term: str | None = None) -> tuple[CatalogIngestOutcome, ...]:
        """Dedupe, exclude, stage, resolve, and promote catalog hits in one batched pass.

        Per-track evaluation keeps per-track isolation (one failure never aborts the whole
        batch and a failed hit stays staged, fail closed), while canonical persistence is
        amortized across the batch: at most one model load and, when anything changed, one
        atomic batch save (P20 Performance Fix 01).
        """
        if not isinstance(tracks, (tuple, list)) or not all(
            isinstance(track, CatalogTrack) for track in tracks
        ):
            raise CatalogIngestionError("tracks must be a sequence of CatalogTrack values")
        batch = _DiscoveryBatchState()
        outcomes: list[CatalogIngestOutcome] = []
        for track in tracks:
            try:
                outcomes.append(self._evaluate_track(track, batch))
            except Exception as error:  # per-track isolation: one failure never aborts the batch
                logger.exception("catalog ingest failed for %s", track.catalog_id)
                outcomes.append(
                    CatalogIngestOutcome(track.catalog_id, CatalogIngestStatus.STAGING_FAILED, error=str(error))
                )
        if batch.model_changed:
            try:
                self._repository.commit_catalog_discovery(batch.model, batch.promotions)
            except Exception as error:
                logger.exception("batched catalog discovery commit failed")
                outcomes = [
                    CatalogIngestOutcome(
                        outcome.catalog_id,
                        CatalogIngestStatus.STAGING_FAILED,
                        error=f"batch persistence failed: {error}".strip(),
                    )
                    if outcome.status is CatalogIngestStatus.PROMOTED
                    else outcome
                    for outcome in outcomes
                ]
        for index, outcome in enumerate(outcomes):
            try:
                self._record_discovery_event(outcome, tracks[index], term)
            except Exception as error:
                logger.exception("discovery event recording failed for %s", outcome.catalog_id)
                outcomes[index] = CatalogIngestOutcome(
                    outcome.catalog_id, CatalogIngestStatus.STAGING_FAILED, error=str(error)
                )
        return tuple(outcomes)

    def _record_discovery_event(
        self, outcome: CatalogIngestOutcome, track: CatalogTrack, term: str | None
    ) -> None:
        """Record one real discovery occurrence in ``catalog_track_state``.

        Only a successful ingestion event qualifies: a fresh ``PROMOTED`` track or an
        ``ALREADY_BOUND`` hit (the current catalog query really did encounter this canonical
        track again -- the repeated-search memory S3 targets). STAGED / STAGED_BLOCKED /
        IDENTITY_CONFLICT / STAGING_FAILED produce no canonical track, and LIBRARY_KNOWN proves
        the hit is already library music rather than a catalog discovery, so neither records.
        """
        if outcome.status not in (CatalogIngestStatus.PROMOTED, CatalogIngestStatus.ALREADY_BOUND):
            return
        if outcome.canonical_id is None:
            return
        now = self._now_fn() if self._now_fn is not None else datetime.now(timezone.utc)
        self._track_state.record_discovery_occurrence(
            outcome.canonical_id,
            source_system=track.source_system,
            term=term,
            now=now,
        )

    def _evaluate_track(self, track: CatalogTrack, batch: _DiscoveryBatchState) -> CatalogIngestOutcome:
        catalog_key = ExternalIdentityKey(track.source_system, EntityType.TRACK, track.catalog_id)
        bound = self._lookup_identity_overlay(catalog_key, batch)
        if bound is not None:
            return CatalogIngestOutcome(track.catalog_id, CatalogIngestStatus.ALREADY_BOUND, bound)
        if track.isrc is not None:
            isrc_key = ExternalIdentityKey("isrc", EntityType.TRACK, track.isrc)
            isrc_bound = self._lookup_identity_overlay(isrc_key, batch)
            if isrc_bound is not None:
                # Same proven recording already canonical: never stage a duplicate identity.
                presence = self._repository.get_source_presence(
                    "apple_music", EntityType.TRACK, isrc_bound, LIBRARY_TRACKS_SCOPE
                )
                if presence is SourcePresence.PRESENT:
                    return CatalogIngestOutcome(
                        track.catalog_id, CatalogIngestStatus.LIBRARY_KNOWN, isrc_bound
                    )
                return CatalogIngestOutcome(
                    track.catalog_id, CatalogIngestStatus.ALREADY_BOUND, isrc_bound
                )
        self._staging.stage_candidate(build_catalog_candidate(track), CATALOG_SCOPE)
        return self._resolve_relations_and_promote(track, catalog_key, batch)

    def _lookup_identity_overlay(
        self, key: ExternalIdentityKey, batch: _DiscoveryBatchState
    ) -> str | None:
        """SQL-first identity lookup with the batch-local overlay for in-batch creations.

        The overlay only ever holds bindings this very batch created, so hitting it is
        exactly equivalent to what a sequentially persisted batch would have made the
        next track's SQL lookup see.
        """
        cached = batch.batch_bindings.get((key.source_system, key.entity_type, key.external_id))
        if cached is not None:
            return cached
        return self._repository.lookup_external_identity(key)

    def _ensure_batch_model(self, batch: _DiscoveryBatchState) -> dict[str, Any]:
        if batch.model is None:
            batch.model = self._repository.load_model()
        return batch.model

    def _resolve_relations_and_promote(
        self, track: CatalogTrack, catalog_key: ExternalIdentityKey, batch: _DiscoveryBatchState
    ) -> CatalogIngestOutcome:
        """Resolve authoritative relations, then promote into the batch's working model.

        Fails closed: without authoritative Catalog Artist / Album identity (or the album name
        required to create a canonical Album) the candidate stays staged with an explicit
        blocker; a binding conflict stays staged as an identity conflict. Promotion appends
        the built Track to the shared working model and registers the entry for the batch's
        single atomic commit.
        """
        if not track.artist_catalog_ids:
            return CatalogIngestOutcome(
                track.catalog_id,
                CatalogIngestStatus.STAGED_BLOCKED,
                error="catalog payload provides no authoritative artist identity; "
                "display names never resolve a relation",
                blocker_codes=("artist_relation_unresolved",),
            )
        if track.album_catalog_id is None or not track.album_name:
            return CatalogIngestOutcome(
                track.catalog_id,
                CatalogIngestStatus.STAGED_BLOCKED,
                error="catalog payload provides no authoritative album identity "
                "(or no album name to create one)",
                blocker_codes=("album_relation_unresolved",),
            )
        try:
            artist_resolution, album_resolution = self._resolve_relations(track, batch)
        except IdentityConflictError as error:
            return CatalogIngestOutcome(
                track.catalog_id,
                CatalogIngestStatus.IDENTITY_CONFLICT,
                error=str(error),
            )
        self._staging.update_relation_resolution(
            catalog_key,
            artist_resolution=artist_resolution,
            album_resolution=album_resolution,
        )
        candidate = self._staging.get_candidate(catalog_key)
        if candidate is None:
            raise StagedCandidateNotFoundError("staged Candidate does not exist")
        model = self._ensure_batch_model(batch)
        evaluation = evaluate_track_promotion(candidate, model)
        if not evaluation.is_promotable:
            return CatalogIngestOutcome(
                track.catalog_id,
                CatalogIngestStatus.STAGED_BLOCKED,
                error="promotion blocked",
                blocker_codes=tuple(blocker.code.value for blocker in evaluation.blockers),
            )
        canonical_id = generate_canonical_id(EntityType.TRACK)
        promoted_track = build_promotable_track(candidate, canonical_id, model)
        model["tracks"].append(promoted_track)
        batch.model_changed = True
        batch.promotions.append((catalog_key, canonical_id))
        batch.batch_bindings[
            (catalog_key.source_system, catalog_key.entity_type, catalog_key.external_id)
        ] = canonical_id
        if track.isrc is not None:
            batch.batch_bindings[("isrc", EntityType.TRACK, track.isrc)] = canonical_id
        return CatalogIngestOutcome(track.catalog_id, CatalogIngestStatus.PROMOTED, canonical_id)

    def _resolve_relations(
        self, track: CatalogTrack, batch: _DiscoveryBatchState
    ) -> tuple[ArtistRelationResolution, AlbumRelationResolution]:
        """Get-or-create canonical Artist / Album entities bound by Catalog identity.

        Reuses an already-bound canonical entity when the authoritative Catalog ID is known;
        otherwise creates the minimum canonical entity carrying that binding. Display names are
        presentation-only. One model load serves the whole batch; creations mutate the shared
        working model and are persisted once at the batch commit.
        """
        model = self._ensure_batch_model(batch)
        artist_ids: list[str] = []
        combined_name = ", ".join(track.artist_names)
        source_fixture_key = EXTERNAL_ID_FIXTURE_KEYS[track.source_system]
        for catalog_artist_id in track.artist_catalog_ids:
            key = ExternalIdentityKey(
                track.source_system, EntityType.ARTIST, catalog_artist_id
            )
            canonical_id = self._lookup_identity_overlay(key, batch)
            if canonical_id is None:
                canonical_id = generate_canonical_id(EntityType.ARTIST)
                # ponytail: songs carry one combined artistName string for every artist;
                # multi-artist entities share it as presentation until per-artist names
                # are fetched (per-artist source lookups).
                name = (
                    track.artist_names[0]
                    if len(track.artist_catalog_ids) == 1
                    else combined_name
                )
                model["artists"].append(
                    {
                        "id": canonical_id,
                        "external_ids": {
                            "apple_music_persistent_id": None,
                            source_fixture_key: catalog_artist_id,
                        },
                        "name": name,
                    }
                )
                batch.batch_bindings[
                    (key.source_system, key.entity_type, key.external_id)
                ] = canonical_id
                batch.model_changed = True
            artist_ids.append(canonical_id)

        album_key = ExternalIdentityKey(
            track.source_system, EntityType.ALBUM, track.album_catalog_id
        )
        album_id = self._lookup_identity_overlay(album_key, batch)
        if album_id is None:
            album_id = generate_canonical_id(EntityType.ALBUM)
            model["albums"].append(
                {
                    "id": album_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        source_fixture_key: track.album_catalog_id,
                    },
                    "name": track.album_name,
                    "artist_ids": list(dict.fromkeys(artist_ids)),
                    "release_date": track.release_date,
                }
            )
            batch.batch_bindings[
                (album_key.source_system, album_key.entity_type, album_key.external_id)
            ] = album_id
            batch.model_changed = True
        return (
            ArtistRelationResolution.resolved_to_artists(artist_ids),
            AlbumRelationResolution.resolved_to_album(album_id),
        )
