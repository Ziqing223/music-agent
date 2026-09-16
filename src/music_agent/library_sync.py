"""P10.11: Library discovery / full-sync orchestration (production daily path).

One coherent path from the real Music.app library into the durable Music Agent store,
preserving every sealed P02/P03/P06 semantic:

    enumerate real persistent IDs (discovery adapter, read-only)
    -> for each id: the existing production per-track read
    -> canonical identity binding (persistent ID is the ONLY authority)
    -> sealed merge/save semantics (refresh_known_track for known tracks)
    -> confirmed source presence (PRESENT, library_tracks scope) for new tracks
    -> sealed P06 ingest_track_observation (same observation, shared path)

Safety semantics:

- Enumeration failure aborts the whole scan with ``enumeration_failed=True`` and NO
  presence/absence inference of any kind (fail closed: an incomplete scan can never
  mark anything absent).
- Absence is reported only: bound persistent IDs missing from a successful enumeration
  appear in ``absent_this_scan``; no canonical entity is ever deleted and no
  ``CONFIRMED_DELETED`` is written (the sealed P04 snapshot deletion-authority
  machinery is the only absence writer, and this path does not trigger it).
- Per-track isolation: one track's read/refresh failure never aborts the scan.
- New tracks are ingested as one atomic canonical batch (save + bindings + presence);
  a batch failure leaves canonical state unchanged and is reported.
- Preference ingestion for new tracks runs after the batch and fails per-track without
  touching the committed canonical state.
- Repeated scans are idempotent by construction: the binding lookup converges every
  re-discovery on the same canonical id, refresh returns UNCHANGED for stable tracks,
  and the sealed P06 repository confirms identical values without new revisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from music_agent.apple_music import (
    add_library_relation_values,
    AppleMusicSourceAdapter,
    materialize_library_track_relations,
    SourceReadStatus,
)
from music_agent.apple_music_library_discovery import AppleMusicLibraryDiscoveryAdapter
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.merge import merge_observations
from music_agent.refresh import refresh_known_track
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.runtime import Clock, utc_now
from music_agent.source_observation import ObservedValue, SourceObservation, SourcePresence

logger = logging.getLogger("music_agent.library_sync")

LIBRARY_TRACKS_SCOPE = "library_tracks"


class LibrarySyncError(ValueError):
    code = "library_sync_error"


class LibrarySyncStatus(StrEnum):
    NEW = "new"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SOURCE_NOT_FOUND = "source_not_found"
    READ_FAILED = "read_failed"
    MERGE_FAILED = "merge_failed"
    INGESTION_FAILED = "ingestion_failed"
    BATCH_FAILED = "batch_failed"


@dataclass(frozen=True, slots=True)
class LibrarySyncTrackOutcome:
    """One per-track outcome of a full-sync scan."""

    persistent_id: str
    canonical_id: str | None
    status: LibrarySyncStatus
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LibrarySyncReport:
    """Aggregate outcome of one full-sync scan."""

    started_at: datetime
    finished_at: datetime
    enumeration_failed: bool
    enumeration_error: str | None
    enumerated_count: int
    unique_count: int
    outcomes: tuple[LibrarySyncTrackOutcome, ...]
    absent_this_scan: tuple[str, ...]
    genre_counts: dict[str, int] | None = None

    @property
    def succeeded(self) -> bool:
        return not self.enumeration_failed and all(
            outcome.status is not LibrarySyncStatus.BATCH_FAILED for outcome in self.outcomes
        )

    def counts(self) -> dict[str, int]:
        tallies = {status.value: 0 for status in LibrarySyncStatus}
        tallies["absent_this_scan"] = len(self.absent_this_scan)
        for outcome in self.outcomes:
            tallies[outcome.status.value] += 1
        return tallies


def build_genre_observation(canonical_id: str, genre: str | None) -> SourceObservation:
    """One genre-only Track observation through the sealed merge path.

    ``genre`` is the exact source string; None/empty yields MISSING (no claim). The
    ``genres`` field is APPLE_MUSIC-owned in the sealed ownership policy, so the existing
    merge authority applies exactly as for any other library fact.
    """
    return SourceObservation(
        EntityType.TRACK,
        canonical_id,
        "apple_music",
        {
            "genres": (
                ObservedValue.value([genre])
                if genre
                else ObservedValue.missing()
            )
        },
    )


def enrich_genres(
    repository: CanonicalRepository,
    genre_adapter,
    canonical_ids: list[str],
    persistent_id_by_canonical: dict[str, str],
) -> dict[str, int]:
    """Read exact genre facts and merge them through the sealed canonical path.

    Per-track isolation: one failed genre read is recorded and never aborts the batch.
    The merge machinery owns every change (APPLE_MUSIC-owned genres, atomic save).
    Synchronous wrapper over :func:`_iter_genre_steps` (one-shot callers).
    Returns {"enriched", "unchanged", "failed"}.
    """
    steps = _iter_genre_steps(
        repository, genre_adapter, canonical_ids, persistent_id_by_canonical
    )
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def _iter_genre_steps(
    repository: CanonicalRepository,
    genre_adapter,
    canonical_ids: list[str],
    persistent_id_by_canonical: dict[str, str],
):
    """Per-track genre enrichment as bounded steps (cooperative resumption).

    Each step reads one exact genre fact and merges it through the sealed
    canonical path -- fresh ``load_model`` per step, atomic save on change,
    the same per-track unit the one-shot cycle always performed. Yields step
    progress and returns {"enriched", "unchanged", "failed"}.
    """
    enriched = 0
    unchanged = 0
    failed = 0
    for canonical_id in canonical_ids:
        persistent_id = persistent_id_by_canonical.get(canonical_id)
        if persistent_id is None:
            failed += 1
            yield {"canonical_id": canonical_id, "status": "failed"}
            continue
        try:
            genre = genre_adapter.read_genre(persistent_id)
            if genre is not None:
                genre = str(genre).strip() or None
        except Exception as error:
            logger.exception("genre enrichment failed for %s", canonical_id)
            failed += 1
            yield {"canonical_id": canonical_id, "status": "failed"}
            continue
        model = repository.load_model()
        observation = build_genre_observation(canonical_id, genre)
        merge_result = merge_observations(model, [observation])
        if merge_result.changed_fields:
            repository.save_model(merge_result.model)
            enriched += 1
            yield {"canonical_id": canonical_id, "status": "enriched"}
        else:
            unchanged += 1
            yield {"canonical_id": canonical_id, "status": "unchanged"}
    return {"enriched": enriched, "unchanged": unchanged, "failed": failed}


def build_canonical_track_from_read(persistent_id: str, fields: dict) -> dict:
    """One canonical track from the sealed read adapter's live-read fields.

    The persistent ID is the only identity authority; fields the adapter does not
    provide (artist/album relations, genres, duration) stay null/empty -- the store
    only claims what the source confirmed. Shared by the full-sync path and
    ``tools/bootstrap_real_tracks.py`` (single production definition).
    """
    library_state = {
        "favorited": None,
        "disliked": None,
        "rating": None,
        "play_count": None,
        "skip_count": None,
        "added_to_library_at": None,
        "last_played_at": None,
    }
    library_state["favorited"] = fields.get("favorited")
    library_state["disliked"] = fields.get("disliked")
    library_state["rating"] = fields.get("rating")
    library_state["play_count"] = fields.get("played_count")
    library_state["added_to_library_at"] = fields.get("date_added")
    library_state["last_played_at"] = fields.get("played_date")
    name = fields.get("name")
    return {
        "id": f"trk_{uuid4()}",
        "external_ids": {"apple_music_persistent_id": persistent_id},
        "name": name if isinstance(name, str) and name else f"待核实 ({persistent_id})",
        "artist_ids": [],
        "album_id": None,
        "duration_ms": None,
        "genres": [],
        "track_number": None,
        "disc_number": None,
        "release_date": None,
        "composer": None,
        "library_state": library_state,
        "agent_metadata": {"tags": []},
    }


class LibrarySyncOrchestrator:
    """Runs one full library scan: enumerate -> per-track read -> canonical -> P06.

    ``iter_steps`` exposes the scan as bounded cooperative steps (one idle
    point per persistent-id / ingest batch / genre track); ``run_cycle`` is
    the synchronous drain wrapper with an identical report.
    """

    def __init__(
        self,
        repository: CanonicalRepository,
        per_track_adapter: AppleMusicSourceAdapter,
        discovery_adapter: AppleMusicLibraryDiscoveryAdapter,
        *,
        clock: Clock = utc_now,
        preference_repository=None,
        genre_adapter=None,
    ) -> None:
        if not isinstance(repository, CanonicalRepository):
            raise LibrarySyncError("repository must be a CanonicalRepository")
        if not callable(getattr(per_track_adapter, "read_track", None)) or not callable(
            getattr(per_track_adapter, "build_observation", None)
        ):
            raise LibrarySyncError(
                "per_track_adapter must provide read_track() and build_observation()"
            )
        if not callable(getattr(discovery_adapter, "list_persistent_ids", None)):
            raise LibrarySyncError(
                "discovery_adapter must provide list_persistent_ids()"
            )
        if genre_adapter is not None and not callable(
            getattr(genre_adapter, "read_genre", None)
        ):
            raise LibrarySyncError("genre_adapter must provide read_genre()")
        self._repository = repository
        self._adapter = per_track_adapter
        self._discovery = discovery_adapter
        self._clock = clock
        self._preference_repository = preference_repository
        self._genre_adapter = genre_adapter

    def iter_steps(self):
        """One full library scan as bounded steps (cooperative resumption).

        Phase order (each phase one or more steps): enumerate (one discovery
        call) -> one step per persistent id (known tracks through the sealed
        refresh path, new tracks through the read adapter) -> ONE atomic
        ingest-batch step (``_ingest_new_tracks`` exact semantics: single
        ``save_model_with_source_presence``) -> one genre-enrichment step per
        canonical track -> return the aggregate report. Between steps the
        runtime main loop is free to service the other components.
        """
        started_at = self._clock()
        try:
            identifiers = self._discovery.list_persistent_ids()
        except Exception as error:
            logger.exception("library enumeration failed; no scan performed")
            return LibrarySyncReport(
                started_at=started_at,
                finished_at=self._clock(),
                enumeration_failed=True,
                enumeration_error=str(error),
                enumerated_count=0,
                unique_count=0,
                outcomes=(),
                absent_this_scan=(),
            )
        yield {"phase": "enumerate", "enumerated_count": len(identifiers)}
        unique = tuple(dict.fromkeys(identifiers))  # deterministic dedupe, order kept

        model = self._repository.load_model()
        bound_ids = {
            track["external_ids"]["apple_music_persistent_id"]
            for track in model["tracks"]
            if track["external_ids"]["apple_music_persistent_id"] is not None
        }
        outcomes: list[LibrarySyncTrackOutcome] = []
        new_tracks: list[dict] = []
        new_reads: list[tuple[str, object]] = []  # (persistent_id, SourceReadResult)
        for persistent_id in unique:
            step_status: str
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
            bound = self._repository.lookup_external_identity(key)
            if bound is not None:
                refreshed = self._refresh_known(bound, persistent_id)
                outcomes.append(refreshed)
                step_status = refreshed.status.value
                yield {"phase": "track", "persistent_id": persistent_id, "status": step_status}
                continue
            try:
                read_result = self._adapter.read_track(persistent_id)
            except Exception as error:  # per-track isolation: one failure never aborts
                logger.exception("library sync: read failed for %s", persistent_id)
                outcomes.append(
                    LibrarySyncTrackOutcome(
                        persistent_id, None, LibrarySyncStatus.READ_FAILED, str(error)
                    )
                )
                step_status = LibrarySyncStatus.READ_FAILED.value
                yield {"phase": "track", "persistent_id": persistent_id, "status": step_status}
                continue
            if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
                status = (
                    LibrarySyncStatus.SOURCE_NOT_FOUND
                    if read_result.status is SourceReadStatus.CONFIRMED_NOT_FOUND
                    else LibrarySyncStatus.READ_FAILED
                )
                outcomes.append(
                    LibrarySyncTrackOutcome(
                        persistent_id,
                        None,
                        status,
                        read_result.error,
                    )
                )
                step_status = status.value
                yield {"phase": "track", "persistent_id": persistent_id, "status": step_status}
                continue
            track = build_canonical_track_from_read(
                persistent_id, dict(read_result.record.fields)
            )
            new_tracks.append(track)
            new_reads.append((persistent_id, read_result))
            step_status = "new"
            yield {"phase": "track", "persistent_id": persistent_id, "status": step_status}

        if new_tracks:
            outcomes.extend(
                self._ingest_new_tracks(new_tracks, new_reads)
            )
            yield {"phase": "ingest", "new_tracks": len(new_tracks)}

        genre_counts: dict[str, int] | None = None
        if self._genre_adapter is not None:
            persistent_id_by_canonical = {}
            current_model = self._repository.load_model()
            for track in current_model["tracks"]:
                persistent_id = track["external_ids"]["apple_music_persistent_id"]
                if persistent_id is not None:
                    persistent_id_by_canonical[track["id"]] = persistent_id
            genre_steps = _iter_genre_steps(
                self._repository,
                self._genre_adapter,
                list(persistent_id_by_canonical),
                persistent_id_by_canonical,
            )
            genre_counts = yield from genre_steps
        absent_this_scan = tuple(sorted(bound_ids - set(unique)))
        report = LibrarySyncReport(
            started_at=started_at,
            finished_at=self._clock(),
            enumeration_failed=False,
            enumeration_error=None,
            enumerated_count=len(identifiers),
            unique_count=len(unique),
            outcomes=tuple(outcomes),
            absent_this_scan=absent_this_scan,
            genre_counts=genre_counts,
        )
        counts = report.counts()
        logger.info(
            "library sync complete: enumerated=%d new=%d updated=%d unchanged=%d "
            "read_failed=%d absent_this_scan=%d",
            report.unique_count,
            counts["new"],
            counts["updated"],
            counts["unchanged"],
            counts["read_failed"],
            len(absent_this_scan),
        )
        return report

    def run_cycle(self) -> LibrarySyncReport:
        """Synchronous compatibility wrapper: drain ``iter_steps`` to completion.

        The report is field-identical to the previous one-shot scan.
        """
        steps = self.iter_steps()
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value

    def _refresh_known(self, canonical_id: str, persistent_id: str) -> LibrarySyncTrackOutcome:
        """Known track: the sealed refresh path (+ sealed P06 hook) owns the semantics."""
        try:
            result = refresh_known_track(
                self._repository,
                self._adapter,
                canonical_id,
                preference_repository=self._preference_repository,
            )
        except Exception as error:
            logger.exception("library sync: refresh failed for %s", canonical_id)
            return LibrarySyncTrackOutcome(persistent_id, canonical_id, LibrarySyncStatus.READ_FAILED, str(error))
        status = {
            "updated": LibrarySyncStatus.UPDATED,
            "unchanged": LibrarySyncStatus.UNCHANGED,
            "source_not_found": LibrarySyncStatus.SOURCE_NOT_FOUND,
            "source_lookup_failed": LibrarySyncStatus.READ_FAILED,
            "merge_failed": LibrarySyncStatus.MERGE_FAILED,
            "no_source_binding": LibrarySyncStatus.READ_FAILED,
            "canonical_not_found": LibrarySyncStatus.READ_FAILED,
        }[result.status.value]
        return LibrarySyncTrackOutcome(persistent_id, canonical_id, status, result.error)

    def _ingest_new_tracks(self, new_tracks, new_reads) -> list[LibrarySyncTrackOutcome]:
        """One atomic canonical batch (save + bindings + presence), then per-track P06.

        The batch starts from a FRESH load: known-track refreshes may have committed
        updates earlier in the cycle, and the batch must never clobber them with a
        stale pre-refresh snapshot.
        """
        presence_records = [
            SourcePresenceRecord(
                "apple_music",
                EntityType.TRACK,
                track["id"],
                LIBRARY_TRACKS_SCOPE,
                SourcePresence.PRESENT,
            )
            for track in new_tracks
        ]
        outcomes: list[LibrarySyncTrackOutcome] = []
        observations: list[tuple[str, object]] = []
        try:
            current = self._repository.load_model()
            current["tracks"] = [*current["tracks"], *new_tracks]
            for persistent_id, read_result in new_reads:
                canonical_id = next(
                    track["id"] for track in new_tracks
                    if track["external_ids"]["apple_music_persistent_id"] == persistent_id
                )
                relation_fields = materialize_library_track_relations(
                    current, canonical_id, read_result.record.fields
                )
                observation = self._adapter.build_observation(canonical_id, read_result)
                observations.append(
                    (persistent_id, add_library_relation_values(observation, relation_fields))
                )
            merged = merge_observations(
                current, [observation for _, observation in observations]
            )
            self._repository.save_model_with_source_presence(
                merged.model, presence_records
            )
        except Exception as error:
            logger.exception("library sync: canonical batch failed")
            return [
                LibrarySyncTrackOutcome(
                    persistent_id, None, LibrarySyncStatus.BATCH_FAILED, str(error)
                )
                for persistent_id, _ in new_reads
            ]
        for persistent_id, observation in observations:
            canonical_id = next(
                track["id"] for track in new_tracks
                if track["external_ids"]["apple_music_persistent_id"] == persistent_id
            )
            status = LibrarySyncStatus.NEW
            error = None
            if self._preference_repository is not None:
                try:
                    from music_agent.preference_ingestion import ingest_track_observation

                    ingest_track_observation(self._preference_repository, observation)
                except Exception as ingest_error:
                    logger.exception("library sync: P06 ingestion failed for %s", persistent_id)
                    status = LibrarySyncStatus.INGESTION_FAILED
                    error = str(ingest_error)
            outcomes.append(
                LibrarySyncTrackOutcome(persistent_id, canonical_id, status, error)
            )
        return outcomes
