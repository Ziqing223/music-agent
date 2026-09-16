"""P10.2: Integrated Music.app refresh cycle over bound canonical tracks.

The cycle reuses the sealed known-entity refresh path exactly -- one
``refresh_known_track`` call per canonical track that carries an ``apple_music_persistent_id``
binding -- and adds the runtime concerns around it: whole-store iteration, per-track failure
isolation, and an aggregate report. It does NOT recreate a synchronization model: merge
authority, source-presence semantics, and atomic save behavior all come from the existing
``refresh`` / ``merge`` / ``repository`` boundaries.

Semantics:

- Only tracks whose external binding resolves to their own canonical id are candidates;
  tracks without a binding are outside the Music.app refresh surface and are simply counted
  as ``skipped_no_binding``.
- Expected source-level outcomes (not found, lookup failed, merge failed, unchanged) surface
  as typed ``RefreshResult`` rows -- exactly the ``refresh_known_track`` vocabulary.
- An unexpected exception in one track's refresh never aborts the cycle: it is logged and
  recorded as a ``RefreshFailure``, and the loop continues with the next track.
- Repeated refresh is safe by construction: a track that did not change returns
  ``UNCHANGED``, and a changed track is saved atomically through the production
  ``save_model`` path (same store transaction discipline as P01-P09 refreshes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.refresh import RefreshResult, refresh_known_track
from music_agent.repository import CanonicalRepository
from music_agent.runtime import Clock, utc_now

logger = logging.getLogger("music_agent.refresh_cycle")


@dataclass(frozen=True, slots=True)
class RefreshFailure:
    """One unexpected per-track failure that did not abort the cycle."""

    canonical_id: str
    error: str


@dataclass(frozen=True, slots=True)
class MusicRefreshReport:
    """Aggregate outcome of one integrated refresh cycle."""

    started_at: datetime
    finished_at: datetime
    bound_track_count: int
    skipped_no_binding: int
    results: tuple[RefreshResult, ...]
    failures: tuple[RefreshFailure, ...]

    @property
    def succeeded(self) -> bool:
        """True when the cycle completed with no unexpected per-track failures."""
        return not self.failures

    def counts(self) -> dict[str, int]:
        """Per-status tallies for observability surfaces (stable keys)."""
        tallies = {
            "updated": 0,
            "unchanged": 0,
            "no_source_binding": 0,
            "canonical_not_found": 0,
            "source_not_found": 0,
            "source_lookup_failed": 0,
            "merge_failed": 0,
            "failed": len(self.failures),
        }
        for result in self.results:
            tallies[result.status.value] += 1
        return tallies


class MusicRefreshOrchestrator:
    """Runs one refresh cycle over every bound canonical track in the store.

    ``preference_repository`` (optional) wires the sealed P06 ingestion into the cycle:
    every FOUND observation that refreshed (or re-confirmed) canonical state is also fed
    through ``ingest_track_observation``. Preference-ingestion failures surface as
    per-track ``RefreshFailure`` entries and never affect the already-saved canonical
    state; identical re-observations confirm without new revisions (sealed repository
    semantics), so repeated cycles never inflate evidence.
    """

    def __init__(
        self,
        repository: CanonicalRepository,
        adapter: AppleMusicSourceAdapter,
        *,
        clock: Clock = utc_now,
        preference_repository=None,
    ) -> None:
        self._repository = repository
        self._adapter = adapter
        self._clock = clock
        self._preference_repository = preference_repository

    def iter_steps(self):
        """One refresh cycle as bounded per-track steps (cooperative resumption).

        Each step refreshes exactly one bound track through the sealed
        ``refresh_known_track`` path -- a self-contained read/merge/atomic-save
        unit, identical to the one-shot cycle's per-track work -- and yields
        step progress. The generator returns the aggregate
        :class:`MusicRefreshReport`; between steps the runtime main loop is
        free to service the other components (notably the agent socket).
        """
        started_at = self._clock()
        model = self._repository.load_model()
        bound_ids: list[str] = []
        skipped_no_binding = 0
        for track in model["tracks"]:
            persistent_id = track["external_ids"]["apple_music_persistent_id"]
            if persistent_id is None:
                skipped_no_binding += 1
                continue
            bound_ids.append(track["id"])
        results: list[RefreshResult] = []
        failures: list[RefreshFailure] = []
        for canonical_id in bound_ids:
            result: RefreshResult | None = None
            try:
                result = refresh_known_track(
                    self._repository,
                    self._adapter,
                    canonical_id,
                    preference_repository=self._preference_repository,
                )
            except Exception as error:  # per-track isolation: one failure never aborts the cycle
                logger.exception(
                    "refresh cycle: unexpected failure for track %s", canonical_id
                )
                failures.append(RefreshFailure(canonical_id, str(error)))
            else:
                results.append(result)
            yield {
                "canonical_id": canonical_id,
                "status": result.status.value if result is not None else "failed",
            }
        report = MusicRefreshReport(
            started_at=started_at,
            finished_at=self._clock(),
            bound_track_count=len(bound_ids),
            skipped_no_binding=skipped_no_binding,
            results=tuple(results),
            failures=tuple(failures),
        )
        counts = report.counts()
        logger.info(
            "refresh cycle complete: bound=%d updated=%d unchanged=%d "
            "source_not_found=%d lookup_failed=%d merge_failed=%d failed=%d",
            report.bound_track_count,
            counts["updated"],
            counts["unchanged"],
            counts["source_not_found"],
            counts["source_lookup_failed"],
            counts["merge_failed"],
            counts["failed"],
        )
        return report

    def run_cycle(self) -> MusicRefreshReport:
        """Synchronous compatibility wrapper: drain ``iter_steps`` to completion.

        The report is field-identical to the previous one-shot cycle.
        """
        steps = self.iter_steps()
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value
