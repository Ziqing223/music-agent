"""P10.16: Simplified daily entrypoint and local backup/recovery.

- ``run_daily``: one daily tick -- a full library sync, then a bound-track refresh,
  then the durable status -- suitable for a launchd StartCalendarInterval job or a
  manual "morning sync". Read-only toward Music.app; writes only the durable store.
- ``backup_store`` / ``restore_store``: file-level local backup (copy + verify-by-
  reopen). Retention: backups are rotated by the OPERATOR (immutable in-store tables
  are never destructively pruned). ``restore`` refuses to overwrite an existing
  store without ``force=True``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from music_agent.repository import CanonicalRepository


@dataclass(frozen=True, slots=True)
class DailyRunResult:
    library_sync_counts: dict[str, int]
    refresh_counts: dict[str, int]
    status: dict
    succeeded: bool


def run_daily(database_path: str | Path, *, timeout_seconds: float = 10.0) -> DailyRunResult:
    """One daily tick: library-sync -> refresh -> status (all production paths)."""
    from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner
    from music_agent.apple_music_library_discovery import (
        AppleMusicLibraryDiscoveryAdapter,
        OsascriptLibraryTrackIdsRunner,
    )
    from music_agent.library_sync import LibrarySyncOrchestrator
    from music_agent.preference_persistence_repository import PreferencePersistenceRepository
    from music_agent.runtime_refresh import MusicRefreshOrchestrator
    from music_agent.runtime_status import build_store_status

    database_path = Path(database_path)
    adapter = AppleMusicSourceAdapter(OsascriptMusicRunner(timeout_seconds=timeout_seconds))
    discovery = AppleMusicLibraryDiscoveryAdapter(
        OsascriptLibraryTrackIdsRunner(timeout_seconds=timeout_seconds)
    )
    succeeded = True
    with PreferencePersistenceRepository(database_path) as preference:
        with CanonicalRepository(database_path) as repository:
            sync_report = LibrarySyncOrchestrator(
                repository, adapter, discovery, preference_repository=preference
            ).run_cycle()
            if sync_report.enumeration_failed:
                succeeded = False
            refresh_report = MusicRefreshOrchestrator(
                repository, adapter, preference_repository=preference
            ).run_cycle()
            if not refresh_report.succeeded:
                succeeded = False
    return DailyRunResult(
        library_sync_counts=sync_report.counts(),
        refresh_counts=refresh_report.counts(),
        status=build_store_status(database_path),
        succeeded=succeeded,
    )


def backup_store(
    database_path: str | Path, *, backup_dir: Path | None = None, now: datetime | None = None
) -> Path:
    """Copy the store file (+ process-state sidecar) to a timestamped backup and verify it."""
    database_path = Path(database_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"store does not exist: {database_path}")
    now = now or datetime.now()
    backup_dir = backup_dir or database_path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{database_path.name}.backup-{now:%Y%m%d-%H%M%S}"
    shutil.copy2(database_path, backup_path)
    # Verify the copy reopens at the expected schema version.
    with CanonicalRepository(backup_path) as repository:
        repository.schema_version
    return backup_path


def restore_store(
    database_path: str | Path, backup_path: str | Path, *, force: bool = False
) -> None:
    """Restore a backup over the store; refuses to overwrite unless force=True."""
    database_path = Path(database_path)
    backup_path = Path(backup_path)
    if not backup_path.is_file():
        raise FileNotFoundError(f"backup does not exist: {backup_path}")
    if database_path.exists() and not force:
        raise ValueError(
            f"store already exists at {database_path}; pass force=True to overwrite"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, database_path)
    with CanonicalRepository(database_path) as repository:
        repository.schema_version
