"""P10.10: Bootstrap the durable daily store from REAL Apple Music tracks.

Read-only toward Music.app: each explicit persistent ID is read once through the
production read adapter (the sealed P01 read path) and ingested through the existing
canonical/source-of-truth path -- ``save_model_with_source_presence`` with the
apple_music external binding and confirmed ``PRESENT`` source-presence records, exactly
the pattern P10.7's live validation used. The persistent ID is the ONLY identity
authority: no title/artist matching, no name-derived identity, no favorites.csv import,
no sealed-model changes.

Fields: whatever the sealed read adapter actually reads (name, favorited, disliked,
rating, play_count, added/last-played timestamps). Fields the adapter does not provide
(artist/album relations, genres, duration) stay null/empty -- the store only claims
what the source confirmed.

Idempotent: a persistent ID already bound in the store is skipped, never duplicated.

The live Music.app reads require the macOS automation permission of the invoking
session -- run this in YOUR terminal (the runtime harness may not drive osascript):

  PYTHONPATH=src .venv/bin/python tools/bootstrap_real_tracks.py \\
      --db ~/MusicAgent/music_agent.db \\
      --persistent-id 0A50922A7206CB39 --persistent-id <ID2> --persistent-id <ID3>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner, SourceReadStatus
from music_agent.apple_music_genre_read import AppleMusicGenreReadAdapter, OsascriptGenreReadRunner
from music_agent.library_sync import enrich_genres
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.library_sync import build_canonical_track_from_read
from music_agent.preference_ingestion import ingest_track_observation
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import SourcePresence

DEFAULT_DATABASE_PATH = Path.home() / "MusicAgent" / "music_agent.db"




def bootstrap(
    database_path: Path, persistent_ids: list[str], *, timeout_seconds: float = 10.0
) -> dict:
    """Read + ingest the given real persistent IDs; returns {added, skipped, read}."""
    adapter = AppleMusicSourceAdapter(OsascriptMusicRunner(timeout_seconds=timeout_seconds))
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with PreferencePersistenceRepository(database_path) as preference:
        with CanonicalRepository(database_path) as repository:
            model = repository.load_model()
            presence_records: list[SourcePresenceRecord] = []
            added = 0
            skipped = 0
            seen: set[str] = set()
            for persistent_id in persistent_ids:
                if persistent_id in seen:
                    continue
                seen.add(persistent_id)
                read_result = adapter.read_track(persistent_id)
                if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
                    raise RuntimeError(
                        f"cannot ingest {persistent_id}: {read_result.status.value} "
                        f"({read_result.error or 'no record'})"
                    )
                key = ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
                bound_id = repository.lookup_external_identity(key)
                if bound_id is None:
                    track = build_canonical_track_from_read(persistent_id, dict(read_result.record.fields))
                    model["tracks"].append(track)
                    presence_records.append(
                        SourcePresenceRecord(
                            "apple_music",
                            EntityType.TRACK,
                            track["id"],
                            "library_tracks",
                            SourcePresence.PRESENT,
                        )
                    )
                    added += 1
                    canonical_id = track["id"]
                    print(f"read: {persistent_id} -> {track['name']}")
                else:
                    canonical_id = bound_id
                    skipped += 1
                    print(f"skip (already bound): {persistent_id}")
                # The same observation also feeds the sealed P06 preference ingestion
                # (idempotent: identical values confirm without new revisions).
                observation = adapter.build_observation(canonical_id, read_result)
                ingest_track_observation(preference, observation)
            if presence_records:
                repository.save_model_with_source_presence(model, presence_records)
            genre_adapter = AppleMusicGenreReadAdapter(
                OsascriptGenreReadRunner(timeout_seconds=timeout_seconds)
            )
            current = repository.load_model()
            bound = {
                track["id"]: track["external_ids"]["apple_music_persistent_id"]
                for track in current["tracks"]
                if track["external_ids"]["apple_music_persistent_id"] is not None
            }
            enrich_genres(repository, genre_adapter, list(bound), bound)
    return {"added": added, "skipped": skipped, "read": len(seen)}


def readback(database_path: Path) -> None:
    """Read the store back in THIS process and print the canonical truth."""
    with CanonicalRepository(database_path) as repository:
        model = repository.load_model()
        print(f"canonical tracks: {len(model['tracks'])}  artists: {len(model['artists'])}")
        for track in model["tracks"]:
            persistent_id = track["external_ids"]["apple_music_persistent_id"]
            binding = repository.lookup_external_identity(
                ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
            )
            print(
                f"  {track['id']}  name={track['name']!r}  "
                f"persistent_id={persistent_id}  binding_ok={binding == track['id']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DATABASE_PATH, metavar="PATH",
        help=f"durable store (default: {DEFAULT_DATABASE_PATH})",
    )
    parser.add_argument(
        "--persistent-id", action="append", required=True, metavar="ID",
        help="one real Apple Music persistent ID (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        result = bootstrap(args.db, args.persistent_id)
    except RuntimeError as error:
        print(f"bootstrap_real_tracks: {error}", file=sys.stderr)
        return 1
    print(
        f"bootstrapped: read={result['read']} added={result['added']} "
        f"skipped={result['skipped']}"
    )
    readback(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
