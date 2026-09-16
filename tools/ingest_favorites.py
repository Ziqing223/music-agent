"""P10.10: One-time read-only importer of the user's real Vault favorites into canonical tracks.

Reads ``<vault>/03_Music/Music_Agent/01_Library/favorites.csv`` (title,artist,language,category --
the user's genuine seed data) and creates canonical tracks (plus one Artist entity per
unique artist name, linked through ``artist_ids``) in the given durable store. This is a
projection INTO the canonical store (SQLite remains the source of truth), read-only
toward the Vault: no Vault file is ever modified here.

Mapping: the CSV has no persistent IDs, so ``apple_music_persistent_id`` stays null; the
category becomes both the genre and the agent tag. Re-running is idempotent: an existing
track with the same (title, artist) is skipped, never duplicated; artists dedupe by name.

Usage: PYTHONPATH=src .venv/bin/python tools/ingest_favorites.py --vault /path/to/Vault --db /path/to/store.db
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from uuid import uuid4

from music_agent.repository import CanonicalRepository

CSV_RELATIVE_PATH = Path("03_Music") / "Music_Agent" / "01_Library" / "favorites.csv"


def load_favorites(vault_root: Path) -> list[dict[str, str]]:
    path = vault_root / CSV_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"favorites.csv not found at {path}")
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            artist = (row.get("artist") or "").strip()
            category = (row.get("category") or "").strip()
            if not title or not artist:
                continue
            rows.append({"title": title, "artist": artist, "category": category})
    return rows


def _library_state() -> dict:
    return {
        "favorited": None,
        "disliked": None,
        "rating": None,
        "play_count": None,
        "skip_count": None,
        "added_to_library_at": None,
        "last_played_at": None,
    }


def ingest(vault_root: Path, database_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Import all CSV rows once; returns {added, skipped, artists_added, total}."""
    rows = load_favorites(vault_root)
    with CanonicalRepository(database_path) as repository:
        model = repository.load_model()
        existing_tracks = {(track["name"], tuple(track["artist_ids"])) for track in model["tracks"]}
        artist_id_by_name = {artist["name"]: artist["id"] for artist in model["artists"]}
        existing_track_names = {name for name, _ in existing_tracks}
        artists_added = 0
        added = 0
        skipped = 0
        for row in rows:
            title = row["title"]
            artist_name = row["artist"]
            if artist_name not in artist_id_by_name:
                artist_id_by_name[artist_name] = f"art_{uuid4()}"
                model["artists"].append(
                    {
                        "id": artist_id_by_name[artist_name],
                        "external_ids": {"apple_music_persistent_id": None},
                        "name": artist_name,
                    }
                )
                artists_added += 1
            if title in existing_track_names:
                skipped += 1
                continue
            model["tracks"].append(
                {
                    "id": f"trk_{uuid4()}",
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": title,
                    "artist_ids": [artist_id_by_name[artist_name]],
                    "album_id": None,
                    "duration_ms": None,
                    "genres": [row["category"]] if row["category"] else [],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": None,
                    "composer": None,
                    "library_state": _library_state(),
                    "agent_metadata": {"tags": [row["category"]] if row["category"] else []},
                }
            )
            existing_track_names.add(title)
            added += 1
        if (added or artists_added) and not dry_run:
            repository.save_model(model)
    return {
        "added": added,
        "skipped": skipped,
        "artists_added": artists_added,
        "total": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True, metavar="PATH")
    parser.add_argument("--db", type=Path, required=True, metavar="PATH")
    args = parser.parse_args(argv)
    try:
        result = ingest(args.vault, args.db)
    except (FileNotFoundError, ValueError) as error:
        print(f"ingest_favorites: {error}", file=sys.stderr)
        return 1
    print(
        f"ingested favorites: total={result['total']} added={result['added']} "
        f"skipped={result['skipped']} artists_added={result['artists_added']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
