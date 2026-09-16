"""P11: User-run live Apple Music Catalog validation (real MusicKit boundary).

These checks talk to the real Apple Music Web API and are deliberately NOT part of the
automated test suite -- the automated suite uses fakes/fixtures only. Run these yourself
with your own credentials so the live gate produces real evidence:

Required environment:
  MUSIC_AGENT_APPLE_MUSIC_DEVELOPER_TOKEN   a signed MusicKit developer JWT
  MUSIC_AGENT_APPLE_MUSIC_USER_TOKEN        a Music User Token (add/readback only)

Commands (run from the repository root, with the virtualenv):

  PYTHONPATH=src .venv/bin/python tools/validate_catalog_live.py search --term "起风了"
      Search the real catalog and print the parsed CatalogTrack records (id, name,
      artists, genres, ISRC where supplied). Read-only; proves the transport and the
      parser against the real payload shape.

  PYTHONPATH=src .venv/bin/python tools/validate_catalog_live.py add --db PATH --catalog-id ID --canonical-id trk_... --term "song name"
      Run the full production flow against the real library: baseline -> add ->
      readback -> reconciliation. This MUTATES your Apple Music library (adds one
      song; the add is idempotent upstream). Prints the typed reconciliation result
      and the resulting bindings on the canonical Track. Verifies, with real
      evidence, that the MusicKit library-song id and Music.app persistent ID agree.

Run order for the P11 live gate: search first; only then add, with a song you actually
want in your library. Nothing here is marked VERIFIED until you run it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from music_agent.apple_music_catalog import AppleMusicCatalogAdapter, MusicKitTransport
from music_agent.catalog_library import (
    MusicKitLibraryTransport,
    add_catalog_song_to_library,
)
from music_agent.repository import CanonicalRepository


def _search(args: argparse.Namespace) -> int:
    transport = MusicKitTransport(storefront=args.storefront)
    adapter = AppleMusicCatalogAdapter(transport)
    tracks = adapter.search(args.term, args.limit)
    print(f"catalog search {args.term!r}: {len(tracks)} hit(s)")
    for track in tracks:
        print(
            f"  {track.catalog_id}  {track.name}  by {', '.join(track.artist_names)}"
            f"  album={track.album_name or '-'}  genres={', '.join(track.genres) or '-'}"
            f"  isrc={track.isrc or '-'}  duration_ms={track.duration_ms or '-'}"
        )
    return 0


def _add(args: argparse.Namespace) -> int:
    transport = MusicKitLibraryTransport(storefront=args.storefront)
    with CanonicalRepository(args.db) as repository:
        result = add_catalog_song_to_library(
            repository,
            transport,
            args.catalog_id,
            args.canonical_id,
            term=args.term,
        )
    print(
        f"add={result.add_status} readback={result.readback_status}"
        f" bound_persistent_id={result.bound_persistent_id}"
        f" bound_isrc={result.bound_isrc}"
        + (f" error={result.error}" if result.error else "")
    )
    if result.succeeded:
        with CanonicalRepository(args.db) as repository:
            track = next(
                track for track in repository.load_model()["tracks"]
                if track["id"] == result.canonical_id
            )
            print("canonical track external_ids:", track["external_ids"])
    return 0 if result.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate_catalog_live")
    parser.add_argument("--storefront", default="us")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--term", required=True)
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(func=_search)
    add = subparsers.add_parser("add")
    add.add_argument("--db", required=True)
    add.add_argument("--catalog-id", required=True)
    add.add_argument("--canonical-id", required=True)
    add.add_argument("--term")
    add.set_defaults(func=_add)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
