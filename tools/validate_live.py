"""P10.7: User-run live Music.app validation (real osascript boundary).

These checks talk to the real Music.app and are deliberately NOT part of the automated
test suite -- run them yourself in the terminal so the macOS automation permission
applies to your session. All Music.app interaction here is read-only except the optional
pause check in ``audio-transition``.

Commands (run from the repository root, with the virtualenv):

  PYTHONPATH=src .venv/bin/python tools/validate_live.py player-state
      Read the real Music player state through the production playback adapter.

  PYTHONPATH=src .venv/bin/python tools/validate_live.py refresh-track --persistent-id ID
      Build a fresh temporary store, bind one canonical track to the given real Music
      persistent ID, and run the production refresh cycle against the real Music.app.
      Read-only on Music.app; the canonical store lives in a temp directory and is
      deleted afterwards. Prints the report and the refreshed entity.

  PYTHONPATH=src .venv/bin/python tools/validate_live.py audio
      Print the current default audio output device and its safety classification.

  PYTHONPATH=src .venv/bin/python tools/validate_live.py audio-transition --db PATH
      Start the production runtime (audio safety enabled) against the given store, print
      the monitor state, and wait. WITH MUSIC PLAYING, physically disconnect your
      headphones / turn off your Bluetooth device; the monitor should pause playback
      within a few seconds. Press Ctrl-C twice to stop. This issues a real pause only
      after a proven transition while playing; it never resumes playback.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.repository import SourcePresenceRecord
from music_agent.source_observation import SourcePresence


def player_state() -> int:
    from music_agent.playback_control import MusicPlaybackAdapter, OsascriptPlaybackRunner

    adapter = MusicPlaybackAdapter(OsascriptPlaybackRunner(timeout_seconds=10.0))
    state = adapter.read_player_state()
    print(f"player state: {state.value}")
    return 0


def _live_model(persistent_id: str) -> dict:
    """The canonical fixture with track 0 re-bound to the given live persistent id.

    The full fixture is kept so the model stays graph-valid; the other synthetic
    persistent ids simply read as confirmed_not_found against the real Music.app,
    which the report shows honestly.
    """
    fixture_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "canonical_music_model.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["tracks"][0]["external_ids"]["apple_music_persistent_id"] = persistent_id
    return fixture


def refresh_track(persistent_id: str) -> int:
    from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner
    from music_agent.repository import CanonicalRepository
    from music_agent.runtime_refresh import MusicRefreshOrchestrator

    with tempfile.TemporaryDirectory() as tmp:
        database_path = Path(tmp) / "live_refresh.db"
        model = _live_model(persistent_id)
        with CanonicalRepository(database_path) as repository:
            repository.save_model_with_source_presence(
                model,
                [
                    SourcePresenceRecord(
                        "apple_music",
                        EntityType.TRACK,
                        model["tracks"][0]["id"],
                        "library_tracks",
                        SourcePresence.PRESENT,
                    )
                ],
            )
            adapter = AppleMusicSourceAdapter(OsascriptMusicRunner(timeout_seconds=10.0))
            report = MusicRefreshOrchestrator(repository, adapter).run_cycle()
            counts = report.counts()
            print("refresh cycle:")
            print(f"  bound={report.bound_track_count} skipped_no_binding={report.skipped_no_binding}")
            print(f"  updated={counts['updated']} unchanged={counts['unchanged']} "
                  f"source_not_found={counts['source_not_found']} "
                  f"lookup_failed={counts['source_lookup_failed']} "
                  f"merge_failed={counts['merge_failed']} failed={counts['failed']}")
            for failure in report.failures:
                print(f"  failure: {failure.canonical_id}: {failure.error}")
            track = repository.load_model()["tracks"][0]
            print("refreshed canonical track:")
            print(json.dumps(track, indent=2, ensure_ascii=False))
    return 0


def audio() -> int:
    from music_agent.audio_safety import CoreAudioDefaultOutputReader

    reader = CoreAudioDefaultOutputReader()
    state = reader.read_default_output()
    if state is None:
        print("default output device: UNKNOWN (fail closed)")
        return 1
    print(f"default output device: id={state.device_id} transport={state.transport_type} "
          f"uid={state.uid}")
    print(f"safety classification: {'built-in public speaker' if state.is_built_in_speaker else 'private output'}")
    return 0


def audio_transition(database_path: Path) -> int:
    from music_agent.audio_safety import AudioSafetyEventKind
    from music_agent.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            database_path=database_path,
            audio_safety_enabled=True,
            audio_safety_poll_interval_seconds=1.0,
        )
    )
    runtime.start()
    print("audio safety monitor running. With music playing, disconnect your headphones.")
    print("Press Ctrl-C to stop.")
    try:
        while True:
            event = runtime.audio_monitor.last_event
            if event is not None and event.kind is not AudioSafetyEventKind.NO_ACTION:
                print(f"event: kind={event.kind.value} device={event.device_state} "
                      f"player={event.player_state} error={event.error}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        runtime.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="validate_live.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("player-state", help="read the real Music player state")
    refresh_parser = subparsers.add_parser("refresh-track", help="refresh one real bound track")
    refresh_parser.add_argument("--persistent-id", required=True, metavar="ID")
    subparsers.add_parser("audio", help="print the current default output device")
    transition_parser = subparsers.add_parser(
        "audio-transition", help="watch for a real headphone-disconnect transition"
    )
    transition_parser.add_argument("--db", type=Path, required=True, metavar="PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "player-state":
        return player_state()
    if args.command == "refresh-track":
        return refresh_track(args.persistent_id)
    if args.command == "audio":
        return audio()
    if args.command == "audio-transition":
        return audio_transition(args.db)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
