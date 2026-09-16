"""P10 daily-use: low-latency playback CLI tests (fake runner; no real Music.app)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from music_agent.agent_request_journal_repository import AgentRequestJournalRepository
from music_agent.repository import CanonicalRepository

CLIENT = "agt_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
FULL = f"{CLIENT}:full"
READONLY = f"{CLIENT}:read_only"

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
BOUND_TRACK = "trk_11111111-1111-4111-8111-111111111111"  # SYNTH-TRACK-001
UNBOUND_TRACK = "trk_33333333-3333-4333-8333-333333333333"


class FakeRunner:
    """Scriptable playback runner (production adapter wraps it)."""

    def __init__(self, now_playing_raw: str = "stopped", failure: Exception | None = None) -> None:
        self.commands: list[tuple[str, tuple]] = []
        self.now_playing_raw = now_playing_raw
        self.failure = failure

    def _record(self, name: str, args: tuple = ()) -> None:
        self.commands.append((name, args))
        if self.failure is not None:
            raise self.failure

    def read_player_state(self) -> str:
        return "stopped"

    def read_now_playing(self) -> str:
        self._record("read_now_playing")
        return self.now_playing_raw

    def pause(self) -> None:
        self._record("pause")

    def play(self) -> None:
        self._record("play")

    def next_track(self) -> None:
        self._record("next_track")

    def previous_track(self) -> None:
        self._record("previous_track")

    def play_track(self, persistent_id: str) -> None:
        self._record("play_track", (persistent_id,))


class PlaybackCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _run(self, argv: list[str], runner: FakeRunner) -> tuple[int, str, str]:
        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("sys.stdout", stdout),
            redirect_stderr(stderr),
            patch(
                "music_agent.playback_control.OsascriptPlaybackRunner",
                return_value=runner,
            ),
        ):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_pause_play_next_previous_commands(self) -> None:
        runner = FakeRunner()
        for subcommand, expected in (
            ("pause", "pause"),
            ("play", "play"),
            ("next", "next_track"),
            ("previous", "previous_track"),
        ):
            exit_code, stdout, _ = self._run(
                ["playback", subcommand, "--db", str(self.database_path),
                 "--agent-client", FULL],
                runner,
            )
            self.assertEqual(exit_code, 0, subcommand)
            summary = json.loads(stdout)
            self.assertEqual(summary["outcome"], "ok")
            self.assertEqual(summary["payload"]["command"], expected)
            self.assertGreater(summary["total_ms"], 0)
            self.assertEqual(summary["replayed"], False)
        self.assertEqual(
            [name for name, _ in runner.commands], ["pause", "play", "next_track", "previous_track"]
        )

    def test_play_track_resolves_binding(self) -> None:
        runner = FakeRunner()
        exit_code, stdout, _ = self._run(
            ["playback", "play-track", BOUND_TRACK, "--db", str(self.database_path),
             "--agent-client", FULL],
            runner,
        )
        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["payload"]["persistent_id"], "SYNTH-TRACK-001")
        self.assertEqual(runner.commands, [("play_track", ("SYNTH-TRACK-001",))])

    def test_play_track_unknown_track_fails_closed(self) -> None:
        runner = FakeRunner()
        exit_code, stdout, stderr = self._run(
            ["playback", "play-track", "trk_00000000-0000-4000-8000-000000000000",
             "--db", str(self.database_path), "--agent-client", FULL],
            runner,
        )
        self.assertEqual(exit_code, 1)
        summary = json.loads(stdout)
        self.assertEqual(summary["outcome"], "execution_error")
        self.assertEqual(summary["error_code"], "canonical_entity_not_found")
        self.assertEqual(runner.commands, [])
        self.assertNotIn("Traceback", stderr)

    def test_play_track_missing_binding_fails_closed(self) -> None:
        runner = FakeRunner()
        exit_code, stdout, _ = self._run(
            ["playback", "play-track", UNBOUND_TRACK, "--db", str(self.database_path),
             "--agent-client", FULL],
            runner,
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(stdout)["error_code"], "playback_unavailable")
        self.assertEqual(runner.commands, [])

    def test_now_playing_typed_payload(self) -> None:
        runner = FakeRunner(now_playing_raw="REAL-PID-9\t某歌\t某艺人\t某专辑\tplaying")
        exit_code, stdout, _ = self._run(
            ["playback", "now-playing", "--db", str(self.database_path),
             "--agent-client", FULL],
            runner,
        )
        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["payload"]["now_playing"]["name"], "某歌")
        self.assertEqual(summary["payload"]["now_playing"]["persistent_id"], "REAL-PID-9")

    def test_permission_denial_returns_typed_exit(self) -> None:
        runner = FakeRunner()
        exit_code, stdout, _ = self._run(
            ["playback", "pause", "--db", str(self.database_path),
             "--agent-client", READONLY],
            runner,
        )
        self.assertEqual(exit_code, 1)
        summary = json.loads(stdout)
        self.assertEqual(summary["outcome"], "permission_denied")
        self.assertEqual(runner.commands, [])

    def test_command_failure_is_typed_without_traceback(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        runner = FakeRunner(failure=PlaybackControlUnavailableError("Application isn't running"))
        exit_code, stdout, stderr = self._run(
            ["playback", "pause", "--db", str(self.database_path), "--agent-client", FULL],
            runner,
        )
        self.assertEqual(exit_code, 1)
        summary = json.loads(stdout)
        self.assertEqual(summary["error_code"], "playback_command_failed")
        self.assertIn("Application isn't running", summary["error_message"])
        self.assertNotIn("Traceback", stderr)

    def test_each_invocation_journals_one_fresh_request(self) -> None:
        runner = FakeRunner()
        self._run(
            ["playback", "pause", "--db", str(self.database_path), "--agent-client", FULL],
            runner,
        )
        with AgentRequestJournalRepository(self.database_path) as journal:
            first = journal.list()
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].request.tool, "pause")
        self._run(
            ["playback", "pause", "--db", str(self.database_path), "--agent-client", FULL],
            runner,
        )
        with AgentRequestJournalRepository(self.database_path) as journal:
            rows = journal.list()
            self.assertEqual(len(rows), 2)  # fresh request per explicit user action
            self.assertNotEqual(rows[0].request.request_id, rows[1].request.request_id)
            self.assertEqual(runner.commands.count(("pause", ())), 2)

    def test_missing_agent_client_is_config_error(self) -> None:
        runner = FakeRunner()
        exit_code, _, stderr = self._run(
            ["playback", "pause", "--db", str(self.database_path)],
            runner,
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("exactly one --agent-client", stderr)

    def test_parser_surface(self) -> None:
        from music_agent.cli import build_parser

        parser = build_parser()
        for subcommand in ("pause", "play", "next", "previous", "now-playing"):
            args = parser.parse_args(["playback", subcommand, "--db", "s.db", "--agent-client", FULL])
            self.assertEqual(args.playback_command, subcommand)
        args = parser.parse_args(
            ["playback", "play-track", BOUND_TRACK, "--db", "s.db", "--agent-client", FULL]
        )
        self.assertEqual(args.track_id, BOUND_TRACK)


if __name__ == "__main__":
    unittest.main()
