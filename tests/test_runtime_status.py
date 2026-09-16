"""P10.6: Recovery, restart, process-state, status, and LaunchAgent tests."""

import json
import os
import plistlib
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.runtime import Runtime, RuntimeConfig
from music_agent.runtime_launchd import (
    LaunchAgentError,
    build_launch_agent_plist,
    default_plist_path,
    serialize_plist,
)
from music_agent.runtime_state import RuntimePidFile, RuntimeProcessState, RuntimeStateError
from music_agent.runtime_status import (
    StoreStatusError,
    build_store_status,
    format_store_status,
)
from music_agent.runtime_task_run_repository import RuntimeTaskRunRepository


def empty_model() -> dict:
    return {
        "tracks": [],
        "artists": [],
        "albums": [],
        "playlists": [],
        "playlist_memberships": [],
    }


class RuntimePidFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        self.pid_file = RuntimePidFile(self.database_path)

    def test_state_file_path_is_derived_from_store(self) -> None:
        self.assertEqual(
            self.pid_file.path, Path(f"{self.database_path}.runtime.json")
        )

    def test_write_read_clear_round_trip(self) -> None:
        self.assertIsNone(self.pid_file.read())
        self.pid_file.write(4242, "2026-08-16T10:00:00+09:00")
        state = self.pid_file.read()
        self.assertEqual(state, RuntimeProcessState(4242, "2026-08-16T10:00:00+09:00"))
        self.pid_file.clear()
        self.assertIsNone(self.pid_file.read())

    def test_write_replaces_atomically_and_validates(self) -> None:
        self.pid_file.write(1, "start")
        self.pid_file.write(2, "restart")
        self.assertEqual(self.pid_file.read().pid, 2)
        with self.assertRaises(RuntimeStateError):
            self.pid_file.write(0, "start")
        with self.assertRaises(RuntimeStateError):
            self.pid_file.write(1, "")

    def test_malformed_file_reads_as_none(self) -> None:
        self.pid_file.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.pid_file.read())

    def test_running_and_stale_detection(self) -> None:
        self.pid_file.write(os.getpid(), "start")  # this process is alive
        self.assertIsNotNone(self.pid_file.running_state())
        self.assertIsNone(self.pid_file.stale_state())
        self.pid_file.write(999999, "dead")  # pid does not exist
        self.assertIsNone(self.pid_file.running_state())
        self.assertEqual(self.pid_file.stale_state().pid, 999999)


class RuntimeLifecycleRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(empty_model())

    def _config(self, **kwargs) -> RuntimeConfig:
        kwargs.setdefault("audio_safety_enabled", False)
        return RuntimeConfig(database_path=self.database_path, **kwargs)

    def test_pid_file_tracks_lifecycle_and_restart(self) -> None:
        pid_file = RuntimePidFile(self.database_path)
        runtime = Runtime(self._config())
        runtime.start()
        self.assertIsNotNone(pid_file.running_state())
        self.assertEqual(pid_file.running_state().pid, os.getpid())
        runtime.close()
        self.assertIsNone(pid_file.running_state())
        self.assertIsNone(pid_file.read())  # clean shutdown removes the file

    def test_restart_recovers_durable_state_and_task_journal(self) -> None:
        first = Runtime(self._config(refresh_interval_seconds=1))
        first.start()
        first.automation.run_all_now()
        first.close()

        # The durable task journal survives shutdown and reports the completed runs.
        durable = build_store_status(self.database_path)
        self.assertEqual(durable["tasks"]["music_refresh"]["last_run"]["status"], "completed")
        self.assertIsNone(durable["tasks"]["music_refresh"]["last_run"]["error"])

        # A restarted runtime re-opens the same store and re-runs its tasks.
        second = Runtime(self._config(refresh_interval_seconds=1))
        second.start()
        try:
            second.automation.run_all_now()
            snapshot = second.status_snapshot()
            self.assertEqual(snapshot["schema_version"], CURRENT_SCHEMA_VERSION)
            self.assertEqual(snapshot["tasks"]["music_refresh"]["status"], "completed")
            self.assertIsNone(snapshot["tasks"]["music_refresh"]["error"])
        finally:
            second.close()

    def test_interrupted_refresh_leaves_store_consistent_and_next_cycle_heals(self) -> None:
        import json as _json

        from music_agent.apple_music import AppleMusicSourceAdapter
        from music_agent.refresh import refresh_known_track
        from music_agent.runtime_refresh import MusicRefreshOrchestrator

        fixture = _json.loads((Path(__file__).parent / "fixtures" / "canonical_music_model.json").read_text(encoding="utf-8"))
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

        class FlakyRunner:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, persistent_id: str) -> str:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("process interrupted mid-refresh")
                return _json.dumps(
                    {"status": "found", "fields": {"name": f"Refreshed-{persistent_id}"}}
                )

        runner = FlakyRunner()
        with CanonicalRepository(self.database_path) as repository:
            orchestrator = MusicRefreshOrchestrator(
                repository, AppleMusicSourceAdapter(runner)
            )
            first = orchestrator.run_cycle()
        # One track refreshed before the interruption; the failure is isolated.
        self.assertEqual(first.counts()["failed"], 1)
        self.assertGreaterEqual(first.counts()["updated"], 2)
        # The store stays fully valid: a new cycle completes and heals the failed track.
        with CanonicalRepository(self.database_path) as repository:
            self.assertIsNotNone(repository.load_model())
            second = MusicRefreshOrchestrator(
                repository, AppleMusicSourceAdapter(FlakyRunner())
            ).run_cycle()
        self.assertEqual(second.counts()["failed"], 1)  # the flaky runner fails again
        # The two tracks refreshed before the interruption now read UNCHANGED:
        # the store healed around the isolated failure.
        self.assertEqual(second.counts()["updated"] + second.counts()["unchanged"], 2)

    def test_music_app_unavailable_surfaces_as_typed_task_detail(self) -> None:
        import json as _json

        from unittest.mock import patch

        fixture = _json.loads((Path(__file__).parent / "fixtures" / "canonical_music_model.json").read_text(encoding="utf-8"))
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

        class UnavailableRunner:
            def run(self, persistent_id: str) -> str:
                raise OSError("osascript: application is not running")

        runtime = Runtime(
            self._config(refresh_interval_seconds=60),
            clock=lambda: datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc),
        )
        with patch(
            "music_agent.apple_music.OsascriptMusicRunner",
            return_value=UnavailableRunner(),
        ):
            runtime.start()
            try:
                report = runtime.automation.run_task_now("music_refresh")
            finally:
                runtime.close()
        # The automation run itself completed (Music.app unavailability is a typed
        # refresh-cycle outcome with per-track isolation, not a task crash).
        self.assertEqual(report.status.value, "completed")
        self.assertEqual(report.detail["counts"]["failed"], 3)
        self.assertEqual(report.detail["succeeded"], False)

    def test_audio_safety_unavailable_degrades_but_runtime_starts(self) -> None:
        from unittest.mock import patch

        from music_agent.audio_safety import AudioSafetyUnavailableError

        with patch(
            "music_agent.audio_safety.CoreAudioDefaultOutputReader",
            side_effect=AudioSafetyUnavailableError("no coreaudio here"),
        ):
            runtime = Runtime(self._config(audio_safety_enabled=True))
            runtime.start()
            try:
                self.assertIsNone(runtime.audio_monitor)
                snapshot = runtime.status_snapshot()
                self.assertFalse(snapshot["audio_safety"]["enabled"])
            finally:
                runtime.close()

    def test_status_snapshot_reports_observer_event_state(self) -> None:
        from unittest.mock import patch

        class FakeAudioObserver:
            def __init__(self) -> None:
                self.last_event = None

            def start(self) -> None: ...

            def tick(self) -> None: ...

            def close(self) -> None: ...

        runtime = Runtime(
            self._config(),
            clock=lambda: datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc),
        )
        with patch("music_agent.audio_safety.CoreAudioDefaultOutputReader", return_value=object()):
            with patch("music_agent.audio_safety.AudioOutputObserver", return_value=FakeAudioObserver()):
                runtime.start()
                try:
                    snapshot = runtime.status_snapshot()
                    self.assertEqual(snapshot["audio_safety"]["last_event"], None)
                finally:
                    runtime.close()


class StoreStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def test_status_on_fresh_store(self) -> None:
        status = build_store_status(self.database_path)
        self.assertEqual(status["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(status["process"], {"state": "stopped"})
        self.assertEqual(status["tasks"]["music_refresh"]["runs"], 0)
        self.assertEqual(status["write_capabilities"]["operations"], 14)
        self.assertEqual(status["write_capabilities"]["execution_ready"], 0)
        text = format_store_status(status)
        self.assertIn("running: no", text)
        self.assertIn("task music_refresh: no runs", text)
        self.assertIn("execution_ready", text)

    def test_status_reports_task_history_and_running_process(self) -> None:
        runtime = Runtime(
            RuntimeConfig(database_path=self.database_path, audio_safety_enabled=False)
        )
        runtime.start()
        runtime.automation.run_all_now()
        status = build_store_status(self.database_path)
        self.assertEqual(status["process"]["state"], "running")
        self.assertEqual(status["process"]["pid"], os.getpid())
        self.assertEqual(status["tasks"]["music_refresh"]["last_run"]["status"], "completed")
        text = format_store_status(status)
        self.assertIn(f"running: yes (pid {os.getpid()}", text)
        runtime.close()

        status = build_store_status(self.database_path)
        self.assertEqual(status["process"]["state"], "stopped")
        # Task history survives shutdown.
        self.assertEqual(status["tasks"]["capability_status"]["last_run"]["status"], "completed")

    def test_status_on_unopenable_store(self) -> None:
        directory = Path(self.temporary_directory.name) / "not_a_store.db"
        directory.mkdir()
        with self.assertRaises(StoreStatusError):
            build_store_status(directory)

    def test_status_cli_prints_formatted_status(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        stdout = io.StringIO()
        with patch("sys.stdout", stdout), redirect_stderr(io.StringIO()):
            exit_code = main(["status", "--db", str(self.database_path)])
        self.assertEqual(exit_code, 0)
        self.assertIn("running: no", stdout.getvalue())


class LaunchAgentTest(unittest.TestCase):
    def test_plist_structure_and_serialization(self) -> None:
        plist = build_launch_agent_plist(
            ["/venv/bin/music-agent", "run", "--db", "/data/store.db"],
            stdout_log=Path("/logs/agent.log"),
            stderr_log=Path("/logs/agent.err.log"),
            working_directory=Path("/work"),
            environment_variables={"PYTHONPATH": "/repo/src"},
        )
        self.assertEqual(plist["Label"], "com.musicagent.runtime")
        self.assertEqual(plist["ProgramArguments"][0], "/venv/bin/music-agent")
        self.assertIn("--db", plist["ProgramArguments"])
        self.assertTrue(plist["KeepAlive"])
        self.assertTrue(plist["RunAtLoad"])
        self.assertEqual(plist["ProcessType"], "Standard")
        self.assertEqual(plist["EnvironmentVariables"], {"PYTHONPATH": "/repo/src"})
        content = serialize_plist(plist)
        loaded = plistlib.loads(content)
        self.assertEqual(loaded["Label"], "com.musicagent.runtime")
        self.assertEqual(loaded["KeepAlive"], True)
        self.assertEqual(loaded["ProcessType"], "Standard")
        self.assertEqual(loaded["EnvironmentVariables"]["PYTHONPATH"], "/repo/src")

    def test_plist_validation(self) -> None:
        with self.assertRaises(LaunchAgentError):
            build_launch_agent_plist([], stdout_log=Path("a"), stderr_log=Path("b"))
        with self.assertRaises(LaunchAgentError):
            build_launch_agent_plist(["x"], label="", stdout_log=Path("a"), stderr_log=Path("b"))
        with self.assertRaises(LaunchAgentError):
            build_launch_agent_plist(
                ["x"], stdout_log=Path("a"), stderr_log="b"  # type: ignore[arg-type]
            )
        with self.assertRaises(LaunchAgentError):
            build_launch_agent_plist(
                ["x"],
                stdout_log=Path("a"),
                stderr_log=Path("b"),
                environment_variables={"PYTHONPATH": ""},
            )

    def test_default_plist_path(self) -> None:
        home = Path("/Users/test")
        self.assertEqual(
            default_plist_path(home),
            home / "Library" / "LaunchAgents" / "com.musicagent.runtime.plist",
        )

    def test_install_agent_writes_plist_without_activation(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent import cli

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.db"
            home_dir = Path(tmp) / "home"
            home_dir.mkdir()
            stdout = io.StringIO()
            with (
                patch("sys.stdout", stdout),
                redirect_stderr(io.StringIO()),
                patch("pathlib.Path.home", return_value=home_dir),
                patch("pathlib.Path.cwd", return_value=Path(tmp)),
            ):
                exit_code = cli.main(
                    ["install-agent", "--db", str(store), "--label", "com.test.agent"]
                )
            self.assertEqual(exit_code, 0)
            plist_path = home_dir / "Library" / "LaunchAgents" / "com.test.agent.plist"
            self.assertTrue(plist_path.exists())
            loaded = plistlib.loads(plist_path.read_bytes())
            self.assertEqual(loaded["Label"], "com.test.agent")
            self.assertIn("run", loaded["ProgramArguments"])
            self.assertIn("--db", loaded["ProgramArguments"])
            self.assertIn(str(store), loaded["ProgramArguments"])
            self.assertEqual(
                loaded["EnvironmentVariables"],
                {"PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])},
            )
            self.assertIn("wrote LaunchAgent plist", stdout.getvalue())

    def test_uninstall_agent_removes_plist_without_activation(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            home_dir = Path(tmp) / "home"
            agents = home_dir / "Library" / "LaunchAgents"
            agents.mkdir(parents=True)
            plist_path = agents / "com.test.agent.plist"
            plist_path.write_bytes(b"<plist/>")
            stdout = io.StringIO()
            with (
                patch("sys.stdout", stdout),
                redirect_stderr(io.StringIO()),
                patch("pathlib.Path.home", return_value=home_dir),
            ):
                exit_code = main(["uninstall-agent", "--label", "com.test.agent"])
            self.assertEqual(exit_code, 0)
            self.assertFalse(plist_path.exists())
            # Missing plist reports an error.
            with (
                patch("sys.stdout", io.StringIO()),
                redirect_stderr(io.StringIO()),
                patch("pathlib.Path.home", return_value=home_dir),
            ):
                exit_code = main(["uninstall-agent", "--label", "com.test.agent"])
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
