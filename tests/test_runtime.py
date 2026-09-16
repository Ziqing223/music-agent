"""P10.1: Runtime composition root and lifecycle tests."""

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.cli import build_parser
from music_agent.repository import CURRENT_SCHEMA_VERSION
from music_agent.runtime import (
    Runtime,
    RuntimeConfig,
    RuntimeLifecycleError,
    RuntimeStartupError,
    utc_now,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_valid_config_accepts_defaults(self) -> None:
        config = RuntimeConfig(database_path=Path("store.db"))
        self.assertEqual(config.music_command_timeout_seconds, 10.0)
        self.assertEqual(config.log_level, "INFO")

    def test_rejects_non_path_database_path(self) -> None:
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path="store.db")  # type: ignore[arg-type]

    def test_rejects_empty_database_path(self) -> None:
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path(""))

    def test_rejects_non_positive_timeout(self) -> None:
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), music_command_timeout_seconds=0)

    def test_rejects_unsupported_log_level(self) -> None:
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), log_level="VERBOSE")


class RuntimeLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.database_path = Path(self._tmp.name) / "store.db"

    def _config(self) -> RuntimeConfig:
        return RuntimeConfig(database_path=self.database_path, audio_safety_enabled=False)

    def test_start_opens_store_and_applies_migrations(self) -> None:
        runtime = Runtime(self._config())
        self.assertFalse(runtime.started)
        runtime.start()
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.schema_version, CURRENT_SCHEMA_VERSION)
        runtime.close()
        self.assertTrue(runtime.closed)

    def test_start_is_idempotent(self) -> None:
        runtime = Runtime(self._config())
        runtime.start()
        runtime.start()
        self.assertEqual(runtime.schema_version, CURRENT_SCHEMA_VERSION)
        runtime.close()

    def test_close_is_idempotent(self) -> None:
        runtime = Runtime(self._config())
        runtime.start()
        runtime.close()
        runtime.close()
        self.assertTrue(runtime.closed)

    def test_run_refuses_after_close(self) -> None:
        runtime = Runtime(self._config())
        runtime.start()
        runtime.close()
        with self.assertRaises(RuntimeLifecycleError):
            runtime.run(threading.Event())

    def test_run_starts_implicitly_and_stops_on_event(self) -> None:
        runtime = Runtime(self._config())
        stop_event = threading.Event()
        stop_event.set()
        runtime.run(stop_event)
        self.assertTrue(runtime.started)
        runtime.close()

    def test_run_refuses_non_event(self) -> None:
        runtime = Runtime(self._config())
        runtime.start()
        with self.assertRaises(RuntimeLifecycleError):
            runtime.run(None)  # type: ignore[arg-type]
        runtime.close()

    def test_schema_mismatch_fails_startup(self) -> None:
        runtime = Runtime(self._config())
        runtime.start()
        runtime.close()
        # A foreign schema version (as if written by newer code) must refuse to run.
        import sqlite3

        connection = sqlite3.connect(self.database_path)
        connection.execute("INSERT INTO schema_migrations(version) VALUES (999)")
        connection.commit()
        connection.close()
        with self.assertRaises(RuntimeStartupError):
            Runtime(self._config()).start()

    def test_component_hooks_run_in_order(self) -> None:
        events: list[str] = []

        class Probe:
            def __init__(self, name: str) -> None:
                self.name = name

            def start(self) -> None:
                events.append(f"start:{self.name}")

            def tick(self) -> None:
                events.append(f"tick:{self.name}")

            def close(self) -> None:
                events.append(f"close:{self.name}")

        runtime = Runtime(self._config())
        runtime._register_component("a", Probe("a"))
        runtime._register_component("b", Probe("b"))
        runtime.start()
        self.assertEqual(events, ["start:a", "start:b"])
        events.clear()
        runtime.close()
        self.assertEqual(events, ["close:b", "close:a"])

    def test_component_requires_all_hooks(self) -> None:
        runtime = Runtime(self._config())
        with self.assertRaises(RuntimeLifecycleError):
            runtime._register_component("bad", object())

    def test_component_tick_runs_until_stop_event(self) -> None:
        ticks: list[int] = []

        class TickingProbe:
            def __init__(self) -> None:
                self.count = 0

            def start(self) -> None: ...

            def tick(self) -> None:
                self.count += 1
                ticks.append(self.count)
                if self.count >= 2:
                    # Simulate a component that triggers shutdown from inside a tick.
                    stop_event.set()

            def close(self) -> None: ...

        runtime = Runtime(self._config())
        stop_event = threading.Event()
        runtime._register_component("ticker", TickingProbe())
        runtime.run(stop_event)
        self.assertEqual(ticks, [1, 2])
        runtime.close()

    def test_utc_now_is_timezone_aware(self) -> None:
        self.assertIsNotNone(utc_now().tzinfo)
        self.assertEqual(utc_now().tzinfo, timezone.utc)
        self.assertIsInstance(utc_now(), datetime)


class CliTest(unittest.TestCase):
    def test_run_requires_db(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run"])

    def test_run_parses_db_and_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--db", "store.db"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.db, Path("store.db"))
        self.assertEqual(args.music_command_timeout, 10.0)
        self.assertEqual(args.log_level, "INFO")

    def test_run_rejects_unknown_log_level(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--db", "store.db", "--log-level", "LOUD"])

    def test_requires_subcommand(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_main_entry_point_exists(self) -> None:
        from music_agent.cli import main

        self.assertIsNotNone(main)
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
