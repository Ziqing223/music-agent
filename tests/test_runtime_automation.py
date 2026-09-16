"""P10.4: Automation engine, task-run journal, and runtime wiring tests."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.runtime import Runtime, RuntimeConfig
from music_agent.runtime_automation import (
    AutomationEngine,
    AutomationTask,
    AutomationValidationError,
    TaskRunReport,
)
from music_agent.runtime_task_run_repository import (
    RuntimeTaskRunRepository,
    TaskRunRecord,
    TaskRunRepositoryValidationError,
    TaskRunStatus,
    generate_task_run_id,
)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 8, 16, 9, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class AutomationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def _engine(self, clock: FakeClock | None = None) -> tuple[AutomationEngine, FakeClock]:
        clock = clock or FakeClock()
        engine = AutomationEngine(self.database_path, clock=clock)
        self.addCleanup(engine.close)
        return engine, clock

    def test_registration_validates(self) -> None:
        engine, _ = self._engine()
        engine.register(AutomationTask("task_a", 60, lambda: {"ok": True}))
        self.assertEqual(engine.task_names, ("task_a",))
        with self.assertRaises(AutomationValidationError):
            engine.register(AutomationTask("task_a", 60, lambda: {}))
        with self.assertRaises(AutomationValidationError):
            engine.register(AutomationTask("", 60, lambda: {}))  # type: ignore[arg-type]
        with self.assertRaises(AutomationValidationError):
            engine.register(AutomationTask("task_b", 0, lambda: {}))  # type: ignore[arg-type]

    def test_first_ticks_drain_due_tasks_one_per_tick(self) -> None:
        engine, clock = self._engine()
        calls: list[str] = []
        engine.register(AutomationTask("refresh", 900, lambda: calls.append("refresh") or {}))
        engine.register(AutomationTask("status", 3600, lambda: calls.append("status") or {}))
        engine.start()
        engine.tick()
        self.assertEqual(calls, ["refresh"])
        # The second due task waits for the next tick: one task per tick.
        engine.tick()
        self.assertEqual(calls, ["refresh", "status"])
        # Nothing is due any more.
        engine.tick()
        self.assertEqual(calls, ["refresh", "status"])

    def test_interval_scheduling_with_injected_clock(self) -> None:
        engine, clock = self._engine()
        calls: list[tuple[str, datetime]] = []
        engine.register(AutomationTask("refresh", 900, lambda: calls.append(("refresh", clock())) or {}))
        engine.register(AutomationTask("status", 3600, lambda: calls.append(("status", clock())) or {}))
        engine.start()
        engine.tick()
        engine.tick()
        self.assertEqual([name for name, _ in calls], ["refresh", "status"])
        clock.advance(899)
        engine.tick()
        self.assertEqual(len(calls), 2)  # refresh at 899s: not yet due
        clock.advance(1)  # 900s: refresh due, status not
        engine.tick()
        self.assertEqual([name for name, _ in calls], ["refresh", "status", "refresh"])
        clock.advance(2700)  # 3600s total: both due -- one per tick, in order
        engine.tick()
        self.assertEqual([name for name, _ in calls], ["refresh", "status", "refresh", "refresh"])
        engine.tick()
        self.assertEqual([name for name, _ in calls], ["refresh", "status", "refresh", "refresh", "status"])

    def test_cadence_survives_one_per_tick_staggering(self) -> None:
        # H: staggering due tasks across ticks must not lose any cadence --
        # every task still fires at its own interval, just one per tick.
        engine, clock = self._engine()
        calls: list[str] = []
        engine.register(AutomationTask("a", 60, lambda: calls.append("a") or {}))
        engine.register(AutomationTask("b", 60, lambda: calls.append("b") or {}))
        engine.start()
        engine.tick()
        engine.tick()
        self.assertEqual(calls, ["a", "b"])
        clock.advance(60)  # both due again
        engine.tick()
        self.assertEqual(calls, ["a", "b", "a"])
        engine.tick()
        self.assertEqual(calls, ["a", "b", "a", "b"])

    def test_failed_task_is_isolated_and_retries_after_full_interval(self) -> None:
        engine, clock = self._engine()
        attempts: list[str] = []

        def flaky() -> dict[str, object]:
            attempts.append("attempt")
            if len(attempts) == 1:
                raise RuntimeError("music unavailable")
            return {"ok": True}

        engine.register(AutomationTask("refresh", 900, flaky))
        engine.register(AutomationTask("status", 3600, lambda: {"healthy": True}))
        engine.start()
        engine.tick()  # refresh fails
        engine.tick()  # status still runs next tick: failure isolation
        self.assertEqual(attempts, ["attempt"])
        reports = engine.last_reports()
        self.assertEqual(reports["refresh"].status, TaskRunStatus.FAILED)
        self.assertIn("music unavailable", reports["refresh"].error or "")
        self.assertEqual(reports["status"].status, TaskRunStatus.COMPLETED)
        # No hammering: the failed task does not re-run until its interval elapses.
        clock.advance(10)
        engine.tick()
        self.assertEqual(attempts, ["attempt"])
        clock.advance(890)
        engine.tick()
        self.assertEqual(attempts, ["attempt", "attempt"])
        self.assertEqual(engine.last_reports()["refresh"].status, TaskRunStatus.COMPLETED)

    def test_non_mapping_task_result_fails_closed(self) -> None:
        engine, _ = self._engine()
        engine.register(AutomationTask("bad", 60, lambda: "not-a-mapping"))  # type: ignore[arg-type, return-value]
        engine.start()
        engine.tick()
        report = engine.last_reports()["bad"]
        self.assertEqual(report.status, TaskRunStatus.FAILED)
        self.assertIn("expected a mapping", report.error or "")

    def test_run_task_now_and_run_all_now(self) -> None:
        engine, _ = self._engine()
        engine.register(AutomationTask("a", 60, lambda: {"v": 1}))
        engine.register(AutomationTask("b", 60, lambda: {"v": 2}))
        engine.start()
        with self.assertRaises(AutomationValidationError):
            engine.run_task_now("unknown")
        report = engine.run_task_now("a")
        self.assertEqual(report.task_name, "a")
        self.assertEqual(report.detail, {"v": 1})
        all_reports = engine.run_all_now()
        self.assertEqual([r.task_name for r in all_reports], ["a", "b"])

    def test_runs_are_durable_and_survive_engine_restart(self) -> None:
        engine, clock = self._engine()
        engine.register(AutomationTask("refresh", 900, lambda: {"count": 3}))
        engine.start()
        engine.tick()
        engine.close()

        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("refresh")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, TaskRunStatus.COMPLETED)
        self.assertEqual(runs[0].detail, {"count": 3})
        self.assertEqual(runs[0].task_name, "refresh")

        # A new engine over the same store schedules from scratch (in-memory state),
        # re-runs the task, and appends a second run: refresh is idempotent by design.
        engine2, _ = self._engine(clock)
        engine2.register(AutomationTask("refresh", 900, lambda: {"count": 3}))
        engine2.start()
        engine2.tick()
        engine2.close()
        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("refresh")
        self.assertEqual(len(runs), 2)
        self.assertNotEqual(runs[0].run_id, runs[1].run_id)

    def test_engine_never_runs_before_start(self) -> None:
        engine, _ = self._engine()
        engine.register(AutomationTask("refresh", 900, lambda: {"x": 1}))
        engine.tick()  # no-op: not started
        self.assertEqual(dict(engine.last_reports()), {})

    def test_close_is_idempotent_and_refuses_restart(self) -> None:
        engine, _ = self._engine()
        engine.start()
        engine.close()
        engine.close()
        from music_agent.runtime_automation import AutomationLifecycleError

        with self.assertRaises(AutomationLifecycleError):
            engine.start()


class CooperativeAutomationTest(unittest.TestCase):
    """P15-S2 R8: generator tasks resume one bounded step per tick and wrap
    up (one journal row, one last_run update) only at run completion."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def _engine(self, clock: FakeClock | None = None) -> tuple[AutomationEngine, FakeClock]:
        clock = clock or FakeClock()
        engine = AutomationEngine(self.database_path, clock=clock)
        self.addCleanup(engine.close)
        return engine, clock

    def test_generator_advances_one_step_per_tick_and_serializes(self) -> None:
        engine, _ = self._engine()
        calls: list[str] = []

        def stepped() -> dict[str, object]:
            calls.append("step1")
            yield None
            calls.append("step2")
            yield None
            calls.append("step3")
            return {"steps": 3}

        engine.register(AutomationTask("stepped", 900, stepped))
        engine.register(AutomationTask("other", 900, lambda: calls.append("other") or {}))
        engine.start()
        engine.tick()  # launch + first step
        self.assertEqual(calls, ["step1"])
        self.assertNotIn("stepped", engine.last_reports())  # no wrap-up per step
        engine.tick()  # the active run has priority over the due "other"
        self.assertEqual(calls, ["step1", "step2"])
        engine.tick()  # final step finishes the run
        self.assertEqual(calls, ["step1", "step2", "step3"])
        report = engine.last_reports()["stepped"]
        self.assertEqual(report.status, TaskRunStatus.COMPLETED)
        self.assertEqual(dict(report.detail), {"steps": 3})
        engine.tick()  # only now does the next due task run
        self.assertEqual(calls, ["step1", "step2", "step3", "other"])

    def test_journal_and_last_run_update_once_at_completion(self) -> None:
        engine, clock = self._engine()
        steps: list[int] = []

        def stepped() -> dict[str, object]:
            for index in (1, 2, 3):
                steps.append(index)
                yield None
            return {"steps": 3}

        engine.register(AutomationTask("stepped", 900, stepped))
        engine.start()
        engine.tick()
        clock.advance(890)  # long mid-run pause on the injected clock
        engine.tick()
        engine.tick()
        engine.tick()  # final advance delivers StopIteration: run completes
        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("stepped")
            self.assertEqual(len(runs), 1)  # exactly one row for the whole run
            self.assertEqual(runs[0].status, TaskRunStatus.COMPLETED)
            self.assertEqual(dict(runs[0].detail), {"steps": 3})
            self.assertEqual(runs[0].started_at, "2026-08-16T09:00:00+00:00")
            self.assertEqual(runs[0].finished_at, "2026-08-16T09:14:50+00:00")
        # last_run anchors completion (t0+890), not launch (t0): 10s after
        # completion the task is NOT due -- a launch-anchored last_run would
        # have made it due 900s after t0, i.e. immediately here.
        clock.advance(10)
        engine.tick()
        self.assertEqual(steps, [1, 2, 3])
        clock.advance(890)  # 900s after completion: due again
        engine.tick()
        self.assertEqual(steps, [1, 2, 3, 1])  # a fresh run starts from scratch

    def test_step_exception_fails_run_clears_cursor_and_releases_serialization(self) -> None:
        engine, _ = self._engine()
        calls: list[str] = []

        def flaky():
            calls.append("step1")
            yield None
            calls.append("step2")
            raise RuntimeError("osascript blew up")
            yield None  # pragma: no cover -- makes this a generator function

        engine.register(AutomationTask("flaky", 900, flaky))
        engine.register(AutomationTask("other", 900, lambda: calls.append("other") or {}))
        engine.start()
        engine.tick()
        engine.tick()  # step 2 raises: the run fails and the cursor is released
        report = engine.last_reports()["flaky"]
        self.assertEqual(report.status, TaskRunStatus.FAILED)
        self.assertIn("osascript blew up", report.error or "")
        engine.tick()  # serialization released: "other" runs on the next tick
        self.assertEqual(calls, ["step1", "step2", "other"])
        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("flaky")
            self.assertEqual(len(runs), 1)  # one FAILED row for the whole run
            self.assertEqual(runs[0].status, TaskRunStatus.FAILED)
            self.assertIn("osascript blew up", runs[0].error or "")

    def test_generator_returning_non_mapping_fails_closed(self) -> None:
        engine, _ = self._engine()

        def bad():
            yield None
            return "not-a-mapping"

        engine.register(AutomationTask("bad", 60, bad))
        engine.start()
        engine.tick()
        engine.tick()
        report = engine.last_reports()["bad"]
        self.assertEqual(report.status, TaskRunStatus.FAILED)
        self.assertIn("expected a mapping", report.error or "")

    def test_engine_restart_reruns_generator_from_scratch(self) -> None:
        engine, clock = self._engine()
        steps: list[int] = []

        def stepped() -> dict[str, object]:
            for index in (1, 2, 3):
                steps.append(index)
                yield None
            return {"steps": 3}

        engine.register(AutomationTask("stepped", 900, stepped))
        engine.start()
        engine.tick()
        engine.tick()
        engine.tick()
        engine.tick()  # the last advance delivers StopIteration
        self.assertEqual(steps, [1, 2, 3])
        engine.close()

        # Full restart: the in-memory cursor is gone and scheduling state is
        # empty, so the task re-runs from scratch and appends a second row.
        engine2 = AutomationEngine(self.database_path, clock=clock)
        self.addCleanup(engine2.close)
        engine2.register(AutomationTask("stepped", 900, stepped))
        engine2.start()
        engine2.tick()
        self.assertEqual(steps, [1, 2, 3, 1])  # fresh cursor: from the top
        engine2.tick()
        engine2.tick()
        engine2.tick()  # the last advance delivers StopIteration
        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("stepped")
            self.assertEqual(len(runs), 2)
            self.assertNotEqual(runs[0].run_id, runs[1].run_id)

    def test_run_task_now_and_run_all_now_drain_generators_synchronously(self) -> None:
        engine, _ = self._engine()
        calls: list[str] = []

        def stepped() -> dict[str, object]:
            calls.append("step1")
            yield None
            calls.append("step2")
            yield None
            return {"steps": 2}

        engine.register(AutomationTask("stepped", 900, stepped))
        engine.register(AutomationTask("plain", 900, lambda: {"v": 1}))
        engine.start()
        report = engine.run_task_now("stepped")
        self.assertEqual(report.status, TaskRunStatus.COMPLETED)
        self.assertEqual(dict(report.detail), {"steps": 2})
        self.assertEqual(calls, ["step1", "step2"])
        all_reports = engine.run_all_now()
        self.assertEqual([r.task_name for r in all_reports], ["stepped", "plain"])
        self.assertEqual(calls, ["step1", "step2", "step1", "step2"])
        self.assertEqual(all_reports[1].detail, {"v": 1})


class TaskRunRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def _record(self, **kwargs) -> TaskRunRecord:
        defaults = dict(
            run_id=generate_task_run_id(),
            task_name="music_refresh",
            status=TaskRunStatus.COMPLETED,
            error=None,
            detail={"updated": 1},
            started_at="2026-08-16T09:00:00+00:00",
            finished_at="2026-08-16T09:00:01+00:00",
        )
        defaults.update(kwargs)
        return TaskRunRecord(**defaults)

    def test_record_and_list_round_trip(self) -> None:
        with RuntimeTaskRunRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            first = self._record()
            second = self._record(task_name="capability_status", detail={"operations": 14})
            repository.record_run(first)
            repository.record_run(second)
        with RuntimeTaskRunRepository(self.database_path) as repository:
            runs = repository.list_runs("music_refresh")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].run_id, first.run_id)
            self.assertEqual(runs[0].detail, {"updated": 1})
            self.assertEqual(repository.latest_run("music_refresh").run_id, first.run_id)  # type: ignore[union-attr]
            self.assertEqual(repository.latest_run("unknown_task"), None)

    def test_duplicate_run_id_fails_closed(self) -> None:
        import sqlite3 as _sqlite3

        record = self._record()
        with RuntimeTaskRunRepository(self.database_path) as repository:
            repository.record_run(record)
            with self.assertRaises(_sqlite3.IntegrityError):
                repository.record_run(record)  # same run_id twice

    def test_rows_are_immutable(self) -> None:
        with RuntimeTaskRunRepository(self.database_path) as repository:
            repository.record_run(self._record())
        connection = sqlite3.connect(self.database_path)
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute("UPDATE runtime_task_runs SET status = 'failed'")
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM runtime_task_runs")
        connection.close()

    def test_failed_run_requires_error_and_completed_forbids_it(self) -> None:
        with self.assertRaises(TaskRunRepositoryValidationError):
            self._record(status=TaskRunStatus.FAILED, error=None)
        with self.assertRaises(TaskRunRepositoryValidationError):
            self._record(status=TaskRunStatus.FAILED, error="")
        with self.assertRaises(TaskRunRepositoryValidationError):
            self._record(status=TaskRunStatus.COMPLETED, error="unexpected")

    def test_fresh_store_and_v15_upgrade(self) -> None:
        from unittest.mock import patch

        from music_agent.repository import MIGRATIONS

        # A v15 store (without runtime_task_runs) upgrades in place with data intact.
        v15_migrations = tuple(MIGRATIONS[:15])
        with patch("music_agent.repository.MIGRATIONS", v15_migrations):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 15)
                repository.save_model(
                    {
                        "tracks": [],
                        "artists": [],
                        "albums": [],
                        "playlists": [],
                        "playlist_memberships": [],
                    }
                )
        with RuntimeTaskRunRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)
            repository.record_run(self._record())
        # The pre-existing P01-P09 tables survived the upgrade.
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model()["tracks"], [])
            self.assertEqual(repository.schema_version, 19)


class RuntimeAutomationWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(
                {
                    "tracks": [],
                    "artists": [],
                    "albums": [],
                    "playlists": [],
                    "playlist_memberships": [],
                }
            )

    def test_runtime_wires_automation_engine_with_production_tasks(self) -> None:
        from unittest.mock import patch

        class EmptyDiscovery:
            def list_persistent_ids(self) -> tuple[str, ...]:
                return ()

        runtime = Runtime(
            RuntimeConfig(database_path=self.database_path, audio_safety_enabled=False)
        )
        runtime.start()
        try:
            engine = runtime.automation
            self.assertEqual(
                engine.task_names,
                ("music_refresh", "capability_status", "library_discovery"),
            )
            # One task per tick: the three startup-due tasks drain over ticks
            # against the empty store (discovery enumerates an empty library
            # via a fake, so no osascript is ever spawned in unit tests).
            # music_refresh and capability_status complete in one tick each;
            # library_discovery is a cooperative generator whose enumerate
            # step and completion land on two separate ticks.
            with patch(
                "music_agent.apple_music_library_discovery.AppleMusicLibraryDiscoveryAdapter",
                return_value=EmptyDiscovery(),
            ):
                engine.tick()
                engine.tick()
                engine.tick()
                engine.tick()
            reports = engine.last_reports()
            self.assertEqual(reports["music_refresh"].status, TaskRunStatus.COMPLETED)
            self.assertEqual(reports["music_refresh"].detail["bound_track_count"], 0)
            self.assertEqual(reports["capability_status"].status, TaskRunStatus.COMPLETED)
            self.assertEqual(reports["capability_status"].detail["operations"], 14)
            self.assertEqual(reports["capability_status"].detail["execution_ready"], 0)
            self.assertEqual(reports["library_discovery"].status, TaskRunStatus.COMPLETED)
            self.assertEqual(reports["library_discovery"].detail["counts"]["new"], 0)
        finally:
            runtime.close()

    def test_runtime_automation_records_durable_runs(self) -> None:
        from unittest.mock import patch

        class EmptyDiscovery:
            def list_persistent_ids(self) -> tuple[str, ...]:
                return ()

        runtime = Runtime(
            RuntimeConfig(database_path=self.database_path, audio_safety_enabled=False)
        )
        runtime.start()
        try:
            with patch(
                "music_agent.apple_music_library_discovery.AppleMusicLibraryDiscoveryAdapter",
                return_value=EmptyDiscovery(),
            ):
                runtime.automation.tick()
                runtime.automation.tick()
                runtime.automation.tick()
                runtime.automation.tick()
        finally:
            runtime.close()
        with RuntimeTaskRunRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_runs("music_refresh")), 1)
            self.assertEqual(len(repository.list_runs("capability_status")), 1)
            self.assertEqual(len(repository.list_runs("library_discovery")), 1)

    def test_config_validates_intervals(self) -> None:
        from music_agent.runtime import RuntimeStartupError

        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=self.database_path, refresh_interval_seconds=0)
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(
                database_path=self.database_path, capability_status_interval_seconds=-5
            )


class AutomationCliFlagTest(unittest.TestCase):
    def test_run_accepts_interval_flags(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(
            ["run", "--db", "store.db", "--refresh-interval", "60",
             "--capability-status-interval", "120"]
        )
        self.assertEqual(args.refresh_interval, 60)
        self.assertEqual(args.capability_status_interval, 120)


if __name__ == "__main__":
    unittest.main()
