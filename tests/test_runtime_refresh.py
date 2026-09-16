"""P10.2: Integrated refresh cycle tests (orchestrator + report + CLI)."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.refresh import RefreshStatus
from music_agent.repository import CanonicalRepository
from music_agent.runtime_refresh import MusicRefreshOrchestrator, MusicRefreshReport

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeRunner:
    def __init__(self, outputs: dict[str, str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        output = self.outputs.get(persistent_id)
        if isinstance(output, Exception):
            raise output
        if output is None:
            raise AssertionError(f"unexpected persistent id: {persistent_id}")
        return output


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class MusicRefreshOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def _adapter(self, outputs: dict[str, str | Exception]) -> tuple[AppleMusicSourceAdapter, FakeRunner]:
        runner = FakeRunner(outputs)
        return AppleMusicSourceAdapter(runner), runner

    def test_cycle_refreshes_every_bound_track_and_skips_unbound(self) -> None:
        found = json.dumps({"status": "found", "fields": {"name": "Renamed"}})
        adapter, runner = self._adapter(
            {
                "SYNTH-TRACK-001": found,
                "SYNTH-TRACK-002": found,
                "SYNTH-TRACK-004": '{"status":"confirmed_not_found"}',
            }
        )
        with CanonicalRepository(self.database_path) as repository:
            report = MusicRefreshOrchestrator(repository, adapter).run_cycle()
        self.assertEqual(report.bound_track_count, 3)
        self.assertEqual(report.skipped_no_binding, 1)
        self.assertEqual(sorted(runner.calls), [
            "SYNTH-TRACK-001", "SYNTH-TRACK-002", "SYNTH-TRACK-004",
        ])
        counts = report.counts()
        self.assertEqual(counts["updated"], 2)
        self.assertEqual(counts["source_not_found"], 1)
        self.assertEqual(counts["failed"], 0)
        self.assertTrue(report.succeeded)

    def test_updated_tracks_persist_through_production_save_path(self) -> None:
        found = json.dumps({"status": "found", "fields": {"played_count": 99}})
        adapter, _ = self._adapter({f"SYNTH-TRACK-00{i}": found for i in (1, 2, 4)})
        with CanonicalRepository(self.database_path) as repository:
            MusicRefreshOrchestrator(repository, adapter).run_cycle()
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
        counts = {t["library_state"]["play_count"] for t in loaded["tracks"]}
        self.assertIn(99, counts)

    def test_repeated_cycle_is_idempotent(self) -> None:
        found = json.dumps({"status": "found", "fields": {"name": "Renamed"}})
        adapter, runner = self._adapter({f"SYNTH-TRACK-00{i}": found for i in (1, 2, 4)})
        with CanonicalRepository(self.database_path) as repository:
            orchestrator = MusicRefreshOrchestrator(repository, adapter)
            first = orchestrator.run_cycle()
            second = orchestrator.run_cycle()
        self.assertEqual(first.counts()["updated"], 3)
        self.assertEqual(second.counts()["unchanged"], 3)
        self.assertEqual(len(runner.calls), 6)

    def test_unexpected_per_track_exception_is_isolated(self) -> None:
        found = json.dumps({"status": "found", "fields": {"name": "Renamed"}})
        adapter, _ = self._adapter(
            {
                "SYNTH-TRACK-001": RuntimeError("unexpected adapter infrastructure failure"),
                "SYNTH-TRACK-002": found,
                "SYNTH-TRACK-004": found,
            }
        )
        with CanonicalRepository(self.database_path) as repository:
            report = MusicRefreshOrchestrator(repository, adapter).run_cycle()
        self.assertFalse(report.succeeded)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].canonical_id, "trk_11111111-1111-4111-8111-111111111111")
        self.assertIn("unexpected adapter infrastructure failure", report.failures[0].error)
        self.assertEqual(report.counts()["updated"], 2)
        # The other tracks still refreshed: the failure did not abort the cycle.
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
        renamed = [t for t in loaded["tracks"] if t["name"] == "Renamed"]
        self.assertEqual(len(renamed), 2)

    def test_report_timestamps_follow_injected_clock(self) -> None:
        clock = FakeClock(datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc))
        runner = FakeRunner({"SYNTH-TRACK-001": '{"status":"confirmed_not_found"}'})
        with CanonicalRepository(self.database_path) as repository:
            orchestrator = MusicRefreshOrchestrator(
                repository, AppleMusicSourceAdapter(runner), clock=clock
            )
            clock.advance(5.0)
            report = orchestrator.run_cycle()
        self.assertEqual(report.started_at, datetime(2026, 8, 16, 12, 0, 5, tzinfo=timezone.utc))
        self.assertEqual(report.finished_at, report.started_at)

    def test_empty_store_cycle_reports_zeroes(self) -> None:
        empty_path = Path(self.temporary_directory.name) / "empty.sqlite3"
        with CanonicalRepository(empty_path) as repository:
            repository.save_model(
                {
                    "tracks": [],
                    "artists": [],
                    "albums": [],
                    "playlists": [],
                    "playlist_memberships": [],
                }
            )
        with CanonicalRepository(empty_path) as repository:
            report = MusicRefreshOrchestrator(
                repository, AppleMusicSourceAdapter(FakeRunner({}))
            ).run_cycle()
        self.assertEqual(report.bound_track_count, 0)
        self.assertEqual(report.skipped_no_binding, 0)
        self.assertEqual(report.counts()["failed"], 0)
        self.assertTrue(report.succeeded)

    def test_unbound_tracks_are_skipped_and_bound_ones_are_queried(self) -> None:
        not_found = '{"status":"confirmed_not_found"}'
        adapter, runner = self._adapter(
            {
                "SYNTH-TRACK-001": not_found,
                "SYNTH-TRACK-002": not_found,
                "SYNTH-TRACK-004": not_found,
            }
        )
        with CanonicalRepository(self.database_path) as repository:
            report = MusicRefreshOrchestrator(repository, adapter).run_cycle()
        # Fixture track index 2 has no persistent id: skipped without a source call;
        # the other three are queried exactly once each.
        self.assertEqual(report.skipped_no_binding, 1)
        self.assertEqual(sorted(runner.calls), [
            "SYNTH-TRACK-001", "SYNTH-TRACK-002", "SYNTH-TRACK-004",
        ])
        self.assertEqual(report.counts()["source_not_found"], 3)

    def test_merge_failed_rows_count_as_merge_failed(self) -> None:
        # A found record that maps to an invalid observation triggers MERGE_FAILED
        # through the production mapping/merge path.
        bad = json.dumps({"status": "found", "fields": {"name": ""}})
        adapter, _ = self._adapter(
            {"SYNTH-TRACK-001": bad, "SYNTH-TRACK-002": bad, "SYNTH-TRACK-004": bad}
        )
        with CanonicalRepository(self.database_path) as repository:
            report = MusicRefreshOrchestrator(repository, adapter).run_cycle()
        self.assertEqual(report.counts()["merge_failed"], 3)
        self.assertTrue(report.succeeded)  # typed domain outcomes are not infra failures


class MusicRefreshReportTest(unittest.TestCase):
    def test_counts_cover_all_statuses(self) -> None:
        report = MusicRefreshReport(
            started_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            finished_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            bound_track_count=0,
            skipped_no_binding=0,
            results=(),
            failures=(),
        )
        self.assertEqual(
            set(report.counts()),
            {
                "updated", "unchanged", "no_source_binding", "canonical_not_found",
                "source_not_found", "source_lookup_failed", "merge_failed", "failed",
            },
        )
        self.assertTrue(report.succeeded)


class RefreshCliTest(unittest.TestCase):
    def test_parser_accepts_refresh_subcommand(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(["refresh", "--db", "store.db"])
        self.assertEqual(args.command, "refresh")
        self.assertEqual(args.db, Path("store.db"))
        self.assertEqual(args.music_command_timeout, 10.0)

    def test_refresh_command_reports_zeroes_on_empty_store(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "empty.sqlite3"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {
                        "tracks": [],
                        "artists": [],
                        "albums": [],
                        "playlists": [],
                        "playlist_memberships": [],
                    }
                )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.stdout", stdout), redirect_stderr(stderr):
                exit_code = main(["refresh", "--db", str(database_path)])
        self.assertEqual(exit_code, 0)
        self.assertIn("refresh complete: bound=0", stdout.getvalue())
        self.assertIn("failed=0", stdout.getvalue())


class CooperativeRefreshTest(unittest.TestCase):
    """P15-S2 R8: iter_steps yields one bounded per-track step and, drained to
    completion, produces a report field-identical to the one-shot run_cycle."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def _reset_store(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def _adapter(self, outputs: dict[str, str | Exception]) -> AppleMusicSourceAdapter:
        runner = FakeRunner(outputs)
        return AppleMusicSourceAdapter(runner)

    def _drain(self, orchestrator: MusicRefreshOrchestrator) -> tuple[MusicRefreshReport, list[dict]]:
        steps = orchestrator.iter_steps()
        yields: list[dict] = []
        report: MusicRefreshReport | None = None
        while True:
            try:
                yields.append(next(steps))
            except StopIteration as done:
                report = done.value
                break
        assert report is not None
        return report, yields

    def test_iter_steps_matches_run_cycle_field_for_field(self) -> None:
        found = json.dumps({"status": "found", "fields": {"name": "Renamed"}})
        outputs = {
            "SYNTH-TRACK-001": found,
            "SYNTH-TRACK-002": found,
            "SYNTH-TRACK-004": '{"status":"confirmed_not_found"}',
        }
        clock = FakeClock(datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc))
        with CanonicalRepository(self.database_path) as repository:
            one_shot = MusicRefreshOrchestrator(repository, self._adapter(outputs), clock=clock).run_cycle()
        # Reset to the pristine fixture: the stepped run must start from the
        # same store state for a field-for-field comparison.
        self._reset_store()
        with CanonicalRepository(self.database_path) as repository:
            stepped, yields = self._drain(
                MusicRefreshOrchestrator(repository, self._adapter(outputs), clock=clock)
            )
        self.assertEqual(len(yields), 3)  # one step per bound track
        self.assertEqual(
            [y["status"] for y in yields],
            ["updated", "updated", "source_not_found"],
        )
        self.assertEqual(stepped.bound_track_count, one_shot.bound_track_count)
        self.assertEqual(stepped.skipped_no_binding, one_shot.skipped_no_binding)
        self.assertEqual(stepped.counts(), one_shot.counts())
        self.assertEqual(stepped.succeeded, one_shot.succeeded)
        self.assertEqual(
            [(r.canonical_id, r.status, r.changed_fields, r.error) for r in stepped.results],
            [(r.canonical_id, r.status, r.changed_fields, r.error) for r in one_shot.results],
        )
        self.assertEqual(
            [(f.canonical_id, f.error) for f in stepped.failures],
            [(f.canonical_id, f.error) for f in one_shot.failures],
        )
        self.assertEqual(stepped.started_at, one_shot.started_at)
        self.assertEqual(stepped.finished_at, one_shot.finished_at)

    def test_iter_steps_isolates_per_track_failures(self) -> None:
        found = json.dumps({"status": "found", "fields": {"name": "Renamed"}})
        outputs: dict[str, str | Exception] = {
            "SYNTH-TRACK-001": found,
            "SYNTH-TRACK-002": RuntimeError("osascript blew up"),
            "SYNTH-TRACK-004": found,
        }
        with CanonicalRepository(self.database_path) as repository:
            report, yields = self._drain(
                MusicRefreshOrchestrator(repository, self._adapter(outputs))
            )
        counts = report.counts()
        self.assertEqual(counts["updated"], 2)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual([y["status"] for y in yields], [
            "updated", "failed", "updated",
        ])
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].canonical_id, "trk_22222222-2222-4222-8222-222222222222")
        self.assertFalse(report.succeeded)


if __name__ == "__main__":
    unittest.main()
