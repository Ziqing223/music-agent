"""P10.14-16: Daily-ops, storage-report, and homepage library-count tests (fakes only)."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from music_agent.daily_ops import DailyRunResult, backup_store, restore_store, run_daily
from music_agent.obsidian_projection import (
    MUSIC_AGENT_PARTITION,
    ObsidianProjectionConfig,
    ObsidianProjectionStructureError,
    ObsidianProjectionWriter,
    ObsidianWriteAction,
)
from music_agent.repository import CanonicalRepository
from music_agent.storage_report import build_storage_report

HOMEPAGE_TEMPLATE = """# Music Agent

## 当前状态

- 自动同步：P10 runtime 接入中；尚未启用常驻同步

## 导航

- [[01_Library/收藏歌曲]]
"""


def empty_model() -> dict:
    return {
        "tracks": [],
        "artists": [],
        "albums": [],
        "playlists": [],
        "playlist_memberships": [],
    }


class HomepageLibraryCountTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.vault_root = Path(self.temporary_directory.name) / "Vault"
        partition = self.vault_root / MUSIC_AGENT_PARTITION
        partition.mkdir(parents=True)
        self.homepage = partition / "首页.md"
        self.homepage.write_text(HOMEPAGE_TEMPLATE, encoding="utf-8")
        self.writer = ObsidianProjectionWriter(ObsidianProjectionConfig(vault_root=self.vault_root))

    def test_inserts_then_updates_one_bullet_idempotently(self) -> None:
        first = self.writer.update_homepage_library_count(130)
        self.assertEqual(first.action, ObsidianWriteAction.UPDATED)
        content = self.homepage.read_text(encoding="utf-8")
        self.assertIn("- 同步曲库：130 首", content)
        self.assertEqual(content.count("- 同步曲库："), 1)
        before = content
        second = self.writer.update_homepage_library_count(130)
        self.assertEqual(second.action, ObsidianWriteAction.NOOP_UNCHANGED)
        self.assertEqual(self.homepage.read_text(encoding="utf-8"), before)
        third = self.writer.update_homepage_library_count(131)
        self.assertEqual(third.action, ObsidianWriteAction.UPDATED)
        self.assertIn("- 同步曲库：131 首", self.homepage.read_text(encoding="utf-8"))
        self.assertEqual(self.homepage.read_text(encoding="utf-8").count("- 同步曲库："), 1)

    def test_requires_existing_status_bullet(self) -> None:
        self.homepage.write_text(
            HOMEPAGE_TEMPLATE.replace("- 自动同步：P10 runtime 接入中；尚未启用常驻同步", ""),
            encoding="utf-8",
        )
        with self.assertRaises(ObsidianProjectionStructureError):
            self.writer.update_homepage_library_count(130)

    def test_rejects_negative_count(self) -> None:
        from music_agent.obsidian_projection import ObsidianProjectionValidationError

        with self.assertRaises(ObsidianProjectionValidationError):
            self.writer.update_homepage_library_count(-1)


class StorageReportTest(unittest.TestCase):
    def test_report_counts_tables_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(empty_model())
            report = build_storage_report(database_path)
            self.assertGreater(report.size_bytes, 0)
            self.assertEqual(report.table_rows["canonical_entities"], 0)
            self.assertIn("schema_migrations", report.table_rows)
            self.assertIn("table canonical_entities", report.summary())


class DailyOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(empty_model())

    def test_run_daily_with_empty_discovery(self) -> None:
        class EmptyDiscovery:
            def list_persistent_ids(self) -> tuple[str, ...]:
                return ()

        class FakeRunner:
            def run(self, persistent_id: str) -> str:
                raise AssertionError("no per-track reads expected for an empty library")

        with (
            patch(
                "music_agent.apple_music_library_discovery.AppleMusicLibraryDiscoveryAdapter",
                return_value=EmptyDiscovery(),
            ),
            patch("music_agent.apple_music.OsascriptMusicRunner", return_value=FakeRunner()),
        ):
            result = run_daily(self.database_path)
        self.assertIsInstance(result, DailyRunResult)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.library_sync_counts["new"], 0)
        self.assertEqual(result.refresh_counts["failed"], 0)
        self.assertIn("tasks", result.status)

    def test_backup_and_restore_round_trip(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(empty_model())
        backup_path = backup_store(
            self.database_path, now=datetime(2026, 8, 16, 15, 0, 0)
        )
        self.assertTrue(backup_path.is_file())
        self.assertIn(".backup-20260816-150000", backup_path.name)
        with CanonicalRepository(backup_path) as repository:
            self.assertEqual(repository.schema_version, 19)
        # Restore refuses to overwrite without force; then restores with force.
        target = Path(self.temporary_directory.name) / "restored.db"
        restore_store(target, backup_path)
        with CanonicalRepository(target) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(repository.load_model()["tracks"], [])
        with self.assertRaises(ValueError):
            restore_store(target, backup_path)  # exists, no force
        restore_store(target, backup_path, force=True)

    def test_backup_missing_store_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            backup_store(Path(self.temporary_directory.name) / "missing.db")


class DailyCliParseTest(unittest.TestCase):
    def test_new_subcommands_parse(self) -> None:
        from music_agent.cli import build_parser

        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["storage-report", "--db", "s.db"]).command, "storage-report"
        )
        self.assertEqual(parser.parse_args(["daily", "--db", "s.db"]).command, "daily")
        args = parser.parse_args(["backup", "--db", "s.db", "--backup-dir", "/tmp/b"])
        self.assertEqual(args.backup_dir, Path("/tmp/b"))
        args = parser.parse_args(["restore", "--db", "s.db", "--from", "b.db", "--force"])
        self.assertTrue(args.force)
        args = parser.parse_args(
            ["obsidian", "update-library-count", "--vault", "/v", "--db", "s.db"]
        )
        self.assertEqual(args.obsidian_command, "update-library-count")

    def test_daily_cli_one_shot(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(empty_model())

            class EmptyDiscovery:
                def list_persistent_ids(self) -> tuple[str, ...]:
                    return ()

            class FakeRunner:
                def run(self, persistent_id: str) -> str:
                    raise AssertionError("no per-track reads")

            stdout = io.StringIO()
            with (
                patch("sys.stdout", stdout),
                redirect_stderr(io.StringIO()),
                patch(
                    "music_agent.apple_music_library_discovery.AppleMusicLibraryDiscoveryAdapter",
                    return_value=EmptyDiscovery(),
                ),
                patch("music_agent.apple_music.OsascriptMusicRunner", return_value=FakeRunner()),
            ):
                exit_code = main(["daily", "--db", str(database_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("daily complete", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
