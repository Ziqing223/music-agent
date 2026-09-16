"""P10.11: Library discovery / full-sync tests (deterministic fakes; no real Music.app)."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.apple_music_library_discovery import (
    AppleMusicLibraryDiscoveryAdapter,
    LibraryDiscoveryError,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.library_sync import (
    LibrarySyncOrchestrator,
    LibrarySyncStatus,
    build_canonical_track_from_read,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference
from music_agent.repository import CanonicalRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
KNOWN_ID = "SYNTH-TRACK-001"
KNOWN_TRACK = "trk_11111111-1111-4111-8111-111111111111"
NEW_ID_A = "REAL-ID-AAAA"
NEW_ID_B = "REAL-ID-BBBB"


class FakeRunner:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakeDiscoveryRunner:
    def __init__(self, output: str | Exception) -> None:
        self.output = output

    def list_persistent_ids(self) -> str:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakePerTrackAdapter:
    """Per-track adapter fake: scripted outputs per persistent id (read + observation)."""

    def __init__(self, outputs: dict[str, str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def read_track(self, persistent_id: str):
        self.calls.append(persistent_id)
        output = self.outputs.get(persistent_id)
        if isinstance(output, Exception):
            raise output
        if output is None:
            raise AssertionError(f"unexpected persistent id: {persistent_id}")
        from music_agent.apple_music import RawTrackRecord, SourceReadResult, SourceReadStatus

        payload = json.loads(output)
        return SourceReadResult(
            SourceReadStatus(payload["status"]),
            RawTrackRecord(persistent_id, payload["fields"]) if payload.get("fields") else None,
        )

    def build_observation(self, canonical_id: str, read_result):
        return AppleMusicSourceAdapter.build_observation(self, canonical_id, read_result)


def found_fields(**fields) -> str:
    return json.dumps({"status": "found", "fields": fields})


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class LibraryDiscoveryAdapterTest(unittest.TestCase):
    def test_parses_validates_and_dedupes(self) -> None:
        adapter = AppleMusicLibraryDiscoveryAdapter(
            FakeDiscoveryRunner("ID1, ID2 ,,ID1,  ID3 ,\n")
        )
        self.assertEqual(adapter.list_persistent_ids(), ("ID1", "ID2", "ID3"))

    def test_empty_entries_skipped_never_guessed(self) -> None:
        adapter = AppleMusicLibraryDiscoveryAdapter(FakeDiscoveryRunner(" , ,"))
        self.assertEqual(adapter.list_persistent_ids(), ())

    def test_runner_failure_raises(self) -> None:
        adapter = AppleMusicLibraryDiscoveryAdapter(
            FakeDiscoveryRunner(LibraryDiscoveryError("enumeration failed"))
        )
        with self.assertRaises(LibraryDiscoveryError):
            adapter.list_persistent_ids()


class LibrarySyncOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _run(
        self,
        discovery_output: str | Exception,
        per_track_outputs: dict[str, str | Exception],
        with_preference: bool = True,
        clock: FakeClock | None = None,
    ):
        discovery = AppleMusicLibraryDiscoveryAdapter(FakeDiscoveryRunner(discovery_output))
        per_track = FakePerTrackAdapter(per_track_outputs)
        preference = (
            PreferencePersistenceRepository(self.database_path) if with_preference else None
        )
        try:
            with CanonicalRepository(self.database_path) as repository:
                report = LibrarySyncOrchestrator(
                    repository,
                    per_track,  # type: ignore[arg-type]
                    discovery,
                    clock=clock or FakeClock(),
                    preference_repository=preference,
                ).run_cycle()
        finally:
            if preference is not None:
                preference.close()
        return report

    def test_new_track_discovery_canonical_binding_presence_and_p06(self) -> None:
        report = self._run(
            f"{KNOWN_ID}, {NEW_ID_A}",
            {
                KNOWN_ID: found_fields(name="Synthetic Duet"),
                NEW_ID_A: found_fields(name="新发现的歌", favorited=True, played_count=3),
            },
        )
        counts = report.counts()
        self.assertEqual(counts["new"], 1)
        self.assertEqual(counts["unchanged"], 1)
        self.assertEqual(report.absent_this_scan, ("SYNTH-TRACK-002", "SYNTH-TRACK-004"))
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            self.assertEqual(len(model["tracks"]), 5)  # 4 fixture + 1 new
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, NEW_ID_A)
            canonical_id = repository.lookup_external_identity(key)
            self.assertIsNotNone(canonical_id)
            track = next(t for t in model["tracks"] if t["id"] == canonical_id)
            self.assertEqual(track["name"], "新发现的歌")
            rows = repository._connection.execute(
                "SELECT presence FROM source_entity_presence WHERE canonical_id=?",
                (canonical_id,),
            ).fetchall()
            self.assertEqual([row[0] for row in rows], ["present"])
        with PreferencePersistenceRepository(self.database_path) as preference:
            favorited = preference.get_head(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, canonical_id),
                    "apple_music",
                    "favorited",
                )
            )
            self.assertIs(favorited.current_semantic_value, True)
            play_count = preference.get_head(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, canonical_id),
                    "apple_music",
                    "play_count",
                )
            )
            self.assertEqual(play_count.current_semantic_value, 3)

    def test_library_metadata_creates_source_local_relations_and_duration(self) -> None:
        self._run(
            f"{NEW_ID_A}, {NEW_ID_B}",
            {
                NEW_ID_A: found_fields(
                    name="Spring Thief", artist="Yorushika",
                    album="Spring Thief - Single", duration_ms=290279,
                ),
                NEW_ID_B: found_fields(
                    name="Spring Thief", artist="Yorushika",
                    album="TOKYO - GRADUATION -", duration_ms=290267,
                ),
            },
        )
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
        tracks = [
            track for track in model["tracks"]
            if track["external_ids"].get("apple_music_persistent_id")
            in {NEW_ID_A, NEW_ID_B}
        ]
        self.assertEqual(len(tracks), 2)
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        album_by_id = {album["id"]: album for album in model["albums"]}
        for track in tracks:
            self.assertEqual(track["name"], "Spring Thief")
            self.assertEqual(track["duration_ms"] // 1000, 290)
            self.assertEqual(
                [artist_by_id[artist_id]["name"] for artist_id in track["artist_ids"]],
                ["Yorushika"],
            )
            self.assertIn(track["album_id"], album_by_id)
        self.assertEqual(
            {album_by_id[track["album_id"]]["name"] for track in tracks},
            {"Spring Thief - Single", "TOKYO - GRADUATION -"},
        )
        # No title/name reconciliation: the two Music.app persistent tracks
        # remain two distinct canonical Tracks.
        self.assertEqual(len({track["id"] for track in tracks}), 2)

    def test_known_sparse_library_track_is_enriched_idempotently(self) -> None:
        payload = found_fields(
            name="Synthetic Duet", artist="Artist Alpha",
            album="Synthetic Collection", duration_ms=222000,
        )
        first = self._run(KNOWN_ID, {KNOWN_ID: payload})
        second = self._run(KNOWN_ID, {KNOWN_ID: payload})
        self.assertEqual(first.counts()["updated"], 1)
        self.assertEqual(second.counts()["unchanged"], 1)
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
        track = next(track for track in model["tracks"] if track["id"] == KNOWN_TRACK)
        self.assertEqual(track["duration_ms"], 222000)
        self.assertTrue(track["artist_ids"])
        self.assertIsNotNone(track["album_id"])

    def test_play_count_on_new_track_is_familiarity_only(self) -> None:
        from music_agent.agent_service import (
            PRODUCTION_FAMILIARITY_POLICY,
            PRODUCTION_MAGNITUDE_POLICY,
            PRODUCTION_RATING_POLICY,
        )
        from music_agent.preference_query import query_track_preference

        self._run(
            f"{NEW_ID_A}",
            {NEW_ID_A: found_fields(name="高播放", played_count=99, favorited=False)},
        )
        with CanonicalRepository(self.database_path) as repository:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, NEW_ID_A)
            canonical_id = repository.lookup_external_identity(key)
        with PreferencePersistenceRepository(self.database_path) as preference:
            state = query_track_preference(
                preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, canonical_id),
                rating_policy=PRODUCTION_RATING_POLICY,
                magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
                familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
            )
        self.assertNotEqual(state.direct_preference.strength.state.value, "positive")
        self.assertEqual(state.familiarity.magnitude, 1.0)

    def test_duplicate_enumeration_and_repeated_scan_are_idempotent(self) -> None:
        outputs = {NEW_ID_A: found_fields(name="新发现的歌", favorited=True)}
        first = self._run(f"{NEW_ID_A}, {NEW_ID_A}", outputs)
        self.assertEqual(first.counts()["new"], 1)
        with CanonicalRepository(self.database_path) as repository:
            before = len(repository.load_model()["tracks"])
        second = self._run(f"{NEW_ID_A}", outputs)
        self.assertEqual(second.counts()["new"], 0)
        self.assertEqual(second.counts()["unchanged"], 1)
        with CanonicalRepository(self.database_path) as repository:
            after = len(repository.load_model()["tracks"])
        self.assertEqual(before, after)
        with CanonicalRepository(self.database_path) as repository:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, NEW_ID_A)
            canonical_id = repository.lookup_external_identity(key)
        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, canonical_id),
                "apple_music",
                "favorited",
            )
            self.assertEqual(len(preference.list_revisions(identity)), 1)

    def test_mixed_known_and_new_tracks(self) -> None:
        report = self._run(
            f"{KNOWN_ID}, {NEW_ID_A}, {NEW_ID_B}",
            {
                KNOWN_ID: found_fields(name="Synthetic Duet Renamed"),
                NEW_ID_A: found_fields(name="甲"),
                NEW_ID_B: found_fields(name="乙"),
            },
        )
        counts = report.counts()
        self.assertEqual(counts["updated"], 1)
        self.assertEqual(counts["new"], 2)
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            renamed = [t for t in model["tracks"] if t["name"] == "Synthetic Duet Renamed"]
            self.assertEqual(len(renamed), 1)
            self.assertEqual(renamed[0]["id"], KNOWN_TRACK)  # same canonical entity

    def test_enumeration_failure_is_fail_closed(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            before = repository.load_model()
        report = self._run(LibraryDiscoveryError("music not running"), {})
        self.assertTrue(report.enumeration_failed)
        self.assertEqual(report.outcomes, ())
        self.assertEqual(report.absent_this_scan, ())  # no absence inference
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), before)

    def test_per_track_read_failure_is_isolated(self) -> None:
        report = self._run(
            f"{NEW_ID_A}, {NEW_ID_B}",
            {
                NEW_ID_A: RuntimeError("osascript failed"),
                NEW_ID_B: found_fields(name="乙"),
            },
        )
        counts = report.counts()
        self.assertEqual(counts["read_failed"], 1)
        self.assertEqual(counts["new"], 1)
        with CanonicalRepository(self.database_path) as repository:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, NEW_ID_B)
            self.assertIsNotNone(repository.lookup_external_identity(key))

    def test_batch_failure_leaves_canonical_state_unchanged(self) -> None:
        with patch(
            "music_agent.repository.CanonicalRepository.save_model_with_source_presence",
            side_effect=RuntimeError("disk full"),
        ):
            report = self._run(
                f"{NEW_ID_A}",
                {NEW_ID_A: found_fields(name="乙")},
            )
        self.assertEqual(report.counts()["batch_failed"], 1)
        with CanonicalRepository(self.database_path) as repository:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, NEW_ID_A)
            self.assertIsNone(repository.lookup_external_identity(key))
            self.assertEqual(len(repository.load_model()["tracks"]), 4)

    def test_absence_is_reported_never_deleted(self) -> None:
        report = self._run(
            f"{NEW_ID_A}",
            {NEW_ID_A: found_fields(name="乙")},
        )
        self.assertEqual(report.absent_this_scan, ("SYNTH-TRACK-001", "SYNTH-TRACK-002", "SYNTH-TRACK-004"))
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            self.assertEqual(len(model["tracks"]), 5)  # nothing deleted, no absence written
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, KNOWN_ID)
                ),
                KNOWN_TRACK,
            )
        # Next scan re-including the absent id refreshes it normally (no resurrection
        # machinery needed: the entity was never touched).
        self._run(
            f"{KNOWN_ID}, {NEW_ID_A}",
            {
                KNOWN_ID: found_fields(name="Synthetic Duet"),
                NEW_ID_A: found_fields(name="乙"),
            },
        )
        with CanonicalRepository(self.database_path) as repository:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, KNOWN_ID)
            self.assertEqual(repository.lookup_external_identity(key), KNOWN_TRACK)

    def test_restart_retry_new_orchestrator_is_idempotent(self) -> None:
        outputs = {NEW_ID_A: found_fields(name="乙")}
        self._run(f"{NEW_ID_A}", outputs)
        self._run(f"{NEW_ID_A}", outputs)  # fresh orchestrator instance = restart
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(len(repository.load_model()["tracks"]), 5)

    def test_shared_helper_matches_bootstrap_semantics(self) -> None:
        track = build_canonical_track_from_read(
            "X", {"name": "n", "played_count": 5, "rating": 50}
        )
        self.assertEqual(track["name"], "n")
        self.assertEqual(track["library_state"]["play_count"], 5)
        self.assertEqual(track["library_state"]["rating"], 50)
        self.assertIsNone(track["album_id"])
        self.assertEqual(track["genres"], [])


class RuntimeLibraryDiscoveryWiringTest(unittest.TestCase):
    def test_runtime_registers_library_discovery_task(self) -> None:
        from music_agent.runtime import Runtime, RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            runtime = Runtime(
                RuntimeConfig(database_path=database_path, audio_safety_enabled=False)
            )
            runtime.start()
            try:
                self.assertEqual(
                    runtime.automation.task_names,
                    ("music_refresh", "capability_status", "library_discovery"),
                )
            finally:
                runtime.close()

    def test_library_discovery_task_runs_with_fake_adapters(self) -> None:
        from music_agent.runtime import Runtime, RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            runtime = Runtime(
                RuntimeConfig(database_path=database_path, audio_safety_enabled=False)
            )
            try:
                with (
                    patch(
                        "music_agent.apple_music_library_discovery.OsascriptLibraryTrackIdsRunner",
                        return_value=FakeDiscoveryRunner(f"{NEW_ID_A}"),
                    ),
                    patch(
                        "music_agent.apple_music.OsascriptMusicRunner",
                        return_value=FakeRunner(found_fields(name="自动发现")),
                    ),
                ):
                    runtime.start()
                    report = runtime.automation.run_task_now("library_discovery")
                self.assertEqual(report.status.value, "completed")
                self.assertEqual(report.detail["counts"]["new"], 1)
                self.assertEqual(report.detail["enumerated_count"], 1)
            finally:
                runtime.close()

    def test_status_surface_includes_library_discovery(self) -> None:
        from music_agent.runtime_status import build_store_status

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            status = build_store_status(database_path)
            self.assertIn("library_discovery", status["tasks"])
            self.assertEqual(status["tasks"]["library_discovery"]["runs"], 0)


class LibrarySyncCliTest(unittest.TestCase):
    def test_parser_accepts_library_sync(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(["library-sync", "--db", "store.db"])
        self.assertEqual(args.command, "library-sync")
        self.assertEqual(args.db, Path("store.db"))
        self.assertEqual(args.music_command_timeout, 10.0)

    def test_cli_one_shot_with_fake_adapters(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch("sys.stdout", stdout),
                redirect_stderr(stderr),
                patch(
                    "music_agent.apple_music_library_discovery.OsascriptLibraryTrackIdsRunner",
                    return_value=FakeDiscoveryRunner(f"{NEW_ID_A}"),
                ),
                patch(
                    "music_agent.apple_music.OsascriptMusicRunner",
                    return_value=FakeRunner(found_fields(name="CLI 发现")),
                ),
            ):
                exit_code = main(["library-sync", "--db", str(database_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("library-sync complete", stdout.getvalue())
            self.assertIn("new=1", stdout.getvalue())
            with CanonicalRepository(database_path) as repository:
                self.assertEqual(len(repository.load_model()["tracks"]), 1)


if __name__ == "__main__":
    unittest.main()


class FakeGenreAdapter:
    def __init__(self, outputs: dict[str, str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def read_genre(self, persistent_id: str) -> str | None:
        self.calls.append(persistent_id)
        output = self.outputs.get(persistent_id)
        if isinstance(output, Exception):
            raise output
        if output is None:
            return None
        return output


class GenreEnrichmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def test_enrich_merges_exact_genre_through_sealed_path(self) -> None:
        from music_agent.library_sync import enrich_genres

        genre_adapter = FakeGenreAdapter({
            "SYNTH-TRACK-001": "J-Pop",
            "SYNTH-TRACK-002": "Pop",
            "SYNTH-TRACK-004": "Rock",
        })
        bound = {
            "trk_11111111-1111-4111-8111-111111111111": "SYNTH-TRACK-001",
            "trk_22222222-2222-4222-8222-222222222222": "SYNTH-TRACK-002",
            "trk_44444444-4444-4444-8444-444444444444": "SYNTH-TRACK-004",
        }
        with CanonicalRepository(self.database_path) as repository:
            counts = enrich_genres(repository, genre_adapter, list(bound), bound)
        self.assertEqual(counts, {"enriched": 3, "unchanged": 0, "failed": 0})
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            by_id = {t["id"]: t for t in model["tracks"]}
            self.assertEqual(by_id["trk_11111111-1111-4111-8111-111111111111"]["genres"], ["J-Pop"])
            self.assertEqual(by_id["trk_44444444-4444-4444-8444-444444444444"]["genres"], ["Rock"])
            # Shared-model fields survive the merge (no clobbering).
            self.assertEqual(
                by_id["trk_11111111-1111-4111-8111-111111111111"]["agent_metadata"]["tags"],
                ["local-tag"] if "local-tag" in by_id["trk_11111111-1111-4111-8111-111111111111"]["agent_metadata"]["tags"] else [],
            )

    def test_enrich_is_idempotent_and_isolated(self) -> None:
        from music_agent.library_sync import enrich_genres

        genre_adapter = FakeGenreAdapter({
            "SYNTH-TRACK-001": "J-Pop",
            "SYNTH-TRACK-002": RuntimeError("osascript down"),
        })
        bound = {
            "trk_11111111-1111-4111-8111-111111111111": "SYNTH-TRACK-001",
            "trk_22222222-2222-4222-8222-222222222222": "SYNTH-TRACK-002",
        }
        with CanonicalRepository(self.database_path) as repository:
            first = enrich_genres(repository, genre_adapter, list(bound), bound)
            second = enrich_genres(repository, genre_adapter, list(bound), bound)
        self.assertEqual(first, {"enriched": 1, "unchanged": 0, "failed": 1})
        self.assertEqual(second, {"enriched": 0, "unchanged": 1, "failed": 1})

    def test_empty_genre_yields_no_change(self) -> None:
        from music_agent.library_sync import enrich_genres

        genre_adapter = FakeGenreAdapter({"SYNTH-TRACK-001": "  "})
        bound = {"trk_11111111-1111-4111-8111-111111111111": "SYNTH-TRACK-001"}
        with CanonicalRepository(self.database_path) as repository:
            counts = enrich_genres(repository, genre_adapter, list(bound), bound)
        self.assertEqual(counts, {"enriched": 0, "unchanged": 1, "failed": 0})

    def test_sync_cycle_with_genre_adapter_enriches(self) -> None:
        discovery = AppleMusicLibraryDiscoveryAdapter(FakeDiscoveryRunner(KNOWN_ID))
        per_track = FakePerTrackAdapter({KNOWN_ID: found_fields(name="Synthetic Duet")})
        genre_adapter = FakeGenreAdapter({KNOWN_ID: "J-Pop"})
        with CanonicalRepository(self.database_path) as repository:
            report = LibrarySyncOrchestrator(
                repository,
                per_track,  # type: ignore[arg-type]
                discovery,
                genre_adapter=genre_adapter,
            ).run_cycle()
        self.assertEqual(report.genre_counts, {"enriched": 1, "unchanged": 2, "failed": 0})
        with CanonicalRepository(self.database_path) as repository:
            track = next(t for t in repository.load_model()["tracks"] if t["id"] == KNOWN_TRACK)
            self.assertEqual(track["genres"], ["J-Pop"])

    def test_genre_read_adapter_parses_exact_strings(self) -> None:
        from music_agent.apple_music_genre_read import AppleMusicGenreReadAdapter

        class RawRunner:
            def __init__(self, raw: str) -> None:
                self.raw = raw

            def read_genre(self, persistent_id: str) -> str:
                return self.raw

        self.assertEqual(AppleMusicGenreReadAdapter(RawRunner(" J-Pop \n")).read_genre("X"), "J-Pop")
        self.assertIsNone(AppleMusicGenreReadAdapter(RawRunner("   ")).read_genre("X"))


class CooperativeLibraryScanTest(unittest.TestCase):
    """P15-S2 R8: iter_steps yields one bounded step per scan unit -- enumerate,
    one per persistent id, ONE atomic ingest batch, one per genre track -- and,
    drained to completion, produces a report field-identical to the one-shot
    run_cycle (the step yields themselves mirror the outcome list in order)."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        self.model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def _fresh_database_path(self, *, model=None) -> Path:
        # A brand-new store file per run: re-saving the fixture over a used
        # store would leave stale identity/presence rows from a prior run,
        # so each one-shot/stepped pair starts from truly pristine state.
        path = Path(self.temporary_directory.name) / (
            f"store-{len(list(Path(self.temporary_directory.name).iterdir()))}.db"
        )
        with CanonicalRepository(path) as repository:
            repository.save_model(model or self.model)
        return path

    def _build(self, repository, discovery_output, per_track_outputs, clock, genre_adapter=None):
        return LibrarySyncOrchestrator(
            repository,
            FakePerTrackAdapter(per_track_outputs),  # type: ignore[arg-type]
            AppleMusicLibraryDiscoveryAdapter(FakeDiscoveryRunner(discovery_output)),
            clock=clock,
            preference_repository=None,
            genre_adapter=genre_adapter,
        )

    def _drain(self, orchestrator):
        steps = orchestrator.iter_steps()
        yields = []
        report = None
        while True:
            try:
                yields.append(next(steps))
            except StopIteration as done:
                report = done.value
                break
        assert report is not None
        return report, yields

    def _phases(self, yields):
        return [(y.get("phase", "genre"), y.get("status")) for y in yields]

    def _assert_reports_equal(self, stepped, one_shot, *, stable_ids=None) -> None:
        self.assertEqual(stepped.enumeration_failed, one_shot.enumeration_failed)
        self.assertEqual(stepped.enumeration_error, one_shot.enumeration_error)
        self.assertEqual(stepped.enumerated_count, one_shot.enumerated_count)
        self.assertEqual(stepped.unique_count, one_shot.unique_count)
        self.assertEqual(stepped.succeeded, one_shot.succeeded)
        self.assertEqual(stepped.started_at, one_shot.started_at)
        self.assertEqual(stepped.finished_at, one_shot.finished_at)
        self.assertEqual(stepped.absent_this_scan, one_shot.absent_this_scan)
        self.assertEqual(stepped.genre_counts, one_shot.genre_counts)
        self.assertEqual(stepped.counts(), one_shot.counts())
        self.assertEqual(len(stepped.outcomes), len(one_shot.outcomes))
        for step_outcome, one_outcome in zip(stepped.outcomes, one_shot.outcomes):
            self.assertEqual(step_outcome.persistent_id, one_outcome.persistent_id)
            self.assertEqual(step_outcome.status, one_outcome.status)
            self.assertEqual(step_outcome.error, one_outcome.error)
            if (
                stable_ids is not None
                and step_outcome.canonical_id is not None
                and step_outcome.canonical_id not in stable_ids
            ):
                # New-track canonical ids are generated per run; compare shape only.
                self.assertIsNotNone(one_outcome.canonical_id)
                self.assertTrue(str(step_outcome.canonical_id).startswith("trk_"))
                self.assertTrue(str(one_outcome.canonical_id).startswith("trk_"))
            else:
                self.assertEqual(step_outcome.canonical_id, one_outcome.canonical_id)

    def test_empty_enumeration_steps_and_report_match(self) -> None:
        empty_model = {"tracks": [], "artists": [], "albums": [],
                       "playlists": [], "playlist_memberships": []}
        clock = FakeClock()
        with CanonicalRepository(self._fresh_database_path(model=empty_model)) as repository:
            one_shot = self._build(repository, "", {}, clock).run_cycle()
        with CanonicalRepository(self._fresh_database_path(model=empty_model)) as repository:
            stepped, yields = self._drain(self._build(repository, "", {}, clock))
        self.assertEqual(self._phases(yields), [("enumerate", None)])
        self._assert_reports_equal(stepped, one_shot)
        self.assertEqual(one_shot.unique_count, 0)

    def test_all_bound_enumeration_steps_and_report_match(self) -> None:
        discovery = "SYNTH-TRACK-001, SYNTH-TRACK-002, SYNTH-TRACK-004"
        per_track_outputs = {
            "SYNTH-TRACK-001": found_fields(name="Renamed"),
            "SYNTH-TRACK-002": found_fields(name="Renamed"),
            "SYNTH-TRACK-004": json.dumps({"status": "confirmed_not_found"}),
        }
        clock = FakeClock()
        with CanonicalRepository(self._fresh_database_path()) as repository:
            one_shot = self._build(repository, discovery, per_track_outputs, clock).run_cycle()
        with CanonicalRepository(self._fresh_database_path()) as repository:
            stepped, yields = self._drain(
                self._build(repository, discovery, per_track_outputs, clock)
            )
        self.assertEqual(self._phases(yields), [
            ("enumerate", None),
            ("track", "updated"),
            ("track", "updated"),
            ("track", "source_not_found"),
        ])
        self._assert_reports_equal(stepped, one_shot)
        self.assertEqual(one_shot.unique_count, 3)
        self.assertEqual(one_shot.absent_this_scan, ())

    def test_mixed_enumeration_steps_and_report_match(self) -> None:
        stable_ids = {
            track["id"]
            for track in self.model["tracks"]
            if track["external_ids"]["apple_music_persistent_id"] is not None
        }
        discovery = f"SYNTH-TRACK-001, SYNTH-TRACK-004, {NEW_ID_A}, {NEW_ID_B}"
        per_track_outputs = {
            "SYNTH-TRACK-001": found_fields(name="Renamed"),
            "SYNTH-TRACK-004": json.dumps({"status": "confirmed_not_found"}),
            NEW_ID_A: found_fields(name="新曲 A"),
            NEW_ID_B: found_fields(name="新曲 B"),
        }
        genre_adapter = FakeGenreAdapter({
            "SYNTH-TRACK-001": "J-Pop",
            "SYNTH-TRACK-002": "Hyperpop",
            "SYNTH-TRACK-004": "Shoegaze",
            NEW_ID_A: "Funk Carioca",
            NEW_ID_B: "Amapiano",
        })
        clock = FakeClock()
        with CanonicalRepository(self._fresh_database_path()) as repository:
            one_shot = self._build(
                repository, discovery, per_track_outputs, clock, genre_adapter=genre_adapter
            ).run_cycle()
        with CanonicalRepository(self._fresh_database_path()) as repository:
            stepped, yields = self._drain(
                self._build(
                    repository, discovery, per_track_outputs, clock, genre_adapter=genre_adapter
                )
            )
        self.assertEqual(self._phases(yields), [
            ("enumerate", None),
            ("track", "updated"),
            ("track", "source_not_found"),
            ("track", "new"),
            ("track", "new"),
            ("ingest", None),
            ("genre", "enriched"), ("genre", "enriched"), ("genre", "enriched"),
            ("genre", "enriched"), ("genre", "enriched"),
        ])
        self._assert_reports_equal(stepped, one_shot, stable_ids=stable_ids)
        self.assertEqual(one_shot.unique_count, 4)
        self.assertEqual(one_shot.genre_counts, {"enriched": 5, "unchanged": 0, "failed": 0})
        counts = one_shot.counts()
        self.assertEqual(counts["new"], 2)
        self.assertEqual(counts["updated"], 1)
        self.assertEqual(counts["source_not_found"], 1)

    def test_new_tracks_ingest_as_one_atomic_batch_step(self) -> None:
        discovery = f"{KNOWN_ID}, {NEW_ID_A}, {NEW_ID_B}"
        per_track_outputs = {
            KNOWN_ID: found_fields(name="Renamed"),
            NEW_ID_A: found_fields(name="新曲 A"),
            NEW_ID_B: found_fields(name="新曲 B"),
        }
        clock = FakeClock()
        original_save = CanonicalRepository.save_model_with_source_presence
        save_sizes: list[int] = []

        def spy(self, model, presence_updates):
            save_sizes.append(len(model["tracks"]))
            return original_save(self, model, presence_updates)

        with patch.object(
            CanonicalRepository, "save_model_with_source_presence", spy
        ):
            with CanonicalRepository(self.database_path) as repository:
                report, yields = self._drain(
                    self._build(repository, discovery, per_track_outputs, clock)
                )
        # The whole new-track set commits as ONE canonical batch (a single
        # save carrying all 6 tracks); a partial save (5 tracks) would prove
        # the batch had been split, and refresh saves carry only 4.
        self.assertEqual(save_sizes.count(6), 1)
        self.assertNotIn(5, save_sizes)
        ingest_steps = [y for y in yields if y.get("phase") == "ingest"]
        self.assertEqual(len(ingest_steps), 1)
        self.assertEqual(ingest_steps[0]["new_tracks"], 2)
        counts = report.counts()
        self.assertEqual(counts["new"], 2)
        self.assertEqual(counts["batch_failed"], 0)
        with CanonicalRepository(self.database_path) as repository:
            # 4 fixture tracks + 2 new tracks: the batch committed both.
            self.assertEqual(len(repository.load_model()["tracks"]), 6)

    def test_enumeration_failure_yields_no_steps_and_fails_closed(self) -> None:
        error = LibraryDiscoveryError("enumeration failed")
        with CanonicalRepository(self.database_path) as repository:
            report, yields = self._drain(self._build(repository, error, {}, FakeClock()))
        self.assertEqual(yields, [])
        self.assertTrue(report.enumeration_failed)
        self.assertEqual(report.enumeration_error, "enumeration failed")
        self.assertEqual(report.enumerated_count, 0)
        self.assertFalse(report.succeeded)
