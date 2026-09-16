"""P10.10: Real-track bootstrap tests (fake read adapter; no Music.app, no real store)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.repository import CanonicalRepository
from tools.bootstrap_real_tracks import bootstrap, readback

REAL_ID_A = "0A50922A7206CB39"
REAL_ID_B = "1111111111111111"


class FakeReadAdapter:
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
        from music_agent.apple_music import AppleMusicSourceAdapter

        return AppleMusicSourceAdapter.build_observation(self, canonical_id, read_result)


class FakeReadAdapterFactory:
    instance: FakeReadAdapter | None = None

    def __call__(self, *args, **kwargs):
        assert FakeReadAdapterFactory.instance is not None
        return FakeReadAdapterFactory.instance


class _EmptyGenreAdapter:
    def read_genre(self, persistent_id: str) -> None:
        return None


def found_fields(**fields) -> str:
    return json.dumps({"status": "found", "fields": fields})


class BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def test_bootstrap_reads_binds_and_persists_real_tracks(self) -> None:
        adapter = FakeReadAdapter({
            REAL_ID_A: found_fields(
                name="起风了 (旧版)", played_count=46, favorited=False, rating=0,
            ),
            REAL_ID_B: found_fields(name="Another Real Song"),
        })
        FakeReadAdapterFactory.instance = adapter
        with (
            patch("tools.bootstrap_real_tracks.AppleMusicSourceAdapter", FakeReadAdapterFactory()),
            patch(
                "tools.bootstrap_real_tracks.AppleMusicGenreReadAdapter",
                return_value=_EmptyGenreAdapter(),
            ),
        ):
            result = bootstrap(self.database_path, [REAL_ID_A, REAL_ID_B])
        self.assertEqual(result, {"added": 2, "skipped": 0, "read": 2})
        self.assertEqual(sorted(adapter.calls), [REAL_ID_A, REAL_ID_B])
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            self.assertEqual(len(model["tracks"]), 2)
            track = next(t for t in model["tracks"] if t["external_ids"]["apple_music_persistent_id"] == REAL_ID_A)
            self.assertEqual(track["name"], "起风了 (旧版)")
            self.assertEqual(track["library_state"]["play_count"], 46)
            self.assertIs(track["library_state"]["favorited"], False)
            self.assertEqual(track["artist_ids"], [])
            # External identity binding established by the canonical save path.
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, REAL_ID_A)
            self.assertEqual(repository.lookup_external_identity(key), track["id"])

    def test_reimport_is_skipped_never_duplicated(self) -> None:
        adapter = FakeReadAdapter({REAL_ID_A: found_fields(name="起风了 (旧版)")})
        FakeReadAdapterFactory.instance = adapter
        with (
            patch("tools.bootstrap_real_tracks.AppleMusicSourceAdapter", FakeReadAdapterFactory()),
            patch(
                "tools.bootstrap_real_tracks.AppleMusicGenreReadAdapter",
                return_value=_EmptyGenreAdapter(),
            ),
        ):
            bootstrap(self.database_path, [REAL_ID_A])
            second = bootstrap(self.database_path, [REAL_ID_A])
        self.assertEqual(second, {"added": 0, "skipped": 1, "read": 1})
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(len(repository.load_model()["tracks"]), 1)

    def test_confirmed_not_found_fails_closed(self) -> None:
        adapter = FakeReadAdapter({REAL_ID_A: '{"status":"confirmed_not_found"}'})
        FakeReadAdapterFactory.instance = adapter
        with (
            patch("tools.bootstrap_real_tracks.AppleMusicSourceAdapter", FakeReadAdapterFactory()),
            patch(
                "tools.bootstrap_real_tracks.AppleMusicGenreReadAdapter",
                return_value=_EmptyGenreAdapter(),
            ),
        ):
            with self.assertRaises(RuntimeError):
                bootstrap(self.database_path, [REAL_ID_A])
        # Nothing was partially written: the store was never created beyond migrations.
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(len(repository.load_model()["tracks"]), 0)

    def test_readback_reports_bindings(self) -> None:
        import io
        from contextlib import redirect_stdout

        adapter = FakeReadAdapter({REAL_ID_A: found_fields(name="起风了 (旧版)")})
        FakeReadAdapterFactory.instance = adapter
        with (
            patch("tools.bootstrap_real_tracks.AppleMusicSourceAdapter", FakeReadAdapterFactory()),
            patch(
                "tools.bootstrap_real_tracks.AppleMusicGenreReadAdapter",
                return_value=_EmptyGenreAdapter(),
            ),
        ):
            bootstrap(self.database_path, [REAL_ID_A])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            readback(self.database_path)
        self.assertIn("canonical tracks: 1", stdout.getvalue())
        self.assertIn("binding_ok=True", stdout.getvalue())

    def test_missing_fields_stay_null(self) -> None:
        adapter = FakeReadAdapter({REAL_ID_A: found_fields(name="Only a Name")})
        FakeReadAdapterFactory.instance = adapter
        with (
            patch("tools.bootstrap_real_tracks.AppleMusicSourceAdapter", FakeReadAdapterFactory()),
            patch(
                "tools.bootstrap_real_tracks.AppleMusicGenreReadAdapter",
                return_value=_EmptyGenreAdapter(),
            ),
        ):
            bootstrap(self.database_path, [REAL_ID_A])
        with CanonicalRepository(self.database_path) as repository:
            track = repository.load_model()["tracks"][0]
            self.assertEqual(track["library_state"]["rating"], None)
            self.assertEqual(track["library_state"]["skip_count"], None)
            self.assertEqual(track["album_id"], None)
            self.assertEqual(track["genres"], [])


if __name__ == "__main__":
    unittest.main()
