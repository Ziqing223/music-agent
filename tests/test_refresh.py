import json
import tempfile
import unittest
from pathlib import Path

from music_agent.apple_music import AppleMusicReadError, AppleMusicSourceAdapter
from music_agent.refresh import RefreshStatus, refresh_known_track
from music_agent.repository import CanonicalRepository


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeRunner:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class KnownTrackRefreshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.track_id = self.model["tracks"][0]["id"]
        self.model["tracks"][0]["agent_metadata"]["tags"] = ["local-tag"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_production_adapter_path_refreshes_bound_track_and_persists_after_reopen(self) -> None:
        runner = FakeRunner(json.dumps({
            "status": "found",
            "fields": {"name": "Read-only Refresh", "played_count": 42, "favorited": False},
        }))
        with CanonicalRepository(self.database_path) as repository:
            result = refresh_known_track(repository, AppleMusicSourceAdapter(runner), self.track_id)
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
        track = loaded["tracks"][0]
        self.assertIs(result.status, RefreshStatus.UPDATED)
        self.assertEqual(runner.calls, ["SYNTH-TRACK-001"])
        self.assertEqual(track["id"], self.track_id)
        self.assertEqual(track["name"], "Read-only Refresh")
        self.assertEqual(track["library_state"]["play_count"], 42)
        self.assertIs(track["library_state"]["favorited"], False)
        self.assertEqual(track["agent_metadata"]["tags"], ["local-tag"])
        self.assertEqual(track["artist_ids"], self.model["tracks"][0]["artist_ids"])
        self.assertEqual(track["album_id"], self.model["tracks"][0]["album_id"])

    def test_unbound_track_and_unknown_canonical_id_do_not_call_source(self) -> None:
        runner = FakeRunner('{"status":"found","fields":{}}')
        unbound_id = self.model["tracks"][2]["id"]
        unknown_id = "trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        with CanonicalRepository(self.database_path) as repository:
            unbound = refresh_known_track(repository, AppleMusicSourceAdapter(runner), unbound_id)
            unknown = refresh_known_track(repository, AppleMusicSourceAdapter(runner), unknown_id)
        self.assertIs(unbound.status, RefreshStatus.NO_SOURCE_BINDING)
        self.assertIs(unknown.status, RefreshStatus.CANONICAL_NOT_FOUND)
        self.assertEqual(runner.calls, [])

    def test_lookup_failure_malformed_output_and_confirmed_not_found_do_not_save(self) -> None:
        cases = (
            (AppleMusicReadError("automation denied"), RefreshStatus.SOURCE_LOOKUP_FAILED),
            ("not-json", RefreshStatus.SOURCE_LOOKUP_FAILED),
            ('{"status":"confirmed_not_found"}', RefreshStatus.SOURCE_NOT_FOUND),
        )
        for output, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                with CanonicalRepository(self.database_path) as repository:
                    result = refresh_known_track(
                        repository, AppleMusicSourceAdapter(FakeRunner(output)), self.track_id
                    )
                with CanonicalRepository(self.database_path) as repository:
                    self.assertEqual(repository.load_model(), self.model)
                self.assertIs(result.status, expected_status)

    def test_invalid_mapped_value_does_not_save(self) -> None:
        runner = FakeRunner('{"status":"found","fields":{"rating":101}}')
        with CanonicalRepository(self.database_path) as repository:
            result = refresh_known_track(repository, AppleMusicSourceAdapter(runner), self.track_id)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), self.model)
        self.assertIs(result.status, RefreshStatus.MERGE_FAILED)

    def test_repeated_identical_refresh_is_idempotent(self) -> None:
        output = '{"status":"found","fields":{"name":"Stable Source State"}}'
        adapter = AppleMusicSourceAdapter(FakeRunner(output))
        with CanonicalRepository(self.database_path) as repository:
            first = refresh_known_track(repository, adapter, self.track_id)
            first_counts = repository.counts()
            second = refresh_known_track(repository, adapter, self.track_id)
            second_counts = repository.counts()
        self.assertIs(first.status, RefreshStatus.UPDATED)
        self.assertTrue(first.changed_fields)
        self.assertIs(second.status, RefreshStatus.UNCHANGED)
        self.assertEqual(second.changed_fields, ())
        self.assertEqual(second_counts, first_counts)


if __name__ == "__main__":
    unittest.main()
