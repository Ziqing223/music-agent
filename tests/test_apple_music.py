import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.apple_music import (
    AppleMusicMappingError,
    AppleMusicReadError,
    AppleMusicSourceAdapter,
    OsascriptMusicRunner,
    SourceReadStatus,
)
from music_agent.source_observation import ObservationState


TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"


class FakeRunner:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class OsascriptMusicRunnerTest(unittest.TestCase):
    def test_runner_uses_argv_without_shell_interpolation(self) -> None:
        external_id = 'opaque ID; $(touch /tmp/never) "quoted"'
        completed = SimpleNamespace(returncode=0, stdout='{"status":"confirmed_not_found"}\n', stderr="")
        with patch("music_agent.apple_music.subprocess.run", return_value=completed) as run:
            output = OsascriptMusicRunner(timeout_seconds=3).run(external_id)
        argv = run.call_args.args[0]
        self.assertEqual(argv[-1], external_id)
        self.assertEqual(argv[:2], ["osascript", "-e"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 3)
        self.assertEqual(output, '{"status":"confirmed_not_found"}')

    def test_nonzero_exit_and_timeout_are_lookup_errors(self) -> None:
        failed = SimpleNamespace(returncode=1, stdout="", stderr="Music unavailable")
        with patch("music_agent.apple_music.subprocess.run", return_value=failed):
            with self.assertRaises(AppleMusicReadError):
                OsascriptMusicRunner().run("PID")
        with patch(
            "music_agent.apple_music.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["osascript"], 1),
        ):
            with self.assertRaises(AppleMusicReadError):
                OsascriptMusicRunner().run("PID")


class AppleMusicSourceAdapterTest(unittest.TestCase):
    def test_found_record_maps_string_false_zero_rating_and_timezone_datetimes(self) -> None:
        output = json.dumps({
            "status": "found",
            "fields": {
                "name": "Source Track",
                "artist": "Yorushika",
                "album": "Spring Thief - Single",
                "duration_ms": 290279,
                "favorited": False,
                "disliked": False,
                "rating": 0,
                "played_count": 0,
                "date_added": "2024-01-02T03:04:05+08:00",
                "played_date": "2025-02-03T04:05:06Z",
            },
        })
        adapter = AppleMusicSourceAdapter(FakeRunner(output))
        read = adapter.read_track("Opaque-PID")
        observation = adapter.build_observation(TRACK_ID, read)
        fields = observation.fields
        self.assertIs(read.status, SourceReadStatus.FOUND)
        self.assertEqual(fields["name"].payload, "Source Track")
        self.assertEqual(fields["duration_ms"].payload, 290279)
        # Artist and album names require canonical relation materialization;
        # the adapter never writes names into ID fields by itself.
        self.assertIs(fields["artist_ids"].state, ObservationState.MISSING)
        self.assertIs(fields["album_id"].state, ObservationState.MISSING)
        self.assertIs(fields["library_state.favorited"].payload, False)
        self.assertEqual(fields["library_state.rating"].payload, 0)
        self.assertEqual(fields["library_state.play_count"].payload, 0)
        self.assertEqual(
            fields["library_state.added_to_library_at"].payload,
            "2024-01-02T03:04:05+08:00",
        )
        self.assertEqual(fields["library_state.last_played_at"].payload, "2025-02-03T04:05:06Z")

    def test_absent_and_null_are_missing_not_null(self) -> None:
        output = json.dumps({"status": "found", "fields": {"played_date": None}})
        adapter = AppleMusicSourceAdapter(FakeRunner(output))
        observation = adapter.build_observation(TRACK_ID, adapter.read_track("PID"))
        self.assertIs(
            observation.fields["library_state.last_played_at"].state,
            ObservationState.MISSING,
        )
        self.assertIs(observation.fields["library_state.play_count"].state, ObservationState.MISSING)

    def test_unverified_properties_and_relation_metadata_do_not_create_canonical_relations(self) -> None:
        output = json.dumps({
            "status": "found",
            "fields": {
                "artist_name": "Same Name Is Not Identity",
                "album_name": "Display Album",
                "duration": 123.4,
                "skip_count": 8,
            },
        })
        adapter = AppleMusicSourceAdapter(FakeRunner(output))
        observation = adapter.build_observation(TRACK_ID, adapter.read_track("PID"))
        for path in ("artist_ids", "album_id", "duration_ms", "library_state.skip_count"):
            with self.subTest(path=path):
                self.assertIs(observation.fields[path].state, ObservationState.MISSING)
        self.assertNotIn("agent_metadata.tags", observation.fields)

    def test_confirmed_not_found_is_distinct_from_lookup_failure(self) -> None:
        not_found = AppleMusicSourceAdapter(
            FakeRunner('{"status":"confirmed_not_found"}')
        ).read_track("PID")
        failed = AppleMusicSourceAdapter(FakeRunner(AppleMusicReadError("denied"))).read_track("PID")
        malformed = AppleMusicSourceAdapter(FakeRunner("not-json")).read_track("PID")
        self.assertIs(not_found.status, SourceReadStatus.CONFIRMED_NOT_FOUND)
        self.assertIs(failed.status, SourceReadStatus.LOOKUP_FAILED)
        self.assertIs(malformed.status, SourceReadStatus.LOOKUP_FAILED)

    def test_invalid_rating_count_boolean_and_datetime_are_rejected_without_coercion(self) -> None:
        cases = (
            ("rating", 101),
            ("played_count", -1),
            ("favorited", 0),
            ("played_date", "2025-01-01T12:00:00"),
            ("duration_ms", -1),
        )
        for property_name, value in cases:
            with self.subTest(property_name=property_name):
                adapter = AppleMusicSourceAdapter(FakeRunner(json.dumps({
                    "status": "found", "fields": {property_name: value}
                })))
                read = adapter.read_track("PID")
                with self.assertRaises(AppleMusicMappingError):
                    adapter.build_observation(TRACK_ID, read)


if __name__ == "__main__":
    unittest.main()
