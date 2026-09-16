"""P12: playback-equivalent library resolver tests (deterministic fakes; no real Music.app).

The resolver turns one live Music.app library candidate query into at most one
playback persistent ID under a strict evidence gate. It writes nothing: no canonical
state, no bindings, no identity facts. Everything here is fail-closed by contract.
"""

import unittest
from unittest.mock import patch

from music_agent.playback_control import (
    LIBRARY_RESOLVE_SCRIPT,
    MusicLibraryResolver,
    OsascriptLibraryResolveRunner,
    PlaybackControlError,
    PlaybackControlUnavailableError,
)

SPRING_THIEF_LINE = (
    "398490020FF165D3\tSpring Thief\tYorushika\tSpring Thief - Single\t290.278991699219\n"
)


class FakeResolveRunner:
    def __init__(self, raw: str = "") -> None:
        self.raw = raw
        self.names: list[str] = []

    def resolve_candidates(self, name: str) -> str:
        self.names.append(name)
        return self.raw


class MusicLibraryResolverTest(unittest.TestCase):
    TARGET = dict(
        name="Spring Thief",
        artist="Yorushika",
        album="Spring Thief - Single",
        duration_ms=290279,
    )

    def _resolve(self, raw: str, **target):
        runner = FakeResolveRunner(raw)
        result = MusicLibraryResolver(runner).resolve_playback_track(**target)
        return result, runner

    def test_unique_match_resolves_persistent_id(self) -> None:
        result, runner = self._resolve(SPRING_THIEF_LINE, **self.TARGET)
        self.assertEqual(result, "398490020FF165D3")
        self.assertEqual(runner.names, ["Spring Thief"])

    def test_normalization_case_and_whitespace(self) -> None:
        raw = "PID-1\tspring  thief\tyorushika\tSPRING THIEF - SINGLE\t290.2789\n"
        result, _ = self._resolve(
            raw,
            name="  SprIng   THIEF ",
            artist="YORUSHIKA",
            album=" spring thief - single ",
            duration_ms=290279,
        )
        self.assertEqual(result, "PID-1")

    def test_normalization_nfc(self) -> None:
        # Decomposed e+combining-acute must equal the composed form after NFC.
        raw = "PID-1\tCafe\u0301 Song\tArt\tAlbum\t100.0\n"  # decomposed
        result, _ = self._resolve(
            raw,
            name="Caf\u00e9 Song",  # composed
            artist="Art",
            album="Album",
            duration_ms=100000,
        )
        self.assertEqual(result, "PID-1")

    def test_zero_candidates_returns_none(self) -> None:
        result, _ = self._resolve("", **self.TARGET)
        self.assertIsNone(result)

    def test_multiple_full_matches_fail_closed(self) -> None:
        raw = SPRING_THIEF_LINE + "398490020FF165D4\tSpring Thief\tYorushika\tSpring Thief - Single\t290.279\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertIsNone(result)

    def test_missing_target_artist_fails_without_query(self) -> None:
        result, runner = self._resolve(SPRING_THIEF_LINE, **{**self.TARGET, "artist": None})
        self.assertIsNone(result)
        self.assertEqual(runner.names, [])

    def test_missing_target_duration_fails_without_query(self) -> None:
        result, runner = self._resolve(SPRING_THIEF_LINE, **{**self.TARGET, "duration_ms": None})
        self.assertIsNone(result)
        self.assertEqual(runner.names, [])

    def test_missing_target_name_fails_without_query(self) -> None:
        result, runner = self._resolve(SPRING_THIEF_LINE, **{**self.TARGET, "name": "  "})
        self.assertIsNone(result)
        self.assertEqual(runner.names, [])

    def test_artist_mismatch_returns_none(self) -> None:
        raw = "PID-1\tSpring Thief\tSomeone Else\tSpring Thief - Single\t290.279\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertIsNone(result)

    def test_duration_at_tolerance_boundary_matches(self) -> None:
        # |289.279 - 290.279| == 1.0 -> within tolerance.
        raw = "PID-1\tSpring Thief\tYorushika\tSpring Thief - Single\t289.279\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertEqual(result, "PID-1")

    def test_duration_over_tolerance_fails(self) -> None:
        raw = "PID-1\tSpring Thief\tYorushika\tSpring Thief - Single\t289.278\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertIsNone(result)

    def test_album_mismatch_does_not_veto_unique_primary_match(self) -> None:
        raw = "PID-1\tSpring Thief\tYorushika\tDifferent Album\t290.279\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertEqual(result, "PID-1")

    def test_incident_compilation_target_resolves_against_single_library_track(self) -> None:
        # Real P12 incident: catalog pick from the "TOKYO - GRADUATION -"
        # compilation vs the library's "Spring Thief - Single" recording.
        result, _ = self._resolve(
            SPRING_THIEF_LINE,
            name="Spring Thief",
            artist="Yorushika",
            album="TOKYO - GRADUATION -",
            duration_ms=290267,
        )
        self.assertEqual(result, "398490020FF165D3")

    def test_multiple_primary_matches_album_tie_break_disambiguates(self) -> None:
        raw = (
            "PID-SINGLE\tSpring Thief\tYorushika\tSpring Thief - Single\t290.279\n"
            "PID-COMP\tSpring Thief\tYorushika\tTOKYO - GRADUATION -\t290.267\n"
        )
        result, _ = self._resolve(
            raw,
            name="Spring Thief",
            artist="Yorushika",
            album="Spring Thief - Single",
            duration_ms=290279,
        )
        self.assertEqual(result, "PID-SINGLE")
        result, _ = self._resolve(
            raw,
            name="Spring Thief",
            artist="Yorushika",
            album="TOKYO - GRADUATION -",
            duration_ms=290267,
        )
        self.assertEqual(result, "PID-COMP")

    def test_multiple_primary_matches_album_matches_none_fails_closed(self) -> None:
        raw = (
            "PID-1\tSpring Thief\tYorushika\tSpring Thief - Single\t290.279\n"
            "PID-2\tSpring Thief\tYorushika\tTOKYO - GRADUATION -\t290.267\n"
        )
        result, _ = self._resolve(
            raw,
            name="Spring Thief",
            artist="Yorushika",
            album="Some Other Release",
            duration_ms=290270,
        )
        self.assertIsNone(result)

    def test_multiple_primary_matches_without_target_album_fails_closed(self) -> None:
        raw = (
            "PID-1\tSpring Thief\tYorushika\tSpring Thief - Single\t290.279\n"
            "PID-2\tSpring Thief\tYorushika\tTOKYO - GRADUATION -\t290.267\n"
        )
        result, _ = self._resolve(
            raw,
            name="Spring Thief",
            artist="Yorushika",
            album=None,
            duration_ms=290270,
        )
        self.assertIsNone(result)

    def test_album_absent_on_target_skips_album_clause(self) -> None:
        raw = "PID-1\tSpring Thief\tYorushika\t\t290.279\n"
        result, _ = self._resolve(raw, **{**self.TARGET, "album": None})
        self.assertEqual(result, "PID-1")

    def test_album_absent_on_candidate_skips_album_clause(self) -> None:
        raw = "PID-1\tSpring Thief\tYorushika\t\t290.279\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertEqual(result, "PID-1")

    def test_candidate_without_duration_fails(self) -> None:
        raw = "PID-1\tSpring Thief\tYorushika\tSpring Thief - Single\t\n"
        result, _ = self._resolve(raw, **self.TARGET)
        self.assertIsNone(result)

    def test_malformed_output_fails_closed(self) -> None:
        with self.assertRaises(PlaybackControlUnavailableError):
            self._resolve("PID-1\tSpring Thief\tYorushika\n", **self.TARGET)

    def test_runner_error_propagates_as_unavailable(self) -> None:
        class FailingRunner:
            def resolve_candidates(self, name: str) -> str:
                raise PlaybackControlUnavailableError("Application isn't running")

        resolver = MusicLibraryResolver(FailingRunner())
        with self.assertRaises(PlaybackControlUnavailableError):
            resolver.resolve_playback_track(**self.TARGET)

    def test_runner_validation(self) -> None:
        with self.assertRaises(PlaybackControlError):
            MusicLibraryResolver(object())  # type: ignore[arg-type]

    def test_osascript_runner_passes_name_via_argv_and_fails_closed(self) -> None:
        runner = OsascriptLibraryResolveRunner(timeout_seconds=5.0)
        with patch("music_agent.playback_control.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = SPRING_THIEF_LINE
            output = runner.resolve_candidates("Spring Thief")
        call = run.call_args
        self.assertEqual(call.args[0][:3], ["osascript", "-e", LIBRARY_RESOLVE_SCRIPT])
        self.assertEqual(call.args[0][3:], ["--", "Spring Thief"])
        self.assertEqual(output, SPRING_THIEF_LINE)

        with patch("music_agent.playback_control.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "AppleEvent timed out"
            with self.assertRaises(PlaybackControlUnavailableError):
                runner.resolve_candidates("Spring Thief")

    def test_resolve_script_has_no_name_interpolation(self) -> None:
        self.assertNotIn("Spring Thief", LIBRARY_RESOLVE_SCRIPT)
        self.assertIn("item 1 of argv", LIBRARY_RESOLVE_SCRIPT)


if __name__ == "__main__":
    unittest.main()
