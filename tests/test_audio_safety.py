"""P15-S2 r3: audio-output observer tests (formerly the P10.5 safety monitor).

The observer is driven deterministically through ``poll_once`` with a fake device reader
and a fake authority (the agent-service entry point); it never touches playback. The
CoreAudio/osascript production boundaries are live-verified on the real Mac (see the P10.7
validation), never exercised by unit tests.
"""

import threading
import time
import unittest

from music_agent.audio_safety import (
    AudioObserverEventKind,
    AudioOutputObserver,
    CoreAudioDefaultOutputReader,
    OutputDeviceState,
)
from music_agent.playback_control import (
    MusicPlaybackAdapter,
    PlaybackControlUnavailableError,
    PlayerState,
)


def speaker_state(device_id: int = 70) -> OutputDeviceState:
    return OutputDeviceState(device_id=device_id, transport_type="bltn", uid="BuiltInSpeakerDevice")


def private_state(device_id: int = 99) -> OutputDeviceState:
    return OutputDeviceState(device_id=device_id, transport_type="blue", uid="AirPodsPro")


class FakeDeviceReader:
    def __init__(self, states: list[OutputDeviceState | None]) -> None:
        self.states = list(states)
        self.last: OutputDeviceState | None = None
        self.calls = 0

    def read_default_output(self) -> OutputDeviceState | None:
        self.calls += 1
        if self.states:
            self.last = self.states.pop(0)
        return self.last  # steady state once the scripted sequence ends


class FakeAuthority:
    """Stands in for ``SharedAgentService`` at the observer boundary: records forwards.

    The service entry ``handle_default_output_transition`` is the single safety
    decision point; the observer must deliver (before, after) snapshot pairs and
    never interpret the returned action.
    """

    def __init__(self, action: object = "decided") -> None:
        self.calls: list[tuple[object, object]] = []
        self.action = action
        self.raise_once: Exception | None = None

    def handle_default_output_transition(self, before: object, after: object) -> object:
        self.calls.append((before, after))
        if self.raise_once is not None:
            error = self.raise_once
            self.raise_once = None
            raise error
        return self.action


def observer(
    states: list[OutputDeviceState | None],
    authority: FakeAuthority | None = None,
    *,
    poll_interval_seconds: float = 1.0,
) -> tuple[AudioOutputObserver, FakeDeviceReader]:
    reader = FakeDeviceReader(states)
    # The observer requires the production reader type; fakes subclass it.
    class FakeReaderWithType(CoreAudioDefaultOutputReader):
        def __init__(self, fake: FakeDeviceReader) -> None:
            self.fake = fake  # bypasses the CoreAudio load entirely

        def read_default_output(self):  # type: ignore[override]
            return self.fake.read_default_output()

    wrapped = FakeReaderWithType(reader)
    obs = AudioOutputObserver(
        wrapped,
        authority or FakeAuthority(),
        poll_interval_seconds=poll_interval_seconds,
        clock=lambda: "2026-08-19T00:00:00Z",
    )
    return obs, reader


class FourccTest(unittest.TestCase):
    def test_decodes_known_tags(self) -> None:
        from music_agent.audio_safety import _fourcc

        self.assertEqual(_fourcc(0x626C746E), "bltn")
        self.assertEqual(_fourcc(0x626C7565), "blue")
        self.assertEqual(_fourcc(0x7472616E), "tran")
        self.assertEqual(_fourcc(0x644F7574), "dOut")

    def test_undecodable_tag_returns_empty(self) -> None:
        from music_agent.audio_safety import _fourcc

        self.assertEqual(_fourcc(0x0000FF00), "")


class PlaybackAdapterTest(unittest.TestCase):
    def test_state_mapping_and_unknown_inputs(self) -> None:
        class Runner:
            def read_player_state(self) -> str:
                return self.output  # type: ignore[attr-defined]

            def pause(self) -> None: ...

            def read_now_playing(self) -> str:
                return "{}"

            def play(self) -> None: ...

            def next_track(self) -> None: ...

            def previous_track(self) -> None: ...

            def play_track(self, persistent_id: str) -> None: ...

        runner = Runner()
        adapter = MusicPlaybackAdapter(runner)  # type: ignore[arg-type]
        for raw, expected in (
            ("playing", PlayerState.PLAYING),
            ("paused", PlayerState.PAUSED),
            ("stopped", PlayerState.STOPPED),
            ("PLAYING\n", PlayerState.PLAYING),
            ("  paused ", PlayerState.PAUSED),
            ("buffering", PlayerState.UNKNOWN),
            ("", PlayerState.UNKNOWN),
        ):
            runner.output = raw
            self.assertEqual(adapter.read_player_state(), expected)

    def test_runner_failure_maps_to_unavailable(self) -> None:
        class Runner:
            def read_player_state(self) -> str:
                raise PlaybackControlUnavailableError("Application isn't running")

            def pause(self) -> None: ...

            def read_now_playing(self) -> str:
                return "{}"

            def play(self) -> None: ...

            def next_track(self) -> None: ...

            def previous_track(self) -> None: ...

            def play_track(self, persistent_id: str) -> None: ...

        adapter = MusicPlaybackAdapter(Runner())  # type: ignore[arg-type]
        with self.assertRaises(PlaybackControlUnavailableError):
            adapter.read_player_state()


class AudioOutputObserverTest(unittest.TestCase):
    def test_baseline_first_read_records_but_never_forwards(self) -> None:
        authority = FakeAuthority()
        obs, _ = observer([private_state()], authority)
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.BASELINE)
        self.assertIsNotNone(event.snapshot)
        self.assertEqual(event.snapshot.device_uid, "AirPodsPro")
        self.assertEqual(event.snapshot.event_type.value, "baseline")
        self.assertEqual(
            event.snapshot.unavailable_fields,
            frozenset({"device_name", "data_source_id", "data_source_name", "device_alive"}),
        )
        self.assertEqual(authority.calls, [])

    def test_private_to_speaker_forwards_exactly_one_snapshot_pair(self) -> None:
        authority = FakeAuthority(action="paused-binding")
        obs, _ = observer([private_state(), speaker_state()], authority)
        obs.poll_once()  # baseline
        event = obs.poll_once()  # the fallback transition
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARDED)
        self.assertEqual(len(authority.calls), 1)
        before, after = authority.calls[0]
        self.assertEqual(before.device_id, "99")
        self.assertEqual(before.device_uid, "AirPodsPro")
        self.assertEqual(after.device_uid, "BuiltInSpeakerDevice")
        self.assertEqual(before.event_type.value, "default_device_changed")
        self.assertEqual(after.event_type.value, "default_device_changed")
        self.assertEqual(event.action, "paused-binding")  # echoed verbatim, never interpreted
        self.assertIsNone(event.error)

    def test_steady_speaker_polls_do_not_reforward(self) -> None:
        authority = FakeAuthority()
        obs, _ = observer([private_state(), speaker_state(), speaker_state()], authority)
        obs.poll_once()  # baseline
        self.assertEqual(obs.poll_once().kind, AudioObserverEventKind.FORWARDED)
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.NO_CHANGE)
        self.assertEqual(len(authority.calls), 1)  # one transition, one forward

    def test_speaker_to_private_is_still_forwarded_for_the_authority_to_rearm(self) -> None:
        # The observer is detection-only: leaving the speaker is a fact the single
        # authority consumes to clear its dedup key; it must never be filtered here.
        authority = FakeAuthority(action=None)
        obs, _ = observer([speaker_state(), private_state()], authority)
        obs.poll_once()  # baseline
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARDED)
        self.assertEqual(len(authority.calls), 1)
        self.assertEqual(authority.calls[0][1].device_uid, "AirPodsPro")
        self.assertIsNone(event.action)  # authority decided "no action" -- observer obeys

    def test_private_to_private_forwarded_once(self) -> None:
        authority = FakeAuthority()
        second_private = OutputDeviceState(100, "usb ", "USBHeadset")
        obs, _ = observer([private_state(99), second_private], authority)
        obs.poll_once()
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARDED)
        self.assertEqual(event.snapshot.device_uid, "USBHeadset")
        self.assertEqual(len(authority.calls), 1)

    def test_unreadable_read_fails_closed_and_next_transition_still_fires(self) -> None:
        authority = FakeAuthority()
        obs, _ = observer([private_state(), None, speaker_state()], authority)
        obs.poll_once()  # baseline private
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.UNAVAILABLE)
        self.assertEqual(len(authority.calls), 0)
        event = obs.poll_once()  # last known good -> speaker: the hazard still forwards
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARDED)
        self.assertEqual(len(authority.calls), 1)

    def test_service_failure_is_recorded_and_the_loop_survives(self) -> None:
        authority = FakeAuthority()
        authority.raise_once = ValueError("decision exploded")
        obs, _ = observer([private_state(), speaker_state(), private_state(98)], authority)
        obs.poll_once()  # baseline
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARD_FAILED)
        self.assertEqual(event.error, "decision exploded")
        self.assertIsNotNone(event.snapshot)  # the observed after fact survives
        # A later distinct transition forwards again: the pump never dies on a bad decision.
        event = obs.poll_once()
        self.assertEqual(event.kind, AudioObserverEventKind.FORWARDED)
        self.assertEqual(len(authority.calls), 2)
        self.assertEqual(authority.calls[1][1].device_uid, "AirPodsPro")

    def test_thread_lifecycle_starts_polls_and_stops(self) -> None:
        obs, _ = observer([private_state()])
        obs.start()
        try:
            deadline = time.monotonic() + 5.0
            while obs.last_event is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(obs.last_event)
        finally:
            obs.close()
        self.assertIsNone(obs._thread)

    def test_observer_owns_no_playback_or_old_policy_surface(self) -> None:
        obs, _ = observer([speaker_state()])
        self.assertFalse(hasattr(obs, "_playback"))
        self.assertFalse(hasattr(obs, "pending_pause"))
        self.assertFalse(hasattr(obs, "_pause_attempts"))
        self.assertFalse(hasattr(obs, "pause_retry_limit"))

    def test_output_device_state_keeps_no_builtin_heuristic(self) -> None:
        # r3 deletes the transport-based builtin judgement: identity is uid-only,
        # decided by the authority -- never a transport/name shortcut in the observer.
        self.assertFalse(hasattr(speaker_state(), "is_built_in_speaker"))
        self.assertFalse(hasattr(private_state(), "is_built_in_speaker"))

    def test_validation_rejects_bad_arguments(self) -> None:
        from music_agent.audio_safety import AudioSafetyError

        reader = CoreAudioDefaultOutputReader.__new__(CoreAudioDefaultOutputReader)

        class MissingEntry:
            pass

        with self.assertRaises(AudioSafetyError):
            AudioOutputObserver(reader, MissingEntry())  # type: ignore[arg-type]
        with self.assertRaises(AudioSafetyError):
            AudioOutputObserver(reader, FakeAuthority(), poll_interval_seconds=0)  # type: ignore[arg-type]
        with self.assertRaises(AudioSafetyError):
            AudioOutputObserver(object(), FakeAuthority())  # type: ignore[arg-type]


class RuntimeAudioSafetyWiringTest(unittest.TestCase):
    def test_runtime_wires_monitor_when_enabled(self) -> None:
        import tempfile
        from pathlib import Path

        from music_agent.runtime import Runtime, RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(
                RuntimeConfig(
                    database_path=Path(tmp) / "store.db",
                    audio_safety_enabled=True,
                    audio_safety_poll_interval_seconds=1.0,
                )
            )
            runtime.start()
            try:
                self.assertIsNotNone(runtime.audio_monitor)
            finally:
                runtime.close()

    def test_runtime_skips_monitor_when_disabled(self) -> None:
        import tempfile
        from pathlib import Path

        from music_agent.runtime import Runtime, RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(
                RuntimeConfig(database_path=Path(tmp) / "store.db", audio_safety_enabled=False)
            )
            runtime.start()
            try:
                self.assertIsNone(runtime.audio_monitor)
            finally:
                runtime.close()

    def test_config_validates_audio_fields(self) -> None:
        from pathlib import Path

        from music_agent.runtime import RuntimeConfig, RuntimeStartupError

        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), audio_safety_poll_interval_seconds=0)
        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), audio_safety_enabled="yes")  # type: ignore[arg-type]


class AudioSafetyCliFlagTest(unittest.TestCase):
    def test_run_accepts_audio_safety_flags(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(["run", "--db", "store.db", "--no-audio-safety"])
        self.assertTrue(args.no_audio_safety)
        args = build_parser().parse_args(
            ["run", "--db", "store.db", "--audio-safety-poll-interval", "1.5"]
        )
        self.assertEqual(args.audio_safety_poll_interval, 1.5)


class DeviceSafetyTraceTest(unittest.TestCase):
    """The temporary P15-S2 [device-safety] diagnosis channel is flag-gated:
    silently off in every normal run, one stderr line per message when on."""

    def test_trace_is_silent_when_flag_unset(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.audio_safety import device_safety_trace

        buffer = io.StringIO()
        with patch.dict("os.environ", {}, clear=True):
            with redirect_stderr(buffer):
                device_safety_trace("should not appear")
        self.assertEqual(buffer.getvalue(), "")

    def test_trace_prints_prefixed_line_to_stderr_when_enabled(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.audio_safety import (
            device_safety_trace,
            device_safety_trace_enabled,
        )

        buffer = io.StringIO()
        with patch.dict("os.environ", {"MUSIC_AGENT_DEVICE_SAFETY_TRACE": "1"}, clear=True):
            self.assertTrue(device_safety_trace_enabled())
            with redirect_stderr(buffer):
                device_safety_trace("hello live")
        self.assertEqual(buffer.getvalue(), "[device-safety] hello live\n")


if __name__ == "__main__":
    unittest.main()
