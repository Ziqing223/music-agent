"""P11-T4 + P3B batch 2: afplay preview boundary -- deterministic runner tests.

The production runner downloads the signed iTunes clip to a temporary file (bounded,
synchronous), starts ``afplay`` non-blocking, and removes the temp file after a stop or a
natural exit. These tests pin the command sequence, the fail-closed URL gate, the
single-channel replace/stop semantics, and the no-media-left-behind guarantee without ever
invoking curl/afplay for real.
"""

import os
import subprocess
import threading
import time
import unittest
from unittest import mock

from music_agent.catalog_preview import (
    AfplayPlayback,
    AfplayPreviewRunner,
    CatalogPreviewError,
)


class FakeAfplayProcess:
    """The Popen shape the runner depends on: poll/terminate/kill/wait plus an exit event."""

    def __init__(self) -> None:
        self._exited = threading.Event()
        self.returncode = 0
        self.terminated = False
        self.killed = False
        self._terminate_exits = True

    def poll(self):
        return self.returncode if self._exited.is_set() else None

    def wait(self, timeout=None):
        if self._exited.wait(timeout):
            return self.returncode
        raise subprocess.TimeoutExpired("afplay", timeout)

    def terminate(self):
        self.terminated = True
        if self._terminate_exits:
            self.returncode = -15
            self._exited.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._exited.set()

    def exit_naturally(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._exited.set()


class AfplayPreviewRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processes: list[FakeAfplayProcess] = []
        popen_patch = mock.patch(
            "music_agent.catalog_preview.subprocess.Popen",
            side_effect=self._record_popen,
        )
        self.popen = popen_patch.start()
        self.addCleanup(popen_patch.stop)
        # Fake processes are event-driven; keep the stop-wait tiny so the kill fallback
        # test does not block for the production 5 seconds.
        wait_patch = mock.patch.object(AfplayPlayback, "_STOP_WAIT_SECONDS", 0.05)
        wait_patch.start()
        self.addCleanup(wait_patch.stop)

    def _record_popen(self, command, **_kwargs):
        process = FakeAfplayProcess()
        self.processes.append(process)
        return process

    def _temp_path_from(self, call) -> str:
        args = call.args[0]
        return args[args.index("-o") + 1]

    @staticmethod
    def _wait_until(predicate, seconds: float = 2.0) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_start_audio_downloads_spawns_afplay_and_returns_immediately(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            AfplayPreviewRunner().start_audio("https://example.com/preview.m4a")
        self.assertEqual(len(run.call_args_list), 1)  # only the download is synchronous
        curl = run.call_args_list[0]
        self.assertEqual(curl.args[0][0], "curl")
        self.assertEqual(curl.args[0][-1], "https://example.com/preview.m4a")
        temp_path = self._temp_path_from(curl)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.popen.call_args.args[0], ["afplay", temp_path])
        self.assertTrue(os.path.exists(temp_path))  # still sounding: the clip is playing

    def test_non_http_urls_fail_closed_without_commands(self) -> None:
        with mock.patch("music_agent.catalog_preview.subprocess.run") as run:
            for url in ("", "file:///etc/passwd", "ftp://example.com/x.m4a"):
                with self.assertRaises(CatalogPreviewError, msg=url):
                    AfplayPreviewRunner().start_audio(url)
        run.assert_not_called()
        self.popen.assert_not_called()

    def test_curl_failure_raises_and_removes_temp(self) -> None:
        completed = mock.Mock(returncode=22, stderr="curl failed")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            with self.assertRaises(CatalogPreviewError):
                AfplayPreviewRunner().start_audio("https://example.com/p.m4a")
        self.assertFalse(os.path.exists(self._temp_path_from(run.call_args_list[0])))
        self.popen.assert_not_called()

    def test_afplay_spawn_failure_raises_and_removes_temp(self) -> None:
        def side_effect(command, **kwargs):
            if command[0] == "curl":
                return mock.Mock(returncode=0, stderr="")
            raise OSError("afplay not found")

        self.popen.side_effect = lambda *a, **k: (_ for _ in ()).throw(
            OSError("afplay not found")
        )
        with mock.patch("music_agent.catalog_preview.subprocess.run", side_effect=side_effect) as run:
            with self.assertRaises(CatalogPreviewError):
                AfplayPreviewRunner().start_audio("https://example.com/p.m4a")
        self.assertFalse(os.path.exists(self._temp_path_from(run.call_args_list[0])))

    def test_stop_preview_terminates_process_and_removes_temp(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            runner = AfplayPreviewRunner()
            runner.start_audio("https://example.com/p.m4a")
            temp_path = self._temp_path_from(run.call_args_list[0])
            self.assertTrue(runner.stop_preview())
        self.assertTrue(self.processes[0].terminated)
        self.assertTrue(self._wait_until(lambda: not os.path.exists(temp_path)))

    def test_stop_preview_idempotent_without_active_playback(self) -> None:
        self.assertFalse(AfplayPreviewRunner().stop_preview())

    def test_stop_falls_back_to_kill_when_terminate_hangs(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        runner = AfplayPreviewRunner()
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed):
            runner.start_audio("https://example.com/p.m4a")
        process = self.processes[0]
        process._terminate_exits = False
        self.assertTrue(runner.stop_preview())
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_starting_a_new_preview_stops_the_sounding_one(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        runner = AfplayPreviewRunner()
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed):
            runner.start_audio("https://example.com/one.m4a")
            runner.start_audio("https://example.com/two.m4a")
        self.assertTrue(self.processes[0].terminated)  # previous clip was replaced
        self.assertFalse(self.processes[1].terminated)
        self.assertTrue(runner.stop_preview())
        self.assertTrue(self.processes[1].terminated)

    def test_natural_exit_reaps_temp_file_and_clears_current(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            runner = AfplayPreviewRunner()
            runner.start_audio("https://example.com/p.m4a")
            temp_path = self._temp_path_from(run.call_args_list[0])
            self.processes[0].exit_naturally(0)
            self.assertTrue(self._wait_until(lambda: not os.path.exists(temp_path)))
            self.assertIsNone(runner._current)

    # --- P14-C06.3b: read-only preview truth -------------------------------------

    def test_is_preview_active_false_before_start_and_after_stop(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        runner = AfplayPreviewRunner()
        self.assertFalse(runner.is_preview_active())
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed):
            runner.start_audio("https://example.com/one.m4a")
        self.assertTrue(runner.is_preview_active())
        self.assertTrue(runner.stop_preview())
        self.assertFalse(runner.is_preview_active())

    def test_is_preview_active_reports_false_after_natural_exit(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed):
            runner = AfplayPreviewRunner()
            runner.start_audio("https://example.com/p.m4a")
        self.assertTrue(runner.is_preview_active())
        self.processes[0].exit_naturally(0)
        self.assertTrue(self._wait_until(lambda: not runner.is_preview_active()))

    def test_is_preview_active_is_purely_read_only(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed):
            runner = AfplayPreviewRunner()
            runner.start_audio("https://example.com/p.m4a")
        current = runner._current
        self.assertTrue(runner.is_preview_active())
        self.assertTrue(runner.is_preview_active())
        self.assertIs(runner._current, current)  # probing mutates nothing
        self.assertFalse(self.processes[0].terminated)  # probing stops nothing

    def test_timeout_maps_to_preview_error(self) -> None:
        with mock.patch(
            "music_agent.catalog_preview.subprocess.run",
            side_effect=subprocess.TimeoutExpired("curl", 1),
        ):
            with self.assertRaises(CatalogPreviewError):
                AfplayPreviewRunner().start_audio("https://example.com/p.m4a")

    def test_invalid_timeout_rejected(self) -> None:
        for timeout in (0, -1):
            with self.assertRaises(CatalogPreviewError, msg=timeout):
                AfplayPreviewRunner(timeout_seconds=timeout)


class NaturalFinishHookTest(unittest.TestCase):
    """P15-S1: the reaper distinguishes a natural end from a stop, and only the
    natural end fires ``on_natural_finish`` -- once, with listener failures
    swallowed so cleanup always completes.
    """

    def setUp(self) -> None:
        self.processes: list[FakeAfplayProcess] = []
        popen_patch = mock.patch(
            "music_agent.catalog_preview.subprocess.Popen",
            side_effect=self._record_popen,
        )
        self.popen = popen_patch.start()
        self.addCleanup(popen_patch.stop)
        wait_patch = mock.patch.object(AfplayPlayback, "_STOP_WAIT_SECONDS", 0.05)
        wait_patch.start()
        self.addCleanup(wait_patch.stop)

    def _record_popen(self, command, **_kwargs):
        process = FakeAfplayProcess()
        self.processes.append(process)
        return process

    def _start(self, events: list) -> tuple[AfplayPreviewRunner, str]:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            runner = AfplayPreviewRunner(on_natural_finish=lambda: events.append("finished"))
            runner.start_audio("https://example.com/p.m4a")
            temp_path = run.call_args_list[0].args[0][
                run.call_args_list[0].args[0].index("-o") + 1
            ]
        return runner, temp_path

    def test_hook_is_optional_and_settable(self) -> None:
        runner = AfplayPreviewRunner()
        self.assertIsNone(runner.on_natural_finish)
        marker = []
        runner.on_natural_finish = lambda: marker.append(1)
        self.assertTrue(callable(runner.on_natural_finish))
        runner.on_natural_finish = None
        self.assertIsNone(runner.on_natural_finish)

    def test_non_callable_hook_refused(self) -> None:
        with self.assertRaises(CatalogPreviewError):
            AfplayPreviewRunner().on_natural_finish = "not callable"  # type: ignore[assignment]

    def test_natural_end_fires_hook_once_and_still_reaps(self) -> None:
        events: list[str] = []
        runner, temp_path = self._start(events)
        self.assertTrue(os.path.exists(temp_path))
        self.processes[0].exit_naturally(0)
        self.assertTrue(self._wait_until(lambda: bool(events)))
        self.assertTrue(self._wait_until(lambda: not os.path.exists(temp_path)))
        self.assertEqual(events, ["finished"])  # exactly once
        self.assertIsNone(runner._current)

    def test_stop_end_never_fires_hook(self) -> None:
        events: list[str] = []
        runner, _ = self._start(events)
        self.assertTrue(runner.stop_preview())
        self._wait_until(lambda: runner._current is None)
        self.assertEqual(events, [])

    def test_nonzero_exit_mid_preview_is_a_failure_not_a_natural_end(self) -> None:
        """P17 acceptance: an afplay that dies nonzero (crash, device loss) must
        not fire the natural-finish hook -- no session advance, no auto-restore
        of the formal playback the preview interrupted -- while cleanup still
        completes."""
        events: list[str] = []
        runner, temp_path = self._start(events)
        self.processes[0].exit_naturally(1)
        self.assertTrue(self._wait_until(lambda: runner._current is None))
        self.assertTrue(self._wait_until(lambda: not os.path.exists(temp_path)))
        self.assertEqual(events, [])

    def test_listener_failure_is_swallowed_and_cleanup_completes(self) -> None:
        def explodes() -> None:
            raise RuntimeError("presenter broken")

        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch("music_agent.catalog_preview.subprocess.run", return_value=completed) as run:
            runner = AfplayPreviewRunner(on_natural_finish=explodes)
            runner.start_audio("https://example.com/p.m4a")
            temp_path = run.call_args_list[0].args[0][
                run.call_args_list[0].args[0].index("-o") + 1
            ]
        self.processes[0].exit_naturally(0)
        self.assertTrue(self._wait_until(lambda: not os.path.exists(temp_path)))
        self.assertIsNone(runner._current)

    @staticmethod
    def _wait_until(predicate, seconds: float = 2.0) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


if __name__ == "__main__":
    unittest.main()