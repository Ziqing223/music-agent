"""P15-S2-IPC S4: the preview-event return channel.

Covers, in the order of the design:

* the event frame codec (JSON ``{event, session, suspended}`` texts on the
  S1 framing, fail-closed);
* the run-side PreviewEventForwarder (one event per connection, terminal
  events end the bridge, every send failure swallowed);
* the chat-side AgentEventSocketListener (bind/refuse/unlink ownership,
  presenter dispatch, garbage frames and raising presenters survive);
* AgentSocketServer arming (sidecar stripped before execute, forwarder on
  the presenter slot, restore identity-guarded, disarmed on a raising
  execute, events after the response still bridge);
* RoutedAgentClient injection (sidecar only on the preview family, lazy
  listener bind, degrade-without-sidecar on bind failure, close/re-bind);
* one full chat->run->chat round trip (client injects, server strips+arms,
  the service fires a session event, the chat presenter receives it).

Real AF_UNIX sockets in temp directories throughout. Known environment
precondition: Claude Code's sandbox denies filesystem bind(); run this
module unsandboxed or from a normal terminal (the same S2/S3 condition).
"""

import os
import socket
import tempfile
import threading
import time
import unittest
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from music_agent.agent_client import AgentClientValidationError
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_socket import (
    AgentEventSocketListener,
    AgentSocketError,
    AgentSocketServer,
    PreviewEventForwarder,
    SocketFrameError,
    decode_event_frame,
    encode_event_frame,
    frame_to_result,
    receive_framed,
    request_to_frame,
    send_framed,
)
from music_agent.routed_client import RoutedAgentClient

_CLIENT_ID = "agt_50000000-0000-4000-8000-000000000004"
_CLIENT = AgentClientIdentity(client_id=_CLIENT_ID, model_id="s4-test", label="test")
_INSTANT = datetime(2026, 8, 19, 2, 0, tzinfo=timezone.utc)
_REQ_B1 = "req_00000000-0000-4000-8000-0000000000b1"

_PROGRESS_EVENT = {
    "event": "progress",
    "session": {
        "current_name": "夜空中最亮的星",
        "queue_position": 2,
        "total": 5,
        "elapsed_ms": 1024,
    },
    "suspended": None,
}

_CANCELLED_EVENT = {
    "event": "cancelled",
    "session": {"queue_position": 3, "total": 3},
    "suspended": {"restore_hint": "恢复播放：输入 继续播放", "origin_tool": "preview_batch"},
}


def wait_until(fn, timeout: float = 5.0, step: float = 0.02) -> bool:
    """Poll an outcome up to ``timeout``; returns the final truth value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(step)
    return fn()


class RecordingPresenter:
    """The chat-side event sink; optionally broken to prove survival."""

    def __init__(self, raise_errors: bool = False) -> None:
        self.events: list[dict] = []
        self.raise_errors = raise_errors

    def __call__(self, event: Mapping) -> None:
        if self.raise_errors:
            raise RuntimeError("presenter broken")
        self.events.append(dict(event))


def start_sink(root: Path, raise_errors: bool = False):
    presenter = RecordingPresenter(raise_errors=raise_errors)
    listener = AgentEventSocketListener(root / "events.sock", presenter)
    listener.start()
    return listener, presenter


def start_agent_server(root: Path, backend, name: str = "agent.sock") -> AgentSocketServer:
    server = AgentSocketServer(root / name, backend)
    server.start()
    return server


def make_request(tool: str, payload: Mapping) -> AgentRequest:
    return AgentRequest(
        request_id=_REQ_B1,
        client=_CLIENT,
        tool=tool,
        payload=payload,
        issued_at=_INSTANT,
    )


def exchange(server: AgentSocketServer, request: AgentRequest, timeout: float = 10.0) -> AgentToolResult:
    """One raw framed request -> response while the main thread pumps tick."""
    holder: dict = {}

    def worker() -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect(str(server.socket_path))
            send_framed(sock, request_to_frame(request))
            frame = receive_framed(sock)
        finally:
            sock.close()
        holder["frame"] = frame

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        server.tick()
        time.sleep(0.02)
    thread.join(timeout=2.0)
    if thread.is_alive() or "frame" not in holder:
        raise AssertionError("exchange did not complete under tick pumping")
    return frame_to_result(holder["frame"])


class EventFrameCodecTest(unittest.TestCase):
    def test_roundtrip_preserves_presenter_shape(self) -> None:
        frame = encode_event_frame(_PROGRESS_EVENT)
        self.assertIsInstance(frame, bytes)
        self.assertEqual(decode_event_frame(frame), _PROGRESS_EVENT)

    def test_empty_mapping_roundtrips(self) -> None:
        self.assertEqual(decode_event_frame(encode_event_frame({})), {})

    def test_encode_requires_a_mapping(self) -> None:
        with self.assertRaises(AgentSocketError):
            encode_event_frame([1, 2])  # type: ignore[arg-type]

    def test_decode_requires_bytes(self) -> None:
        with self.assertRaises(AgentSocketError):
            decode_event_frame("{}")  # type: ignore[arg-type]

    def test_non_json_frame_fails_closed(self) -> None:
        with self.assertRaises(SocketFrameError):
            decode_event_frame(b"not json")

    def test_json_array_frame_fails_closed(self) -> None:
        with self.assertRaises(SocketFrameError):
            decode_event_frame(b"[1, 2]")


class PreviewEventForwarderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_delivers_events_verbatim_to_the_listener(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        forwarder = PreviewEventForwarder(listener.socket_path)
        forwarder(_PROGRESS_EVENT)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0], _PROGRESS_EVENT)
        self.assertFalse(forwarder.terminal)

    def test_terminal_event_is_delivered_and_ends_the_bridge(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        terminals: list[int] = []
        forwarder = PreviewEventForwarder(
            listener.socket_path, on_terminal=lambda: terminals.append(1)
        )
        terminal_event = {"event": "completed", "session": {"total": 3}, "suspended": None}
        forwarder(terminal_event)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0]["event"], "completed")
        self.assertTrue(forwarder.terminal)
        self.assertEqual(terminals, [1])
        forwarder(_PROGRESS_EVENT)  # after the end: silently dropped
        time.sleep(0.1)
        self.assertEqual(len(presenter.events), 1)
        self.assertEqual(terminals, [1])

    def test_close_fires_on_terminal_once_and_stops_delivery(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        terminals: list[int] = []
        forwarder = PreviewEventForwarder(
            listener.socket_path, on_terminal=lambda: terminals.append(1)
        )
        forwarder.close()
        forwarder.close()
        self.assertEqual(terminals, [1])
        self.assertTrue(forwarder.terminal)
        forwarder(_PROGRESS_EVENT)
        time.sleep(0.1)
        self.assertEqual(presenter.events, [])

    def test_missing_listener_is_swallowed(self) -> None:
        forwarder = PreviewEventForwarder(self.root / "nobody.sock")
        forwarder(_PROGRESS_EVENT)  # no raise, nothing delivered, still live
        self.assertFalse(forwarder.terminal)

    def test_validation(self) -> None:
        with self.assertRaises(AgentSocketError):
            PreviewEventForwarder(Path(""))
        with self.assertRaises(AgentSocketError):
            PreviewEventForwarder(Path("x.sock"), on_terminal=42)  # type: ignore[arg-type]
        with self.assertRaises(AgentSocketError):
            PreviewEventForwarder(Path("x.sock"), timeout=0)


class AgentEventSocketListenerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _send_event(self, path: Path, event: Mapping) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect(str(path))
            send_framed(sock, encode_event_frame(event))
        finally:
            sock.close()

    def test_presenter_receives_events_from_separate_connections(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        self._send_event(listener.socket_path, _PROGRESS_EVENT)
        self._send_event(listener.socket_path, _CANCELLED_EVENT)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 2))
        self.assertEqual(presenter.events[0], _PROGRESS_EVENT)
        self.assertEqual(presenter.events[1], _CANCELLED_EVENT)

    def test_garbage_frame_drops_only_that_event(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(listener.socket_path))
        try:
            sock.sendall(b"\x00\x00\x00\x08notjson!")
        finally:
            sock.close()
        self._send_event(listener.socket_path, _PROGRESS_EVENT)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))

    def test_raising_presenter_survives(self) -> None:
        listener, presenter = start_sink(self.root, raise_errors=True)
        self.addCleanup(listener.close)
        self._send_event(listener.socket_path, _PROGRESS_EVENT)
        time.sleep(0.1)
        presenter.raise_errors = False
        self._send_event(listener.socket_path, _CANCELLED_EVENT)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0], _CANCELLED_EVENT)

    def test_concurrent_bind_refused_and_close_unlinks(self) -> None:
        listener, _presenter = start_sink(self.root)
        path = listener.socket_path
        second = AgentEventSocketListener(path, RecordingPresenter())
        with self.assertRaises(AgentSocketError):
            second.start()
        listener.close()
        self.assertFalse(path.exists())
        third = AgentEventSocketListener(path, RecordingPresenter())
        third.start()  # the unlinked path binds again cleanly
        third.close()

    def test_validation(self) -> None:
        with self.assertRaises(AgentSocketError):
            AgentEventSocketListener(Path(""), RecordingPresenter())
        with self.assertRaises(AgentSocketError):
            AgentEventSocketListener(Path("x.sock"), 42)  # type: ignore[arg-type]


class EventAwareService:
    """Enough service surface to host sockets: an execute() returning a valid
    envelope, plus the P15 presenter slot the server arms."""

    def __init__(self, fire_event: Mapping | None = None, raise_error: Exception | None = None) -> None:
        self.preview_event_handler = None
        self.arms_seen: list[object] = []
        self.executed: list[AgentRequest] = []
        self.fire_event = fire_event
        self.raise_error = raise_error

    def execute(self, request: AgentRequest, *, completed_at=None) -> AgentToolResult:
        self.executed.append(request)
        self.arms_seen.append(self.preview_event_handler)
        if self.raise_error is not None:
            raise self.raise_error
        if self.fire_event is not None and self.preview_event_handler is not None:
            self.preview_event_handler(self.fire_event)
        return AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=AgentToolOutcome.OK,
            payload={"ok": True},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )


class HandlerlessService:
    """No ``preview_event_handler`` attribute at all: the no-arm branch."""

    def __init__(self) -> None:
        self.executed: list[AgentRequest] = []

    def execute(self, request: AgentRequest, *, completed_at=None) -> AgentToolResult:
        self.executed.append(request)
        return AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=AgentToolOutcome.OK,
            payload={"ok": True},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )


class ServerEventArmingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _started_server(self, backend) -> AgentSocketServer:
        server = start_agent_server(self.root, backend)
        self.addCleanup(server.close)
        return server

    def test_arming_strips_sidecar_arms_bridge_and_restores(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService(fire_event=_CANCELLED_EVENT)
        server = self._started_server(service)
        result = exchange(
            server,
            make_request("preview_batch", {"queue": "next", "event_socket": str(listener.socket_path)}),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        self.assertIsInstance(service.arms_seen[0], PreviewEventForwarder)
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0], _CANCELLED_EVENT)
        self.assertIsNone(service.preview_event_handler)  # terminal restored
        self.assertEqual(len(server._armed_forwarders), 0)  # and drained the registry

    def test_no_sidecar_no_arm(self) -> None:
        service = EventAwareService()
        server = self._started_server(service)
        result = exchange(server, make_request("preview_batch", {"queue": "next"}))
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        self.assertIsNone(service.arms_seen[0])
        self.assertIsNone(service.preview_event_handler)

    def test_non_preview_tool_never_arms_nor_strips(self) -> None:
        service = EventAwareService()
        server = self._started_server(service)
        result = exchange(server, make_request("play", {"event_socket": "/tmp/x.sock"}))
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {"event_socket": "/tmp/x.sock"})
        self.assertIsNone(service.preview_event_handler)

    def test_unusable_sidecar_strips_without_arm(self) -> None:
        service = EventAwareService()
        server = self._started_server(service)
        result = exchange(server, make_request("preview_batch", {"event_socket": ""}))
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {})  # stripped anyway
        self.assertIsNone(service.preview_event_handler)

    def test_raising_execute_disarms_the_bridge(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService(raise_error=RuntimeError("boom"))
        service.preview_event_handler = "sentinel"
        server = self._started_server(service)
        result = exchange(
            server,
            make_request("preview_batch", {"event_socket": str(listener.socket_path)}),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_message, "boom")
        self.assertEqual(service.preview_event_handler, "sentinel")  # restored
        time.sleep(0.1)
        self.assertEqual(presenter.events, [])

    def test_events_after_the_response_still_bridge(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService()
        server = self._started_server(service)
        result = exchange(
            server,
            make_request("preview_batch", {"event_socket": str(listener.socket_path)}),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        forwarder = service.preview_event_handler
        self.assertIsInstance(forwarder, PreviewEventForwarder)
        # a natural finish fires minutes later on the reaper thread:
        forwarder({"event": "completed", "session": {"total": 3}, "suspended": None})
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0]["event"], "completed")
        self.assertTrue(forwarder.terminal)
        self.assertIsNone(service.preview_event_handler)

    def test_server_close_closes_armed_bridges(self) -> None:
        service = EventAwareService()
        server = self._started_server(service)
        exchange(server, make_request("preview_batch", {"event_socket": "/tmp/x.sock"}))
        forwarder = service.preview_event_handler
        self.assertIsInstance(forwarder, PreviewEventForwarder)
        self.assertFalse(forwarder.terminal)
        server.close()
        self.assertTrue(forwarder.terminal)

    def test_eventless_service_strips_without_arm(self) -> None:
        service = HandlerlessService()
        server = self._started_server(service)
        result = exchange(server, make_request("preview_batch", {"event_socket": "/tmp/x.sock"}))
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {})


class RecordingService:
    """Server-side recorder returning an honest echo envelope."""

    def __init__(self) -> None:
        self.executed: list[AgentRequest] = []

    def execute(self, request: AgentRequest, *, completed_at=None) -> AgentToolResult:
        self.executed.append(request)
        return AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=AgentToolOutcome.OK,
            payload={"echo": dict(request.payload)},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )


def fake_local_service() -> mock.Mock:
    """Autospec instance: passes the client's isinstance boundary, records calls."""
    from music_agent.agent_service import SharedAgentService

    return mock.create_autospec(SharedAgentService, instance=True)


def call_with_pump(pump, fn, timeout: float = 10.0):
    """Run ``fn`` on a helper thread while the main thread pumps ``tick``."""
    holder: dict = {}

    def worker() -> None:
        holder["result"] = fn()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        pump()
        time.sleep(0.02)
    thread.join(timeout=2.0)
    if thread.is_alive() or "result" not in holder:
        raise AssertionError("routed call did not complete under tick pumping")
    return holder["result"]


class RoutedClientEventInjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.agent_socket_path = self.root / "store.db.agent.sock"
        self.recording = RecordingService()
        self.local = fake_local_service()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _started_server(self, backend) -> AgentSocketServer:
        server = start_agent_server(self.root, backend, name="store.db.agent.sock")
        self.addCleanup(server.close)
        return server

    def _client(self, event_socket_path=None, event_listener=None) -> RoutedAgentClient:
        return RoutedAgentClient(
            _CLIENT,
            self.local,
            remote_socket_path=self.agent_socket_path,
            event_socket_path=event_socket_path,
            event_listener=event_listener,
        )

    def test_preview_call_injects_sidecar_and_binds_listener(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService()
        server = self._started_server(service)
        client = self._client(event_socket_path=listener.socket_path, event_listener=listener)
        result = call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        # The server strips the sidecar before execute, so the service never
        # sees the transport key -- the armed bridge is its only trace, and
        # it carries the injected path verbatim.
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        bridge = service.arms_seen[0]
        self.assertIsInstance(bridge, PreviewEventForwarder)
        self.assertEqual(bridge.event_socket_path, listener.socket_path)
        self.assertTrue(listener.socket_path.exists())  # lazily bound
        self.local.execute.assert_not_called()
        self.assertEqual(presenter.events, [])  # no session started: no events

    def test_remote_non_preview_tool_never_injects_or_binds(self) -> None:
        listener, _presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        listener.close()
        self.assertFalse(listener.socket_path.exists())
        server = self._started_server(self.recording)
        client = self._client(event_socket_path=listener.socket_path, event_listener=listener)
        result = call_with_pump(
            server.tick,
            lambda: client.call("get_playback_context", {}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertFalse(listener.socket_path.exists())  # stayed unbound
        self.assertEqual(dict(self.recording.executed[0].payload), {})

    def test_without_event_channel_nothing_injected(self) -> None:
        service = EventAwareService()
        server = self._started_server(service)
        client = self._client()
        result = call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        self.assertIsNone(service.arms_seen[0])  # no sidecar -> no bridge

    def test_local_tool_never_binds_the_listener(self) -> None:
        listener = AgentEventSocketListener(self.root / "never.sock", RecordingPresenter())
        self.local.execute.return_value = AgentToolResult(
            request_id=_REQ_B1,
            tool="pause",
            outcome=AgentToolOutcome.OK,
            payload={"command": "pause"},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )
        client = self._client(event_socket_path=listener.socket_path, event_listener=listener)
        result = client.call("pause", {}, issued_at=_INSTANT)
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.local.execute.assert_called_once()
        self.assertFalse(listener.socket_path.exists())

    def test_bind_failure_degrades_without_sidecar(self) -> None:
        unreachable = self.root / "no" / "such" / "dir" / "events.sock"
        listener = AgentEventSocketListener(unreachable, RecordingPresenter())
        service = EventAwareService()
        server = self._started_server(service)
        client = self._client(event_socket_path=unreachable, event_listener=listener)
        result = call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)  # the call still executes
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        self.assertIsNone(service.arms_seen[0])  # no channel -> no sidecar at all

    def test_close_event_listener_unlinks_and_rebinds(self) -> None:
        listener, _presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService()
        server = self._started_server(service)
        client = self._client(event_socket_path=listener.socket_path, event_listener=listener)
        call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        first_bridge = service.arms_seen[0]
        self.assertIsInstance(first_bridge, PreviewEventForwarder)
        client.close_event_listener()
        self.assertFalse(listener.socket_path.exists())
        client.close_event_listener()  # idempotent
        call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "after"}, issued_at=_INSTANT),
        )
        self.assertTrue(listener.socket_path.exists())  # lazy re-bind
        second_bridge = service.arms_seen[1]
        self.assertIsInstance(second_bridge, PreviewEventForwarder)
        self.assertEqual(second_bridge.event_socket_path, listener.socket_path)

    def test_full_round_trip_chat_run_chat(self) -> None:
        listener, presenter = start_sink(self.root)
        self.addCleanup(listener.close)
        service = EventAwareService(fire_event=_CANCELLED_EVENT)
        server = self._started_server(service)
        client = self._client(event_socket_path=listener.socket_path, event_listener=listener)
        result = call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(service.executed[0].payload), {"queue": "next"})
        self.assertTrue(wait_until(lambda: len(presenter.events) == 1))
        self.assertEqual(presenter.events[0], _CANCELLED_EVENT)


class CliEventWiringTest(unittest.TestCase):
    """The real builder arms the event channel with the pid-stamped default
    and the shared presenter; nothing binds until the first remote preview."""

    def _build(self, db_path: str, agent_service=None) -> RoutedAgentClient:
        from types import SimpleNamespace
        from unittest.mock import patch

        from music_agent.agent_service import SharedAgentService
        from music_agent.cli import _build_chat_session

        class SpyService(SharedAgentService):
            def __init__(self, db, **kwargs):
                pass

            def close(self):
                pass

        args = SimpleNamespace(
            db=db_path,
            agent_client=[f"{_CLIENT_ID}:full"],
            provider="deepseek",
            model=None,
            api_key_env=None,
            base_url=None,
            timeout=120.0,
            max_rounds=8,
            verbose=False,
            agent_service=agent_service,
        )
        with patch("music_agent.agent_service.SharedAgentService", SpyService):
            _service, loop = _build_chat_session(args)
        return loop.client

    def test_event_channel_defaults_to_pid_stamped_path_lazy(self) -> None:
        from music_agent.agent_socket import preview_event_socket_path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "store.db")
            client = self._build(db_path)
        expected = preview_event_socket_path(Path(db_path), os.getpid())
        self.assertEqual(client.event_socket_path, expected)
        self.assertFalse(client.event_socket_path.exists())  # lazy: never bound


if __name__ == "__main__":
    unittest.main()