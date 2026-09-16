"""P15-S2-IPC S2: agent-socket host tests (server + runtime wiring).

The server is exercised over a real UDS in a temp directory. Execution is
tick-drained: the accept thread only receives/decodes/waits, and every request
is executed by ``tick()`` on the thread that started the server -- the same
contract the ``run`` main loop provides, and the reason the store's
single-thread SQLite connections are never entered from another thread.
The tests therefore pump ``tick()`` on the main thread while the socket
exchange runs on a helper thread, mirroring the real run loop.
"""

import os
import socket
import stat
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_socket import (
    AgentSocketError,
    AgentSocketServer,
    agent_socket_path,
    frame_to_result,
    receive_framed,
    request_to_frame,
    send_framed,
)

_REQUEST_ID = "req_00000000-0000-4000-8000-000000000002"
_CLIENT_ID = "agt_20000000-0000-4000-8000-000000000002"
_INSTANT = datetime(2026, 8, 19, 1, 0, tzinfo=timezone.utc)


def example_request(tool: str = "get_playback_context") -> AgentRequest:
    return AgentRequest(
        request_id=_REQUEST_ID,
        client=AgentClientIdentity(client_id=_CLIENT_ID, model_id="socket-test"),
        tool=tool,
        payload={},
        issued_at=_INSTANT,
    )


class StubService:
    """Decision-free stand-in: returns a fixed valid envelope or raises."""

    def __init__(self, *, results: dict[str, AgentToolResult] | None = None,
                 raise_error: Exception | None = None) -> None:
        self.executed: list[AgentRequest] = []
        self._results = results or {}
        self._raise_error = raise_error

    def execute(self, request: AgentRequest, *, completed_at: str | None = None) -> AgentToolResult:
        self.executed.append(request)
        if self._raise_error is not None:
            raise self._raise_error
        result = self._results.get(request.tool)
        if result is not None:
            return result
        return AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=AgentToolOutcome.OK,
            payload={"echo": dict(request.payload)},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )


def connect_client(socket_path: Path, timeout: float = 5.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(socket_path))
            return client
        except OSError:
            client.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def raw_roundtrip(socket_path: Path, request: AgentRequest) -> AgentToolResult:
    """One socket exchange WITHOUT pumping tick (helper-thread side only)."""
    with connect_client(socket_path) as client:
        send_framed(client, request_to_frame(request))
        return frame_to_result(receive_framed(client))


def roundtrip(socket_path: Path, request: AgentRequest, pump, timeout: float = 10.0) -> AgentToolResult:
    """Socket exchange on a helper thread; the main thread pumps ``tick()``."""
    holder: dict[str, AgentToolResult] = {}

    def worker() -> None:
        holder["result"] = raw_roundtrip(socket_path, request)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        pump()
        time.sleep(0.02)
    thread.join(timeout=2.0)
    if thread.is_alive() or "result" not in holder:
        raise AssertionError("routed roundtrip did not complete under tick pumping")
    return holder["result"]


class AgentSocketServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "store.db.agent.sock"
        self.service = StubService()
        self.server = AgentSocketServer(self.socket_path, self.service)

    def tearDown(self) -> None:
        self.server.close()
        self._tmp.cleanup()

    def test_roundtrip_executes_request_on_the_tick_thread(self) -> None:
        self.server.start()
        try:
            result = roundtrip(
                self.socket_path, example_request("stop_preview"), self.server.tick
            )
        finally:
            self.server.close()
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.tool, "stop_preview")
        self.assertEqual(dict(result.payload or {}), {"echo": {}})
        self.assertEqual(len(self.service.executed), 1)
        self.assertEqual(self.service.executed[0].tool, "stop_preview")

    def test_execute_exception_becomes_execution_error_envelope(self) -> None:
        self.service._raise_error = ValueError("osascript blew up")
        self.server.start()
        try:
            result = roundtrip(self.socket_path, example_request(), self.server.tick)
        finally:
            self.server.close()
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "execution_error")
        self.assertEqual(result.error_message, "osascript blew up")
        self.assertIsNone(result.payload)

    def test_garbage_frame_closes_connection_and_server_survives(self) -> None:
        self.server.start()
        try:
            with connect_client(self.socket_path) as client:
                client.sendall(b"garbage!!")
                client.shutdown(socket.SHUT_WR)
            result = roundtrip(
                self.socket_path, example_request("advance_preview"), self.server.tick
            )
        finally:
            self.server.close()
        self.assertEqual(result.outcome, AgentToolOutcome.OK)

    def test_start_binds_private_socket_and_close_unlinks(self) -> None:
        self.server.start()
        self.assertTrue(self.socket_path.exists())
        mode = stat.S_IMODE(os.stat(self.socket_path).st_mode)
        self.assertEqual(mode, 0o600)
        self.server.close()
        self.assertFalse(self.socket_path.exists())

    def test_live_conflict_is_refused_and_left_untouched(self) -> None:
        occupier = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        occupier.bind(str(self.socket_path))
        occupier.listen(1)
        try:
            with self.assertRaises(AgentSocketError):
                self.server.start()
            self.server.close()  # must NOT unlink the occupier's socket
            self.assertTrue(self.socket_path.exists())
        finally:
            occupier.close()
            self.socket_path.unlink(missing_ok=True)

    def test_stale_regular_file_at_path_is_recovered(self) -> None:
        self.socket_path.write_text("leftover from a crashed process", encoding="utf-8")
        self.server.start()
        try:
            result = roundtrip(
                self.socket_path, example_request("preview_batch"), self.server.tick
            )
        finally:
            self.server.close()
        self.assertEqual(result.outcome, AgentToolOutcome.OK)

    def test_closing_while_a_request_waits_drops_it_cleanly(self) -> None:
        # The stall holder keeps the first roundtrip pending; close() must
        # unblock the handler's wait without corrupting the server.
        self.server.start()
        try:
            with connect_client(self.socket_path) as client:
                send_framed(client, request_to_frame(example_request("preview_batch")))
            self.server.close()
        finally:
            self.server.close()

    def test_closed_client_request_is_abandoned_before_execute(self) -> None:
        # D: a request whose client has gone away is dropped before the
        # service boundary -- zero executes, no side effects, no retry.
        self.server.start()
        try:
            client = connect_client(self.socket_path)
            send_framed(client, request_to_frame(example_request("play")))
            client.close()  # the client is gone while the request is still queued
            pending = self.server._pending
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pending.empty():
                time.sleep(0.02)
            self.assertFalse(pending.empty(), "request never reached the queue")
            self.server.tick()
            self.assertEqual(self.service.executed, [])
            self.assertTrue(pending.empty())  # drained by abandonment, not execution
        finally:
            self.server.close()

    def test_abandoned_preview_batch_has_no_side_effects(self) -> None:
        # E: an abandoned preview_batch never reaches the service, so no
        # PreviewSession or preview spawn can be created -- execute is the
        # only side-effect gateway.
        self.server.start()
        try:
            client = connect_client(self.socket_path)
            send_framed(client, request_to_frame(example_request("preview_batch")))
            client.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and self.server._pending.empty():
                time.sleep(0.02)
            self.assertFalse(self.server._pending.empty())
            self.server.tick()
            self.assertEqual(self.service.executed, [])
        finally:
            self.server.close()

    def test_waiting_client_is_not_abandoned_across_ticks(self) -> None:
        # F: a silent but connected client survives many probe windows and
        # its request executes normally with a response.
        self.server.start()
        try:
            holder: dict[str, AgentToolResult] = {}

            def worker() -> None:
                with connect_client(self.socket_path) as client:
                    send_framed(client, request_to_frame(example_request("preview_batch")))
                    time.sleep(0.4)  # silent across several tick/probe windows
                    holder["result"] = frame_to_result(receive_framed(client))

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            deadline = time.monotonic() + 10.0
            while thread.is_alive() and time.monotonic() < deadline:
                self.server.tick()
                time.sleep(0.02)
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive(), "roundtrip did not complete")
            self.assertEqual(len(self.service.executed), 1)
            self.assertEqual(self.service.executed[0].tool, "preview_batch")
            self.assertEqual(holder["result"].outcome, AgentToolOutcome.OK)
        finally:
            self.server.close()

    def test_construction_validates_path_and_service(self) -> None:
        with self.assertRaises(AgentSocketError):
            AgentSocketServer(Path(""), self.service)
        with self.assertRaises(AgentSocketError):
            AgentSocketServer(Path("x.sock"), object())  # type: ignore[arg-type]


class RuntimeAgentSocketWiringTest(unittest.TestCase):
    def _start_runtime(self, *, enabled: bool = True) -> tuple[object, Path]:
        from music_agent.runtime import Runtime, RuntimeConfig

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        database_path = Path(tmp.name) / "store.db"
        runtime = Runtime(
            RuntimeConfig(
                database_path=database_path,
                audio_safety_enabled=False,
                agent_socket_enabled=enabled,
                agent_clients={},  # every client unknown: fail closed
            )
        )
        runtime.start()
        return runtime, agent_socket_path(database_path)

    def _routed(self, runtime: object, socket_path: Path, request: AgentRequest) -> AgentToolResult:
        return roundtrip(socket_path, request, runtime.agent_socket.tick)

    def test_runtime_hosts_socket_and_serves_the_real_service(self) -> None:
        runtime, socket_path = self._start_runtime()
        try:
            self.assertIsNotNone(runtime.agent_socket)
            result = self._routed(runtime, socket_path, example_request("not_a_tool"))
            self.assertEqual(result.outcome, AgentToolOutcome.TOOL_NOT_SUPPORTED)
            self.assertEqual(result.error_code, "tool_not_supported")
        finally:
            runtime.close()
        self.assertFalse(socket_path.exists())

    def test_runtime_refuses_unknown_client_fail_closed(self) -> None:
        runtime, socket_path = self._start_runtime()
        try:
            unknown = AgentRequest(
                request_id="req_00000000-0000-4000-8000-000000000003",
                client=AgentClientIdentity(
                    client_id="agt_30000000-0000-4000-8000-000000000003",
                    model_id="stranger",
                ),
                tool="get_playback_context",
                payload={},
                issued_at=_INSTANT,
            )
            result = self._routed(runtime, socket_path, unknown)
            self.assertEqual(result.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        finally:
            runtime.close()

    def test_replayed_request_served_from_the_durable_journal(self) -> None:
        # The same request id + payload twice: the second roundtrip returns the
        # journaled result with replayed=True -- the execute boundary is intact.
        runtime, socket_path = self._start_runtime()
        try:
            first = self._routed(runtime, socket_path, example_request("not_a_tool"))
            self.assertFalse(first.replayed)
            second = self._routed(runtime, socket_path, example_request("not_a_tool"))
            self.assertTrue(second.replayed)
            self.assertEqual(second.outcome, first.outcome)
        finally:
            runtime.close()

    def test_disabled_config_skips_the_socket(self) -> None:
        runtime, socket_path = self._start_runtime(enabled=False)
        try:
            self.assertIsNone(runtime.agent_socket)
            self.assertFalse(socket_path.exists())
        finally:
            runtime.close()

    def test_agent_socket_is_registered_before_automation(self) -> None:
        # A: the tick loop visits the agent-socket dispatcher before the
        # automation engine, so queued routed requests never wait behind a
        # long synchronous automation task start.
        runtime, _ = self._start_runtime()
        try:
            names = [name for name, _ in runtime._components]  # type: ignore[attr-defined]
            self.assertLess(names.index("agent_socket"), names.index("automation"))
        finally:
            runtime.close()

    def test_config_validates_agent_socket_field(self) -> None:
        from pathlib import Path as _Path

        from music_agent.runtime import RuntimeConfig, RuntimeStartupError

        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=_Path("store.db"), agent_socket_enabled="yes")  # type: ignore[arg-type]


class DispatchOrderingIntegrationTest(unittest.TestCase):
    """C: with the real ``Runtime.run`` tick loop, a routed request that
    arrives while a slow automation task occupies the main thread dispatches
    as soon as that task finishes -- BEFORE the next due automation task."""

    def test_pending_ipc_dispatches_between_automation_tasks(self) -> None:
        from music_agent.runtime import Runtime, RuntimeConfig
        from music_agent.runtime_automation import AutomationEngine, AutomationTask

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        database_path = Path(tmp.name) / "store.db"

        order: list[str] = []
        service = StubService()
        base_execute = service.execute

        def tracked_execute(request: AgentRequest, *, completed_at=None) -> AgentToolResult:
            order.append(f"execute:{request.tool}")
            return base_execute(request, completed_at=completed_at)

        service.execute = tracked_execute  # type: ignore[method-assign]

        def slow_task() -> dict[str, object]:
            order.append("slow:start")
            time.sleep(0.6)
            order.append("slow:end")
            return {"task": "slow"}

        engine = AutomationEngine(database_path)
        engine.register(AutomationTask("slow", 900, slow_task))
        engine.register(
            AutomationTask("next", 900, lambda: order.append("next") or {"task": "next"})
        )
        server = AgentSocketServer(agent_socket_path(database_path), service)

        runtime = Runtime(
            RuntimeConfig(database_path=database_path, audio_safety_enabled=False)
        )
        runtime._build_automation_engine = lambda: engine  # type: ignore[method-assign]
        runtime._build_agent_socket_server = lambda: server  # type: ignore[method-assign]
        runtime.start()
        stop_event = threading.Event()
        loop = threading.Thread(target=runtime.run, args=(stop_event,), daemon=True)
        loop.start()
        try:
            # Wait until the slow task is actually occupying the main thread,
            # then deliver the request mid-task (like a real user).
            deadline = time.monotonic() + 5.0
            while "slow:start" not in order and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn("slow:start", order)
            result = raw_roundtrip(
                agent_socket_path(database_path), example_request("play")
            )
            self.assertEqual(result.outcome, AgentToolOutcome.OK)
            # Let the loop settle so the next automation task gets its chance
            # (under the old chained tick it would have run before execute).
            deadline = time.monotonic() + 5.0
            while "next" not in order and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            stop_event.set()
            loop.join(timeout=5.0)
            runtime.close()
        self.assertIn("next", order)
        self.assertIn("execute:play", order)
        self.assertLess(order.index("slow:end"), order.index("execute:play"))
        self.assertLess(order.index("execute:play"), order.index("next"))

    def test_pending_ipc_dispatches_between_task_steps(self) -> None:
        """C13: a multi-step automation task that advances one bounded step per
        tick yields the main loop between steps -- a routed request dispatching
        mid-run executes AFTER the current step and BEFORE the next step, not
        after the whole task drains."""
        from music_agent.runtime import Runtime, RuntimeConfig
        from music_agent.runtime_automation import AutomationEngine, AutomationTask

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        database_path = Path(tmp.name) / "store.db"

        order: list[str] = []
        service = StubService()
        base_execute = service.execute

        def tracked_execute(request: AgentRequest, *, completed_at=None) -> AgentToolResult:
            order.append(f"execute:{request.tool}")
            return base_execute(request, completed_at=completed_at)

        service.execute = tracked_execute  # type: ignore[method-assign]

        def chunked_task():
            order.append("step1:start")
            time.sleep(0.6)
            order.append("step1:end")
            yield {"step": 1}
            order.append("step2:start")
            time.sleep(0.6)
            order.append("step2:end")
            yield {"step": 2}
            return {"task": "chunked"}

        engine = AutomationEngine(database_path)
        engine.register(AutomationTask("chunked", 900, chunked_task))
        server = AgentSocketServer(agent_socket_path(database_path), service)

        runtime = Runtime(
            RuntimeConfig(database_path=database_path, audio_safety_enabled=False)
        )
        runtime._build_automation_engine = lambda: engine  # type: ignore[method-assign]
        runtime._build_agent_socket_server = lambda: server  # type: ignore[method-assign]
        runtime.start()
        stop_event = threading.Event()
        loop = threading.Thread(target=runtime.run, args=(stop_event,), daemon=True)
        loop.start()
        try:
            # Wait until step 1 of the chunked task is occupying the main
            # thread, then deliver the request mid-step (like a real user).
            deadline = time.monotonic() + 5.0
            while "step1:start" not in order and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn("step1:start", order)
            result = raw_roundtrip(
                agent_socket_path(database_path), example_request("play")
            )
            self.assertEqual(result.outcome, AgentToolOutcome.OK)
            # Let the loop settle so step 2 gets its chance.
            deadline = time.monotonic() + 5.0
            while "step2:start" not in order and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            stop_event.set()
            loop.join(timeout=5.0)
            runtime.close()
        # The response landed between steps of the SAME run: the main loop was
        # never blocked for the whole task.
        self.assertIn("execute:play", order)
        self.assertLess(order.index("step1:end"), order.index("execute:play"))
        self.assertLess(order.index("execute:play"), order.index("step2:start"))


if __name__ == "__main__":
    unittest.main()