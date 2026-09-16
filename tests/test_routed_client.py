"""P15-S2-IPC S3: RoutedAgentClient -- routing table, remote execution and the
offline fail-closed surface (decision b + D1/D3/D4).

Two server backends are used over a real UDS in a temp directory:

* a decision-free ``RecordingService`` that proves *which* side executed and
  with what request identity;
* a real :class:`SharedAgentService` built on the deterministic playback
  fakes, proving the full wire + execute chain from the routed client.

The server executes requests on its ``tick()`` (the run main-loop contract),
so every remote call runs on a helper thread while the main thread pumps
``tick()`` -- exactly the shape the real ``run`` loop provides.
"""

import tempfile
import threading
import time
import unittest
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from music_agent.agent_client import AgentClient, AgentClientValidationError
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.agent_socket import (
    AGENT_CLIENT_NOT_REGISTERED_CODE,
    AGENT_RESPONSE_LOST_CODE,
    AGENT_RUNTIME_OFFLINE_CODE,
    AgentSocketServer,
    SocketFrameError,
)
from music_agent.agent_tools import AgentToolName
from music_agent.routed_client import REMOTE_TOOL_NAMES, RoutedAgentClient
from music_agent.track_similarity import SimilarityExecutionContext

_CLIENT_ID = "agt_40000000-0000-4000-8000-000000000004"
_REGISTERED_ELSEWHERE_ID = "agt_90000000-0000-4000-8000-000000000099"
_CLIENT = AgentClientIdentity(client_id=_CLIENT_ID, model_id="routed-test", label="test")
_INSTANT = datetime(2026, 8, 19, 2, 0, tzinfo=timezone.utc)
_REQ_A1 = "req_00000000-0000-4000-8000-0000000000a1"
_REQ_A2 = "req_00000000-0000-4000-8000-0000000000a2"
_REQ_A3 = "req_00000000-0000-4000-8000-0000000000a3"


class RecordingService:
    """Decision-free server side: returns a fixed valid envelope or raises."""

    def __init__(self, raise_error: Exception | None = None) -> None:
        self.executed: list[AgentRequest] = []
        self._raise_error = raise_error

    def execute(self, request: AgentRequest, *, completed_at=None) -> AgentToolResult:
        self.executed.append(request)
        if self._raise_error is not None:
            raise self._raise_error
        return AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=AgentToolOutcome.OK,
            payload={"echo": dict(request.payload)},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )


class FakePlaybackAdapter:
    """Deterministic playback double (same proven shape as the playback-tool
    fakes: every observation returns a stopped player with no track)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.failures: dict[str, Exception] = {}

    def read_player_state(self) -> str:
        return "stopped"

    def read_now_playing(self):
        from music_agent.playback_control import NowPlaying, PlayerState

        return NowPlaying(
            state=PlayerState.STOPPED,
            persistent_id=None,
            name=None,
            artist=None,
            album=None,
        )

    def play(self) -> None:
        self.calls.append(("play", ()))
        self._maybe_fail("play")

    def pause(self) -> None:
        self.calls.append(("pause", ()))
        self._maybe_fail("pause")

    def next_track(self) -> None:
        self.calls.append(("next_track", ()))
        self._maybe_fail("next_track")

    def previous_track(self) -> None:
        self.calls.append(("previous_track", ()))
        self._maybe_fail("previous_track")

    def play_track(self, persistent_id: str) -> None:
        self.calls.append(("play_track", (persistent_id,)))
        self._maybe_fail("play_track")

    def _maybe_fail(self, name: str) -> None:
        failure = self.failures.get(name)
        if failure is not None:
            raise failure


class FakePreviewRunner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stop_calls = 0
        self.active = False
        self.on_natural_finish = None

    def start_audio(self, url: str) -> None:
        self.started.append(url)
        self.active = True

    def stop_preview(self) -> bool:
        self.stop_calls += 1
        was_active = self.active
        self.active = False
        return was_active

    def is_preview_active(self) -> bool:
        return self.active


class FakePlaybackResolver:
    def resolve_playback_track(self, **kwargs):
        return None


def build_service(db_path: Path) -> tuple[SharedAgentService, FakePlaybackAdapter]:
    adapter = FakePlaybackAdapter()
    service = SharedAgentService(
        db_path,
        clients=AgentClientRegistry({_CLIENT_ID: AgentClientPolicy.FULL}),
        playback_adapter=adapter,
        playback_resolver=FakePlaybackResolver(),
        preview_runner=FakePreviewRunner(),
    )
    return service, adapter


def build_unknown_client_service(db_path: Path) -> tuple[SharedAgentService, FakePreviewRunner]:
    """A hosted service whose registry holds a DIFFERENT client id -- the
    routed client ``_CLIENT`` is unknown to it and the permission gate must
    refuse every tool fail-closed (P20-Fix01 not-registered harness)."""
    runner = FakePreviewRunner()
    service = SharedAgentService(
        db_path,
        clients=AgentClientRegistry(
            {_REGISTERED_ELSEWHERE_ID: AgentClientPolicy.FULL}
        ),
        playback_adapter=FakePlaybackAdapter(),
        playback_resolver=FakePlaybackResolver(),
        preview_runner=runner,
    )
    return service, runner


def fake_local_service() -> mock.Mock:
    """Autospec instance: passes the isinstance boundary, records every call."""
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


class RoutingTableTest(unittest.TestCase):
    def test_remote_table_is_exactly_the_approved_family(self) -> None:
        self.assertEqual(
            REMOTE_TOOL_NAMES,
            {
                AgentToolName.PREVIEW_BATCH.value,
                AgentToolName.PREVIEW_CATALOG_TRACK.value,
                AgentToolName.STOP_PREVIEW.value,
                AgentToolName.ADVANCE_PREVIEW.value,
                AgentToolName.GET_PLAYBACK_CONTEXT.value,
                AgentToolName.PLAY.value,
            },
        )

    def test_formal_playback_controls_stay_local(self) -> None:
        for tool in (
            AgentToolName.PAUSE.value,
            AgentToolName.NEXT_TRACK.value,
            AgentToolName.PREVIOUS_TRACK.value,
            AgentToolName.PLAY_TRACK.value,
            AgentToolName.GET_ACTIVE_CONTEXT.value,
            AgentToolName.GET_NOW_PLAYING.value,
        ):
            self.assertNotIn(tool, REMOTE_TOOL_NAMES, tool)


class RemoteExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "store.db.agent.sock"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _started_server(self, backend) -> AgentSocketServer:
        server = AgentSocketServer(self.socket_path, backend)
        server.start()
        self.addCleanup(server.close)
        return server

    def test_remote_tool_executes_on_the_run_server(self) -> None:
        recording = RecordingService()
        server = self._started_server(recording)
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = call_with_pump(
            server.tick,
            lambda: client.call("preview_batch", {"queue": "next"}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(result.payload or {}), {"echo": {"queue": "next"}})
        self.assertEqual(len(recording.executed), 1)
        request = recording.executed[0]
        self.assertEqual(request.tool, "preview_batch")
        self.assertEqual(request.client.client_id, _CLIENT_ID)
        self.assertTrue(request.request_id.startswith("req_"))
        local.execute.assert_not_called()

    def test_get_playback_context_reads_run_state_over_the_wire(self) -> None:
        service, _adapter = build_service(Path(self._tmp.name) / "store.db")
        server = self._started_server(service)
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = call_with_pump(
            server.tick,
            lambda: client.call("get_playback_context", {}, issued_at=_INSTANT),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        payload = result.payload
        self.assertIsInstance(payload, Mapping)
        self.assertIsInstance(payload["channel"], Mapping)
        self.assertIsInstance(payload["preview_sounding"], bool)
        self.assertIsNone(payload["session"])
        self.assertIsNone(payload["suspended"])
        local.execute.assert_not_called()

    def test_play_executes_in_run(self) -> None:
        # D3: the continue/restore terminal runs on the authority side. The
        # fake adapter records the actual play hitting the run-side service.
        service, adapter = build_service(Path(self._tmp.name) / "store.db")
        server = self._started_server(service)
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = call_with_pump(server.tick, lambda: client.call("play", {}, issued_at=_INSTANT))
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["command"], "play")
        self.assertEqual(adapter.calls, [("play", ())])
        local.execute.assert_not_called()

    def test_local_tool_stays_local_even_with_the_server_up(self) -> None:
        self._started_server(RecordingService())
        local = fake_local_service()
        local.execute.return_value = AgentToolResult(
            request_id=_REQ_A2,
            tool="pause",
            outcome=AgentToolOutcome.OK,
            payload={"command": "pause", "ok": True, "elapsed_ms": None},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = client.call("pause", {}, issued_at=_INSTANT)
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        local.execute.assert_called_once()
        self.assertEqual(local.execute.call_args.args[0].tool, "pause")

    def test_each_remote_call_mints_a_fresh_request_id(self) -> None:
        recording = RecordingService()
        server = self._started_server(recording)
        client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
        call_with_pump(server.tick, lambda: client.call("stop_preview", {}, issued_at=_INSTANT))
        call_with_pump(server.tick, lambda: client.call("advance_preview", {}, issued_at=_INSTANT))
        ids = [request.request_id for request in recording.executed]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(i.startswith("req_") for i in ids))


class OfflineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "store.db.agent.sock"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_remote_tool_offline_refuses_honestly(self) -> None:
        client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
        result = client.call(
            "preview_batch", {}, issued_at=_INSTANT, request_id=_REQ_A1
        )
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        self.assertIsNone(result.payload)
        self.assertTrue(result.error_message)
        self.assertEqual(result.request_id, _REQ_A1)

    def test_play_refuses_offline_never_falls_back_local(self) -> None:
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = client.call("play", {}, issued_at=_INSTANT)
        self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        local.execute.assert_not_called()

    def test_get_playback_context_offline_has_no_local_composite(self) -> None:
        # D4: not even the session-absent fallback is fabricated locally --
        # the reading tool refuses; the chat fast path treats it as an
        # unreadable context and fails closed.
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = client.call("get_playback_context", {}, issued_at=_INSTANT)
        self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        local.execute.assert_not_called()

    def test_local_tools_keep_working_while_offline(self) -> None:
        local = fake_local_service()
        local.execute.return_value = AgentToolResult(
            request_id=_REQ_A3,
            tool="pause",
            outcome=AgentToolOutcome.OK,
            payload={"command": "pause", "ok": True, "elapsed_ms": None},
            error_code=None,
            error_message=None,
            completed_at=_INSTANT,
        )
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = client.call("pause", {}, issued_at=_INSTANT)
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        local.execute.assert_called_once()

    def test_stale_regular_file_at_the_socket_path_stays_a_refusal(self) -> None:
        self.socket_path.write_text("stale leftover", encoding="utf-8")
        client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
        result = client.call("stop_preview", {}, issued_at=_INSTANT)
        self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)


class ClientNotRegisteredTest(unittest.TestCase):
    """P20-Fix01: a reachable run whose permission gate refuses the session
    client (its id is absent from run's own registrations) surfaces the
    stable natural not-registered envelope -- the gate itself stays
    fail-closed, nothing executes, and the internal ``unknown_client``
    registry sentence never reaches the caller."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "store.db.agent.sock"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _started_server(self, backend) -> AgentSocketServer:
        server = AgentSocketServer(self.socket_path, backend)
        server.start()
        self.addCleanup(server.close)
        return server

    def _unknown_client_service(self) -> tuple[SharedAgentService, FakePreviewRunner]:
        return build_unknown_client_service(Path(self._tmp.name) / "store.db")

    def test_remote_read_refusal_translates_to_the_stable_envelope(self) -> None:
        service, _runner = self._unknown_client_service()
        server = self._started_server(service)
        local = fake_local_service()
        client = RoutedAgentClient(_CLIENT, local, remote_socket_path=self.socket_path)
        result = call_with_pump(
            server.tick,
            lambda: client.call(
                "get_playback_context", {}, issued_at=_INSTANT, request_id=_REQ_A1
            ),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, AGENT_CLIENT_NOT_REGISTERED_CODE)
        self.assertIsNone(result.payload)  # nothing executed, nothing fabricated locally
        self.assertIn("后台运行时未注册本会话的客户端身份", result.error_message or "")
        self.assertNotIn("unknown_client", result.error_message or "")
        self.assertNotIn(
            "not registered with the shared agent service", result.error_message or ""
        )
        self.assertEqual(result.request_id, _REQ_A1)  # envelope fields are preserved
        local.execute.assert_not_called()

    def test_preview_mutate_gate_refuses_without_any_execution(self) -> None:
        service, runner = self._unknown_client_service()
        server = self._started_server(service)
        client = RoutedAgentClient(
            _CLIENT, fake_local_service(), remote_socket_path=self.socket_path
        )
        result = call_with_pump(
            server.tick,
            lambda: client.call(
                "preview_catalog_track",
                {"canonical_id": "trk_40000000-0000-4000-8000-000000000004"},
                issued_at=_INSTANT,
            ),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, AGENT_CLIENT_NOT_REGISTERED_CODE)
        self.assertIsNone(result.payload)
        self.assertEqual(runner.started, [])  # refused BEFORE the preview started
        self.assertEqual(runner.stop_calls, 0)

    def test_service_gate_still_refuses_the_unknown_client_fail_closed(self) -> None:
        # The run-side permission gate is untouched: the same request via the
        # service boundary directly still lands in the UNKNOWN_CLIENT outcome --
        # the translation is a chat-boundary surface, never a permission change.
        service, runner = self._unknown_client_service()
        direct = service.execute(
            AgentRequest(
                request_id=_REQ_A3,
                client=_CLIENT,
                tool="preview_catalog_track",
                payload={"canonical_id": "trk_40000000-0000-4000-8000-000000000004"},
                issued_at=_INSTANT,
            )
        )
        self.assertEqual(direct.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        self.assertEqual(runner.started, [])
        self.assertEqual(runner.stop_calls, 0)

    def test_not_registered_stays_distinct_from_offline(self) -> None:
        # Offline = the socket is unreachable and no gate ever runs (existing
        # D4 behavior). Not-registered = the socket served the request AND the
        # gate refused it. The two stable codes never collapse into one.
        self.assertNotEqual(
            AGENT_CLIENT_NOT_REGISTERED_CODE, AGENT_RUNTIME_OFFLINE_CODE
        )
        client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
        offline = client.call(
            "get_playback_context", {}, issued_at=_INSTANT, request_id=_REQ_A2
        )
        self.assertEqual(offline.error_code, AGENT_RUNTIME_OFFLINE_CODE)


class ResponseLostTest(unittest.TestCase):
    """S3 repair (live Broken-pipe diagnosis): a reachable run whose reply
    never arrives is an unknown-outcome failure, not an offline fact."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "store.db.agent.sock"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_silent_server_times_out_to_a_response_lost_failure(self) -> None:
        server = AgentSocketServer(self.socket_path, RecordingService())
        server.start()
        try:
            client = RoutedAgentClient(
                _CLIENT,
                fake_local_service(),
                remote_socket_path=self.socket_path,
                timeout=0.5,
            )
            # The accept thread reads the request but no tick() ever executes
            # it -- the client's socket read times out. The request was
            # delivered: the outcome is unknown, never agent_runtime_offline.
            result = client.call("stop_preview", {}, issued_at=_INSTANT, request_id=_REQ_A2)
            self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
            self.assertEqual(result.error_code, AGENT_RESPONSE_LOST_CODE)
            self.assertIsNone(result.payload)
            self.assertTrue(result.error_message)
            self.assertEqual(result.request_id, _REQ_A2)
        finally:
            server.close()

    def test_garbage_reply_is_response_lost_not_offline(self) -> None:
        server = AgentSocketServer(self.socket_path, RecordingService())
        server.start()
        try:
            client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
            with mock.patch(
                "music_agent.routed_client.receive_framed",
                side_effect=SocketFrameError("response frame corrupt"),
            ):
                result = client.call("stop_preview", {}, issued_at=_INSTANT)
            self.assertEqual(result.error_code, AGENT_RESPONSE_LOST_CODE)
        finally:
            server.close()

    def test_undeliverable_request_stays_an_offline_refusal(self) -> None:
        # The runtime went away between accept and execute: the request frame
        # never left the wire -- the honest answer remains agent_runtime_offline.
        server = AgentSocketServer(self.socket_path, RecordingService())
        server.start()
        try:
            client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=self.socket_path)
            with mock.patch(
                "music_agent.routed_client.send_framed",
                side_effect=OSError("broken pipe"),
            ):
                result = client.call("stop_preview", {}, issued_at=_INSTANT)
            self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        finally:
            server.close()

    def test_response_lost_sends_exactly_once(self) -> None:
        """G: a lost response never triggers a transport retry -- the request
        frame goes out exactly once, so a duplicated side effect is impossible
        by construction (a second send would re-run the tool)."""
        import music_agent.routed_client as routed_module

        server = AgentSocketServer(self.socket_path, RecordingService())
        server.start()
        try:
            client = RoutedAgentClient(
                _CLIENT,
                fake_local_service(),
                remote_socket_path=self.socket_path,
                timeout=0.5,
            )
            sent: list[int] = []
            real_send = routed_module.send_framed

            def counting_send(sock, frame) -> None:
                sent.append(1)
                return real_send(sock, frame)

            with mock.patch(
                "music_agent.routed_client.send_framed", side_effect=counting_send
            ):
                result = client.call(
                    "stop_preview", {}, issued_at=_INSTANT, request_id=_REQ_A3
                )
            self.assertEqual(result.error_code, AGENT_RESPONSE_LOST_CODE)
            self.assertEqual(len(sent), 1)
        finally:
            server.close()


class ValidationTest(unittest.TestCase):
    def test_construction_validates(self) -> None:
        good_local = fake_local_service()
        with self.assertRaises(AgentClientValidationError):
            RoutedAgentClient("not-an-identity", good_local, remote_socket_path=Path("x.sock"))  # type: ignore[arg-type]
        with self.assertRaises(AgentClientValidationError):
            RoutedAgentClient(_CLIENT, object(), remote_socket_path=Path("x.sock"))  # type: ignore[arg-type]
        with self.assertRaises(AgentClientValidationError):
            RoutedAgentClient(_CLIENT, good_local, remote_socket_path=Path(""))
        with self.assertRaises(AgentClientValidationError):
            RoutedAgentClient(_CLIENT, good_local, remote_socket_path=Path("x.sock"), timeout=0)

    def test_call_validation_mirrors_the_plain_client(self) -> None:
        client = RoutedAgentClient(_CLIENT, fake_local_service(), remote_socket_path=Path("x.sock"))
        with self.assertRaises(AgentClientValidationError):
            client.call("", {})
        with self.assertRaises(AgentClientValidationError):
            client.call("pause", [])  # type: ignore[arg-type]


class ChatSessionWiringTest(unittest.TestCase):
    """The real builder wires the routed client with the derived default."""

    def _build(self, agent_service=None) -> object:
        from types import SimpleNamespace
        from unittest.mock import patch

        from music_agent.cli import _build_chat_session

        class SpyService(SharedAgentService):
            def __init__(self, db, **kwargs):
                pass

            def close(self):
                pass

        args = SimpleNamespace(
            db="/tmp/chat/store.db",
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

    def test_default_socket_derives_from_db(self) -> None:
        client = self._build()
        self.assertIsInstance(client, RoutedAgentClient)
        self.assertEqual(client.remote_socket_path, Path("/tmp/chat/store.db.agent.sock"))

    def test_agent_service_flag_overrides_the_default(self) -> None:
        client = self._build(agent_service="/tmp/custom/run.sock")
        self.assertEqual(client.remote_socket_path, Path("/tmp/custom/run.sock"))

    def test_chat_client_wires_the_extended_request_timeout(self) -> None:
        # S3 repair: the routed exchange must outlast the slowest honest
        # execute (preview_batch starts its first clip synchronously, incl.
        # the bounded clip download) -- the cli builder relies on the default.
        client = self._build()
        self.assertEqual(client._timeout, 120.0)


class RoutedClientInternalContextTest(unittest.TestCase):
    """P15-S3-S3D/E wiring fix: the real chat session binds the provider loop
    to RoutedAgentClient, so the internal Fresh context must survive THIS
    override -- never swallowed, never aimed at the wire. Plain calls stay
    byte-for-byte the base client; local delegation forwards the set verbatim;
    a remote call carrying a truthy set is refused fail-closed before any
    socket dial; generation tools are pinned as local-only routing entries
    (the live TypeError's exact path)."""

    def _local_client(self, local) -> RoutedAgentClient:
        return RoutedAgentClient(_CLIENT, local, remote_socket_path=Path("unused.sock"))

    def test_plain_call_is_byte_for_byte_the_base_client(self) -> None:
        # Same identity, same injected request_id/issued_at: the routed local
        # path must produce the IDENTICAL request + execute kwargs as the
        # plain client (internal kwarg absent/None, nothing invented).
        plain_local = fake_local_service()
        routed_local = fake_local_service()
        plain = AgentClient(_CLIENT, plain_local)
        routed = self._local_client(routed_local)
        payload = {"command": "pause"}
        plain.call("pause", payload, request_id=_REQ_A1, issued_at=_INSTANT)
        routed.call("pause", payload, request_id=_REQ_A1, issued_at=_INSTANT)
        self.assertEqual(routed_local.execute.call_args, plain_local.execute.call_args)
        self.assertIsNone(routed_local.execute.call_args.kwargs["fresh_canonical_ids"])
        self.assertIsNone(routed_local.execute.call_args.kwargs["recommendation_scope_ids"])

    def test_local_delegate_forwards_fresh_ids_verbatim(self) -> None:
        local = fake_local_service()
        client = self._local_client(local)
        client.call(
            "generate_inferred_recommendation",
            {"limit": 5},
            fresh_canonical_ids=("trk_a", "trk_b"),
        )
        execute_kwargs = local.execute.call_args.kwargs
        self.assertEqual(execute_kwargs["fresh_canonical_ids"], ("trk_a", "trk_b"))
        # Internal context travels OUTSIDE payload: the journaled request
        # payload carries no such key, ever.
        request = local.execute.call_args.args[0]
        self.assertNotIn("fresh_canonical_ids", request.payload)

    def test_truthy_fresh_ids_on_a_remote_tool_refuse_fail_closed(self) -> None:
        # A truthy set aimed at the remote family would need a wire slot that
        # does not exist: refused before any socket activity, and never
        # laundered through the local service either.
        local = fake_local_service()
        client = self._local_client(local)
        with self.assertRaises(AgentClientValidationError):
            client.call("play", {}, fresh_canonical_ids=("trk_a",))
        local.execute.assert_not_called()

    def test_artist_scope_stays_local_and_outside_request_payload(self) -> None:
        local = fake_local_service()
        client = self._local_client(local)
        client.call(
            "generate_recommendation",
            {"target_ids": ["trk_a"], "limit": 5},
            recommendation_scope_ids=("trk_a",),
        )
        kwargs = local.execute.call_args.kwargs
        self.assertEqual(kwargs["recommendation_scope_ids"], ("trk_a",))
        request = local.execute.call_args.args[0]
        self.assertNotIn("recommendation_scope_ids", request.payload)
        local.reset_mock()
        with self.assertRaises(AgentClientValidationError):
            client.call("play", {}, recommendation_scope_ids=("trk_a",))
        local.execute.assert_not_called()

    def test_similarity_context_stays_local_and_outside_provider_payload(self) -> None:
        local = fake_local_service()
        client = self._local_client(local)
        seed = SimilarityExecutionContext(
            "trk_11111111-1111-4111-8111-111111111111"
        )
        client.call(
            "generate_inferred_recommendation",
            {"target_ids": [seed.seed_canonical_id], "limit": 5},
            similarity_context=seed,
        )
        kwargs = local.execute.call_args.kwargs
        self.assertEqual(kwargs["similarity_context"], seed)
        request = local.execute.call_args.args[0]
        self.assertNotIn("similarity_context", request.payload)
        local.reset_mock()
        with self.assertRaises(AgentClientValidationError):
            client.call("play", {}, similarity_context=seed)
        local.execute.assert_not_called()

    def test_falsy_fresh_ids_leave_remote_calls_unchanged(self) -> None:
        # () and None are the ordinary non-Fresh cases: remote behavior is
        # byte-identical to before the fix (offline refusal on this fixture).
        local = fake_local_service()
        client = self._local_client(local)
        for falsy in ((), None):
            with self.subTest(falsy=falsy):
                result = client.call(
                    "play", {}, issued_at=_INSTANT, fresh_canonical_ids=falsy
                )
                self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
                self.assertEqual(result.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        local.execute.assert_not_called()

    def test_generation_tools_are_local_only_routing_entries(self) -> None:
        from music_agent.provider_agent import _GENERATION_TOOL_NAMES

        # The pin that makes the whole fix a local-delegate exercise: no
        # generation tool can ever be routed remotely, so the Fresh set can
        # never need a wire representation.
        self.assertFalse(REMOTE_TOOL_NAMES & _GENERATION_TOOL_NAMES)
        local = fake_local_service()
        client = self._local_client(local)
        for tool in ("generate_recommendation", "generate_inferred_recommendation"):
            with self.subTest(tool=tool):
                local.reset_mock()
                client.call(tool, {}, fresh_canonical_ids=("trk_z",))
                local.execute.assert_called_once()
                self.assertEqual(
                    local.execute.call_args.kwargs["fresh_canonical_ids"], ("trk_z",)
                )


if __name__ == "__main__":
    unittest.main()
