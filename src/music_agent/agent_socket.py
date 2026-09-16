"""P15-S2-IPC S1: agent-socket transport primitives (framing + canonical interchange).

The long-run runtime boundary (design ``P15-S2-IPC-DESIGN.md``): one Unix-domain
socket host inside ``run`` executes routed tool calls for chat-session clients.
This module is the transport layer *below* the decision layer -- it moves the
canonical agent interchange across one socket connection.

- **No envelope of its own.** Frames carry ``encode_agent_request`` /
  ``encode_agent_tool_result`` texts verbatim, so the P09 provider boundary
  (validation, outcome vocabulary, ``contract_version``, replay shape) is the
  wire contract too. The S3 routed client decides *which tool* travels; here
  nothing routes, nothing is permitted or refused.
- **Framing.** 4-byte big-endian length prefix + UTF-8 body. An oversized
  length is refused *before* the body is read; a zero-length body is refused
  outright. Reads loop over ``recv``, so partial deliveries assemble correctly.
- **Fail closed.** Truncated headers/bodies, frames over ``MAX_FRAME_BYTES``,
  invalid UTF-8, and undecodable bodies raise ``SocketFrameError`` / contract
  validation errors -- nothing is ever coerced or guessed.
- **No state, no policy.** No socket registry, no routing table, no device
  knowledge. The ``agent_runtime_offline`` refusal (decision b) is the S3
  client's semantics and appears here only as a result-construction helper.
- **Paths.** Sockets live next to the store (``<store>.agent.sock``), mirroring
  the P10.6 runtime process-state file; the S4 preview-event channel lives at
  ``<store>.agent-events-<pid>.sock`` (chat-session owned).

The S4 preview-event return channel reuses this framing byte-for-byte; its
frames carry ``{event, session, suspended}`` JSON texts instead of the agent
interchange, with the same length-capped, fail-closed transport.

``AgentSocketServer`` (S2) is the run-side host: it owns the listener thread
and the socket file, decodes every frame through the canonical interchange,
and executes each request through the service's existing ``execute`` boundary.
It makes no permission or policy decision of its own -- a live existing
listener is refused as the only honest reading of "already hosted", and a
stale socket file from a crashed process is recovered (never guessed around).
For preview-family requests it additionally strips the transport sidecar
``event_socket`` from the payload (S4) and arms a :class:`PreviewEventForwarder`
on the service's preview-event presenter slot, so session events stream back
to the requesting chat-session until a terminal state. Request execution is
tick-drained from the runtime main loop; a queued request whose client has
already gone away is abandoned before ``execute`` (no side effects, no retry),
while a response lost after execute keeps the client's ``agent_response_lost``
unknown-outcome semantics.

``AgentEventSocketListener`` (S4) is the chat-session-side sink: it owns
``<store>.agent-events-<pid>.sock`` and hands each decoded event mapping to a
presenter -- no responses, no state, nothing executed.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import queue
import select
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from music_agent.agent_contract import (
    AgentContractValidationError,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
    decode_agent_request,
    decode_agent_tool_result,
    encode_agent_request,
    encode_agent_tool_result,
)
from music_agent.agent_tools import AgentToolName

logger = logging.getLogger("music_agent.agent_socket")

FRAME_LENGTH_BYTES = 4
MAX_FRAME_BYTES = 4 * 1024 * 1024  # bounded before read: tool payloads are small objects

AGENT_SOCKET_SUFFIX = ".agent.sock"
EVENT_SOCKET_STEM = ".agent-events"

AGENT_RUNTIME_OFFLINE_CODE = "agent_runtime_offline"
AGENT_RUNTIME_OFFLINE_HINT = (
    "试听与设备安全由后台 runtime 承载：请先运行 music-agent run --db <库路径>"
)

# P20-Fix01: a runtime that IS reachable but does not register the calling
# client is a distinct surface from an offline runtime -- the run-side
# permission gate still refuses (fail closed, unchanged), and the routed
# client translates that refusal into this stable, natural envelope so the
# internal `unknown_client` registry sentence never reaches a user. The chat
# playback-status fast path uses the code to pick its own honest degrade note.
AGENT_CLIENT_NOT_REGISTERED_CODE = "agent_client_not_registered"
AGENT_CLIENT_NOT_REGISTERED_HINT = (
    "后台运行时未注册本会话的客户端身份：试听与播放上下文等远程功能暂不可用"
    "（本地功能不受影响）。请用与 run 一致的 --agent-client 启动本会话，"
    "或在 run 的启动参数中登记同一客户端身份后重试。"
)

# S3 repair (live Broken-pipe diagnosis): response-stage failures are a
# distinct outcome class -- the request WAS delivered and may have executed,
# so the runtime is demonstrably not offline and the result is unknown.
AGENT_RESPONSE_LOST_CODE = "agent_response_lost"
AGENT_RESPONSE_LOST_HINT = (
    "请求已送达后台 runtime，但响应在到达前丢失（超时或连接中断）："
    "本次执行结果未知，副作用可能已经发生。请勿盲目重试；先用 "
    "get_playback_context 查询当前状态，或确认 run 仍在运行后再决定。"
)

# S4: the preview-event return channel. ``event_socket`` is a transport-only
# sidecar key the routed client stamps into preview-family payloads; run peels
# it off before the service ever sees the payload and uses it to arm the event
# bridge. The preview tool names ARE the transport's arming set -- the S3
# routing table independently routes the same family.
EVENT_SOCKET_KEY = "event_socket"
PREVIEW_EVENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        AgentToolName.PREVIEW_BATCH.value,
        AgentToolName.PREVIEW_CATALOG_TRACK.value,
        AgentToolName.STOP_PREVIEW.value,
        AgentToolName.ADVANCE_PREVIEW.value,
    }
)
TERMINAL_PREVIEW_EVENTS: frozenset[str] = frozenset({"cancelled", "completed", "failed"})


class AgentSocketError(ValueError):
    code = "agent_socket_error"


class SocketFrameError(AgentSocketError):
    """The byte stream could not be split into one honest frame (fail closed)."""


class AgentResponseLostError(AgentSocketError):
    """The request frame was delivered but no usable response came back.

    Raised by the routed client for every failure AFTER the request frame left
    the wire: a read timeout, a dropped link, an empty or malformed reply.
    The runtime was reachable when the request went out -- the execution
    outcome is unknown and must surface as such, never as an offline fact.
    """

    code = "frame_error"


def _encode_text(text: str, *, what: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SocketFrameError(f"{what} is not valid UTF-8: {error}") from error


def _decode_text(body: bytes, *, what: str) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SocketFrameError(f"{what} is not valid UTF-8: {error}") from error


def agent_socket_path(database_path: str | Path) -> Path:
    """The run-side service socket, derived from the store the runtime hosts."""
    return Path(f"{database_path}{AGENT_SOCKET_SUFFIX}")


def preview_event_socket_path(database_path: str | Path, pid: int) -> Path:
    """The chat-session-side preview-event socket for one client process."""
    if not isinstance(pid, int) or pid <= 0:
        raise AgentSocketError("pid must be a positive integer")
    return Path(f"{database_path}{EVENT_SOCKET_STEM}-{pid}.sock")


def send_framed(sock: socket.socket, body: bytes) -> None:
    """Write one frame (4-byte big-endian length + body) to ``sock``."""
    if not isinstance(body, bytes):
        raise AgentSocketError("frame body must be bytes")
    if len(body) == 0:
        raise SocketFrameError("frame body must not be empty")
    if len(body) > MAX_FRAME_BYTES:
        raise SocketFrameError(
            f"frame of {len(body)} bytes exceeds the {MAX_FRAME_BYTES}-byte cap"
        )
    sock.sendall(struct.pack(">I", len(body)) + body)


def _exact_recv(sock: socket.socket, length: int) -> bytes | None:
    """Read exactly ``length`` bytes; ``None`` on a clean EOF before completion."""
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except OSError as error:
            raise SocketFrameError(f"socket read failed: {error}") from error
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_framed(sock: socket.socket) -> bytes:
    """Read one frame; the length cap is enforced before the body is read."""
    header = _exact_recv(sock, FRAME_LENGTH_BYTES)
    if header is None:
        raise SocketFrameError("connection closed before the frame header completed")
    (length,) = struct.unpack(">I", header)
    if length == 0:
        raise SocketFrameError("frame body must not be empty")
    if length > MAX_FRAME_BYTES:
        raise SocketFrameError(
            f"frame of {length} bytes exceeds the {MAX_FRAME_BYTES}-byte cap"
        )
    body = _exact_recv(sock, length)
    if body is None:
        raise SocketFrameError("connection closed before the frame body completed")
    return body


def request_to_frame(request: AgentRequest) -> bytes:
    """Frame one canonical agent request (the P09 provider interchange text)."""
    if not isinstance(request, AgentRequest):
        raise AgentSocketError("request must be an AgentRequest")
    return _encode_text(encode_agent_request(request), what="agent request")


def frame_to_request(body: bytes) -> AgentRequest:
    """Decode one framed body back to a validated :class:`AgentRequest`."""
    if not isinstance(body, bytes):
        raise AgentSocketError("frame body must be bytes")
    return decode_agent_request(_decode_text(body, what="agent request frame"))


def result_to_frame(result: AgentToolResult) -> bytes:
    """Frame one canonical agent tool result (the P09 provider interchange text)."""
    if not isinstance(result, AgentToolResult):
        raise AgentSocketError("result must be an AgentToolResult")
    return _encode_text(encode_agent_tool_result(result), what="agent tool result")


def frame_to_result(body: bytes) -> AgentToolResult:
    """Decode one framed body back to a validated :class:`AgentToolResult`."""
    if not isinstance(body, bytes):
        raise AgentSocketError("frame body must be bytes")
    return decode_agent_tool_result(_decode_text(body, what="agent result frame"))


def encode_event_frame(event: Mapping[str, Any]) -> bytes:
    """Frame one preview event for the S4 return channel.

    The event carries the S1 presenter contract verbatim -- ``{event, session,
    suspended}`` -- as one JSON object text; the transport adds nothing.
    Unlike the agent interchange this is not a P09 decision: no exact-key
    canonicalization applies, only UTF-8 JSON that decodes back to a mapping
    (the presenter validates its own shapes, exactly as it does locally).
    """
    if not isinstance(event, Mapping):
        raise AgentSocketError("event must be a mapping")
    return _encode_text(
        json.dumps(dict(event), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        what="preview event",
    )


def decode_event_frame(body: bytes) -> dict[str, Any]:
    """Decode one framed body back to the event mapping; bad JSON fails closed."""
    if not isinstance(body, bytes):
        raise AgentSocketError("frame body must be bytes")
    text = _decode_text(body, what="preview event frame")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise SocketFrameError(f"preview event frame is not JSON: {error}") from error
    if not isinstance(data, dict):
        raise SocketFrameError("preview event frame must be a JSON object")
    return data


def build_offline_refusal(
    request: AgentRequest,
    *,
    completed_at: datetime | None = None,
) -> AgentToolResult:
    """Decision b: one routed request the runtime could not host (it is offline).

    The refusal is itself a fully valid execution envelope -- EXECUTION_ERROR
    with the stable ``agent_runtime_offline`` code -- so the routed client can
    return it to the provider loop unchanged. It deliberately carries no
    payload: the offline fact must reach the model and the user, never a
    locally-fabricated session answer (Preview / Device Safety never degrade
    locally).
    """
    if not isinstance(request, AgentRequest):
        raise AgentSocketError("request must be an AgentRequest")
    return AgentToolResult(
        request_id=request.request_id,
        tool=request.tool,
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code=AGENT_RUNTIME_OFFLINE_CODE,
        error_message=AGENT_RUNTIME_OFFLINE_HINT,
        completed_at=completed_at if completed_at is not None else datetime.now(timezone.utc),
    )


def build_response_lost_result(
    request: AgentRequest,
    *,
    completed_at: datetime | None = None,
) -> AgentToolResult:
    """S3 repair: the honest surface for a delivered-but-unanswered request.

    Exactly like the offline refusal it is a fully valid execution envelope
    (EXECUTION_ERROR, no payload), but the error code is
    ``agent_response_lost``: the runtime WAS reachable -- the request may have
    executed and its side effects may already exist -- and the execution
    outcome is unknown. No auto-retry may ride on this code; the provider sees
    an explicit "check current state first" guidance instead.
    """
    if not isinstance(request, AgentRequest):
        raise AgentSocketError("request must be an AgentRequest")
    return AgentToolResult(
        request_id=request.request_id,
        tool=request.tool,
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code=AGENT_RESPONSE_LOST_CODE,
        error_message=AGENT_RESPONSE_LOST_HINT,
        completed_at=completed_at if completed_at is not None else datetime.now(timezone.utc),
    )


_CONNECTION_TIMEOUT_SECONDS = 30.0
_ACCEPT_TIMEOUT_SECONDS = 0.5
_LISTEN_BACKLOG = 8
_SOCKET_MODE = 0o600
_RESPONSE_POLL_SECONDS = 0.2
_EVENT_SEND_TIMEOUT_SECONDS = 5.0


def _conn_peer(sock: socket.socket) -> str:
    """Best-effort peer identity for [agent-ipc] traces (never raises).

    For AF_UNIX the peer name is the peer's bound path when it has one --
    a chat-session exchange socket is anonymous (''), a named peer pointing
    at the agent socket would betray a cross-wired client.
    """
    try:
        peer = sock.getpeername()
    except OSError:
        return "?"
    return peer if isinstance(peer, str) else repr(peer)


def _thread_name() -> str:
    return threading.current_thread().name


def _connection_alive(sock: socket.socket) -> bool:
    """Best-effort pre-execute liveness probe for a pending request connection.

    Strong-evidence-only: the request is abandoned only when the peer has
    observably gone away (EOF or a reset/unusable connection). A silent but
    open connection -- the normal "client is waiting for its response" state
    -- probes as alive, and an ambiguous probe error (EINTR and friends) also
    resolves to alive so an uncertain transport can never swallow a side
    effect the client asked for. Never raises.
    """
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return True
        return sock.recv(1, socket.MSG_PEEK) != b""
    except OSError as error:
        if error.errno in (errno.ECONNRESET, errno.ENOTCONN):
            return False
        return True


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    request: AgentRequest
    slot: queue.Queue[AgentToolResult | None]
    connection: socket.socket


def _unlink_socket_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _socket_file_state(path: Path) -> str:
    """The honest reading of an existing socket file: ``live``, ``stale``, ``absent``."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1.0)
        probe.connect(str(path))
    except OSError as error:
        if error.errno == errno.ECONNREFUSED:
            return "stale"
        if error.errno == errno.ENOTSOCK:
            # Not a socket at all: whatever is there is not a live server.
            return "stale"
        if error.errno == errno.ENOENT:
            return "absent"
        raise AgentSocketError(f"cannot probe agent socket {path}: {error}") from error
    else:
        return "live"
    finally:
        probe.close()


def _bind_unix_listener(path: Path) -> socket.socket:
    """Bind one AF_UNIX listener at ``path`` (0600, already listening).

    Shared by the request server and the event listener: a live socket is
    refused -- the only honest reading of concurrent hosting -- and a stale
    file from a dead process is recovered. The caller tracks ownership so
    only a successful bind licenses the unlink on close.
    """
    state = _socket_file_state(path)
    if state == "live":
        raise AgentSocketError(f"socket {path} is already served by a live process")
    if state == "stale":
        _unlink_socket_quietly(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
    except OSError as error:
        listener.close()
        raise AgentSocketError(f"cannot bind socket {path}: {error}") from error
    try:
        os.chmod(path, _SOCKET_MODE)
    except OSError:
        logger.warning("could not chmod socket %s", path)
    listener.listen(_LISTEN_BACKLOG)
    listener.settimeout(_ACCEPT_TIMEOUT_SECONDS)
    return listener


class PreviewEventForwarder:
    """S4: the run-side one-shot presenter bridge for one chat-session.

    While a routed preview-family request is in flight the server puts this
    callable in the service's ``preview_event_handler`` slot; every session
    event is pushed verbatim over the requesting chat-session's event socket.
    The bridge must outlive its request -- a natural-finish completion arrives
    on the reaper thread minutes after the response left -- so it closes on a
    terminal event (``cancelled`` / ``completed`` / ``failed``) or an explicit
    :meth:`close`, never on the request's return.

    Best-effort by contract: a vanished chat-session or a dead connection
    drops the event silently -- presentation may never interrupt audio -- and
    malformed events are dropped.
    """

    def __init__(
        self,
        event_socket_path: Path,
        *,
        on_terminal: Callable[[], None] | None = None,
        timeout: float = _EVENT_SEND_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(event_socket_path, Path) or not event_socket_path.parts:
            raise AgentSocketError("event_socket_path must be a non-empty Path")
        if on_terminal is not None and not callable(on_terminal):
            raise AgentSocketError("on_terminal must be callable or None")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise AgentSocketError("timeout must be a positive number of seconds")
        self._event_socket_path = event_socket_path
        self._on_terminal = on_terminal
        self._timeout = float(timeout)
        self._terminal = False

    @property
    def event_socket_path(self) -> Path:
        return self._event_socket_path

    @property
    def terminal(self) -> bool:
        """True once a terminal event (or :meth:`close`) ended the bridge."""
        return self._terminal

    def __call__(self, event: Mapping[str, Any]) -> None:
        if self._terminal:
            if isinstance(event, Mapping):
                logger.debug(
                    "[preview-ipc] run: forwarder called after terminal -- "
                    "dropped kind=%s",
                    event.get("event"),
                )
            return  # a bridged session is strictly monotone: nothing after the end
        if not isinstance(event, Mapping):
            return  # presenter contract: malformed events are dropped, not raised
        logger.debug(
            "[preview-ipc] run: forwarder will relay kind=%s to %s",
            event.get("event"),
            self._event_socket_path,
        )
        self._send(event)
        if event.get("event") in TERMINAL_PREVIEW_EVENTS:
            self.close()  # the terminal event itself IS delivered before the close

    def close(self) -> None:
        """Stop delivery and fire the arm-side restore exactly once (idempotent)."""
        if self._terminal:
            return
        self._terminal = True
        on_terminal = self._on_terminal
        if on_terminal is not None:
            on_terminal()

    def _send(self, event: Mapping[str, Any]) -> None:
        """One event over one fresh connection; every failure is swallowed."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            sock.connect(str(self._event_socket_path))
        except OSError as error:
            logger.warning(
                "[preview-ipc] run: forwarder connect to %s failed: %s",
                self._event_socket_path,
                error,
            )
            sock.close()
            return  # no listener: the chat-session is gone or not yet bound
        logger.debug(
            "[preview-ipc] run: forwarder connected %s (kind=%s)",
            self._event_socket_path,
            event.get("event"),
        )
        try:
            frame = encode_event_frame(event)
            send_framed(sock, frame)
            logger.debug(
                "[preview-ipc] run: forwarder sent %d bytes (kind=%s)",
                len(frame),
                event.get("event"),
            )
        except (AgentSocketError, OSError) as error:
            logger.warning("[preview-ipc] run: forwarder send failed: %s", error)
        finally:
            try:
                sock.close()
            except OSError:
                pass


class AgentSocketServer:
    """S2: the run-side host executing routed agent requests over the service UDS.

    Transport machinery only: it owns the listener thread and the socket file;
    every permission, replay, and policy decision stays inside the service's
    ``execute``. The accept thread receives, decodes, and waits; execution
    happens exclusively on the runtime's own thread through the component
    ``tick()`` hook -- the store's SQLite connections are single-threaded by
    construction, so the service may only ever be entered from the thread that
    started the runtime, exactly like the automation engine. A live listener
    already bound to the socket path is the one case the server decides for
    itself -- the runtime is already hosted, and guessing around it is refused;
    a stale file from a crashed process is recovered by unlink.
    """

    def __init__(self, socket_path: Path, service: object) -> None:
        if not isinstance(socket_path, Path) or socket_path == Path(""):
            raise AgentSocketError("socket_path must be a non-empty Path")
        if not callable(getattr(service, "execute", None)):
            raise AgentSocketError("service must expose execute(request)")
        self.socket_path = socket_path
        self._service = service
        self._stop_event = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._pending: queue.Queue[_PendingRequest] = queue.Queue()
        # The socket file is unlinked only when this server bound it successfully;
        # a failed bind must never remove another live process's socket.
        self._owns_path = False
        # S4: forwarders armed while their chat-session lives; closed when the
        # bridge terminates or when the server itself stops (unreapable arms
        # must never leak past the runtime's life).
        self._armed_forwarders: set[PreviewEventForwarder] = set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._listener = self._bound_listener()
        self._thread = threading.Thread(
            target=self._serve_forever, name="agent-socket", daemon=True
        )
        self._thread.start()
        logger.info("agent socket listening at %s", self.socket_path)

    def tick(self) -> None:
        """Runtime component hook: drain pending requests on the runtime's own thread.

        The runtime loop calls this on the thread that owns the store's SQLite
        connections -- the only thread allowed to enter the service. Requests
        arrive from the accept thread fully decoded; each is executed here and
        its result handed back through the per-request slot, which wakes the
        connection handler to write the response frame.

        Before execute, each request's connection is probed for liveness: a
        request whose client has already gone away is abandoned without
        entering ``service.execute`` -- no preview/playback side effects, no
        retry -- and the handler is released to close the connection.
        """
        while True:
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                return
            logger.debug(
                "[runtime-dispatch] socket: drain begin req=%s qsize=%s",
                pending.request.request_id,
                self._pending.qsize(),
            )
            if not _connection_alive(pending.connection):
                logger.warning(
                    "[runtime-dispatch] socket: abandoned-before-execute req=%s tool=%s "
                    "-- client connection gone before execute; dropping the queued "
                    "request without side effects and without retry",
                    pending.request.request_id,
                    pending.request.tool,
                )
                pending.slot.put(None)
                logger.debug(
                    "[runtime-dispatch] socket: drain end req=%s qsize=%s result=abandoned",
                    pending.request.request_id,
                    self._pending.qsize(),
                )
                continue
            logger.debug(
                "[agent-ipc] run: executing req=%s tool=%s thr=%s",
                pending.request.request_id,
                pending.request.tool,
                _thread_name(),
            )
            try:
                result = self._execute(pending.request)
            except Exception:
                logger.exception(
                    "[runtime-dispatch] socket: drain execute raised req=%s",
                    pending.request.request_id,
                )
                result = None
            pending.slot.put(result)
            logger.debug(
                "[agent-ipc] run: tick put result req=%s thr=%s",
                pending.request.request_id,
                _thread_name(),
            )
            logger.debug(
                "[runtime-dispatch] socket: drain end req=%s qsize=%s result=%s",
                pending.request.request_id,
                self._pending.qsize(),
                "ok" if result is not None else "None",
            )

    def close(self) -> None:
        """Stop the accept thread and remove the socket file. Idempotent."""
        self._stop_event.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        self._listener = None
        if self._owns_path:
            _unlink_socket_quietly(self.socket_path)
            self._owns_path = False
        for forwarder in list(self._armed_forwarders):
            forwarder.close()
        self._armed_forwarders.clear()
        logger.info("agent socket closed at %s", self.socket_path)

    def _bound_listener(self) -> socket.socket:
        listener = _bind_unix_listener(self.socket_path)
        self._owns_path = True
        return listener

    def _serve_forever(self) -> None:
        assert self._listener is not None
        try:
            while not self._stop_event.is_set():
                try:
                    connection, _address = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        logger.exception("agent socket accept failed")
                    continue
                fd = connection.fileno()
                logger.debug(
                    "[agent-ipc] run: accepted fd=%s peer=%r thr=%s",
                    fd,
                    _conn_peer(connection),
                    _thread_name(),
                )
                try:
                    self._handle_connection(connection)
                finally:
                    try:
                        connection.close()
                    except OSError:
                        pass
                logger.debug("[agent-ipc] run: connection closed fd=%s", fd)
        finally:
            try:
                self._listener.close()
            except OSError:
                pass
            _unlink_socket_quietly(self.socket_path)
        logger.info("agent socket loop stopped")

    def _handle_connection(self, connection: socket.socket) -> None:
        connection.settimeout(_CONNECTION_TIMEOUT_SECONDS)
        try:
            body = receive_framed(connection)
        except (SocketFrameError, socket.timeout, OSError) as error:
            logger.warning("agent socket: frame read failed: %s", error)
            return
        try:
            request = frame_to_request(body)
        except AgentContractValidationError as error:
            logger.warning("agent socket: undecodable request dropped: %s", error)
            return
        logger.debug(
            "[agent-ipc] run: request decoded req=%s tool=%s fd=%s",
            request.request_id,
            request.tool,
            connection.fileno(),
        )
        slot: queue.Queue[AgentToolResult | None] = queue.Queue(maxsize=1)
        self._pending.put(_PendingRequest(request=request, slot=slot, connection=connection))
        logger.debug(
            "[agent-ipc] run: request enqueued req=%s tool=%s fd=%s",
            request.request_id,
            request.tool,
            connection.fileno(),
        )
        result: AgentToolResult | None = None
        while True:
            try:
                result = slot.get(timeout=_RESPONSE_POLL_SECONDS)
                break
            except queue.Empty:
                if self._stop_event.is_set():
                    return  # closing: the runtime thread will never drain again
        if result is None:
            return
        logger.debug(
            "[agent-ipc] run: writing response req=%s fd=%s thr=%s",
            request.request_id,
            connection.fileno(),
            _thread_name(),
        )
        try:
            send_framed(connection, result_to_frame(result))
        except OSError as error:
            logger.warning(
                "[agent-ipc] run: response write FAILED req=%s fd=%s peer=%r error=%s",
                request.request_id,
                connection.fileno(),
                _conn_peer(connection),
                error,
            )
        else:
            logger.debug(
                "[agent-ipc] run: response written req=%s fd=%s",
                request.request_id,
                connection.fileno(),
            )

    def _execute(self, request: AgentRequest) -> AgentToolResult:
        """The service boundary, one request at a time; escaping exceptions
        (the osascript family) become honest EXECUTION_ERROR envelopes so the
        routed client always receives one validated result frame.

        S4: a preview-family request carrying the ``event_socket`` sidecar
        arms a :class:`PreviewEventForwarder` on the service presenter slot
        before execute; a raising execute (no session ever started) disarms
        it -- a successful one leaves the bridge live until a terminal event
        arrives, which can be minutes later on the reaper thread."""
        forwarder: PreviewEventForwarder | None = None
        prepared = request
        if request.tool in PREVIEW_EVENT_TOOL_NAMES:
            prepared, forwarder = self._detach_event_sidecar(request)
        try:
            started = time.monotonic()
            result = self._service.execute(prepared)  # type: ignore[attr-defined]
            logger.info(
                "agent socket: routed execute tool=%s finished in %.1fs",
                request.tool,
                time.monotonic() - started,
            )
            return result
        except Exception as error:
            if forwarder is not None:
                forwarder.close()  # no session: no events will ever arrive
            logger.exception("agent socket: routed execute raised")
            return AgentToolResult(
                request_id=request.request_id,
                tool=request.tool,
                outcome=AgentToolOutcome.EXECUTION_ERROR,
                payload=None,
                error_code=getattr(error, "code", None) or "execution_error",
                error_message=str(error) or "routed execution failed",
                completed_at=datetime.now(timezone.utc),
            )

    def _detach_event_sidecar(
        self, request: AgentRequest
    ) -> tuple[AgentRequest, PreviewEventForwarder | None]:
        """S4: peel the ``event_socket`` transport sidecar off a preview-family
        payload and, when a chat-session supplied a usable path, arm the event
        bridge on the service's presenter slot.

        The sidecar is stripped whether or not a bridge arms -- transport
        keys never reach executors. Arming swaps the service handler and
        restores the previous one on the bridge's terminal event
        (identity-guarded so a late terminal can never clobber a newer arm).
        """
        payload = request.payload
        prepared = request
        if isinstance(payload, Mapping) and EVENT_SOCKET_KEY in payload:
            prepared = replace(
                request,
                payload={
                    key: value for key, value in payload.items() if key != EVENT_SOCKET_KEY
                },
            )
            raw = payload[EVENT_SOCKET_KEY]
            if isinstance(raw, str) and Path(raw).parts:
                service = self._service
                if not hasattr(service, "preview_event_handler"):
                    # Event-less service (stub / disabled presenter): nothing
                    # to bridge -- events simply will not be surfaced.
                    logger.debug(
                        "[preview-ipc] run: service exposes no preview_event_handler "
                        "slot -- no bridge (tool=%s), events will not be surfaced",
                        request.tool,
                    )
                    return prepared, None
                previous = getattr(service, "preview_event_handler", None)
                forwarder = PreviewEventForwarder(
                    Path(raw),
                    on_terminal=lambda: self._restore_handler(service, forwarder, previous),
                )
                self._armed_forwarders.add(forwarder)
                service.preview_event_handler = forwarder  # type: ignore[attr-defined]
                logger.debug(
                    "[preview-ipc] run: event bridge armed (tool=%s) -> %s",
                    request.tool,
                    raw,
                )
                return prepared, forwarder
            logger.debug(
                "[preview-ipc] run: event_socket sidecar unusable (%r) -- "
                "no bridge (tool=%s)",
                raw,
                request.tool,
            )
        return prepared, None

    def _restore_handler(
        self, service: object, forwarder: PreviewEventForwarder, previous: Any
    ) -> None:
        """Terminal callback: put the presenter slot back the way it was."""
        if getattr(service, "preview_event_handler", None) is forwarder:
            service.preview_event_handler = previous  # type: ignore[attr-defined]
            logger.debug(
                "[preview-ipc] run: bridge disarmed, presenter slot restored "
                "(previous=%s)",
                type(previous).__name__ if previous is not None else "none",
            )
        else:
            logger.debug("[preview-ipc] run: bridge disarm skipped (slot no longer ours)")
        self._armed_forwarders.discard(forwarder)


class AgentEventSocketListener:
    """S4: the chat-session-side preview-event sink.

    Owns ``<store>.agent-events-<pid>.sock`` for the life of the chat process
    and hands every decoded event mapping to the presenter -- the same
    ``{event, session, suspended}`` contract the run-side forwarder sends --
    so presentation stays identical whether events come from the local service
    or over the socket. The presenter runs on the accept thread and must be
    cheap and non-raising; the CLI presenter only prints. There are no
    responses and no state on this end.

    Ownership mirrors :class:`AgentSocketServer`: a live socket at the path
    is refused (this pid already hosts one), a stale file from a dead process
    is recovered, and only a successful bind licenses the unlink on close.
    """

    def __init__(self, socket_path: Path, presenter: Callable[[Any], None]) -> None:
        if not isinstance(socket_path, Path) or not socket_path.parts:
            raise AgentSocketError("socket_path must be a non-empty Path")
        if not callable(presenter):
            raise AgentSocketError("presenter must be callable")
        self.socket_path = socket_path
        self._presenter = presenter
        self._stop_event = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._owns_path = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._listener = _bind_unix_listener(self.socket_path)
        self._owns_path = True
        self._thread = threading.Thread(
            target=self._serve_forever, name="agent-event-socket", daemon=True
        )
        self._thread.start()
        logger.info("preview event socket listening at %s", self.socket_path)

    def close(self) -> None:
        """Stop the accept thread and remove the socket file. Idempotent."""
        self._stop_event.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        self._listener = None
        if self._owns_path:
            _unlink_socket_quietly(self.socket_path)
            self._owns_path = False
        logger.info("preview event socket closed at %s", self.socket_path)

    def _serve_forever(self) -> None:
        assert self._listener is not None
        try:
            while not self._stop_event.is_set():
                try:
                    connection, _address = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        logger.exception("agent event socket accept failed")
                    continue
                try:
                    self._consume_connection(connection)
                finally:
                    try:
                        connection.close()
                    except OSError:
                        pass
        finally:
            try:
                self._listener.close()
            except OSError:
                pass
            _unlink_socket_quietly(self.socket_path)
        logger.info("agent event socket loop stopped")

    def _consume_connection(self, connection: socket.socket) -> None:
        """One connection = one event: decode it and hand it to the presenter.

        A bad frame drops that event only, and a raising presenter is
        swallowed -- the pipeline must survive a broken print target."""
        connection.settimeout(_CONNECTION_TIMEOUT_SECONDS)
        try:
            body = receive_framed(connection)
            event = decode_event_frame(body)
        except (SocketFrameError, socket.timeout, OSError) as error:
            logger.warning("agent event socket: frame dropped: %s", error)
            return
        logger.debug(
            "[preview-ipc] chat: event received bytes=%d kind=%s",
            len(body),
            event.get("event"),
        )
        try:
            self._presenter(event)
            logger.debug(
                "[preview-ipc] chat: presenter returned (kind=%s)", event.get("event")
            )
        except Exception:
            logger.exception("agent event socket: presenter raised (swallowed)")