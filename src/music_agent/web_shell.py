"""P17-B: the minimal local product shell -- a Terminal-free daily-use surface.

One process hosts a tiny static UI (``ui/index.html``) plus a stdlib HTTP
server over loopback, layered strictly as:

    existing Backend / Agent  ->  UI adapter  ->  thin HTML surface

Nothing here re-implements recommendation, feedback-learning, playback or
open-in-Apple-Music semantics: every user action is one existing agent tool
executed through the production service / routed client, exactly the calls
``chat-session`` makes. The shell never writes the SQLite store directly,
never composes Apple Music URLs, and its UI knows no intent parsing.

Thread model (P15-S2-IPC audit: SQLite access is single-threaded by
construction, one service per thread):

    worker-A (authority)  transient card reads + UI commands + the embedded
                          authority when this process hosts it (agent socket
                          tick + audio-safety observer, the same pieces
                          ``Runtime`` wires for ``music-agent run``)
    worker-B (chat)       one provider loop per message over its own service
    HTTP handler threads  enqueue + wait only; they never enter a service

Authority attach-or-embed: the CLI probes ``<db>.agent.sock`` up front and
selects the mode. ``embed`` -- no other run is alive, this process binds the
socket and hosts preview/device-safety authority for its lifetime; ``attach``
-- a real run is alive, preview/playback-context commands ride its socket
(the production IPC), everything local stays local. ``standalone`` is the
test/lab mode: direct clients, no UDS anywhere.

UI-side status is never optimistic: ``/api/state`` reads the authoritative
``get_now_playing`` + ``get_playback_context`` tool results (Music.app
readback), and every card action returns the real tool envelope -- errors
surface as an honest banner.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger("music_agent.web_shell")

# One user action == one existing agent tool. The whitelist is closed: the
# browser can trigger exactly these, with exactly their documented payload
# shapes, and nothing else. ``play`` routes to the run authority (the
# continue/restore semantic), the rest are the stateless or adapter-only
# controls the routing table already classifies.
SHELL_COMMANDS: frozenset[str] = frozenset(
    {
        "pause",
        "play",
        "next_track",
        "previous_track",
        "play_track",
        "preview_catalog_track",
        "open_in_apple_music",
    }
)

# Commands whose payload carries one canonical track id (validated as a
# non-empty string server-side; the UI never gets to name payload keys).
_CANONICAL_ID_COMMANDS: frozenset[str] = frozenset(
    {"play_track", "preview_catalog_track", "open_in_apple_music"}
)

_CHAT_MAX_CHARS = 4000
_STATUS_JOB_TIMEOUT_SECONDS = 60.0
_CHAT_RESPONSE_TIMEOUT_SECONDS = 900.0


class ShellStartupError(RuntimeError):
    """Startup failure surfaced to the launcher window before the browser opens."""

    code = "shell_startup_error"


class ShellProjection:
    """Pure view-model projections: tool payloads -> UI JSON.

    Presentational only -- never recommendation, playback or identity logic.
    Additive keys are fine here; changing any input fact is not.
    """

    @staticmethod
    def formal_player(
        now_playing_payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Formal-player facts only, from the cheap authoritative readback."""
        source: Mapping[str, Any] = {}
        if isinstance(now_playing_payload, Mapping) and isinstance(
            now_playing_payload.get("now_playing"), Mapping
        ):
            source = now_playing_payload["now_playing"]
        return {
            "state": source.get("state"),
            "name": source.get("name"),
            "artist": source.get("artist"),
            "album": source.get("album"),
        }

    @staticmethod
    def now_playing(
        now_playing_payload: Mapping[str, Any] | None,
        context_payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Player strip model from the authoritative tool readbacks.

        ``now_playing_payload`` come from ``get_now_playing``, ``context_payload``
        from ``get_playback_context`` (null-safe on tool failure -- the strip
        renders an honest "不可用" instead of inventing transport state).
        """
        formal = ShellProjection.formal_player(now_playing_payload)
        session: Mapping[str, Any] | None = None
        preview_suspension: Mapping[str, Any] | None = None
        preview_sounding = False
        if isinstance(context_payload, Mapping):
            if isinstance(context_payload.get("session"), Mapping):
                session = context_payload["session"]
            if isinstance(context_payload.get("preview_suspension"), Mapping):
                preview_suspension = context_payload["preview_suspension"]
            preview_sounding = context_payload.get("preview_sounding") is True
        return {
            **formal,
            "preview_session": (
                {
                    "state": session.get("state"),
                    "position": session.get("position"),
                    "total": session.get("total"),
                    "current_name": session.get("current_name"),
                    "failure_reason": session.get("failure_reason"),
                }
                if session is not None
                else None
            ),
            # P18-S1/P20: the runner truth plus the restore arm's lifecycle
            # projection. The historical ``suspended`` memo intentionally does
            # not cross this presentation boundary: once the obligation is
            # completed or cancelled, the UI note must disappear immediately.
            "preview_sounding": preview_sounding,
            "suspended": (
                {
                    "name": preview_suspension.get("name"),
                    "player_state": preview_suspension.get("player_state"),
                    "pause_ok": preview_suspension.get("pause_ok"),
                }
                if preview_suspension is not None
                and preview_suspension.get("pause_ok") is True
                else None
            ),
        }

    @staticmethod
    def cards(
        items: Iterable[Mapping[str, Any]],
        openable_by_id: Mapping[str, bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Recommendation card model: display facts + the action identifier.

        Drops ``target_id``-adjacent internal surface the UI must not render
        (candidate ids, scores, provenance, fresh markers). The buttons are
        derived from the existing playback.route fact -- the UI decides
        nothing about capability.

        P18-S2: ``apple_music_openable`` mirrors the real ``itunes_store``
        binding presence (the same fact ``open_in_apple_music`` fail-closes
        on); absent map = False for every card, so the UI can never offer an
        action the backend would honestly refuse.
        """
        cards: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            artist = item.get("artist_name")
            album = item.get("album")
            canonical_id = item.get("target_id")
            if not isinstance(canonical_id, str) or not canonical_id:
                continue
            playback = item.get("playback") if isinstance(item.get("playback"), Mapping) else {}
            route = playback.get("route")
            cards.append(
                {
                    "canonical_id": canonical_id,
                    "name": name if isinstance(name, str) else "",
                    "artist": artist if isinstance(artist, str) else "",
                    # P18-S1.3: version/source disambiguation (single vs album),
                    # display-only -- identity stays the canonical id.
                    "album": album if isinstance(album, str) else "",
                    "route": route if isinstance(route, str) else "unavailable",
                    "apple_music_openable": (
                        bool(openable_by_id.get(canonical_id))
                        if openable_by_id is not None
                        else False
                    ),
                }
            )
        return cards

    @staticmethod
    def command_error(result: Any) -> dict[str, str] | None:
        """Honest tool-envelope error for the UI, or None on success.

        ``result`` is an AgentToolResult; every non-ok outcome becomes
        ``{code, message}`` for the banner (never a fabricated ok).
        """
        # Import-lazy: projection module must stay light for unit tests.
        from music_agent.agent_contract import AgentToolOutcome

        outcome = getattr(result, "outcome", None)
        if outcome is None or outcome == AgentToolOutcome.OK:
            return None
        return {
            "code": getattr(result, "error_code", None) or "command_failed",
            "message": getattr(result, "error_message", None) or "操作失败，请重试。",
        }

    @staticmethod
    def verified_action_response(result: Any) -> dict[str, Any] | None:
        """Project an ActionAttempt without recreating its success policy."""

        from music_agent.action_attempt import (
            ActionAttempt,
            ActionAttemptStatus,
            PlaybackControlAttempt,
            render_verified_action_result,
            render_verified_playback_control_result,
        )

        if not isinstance(result, (ActionAttempt, PlaybackControlAttempt)):
            return None
        message = (
            render_verified_action_result(result)
            if isinstance(result, ActionAttempt)
            else render_verified_playback_control_result(result)
        )
        if result.status is ActionAttemptStatus.COMPLETED:
            return {
                "ok": True,
                "result": {
                    "status": result.status.value,
                    "readback_state": result.readback_state.value,
                },
                "message": message,
            }
        return {
            "ok": False,
            "error": {
                "code": (
                    result.execution_error_code
                    if result.execution_error_code
                    else result.failure_reason or "action_not_verified"
                ),
                "message": message,
            },
        }


class ShellEventLog:
    """Thread-safe sequential store of preview-session events, tail-polled by
    the browser (``/api/events?after=``). The presenter is registered exactly
    where chat-session registers its console presenter; the UI is one more
    client of the same P15-S1 event stream."""

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._next_seq = 1
        self._events: list[dict[str, Any]] = []

    def present(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        with self._lock:
            self._events.append({"seq": self._next_seq, "event": dict(event)})
            self._next_seq += 1
            if len(self._events) > self._capacity:
                self._events = self._events[-self._capacity :]

    def snapshot(self, after: int) -> list[dict[str, Any]]:
        with self._lock:
            return [entry for entry in self._events if entry["seq"] > after]


class DurableCardsSource:
    """Transient, read-only cards reader: latest recommendation run projected
    with the production ``_item_playback_summary`` view (identical facts the
    model sees), then narrowed by :meth:`ShellProjection.cards` to the UI
    shape. Connections are opened and closed inside one call on the calling
    worker thread -- never shared, never written."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def latest_batch_state(self, run_id: str | None = None) -> tuple[str | None, list[dict[str, Any]]]:
        """(newest run id, projected cards for exactly that run) from ONE
        history read.

        P19-T14-B: the id and its cards must never disagree about which run
        is newest -- the reply door compares ids across a chat turn and the
        UI takes over on the attached pairs, so both halves come from the
        same repository connection and the same insertion-chronology key
        (``newest_run_id``). The item-key projection inside the cards is a
        display detail; batch identity stays the run id.
        """
        from music_agent.agent_service import SharedAgentService
        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )
        from music_agent.repository import CanonicalRepository

        with RecommendationHistoryRepository(self._database_path) as history:
            run_id = run_id if run_id is not None else history.newest_run_id()
            run = history.get_result(run_id) if run_id is not None else None
        if run is None:
            return None, []
        with CanonicalRepository(self._database_path) as canonical:
            model = canonical.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        album_by_id = {album["id"]: album for album in model["albums"]}
        items = [
            SharedAgentService._item_playback_summary(
                item, track_by_id, artist_by_id, album_by_id
            )
            for item in run.items
        ]
        # P18-S2: the Apple Music action gate is the same itunes_store binding
        # the backend fail-closes on -- projected here so the card buttons can
        # never offer what the command would refuse.
        openable_by_id = {
            track_id: bool(track.get("external_ids", {}).get("itunes_store_id"))
            for track_id, track in track_by_id.items()
        }
        return run_id, ShellProjection.cards(items, openable_by_id)

    def latest_cards(self) -> list[dict[str, Any]]:
        return self.latest_batch_state()[1]

    def latest_run_id(self) -> str | None:
        """Newest persisted recommendation run id, or None when none exist.

        The run id -- not the projected item keys -- is the freshness truth
        for the structured takeover: two runs whose items project to the same
        key are still two different batches. Like :meth:`latest_cards`, the
        repository connection is opened and closed inside this call on the
        calling worker thread -- never shared, never written.
        """
        return self.latest_batch_state()[0]


@dataclass(frozen=True, slots=True)
class ShellConfig:
    """CLI-supplied composition inputs (verbatim chat-session knobs + shell knobs)."""

    database_path: Path
    provider_factory: Callable[[], Any]
    agent_client: tuple[str, str]  # (client_id, policy)
    max_rounds: int = 8
    trace_path: str | None = None
    mode: str | None = None  # None = probe <db>.agent.sock (attach-or-embed)
    host: str = "127.0.0.1"
    port: int = 0
    open_browser: bool = True
    # Test seam: build one SharedAgentService (fake adapters) instead of the
    # production wiring. Never used for mode decisions or client routing.
    service_factory: Callable[[], Any] | None = None


def agent_socket_is_served(socket_path: Path) -> bool:
    """A served socket accepts a connect; a dead path refuses. Best-effort
    probe: any failure answers False (then embed governs under probe rules)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
        return True
    except OSError:
        return False
    finally:
        sock.close()


class _SerialWorker(threading.Thread):
    """One service-entering thread (P15-S2-IPC single-threaded construction).

    Jobs are plain callables dispatched through a queue; the executing thread
    is the only thread that ever enters its service's repositories. An
    optional pump hook (the embedded agent-socket ``tick``) runs between jobs.
    """

    def __init__(self, name: str, pump: Callable[[], None] | None = None) -> None:
        super().__init__(name=name, daemon=True)
        self._queue: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        # Note: the event must NOT be named ``_stop`` -- that private slot is
        # Thread's own shutdown callable and ``join`` would call the Event.
        self._stop_event = threading.Event()
        self._pump = pump

    def submit(self, fn: Callable[[], Any]) -> Future:
        future: Future = Future()
        self._queue.put((fn, future))
        return future

    def set_pump(self, pump: Callable[[], None]) -> None:
        """Install the authority tick (embedded agent socket) after start.

        Wired late because the socket server is built after the workers; the
        run loop reads the pump on every idle iteration."""
        self._pump = pump

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                fn, future = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._pump is not None:
                    try:
                        self._pump()
                    except Exception:
                        logger.exception("[shell] pump tick raised")
                continue
            try:
                future.set_result(fn())
            except Exception as error:  # a job result is never a thread death
                logger.exception("[shell] job raised")
                future.set_exception(error)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self.join(timeout)


class _ShellHttpHandler(BaseHTTPRequestHandler):
    """Transport only: parse, hand off to :class:`WebShellApp`, write JSON.
    No tool, projection or product logic lives here (P17-B layering)."""

    server_version = "MusicAgentShell/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        app: WebShellApp = self.server.app  # type: ignore[attr-defined]
        if self.path == "/" or self.path.startswith("/index.html"):
            self._send_html(app.index_html)
        elif self.path == "/healthz":
            self._send_json(200, app.health())
        elif self.path == "/api/state":
            self._send_json(200, app.state_snapshot())
        elif self.path == "/api/player-state":
            self._send_json(200, app.player_snapshot())
        elif self.path == "/api/cards":
            self._send_json(200, app.latest_batch_state())
        elif self.path.startswith("/api/events"):
            self._send_json(200, app.events_since(self._after_param()))
        else:
            self._send_json(404, {"error": {"code": "not_found", "message": "未知接口。"}})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        app: WebShellApp = self.server.app  # type: ignore[attr-defined]
        if self.path == "/api/chat":
            self._handle_chat(app)
        elif self.path == "/api/command":
            self._handle_command(app)
        elif self.path == "/api/apple-music-target":
            self._handle_apple_music_target(app)
        elif self.path == "/api/shutdown":
            self._send_json(200, {"ok": True, "message": "正在退出。"})
            app.request_shutdown()
        else:
            self._send_json(404, {"error": {"code": "not_found", "message": "未知接口。"}})

    def _handle_chat(self, app: WebShellApp) -> None:
        body = self._read_json_body()
        if body is None:
            return
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": {"code": "empty_message", "message": "消息为空。"}})
            return
        if len(text) > _CHAT_MAX_CHARS:
            self._send_json(400, {"error": {"code": "message_too_long", "message": "消息过长。"}})
            return
        future = app.submit_chat(text.strip())
        try:
            reply = future.result(timeout=_CHAT_RESPONSE_TIMEOUT_SECONDS)
        except Exception as error:
            logger.exception("[shell] chat job failed")
            self._send_json(
                500,
                {"error": {"code": "chat_failed", "message": f"回复生成失败：{error}"}},
            )
            return
        self._send_json(200, reply)

    def _handle_command(self, app: WebShellApp) -> None:
        body = self._read_json_body()
        if body is None:
            return
        payload = body if isinstance(body, dict) else {}
        tool = payload.get("tool")
        if tool not in SHELL_COMMANDS:
            self._send_json(
                400, {"error": {"code": "unknown_command", "message": "未知操作。"}}
            )
            return
        tool_payload: dict[str, Any] = {}
        if tool in _CANONICAL_ID_COMMANDS:
            canonical_id = payload.get("canonical_id")
            if not isinstance(canonical_id, str) or not canonical_id.strip():
                self._send_json(
                    400, {"error": {"code": "missing_track", "message": "缺少曲目标识。"}}
                )
                return
            tool_payload["canonical_id"] = canonical_id
        result = app.run_command(tool, tool_payload)
        verified = ShellProjection.verified_action_response(result)
        if verified is not None:
            self._send_json(200, verified)
            return
        error = ShellProjection.command_error(result)
        if error is not None:
            self._send_json(200, {"ok": False, "error": error})
            return
        self._send_json(200, {"ok": True, "result": dict(result.payload or {})})

    def _handle_apple_music_target(self, app: WebShellApp) -> None:
        """Native-only resolve door: identity lookup without any OS handoff."""
        body = self._read_json_body()
        if body is None:
            return
        canonical_id = body.get("canonical_id") if isinstance(body, dict) else None
        if not isinstance(canonical_id, str) or not canonical_id.strip():
            self._send_json(
                400, {"error": {"code": "missing_track", "message": "缺少曲目标识。"}}
            )
            return
        self._send_json(200, app.apple_music_target(canonical_id))

    def _after_param(self) -> int:
        try:
            prefix = "after="
            start = self.path.index(prefix) + len(prefix)
            return max(0, int(self.path[start:].split("&")[0]))
        except (ValueError, IndexError):
            return 0

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 64 * 1024:
            self._send_json(400, {"error": {"code": "bad_body", "message": "请求体无效。"}})
            return None
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": {"code": "bad_json", "message": "JSON 无效。"}})
            return None
        if not isinstance(parsed, dict):
            self._send_json(400, {"error": {"code": "bad_json", "message": "JSON 无效。"}})
            return None
        return parsed

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, document: str) -> None:
        encoded = document.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("http %s", fmt % args)


class WebShellApp:
    """P17-B composition root: workers, authority wiring, HTTP surface.

    Lifecycle: :meth:`start` (build services, bind authority in embed mode,
    bind the loopback server) -> :meth:`serve_forever` -> :meth:`close`.
    """

    def __init__(self, config: ShellConfig) -> None:
        if not isinstance(config, ShellConfig):
            raise ShellStartupError("config must be a ShellConfig")
        self._config = config
        self.index_html = (
            Path(__file__).parent / "ui" / "index.html"
        ).read_text(encoding="utf-8")
        self._closed = False
        self._shutdown_requested = threading.Event()
        self._events = ShellEventLog()
        self._mode: str | None = None
        self._service_a: Any = None
        self._service_b: Any = None
        self._ui_client: Any = None
        self._chat_client: Any = None
        self._loop: Any = None
        self._socket_server: Any = None
        self._observer: Any = None
        self._event_listener: Any = None
        self._worker_a: _SerialWorker | None = None
        self._worker_b: _SerialWorker | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._cards = DurableCardsSource(config.database_path)
        # P22-S2.1: conversation continuation is owned by the Web session,
        # not ProviderAgentLoop.  The provider still starts fresh per call;
        # this single slot remembers only one code-authorized offered action.
        from music_agent.conversation_continuation import OfferedActionRegister

        self._offered_actions = OfferedActionRegister()

    # -- wiring ----------------------------------------------------------------

    def _build_service(self) -> Any:
        from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
        from music_agent.agent_service import SharedAgentService
        from music_agent.catalog_ingestion import default_catalog_search_source
        from music_agent.playback_control import (
            MusicLibraryResolver,
            MusicPlaybackAdapter,
            OsascriptLibraryResolveRunner,
            OsascriptPlaybackRunner,
        )

        client_id, policy = self._config.agent_client
        return SharedAgentService(
            self._config.database_path,
            clients=AgentClientRegistry({client_id: AgentClientPolicy(policy)}),
            playback_adapter=MusicPlaybackAdapter(OsascriptPlaybackRunner(timeout_seconds=60.0)),
            playback_resolver=MusicLibraryResolver(
                OsascriptLibraryResolveRunner(timeout_seconds=60.0)
            ),
            catalog_search_source=default_catalog_search_source(),
        )

    def _resolve_mode(self) -> str:
        if self._config.mode is not None:
            return self._config.mode
        from music_agent.agent_socket import agent_socket_path

        if agent_socket_is_served(agent_socket_path(self._config.database_path)):
            return "attach"
        return "embed"

    def _build_ui_client(self, client_id: str) -> Any:
        """The command client: direct on the authority in embed/standalone,
        routed over the production IPC in attach mode (authority = the run)."""
        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentClientIdentity
        from music_agent.agent_socket import agent_socket_path
        from music_agent.routed_client import RoutedAgentClient

        identity = AgentClientIdentity(client_id=client_id, model_id="web-ui", label="web-shell")
        if self._mode == "attach":
            return RoutedAgentClient(
                identity,
                self._service_a,
                remote_socket_path=agent_socket_path(self._config.database_path),
            )
        return AgentClient(identity, self._service_a)

    def start(self) -> None:
        if self._mode is not None:
            return
        self._mode = self._resolve_mode()
        client_id, _policy = self._config.agent_client

        # P15-S2-IPC: a SharedAgentService is single-threaded by construction
        # -- its request journal opens a SQLite connection eagerly in the
        # constructor -- so each service is constructed on the worker thread
        # that will execute it, never here. Workers come up first for exactly
        # that reason.
        self._worker_a = _SerialWorker("web-shell-authority")
        self._worker_b = _SerialWorker("web-shell-chat")
        self._worker_a.start()
        self._worker_b.start()
        build_service = self._config.service_factory or self._build_service
        try:
            self._service_a = self._worker_a.submit(build_service).result(
                timeout=120.0
            )
            self._service_b = self._worker_b.submit(
                self._build_chat_service
            ).result(timeout=120.0)
        except Exception as error:
            self.close()
            raise ShellStartupError(f"服务初始化失败（{error}）。") from error

        # Authority: embed hosts the routed tool boundary itself so preview /
        # device-safety capability works from a cold launch, with the same
        # components Runtime wires (socket server + observer, no automation).
        if self._mode == "embed":
            from music_agent.agent_socket import AgentSocketServer, agent_socket_path

            try:
                self._socket_server = AgentSocketServer(
                    agent_socket_path(self._config.database_path), self._service_a
                )
                self._socket_server.start()
            except Exception as error:
                self.close()
                raise ShellStartupError(
                    f"无法在本机绑定 agent socket（{error}）——可能已有其他实例在运行。"
                ) from error
            self._observer = self._build_audio_observer()
            self._worker_a.set_pump(self._socket_server.tick)
            logger.info(
                "web shell embedding authority at %s",
                agent_socket_path(self._config.database_path),
            )

        # Chat: always the production routed client shape (preview family over
        # the authority socket, everything else local); standalone is the
        # test-only direct wiring. The preview-event sink streams session
        # events to the shared log the browser polls.
        event_paths = None
        if self._mode in ("attach", "embed"):
            from music_agent.agent_socket import (
                AgentEventSocketListener,
                preview_event_socket_path,
            )

            event_socket_path = preview_event_socket_path(
                self._config.database_path, os.getpid()
            )
            self._event_listener = AgentEventSocketListener(
                event_socket_path, self._events.present
            )
            event_paths = (event_socket_path, self._event_listener)
        self._chat_client = self._build_chat_client(client_id, event_paths)
        self._ui_client = self._build_ui_client(client_id)

        # The loop (incl. provider construction) is built on the chat worker
        # too: it is only ever entered from there, so its whole life happens
        # on one thread.
        try:
            self._loop = self._worker_b.submit(self._build_loop).result(
                timeout=120.0
            )
        except Exception as error:
            self.close()
            raise ShellStartupError(f"对话循环初始化失败（{error}）。") from error

        self._httpd = ThreadingHTTPServer(
            (self._config.host, self._config.port), _shell_handler
        )
        self._httpd.daemon_threads = True
        self._httpd.app = self  # type: ignore[attr-defined]
        logger.info(
            "web shell listening at http://%s:%s (mode=%s)",
            self._config.host,
            self._httpd.server_port,
            self._mode,
        )

    def _build_chat_service(self) -> Any:
        """Worker-B-side constructor: the service plus the P15-S1 presenter
        hook (the same knob chat-session sets for console events)."""
        build_service = self._config.service_factory or self._build_service
        service = build_service()
        service.preview_event_handler = self._events.present
        return service

    def _build_loop(self) -> Any:
        from music_agent.provider_agent import ProviderAgentLoop, ProviderLoopConfig
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

        return ProviderAgentLoop(
            self._config.provider_factory(),
            self._chat_client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(
                max_tool_rounds=self._config.max_rounds,
                instrument=bool(self._config.trace_path),
            ),
        )

    def _build_chat_client(self, client_id: str, event_paths: object | None) -> Any:
        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentClientIdentity
        from music_agent.agent_socket import agent_socket_path
        from music_agent.routed_client import RoutedAgentClient

        identity = AgentClientIdentity(client_id=client_id, model_id="chat", label="web-shell")
        if self._mode == "standalone":
            return AgentClient(identity, self._service_b)
        event_socket_path, event_listener = event_paths  # type: ignore[misc]
        return RoutedAgentClient(
            identity,
            self._service_b,
            remote_socket_path=agent_socket_path(self._config.database_path),
            event_socket_path=event_socket_path,
            event_listener=event_listener,
        )

    def _build_audio_observer(self) -> Any:
        """Device-safety pump for the embedded authority (Runtime parity); the
        observer is detection-only and forwards to service_A's decision point.
        Degrades to None exactly like ``music-agent run`` does on machines
        without the reader."""

        from music_agent.audio_safety import (
            AudioOutputObserver,
            AudioSafetyUnavailableError,
            CoreAudioDefaultOutputReader,
        )

        try:
            reader = CoreAudioDefaultOutputReader()
        except AudioSafetyUnavailableError as error:
            logger.warning("audio-safety observer unavailable (%s); degraded", error)
            return None
        observer = AudioOutputObserver(
            reader, self._service_a, poll_interval_seconds=2.0
        )
        observer.start()
        return observer

    # -- HTTP model (handler threads enter here; work runs on the workers) ----

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "mode": self._mode,
            "server": "music-agent web shell",
        }

    def submit_chat(self, text: str) -> Future:
        assert self._worker_b is not None and self._loop is not None
        return self._worker_b.submit(lambda: self._run_chat(text))

    def _stop_family_fast_path(self, text: str) -> dict[str, Any] | None:
        # P19-T14-A: the stop family must truly stop the sounding Preview.
        #
        # The web shell had no deterministic fast path -- every line went to
        # the provider loop, whose stop gate read ``get_active_context`` from
        # the LOCAL chat-side service. That service never started the clip,
        # so the snapshot always said silent: the model saw no preview to
        # stop, called a Music.app-only pause (or nothing), and replied
        # success while the authority's afplay kept playing. Mirror
        # chat-session's fast path for the stop family only: route the line,
        # read the live Playback Context through the same routed client the
        # loop uses (the AUTHORITY runner truth), and when the route is
        # ``stop_preview`` execute it through that same client and answer
        # with the honest fixed sentences -- every reply is derived from the
        # authority's own stop result, never claimed. Resume semantics are
        # untouched: stop_preview itself disarms any AudioSuspension the
        # preview caused, exactly as the P15-S1 contract defines. P22-S1.12
        # extends this existing execution boundary to the other closed playback
        # controls (继续/下一首/上一首) instead of creating a second executor:
        # route through the same authoritative context, execute through the
        # same client, preserve the session/suspension gates, and never fall
        # back to Provider after a closed command has been recognized.
        from music_agent.agent_contract import AgentToolOutcome
        from music_agent.action_attempt import (
            render_verified_playback_control_result,
            run_playback_control_attempt,
        )
        from music_agent.intent_router import needs_active_context, route_intent
        from music_agent.markdown_presentation import render_assistant_markdown

        initially_routed: str | None = None
        closed_playback_command = False

        def fixed(reply: str) -> dict[str, Any]:
            return {
                "reply": reply,
                "reply_html": render_assistant_markdown(reply),
                "rounds_capped": False,
                "batch": None,
            }

        def context_unavailable_reply(tool: str | None) -> str:
            if tool == "pause":
                return "暂时无法确认当前播放或试听状态，未执行暂停。"
            if tool == "play":
                return "暂时无法确认当前播放或试听状态，未执行继续播放。"
            if tool == "next_track":
                return "暂时无法确认当前播放或试听状态，未执行下一首。"
            return "暂时无法确认当前播放状态，未执行该操作。"

        def operation_unconfirmed_reply(tool: str | None) -> str:
            if tool == "pause":
                return "暂停操作未能确认是否成功，当前播放或试听状态未确认。"
            if tool == "play":
                return "继续播放操作未能确认是否成功，当前播放状态未确认。"
            if tool == "next_track":
                return "下一首操作未能确认是否成功，当前播放或试听状态未确认。"
            if tool == "previous_track":
                return "上一首操作未能确认是否成功，当前播放状态未确认。"
            return "播放操作未能确认是否成功，当前状态未确认。"

        try:
            routed = route_intent(text)
            initially_routed = routed
            closed_playback_command = routed in {
                "pause",
                "play",
                "next_track",
                "previous_track",
            }
            suspended = None
            session: Mapping[str, Any] = {}
            session_state = None
            if needs_active_context(text):
                try:
                    result = self._loop.client.call("get_playback_context", {})
                except Exception:
                    if closed_playback_command:
                        return fixed(context_unavailable_reply(initially_routed))
                    return None
                if result.outcome is not AgentToolOutcome.OK:
                    if closed_playback_command:
                        return fixed(context_unavailable_reply(initially_routed))
                    return None
                payload = result.payload if isinstance(result.payload, Mapping) else {}
                channel = payload.get("channel")
                channel = channel if isinstance(channel, Mapping) else {}
                state = channel.get("state")
                channel_state = state if isinstance(state, str) else None
                sounding = payload.get("preview_sounding") is True
                session_value = payload.get("session")
                session = session_value if isinstance(session_value, Mapping) else {}
                state = session.get("state")
                session_state = state if isinstance(state, str) else None
                suspended_value = payload.get("suspended")
                suspended = (
                    suspended_value if isinstance(suspended_value, Mapping) else None
                )
                routed = route_intent(
                    text,
                    channel=channel_state,
                    preview_sounding=sounding,
                    preview_session_state=session_state,
                )
            if routed == "pause":
                attempt = run_playback_control_attempt(
                    "pause",
                    expected_state="paused",
                    invoke=self._loop.client.call,
                )
                reply = render_verified_playback_control_result(attempt)
            elif routed == "play":
                if session_state == "running":
                    position = session.get("position")
                    total = session.get("total")
                    if isinstance(position, int) and isinstance(total, int):
                        reply = f"连播进行中（第 {position}/共 {total} 首）。"
                    else:
                        reply = "连播进行中。"
                elif suspended is None:
                    reply = "没有可恢复的播放。"
                else:
                    attempt = run_playback_control_attempt(
                        "play",
                        expected_state="playing",
                        invoke=self._loop.client.call,
                    )
                    reply = render_verified_playback_control_result(attempt)
            elif routed in {"next_track", "previous_track"}:
                result = self._loop.client.call(routed, {})
                if result.outcome is not AgentToolOutcome.OK:
                    reply = (
                        "切换下一首失败，当前播放状态未确认。"
                        if routed == "next_track"
                        else "切换上一首失败，当前播放状态未确认。"
                    )
                else:
                    reply = (
                        "已切换到下一首。"
                        if routed == "next_track"
                        else "已切换到上一首。"
                    )
            elif routed == "advance_preview" and closed_playback_command:
                # Plain 下一首 during a RUNNING preview session belongs to the
                # preview authority, never Music.app. The tool's typed outcome
                # and payload are the existing terminal truth for this skip.
                result = self._loop.client.call("advance_preview", {})
                payload = result.payload if isinstance(result.payload, Mapping) else {}
                if result.outcome is not AgentToolOutcome.OK:
                    reply = "切换下一首试听失败，当前试听状态未确认。"
                elif payload.get("advanced") is True:
                    reply = "已切换到下一首试听。"
                elif payload.get("completed") is True:
                    reply = "试听连播已结束。"
                else:
                    reply = "下一首试听指令已执行，但当前试听状态未确认。"
            elif routed != "stop_preview":
                return None
            else:
                result = self._loop.client.call("stop_preview", {})
                payload = result.payload if isinstance(result.payload, Mapping) else {}
                if result.outcome is not AgentToolOutcome.OK:
                    if closed_playback_command:
                        reply = "暂停试听失败，当前试听状态未确认。"
                    else:
                        # Non-P22 stop-family forms retain their historical
                        # fallback behavior; S1.12 changes only the closed
                        # playback quartet.
                        return None
                elif payload.get("stopped") or payload.get("preview_session_cancelled"):
                    reply = "已停止试听。"
                else:
                    reply = "当前没有正在播放的试听。"
        except Exception:
            if closed_playback_command:
                return fixed(operation_unconfirmed_reply(initially_routed))
            # Non-P22 stop-family forms keep their historical fallback.
            return None
        return fixed(reply)

    def _direction_shift_fast_path(self, text: str) -> dict[str, Any] | None:
        # P20-Fix05: 再来一批换个方向 (and the closed shift family, plus a
        # mapped explicit direction word like 换成日系) must truly change the
        # recommendation direction -- the old path handed the line to the
        # provider, which only excluded the previous batch and kept the same
        # genre. The deterministic coach runs instead, zero provider rounds:
        # the previous batch's DURABLE direction is read, a real different
        # direction from the user's durable positive evidence is selected (or
        # the explicit word honored), one generate_inferred_recommendation
        # fires with genres=[new direction] AND the previous batch excluded,
        # and the reply presents what actually happened. Fail-honest by
        # construction: no alternative direction / unreadable state / failed
        # generation collapses to a fixed sentence (no fabrication) and any
        # non-shift line remits to the provider loop untouched. The batch
        # identity travels on the reply exactly like the loop path (newest
        # persisted run id compared against the cards source) so the UI's
        # structured takeover works on the same fetch.
        from music_agent.direction_coach import run_direction_shift
        from music_agent.markdown_presentation import render_assistant_markdown

        try:
            outcome = run_direction_shift(self._loop.client, text)
        except Exception:
            logger.exception("[shell] direction-shift executor failed")
            return None
        if outcome is None:
            return None
        if outcome.get("kind") == "reply":
            reply: str = outcome.get("text", "")
            return {
                "reply": reply,
                "reply_html": render_assistant_markdown(reply),
                "rounds_capped": False,
                "batch": None,
            }
        if outcome.get("kind") != "generated":
            return None
        # P20-Fix10: the shifted batch's items render through the SAME
        # deterministic presenter as a provider-loop recommendation success
        # (evidence-carrying reasons, the Fix09 copy discipline); the Fix05
        # direction note keeps its own line. A payload outside the post-Fix09
        # item contract fails open to the prior name-only lines (never
        # invented), keeping the two textual surfaces identical.
        from music_agent.recommendation_presenter import (
            render_recommendation_cue,
            render_recommendation_items,
        )

        lines = [outcome.get("note", "")]
        lines_text = render_recommendation_items(outcome)
        if lines_text is None:
            for position, item in enumerate(outcome.get("items", []), start=1):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                artist = item.get("artist_name")
                lines.append(
                    f"{position}. {name} — {artist}"
                    if isinstance(artist, str) and artist
                    else f"{position}. {name}"
                )
        else:
            lines.append("")
            lines.append(lines_text)
            cue = render_recommendation_cue(outcome)
            if cue is not None:
                lines.append("")
                lines.append(cue)
        reply = "\n".join(lines)
        batch: dict[str, Any] | None = None
        run_id = outcome.get("run_id")
        if isinstance(run_id, str) and run_id == self._cards.latest_run_id():
            batch = {
                "run_id": run_id,
                "cards": self._cards.latest_cards(),
            }
        return {
            "reply": reply,
            "reply_html": render_assistant_markdown(reply),
            "rounds_capped": False,
            "batch": batch,
        }

    def _explanation_fast_path(self, text: str) -> dict[str, Any] | None:
        # P20-Fix11: the closed explanation forms (为什么推荐这些？ /
        # 为什么这一批适合我？ / …) render deterministically from the
        # authoritative recommendation run through the shared presenter --
        # zero provider rounds, so the SSL traceback and planning-narration
        # UAT failures are structurally unreachable for these lines. The
        # active batch resolves through the household active-context rule,
        # one run read yields the evidence-carrying items, and the reply
        # travels plain (no batch takeover: an explanation never generates
        # one). Fail-honest: no active batch or an unreadable run prints the
        # fixed natural sentence; any other line remits to the provider loop
        # untouched. The Fix08 boundary still closes the presented text (the
        # deterministic copy passes byte-identical; the per-task explanation
        # fallback backs any impossible contamination).
        from music_agent.explanation_coach import run_recommendation_explanation
        from music_agent.final_response_boundary import present_final_text
        from music_agent.markdown_presentation import render_assistant_markdown

        try:
            outcome = run_recommendation_explanation(self._loop.client, text)
        except Exception:
            logger.exception("[shell] recommendation-explanation executor failed")
            return None
        if outcome is None:
            return None
        reply = outcome.get("text")
        if not isinstance(reply, str) or not reply:
            return None
        presented = present_final_text(reply, fallback_kind="explanation")
        return {
            "reply": presented,
            "reply_html": render_assistant_markdown(presented),
            "rounds_capped": False,
            "batch": None,
        }

    def _arm_preview_offer(
        self,
        target_canonical_id: str,
        *,
        source: str,
        verified_title: str | None = None,
        verified_artist: str | None = None,
    ) -> None:
        """Arm the one session-local target-bound Preview continuation."""
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        self._offered_actions.arm(
            OfferedAction(
                kind=PREVIEW_TRACK,
                target_canonical_id=target_canonical_id,
                source=source,
                verified_title=verified_title,
                verified_artist=verified_artist,
            )
        )

    def _continuation_fast_path(
        self, text: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve a pending offered action before TurnPlan/routing/provider.

        Acceptance/decline completes here.  An override expires the old offer;
        when it is a clearly delimited ``decline + substantive request`` the
        continuation layer removes only the decline clause and hands the new
        request to the existing semantic pipeline.  It never parses assistant
        prose or reconstructs target authority.
        """
        from music_agent.action_attempt import (
            create_direct_action_attempt,
            render_verified_action_result,
            run_action_attempt,
        )
        from music_agent.markdown_presentation import render_assistant_markdown

        decision = self._offered_actions.resolve(text)
        if decision.outcome == "none":
            return None, text
        if decision.outcome == "override":
            return None, decision.replacement_text or text
        if decision.outcome == "declined":
            reply = "好的，不试听了。"
        else:
            action = decision.action
            assert action is not None
            # S2.1 supports preview_track only.  The exact target comes from
            # the consumed OfferedAction; Provider/LLM does not re-resolve it.
            attempt = run_action_attempt(
                create_direct_action_attempt(
                    action.target_canonical_id,
                    route="preview_only",
                    title=action.verified_title,
                    artist=action.verified_artist,
                ),
                self._loop.client.call,
            )
            reply = render_verified_action_result(attempt)
        return ({
            "reply": reply,
            "reply_html": render_assistant_markdown(reply),
            "rounds_capped": False,
            "batch": None,
        }, text)

    def _track_reference_fast_path(self, text: str) -> dict[str, Any] | None:
        # P19-T14-F-R2: deterministic pronoun binding. T14-F made every
        # 它/他/她 spelling reach the same text, but the referent resolution
        # stayed model-side -- and the real-browser failure wandered into
        # recommendation generation (「暂时没有找到合适的推荐…」) instead
        # of previewing the current track. The pronoun now binds BEFORE any
        # provider round: the service's own channel register
        # (get_playback_context -> channel.canonical_id, the one existing
        # current/recent-track source) is the single authoritative referent.
        #
        # - referent present: 试听它 -> preview_catalog_track(id);
        #   播放它 -> play_track(id) only -- the T14-E contract is preserved
        #   by construction (play never degrades into a preview here).
        # - referent absent (channel none/malformed): one fixed honest
        #   question -- the provider never sees the line, so no
        #   recommendation tool/generation can be reached from it.
        # - read failed / agent offline / continuous session running: the
        #   referent is contested or unreadable -- fall back to the provider
        #   loop untouched (its session and offline rules own that surface).
        from music_agent.agent_contract import AgentToolOutcome
        from music_agent.action_attempt import (
            ActionAttemptStatus,
            create_direct_action_attempt,
            run_action_attempt,
        )
        from music_agent.intent_router import (
            PRONOUN_PLAY_ASK,
            PRONOUN_PREVIEW_ASK,
            PRONOUN_PREVIEW_START_REPLY,
            PRONOUN_PREVIEW_UNAVAILABLE_REPLY,
            PRONOUN_PLAY_START_REPLY,
            continuous_preview_session_running,
            pronoun_track_reference,
            pronoun_track_reference_target,
        )
        from music_agent.markdown_presentation import render_assistant_markdown
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        kind = pronoun_track_reference(text)
        if kind is None:
            return None
        try:
            result = self._loop.client.call("get_playback_context", {})
        except Exception:
            # unreadable live truth: refuse to fabricate a referent; fall back
            # to the loop, whose own reads report the failure their way.
            return None
        if result.outcome is not AgentToolOutcome.OK:
            return None
        payload = result.payload if isinstance(result.payload, Mapping) else {}
        if continuous_preview_session_running(payload):
            return None
        target_id = pronoun_track_reference_target(payload)
        if target_id is None:
            reply = PRONOUN_PREVIEW_ASK if kind == "preview" else PRONOUN_PLAY_ASK
            return {
                "reply": reply,
                "reply_html": render_assistant_markdown(reply),
                "rounds_capped": False,
                "batch": None,
            }
        if kind == "preview":
            attempt = run_action_attempt(
                create_direct_action_attempt(target_id, route="preview_only"),
                self._loop.client.call,
            )
            if attempt.status is ActionAttemptStatus.COMPLETED:
                reply = PRONOUN_PREVIEW_START_REPLY
            else:
                reply = PRONOUN_PREVIEW_UNAVAILABLE_REPLY
        else:
            attempt = run_action_attempt(
                create_direct_action_attempt(target_id, route="library"),
                self._loop.client.call,
            )
            if attempt.status is ActionAttemptStatus.COMPLETED:
                reply = PRONOUN_PLAY_START_REPLY
            else:
                self._arm_preview_offer(
                    target_id, source="direct_track_play_unavailable"
                )
                reply = PLAY_PREVIEW_DOWNGRADE_FALLBACK
        return {
            "reply": reply,
            "reply_html": render_assistant_markdown(reply),
            "rounds_capped": False,
            "batch": None,
        }

    def _run_chat(self, text: str) -> dict[str, Any]:
        # P19-T14-F: pronoun normalization at the intent/reference parse
        # boundary. 试听它 resolves through the live-track context today, but
        # the spoken variants 试听他/她 reach the model, which asks "which
        # song" instead of reusing the resolved track. The final-object
        # pronouns of track-reference verbs are rewritten to 它 BEFORE any
        # fast path, door, or provider round, so every variant reuses the
        # proven 它 path. Purely a text rewrite: routing (no 它-object form
        # is routed), the T14-B/E doors and the loop below all see the same
        # normalized line; nothing else about this turn changes.
        from music_agent.intent_router import (
            normalize_track_reference_pronouns,
            resolve_turn_plan,
        )

        text = normalize_track_reference_pronouns(text)
        # P22-S2.1: continuation arbitration must happen before TurnPlan, the
        # S1.12 playback fast path, or any provider planning.  A non-match only
        # expires the old offer and falls through to those existing owners.
        continuation, text = self._continuation_fast_path(text)
        if continuation is not None:
            return continuation
        # Production semantic ownership lives in ProviderAgentLoop.resolve_turn:
        # deterministic parser first, then the narrow P22 interpreter only for
        # unresolved turns.  Test/host stand-ins that do not implement the new
        # resolver retain the deterministic compatibility fallback.
        resolver = getattr(self._loop, "resolve_turn", None)
        turn_plan = (
            resolver(text) if callable(resolver) else resolve_turn_plan(text)
        )
        fast = self._track_reference_fast_path(text)
        if fast is not None:
            return fast
        fast = self._stop_family_fast_path(text)
        if fast is not None:
            return fast
        fast = self._direction_shift_fast_path(text)
        if fast is not None:
            return fast
        fast = self._explanation_fast_path(text)
        if fast is not None:
            return fast
        # P20 consolidation: the reply door consumes the SAME current-turn
        # recommendation expectation as provider routing. A recommendation-shaped request is only
        # ever answered by a real generated batch; when this run produced no
        # success, the final answer is replaced with ONE honest short sentence
        # -- the model's own reply (which otherwise degrades into a long
        # hand-enumerated "1. 2. 3." pseudo-list) is never relayed. The guard
        # lives here, at the harness boundary, not inside ProviderAgentLoop,
        # which stays provider-agnostic. The batch identity travels in this
        # reply so the UI can take over on the same fetch that carried it.
        # The successful generation payload is the ownership authority.
        from music_agent.final_response_boundary import (
            present_final_text,
            presentation_fallback_kind,
        )  # P20 Fix 08
        from music_agent.markdown_presentation import render_assistant_markdown  # P18-S1
        from music_agent.provider_agent import (
            PLAY_PREVIEW_DOWNGRADE_FALLBACK,
            RECOMMENDATION_UNFULFILLED_FALLBACK,
            action_result_play_preview_downgrade,
            action_result_preview_started,
            generation_succeeded,
            looks_like_numbered_song_list,
        )

        result = self._loop.run(text, turn_plan=turn_plan)
        # P22-S2.1 follow-up: a named-play Preview offer is armed only from
        # the structured per-turn result that owns its exact canonical target.
        # The visible offer text and pending authority therefore arrive
        # atomically; previous assistant prose is never parsed for identity.
        structured_offer = getattr(result, "offered_action", None)
        if structured_offer is not None:
            from music_agent.conversation_continuation import OfferedAction

            if isinstance(structured_offer, OfferedAction):
                self._offered_actions.arm(structured_offer)
        final_text = result.final_text
        generated = generation_succeeded(result.tool_executions)
        if not result.rounds_capped and not generated:
            # No batch this run. A closed recommendation intent, or a reply
            # the model shaped like a numbered song list, collapses to the one
            # honest sentence. Plain prose for non-recommendation turns stays
            # untouched: the door only shuts on the recommendation surface.
            if turn_plan.expects_recommendation_batch or looks_like_numbered_song_list(
                final_text
            ):
                final_text = RECOMMENDATION_UNFULFILLED_FALLBACK
        # P19-T14-E: a play-intent turn must never leave preview audio sounding.
        # The run's own execution record is the evidence (the same record the
        # B-door reads): a preview that started without any formal playback is
        # the explicit-play degradation the product forbids -- stop it through
        # the authoritative client and replace the reply with the one honest
        # sentence. A stray preview alongside successful formal playback is
        # also stopped (single audio source), but the run's answer stands.
        if turn_plan.explicit_play_intent and action_result_preview_started(result):
            try:
                self._loop.client.call("stop_preview", {})
            except Exception:
                logger.exception("[shell] play-turn preview stop failed")
            if action_result_play_preview_downgrade(text, result):
                # The provider run already has structured terminal truth for a
                # direct track action.  Preserve that exact canonical target in
                # the session before replacing the prose; never recover it from
                # the rendered sentence on the next turn.
                attempt = getattr(result, "action_attempt", None)
                target_id = getattr(attempt, "selected_canonical_id", None)
                expected_route = getattr(attempt, "expected_route", None)
                if (
                    isinstance(target_id, str)
                    and target_id
                    and expected_route == "preview_only"
                ):
                    self._arm_preview_offer(
                        target_id,
                        source="play_preview_downgrade",
                        verified_title=getattr(attempt, "selected_title", None),
                        verified_artist=getattr(attempt, "selected_artist", None),
                    )
                final_text = PLAY_PREVIEW_DOWNGRADE_FALLBACK
        batch: dict[str, Any] | None = None
        # The successful payload is the ownership proof: it was captured from
        # THIS ProviderAgentLoop invocation's actual generation execution, not
        # inferred from global history or a second text classifier. Search,
        # playback and library tool surfaces cannot generate; if a future/full
        # surface genuinely does produce a same-turn run, prose-only delivery
        # would be the wrong projection of that authoritative result.
        if generated:
            owned = result.recommendation_payload
            run_id = owned.get("run_id") if isinstance(owned, Mapping) else None
            if isinstance(run_id, str) and run_id:
                owned_id, cards = self._cards.latest_batch_state(run_id)
                batch = {
                    "run_id": owned_id,
                    "cards": cards,
                    "label": turn_plan.recommendation_label,
                } if owned_id is not None else None
        # P20 Fix 08: the final response boundary closes the presented text at
        # the web door too (Layer-1 scrubbing + fail-closed Layer-2 leakage
        # validation; the CLI door shares the same present_final_text). The
        # B/E doors above decided on the RAW text; only presentation passes
        # the boundary, and a contaminated text can only become its clean
        # region or the stable per-task fallback -- never the raw reply.
        kind = presentation_fallback_kind(text, result.tool_executions)
        presented = present_final_text(final_text, fallback_kind=kind)
        reply: dict[str, Any] = {
            "reply": presented,
            # Server-rendered presentation channel: escape-first whitelisted
            # Markdown. The UI may innerHTML THIS field only; user messages
            # and every other channel stay textContent.
            "reply_html": render_assistant_markdown(presented),
            "rounds_capped": bool(result.rounds_capped),
            "batch": batch,
        }
        return reply

    def player_snapshot(self) -> dict[str, Any]:
        """Cheap formal-player read for the 2s UI heartbeat.

        Preview/session truth is intentionally absent here. The browser merges
        only these four formal-player fields into its last full state, so the
        expensive ``get_playback_context`` read never gets synthesized as
        false simply because this fast endpoint did not ask for it.
        """
        assert self._worker_a is not None
        now: dict[str, Any] | None = None
        try:
            result = self._worker_a.submit(
                lambda: self._call_ui("get_now_playing", {})
            ).result(timeout=_STATUS_JOB_TIMEOUT_SECONDS)
            payload = getattr(result, "payload", None)
            if isinstance(payload, Mapping):
                now = dict(payload)
        except Exception:
            logger.exception("[shell] player status read failed")
        return {
            "player": ShellProjection.formal_player(now),
            "mode": self._mode,
        }

    def state_snapshot(self) -> dict[str, Any]:
        """Authoritative player/session state, read on the authority worker.
        Tool failures degrade to an honest unavailable state, never a 500:
        the strip must keep rendering while Music.app or the run is away."""
        assert self._worker_a is not None
        now_future = self._worker_a.submit(
            lambda: self._call_ui("get_now_playing", {})
        )
        context_future = self._worker_a.submit(
            lambda: self._call_ui("get_playback_context", {})
        )
        now: dict[str, Any] | None = None
        context: dict[str, Any] | None = None
        for future, target in ((now_future, "player"), (context_future, "context")):
            try:
                result = future.result(timeout=_STATUS_JOB_TIMEOUT_SECONDS)
                payload = getattr(result, "payload", None)
                if isinstance(payload, Mapping):
                    if target == "player":
                        now = dict(payload)
                    else:
                        context = dict(payload)
            except Exception:
                logger.exception("[shell] status read failed")
        return {
            "player": ShellProjection.now_playing(now, context),
            "mode": self._mode,
        }

    def latest_batch_state(self) -> dict[str, Any]:
        """{cards, run_id} for /api/cards from ONE authority-worker read.

        P19-T14-B: the run id travels with the cards so the UI decides batch
        freshness by identity, never by item-key equality (same items,
        different run = still a new batch). One submitted read keeps the id
        and its cards from disagreeing about the newest run.
        """
        assert self._worker_a is not None
        try:
            run_id, cards = self._worker_a.submit(
                self._cards.latest_batch_state
            ).result(timeout=30.0)
            return {"cards": cards, "run_id": run_id}
        except Exception:
            logger.exception("[shell] cards read failed")
            return {"cards": [], "run_id": None}

    def apple_music_target(self, canonical_id: str) -> dict[str, Any]:
        """Resolve the native handoff target on the authority worker, side-effect free."""
        assert self._worker_a is not None
        assert self._service_a is not None
        try:
            target = self._worker_a.submit(
                lambda: self._service_a.resolve_apple_music_target(canonical_id)
            ).result(timeout=60.0)
            return {"ok": True, "result": dict(target)}
        except Exception as error:
            logger.exception("[shell] Apple Music target resolution failed")
            return {
                "ok": False,
                "error": {
                    "code": getattr(error, "code", None) or "apple_music_target_failed",
                    "message": str(error) or "无法解析 Apple Music 曲目地址。",
                },
            }

    def run_command(self, tool: str, payload: Mapping[str, Any]) -> Any:
        assert self._worker_a is not None
        try:
            if tool in {"play_track", "preview_catalog_track"}:
                return self._worker_a.submit(
                    lambda: self._run_verified_track_command(tool, dict(payload))
                ).result(timeout=300.0)
            if tool in {"play", "pause"}:
                return self._worker_a.submit(
                    lambda: self._run_verified_playback_control(tool)
                ).result(timeout=300.0)
            return self._worker_a.submit(
                lambda: self._call_ui(tool, dict(payload))
            ).result(timeout=300.0)
        except Exception as error:
            logger.exception("[shell] command %s failed", tool)
            return _synthetic_command_failure(tool, str(error))

    def _run_verified_track_command(
        self, tool: str, payload: Mapping[str, Any]
    ) -> Any:
        """Run one Web track command through the shared terminal workflow."""

        from music_agent.action_attempt import (
            create_direct_action_attempt,
            run_action_attempt,
        )

        route = "library" if tool == "play_track" else "preview_only"
        attempt = create_direct_action_attempt(
            payload["canonical_id"], route=route
        )
        return run_action_attempt(attempt, self._call_ui)

    def _run_verified_playback_control(self, tool: str) -> Any:
        """Run play/pause and confirm the resulting authoritative player state."""

        from music_agent.action_attempt import run_playback_control_attempt

        return run_playback_control_attempt(
            tool,
            expected_state="playing" if tool == "play" else "paused",
            invoke=self._call_ui,
        )

    def _call_ui(self, tool: str, payload: Mapping[str, Any]) -> Any:
        attempt = getattr(self._ui_client, "call")
        return attempt(tool, payload)

    def events_since(self, after: int) -> dict[str, Any]:
        entries = self._events.snapshot(after)
        return {
            "events": entries,
            "next": entries[-1]["seq"] if entries else after,
        }

    # -- lifecycle -------------------------------------------------------------

    @property
    def port(self) -> int:
        assert self._httpd is not None
        return self._httpd.server_port

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def request_shutdown(self) -> None:
        """Browser-initiated exit: kick the stopper on a detached thread (the
        serving thread must never shutdown the server it serves)."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        threading.Thread(target=self._stop_server, daemon=True).start()

    def _stop_server(self) -> None:
        self.close()
        self.stop_httpd()

    def stop_httpd(self) -> None:
        """Unblock ``serve_forever`` from outside its own thread (signal path),
        then release the listening socket (idempotent: a second call returns
        immediately)."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    def serve_forever(self) -> None:
        assert self._httpd is not None
        self._httpd.serve_forever()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Services release on their own worker threads: the journal connection
        # is thread-affine, so even the close must happen on the constructing
        # side. Jobs drain in FIFO order before ``stop``, so a latex close job
        # waits behind any in-flight command/chat (bounded by the timeout).
        for worker, service in (
            (self._worker_a, self._service_a),
            (self._worker_b, self._service_b),
        ):
            if worker is None or service is None:
                continue
            try:
                worker.submit(service.close).result(timeout=5.0)
            except Exception:
                logger.exception("[shell] service close job failed")
        for worker in (self._worker_a, self._worker_b):
            if worker is not None:
                worker.stop()
        if self._event_listener is not None:
            try:
                self._event_listener.close()
            except Exception:
                logger.exception("[shell] event listener close failed")
        if self._observer is not None:
            try:
                self._observer.close()
            except Exception:
                logger.exception("[shell] audio observer close failed")
        if self._socket_server is not None:
            try:
                self._socket_server.close()
            except Exception:
                logger.exception("[shell] socket server close failed")
        # Services were already released on their workers above; the refs are
        # dropped here only (a main-thread close would violate the thread
        # affinity of their SQLite connections).
        self._service_a = None
        self._service_b = None
        logger.info("web shell closed")


def _shell_handler(
    request: Any, client_address: Any, server: Any
) -> _ShellHttpHandler:
    """Per-request handler factory for the ThreadingHTTPServer (``server.app``
    carries the WebShellApp every handler delegates to)."""
    return _ShellHttpHandler(request, client_address, server)


def _synthetic_command_failure(tool: str, detail: str) -> Any:
    from types import SimpleNamespace

    from music_agent.agent_contract import AgentToolOutcome

    return SimpleNamespace(
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        error_code="command_failed",
        error_message=f"{tool} 执行失败：{detail}",
        payload=None,
    )
