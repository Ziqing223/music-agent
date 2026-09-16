"""P15-S2-IPC S3: RoutedAgentClient -- the chat-facing request boundary that routes
the Preview/Device-Safety tool family to the run-hosted agent socket and serves
everything else through the process-local shared service (decision b + D1/D3/D4).

Routing table (owner-approved):
    remote (run authority)   preview_batch, preview_catalog_track, stop_preview,
                             advance_preview, get_playback_context, play
    local (local service)    every other tool: recommendation, feedback, search,
                             pause / next_track / previous_track / play_track,
                             get_active_context / get_now_playing, ...

``play`` rides remotely because the owner's D3 amendment makes the continue /
restore semantic observable to run (the authority on AudioSuspension): one
execute boundary, zero ambiguity between a restore play and a plain play. The
D3 minimal audit stands behind the rest of the split -- ``_execute_play`` and
``_execute_pause`` carry no PlaybackContext / AudioSuspension / PreviewSession
side effects on either side, so remote execution changes no P15-PC semantics
and local execution creates no authority split for the formal controls.

Offline semantics (D4): a remote tool whose run socket is UNREACHABLE --
connect refused or the request frame undeliverable -- returns a VALID
AgentToolResult with ``outcome=EXECUTION_ERROR`` and
``error_code="agent_runtime_offline"``. Preview/Device-Safety capability is
never degraded to a local answer: no session is started here, and no
``session``/``suspended`` fact is invented ("run offline => no session exists"
licenses *refusing*, not answering). The refusal is an ordinary tool result,
so the provider loop surfaces it as a normal tool error and the conversation
stays alive; ``_fetch_routing_context`` absorbs it through its existing
unknown-context fail-closed shape (宁可不路由，不误触发).

S3 repair (live Broken-pipe diagnosis): a request that WAS delivered but got
no usable response -- a read timeout, a dropped link, a garbage reply -- is a
DIFFERENT outcome class: ``agent_response_lost``. The runtime was reachable
when the request went out, so the execution outcome is unknown and side
effects may already exist; this code must never trigger a blind re-issue of
the same tool (fail closed, check-state-first guidance in the message).
There is no transport-level retry anywhere -- one request, one attempt.

The request timeout defaults to 120s, sized for the slowest honest execute:
``preview_batch`` starts its first clip synchronously inside the run-side
execute (catalog lookup + full 30s-clip download, the download itself bounded
at 60s by the runner), so a short client budget would abandon a healthily
executing request mid-flight and misread its late response as an error.

S4: preview events stream back to the chat side over the event channel. A
client built with ``event_socket_path`` (plus a lazy listener bound just
before the first remote preview call) stamps that path into every preview-
family payload under the transport-only ``event_socket`` key; run peels it
off, arms its forwarder, and pushes ``{event, session, suspended}`` back to
the chat presenter. The listener is chat-process-owned and bound lazily, so
a chat process that never previews never binds; if the bind fails the
remote call still executes -- only event printing is lost, never the audio.

P20-Fix01 (client identity consistency): a reachable run whose permission
gate refuses the calling client (its id is missing from run's OWN
``--agent-client`` registrations -- the chat-side registration never flows
to the run) answers a normal UNKNOWN_CLIENT envelope, and this boundary
translates it into the stable ``agent_client_not_registered`` envelope with
a natural guidance message. The run-side gate is untouched -- nothing is
executed, no identity is swapped or invented, and offline remains the
distinct ``agent_runtime_offline`` surface it always was.

The class subclasses :class:`AgentClient` (the provider loop's type guard
accepts it unchanged, and the local half keeps the exact constructor
validation, journal/replay and permission chain), then overrides only
``call``: validate, route, and delegate the local remainder to the inherited
implementation -- the local path is byte-for-byte the P09 client.

S3D/E internal context: ``call`` also accepts the base client's internal
recommendation execution context and forwards it verbatim down the local delegate path
(it lives outside ``payload``, so it has no wire representation and never
travels remotely -- generation tools are local-only routing entries). The
real chat-session binds the provider loop to this class, so the override
signature must stay a superset of the base client's; a truthy Fresh set aimed
at a remote tool is refused fail-closed as an internal invariant violation.
"""

from __future__ import annotations

import atexit
import logging
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from music_agent.agent_client import AgentClient, AgentClientValidationError
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentContractValidationError,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
    generate_request_id,
)
from music_agent.agent_service import SharedAgentService
from music_agent.agent_socket import (
    AGENT_CLIENT_NOT_REGISTERED_CODE,
    AGENT_CLIENT_NOT_REGISTERED_HINT,
    EVENT_SOCKET_KEY,
    PREVIEW_EVENT_TOOL_NAMES,
    AgentResponseLostError,
    AgentSocketError,
    SocketFrameError,
    build_offline_refusal,
    build_response_lost_result,
    frame_to_result,
    receive_framed,
    request_to_frame,
    send_framed,
)
from music_agent.agent_tools import AgentToolName
from music_agent.track_similarity import SimilarityExecutionContext

logger = logging.getLogger("music_agent.routed_client")

# The Preview/Device-Safety family -- run is the sole authority on
# PreviewSessionRegistry / AudioSuspension / AfplayPreviewRunner, so every
# tool that reads, consumes or changes that state executes remotely (D3),
# plus ``play`` as the one carrier of the continue/restore semantic. The
# preview half of the table is also the event-channel family: exactly those
# remote calls carry the ``event_socket`` sidecar (S4).
REMOTE_TOOL_NAMES: frozenset[str] = frozenset(
    PREVIEW_EVENT_TOOL_NAMES
    | {
        AgentToolName.GET_PLAYBACK_CONTEXT.value,
        AgentToolName.PLAY.value,
    }
)


class RoutedAgentClient(AgentClient):
    """One chat-bound client: remote run for the preview/device-safety family,
    the local shared service for everything else."""

    def __init__(
        self,
        client: AgentClientIdentity,
        service: SharedAgentService,
        *,
        remote_socket_path: Path,
        timeout: float = 120.0,
        event_socket_path: Path | None = None,
        event_listener: object | None = None,
    ) -> None:
        super().__init__(client, service)  # the same validation as the plain client
        if not isinstance(remote_socket_path, Path) or not remote_socket_path.parts:
            raise AgentClientValidationError("remote_socket_path must be a non-empty Path")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise AgentClientValidationError("timeout must be a positive number of seconds")
        if event_socket_path is not None and (
            not isinstance(event_socket_path, Path) or not event_socket_path.parts
        ):
            raise AgentClientValidationError("event_socket_path must be a non-empty Path or None")
        self._remote_socket_path = remote_socket_path
        self._timeout = float(timeout)
        self._event_socket_path = event_socket_path
        self._event_listener = event_listener
        self._event_listener_bound = False
        self._event_atexit_hooked = False

    @property
    def remote_socket_path(self) -> Path:
        return self._remote_socket_path

    @property
    def event_socket_path(self) -> Path | None:
        """The chat-side event sink path (``None`` = no event channel)."""
        return self._event_socket_path

    def close_event_listener(self) -> None:
        """Chat-process teardown for the event sink. Idempotent; a later
        preview call re-binds it lazily."""
        listener = self._event_listener
        if listener is not None:
            listener.close()  # type: ignore[attr-defined]
        self._event_listener_bound = False

    def _ensure_event_listener(self) -> bool:
        """Bind the chat-side event sink just before the first remote preview
        call -- a chat process that never previews never binds. A failed bind
        degrades to "no event channel": the remote call still executes, its
        events are simply not surfaced."""
        if self._event_socket_path is None or self._event_listener is None:
            return False
        if self._event_listener_bound:
            return True
        try:
            self._event_listener.start()  # type: ignore[attr-defined]
        except AgentSocketError:
            logger.warning(
                "preview event listener could not bind at %s", self._event_socket_path
            )
            return False
        self._event_listener_bound = True
        logger.info(
            "[preview-ipc] chat: event listener bound at %s", self._event_socket_path
        )
        if not self._event_atexit_hooked:
            atexit.register(self._event_listener.close)  # type: ignore[attr-defined]
            self._event_atexit_hooked = True
        return True

    def call(
        self,
        tool: str | AgentToolName,
        payload: Mapping[str, Any],
        *,
        request_id: str | None = None,
        issued_at: datetime | None = None,
        completed_at: str | None = None,
        fresh_canonical_ids: tuple[str, ...] | None = None,
        recommendation_scope_ids: tuple[str, ...] | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> AgentToolResult:
        """Invoke one agent tool: the remote family over the run socket, the
        rest through the inherited local service call.

        ``request_id`` / ``issued_at`` keep the AgentClient contract (client-owned
        idempotency keys; deterministic tests inject both). ``completed_at``
        only reaches the local path, exactly as before.

        P15-S3-S3D: ``fresh_canonical_ids`` is the same internal same-run Fresh
        context the base client carries -- call-side runtime provenance, never
        model input and never part of ``payload``. Generation
        tools are local-only tools (they are not in REMOTE_TOOL_NAMES), so the
        set travels the local delegate path to the service verbatim; it has no
        wire representation, and a truthy set aimed at a remote tool is an
        internal invariant violation, refused fail-closed.

        ``similarity_context`` follows the same local-only transport boundary.
        The service records its seed discriminator in the durable request journal
        solely to keep replay identity seed-bound.
        """
        if isinstance(tool, AgentToolName):
            tool = tool.value
        if not isinstance(tool, str) or tool == "":
            raise AgentClientValidationError("tool must be a non-empty string or AgentToolName")
        if not isinstance(payload, Mapping):
            raise AgentClientValidationError("payload must be a JSON object")
        if tool in REMOTE_TOOL_NAMES:
            if (
                fresh_canonical_ids
                or recommendation_scope_ids is not None
                or similarity_context is not None
            ):
                raise AgentClientValidationError(
                    "recommendation execution context has no remote wire representation; "
                    "it is only valid for local generation tools and "
                    f"cannot accompany a remote call to {tool!r}"
                )
            return self._call_remote(tool, payload, request_id=request_id, issued_at=issued_at)
        return super().call(
            tool,
            payload,
            request_id=request_id,
            issued_at=issued_at,
            completed_at=completed_at,
            fresh_canonical_ids=fresh_canonical_ids,
            recommendation_scope_ids=recommendation_scope_ids,
            similarity_context=similarity_context,
        )

    def _call_remote(
        self,
        tool: str,
        payload: Mapping[str, Any],
        *,
        request_id: str | None,
        issued_at: datetime | None,
    ) -> AgentToolResult:
        if issued_at is None:
            issued_at = datetime.now(timezone.utc)
        wire_payload: Mapping[str, Any] = payload
        if tool in PREVIEW_EVENT_TOOL_NAMES and self._ensure_event_listener():
            # S4: stamp the event sidecar into preview-family payloads only;
            # run peels it off before the service ever sees the request.
            wire_payload = dict(payload)
            wire_payload[EVENT_SOCKET_KEY] = str(self._event_socket_path)
            logger.debug(
                "[preview-ipc] chat: event sidecar stamped into %s -> %s",
                tool,
                self._event_socket_path,
            )
        request = AgentRequest(
            request_id=request_id or generate_request_id(),
            client=self._client,
            tool=tool,
            payload=wire_payload,
            issued_at=issued_at,
        )
        try:
            frame = self._exchange(request)
        except AgentResponseLostError as error:
            logger.warning(
                "routed call %s: response lost after delivery (%s) -- "
                "outcome unknown, no auto-retry",
                tool,
                error,
            )
            return build_response_lost_result(request)
        except (AgentSocketError, OSError) as error:
            logger.warning("routed call %s: runtime unreachable (%s)", tool, error)
            return build_offline_refusal(request)
        try:
            result = frame_to_result(frame)
        except AgentContractValidationError as error:
            # The peer answered garbage: the capability is unusable, and the
            # honest surface is the same fail-closed response-lost refusal,
            # never a crash and never a fabricated offline fact.
            logger.warning("routed call %s: unusable reply (%s)", tool, error)
            return build_response_lost_result(request)
        if result.outcome is AgentToolOutcome.UNKNOWN_CLIENT:
            # P20-Fix01: the run-side permission gate REFUSED this client
            # (its id is absent from run's own --agent-client registrations).
            # The gate stays exactly as it is -- nothing is executed, no
            # policy is bridged -- only the surface is translated: the model
            # and the user get the stable natural not-registered envelope,
            # never the internal ``unknown_client`` registry sentence. The
            # distinct code keeps runtime-offline and client-not-registered
            # separable for every consumer (fast paths, traces, diagnostics).
            logger.warning(
                "routed call %s: run refused the session client %r "
                "(not in run's --agent-client registrations) -- translated to %s",
                tool,
                self._client.client_id,
                AGENT_CLIENT_NOT_REGISTERED_CODE,
            )
            return AgentToolResult(
                request_id=result.request_id,
                tool=result.tool,
                outcome=AgentToolOutcome.EXECUTION_ERROR,
                payload=None,
                error_code=AGENT_CLIENT_NOT_REGISTERED_CODE,
                error_message=AGENT_CLIENT_NOT_REGISTERED_HINT,
                completed_at=result.completed_at,
                contract_version=result.contract_version,
                replayed=result.replayed,
            )
        return result

    def _exchange(self, request: AgentRequest) -> bytes:
        """One synchronous request over one fresh socket (D5: single request,
        single response), with the outcome phase made explicit.

        Failures BEFORE the request frame leaves the wire (connect refused,
        an undeliverable frame) mean the runtime was unreachable -- callers
        map those to the offline refusal. Every failure AFTER full delivery
        (read timeout, dropped link, empty reply) raises
        :class:`AgentResponseLostError` instead: the request may already be
        executing and its result is unknown.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            try:
                sock.connect(str(self._remote_socket_path))
            except OSError as error:
                raise AgentSocketError(f"agent socket connect failed: {error}") from error
            logger.debug(
                "[agent-ipc] chat: connected req=%s fd=%s thr=%s remote=%s",
                request.request_id,
                sock.fileno(),
                threading.current_thread().name,
                self._remote_socket_path,
            )
            try:
                frame = request_to_frame(request)
                send_framed(sock, frame)
            except (AgentSocketError, OSError) as error:
                raise AgentSocketError(f"request frame delivery failed: {error}") from error
            logger.debug(
                "[agent-ipc] chat: request sent req=%s fd=%s bytes=%s",
                request.request_id,
                sock.fileno(),
                len(frame),
            )
            # Delivered: from here on the run may be executing -- every
            # failure is a lost response, never an offline signal.
            logger.debug(
                "[agent-ipc] chat: recv waiting req=%s fd=%s thr=%s",
                request.request_id,
                sock.fileno(),
                threading.current_thread().name,
            )
            try:
                frame = receive_framed(sock)  # None on a dropped link
            except (SocketFrameError, OSError) as error:
                raise AgentResponseLostError(
                    f"response never arrived: {error}"
                ) from error
            if frame is None:
                raise AgentResponseLostError("connection closed before a response frame")
            logger.debug(
                "[agent-ipc] chat: response received req=%s fd=%s bytes=%s",
                request.request_id,
                sock.fileno(),
                len(frame),
            )
            return frame
        finally:
            sock.close()
            logger.debug(
                "[agent-ipc] chat: socket closed req=%s fd=%s",
                request.request_id,
                sock.fileno(),
            )
