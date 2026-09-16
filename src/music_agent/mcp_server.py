"""MCP M1-M4: the local stdio MCP adapter over SharedAgentService.execute().

Contract-frozen (MUSIC_AGENT_MCP_CONTRACT_v0.2.1). The adapter is transport
plus identity only; every domain fact stays behind the existing boundary:

* boundary -- every allowlisted domain call enters through the existing agent
  boundary; the adapter contains no domain logic, opens no SQLite, and
  imports no repository (architectural acceptance test, contract §20);
* transport -- newline-delimited JSON-RPC over stdio, local machine only
  (§3.1); the startup surface is ``python -m music_agent.mcp_server``;
* identity -- the wrapper owns the ``agt_`` client identity (validator-shaped,
  generated via ``generate_client_id()`` unless one is injected) and injects
  its policy into the in-memory AgentClientRegistry at construction time --
  never durable state, never model-derived (§4);
* request identity -- one fresh ``req_<uuid4>`` per tools/call, minted by the
  existing AgentClient boundary; the service journal/replay contract is
  preserved untouched: a first-seen request appends one append-only journal
  row, a verbatim replay returns the stored result marked ``replayed``
  without a new row (§5.1/§6);
* routing -- the exact six-tool routed set (preview_batch,
  preview_catalog_track, stop_preview, advance_preview, get_playback_context,
  play) executes through the existing RoutedAgentClient / run-hosted UDS
  authority and fails closed as ``agent_runtime_offline`` (a valid envelope
  with outcome execution_error) when run is not serving; pause / next_track /
  previous_track / play_track and every other tool stay process-local.
  advance_preview is absent from the MCP surface entirely (§14). The MCP
  process never binds a preview-event sink, never starts a runtime pump, and
  never degrades a routed call to a local answer -- preview/session progress
  stays readable through get_playback_context and the preview_batch result
  snapshot;
* results -- the canonical AgentToolResult envelope passes through verbatim
  via ``encode_agent_tool_result``: complete payload, never truncated, never
  rewritten; every non-OK outcome rides the MCP tool result with ``isError``
  true so a refusal can never masquerade as success (§6/§15).

No MCP code retries: the ``agent_runtime_offline`` and ``agent_response_lost``
envelopes of the shared routed client pass through field-by-field (§15.2),
and the wrapper mints exactly one request identity per tools/call (§16).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    AgentToolResult,
    encode_agent_tool_result,
    generate_client_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.agent_socket import agent_socket_path
from music_agent.mcp_projection import (
    MCP_EXPOSABLE_TOOL_NAMES,
    build_mcp_tool_definitions,
)
from music_agent.routed_client import RoutedAgentClient

logger = logging.getLogger("music_agent.mcp_server")

DEFAULT_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS: frozenset[str] = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18"}
)
DEFAULT_MODEL_ID = "mcp-host"
DEFAULT_LABEL = "mcp"

_SERVER_NAME = "music-agent-core"
_SERVER_VERSION = "0.1.0"

# Constructor sentinel: "build the same production adapter chain the
# chat-session boundary uses". Tests inject fakes or None; the production
# default never runs inside a test.
_DEFAULT_ADAPTER = object()


class McpServerError(ValueError):
    code = "mcp_server_error"


class McpServerValidationError(McpServerError):
    code = "validation_error"


class _JsonRpcError(Exception):
    """An internal JSON-RPC error code for the serve loop (never leaked as an
    agent outcome -- the domain layer never saw the misformed message)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def map_tool_result(result: AgentToolResult) -> dict[str, Any]:
    """Project one agent envelope into the MCP CallToolResult shape.

    The complete canonical envelope (payload included -- no truncation, no
    rewrite, ``replayed`` and error layers intact) is the single text content;
    every non-OK outcome flags ``isError`` so a refusal can never be rewritten
    into an apparent success.
    """
    if not isinstance(result, AgentToolResult):
        raise McpServerValidationError("result must be an AgentToolResult")
    return {
        "content": [{"type": "text", "text": encode_agent_tool_result(result)}],
        "isError": result.outcome is not AgentToolOutcome.OK,
    }


class MusicAgentMcpServer:
    """The thin stdio MCP adapter: one frozen surface, one in-memory client,
    the existing local-service / routed-client execution boundary."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        client_id: str | None = None,
        client_policy: str | AgentClientPolicy = AgentClientPolicy.FULL,
        model_id: str = DEFAULT_MODEL_ID,
        label: str = DEFAULT_LABEL,
        remote_socket_path: Path | str | None = None,
        playback_adapter: object = _DEFAULT_ADAPTER,
        playback_resolver: object = _DEFAULT_ADAPTER,
        catalog_search_source: object = _DEFAULT_ADAPTER,
        tool_definitions: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if isinstance(database_path, str):
            database_path = Path(database_path)
        if not isinstance(database_path, Path) or not database_path.parts:
            raise McpServerValidationError("database_path must be a non-empty Path")
        if client_id is None:
            client_id = generate_client_id()
        if not isinstance(client_policy, AgentClientPolicy):
            try:
                client_policy = AgentClientPolicy(client_policy)
            except (TypeError, ValueError) as error:
                raise McpServerValidationError(
                    f"client_policy must be one of {[p.value for p in AgentClientPolicy]}"
                ) from error
        registry = AgentClientRegistry({client_id: client_policy})
        if playback_adapter is _DEFAULT_ADAPTER:
            playback_adapter = _build_default_playback_adapter()
        if playback_resolver is _DEFAULT_ADAPTER:
            playback_resolver = _build_default_playback_resolver()
        if catalog_search_source is _DEFAULT_ADAPTER:
            catalog_search_source = _build_default_catalog_search_source()
        service = SharedAgentService(
            database_path,
            clients=registry,
            playback_adapter=playback_adapter,
            playback_resolver=playback_resolver,
            catalog_search_source=catalog_search_source,
        )
        identity = AgentClientIdentity(client_id=client_id, model_id=model_id, label=label)
        remote_path = (
            Path(remote_socket_path)
            if remote_socket_path is not None
            else agent_socket_path(database_path)
        )
        self._service = service
        self._client = RoutedAgentClient(identity, service, remote_socket_path=remote_path)
        if tool_definitions is None:
            self._definitions = build_mcp_tool_definitions()
        else:
            self._definitions = tuple(dict(definition) for definition in tool_definitions)
        if not self._definitions:
            raise McpServerValidationError("the MCP tool surface must be non-empty")
        self._closed = False

    # --- observability (tests / diagnostics) --------------------------------

    @property
    def client_id(self) -> str:
        return self._client.client.client_id

    @property
    def client(self) -> RoutedAgentClient:
        return self._client

    @property
    def service(self) -> SharedAgentService:
        return self._service

    @property
    def tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return self._definitions

    # --- domain invocation (the only path out of the adapter) ---------------

    def invoke_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> AgentToolResult:
        """Invoke one allowlisted tool through the existing agent boundary.

        ``request_id`` is the wrapper-owned idempotency key: omitted (the
        normal MCP path), the existing client boundary mints a fresh
        ``req_<uuid4>``; reusing a value exercises the service journal replay
        contract unchanged. Names outside the frozen allowlist are refused
        here, before the domain layer -- DO_NOT_EXPOSE is enforced by
        construction, not by the model's good behavior.
        """
        if not isinstance(name, str) or name not in MCP_EXPOSABLE_TOOL_NAMES:
            raise McpServerValidationError(
                f"unknown or unexposed tool: {name!r} (the MCP surface is the "
                "contract-frozen allowlist; LIVE_WRITE and CLI-only tools are "
                "not callable through MCP)"
            )
        if not isinstance(arguments, Mapping) or any(
            not isinstance(key, str) for key in arguments
        ):
            raise McpServerValidationError("arguments must be a JSON object")
        return self._client.call(name, dict(arguments), request_id=request_id)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One MCP tools/call -> the wire CallToolResult (invocation + mapping)."""
        return map_tool_result(self.invoke_tool(name, arguments))

    # --- MCP protocol surface -----------------------------------------------

    def initialize(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The MCP initialize handshake: echo a supported client version."""
        requested = (params or {}).get("protocolVersion")
        version = (
            requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        )
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": _SERVER_NAME,
                "title": "Music Agent",
                "version": _SERVER_VERSION,
            },
        }

    def list_tools(self) -> dict[str, Any]:
        """The stable tools/list payload (frozen surface, provider order)."""
        return {"tools": list(self._definitions)}

    def handle_message(
        self, message: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """One parsed JSON-RPC message -> ``(response, should_exit)``.

        ``None`` response = a notification (no reply). ``should_exit`` flips
        only on the MCP ``exit`` method; EOF also ends the serve loop.
        """
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params")
        if not isinstance(params, Mapping):
            params = {}
        if isinstance(method, str) and method.startswith("notifications/"):
            return None, False
        if method == "initialize":
            return _jsonrpc_success(request_id, self.initialize(params)), False
        if method == "ping":
            return _jsonrpc_success(request_id, {}), False
        if method == "tools/list":
            return _jsonrpc_success(request_id, self.list_tools()), False
        if method == "tools/call":
            return _jsonrpc_success(request_id, self._handle_tools_call(params)), False
        if method == "exit":
            # MCP 2025-06-18: the server exits on receipt without a response.
            return None, True
        if not isinstance(method, str) or method == "":
            raise _JsonRpcError(-32600, "Invalid Request")
        raise _JsonRpcError(-32601, f"Method not found: {method!r}")

    def _handle_tools_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or name == "":
            raise McpServerValidationError("tools/call requires a non-empty string 'name'")
        arguments = params.get("arguments", {})
        if arguments is None:
            # Transport-level tolerance only: an absent arguments object is an
            # empty payload; the existing validators still see and reject any
            # unknown key the model injects (never coerced away).
            arguments = {}
        return self.call_tool(name, arguments)

    # --- stdio lifecycle ------------------------------------------------------

    def serve(self, reader: Iterable[bytes], writer: Callable[[bytes], None]) -> bool:
        """Serve newline-delimited JSON-RPC until EOF or ``exit``.

        ``reader`` yields one raw message line per iteration (``sys.stdin.
        buffer`` qualifies); ``writer`` receives one encoded response line per
        reply. A malformed line never kills the loop -- the parse error is
        answered (-32700) and the next line is served.
        """
        for raw in reader:
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if not isinstance(raw, (bytes, bytearray)):
                writer(_error_line(None, -32700, "Parse error"))
                continue
            try:
                text = bytes(raw).decode("utf-8").strip()
            except UnicodeDecodeError:
                writer(_error_line(None, -32700, "Parse error"))
                continue
            if not text:
                continue
            try:
                message = json.loads(text)
            except (TypeError, ValueError):
                writer(_error_line(None, -32700, "Parse error"))
                continue
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                request_id = message.get("id") if isinstance(message, dict) else None
                writer(_error_line(request_id, -32600, "Invalid Request"))
                continue
            try:
                response, should_exit = self.handle_message(message)
            except _JsonRpcError as error:
                response, should_exit = (
                    _jsonrpc_error(message.get("id"), error.code, error.message),
                    False,
                )
            except McpServerValidationError as error:
                response, should_exit = (
                    _jsonrpc_error(message.get("id"), -32602, str(error)),
                    False,
                )
            except Exception as error:  # fail closed; the loop stays alive
                logger.warning("mcp internal error on %r: %s", message.get("method"), error)
                response, should_exit = (
                    _jsonrpc_error(message.get("id"), -32603, f"Internal error: {error!r}"),
                    False,
                )
            if response is not None:
                writer((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            if should_exit:
                return True
        return False

    def serve_stdio(self) -> None:
        """Serve on the process stdio until EOF or exit; always closes."""
        try:
            self.serve(sys.stdin.buffer, _write_stdout)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._service.close()


# --- defaults (the same production chain the chat-session boundary uses) -----


def _build_default_playback_adapter() -> object:
    from music_agent.playback_control import MusicPlaybackAdapter, OsascriptPlaybackRunner

    return MusicPlaybackAdapter(OsascriptPlaybackRunner(timeout_seconds=60.0))


def _build_default_playback_resolver() -> object:
    from music_agent.playback_control import (
        MusicLibraryResolver,
        OsascriptLibraryResolveRunner,
    )

    return MusicLibraryResolver(OsascriptLibraryResolveRunner(timeout_seconds=60.0))


def _build_default_catalog_search_source() -> object:
    from music_agent.catalog_ingestion import default_catalog_search_source

    return default_catalog_search_source()


# --- JSON-RPC wire helpers ----------------------------------------------------


def _jsonrpc_success(request_id: object, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: object, code: int, message: str) -> dict[str, Any]:
    if not isinstance(message, str):
        message = str(message)
    return {
        "jsonrpc": "2.0",
        "id": request_id if request_id is not None else None,
        "error": {"code": code, "message": message},
    }


def _error_line(request_id: object, code: int, message: str) -> bytes:
    return (json.dumps(_jsonrpc_error(request_id, code, message), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _write_stdout(data: bytes) -> None:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


# --- CLI ----------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_agent.mcp_server",
        description=(
            "Local stdio MCP adapter over the shared agent service "
            "(MUSIC_AGENT_MCP_CONTRACT_v0.2.1). One MCP tool = one existing "
            "AgentToolName; no SQLite or repository access exists in this process."
        ),
    )
    parser.add_argument("--db", required=True, help="path to the music-agent SQLite store")
    parser.add_argument(
        "--client-policy",
        default="full",
        choices=["full", "read_only", "none"],
        help="in-memory policy for this adapter's generated agt_ client "
        "(default: full; never durable)",
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="optional validator-shaped agt_ client identity "
        "(default: generate_client_id())",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="provenance metadata only")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="provenance metadata only")
    parser.add_argument(
        "--agent-service",
        default=None,
        help="run-hosted agent socket path override for the exact-six routed "
        "family (default: <db>.agent.sock)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        server = MusicAgentMcpServer(
            args.db,
            client_id=args.client_id,
            client_policy=args.client_policy,
            model_id=args.model_id,
            label=args.label,
            remote_socket_path=Path(args.agent_service) if args.agent_service else None,
        )
    except (McpServerError, ValueError) as error:
        parser.error(str(error))
    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())