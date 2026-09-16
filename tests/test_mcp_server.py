"""M1-M4: the local stdio MCP adapter over SharedAgentService.execute().

Milestone-classed verification of MUSIC_AGENT_MCP_CONTRACT_v0.2.1 against the
real adapter:

* M1 -- stdio JSON-RPC transport skeleton + the frozen 29-tool projection
  (handshake, ping, tools/list, notifications, exit, parse/invalid/method
  errors, DO_NOT_EXPOSE hard wall, constructor validation, zero-dependency /
  zero-SQLite architectural scan);
* M2 -- the 15 READ tools through the real service boundary with the seeded
  canonical fixture, read_only / none client policy, the append-only journal
  and its replay semantics (first-seen row, replay-no-new-row, conflict), and
  the untruncated envelope mapping;
* M3 -- the 5 agent-owned durable MUTATE tools as a closed loop over the real
  SQLite store: record -> interpret -> apply -> list + preference visibility,
  generation -> run readback, discovery promotion, service-authoritative
  time, fail-closed duplicates;
* M4 -- the exact-six routed family (repo-truth guard) failing closed as
  ``agent_runtime_offline`` when run is not serving, the process-local
  playback remainder executing through the injected adapter only, ``play``
  never degrading locally, open_in_apple_music local + fail-closed, and the
  MCP process never owning a preview event sink or a second authority.

The tests never bind a socket (no UDS server, no run process): the remote
path default socket ``<db>.agent.sock`` simply does not exist, which is
exactly the offline scenario the routing contract defines.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    AgentToolResult,
    decode_agent_tool_result,
    encode_agent_tool_result,
    generate_request_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.mcp_server import (
    DEFAULT_MODEL_ID,
    MusicAgentMcpServer,
    McpServerValidationError,
    map_tool_result,
)
from music_agent.playback_control import NowPlaying, PlayerState
from music_agent.repository import CanonicalRepository
from music_agent.routed_client import REMOTE_TOOL_NAMES
from music_agent.validation import validate_fixture

logging.disable(logging.CRITICAL)  # the routed client logs offline warnings on stderr

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"

CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"
CLIENT_READ_ONLY = "agt_22222222-2222-4222-8222-222222222222"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
ISO = "2026-08-16T00:00:00+00:00"
FBK = "fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
REQ_1 = "req_11111111-1111-4111-8111-111111111111"
REQ_2 = "req_22222222-2222-4222-8222-222222222222"

# The contract-frozen exact-six routed family (verified against REPO-truth in
# `test_remote_tool_names_are_exactly_the_repo_truth`).
EXACT_SIX = {
    "preview_batch",
    "preview_catalog_track",
    "stop_preview",
    "advance_preview",
    "get_playback_context",
    "play",
}
# The six minus advance_preview (registry-only, CLI-routed, never on MCP).
EXACT_SIX_ON_MCP = EXACT_SIX - {"advance_preview"}

from music_agent.mcp_projection import (  # noqa: E402  (after the six above)
    MCP_EXPOSABLE_TOOL_NAMES,
    READ_TOOL_NAMES,
)


class RecordingPlaybackAdapter:
    """Records every command; read_now_playing reports a stopped state."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def play(self, **kwargs: object) -> None:
        self.calls.append(("play", ()))

    def pause(self, **kwargs: object) -> None:
        self.calls.append(("pause", ()))

    def next_track(self, **kwargs: object) -> None:
        self.calls.append(("next_track", ()))

    def previous_track(self, **kwargs: object) -> None:
        self.calls.append(("previous_track", ()))

    def play_track(self, persistent_id: str) -> None:
        self.calls.append(("play_track", (persistent_id,)))

    def read_now_playing(self) -> NowPlaying:
        self.calls.append(("read_now_playing", ()))
        return NowPlaying(state=PlayerState.STOPPED)

    def method_calls(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)


class RecordingResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve_playback_track(
        self, name: str, artist: str | None, album: str | None, duration_ms: int | None
    ) -> str | None:
        self.calls += 1
        return None


class RecordingCatalogSource:
    def __init__(self, *, track_view_url: str | None = None) -> None:
        self.terms: list[tuple[str, int]] = []
        self._track_view_url = track_view_url

    def search(self, term: str, limit: int) -> tuple:
        self.terms.append((term, limit))
        return ()

    def lookup_track_view_url(self, itunes_id: str) -> str | None:
        return self._track_view_url


_UNWIRED = object()


def make_server(  # noqa: PLR0913 (test builder)
    database_path: Path,
    *,
    client_policy: AgentClientPolicy = AgentClientPolicy.FULL,
    client_id: str | None = CLIENT_FULL,
    playback_adapter: object | None = None,
    catalog_search_source: object = _UNWIRED,
    remote_socket_path: Path | None = None,
) -> MusicAgentMcpServer:
    if playback_adapter is None:
        playback_adapter = RecordingPlaybackAdapter()
    if catalog_search_source is _UNWIRED:
        catalog_search_source = RecordingCatalogSource()
    server = MusicAgentMcpServer(
        database_path,
        client_id=client_id,
        client_policy=client_policy,
        playback_adapter=playback_adapter,
        playback_resolver=RecordingResolver(),
        catalog_search_source=catalog_search_source,
        remote_socket_path=remote_socket_path,
    )
    return server


class McpServerTestCase(unittest.TestCase):
    """Temp-dir store + fakes + a server; journal row counting for the M2 sweep."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        self.playback_adapter = RecordingPlaybackAdapter()
        self.catalog_source = RecordingCatalogSource()
        self._server: MusicAgentMcpServer | None = None

    def tearDown(self) -> None:
        if self._server is not None:
            self._server.close()
        self.temporary_directory.cleanup()

    def server(
        self,
        policy: AgentClientPolicy = AgentClientPolicy.FULL,
        client_id: str = CLIENT_FULL,
        catalog_source: object = _UNWIRED,
        remote_socket_path: Path | None = None,
    ) -> MusicAgentMcpServer:
        self._server = make_server(
            self.database_path,
            client_policy=policy,
            client_id=client_id,
            playback_adapter=self.playback_adapter,
            catalog_search_source=catalog_source,
            remote_socket_path=remote_socket_path,
        )
        return self._server

    def seed_fixture(self) -> None:
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def seed_annotated_fixture(self) -> None:
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        for track_id, values in (
            (TRACK_A, {"apple_music_persistent_id": "PERSIST-A", "itunes_store_id": None}),
            (TRACK_B, {"apple_music_persistent_id": None, "itunes_store_id": None}),
            (TRACK_C, {"apple_music_persistent_id": None, "itunes_store_id": "333"}),
        ):
            for track in fixture["tracks"]:
                if track["id"] == track_id:
                    track["external_ids"].update(values)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def seed_positive(self, track_id: str) -> None:
        from music_agent.preference_attribution import (
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import (
            PreferencePersistenceRepository,
        )
        from music_agent.source_observation import ObservedValue

        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                    "apple_music",
                    "favorited",
                ),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance="fixture_seed",
            )

    def journal_rows(self) -> int:
        with sqlite3.connect(self.database_path) as connection:
            (count,) = connection.execute(
                "SELECT COUNT(*) FROM agent_requests"
            ).fetchone()
        return count

    # --- wire harness -------------------------------------------------------

    @staticmethod
    def serve_lines(server: MusicAgentMcpServer, messages: list[dict]) -> list[dict]:
        """Feed JSON objects one per line through serve(); return reply objects."""
        lines = [json.dumps(message).encode("utf-8") for message in messages]
        out: list[dict] = []
        server.serve(iter(lines), lambda data: out.extend(_decode_wire_lines(data)))
        return out

    @staticmethod
    def serve_raw(server: MusicAgentMcpServer, raw_lines: list[bytes]) -> list[dict]:
        out: list[dict] = []
        server.serve(iter(raw_lines), lambda data: out.extend(_decode_wire_lines(data)))
        return out


def _decode_wire_lines(data: bytes) -> list[dict]:
    return [
        json.loads(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]


def invoke(server: MusicAgentMcpServer, tool: str, arguments: dict, request_id: str | None = None):
    return server.invoke_tool(tool, arguments, request_id=request_id)


# =============================================================================
# M1 -- stdio transport skeleton + safe projection
# =============================================================================


class M1TransportTest(McpServerTestCase):
    def test_initialize_handshake_echoes_supported_version(self) -> None:
        replies = self.serve_lines(
            self.server(),
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                            "clientInfo": {"name": "test", "version": "0"}}},
            ],
        )
        self.assertEqual(replies[0]["id"], 1)
        result = replies[0]["result"]
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        self.assertEqual(result["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(
            result["serverInfo"],
            {"name": "music-agent-core", "title": "Music Agent", "version": "0.1.0"},
        )

    def test_initialize_unknown_version_falls_back_to_default(self) -> None:
        replies = self.serve_lines(
            self.server(),
            [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2099-01-01"}}],
        )
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2024-11-05")

    def test_initialize_without_params_still_succeeds(self) -> None:
        replies = self.serve_lines(
            self.server(), [{"jsonrpc": "2.0", "id": 1, "method": "initialize"}]
        )
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2024-11-05")

    def test_ping(self) -> None:
        replies = self.serve_lines(
            self.server(), [{"jsonrpc": "2.0", "id": 2, "method": "ping"}]
        )
        self.assertEqual(replies[0], {"jsonrpc": "2.0", "id": 2, "result": {}})

    def test_tools_list_carries_the_frozen_surface(self) -> None:
        replies = self.serve_lines(
            self.server(), [{"jsonrpc": "2.0", "id": 3, "method": "tools/list"}]
        )
        tools = replies[0]["result"]["tools"]
        self.assertEqual(len(tools), 29)
        names = [tool["name"] for tool in tools]
        self.assertEqual(set(names), MCP_EXPOSABLE_TOOL_NAMES)
        self.assertEqual(len(set(names)), 29)

    def test_notification_is_not_answered(self) -> None:
        replies = self.serve_lines(
            self.server(),
            [{"jsonrpc": "2.0", "method": "notifications/initialized"}],
        )
        self.assertEqual(replies, [])

    def test_exit_ends_the_loop_without_a_response(self) -> None:
        out: list[dict] = []
        ended = self.server().serve(
            iter([b'{"jsonrpc":"2.0","method":"exit"}']),
            lambda data: out.extend(_decode_wire_lines(data)),
        )
        self.assertTrue(ended)
        self.assertEqual(out, [])

    def test_eof_ends_the_loop(self) -> None:
        out: list[dict] = []
        ended = self.server().serve(iter([]), lambda data: out.extend(_decode_wire_lines(data)))
        self.assertFalse(ended)
        self.assertEqual(out, [])

    def test_parse_error_answers_32700_and_continues(self) -> None:
        replies = self.serve_raw(
            self.server(),
            [
                b"not json at all",
                b'{"jsonrpc":"2.0","id":7,"method":"ping"}',
            ],
        )
        self.assertEqual(replies[0]["error"]["code"], -32700)
        self.assertEqual(
            replies[1], {"jsonrpc": "2.0", "id": 7, "result": {}}
        )

    def test_non_jsonrpc_message_answers_32600(self) -> None:
        replies = self.serve_raw(
            self.server(), [b'{"jsonrpc":"1.0","id":8,"method":"ping"}']
        )
        self.assertEqual(replies[0]["id"], 8)
        self.assertEqual(replies[0]["error"]["code"], -32600)

    def test_unknown_method_answers_32601(self) -> None:
        replies = self.serve_lines(
            self.server(),
            [{"jsonrpc": "2.0", "id": 9, "method": "tools/no_such_thing"}],
        )
        self.assertEqual(replies[0]["id"], 9)
        self.assertEqual(replies[0]["error"]["code"], -32601)
        self.assertIn("tools/no_such_thing", replies[0]["error"]["message"])

    def test_tools_call_without_name_answers_32602(self) -> None:
        replies = self.serve_lines(
            self.server(),
            [{"jsonrpc": "2.0", "id": 10, "method": "tools/call",
              "params": {"arguments": {}}}],
        )
        self.assertEqual(replies[0]["id"], 10)
        self.assertEqual(replies[0]["error"]["code"], -32602)

    def test_tools_call_with_missing_arguments_object_is_tolerated(self) -> None:
        """Transport tolerance only: an absent arguments object means an empty
        payload; validator discipline is the domain layer's, untouched."""
        replies = self.serve_lines(
            self.server(),
            [{"jsonrpc": "2.0", "id": 11, "method": "tools/call",
              "params": {"name": "get_agent_capabilities"}}],
        )
        result = replies[0]["result"]
        decoded = decode_agent_tool_result(result["content"][0]["text"])
        self.assertEqual(decoded.outcome, AgentToolOutcome.OK)

    def test_do_not_expose_tools_are_blocked_at_the_adapter(self) -> None:
        for name in ("execute_write_intent", "add_catalog_to_library", "advance_preview"):
            with self.subTest(name=name):
                with self.assertRaises(McpServerValidationError):
                    invoke(self.server(), name, {})

    def test_unknown_tool_is_blocked(self) -> None:
        with self.assertRaises(McpServerValidationError):
            invoke(self.server(), "explain_recommendation", {})

    def test_arguments_must_be_a_string_keyed_mapping(self) -> None:
        with self.assertRaises(McpServerValidationError):
            invoke(self.server(), "get_agent_capabilities", ["not", "a", "mapping"])  # type: ignore[arg-type]
        with self.assertRaises(McpServerValidationError):
            invoke(self.server(), "get_agent_capabilities", {1: "non-string key"})  # type: ignore[dict-item]

    def test_generated_client_id_is_validator_shaped(self) -> None:
        server = self.server(client_id=None)
        self.assertTrue(server.client_id.startswith("agt_"))
        # The generated identity is usable through the boundary.
        result = invoke(server, "get_agent_capabilities", {})
        self.assertEqual(result.outcome, AgentToolOutcome.OK)

    def test_injected_client_id_is_preserved(self) -> None:
        self.assertEqual(
            self.server(client_id="agt_33333333-3333-4333-8333-333333333333").client_id,
            "agt_33333333-3333-4333-8333-333333333333",
        )

    def test_bad_policy_is_refused_at_construction(self) -> None:
        with self.assertRaises(McpServerValidationError):
            make_server(self.database_path, client_policy="super_admin")  # type: ignore[arg-type]

    def test_bad_database_path_is_refused(self) -> None:
        with self.assertRaises(McpServerValidationError):
            MusicAgentMcpServer(Path(""))  # empty path refused

    def test_close_is_idempotent_and_second_close_is_quiet(self) -> None:
        server = self.server()
        server.close()
        server.close()

    def test_unexpected_handler_failure_answers_32603_and_keeps_serving(self) -> None:
        server = self.server()
        original = server.handle_message
        state = {"raised": False}

        def flaky(message):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("boom")
            return original(message)

        with mock.patch.object(server, "handle_message", flaky):
            replies = self.serve_lines(
                server,
                [
                    {"jsonrpc": "2.0", "id": 12, "method": "ping"},
                    {"jsonrpc": "2.0", "id": 13, "method": "ping"},
                ],
            )
        self.assertEqual(replies[0]["error"]["code"], -32603)
        self.assertEqual(replies[1], {"jsonrpc": "2.0", "id": 13, "result": {}})

    def test_cli_parser_accepts_the_documented_startup_surface(self) -> None:
        from music_agent.mcp_server import build_argument_parser

        arguments = build_argument_parser().parse_args(
            ["--db", "/tmp/example.sqlite3", "--client-policy", "read_only"]
        )
        self.assertEqual(arguments.db, "/tmp/example.sqlite3")
        self.assertEqual(arguments.client_policy, "read_only")

    def test_mcp_modules_contain_no_sqlite_or_repository_access(self) -> None:
        """Contract §20 architectural acceptance test: the adapter layer is
        transport + identity only. No sqlite3 / repository import may exist in
        either MCP module -- every domain fact stays behind execute()."""
        root = Path(__file__).resolve().parent.parent / "src" / "music_agent"
        combined = ""
        for module_name in ("mcp_projection.py", "mcp_server.py"):
            with open(root / module_name, encoding="utf-8") as module_file:
                source = module_file.read()
            combined += source
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                else:
                    continue
                for module in imported:
                    self.assertFalse(
                        module == "sqlite3" or module.endswith("_repository")
                        or module == "music_agent.repository",
                        f"{module_name} imports {module}",
                    )
        # Soft guard: the scan above read real code.
        self.assertIn("SharedAgentService", combined)
        # The production osascript/afplay chain is built by functions only
        # (lazy, construction-time); no module-import side effect may spawn a
        # runner at import level.
        construction_tokens = (
            "OsascriptPlaybackRunner(",
            "OsascriptLibraryResolveRunner(",
            "default_catalog_search_source()",
        )
        for line in combined.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("def ", "class ", "#", '"""')):
                continue  # declarations/docstrings, not construction side effects
            for token in construction_tokens:
                if token in line:
                    self.assertTrue(
                        line.startswith((" ", "\t")),
                        f"runner construction must live inside a function body: {line!r}",
                    )


# =============================================================================
# M2 -- READ surface (15 tools) + read-only policy + journal/replay
# =============================================================================


class M2ReadSurfaceTest(McpServerTestCase):
    def test_read_tools_execute_process_local_over_the_real_boundary(self) -> None:
        """The non-routed READ tools succeed against the seeded store; each is a
        fresh journaled request; the fake adapter proves local execution."""
        self.seed_fixture()
        self.seed_positive(TRACK_A)
        server = self.server()
        cases = {
            "get_canonical_entity": {"canonical_id": TRACK_A},
            "query_track_preference": {
                "target_id": TRACK_A, "source_system": "apple_music"
            },
            "list_recommendation_runs": {},
            "list_feedback_observations": {},
            "get_agent_capabilities": {},
            "get_active_context": {},
            "get_now_playing": {},
            "search_library_tracks": {"term": "anything"},
            "query_catalog_discovery_state": {"canonical_id": TRACK_A},
        }
        for name, arguments in cases.items():
            with self.subTest(tool=name):
                result = invoke(server, name, arguments)
                self.assertEqual(result.outcome, AgentToolOutcome.OK, result)
        # get_now_playing read the injected adapter -> local, not invented.
        self.assertGreaterEqual(
            self.playback_adapter.method_calls("read_now_playing"), 1
        )
        # Every READ request was journaled (C2: reads also grow the journal).
        self.assertEqual(self.journal_rows(), len(cases))

    def test_never_recommended_track_state_is_honest(self) -> None:
        self.seed_fixture()
        result = invoke(
            self.server(),
            "query_catalog_discovery_state",
            {"canonical_id": TRACK_A},
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        found = result.payload["found"]
        self.assertIsInstance(found, bool)
        if found:
            state = result.payload["state"]
            # Facts, never ranking/score/eligibility (P15-S3-S2 XOR contract).
            for key in ("score", "ranking", "eligible_for_exploration"):
                self.assertNotIn(key, state)
            # Never recommended (this test generates nothing) must be stated.
            self.assertTrue(state.get("never_recommended"))

    def test_missing_entities_fail_closed_with_stable_codes(self) -> None:
        self.seed_fixture()
        server = self.server()
        missing_feedback = invoke(
            server, "get_feedback_observation", {"feedback_id": FBK}
        )
        self.assertEqual(missing_feedback.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(missing_feedback.error_code, "feedback_not_found")
        missing_run = invoke(
            server,
            "get_recommendation_run",
            {"run_id": "rcm_99999999-9999-4999-8999-999999999999"},
        )
        self.assertEqual(missing_run.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(missing_run.error_code, "recommendation_run_not_found")

    def test_unknown_payload_key_is_refused_as_validation_error(self) -> None:
        self.seed_fixture()
        result = invoke(
            self.server(),
            "query_track_preference",
            {"target_id": TRACK_A, "invented_key": "hallucination"},
        )
        self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(result.error_code, "validation_error")

    def test_read_only_policy_allows_reads_and_blocks_mutations(self) -> None:
        self.seed_fixture()
        server = self.server(policy=AgentClientPolicy.READ_ONLY,
                             client_id=CLIENT_READ_ONLY)
        self.assertEqual(
            invoke(server, "get_agent_capabilities", {}).outcome,
            AgentToolOutcome.OK,
        )
        blocked = invoke(server, "record_feedback", record_feedback_payload())
        self.assertEqual(blocked.outcome, AgentToolOutcome.PERMISSION_DENIED)
        self.assertEqual(blocked.error_code, "permission_denied")

    def test_none_policy_blocks_even_reads(self) -> None:
        server = self.server(policy=AgentClientPolicy.NONE,
                             client_id="agt_44444444-4444-4444-8444-444444444444")
        blocked = invoke(server, "get_agent_capabilities", {})
        self.assertEqual(blocked.outcome, AgentToolOutcome.PERMISSION_DENIED)

    def test_unregistered_client_is_refused_unknown_client(self) -> None:
        """The policy boundary is per-client (in-memory registry), not
        per-process: an identity absent from the registry fails closed with
        the stable unknown_client outcome."""
        from music_agent.agent_service import SharedAgentService
        from music_agent.agent_socket import agent_socket_path
        from music_agent.routed_client import RoutedAgentClient

        registry = AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.FULL})
        service = SharedAgentService(
            self.database_path,
            clients=registry,
            playback_adapter=self.playback_adapter,
            playback_resolver=RecordingResolver(),
            catalog_search_source=self.catalog_source,
        )
        self.addCleanup(service.close)
        unregistered = RoutedAgentClient(
            AgentClientIdentity(
                client_id="agt_55555555-5555-4555-8555-555555555555",
                model_id=DEFAULT_MODEL_ID,
            ),
            service,
            remote_socket_path=agent_socket_path(self.database_path),
        )
        result = unregistered.call("get_agent_capabilities", {})
        self.assertEqual(result.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        self.assertEqual(result.error_code, "unknown_client")

    def test_journal_first_seen_writes_one_row(self) -> None:
        """Final Verification Note 1: a first-seen request_id appends exactly
        one journal row; no row for any other identity."""
        server = self.server()
        result = invoke(server, "get_agent_capabilities", {}, request_id=REQ_1)
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertFalse(result.replayed)
        self.assertEqual(self.journal_rows(), 1)

    def test_journal_replay_returns_stored_result_without_a_new_row(self) -> None:
        server = self.server()
        first = invoke(server, "get_agent_capabilities", {}, request_id=REQ_1)
        second = invoke(server, "get_agent_capabilities", {}, request_id=REQ_1)
        self.assertTrue(second.replayed)
        self.assertEqual(second.outcome, first.outcome)
        self.assertEqual(second.payload, first.payload)
        self.assertEqual(self.journal_rows(), 1)

    def test_journal_conflict_refuses_and_writes_no_row(self) -> None:
        server = self.server()
        invoke(server, "get_agent_capabilities", {}, request_id=REQ_1)
        conflict = invoke(
            server, "get_agent_capabilities", {"invented": True}, request_id=REQ_1
        )
        self.assertEqual(conflict.outcome, AgentToolOutcome.REPLAY_CONFLICT)
        self.assertFalse(conflict.replayed)
        self.assertEqual(self.journal_rows(), 1)

    def test_omitted_request_ids_mint_fresh_req_identities(self) -> None:
        server = self.server()
        first = invoke(server, "get_agent_capabilities", {})
        second = invoke(server, "get_agent_capabilities", {})
        self.assertNotEqual(first.request_id, second.request_id)
        self.assertTrue(first.request_id.startswith("req_"))
        self.assertTrue(second.request_id.startswith("req_"))

    def test_tools_call_wire_round_trip_is_the_encode_tool_result_text(self) -> None:
        """The wire content is exactly the canonical encode_agent_tool_result
        text of this call's own envelope (lossless decode-encode round trip)."""
        server = self.server()
        replies = self.serve_lines(
            server,
            [{"jsonrpc": "2.0", "id": 20, "method": "tools/call",
              "params": {"name": "get_agent_capabilities", "arguments": {}}}],
        )
        result = replies[0]["result"]
        self.assertIs(result["isError"], False)
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        text = result["content"][0]["text"]
        decoded = decode_agent_tool_result(text)
        self.assertEqual(decoded.outcome, AgentToolOutcome.OK)
        self.assertEqual(decoded.tool, "get_agent_capabilities")
        self.assertEqual(text, encode_agent_tool_result(decoded))
        # The wire invoke went through the same envelope mapping as a stale
        # direct call (completed_at may differ between two executions; the
        # payload of a deterministic READ tool does not).
        direct = invoke(server, "get_agent_capabilities", {})
        self.assertEqual(decoded.payload, direct.payload)

    def test_non_ok_outcome_rides_with_is_error_true(self) -> None:
        """A refusal can never masquerade as success on the MCP wire."""
        server = self.server(policy=AgentClientPolicy.READ_ONLY,
                             client_id=CLIENT_READ_ONLY)
        replies = self.serve_lines(
            server,
            [{"jsonrpc": "2.0", "id": 21, "method": "tools/call",
              "params": {"name": "record_feedback",
                         "arguments": {"kind": "liked", "target_id": TRACK_A,
                                       "source_system": "recommendation_ui",
                                       "source_path": "card_actions"}}}],
        )
        result = replies[0]["result"]
        self.assertIs(result["isError"], True)
        decoded = decode_agent_tool_result(result["content"][0]["text"])
        self.assertEqual(decoded.outcome, AgentToolOutcome.PERMISSION_DENIED)

    def test_envelope_mapping_is_complete_and_untruncated(self) -> None:
        """Contract §6: no 2000-character truncation at this boundary -- the
        canonical envelope passes through verbatim."""
        payload = {"state": "stopped", "blob": "x" * 5000}
        result = AgentToolResult(
            request_id=generate_request_id(),
            tool="get_now_playing",
            outcome=AgentToolOutcome.OK,
            payload=payload,
            error_code=None,
            error_message=None,
            completed_at=datetime.now(timezone.utc),
        )
        mapped = map_tool_result(result)
        text = mapped["content"][0]["text"]
        self.assertGreater(len(text), 5000)
        self.assertEqual(text, encode_agent_tool_result(result))
        self.assertIs(mapped["isError"], False)

    def test_argument_mapping_preserves_payload_bytes(self) -> None:
        """The invoke boundary forwards the exact payload, no coercion."""
        server = self.server()
        result = invoke(
            server,
            "query_track_preference",
            {"target_id": TRACK_A, "source_system": "apple_music", "extra_number": 0},
            request_id=REQ_2,
        )
        # extra key refused -> the payload reached the validator intact.
        self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)


# =============================================================================
# M3 -- the 5 agent-owned durable MUTATE tools as a closed loop
# =============================================================================


def record_feedback_payload(track_id: str = TRACK_A, feedback_id: str = FBK) -> dict:
    return {
        "kind": "liked",
        "source_system": "recommendation_ui",
        "source_path": "card_actions",
        "target_id": track_id,
        "feedback_id": feedback_id,
    }


class M3MutationTest(McpServerTestCase):
    def test_feedback_to_application_closed_loop(self) -> None:
        self.seed_fixture()
        server = self.server()

        recorded = invoke(server, "record_feedback", record_feedback_payload())
        self.assertEqual(recorded.outcome, AgentToolOutcome.OK)
        feedback_id = recorded.payload["feedback_id"]

        interpreted = invoke(
            server, "interpret_feedback", {"feedback_id": feedback_id}
        )
        self.assertEqual(interpreted.outcome, AgentToolOutcome.OK)
        self.assertEqual(interpreted.payload["direction"], "positive")
        self.assertEqual(interpreted.payload["explicitness"], "explicit")

        applied = invoke(server, "apply_learning", {"feedback_id": feedback_id})
        self.assertEqual(applied.outcome, AgentToolOutcome.OK)
        self.assertTrue(applied.payload["applied"])
        self.assertEqual(applied.payload["proposal_kind"], "evidence_observation")

        applications = invoke(server, "list_learning_applications", {})
        self.assertEqual(len(applications.payload["applications"]), 1)
        self.assertEqual(
            applications.payload["applications"][0]["feedback_id"], feedback_id
        )

        preference = invoke(
            server,
            "query_track_preference",
            {"target_id": TRACK_A, "source_system": "feedback_learning"},
        )
        self.assertEqual(preference.payload["preference_state"], "positive")

    def test_skip_feedback_applies_nothing(self) -> None:
        self.seed_fixture()
        server = self.server()
        recorded = invoke(
            server,
            "record_feedback",
            {**record_feedback_payload(feedback_id="fbk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
             "kind": "skipped"},
        )
        applied = invoke(
            server, "apply_learning", {"feedback_id": recorded.payload["feedback_id"]}
        )
        self.assertEqual(applied.outcome, AgentToolOutcome.OK)
        self.assertFalse(applied.payload["applied"])
        self.assertEqual(applied.payload["reason"], "no_proposal")

    def test_duplicate_feedback_fails_closed(self) -> None:
        self.seed_fixture()
        server = self.server()
        payload = record_feedback_payload()
        self.assertEqual(invoke(server, "record_feedback", payload).outcome,
                         AgentToolOutcome.OK)
        duplicate = invoke(server, "record_feedback", payload)
        self.assertEqual(duplicate.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(duplicate.error_code, "duplicate_feedback_observation")

    def test_generate_recommendation_persists_and_reads_back(self) -> None:
        self.seed_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        server = self.server()

        generated = invoke(
            server,
            "generate_recommendation",
            {"target_ids": [TRACK_A, TRACK_B], "limit": 5,
             "source_system": "apple_music"},
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        run_id = generated.payload["run_id"]

        runs = invoke(server, "list_recommendation_runs", {})
        self.assertEqual([run["run_id"] for run in runs.payload["runs"]], [run_id])

        fetched = invoke(
            server, "get_recommendation_run", {"run_id": run_id}
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        self.assertEqual(fetched.payload["run_id"], run_id)
        self.assertIn("encoded_result", fetched.payload)

    def test_plain_generate_rejects_unknown_key_without_history(self) -> None:
        """§7.2: min_exploration is an inferred-only knob; the plain tool
        refuses it before execution."""
        self.seed_fixture()
        self.seed_positive(TRACK_A)
        server = self.server()
        refused = invoke(
            server,
            "generate_recommendation",
            {"target_ids": [TRACK_A], "limit": 5, "min_exploration": 2},
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(refused.error_code, "validation_error")
        runs = invoke(server, "list_recommendation_runs", {})
        self.assertEqual(runs.payload["runs"], [])

    def test_inferred_generate_records_an_active_batch(self) -> None:
        self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        server = self.server()
        inferred = invoke(
            server,
            "generate_inferred_recommendation",
            {"target_ids": [TRACK_A, TRACK_B], "limit": 5},
        )
        self.assertEqual(inferred.outcome, AgentToolOutcome.OK)
        run_id = inferred.payload["run_id"]
        context = invoke(server, "get_active_context", {})
        self.assertEqual(context.payload["active_batch"]["run_id"], run_id)

    def test_discover_catalog_tracks_persists_through_the_boundary(self) -> None:
        from music_agent.apple_music_catalog import CatalogTrack

        self.seed_fixture()
        tracks = (
            CatalogTrack(
                catalog_id="CATALOG-SONG-1",
                name="Catalog Song CATALOG-SONG-1",
                artist_names=("Artist Alpha",),
                album_name="Catalog Album",
                genres=("Synthetic",),
                isrc="USSYN2400001",
                duration_ms=201000,
                release_date="2024-01-15",
                url=None,
                artist_catalog_ids=("CATALOG-ARTIST-1",),
                album_catalog_id="CATALOG-ALBUM-1",
            ),
        )
        source = type("_FakeCatalogSource", (), {
            "search": lambda self, term, limit: tracks,
            "lookup_track_view_url": lambda self, itunes_id: None,
        })()
        server = self.server(catalog_source=source)
        result = invoke(server, "discover_catalog_tracks", {"term": "catalog song"})
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["discovered_count"], 1)
        self.assertEqual(result.payload["promoted_count"], 1)

    def test_unwired_search_source_fails_closed(self) -> None:
        self.seed_fixture()
        server = self.server(catalog_source=None)
        result = invoke(server, "discover_catalog_tracks", {"term": "anything"})
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "catalog_discovery_unavailable")


# =============================================================================
# M4 -- playback/preview wrapper with exact-six routing
# =============================================================================


class M4RoutingTest(McpServerTestCase):
    def test_route_table_is_exactly_the_repo_truth(self) -> None:
        """The MCP adapter consumes the existing routed table verbatim; this
        test guards the repo-truth it depends on (any repo change to the
        table must be a deliberate, contract-reviewed decision)."""
        self.assertEqual(REMOTE_TOOL_NAMES, EXACT_SIX)
        self.assertEqual(
            EXACT_SIX_ON_MCP & MCP_EXPOSABLE_TOOL_NAMES, EXACT_SIX_ON_MCP
        )
        self.assertNotIn("advance_preview", MCP_EXPOSABLE_TOOL_NAMES)

    def test_exact_six_fail_closed_offline_as_valid_envelopes(self) -> None:
        """No run process is serving (the default socket does not exist): every
        routed tool returns a VALID result with the stable offline refusal --
        never a degraded local answer."""
        server = self.server()
        for name in sorted(EXACT_SIX_ON_MCP):
            with self.subTest(tool=name):
                result = invoke(server, name, {}, request_id=None)
                self.assertIsInstance(result, AgentToolResult, name)
                self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR, name)
                self.assertEqual(result.error_code, "agent_runtime_offline", name)
                self.assertFalse(result.replayed, name)

    def test_offline_refusal_preserves_the_requested_identity(self) -> None:
        server = self.server()
        result = invoke(server, "play", {}, request_id=REQ_2)
        self.assertEqual(result.request_id, REQ_2)
        self.assertEqual(result.error_code, "agent_runtime_offline")

    def test_play_never_executes_locally(self) -> None:
        server = self.server()
        result = invoke(server, "play", {})
        self.assertEqual(result.error_code, "agent_runtime_offline")
        self.assertEqual(self.playback_adapter.method_calls("play"), 0)
        self.assertEqual(self.playback_adapter.method_calls("pause"), 0)

    def test_local_playback_family_executes_through_the_injected_adapter(self) -> None:
        """pause / next_track / previous_track / play_track are NOT routed --
        they execute process-locally through the service's own adapter."""
        self.seed_annotated_fixture()
        server = self.server()
        paused = invoke(server, "pause", {})
        self.assertEqual(paused.outcome, AgentToolOutcome.OK)
        next_result = invoke(server, "next_track", {})
        self.assertEqual(next_result.outcome, AgentToolOutcome.OK)
        previous = invoke(server, "previous_track", {})
        self.assertEqual(previous.outcome, AgentToolOutcome.OK)
        play_track = invoke(server, "play_track", {"canonical_id": TRACK_A})
        self.assertEqual(play_track.outcome, AgentToolOutcome.OK)
        self.assertEqual(play_track.payload["persistent_id"], "PERSIST-A")
        self.assertEqual(play_track.payload["resolution"], "binding")
        self.assertEqual(self.playback_adapter.method_calls("pause"), 1)
        self.assertEqual(self.playback_adapter.method_calls("next_track"), 1)
        self.assertEqual(self.playback_adapter.method_calls("previous_track"), 1)
        self.assertEqual(self.playback_adapter.method_calls("play_track"), 1)

    def test_get_playback_context_never_invents_a_local_answer(self) -> None:
        """Route truth: get_playback_context is run's authority -- offline means
        offline, never a fabricated local session fact."""
        server = self.server()
        result = invoke(server, "get_playback_context", {})
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "agent_runtime_offline")

    def test_preview_state_is_readable_offline_only_as_refusal(self) -> None:
        server = self.server()
        for name in ("preview_batch", "preview_catalog_track", "stop_preview"):
            result = invoke(server, name, {"canonical_id": TRACK_A})
            self.assertEqual(result.error_code, "agent_runtime_offline", name)

    def test_mcp_never_binds_a_preview_event_sink(self) -> None:
        """The MCP process is not a preview presenter: no event listener, no
        event socket, no second authority over AudioSuspension."""
        server = self.server()
        self.assertIsNone(server.client.event_socket_path)

    def test_advance_preview_is_the_seventh_ghost_and_never_reachable(self) -> None:
        """advance_preview IS routed in the repo (CLI preview family) but is
        absent from the MCP surface: the adapter refuses it before any socket
        work, and it cannot be selected from tools/list either."""
        server = self.server()
        with self.assertRaises(McpServerValidationError):
            invoke(server, "advance_preview", {})
        self.assertIn("advance_preview", REMOTE_TOOL_NAMES)
        # Serve-side: a tools/call for it is a -32602 invalid params error,
        # not an agent result and not a network call.
        replies = self.serve_lines(
            server,
            [{"jsonrpc": "2.0", "id": 30, "method": "tools/call",
              "params": {"name": "advance_preview", "arguments": {}}}],
        )
        self.assertEqual(replies[0]["error"]["code"], -32602)

    def test_open_in_apple_music_is_local_and_fails_closed_without_binding(self) -> None:
        # TRACK_C exists in the plain fixture with NO itunes_store binding:
        # the local handler refuses with the honest unavailable code (never a
        # routed offline refusal and never a Music.app handoff).
        self.seed_fixture()
        server = self.server()
        result = invoke(server, "open_in_apple_music", {"canonical_id": TRACK_C})
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "apple_music_open_unavailable")

    def test_open_in_apple_music_uses_lookup_identity_for_music_app_handoff(self) -> None:
        self.seed_annotated_fixture()
        source = RecordingCatalogSource(
            track_view_url="https://music.apple.com/us/album/x?i=333"
        )
        server = self.server(catalog_source=source)
        with mock.patch(
            "music_agent.apple_music_open.open_music_app",
            return_value="https://music.apple.com/us/song/333",
        ) as handoff:
            result = invoke(server, "open_in_apple_music", {"canonical_id": TRACK_C})
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(
            result.payload["url"], "https://music.apple.com/us/album/x?i=333"
        )
        self.assertEqual(
            result.payload["client_url"], "https://music.apple.com/us/song/333"
        )
        self.assertTrue(result.payload["handoff_requested"])
        self.assertNotIn("opened", result.payload)
        self.assertEqual(result.payload["source"], "itunes_store_lookup")
        handoff.assert_called_once_with(
            "https://music.apple.com/us/album/x?i=333", "333"
        )


if __name__ == "__main__":
    unittest.main()