"""P09.6: AgentClient -- the client adapter boundary over the shared service.

These tests prove the thin client side of the shared agent layer: constructor fail-closed on
non-identity/non-service wiring, tool invocation by name (str or enum) with client-owned request
identity (reused ids exercise the service's replay contract), injected issued_at defaults and
overrides, and the provider-independent canonical interchange a real SDK boundary would use
(encode request -> decode -> execute -> encode result -> decode round trip). Two clients with
different model metadata share one service and observe identical state, proving model identity
is provenance, not a state or permission dimension.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient, AgentClientValidationError
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    decode_agent_request,
    decode_agent_tool_result,
    encode_agent_request,
    encode_agent_tool_result,
    generate_request_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.agent_tools import AgentToolName

CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"
CLIENT_READ_ONLY = "agt_22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
ISO = "2026-08-16T00:00:00+00:00"


class AgentClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        self.service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry(
                {
                    CLIENT_FULL: AgentClientPolicy.FULL,
                    CLIENT_READ_ONLY: AgentClientPolicy.READ_ONLY,
                }
            ),
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary_directory.cleanup()

    def client(self, model_id: str, label: str | None = None) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(CLIENT_FULL, model_id, label), self.service
        )

    def test_constructor_fails_closed_on_invalid_wiring(self) -> None:
        with self.assertRaises(AgentClientValidationError):
            AgentClient("not-an-identity", self.service)  # type: ignore[arg-type]
        with self.assertRaises(AgentClientValidationError):
            AgentClient(AgentClientIdentity(CLIENT_FULL, "codex"), "not-a-service")  # type: ignore[arg-type]

    def test_call_invokes_the_service_for_str_and_enum_tool_names(self) -> None:
        by_str = self.client("codex").call(
            "get_agent_capabilities", {}, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(by_str.outcome, AgentToolOutcome.OK)
        by_enum = self.client("codex").call(
            AgentToolName.GET_AGENT_CAPABILITIES, {}, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(by_enum.outcome, AgentToolOutcome.OK)
        self.assertEqual(by_str.payload, by_enum.payload)

    def test_call_fails_closed_on_invalid_arguments(self) -> None:
        with self.assertRaises(AgentClientValidationError):
            self.client("codex").call("", {}, issued_at=NOW)
        with self.assertRaises(AgentClientValidationError):
            self.client("codex").call("tool", "not-a-payload", issued_at=NOW)  # type: ignore[arg-type]

    def test_client_owned_request_ids_drive_the_replay_contract(self) -> None:
        request_id = generate_request_id()
        first = self.client("codex").call(
            "get_agent_capabilities", {}, request_id=request_id, issued_at=NOW, completed_at=ISO
        )
        self.assertFalse(first.replayed)
        replay = self.client("codex").call(
            "get_agent_capabilities", {}, request_id=request_id, issued_at=NOW, completed_at=ISO
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.payload, first.payload)

    def test_issued_at_defaults_to_client_time_and_accepts_injection(self) -> None:
        injected = self.client("codex").call(
            "get_agent_capabilities", {}, issued_at=NOW, completed_at=ISO
        )
        self.assertFalse(injected.replayed)
        defaulted = self.client("codex").call(
            "get_agent_capabilities", {}, completed_at=ISO
        )
        self.assertEqual(defaulted.outcome, AgentToolOutcome.OK)

    def test_canonical_interchange_is_the_provider_boundary(self) -> None:
        """A provider SDK translates provider-native calls through the canonical JSON texts."""
        from music_agent.agent_contract import AgentRequest

        wire_request = encode_agent_request(
            AgentRequest(
                generate_request_id(),
                AgentClientIdentity(CLIENT_FULL, "deepseek", "deepseek-chat adapter"),
                "get_agent_capabilities",
                {},
                NOW,
            )
        )
        decoded = decode_agent_request(wire_request)
        self.assertEqual(decoded.client.model_id, "deepseek")
        result = self.service.execute(decoded, completed_at=ISO)
        wire_result = encode_agent_tool_result(result)
        decoded_result = decode_agent_tool_result(wire_result)
        self.assertEqual(decoded_result.outcome, AgentToolOutcome.OK)
        self.assertEqual(decoded_result.payload, result.payload)

    def test_two_model_clients_share_identical_state(self) -> None:
        """Model identity is provenance: two clients over one service see the same facts."""
        codex = AgentClient(AgentClientIdentity(CLIENT_FULL, "codex"), self.service)
        deepseek = AgentClient(AgentClientIdentity(CLIENT_FULL, "deepseek"), self.service)
        codex_view = codex.call("get_agent_capabilities", {}, issued_at=NOW, completed_at=ISO)
        deepseek_view = deepseek.call(
            "get_agent_capabilities", {}, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(codex_view.payload, deepseek_view.payload)

    def test_read_only_policy_follows_client_id_not_model(self) -> None:
        """The same client id keeps READ_ONLY whatever model it claims."""
        read_only = AgentClient(
            AgentClientIdentity(CLIENT_READ_ONLY, "deepseek"), self.service
        )
        result = read_only.call(
            "get_agent_capabilities", {}, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        denied = read_only.call(
            "record_feedback",
            {
                "kind": "liked",
                "source_system": "recommendation_ui",
                "source_path": "card_actions",
                "target_id": "trk_11111111-1111-4111-8111-111111111111",
            },
            issued_at=NOW,
            completed_at=ISO,
        )
        self.assertEqual(denied.outcome, AgentToolOutcome.PERMISSION_DENIED)


if __name__ == "__main__":
    unittest.main()
