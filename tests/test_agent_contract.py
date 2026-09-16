"""P09.1: the shared agent service contract -- identity, envelopes, and canonical interchange.

These tests prove the side-effect-free contract the shared agent layer is built on: the disjoint
``agt_`` / ``req_`` identity namespaces (generation and fail-closed validation), the validated
frozen request/result envelopes (client identity as provenance-only metadata, bounded JSON
payloads, injected tz-aware timestamps, the fixed outcome vocabulary with its payload/error
invariants), and the deterministic canonical JSON interchange that is the provider-independent
boundary -- round-trips preserve structure, encoding is deterministic, and decoding fails closed
on malformed or out-of-contract text.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from music_agent.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentClientIdentity,
    AgentContractValidationError,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
    decode_agent_request,
    decode_agent_tool_result,
    encode_agent_payload,
    encode_agent_request,
    encode_agent_tool_result,
    generate_client_id,
    generate_request_id,
    validate_client_id,
    validate_request_id,
)

CLIENT_ID = "agt_11111111-1111-4111-8111-111111111111"
REQUEST_ID = "req_22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def client(model_id: str = "codex", label: str | None = None) -> AgentClientIdentity:
    return AgentClientIdentity(CLIENT_ID, model_id, label)


def request(tool: str = "get_canonical_entity", payload: dict | None = None) -> AgentRequest:
    return AgentRequest(
        REQUEST_ID,
        client(),
        tool,
        {} if payload is None else payload,
        NOW,
    )


def ok_result(payload: dict | None = None) -> AgentToolResult:
    return AgentToolResult(
        REQUEST_ID,
        "get_canonical_entity",
        AgentToolOutcome.OK,
        {} if payload is None else payload,
        None,
        None,
        NOW,
    )


class AgentIdentityTest(unittest.TestCase):
    def test_client_ids_use_a_disjoint_namespace(self) -> None:
        generated = generate_client_id()
        self.assertTrue(generated.startswith("agt_"))
        validate_client_id(generated)
        # agt_ is disjoint from every canonical and operational namespace.
        for prefix in ("trk_", "int_", "rcm_", "cnd_", "fbk_", "req_"):
            self.assertFalse(generated.startswith(prefix))

    def test_request_ids_use_a_disjoint_namespace(self) -> None:
        generated = generate_request_id()
        self.assertTrue(generated.startswith("req_"))
        validate_request_id(generated)
        for prefix in ("trk_", "int_", "rcm_", "cnd_", "fbk_", "agt_"):
            self.assertFalse(generated.startswith(prefix))

    def test_id_validation_fails_closed_on_foreign_namespaces(self) -> None:
        for foreign in (
            "rcm_11111111-1111-4111-8111-111111111111",
            "agt_",
            "agt_11111111",
            "agt_not-a-uuid",
            "",
        ):
            with self.assertRaises(AgentContractValidationError):
                validate_client_id(foreign)
        for foreign in (
            "agt_22222222-2222-4222-8222-222222222222",
            "req_",
            "req_22222222",
            "req_not-a-uuid",
            "",
        ):
            with self.assertRaises(AgentContractValidationError):
                validate_request_id(foreign)


class AgentClientIdentityTest(unittest.TestCase):
    def test_valid_identity(self) -> None:
        identity = client(model_id="deepseek", label="deepseek-chat adapter")
        self.assertEqual(identity.client_id, CLIENT_ID)
        self.assertEqual(identity.model_id, "deepseek")

    def test_fails_closed_on_invalid_fields(self) -> None:
        with self.assertRaises(AgentContractValidationError):
            AgentClientIdentity("trk_11111111-1111-4111-8111-111111111111", "codex")
        with self.assertRaises(AgentContractValidationError):
            AgentClientIdentity(CLIENT_ID, "")
        with self.assertRaises(AgentContractValidationError):
            AgentClientIdentity(CLIENT_ID, "codex", label="")
        with self.assertRaises(AgentContractValidationError):
            AgentClientIdentity(CLIENT_ID, "codex", label=7)  # type: ignore[arg-type]


class AgentRequestTest(unittest.TestCase):
    def test_valid_request_normalizes_payload(self) -> None:
        envelope = request(payload={"target_id": "trk_11111111-1111-4111-8111-111111111111"})
        self.assertEqual(envelope.tool, "get_canonical_entity")
        self.assertEqual(envelope.contract_version, AGENT_CONTRACT_VERSION)
        self.assertEqual(envelope.issued_at, NOW)
        self.assertIsNotNone(envelope.payload["target_id"])

    def test_fails_closed_on_invalid_requests(self) -> None:
        with self.assertRaises(AgentContractValidationError):
            AgentRequest("int_1", client(), "tool", {}, NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, "not-a-client", "tool", {}, NOW)  # type: ignore[arg-type]
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "", {}, NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "tool", "not-a-dict", NOW)  # type: ignore[arg-type]
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "tool", {7: "v"}, NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "tool", {}, NOW.replace(tzinfo=None))
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "tool", {}, NOW, contract_version=0)
        with self.assertRaises(AgentContractValidationError):
            AgentRequest(REQUEST_ID, client(), "tool", {}, NOW, contract_version="1")  # type: ignore[arg-type]


class AgentToolResultTest(unittest.TestCase):
    def test_ok_result_requires_payload_without_error(self) -> None:
        result = ok_result(payload={"entity": "trk_1"})
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertIsNone(result.error_code)

    def test_refusal_results_require_error_without_payload(self) -> None:
        for outcome in (
            AgentToolOutcome.INVALID_REQUEST,
            AgentToolOutcome.REPLAY_CONFLICT,
            AgentToolOutcome.TOOL_NOT_SUPPORTED,
            AgentToolOutcome.UNKNOWN_CLIENT,
            AgentToolOutcome.PERMISSION_DENIED,
            AgentToolOutcome.NOT_EXECUTION_READY,
            AgentToolOutcome.EXECUTION_ERROR,
        ):
            result = AgentToolResult(
                REQUEST_ID, "tool", outcome, None, "code", "message", NOW
            )
            self.assertIsNone(result.payload)
            self.assertEqual(result.error_code, "code")

    def test_outcome_invariants_fail_closed(self) -> None:
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(REQUEST_ID, "tool", AgentToolOutcome.OK, None, None, None, NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(REQUEST_ID, "tool", AgentToolOutcome.OK, {}, "code", None, NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(
                REQUEST_ID, "tool", AgentToolOutcome.EXECUTION_ERROR, {"x": 1}, "code", None, NOW
            )
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(REQUEST_ID, "tool", AgentToolOutcome.EXECUTION_ERROR, None, None, "msg", NOW)
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(
                REQUEST_ID, "tool", AgentToolOutcome.EXECUTION_ERROR, None, "code", None, NOW, replayed=1  # type: ignore[arg-type]
            )
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(
                REQUEST_ID, "tool", AgentToolOutcome.OK, {}, None, None, NOW.replace(tzinfo=None)
            )
        with self.assertRaises(AgentContractValidationError):
            AgentToolResult(REQUEST_ID, "tool", "not-an-outcome", None, "code", "msg", NOW)  # type: ignore[arg-type]


class CanonicalInterchangeTest(unittest.TestCase):
    def test_request_round_trip_preserves_structure(self) -> None:
        envelope = request(payload={"target_id": "trk_11111111-1111-4111-8111-111111111111"})
        decoded = decode_agent_request(encode_agent_request(envelope))
        self.assertEqual(decoded, envelope)

    def test_result_round_trip_preserves_structure(self) -> None:
        for result in (
            ok_result(payload={"state": "positive", "magnitude": 0.9}),
            AgentToolResult(
                REQUEST_ID, "tool", AgentToolOutcome.PERMISSION_DENIED, None, "permission_denied", "read-only client", NOW
            ),
        ):
            self.assertEqual(decode_agent_tool_result(encode_agent_tool_result(result)), result)

    def test_encoding_is_deterministic_and_order_independent(self) -> None:
        first = request(payload={"b": 2, "a": 1})
        second = request(payload={"a": 1, "b": 2})
        self.assertEqual(encode_agent_request(first), encode_agent_request(second))

    def test_payload_encoding_rejects_non_json_values(self) -> None:
        with self.assertRaises(AgentContractValidationError):
            encode_agent_payload({"value": float("nan")})
        with self.assertRaises(AgentContractValidationError):
            encode_agent_payload({"value": {1, 2}})

    def test_decode_fails_closed_on_malformed_text(self) -> None:
        for text in ("", "not json", "[1, 2]", '{"tool": "x"}', 42):  # type: ignore[arg-type]
            with self.assertRaises(AgentContractValidationError):
                decode_agent_request(text)
        for text in ("", "not json", "[1, 2]", '{"outcome": "ok"}', '{"outcome": "mystery"}', 42):  # type: ignore[arg-type]
            with self.assertRaises(AgentContractValidationError):
                decode_agent_tool_result(text)

    def test_decode_fails_closed_on_naive_timestamps(self) -> None:
        envelope = request()
        text = encode_agent_request(envelope).replace("+00:00", "", 1)
        with self.assertRaises(AgentContractValidationError):
            decode_agent_request(text)

    def test_replayed_flag_survives_interchange(self) -> None:
        result = ok_result(payload={"replayed": True})
        replayed = AgentToolResult(
            REQUEST_ID, result.tool, result.outcome, result.payload, None, None, NOW, replayed=True
        )
        self.assertTrue(decode_agent_tool_result(encode_agent_tool_result(replayed)).replayed)


if __name__ == "__main__":
    unittest.main()
