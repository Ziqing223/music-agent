"""P15-S2-IPC S1: agent-socket transport primitives (framing/canonical interchange).

The primitives are exercised over a real ``socketpair`` where possible; the wire
contract is the P09 provider interchange (``agent_contract`` encode/decode
pairs), so every roundtrip here also proves the framed bytes decode back through
that boundary unchanged.
"""

import socket
import struct
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentContractValidationError,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_socket import (
    AGENT_RUNTIME_OFFLINE_CODE,
    MAX_FRAME_BYTES,
    AgentSocketError,
    SocketFrameError,
    agent_socket_path,
    build_offline_refusal,
    frame_to_request,
    frame_to_result,
    preview_event_socket_path,
    receive_framed,
    request_to_frame,
    result_to_frame,
    send_framed,
)

_REQUEST_ID = "req_00000000-0000-4000-8000-000000000001"
_CLIENT_ID = "agt_10000000-0000-4000-8000-000000000001"
_INSTANT = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)


def example_request(payload: dict | None = None) -> AgentRequest:
    return AgentRequest(
        request_id=_REQUEST_ID,
        client=AgentClientIdentity(client_id=_CLIENT_ID, model_id="test-model"),
        tool="preview_batch",
        payload=payload if payload is not None else {},
        issued_at=_INSTANT,
    )


def example_ok_result() -> AgentToolResult:
    return AgentToolResult(
        request_id=_REQUEST_ID,
        tool="preview_batch",
        outcome=AgentToolOutcome.OK,
        payload={"started": True, "session": {"state": "running"}},
        error_code=None,
        error_message=None,
        completed_at=_INSTANT,
        replayed=True,
    )


class SocketPair:
    def __enter__(self):
        left, right = socket.socketpair()
        self.left = left
        self.right = right
        return left, right

    def __exit__(self, exc_type, exc, tb):
        self.left.close()
        self.right.close()


class FramingTest(unittest.TestCase):
    def test_frame_exact_bytes(self) -> None:
        with SocketPair() as (left, right):
            send_framed(left, b"hi")
            self.assertEqual(right.recv(6), b"\x00\x00\x00\x02hi")

    def test_request_roundtrip_over_socketpair(self) -> None:
        with SocketPair() as (left, right):
            send_framed(left, request_to_frame(example_request({"queue": "next"})))
            decoded = frame_to_request(receive_framed(right))
        self.assertEqual(decoded.request_id, _REQUEST_ID)
        self.assertEqual(decoded.client.client_id, _CLIENT_ID)
        self.assertEqual(decoded.tool, "preview_batch")
        self.assertEqual(dict(decoded.payload), {"queue": "next"})
        self.assertEqual(decoded.issued_at, _INSTANT)

    def test_result_roundtrip_ok_payload_with_replayed_flag(self) -> None:
        with SocketPair() as (left, right):
            send_framed(left, result_to_frame(example_ok_result()))
            decoded = frame_to_result(receive_framed(right))
        self.assertEqual(decoded.outcome, AgentToolOutcome.OK)
        self.assertEqual(dict(decoded.payload or {}), {"started": True, "session": {"state": "running"}})
        self.assertTrue(decoded.replayed)
        self.assertIsNone(decoded.error_code)

    def test_result_roundtrip_error_result(self) -> None:
        refusal = build_offline_refusal(example_request(), completed_at=_INSTANT)
        with SocketPair() as (left, right):
            send_framed(left, result_to_frame(refusal))
            decoded = frame_to_result(receive_framed(right))
        self.assertEqual(decoded.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(decoded.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        self.assertIsNone(decoded.payload)

    def test_truncated_header_fails_closed(self) -> None:
        with SocketPair() as (left, right):
            left.send(struct.pack(">I", 5)[:2])
            left.shutdown(socket.SHUT_WR)
            with self.assertRaises(SocketFrameError):
                receive_framed(right)

    def test_truncated_body_fails_closed(self) -> None:
        with SocketPair() as (left, right):
            left.send(struct.pack(">I", 10) + b"abc")
            left.shutdown(socket.SHUT_WR)
            with self.assertRaises(SocketFrameError):
                receive_framed(right)

    def test_zero_length_body_refused(self) -> None:
        with SocketPair() as (left, right):
            left.send(struct.pack(">I", 0))
            left.shutdown(socket.SHUT_WR)
            with self.assertRaises(SocketFrameError):
                receive_framed(right)

    def test_oversized_length_refused_before_body_read(self) -> None:
        with SocketPair() as (left, right):
            left.send(struct.pack(">I", MAX_FRAME_BYTES + 1))
            with self.assertRaises(SocketFrameError):
                receive_framed(right)

    def test_send_rejects_oversized_body(self) -> None:
        with SocketPair() as (left, _right):
            with self.assertRaises(SocketFrameError):
                send_framed(left, b"x" * (MAX_FRAME_BYTES + 1))

    def test_send_rejects_non_bytes_and_empty_body(self) -> None:
        with SocketPair() as (left, _right):
            with self.assertRaises(AgentSocketError):
                send_framed(left, "text")  # type: ignore[arg-type]
            with self.assertRaises(SocketFrameError):
                send_framed(left, b"")

    def test_invalid_utf8_frame_fails_closed(self) -> None:
        with SocketPair() as (left, right):
            send_framed(left, b'\xff\xfe{"not": "text"}')
            with self.assertRaises(SocketFrameError):
                frame_to_request(receive_framed(right))

    def test_undecodable_json_fails_closed(self) -> None:
        with SocketPair() as (left, right):
            send_framed(left, b"not json at all")
            with self.assertRaises(AgentContractValidationError):
                frame_to_request(receive_framed(right))

    def test_construction_validates_argument_types(self) -> None:
        self.assertRaises(AgentSocketError, request_to_frame, object())  # type: ignore[arg-type]
        self.assertRaises(AgentSocketError, result_to_frame, object())  # type: ignore[arg-type]
        self.assertRaises(AgentSocketError, frame_to_request, "text")  # type: ignore[arg-type]
        self.assertRaises(AgentSocketError, frame_to_result, "text")  # type: ignore[arg-type]


class PathDerivationTest(unittest.TestCase):
    def test_service_socket_derives_from_store(self) -> None:
        self.assertEqual(
            agent_socket_path(Path("/Users/x/store.db")),
            Path("/Users/x/store.db.agent.sock"),
        )

    def test_event_socket_carries_client_pid(self) -> None:
        self.assertEqual(
            preview_event_socket_path(Path("/Users/x/store.db"), 4242),
            Path("/Users/x/store.db.agent-events-4242.sock"),
        )

    def test_event_socket_rejects_bad_pid(self) -> None:
        self.assertRaises(AgentSocketError, preview_event_socket_path, Path("store.db"), 0)
        self.assertRaises(AgentSocketError, preview_event_socket_path, Path("store.db"), "9")  # type: ignore[arg-type]


class OfflineRefusalTest(unittest.TestCase):
    def test_refusal_is_a_valid_execution_envelope(self) -> None:
        refusal = build_offline_refusal(example_request(), completed_at=_INSTANT)
        self.assertEqual(refusal.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refusal.error_code, AGENT_RUNTIME_OFFLINE_CODE)
        self.assertIsNotNone(refusal.error_message)
        self.assertIsNone(refusal.payload)
        self.assertEqual(refusal.request_id, _REQUEST_ID)
        self.assertEqual(refusal.tool, "preview_batch")

    def test_refusal_rejects_non_request(self) -> None:
        self.assertRaises(AgentSocketError, build_offline_refusal, object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()