"""P09.1: the shared agent service contract -- identity, request/result envelopes, serialization.

The shared agent / service layer lets multiple model clients (Codex, DeepSeek, future models) act
as agents over one Music Agent core without owning any user-state truth. This module is the
side-effect-free contract both sides of that boundary share:

* ``AgentClientIdentity`` names a registered client (opaque ``agt_`` ID plus a model id as
  *provenance metadata only* -- the model id never grants, restricts, or stores anything).
* ``AgentRequest`` is one validated tool invocation: opaque ``req_`` identity, client, tool name,
  bounded JSON payload, injected tz-aware ``issued_at``, and the contract version.
* ``AgentToolResult`` is the execution envelope: a fixed fail-closed outcome vocabulary, a
  bounded JSON payload on success, and a stable ``error_code`` from the failure taxonomy on
  failure.
* The canonical deterministic JSON interchange (encode/decode pairs) is the provider-independent
  boundary: any model client serializes an ``AgentRequest`` and deserializes an
  ``AgentToolResult`` with these functions; no provider SDK or transport is implied.

Identity namespaces: ``agt_`` (agent-client) and ``req_`` (agent request) are disjoint from every
canonical (``trk_/art_/alb_/pl_/pm_``) and operational (``int_/att_/prb_/rec_/rcm_/cnd_/fbk_``)
namespace. Request identity is never derived from the payload, the client, or the tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4


class AgentContractError(ValueError):
    code = "agent_contract_error"


class AgentContractValidationError(AgentContractError):
    code = "validation_error"


# The current agent service contract. It scopes the request/result envelope semantics and the
# serialization identity under which a request executes. Bump it on any material change to those
# semantics; it does not version the tool set, permission policy, or domain pipelines the service
# orchestrates.
AGENT_CONTRACT_VERSION = 1

_CLIENT_ID_PREFIX = "agt_"
_REQUEST_ID_PREFIX = "req_"


class AgentToolOutcome(StrEnum):
    """The fail-closed outcome vocabulary of the shared agent service.

    Every executed (or refused) request lands in exactly one of these states. ``OK`` is the only
    outcome that carries a payload; every other outcome carries a stable ``error_code``.
    """

    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    REPLAY_CONFLICT = "replay_conflict"
    TOOL_NOT_SUPPORTED = "tool_not_supported"
    UNKNOWN_CLIENT = "unknown_client"
    PERMISSION_DENIED = "permission_denied"
    NOT_EXECUTION_READY = "not_execution_ready"
    EXECUTION_ERROR = "execution_error"


def generate_client_id() -> str:
    """Generate a stable agent-client identity.

    The ``agt_`` prefix is outside every canonical and operational namespace, so a client ID can
    never be confused with a track, intent, recommendation run, feedback observation, or request,
    and it is never derived from the client's model or label.
    """
    return f"{_CLIENT_ID_PREFIX}{uuid4()}"


def validate_client_id(client_id: str) -> None:
    if not isinstance(client_id, str) or not client_id.startswith(_CLIENT_ID_PREFIX):
        raise AgentContractValidationError(
            f"client_id must use the {_CLIENT_ID_PREFIX} namespace"
        )
    _require_uuid_suffix(client_id[len(_CLIENT_ID_PREFIX) :], label="client_id")


def generate_request_id() -> str:
    """Generate a stable agent-request identity.

    The ``req_`` prefix is outside every canonical, operational, and agent-client namespace, and a
    request ID is never derived from the payload, the client, or the tool, so replayed payloads
    cannot collide with one another by construction.
    """
    return f"{_REQUEST_ID_PREFIX}{uuid4()}"


def validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id.startswith(_REQUEST_ID_PREFIX):
        raise AgentContractValidationError(
            f"request_id must use the {_REQUEST_ID_PREFIX} namespace"
        )
    _require_uuid_suffix(request_id[len(_REQUEST_ID_PREFIX) :], label="request_id")


def _require_uuid_suffix(suffix: str, *, label: str) -> None:
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise AgentContractValidationError(f"{label} suffix must be a canonical UUID") from error
    if str(parsed) != suffix:
        raise AgentContractValidationError(f"{label} suffix must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class AgentClientIdentity:
    """The registered identity of one model client.

    ``client_id`` is the opaque registered identity (``agt_`` namespace). ``model_id`` and
    ``label`` are provenance metadata for humans and journals only: no permission, state, or
    routing decision may ever be derived from them.
    """

    client_id: str
    model_id: str
    label: str | None = None

    def __post_init__(self) -> None:
        validate_client_id(self.client_id)
        if not isinstance(self.model_id, str) or self.model_id == "":
            raise AgentContractValidationError("model_id must be a non-empty string")
        if self.label is not None and (
            not isinstance(self.label, str) or self.label == ""
        ):
            raise AgentContractValidationError("label must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """One validated tool invocation submitted by a model client.

    ``payload`` is bounded application input (a JSON object with string keys); it is request-scoped
    context, never durable state. ``issued_at`` is the injected tz-aware submission instant (the
    caller's clock; the service never timestamps a request itself).
    """

    request_id: str
    client: AgentClientIdentity
    tool: str
    payload: Mapping[str, Any]
    issued_at: datetime
    contract_version: int = AGENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        validate_request_id(self.request_id)
        if not isinstance(self.client, AgentClientIdentity):
            raise AgentContractValidationError("client must be an AgentClientIdentity")
        if not isinstance(self.tool, str) or self.tool == "":
            raise AgentContractValidationError("tool must be a non-empty string")
        normalized = _require_payload_object(self.payload)
        object.__setattr__(self, "payload", normalized)
        if (
            not isinstance(self.contract_version, int)
            or isinstance(self.contract_version, bool)
            or self.contract_version < 1
        ):
            raise AgentContractValidationError("contract_version must be a positive integer")
        _require_aware_datetime(self.issued_at, label="issued_at")


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """The execution envelope returned for one agent request.

    ``OK`` is the only outcome that carries a ``payload`` (a bounded JSON object) and no
    ``error_code``; every refusal or failure outcome carries a stable ``error_code`` and
    ``error_message`` and no payload. ``replayed`` marks a result returned from the durable
    request journal instead of a fresh execution. ``completed_at`` is the injected tz-aware
    completion instant.
    """

    request_id: str
    tool: str
    outcome: AgentToolOutcome
    payload: Mapping[str, Any] | None
    error_code: str | None
    error_message: str | None
    completed_at: datetime
    contract_version: int = AGENT_CONTRACT_VERSION
    replayed: bool = False

    def __post_init__(self) -> None:
        validate_request_id(self.request_id)
        if not isinstance(self.tool, str) or self.tool == "":
            raise AgentContractValidationError("tool must be a non-empty string")
        if not isinstance(self.outcome, AgentToolOutcome):
            raise AgentContractValidationError("outcome must be an AgentToolOutcome")
        if self.payload is not None:
            object.__setattr__(self, "payload", _require_payload_object(self.payload))
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or self.error_code == ""
        ):
            raise AgentContractValidationError("error_code must be None or a non-empty string")
        if self.error_message is not None and (
            not isinstance(self.error_message, str) or self.error_message == ""
        ):
            raise AgentContractValidationError("error_message must be None or a non-empty string")
        if not isinstance(self.replayed, bool):
            raise AgentContractValidationError("replayed must be a bool")
        if (
            not isinstance(self.contract_version, int)
            or isinstance(self.contract_version, bool)
            or self.contract_version < 1
        ):
            raise AgentContractValidationError("contract_version must be a positive integer")
        _require_aware_datetime(self.completed_at, label="completed_at")
        if self.outcome is AgentToolOutcome.OK:
            if self.payload is None:
                raise AgentContractValidationError("OK results must carry a payload")
            if self.error_code is not None:
                raise AgentContractValidationError("OK results must not carry an error_code")
        else:
            if self.payload is not None:
                raise AgentContractValidationError(
                    f"{self.outcome.value} results must not carry a payload"
                )
            if self.error_code is None:
                raise AgentContractValidationError(
                    f"{self.outcome.value} results must carry an error_code"
                )


def _require_payload_object(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise AgentContractValidationError("payload must be a JSON object")
    copied: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise AgentContractValidationError("payload keys must be strings")
        copied[key] = value
    return MappingProxyType(copied)


def _require_aware_datetime(value: object, *, label: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise AgentContractValidationError(f"{label} must be a timezone-aware datetime")


# --- canonical serialization ---------------------------------------------


def encode_agent_payload(payload: Mapping[str, Any]) -> str:
    """Encode a request/result payload object to its canonical deterministic JSON text.

    This is the equality and persistence form for payloads: sorted keys, no whitespace, no NaN or
    infinity. Fails closed on non-JSON-serializable payload values.
    """
    _require_payload_object(payload)
    try:
        return json.dumps(
            dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise AgentContractValidationError(f"payload is not JSON-serializable: {error}") from error


def encode_agent_request(request: AgentRequest) -> str:
    """Encode an agent request to its canonical JSON text form (the provider interchange)."""
    if not isinstance(request, AgentRequest):
        raise AgentContractValidationError("request must be an AgentRequest")
    return json.dumps(
        {
            "request_id": request.request_id,
            "client": {
                "client_id": request.client.client_id,
                "model_id": request.client.model_id,
                "label": request.client.label,
            },
            "tool": request.tool,
            "payload": dict(request.payload),
            "issued_at": request.issued_at.isoformat(),
            "contract_version": request.contract_version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def decode_agent_request(text: str) -> AgentRequest:
    """Decode a canonical JSON agent request back to an :class:`AgentRequest`.

    Fails closed on a non-string payload, unparseable JSON, or a decoded structure outside the
    contract, rather than coercing an unknown value.
    """
    if not isinstance(text, str):
        raise AgentContractValidationError("encoded request must be a string")
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise AgentContractValidationError(f"agent request is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise AgentContractValidationError("agent request must be a JSON object")
    expected = {"request_id", "client", "tool", "payload", "issued_at", "contract_version"}
    if set(data) != expected:
        raise AgentContractValidationError(
            f"agent request keys must be exactly {sorted(expected)}"
        )
    client_data = data["client"]
    if not isinstance(client_data, dict) or set(client_data) != {
        "client_id",
        "model_id",
        "label",
    }:
        raise AgentContractValidationError("request client must carry exactly its identity keys")
    client = AgentClientIdentity(
        client_id=client_data["client_id"],
        model_id=client_data["model_id"],
        label=client_data["label"],
    )
    payload = data["payload"]
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise AgentContractValidationError("request payload must be a JSON object")
    return AgentRequest(
        request_id=data["request_id"],
        client=client,
        tool=data["tool"],
        payload=payload,
        issued_at=_decode_aware_datetime(data["issued_at"], label="issued_at"),
        contract_version=data["contract_version"],
    )


def encode_agent_tool_result(result: AgentToolResult) -> str:
    """Encode an agent tool result to its canonical JSON text form (the provider interchange)."""
    if not isinstance(result, AgentToolResult):
        raise AgentContractValidationError("result must be an AgentToolResult")
    return json.dumps(
        {
            "request_id": result.request_id,
            "tool": result.tool,
            "outcome": result.outcome.value,
            "payload": None if result.payload is None else dict(result.payload),
            "error_code": result.error_code,
            "error_message": result.error_message,
            "completed_at": result.completed_at.isoformat(),
            "contract_version": result.contract_version,
            "replayed": result.replayed,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def decode_agent_tool_result(text: str) -> AgentToolResult:
    """Decode a canonical JSON agent tool result back to an :class:`AgentToolResult`.

    Fails closed on a non-string payload, unparseable JSON, or a decoded structure outside the
    contract.
    """
    if not isinstance(text, str):
        raise AgentContractValidationError("encoded result must be a string")
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise AgentContractValidationError(f"agent tool result is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise AgentContractValidationError("agent tool result must be a JSON object")
    expected = {
        "request_id",
        "tool",
        "outcome",
        "payload",
        "error_code",
        "error_message",
        "completed_at",
        "contract_version",
        "replayed",
    }
    if set(data) != expected:
        raise AgentContractValidationError(
            f"agent tool result keys must be exactly {sorted(expected)}"
        )
    try:
        outcome = AgentToolOutcome(data["outcome"])
    except ValueError as error:
        raise AgentContractValidationError(
            f"unknown outcome {data['outcome']!r}"
        ) from error
    payload = data["payload"]
    if payload is not None and (
        not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload)
    ):
        raise AgentContractValidationError("result payload must be a JSON object or null")
    return AgentToolResult(
        request_id=data["request_id"],
        tool=data["tool"],
        outcome=outcome,
        payload=payload,
        error_code=data["error_code"],
        error_message=data["error_message"],
        completed_at=_decode_aware_datetime(data["completed_at"], label="completed_at"),
        contract_version=data["contract_version"],
        replayed=data["replayed"],
    )


def _decode_aware_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise AgentContractValidationError(f"{label} must be an ISO datetime") from error
    _require_aware_datetime(parsed, label=label)
    return parsed
