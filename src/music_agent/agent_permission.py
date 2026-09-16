"""P09.3: the agent-facing permission decision point -- policy above raw capability.

Every agent request passes one permission decision before any execution. The decision composes
two layers, evaluated strictly in order:

1. **Client policy** -- ``AgentClientRegistry`` maps a registered ``agt_`` client identity to an
   :class:`AgentClientPolicy` (FULL / READ_ONLY / NONE). Policy binds to the *client id*, never to
   the model id: a client's model metadata (Codex, DeepSeek, anything) is provenance, and no
   model identity can elevate or bypass the registered policy. An unregistered client id is
   unknown and fails closed.
2. **Capability / readiness** -- tools in the ``live_write`` class (Apple Music writes) are
   additionally gated by the existing ``WRITE_CAPABILITY_MATRIX`` +
   ``is_execution_ready`` predicate from P02--P05. This module never reimplements that predicate:
   it delegates, so agent requests can never become a path around the fail-closed write safety
   the write orchestrator already enforces. Today no write operation is execution-ready, so
   every live-write tool call fails closed with ``not_execution_ready``.

The decision output is a :class:`PermissionDenial` (or ``None`` for permitted), carrying the
stable outcome and error code the service surface echoes into the result envelope. The module is
side-effect-free: the service resolves any intent/operation context and passes it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from music_agent.agent_contract import AgentToolOutcome, validate_client_id
from music_agent.agent_tools import AgentToolPermissionClass
from music_agent.write_intent import (
    WRITE_CAPABILITY_MATRIX,
    DomainPermission,
    WriteCapability,
    WriteOperation,
    is_execution_ready,
)


class AgentPermissionError(ValueError):
    code = "agent_permission_error"


class AgentPermissionValidationError(AgentPermissionError):
    code = "validation_error"


class AgentClientPolicy(StrEnum):
    """What a registered client is permitted to invoke.

    ``FULL`` permits read and mutate tools (live writes remain capability-gated). ``READ_ONLY``
    permits read tools only. ``NONE`` permits nothing.
    """

    FULL = "full"
    READ_ONLY = "read_only"
    NONE = "none"


class AgentClientRegistry:
    """The registered client surface, keyed by ``agt_`` client id.

    Registration is in-memory configuration (who may call what), not durable user state. Policy
    is keyed strictly by client id; model/label metadata never participates in lookup.
    """

    def __init__(self, registrations: Mapping[str, AgentClientPolicy]) -> None:
        copied: dict[str, AgentClientPolicy] = {}
        for client_id, policy in registrations.items():
            try:
                validate_client_id(client_id)
            except ValueError as error:
                raise AgentPermissionValidationError(str(error)) from error
            if not isinstance(policy, AgentClientPolicy):
                raise AgentPermissionValidationError(
                    f"policy for {client_id} must be an AgentClientPolicy"
                )
            copied[client_id] = policy
        self._policies: Mapping[str, AgentClientPolicy] = MappingProxyType(copied)

    def policy_for(self, client_id: str) -> AgentClientPolicy | None:
        """Return the registered policy for a client id, or ``None`` for an unknown client."""
        if not isinstance(client_id, str):
            return None
        return self._policies.get(client_id)

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(self._policies)


@dataclass(frozen=True, slots=True)
class PermissionDenial:
    """A fail-closed refusal: the outcome, stable error code, and human-readable message."""

    outcome: AgentToolOutcome
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, AgentToolOutcome):
            raise AgentPermissionValidationError("outcome must be an AgentToolOutcome")
        if not isinstance(self.code, str) or self.code == "":
            raise AgentPermissionValidationError("code must be a non-empty string")
        if not isinstance(self.message, str) or self.message == "":
            raise AgentPermissionValidationError("message must be a non-empty string")


def decide_agent_permission(
    permission_class: AgentToolPermissionClass,
    policy: AgentClientPolicy | None,
    *,
    live_write_operation: WriteOperation | None = None,
    capability_matrix: Mapping[WriteOperation, WriteCapability] = WRITE_CAPABILITY_MATRIX,
) -> PermissionDenial | None:
    """Decide whether one tool invocation is permitted; ``None`` means permitted.

    ``policy`` is the registered policy for the request's client id (``None`` when the client is
    unregistered). ``live_write_operation`` is required for live-write tools: the resolved write
    operation the intent targets. ``capability_matrix`` is injectable for tests but defaults to
    the sealed production matrix.
    """
    if not isinstance(permission_class, AgentToolPermissionClass):
        raise AgentPermissionValidationError(
            "permission_class must be an AgentToolPermissionClass"
        )
    if policy is None:
        return PermissionDenial(
            AgentToolOutcome.UNKNOWN_CLIENT,
            "unknown_client",
            "client id is not registered with the shared agent service",
        )
    if policy is AgentClientPolicy.NONE:
        return PermissionDenial(
            AgentToolOutcome.PERMISSION_DENIED,
            "permission_denied",
            "client policy forbids every tool invocation",
        )
    if (
        policy is AgentClientPolicy.READ_ONLY
        and permission_class is not AgentToolPermissionClass.READ
    ):
        return PermissionDenial(
            AgentToolOutcome.PERMISSION_DENIED,
            "permission_denied",
            f"read-only client policy forbids {permission_class.value} tools",
        )
    if permission_class is AgentToolPermissionClass.LIVE_WRITE:
        if live_write_operation is None:
            raise AgentPermissionValidationError(
                "live-write permission decisions require a resolved write operation"
            )
        capability = capability_matrix.get(live_write_operation)
        if capability is None:
            return PermissionDenial(
                AgentToolOutcome.NOT_EXECUTION_READY,
                "not_execution_ready",
                f"write operation {live_write_operation.value} has no capability entry",
            )
        if not is_execution_ready(capability):
            return PermissionDenial(
                AgentToolOutcome.NOT_EXECUTION_READY,
                "not_execution_ready",
                _readiness_message(capability),
            )
    return None


def _readiness_message(capability: WriteCapability) -> str:
    missing: list[str] = []
    if capability.domain_permission is not DomainPermission.ALLOWED:
        missing.append("domain permission is not allowed")
    if not capability.capability_verified:
        missing.append("capability is not verified")
    if not capability.adapter_implemented:
        missing.append("adapter is not implemented")
    if not capability.readback_implemented:
        missing.append("readback is not implemented")
    return (
        f"write operation {capability.operation.value} is not execution-ready: "
        + "; ".join(missing)
    )


def write_capability_summary(
    capability_matrix: Mapping[WriteOperation, WriteCapability] = WRITE_CAPABILITY_MATRIX,
) -> tuple[dict[str, object], ...]:
    """Project the write capability matrix into the stable agent-facing read surface.

    Each entry carries the operation, its sealed permission/verification/implementation facts,
    and the derived ``execution_ready`` verdict. This is the read-only shape the
    ``get_agent_capabilities`` tool returns; it exposes the gate, never a way around it.
    """
    return tuple(
        {
            "operation": capability.operation.value,
            "entity_type": capability.entity_type.value,
            "domain_permission": capability.domain_permission.value,
            "capability_verified": capability.capability_verified,
            "adapter_implemented": capability.adapter_implemented,
            "readback_verified": capability.readback_verified,
            "readback_implemented": capability.readback_implemented,
            "readback_strategy": capability.readback_strategy.value,
            "execution_ready": is_execution_ready(capability),
        }
        for capability in capability_matrix.values()
    )
