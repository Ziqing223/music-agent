"""P09.6: AgentClient -- the model/client adapter boundary over the shared service.

An :class:`AgentClient` is the thinnest possible client side of the shared agent layer: one
registered :class:`AgentClientIdentity` plus one :class:`SharedAgentService` reference. It turns
a tool name and a bounded payload into an :class:`AgentRequest` and returns the service's
:class:`AgentToolResult`. It owns nothing else: no state, no caching, no user truth.

Provider independence is structural:

* The client's request identity is **client-owned**: ``call`` accepts an optional ``request_id``,
  so a client controls its own idempotency keys (same id + same payload = journaled replay;
  same id + different payload = fail-closed replay conflict).
* The client's model metadata (``model_id`` / ``label``) is provenance carried inside the
  request envelope; the service resolves permission strictly from the registered ``client_id``.
* Real provider SDKs (Codex, DeepSeek, ...) sit *outside* the core contract: they translate
  provider-native tool-call shapes into an ``AgentRequest`` (serialized with
  ``encode_agent_request``) and translate the returned ``AgentToolResult`` back
  (``decode_agent_tool_result``). No network dependency or provider SDK lives in this package;
  deterministic in-memory clients (as in the P09 integration tests) exercise the same boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentRequest,
    AgentToolResult,
    generate_request_id,
)
from music_agent.agent_service import SharedAgentService
from music_agent.agent_tools import AgentToolName
from music_agent.track_similarity import SimilarityExecutionContext


class AgentClientError(ValueError):
    code = "agent_client_error"


class AgentClientValidationError(AgentClientError):
    code = "validation_error"


class AgentClient:
    """One model client bound to one shared agent service."""

    def __init__(self, client: AgentClientIdentity, service: SharedAgentService) -> None:
        if not isinstance(client, AgentClientIdentity):
            raise AgentClientValidationError("client must be an AgentClientIdentity")
        if not isinstance(service, SharedAgentService):
            raise AgentClientValidationError("service must be a SharedAgentService")
        self._client = client
        self._service = service

    @property
    def client(self) -> AgentClientIdentity:
        return self._client

    @property
    def service(self) -> SharedAgentService:
        return self._service

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
        """Invoke one agent tool through the shared service.

        ``request_id`` is client-owned: omit it to mint a fresh one, or reuse it to exercise the
        service's replay contract. ``issued_at`` is the client's submission instant (injected for
        deterministic tests; defaults to the client's current UTC time). ``completed_at`` is
        passed through to the service for the same reason.

        P15-S3-S3D: ``fresh_canonical_ids`` is the INTERNAL same-run Fresh provenance transport
        (the provider loop's own capture from genuinely executed discoveries). It is a call-side
        execution fact, never part of the model-supplied ``payload`` or any tool schema; the
        model cannot set or influence it.
        ``recommendation_scope_ids`` is the analogous current-turn artist
        constraint: ``None`` means no hard scope, while a tuple is an
        application-resolved allow-list consumed only by recommendation
        generation.
        ``similarity_context`` is the analogous strict, code-owned seed authority. The service
        journals only its seed discriminator for durable replay protection; it never becomes a
        public Provider argument or handler payload.
        """
        if isinstance(tool, AgentToolName):
            tool = tool.value
        if not isinstance(tool, str) or tool == "":
            raise AgentClientValidationError("tool must be a non-empty string or AgentToolName")
        if not isinstance(payload, Mapping):
            raise AgentClientValidationError("payload must be a JSON object")
        if issued_at is None:
            issued_at = datetime.now(timezone.utc)
        request = AgentRequest(
            request_id=request_id or generate_request_id(),
            client=self._client,
            tool=tool,
            payload=payload,
            issued_at=issued_at,
        )
        return self._service.execute(
            request,
            completed_at=completed_at,
            fresh_canonical_ids=fresh_canonical_ids,
            recommendation_scope_ids=recommendation_scope_ids,
            similarity_context=similarity_context,
        )
