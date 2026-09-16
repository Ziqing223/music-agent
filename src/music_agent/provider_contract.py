"""P10.8: Provider-independent LLM chat contract.

The contract models exactly what a tool-calling chat provider needs: one round trip of
``chat(system, messages, tools) -> ProviderResponse``. Provider specifics (DeepSeek
endpoints, HTTP details, the Codex CLI invocation) live behind each adapter; the agent loop
(:class:`~music_agent.provider_agent.ProviderAgentLoop`) and everything else depend only
on this contract. P09 stays the only execution boundary: tool calls are never executed
here, only described and returned.

Credentials policy: an API key is configuration or environment only, never a default,
never logged, and excluded from dataclass ``repr``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


class ProviderError(RuntimeError):
    code = "provider_error"


class ProviderAuthError(ProviderError):
    """Credentials missing or rejected (401/403)."""

    code = "provider_auth_error"


class ProviderRateLimitError(ProviderError):
    """The provider throttled the request (429)."""

    code = "provider_rate_limit_error"


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached or failed (5xx, connection)."""

    code = "provider_unavailable_error"


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout_error"


class ProviderInvalidResponseError(ProviderError):
    """The provider returned something the adapter could not interpret."""

    code = "provider_invalid_response_error"


class ProviderMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ProviderStopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    """One tool invocation the provider requested (arguments stay JSON text)."""

    call_id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        for label, value in (("call_id", self.call_id), ("name", self.name)):
            if not isinstance(value, str) or value == "":
                raise ProviderError(f"{label} must be a non-empty string")
        if not isinstance(self.arguments, str):
            raise ProviderError("arguments must be a string")


@dataclass(frozen=True, slots=True)
class ProviderToolResult:
    """One tool execution result fed back to the provider."""

    call_id: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or self.call_id == "":
            raise ProviderError("call_id must be a non-empty string")
        if not isinstance(self.content, str):
            raise ProviderError("content must be a string")


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """One conversation message in the provider-independent shape.

    Kinds: plain ``text`` (system/user/assistant); assistant ``tool_calls``; assistant
    ``text`` AND ``tool_calls`` together (real providers emit a short preamble alongside
    tool calls -- the text is part of the assistant turn and must never be silently
    discarded); or user-side ``tool_results`` (one per tool_use id -- the transport
    contract requires a result for every requested call). ``tool_results`` is exclusive.

    ``wire_metadata`` is an opaque, adapter-owned passthrough for wire-protocol fields a
    provider REQUIRES to be echoed back on tool-call continuation (e.g. DeepSeek
    thinking-mode ``reasoning_content``). It is allowed only on assistant tool-call
    messages, the loop carries it back untouched without inspecting it, and it is never
    Music Agent semantic state -- P09 and every durable store never see it.
    """

    role: ProviderMessageRole
    text: str | None = None
    tool_calls: tuple[ProviderToolCall, ...] | None = None
    tool_results: tuple[ProviderToolResult, ...] | None = None
    wire_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ProviderMessageRole):
            raise ProviderError("role must be a ProviderMessageRole")
        if self.tool_results is not None and (self.text is not None or self.tool_calls is not None):
            raise ProviderError("tool_results must be the only payload kind")
        if self.text is None and self.tool_calls is None and self.tool_results is None:
            raise ProviderError("at least one of text/tool_calls/tool_results must be set")
        if self.role is not ProviderMessageRole.ASSISTANT and self.tool_calls is not None:
            raise ProviderError("only assistant messages can carry tool calls")
        if self.text is not None and not isinstance(self.text, str):
            raise ProviderError("text must be a string")
        if self.tool_calls is not None:
            if not self.tool_calls or not all(
                isinstance(call, ProviderToolCall) for call in self.tool_calls
            ):
                raise ProviderError("tool_calls must be a non-empty tuple of ProviderToolCall")
        if self.tool_results is not None:
            if not self.tool_results or not all(
                isinstance(result, ProviderToolResult) for result in self.tool_results
            ):
                raise ProviderError("tool_results must be a non-empty tuple of ProviderToolResult")
            if self.role is not ProviderMessageRole.USER:
                raise ProviderError("tool_result messages must be user-role")
        if self.wire_metadata is not None:
            if self.tool_calls is None:
                raise ProviderError("wire_metadata is only valid on assistant tool-call messages")
            if not isinstance(self.wire_metadata, Mapping):
                raise ProviderError("wire_metadata must be a mapping")
            object.__setattr__(
                self, "wire_metadata", MappingProxyType(dict(self.wire_metadata))
            )


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """One provider round-trip: the assistant message plus stop reason and usage."""

    message: ProviderMessage
    stop_reason: ProviderStopReason
    usage: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.message, ProviderMessage):
            raise ProviderError("message must be a ProviderMessage")
        if self.message.role is not ProviderMessageRole.ASSISTANT:
            raise ProviderError("provider responses must be assistant messages")
        if not isinstance(self.stop_reason, ProviderStopReason):
            raise ProviderError("stop_reason must be a ProviderStopReason")
        if not isinstance(self.usage, Mapping):
            raise ProviderError("usage must be a mapping")
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


@dataclass(frozen=True, slots=True)
class ProviderToolSchema:
    """The provider-facing description of one P09 agent tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ProviderError("name must be a non-empty string")
        if not isinstance(self.description, str) or self.description == "":
            raise ProviderError("description must be a non-empty string")
        if not isinstance(self.input_schema, Mapping):
            raise ProviderError("input_schema must be a mapping")
        if self.input_schema.get("type") != "object":
            raise ProviderError("input_schema must declare type object")
        object.__setattr__(self, "input_schema", MappingProxyType(dict(self.input_schema)))


class ChatProvider(Protocol):
    """One stateless chat round trip (providers are isolated behind this boundary)."""

    def chat(
        self,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ProviderToolSchema],
    ) -> ProviderResponse: ...


def resolve_api_key(api_key: str | None, api_key_env: str | None) -> str:
    """Resolve a provider credential; fail clearly (naming the variable, never the value)."""
    if api_key:
        return api_key
    if api_key_env:
        value = os.environ.get(api_key_env, "")
        if not value:
            raise ProviderAuthError(
                f"missing provider credential: environment variable {api_key_env!r} is not set"
            )
        return value
    raise ProviderAuthError(
        "missing provider credential: provide api_key or api_key_env in the provider config"
    )


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider configuration for one adapter instance.

    ``api_key`` holds the credential directly; ``api_key_env`` names an environment
    variable to read instead. Exactly one should be provided at use time -- the adapter
    fails with :class:`ProviderAuthError` naming the missing variable, never the value.
    """

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith("https://"):
            raise ProviderError("base_url must be an https:// string")
        if not isinstance(self.model, str) or self.model == "":
            raise ProviderError("model must be a non-empty string")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ProviderError("api_key must be a string or None")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str) or self.api_key_env == ""
        ):
            raise ProviderError("api_key_env must be a non-empty string or None")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ProviderError("timeout_seconds must be positive")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ProviderError("max_tokens must be a positive integer")
