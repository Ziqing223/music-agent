"""P10.8: DeepSeek API provider (OpenAI-compatible chat completions, stdlib HTTP).

First-class provider per the P10.8 amendment: the DeepSeek API with credentials from
``DEEPSEEK_API_KEY`` (or explicit runtime configuration) only. The adapter speaks the
OpenAI-compatible shape (``POST {base}/chat/completions``, ``Authorization: Bearer``,
function-style tools) against ``https://api.deepseek.com/v1`` by default, and maps every
failure to the shared typed provider-error vocabulary through the shared transport.
Secrets are never logged and never embedded in error messages.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from music_agent.provider_contract import (
    ProviderAuthError,
    resolve_api_key,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderStopReason,
    ProviderToolCall,
    ProviderToolSchema,
    ProviderUnavailableError,
)
from music_agent.provider_transport import HttpTransport, UrllibHttpTransport, guarded_post

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekApiProvider:
    supports_turn_interpreter = True

    """ChatProvider over the DeepSeek OpenAI-compatible chat completions API."""

    def __init__(self, config, transport: HttpTransport | None = None) -> None:
        from music_agent.provider_contract import ProviderConfig

        if not isinstance(config, ProviderConfig):
            raise ProviderError("config must be a ProviderConfig")
        if transport is not None and not isinstance(transport, HttpTransport):
            raise ProviderError("transport must implement HttpTransport")
        self.config = config
        self._transport = transport or UrllibHttpTransport()

    def chat(
        self,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ProviderToolSchema],
    ) -> ProviderResponse:
        if not isinstance(system, str):
            raise ProviderError("system must be a string")
        api_key = resolve_api_key(self.config.api_key, self.config.api_key_env)
        wire_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            wire_messages.extend(_encode_message(message))
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": wire_messages,
        }
        # S1: a zero-tool turn (plain chat) omits BOTH keys entirely -- an
        # empty tools array with tool_choice "auto" is a rejected wire shape on
        # OpenAI-compatible APIs, and sending no schema is the whole point.
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in tools
            ]
            body["tool_choice"] = "auto"
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        status, raw = guarded_post(
            self._transport,
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            body=body,
            timeout=self.config.timeout_seconds,
        )
        return _parse_response(status, raw)


def _encode_message(message: ProviderMessage) -> list[dict[str, Any]]:
    """Encode one contract message as one-or-more wire messages (tool results flatten).

    Tool-call messages are checked first: a mixed message (preamble text + tool_calls)
    must encode as the assistant tool-call wire message with the text, the echoed
    reasoning metadata, and the tool calls together.
    """
    if message.tool_calls is not None:
        wire_message: dict[str, Any] = {
            "role": "assistant",
            # DeepSeek documents the continuation assistant tool-call message with an
            # empty-string content (never null) plus the echoed reasoning_content.
            "content": message.text or "",
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ],
        }
        if message.wire_metadata:
            wire_message.update(dict(message.wire_metadata))
        return [wire_message]
    if message.text is not None:
        return [{"role": message.role.value, "content": message.text}]
    assert message.tool_results is not None
    return [
        {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.content,
        }
        for result in message.tool_results
    ]


def _parse_response(status: int, raw: str) -> ProviderResponse:
    if status == 401 or status == 403:
        raise ProviderAuthError("provider rejected the credential (HTTP 401/403)")
    if status == 429:
        raise ProviderRateLimitError("provider rate limit reached (HTTP 429)")
    if status >= 500:
        raise ProviderUnavailableError(_format_http_error(status, raw))
    if status != 200:
        raise ProviderUnavailableError(_format_http_error(status, raw))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProviderInvalidResponseError("provider response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ProviderInvalidResponseError("provider response must be a JSON object")
    try:
        choices = payload["choices"]
        if not isinstance(choices, list) or not choices:
            raise ProviderInvalidResponseError("provider choices must be a non-empty list")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ProviderInvalidResponseError("provider message must be an object")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderInvalidResponseError("provider content must be a string or null")
        tool_calls_raw = message.get("tool_calls") or []
        tool_calls: list[ProviderToolCall] = []
        for raw_call in tool_calls_raw:
            if not isinstance(raw_call, dict):
                raise ProviderInvalidResponseError("tool_calls entries must be objects")
            function = raw_call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ProviderInvalidResponseError("tool_call function must carry a name")
            if not isinstance(function.get("arguments"), str) and function.get("arguments") is not None:
                raise ProviderInvalidResponseError("tool_call arguments must be a JSON string")
            tool_calls.append(
                ProviderToolCall(
                    call_id=str(raw_call.get("id", "")),
                    name=function["name"],
                    arguments=(
                        function.get("arguments") if isinstance(function.get("arguments"), str) else "{}"
                    ),
                )
            )
        if content is None and not tool_calls:
            raise ProviderInvalidResponseError(
                "provider message must carry content and/or tool_calls"
            )
        wire_metadata: dict[str, Any] | None = None
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            if not isinstance(reasoning_content, str):
                raise ProviderInvalidResponseError("reasoning_content must be a string")
            wire_metadata = {"reasoning_content": reasoning_content}
        message_obj = ProviderMessage(
            role=ProviderMessageRole.ASSISTANT,
            text=content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
            wire_metadata=wire_metadata,
        )
        stop_reason = _parse_finish_reason(str(choices[0].get("finish_reason", "")))
        return ProviderResponse(
            message=message_obj,
            stop_reason=stop_reason,
            usage=_normalize_usage(payload.get("usage")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderInvalidResponseError(str(error)) from error


_REQUIRED_NUMERIC_USAGE_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})


def _format_http_error(status: int, raw: str) -> str:
    """Render a sanitized provider error: only the provider's own error code/message.

    Never the request content, never headers, never the credential -- the error text is
    built exclusively from the provider's error envelope.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return f"provider returned HTTP {status}"
    if not isinstance(payload, dict):
        return f"provider returned HTTP {status}"
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, (str, int)) or (isinstance(message, str) and message):
            parts = [f"provider returned HTTP {status}"]
            if isinstance(code, (str, int)):
                parts.append(f"code={code}")
            if isinstance(message, str) and message.strip():
                parts.append(f"message={message.strip()[:500]}")
            return ": ".join(parts)
    if isinstance(error, str) and error.strip():
        return f"provider returned HTTP {status}: {error.strip()[:500]}"
    return f"provider returned HTTP {status}"


def _normalize_usage(usage_raw: Any) -> dict[str, int]:
    """Normalize the provider usage object into the contract's Mapping[str, int].

    Real OpenAI-compatible responses nest detail objects (``prompt_tokens_details``,
    ``completion_tokens_details``, ...); those are provider metadata the contract does
    not model and are safely ignored -- never ``int()``-coerced. Scalar numeric values
    (int, integral float, numeric string) are kept as ints. A core token-count field
    that is present but non-numeric is a contract-shape violation and fails closed;
    other non-numeric optional scalar fields are ignored.
    """
    if not isinstance(usage_raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in usage_raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, dict):
            if key in _REQUIRED_NUMERIC_USAGE_FIELDS:
                raise ProviderInvalidResponseError(f"usage {key!r} must be numeric")
            continue  # nested detail objects: unsupported optional detail, safely ignored
        if isinstance(value, bool):
            if key in _REQUIRED_NUMERIC_USAGE_FIELDS:
                raise ProviderInvalidResponseError(f"usage {key!r} must be numeric")
            continue
        if isinstance(value, int):
            normalized[key] = value
            continue
        if isinstance(value, float) and value.is_integer():
            normalized[key] = int(value)
            continue
        if isinstance(value, str) and value.strip().isdigit():
            normalized[key] = int(value)
            continue
        if key in _REQUIRED_NUMERIC_USAGE_FIELDS:
            raise ProviderInvalidResponseError(f"usage {key!r} must be numeric")
        # Unsupported optional scalar fields: safely ignored.
    return normalized


def _parse_finish_reason(raw: str) -> ProviderStopReason:
    if raw == "stop":
        return ProviderStopReason.END_TURN
    if raw == "tool_calls":
        return ProviderStopReason.TOOL_USE
    if raw == "length":
        return ProviderStopReason.MAX_TOKENS
    return ProviderStopReason.OTHER
