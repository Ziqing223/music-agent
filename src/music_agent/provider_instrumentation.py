"""P15-S4-M1: cost instrumentation for the provider agent loop.

Pure measurement and rendering -- the loop (:class:`~music_agent.provider_agent.ProviderAgentLoop`)
is wired to accumulate these records when ``ProviderLoopConfig.instrument`` is
enabled (default off), and the CLI writes them as one JSONL line per request.

Truthfulness rules (never violated):

- ``usage`` carries exactly what the provider returned (``ProviderResponse.usage``),
  as a ``Mapping[str, int]`` -- empty when the provider returned none. No token
  estimates are ever invented here.
- All sizes are *character counts* of payloads this process actually built
  (``*_chars``), never token counts and never wire bytes: ``input_chars`` is the
  contract-level input (system prompt + message payloads + tool schemas), which
  is what both providers re-send every round, and the name says so.
- All latencies are local ``time.monotonic`` measurements in milliseconds
  (``*_ms``). A ``duration_ms`` of ``None`` on a tool record means the tool call
  was not executed (dedupe / in-run read-cache hit / synthesized refusal).
- Missing data is explicit (empty mapping / ``None``), never a fabricated value.
- Tool ``arguments`` are recorded only when instrumentation is enabled, parsed
  to structured JSON and passed through a minimal sensitive-key redaction (see
  :func:`safe_tool_arguments`). Unparseable arguments become an explicit
  unavailable marker -- never the raw text, never an invented value, and never
  a failure the loop can see.

Nothing here changes loop semantics: the loop executes the same tool calls in the
same order with the same results whether instrumentation is on or off -- it only
also records measurements alongside.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


# P15-S4-M2 (diagnostic): minimal defense-in-depth redaction for tool arguments.
# The current agent tool schemas carry no sensitive fields (code-verified:
# every argument key is a domain id -- trk_/rcm_/cnd_ target ids, candidate
# ids, genres, terms, limits -- and credentials live in provider configuration,
# never tool payloads). The filter below is a cheap guarantee that a future
# tool surface cannot leak a credential into a trace file.
_SENSITIVE_ARGUMENT_KEY_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "credential",
    "authorization",
    "secret",
    "password",
    "passwd",
    "private",
    "signature",
    "signed",
)
_REDACTED_ARGUMENT_VALUE = "[REDACTED]"
_ARGUMENTS_UNAVAILABLE_KEY = "__unavailable__"


def _redact_arguments(value: Any) -> Any:
    """Recursive minimal redaction for trace-safe tool arguments.

    A value under a key whose lowercase name contains a sensitive marker is
    replaced wholesale; every other container is traversed so nested secrets
    are caught too. Non-finite floats are replaced by their string form so the
    emitted JSONL stays strict JSON even for model-provided NaN/inf payloads.
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(marker in lower for marker in _SENSITIVE_ARGUMENT_KEY_MARKERS):
                redacted[str(key)] = _REDACTED_ARGUMENT_VALUE
            else:
                redacted[str(key)] = _redact_arguments(item)
        return redacted
    if isinstance(value, list):
        return [_redact_arguments(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def safe_tool_arguments(arguments_text: str) -> dict[str, Any]:
    """Parse and minimally redact one tool call's raw arguments JSON for the
    trace. Never raises, never affects the agent loop, and its result is always
    a JSON-serializable mapping: either the structured (redacted) arguments, or
    an explicit ``{"__unavailable__": <reason>}`` marker. The raw text is never
    recorded -- unparseable free-form strings could contain anything.
    """
    try:
        parsed = json.loads(arguments_text)
    except (json.JSONDecodeError, RecursionError):
        return {_ARGUMENTS_UNAVAILABLE_KEY: "arguments are not valid JSON"}
    if not isinstance(parsed, dict):
        return {_ARGUMENTS_UNAVAILABLE_KEY: "arguments are not a JSON object"}
    try:
        redacted = _redact_arguments(dict(parsed))
    except RecursionError:
        # A pathological payload can parse in the C scanner but still overflow
        # the redaction walk -- the marker keeps the trace safe and the loop
        # untouched.
        return {_ARGUMENTS_UNAVAILABLE_KEY: "arguments are too deeply nested"}
    try:
        json.dumps(redacted, ensure_ascii=False)  # guarantee: trace lines always serialize
    except (TypeError, ValueError, RecursionError):
        return {_ARGUMENTS_UNAVAILABLE_KEY: "arguments are not JSON-serializable"}
    return redacted


def _ms(value: float) -> float:
    """One-decimal millisecond rounding for emitted records."""
    return round(value, 1)


@dataclass(frozen=True, slots=True)
class RoundInputMeasure:
    """Measurable input size of one provider round (chars, not tokens)."""

    input_chars: int
    tool_schemas_chars: int
    tool_schemas_count: int
    messages_count: int


@dataclass(frozen=True, slots=True)
class ProviderRoundMeasure:
    """One provider round trip as measured by the loop."""

    round_index: int
    provider: str
    provider_latency_ms: float
    input_chars: int
    tool_schemas_chars: int
    tool_schemas_count: int
    messages_count: int
    tool_calls_count: int
    usage: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "provider": self.provider,
            "provider_latency_ms": _ms(self.provider_latency_ms),
            "input_chars": self.input_chars,
            "tool_schemas_chars": self.tool_schemas_chars,
            "tool_schemas_count": self.tool_schemas_count,
            "messages_count": self.messages_count,
            "tool_calls_count": self.tool_calls_count,
            "usage": dict(self.usage),
        }


@dataclass(frozen=True, slots=True)
class ProviderToolMeasure:
    """One tool call measured by the loop.

    ``origin`` distinguishes provider-requested calls from deterministic
    policy injections and the S5 prefetch.  It is trace-only metadata: the
    durable P09 request journal remains unchanged.

    ``duration_ms`` is ``None`` when the call was not executed: same-round
    dedupe (``outcome="deduped"``), in-run read-cache hit (``outcome="cached"``),
    a generation-budget synthesized refusal, or invalid arguments. Every such
    call still records its delivered size -- the provider gets a result either
    way.
    """

    round_index: int
    call_index: int
    name: str
    outcome: str
    error_code: str | None
    duration_ms: float | None
    raw_result_chars: int
    delivered_result_chars: int
    truncated: bool
    replayed: bool
    executed: bool
    # P15-S4-M2 (diagnostic): the call's raw arguments JSON, parsed and
    # minimally redacted (see safe_tool_arguments). The loop always supplies
    # it under instrumentation; None only on records constructed without it.
    arguments: Mapping[str, Any] | None = None
    origin: str = "provider_requested"

    @property
    def succeeded(self) -> bool:
        return self.executed and self.outcome == "ok" and self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "call_index": self.call_index,
            "name": self.name,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "duration_ms": _ms(self.duration_ms) if self.duration_ms is not None else None,
            "raw_result_chars": self.raw_result_chars,
            "delivered_result_chars": self.delivered_result_chars,
            "truncated": self.truncated,
            "replayed": self.replayed,
            "executed": self.executed,
            "arguments": dict(self.arguments) if self.arguments is not None else None,
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class ProviderRunTrace:
    """All measurements of one agent run (one user request)."""

    total_ms: float
    rounds_capped: bool
    context_trimmed: bool
    rounds: tuple[ProviderRoundMeasure, ...]
    tools: tuple[ProviderToolMeasure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", tuple(self.rounds))
        object.__setattr__(self, "tools", tuple(self.tools))

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def total_provider_ms(self) -> float:
        """Sum of per-round provider latency (local measurements)."""
        return sum(record.provider_latency_ms for record in self.rounds)

    @property
    def total_tool_ms(self) -> float:
        """Sum of tool execution durations; non-executed calls contribute 0."""
        return sum(
            record.duration_ms
            for record in self.tools
            if record.duration_ms is not None
        )

    @property
    def tool_calls_total(self) -> int:
        """All recorded calls, including deterministic policy injections."""
        return len(self.tools)

    @property
    def executed_tool_calls_total(self) -> int:
        return sum(1 for record in self.tools if record.executed)

    @property
    def truncated_count(self) -> int:
        return sum(1 for record in self.tools if record.truncated)

    @property
    def usage_total(self) -> dict[str, int]:
        """Real provider usage summed per field across rounds. Empty when the
        provider returned no usage at all -- never an estimate."""
        totals: dict[str, int] = {}
        for record in self.rounds:
            for key, value in record.usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return totals

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable request aggregate (nested round/tool records)."""
        return {
            "total_ms": _ms(self.total_ms),
            "total_rounds": self.total_rounds,
            "total_provider_ms": _ms(self.total_provider_ms),
            "total_tool_ms": _ms(self.total_tool_ms),
            "tool_calls_total": self.tool_calls_total,
            "executed_tool_calls_total": self.executed_tool_calls_total,
            "truncated_count": self.truncated_count,
            "usage_total": self.usage_total,
            "rounds_capped": self.rounds_capped,
            "context_trimmed": self.context_trimmed,
            "rounds": [record.to_dict() for record in self.rounds],
            "tools": [record.to_dict() for record in self.tools],
        }


def measure_round_input(system: str, messages: Sequence[Any], tools: Sequence[Any]) -> RoundInputMeasure:
    """Contract-level input size of one round, in chars (never tokens).

    ``input_chars`` counts the system prompt, every message payload the loop will
    pass to the provider (user/assistant text, tool-call argument JSON, tool
    result content), and every tool schema name/description/input_schema. Both
    providers re-send exactly this material every round; the exact wire bytes
    differ per provider (JSON structure, envelope wrappers) and are deliberately
    not claimed here.
    """
    schema_chars = 0
    for tool in tools:
        schema_chars += (
            len(tool.name)
            + len(tool.description)
            + len(json.dumps(dict(tool.input_schema), ensure_ascii=False))
        )
    message_chars = 0
    for message in messages:
        if message.text is not None:
            message_chars += len(message.text)
        if message.tool_calls is not None:
            message_chars += sum(len(call.arguments) for call in message.tool_calls)
        if message.tool_results is not None:
            message_chars += sum(len(result.content) for result in message.tool_results)
    return RoundInputMeasure(
        input_chars=len(system) + schema_chars + message_chars,
        tool_schemas_chars=schema_chars,
        tool_schemas_count=len(tools),
        messages_count=len(messages),
    )


def trace_jsonl_line(
    trace: ProviderRunTrace, *, provider_label: str, logged_at: str
) -> str:
    """One JSONL line for one agent request (kind=agent_request)."""
    payload = trace.to_dict()
    payload["kind"] = "agent_request"
    payload["provider"] = provider_label
    payload["logged_at_utc"] = logged_at
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_trace_jsonl(
    path: Path,
    trace: ProviderRunTrace,
    *,
    provider_label: str,
    logged_at: str | None = None,
) -> None:
    """Append one trace line to a JSONL file (diagnostic-only, never stdout)."""
    if logged_at is None:
        logged_at = datetime.now(timezone.utc).isoformat()
    line = trace_jsonl_line(trace, provider_label=provider_label, logged_at=logged_at)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
