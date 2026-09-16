"""P15-S4-M1: provider-agent cost instrumentation (loop-level, self-contained).

Every test drives :class:`music_agent.provider_agent.ProviderAgentLoop` with a
scripted provider and an :class:`~music_agent.agent_client.AgentClient`
subclass whose ``call`` returns canned results without touching the P09
service (the service exists only to satisfy the construction contract, as in
test_provider_agent.py's RecordingClient). No provider HTTP, no sockets -- so
assertions are deterministic except the ``time.monotonic`` latencies, which
are checked for structure (float, >= 0, ordering) rather than exact values.

Truthfulness rules under test: ``usage`` carries exactly what
``ProviderResponse.usage`` held (real DeepSeek / Codex shapes), missing usage
degrades to an empty mapping, all sizes are character counts of payloads this
process built (never tokens / never wire bytes), non-executed tool calls record
``duration_ms=None, executed=False``, and the aggregate only sums int usage
fields. Also: instrumentation must not change the loop's observable behavior
(message content, tool results, final text) and must stay off by default.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from uuid import uuid4

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.provider_agent import ProviderAgentLoop, ProviderLoopConfig
from music_agent.provider_contract import (
    ProviderError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
    ProviderToolCall,
    ProviderToolSchema,
)
from music_agent.provider_instrumentation import (
    ProviderRoundMeasure,
    ProviderRunTrace,
    ProviderToolMeasure,
    measure_round_input,
    safe_tool_arguments,
    trace_jsonl_line,
    write_trace_jsonl,
)
from music_agent.repository import CanonicalRepository

SYSTEM_PROMPT = "You are a test music agent."
SENTENCE = "play me a song"
CLIENT_ID = "agt_11111111-1111-4111-8111-111111111111"

TOOLS = (
    ProviderToolSchema(
        name="search_songs",
        description="search the song catalog",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    ProviderToolSchema(
        name="play_song",
        description="play one song",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    ),
)

# Real wire shapes, as the adapters normalize them into ProviderResponse.usage:
# DeepSeek = OpenAI-compatible scalar fields (detail objects dropped upstream);
# Codex = the numeric fields of the CLI's turn.completed event.
DEEPSEEK_ROUND_1 = {
    "prompt_tokens": 120,
    "completion_tokens": 30,
    "total_tokens": 150,
    "prompt_cache_hit_tokens": 64,
    "prompt_cache_miss_tokens": 56,
}
DEEPSEEK_ROUND_2 = {"prompt_tokens": 130, "completion_tokens": 10, "total_tokens": 140}
CODEX_ROUND_1 = {"input_tokens": 500, "output_tokens": 40}
CODEX_ROUND_2 = {"input_tokens": 510, "output_tokens": 12}


def fixture_model() -> dict:
    return json.loads(
        (Path(__file__).parent / "fixtures" / "canonical_music_model.json").read_text(
            encoding="utf-8"
        )
    )


class ScriptedProvider:
    """Answers ``chat`` from a script; records every call for byte-level checks."""

    def __init__(self, responses: list[ProviderResponse]):
        self.responses = list(responses)
        self.calls: list[tuple[str, list[ProviderMessage], tuple]] = []

    def chat(self, system, messages, tools):
        self.calls.append((system, list(messages), tuple(tools)))
        return self.responses.pop(0)


class FakeClient(AgentClient):
    """AgentClient whose ``call`` returns name-keyed canned results (no P09)."""

    def __init__(self, service: SharedAgentService, handlers: dict[str, object]):
        super().__init__(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
        )
        self.handlers = dict(handlers)
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool, payload, *, request_id=None, issued_at=None, completed_at=None):
        name = str(tool)
        self.calls.append((name, dict(payload)))
        handler = self.handlers.get(name)
        if handler is None:
            raise AssertionError(f"unexpected tool call: {name}")
        return handler(dict(payload))


def _tool_message(*calls: ProviderToolCall) -> ProviderMessage:
    return ProviderMessage(ProviderMessageRole.ASSISTANT, tool_calls=tuple(calls))


def _text_message(text: str) -> ProviderMessage:
    return ProviderMessage(ProviderMessageRole.ASSISTANT, text=text)


def _response(
    message: ProviderMessage,
    *,
    stop_reason: ProviderStopReason = ProviderStopReason.TOOL_USE,
    usage=None,
) -> ProviderResponse:
    return ProviderResponse(message=message, stop_reason=stop_reason, usage=usage or {})


def _tool_result(payload=None, *, outcome=AgentToolOutcome.OK, replayed=False):
    return AgentToolResult(
        request_id=f"req_{uuid4()}",
        tool="search_songs",
        outcome=outcome,
        payload=payload,
        error_code=None,
        error_message=None,
        completed_at=datetime.now(timezone.utc),
        replayed=replayed,
    )


def _search_call(call_id: str = "c1", arguments: str = '{"q": "rain"}') -> ProviderToolCall:
    return ProviderToolCall(call_id, "search_songs", arguments)


class InstrumentationTestCase(TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture_model())
        self.service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
        )
        self.addCleanup(self.service.close)

    def make_client(self, handlers: dict[str, object]) -> FakeClient:
        return FakeClient(self.service, handlers)

    def run_scripted(
        self,
        provider: ScriptedProvider,
        client: FakeClient,
        *,
        instrument: bool = True,
    ) -> tuple:
        loop = ProviderAgentLoop(
            provider,
            client,
            TOOLS,
            config=ProviderLoopConfig(
                system_prompt=SYSTEM_PROMPT,
                max_tool_rounds=4,
                instrument=instrument,
            ),
        )
        return loop.run(SENTENCE), loop


def _trace_of(result: object) -> ProviderRunTrace:
    trace = getattr(result, "trace", None)
    if trace is None:
        raise AssertionError("instrumented run carries no trace")
    return trace


class UsageCarryTests(InstrumentationTestCase):
    def test_deepseek_usage_preserved_round_by_round(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage=DEEPSEEK_ROUND_1),
                _response(
                    _text_message("Found it."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage=DEEPSEEK_ROUND_2,
                ),
            ]
        )
        client = self.make_client(
            {"search_songs": lambda args: _tool_result({"matches": [101]})}
        )
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(dict(trace.rounds[0].usage), DEEPSEEK_ROUND_1)
        self.assertEqual(dict(trace.rounds[1].usage), DEEPSEEK_ROUND_2)
        self.assertEqual(
            trace.usage_total,
            {
                "prompt_tokens": 250,
                "completion_tokens": 40,
                "total_tokens": 290,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 56,
            },
        )

    def test_codex_usage_preserved_round_by_round(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage=CODEX_ROUND_1),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage=CODEX_ROUND_2,
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(dict(trace.rounds[0].usage), CODEX_ROUND_1)
        self.assertEqual(dict(trace.rounds[1].usage), CODEX_ROUND_2)
        self.assertEqual(trace.usage_total, {"input_tokens": 1010, "output_tokens": 52})

    def test_missing_usage_fails_graceful(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage=None),
                _response(
                    _text_message("Nothing."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage=None,
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(dict(trace.rounds[0].usage), {})
        self.assertEqual(dict(trace.rounds[1].usage), {})
        self.assertEqual(trace.usage_total, {})
        self.assertEqual(trace.to_dict()["usage_total"], {})


class RoundMeasureTests(InstrumentationTestCase):
    def test_per_round_latency_and_shape(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage={"total_tokens": 7}),
                _response(
                    _text_message("Ok."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={"total_tokens": 3},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": [1]})})
        result, loop = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(len(trace.rounds), 2)
        for index, record in enumerate(trace.rounds, start=1):
            self.assertEqual(record.round_index, index)
            self.assertEqual(record.provider, "ScriptedProvider")
            self.assertIsInstance(record.provider_latency_ms, float)
            self.assertGreaterEqual(record.provider_latency_ms, 0.0)
            self.assertEqual(record.tool_schemas_count, len(TOOLS))
            self.assertGreater(record.input_chars, 0)
            self.assertGreaterEqual(record.messages_count, 1)
        self.assertEqual(trace.rounds[0].tool_calls_count, 1)
        self.assertEqual(trace.rounds[1].tool_calls_count, 0)
        # the loop really called the provider twice -- the latency was measured
        # around those calls, not invented
        self.assertEqual(len(loop.provider.calls), 2)
        self.assertEqual(
            trace.total_provider_ms, sum(r.provider_latency_ms for r in trace.rounds)
        )

    def test_measure_round_input_exact_math(self):
        schema_chars = sum(
            len(tool.name)
            + len(tool.description)
            + len(json.dumps(dict(tool.input_schema), ensure_ascii=False))
            for tool in TOOLS
        )
        messages = [
            ProviderMessage(ProviderMessageRole.USER, text=SENTENCE),
            _tool_message(_search_call()),
        ]
        measure = measure_round_input(SYSTEM_PROMPT, messages, TOOLS)
        expected = len(SYSTEM_PROMPT) + schema_chars + len(SENTENCE) + len(
            '{"q": "rain"}'
        )
        self.assertEqual(measure.input_chars, expected)
        self.assertEqual(measure.tool_schemas_chars, schema_chars)
        self.assertEqual(measure.tool_schemas_count, 2)
        self.assertEqual(measure.messages_count, 2)


class ToolMeasureTests(InstrumentationTestCase):
    def test_tool_duration_recorded_and_matches_execution_record(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage={"total_tokens": 5}),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={"total_tokens": 2},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": [1]})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        record = trace.tools[0]
        self.assertTrue(record.executed)
        self.assertIsNotNone(record.duration_ms)
        self.assertGreaterEqual(record.duration_ms, 0.0)
        # the same timer produced both numbers: |record - execution| is only the
        # one-decimal rounding (<= 0.05 ms)
        execution = result.tool_executions[0]
        self.assertLessEqual(abs(record.duration_ms - execution.elapsed_ms), 0.05)

    def test_raw_and_delivered_size_distinguished(self):
        payload = {"results": [{"title": f"song{i}", "artist": "someone"} for i in range(20)]}
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage={"total_tokens": 5}),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={"total_tokens": 2},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result(payload)})
        result, _ = self.run_scripted(provider, client)
        record = _trace_of(result).tools[0]
        raw_text = json.dumps(dict(payload), ensure_ascii=False)
        envelope_text = json.dumps(
            {
                "outcome": "ok",
                "error_code": None,
                "error_message": None,
                "payload": dict(payload),
                "replayed": False,
            },
            ensure_ascii=False,
        )
        self.assertEqual(record.raw_result_chars, len(raw_text))
        self.assertEqual(record.delivered_result_chars, len(envelope_text))
        self.assertGreater(record.delivered_result_chars, record.raw_result_chars)

    def test_truncated_flag_and_count(self):
        big = {"blob": "x" * 3000}
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(
                        _search_call("c1", '{"q": "big"}'),
                        ProviderToolCall("c2", "play_song", '{"id": "let-it-be"}'),
                    ),
                    usage={"total_tokens": 6},
                ),
                _response(
                    _text_message("Playing."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={"total_tokens": 3},
                ),
            ]
        )
        client = self.make_client(
            {
                "search_songs": lambda args: _tool_result(big),
                "play_song": lambda args: _tool_result({"queued": True}),
            }
        )
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        big_record = trace.tools[0]
        small_record = trace.tools[1]
        self.assertTrue(big_record.truncated)
        self.assertEqual(
            big_record.raw_result_chars, len(json.dumps(big, ensure_ascii=False))
        )
        self.assertLess(big_record.delivered_result_chars, big_record.raw_result_chars)
        self.assertFalse(small_record.truncated)
        self.assertEqual(trace.truncated_count, 1)
        # the truncated payload reached the provider as the preview marker
        tool_result_message = provider.calls[1][1][-1]
        first_fed_back = json.loads(tool_result_message.tool_results[0].content)
        self.assertTrue(first_fed_back["payload"]["truncated"])
        self.assertEqual(
            first_fed_back["payload"]["preview"],
            json.dumps(big, ensure_ascii=False)[:2000],
        )

    def test_cached_tool_call_recorded_not_executed(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call("c1", '{"q": "rain"}')), usage={}),
                _response(_tool_message(_search_call("c2", '{"q": "rain"}')), usage={}),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": [7]})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        executed = trace.tools[0]
        cached = trace.tools[1]
        self.assertTrue(executed.executed)
        self.assertIsNotNone(executed.duration_ms)
        self.assertEqual(executed.outcome, "ok")
        self.assertFalse(cached.executed)
        self.assertEqual(cached.outcome, "cached")
        self.assertIsNone(cached.duration_ms)
        self.assertEqual(cached.raw_result_chars, 0)
        self.assertEqual(cached.delivered_result_chars, executed.delivered_result_chars)
        self.assertEqual(len(client.calls), 1)  # P09 really ran once
        self.assertEqual(trace.executed_tool_calls_total, 1)
        self.assertEqual(trace.tool_calls_total, 2)

    def test_deduped_tool_call_in_same_round(self):
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(
                        _search_call("c1", '{"q": "rain"}'),
                        _search_call("c2", '{"q": "rain"}'),
                    ),
                    usage={},
                ),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": [1]})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        first, second = trace.tools
        self.assertEqual((first.call_index, second.call_index), (1, 2))
        self.assertTrue(first.executed)
        self.assertFalse(second.executed)
        self.assertEqual(second.outcome, "deduped")
        self.assertIsNone(second.duration_ms)
        self.assertEqual(second.delivered_result_chars, first.delivered_result_chars)
        self.assertEqual(len(client.calls), 1)

    def test_invalid_arguments_recorded_not_executed(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call("c1", "not json at all")), usage={}),
                _response(
                    _text_message("Please rephrase."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        record = trace.tools[0]
        self.assertFalse(record.executed)
        self.assertEqual(record.outcome, "invalid_arguments")
        self.assertIsNone(record.duration_ms)
        self.assertEqual(record.raw_result_chars, 0)
        self.assertGreater(record.delivered_result_chars, 0)
        self.assertEqual(len(client.calls), 0)
        # pre-existing behavior: the loop survives and answers the next round
        self.assertEqual(result.final_text, "Please rephrase.")


class ToolArgumentsTracingTests(InstrumentationTestCase):
    """P15-S4-M2 (diagnostic): tool arguments land in the trace -- structured
    when parseable (minimally redacted), an explicit unavailable marker when
    not, never the raw text, and never a serialization or agent-flow hazard."""

    def test_arguments_recorded_when_instrumented(self):
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(_search_call("c1", '{"q": "rain", "limit": 5}')),
                    usage={},
                ),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": [1]})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(trace.tools[0].arguments, {"q": "rain", "limit": 5})
        line = json.loads(
            trace_jsonl_line(
                trace, provider_label="deepseek", logged_at="2026-08-20T00:00:00+00:00"
            )
        )
        self.assertEqual(line["tools"][0]["arguments"], {"q": "rain", "limit": 5})

    def test_multi_tool_round_arguments_map_to_their_own_calls(self):
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(
                        _search_call("c1", '{"q": "a"}'),
                        ProviderToolCall("c2", "play_song", '{"id": "1"}'),
                    ),
                    usage={},
                ),
                _response(
                    _text_message("Both done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client(
            {
                "search_songs": lambda args: _tool_result({"matches": []}),
                "play_song": lambda args: _tool_result({"queued": True}),
            }
        )
        result, _ = self.run_scripted(provider, client)
        first, second = _trace_of(result).tools
        self.assertEqual((first.call_index, second.call_index), (1, 2))
        self.assertEqual(first.arguments, {"q": "a"})
        self.assertEqual(second.arguments, {"id": "1"})

    def test_default_record_without_arguments_stays_none(self):
        # Backward compatibility of the schema: records built without the new
        # field serialize an explicit null, never an invented value.
        record = ProviderToolMeasure(
            round_index=1,
            call_index=1,
            name="x",
            outcome="ok",
            error_code=None,
            duration_ms=1.0,
            raw_result_chars=0,
            delivered_result_chars=0,
            truncated=False,
            replayed=False,
            executed=True,
        )
        self.assertIsNone(record.arguments)
        self.assertIsNone(record.to_dict()["arguments"])

    def test_disabled_instrumentation_records_no_arguments(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call("c1", '{"q": "rain"}')), usage={}),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client, instrument=False)
        self.assertIsNone(result.trace)  # no trace, no arguments, anything recorded
        self.assertEqual(result.final_text, "Done.")
        self.assertEqual(len(client.calls), 1)  # execution path untouched

    def test_invalid_arguments_record_safe_marker(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call("c1", "not json at all")), usage={}),
                _response(
                    _text_message("Please rephrase."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(
            trace.tools[0].arguments,
            {"__unavailable__": "arguments are not valid JSON"},
        )
        line = trace_jsonl_line(
            trace, provider_label="deepseek", logged_at="2026-08-20T00:00:00+00:00"
        )
        self.assertNotIn("not json", line)  # the raw text never leaks into the trace
        json.loads(line)
        # non-object JSON gets the same explicit treatment
        self.assertEqual(
            safe_tool_arguments('["a", "b"]'),
            {"__unavailable__": "arguments are not a JSON object"},
        )
        # pathological nesting is still handled explicitly and safely: the parsed
        # non-object payload gets the non-object marker -- never the raw text,
        # never a raised error, never traversal into the deep structure
        self.assertEqual(
            safe_tool_arguments("[" * 1200 + "]" * 1200),
            {"__unavailable__": "arguments are not a JSON object"},
        )
        # an enormously nested object may parse but overflow the redaction
        # walk depending on the interpreter recursion limit -- the invariant
        # is: never raises, always an explicit marker
        deep_marker = safe_tool_arguments('{"a":' * 1500 + "1" + "}" * 1500)
        self.assertTrue(deep_marker.get("__unavailable__"))

    def test_non_finite_arguments_stay_strict_json(self):
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(_search_call("c1", '{"q": NaN, "t": -Infinity}')),
                    usage={},
                ),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        self.assertEqual(_trace_of(result).tools[0].arguments, {"q": "nan", "t": "-inf"})
        line = trace_jsonl_line(
            _trace_of(result), provider_label="deepseek", logged_at="2026-08-20T00:00:00+00:00"
        )
        parsed = json.loads(line)  # strict JSON: no bare NaN/inf tokens survive
        self.assertEqual(parsed["tools"][0]["arguments"], {"q": "nan", "t": "-inf"})

    def test_sensitive_arguments_are_redacted(self):
        redacted = safe_tool_arguments(
            json.dumps(
                {
                    "api_key": "sk-abc",
                    "Api_Key": "sk-case",
                    "data": {
                        "access_token": "tok",
                        "Authorization": "Bearer z",
                        "notes": [{"token": "t1"}, {"q": "ok"}],
                    },
                    "term": "夜晚",
                }
            )
        )
        self.assertEqual(
            redacted,
            {
                "api_key": "[REDACTED]",
                "Api_Key": "[REDACTED]",
                "data": {
                    "access_token": "[REDACTED]",
                    "Authorization": "[REDACTED]",
                    "notes": [{"token": "[REDACTED]"}, {"q": "ok"}],
                },
                "term": "夜晚",
            },
        )
        # loop level: a redacted trace line never contains the live value
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(_search_call("c1", '{"api_key": "sk-LIVE-7", "q": "rain"}')),
                    usage={},
                ),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client)
        line = trace_jsonl_line(
            _trace_of(result), provider_label="deepseek", logged_at="2026-08-20T00:00:00+00:00"
        )
        self.assertNotIn("sk-LIVE-7", line)
        self.assertIn("[REDACTED]", line)


class AggregateAndOutputTests(InstrumentationTestCase):
    def test_request_aggregate_fields(self):
        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage=DEEPSEEK_ROUND_1),
                _response(
                    _tool_message(ProviderToolCall("c2", "play_song", '{"id": "x"}')),
                    usage=CODEX_ROUND_1,
                ),
                _response(
                    _text_message("Enjoy."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={"total_tokens": 3},
                ),
            ]
        )
        client = self.make_client(
            {
                "search_songs": lambda args: _tool_result({"matches": [1]}),
                "play_song": lambda args: _tool_result({"queued": True}),
            }
        )
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(trace.total_rounds, 3)
        self.assertEqual(trace.tool_calls_total, 2)
        self.assertEqual(trace.executed_tool_calls_total, 2)
        self.assertEqual(trace.truncated_count, 0)
        self.assertEqual(
            trace.total_provider_ms, sum(r.provider_latency_ms for r in trace.rounds)
        )
        self.assertEqual(trace.total_tool_ms, sum(r.duration_ms for r in trace.tools))
        self.assertEqual(trace.total_ms, result.total_elapsed_ms)
        self.assertFalse(trace.rounds_capped)
        self.assertFalse(trace.context_trimmed)
        self.assertEqual(
            trace.usage_total,
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150 + 3,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 56,
                "input_tokens": 500,
                "output_tokens": 40,
            },
        )
        line = trace.to_dict()
        self.assertEqual(line["total_rounds"], 3)
        self.assertEqual(len(line["rounds"]), 3)
        self.assertEqual(len(line["tools"]), 2)
        json.dumps(line)  # the aggregate is JSON-serializable as emitted

    def test_usage_total_sums_only_integers(self):
        rounds = (
            ProviderRoundMeasure(
                round_index=1,
                provider="x",
                provider_latency_ms=1.0,
                input_chars=0,
                tool_schemas_chars=0,
                tool_schemas_count=1,
                messages_count=1,
                tool_calls_count=1,
                usage={"tokens": 5, "note": "n/a"},  # type: ignore[dict-item]
            ),
        )
        trace = ProviderRunTrace(
            total_ms=1.0, rounds_capped=False, context_trimmed=False, rounds=rounds, tools=()
        )
        self.assertEqual(trace.usage_total, {"tokens": 5})

    def test_multi_tool_round_aggregation_and_ordering(self):
        provider = ScriptedProvider(
            [
                _response(
                    _tool_message(
                        _search_call("c1", '{"q": "a"}'),
                        ProviderToolCall("c2", "play_song", '{"id": "1"}'),
                    ),
                    usage={},
                ),
                _response(
                    _text_message("Both done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client(
            {
                "search_songs": lambda args: _tool_result({"matches": []}),
                "play_song": lambda args: _tool_result({"queued": True}),
            }
        )
        result, _ = self.run_scripted(provider, client)
        trace = _trace_of(result)
        self.assertEqual(trace.rounds[0].tool_calls_count, 2)
        self.assertEqual(
            [
                (record.round_index, record.call_index, record.executed)
                for record in trace.tools
            ],
            [(1, 1, True), (1, 2, True)],
        )
        self.assertEqual(trace.tool_calls_total, 2)
        self.assertEqual(trace.executed_tool_calls_total, 2)
        self.assertEqual(trace.total_tool_ms, sum(r.duration_ms for r in trace.tools))


class NoBehaviorChangeTests(InstrumentationTestCase):
    def _scenario_pair(self):
        """One identical scenario run with instrumentation off and on."""

        def build():
            provider = ScriptedProvider(
                [
                    _response(_tool_message(_search_call()), usage=DEEPSEEK_ROUND_1),
                    _response(
                        _tool_message(ProviderToolCall("c2", "play_song", '{"id": "x"}')),
                        usage=CODEX_ROUND_1,
                    ),
                    _response(
                        _text_message("Enjoy the song."),
                        stop_reason=ProviderStopReason.END_TURN,
                        usage={"total_tokens": 3},
                    ),
                ]
            )
            client = self.make_client(
                {
                    "search_songs": lambda args: _tool_result({"matches": [1]}),
                    "play_song": lambda args: _tool_result({"queued": True}),
                }
            )
            return provider, client

        provider_off, client_off = build()
        result_off, _ = self.run_scripted(provider_off, client_off, instrument=False)
        provider_on, client_on = build()
        result_on, _ = self.run_scripted(provider_on, client_on, instrument=True)
        return (result_off, provider_off), (result_on, provider_on)

    def test_instrumentation_does_not_change_observable_behavior(self):
        (result_off, provider_off), (result_on, provider_on) = self._scenario_pair()

        self.assertEqual(result_off.final_text, result_on.final_text)
        self.assertEqual(result_off.rounds, result_on.rounds)
        self.assertEqual(result_off.context_trimmed, result_on.context_trimmed)
        self.assertEqual(result_off.rounds_capped, result_on.rounds_capped)
        self.assertEqual(
            [(e.name, e.outcome, e.error_code) for e in result_off.tool_executions],
            [(e.name, e.outcome, e.error_code) for e in result_on.tool_executions],
        )
        self.assertIsNone(result_off.trace)
        self.assertIsNotNone(result_on.trace)
        # identical tool-result bytes reached the provider in each round
        for (_, messages_off, _), (_, messages_on, _) in zip(
            provider_off.calls, provider_on.calls
        ):
            self.assertEqual(
                [
                    tuple(r.content for r in m.tool_results)
                    for m in messages_off
                    if m.tool_results is not None
                ],
                [
                    tuple(r.content for r in m.tool_results)
                    for m in messages_on
                    if m.tool_results is not None
                ],
            )

    def test_disabled_by_default(self):
        config = ProviderLoopConfig(system_prompt=SYSTEM_PROMPT)
        self.assertFalse(config.instrument)
        with self.assertRaises(ProviderError):
            ProviderLoopConfig(system_prompt=SYSTEM_PROMPT, instrument="yes")

        provider = ScriptedProvider(
            [
                _response(_tool_message(_search_call()), usage={}),
                _response(
                    _text_message("Done."),
                    stop_reason=ProviderStopReason.END_TURN,
                    usage={},
                ),
            ]
        )
        client = self.make_client({"search_songs": lambda args: _tool_result({"matches": []})})
        result, _ = self.run_scripted(provider, client, instrument=False)
        self.assertIsNone(result.trace)
        self.assertEqual(result.final_text, "Done.")
        # default config behaves exactly as before instrumentation existed
        loop = ProviderAgentLoop(provider, client, TOOLS)
        self.assertFalse(loop.config.instrument)


class JsonlOutputTests(TestCase):
    def test_trace_jsonl_line_shape(self):
        trace = ProviderRunTrace(
            total_ms=12.3,
            rounds_capped=False,
            context_trimmed=True,
            rounds=(
                ProviderRoundMeasure(
                    round_index=1,
                    provider="deepseek",
                    provider_latency_ms=4.5,
                    input_chars=100,
                    tool_schemas_chars=40,
                    tool_schemas_count=2,
                    messages_count=2,
                    tool_calls_count=1,
                    usage={"prompt_tokens": 5},
                ),
            ),
            tools=(),
        )
        line = trace_jsonl_line(
            trace, provider_label="deepseek", logged_at="2026-08-20T00:00:00+00:00"
        )
        parsed = json.loads(line)
        self.assertEqual(parsed["kind"], "agent_request")
        self.assertEqual(parsed["provider"], "deepseek")
        self.assertEqual(parsed["logged_at_utc"], "2026-08-20T00:00:00+00:00")
        self.assertEqual(parsed["total_rounds"], 1)
        self.assertEqual(parsed["context_trimmed"], True)
        self.assertEqual(parsed["rounds"][0]["usage"], {"prompt_tokens": 5})
        self.assertFalse(line.endswith("\n"))  # the writer owns the newline

    def test_write_trace_jsonl_appends_machine_analyzable_lines(self):
        def trace_of(total: float) -> ProviderRunTrace:
            return ProviderRunTrace(
                total_ms=total,
                rounds_capped=False,
                context_trimmed=False,
                rounds=(),
                tools=(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            write_trace_jsonl(path, trace_of(1.0), provider_label="codex")
            write_trace_jsonl(path, trace_of(2.0), provider_label="codex")
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            first, second = (json.loads(line) for line in lines)
            self.assertEqual(first["total_ms"], 1.0)
            self.assertEqual(second["total_ms"], 2.0)
            self.assertTrue(first["logged_at_utc"])
            self.assertEqual(first["kind"], "agent_request")


class CliTraceWriteTests(TestCase):
    """The cli._write_trace glue: trace file written when --trace asked,
    silent skip when no trace exists, stderr warning (never a crash) on a
    failed write."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.trace = ProviderRunTrace(
            total_ms=7.5,
            rounds_capped=False,
            context_trimmed=False,
            rounds=(),
            tools=(),
        )

    def _args(self, *, trace: object = "trace-file", provider: str = "deepseek") -> SimpleNamespace:
        namespace = SimpleNamespace(provider=provider)
        if trace is not None:
            namespace.trace = trace
        return namespace

    def test_trace_option_writes_one_jsonl_line(self):
        from music_agent.cli import _write_trace

        path = Path(self.temporary_directory.name) / "trace.jsonl"
        _write_trace(SimpleNamespace(trace=self.trace), self._args(trace=str(path)))
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["kind"], "agent_request")
        self.assertEqual(parsed["provider"], "deepseek")
        self.assertEqual(parsed["total_ms"], 7.5)

    def test_result_without_trace_is_silently_skipped(self):
        from music_agent.cli import _write_trace

        path = Path(self.temporary_directory.name) / "trace.jsonl"
        _write_trace(SimpleNamespace(), self._args(trace=str(path)))  # no .trace attr
        self.assertFalse(path.exists())

    def test_no_trace_option_is_silently_skipped(self):
        from music_agent.cli import _write_trace

        path = Path(self.temporary_directory.name) / "trace.jsonl"
        _write_trace(SimpleNamespace(trace=self.trace), self._args(trace=None))
        self.assertFalse(path.exists())

    def test_failed_write_warns_on_stderr_and_survives(self):
        import io
        import sys
        from contextlib import redirect_stderr

        from music_agent.cli import _write_trace

        # a path whose parent is a file: open() must fail with OSError
        blocker = Path(self.temporary_directory.name) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        target = blocker / "trace.jsonl"
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            _write_trace(SimpleNamespace(trace=self.trace), self._args(trace=str(target)))
        output = buffer.getvalue()
        self.assertIn("failed to write trace", output)
        self.assertIn(str(target), output)


if __name__ == "__main__":
    import unittest

    unittest.main()