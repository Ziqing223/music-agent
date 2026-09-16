"""P10.8: DeepSeek (OpenAI-compatible) and Codex CLI provider tests.

Deterministic fakes only: no real DeepSeek quota and no real Codex CLI invocation
anywhere in this suite (the live-validation gate is user-approved and separate).
"""

import json
import unittest
from unittest.mock import patch

from music_agent.codex_provider import (
    CodexCliConfig,
    CodexCliProvider,
    CodexCliError,
    SubprocessCodexRunner,
)
from music_agent.deepseek_provider import (
    DEFAULT_DEEPSEEK_API_KEY_ENV,
    DeepSeekApiProvider,
)
from music_agent.prompts.workflow import _S4_RECOMMENDATION_PROMPT
from music_agent.provider_contract import (
    ProviderAuthError,
    ProviderConfig,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderStopReason,
    ProviderTimeoutError,
    ProviderToolCall,
    ProviderToolSchema,
    ProviderUnavailableError,
)


class FakeTransport:
    def __init__(self, response: tuple[int, str] | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post_json(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": dict(body), "timeout": timeout})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def deepseek_provider(transport=None, **config) -> DeepSeekApiProvider:
    defaults = dict(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key="test-key",
    )
    defaults.update(config)
    return DeepSeekApiProvider(ProviderConfig(**defaults), transport=transport)


def tools() -> tuple[ProviderToolSchema, ...]:
    return (
        ProviderToolSchema(
            "get_canonical_entity", "读取实体",
            {"type": "object", "properties": {"canonical_id": {"type": "string"}}},
        ),
    )


class DeepSeekRequestShapeTest(unittest.TestCase):
    def test_request_shape_and_bearer_header(self) -> None:
        fake = FakeTransport((200, json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })))
        deepseek_provider(transport=fake).chat(
            "system",
            [ProviderMessage(ProviderMessageRole.USER, text="hello")],
            tools(),
        )
        call = fake.calls[0]
        self.assertEqual(call["url"], "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["body"]["model"], "deepseek-chat")
        self.assertEqual(call["body"]["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(call["body"]["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(call["body"]["tools"][0]["type"], "function")
        self.assertEqual(call["body"]["tools"][0]["function"]["name"], "get_canonical_entity")
        self.assertEqual(call["body"]["tool_choice"], "auto")

    def test_zero_tools_omits_tools_and_tool_choice_keys(self) -> None:
        # S1: a plain-chat turn travels with NO tool schema at all -- an empty
        # tools array with tool_choice "auto" is a rejected wire shape on
        # OpenAI-compatible APIs, so both keys are omitted entirely.
        fake = FakeTransport((200, json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })))
        deepseek_provider(transport=fake).chat(
            "system",
            [ProviderMessage(ProviderMessageRole.USER, text="你好")],
            (),
        )
        call = fake.calls[0]
        self.assertNotIn("tools", call["body"])
        self.assertNotIn("tool_choice", call["body"])
        self.assertEqual(call["body"]["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(call["body"]["messages"][1], {"role": "user", "content": "你好"})

    def test_tool_call_and_result_encoding(self) -> None:
        from music_agent.provider_contract import ProviderToolResult

        fake = FakeTransport((200, json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })))
        deepseek_provider(transport=fake).chat(
            "s",
            [
                ProviderMessage(
                    ProviderMessageRole.ASSISTANT,
                    tool_calls=(ProviderToolCall("c1", "get_canonical_entity", '{"canonical_id": "x"}'),),
                ),
                ProviderMessage(
                    ProviderMessageRole.USER,
                    tool_results=(ProviderToolResult("c1", "r1"), ProviderToolResult("c2", "r2")),
                ),
            ],
            tools(),
        )
        messages = fake.calls[0]["body"]["messages"]
        self.assertEqual(
            messages[1],
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_canonical_entity", "arguments": '{"canonical_id": "x"}'}}]},
        )
        # Two tool results flatten into two wire messages.
        self.assertEqual(messages[2], {"role": "tool", "tool_call_id": "c1", "content": "r1"})
        self.assertEqual(messages[3], {"role": "tool", "tool_call_id": "c2", "content": "r2"})


class DeepSeekResponseParsingTest(unittest.TestCase):
    def test_text_response(self) -> None:
        fake = FakeTransport((200, json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        })))
        response = deepseek_provider(transport=fake).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.message.text, "你好")
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)
        self.assertEqual(dict(response.usage), {"prompt_tokens": 3, "completion_tokens": 4})

    def test_tool_calls_response(self) -> None:
        fake = FakeTransport((200, json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c9", "type": "function",
                                    "function": {"name": "get_canonical_entity",
                                                 "arguments": '{"canonical_id": "t"}'}}],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })))
        response = deepseek_provider(transport=fake).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.TOOL_USE)
        self.assertEqual(response.message.tool_calls[0].name, "get_canonical_entity")
        self.assertEqual(response.message.tool_calls[0].arguments, '{"canonical_id": "t"}')

    def test_error_mapping(self) -> None:
        for status, error_type in (
            (401, ProviderAuthError),
            (403, ProviderAuthError),
            (429, ProviderRateLimitError),
            (500, ProviderUnavailableError),
            (400, ProviderUnavailableError),
        ):
            with self.assertRaises(error_type):
                deepseek_provider(transport=FakeTransport((status, "{}"))).chat(
                    "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
                )
        with self.assertRaises(ProviderTimeoutError):
            deepseek_provider(transport=FakeTransport(TimeoutError("slow"))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, "not-json"))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_missing_credential_names_the_variable(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
            provider = deepseek_provider(api_key=None, api_key_env=DEFAULT_DEEPSEEK_API_KEY_ENV)
            with self.assertRaises(ProviderAuthError) as raised:
                provider.chat("s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools())
            self.assertIn("DEEPSEEK_API_KEY", str(raised.exception))


MIXED_DEEPSEEK_RESPONSE = json.dumps({
    "choices": [{
        "message": {
            "role": "assistant",
            # Thinking-mode tool-call round: reasoning_content + preamble + tool_calls.
            "reasoning_content": "需要先确认能力边界。",
            "content": "好的，我来确认能力边界。",
            "tool_calls": [{
                "id": "call_ds_001",
                "type": "function",
                "function": {"name": "get_agent_capabilities", "arguments": "{}"},
            }],
        },
        "finish_reason": "tool_calls",
    }],
    # Real OpenAI-compatible usage: scalar token counts PLUS nested detail objects.
    "usage": {
        "prompt_tokens": 500,
        "completion_tokens": 40,
        "total_tokens": 540,
        "prompt_tokens_details": {"cached_tokens": 120},
        "completion_tokens_details": {"reasoning_tokens": 10},
    },
})


class DeepSeekMixedResponseTest(unittest.TestCase):
    def test_regression_mixed_content_and_tool_calls_parsed(self) -> None:
        """The exact live failure shape: reasoning + preamble + tool_calls + nested usage."""
        response = deepseek_provider(transport=FakeTransport((200, MIXED_DEEPSEEK_RESPONSE))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="确认能力")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.TOOL_USE)
        self.assertEqual(response.message.text, "好的，我来确认能力边界。")
        self.assertEqual(len(response.message.tool_calls), 1)
        self.assertEqual(response.message.tool_calls[0].name, "get_agent_capabilities")
        self.assertEqual(json.loads(response.message.tool_calls[0].arguments), {})
        # Thinking-mode wire metadata is preserved adapter-side for the continuation.
        self.assertEqual(
            dict(response.message.wire_metadata or {}),
            {"reasoning_content": "需要先确认能力边界。"},
        )

    def test_reasoning_metadata_echoed_on_tool_call_continuation(self) -> None:
        """The assistant tool-call message goes back with reasoning_content + matching id."""
        from music_agent.provider_contract import ProviderToolResult

        provider = deepseek_provider(transport=FakeTransport((200, MIXED_DEEPSEEK_RESPONSE)))
        response = provider.chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="确认能力")], tools()
        )
        final_body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "最终回答"},
                         "finish_reason": "stop"}],
            "usage": {},
        })
        provider = deepseek_provider(transport=FakeTransport((200, final_body)))
        provider.chat(
            "s",
            [
                ProviderMessage(ProviderMessageRole.USER, text="确认能力"),
                response.message,
                ProviderMessage(
                    ProviderMessageRole.USER,
                    tool_results=(ProviderToolResult("call_ds_001", "result"),),
                ),
            ],
            tools(),
        )
        wire_messages = provider._transport.calls[0]["body"]["messages"]
        assistant_wire = next(m for m in wire_messages if m["role"] == "assistant")
        self.assertEqual(assistant_wire["reasoning_content"], "需要先确认能力边界。")
        self.assertEqual(assistant_wire["content"], "好的，我来确认能力边界。")
        self.assertEqual(assistant_wire["tool_calls"][0]["id"], "call_ds_001")
        tool_wire = next(m for m in wire_messages if m["role"] == "tool")
        self.assertEqual(tool_wire["tool_call_id"], "call_ds_001")  # matching id

    def test_absent_reasoning_omitted_from_continuation(self) -> None:
        """Non-thinking models: no reasoning_content key on the continuation message."""
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        del payload["choices"][0]["message"]["reasoning_content"]
        payload["choices"][0]["message"]["content"] = None
        provider = deepseek_provider(transport=FakeTransport((200, json.dumps(payload))))
        response = provider.chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertIsNone(response.message.wire_metadata)
        provider = deepseek_provider(transport=FakeTransport((200, json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }))))
        provider.chat(
            "s",
            [
                ProviderMessage(ProviderMessageRole.USER, text="x"),
                response.message,
            ],
            tools(),
        )
        assistant_wire = next(
            m for m in provider._transport.calls[0]["body"]["messages"] if m["role"] == "assistant"
        )
        self.assertNotIn("reasoning_content", assistant_wire)
        self.assertEqual(assistant_wire["content"], "")  # documented empty-string form

    def test_non_string_reasoning_content_fails_closed(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["choices"][0]["message"]["reasoning_content"] = {"chunks": 3}
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_http_400_error_body_mapped_to_sanitized_message(self) -> None:
        body = json.dumps({
            "error": {
                "code": "invalid_request_error",
                "message": "The input (number of tokens) is too long",
                "type": "invalid_request_error",
            }
        })
        with self.assertRaises(ProviderUnavailableError) as raised:
            deepseek_provider(transport=FakeTransport((400, body))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )
        message = str(raised.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("invalid_request_error", message)
        self.assertIn("too long", message)
        # Sanitization: the credential and any request content never appear.
        self.assertNotIn("test-key", message)
        self.assertNotIn("Bearer", message)

    def test_http_400_without_error_envelope_stays_generic(self) -> None:
        with self.assertRaises(ProviderUnavailableError) as raised:
            deepseek_provider(transport=FakeTransport((400, "plain"))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )
        self.assertEqual(str(raised.exception), "provider returned HTTP 400")

    def test_usage_normalization_with_nested_detail_objects(self) -> None:
        """Nested usage detail objects are ignored; scalar token counts survive."""
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["usage"]["completion_tokens_details"]["audio_tokens"] = 3
        payload["usage"]["custom_metadata"] = {"anything": [1, 2]}
        response = deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(dict(response.usage), {
            "prompt_tokens": 500, "completion_tokens": 40, "total_tokens": 540,
        })

    def test_scalar_usage_fields_including_strings(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["usage"] = {"prompt_tokens": "500", "completion_tokens": 40.0}
        response = deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(dict(response.usage), {"prompt_tokens": 500, "completion_tokens": 40})

    def test_absent_usage_is_empty(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        del payload["usage"]
        response = deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(dict(response.usage), {})

    def test_malformed_required_usage_field_fails_closed(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["usage"]["prompt_tokens"] = {"nested": 1}  # core field as object: violation
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["usage"]["completion_tokens"] = "not-a-number"
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_mixed_response_with_multiple_tool_calls(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["choices"][0]["message"]["tool_calls"].append({
            "id": "call_ds_002", "type": "function",
            "function": {"name": "list_recommendation_runs", "arguments": "{}"},
        })
        response = deepseek_provider(
            transport=FakeTransport((200, json.dumps(payload)))
        ).chat("s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools())
        self.assertEqual([c.name for c in response.message.tool_calls],
                         ["get_agent_capabilities", "list_recommendation_runs"])

    def test_mixed_response_with_malformed_tool_call_fails_closed(self) -> None:
        payload = json.loads(MIXED_DEEPSEEK_RESPONSE)
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = None
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, json.dumps(payload)))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_content_only_and_tool_call_only_unchanged(self) -> None:
        content_only = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "只有文字"},
                         "finish_reason": "stop"}],
            "usage": {},
        })
        response = deepseek_provider(transport=FakeTransport((200, content_only))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.message.text, "只有文字")
        self.assertIsNone(response.message.tool_calls)

        tool_only = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "c", "type": "function",
                                                     "function": {"name": "get_agent_capabilities",
                                                                  "arguments": "{}"}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {},
        })
        response = deepseek_provider(transport=FakeTransport((200, tool_only))).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertIsNone(response.message.text)
        self.assertEqual(len(response.message.tool_calls), 1)

    def test_empty_message_fails_closed(self) -> None:
        empty = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {},
        })
        with self.assertRaises(ProviderInvalidResponseError):
            deepseek_provider(transport=FakeTransport((200, empty))).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_full_multi_round_flow_through_p09(self) -> None:
        """DeepSeek mixed response -> generic normalization -> P09 executes
        get_agent_capabilities -> tool result returns -> final natural-language answer."""
        import tempfile
        from pathlib import Path

        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentClientIdentity
        from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
        from music_agent.agent_service import SharedAgentService
        from music_agent.provider_agent import ProviderAgentLoop, ProviderLoopConfig
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS
        from music_agent.repository import CanonicalRepository

        final_response = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "能力边界已确认，我是你的音乐推荐助手。"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 60},
        })
        interpreter_unsupported = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": json.dumps({
                "intent": "unsupported",
                "recommendation": None,
                "action": None,
                "requires_clarification": False,
                "reason": "outside_p22_s1_scope",
            }, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 30},
        })

        class ScriptedTransport(FakeTransport):
            def __init__(self) -> None:
                super().__init__((200, final_response))
                self.responses = [
                    (200, interpreter_unsupported),
                    (200, MIXED_DEEPSEEK_RESPONSE),
                    (200, final_response),
                ]

            def post_json(self, url, headers, body, timeout):
                self.calls.append({"url": url, "headers": dict(headers), "body": dict(body), "timeout": timeout})
                return self.responses.pop(0)

        provider = DeepSeekApiProvider(
            ProviderConfig(base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                           api_key="test-key"),
            transport=ScriptedTransport(),
        )
        client_id = "agt_77777777-7777-4777-8777-777777777777"
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            with SharedAgentService(
                database_path, clients=AgentClientRegistry({client_id: AgentClientPolicy.FULL})
            ) as service:
                client = AgentClient(
                    AgentClientIdentity(client_id=client_id, model_id="deepseek-chat", label="tests"),
                    service,
                )
                loop = ProviderAgentLoop(
                    provider, client, PROVIDER_TOOL_SCHEMAS,
                    config=ProviderLoopConfig(max_tool_rounds=4),
                )
                result = loop.run("确认能力边界")
            self.assertEqual(result.final_text, "能力边界已确认，我是你的音乐推荐助手。")
            self.assertFalse(result.rounds_capped)
            self.assertEqual(
                [execution.name for execution in result.tool_executions],
                ["get_agent_capabilities"],
            )
            self.assertEqual(result.tool_executions[0].outcome, "ok")
            # The second round carried the preamble, the echoed reasoning metadata,
            # and the tool result into the request.
            second_body = provider._transport.calls[2]["body"]
            assistant_wire = next(
                m for m in second_body["messages"] if m["role"] == "assistant"
            )
            self.assertEqual(assistant_wire["content"], "好的，我来确认能力边界。")
            self.assertEqual(assistant_wire["reasoning_content"], "需要先确认能力边界。")
            self.assertEqual(assistant_wire["tool_calls"][0]["id"], "call_ds_001")
            tool_result_wire = next(
                m for m in second_body["messages"] if m["role"] == "tool"
            )
            self.assertEqual(tool_result_wire["tool_call_id"], "call_ds_001")
            self.assertIn("get_agent_capabilities", tool_result_wire["content"])
            # Reasoning metadata is wire protocol only: nothing persisted to the
            # canonical store references it.
            from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

            with AgentRequestJournalRepository(database_path) as journal:
                records = journal.list()
                # P22 adds one read-only get_active_context call to build the
                # interpreter's high-level context before the legacy FULL path.
                self.assertEqual(len(records), 2)
                self.assertEqual(
                    [record.request.tool for record in records],
                    ["get_active_context", "get_agent_capabilities"],
                )
                for record in records:
                    # The canonical journal sees only clean code-owned read payloads:
                    # no reasoning metadata is persisted anywhere in Music Agent state.
                    self.assertEqual(dict(record.request.payload), {})
                    self.assertNotIn("reasoning", json.dumps(dict(record.request.payload)))
                    self.assertNotIn("确认能力边界", json.dumps(dict(record.request.payload)))


class FakeCodexRunner:
    def __init__(self, responses: list[tuple[int, str, str] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def run(self, command, prompt, timeout):
        self.calls.append({"command": list(command), "prompt": prompt, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeCodexRunner exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def codex_provider(runner=None, **config) -> CodexCliProvider:
    return CodexCliProvider(CodexCliConfig(**config), runner=runner)


REAL_EVENT_ENVELOPE = (
    '{"type":"thread.started","thread_id":"01a008d4-4b9d-7200-8d7f-f55d8b8eb83d"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
    '"text":"{\\"tool_calls\\":[{\\"id\\":\\"call_get_agent_capabilities_001\\",'
    '\\"name\\":\\"get_agent_capabilities\\",\\"arguments\\":{}}]}"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":19863,"cached_input_tokens":11008,'
    '"cache_write_input_tokens":0,"output_tokens":63,"reasoning_output_tokens":30}}'
)


def real_event_text(text: str) -> str:
    """The real codex exec --json event stream carrying one agent_message text."""
    return (
        '{"type":"thread.started","thread_id":"t1"}\n'
        '{"type":"turn.started"}\n'
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":'
        + json.dumps(text, ensure_ascii=False)
        + "}}\n"
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}'
    )


class CodexProviderTest(unittest.TestCase):
    def test_plain_text_answer_legacy_wrapper(self) -> None:
        runner = FakeCodexRunner([(0, json.dumps({"result": "这是最终回答。"}), "")])
        provider = codex_provider(runner=runner)
        response = provider.chat("s", [ProviderMessage(ProviderMessageRole.USER, text="你好")], tools())
        self.assertEqual(response.message.text, "这是最终回答。")
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)
        self.assertEqual(runner.calls[0]["command"], ["codex", "exec", "--json"])

    def test_zero_tools_renders_no_tool_block_and_asks_for_a_direct_answer(self) -> None:
        # S1: the codex prompt of a zero-tool turn must not dangle a tool
        # instruction with nothing to call -- it asks for the answer directly.
        runner = FakeCodexRunner([(0, json.dumps({"result": "好的。"}), "")])
        provider = codex_provider(runner=runner)
        provider.chat("s", [ProviderMessage(ProviderMessageRole.USER, text="你好")], ())
        prompt = runner.calls[0]["prompt"]
        self.assertNotIn("你可以使用以下工具", prompt)
        self.assertNotIn("当需要调用工具时", prompt)
        self.assertIn("本轮不提供任何工具：请直接输出给用户的最终回答。", prompt)

    def test_regression_real_event_stream_envelope_parsed(self) -> None:
        """The exact live event shape from the failed validation MUST yield a tool call."""
        runner = FakeCodexRunner([(0, REAL_EVENT_ENVELOPE, "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="确认能力")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.TOOL_USE)
        self.assertIsNone(response.message.text)
        self.assertEqual(len(response.message.tool_calls), 1)
        call = response.message.tool_calls[0]
        self.assertEqual(call.name, "get_agent_capabilities")
        self.assertEqual(call.call_id, "call_get_agent_capabilities_001")
        self.assertEqual(json.loads(call.arguments), {})
        self.assertEqual(dict(response.usage)["input_tokens"], 19863)
        self.assertEqual(dict(response.usage)["output_tokens"], 63)

    def test_real_event_stream_plain_text_answer(self) -> None:
        runner = FakeCodexRunner([(0, real_event_text("这是最终回答。"), "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="你好")], tools()
        )
        self.assertEqual(response.message.text, "这是最终回答。")
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)
        self.assertEqual(dict(response.usage)["input_tokens"], 10)

    def test_real_event_stream_garbled_envelope_fails_soft(self) -> None:
        garbled = '{"tool_calls": [{"name": "x"}]}'  # missing id/arguments
        runner = FakeCodexRunner([(0, real_event_text(garbled), "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)
        self.assertIsNotNone(response.message.text)

    def test_native_codex_function_call_items_are_ignored(self) -> None:
        """Codex's own tool items never reach the Music Agent loop (no execution here)."""
        stream = (
            '{"type":"item.completed","item":{"id":"i1","type":"function_call",'
            '"name":"shell","arguments":"{\\"command\\":\\"rm -rf /\\"}"}}\n'
            + real_event_text("我没有使用工具。")
        )
        runner = FakeCodexRunner([(0, stream, "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.message.text, "我没有使用工具。")
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)

    def test_multi_round_tool_result_continuation_through_p09(self) -> None:
        """Round 1: envelope (real event shape) -> P09 executes -> round 2 prompt carries
        the tool result text -> final answer. End-to-end through the provider loop."""
        import tempfile
        from pathlib import Path

        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentClientIdentity
        from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
        from music_agent.agent_service import SharedAgentService
        from music_agent.provider_agent import ProviderAgentLoop, ProviderLoopConfig
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS
        from music_agent.repository import CanonicalRepository

        interpreter_unsupported = json.dumps({
            "intent": "unsupported",
            "recommendation": None,
            "action": None,
            "requires_clarification": False,
            "reason": "outside_p22_s1_scope",
        }, ensure_ascii=False)
        runner = FakeCodexRunner([
            (0, real_event_text(interpreter_unsupported), ""),
            (0, REAL_EVENT_ENVELOPE, ""),
            (0, real_event_text("能力边界已确认，我是你的音乐推荐助手。"), ""),
        ])
        provider = CodexCliProvider(CodexCliConfig(), runner=runner)
        client_id = "agt_66666666-6666-4666-8666-666666666666"
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {
                        "tracks": [],
                        "artists": [],
                        "albums": [],
                        "playlists": [],
                        "playlist_memberships": [],
                    }
                )
            with SharedAgentService(
                database_path, clients=AgentClientRegistry({client_id: AgentClientPolicy.FULL})
            ) as service:
                client = AgentClient(
                    AgentClientIdentity(client_id=client_id, model_id="codex", label="tests"),
                    service,
                )
                loop = ProviderAgentLoop(
                    provider, client, PROVIDER_TOOL_SCHEMAS,
                    config=ProviderLoopConfig(max_tool_rounds=4),
                )
                result = loop.run("确认能力边界")
            self.assertEqual(result.final_text, "能力边界已确认，我是你的音乐推荐助手。")
            self.assertFalse(result.rounds_capped)
            self.assertEqual(
                [execution.name for execution in result.tool_executions],
                ["get_agent_capabilities"],
            )
            self.assertEqual(result.tool_executions[0].outcome, "ok")
            # The continuation prompt carried the tool result text back to Codex.
            second_prompt = runner.calls[2]["prompt"]
            self.assertIn("[工具结果", second_prompt)
            self.assertIn("get_agent_capabilities", second_prompt)

    def test_tool_call_envelope_parsed(self) -> None:
        envelope = {"tool_calls": [{"id": "c1", "name": "get_canonical_entity",
                                    "arguments": {"canonical_id": "trk_1"}}]}
        runner = FakeCodexRunner([(0, json.dumps(envelope), "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="查一下")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.TOOL_USE)
        self.assertEqual(response.message.tool_calls[0].name, "get_canonical_entity")
        self.assertEqual(json.loads(response.message.tool_calls[0].arguments), {"canonical_id": "trk_1"})

    def test_garbled_envelope_fails_soft_to_text(self) -> None:
        garbled = {"tool_calls": [{"name": "x"}]}  # missing id/arguments
        runner = FakeCodexRunner([(0, json.dumps(garbled), "")])
        response = codex_provider(runner=runner).chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(response.stop_reason, ProviderStopReason.END_TURN)
        self.assertIsNotNone(response.message.text)

    def test_prompt_contains_system_tools_and_conversation(self) -> None:
        runner = FakeCodexRunner([(0, "ok", "")])
        codex_provider(runner=runner).chat(
            "系统提示", [ProviderMessage(ProviderMessageRole.USER, text="用户问题")], tools()
        )
        prompt = runner.calls[0]["prompt"]
        self.assertIn("系统提示", prompt)
        self.assertIn("get_canonical_entity", prompt)
        self.assertIn("用户问题", prompt)
        self.assertIn("tool_calls", prompt)
        # The model flag appends to the command when configured.
        self.assertEqual(runner.calls[0]["command"], ["codex", "exec", "--json"])

    def test_model_flag_appended(self) -> None:
        runner = FakeCodexRunner([(0, "ok", "")])
        codex_provider(runner=runner, model="gpt-5-codex").chat(
            "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
        )
        self.assertEqual(runner.calls[0]["command"], ["codex", "exec", "--json", "--model", "gpt-5-codex"])

    def test_auth_failure_maps_typed(self) -> None:
        runner = FakeCodexRunner([(1, "", "Error: not logged in, run codex login")])
        with self.assertRaises(ProviderAuthError):
            codex_provider(runner=runner).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_other_failure_maps_unavailable(self) -> None:
        runner = FakeCodexRunner([(2, "", "crash")])
        with self.assertRaises(ProviderUnavailableError):
            codex_provider(runner=runner).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_timeout_maps_typed(self) -> None:
        runner = FakeCodexRunner([TimeoutError("slow")])
        with self.assertRaises(ProviderTimeoutError):
            codex_provider(runner=runner).chat(
                "s", [ProviderMessage(ProviderMessageRole.USER, text="x")], tools()
            )

    def test_config_validation(self) -> None:
        with self.assertRaises(CodexCliError):
            CodexCliConfig(command=())
        with self.assertRaises(CodexCliError):
            CodexCliConfig(model="")
        with self.assertRaises(CodexCliError):
            CodexCliConfig(timeout_seconds=0)


class PromptTransportParityTest(unittest.TestCase):
    def test_deepseek_and_codex_preserve_same_upper_layer_prompt_and_tool_semantics(self) -> None:
        selected_tools = tools()
        deepseek_transport = FakeTransport((200, json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {},
        })))
        deepseek_provider(transport=deepseek_transport).chat(
            _S4_RECOMMENDATION_PROMPT,
            [ProviderMessage(ProviderMessageRole.USER, text="推荐")],
            selected_tools,
        )
        deepseek_body = deepseek_transport.calls[0]["body"]
        self.assertEqual(
            deepseek_body["messages"][0],
            {"role": "system", "content": _S4_RECOMMENDATION_PROMPT},
        )
        deepseek_function = deepseek_body["tools"][0]["function"]
        self.assertEqual(deepseek_function["name"], selected_tools[0].name)
        self.assertEqual(deepseek_function["description"], selected_tools[0].description)
        self.assertEqual(deepseek_function["parameters"], dict(selected_tools[0].input_schema))

        runner = FakeCodexRunner([(0, json.dumps({"result": "ok"}), "")])
        codex_provider(runner=runner).chat(
            _S4_RECOMMENDATION_PROMPT,
            [ProviderMessage(ProviderMessageRole.USER, text="推荐")],
            selected_tools,
        )
        codex_prompt = runner.calls[0]["prompt"]
        self.assertTrue(codex_prompt.startswith(_S4_RECOMMENDATION_PROMPT + "\n\n"))
        expected_tool_line = (
            f"- {selected_tools[0].name}: {selected_tools[0].description} "
            f"参数 JSON Schema: {json.dumps(dict(selected_tools[0].input_schema), ensure_ascii=False)}"
        )
        self.assertIn(expected_tool_line, codex_prompt)



class ChatProviderSelectionTest(unittest.TestCase):
    def test_build_chat_provider_dispatch(self) -> None:
        import argparse
        from unittest.mock import patch

        from music_agent.cli import _build_chat_provider
        from music_agent.codex_provider import CodexCliProvider
        from music_agent.deepseek_provider import DeepSeekApiProvider

        args = argparse.Namespace(
            provider="deepseek", base_url=None, model=None, api_key_env=None, timeout=30.0
        )
        self.assertIsInstance(_build_chat_provider(args), DeepSeekApiProvider)
        args = argparse.Namespace(provider="codex", base_url=None, model=None,
                                  api_key_env=None, timeout=30.0)
        self.assertIsInstance(_build_chat_provider(args), CodexCliProvider)

    def test_chat_parser_provider_choice(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args([
            "chat", "--db", "s.db", "--message", "hi",
            "--agent-client", "agt_11111111-1111-4111-8111-111111111111:full",
            "--provider", "codex",
        ])
        self.assertEqual(args.provider, "codex")
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "chat", "--db", "s.db", "--message", "hi", "--provider", "unknown"
            ])


if __name__ == "__main__":
    unittest.main()
