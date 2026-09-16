"""P10.8: Provider contract validation tests."""

import unittest

from music_agent.provider_contract import (
    ProviderConfig,
    ProviderError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
    ProviderToolCall,
    ProviderToolResult,
    ProviderToolSchema,
)


class ProviderMessageTest(unittest.TestCase):
    def test_at_least_one_kind_required(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderMessage(ProviderMessageRole.USER)

    def test_tool_results_must_be_the_only_kind(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderMessage(
                ProviderMessageRole.USER, text="a",
                tool_results=(ProviderToolResult("id", "content"),),
            )
        with self.assertRaises(ProviderError):
            ProviderMessage(
                ProviderMessageRole.USER,
                tool_calls=(ProviderToolCall("c1", "t", "{}"),),
                tool_results=(ProviderToolResult("id", "content"),),
            )

    def test_assistant_tool_calls_valid(self) -> None:
        message = ProviderMessage(
            ProviderMessageRole.ASSISTANT,
            tool_calls=(ProviderToolCall("c1", "get_canonical_entity", "{}"),),
        )
        self.assertEqual(message.tool_calls[0].name, "get_canonical_entity")

    def test_mixed_text_and_tool_calls_allowed_for_assistant(self) -> None:
        """Real providers emit a preamble alongside tool calls; the contract keeps it."""
        message = ProviderMessage(
            ProviderMessageRole.ASSISTANT,
            text="好的，我来查询。",
            tool_calls=(ProviderToolCall("c1", "get_canonical_entity", "{}"),),
        )
        self.assertEqual(message.text, "好的，我来查询。")
        self.assertEqual(message.tool_calls[0].name, "get_canonical_entity")
        # Non-assistant roles cannot carry tool calls, mixed or not.
        with self.assertRaises(ProviderError):
            ProviderMessage(
                ProviderMessageRole.USER,
                text="x",
                tool_calls=(ProviderToolCall("c1", "t", "{}"),),
            )

    def test_tool_result_must_be_user_role(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderMessage(
                ProviderMessageRole.ASSISTANT,
                tool_results=(ProviderToolResult("id", "content"),),
            )

    def test_system_cannot_carry_tool_calls(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderMessage(
                ProviderMessageRole.SYSTEM,
                tool_calls=(ProviderToolCall("c1", "x", "{}"),),
            )


class ProviderResponseTest(unittest.TestCase):
    def test_response_must_be_assistant(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderResponse(
                message=ProviderMessage(ProviderMessageRole.USER, text="x"),
                stop_reason=ProviderStopReason.END_TURN,
                usage={},
            )


class ProviderConfigTest(unittest.TestCase):
    def test_api_key_excluded_from_repr(self) -> None:
        config = ProviderConfig(base_url="https://example.com", model="m", api_key="SECRET")
        self.assertNotIn("SECRET", repr(config))

    def test_validates_fields(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderConfig(base_url="http://insecure", model="m")
        with self.assertRaises(ProviderError):
            ProviderConfig(base_url="https://example.com", model="")
        with self.assertRaises(ProviderError):
            ProviderConfig(base_url="https://example.com", model="m", timeout_seconds=0)
        with self.assertRaises(ProviderError):
            ProviderConfig(base_url="https://example.com", model="m", max_tokens=0)


class ProviderToolSchemaTest(unittest.TestCase):
    def test_requires_object_schema(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderToolSchema("t", "d", {"type": "string"})
        schema = ProviderToolSchema("t", "d", {"type": "object", "properties": {}})
        self.assertEqual(schema.name, "t")


if __name__ == "__main__":
    unittest.main()
