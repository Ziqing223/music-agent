"""P22-S1 tests for the deterministic-first Conversation Interpreter."""

import hashlib
import json
import unittest

from music_agent.intent_router import (
    TurnExpectedResult,
    TurnPrimarySemantic,
    TurnSemanticSource,
    TurnTaskSurface,
    resolve_turn_plan,
)
from music_agent.prompts.conversation_interpreter import INTERPRETER_SYSTEM_PROMPT
from music_agent.provider_contract import (
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
)
from music_agent.turn_interpreter import (
    TurnInterpreterContext,
    interpret_turn,
)


def _response(payload) -> ProviderResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return ProviderResponse(
        ProviderMessage(ProviderMessageRole.ASSISTANT, text=text),
        ProviderStopReason.END_TURN,
        {},
    )


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, system, messages, tools):
        self.calls.append((system, tuple(messages), tuple(tools)))
        if not self.responses:
            raise AssertionError("provider exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class PatternProvider:
    """Test-only semantic provider for the required P22 natural-language matrix."""

    def __init__(self):
        self.calls = []

    def chat(self, system, messages, tools):
        self.calls.append((system, tuple(messages), tuple(tools)))
        request = json.loads(messages[0].text)
        text = request["user_text"]
        generic = {
            "intent": "recommendation",
            "recommendation": {
                "mode": "generic",
                "requested_count": 5,
                "scene": None,
                "seed": None,
            },
            "action": None,
            "requires_clarification": False,
            "reason": None,
        }
        if text in {
            "推荐",
            "推荐一下",
            "来点歌",
            "给我来几首",
            "我想听点歌",
            "放点好听的",
            "再来几首",
        }:
            return _response(generic)
        if text in {"我想听类似这首的歌", "来点和这首感觉接近的"}:
            return _response(
                {
                    "intent": "recommendation",
                    "recommendation": {
                        "mode": "similarity",
                        "requested_count": 5,
                        "scene": None,
                        "seed": {"kind": "current_track", "value": None},
                    },
                    "action": None,
                    "requires_clarification": False,
                    "reason": None,
                }
            )
        if text == "给我找几首和 Spring Thief 差不多的":
            return _response(
                {
                    "intent": "recommendation",
                    "recommendation": {
                        "mode": "similarity",
                        "requested_count": 5,
                        "scene": None,
                        "seed": {"kind": "free_text", "value": "Spring Thief"},
                    },
                    "action": None,
                    "requires_clarification": False,
                    "reason": None,
                }
            )
        if text in {"随便来一首", "你帮我挑一首", "从刚才那些里选一首"}:
            return _response(
                {
                    "intent": "delegated_selection",
                    "recommendation": None,
                    "action": {
                        "source": "active_recommendation",
                        "selection_mode": "agent_choose_one",
                    },
                    "requires_clarification": False,
                    "reason": None,
                }
            )
        raise AssertionError(text)


class TurnInterpreterTest(unittest.TestCase):
    def test_extracted_system_prompt_matches_pre_extraction_snapshot_exactly(self):
        self.assertEqual(len(INTERPRETER_SYSTEM_PROMPT), 1090)
        self.assertEqual(
            hashlib.sha256(INTERPRETER_SYSTEM_PROMPT.encode()).hexdigest(),
            "f169ca1b9853019d2cf8a8137b7a7af1ebd1a89b034789b436cad21e156e388b",
        )

    def _interpret(self, text, provider, context=None):
        deterministic = resolve_turn_plan(text)
        self.assertEqual(deterministic.primary, TurnPrimarySemantic.UNKNOWN)
        self.assertEqual(deterministic.task_surface, TurnTaskSurface.FULL)
        self.assertEqual(deterministic.semantic_source, TurnSemanticSource.UNRESOLVED)
        return interpret_turn(
            provider,
            text,
            deterministic,
            context or TurnInterpreterContext(),
        )

    def test_required_open_forms_remain_deterministic_unknown_then_interpret(self):
        provider = PatternProvider()
        generic_forms = (
            "推荐",
            "推荐一下",
            "来点歌",
            "给我来几首",
            "我想听点歌",
            "放点好听的",
            "再来几首",
        )
        for text in generic_forms:
            with self.subTest(text=text):
                result = self._interpret(text, provider)
                self.assertEqual(result.status, "interpreted")
                self.assertEqual(result.plan.primary, TurnPrimarySemantic.RECOMMENDATION)
                self.assertEqual(result.plan.recommendation.mode, "generic")
                self.assertEqual(result.plan.semantic_source, TurnSemanticSource.LLM_INTERPRETED)
        self.assertTrue(all(call[2] == () for call in provider.calls))

    def test_similarity_forms_map_only_raw_referent_semantics(self):
        provider = PatternProvider()
        for text in ("我想听类似这首的歌", "来点和这首感觉接近的"):
            result = self._interpret(
                text,
                provider,
                TurnInterpreterContext(has_current_playback=True),
            )
            self.assertEqual(result.plan.recommendation.mode, "similarity_seed")
            self.assertEqual(result.plan.recommendation.seed_source, "current_track")
            self.assertIsNone(result.plan.recommendation.target)
        named = self._interpret(
            "给我找几首和 Spring Thief 差不多的", provider
        )
        self.assertEqual(named.plan.recommendation.mode, "similarity_seed")
        self.assertEqual(named.plan.recommendation.target, "Spring Thief")
        self.assertFalse(hasattr(named.plan.recommendation, "canonical_id"))

    def test_current_track_similarity_requires_verified_current_playback_even_with_referenced_item(self):
        provider = PatternProvider()
        result = self._interpret(
            "我想听类似这首的歌",
            provider,
            TurnInterpreterContext(
                has_current_playback=False,
                has_referenced_item=True,
            ),
        )
        self.assertEqual(result.status, "clarification")
        self.assertTrue(result.plan.requires_clarification)
        self.assertEqual(result.plan.expected_result, TurnExpectedResult.CLARIFICATION)
        self.assertIsNone(result.plan.recommendation)

        named = self._interpret(
            "给我找几首和 Spring Thief 差不多的",
            provider,
            TurnInterpreterContext(
                has_current_playback=False,
                has_referenced_item=True,
            ),
        )
        self.assertEqual(named.status, "interpreted")
        self.assertEqual(named.plan.recommendation.mode, "similarity_seed")
        self.assertEqual(named.plan.recommendation.target, "Spring Thief")

    def test_delegated_selection_carries_no_selected_item_route_or_tool(self):
        provider = PatternProvider()
        context = TurnInterpreterContext(
            has_active_recommendation=True,
            active_recommendation_item_count=5,
        )
        for text in ("随便来一首", "你帮我挑一首", "从刚才那些里选一首"):
            with self.subTest(text=text):
                result = self._interpret(text, provider, context)
                action = result.plan.playback_action
                self.assertEqual(result.status, "interpreted")
                self.assertEqual(result.plan.primary, TurnPrimarySemantic.PLAYBACK_ACTION)
                self.assertEqual(action.source, "active_recommendation")
                self.assertEqual(action.selection_mode, "agent_choose_one")
                self.assertTrue(action.delegated)
                self.assertFalse(hasattr(action, "canonical_id"))
                self.assertFalse(hasattr(action, "route"))
                self.assertFalse(hasattr(action, "tool"))

    def test_context_can_fail_closed_for_missing_active_recommendation(self):
        provider = PatternProvider()
        result = self._interpret(
            "随便来一首",
            provider,
            TurnInterpreterContext(has_active_recommendation=False),
        )
        self.assertEqual(result.status, "clarification")
        self.assertTrue(result.plan.requires_clarification)
        self.assertEqual(result.plan.expected_result, TurnExpectedResult.CLARIFICATION)

    def test_ambiguous_model_result_becomes_clarification(self):
        provider = ScriptedProvider(
            [
                _response(
                    {
                        "intent": "clarification",
                        "recommendation": None,
                        "action": None,
                        "requires_clarification": True,
                        "reason": "referent_missing",
                    }
                )
            ]
        )
        result = self._interpret("就这个", provider)
        self.assertEqual(result.status, "clarification")
        self.assertTrue(result.plan.requires_clarification)
        self.assertEqual(result.plan.semantic_source, TurnSemanticSource.UNRESOLVED)

    def test_authoritative_fields_are_rejected_not_accepted_as_truth(self):
        provider = ScriptedProvider(
            [
                _response(
                    {
                        "intent": "delegated_selection",
                        "recommendation": None,
                        "action": {
                            "source": "active_recommendation",
                            "selection_mode": "agent_choose_one",
                            "canonical_id": "trk_model_invented",
                            "route": "library",
                            "tool": "play_track",
                            "selected_item": 2,
                            "success": True,
                        },
                        "requires_clarification": False,
                        "reason": None,
                    }
                )
            ]
        )
        result = self._interpret(
            "随便来一首",
            provider,
            TurnInterpreterContext(has_active_recommendation=True),
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.plan.requires_clarification)
        self.assertEqual(result.error_code, "invalid_structured_output")

    def test_failure_matrix_fails_closed(self):
        invalid_payloads = [
            "not-json",
            {
                "intent": "invented_intent",
                "recommendation": None,
                "action": None,
                "requires_clarification": False,
                "reason": None,
            },
            {
                "intent": "recommendation",
                "recommendation": {
                    "mode": "generic",
                    "requested_count": 5,
                    "scene": None,
                    "seed": None,
                },
                "action": {
                    "source": "active_recommendation",
                    "selection_mode": "agent_choose_one",
                },
                "requires_clarification": False,
                "reason": None,
            },
            {
                "intent": "recommendation",
                "recommendation": {
                    "mode": "similarity",
                    "requested_count": 5,
                    "scene": None,
                    "seed": {"kind": "free_text", "value": None},
                },
                "action": None,
                "requires_clarification": False,
                "reason": None,
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                provider = ScriptedProvider([_response(payload)])
                result = self._interpret("弄一下", provider)
                self.assertEqual(result.status, "failed")
                self.assertTrue(result.plan.requires_clarification)
                self.assertEqual(result.error_code, "invalid_structured_output")

        provider = ScriptedProvider([RuntimeError("boom")])
        result = self._interpret("来一个", provider)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.plan.requires_clarification)
        self.assertEqual(result.error_code, "provider_error")

    def test_valid_unsupported_preserves_safe_existing_full_fallback(self):
        provider = ScriptedProvider(
            [
                _response(
                    {
                        "intent": "unsupported",
                        "recommendation": None,
                        "action": None,
                        "requires_clarification": False,
                        "reason": "outside_p22_s1_scope",
                    }
                )
            ]
        )
        result = self._interpret("查两次同一实体", provider)
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.plan.primary, TurnPrimarySemantic.UNKNOWN)
        self.assertEqual(result.plan.task_surface, TurnTaskSurface.FULL)
        self.assertFalse(result.plan.requires_clarification)
        self.assertEqual(result.plan.semantic_source, TurnSemanticSource.UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
