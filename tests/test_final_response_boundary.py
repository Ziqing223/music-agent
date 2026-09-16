"""P20 Fix 08 targeted tests: the final user response boundary.

The fixtures are the REAL leaked shapes from the UAT electrocardiogram
(/tmp/p20-final-uat-trace.jsonl) and the mandate's sec.11 pins -- every
abbreviated internal id, every process-narration block and every normal
capability sentence the product presents today. The suite proves:

* A/B/C: UAT ids (rcm_1b4e0124, trk_a7827505, trk_860deec5) and whole process
  blocks (让我核对 / 我不能基于歌名/艺人自编 / 规则说 / 我不该说 / 让我组织
  回答 / 我只能严格按照…字段说 / 向用户如实呈现 / 用户当前正在播放) never
  reach the presented text;
* D: a clean final answer after the process block survives whole;
* E: empty input and whole-text contamination collapse to the stable
  per-task fallback -- never the raw text, never an empty string;
* F: a sanitizer exception also falls back -- no code path echoes raw;
* G/H: the CLI door (_print_chat_result, shared by chat and chat-session)
  applies the same boundary;
* I: covered by the web /api/chat doors in tests/test_web_shell.py;
* J: normal-language answers (sec.9 pins) pass byte-identical;
* K/L/M/N: Fix03/Fix06 explanations, Fix05 direction shifts, PerfFix02
  fresh-batch replies and the real Fix07 live reply are untouched.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from music_agent.final_response_boundary import (
    FINAL_RESPONSE_FALLBACKS,
    present_final_text,
    presentation_fallback_kind,
)

_EXPLANATION_FALLBACK = FINAL_RESPONSE_FALLBACKS["explanation"]
_RECOMMENDATION_FALLBACK = FINAL_RESPONSE_FALLBACKS["recommendation"]
_DEFAULT_FALLBACK = FINAL_RESPONSE_FALLBACKS["default"]


class ExplicitFinalDeliveryTest(unittest.TestCase):
    def test_scratch_and_explicit_final_only_delivers_answer(self):
        answer = "找到《Spring Thief》— Yorushika，你的资料库中有两个可正式播放的版本。"
        raw = ("Let me note the context...\nLet me determine what action they want...\n"
               "This appears to be a search request...\n"
               "Since the message literally just asks to search...\n"
               "Let me present the results clearly...\n"
               f"<final_answer>{answer}</final_answer>")
        self.assertEqual(present_final_text(raw), answer)

    def test_unframed_scratch_fails_closed_instead_of_guessing_a_tail(self):
        for raw in ("Let me note the context...", "This appears to be a search request...",
                    "Let me determine what action they want...\n找到两条歌曲。",
                    "<final_answer>Let me present the results clearly...</final_answer>",
                    "<final_answer>one</final_answer><final_answer>two</final_answer>"):
            with self.subTest(raw=raw):
                self.assertEqual(present_final_text(raw), _DEFAULT_FALLBACK)

    def test_normal_english_is_preserved(self):
        for answer in ("I found two versions of Spring Thief in your Apple Music library.",
                       "Let me know which version you prefer.",
                       "This appears to be a live recording.",
                       "I can play either version for you."):
            self.assertEqual(present_final_text(answer), answer)

# The real UAT process-narration block (sec.11-B fixture, shape verbatim from
# the leaked "为什么这些适合我？" turn).
_UAT_PROCESS_BLOCK = (
    "让我核对一下刚刚拿到的信息。\n"
    "我不能基于歌名/艺人自编。\n"
    "规则说解释必须依据已有证据。\n"
    "我不该说更多内部内容。\n"
    "让我组织回答。\n"
    "我只能严格按照 推荐依据 字段说。\n"
)

# The honest final answer that followed the block (sec.11-D shape -- the
# clean tail of the UAT explanation turn). No trailing newline: Layer 1
# strips trailing whitespace, so byte-identical survival means this shape.
_UAT_CLEAN_EXPLANATION = (
    "根据你的偏好记录，这 5 首分别来自这些方向：\n"
    "• Every Breath You Take → Rock\n"
    "• Girls Just Want to Have Fun → Pop\n"
    "• All I Want → Soundtrack\n"
    "• Ashitamo → Rock\n"
    "• 晚风轻轻吹 → Pop"
)

# The real "再来一批" leakage (sec.11-C fixture, shape verbatim).
_UAT_ANOTHER_BATCH_LEAK = (
    "这一批只生成了 1 首，正是用户当前正在播放的那首歌本身。\n"
    "但这一批只有当前正在播的歌本身。\n"
    "向用户如实呈现这一结果，并给出后续选项。\n"
)

# The real Fix07 live batch reply (sec.11-N: byte-identical survival).
_FIX07_LIVE_REPLY = (
    "顺利生成了新批次，包含 3 首本次新发现的歌曲。让我按照返回顺序为你整理推荐。\n"
    "为你找来了这批新歌，其中有 3 首是本次目录搜索的新发现，3 首来自和你偏好方向匹配的目录曲目：\n"
    "**本次新发现的 3 首：**\n"
    "1. **FRESH — Lucky Kilimanjaro** — 全新的 J-Pop 单曲，和你的日系偏好方向一致，这次目录搜索刚找到，暂时还没有偏好匹配的详细证据，值得一听。\n"
    "2. **Comedy — Gen Hoshino** — 星野源的招牌轻快之作，本次目录搜索新发现，旋律悦耳、节奏明快。\n"
    "3. **FRESH — TVXQ!** — 东方神起的同名新曲，J-Pop 风格，同样是这次搜索到的目录新歌。\n"
    "**与你偏好匹配的熟悉目录曲目：**\n"
    "1. **Into The Night (Anonymouz Version) — Anonymouz** — 根据你对 Alternative 风格的偏好推断而来，翻唱自这首名曲的别样版本。\n"
    "2. **Into The Night (feat. 宵崎奏 & Hatsune Miku) — Nightcord at 25:00** — 基于你 Anime 方向的偏好推断，初音未来的合作演绎。\n"
    "这批只能试听 30 秒，需要我播放哪一首试试吗？或者你想换个方向再找一批？"
)


class AbbreviatedInternalIdTest(unittest.TestCase):
    """A: no internal id -- abbreviated or full -- reaches the presentation."""

    @staticmethod
    def _assert_no_id_residue(presented: str) -> None:
        for prefix in ("rcm_", "cnd_", "fbk_", "trk_", "art_", "alb_", "pl_",
                       "pm_", "int_", "agt_"):
            assert prefix not in presented, (prefix, presented)

    def test_uat_abbreviated_ids_never_reach_output(self) -> None:
        text = _UAT_PROCESS_BLOCK + (
            "我已经知道 rcm_1b4e0124 的候选 trk_a7827505 与 trk_860deec5 "
            "都来自 Rock 方向。\n"
        )
        presented = present_final_text(text, fallback_kind="explanation")
        for token in ("rcm_1b4e0124", "trk_a7827505", "trk_860deec5"):
            self.assertNotIn(token, presented)
        self._assert_no_id_residue(presented)

    def test_every_id_prefix_family_is_blocked(self) -> None:
        text = "这里有 rcm_a cnd_b fbk_c trk_d art_e alb_f pl_g pm_h int_i agt_j。"
        presented = present_final_text(text)
        self._assert_no_id_residue(presented)
        self.assertEqual(presented, _DEFAULT_FALLBACK)

    def test_full_uuid_ids_still_scrubbed_by_layer_1(self) -> None:
        text = "本批生成成功（rcm_11111111-1111-4111-8111-111111111111），可以开始试听。"
        presented = present_final_text(text)
        self.assertNotIn("rcm_", presented)
        self.assertIn("推荐编号", presented)  # layer-1's safe replacement

    def test_short_ids_blocked_even_mid_sentence_with_clean_region(self) -> None:
        # The clean tail survives whole; the contaminated sentence never does.
        presented = present_final_text(
            f"这首歌的 id 是 trk_a7827505。\n{_UAT_CLEAN_EXPLANATION}",
            fallback_kind="explanation",
        )
        self.assertNotIn("trk_", presented)
        self.assertIn("Every Breath You Take", presented)
        self.assertIn("晚风轻轻吹", presented)

    def test_fullwidth_abbreviated_id_is_blocked(self) -> None:
        presented = present_final_text("内部编号是 ｒｃｍ＿１ｂ４ｅ０１２４。")
        self.assertNotIn("ｒｃｍ", presented)
        self.assertEqual(presented, _DEFAULT_FALLBACK)


class ProcessLeakBlockTest(unittest.TestCase):
    """B/C: whole process-narration blocks never reach the presentation."""

    def test_uat_process_block_falls_back_when_no_clean_region(self) -> None:
        presented = present_final_text(
            _UAT_PROCESS_BLOCK, fallback_kind="explanation"
        )
        for leak in (
            "让我核对", "我不能基于歌名/艺人自编", "规则说", "我不该说",
            "让我组织回答", "我只能严格按照", "推荐依据 字段",
        ):
            self.assertNotIn(leak, presented)
        self.assertEqual(presented, _EXPLANATION_FALLBACK)

    def test_uat_process_block_then_clean_answer_preserves_answer(self) -> None:
        presented = present_final_text(
            _UAT_PROCESS_BLOCK + _UAT_CLEAN_EXPLANATION,
            fallback_kind="explanation",
        )
        for leak in ("让我核对", "我不能基于", "规则说", "我不该说", "让我组织回答"):
            self.assertNotIn(leak, presented)
        for kept in (
            "Every Breath You Take → Rock",
            "Girls Just Want to Have Fun → Pop",
            "All I Want → Soundtrack",
            "Ashitamo → Rock",
            "晚风轻轻吹 → Pop",
        ):
            self.assertIn(kept, presented)

    def test_uat_another_batch_leak_falls_back(self) -> None:
        presented = present_final_text(
            _UAT_ANOTHER_BATCH_LEAK, fallback_kind="recommendation"
        )
        for leak in ("正是用户当前正在播放", "用户当前", "向用户如实呈现这一结果",
                     "向用户", "并给出后续选项"):
            self.assertNotIn(leak, presented)
        self.assertEqual(presented, _RECOMMENDATION_FALLBACK)

    def test_each_marker_family_classifies_alone(self) -> None:
        # Only Layer-2 markers here (no Layer-1 prefix overlaps): each family
        # alone must close its unit to the per-task fallback.
        markers_and_kinds = [
            # (marker sentence, expected fallback kind)
            ("让我核对一下刚拿到的数据。", "recommendation"),
            ("我不能基于歌名/艺人自编。", "explanation"),
            ("规则说解释要基于已有证据。", "explanation"),
            ("我不该说这些。", "default"),
            ("让我组织回答。", "default"),
            ("我只能严格按照 推荐依据 字段说。", "explanation"),
            ("向用户如实呈现这一结果。", "default"),
            ("这是用户当前正在播放的歌。", "default"),
            ("我需要先看看这批的数量。", "default"),
        ]
        for sentence, kind in markers_and_kinds:
            with self.subTest(sentence=sentence):
                presented = present_final_text(sentence, fallback_kind=kind)
                self.assertEqual(presented, FINAL_RESPONSE_FALLBACKS[kind])

    def test_clean_head_then_process_tail_keeps_the_answer(self) -> None:
        text = (
            "为你找到这 5 首和你偏好方向一致的新歌，试试看吧。\n"
            "让我核对一下刚才生成的数量是否符合要求。\n"
        )
        presented = present_final_text(text, fallback_kind="recommendation")
        self.assertIn("为你找到这 5 首和你偏好方向一致的新歌", presented)
        self.assertNotIn("让我核对", presented)

    def test_dangling_process_tail_is_never_salvaged(self) -> None:
        # "并给出后续选项。"-shaped leftovers are too short to be a real
        # answer and must not survive as a broken remnant (sec.5 anti-patch).
        presented = present_final_text(
            "这批推荐有内部编号，需要如实说明。\n并给出后续选项。\n",
            fallback_kind="recommendation",
        )
        self.assertEqual(presented, _RECOMMENDATION_FALLBACK)
        self.assertNotIn("后续选项", presented)


class BareInternalFieldTest(unittest.TestCase):
    """B: bare unmapped internal field names are blocked."""

    def test_bare_field_names_never_reach_output(self) -> None:
        # Unmapped field names (Layer-1 leaves them verbatim) close the whole
        # unit at Layer 2: the contaminated sentence never reaches the user.
        for field in ("basis_targets", "source_path", "source_system",
                      "source_event_id"):
            with self.subTest(field=field):
                presented = present_final_text(
                    f"我把 {field} 的值写在这里。", fallback_kind="default"
                )
                self.assertNotIn(field, presented)
                self.assertEqual(presented, _DEFAULT_FALLBACK)

    def test_layer1_mapped_field_names_stay_replaced_never_english(self) -> None:
        # canonical_id/target_id are Layer-1-mapped names; what reaches the
        # user is the Chinese replacement and never the English token.
        for field in ("canonical_id", "target_id", "candidate_id", "run_id"):
            with self.subTest(field=field):
                presented = present_final_text(
                    f"我把 {field} 的值写在这里。", fallback_kind="default"
                )
                self.assertNotIn(field, presented)
                self.assertNotEqual(presented, _DEFAULT_FALLBACK)

    def test_field_in_process_head_preserves_clean_tail(self) -> None:
        presented = present_final_text(
            f"这条候选的 source_path 是偏好证据。\n{_UAT_CLEAN_EXPLANATION}",
            fallback_kind="explanation",
        )
        self.assertNotIn("source_path", presented)
        self.assertIn("Ashitamo → Rock", presented)

    def test_library_projection_fields_never_reach_user_output(self) -> None:
        fields = (
            "provenance", "provenance.kind", "route", "artist_ids",
            "artist_name", "play_count", "apple_music_persistent_id",
            "itunes_store_id", "persistent ID",
        )
        for field in fields:
            with self.subTest(field=field):
                presented = present_final_text(
                    f"内部结果的 {field}=value。", fallback_kind="default"
                )
                self.assertNotIn(field, presented)
                self.assertEqual(presented, _DEFAULT_FALLBACK)

    def test_scratch_reasoning_markers_fail_closed(self) -> None:
        for text in (
            "我应该先选择资料库候选。",
            "让我重新考虑这次选择。",
            "用户编号②暗示应选择第一条。",
            "I should inspect the route before answering.",
            "Let me reconsider the tool result.",
        ):
            with self.subTest(text=text):
                self.assertEqual(present_final_text(text), _DEFAULT_FALLBACK)


class FailClosedFallbackTest(unittest.TestCase):
    """E/F: no input state can ever fall through to the raw provider text."""

    def test_empty_input_uses_stable_fallback(self) -> None:
        for raw in ("", "   ", "\n\n"):
            with self.subTest(raw=repr(raw)):
                presented = present_final_text(raw)
                self.assertNotEqual(presented, raw)
                self.assertEqual(presented, _DEFAULT_FALLBACK)

    def test_whole_text_contamination_uses_stable_fallback(self) -> None:
        presented = present_final_text(
            "让我核对一下 rcm_1b4e0124 的内容。", fallback_kind="recommendation"
        )
        self.assertEqual(presented, _RECOMMENDATION_FALLBACK)

    def test_sanitizer_exception_falls_back_never_raw(self) -> None:
        with mock.patch(
            "music_agent.final_response_boundary.sanitize_user_text",
            side_effect=RuntimeError("scrubber exploded"),
        ):
            presented = present_final_text(
                "一切正常的回答。", fallback_kind="recommendation"
            )
        self.assertNotEqual(presented, "一切正常的回答。")
        self.assertEqual(presented, _RECOMMENDATION_FALLBACK)

    def test_unknown_fallback_kind_degrades_to_default(self) -> None:
        presented = present_final_text("让我核对。", fallback_kind="nonsense")
        self.assertEqual(presented, _DEFAULT_FALLBACK)

    def test_non_string_input_fails_closed(self) -> None:
        self.assertEqual(present_final_text(None), _DEFAULT_FALLBACK)  # type: ignore[arg-type]


class NormalLanguagePreservationTest(unittest.TestCase):
    """J: sec.9 pins + everyday capability sentences survive byte-identical."""

    def test_normal_language_is_never_touched(self) -> None:
        pins = (
            "我可以帮你试听这首歌。",
            "如果你愿意，我可以再换一个方向。",
            "根据你的偏好记录，这首主要来自 Rock 方向。",
            "这首目前无法正式播放，只能试听 30 秒。",
            "我不能正式播放这首歌，但可以试听 30 秒。",
            "这批推荐只有 1 首。",
            "目前没有足够的其它方向，我可以按你指定的风格继续找。",
            "好的，我已经为你播放这首歌。",
            "换个方向后的结果如下：这 5 首主要来自 J-Pop。",
            "这首歌不在你的资料库里，但我找到了它的试听。",
            "今天想听什么风格，我都可以试着找找看。",
            "根据你最近的反馈，我优先考虑 Rock 方向的歌。",
        )
        for sentence in pins:
            with self.subTest(sentence=sentence):
                self.assertEqual(present_final_text(sentence), sentence)


class ExplanationAndRegressionSurvivalTest(unittest.TestCase):
    """K/L/M/N: proven-good outputs from every later Fix pass untouched."""

    def test_fix03_fix06_explanation_evidence_survives(self) -> None:
        presented = present_final_text(_UAT_CLEAN_EXPLANATION)
        self.assertEqual(presented, _UAT_CLEAN_EXPLANATION)

    def test_fix05_direction_shift_reply_survives(self) -> None:
        text = "好的，换到 J-Pop 方向。这一批主要来自 J-Pop，只有 3 首符合方向。"
        self.assertEqual(present_final_text(text), text)

    def test_perffix02_fresh_batch_reply_survives(self) -> None:
        text = (
            "为你找来了这批新歌，其中有 3 首是本次目录搜索的新发现：\n"
            "1. **FRESH — Lucky Kilimanjaro** — 全新的 J-Pop 单曲，值得一听。\n"
            "这批只能试听 30 秒，需要我播放哪一首试试吗？"
        )
        self.assertEqual(present_final_text(text), text)

    def test_fix07_live_delivered_reply_survives_byte_identical(self) -> None:
        self.assertEqual(present_final_text(_FIX07_LIVE_REPLY), _FIX07_LIVE_REPLY)

    def test_short_hex_blocks_layer_1_contract_still_untouched_by_layer_1(self) -> None:
        # Fix04's pinned contract is preserved: layer-1 alone still leaves
        # short blocks alone; the Layer-2 boundary is what closes them.
        from music_agent.output_sanitizer import sanitize_user_text

        self.assertEqual(
            sanitize_user_text("小段字 trk_abc 保持原样。"), "小段字 trk_abc 保持原样。"
        )
        self.assertNotIn("trk_", present_final_text("小段字 trk_abc 保持原样。"))


class FallbackKindSelectionTest(unittest.TestCase):
    def test_explanation_shapes_select_explanation_fallback(self) -> None:
        # The REAL closed explanation forms (Fix06's _RECOMMENDATION_EXPLANATION_
        # FORMS), with the punctuated live shapes the door does see.
        for line in ("为什么这一批适合我？", "为什么这些适合我？", "为什么推荐这些",
                     "推荐理由是什么", "这几首为什么适合我"):
            with self.subTest(line=line):
                self.assertEqual(presentation_fallback_kind(line), "explanation")

    def test_recommendation_shapes_select_recommendation_fallback(self) -> None:
        # Recommendation AND fresh-discovery intents both own the
        # recommendation fallback (fresh discovery is its own classifier).
        for line in ("推荐几首歌", "再来一批", "推荐一些我没听过的新歌"):
            with self.subTest(line=line):
                self.assertEqual(presentation_fallback_kind(line), "recommendation")

    def test_play_and_plain_chat_default(self) -> None:
        for line in ("播放 Hanataba", "你好", "暂停"):
            with self.subTest(line=line):
                self.assertEqual(presentation_fallback_kind(line), "default")

    def test_generated_run_upgrades_to_recommendation_fallback(self) -> None:
        from music_agent.provider_agent import ProviderLoopToolExecution

        generated = (ProviderLoopToolExecution(name="generate_recommendation", outcome="ok"),)
        self.assertEqual(presentation_fallback_kind("谢谢", generated), "recommendation")
        self.assertEqual(presentation_fallback_kind("谢谢", ()), "default")


class CliDoorTest(unittest.TestCase):
    """G/H: the CLI print door (chat and chat-session share it) is closed."""

    @staticmethod
    def _print(as_text: str, *, user_text: str | None = None,
               executions: tuple = (), final_text: str | None = None) -> str:
        from music_agent.cli import _print_chat_result

        result = SimpleNamespace(
            final_text=as_text,
            tool_executions=executions,
            rounds=1,
            context_trimmed=False,
            rounds_capped=False,
            total_elapsed_ms=1,
        )
        with redirect_stdout(io.StringIO()) as captured:
            _print_chat_result(
                result, "deepseek", user_text=user_text, final_text=final_text
            )
        return captured.getvalue()

    def test_process_block_never_reaches_cli_stdout(self) -> None:
        out = self._print(_UAT_PROCESS_BLOCK, user_text="为什么这一批适合我？")
        self.assertIn(_EXPLANATION_FALLBACK + "\n", out)
        for leak in ("让我核对", "规则说", "我不该说"):
            self.assertNotIn(leak, out)

    def test_abbreviated_id_never_reaches_cli_stdout(self) -> None:
        out = self._print(
            _UAT_PROCESS_BLOCK + "候选 rcm_1b4e0124 是 Rock 方向。\n" + _UAT_CLEAN_EXPLANATION,
            user_text="为什么这一批适合我？",
        )
        self.assertNotIn("rcm_", out)
        self.assertIn("Every Breath You Take", out)

    def test_override_door_text_still_passes_boundary(self) -> None:
        # The T14-E override path goes through the same boundary, and canned
        # honest sentences pass unchanged.
        out = self._print(
            "好的。", final_text="让我核对一下内部状态。", user_text="播放这首歌"
        )
        self.assertNotIn("让我核对", out)
        self.assertEqual(out, _DEFAULT_FALLBACK + "\n")

    def test_normal_answer_reaches_cli_stdout_unchanged(self) -> None:
        text = "我可以帮你试听这首歌。\n"
        self.assertEqual(self._print(text, user_text="你好"), text)


if __name__ == "__main__":
    unittest.main()
