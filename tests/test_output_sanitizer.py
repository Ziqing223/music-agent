"""P14-R4.1 + P20 Fix 04: deterministic output-gate scrub unit tests.

Every rule of the closed vocabulary gets a direct case: internal-id
patterns (each prefix, CJK embedding, uppercase hex), field/value/route
literals in half- and full-width forms, fixed self-narration prefixes, and
the Fix 04 additions — route composites, boolean/count fields, internal id
field names, the extended process-narration family, and the empty fallback.
Clean text must pass through byte-identical.
"""

import unittest

from music_agent.output_sanitizer import (
    _EMPTY_TEXT_FALLBACK,
    sanitize_user_text,
)

UUID = "58f121dd-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CANDIDATE = "11111111-1111-4111-8111-111111111111"


class InternalIdScrubTest(unittest.TestCase):
    def test_each_prefix_maps_to_readable_class_word(self) -> None:
        samples = {
            "rcm": "推荐编号",
            "cnd": "候选编号",
            "fbk": "反馈编号",
            "trk": "曲目编号",
            "art": "艺人编号",
            "alb": "专辑编号",
            "pl": "歌单编号",
            "pm": "歌单成员编号",
            "int": "意图编号",
            "agt": "服务请求编号",
        }
        for prefix, expected in samples.items():
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"编号为 {prefix}_{UUID}。")
                self.assertNotIn(f"{prefix}_", scrubbed)
                self.assertEqual(scrubbed, f"编号为 {expected}。")

    def test_ids_embedded_in_chinese_sentence_stay_readable(self) -> None:
        text = (
            f"成功生成新一批（rcm_{UUID}），共 5 首，"
            f"首条候选 cnd_{CANDIDATE}。"
        )
        scrubbed = sanitize_user_text(text)
        self.assertIn("（推荐编号）", scrubbed)
        self.assertIn("候选编号", scrubbed)
        self.assertNotIn("rcm_", scrubbed)
        self.assertNotIn("cnd_", scrubbed)

    def test_multiple_ids_and_uppercase_hex(self) -> None:
        text = (
            "目标 trk_ABCDEF12-ABCD-4ABC-8ABC-ABCDEFABCDEF "
            f"艺人 art_{UUID}"
        )
        scrubbed = sanitize_user_text(text)
        self.assertEqual(scrubbed, "目标 曲目编号 艺人 艺人编号")

    def test_short_hex_blocks_are_not_touched(self) -> None:
        text = "小段字 trk_abc 保持原样。"
        self.assertEqual(sanitize_user_text(text), text)


class FieldAndValueLiteralTest(unittest.TestCase):
    def test_field_names_map_to_readable_words(self) -> None:
        text = "active_batch 有 5 首，runs_total: 3，preview_sounding 为 true。"
        scrubbed = sanitize_user_text(text)
        self.assertNotIn("active_batch", scrubbed)
        self.assertNotIn("runs_total", scrubbed)
        self.assertNotIn("preview_sounding", scrubbed)
        # Fix 04: the boolean field collapses to the honest audible-state
        # sentence instead of the P14 「试听状态 为 true」 half-translation.
        self.assertEqual(
            scrubbed, "当前批次 有 5 首，推荐批总数: 3，有试听正在播放。"
        )

    def test_route_labels_map_standalone(self) -> None:
        scrubbed = sanitize_user_text(
            "本批第一首 library，第二首 preview_only，第三首 unavailable。"
        )
        self.assertEqual(
            scrubbed, "本批第一首 可正式播放，第二首 只能试听，第三首 不可播放。"
        )

    def test_route_composite_collapses_to_full_natural_phrase(self) -> None:
        # P20 Fix 04 A: the P14 output shape 「playback.route 为 只能试听」
        # must collapse to the full capability phrase, not a half-translation.
        for form, expected in (
            ("playback.route 为 preview_only", "只能试听 30 秒"),
            ("playback.route 为 library", "可以正式播放"),
            ("playback.route 为 unavailable", "无法播放"),
        ):
            with self.subTest(form=form):
                scrubbed = sanitize_user_text(f"该曲 {form}。")
                self.assertNotIn("playback.route", scrubbed)
                self.assertNotIn("preview_only", scrubbed)
                self.assertNotIn("library", scrubbed)
                self.assertNotIn("unavailable", scrubbed)
                self.assertEqual(scrubbed, f"该曲 {expected}。")

    def test_bare_route_composite_and_contracted_forms(self) -> None:
        samples = {
            "第二首 route 是 preview_only。": "第二首 只能试听 30 秒。",
            "第二首 route=library。": "第二首 可以正式播放。",
            "第二首 route为unavailable。": "第二首 无法播放。",
            "第二首 playback_route 为 preview_only。": "第二首 只能试听 30 秒。",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                self.assertEqual(sanitize_user_text(text), expected)

    def test_route_field_name_maps_when_value_is_absent(self) -> None:
        scrubbed = sanitize_user_text("这首的 playback.route 由工具给出。")
        self.assertNotIn("playback.route", scrubbed)
        self.assertIn("播放情况", scrubbed)

    def test_tokens_inside_longer_words_are_not_matched(self) -> None:
        text = "libraries pinpoint_library unavailable_days keep intact."
        self.assertEqual(sanitize_user_text(text), text)

    def test_context_values_map_standalone(self) -> None:
        scrubbed = sanitize_user_text(
            "当前曲目 agent_selected，另一首 own_queue。"
        )
        self.assertEqual(
            scrubbed, "当前曲目 Agent 选的曲目，另一首 你自己的队列。"
        )

    def test_context_unknown_composites_collapse_without_dangling_field(self) -> None:
        for form in (
            "context 为 unknown",
            "context unknown",
            "context: unknown",
            "context:unknown",
            "context= unknown",
            "context：unknown",
        ):
            with self.subTest(form=form):
                scrubbed = sanitize_user_text(f"归属 {form}。")
                self.assertNotIn("context", scrubbed)
                self.assertNotIn("unknown", scrubbed)
                self.assertEqual(scrubbed, "归属 无法判断。")

    def test_bare_unknown_stays_untouched(self) -> None:
        text = "这首歌的归属 unknown 无法确定。"
        self.assertEqual(sanitize_user_text(text), text)


class Fix04BooleanAndCountFieldTest(unittest.TestCase):
    def test_preview_sounding_false_collapses_to_honest_state(self) -> None:
        # P20 Fix 04 C: 「试听状态 为 false」 must never reach the user.
        scrubbed = sanitize_user_text("preview_sounding=false。")
        self.assertNotIn("preview_sounding", scrubbed)
        self.assertNotIn("false", scrubbed)
        self.assertEqual(scrubbed, "当前没有正在播放的试听。")

    def test_preview_sounding_true_and_wrapped_forms(self) -> None:
        samples = {
            "试听中（preview_sounding 为 true）。": "试听中（有试听正在播放）。",
            "preview_sounding: ｆａｌｓｅ。": "当前没有正在播放的试听。",
            "preview_sounding是False。": "当前没有正在播放的试听。",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                self.assertEqual(sanitize_user_text(text), expected)

    def test_fresh_item_count_collapses_to_natural_claim(self) -> None:
        # P20 Fix 04 B: 「fresh_item_count = 5」 → 「这 5 首都是本次新发现」.
        samples = {
            "fresh_item_count = 5。": "这 5 首都是本次新发现。",
            "这批 fresh_item_count: ３（全是目录新歌）。": "这批 这 ３ 首都是本次新发现（全是目录新歌）。",
            "fresh_item_count=0。": "这 0 首都是本次新发现。",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                scrubbed = sanitize_user_text(text)
                self.assertNotIn("fresh_item_count", scrubbed)
                self.assertEqual(scrubbed, expected)

    def test_fresh_item_count_name_only_glosses(self) -> None:
        scrubbed = sanitize_user_text("这批的 fresh_item_count 不低。")
        self.assertNotIn("fresh_item_count", scrubbed)
        self.assertIn("本次新发现的曲目数", scrubbed)

    def test_fresh_this_request_boolean_collapses_per_item(self) -> None:
        samples = {
            "首条 fresh_this_request 为 true。": "首条 本次新发现。",
            "首条 fresh_this_request=false。": "首条 不是本次新发现。",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                scrubbed = sanitize_user_text(text)
                self.assertNotIn("fresh_this_request", scrubbed)
                self.assertEqual(scrubbed, expected)

    def test_internal_id_field_names_are_mapped(self) -> None:
        # P20 Fix 04 D: the bare internal id field names must never appear.
        text = (
            "canonical_id 已绑定，target_id 对应候选，candidate_id 是首条，"
            "run_id 来自上一批，referent_canonical_id 保留。"
        )
        scrubbed = sanitize_user_text(text)
        for forbidden in (
            "canonical_id",
            "target_id",
            "candidate_id",
            "run_id",
            "referent_canonical_id",
        ):
            self.assertNotIn(forbidden, scrubbed)
        for mapped in (
            "曲目标识",
            "目标曲目标识",
            "候选曲目标识",
            "推荐批次标识",
            "指代曲目标识",
        ):
            self.assertIn(mapped, scrubbed)

    def test_fix03_evidence_field_names_are_mapped(self) -> None:
        # P20 Fix 04 live run: the model echoed the Fix03 evidence-projection
        # field names 「evidence」「basis」 into the answer.
        scrubbed = sanitize_user_text(
            "每条 evidence 依据的真实 basis 都在记录里。"
        )
        self.assertNotIn("evidence", scrubbed)
        self.assertNotIn("basis", scrubbed)
        self.assertIn("推荐依据", scrubbed)


class FullWidthLiteralTest(unittest.TestCase):
    def test_fullwidth_field_and_route_literals_are_mapped(self) -> None:
        self.assertEqual(
            sanitize_user_text("ａｃｔｉｖｅ＿ｂａｔｃｈ 的值"), "当前批次 的值"
        )
        self.assertEqual(
            sanitize_user_text("首条是 ｌｉｂｒａｒｙ 吗"), "首条是 可正式播放 吗"
        )

    def test_fullwidth_context_unknown_composite_is_mapped(self) -> None:
        scrubbed = sanitize_user_text("归属 ｃｏｎｔｅｘｔ 为 ｕｎｋｎｏｗｎ。")
        self.assertEqual(scrubbed, "归属 无法判断。")

    def test_fullwidth_route_composite_is_collapsed(self) -> None:
        scrubbed = sanitize_user_text("该曲 ｐｌａｙｂａｃｋ．ｒｏｕｔｅ 为 ｐｒｅｖｉｅｗ＿ｏｎｌｙ。")
        self.assertEqual(scrubbed, "该曲 只能试听 30 秒。")


class SelfNarrationTest(unittest.TestCase):
    def test_fixed_prefix_sentence_is_stripped(self) -> None:
        text = "我如实告知用户这些编号是内部信息。以下是推荐：A — 甲"
        self.assertEqual(sanitize_user_text(text), "以下是推荐：A — 甲")

    def test_second_prefix_and_unterminated_form_falls_back(self) -> None:
        # Fix 04 fail-safe: fully scrubbed text returns the stable sentence,
        # never an empty answer.
        self.assertEqual(
            sanitize_user_text("我应该向用户说明后续将重试"), _EMPTY_TEXT_FALLBACK
        )

    def test_line_boundary_sentence_stripped_without_blank_line(self) -> None:
        text = "推荐如下：\n我应该向用户说明这是内部状态。\nA — 甲"
        self.assertEqual(sanitize_user_text(text), "推荐如下：\nA — 甲")

    def test_similar_natural_sentences_are_kept(self) -> None:
        text = "我如实告诉玩家规则。用户应该向我提问。"
        self.assertEqual(sanitize_user_text(text), text)

    def test_process_lead_in_sentence_is_stripped(self) -> None:
        text = "让我检查一下推荐历史。以下是推荐：A — 甲"
        self.assertEqual(sanitize_user_text(text), "以下是推荐：A — 甲")

    def test_tool_result_lead_ins_strip_whole_sentence(self) -> None:
        text = "我从工具结果得知这批有三首，根据工具返回第一首可播。推荐：A — 甲"
        self.assertEqual(sanitize_user_text(text), "推荐：A — 甲")

    def test_new_lead_in_line_boundary_sentence_stripped(self) -> None:
        text = "推荐如下：\n让我看看还有没有可播的。\nA — 甲"
        self.assertEqual(sanitize_user_text(text), "推荐如下：\nA — 甲")

    def test_unterminated_new_lead_in_falls_back(self) -> None:
        self.assertEqual(sanitize_user_text("我来查一下候选池"), _EMPTY_TEXT_FALLBACK)

    def test_lead_in_near_misses_are_kept(self) -> None:
        text = "我让你检查一下歌单顺序，让我慢慢再看一遍是否遗漏。"
        self.assertEqual(sanitize_user_text(text), text)


class Fix04ProcessNarrationTest(unittest.TestCase):
    """P20 Fix 04 E: the extended process-narration family (≥8 patterns)."""

    def test_third_person_intent_restatement_is_stripped(self) -> None:
        prefixes = (
            "用户明确说「播放第二首」，这是一个正式播放请求。",
            "用户明确要求换个方向，所以我重新生成。",
            "用户明确表示想要新歌。",
            "用户说了想听平缓的歌。",
            "用户要求我给出推荐。",
            "用户让我确认偏好。",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"{prefix}以下是推荐：A — 甲")
                self.assertEqual(scrubbed, "以下是推荐：A — 甲")

    def test_rule_self_narration_is_stripped(self) -> None:
        for prefix in (
            "按规则我不能自动降级为试听。",
            "根据规则我不能播放。",
        ):
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"{prefix}以下是推荐：A — 甲")
                self.assertEqual(scrubbed, "以下是推荐：A — 甲")

    def test_duty_and_service_lead_ins_are_stripped(self) -> None:
        prefixes = (
            "我应该向用户展示这批推荐。",
            "我需要如实向用户说明这一点。",
            "我来为用户展示这批推荐。",
            "我要为用户整理答案。",
            "我需要如实说明状态。",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"{prefix}以下是推荐：A — 甲")
                self.assertEqual(scrubbed, "以下是推荐：A — 甲")

    def test_sufficiency_and_confirmation_lead_ins_are_stripped(self) -> None:
        prefixes = (
            "信息充足。",
            "现在信息充足，可以回答了。",
            "让我确认一下结果。",
            "让我来确认播放状态。",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"{prefix}以下是推荐：A — 甲")
                self.assertEqual(scrubbed, "以下是推荐：A — 甲")

    def test_tool_result_restatement_lead_ins_are_stripped(self) -> None:
        prefixes = (
            "根据工具结果第一首可播。",
            "工具返回了这批的曲目。",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                scrubbed = sanitize_user_text(f"{prefix}以下是推荐：A — 甲")
                self.assertEqual(scrubbed, "以下是推荐：A — 甲")

    def test_live_rule_restatement_variants_are_stripped(self) -> None:
        # P20 Fix 04 live run: 「根据播放意图规则，该曲无法正式播放时不能
        # 自动降级成试听，应如实告知用户。」 — the internal rule-name
        # restatement family, both self-standing and in-sentence.
        scrubbed = sanitize_user_text(
            "根据播放意图规则，该曲无法正式播放时不能自动降级成试听，"
            "应如实告知用户。"
        )
        self.assertEqual(scrubbed, _EMPTY_TEXT_FALLBACK)
        scrubbed = sanitize_user_text(
            "好的。按播放意图规则不能播放。推荐：A — 甲"
        )
        self.assertEqual(scrubbed, "好的。推荐：A — 甲")

    def test_live_second_person_honesty_lead_in_is_stripped(self) -> None:
        # Live shape: 「我需要如实向你说明每首的依据：」 on its own line --
        # the strip consumes the lead-in, the real list on the next line
        # survives.
        scrubbed = sanitize_user_text(
            "这五首歌走的是目录推断。我需要如实向你说明每首的依据：\n"
            "1. A — 甲"
        )
        self.assertEqual(scrubbed, "这五首歌走的是目录推断。\n1. A — 甲")

    def test_live_yiju_lead_in_is_stripped(self) -> None:
        scrubbed = sanitize_user_text(
            "这批共五条。让我依据每首曲目的真实依据来解释。\nA — 甲"
        )
        self.assertEqual(scrubbed, "这批共五条。\nA — 甲")

    def test_live_english_thinking_preamble_is_stripped(self) -> None:
        # P20 Fix 04 live run: the model wrote its whole self-instruction
        # answer in English before the Chinese one.
        scrubbed = sanitize_user_text(
            "I have all the information I need. This batch is from the current "
            "active recommendation. Let me explain based on the actual evidence.\n"
            "推荐如下：A — 甲"
        )
        self.assertEqual(scrubbed, "推荐如下：A — 甲")

    def test_live_heshi_and_yiju_lead_ins_are_stripped(self) -> None:
        # Second live run (P20 Fix 04): 「让我核实一下——…」「让我梳理一下
        # …」 self-instruction cousins; the substantive list on the next line
        # survives while the stripped sentence leaves no dangling stop.
        scrubbed = sanitize_user_text(
            "让我核实一下——这批每条都是推断。\n- A — 甲"
        )
        self.assertEqual(scrubbed, "- A — 甲")
        scrubbed = sanitize_user_text(
            "我已经掌握了推荐依据。让我梳理一下这批推荐的证据来源。\n"
            "从反馈学习记录来看，你之前收藏过 Mandopop 的歌曲。"
        )
        self.assertEqual(
            scrubbed,
            "从反馈学习记录来看，你之前收藏过 Mandopop 的歌曲。",
        )

    def test_live_self_confirmation_lead_in_is_stripped(self) -> None:
        # 「我已经掌握了这批推荐的完整依据，现在为你说明。」 — the whole
        # self-confirmation sentence goes, the explanation that follows stays.
        scrubbed = sanitize_user_text(
            "我已经掌握了这批推荐的完整依据，现在为你说明。\n"
            "这批 5 首全部是 YOASOBI 的作品。"
        )
        self.assertEqual(scrubbed, "这批 5 首全部是 YOASOBI 的作品。")

    def test_stripped_opening_sentences_leave_no_dangling_stop(self) -> None:
        # Two fully-stripped sentences at the very start must not leave a
        # bare 「。」 before the surviving text.
        scrubbed = sanitize_user_text(
            "我已经掌握了。让我核实一下。\n推荐如下。"
        )
        self.assertEqual(scrubbed, "推荐如下。")

    def test_live_third_run_narration_cousins_are_stripped(self) -> None:
        # Third live run (P20 Fix 04): the model rephrased the banned family
        # — 按照规则 rule restatement, 让我为用户/让我说明 lead-ins, and the
        # Chinese form of the English self-confirmation preamble.
        scrubbed = sanitize_user_text(
            "按照规则，用户要求\"播放\"时只能走正式播放通路，"
            "不能自动降级为试听。\n"
            "第二首目前无法正式播放，只能试听 30 秒。"
        )
        self.assertEqual(scrubbed, "第二首目前无法正式播放，只能试听 30 秒。")
        scrubbed = sanitize_user_text(
            "新一批推荐已生成。让我为用户整理输出。\n1. A — 甲"
        )
        self.assertEqual(scrubbed, "新一批推荐已生成。\n1. A — 甲")
        scrubbed = sanitize_user_text(
            "我已经收集到足够的信息来解释这批推荐的依据了。\n这批共 5 首。"
        )
        self.assertEqual(scrubbed, "这批共 5 首。")
        scrubbed = sanitize_user_text(
            "让我说明得更清楚一些：\n---\n这场说明如下。"
        )
        self.assertEqual(scrubbed, "---\n这场说明如下。")

    def test_live_fourth_run_narration_cousins_are_stripped(self) -> None:
        # Fourth live run (P20 Fix 04): per-item lead-ins (让我逐条说明 /
        # 让我为你说明) and the search-restatement lead-in (搜索确认) — the
        # same genus with new phrasing; the real content that follows stays.
        scrubbed = sanitize_user_text(
            "搜索确认：这首在目录中的播放能力也是只能试听，无法正式播放。\n"
            "那么当前第二首无法正式播放。"
        )
        self.assertEqual(scrubbed, "那么当前第二首无法正式播放。")
        scrubbed = sanitize_user_text("让我逐条说明：\n1. A — 甲")
        self.assertEqual(scrubbed, "1. A — 甲")
        scrubbed = sanitize_user_text(
            "让我为你说明。\n这批推荐共 5 首。"
        )
        self.assertEqual(scrubbed, "这批推荐共 5 首。")
        # The closed 让我-class must never swallow natural 让我+verb speech.
        scrubbed = sanitize_user_text(
            "这首让我想起那年夏天的海边。试试看。"
        )
        self.assertEqual(scrubbed, "这首让我想起那年夏天的海边。试试看。")

    def test_live_fifth_run_mapped_forms_and_bold_values(self) -> None:
        # Fifth live run: markdown bold around values, 「标注为」 separators,
        # and Chinese-dressed mapped fields all collapse to the natural
        # playback phrases.
        scrubbed = sanitize_user_text(
            "该曲 playback.route 为 **preview_only**（preview_only）。"
        )
        self.assertEqual(scrubbed, "该曲 只能试听 30 秒（只能试听）。")
        scrubbed = sanitize_user_text(
            "该曲 playback.route 标注为 preview_only。"
        )
        self.assertEqual(scrubbed, "该曲 只能试听 30 秒。")
        scrubbed = sanitize_user_text(
            "该曲 播放情况 标注为 只能试听。"
        )
        self.assertEqual(scrubbed, "该曲 只能试听 30 秒。")
        scrubbed = sanitize_user_text(
            "preview_sounding 为 **false**。"
        )
        self.assertEqual(scrubbed, "当前没有正在播放的试听。")
        scrubbed = sanitize_user_text(
            "fresh_item_count = **5 首**。"
        )
        self.assertEqual(scrubbed, "这 5 首都是本次新发现。")
        scrubbed = sanitize_user_text(
            "是否本次新发现 标注为 true。"
        )
        self.assertEqual(scrubbed, "本次新发现。")

    def test_live_fifth_run_narration_cousins_are_stripped(self) -> None:
        # Fifth live run: attenuated rule restatement (我不能自动降级),
        # third-person intent restatement (用户点名), and the honesty
        # lead-in between 我应该向用户 and 我需要如实说明 (我应该如实).
        scrubbed = sanitize_user_text(
            "刚才的推荐条目都是 只能试听（目录推断的曲目）。"
            "用户点名「第二首」= 定位到当前批第 2 项，其无法正式播放。\n"
            "第二首目前无法正式播放，只能试听 30 秒。"
        )
        self.assertEqual(
            scrubbed,
            "刚才的推荐条目都是 只能试听（目录推断的曲目）。\n"
            "第二首目前无法正式播放，只能试听 30 秒。",
        )
        scrubbed = sanitize_user_text(
            "我不能自动降级为试听，\n\n第二首目前无法正式播放，"
            "只能试听 30 秒。"
        )
        self.assertEqual(
            scrubbed,
            "第二首目前无法正式播放，只能试听 30 秒。",
        )
        scrubbed = sanitize_user_text(
            "我理解了当前批次的情况。让我向用户解释这批推荐的依据。\n"
            "这批 5 首都是目录推断。"
        )
        self.assertEqual(scrubbed, "这批 5 首都是目录推断。")

    def test_live_sixth_run_narration_cousins_are_stripped(self) -> None:
        # Sixth live run: 我来整理/我来逐条 lead-ins and the 按照顺序给出
        # self-instruction; the real content that follows stays, and the
        # unbanned status sentence in the same reply is left untouched.
        scrubbed = sanitize_user_text(
            "生成成功，返回了一批新歌推荐（本次请求中发现 4 首新歌）。"
            "我来整理最终回答。\n"
            "这批推荐中，前 4 首是本次新发现的 J-Pop。"
        )
        self.assertEqual(
            scrubbed,
            "生成成功，返回了一批新歌推荐（本次请求中发现 4 首新歌）。\n"
            "这批推荐中，前 4 首是本次新发现的 J-Pop。",
        )
        scrubbed = sanitize_user_text(
            "按照顺序给出推荐，并在新歌上标注本次新发现。\n"
            "为你找到了这批新歌。"
        )
        self.assertEqual(scrubbed, "为你找到了这批新歌。")
        scrubbed = sanitize_user_text(
            "我来逐条解释这批推荐与你的偏好的对应关系。\n"
            "这批共 5 首，来自两个方向。"
        )
        self.assertEqual(scrubbed, "这批共 5 首，来自两个方向。")

    def test_seventh_run_xiangni_lead_in_is_stripped(self) -> None:
        # Seventh live run: the 让我向你 lead-in (让我向用户 was already
        # covered); the directional summary stays, the lead-in goes.
        scrubbed = sanitize_user_text(
            "这批推荐的 5 首歌全部来自同一方向（J-Pop / YOASOBI 目录推断），"
            "让我向你说明推荐理由。\n1. A — 甲"
        )
        self.assertEqual(
            scrubbed,
            "这批推荐的 5 首歌全部来自同一方向（J-Pop / YOASOBI 目录推断）\n"
            "1. A — 甲",
        )

    def test_english_process_phrases_are_case_sensitive_verbatim(self) -> None:
        # Only the observed verbatim forms strip; natural English song titles
        # and ordinary mixed lines survive.
        text = "My Songs Know What You Did In The Dark — Fall Out Boy"
        self.assertEqual(sanitize_user_text(text), text)


class Fix04NaturalLanguageSafetyTest(unittest.TestCase):
    """P20 Fix 04 F: natural sentences survive byte-identical."""

    NATURAL_SENTENCES = (
        "根据你的偏好记录，这首主要来自 J-Pop 方向。",
        "这首目前无法正式播放，只能试听 30 秒。",
        "我可以继续帮你找类似的歌。",
    )

    def test_natural_sentences_pass_through(self) -> None:
        for sentence in self.NATURAL_SENTENCES:
            with self.subTest(sentence=sentence):
                self.assertEqual(sanitize_user_text(sentence), sentence)


class Fix04EmptyFallbackTest(unittest.TestCase):
    """P20 Fix 04 G: empty-after-filter → stable fallback, never empty."""

    def test_fully_stripped_text_returns_fallback(self) -> None:
        self.assertEqual(
            sanitize_user_text("用户明确说「播放」，按规则我不能自动试听。"),
            _EMPTY_TEXT_FALLBACK,
        )

    def test_whitespace_only_input_returns_fallback(self) -> None:
        self.assertEqual(sanitize_user_text("   \n "), _EMPTY_TEXT_FALLBACK)

    def test_fallback_itsself_is_stable_and_never_stripped(self) -> None:
        self.assertEqual(sanitize_user_text(_EMPTY_TEXT_FALLBACK), _EMPTY_TEXT_FALLBACK)


class Fix04ExplanationRegressionTest(unittest.TestCase):
    """P20 Fix 04 H: Fix 03 explanation prose must survive unscathed."""

    def test_direct_preference_evidence_prose_passes(self) -> None:
        text = (
            "这首属于直接偏好，依据是《夜に駆ける》本身的偏好记录。"
        )
        self.assertEqual(sanitize_user_text(text), text)

    def test_mixed_direction_explanation_passes(self) -> None:
        text = (
            "这批 5 首里，第 1 首是你明确喜欢过的，其余 4 首是顺着你听的"
            " Mandopop（华语流行）方向找的新歌。"
        )
        self.assertEqual(sanitize_user_text(text), text)

    def test_fresh_no_evidence_honest_note_passes(self) -> None:
        text = "这首歌来自本次目录搜索的新发现，暂无偏好匹配证据。"
        self.assertEqual(sanitize_user_text(text), text)

    def test_artist_identity_is_kept(self) -> None:
        text = "Sunanowakusei — Kenshi Yonezu（米津玄师）的演奏。"
        self.assertEqual(sanitize_user_text(text), text)

    def test_inferred_wording_is_kept(self) -> None:
        text = "这首主要是根据你对 J-Pop 方向的偏好推断出来的。"
        self.assertEqual(sanitize_user_text(text), text)


class IntegrationAndPassthroughTest(unittest.TestCase):
    def test_leaky_reply_shape_is_fully_scrubbed(self) -> None:
        text = (
            f"本批生成成功（rcm_{UUID}），runs_total: 3。"
            "第一首 route 为 library 可正式播放，第二首 route 是 preview_only。"
            "我如实告知用户这是内部状态。"
            "第三首 context 为 unknown。"
        )
        scrubbed = sanitize_user_text(text)
        for forbidden in (
            "rcm_",
            "runs_total",
            "library",
            "preview_only",
            "unknown",
            "context",
            "我如实告知用户",
        ):
            self.assertNotIn(forbidden, scrubbed)
        self.assertIn("（推荐编号）", scrubbed)
        self.assertIn("推荐批总数", scrubbed)
        self.assertIn("只能试听 30 秒", scrubbed)
        self.assertIn("无法判断", scrubbed)

    def test_clean_text_passes_through_byte_identical(self) -> None:
        text = (
            "为你挑了三首：晴天 — 周杰伦（节奏轻快）、"
            "起风了 — 买辣椒也用券（旋律悠扬）。"
        )
        self.assertEqual(sanitize_user_text(text), text)

    def test_non_string_input_passes_through(self) -> None:
        self.assertIsNone(sanitize_user_text(None))


if __name__ == "__main__":
    unittest.main()