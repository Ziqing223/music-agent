"""P20 Fix 06: recommendation-explanation variant grounding regression.

The UAT-live line 「为什么这一批适合我？」 missed the closed explanation
set and ran on the 31-tool full surface with the DEFAULT prompt, so the
answer drifted into score claims, encyclopedia backfill and current-playing
causality.  These tests pin the fix: every verb-first spelling of the
why-question enters the same read-only evidence-grounded explanation path
as 「为什么推荐这些？」, mixed intent keeps the full fail-safe surface,
the explanation prompt re-bans score/encyclopedia/current-playing claims
(with the explicit-score carve-out), and the Fix-04 sanitizer strips the
UAT process-narration sentences while natural language still passes through
byte-identical.  Fix 03 / Fix 04 / Fix 05 behaviour is re-anchored so the
new doors cannot regress the old ones.
"""

import unittest

from music_agent.intent_router import (
    is_fresh_discovery_intent,
    is_recommendation_explanation_intent,
    is_recommendation_request,
)
from music_agent.output_sanitizer import _EMPTY_TEXT_FALLBACK, sanitize_user_text
from music_agent.provider_agent import (
    DEFAULT_SYSTEM_PROMPT,
    _EXPLANATION_TOOL_NAMES,
    _GENERATION_TOOL_NAMES,
    _S4_DISCOVERY_PROMPT,
    _S4_EXPLANATION_PROMPT,
    _S4_RECOMMENDATION_PROMPT,
    _select_system_prompt,
    _select_task_tools,
)
from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

# The six §三 spellings of "explain why this batch fits me" -- all must
# enter the same evidence-grounded explanation path.
VARIANTS = (
    "为什么推荐这些？",
    "为什么这些适合我？",
    "为什么这一批适合我？",
    "为什么这批适合我？",
    "这批为什么适合我？",
    "这批推荐为什么适合我？",
)


class ExplanationVariantRoutingTests(unittest.TestCase):
    # ---- §七-1: every variant enters the read-only explanation path ------

    def test_all_six_variants_classify_as_explanation(self) -> None:
        for text in VARIANTS:
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_explanation_intent(text))
                # Explanation ≠ generation: none of them is a recommendation
                # or fresh-discovery request either.
                self.assertFalse(is_recommendation_request(text))
                self.assertFalse(is_fresh_discovery_intent(text))

    def test_all_six_variants_get_the_evidence_grounded_explanation_prompt(
        self,
    ) -> None:
        for text in VARIANTS:
            with self.subTest(text=text):
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    _S4_EXPLANATION_PROMPT,
                )

    def test_all_six_variants_run_the_12_tool_read_only_surface(self) -> None:
        for text in VARIANTS:
            with self.subTest(text=text):
                offered = {
                    tool.name
                    for tool in _select_task_tools(text, PROVIDER_TOOL_SCHEMAS)
                }
                self.assertEqual(offered, set(_EXPLANATION_TOOL_NAMES))
                self.assertTrue(offered.isdisjoint(_GENERATION_TOOL_NAMES))
                self.assertNotIn("discover_catalog_tracks", offered)

    # ---- §七-2: mixed intent keeps the full fail-safe surface ------------

    def test_mixed_explanation_plus_generation_intent_stays_full(self) -> None:
        # 为什么这一批适合我 + 再给我推荐五首 is NOT a closed explanation
        # form: narrowing would rob the second half of its generation tools.
        text = "为什么这一批适合我，再给我推荐五首"
        self.assertFalse(is_recommendation_explanation_intent(text))
        self.assertEqual(
            _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
            DEFAULT_SYSTEM_PROMPT,
        )
        offered = {
            tool.name for tool in _select_task_tools(text, PROVIDER_TOOL_SCHEMAS)
        }
        self.assertNotEqual(offered, set(_EXPLANATION_TOOL_NAMES))
        self.assertTrue(_GENERATION_TOOL_NAMES <= offered)
        self.assertIn("discover_catalog_tracks", offered)


class ExplanationPromptDisciplineTests(unittest.TestCase):
    # ---- §七-3: score claims stay banned without an explicit question -----

    def test_score_claims_are_banned_with_the_score_question_carve_out(
        self,
    ) -> None:
        self.assertIn("不得说成满分", _S4_EXPLANATION_PROMPT)
        for banned in (
            "满分",
            "评分满分",
            "score 1.0",
            "100% 匹配",
            "百分百",
            "高度吻合",
            "匹配度很高",
        ):
            with self.subTest(banned=banned):
                self.assertIn(banned, _S4_EXPLANATION_PROMPT)
        # The carve-out: only an explicit score question may surface numbers.
        self.assertIn("分数是多少", _S4_EXPLANATION_PROMPT)

    # ---- §七-4: encyclopedia backfill stays banned ------------------------

    def test_encyclopedia_backfill_is_banned_without_recorded_evidence(
        self,
    ) -> None:
        self.assertIn(
            "不得凭空补写歌曲的具体出处作品", _S4_EXPLANATION_PROMPT
        )
        for banned in ("主题曲", "插曲", "代表作", "流派", "历史背景"):
            with self.subTest(banned=banned):
                self.assertIn(banned, _S4_EXPLANATION_PROMPT)

    # ---- §七-5: the current-playing song is not new causal evidence -------

    def test_current_playing_song_cannot_become_answer_causal_evidence(
        self,
    ) -> None:
        self.assertIn(
            "严禁自行把正在播放的歌曲拉进来当推荐因果证据",
            _S4_EXPLANATION_PROMPT,
        )
        # … but the durable basis may legitimately cite it.
        self.assertIn(
            "被批次条目的 basis 明确引用", _S4_EXPLANATION_PROMPT
        )

    # ---- §七-7: Fix03 evidence grounding clauses survive unchanged --------

    def test_explanation_prompt_still_carries_fix03_evidence_clauses(
        self,
    ) -> None:
        for clause in (
            "逐条对应刚读到的批次条目",
            "name/artist_name",
            "暂无偏好匹配证据",
            "证据方向混合时如实列出各方向",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, _S4_EXPLANATION_PROMPT)


class ExplanationSanitizerTests(unittest.TestCase):
    # ---- §七-6: the two UAT sentences are process narration ----------------

    def test_uat_process_narration_sentences_are_stripped(self) -> None:
        for sentence in (
            "让我结合当前正在播放的歌曲和偏好情况来回答你。",
            "让我结合偏好证据来说明理由。",
            "让我结合你的收藏与评分依据来回答你。",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(sanitize_user_text(sentence), _EMPTY_TEXT_FALLBACK)

    def test_uat_strip_keeps_the_real_answer_that_follows(self) -> None:
        scrubbed = sanitize_user_text(
            "让我结合偏好证据来说明理由。第一首来自你的收藏。"
        )
        self.assertEqual(scrubbed, "第一首来自你的收藏。")

    def test_natural_sentences_containing_结合_survive(self) -> None:
        # 让我结合 is narration only as the sentence lead-in; plain 结合 in
        # real content must pass through byte-identical.
        for text in (
            "这首歌结合了传统民乐与电子合成器。",
            "结合你的偏好，这首更合口味。",
            "推荐理由结合了评分和收藏两个方向。",
            "让我联想到小时候常听的歌。",
        ):
            with self.subTest(text=text):
                self.assertEqual(sanitize_user_text(text), text)


class NoRegressionTests(unittest.TestCase):
    # ---- §七-8: Fix04 natural-language laundry list ------------------------

    def test_natural_answers_pass_through_unchanged(self) -> None:
        for text in (
            "第一首是周杰伦的《晴天》，因为来自你的收藏。",
            "这五首里有三首属于 J-Pop 方向。",
            "上一批推荐过这首，所以这批换了一批。",
            "试听在响，正式播放已暂停。",
            "谢谢你，我明白了。",
        ):
            with self.subTest(text=text):
                self.assertEqual(sanitize_user_text(text), text)

    # ---- §七-9: Fix05 direction-shift phrasing untouched ------------------

    def test_direction_shift_phrases_are_untouched_by_the_explanation_fix(
        self,
    ) -> None:
        for text in ("再来一批，换个方向", "换一组换个方向", "再推荐一批，换方向"):
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_request(text))
                self.assertFalse(is_recommendation_explanation_intent(text))
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    _S4_RECOMMENDATION_PROMPT,
                )
                offered = {
                    tool.name
                    for tool in _select_task_tools(text, PROVIDER_TOOL_SCHEMAS)
                }
                self.assertTrue(_GENERATION_TOOL_NAMES <= offered)

    def test_fresh_discovery_keeps_its_own_prompt_and_surface(self) -> None:
        text = "推荐一些没听过的新歌"
        self.assertTrue(is_fresh_discovery_intent(text))
        self.assertFalse(is_recommendation_explanation_intent(text))
        self.assertEqual(
            _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
            _S4_DISCOVERY_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()