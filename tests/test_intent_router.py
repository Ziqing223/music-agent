"""P3B fast path: the local control-command intent router.

Exact match only -- playback commands map to their P09 tool name; anything
ambiguous, longer than the command, or conversational returns None so the
provider loop takes it. The router itself is a pure function: no service, no
provider, no state. P14-C06.3a/b: 停止/stop routes on the real preview truth
(preview_sounding, the runner read) and 换一首 on channel + ownership, only on
the prompt-blessed combinations; unknown context never invents a route.
P15-S1: the conditional session table (暂停 ≙ stop_preview, stop over the
inter-clip gap, continue family left to the caller's interception) -- only a
literally "running" session state changes routing, never a terminal one.
P15-S1 C02: 下一首/next/下一首试听 advance the RUNNING session (advance_preview);
off a session 下一首/next keep their pre-P15 route and the 试听 form refuses.
"""

from __future__ import annotations

import unittest

from music_agent.intent_router import (
    TurnExpectedResult,
    TurnPrimarySemantic,
    TurnTaskSurface,
    continuous_preview_session_running,
    is_delegated_playback_intent,
    is_explicit_play_intent,
    is_feedback_intent,
    expects_recommendation_batch,
    is_fresh_discovery_intent,
    is_plain_chat,
    is_preview_intent,
    is_read_only_library_intent,
    is_recommendation_explanation_intent,
    is_recommendation_intent,
    is_recommendation_request,
    recommendation_presentation_label,
    resolve_current_track_feedback_turn_semantics,
    resolve_preference_statement_turn_semantics,
    resolve_recommendation_turn_semantics,
    needs_active_context,
    normalize_track_reference_pronouns,
    pronoun_track_reference,
    pronoun_track_reference_target,
    route_intent,
    resolve_turn_plan,
)


class IntentRouterTest(unittest.TestCase):
    def test_chinese_commands_route_exactly(self) -> None:
        cases = {
            "暂停": "pause",
            "停止试听": "stop_preview",
            "继续播放": "play",
            "继续": "play",
            "下一首": "next_track",
            "上一首": "previous_track",
        }
        for text, tool in cases.items():
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), tool)

    def test_english_commands_route_case_insensitively(self) -> None:
        cases = {
            "pause": "pause",
            "PAUSE": "pause",
            "stop preview": "stop_preview",
            "Stop Preview": "stop_preview",
            "play": "play",
            "resume": "play",
            "continue": "play",
            "next": "next_track",
            "prev": "previous_track",
            "previous": "previous_track",
        }
        for text, tool in cases.items():
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), tool)

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        self.assertEqual(route_intent("  暂停  "), "pause")
        self.assertEqual(route_intent("\tnext\n"), "next_track")

    def test_empty_and_non_string_inputs_return_none(self) -> None:
        for text in ("", "   ", "\t\n", None, 7):  # type: ignore[arg-type]
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))

    def test_ordinary_chat_does_not_misfire(self) -> None:
        for text in (
            "推荐几首新歌",
            "换个别的",
            "你好",
            # extension words around a command are NOT the command
            "帮我暂停",
            "暂停一下",
            "我想继续播放",
            "下一首歌叫什么",
            "上一首吧",
            # near-misses are refused (trailing punctuation is tolerated now,
            # see PlaybackStatusRoutingTest; 停止/stop need the preview truth)
            "下一曲",
            "播放",
            "停止",
            "stop",
            # full conversational intents stay with the provider loop
            "换一首",
            "随便播放一首",
            "你来决定",
        ):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))


class DelegatedPlaybackIntentTest(unittest.TestCase):
    def test_existing_closed_delegation_family_is_authorized(self) -> None:
        for text in (
            "你来决定",
            "你选",
            "随便",
            "随便播放一首",
            "放首歌",
            "你来决定？",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_delegated_playback_intent(text))

    def test_nearby_words_do_not_expand_delegation_authority(self) -> None:
        for text in (
            "推荐几首歌",
            "播放一首歌",
            "随便推荐几首",
            "你来决定推荐什么",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_delegated_playback_intent(text))


class ContextAwareRoutingTest(unittest.TestCase):
    """P14-C06.3a: the two context-sensitive forms route only on the exact
    prompt-blessed combinations; every other context reads as unknown and refuses."""

    def test_stop_routes_on_preview_truth_only(self) -> None:
        """P14-C06.3b: stop needs the real runner truth; the channel register no
        longer routes it on its own."""
        for text in ("stop", "停止"):
            with self.subTest(text=text):
                self.assertEqual(
                    route_intent(text, preview_sounding=True), "stop_preview"
                )
                # The truth wins regardless of what the action log claims.
                self.assertEqual(
                    route_intent(text, channel="preview", preview_sounding=True),
                    "stop_preview",
                )
                self.assertEqual(
                    route_intent(text, channel="none", preview_sounding=True),
                    "stop_preview",
                )
                # No sounding preview: never auto-stop, provider decides.
                for channel in ("preview", "library", "none", None):
                    self.assertIsNone(
                        route_intent(text, channel=channel, preview_sounding=False)
                    )
                self.assertIsNone(
                    route_intent(text, channel=channel, preview_sounding=False, context="own_queue")
                )
                # Channel alone is an action log, not proof of sound.
                self.assertIsNone(route_intent(text, channel="preview"))
                self.assertIsNone(route_intent(text))

    def test_huan_shou_routes_only_on_own_queue_with_no_channel(self) -> None:
        self.assertEqual(
            route_intent("换一首", channel="none", context="own_queue"), "next_track"
        )
        fallbacks = [
            ("none", "agent_selected"),
            ("none", "unknown"),
            ("none", None),
            (None, "own_queue"),
            ("preview", "own_queue"),
            ("library", "own_queue"),
            ("library", "agent_selected"),
            ("preview", None),
        ]
        for channel, context in fallbacks:
            with self.subTest(channel=channel, context=context):
                self.assertIsNone(route_intent("换一首", channel=channel, context=context))

    def test_context_never_changes_the_plain_table(self) -> None:
        self.assertEqual(route_intent("暂停", channel="preview"), "pause")
        self.assertEqual(
            route_intent("next", channel="library", context="agent_selected"), "next_track"
        )

    def test_default_no_context_preserves_legacy_refusals(self) -> None:
        for text in ("停止", "stop", "换一首"):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))

    def test_context_tokens_are_matched_strictly(self) -> None:
        # The caller passes the canonical tokens straight from get_active_context;
        # near-miss spellings must not route (宁可不路由，不误触发).
        self.assertIsNone(route_intent("停止", channel="PREVIEW"))
        self.assertIsNone(route_intent("停止", channel=" preview"))
        self.assertIsNone(route_intent("停止", channel=7))
        self.assertIsNone(route_intent("换一首", channel="none", context="OWN_QUEUE"))
        self.assertIsNone(route_intent("换一首", channel="none", context=7))
        # preview_sounding counts only when literally True -- truthiness never routes.
        for bad in (1, "yes", "True", None):
            with self.subTest(bad=bad):
                self.assertIsNone(route_intent("停止", preview_sounding=bad))

    def test_needs_active_context_pins_only_the_sensitive_forms(self) -> None:
        for text in (
            "stop", " stop ", "停止", " 停止 ", "换一首", "\t换一首",
            # P15-S1: the session-aware forms joined the set (暂停 can flip to
            # stop_preview; the continue family is intercepted while running).
            "暂停", " pause ", "继续", "继续播放", "continue", "play", " resume ",
            # P15-S1 C02: 下一首/next can flip to advance_preview while a session
            # runs, and 下一首试听 only routes off a live session -- all three pay
            # the one local read.
            "下一首", " next ", "下一首试听",
        ):
            with self.subTest(text=text):
                self.assertTrue(needs_active_context(text))
        for text in (
            "停止试听", "stop preview",
            "播放", "停", "帮我停止", "", "   ", None, 7,
        ):
            with self.subTest(text=text):
                self.assertFalse(needs_active_context(text))


class SessionAwareRoutingTest(unittest.TestCase):
    """P15-S1 §8: the session table is conditional -- only a literally "running"
    session changes routing; every other state (or no state) keeps the pre-P15
    behavior verbatim."""

    def test_pause_family_routes_to_stop_preview_only_while_running(self) -> None:
        for text in ("暂停", "pause", " PAUSE "):
            with self.subTest(text=text):
                self.assertEqual(
                    route_intent(text, preview_session_state="running"), "stop_preview"
                )
                # No session: the plain pause route, exactly as before.
                self.assertEqual(route_intent(text, preview_session_state=None), "pause")
                # Terminal sessions: pause is Music.app's again (regression-safe).
                for state in ("completed", "cancelled"):
                    self.assertEqual(
                        route_intent(text, preview_session_state=state), "pause"
                    )

    def test_stop_covers_the_running_session_even_without_sound(self) -> None:
        """A running session between clips: nothing sounding, yet stop/停止 must
        still end the session (the runner truth alone cannot see the gap)."""
        for text in ("stop", "停止"):
            with self.subTest(text=text):
                self.assertEqual(
                    route_intent(
                        text,
                        preview_sounding=False,
                        preview_session_state="running",
                    ),
                    "stop_preview",
                )
                # Terminal session off sound: unchanged refusal.
                self.assertIsNone(
                    route_intent(
                        text,
                        preview_sounding=False,
                        preview_session_state="completed",
                    )
                )

    def test_continue_family_keeps_routing_to_play_under_every_state(self) -> None:
        """The router never blocks the continue family itself: the caller intercepts
        a play route while the session runs (never a false next_track/stop here)."""
        for text in ("继续", "继续播放", "continue", "play", "resume"):
            with self.subTest(text=text):
                for state in ("running", "completed", "cancelled", None):
                    self.assertEqual(
                        route_intent(text, preview_session_state=state), "play"
                    )

    def test_session_state_is_matched_strictly(self) -> None:
        # Near-miss tokens never count as a running session (宁可不路由，不误触发).
        for bad in ("RUNNING", " running", "Running", 7, True):
            with self.subTest(bad=bad):
                self.assertEqual(route_intent("暂停", preview_session_state=bad), "pause")
        self.assertEqual(route_intent("停止", preview_session_state="RUNNING"), None)

    def test_other_commands_are_untouched_by_the_session_state(self) -> None:
        """Only the session-aware forms flip; 停止试听 and 上一首 never consult
        the session (上一首 stays Music.app's -- C02 deliberately left it alone)."""
        for state in ("running", None):
            self.assertEqual(
                route_intent("停止试听", preview_session_state=state), "stop_preview"
            )
            self.assertEqual(
                route_intent("上一首", preview_session_state=state), "previous_track"
            )


class AdvancePreviewRoutingTest(unittest.TestCase):
    """P15-S1 C02: 下一首/next/下一首试听 advance the RUNNING session in place;
    off a session the plain next forms keep their pre-P15 routes verbatim and the
    试听 form refuses (the router never invents a session)."""

    def test_next_forms_advance_the_running_session(self) -> None:
        for text in ("下一首", "next", " NEXT ", "下一首试听", " 下一首试听 "):
            with self.subTest(text=text):
                self.assertEqual(
                    route_intent(text, preview_session_state="running"),
                    "advance_preview",
                )

    def test_next_forms_keep_pre_p15_routes_off_a_session(self) -> None:
        for state in (None, "completed", "cancelled", "failed"):
            with self.subTest(state=state):
                # 下一首/next: the plain next_track route, exactly as before.
                self.assertEqual(
                    route_intent("下一首", preview_session_state=state), "next_track"
                )
                self.assertEqual(
                    route_intent("next", preview_session_state=state), "next_track"
                )
                # Without a live session there is nothing to advance -- refuse
                # and let the provider loop decide what 下一首试听 means.
                self.assertIsNone(
                    route_intent("下一首试听", preview_session_state=state)
                )

    def test_preview_form_matches_the_session_token_strictly(self) -> None:
        # Near-miss session tokens never advance: 宁可不路由，不误触发.
        for bad in ("RUNNING", " running", "Running", 7, True):
            with self.subTest(bad=bad):
                self.assertIsNone(
                    route_intent("下一首试听", preview_session_state=bad)
                )
        self.assertEqual(
            route_intent("下一首", preview_session_state="RUNNING"), "next_track"
        )


class PlaybackStatusRoutingTest(unittest.TestCase):
    """P15-S4-M3-B: the five V1 status phrases route to the playback_status
    pseudo-command -- a closed set: trailing punctuation is tolerated, but any
    nearby/elongated form stays a refusal for the provider loop. The tolerated
    punctuation applies to the whole table (the design §5 extension), so a
    punctuated plain command keeps its route and its context fetch."""

    V1_PHRASES = (
        "现在在播放什么",
        "现在播放的是什么",
        "当前在播放什么",
        "当前播放什么",
        "现在是什么歌",
    )

    def test_v1_phrases_route_to_playback_status(self) -> None:
        for text in self.V1_PHRASES:
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), "playback_status")

    def test_trailing_punctuation_and_whitespace_are_tolerated(self) -> None:
        variants = (
            "现在在播放什么？",
            "现在在播放什么?",
            "现在在播放什么。",
            " 现在在播放什么 ？",
            "现在播放的是什么！",
            "现在是什么歌…",
            "当前在播放什么？?。",
            "当前播放什么。",
            " 当前播放什么 ",
            "当前在播放什么！",
        )
        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), "playback_status")

    def test_status_phrases_do_not_pay_a_context_prefetch(self) -> None:
        for text in self.V1_PHRASES + ("现在在播放什么？",):
            with self.subTest(text=text):
                self.assertFalse(needs_active_context(text))

    def test_nearby_queries_never_route(self) -> None:
        # True near-misses: not in any table, the provider loop decides.
        for text in (
            "这是什么歌",
            "现在在播放什么歌",
            "刚才在播放什么",
            "现在播放的是什么呢",
            "播放什么",
            "播放xxx",
            "播放周杰伦",
            "推荐",
            "反馈",
            "试听",
        ):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))

    def test_design_exclusions_never_become_playback_status(self) -> None:
        """The design's explicit V1 exclusions stay out of the status fast path.
        Commands that exist elsewhere keep their own routes; the rest refuse --
        none of them may flip into playback_status."""
        for text in ("继续播放", "暂停", "下一首", "推荐", "反馈", "试听"):
            with self.subTest(text=text):
                self.assertNotEqual(route_intent(text), "playback_status")
        # The valid commands among them keep their exact pre-M3-B routes.
        self.assertEqual(route_intent("继续播放"), "play")
        self.assertEqual(route_intent("暂停"), "pause")
        self.assertEqual(route_intent("下一首"), "next_track")

    def test_plain_commands_with_trailing_punctuation_keep_routing(self) -> None:
        self.assertEqual(route_intent("暂停。"), "pause")
        self.assertEqual(route_intent("next!"), "next_track")
        self.assertEqual(route_intent(" 停止。", preview_sounding=True), "stop_preview")
        self.assertIsNone(route_intent("停止。"))
        # context-sensitive forms keep their context fetch through punctuation.
        self.assertTrue(needs_active_context("暂停。"))
        self.assertTrue(needs_active_context("停止。"))
        self.assertFalse(needs_active_context("现在在播放什么。"))


class FormalPlayRoutingTest(unittest.TestCase):
    """P16-S3: the five V1 formal-playback phrases route to the formal_play
    pseudo-command -- a closed set with the same discipline as the M3-B
    status set. The delegation family (随便播放一首 等) deliberately stays
    with the provider loop: it permits a preview fallback the deterministic
    runner refuses, so stealing it would change behavior, not just latency."""

    V1_PHRASES = (
        "播放一首正式歌曲",
        "放一首正式歌曲",
        "播放一首正式的歌",
        "来一首正式歌曲",
        "正式播放一首",
    )

    def test_v1_phrases_route_to_formal_play(self) -> None:
        for text in self.V1_PHRASES:
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), "formal_play")

    def test_trailing_punctuation_and_whitespace_are_tolerated(self) -> None:
        for text in (
            "播放一首正式歌曲。",
            " 播放一首正式歌曲？",
            "放一首正式歌曲!",
            "正式播放一首…",
            " 来一首正式歌曲 ！",
        ):
            with self.subTest(text=text):
                self.assertEqual(route_intent(text), "formal_play")

    def test_formal_phrases_do_not_pay_a_context_prefetch(self) -> None:
        for text in self.V1_PHRASES + ("播放一首正式歌曲？",):
            with self.subTest(text=text):
                self.assertFalse(needs_active_context(text))

    def test_delegation_family_is_never_stolen(self) -> None:
        # The plain delegation family keeps its provider-loop route: it may
        # fall back to a preview, which the deterministic runner refuses.
        for text in (
            "随便播放一首",
            "放首歌",
            "你来决定",
            "来一首歌",
            "播放一首歌",
            "推荐一首播放",
        ):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))

    def test_nearby_queries_never_route(self) -> None:
        # Anything beyond the exact five forms stays a refusal -- the provider
        # prompt (now including the explicit-正式 rule) decides the rest.
        for text in (
            "请播放一首正式歌曲",
            "播放一首正式歌曲吧",
            "正式播放",
            "播放正式的",
            "正式歌曲",
            "播放一首好听的正式歌曲",
        ):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(text))

    def test_design_exclusions_never_become_formal_play(self) -> None:
        """Existing commands keep their exact routes; none of them may flip
        into formal_play."""
        for text in (
            "继续播放",
            "暂停",
            "下一首",
            "现在在播放什么",
            "停止试听",
        ):
            with self.subTest(text=text):
                self.assertNotEqual(route_intent(text), "formal_play")
        self.assertEqual(route_intent("继续播放"), "play")
        self.assertEqual(route_intent("暂停"), "pause")
        self.assertEqual(route_intent("现在在播放什么"), "playback_status")


class T14AStopFamilyRoutingTest(unittest.TestCase):
    """P19-T14-A: the three stop-family forms the web shell fast path serves.

    暂停试听 names the preview explicitly, so it is a plain context-free
    table entry (stop_preview, exactly like 停止试听); 不听了 joins stop/停止
    as a context-sensitive form -- it flips to stop_preview only on real
    runner truth (preview_sounding is True / session running) and stays None
    otherwise so the provider loop answers as it always has. The plain 暂停
    rules are untouched by construction."""

    def test_pause_preview_routes_context_free(self) -> None:
        self.assertEqual(route_intent("暂停试听"), "stop_preview")
        self.assertEqual(route_intent(" 暂停试听。"), "stop_preview")
        # never a Music.app pause, whatever the channel/truth claims: the
        # form names the preview, so its route owes nothing to any read.
        for channel in ("preview", "library", "none", None):
            self.assertEqual(
                route_intent("暂停试听", channel=channel), "stop_preview"
            )
        self.assertEqual(
            route_intent("暂停试听", preview_sounding=False), "stop_preview"
        )
        self.assertEqual(
            route_intent("暂停试听", preview_session_state="running"),
            "stop_preview",
        )

    def test_bu_ting_le_flips_only_on_preview_truth(self) -> None:
        self.assertIsNone(route_intent("不听了"))
        self.assertIsNone(route_intent(" 不听了。"))
        self.assertEqual(
            route_intent("不听了", preview_sounding=True), "stop_preview"
        )
        self.assertEqual(
            route_intent("不听了", preview_session_state="running"), "stop_preview"
        )
        # no runner truth: the flip refuses even when the channel assert logs.
        self.assertIsNone(route_intent("不听了", preview_sounding=False))
        self.assertIsNone(route_intent("不听了", channel="preview"))

    def test_stop_family_pays_the_context_read(self) -> None:
        # 不听了 could change with the live context, so it pays the read on
        # the caller's fast path; the table forms never do.
        self.assertTrue(needs_active_context("不听了"))
        self.assertFalse(needs_active_context("暂停试听"))
        self.assertFalse(needs_active_context("停止试听"))
        self.assertTrue(needs_active_context("暂停"))

    def test_plain_pause_rules_are_unchanged(self) -> None:
        # P19-T14-A must not rewrite plain 暂停: it stays pause context-free,
        # and flips to stop_preview only while a preview session literally
        # runs (the pre-existing P15-S1 rule).
        self.assertEqual(route_intent("暂停"), "pause")
        self.assertEqual(route_intent("暂停", preview_sounding=True), "pause")
        self.assertEqual(
            route_intent("暂停", preview_session_state="running"), "stop_preview"
        )


class RecommendationIntentGuardTest(unittest.TestCase):
    """P19-T14-B: ``is_recommendation_intent`` classifies the closed
    recommendation-request form set for the web shell's reply door. It never
    executes anything -- it only feeds the door's decision when a reply
    finished without a generated batch, so false positives silently discard
    legitimate replies. The set is the web shell's own shortcut chips plus
    the similar-to-current and recommend-a-batch families; everything else
    (extra words, playlist creation, questions about the feature) is False."""

    POSITIVE_FORMS = (
        # the web shell's own shortcut chips (ui/index.html)
        "推荐音乐",
        "找类似这首的",
        "换个心情",
        # the similar-to-current family (the Hanataba live failure wording)
        "类似这首的",
        "类似的歌",
        "类似的歌曲",
        "找类似的",
        "找类似的歌",
        "找相似的歌",
        "推荐类似的",
        "推荐和这首类似的",
        "类似刚才这首",
        "相似刚才这首",
        # the recommend-a-batch family
        "推荐几首歌",
        "给我推荐",
        "给我推荐几首歌",
        "帮我推荐",
        "帮忙推荐几首歌",
        "来一批",
        "再来一批",
        "再来点新的",
        "换一批",
        "换一组",
        "推荐新的",
        "推荐点新的",
        "推荐一些新歌",
    )

    def test_closed_recommendation_forms_classify_true(self) -> None:
        for text in self.POSITIVE_FORMS:
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_intent(text))

    def test_trailing_punctuation_and_whitespace_are_tolerated(self) -> None:
        for text in (
            "找类似这首的。",
            " 推荐音乐？",
            "换一批!",
            " 找类似的歌 …",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_intent(text))

    def test_recommendation_classification_never_routes_anything(self) -> None:
        # The guard is a pure classifier: none of its forms gain a playback
        # route, and none of the routed commands classify as recommendation.
        for text in self.POSITIVE_FORMS:
            self.assertIsNone(route_intent(text))
        for text in ("暂停", "下一首", "停止试听", "播放一首正式歌曲"):
            self.assertFalse(is_recommendation_intent(text))

    def test_near_miss_phrasings_stay_false(self) -> None:
        # Anything beyond the exact forms -- extra words, questions about the
        # feature, playlist creation -- returns False so a legitimate reply
        # is never silently discarded.
        for text in (
            "请问怎么推荐",
            "推荐功能怎么用",
            "帮我找一首类似的歌吧",
            "还有没有类似这首歌的",
            "推荐几首好听的",
            "创建歌单",
            "创建一个歌单",
            "随便放一首",
            "放首歌",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_recommendation_intent(text))

    def test_non_string_input_is_never_recommendation(self) -> None:
        for value in (None, 42, ["找类似这首的"], b"find-similar"):
            self.assertFalse(is_recommendation_intent(value))


class T14EPlayIntentClassificationTest(unittest.TestCase):
    """P19-T14-E: explicit-play recognition for the Play-vs-Preview guard.

    The classifier never executes anything (it only feeds the harness door),
    so a false positive costs one redundant post-run check; the admissions and
    exclusions below are still written to keep the play family in and every
    feature/question/试听-mixed phrasing out.
    """

    POSITIVE_FORMS = (
        # bare play
        "播放",
        "播放。",
        # batch-item forms
        "播放这首",
        "播放那首",
        "播放这一首",
        "播放那一首",
        "播放上一首",
        # ordinal forms (arabic + Chinese numerals)
        "播放第2首",
        "播放第10首",
        "播放第三首",
        "播放第十二首",
        # named-track forms (any song/artist suffices)
        "播放 Hanataba",
        "播放周杰伦的晴天",
        "播放 back number 的那首歌",
    )

    NEGATIVE_FORMS = (
        # feature/question phrasings that merely start with 播放
        "播放器怎么用",
        "播放列表",
        "播放队列空了",
        "播放历史",
        "播放功能怎么用",
        "播放状态怎么看",
        "播放这首歌吗",
        "播放吗",
        # 试听-touching (mixed intent stays the preview family's surface)
        "播放前先试听一下",
        "播放或试听",
        # explicit preview family
        "试听 Hanataba",
        "试听这首",
        "试听第2首",
        # delegation family (whitelisted play-OR-preview, not pure play)
        "随便播放一首",
        "放首歌",
        "你来决定",
        # near-miss play phrasings outside the closed grammar
        "帮我播放 Hanataba",
        "怎么播放",
        "我要播放",
        # non-play control phrases
        "暂停",
        "下一首",
        "停止试听",
    )

    def test_play_family_forms_are_recognized(self) -> None:
        for text in self.POSITIVE_FORMS:
            with self.subTest(text=text):
                self.assertTrue(is_explicit_play_intent(text))

    def test_feature_question_and_preview_forms_stay_false(self) -> None:
        for text in self.NEGATIVE_FORMS:
            with self.subTest(text=text):
                self.assertFalse(is_explicit_play_intent(text))

    def test_non_string_input_is_never_play_intent(self) -> None:
        for value in (None, 42, ["播放这首"], b"play"):
            self.assertFalse(is_explicit_play_intent(value))


class ReadOnlyLibraryIntentTest(unittest.TestCase):
    def test_search_and_library_questions_are_read_only(self) -> None:
        for text in (
            "搜索 Spring Thief Yorushika",
            "查找 Spring Thief",
            "有没有 Spring Thief",
            "Spring Thief 在我的资料库里吗？",
            "我的资料库有哪些 Yorushika 的歌？",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_read_only_library_intent(text))

    def test_explicit_playback_never_becomes_read_only(self) -> None:
        for text in (
            "播放 Spring Thief — Yorushika", "开始播放 Spring Thief", "放这首"
        ):
            with self.subTest(text=text):
                self.assertFalse(is_read_only_library_intent(text))
                self.assertTrue(is_explicit_play_intent(text))


class T14FTrackReferencePronounNormalizationTest(unittest.TestCase):
    """P19-T14-F: 它/他/她 equivalence inside track-reference patterns only.

    ``normalize_track_reference_pronouns`` is a pure text rewrite at the
    parse boundary: 他/她 becomes 它 ONLY as the direct, sentence-final
    object of a track-reference verb (试听 / 播放 / 停止试听). Every other
    reading -- possessives, subjects, questions -- passes through untouched,
    and the rewrite never invents or removes a route.
    """

    # The spoken pronoun variants resolve identically to the proven 它 path.
    EQUIVALENT_FORMS = (
        ("试听他", "试听它"),
        ("试听她", "试听它"),
        ("试听他。", "试听它。"),
        ("试听她！", "试听它！"),
        (" 试听他 ", " 试听它 "),
        ("播放他", "播放它"),
        ("播放她", "播放它"),
        ("播放他…", "播放它…"),
        ("播放她。", "播放它。"),
        ("停止试听他", "停止试听它"),
        ("停止试听她", "停止试听它"),
        ("别播放他", "别播放它"),
    )

    # Unrelated 他/她 uses must never be rewritten globally.
    UNCHANGED_FORMS = (
        "试听他的歌",
        "播放她的歌",
        "他试听",
        "她说这首歌很好听",
        "他是歌手",
        "她走了",
        "给他播放一首歌",
        "播放给他听",
        "试听下一首他和她的歌",
        "问问他",
        "播放他吗",
        "试听一下他",
        "她喜欢试听",
        "播放器坏了",
        "他和她都在",
        "试听他 的歌",
        "试听他她",
        "他的是这首歌",
        "谁说他走了",
        "播放这首",
    )

    def test_pronoun_variants_rewrite_to_它_inside_track_reference_patterns(self) -> None:
        for source, expected in self.EQUIVALENT_FORMS:
            with self.subTest(source=source):
                self.assertEqual(
                    normalize_track_reference_pronouns(source), expected
                )

    def test_它_spelling_is_idempotent(self) -> None:
        for text in ("试听它", "播放它", "停止试听它", "试听它。"):
            with self.subTest(text=text):
                self.assertEqual(normalize_track_reference_pronouns(text), text)

    def test_unrelated_pronoun_uses_are_not_rewritten(self) -> None:
        for text in self.UNCHANGED_FORMS:
            with self.subTest(text=text):
                self.assertEqual(normalize_track_reference_pronouns(text), text)

    def test_non_string_or_empty_input_is_unchanged(self) -> None:
        for value in (None, "", 42, ["试听他"], b"\xe8\xaf\x95\xe5\x90\xac"):
            with self.subTest(value=value):
                self.assertEqual(normalize_track_reference_pronouns(value), value)

    def test_normalization_never_creates_or_removes_a_route(self) -> None:
        # T14-F: no 试听/播放 pronoun-object spelling is routed, so the
        # rewrite is routing-neutral in both directions for the preview/play
        # family: variants that do not route today stay unrouted after
        # normalization, and no normalized form collides with the table.
        for text in ("试听他", "试听她", "播放他", "试听它", "播放它", "播放她"):
            with self.subTest(text=text):
                self.assertIsNone(route_intent(normalize_track_reference_pronouns(text)))
        # T14-F-R2: the stop-object form IS routed -- the 停止试听它 table
        # entry (deliberate) owns it, and every 他/她 feedstock reaches it
        # through the same normalization.
        for text in ("停止试听它", "停止试听他", "停止试听她"):
            with self.subTest(text=text):
                self.assertEqual(
                    route_intent(normalize_track_reference_pronouns(text)),
                    "stop_preview",
                )

    def test_play_and_preview_family_classification_is_pronoun_agnostic(self) -> None:
        # T14-E interplay: normalized play-pronoun variants stay inside the
        # explicit-play family (the door keeps covering them); 试听 variants
        # stay out of it.
        self.assertTrue(is_explicit_play_intent(normalize_track_reference_pronouns("播放他")))
        self.assertTrue(is_explicit_play_intent(normalize_track_reference_pronouns("播放她")))
        self.assertFalse(is_explicit_play_intent(normalize_track_reference_pronouns("试听她")))


class T14FR2PronounBindingTest(unittest.TestCase):
    """P19-T14-F-R2: deterministic pronoun binding -- the pure pieces.

    T14-F proved insufficient: the 他/她 → 它 rewrite ran, but the downstream
    referent resolution stayed model-side (live failure: the model wandered
    into recommendation generation instead of previewing the current track).
    R2 binds the pronoun BEFORE any provider round. This class tests the
    side-effect-free primitives the three-session fast paths rely on: the
    closed pronoun-reference classifier, the single authoritative referent
    extractor (the service's channel register), and the continuous-session
    gate. No I/O, no routing surprises.
    """

    TRACK_A = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    def test_classifier_maps_every_spelling_to_the_same_kind(self) -> None:
        # Acceptance 1's classification core: all three spellings of each
        # verb family are ONE reference -- 试听它/他/她 at every seam and
        # 播放它/他/她 at every seam resolve to the same kind, so the same
        # fast path executes for all of them.
        kinds = (
            ("试听它", "preview"),
            ("试听他", "preview"),
            ("试听她", "preview"),
            ("播放它", "play"),
            ("播放他", "play"),
            ("播放她", "play"),
            ("试听她。", "preview"),
            (" 试听她 ", "preview"),
            ("播放他！", "play"),
            ("播放她…", "play"),
        )
        for text, kind in kinds:
            with self.subTest(text=text):
                self.assertEqual(pronoun_track_reference(text), kind)

    def test_classifier_rejects_every_other_form(self) -> None:
        # Closed whole-line forms only: qualifiers, extra objects, named
        # tracks, stop forms, and fragments must NOT classify -- the fast
        # path must never hijack a line the provider owns.
        for text in (
            "试听",
            "播放",
            "试听它的歌",
            "试听他的歌",
            "播放她吗",
            "停止试听它",
            "停止试听她",
            "试听一下它",
            "别播放他",
            "请播放它",
            "试听 Hanataba",
            "听它",
            "",
            "  ",
        ):
            with self.subTest(text=text):
                self.assertIsNone(pronoun_track_reference(text))

    def test_classifier_resists_non_string_input(self) -> None:
        for value in (None, 42, ["试听它"], b"\xe8\xaf\x95\xe5\x90\xac"):
            with self.subTest(value=value):
                self.assertIsNone(pronoun_track_reference(value))

    def test_target_extractor_uses_only_the_channel_register(self) -> None:
        # Legacy channel fallback (F-R2, kept for F-R4): with NO referent in
        # the payload, channel.canonical_id with a library/preview state is
        # the referent source -- the service's register of its last
        # playback/preview action. Absence, ambiguity, and malformed shapes
        # all yield None (never guessed, never invented).
        library = {"channel": {"state": "library", "canonical_id": self.TRACK_A}}
        preview = {"channel": {"state": "preview", "canonical_id": self.TRACK_A}}
        self.assertEqual(pronoun_track_reference_target(library), self.TRACK_A)
        self.assertEqual(pronoun_track_reference_target(preview), self.TRACK_A)
        for payload in (
            {"channel": {"state": "none", "canonical_id": self.TRACK_A}},
            {"channel": {"state": "unknown", "canonical_id": self.TRACK_A}},
            {"channel": {"state": "preview", "canonical_id": None}},
            {"channel": {"state": "preview"}},
            {"channel": {"state": "preview", "canonical_id": ""}},
            {"channel": {}},
            {"channel": "preview"},
            {"preview_sounding": True},
            {},
            None,
            "channel",
            42,
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(pronoun_track_reference_target(payload))

    def test_target_extractor_prefers_the_referent_over_the_channel(self) -> None:
        # P19-T14-F-R4: referent_canonical_id is the conversational target and
        # wins outright -- even while a NEWER action channel exists (the
        # A→B→stop→pronoun proof: channel is an action log; the referent is
        # what the user is talking about).
        trk_b = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
        self.assertEqual(
            pronoun_track_reference_target(
                {
                    "referent_canonical_id": trk_b,
                    "channel": {"state": "preview", "canonical_id": self.TRACK_A},
                }
            ),
            trk_b,
        )
        self.assertEqual(
            pronoun_track_reference_target(
                {
                    "referent_canonical_id": trk_b,
                    "channel": {"state": "none", "canonical_id": None},
                }
            ),
            trk_b,
        )

    def test_target_extractor_referent_survives_a_cleared_channel(self) -> None:
        # The Owner regression: 试听 Hanataba → stop preview → 试听他. The
        # observation after stop has channel=none (cleared by constitution)
        # while the referent still names the explicitly targeted track.
        payload = {
            "referent_canonical_id": self.TRACK_A,
            "channel": {"state": "none", "canonical_id": None},
            "preview_sounding": False,
        }
        self.assertEqual(pronoun_track_reference_target(payload), self.TRACK_A)

    def test_target_extractor_rejects_malformed_referent_values(self) -> None:
        # A malformed/empty referent is absence, not a guess: the payload
        # falls through to the channel fallback (or None when that is also
        # absent). Never invented.
        for bad in (None, "", 42, ["trk_x"], {"id": "trk_x"}):
            with self.subTest(bad=bad):
                self.assertIsNone(
                    pronoun_track_reference_target(
                        {
                            "referent_canonical_id": bad,
                            "channel": {"state": "none", "canonical_id": None},
                        }
                    )
                )
        # And an honest channel fallback still works after a malformed referent.
        self.assertEqual(
            pronoun_track_reference_target(
                {
                    "referent_canonical_id": "",
                    "channel": {"state": "library", "canonical_id": self.TRACK_A},
                }
            ),
            self.TRACK_A,
        )

    def test_session_gate_admits_only_a_running_continuous_session(self) -> None:
        # A literally RUNNING continuous preview session owns its multi-track
        # surface (the referent is contested there; callers defer to the
        # provider's session rules). Any other session state -- including a
        # completed one and the absence of a session -- is not contested.
        self.assertTrue(
            continuous_preview_session_running({"session": {"state": "running"}})
        )
        for payload in (
            {"session": {"state": "completed"}},
            {"session": {"state": "cancelled"}},
            {"session": None},
            {"session": "running"},
            {},
            None,
            "x",
            42,
        ):
            with self.subTest(payload=payload):
                self.assertFalse(continuous_preview_session_running(payload))

    def test_stop_object_form_has_a_table_route_and_no_pronoun_kind(self) -> None:
        # 停止试听它 is a routed stop (deliberate T14-F-R2 table entry), NOT
        # a pronoun track reference: the stop family owns it; the preview/
        # play binding path must stay out of its way.
        self.assertEqual(route_intent("停止试听它"), "stop_preview")
        self.assertEqual(
            route_intent(normalize_track_reference_pronouns("停止试听她")),
            "stop_preview",
        )
        self.assertIsNone(pronoun_track_reference("停止试听它"))


class PlainChatClassificationTest(unittest.TestCase):
    """S1: the closed plain-chat classifier that gates the loop's zero-tool run.

    One-sided by design -- a false positive STRIPS tools from a request that
    may need them, so only the exact closed set classifies; every near-miss,
    mixed sentence, music line and consent form stays False (full tools).
    """

    def test_closed_greeting_thanks_farewell_forms_classify_true(self) -> None:
        for text in (
            "你好",
            "你好！",
            "你好。",
            "您好",
            "你好呀",
            "早上好",
            "晚上好",
            "晚安",
            "在吗",
            "嗨",
            "hi",
            "Hello",
            "HEY",
            "谢谢",
            "谢谢你",
            "谢谢啦",
            "多谢",
            "感谢",
            "辛苦了",
            "thanks",
            "Thank you",
            "再见",
            "拜拜",
            "bye",
            "你是谁",
            "今天天气怎么样",
            "讲个笑话",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_plain_chat(text))

    def test_trailing_punctuation_and_whitespace_are_tolerated(self) -> None:
        self.assertTrue(is_plain_chat("  你好！"))
        self.assertTrue(is_plain_chat("谢谢。" + " "))
        self.assertFalse(is_plain_chat("你好，谢谢"))  # two forms glued: not exact

    def test_music_lines_never_classify(self) -> None:
        for text in (
            "推荐几首歌",
            "播放晴天",
            "暂停",
            "继续",
            "下一首",
            "试听它",
            "换一首",
            "这首歌是什么？",
            "我喜欢这首歌",
            "谢谢你的推荐",
            "你唱首歌",
            "有什么好听的歌吗",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_plain_chat(text))

    def test_consent_family_never_classifies(self) -> None:
        # A bare 好的/可以/行/嗯 often answers a pending offer (要试听吗? -> 好的)
        # and must keep every tool so the action can actually execute.
        for text in ("好的", "好", "行", "行吧", "可以", "嗯", "收到", "没问题", "对"):
            with self.subTest(text=text):
                self.assertFalse(is_plain_chat(text))

    def test_mixed_and_unlisted_lines_stay_false_fail_safe(self) -> None:
        for text in (
            "你好帮我推荐几首歌",  # greeting glued to a request
            "你好，帮我推荐几首歌",
            "最近好吗我想听歌",
            "天气不错啊",  # unlisted chitchat: keep tools
            "今天过的怎么样",
            "周末去哪玩",
            "哈哈哈哈",
            "你能做什么",  # capability questions ride get_agent_capabilities
            "",
            " ",
            None,
            42,
        ):
            with self.subTest(text=text):
                self.assertFalse(is_plain_chat(text))

    def test_plain_chat_forms_never_overlap_the_routing_table(self) -> None:
        # The zero-tool gate must never claim a line the fast path would
        # execute (they never reach the loop, but the classifier stays honest).
        for text in _all_routed_spellings():
            self.assertFalse(is_plain_chat(text))


class S3TaskSurfaceClassificationTest(unittest.TestCase):
    """S3: per-task surface classifiers and their fail-safe boundaries.

    P20 additionally projects recommendation/fresh classification through
    ``expects_recommendation_batch`` so provider routing and the web reply door
    agree on whether this turn expected a new recommendation batch.
    """

    def test_turn_plan_required_semantic_matrix_has_one_primary(self) -> None:
        cases = {
            "推荐几首歌": TurnPrimarySemantic.RECOMMENDATION,
            "推荐 Yorushika 的歌": TurnPrimarySemantic.RECOMMENDATION,
            "推荐类似《Spring Thief》的歌": TurnPrimarySemantic.RECOMMENDATION,
            "我喜欢 IU，给我推荐几首歌": TurnPrimarySemantic.RECOMMENDATION,
            "我喜欢 IU，推荐几首适合晚上听的歌": TurnPrimarySemantic.RECOMMENDATION,
            "我喜欢 Yorushika 的歌": TurnPrimarySemantic.PREFERENCE_STATEMENT,
            "我喜欢这首歌": TurnPrimarySemantic.FEEDBACK,
            "我不喜欢这首歌": TurnPrimarySemantic.FEEDBACK,
            "我喜欢当前正在播放的这首歌": TurnPrimarySemantic.FEEDBACK,
            "我不喜欢当前正在播放的这首歌": TurnPrimarySemantic.FEEDBACK,
            "播放第二首": TurnPrimarySemantic.PLAYBACK_ACTION,
            "试听第四首": TurnPrimarySemantic.PLAYBACK_ACTION,
            "随便播放一首": TurnPrimarySemantic.PLAYBACK_ACTION,
            "再换一首": TurnPrimarySemantic.PLAYBACK_ACTION,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_turn_plan(text).primary, expected)

    def test_turn_plan_reuses_existing_recommendation_semantics(self) -> None:
        text = "推荐类似《Spring Thief》的歌"
        existing = resolve_recommendation_turn_semantics(text)
        plan = resolve_turn_plan(text)

        self.assertEqual(plan.expected_result, TurnExpectedResult.RECOMMENDATION_BATCH)
        self.assertEqual(plan.task_surface, TurnTaskSurface.RECOMMENDATION)
        self.assertEqual(plan.recommendation, existing)
        self.assertEqual(plan.recommendation.mode, "similarity_seed")
        # Slice 1 deliberately preserves the current unnormalized seed text.
        self.assertEqual(plan.recommendation.target, "《Spring Thief》")

    def test_short_recommendation_forms_have_explicit_turn_semantics(self) -> None:
        for text in (
            "推荐音乐",
            "推荐几首歌",
            "给我推荐",
            "给我推荐几首歌",
            "帮我推荐",
            "帮我推荐几首歌",
            "帮忙推荐几首歌",
        ):
            with self.subTest(text=text):
                generic = resolve_turn_plan(text)
                self.assertEqual(
                    generic.primary, TurnPrimarySemantic.RECOMMENDATION
                )
                self.assertEqual(
                    generic.expected_result, TurnExpectedResult.RECOMMENDATION_BATCH
                )
                self.assertEqual(generic.recommendation.mode, "generic")
                self.assertEqual(generic.recommendation.requested_count, 5)
                self.assertIsNone(generic.recommendation.target)
                self.assertIsNone(generic.recommendation.seed_source)

        similar = resolve_turn_plan("找类似这首的")
        self.assertEqual(similar.primary, TurnPrimarySemantic.RECOMMENDATION)
        self.assertEqual(
            similar.expected_result, TurnExpectedResult.RECOMMENDATION_BATCH
        )
        self.assertEqual(similar.recommendation.mode, "similarity_seed")
        self.assertEqual(similar.recommendation.requested_count, 5)
        self.assertIsNone(similar.recommendation.target)
        self.assertEqual(similar.recommendation.target_kind, "track")
        self.assertEqual(similar.recommendation.seed_source, "current_track")

        # Quick-action and manual entry have no separate semantic API: the
        # same exact text necessarily yields the same immutable plan.
        chip_text = "推荐音乐"
        manually_typed_text = "".join(("推荐", "音乐"))
        self.assertEqual(
            resolve_turn_plan(chip_text), resolve_turn_plan(manually_typed_text)
        )

    def test_library_absence_wording_routes_to_fresh_discovery_not_artist(self) -> None:
        text = "推荐几首我没有的歌"

        self.assertTrue(is_fresh_discovery_intent(text))
        self.assertIsNone(resolve_recommendation_turn_semantics(text))
        plan = resolve_turn_plan(text)
        self.assertEqual(plan.primary, TurnPrimarySemantic.RECOMMENDATION)
        self.assertEqual(plan.task_surface, TurnTaskSurface.FRESH_DISCOVERY)
        self.assertEqual(
            plan.expected_result, TurnExpectedResult.RECOMMENDATION_BATCH
        )
        self.assertTrue(plan.fresh_discovery)
        self.assertIsNone(plan.recommendation)

    def test_current_track_similarity_followups_preserve_the_same_turn_semantics(
        self,
    ) -> None:
        baseline = resolve_turn_plan("找类似这首的")
        for text in (
            "再找一些类似这首的",
            "再找几首类似这首的",
            "再来一些类似这首的",
            "再推荐几首类似这首的",
            "再找一些类似这首的。",
        ):
            with self.subTest(text=text):
                plan = resolve_turn_plan(text)
                self.assertEqual(plan.primary, TurnPrimarySemantic.RECOMMENDATION)
                self.assertEqual(
                    plan.expected_result,
                    TurnExpectedResult.RECOMMENDATION_BATCH,
                )
                self.assertEqual(plan.task_surface, TurnTaskSurface.RECOMMENDATION)
                self.assertEqual(plan.recommendation, baseline.recommendation)
                self.assertEqual(plan.recommendation.mode, "similarity_seed")
                self.assertEqual(plan.recommendation.target_kind, "track")
                self.assertEqual(plan.recommendation.seed_source, "current_track")

    def test_generic_continuation_is_not_upgraded_to_current_track_similarity(
        self,
    ) -> None:
        plan = resolve_turn_plan("再推荐几首")
        self.assertEqual(plan.primary, TurnPrimarySemantic.UNKNOWN)
        self.assertEqual(plan.task_surface, TurnTaskSurface.FULL)
        self.assertIsNone(plan.recommendation)

        generic = resolve_turn_plan("推荐音乐")
        self.assertEqual(generic.recommendation.mode, "generic")
        mood = resolve_turn_plan("换个心情")
        self.assertEqual(mood.primary, TurnPrimarySemantic.RECOMMENDATION)
        self.assertIsNone(mood.recommendation)

    def test_direction_control_keeps_its_existing_compatibility_shape(self) -> None:
        plan = resolve_turn_plan("换个心情")
        self.assertEqual(plan.primary, TurnPrimarySemantic.RECOMMENDATION)
        self.assertEqual(plan.task_surface, TurnTaskSurface.RECOMMENDATION)
        self.assertEqual(plan.expected_result, TurnExpectedResult.RECOMMENDATION_BATCH)
        self.assertIsNone(plan.recommendation)

    def test_turn_plan_preserves_preference_and_current_track_feedback_objects(self) -> None:
        preference_text = "我喜欢 Yorushika 的歌"
        preference = resolve_turn_plan(preference_text)
        self.assertEqual(
            preference.preference_statement,
            resolve_preference_statement_turn_semantics(preference_text),
        )
        self.assertIsNone(preference.recommendation)
        self.assertEqual(preference.expected_result, TurnExpectedResult.ACKNOWLEDGEMENT)

        feedback_text = "我不喜欢当前正在播放的这首歌"
        feedback = resolve_turn_plan(feedback_text)
        self.assertEqual(
            feedback.current_track_feedback,
            resolve_current_track_feedback_turn_semantics(feedback_text),
        )
        self.assertEqual(feedback.current_track_feedback.kind, "disliked")
        self.assertEqual(feedback.task_surface, TurnTaskSurface.FEEDBACK)

    def test_turn_plan_actions_are_abstract_and_do_not_select_identity_or_route(self) -> None:
        cases = {
            "播放第二首": ("play", "active_recommendation", "explicit_index", 2, False),
            "试听第四首": ("preview", "active_recommendation", "explicit_index", 4, False),
            "随便播放一首": (
                "play_or_preview",
                "active_recommendation",
                "agent_choose_one",
                None,
                True,
            ),
            "再换一首": (
                "play_or_preview",
                "active_recommendation",
                "choose_another",
                None,
                False,
            ),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                plan = resolve_turn_plan(text)
                action = plan.playback_action
                self.assertIsNotNone(action)
                self.assertEqual(
                    (
                        action.kind,
                        action.source,
                        action.selection_mode,
                        action.explicit_index,
                        action.delegated,
                    ),
                    expected,
                )
                self.assertFalse(hasattr(action, "canonical_id"))
                self.assertFalse(hasattr(action, "route"))
                self.assertFalse(hasattr(plan, "executed"))
                self.assertFalse(hasattr(plan, "completed"))
        self.assertTrue(resolve_turn_plan("再换一首").delegated_action_authorized)
        self.assertFalse(resolve_turn_plan("换一首").delegated_action_authorized)

    def test_recommendation_request_covers_closed_set_and_direction_form(self) -> None:
        for text in ("推荐几首歌", "换一组", "推荐点日系的", "推荐点轻松的", "推荐点什么"):
            self.assertTrue(is_recommendation_request(text))
        self.assertTrue(is_recommendation_request("推荐几首适合晚上听的歌"))
        for text in ("推荐", "推荐一下"):
            self.assertFalse(is_recommendation_request(text))

    def test_unified_batch_expectation_follows_provider_request_semantics(self) -> None:
        self.assertTrue(expects_recommendation_batch("推荐点日系的"))
        self.assertTrue(expects_recommendation_batch("推荐几首歌"))
        self.assertFalse(expects_recommendation_batch("播放和推荐有什么区别？"))

    def test_natural_recommendation_requests_join_tool_routing(self) -> None:
        for text in (
            "推荐 Yorushika 的音乐",
            "给我推荐几首 Yorushika 的歌",
            "推荐一些适合晚上听的歌",
            "找几首和 Spring Thief 类似的歌",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_request(text))

    def test_explicit_recommendation_semantics_are_distinct(self) -> None:
        cases = {
            "推荐 Yorushika 的音乐": ("artist_constraint", "Yorushika", "artist", None),
            "给我推荐几首 Yorushika 的歌": ("artist_constraint", "Yorushika", "artist", None),
            "推荐类似 Yorushika 的音乐": ("similarity_seed", "Yorushika", None, None),
            "找几首和 Spring Thief 类似的歌": ("similarity_seed", "Spring Thief", None, None),
            "我喜欢 Yorushika，给我推荐几首歌": ("preference_seed", "Yorushika", None, None),
            "推荐几首适合晚上听的 Yorushika 的歌": (
                "artist_constraint", "Yorushika", "artist", "evening"
            ),
            "我喜欢 Yorushika，推荐几首适合晚上听的歌": (
                "preference_seed", "Yorushika", None, "evening"
            ),
            "推荐周杰伦的音乐": ("artist_constraint", "周杰伦", "artist", None),
            "推荐类似 Lamp 的音乐": ("similarity_seed", "Lamp", None, None),
            "我喜欢 IU，给我推荐几首歌": ("preference_seed", "IU", None, None),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                value = resolve_recommendation_turn_semantics(text)
                self.assertIsNotNone(value)
                self.assertEqual(
                    (value.mode, value.target, value.target_kind, value.scene), expected
                )
        self.assertEqual(
            recommendation_presentation_label("推荐类似 Yorushika 的音乐"),
            "类似 Yorushika",
        )
        self.assertEqual(
            recommendation_presentation_label("推荐几首适合晚上听的 Yorushika 的歌"),
            "今晚推荐",
        )

    def test_fresh_discovery_closed_set(self) -> None:
        for text in ("找些新的", "找点新的", "推荐没听过的", "找没听过的", "找库外歌曲"):
            self.assertTrue(is_fresh_discovery_intent(text))
        for text in ("找点新歌", "找新歌", "库外新歌", "再来点新的", "找些新歌听听"):
            self.assertFalse(is_fresh_discovery_intent(text))

    def test_collection_preference_statement_is_distinct_from_feedback_and_recommendation(self) -> None:
        for text, polarity, target in (
            ("我喜欢 Yorushika 的歌", "positive", "Yorushika"),
            ("我不喜欢 IU 的音乐", "negative", "IU"),
            ("我讨厌 Artist Alpha 的歌曲", "negative", "Artist Alpha"),
        ):
            with self.subTest(text=text):
                semantics = resolve_preference_statement_turn_semantics(text)
                self.assertIsNotNone(semantics)
                self.assertEqual(semantics.polarity, polarity)
                self.assertEqual(semantics.target, target)
                self.assertFalse(is_feedback_intent(text))
                self.assertFalse(is_recommendation_request(text))
                self.assertFalse(expects_recommendation_batch(text))

        self.assertIsNone(
            resolve_preference_statement_turn_semantics("我喜欢这首歌")
        )
        self.assertTrue(is_feedback_intent("我喜欢这首歌"))
        self.assertIsNone(
            resolve_preference_statement_turn_semantics("推荐 Yorushika 的歌")
        )
        self.assertTrue(is_recommendation_request("推荐 Yorushika 的歌"))
        for recommendation_text in (
            "我喜欢 IU，给我推荐几首歌",
            "我喜欢 IU，推荐几首适合晚上听的歌",
        ):
            self.assertIsNone(
                resolve_preference_statement_turn_semantics(recommendation_text)
            )
            self.assertTrue(is_recommendation_request(recommendation_text))

    def test_feedback_verdict_forms(self) -> None:
        for text in (
            "我喜欢第二首", "我不喜欢这首", "讨厌这首", "喜欢第三首",
            "我喜欢这首歌", "我不喜欢这首歌", "讨厌那首歌曲",
            "这首不错", "这首好听", "这首一般", "这个方向不错", "这方向好",
        ):
            self.assertTrue(is_feedback_intent(text))
        for text in (
            "喜欢", "不喜欢", "喜欢夜曲", "我很喜欢第二首",
            "第二首好听", "收藏这首", "这个方向", "方向不错",
        ):
            self.assertFalse(is_feedback_intent(text))

    def test_current_track_feedback_short_and_long_positive_and_negative_forms(self) -> None:
        cases = {
            "我喜欢这首歌": "liked",
            "我不喜欢这首歌": "disliked",
            "我喜欢当前正在播放的这首歌": "liked",
            "我不喜欢当前正在播放的这首歌": "disliked",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                semantics = resolve_current_track_feedback_turn_semantics(text)
                self.assertIsNotNone(semantics)
                self.assertEqual(semantics.kind, kind)
                self.assertTrue(is_feedback_intent(text))

        self.assertIsNone(
            resolve_current_track_feedback_turn_semantics("我喜欢 Yorushika 的歌")
        )
        self.assertFalse(is_feedback_intent("我喜欢 Yorushika 的歌"))

    def test_preview_forms(self) -> None:
        for text in (
            "试听第二首", "试听夜曲", "试听这首", "试听上一首", "试听一下",
            "都试听一遍", "把这一批都试听一遍", "停止试听", "暂停试听",
            "停", "停止", "别放了", "关掉",
        ):
            self.assertTrue(is_preview_intent(text))
        for text in (
            "试听", "试听吗", "试听什么", "试听哪首好", "试听是什么意思",
            "试听怎么弄", "随便播放一首", "播放第二首", "试听后再播放",
        ):
            self.assertFalse(is_preview_intent(text))

    def test_s3_classifiers_never_claim_routed_commands(self) -> None:
        # The fast path executes these before the loop runs; the surface
        # classifiers must not fire on them either way (except the stop
        # residuals, which are handed BACK to the loop by the routing table).
        for text in ("暂停", "继续播放", "下一首", "上一首", "换一首", "播放它"):
            self.assertFalse(is_feedback_intent(text))
            self.assertFalse(is_fresh_discovery_intent(text))
            self.assertFalse(is_preview_intent(text))
            self.assertFalse(is_recommendation_request(text))


class P20Fix02ClassificationCoverageTest(unittest.TestCase):
    """P20-Fix02: natural-phrase classification coverage (quality audit Fix 02).

    Freezes the after-state of the expanded classifiers: the recommendation
    surface extensions (A), the read-only explanation family (B -- must never
    be classified as a request for a NEW batch), the fresh/catalog additions
    (C), and every pinned negative staying on the full set. The door set
    ``_RECOMMENDATION_INTENT_FORMS`` is frozen by T14-B and stays out of
    every expansion.
    """

    RECOMMENDATION_SURFACE_FORMS = (
        "最近给我推荐几首歌",
        "帮我推荐几首歌",
        "来几首推荐",
        "再推荐一批",
        "推荐点歌",
    )

    RECOMMENDATION_RECOMMEND_FORMS = (
        "再来一批，换个方向",
        "再来一批换个方向",
        "再来一批,换个方向",
        "换一批，换个方向",
        "换一组换个方向",
        "再推荐一批，换方向",
    )

    EXPLANATION_FORMS = (
        "为什么推荐这些",
        "为什么给我推荐这些",
        "为什么这些适合我",
        "这几首为什么适合我",
        "这批为什么适合我",
        "为什么这一批适合我",
        "为什么这批适合我",
        "这批推荐为什么适合我",
        "推荐理由是什么",
        "为什么推荐这几首",
    )

    FRESH_FORMS = (
        "推荐一些新歌",
        "推荐一些没听过的新歌",
        "推荐一些我没听过的新歌",
        "找点我没听过的歌",
        "给我找些新歌",
        "推荐点库外的歌",
        "找一些新的",
        "来点没听过的",
    )

    def test_recommendation_surface_extensions_classify_request(self) -> None:
        for text in self.RECOMMENDATION_SURFACE_FORMS + self.RECOMMENDATION_RECOMMEND_FORMS:
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_request(text))

    def test_recommendation_surface_extensions_share_the_batch_expectation(self) -> None:
        for text in self.RECOMMENDATION_SURFACE_FORMS + self.RECOMMENDATION_RECOMMEND_FORMS:
            with self.subTest(text=text):
                self.assertTrue(expects_recommendation_batch(text))

    def test_natural_and_fresh_requests_share_the_batch_expectation(self) -> None:
        for text in (
            "推荐 Yorushika 的音乐",
            "推荐类似 Yorushika 的音乐",
            "我喜欢 IU，给我推荐几首歌",
            "找点我没听过的歌",
            "推荐一些我没听过的新歌",
        ):
            with self.subTest(text=text):
                self.assertTrue(expects_recommendation_batch(text))

    def test_recommend_re_pattern_refuses_loose_tails(self) -> None:
        for text in ("换一批是什么", "再来一批好听的", "再来一批，换一批", "换一"):
            with self.subTest(text=text):
                self.assertFalse(is_recommendation_request(text))

    def test_explanation_forms_classify_explanation(self) -> None:
        for text in self.EXPLANATION_FORMS:
            with self.subTest(text=text):
                self.assertTrue(is_recommendation_explanation_intent(text))

    def test_explanation_forms_never_classify_new_batch_or_fresh(self) -> None:
        # The structural guarantee: a why-question can never be offered the
        # generation tools (recommendation request) or the catalog path.
        for text in self.EXPLANATION_FORMS:
            with self.subTest(text=text):
                self.assertFalse(is_recommendation_request(text))
                self.assertFalse(is_fresh_discovery_intent(text))

    def test_explanation_negative_phrasings_stay_out(self) -> None:
        for text in (
            "推荐系统是怎么工作的",
            "你会推荐吗",
            "为什么推荐算法这么慢",
            "播放和推荐有什么区别",
            "不要推荐了",
            "我不想听推荐",
            "刚才推荐出错了",
            "为什么推荐",
            "为什么",
            "推荐理由",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_recommendation_explanation_intent(text))

    def test_fresh_extensions_classify_fresh(self) -> None:
        for text in self.FRESH_FORMS:
            with self.subTest(text=text):
                self.assertTrue(is_fresh_discovery_intent(text))

    def test_fresh_extensions_keep_the_pinned_near_misses_excluded(self) -> None:
        # The pre-Fix02 pinned negatives stand: real fresh intent must name
        # 没听过 / 库外 / the 推荐-prefixed 新歌 pairing.
        for text in ("找点新歌", "找新歌", "再来点新的", "找些新歌听听", "库外新歌"):
            with self.subTest(text=text):
                self.assertFalse(is_fresh_discovery_intent(text))

    def test_overlapping_fresh_form_keeps_door_membership(self) -> None:
        # 推荐一些新歌 stays recommendation-shaped for the T14-B reply door
        # while the S3/S4 selectors route it on the fresh/catalog path first.
        self.assertTrue(is_recommendation_intent("推荐一些新歌"))
        self.assertTrue(is_fresh_discovery_intent("推荐一些新歌"))

    def test_feature_comparison_question_is_no_longer_play_intent(self) -> None:
        # P20-Fix02: a 播放-prefixed feature-comparison question must keep the
        # full surface -- it neither narrows to the play family nor classifies
        # as any recommendation family.
        self.assertFalse(is_explicit_play_intent("播放和推荐有什么区别？"))
        self.assertFalse(is_recommendation_request("播放和推荐有什么区别？"))
        self.assertFalse(is_recommendation_explanation_intent("播放和推荐有什么区别？"))

    def test_mixed_and_unknown_phrasings_stay_unclaimed(self) -> None:
        for text in (
            "推荐几首然后播放第二首",
            "推荐还是播放？",
            "推荐完帮我直接试听",
            "我到底该推荐还是继续播放？",
            "推荐",
            "推荐一下",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_recommendation_request(text))
                self.assertFalse(is_recommendation_explanation_intent(text))

    def test_explanation_never_claims_routed_commands(self) -> None:
        for text in ("暂停", "继续播放", "下一首", "上一首", "换一首", "播放它"):
            self.assertFalse(is_recommendation_explanation_intent(text))

    def test_trailing_punctuation_tolerated_for_new_families(self) -> None:
        self.assertTrue(is_recommendation_explanation_intent("为什么推荐这些？"))
        self.assertTrue(is_recommendation_request("再来一批，换个方向。"))
        self.assertTrue(is_fresh_discovery_intent(" 推荐一些新歌？"))


def _all_routed_spellings() -> list[str]:
    """Every routed command spelling the routing table admits (live forms)."""
    return [
        "暂停", "停止试听", "停止试听它", "暂停试听", "继续播放", "继续",
        "下一首", "下一首试听", "上一首", "现在在播放什么", "现在播放的是什么",
        "当前在播放什么", "当前播放什么", "现在是什么歌", "播放一首正式歌曲",
        "放一首正式歌曲", "播放一首正式的歌", "来一首正式歌曲", "正式播放一首",
        "stop", "停止", "换一首", "播放", "播放它",
    ]



class P22S21NamedPlayTargetProjectionTest(unittest.TestCase):
    def test_named_play_projects_user_target_text_only(self) -> None:
        self.assertEqual(
            resolve_turn_plan("播放地球最后一夜").playback_action.target_text,
            "地球最后一夜",
        )
        self.assertEqual(
            resolve_turn_plan("开始播放 Wendy").playback_action.target_text,
            "Wendy",
        )

    def test_closed_playback_forms_do_not_invent_named_target_text(self) -> None:
        for text in ("播放", "播放这首", "播放第二首", "播放一首正式歌曲"):
            with self.subTest(text=text):
                action = resolve_turn_plan(text).playback_action
                self.assertIsNotNone(action)
                self.assertIsNone(action.target_text)

if __name__ == "__main__":
    unittest.main()
