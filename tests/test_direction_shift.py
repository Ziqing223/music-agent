"""P20 Quality Fix 05: pure direction-shift semantics (direction_shift module).

The closed recognition set, the explicit direction-word mapping, the request
classifier and the deterministic replacement picker -- all pure, provider-
free decisions. Every near-miss refuses: a shift only ever fires for the
exact closed set, an explicit direction only for the mapped table or a
verified canonical genre key, and the picker never fabricates a direction.
"""

from __future__ import annotations

import unittest

from music_agent.direction_shift import (
    DIRECTION_WORD_GENRES,
    NO_ALTERNATIVE_DIRECTION_REPLY,
    classify_direction_request,
    explicit_direction_note,
    is_direction_shift_request,
    map_direction_word,
    pick_shifted_direction,
    shifted_direction_note,
)


class DirectionShiftPhrasesTest(unittest.TestCase):
    def test_all_closed_shift_forms_are_recognized(self) -> None:
        forms = [
            # bare shift lines
            "换个方向", "换方向", "换一种风格", "来点不一样的",
            # verb + tail, run together and comma-separated
            "再来一批换个方向", "再来一批，换个方向",
            "再来一批换方向", "再来一批，换方向",
            "再推荐一批换个方向", "再推荐一批，换个方向",
            "再推荐一批换方向", "再推荐一批，换方向",
            "换一批换个方向", "换一批，换个方向",
            "换一批换方向", "换一批，换方向",
            "换一组换个方向", "换一组，换个方向",
            "换一组换方向", "换一组，换方向",
        ]
        self.assertEqual(len(forms), 20)
        for form in forms:
            with self.subTest(form=form):
                self.assertTrue(is_direction_shift_request(form))

    def test_trailing_punctuation_and_casefold(self) -> None:
        self.assertTrue(is_direction_shift_request("再来一批，换个方向。"))
        self.assertTrue(is_direction_shift_request(" 换个方向? "))
        self.assertTrue(is_direction_shift_request("来点不一样的！"))
        self.assertTrue(is_direction_shift_request("再来一批换方向…"))

    def test_near_misses_refuse(self) -> None:
        refusals = [
            "再来一批",  # plain re-recommend: the critical regression surface
            "再来一批好听的",
            "再来一批新的",
            "换一个方向",
            "换个方向吧",
            "来点不一样的歌",
            "换种风格",
            "换个风格",
            "好听的换个方向",
            "换个方向好听的",
            "再来一批，换一种风格",  # not a closed form
            "再推荐一批",
            "换一批",
            "再来一批  换 方向",
            "direction",
            "change direction",
            "",
            "   ",
        ]
        for text in refusals:
            with self.subTest(text=text):
                self.assertFalse(is_direction_shift_request(text))

    def test_non_string_refuses(self) -> None:
        for value in (None, 42, ["换个方向"], {"text": "换个方向"}):
            self.assertFalse(is_direction_shift_request(value))


class DirectionWordMappingTest(unittest.TestCase):
    def test_alias_table(self) -> None:
        self.assertEqual(DIRECTION_WORD_GENRES["日系"], "J-Pop")
        for word, genre in (
            ("日系", "J-Pop"),
            ("日语", "J-Pop"),
            ("日文", "J-Pop"),
            ("摇滚", "Rock"),
            ("摇滚乐", "Rock"),
        ):
            with self.subTest(word=word):
                self.assertEqual(map_direction_word(word), genre)

    def test_verified_genre_key_passthrough(self) -> None:
        for key in ("J-Pop", "Rock", "Mandopop", "K-Pop", "Pop", "Alternative"):
            with self.subTest(key=key):
                self.assertEqual(map_direction_word(key), key)

    def test_passthrough_is_case_insensitive(self) -> None:
        # The classifier casefolds the whole line before mapping; verified
        # keys fold back to their canonical spelling.
        self.assertEqual(map_direction_word("j-pop"), "J-Pop")
        self.assertEqual(map_direction_word("ROCK"), "Rock")
        self.assertEqual(map_direction_word(" POP "), "Pop")

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(map_direction_word(" 日系 "), "J-Pop")

    def test_unknown_words_refuse(self) -> None:
        for word in (
            "轻盈", "平缓", "电子", "古典", "民谣", "日语歌", "摇滚的",
            "JPOP", " JPOP ", "日系风格",
        ):
            with self.subTest(word=word):
                self.assertIsNone(map_direction_word(word))

    def test_non_string_refuses(self) -> None:
        self.assertIsNone(map_direction_word(None))
        self.assertIsNone(map_direction_word(42))
        self.assertIsNone(map_direction_word(""))


class DirectionRequestClassificationTest(unittest.TestCase):
    def test_closed_shift_forms_classify_as_shift(self) -> None:
        for text in ("换个方向", "再来一批换个方向", "换一种风格", "来点不一样的"):
            request = classify_direction_request(text)
            self.assertIsNotNone(request, text)
            self.assertEqual(request.kind, "shift")
            self.assertIsNone(request.genre)

    def test_shift_set_wins_over_explicit_prefix(self) -> None:
        # 来点不一样的 looks like a 来点<word> form; the closed shift set first.
        request = classify_direction_request("来点不一样的")
        self.assertEqual(request.kind, "shift")

    def test_explicit_mapped_words(self) -> None:
        for text, genre in (
            ("换成日系", "J-Pop"),
            ("换成日语", "J-Pop"),
            ("换成摇滚", "Rock"),
            ("来点摇滚乐", "Rock"),
            ("换成J-Pop", "J-Pop"),
            ("来点 K-Pop", "K-Pop"),
            ("换成日系。", "J-Pop"),
        ):
            with self.subTest(text=text):
                request = classify_direction_request(text)
                self.assertIsNotNone(request, text)
                self.assertEqual(request.kind, "explicit")
                self.assertEqual(request.genre, genre)

    def test_explicit_unmapped_words_remit_to_provider(self) -> None:
        # Unmapped direction words stay the provider loop's surface (None).
        for text in ("换成平缓", "来点轻松的", "换成民谣", "来点什么", "换成"):
            with self.subTest(text=text):
                self.assertIsNone(classify_direction_request(text))

    def test_plain_forms_are_not_direction_requests(self) -> None:
        # §十一(9): the plain re-recommend surface is untouched.
        for text in ("再来一批", "再推荐一批", "换一批", "换一组", "推荐几首歌"):
            self.assertIsNone(classify_direction_request(text))
        self.assertIsNone(classify_direction_request(None))
        self.assertIsNone(classify_direction_request(""))


class ShiftPickerTest(unittest.TestCase):
    def test_single_direction_batch_picks_strongest_absent_positive(self) -> None:
        # A pure Mandopop batch: the strongest absent positive wins.
        decision = pick_shifted_direction({"Mandopop": 3}, ("J-Pop", "Rock"))
        self.assertEqual(decision.selected, "J-Pop")
        self.assertEqual(decision.excluded_genres, frozenset({"Mandopop"}))

    def test_absent_direction_beats_strongest_leftover_positive(self) -> None:
        # J-Pop appears once in the batch and is the strongest positive, but an
        # ABSENT direction is a truer shift -- Rock wins by policy.
        decision = pick_shifted_direction(
            {"Mandopop": 3, "J-Pop": 1},
            ("J-Pop", "Rock", "K-Pop"),
        )
        self.assertEqual(decision.selected, "Rock")
        self.assertEqual(decision.excluded_genres, frozenset({"Mandopop"}))

    def test_batch_main_direction_is_excluded_even_when_strongest(self) -> None:
        # Eclipse: the user's strongest positive IS the previous direction;
        # the next real positive wins.
        decision = pick_shifted_direction(
            {"Mandopop": 3},
            ("Mandopop", "J-Pop"),
        )
        self.assertEqual(decision.selected, "J-Pop")
        self.assertEqual(decision.excluded_genres, frozenset({"Mandopop"}))

    def test_mixed_batch_excludes_all_main_share_directions(self) -> None:
        # Mandopop/J-Pop tie for the main share: both excluded; Rock selected.
        decision = pick_shifted_direction(
            {"Mandopop": 2, "J-Pop": 2, "Rock": 1},
            ("Mandopop", "J-Pop", "Rock"),
        )
        self.assertEqual(decision.selected, "Rock")
        self.assertEqual(decision.excluded_genres, frozenset({"Mandopop", "J-Pop"}))

    def test_absent_positive_outranks_present_below_main(self) -> None:
        # K-Pop is present below the main share; an absent positive wins.
        decision = pick_shifted_direction(
            {"Mandopop": 5, "K-Pop": 1},
            ("K-Pop", "Rock"),
        )
        self.assertEqual(decision.selected, "Rock")

    def test_present_below_main_used_when_nothing_absent(self) -> None:
        decision = pick_shifted_direction(
            {"Mandopop": 5, "Rock": 1},
            ("Rock",),
        )
        self.assertEqual(decision.selected, "Rock")

    def test_fail_honest_when_only_positive_is_the_main_direction(self) -> None:
        decision = pick_shifted_direction({"Mandopop": 3}, ("Mandopop",))
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.excluded_genres, frozenset({"Mandopop"}))

    def test_fail_honest_with_no_positive_directions(self) -> None:
        decision = pick_shifted_direction({"Mandopop": 3}, ())
        self.assertIsNone(decision.selected)
        decision = pick_shifted_direction({}, ())
        self.assertIsNone(decision.selected)

    def test_empty_batch_counts_pick_strongest_positive(self) -> None:
        decision = pick_shifted_direction({}, ("Rock", "J-Pop"))
        self.assertEqual(decision.selected, "Rock")
        self.assertEqual(decision.excluded_genres, frozenset())

    def test_garbage_counts_are_ignored(self) -> None:
        decision = pick_shifted_direction(
            {"Mandopop": 0, "J-Pop": -1, 12: 3}, ("J-Pop", "Rock")
        )
        self.assertEqual(decision.selected, "J-Pop")
        self.assertEqual(decision.excluded_genres, frozenset())

    def test_caller_order_is_preserved_for_ties(self) -> None:
        decision = pick_shifted_direction(
            {"Mandopop": 2},
            ("Rock", "J-Pop"),  # caller order wins; no internal re-ranking
        )
        self.assertEqual(decision.selected, "Rock")


class DirectionNotesTest(unittest.TestCase):
    def test_notes_carry_the_selected_direction(self) -> None:
        self.assertIn("J-Pop", shifted_direction_note("J-Pop"))
        self.assertIn("J-Pop", explicit_direction_note("J-Pop"))
        self.assertNotIn("Rock", shifted_direction_note("J-Pop"))

    def test_fail_honest_reply_names_the_supported_directions(self) -> None:
        self.assertIn("日系", NO_ALTERNATIVE_DIRECTION_REPLY)
        self.assertIn("摇滚", NO_ALTERNATIVE_DIRECTION_REPLY)
        self.assertIn("另一个方向", NO_ALTERNATIVE_DIRECTION_REPLY)


if __name__ == "__main__":
    unittest.main()