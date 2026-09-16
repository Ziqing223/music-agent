"""P20-Fix10: the deterministic recommendation presenter (mandate §八 mapping,
§九 internal-term ban, §十 no encyclopedia, §十五 purity, and the strict
fail-safe contract). Pure mapping tests -- no provider, no loop, no network.

The fixtures speak the post-Fix09 item contract exactly as the real generation
result items do (name / artist_name / playback / fresh_this_request /
evidence.mechanism + evidence.basis rows with kind / label / provenance) so the
copy assertions pin the same strings live UAT will surface.
"""

from __future__ import annotations

import unittest

from music_agent.recommendation_presenter import (
    render_recommendation_cue,
    render_recommendation_explanation_for_user,
    render_recommendation_for_user,
    render_recommendation_items,
)


def basis_row(kind: str, label: str, provenance: str) -> dict:
    return {"kind": kind, "label": label, "provenance": provenance}


def item(
    name: str,
    *,
    artist: str | None = None,
    basis: list[dict] | None = None,
    fresh: bool = False,
    route: str = "library",
) -> dict:
    entry: dict = {
        "name": name,
        "playback": {"route": route, "label": route},
        "evidence": {
            "mechanism": "直接偏好" if any(
                row["provenance"] == "直接" for row in (basis or [])
            ) else "推断偏好",
            "basis": basis if basis is not None else [],
        },
        "fresh_this_request": fresh,
    }
    if artist is not None:
        entry["artist_name"] = artist
    return entry


def payload(*items: dict) -> dict:
    return {
        "run_id": "rcm_11111111-1111-4111-8111-111111111111",
        "item_count": len(items),
        "items": list(items),
    }


ROCK = basis_row("genre", "Rock", "推断")
ROCK_DIRECT = basis_row("genre", "Rock", "直接")
TRACK_SELF = basis_row("track", "夜曲", "推断")
TRACK_SELF_DIRECT = basis_row("track", "夜曲", "直接")
ARTIST_INFERRED = basis_row("artist", "LOVE PSYCHEDELICO", "推断")
ANIME = basis_row("genre", "Anime", "推断")
MANDO = basis_row("genre", "Mandopop", "推断")
ALT = basis_row("genre", "Alternative", "推断")


def run_item(
    name: str,
    *,
    artist: str | None = None,
    basis: list[dict] | None = None,
    note: str | None = None,
) -> dict:
    """One batch-HISTORY item: the durable run reader shape (evidence + note,
    no per-item fresh flag -- freshness only travels in the note)."""
    entry = {
        "name": name,
        "evidence": {
            "mechanism": (
                "直接偏好"
                if any(
                    row["provenance"] == "直接" for row in (basis if basis is not None else [])
                )
                else "推断偏好"
            ),
            "basis": basis if basis is not None else [],
        },
    }
    if artist is not None:
        entry["artist_name"] = artist
    if note is not None:
        entry["evidence"]["note"] = note
    return entry


def run_payload(*items: dict) -> dict:
    return {
        "run_id": "rcm_11111111-1111-4111-8111-111111111111",
        "item_count": len(items),
        "items": list(items),
    }


class MandateCopyMappingTest(unittest.TestCase):
    """§八 A--F: one evidence shape maps to exactly one honest reason line."""

    def test_direct_genre_describes_the_direction(self) -> None:
        # A: a direct row with a reliable label may be more specific.
        text = render_recommendation_items(payload(item("Rock Song", basis=[ROCK_DIRECT])))
        self.assertIn("这首来自你对 Rock 方向的已有偏好。", text)
        self.assertNotIn("推断", text)

    def test_direct_unknown_kind_keeps_the_mandate_generic(self) -> None:
        # A: 「这首来自你已有的明确偏好。」 -- never drops below that floor.
        text = render_recommendation_items(payload(item("Mystery", basis=[
            basis_row("collection", "收藏夹", "直接"),
        ])))
        self.assertIn("这首来自你已有的明确偏好。", text)

    def test_inferred_genre_is_never_written_as_direct(self) -> None:
        # B: inferred rock must read 「按 Rock 方向推断出来」 -- no 已有偏好.
        text = render_recommendation_items(payload(item("Rock Song", basis=[ROCK])))
        self.assertIn("这首按 Rock 方向推断出来。", text)
        self.assertNotIn("已有偏好", text)

    def test_inferred_track_self_uses_the_fix09_definition(self) -> None:
        # C: Fix09's own definition for basis 曲目自身.
        text = render_recommendation_items(payload(item("夜曲", basis=[TRACK_SELF])))
        self.assertIn("这首按对这首曲目本身的偏好推断递选。", text)
        self.assertNotIn("Rock", text)

    def test_fresh_with_evidence_combines_identity_and_reason(self) -> None:
        # D: fresh identity + the evidence-derived reason, one sentence.
        text = render_recommendation_items(
            payload(item("Fresh Rock", basis=[ROCK], fresh=True))
        )
        self.assertIn("这是本次新发现，按 Rock 方向推断出来。", text)

    def test_fresh_without_evidence_is_honest(self) -> None:
        # E: fresh with no matching evidence reads the fixed honest sentence.
        text = render_recommendation_items(payload(item("Unknown", fresh=True)))
        self.assertIn("这是本次新发现，目前没有更直接的偏好匹配证据。", text)

    def test_non_fresh_item_is_never_called_new(self) -> None:
        # F: a catalog candidate outside this request's discovery is NOT 本次新发现,
        # even with an inferred basis.
        text = render_recommendation_items(payload(item("Old Favorite", basis=[ROCK])))
        self.assertNotIn("本次新发现", text)
        self.assertNotIn("新发现", text)

    def test_post_fix09_note_only_bytes(self) -> None:
        # The builder's note field is the only fact for a zero-basis item; the
        # renderer must not invent suitability -- it reads the honest non-fresh
        # floor sentence (mirrors the note's fact level, without the field name).
        entry = item("Note Only")
        entry["evidence"] = {"mechanism": "exploration", "basis": [],
                             "note": "本次目录搜索的新发现（暂无偏好匹配证据）"}
        text = render_recommendation_items(payload(entry))
        self.assertIn("这首目前没有更直接的偏好匹配证据。", text)

    def test_multiple_basis_rows_join_on_the_delimiter(self) -> None:
        text = render_recommendation_items(payload(
            item("Mixed", basis=[ROCK, basis_row("artist", "乐队", "直接")])
        ))
        self.assertIn("这首按 Rock 方向推断出来；来自你对「乐队」的已有偏好。", text)

    def test_direct_track_self_reads_this_tracks_preference(self) -> None:
        text = render_recommendation_items(
            payload(item("夜曲", basis=[TRACK_SELF_DIRECT]))
        )
        self.assertIn("这首来自你对这首曲目本身的已有偏好。", text)

    def test_direct_and_inferred_reference_other_tracks_verbatim(self) -> None:
        other_direct = basis_row("track", "晴天", "直接")
        other_inferred = basis_row("track", "雨下一整晚", "推断")
        text = render_recommendation_items(payload(
            item("夜曲", basis=[other_direct]),
            item("七里香", basis=[other_inferred]),
        ))
        self.assertIn("这首来自你对《晴天》的已有偏好。", text)
        self.assertIn("这首根据你对《雨下一整晚》的偏好推断选入。", text)

    def test_artist_basis_copy(self) -> None:
        text = render_recommendation_items(payload(
            item("A", basis=[basis_row("artist", "乐队", "直接")]),
            item("B", basis=[ARTIST_INFERRED]),
        ))
        self.assertIn("这首来自你对「乐队」的已有偏好。", text)
        self.assertIn("这首按「LOVE PSYCHEDELICO」的艺人偏好推断出来。", text)

    def test_unknown_inferred_row_is_dropped_not_guessed(self) -> None:
        # An inferred row of an unknown kind cannot be honestly specific; it is
        # dropped and the item falls to the no-direct-evidence floor.
        text = render_recommendation_items(payload(
            item("X", basis=[basis_row("playlist", "驾车", "推断")])
        ))
        self.assertIn("这首目前没有更直接的偏好匹配证据。", text)
        self.assertNotIn("驾车", text)


class FormatAndOrderTest(unittest.TestCase):
    def test_header_counts_items_and_order_is_the_payload_order(self) -> None:
        batch = payload(
            item("first", basis=[ROCK]),
            item("second", basis=[ROCK_DIRECT]),
            item("third", basis=[]),
        )
        text = render_recommendation_for_user(batch)
        self.assertTrue(text.startswith("为你推荐这 3 首："), text)
        first = text.find("1. first")
        second = text.find("2. second")
        third = text.find("3. third")
        self.assertLess(first, second)
        self.assertLess(second, third)
        self.assertEqual(text.count("1. first"), 1)  # one ordered list only

    def test_artist_line_format_when_present(self) -> None:
        text = render_recommendation_items(payload(
            item("夜曲", artist="测试艺人", basis=[ROCK])
        ))
        self.assertIn("1. 夜曲 — 测试艺人\n", text)

    def test_artist_line_omitted_when_absent(self) -> None:
        text = render_recommendation_items(payload(item("夜曲", basis=[ROCK])))
        self.assertIn("1. 夜曲\n", text)
        self.assertNotIn("—", text)

    def test_full_render_ends_with_the_playback_cue(self) -> None:
        text = render_recommendation_for_user(payload(item("夜曲", basis=[ROCK])))
        self.assertTrue(text.endswith("需要试听哪一首，直接告诉我。"), text)

    def test_all_preview_only_batch_gets_the_30_second_cue(self) -> None:
        text = render_recommendation_cue(payload(
            item("a", basis=[ROCK], route="preview_only"),
            item("b", basis=[ROCK], route="preview_only"),
        ))
        self.assertEqual(text, "这批曲目都只能试听 30 秒。需要试听哪一首，直接告诉我。")

    def test_mixed_or_library_batch_gets_the_plain_cue(self) -> None:
        self.assertEqual(
            render_recommendation_cue(payload(
                item("a", basis=[ROCK], route="preview_only"),
                item("b", basis=[ROCK], route="library"),
            )),
            "需要试听哪一首，直接告诉我。",
        )
        self.assertEqual(
            render_recommendation_cue(payload(item("a", basis=[ROCK]))),
            "需要试听哪一首，直接告诉我。",
        )

    def test_one_item_batch_renders_one_item(self) -> None:
        # §十四: an honest 1-item batch shows exactly 1 item (no padding).
        text = render_recommendation_for_user(payload(item("Only", basis=[ROCK_DIRECT])))
        self.assertTrue(text.startswith("为你推荐这 1 首："), text)
        self.assertEqual(text.count("1. "), 1)
        self.assertNotIn("2. ", text)


class StrictFailSafeContractTest(unittest.TestCase):
    """A payload outside the post-Fix09 item contract renders None -- the
    caller keeps its ordinary text path instead of showing a guessed reason."""

    def _assert_none(self, candidate) -> None:
        self.assertIsNone(render_recommendation_for_user(candidate))

    def test_none_payload(self) -> None:
        self._assert_none(None)

    def test_non_mapping_payload(self) -> None:
        self._assert_none("not a mapping")

    def test_missing_items_field(self) -> None:
        self._assert_none({"run_id": "rcm_11111111-1111-4111-8111-111111111111"})

    def test_empty_items(self) -> None:
        self._assert_none(payload())

    def test_non_list_items(self) -> None:
        self._assert_none({"items": {"0": {"name": "x"}}})

    def test_non_mapping_item(self) -> None:
        self._assert_none(payload("bare string item"))

    def test_missing_name(self) -> None:
        entry = item("x")
        del entry["name"]
        self._assert_none(payload(entry))

    def test_blank_name(self) -> None:
        self._assert_none(payload(item("   ")))

    def test_missing_evidence_block(self) -> None:
        # The legacy replayed / canned shape (name only) must fail closed.
        self._assert_none(payload({"name": "夜曲", "artist_name": "测试艺人"}))

    def test_non_mapping_evidence(self) -> None:
        entry = item("x")
        entry["evidence"] = "直接偏好"
        self._assert_none(payload(entry))


class BannedSurfaceTest(unittest.TestCase):
    """§九/§十/§十一: the rendered surface never carries internal terms,
    encyclopedia filler, or current-playing causality. The renderer is a pure
    projection of the payload, so this is structural (not sanitizer-dependent)."""

    BANNED = [
        "novel",
        "fresh=true",
        "fresh=false",
        "fresh_this_request",
        "mechanism",
        "provenance",
        "basis",
        "source_path",
        "score_total",
        "score",
        "candidate",
        "run_id",
        "rcm_",
        "cnd_",
        "trk_",
        "canonical",
        "target_id",
        "目录推断机制",
        "经典金曲",
        "动漫主题曲",
        "代表作",
        "治愈",
        "招牌风格",
        "基于你正在听",
        "当前播放的",
        "延续当前播放",
    ]

    def _full_render(self) -> str:
        return render_recommendation_for_user(payload(
            item("夜曲", artist="测试艺人", basis=[TRACK_SELF]),
            item("Fresh Rock", basis=[ROCK], fresh=True, route="preview_only"),
            item("Unknown", fresh=True, route="preview_only"),
            item("Old Favorite", basis=[ROCK_DIRECT]),
            item("Note Only"),
        ))

    def test_no_banned_token_anywhere(self) -> None:
        text = self._full_render()
        for banned in self.BANNED:
            self.assertNotIn(banned, text)

    def test_cue_and_items_share_the_ban(self) -> None:
        for banned in self.BANNED:
            self.assertNotIn(banned, render_recommendation_items(
                payload(item("x", basis=[ROCK]))
            ))
            self.assertNotIn(banned, render_recommendation_cue(
                payload(item("x", basis=[ROCK], route="preview_only"))
            ))


class ExplanationDirectionSummaryTest(unittest.TestCase):
    """§六/§七: the direction summary is computed from real genre evidence --
    three directions are three, track-self and no-evidence items never join
    the genre count, and a batch with no genre rows claims no direction."""

    def test_mixed_three_directions_names_all_three(self) -> None:
        # Alternative×1 / Anime×1 / Mandopop×3 must read THREE directions.
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Alt Song", basis=[ALT]),
            run_item("Anime Song", basis=[ANIME]),
            run_item("M1", basis=[MANDO]),
            run_item("M2", basis=[MANDO]),
            run_item("M3", basis=[MANDO]),
        ))
        self.assertIn(
            "这一批主要来自 Alternative、Anime 和 Mandopop 三个方向：", text
        )
        self.assertNotIn("两个方向", text)
        self.assertNotIn("2 个方向", text)
        self.assertIn("1. Alt Song\n这首按 Alternative 方向推断出来。", text)
        self.assertIn("3. M1\n这首按 Mandopop 方向推断出来。", text)

    def test_all_one_direction_batch_summarizes_the_whole_batch(self) -> None:
        items = [run_item(f"Anime {index}", basis=[ANIME]) for index in range(5)]
        text = render_recommendation_explanation_for_user(run_payload(*items))
        self.assertIn("这一批 5 首都来自 Anime 方向。", text)
        self.assertEqual(text.count("按 Anime 方向推断出来。"), 5)

    def test_two_directions_join_with_he_not_a_comma(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("A", basis=[ANIME]),
            run_item("B", basis=[MANDO]),
        ))
        self.assertIn("这一批主要来自 Anime 和 Mandopop 两个方向：", text)
        self.assertNotIn("、", text)

    def test_genre_count_ignores_track_self_and_no_evidence_items(self) -> None:
        # Two Anime rows plus a track-self item and a zero-basis item: the
        # summary counts ONLY the genre evidence; the other two keep their own
        # per-item reasons (stated separately, never forced into a genre).
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("A1", basis=[ANIME]),
            run_item("A2", basis=[ANIME]),
            run_item("夜曲", basis=[TRACK_SELF]),
            run_item("Unknown", basis=[]),
        ))
        self.assertIn("这一批主要来自 Anime 方向。", text)
        self.assertIn("这首按对这首曲目本身的偏好推断递选。", text)
        self.assertIn("这首目前没有更直接的偏好匹配证据。", text)
        self.assertNotIn("两个方向", text)

    def test_no_genre_rows_renders_no_direction_header(self) -> None:
        # A batch with no genre evidence at all claims no direction: the item
        # list alone is the explanation.
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("夜曲", basis=[TRACK_SELF]),
        ))
        self.assertTrue(text.startswith("1. "), text)
        self.assertNotIn("这一批", text)
        self.assertNotIn("方向", text)


class ExplanationPerItemCopyTest(unittest.TestCase):
    """§八: each explanation reason reuses the Fix10 first-presentation copy
    -- one vocabulary, fact-identical per item."""

    def test_direct_genre_never_reads_as_inferred(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Rock Song", basis=[ROCK_DIRECT])
        ))
        self.assertIn("这首来自你对 Rock 方向的已有偏好。", text)
        self.assertNotIn("推断", text)

    def test_inferred_genre_copy(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Rock Song", basis=[ROCK])
        ))
        self.assertIn("这首按 Rock 方向推断出来。", text)

    def test_inferred_track_self_keeps_the_fix09_definition(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("夜曲", basis=[TRACK_SELF])
        ))
        self.assertIn("这首按对这首曲目本身的偏好推断递选。", text)

    def test_artist_basis_copy_matches_first_presentation(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("A", basis=[ARTIST_INFERRED])
        ))
        self.assertIn("这首按「LOVE PSYCHEDELICO」的艺人偏好推断出来。", text)

    def test_fresh_zero_basis_note_restores_first_view_fresh_copy(self) -> None:
        # The durable run reader records this-request discovery as the note on
        # a zero-basis item; the explanation reads the same honest fresh copy
        # its first view showed.
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Unknown", basis=[], note="本次目录搜索的新发现（暂无偏好匹配证据）")
        ))
        self.assertIn("这是本次新发现，目前没有更直接的偏好匹配证据。", text)

    def test_zero_basis_without_note_reads_the_plain_floor(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Unknown", basis=[])
        ))
        self.assertIn("这首目前没有更直接的偏好匹配证据。", text)
        self.assertNotIn("本次新发现", text)

    def test_order_is_the_payload_order(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("first", basis=[ANIME]),
            run_item("second", basis=[MANDO]),
            run_item("third", basis=[ALT]),
        ))
        first = text.find("1. first")
        second = text.find("2. second")
        third = text.find("3. third")
        self.assertLess(first, second)
        self.assertLess(second, third)

    def test_artist_line_format_when_present(self) -> None:
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("夜曲", artist="测试艺人", basis=[ROCK])
        ))
        self.assertIn("1. 夜曲 — 测试艺人\n", text)


class ExplanationSurfaceDisciplineTest(unittest.TestCase):
    """§五/§九/§十/十二: the explanation never invents, never leaks, and
    never carries the presentation cue. The renderer is a pure projection of
    the run payload, so this is structural (not sanitizer-dependent)."""

    BANNED = [
        "满分",
        "100%",
        "高度匹配",
        "评分",
        "score",
        "动漫来源",
        "动漫主题曲",
        "经典",
        "代表作",
        "治愈",
        "招牌风格",
        "基于你正在听",
        "当前播放",
        "延续",
        "rcm_",
        "cnd_",
        "trk_",
        "run_id",
        "candidate",
        "mechanism",
        "provenance",
        "basis",
        "novel",
        "fresh_this_request",
    ]

    def _full_render(self) -> str:
        return render_recommendation_explanation_for_user(run_payload(
            run_item("夜曲", artist="测试艺人", basis=[TRACK_SELF]),
            run_item("Fresh Rock", basis=[ROCK]),
            run_item("Unknown", basis=[], note="本次目录搜索的新发现（暂无偏好匹配证据）"),
            run_item("Old Favorite", basis=[ROCK_DIRECT]),
            run_item("Note Only", basis=[]),
        ))

    def test_no_banned_token_anywhere(self) -> None:
        text = self._full_render()
        for banned in self.BANNED:
            self.assertNotIn(banned, text)

    def test_no_playback_cue_in_the_explanation(self) -> None:
        # A presentation list invites playback; an explanation answers why.
        self.assertNotIn("试听", self._full_render())
        self.assertNotIn("直接告诉我", self._full_render())

    def test_current_playing_never_surfaces_unless_cited_as_basis(self) -> None:
        # The renderer only ever writes names/artists/labels from the payload,
        # so a current-playing track can appear ONLY as a cited basis label.
        text = render_recommendation_explanation_for_user(run_payload(
            run_item("Anime Song", basis=[ANIME]),
            run_item("夜曲", basis=[TRACK_SELF]),
            run_item("Unknown", basis=[]),
        ))
        self.assertNotIn("YOASOBI", text)
        cited = render_recommendation_explanation_for_user(run_payload(
            run_item("A", basis=[basis_row("artist", "YOASOBI", "推断")])
        ))
        self.assertIn("这首按「YOASOBI」的艺人偏好推断出来。", cited)


class ExplanationFailSafeContractTest(unittest.TestCase):
    """A run payload outside the post-Fix09 item contract renders None -- the
    caller replies with the fixed fail-honest sentence, never a guessed
    reason."""

    def _assert_none(self, candidate) -> None:
        self.assertIsNone(render_recommendation_explanation_for_user(candidate))

    def test_none_payload(self) -> None:
        self._assert_none(None)

    def test_non_mapping_payload(self) -> None:
        self._assert_none("not a mapping")

    def test_missing_items_field(self) -> None:
        self._assert_none({"run_id": "rcm_11111111-1111-4111-8111-111111111111"})

    def test_empty_items(self) -> None:
        self._assert_none(run_payload())

    def test_legacy_name_only_item_fails_closed(self) -> None:
        self._assert_none({"items": [{"name": "夜曲", "artist_name": "测试艺人"}]})

    def test_non_mapping_evidence(self) -> None:
        entry = run_item("x")
        entry["evidence"] = "直接偏好"
        self._assert_none(run_payload(entry))


if __name__ == "__main__":
    unittest.main()