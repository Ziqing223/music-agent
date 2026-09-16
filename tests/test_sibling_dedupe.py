"""P20 Fix07 unit tests: same-batch sibling/duplicate suppression.

Covers the pure ``music_agent.sibling_dedupe`` surface (sec.3-11): the
canonical-id / ISRC / title+artist identity ladder, conservative
normalization, fail-open identity handling, first-occurrence retention,
and pool backfill.
"""

from __future__ import annotations

import unittest

from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    PreferenceTargetKind,
    PreferenceTargetReference,
    RecommendationItem,
    ScoreBreakdown,
    ScoreComponent,
)
from music_agent.sibling_dedupe import (
    TrackSiblingIdentity,
    exclude_historical_siblings,
    has_sibling_duplicate,
    normalize_title,
    resolve_sibling_identity,
    select_distinct_works,
)

ARTIST_ALPHA = "art_00000000-0000-4000-8000-aaaaaaaaaaaa"
ARTIST_BETA = "art_00000000-0000-4000-8000-bbbbbbbbbbbb"


def _item(n: int) -> RecommendationItem:
    """One hand-built ranked item with a stable cnd_/trk_ identity for slot n."""
    candidate = Candidate(
        candidate_id=f"cnd_00000000-0000-4000-8000-{n:012d}",
        target=PreferenceTargetReference(
            PreferenceTargetKind.TRACK, f"trk_00000000-0000-4000-8000-{n:012d}"
        ),
        source=CandidateSourceReference("fixture", "unit"),
    )
    return RecommendationItem(
        candidate=candidate,
        score=ScoreBreakdown(
            total=1.0 - n * 0.01, components=(ScoreComponent("unit", 1.0 - n * 0.01),)
        ),
    )


def _track(
    title: str,
    artists: tuple[str, ...] = (ARTIST_ALPHA,),
    isrc: str | None = None,
) -> dict:
    external_ids: dict = {"apple_music_persistent_id": None, "itunes_store_id": None}
    if isrc is not None:
        external_ids["isrc"] = isrc
    return {
        "name": title,
        "artist_ids": list(artists),
        "external_ids": external_ids,
    }


def _ids(items: tuple[RecommendationItem, ...]) -> list[str]:
    return [item.candidate.target.target_id for item in items]


class NormalizationTest(unittest.TestCase):
    """Conservative normalization only (sec.5): no text is ever deleted."""

    def test_unicode_fullwidth_case_and_space_equivalence(self) -> None:
        self.assertEqual(
            normalize_title("ＹＯＡＳＯＢＩ　アンコール"),
            normalize_title("yoasobi アンコール"),
        )
        self.assertEqual(
            normalize_title("yoasobi アンコール"),
            "yoasobi アンコール",
        )
        # ASCII upper/lower + runs of surrounding/collapsible whitespace.
        self.assertEqual(normalize_title("  SONG   Title\t"), normalize_title("song title"))
        # Half/full-width punctuation equivalence (NFKC).
        self.assertEqual(normalize_title("Song（Live）"), normalize_title("Song(Live)"))

    def test_version_markers_are_never_stripped(self) -> None:
        for marker in (
            "(live)",
            "(remix)",
            "(acoustic)",
            "(english version)",
            "(piano version)",
            "(demo)",
            "(edit)",
            "(2024 remaster)",
        ):
            with self.subTest(marker=marker):
                self.assertEqual(
                    normalize_title(f"Song {marker}")[:5], "song "
                )
                self.assertNotEqual(
                    normalize_title(f"Song {marker}"), normalize_title("Song")
                )

    def test_non_string_title_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_title(42)  # type: ignore[arg-type]


class IdentityResolutionTest(unittest.TestCase):
    """Fail-open identity resolution from model track mappings (sec.6)."""

    def test_full_identity_resolves_with_isrc(self) -> None:
        identity = resolve_sibling_identity(
            _track("Yoasobi アンコール", isrc="JPABC-0001")
        )
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.group_keys(), (("isrc", "JPABC-0001"),))

    def test_full_identity_without_isrc_resolves_to_title_artist_key(self) -> None:
        identity = resolve_sibling_identity(_track("Yoasobi アンコール"))
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(
            identity.group_keys(),
            (("tt", "yoasobi アンコール", frozenset({ARTIST_ALPHA})),),
        )

    def test_missing_title_fails_open(self) -> None:
        track = _track("有名字", artists=(ARTIST_ALPHA,))
        track["name"] = None
        self.assertIsNone(resolve_sibling_identity(track))
        track["name"] = "   "
        self.assertIsNone(resolve_sibling_identity(track))

    def test_missing_artist_ids_fail_open(self) -> None:
        for bad in (None, [], ["not-a-canonical-artist?"]):
            track = _track("有名字")
            track["artist_ids"] = bad
            self.assertIsNone(resolve_sibling_identity(track))

    def test_blank_isrc_is_ignored(self) -> None:
        track = _track("有名字", isrc="   ")
        identity = resolve_sibling_identity(track)
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertTrue(identity.group_keys()[0][0] == "tt")


class SiblingSelectionTest(unittest.TestCase):
    """Selection semantics of ``select_distinct_works`` / ``has_sibling_duplicate``."""

    def _select(
        self,
        items: tuple[RecommendationItem, ...],
        tracks: dict[str, dict],
        limit: int,
    ) -> tuple[RecommendationItem, ...]:
        return select_distinct_works(items, tracks, limit)

    # --- the identity ladder (sec.4) ----------------------------------------

    def test_identical_canonical_target_appears_once(self) -> None:
        # Defensive: the same canonical target twice in one list collapses, even
        # with no usable identity lookup at all.
        first, second = _item(1), _item(2)
        duplicated = RecommendationItem(
            candidate=Candidate(
                candidate_id=second.candidate.candidate_id,
                target=first.candidate.target,
                source=second.candidate.source,
            ),
            score=second.score,
        )
        selected = self._select((first, duplicated), {}, 5)
        self.assertEqual(_ids(selected), [first.candidate.target.target_id])

    def test_same_isrc_different_catalog_identity_is_suppressed(self) -> None:
        # Same recording under two catalog identities -- suppressed even with
        # different titles and artists (the ISRC is the recording truth).
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("Song", isrc="USABC-0001"),
            items[1].candidate.target.target_id: _track(
                "Song (Original Mix)", artists=(ARTIST_BETA,), isrc="USABC-0001"
            ),
        }
        self.assertTrue(has_sibling_duplicate(items, tracks))
        selected = self._select(items, tracks, 5)
        self.assertEqual(_ids(selected), [items[0].candidate.target.target_id])

    def test_different_isrcs_same_title_same_artist_both_kept(self) -> None:
        # Two DIFFERENT recordings (different ISRCs) never merge (sec.4-2).
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("Song", isrc="USABC-0001"),
            items[1].candidate.target.target_id: _track("Song", isrc="USABC-0002"),
        }
        self.assertFalse(has_sibling_duplicate(items, tracks))
        self.assertEqual(_ids(self._select(items, tracks, 5)), _ids(items))

    def test_same_title_same_artist_container_only_is_suppressed(self) -> None:
        # The UAT shape: two distinct catalog identities, verbatim title, one
        # canonical artist, differing only in the album container.
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("Ano yume wo nazotte"),
            items[1].candidate.target.target_id: _track("Ano yume wo nazotte"),
        }
        self.assertTrue(has_sibling_duplicate(items, tracks))
        selected = self._select(items, tracks, 5)
        self.assertEqual(_ids(selected), [items[0].candidate.target.target_id])

    def test_different_artist_same_title_both_kept(self) -> None:
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("Song", (ARTIST_ALPHA,)),
            items[1].candidate.target.target_id: _track("Song", (ARTIST_BETA,)),
        }
        self.assertFalse(has_sibling_duplicate(items, tracks))
        self.assertEqual(_ids(self._select(items, tracks, 5)), _ids(items))

    def test_one_shared_artist_of_two_distinct_sets_not_suppressed(self) -> None:
        # Artist identity is the canonical set -- an overlapping set is not an
        # identical set.
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track(
                "Song", (ARTIST_ALPHA, ARTIST_BETA)
            ),
            items[1].candidate.target.target_id: _track("Song", (ARTIST_ALPHA,)),
        }
        self.assertFalse(has_sibling_duplicate(items, tracks))
        self.assertEqual(_ids(self._select(items, tracks, 5)), _ids(items))

    def test_version_marker_variants_are_all_kept(self) -> None:
        titles = ("Song", "Song (Live)", "Song (Remix)", "Song (Acoustic)",
                  "Song (English Version)", "Song (Piano Version)")
        items = tuple(_item(n) for n in range(1, len(titles) + 1))
        tracks = {
            items[n].candidate.target.target_id: _track(titles[n])
            for n in range(len(titles))
        }
        self.assertFalse(has_sibling_duplicate(items, tracks))
        self.assertEqual(_ids(self._select(items, tracks, 20)), _ids(items))

    def test_unicode_case_space_duplicate_is_suppressed(self) -> None:
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("ＹＯＡＳＯＢＩ　アンコール"),
            items[1].candidate.target.target_id: _track("yoasobi アンコール"),
        }
        self.assertTrue(has_sibling_duplicate(items, tracks))
        selected = self._select(items, tracks, 5)
        self.assertEqual(_ids(selected), [items[0].candidate.target.target_id])

    def test_missing_identity_fails_open_on_both_sides(self) -> None:
        # An unknown target neither suppresses nor is suppressed (sec.6).
        items = (_item(1), _item(2), _item(3))
        tracks = {
            items[0].candidate.target.target_id: _track(
                "Song", artists=(ARTIST_ALPHA,), isrc="USX-1"
            ),
            # item 2: no row at all -- unknown identity.
            items[2].candidate.target.target_id: _track(
                "Song", artists=(ARTIST_ALPHA,), isrc="USX-1"
            ),
        }
        self.assertTrue(has_sibling_duplicate(items, tracks))
        selected = self._select(items, tracks, 5)
        # item1 (isrc USX-1) and item3 (isrc USX-1) collide -> item3 dropped;
        # item2 has no identity row and is kept.
        self.assertEqual(
            [i.candidate.candidate_id for i in selected],
            [items[0].candidate.candidate_id, items[1].candidate.candidate_id],
        )

    def test_one_sided_isrc_is_undecidable_and_both_kept(self) -> None:
        # One side carries the recording truth, the other only title+artist:
        # cannot prove either way -> fail open.
        items = (_item(1), _item(2))
        tracks = {
            items[0].candidate.target.target_id: _track("Song", isrc="USABC-0001"),
            items[1].candidate.target.target_id: _track("Song"),
        }
        self.assertFalse(has_sibling_duplicate(items, tracks))
        self.assertEqual(_ids(self._select(items, tracks, 5)), _ids(items))

    # --- retention and backfill (sec.8/9) -----------------------------------

    def test_first_occurrence_in_existing_order_is_retained(self) -> None:
        # The later duplicate is dropped even when it carries the better score.
        items = (_item(2), _item(1))  # order: lower score first
        tracks = {
            items[0].candidate.target.target_id: _track("Song"),
            items[1].candidate.target.target_id: _track("Song"),
        }
        selected = self._select(items, tracks, 5)
        self.assertEqual(_ids(selected), [items[0].candidate.target.target_id])

    def test_limit_5_backfill_continues_past_suppressed_sibling(self) -> None:
        # Pool A, A', B, C, D, E with limit 5 -> A, B, C, D, E (A' suppressed,
        # E backfills the free place -- suppression never shortens a batch).
        items: tuple[RecommendationItem, ...] = tuple(
            _item(n) for n in range(1, 7)
        )
        tracks = {
            items[0].candidate.target.target_id: _track("Alpha"),
            items[1].candidate.target.target_id: _track("Alpha"),
            items[2].candidate.target.target_id: _track("Beta"),
            items[3].candidate.target.target_id: _track("Gamma"),
            items[4].candidate.target.target_id: _track("Delta"),
            items[5].candidate.target.target_id: _track("Epsilon"),
        }
        selected = self._select(items, tracks, 5)
        self.assertEqual(
            _ids(selected),
            [
                items[0].candidate.target.target_id,
                items[2].candidate.target.target_id,
                items[3].candidate.target.target_id,
                items[4].candidate.target.target_id,
                items[5].candidate.target.target_id,
            ],
        )

    def test_exhausted_pool_returns_fewer_items_honestly(self) -> None:
        # No second supply is ever summoned for backfill (sec.10): an exhausted
        # pool returns fewer distinct works instead.
        items = (_item(1), _item(2), _item(3))
        tracks = {
            items[0].candidate.target.target_id: _track("Alpha"),
            items[1].candidate.target.target_id: _track("Alpha"),
            items[2].candidate.target.target_id: _track("Alpha"),
        }
        selected = self._select(items, tracks, 5)
        self.assertEqual(_ids(selected), [items[0].candidate.target.target_id])

    def test_no_collision_selection_is_the_exact_prefix(self) -> None:
        # The no-op path must not reorder or shorten.
        items = tuple(_item(n) for n in range(1, 6))
        tracks = {
            items[n].candidate.target.target_id: _track(f"Distinct {n}")
            for n in range(5)
        }
        selected = self._select(items, tracks, 3)
        self.assertEqual(_ids(selected), _ids(items[:3]))

    def test_suppression_across_backfill_region_is_transitive(self) -> None:
        # A residual pool entry that duplicates a kept head item is skipped;
        # the next distinct residual fills the place.
        items = tuple(_item(n) for n in range(1, 6))
        tracks = {
            items[0].candidate.target.target_id: _track("Alpha"),
            items[1].candidate.target.target_id: _track("Alpha"),
            items[2].candidate.target.target_id: _track("Beta"),
            items[3].candidate.target.target_id: _track("Alpha"),
            items[4].candidate.target.target_id: _track("Gamma"),
        }
        selected = self._select(items, tracks, 3)
        self.assertEqual(
            _ids(selected),
            [
                items[0].candidate.target.target_id,
                items[2].candidate.target.target_id,
                items[4].candidate.target.target_id,
            ],
        )


class HistoricalSiblingExclusionTest(unittest.TestCase):
    """Cross-run filtering reuses the exact same conservative identity ladder."""

    def test_exact_historical_canonical_id_is_excluded(self) -> None:
        item = _item(1)
        self.assertEqual(
            exclude_historical_siblings(
                (item,), {}, (item.candidate.target.target_id,)
            ),
            (),
        )

    def test_owner_release_siblings_are_excluded_across_runs(self) -> None:
        for index, title in enumerate(
            ("Harujion", "Monster", "PINK BLOOD"), start=1
        ):
            with self.subTest(title=title):
                historical = _item(index)
                candidate = _item(index + 10)
                tracks = {
                    historical.candidate.target.target_id: _track(title),
                    candidate.candidate.target.target_id: _track(title),
                }
                self.assertEqual(
                    exclude_historical_siblings(
                        (candidate,),
                        tracks,
                        (historical.candidate.target.target_id,),
                    ),
                    (),
                )

    def test_same_title_different_artist_is_not_excluded(self) -> None:
        historical, candidate = _item(1), _item(2)
        tracks = {
            historical.candidate.target.target_id: _track(
                "Song", (ARTIST_ALPHA,)
            ),
            candidate.candidate.target.target_id: _track(
                "Song", (ARTIST_BETA,)
            ),
        }
        self.assertEqual(
            exclude_historical_siblings(
                (candidate,), tracks, (historical.candidate.target.target_id,)
            ),
            (candidate,),
        )

    def test_near_title_and_explicit_versions_are_not_excluded(self) -> None:
        historical = _item(1)
        candidates = tuple(_item(index) for index in range(2, 6))
        tracks = {
            historical.candidate.target.target_id: _track("Song"),
            candidates[0].candidate.target.target_id: _track("Songs"),
            candidates[1].candidate.target.target_id: _track("Song (Live)"),
            candidates[2].candidate.target.target_id: _track("Song (Remix)"),
            candidates[3].candidate.target.target_id: _track("Song (Acoustic)"),
        }
        self.assertEqual(
            exclude_historical_siblings(
                candidates, tracks, (historical.candidate.target.target_id,)
            ),
            candidates,
        )

    def test_isrc_ladder_remains_authoritative_and_fail_open(self) -> None:
        historical = _item(1)
        same_isrc, different_isrc, no_isrc = _item(2), _item(3), _item(4)
        tracks = {
            historical.candidate.target.target_id: _track(
                "Song", isrc="USABC-0001"
            ),
            same_isrc.candidate.target.target_id: _track(
                "Another title", (ARTIST_BETA,), isrc="USABC-0001"
            ),
            different_isrc.candidate.target.target_id: _track(
                "Song", isrc="USABC-0002"
            ),
            no_isrc.candidate.target.target_id: _track("Song"),
        }
        self.assertEqual(
            exclude_historical_siblings(
                (same_isrc, different_isrc, no_isrc),
                tracks,
                (historical.candidate.target.target_id,),
            ),
            (different_isrc, no_isrc),
        )


class IdentityConstructionTest(unittest.TestCase):
    def test_identity_construction_and_repr(self) -> None:
        identity = TrackSiblingIdentity("Song", (ARTIST_ALPHA,))
        self.assertIn("Song", repr(identity))
        self.assertIsNone(identity.isrc)


if __name__ == "__main__":
    unittest.main()
