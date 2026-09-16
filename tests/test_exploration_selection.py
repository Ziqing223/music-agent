"""P15-S3-S3C: unit tests for the pure exploration-selection floor.

Covers selection items 13-22 of the slice test plan plus the classifier
predicate contract: floor met -> unchanged, the owner's worked rank example,
multiple/depleted supply, best effort without fabrication, original-order
preservation, no score mutation, no resurrection of absent items, and the
catalog-source classification with fail-closed typing.
"""

from __future__ import annotations

import unittest

from music_agent.candidate_generation import (
    CANDIDATE_SOURCE_PATH,
    CANDIDATE_SOURCE_SYSTEM,
)
from music_agent.catalog_candidate_generation import (
    CATALOG_CANDIDATE_SOURCE_PATH,
    CATALOG_CANDIDATE_SOURCE_SYSTEM,
)
from music_agent.exploration_selection import (
    apply_exploration_floor,
    fresh_membership_predicate,
    is_catalog_exploration,
    is_fresh_driven,
)
from music_agent.fresh_candidate_generation import (
    FRESH_CANDIDATE_SOURCE_PATH,
    FRESH_CANDIDATE_SOURCE_SYSTEM,
)
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    PreferenceTargetKind,
    PreferenceTargetReference,
    RecommendationItem,
    ScoreBreakdown,
    ScoreComponent,
)


def _item(rank_suffix: int, total: float, *, catalog: bool) -> RecommendationItem:
    """One hand-built ranked item; ``rank_suffix`` fixes a stable cnd_ identity."""
    system = CATALOG_CANDIDATE_SOURCE_SYSTEM if catalog else CANDIDATE_SOURCE_SYSTEM
    path = CATALOG_CANDIDATE_SOURCE_PATH if catalog else CANDIDATE_SOURCE_PATH
    candidate = Candidate(
        candidate_id=f"cnd_00000000-0000-4000-8000-{rank_suffix:012d}",
        target=PreferenceTargetReference(
            PreferenceTargetKind.TRACK, f"trk_00000000-0000-4000-8000-{rank_suffix:012d}"
        ),
        source=CandidateSourceReference(system, path),
    )
    return RecommendationItem(
        candidate=candidate,
        score=ScoreBreakdown(total=total, components=(ScoreComponent("test", total),)),
    )


class ExplorationFloorSelectionTest(unittest.TestCase):
    """Selection semantics of ``apply_exploration_floor`` (items 13-22)."""

    def _ranked(self, *specs: tuple[str, float]) -> tuple[RecommendationItem, ...]:
        """``specs`` is the complete ranked list in rank order: (kind, total)."""
        items: list[RecommendationItem] = []
        for index, (kind, total) in enumerate(specs):
            items.append(_item(index + 1, total, catalog=(kind == "c")))
        return tuple(items)

    def _ids(self, items: tuple[RecommendationItem, ...]) -> list[str]:
        return [item.candidate.target.target_id for item in items]

    @staticmethod
    def _tid(n: int) -> str:
        return f"trk_00000000-0000-4000-8000-{n:012d}"

    def test_item13_floor_zero_returns_first_limit_unchanged(self) -> None:
        ranked = self._ranked(("f", 0.92), ("f", 0.90), ("c", 0.76))
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=0, limit=2
        )
        self.assertEqual(result, ranked[:2])

    def test_item14_floor_already_met_returns_unchanged(self) -> None:
        ranked = self._ranked(("c", 0.92), ("f", 0.90), ("f", 0.87), ("c", 0.76))
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=1, limit=2
        )
        self.assertEqual(result, ranked[:2])

    def test_item15_owner_worked_example(self) -> None:
        # Rank: 1 Familiar .92, 2 Familiar .90, 3 Familiar .87, 4 Familiar .84,
        # 5 Familiar .81, 6 Catalog .76; limit=5, floor=1 -> 1,2,3,4,6. Catalog
        # gets NO bonus -- it only replaces the LOWEST-ranked Familiar.
        ranked = self._ranked(
            ("f", 0.92), ("f", 0.90), ("f", 0.87), ("f", 0.84), ("f", 0.81),
            ("c", 0.76),
        )
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=1, limit=5
        )
        self.assertEqual(
            self._ids(result)[-1], self._tid(6), "catalog replaces rank-5 familiar"
        )
        self.assertEqual(
            self._ids(result),
            [self._tid(n) for n in (1, 2, 3, 4, 6)],
            "final order must be the ORIGINAL rank order, no re-sort",
        )

    def test_item16_two_floors_replace_two_lowest_familiars(self) -> None:
        ranked = self._ranked(
            ("f", 0.92), ("f", 0.90), ("f", 0.87), ("f", 0.84), ("c", 0.76),
            ("c", 0.75),
        )
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=2, limit=4
        )
        self.assertEqual(self._ids(result), [self._tid(n) for n in (1, 2, 5, 6)])

    def test_item17_zero_qualified_tail_is_best_effort_unchanged(self) -> None:
        ranked = self._ranked(("f", 0.92), ("f", 0.90), ("f", 0.87))
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=1, limit=2
        )
        self.assertEqual(result, ranked[:2])

    def test_item18_floor_above_supply_injects_all_available(self) -> None:
        ranked = self._ranked(
            ("f", 0.92), ("f", 0.90), ("f", 0.87), ("c", 0.76)
        )
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=3, limit=5
        )
        self.assertEqual(self._ids(result), [self._tid(n) for n in (1, 2, 3, 4)])
        result2 = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=3, limit=3
        )
        self.assertEqual(
            self._ids(result2), [self._tid(n) for n in (1, 2, 4)]
        )

    def test_item19_original_order_preserved_among_selected(self) -> None:
        ranked = self._ranked(
            ("f", 0.95), ("c", 0.94), ("f", 0.93), ("c", 0.92), ("f", 0.91),
        )
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=2, limit=3
        )
        # Selected set = {1,2,4}; output must keep their original positions.
        self.assertEqual(self._ids(result), [self._tid(1), self._tid(2), self._tid(4)])

    def test_item20_empty_ranked_returns_empty(self) -> None:
        self.assertEqual(
            apply_exploration_floor(
                (), is_exploration=is_catalog_exploration, floor=1, limit=5
            ),
            (),
        )

    def test_item21_shorter_than_limit_head_covers_all(self) -> None:
        ranked = self._ranked(("f", 0.92), ("c", 0.76))
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=1, limit=5
        )
        self.assertEqual(result, ranked)

    def test_item22_no_score_or_item_mutation(self) -> None:
        ranked = self._ranked(
            ("f", 0.92), ("f", 0.90), ("f", 0.87), ("c", 0.76)
        )
        before = {item.candidate.candidate_id: item.score.total for item in ranked}
        assert before
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=1, limit=2
        )
        for item in result:
            self.assertEqual(item.score.total, before[item.candidate.candidate_id])
        # Every returned item must be present in the input list (no resurrection,
        # no fabrication), object-identical.
        original = {id(item) for item in ranked}
        self.assertTrue(all(id(item) in original for item in result))

    def test_floor_equal_limit_all_familiar_permits_full_swap(self) -> None:
        ranked = self._ranked(("f", 0.95), ("f", 0.90), ("c", 0.85), ("c", 0.80))
        result = apply_exploration_floor(
            ranked, is_exploration=is_catalog_exploration, floor=2, limit=2
        )
        self.assertEqual(self._ids(result), [self._tid(3), self._tid(4)])


class CatalogExplorationPredicateTest(unittest.TestCase):
    """Classifier contract (identity item 23): real source semantics only."""

    def test_catalog_source_is_exploration(self) -> None:
        item = _item(1, 0.76, catalog=True)
        self.assertTrue(is_catalog_exploration(item))
        self.assertEqual(
            item.candidate.source.source_system, CATALOG_CANDIDATE_SOURCE_SYSTEM
        )
        self.assertEqual(
            item.candidate.source.source_path, CATALOG_CANDIDATE_SOURCE_PATH
        )

    def test_preference_source_is_familiar(self) -> None:
        item = _item(1, 0.92, catalog=False)
        self.assertFalse(is_catalog_exploration(item))

    def test_non_contract_input_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            is_catalog_exploration(object())  # type: ignore[arg-type]


class FreshMembershipPredicateTest(unittest.TestCase):
    """P15-S3-S3D classifier contract: fresh identity is TARGET MEMBERSHIP ONLY.

    Freshness equals membership in the authoritative same-run promoted set -- it
    is never inferred from candidate sources or labels, and the exploration
    classifier is deliberately independent from it (a catalog candidate can be
    non-fresh; a preference-driven target can be fresh, as the live
    catalog_driven=False fresh case proved).
    """

    @staticmethod
    def _tid(n: int) -> str:
        return f"trk_00000000-0000-4000-8000-{n:012d}"

    def test_membership_by_target_id_only(self) -> None:
        predicate = fresh_membership_predicate([self._tid(1)])
        self.assertTrue(predicate(_item(1, 0.9, catalog=False)))
        self.assertTrue(predicate(_item(1, 0.9, catalog=True)))
        self.assertFalse(predicate(_item(2, 0.9, catalog=False)))
        self.assertFalse(predicate(_item(2, 0.9, catalog=True)))

    def test_empty_set_classifies_nothing_fresh(self) -> None:
        predicate = fresh_membership_predicate(())
        for item in (_item(1, 0.9, catalog=False), _item(2, 0.9, catalog=True)):
            self.assertFalse(predicate(item))

    def test_predicate_freezes_input_set(self) -> None:
        mutable = [self._tid(1)]
        predicate = fresh_membership_predicate(mutable)
        mutable.append(self._tid(2))
        self.assertTrue(predicate(_item(1, 0.9, catalog=True)))
        self.assertFalse(predicate(_item(2, 0.9, catalog=True)))

    def test_non_contract_input_fails_closed(self) -> None:
        predicate = fresh_membership_predicate([self._tid(1)])
        with self.assertRaises(TypeError):
            predicate(object())  # type: ignore[arg-type]


class FreshFloorSelectionTest(unittest.TestCase):
    """P15-S3-S3D: the fresh floor rides the frozen exploration primitive.

    The Request-2 regression shape is pinned here at the primitive level: a
    complete rank whose visible head carries zero fresh items and whose tail
    holds the eligible fresh candidates (highest at rank 16, echoing the owner's
    deterministic reproduction) must exchange the lowest-ranked non-fresh head
    place for the highest-ranked fresh tail place -- with no score change, no
    re-sort, and best-effort behavior when the fresh supply runs out.
    """

    @staticmethod
    def _tid(n: int) -> str:
        return f"trk_00000000-0000-4000-8000-{n:012d}"

    def _ranked(self, count: int) -> tuple[RecommendationItem, ...]:
        return tuple(_item(n, 0.9 - n * 0.001, catalog=False) for n in range(1, count + 1))

    def _ids(self, items: tuple[RecommendationItem, ...]) -> list[str]:
        return [item.candidate.target.target_id for item in items]

    def test_request2_baseline_floor_zero_is_head_unchanged(self) -> None:
        ranked = self._ranked(23)
        fresh = fresh_membership_predicate([self._tid(16)])
        result = apply_exploration_floor(ranked, is_exploration=fresh, floor=0, limit=5)
        self.assertEqual(result, ranked[:5])

    def test_request2_min_fresh_one_exchanges_lowest_head_for_rank16(self) -> None:
        # The owner's Request-2 shape: 5 all-known head items, the highest-ranked
        # eligible fresh candidate sitting at rank 16. min_fresh=1 must replace
        # the LOWEST head place (rank 5) with rank 16 -- nothing else moves.
        ranked = self._ranked(23)
        fresh = fresh_membership_predicate([self._tid(16)])
        result = apply_exploration_floor(ranked, is_exploration=fresh, floor=1, limit=5)
        self.assertEqual(
            self._ids(result),
            [self._tid(n) for n in (1, 2, 3, 4, 16)],
        )

    def test_request2_min_fresh_two_also_injects_rank22(self) -> None:
        ranked = self._ranked(23)
        fresh = fresh_membership_predicate([self._tid(16), self._tid(22)])
        result = apply_exploration_floor(ranked, is_exploration=fresh, floor=2, limit=5)
        self.assertEqual(
            self._ids(result),
            [self._tid(n) for n in (1, 2, 3, 16, 22)],
        )

    def test_fresh_floor_above_supply_is_best_effort(self) -> None:
        ranked = self._ranked(23)
        fresh = fresh_membership_predicate([self._tid(16)])
        result = apply_exploration_floor(ranked, is_exploration=fresh, floor=3, limit=5)
        self.assertEqual(
            self._ids(result),
            [self._tid(n) for n in (1, 2, 3, 4, 16)],
        )

    def test_fresh_floor_no_scores_or_items_mutated(self) -> None:
        ranked = self._ranked(23)
        before = {item.candidate.candidate_id: item.score.total for item in ranked}
        fresh = fresh_membership_predicate([self._tid(16)])
        result = apply_exploration_floor(ranked, is_exploration=fresh, floor=1, limit=5)
        self.assertEqual([item.score.total for item in result],
                         [before[item.candidate.candidate_id] for item in result])
        original = {id(item) for item in ranked}
        self.assertTrue(all(id(item) in original for item in result))

    def test_fresh_identity_is_independent_of_catalog_source(self) -> None:
        # The exploration classifier and the fresh classifier agree on nothing:
        # a catalog item outside the promoted set is exploration but NOT fresh,
        # and a promoted preference-driven item is fresh though NOT exploration.
        fresh = fresh_membership_predicate([self._tid(1)])
        catalog_familiar = _item(2, 0.8, catalog=True)
        preference_fresh = _item(1, 0.9, catalog=False)
        self.assertTrue(is_catalog_exploration(catalog_familiar))
        self.assertFalse(fresh(catalog_familiar))
        self.assertFalse(is_catalog_exploration(preference_fresh))
        self.assertTrue(fresh(preference_fresh))


class FreshDrivenExplorationPredicateTest(unittest.TestCase):
    """P15-S3-S3E: the ``fresh_driven`` source counts as Catalog exploration."""

    @staticmethod
    def _item(rank_suffix: int, *, source_system: str, source_path: str) -> RecommendationItem:
        candidate = Candidate(
            candidate_id=f"cnd_00000000-0000-4000-8000-{rank_suffix:012d}",
            target=PreferenceTargetReference(
                PreferenceTargetKind.TRACK,
                f"trk_00000000-0000-4000-8000-{rank_suffix:012d}",
            ),
            source=CandidateSourceReference(source_system, source_path),
        )
        return RecommendationItem(
            candidate=candidate,
            score=ScoreBreakdown(
                total=0.0, components=(ScoreComponent("test", 0.0),)
            ),
        )

    def test_fresh_driven_source_is_exploration(self) -> None:
        item = self._item(
            1,
            source_system=FRESH_CANDIDATE_SOURCE_SYSTEM,
            source_path=FRESH_CANDIDATE_SOURCE_PATH,
        )
        self.assertTrue(is_fresh_driven(item))
        self.assertFalse(is_catalog_exploration(item))

    def test_catalog_and_preference_sources_are_not_fresh_driven(self) -> None:
        catalog = self._item(
            1,
            source_system=CATALOG_CANDIDATE_SOURCE_SYSTEM,
            source_path=CATALOG_CANDIDATE_SOURCE_PATH,
        )
        preference = self._item(
            2,
            source_system=CANDIDATE_SOURCE_SYSTEM,
            source_path=CANDIDATE_SOURCE_PATH,
        )
        self.assertFalse(is_fresh_driven(catalog))
        self.assertFalse(is_fresh_driven(preference))
        # The frozen catalog classifier is untouched by the new source.
        self.assertTrue(is_catalog_exploration(catalog))

    def test_fresh_direction_is_the_source_only_never_membership(self) -> None:
        # Fresh *identity* stays membership-only: is_fresh_driven reads the
        # candidate source, while the fresh predicate reads target membership --
        # a fresh_driven item and membership never need to agree here.
        fresh = fresh_membership_predicate([])
        item = self._item(
            3,
            source_system=FRESH_CANDIDATE_SOURCE_SYSTEM,
            source_path=FRESH_CANDIDATE_SOURCE_PATH,
        )
        self.assertTrue(is_fresh_driven(item))
        self.assertFalse(fresh(item))

    def test_non_contract_input_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            is_fresh_driven(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            is_fresh_driven("not-an-item")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()