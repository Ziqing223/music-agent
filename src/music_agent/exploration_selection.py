"""P15-S3-S3C: best-effort exploration floor at final selection only.

The exploration classification reuses the EXISTING candidate source vocabulary --
it never invents a new persistent source field or a fresh marker. An item whose
candidate came from the catalog-driven source (``catalog_driven``) counts as
exploration; everything else (the preference-driven, direct-evidence source) is
Familiar. Because the floor only ever receives the fully-filtered ranked list
coming out of the existing P07 chain, a counted exploration item is by
construction one that already passed affinity matching, eligibility, negative
filtering, repeat/scenario/quality filtering, scoring, and the existing ranking
comparator.

This module holds exactly one selection-stage helper. It never:

* re-scores or re-weights items (no exploration bonus; ``score.total`` is never
  touched and the ranking comparator is never re-run, replaced, or extended);
* resurrects candidates the existing pipeline filtered out (filtered candidates
  are absent from its input, and only already-present items can be selected);
* fabricates candidates or converts an empty tail into a success;
* persists anything -- the floor is request-local and pure.

``apply_exploration_floor`` scans the COMPLETE ranked eligible list (there is no
heuristic widening window, so a qualified exploration item can never fall
outside a window), and the floor is best-effort: fewer qualified exploration
items than requested simply means fewer (or zero) appear -- never a failure,
never a synthetic item.

P15-S3-S3D adds the Fresh floor over the SAME frozen primitive: the generic
selection helper below is reused with a different predicate
(:func:`fresh_membership_predicate`), so the fresh floor shares every property
of the exploration floor -- membership exchange only, no scoring/ordering
change, no resurrection, best effort. Fresh identity is target-membership in
the authoritative same-run promoted set supplied by the caller (the provider
loop's run-local capture); it is never inferred from candidate sources or
labels here.

P15-S3-S3E adds the fresh-driven candidate source: its candidates count as
Catalog exploration via :func:`is_fresh_driven`, so a zero-basis fresh item
lifted by the Fresh floor also satisfies the exploration floor from the same
place. Fresh *identity* remains membership-only -- a promoted track whose
preference evidence produced a normal ``catalog_driven`` candidate is still
fresh, so exploration classification (source) and Fresh classification
(membership) are deliberately never conflated.
"""

from __future__ import annotations

from typing import Callable, Collection, Iterable

from music_agent.catalog_candidate_generation import (
    CATALOG_CANDIDATE_SOURCE_PATH,
    CATALOG_CANDIDATE_SOURCE_SYSTEM,
)
from music_agent.fresh_candidate_generation import (
    FRESH_CANDIDATE_SOURCE_PATH,
    FRESH_CANDIDATE_SOURCE_SYSTEM,
)
from music_agent.recommendation_contract import RecommendationItem


def is_catalog_exploration(item: RecommendationItem) -> bool:
    """The single exploration-classification predicate for the floor.

    Reads the real ``Candidate.source`` semantics -- ``(source_system,
    source_path)`` as defined by the candidate generators -- and compares them
    against the frozen catalog-driven constants. Non-contract input fails closed
    with ``TypeError`` rather than being coerced.
    """
    if not isinstance(item, RecommendationItem):
        raise TypeError("item must be a RecommendationItem")
    source = item.candidate.source
    return (
        source.source_system == CATALOG_CANDIDATE_SOURCE_SYSTEM
        and source.source_path == CATALOG_CANDIDATE_SOURCE_PATH
    )


def is_fresh_driven(item: RecommendationItem) -> bool:
    """P15-S3-S3E: exploration classification for the fresh-driven source.

    A candidate from the explicit Fresh-intent channel (zero-basis, negative-
    vetoed) counts as Catalog exploration -- it exists only because the user
    asked for new songs. Fresh *identity* is NOT this predicate: that stays
    target-membership in the authoritative same-run promoted set
    (:func:`fresh_membership_predicate`), because a promoted track backed by
    real preference evidence is represented by a normal ``catalog_driven``
    candidate and must still read ``fresh_this_request=true``. Non-contract
    input fails closed with ``TypeError``.
    """
    if not isinstance(item, RecommendationItem):
        raise TypeError("item must be a RecommendationItem")
    source = item.candidate.source
    return (
        source.source_system == FRESH_CANDIDATE_SOURCE_SYSTEM
        and source.source_path == FRESH_CANDIDATE_SOURCE_PATH
    )


def fresh_membership_predicate(
    fresh_target_ids: Collection[str],
) -> Callable[[RecommendationItem], bool]:
    """P15-S3-S3D: the Fresh classification predicate for the floor.

    Classifies an item as same-run Fresh iff its target identity belongs to the
    caller-supplied authoritative promoted set (the provider loop's run-local
    capture from genuinely executed ``discover_catalog_tracks`` results). This
    predicate deliberately reads ONLY target membership -- it never consults
    ``Candidate.source``, labels, or any other signal, so an empty set classifies
    nothing as fresh and the floor best-effort no-ops. Non-contract input fails
    closed with ``TypeError``.
    """
    frozen: frozenset[str] = frozenset(fresh_target_ids)

    def is_fresh(item: RecommendationItem) -> bool:
        if not isinstance(item, RecommendationItem):
            raise TypeError("item must be a RecommendationItem")
        return item.candidate.target.target_id in frozen

    return is_fresh


def apply_exploration_floor(
    ranked: Iterable[RecommendationItem],
    is_exploration: Callable[[RecommendationItem], bool],
    floor: int,
    limit: int,
) -> tuple[RecommendationItem, ...]:
    """Best-effort exploration floor over the fully-filtered, ranked item list.

    ``ranked`` is the complete ranked eligible list in the existing rank order
    (``score.total`` descending, canonical ``target_id`` ascending) after every
    existing filter; tuple order is rank. ``is_exploration`` classifies one item
    (see :func:`is_catalog_exploration`). ``floor`` is the minimum count of
    exploration-class items the first ``limit`` places should carry, and
    ``limit`` is the caller-visible batch size.

    Selection rule: take the first ``limit`` places; if they already hold at
    least ``floor`` exploration-class items the list is unchanged. Otherwise
    walk the tail of the complete list in rank order and, for each exploration
    item found, replace the currently LOWEST-ranked Familiar item among the
    selected places -- repeating until the floor is met, the tail is exhausted
    (best effort), or no Familiar place remains to exchange (best effort). The
    returned tuple preserves the ORIGINAL rank order among the selected items.

    ``floor <= 0`` returns the first ``limit`` places unchanged (the
    pre-S3-S3C behavior). Scores, candidates, and ordering are never modified;
    only batch membership changes.
    """
    items = tuple(ranked)
    if floor <= 0 or not items:
        return items[:limit]
    head = list(items[:limit])
    if sum(1 for item in head if is_exploration(item)) >= floor:
        return tuple(head)
    selected_ids = {item.candidate.candidate_id for item in head}
    familiar_head = [item for item in head if not is_exploration(item)]
    needed = floor - (len(head) - len(familiar_head))
    for item in items[limit:]:
        if needed == 0:
            break
        if not is_exploration(item):
            continue
        candidate_id = item.candidate.candidate_id
        if candidate_id in selected_ids:
            continue
        if not familiar_head:
            break
        dropped = familiar_head.pop()
        selected_ids.discard(dropped.candidate.candidate_id)
        selected_ids.add(candidate_id)
        needed -= 1
    kept = tuple(
        item for item in items if item.candidate.candidate_id in selected_ids
    )
    return kept[:limit]