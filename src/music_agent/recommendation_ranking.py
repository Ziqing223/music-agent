"""Production ranking and recommendation chain completion (P07.4).

This module is the ranking slice of the recommendation system. It completes the production chain
from scored eligible candidates to a deterministic ranked :class:`RecommendationResult`, and is
the composition root of the full P07 chain:

``RecommendationRequest`` -> candidates (P07.2) -> scored items (P07.3) -> quality policy
(P07.6, optional) -> ranking (this module) -> ``RecommendationResult`` (P07.1).

Ranking semantics
-----------------

The rank order is a strict total order: ``score.total`` descending, with canonical ``target_id``
ascending as the deterministic final tie-break.  Specialized callers may inject a positive
preference lookup that is consulted only after an exact primary-score tie; ordinary callers omit
it and retain their existing order.  The rank of every item is therefore fully determined by its
score and authoritative inputs -- never by input order, the clock, or randomness. Tuple order in
the assembled result is rank; there is no separate rank field.

P19-T15 novelty tie-break (2026-08): when the caller injects ``recommendation_counts`` (a
``{target_id: count}`` lookup from the persisted ``catalog_track_state`` authority), a
historical ``recommendation_count`` ascending inserts between score and identity:
``score.total`` descending, then ``recommendation_count`` ascending, then ``target_id``
ascending. Relevance stays primary -- the count only discriminates score-tied candidates, so a
lower-relevance item can never outrank a higher-relevance one. Unknown targets count as zero
(absence is "no memory"), and callers that omit the lookup keep the pre-T15 order unchanged.
The lookup values are persisted facts: same facts, same order -- determinism is untouched.

Quality policy placement
------------------------

When ``quality_evidence`` is supplied, the P07.6 quality policy
(:func:`~music_agent.recommendation_quality.apply_quality_policy`) filters the scored items
*before* ranking: excluded items never occupy rank slots, and the surviving items are ranked in
score order and truncated to ``request.limit``. The per-item exclusions are preserved in the
returned :class:`RankingOutcome`'s quality report, so a recommendation produced with quality
controls remains explainable. Without quality evidence the chain is pure ranking.

Determinism and boundaries
--------------------------

``run_id`` and ``produced_at`` are injected by the caller -- this module never reads the clock,
the database, or randomness, and never mutates P06. ``rank_recommendations`` consumes
already-scored items (it is independent of the scoring algorithm);
:func:`build_recommendation` wires the P07.2 candidate generation and P07.3 scoring model
together for the end-to-end chain, and never scores a ``REJECTED`` candidate (the contract
forbids it; rejected candidates are dropped at the candidate layer, where their rejection is
expressed). Assembly goes through the single contract boundary
:func:`~music_agent.recommendation_contract.assemble_recommendation_result`, which stamps the
current ``RECOMMENDATION_CONTRACT_VERSION`` and re-validates the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from music_agent.candidate_generation import generate_candidates
from music_agent.recommendation_contract import (
    Candidate,
    Eligibility,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
    assemble_recommendation_result,
)
from music_agent.recommendation_quality import (
    QualityControlReport,
    QualityEvidence,
    apply_quality_policy,
)
from music_agent.recommendation_scoring import score_candidate


class RecommendationRankingError(ValueError):
    code = "recommendation_ranking_error"


class RecommendationRankingValidationError(RecommendationRankingError):
    code = "validation_error"


@dataclass(frozen=True, slots=True)
class RankingOutcome:
    """A ranked recommendation run plus the quality report that shaped it, if any."""

    result: RecommendationResult
    quality_report: QualityControlReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, RecommendationResult):
            raise RecommendationRankingValidationError(
                "result must be a RecommendationResult"
            )
        if self.quality_report is not None and not isinstance(
            self.quality_report, QualityControlReport
        ):
            raise RecommendationRankingValidationError(
                "quality_report must be a QualityControlReport or None"
            )


def rank_items(
    items: Iterable[RecommendationItem],
    recommendation_counts: Mapping[str, int] | None = None,
    preference_tiebreaks: Mapping[str, float] | None = None,
) -> tuple[RecommendationItem, ...]:
    """Return ``items`` in deterministic rank order.

    Rank is a strict total order: ``score.total`` descending, canonical ``target_id`` ascending
    as the tie-break. P10 reproducibility amendment (2026-08): the previous candidate_id
    tie-break was a per-run UUID, so cross-run top-N membership among score-tied candidates
    was not reproducible for identical inputs. ``candidate_id`` remains the per-run record
    identity but no longer participates in ranking. Input order, the clock, and randomness
    never influence the result.

    P19-T15: ``recommendation_counts`` optionally injects the persisted
    ``catalog_track_state`` recommendation-count authority as ``{target_id: count}``;
    score-tied items then order by count ascending before ``target_id`` ascending, so
    never/rarely-recommended tracks surface ahead of repeated ones without ever outranking a
    higher score. Unknown targets count as zero. Omit (or pass ``None``) for the pre-T15
    order. Wrong types fail closed with :class:`RecommendationRankingValidationError`; every
    element must be a :class:`RecommendationItem`. ``preference_tiebreaks`` is an optional
    ``{target_id: magnitude}`` lookup applied after an exact score tie and before novelty/count;
    it is used by True Similarity V1 so preference cannot outrank seed-relative relevance.
    """
    coerced: list[RecommendationItem] = []
    seen: set[str] = set()
    for item in _require_iterable(items, "items"):
        if not isinstance(item, RecommendationItem):
            raise RecommendationRankingValidationError(
                "each items entry must be a RecommendationItem"
            )
        candidate_id = item.candidate.candidate_id
        if candidate_id in seen:
            raise RecommendationRankingValidationError(
                f"duplicate candidate {candidate_id!r} cannot be ranked"
            )
        seen.add(candidate_id)
        coerced.append(item)
    counts = _coerce_recommendation_counts(recommendation_counts)
    tiebreaks = _coerce_preference_tiebreaks(preference_tiebreaks)
    return tuple(
        sorted(
            coerced,
            key=lambda item: (
                -item.score.total,
                -tiebreaks.get(item.candidate.target.target_id, 0.0),
                counts.get(item.candidate.target.target_id, 0),
                item.candidate.target.target_id,
            ),
        )
    )


def rank_recommendations(
    request: RecommendationRequest,
    items: Iterable[RecommendationItem],
    *,
    run_id: str,
    produced_at: datetime,
    quality_evidence: QualityEvidence | None = None,
    recommendation_counts: Mapping[str, int] | None = None,
    preference_tiebreaks: Mapping[str, float] | None = None,
) -> RankingOutcome:
    """Rank scored items into a deterministic ``RecommendationResult`` for one request.

    ``request`` is the answered request; ``items`` are the already-scored eligible candidates
    (tuple order is irrelevant -- ranking re-establishes order). Every item's target kind must
    match ``request.recommended_kind``. When ``quality_evidence`` is supplied, the P07.6 quality
    policy filters items before ranking and its report is attached to the outcome; otherwise the
    chain is pure ranking. ``recommendation_counts`` (P19-T15) injects the persisted
    recommendation-count tie-break lookup -- see :func:`rank_items`.
    ``preference_tiebreaks`` is likewise documented there and remains secondary to exact
    ``score.total`` equality. The ranked items are
    truncated to ``request.limit`` and assembled through the single contract boundary with the
    injected ``run_id`` / ``produced_at`` -- this function reads neither the clock nor
    randomness.
    """
    request = _require_request(request)
    items = tuple(_require_iterable(items, "items"))
    for item in items:
        if not isinstance(item, RecommendationItem):
            raise RecommendationRankingValidationError(
                "each items entry must be a RecommendationItem"
            )
        if item.candidate.target.kind.value != request.recommended_kind.value:
            raise RecommendationRankingValidationError(
                f"item target kind {item.candidate.target.kind.value} does not match "
                f"requested kind {request.recommended_kind.value}"
            )

    report: QualityControlReport | None = None
    selected = items
    if quality_evidence is not None:
        outcome = apply_quality_policy(items, request.context.now, quality_evidence)
        selected = outcome.selected_items
        report = outcome.report

    ranked = rank_items(
        selected,
        recommendation_counts,
        preference_tiebreaks,
    )[: request.limit]
    result = assemble_recommendation_result(
        request,
        ranked,
        run_id=run_id,
        produced_at=produced_at,
    )
    return RankingOutcome(result=result, quality_report=report)


def build_recommendation(
    request: RecommendationRequest,
    *,
    run_id: str,
    produced_at: datetime,
    quality_evidence: QualityEvidence | None = None,
    extra_candidates: Iterable[Candidate] = (),
    recommendation_counts: Mapping[str, int] | None = None,
) -> RankingOutcome:
    """Run the complete P07 chain for one request: candidates -> scores -> rank -> result.

    ``request`` supplies the context and kind; candidate generation (P07.2) and scoring (P07.3)
    are wired in fixed order, ``REJECTED`` candidates are dropped at the candidate layer (they
    may never be scored), and the surviving scored items are ranked and truncated exactly as in
    :func:`rank_recommendations`, including the optional P07.6 quality policy.
    ``recommendation_counts`` (P19-T15) injects the persisted recommendation-count tie-break
    lookup; omitted it keeps the pre-T15 order. ``run_id`` and ``produced_at`` are injected by
    the caller, so the whole chain is deterministic.

    ``extra_candidates`` (P11.2) appends candidates from other sources (e.g. catalog-driven)
    before scoring. Preference-driven candidates win on duplicate targets: an extra candidate
    whose target a preference-driven candidate already proposes is skipped, never scored twice.
    """
    request = _require_request(request)
    candidates = list(generate_candidates(request.context, request.recommended_kind))
    existing_targets = {candidate.target for candidate in candidates}
    for extra in extra_candidates:
        if not isinstance(extra, Candidate):
            raise RecommendationRankingValidationError(
                "each extra_candidates entry must be a Candidate"
            )
        if extra.target in existing_targets:
            continue
        existing_targets.add(extra.target)
        candidates.append(extra)
    items = tuple(
        RecommendationItem(candidate, score_candidate(candidate, request.context))
        for candidate in candidates
        if candidate.eligibility is Eligibility.ELIGIBLE
    )
    return rank_recommendations(
        request,
        items,
        run_id=run_id,
        produced_at=produced_at,
        quality_evidence=quality_evidence,
        recommendation_counts=recommendation_counts,
    )


def _coerce_recommendation_counts(value: object) -> dict[str, int]:
    """Coerce the injected P19-T15 count lookup, or ``{}`` for the pre-T15 order.

    ``None`` means "no counts supplied" and yields the unchanged (score, target_id) order via
    the zero default. Anything non-``None`` must be a ``{str: int}`` mapping -- strings keep
    identity semantics strict and ints protect the deterministic comparator; wrong types fail
    closed like every other boundary in this module.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecommendationRankingValidationError(
            "recommendation_counts must be a {target_id: count} mapping or None"
        )
    counts: dict[str, int] = {}
    for target_id, count in value.items():
        if not isinstance(target_id, str):
            raise RecommendationRankingValidationError(
                "recommendation_counts keys must be track target ids (str)"
            )
        if not isinstance(count, int) or isinstance(count, bool):
            raise RecommendationRankingValidationError(
                f"recommendation_counts[{target_id!r}] must be an int"
            )
        counts[target_id] = count
    return counts


def _coerce_preference_tiebreaks(value: object) -> dict[str, float]:
    """Validate an optional positive-preference score used only after relevance.

    The primary ``ScoreBreakdown.total`` remains the first comparator.  This
    seam lets a specialized caller (True Similarity V1) reuse the production
    ranking/quality/result assembly while ensuring preference can distinguish
    only exactly score-tied items.  Ordinary callers omit it and retain their
    previous byte-equivalent order.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecommendationRankingValidationError(
            "preference_tiebreaks must be a {target_id: score} mapping or None"
        )
    scores: dict[str, float] = {}
    for target_id, score in value.items():
        if not isinstance(target_id, str):
            raise RecommendationRankingValidationError(
                "preference_tiebreaks keys must be track target ids (str)"
            )
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise RecommendationRankingValidationError(
                f"preference_tiebreaks[{target_id!r}] must be within [0, 1]"
            )
        scores[target_id] = float(score)
    return scores


def _require_request(request: object) -> RecommendationRequest:
    if not isinstance(request, RecommendationRequest):
        raise RecommendationRankingValidationError(
            f"request must be a RecommendationRequest, not {type(request).__name__}"
        )
    return request


def _require_iterable(value: object, label: str):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise RecommendationRankingValidationError(
            f"{label} must be an iterable, not {type(value).__name__}"
        )
    return value
