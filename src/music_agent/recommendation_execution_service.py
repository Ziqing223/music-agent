"""Recommendation execution domain extracted from :mod:`music_agent.agent_service`.

The shared agent service remains the sole permission/replay/journal/dispatch facade.  This
module owns only the recommendation generation execution paths and receives all durable state
and shared projections from that facade by explicit dependency injection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from music_agent.preference_attribution import InferredAffinity, PreferenceTargetKind, PreferenceTargetReference
from music_agent.preference_propagation import canonicalize_genre_key
from music_agent.preference_query import TrackPreferenceState, query_track_preference
from music_agent.preference_strength import PreferenceState
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    assemble_recommendation_result,
    encode_recommendation_result,
    generate_candidate_id,
)
from music_agent.known_catalog_supply import summarize_known_catalog_supply
from music_agent.recommendation_quality import REASON_PREVIOUSLY_RECOMMENDED, QualityEvidence, collect_previous_targets
from music_agent.recommendation_ranking import RankingOutcome, build_recommendation, rank_recommendations
from music_agent.sibling_dedupe import exclude_historical_siblings, has_sibling_duplicate, select_distinct_works
from music_agent.track_similarity import (
    SIMILARITY_SOURCE_PATH,
    SimilarityExecutionContext,
    TrackSimilarityEvidence,
    score_track_similarity,
)

DEFAULT_SOURCE_SYSTEM = "apple_music"
_AVOID_PREVIOUS_RUNS_WINDOW = 5

_REASON_DIRECTION_FILTERED_ALL_TARGETS = "direction_filtered_all_targets"
_REASON_NO_DIRECT_EVIDENCE = "no_direct_evidence"
_REASON_ALL_TARGETS_NEGATIVE = "all_targets_negative_evidence"
_REASON_ALL_EXCLUDED = "all_eligible_candidates_excluded"
_REASON_NO_CANDIDATES_IN_POOL = "no_candidates_in_pool"
_REASON_EMPTY_GENERATION = "empty_generation"

def _quality_exclusion_counts(report: Any) -> tuple[int, int]:
    """(total, previously_recommended) exclusion counts of one quality report.

    Real execution data only: the report is what the quality policy returned for
    this run, and ``None`` (no exclusions injected) yields (0, 0).
    """
    if report is None:
        return (0, 0)
    total = 0
    previous = 0
    for decision in report.decisions:
        for exclusion in decision.exclusions:
            total += 1
            if exclusion.reason == REASON_PREVIOUSLY_RECOMMENDED:
                previous += 1
    return (total, previous)


def _empty_generation_diagnostics(
    *,
    input_target_count: int,
    after_direction_filter_count: int,
    positive_evidence_count: int,
    negative_evidence_count: int,
    inferred_positive_count: int = 0,
    catalog_candidate_count: int = 0,
    exclusion_count: int = 0,
    excluded_previous_count: int = 0,
    is_inferred: bool = False,
    known_catalog_track_count: int | None = None,
    known_never_recommended_count: int | None = None,
    known_eligible_count: int | None = None,
    known_rejected_count: int | None = None,
    known_never_recommended_eligible_count: int | None = None,
    fresh_promoted_count: int | None = None,
    fresh_candidate_count: int | None = None,
    fresh_negative_rejected_count: int | None = None,
) -> dict[str, Any]:
    """P15-S4-M2-2: truthful per-execution funnel counts + reason + next action.

    Every count comes from real variables of this one execution, never an
    invented number. ``candidate_count`` is the pool actually handed to
    scoring/quality: one candidate per positive-conclusion direct input (the
    plain tool's only source), plus one per positive inferred input and every
    catalog candidate for the inferred tool. Negative conclusions produce
    *rejected* candidates and non-directional states produce none -- both
    faithfully excluded from the pool. Reasons reproduce the funnel's real
    order; ``recommended_next_action`` names the specific remedy for each
    reason (the switch-to-inferred pointer appears only where it is the right
    move, and discovery is advised only when the candidate pool itself is
    empty, per the M2-2 contract).

    P15-S3-S3A: the five optional ``known_*`` keys carry the Known-Catalog
    supply facts (``summarize_known_catalog_supply``) and are emitted only for
    the inferred tool -- the supply concept does not exist for the plain
    preference-driven tool, whose empty remedy stays the switch-to-inferred
    pointer. When present they never invent memory: a missing state row is
    reported as "not never-recommended", exactly like the durable table.

    P15-S3-S3E: the three optional ``fresh_*`` keys carry the explicit Fresh
    channel counts and are emitted only when the caller supplies them (the
    inferred handler supplies them whenever the explicit-intent gate was open,
    ``min_fresh > 0``). Zeros are truthful: an empty promoted set or an
    all-negative-vetoed set simply reads as zero counts.
    """
    candidate_count = (
        positive_evidence_count + inferred_positive_count + catalog_candidate_count
    )
    if after_direction_filter_count == 0 and input_target_count > 0:
        reason = _REASON_DIRECTION_FILTERED_ALL_TARGETS
        next_action = (
            "方向过滤（genres）移除了全部输入目标：方向是硬过滤而非软提示。"
            "请放宽或去掉 genres 过滤，或改选与方向匹配的目标曲目。"
        )
    elif positive_evidence_count == 0 and negative_evidence_count > 0 and candidate_count == 0:
        reason = _REASON_ALL_TARGETS_NEGATIVE
        if is_inferred:
            next_action = (
                "所有带方向性证据的目标都只有负向结论（不喜欢/低评分），负向结论"
                "只会把目标标记为被拒绝的候选、永远不会进入推荐结果。请提供带正向"
                "证据（收藏/高评分）的曲目作为目标，或扩大发现范围。"
            )
        else:
            next_action = (
                "所有带方向性证据的目标都只有负向结论（不喜欢/低评分），负向结论"
                "只会把目标标记为被拒绝的候选、永远不会进入推荐结果；请改用 "
                "generate_inferred_recommendation，或提供不同目标曲目。"
            )
    elif candidate_count == 0:
        reason = (
            _REASON_NO_CANDIDATES_IN_POOL
            if is_inferred
            else _REASON_NO_DIRECT_EVIDENCE
        )
        if is_inferred:
            # P15-S3-S3A: the remedy names the real Known-Catalog supply so the
            # model can tell "fresh discovery is the only route" (supply truly
            # exhausted) from "supply exists but nothing matches the current
            # positive directions" (re-aim the search, never the same input).
            if known_catalog_track_count:
                next_action = (
                    f"候选池完全为空：本轮没有任何直接候选、推断候选或目录候选"
                    f"——已知目录供给为 {known_catalog_track_count} 首"
                    f"（其中 {known_never_recommended_count or 0} 首从未推荐过），"
                    "但没有一首与当前的正向偏好方向匹配，因此无法成为候选；"
                    "继续用相同输入重试不会改变结果。可以按已证实正向方向设置"
                    "搜索词、通过 discover_catalog_tracks 扩充候选目录，"
                    "或改换方向，或如实向用户说明当前没有可推荐的新曲目。"
                )
            else:
                next_action = (
                    "候选池完全为空：本轮没有任何直接候选、推断候选或目录候选，"
                    "且已知目录候选供给已耗尽（没有可复用的已知目录曲目）；"
                    "继续用相同输入重试不会改变结果。可以通过 discover_catalog_tracks "
                    "以正向偏好方向为搜索词扩充候选目录，或如实向用户说明"
                    "当前没有可推荐的新曲目。"
                )
        else:
            next_action = (
                "本工具只会依据目标曲目的方向性直接偏好证据（收藏/高评分为正，"
                "不喜欢/低评分为负）产生候选：没有这类证据的目标会被完全跳过。"
                "新发现的目录曲目通常没有任何偏好证据，所以用相同的 target_ids "
                "重试本工具不会得到不同结果；请改用 generate_inferred_recommendation。"
            )
    elif exclusion_count > 0:
        reason = _REASON_ALL_EXCLUDED
        next_action = (
            "产生的候选全部在质量控制阶段被排除"
            + (
                f"（其中 {excluded_previous_count} 条来自重复推荐抑制）"
                if excluded_previous_count > 0
                else ""
            )
            + "——候选池本身非空。请放宽排除条件"
            "（exclude_target_ids / avoid_previous_runs 等）或更换候选目标。"
        )
    else:
        # Unreachable given the derivations above; kept fail-closed rather than
        # pretending to know a specific cause.
        reason = _REASON_EMPTY_GENERATION
        next_action = (
            "生成流水线在无可观测排除的情况下返回了空结果；请报告该错误，"
            "不要盲目重试相同输入。"
        )
    diagnostics: dict[str, Any] = {
        "input_target_count": input_target_count,
        "after_direction_filter_count": after_direction_filter_count,
        "direct_evidence_count": positive_evidence_count + negative_evidence_count,
        "positive_evidence_count": positive_evidence_count,
        "negative_evidence_count": negative_evidence_count,
        "candidate_count": candidate_count,
        "excluded_previous_count": excluded_previous_count,
        "reason": reason,
        "recommended_next_action": next_action,
    }
    if is_inferred:
        diagnostics["inferred_positive_count"] = inferred_positive_count
        diagnostics["catalog_candidate_count"] = catalog_candidate_count
        # P15-S3-S3A: Known-Catalog supply facts (emitted only when the caller
        # supplied them -- the inferred handler always does; the plain handler
        # never does, and its envelope stays byte-compatible with M2-2).
        supply_keys = (
            ("known_catalog_track_count", known_catalog_track_count),
            ("known_never_recommended_count", known_never_recommended_count),
            ("known_eligible_count", known_eligible_count),
            ("known_rejected_count", known_rejected_count),
            (
                "known_never_recommended_eligible_count",
                known_never_recommended_eligible_count,
            ),
        )
        for key, value in supply_keys:
            if value is not None:
                diagnostics[key] = value
        # P15-S3-S3E: Fresh-channel counts, same None-gated additive seam. The
        # inferred handler supplies actual counts whenever min_fresh > 0 (zeros
        # included); the plain handler never does and stays byte-compatible.
        fresh_keys = (
            ("fresh_promoted_count", fresh_promoted_count),
            ("fresh_candidate_count", fresh_candidate_count),
            ("fresh_negative_rejected_count", fresh_negative_rejected_count),
        )
        for key, value in fresh_keys:
            if value is not None:
                diagnostics[key] = value
    return diagnostics


def _exploration_scan_limit(
    context_inputs: Iterable[PreferenceInput], extra_candidates: Iterable[object]
) -> int:
    """Exact upper bound for the P15-S3-S3C exploration floor scan.

    Every scored item in the inferred chain comes either from the preference-driven
    generator (at most one candidate per TRACK-kind context input) or from the extra
    catalog candidates (deduplicated against preference-driven targets inside
    ``build_recommendation``), so ``#TRACK inputs + #extra candidates`` can never
    under-count the complete ranked eligible list. This replaces any heuristic
    widening window: the floor scans the complete ranked list, so a qualified
    exploration item can never fall "outside a window".
    """

    track_inputs = sum(
        1
        for entry in context_inputs
        if isinstance(entry, PreferenceInput)
        and entry.target.kind is PreferenceTargetKind.TRACK
    )
    return track_inputs + sum(1 for _ in extra_candidates)



class RecommendationExecutionService:
    """Execute recommendation generation over state owned by SharedAgentService."""

    def __init__(
        self,
        *,
        canonical,
        preference,
        recommendation_history,
        catalog_track_state,
        active_context,
        rating_policy,
        magnitude_policy,
        familiarity_policy,
        empty_recommendation_error_type,
        similarity_seed_unavailable_error_type,
        preference_inputs_by_target: Callable[..., Any],
        item_evidence_entry: Callable[..., Any],
        playback_annotation: Callable[..., Any],
        item_playback_summary: Callable[..., Any],
        run_id_factory: Callable[[], str],
    ) -> None:
        self._canonical = canonical
        self._preference = preference
        self._recommendation_history = recommendation_history
        self._catalog_track_state = catalog_track_state
        self._active_context = active_context
        self._rating_policy = rating_policy
        self._magnitude_policy = magnitude_policy
        self._familiarity_policy = familiarity_policy
        self._empty_recommendation_error_type = empty_recommendation_error_type
        self._similarity_seed_unavailable_error_type = similarity_seed_unavailable_error_type
        self._preference_inputs_by_target = preference_inputs_by_target
        self._item_evidence_entry = item_evidence_entry
        self._playback_annotation = playback_annotation
        self._item_playback_summary = item_playback_summary
        self._run_id_factory = run_id_factory

    def _repeat_quality_evidence(
        self, payload: Mapping[str, Any]
    ) -> QualityEvidence | None:
        """Repeat-suppression evidence (P07.6, P19-T15): explicit exclusions and/or history.

        ``exclude_target_ids`` names targets to drop from this run; ``avoid_previous_runs``
        folds in every target recommended by the most recent ``_AVOID_PREVIOUS_RUNS_WINDOW``
        runs -- fresh repeats are suppressed, while earlier recommendations may re-enter the
        pool (short-term repeat control, not permanent dedup).

        P19-T15 default: an unqualified generate -- one that provides NEITHER exclusion
        option -- now excludes the recent-run window server-side, so ordinary consecutive
        recommendations stop repeating the same batch without depending on the provider
        remembering to request dedup. Providing ``exclude_target_ids`` keeps the exact-list
        semantics (no window); providing ``avoid_previous_runs`` keeps the caller's value,
        including an explicit ``false`` that restores the pre-T15 repeat-allowed behavior.
        With empty window history the default still yields no evidence (first run unchanged).
        """
        excluded: set[PreferenceTargetReference] = {
            PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id)
            for target_id in (payload.get("exclude_target_ids") or ())
        }
        if "avoid_previous_runs" in payload:
            use_window = bool(payload["avoid_previous_runs"])
        elif "exclude_target_ids" in payload:
            use_window = False
        else:
            # P19-T15: server-authoritative dedup default over the existing window.
            use_window = True
        if use_window:
            excluded |= collect_previous_targets(
                self._recommendation_history.list_runs(
                    limit=_AVOID_PREVIOUS_RUNS_WINDOW
                )
            )
        if not excluded:
            return None
        return QualityEvidence(previous_targets=frozenset(excluded))

    def _requested_genre_keys(
        self, payload: Mapping[str, Any]
    ) -> frozenset[str] | None:
        """Canonicalized genre keys of a directed request, or None for an undirected one."""
        genres = payload.get("genres")
        if genres is None:
            return None
        keys = {canonicalize_genre_key(genre) for genre in genres}
        keys.discard("")
        return frozenset(keys) if keys else None

    def _direction_filtered_target_ids(
        self, payload: Mapping[str, Any]
    ) -> list[str]:
        """A directed request keeps only targets whose canonical genres match the direction.

        Input-side candidate narrowing only: scoring and ranking are untouched. A target the
        store cannot verify (missing genre data) is dropped from a directed request -- the
        direction is a hard filter, never a soft hint.
        """
        keys = self._requested_genre_keys(payload)
        target_ids = list(payload["target_ids"])
        if not keys:
            return target_ids
        model = self._canonical.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        filtered: list[str] = []
        for target_id in target_ids:
            track = track_by_id.get(target_id)
            if track is None:
                continue
            track_keys = {
                canonicalize_genre_key(genre) for genre in track.get("genres", ())
            }
            if keys & track_keys:
                filtered.append(target_id)
        return filtered

    def _direction_filtered_tracks(
        self, payload: Mapping[str, Any], tracks: Iterable[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """Narrow a catalog-track pool to the direction's genres (input-side filter)."""
        keys = self._requested_genre_keys(payload)
        result = list(tracks)
        if not keys:
            return result
        return [
            track
            for track in result
            if keys
            & {canonicalize_genre_key(genre) for genre in track.get("genres", ())}
        ]

    def _execute_generate_recommendation(
        self,
        payload: Mapping[str, Any],
        fresh_canonical_ids: tuple[str, ...] = (),
        recommendation_scope_ids: tuple[str, ...] | None = None,
        produced_at: datetime | None = None,
    ) -> dict[str, Any]:
        source_system = payload.get("source_system") or DEFAULT_SOURCE_SYSTEM
        # P15 burn-down Issue 1: the durable run time is service-authoritative
        # (the execute-path ``completed_dt`` passed through the trusted context
        # seam). The model's payload has no produced_at key (rejected at the
        # validation boundary), and this handler never consults one.
        produced_at = produced_at or datetime.now(timezone.utc)
        filtered_target_ids = self._direction_filtered_target_ids(payload)
        if recommendation_scope_ids is not None:
            allowed = frozenset(recommendation_scope_ids)
            filtered_target_ids = [
                target_id for target_id in filtered_target_ids if target_id in allowed
            ]
        preference_inputs: list[PreferenceInput] = []
        direct_states: list[TrackPreferenceState] = []
        for target_id in filtered_target_ids:
            state = query_track_preference(
                self._preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
                rating_policy=self._rating_policy,
                magnitude_policy=self._magnitude_policy,
                familiarity_policy=self._familiarity_policy,
                source_system=source_system,
            )
            direct_states.append(state)
            preference_inputs.append(PreferenceInput.from_direct(state.direct_preference))
        # P16-S2: deterministic routing out of the provably-empty plain funnel.
        # The plain channel's candidate pool is exactly one candidate per
        # POSITIVE direct state (negative states yield rejected candidates,
        # non-directional states yield none), so with zero positive states
        # ``build_recommendation`` below MUST return empty -- the funnel facts
        # are complete before the expensive build runs. Under the old contract
        # the model then re-called generate_inferred_recommendation with the
        # same payload (the M2-2 envelope advised exactly that): one wasted
        # empty build plus one wasted provider round per request. The service
        # now routes straight to the inferred channel and returns its result,
        # annotated with ``channel=inferred_fallback``; the delivered items are
        # what the inferred tool returns for the same payload, so
        # recommendation semantics are unchanged -- only the wasted rounds are.
        # Two cases keep the error envelope instead (never a fallback):
        # * the direction filter emptied every target -- the envelope's remedy
        #   (relax genres) is a model decision, not a deterministic retry;
        # * same-run Fresh provenance exists (the loop captured promoted
        #   discoveries this run) -- fresh requests route through
        #   generate_inferred_recommendation WITH min_fresh, and a fallback
        #   without it would silently drop the Fresh intent (S3D/S3E). The
        #   plain error keeps correcting an accidental plain call there, so
        #   Fresh behavior stays byte-identical.
        positive_count = sum(
            1
            for state in direct_states
            if state.direct_preference.strength.state is PreferenceState.POSITIVE
        )
        negative_count = sum(
            1
            for state in direct_states
            if state.direct_preference.strength.state is PreferenceState.NEGATIVE
        )
        if filtered_target_ids and positive_count == 0 and not fresh_canonical_ids:
            fallback = self._execute_generate_inferred_recommendation(
                payload,
                fresh_canonical_ids=fresh_canonical_ids,
                recommendation_scope_ids=recommendation_scope_ids,
                produced_at=produced_at,
            )
            fallback["channel"] = "inferred_fallback"
            return fallback
        request = RecommendationRequest(
            RecommendationContext(produced_at, tuple(preference_inputs)),
            RecommendedItemKind.TRACK,
            payload["limit"],
        )
        # P19-T15: the plain channel's candidate universe is exactly the
        # preference-driven targets (positive direct states); inject their
        # persisted recommendation counts (absent = zero, "no memory") so
        # score-tied candidates surface never/rarely-recommended tracks first.
        recommendation_counts = {
            canonical_id: state.recommendation_count
            for canonical_id, state in self._catalog_track_state.load_states(
                filtered_target_ids
            ).items()
        }
        run_id = self._run_id_factory()
        quality_evidence = self._repeat_quality_evidence(payload)
        outcome = build_recommendation(
            request,
            run_id=run_id,
            produced_at=produced_at,
            quality_evidence=quality_evidence,
            recommendation_counts=recommendation_counts,
        )
        if not outcome.result.items:
            # P14-R2: an empty run never persists -- it would pollute history and
            # could become the active batch pointer. The refusal is fed back to
            # the model as a tool error so it can relax criteria and retry.
            # P15-S4-M2-2: the error carries this execution's real funnel counts
            # so the loop can act on the actual cause instead of guessing.
            # P16-S2: this branch is reachable now only for positive-evidence
            # targets (exclusion/quality emptiness) plus the direction-filter
            # and fresh-run signatures, which provably can never enter the
            # deterministic inferred fallback above -- their funnel counts are
            # the same values computed before the build.
            exclusion_count, excluded_previous_count = _quality_exclusion_counts(
                outcome.quality_report
            )
            raise self._empty_recommendation_error_type(
                diagnostics=_empty_generation_diagnostics(
                    input_target_count=len(payload["target_ids"]),
                    after_direction_filter_count=len(filtered_target_ids),
                    positive_evidence_count=positive_count,
                    negative_evidence_count=negative_count,
                    exclusion_count=exclusion_count,
                    excluded_previous_count=excluded_previous_count,
                )
            )
        # P20 Fix07: same-batch sibling suppression at the delivery layer only
        # (see sibling_dedupe.py) -- never touches catalog entities, canonical
        # identity, or catalog_track_state. The model load below was previously
        # lazy after save; it is moved ahead of persistence so the suppression
        # lookup and the result summary share one load (no extra reads).
        model = self._canonical.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        album_by_id = {album["id"]: album for album in model["albums"]}
        if has_sibling_duplicate(outcome.result.items, track_by_id):
            # Backfill pool: the plain channel's complete eligible list is at
            # most one candidate per TRACK input, so the scan bound is exact.
            pool_items = outcome.result.items
            scan_limit = _exploration_scan_limit(preference_inputs, ())
            if scan_limit > request.limit:
                wide_outcome = build_recommendation(
                    RecommendationRequest(
                        request.context, request.recommended_kind, scan_limit
                    ),
                    run_id=run_id,
                    produced_at=produced_at,
                    quality_evidence=quality_evidence,
                    recommendation_counts=recommendation_counts,
                )
                pool_items = wide_outcome.result.items
            selected = select_distinct_works(pool_items, track_by_id, request.limit)
            outcome = RankingOutcome(
                result=assemble_recommendation_result(
                    request, selected, run_id=run_id, produced_at=produced_at
                ),
                quality_report=outcome.quality_report,
            )
        self._recommendation_history.save_result(outcome.result)
        # P15-S3-S2: runtime inverse of the 0019 backfill B -- project this run's
        # persisted track items into catalog_track_state over the same口径 (items
        # only, existing catalog rows only). Loud, never swallowed. The contract
        # enforces target-kind homogeneity within a run, so a TRACK run's items are
        # track items (mirrors B's kind='track' item scope).
        if outcome.result.request.recommended_kind is RecommendedItemKind.TRACK:
            self._catalog_track_state.record_recommendation_items(
                [item.candidate.target.target_id for item in outcome.result.items],
                produced_at=outcome.result.produced_at,
            )
        # P14-C07.3: a durably saved run becomes the active batch pointer (with a clean
        # item cursor) -- the pointer never names a run that history does not hold.
        self._active_context.note_recommendation_batch(outcome.result.run_id)
        # P15-S3-S3D: per-item same-run Fresh provenance (additive result keys only,
        # computed from the authoritative loop-captured set -- never from labels or
        # sources). ``min_fresh`` itself is rejected for the plain tool at the
        # validation boundary; the plain path only ever REPORTS provenance.
        fresh_ids = frozenset(fresh_canonical_ids)
        # P20-Fix09: the generation result now carries each item's durable
        # evidence projection -- the SAME shared builder the run detail reader
        # uses, resolved from the same primitives (source_path / basis_targets /
        # the run's own preference_inputs), so the first presentation of this
        # batch and a later "为什么推荐这些？" read one fact source. Additive
        # result key only: no schema, payload-envelope, or persistence change.
        inputs_by_target = self._preference_inputs_by_target(
            outcome.result.request.context.preference_inputs
        )
        items = [
            {
                **self._item_playback_summary(
                    item, track_by_id, artist_by_id, album_by_id
                ),
                "fresh_this_request": item.candidate.target.target_id in fresh_ids,
                "evidence": self._item_evidence_entry(
                    item, track_by_id, artist_by_id, inputs_by_target
                ),
            }
            for item in outcome.result.items
        ]
        return {
            "run_id": outcome.result.run_id,
            "item_count": len(outcome.result.items),
            "fresh_item_count": sum(
                1
                for item in outcome.result.items
                if item.candidate.target.target_id in fresh_ids
            ),
            "items": items,
            "encoded_result": encode_recommendation_result(outcome.result),
        }

    def _execute_true_similarity_recommendation(
        self,
        payload: Mapping[str, Any],
        *,
        similarity_context: SimilarityExecutionContext,
        produced_at: datetime,
        fresh_canonical_ids: tuple[str, ...],
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
        album_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the seed-relative V1 branch over authoritative canonical tracks.

        The internal ``similarity_context`` is the only seed authority.  Public
        ``target_ids`` and direction arguments cannot replace it or narrow the
        code-owned candidate universe.  Only tracks with an existing executable
        library/preview route enter that universe; the seed, negative preferences,
        recent runs, and metadata-ineligible tracks are then removed fail closed.
        """

        seed_id = similarity_context.seed_canonical_id
        seed = track_by_id.get(seed_id)
        if seed is None:
            raise self._similarity_seed_unavailable_error_type(
                f"strict similarity seed {seed_id} is not a canonical Track"
            )
        source_system = payload.get("source_system") or DEFAULT_SOURCE_SYSTEM
        candidate_tracks = [
            track
            for track in track_by_id.values()
            if self._playback_annotation(track)["route"] != "unavailable"
        ]
        seed_ref = PreferenceTargetReference(PreferenceTargetKind.TRACK, seed_id)
        preference_inputs: list[PreferenceInput] = []
        preference_tiebreaks: dict[str, float] = {}
        states_by_id: dict[str, TrackPreferenceState] = {}
        evidence_by_id: dict[str, TrackSimilarityEvidence] = {}
        items: list[RecommendationItem] = []
        negative_veto_count = 0
        ineligible_count = 0

        for track in sorted(candidate_tracks, key=lambda entry: entry["id"]):
            target_id = track["id"]
            state = query_track_preference(
                self._preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
                rating_policy=self._rating_policy,
                magnitude_policy=self._magnitude_policy,
                familiarity_policy=self._familiarity_policy,
                source_system=source_system,
            )
            states_by_id[target_id] = state
            if state.direct_preference.strength.state in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                preference_inputs.append(
                    PreferenceInput.from_direct(state.direct_preference)
                )
            if target_id == seed_id:
                continue
            if state.direct_preference.strength.state is PreferenceState.NEGATIVE:
                negative_veto_count += 1
                continue
            evidence = score_track_similarity(seed, track)
            if not evidence.categorical_eligible:
                ineligible_count += 1
                continue
            evidence_by_id[target_id] = evidence
            preference_tiebreaks[target_id] = (
                state.direct_preference.strength.magnitude
                if state.direct_preference.strength.state is PreferenceState.POSITIVE
                else 0.0
            )
            items.append(
                RecommendationItem(
                    Candidate(
                        candidate_id=generate_candidate_id(),
                        target=PreferenceTargetReference(
                            PreferenceTargetKind.TRACK, target_id
                        ),
                        source=CandidateSourceReference(
                            "music_agent", SIMILARITY_SOURCE_PATH
                        ),
                        # The strict seed is persisted as the factual basis role;
                        # it is not a preference claim and is never scored as one.
                        basis_targets=(seed_ref,),
                    ),
                    evidence.score_breakdown,
                )
            )

        context = RecommendationContext(produced_at, tuple(preference_inputs))
        request = RecommendationRequest(
            context, RecommendedItemKind.TRACK, payload["limit"]
        )
        quality_payload = dict(payload)
        excluded = [
            value
            for value in (quality_payload.get("exclude_target_ids") or ())
            if isinstance(value, str) and value
        ]
        quality_payload["exclude_target_ids"] = list(
            dict.fromkeys((*excluded, seed_id))
        )
        quality_payload["avoid_previous_runs"] = True
        quality_evidence = self._repeat_quality_evidence(quality_payload)
        recent_results = self._recommendation_history.list_runs(
            limit=_AVOID_PREVIOUS_RUNS_WINDOW
        )
        historical_target_ids = tuple(
            target.target_id
            for target in collect_previous_targets(recent_results)
            if target.kind is PreferenceTargetKind.TRACK
        )
        run_id = self._run_id_factory()

        # Rank the full eligible pool once so sibling suppression can backfill
        # from this same seed-relative universe without invoking a generic
        # generator or widening discovery.
        rank_request = RecommendationRequest(
            context,
            RecommendedItemKind.TRACK,
            max(1, len(items)),
        )
        ranked = rank_recommendations(
            rank_request,
            items,
            run_id=run_id,
            produced_at=produced_at,
            quality_evidence=quality_evidence,
            preference_tiebreaks=preference_tiebreaks,
        )
        history_filtered = exclude_historical_siblings(
            ranked.result.items,
            track_by_id,
            historical_target_ids,
        )
        historical_sibling_exclusion_count = (
            len(ranked.result.items) - len(history_filtered)
        )
        selected = select_distinct_works(
            history_filtered, track_by_id, request.limit
        )
        outcome = RankingOutcome(
            result=assemble_recommendation_result(
                request, selected, run_id=run_id, produced_at=produced_at
            ),
            quality_report=ranked.quality_report,
        )
        if not outcome.result.items:
            exclusion_count, excluded_previous_count = _quality_exclusion_counts(
                outcome.quality_report
            )
            raise self._empty_recommendation_error_type(
                diagnostics={
                    "reason": "no_eligible_seed_relative_candidates",
                    "seed_canonical_id": seed_id,
                    "candidate_pool_count": len(candidate_tracks),
                    "metadata_ineligible_count": ineligible_count,
                    "negative_veto_count": negative_veto_count,
                    "exclusion_count": exclusion_count,
                    "excluded_previous_count": excluded_previous_count,
                    "excluded_historical_sibling_count": (
                        historical_sibling_exclusion_count
                    ),
                    "requested_count": request.limit,
                    "recommended_next_action": "return_honest_empty",
                }
            )

        self._recommendation_history.save_result(outcome.result)
        self._catalog_track_state.record_recommendation_items(
            [item.candidate.target.target_id for item in outcome.result.items],
            produced_at=outcome.result.produced_at,
        )
        self._active_context.note_recommendation_batch(outcome.result.run_id)
        inputs_by_target = self._preference_inputs_by_target(
            outcome.result.request.context.preference_inputs
        )
        fresh_ids = frozenset(fresh_canonical_ids)
        projected_items: list[dict[str, Any]] = []
        for item in outcome.result.items:
            target_id = item.candidate.target.target_id
            track = track_by_id[target_id]
            state = states_by_id[target_id]
            evidence = evidence_by_id[target_id]
            artist_names = [
                artist_by_id[artist_id]["name"]
                for artist_id in track["artist_ids"]
                if artist_id in artist_by_id
            ]
            projected_items.append(
                {
                    "target_id": target_id,
                    "name": track["name"],
                    "artist_name": ", ".join(artist_names) if artist_names else None,
                    "album": (
                        (album_by_id.get(track["album_id"]) or {}).get("name")
                        if track.get("album_id") is not None
                        else None
                    ),
                    "direct_state": state.direct_preference.strength.state.value,
                    "explanation": self._similarity_explanation(evidence, artist_by_id),
                    "label": "similarity",
                    "score_total": item.score.total,
                    "playback": self._playback_annotation(track),
                    "fresh_this_request": target_id in fresh_ids,
                    "evidence": self._item_evidence_entry(
                        item, track_by_id, artist_by_id, inputs_by_target
                    ),
                }
            )
        return {
            "run_id": outcome.result.run_id,
            "item_count": len(outcome.result.items),
            "fresh_item_count": sum(
                1
                for item in outcome.result.items
                if item.candidate.target.target_id in fresh_ids
            ),
            "source_system": source_system,
            "similarity_seed_canonical_id": seed_id,
            "items": projected_items,
            "encoded_result": encode_recommendation_result(outcome.result),
        }

    @staticmethod
    def _similarity_explanation(
        evidence: TrackSimilarityEvidence,
        artist_by_id: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Compact deterministic description of shared categorical metadata."""

        labels: list[str] = []
        if evidence.shared_genres:
            labels.append("genre=" + "/".join(evidence.shared_genres))
        if evidence.shared_artist_ids:
            names = [
                artist_by_id[artist_id]["name"]
                for artist_id in evidence.shared_artist_ids
                if artist_id in artist_by_id
            ]
            labels.append("artist=" + "/".join(names or evidence.shared_artist_ids))
        if evidence.shared_composer is not None:
            labels.append("composer=" + evidence.shared_composer)
        if evidence.shared_tags:
            labels.append("tag=" + "/".join(evidence.shared_tags))
        return "与 seed 共享 " + "；".join(labels)

    def _execute_generate_inferred_recommendation(
        self,
        payload: Mapping[str, Any],
        fresh_canonical_ids: tuple[str, ...] = (),
        recommendation_scope_ids: tuple[str, ...] | None = None,
        produced_at: datetime | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> dict[str, Any]:
        """P10.17b: source-scoped direct inputs + genre-inferred inputs (frozen policy).

        Strictly single-source: direct inputs come from the sealed per-source query,
        inferred inputs come ONLY from that same source's directional evidence reduced
        by the P10 genre-affinity reducer. No cross-source merging exists anywhere.

        P15-S3-S3D: ``fresh_canonical_ids`` is the authoritative same-run Fresh
        provenance captured by the provider loop (PROMOTED entries of genuinely
        executed discoveries). It is NEVER supplied through the model payload --
        the tool schema has no key for it and validation would reject one. Driven
        by it: the best-effort ``min_fresh`` final-selection floor and the
        per-item ``fresh_this_request`` / batch ``fresh_item_count`` result keys.
        """
        from music_agent.artist_affinity import (
            SourcedArtistContribution,
            build_artist_affinities,
        )
        from music_agent.catalog_candidate_generation import generate_catalog_candidates
        from music_agent.catalog_recommendation import EvenSplitPolicy, affinity_inputs
        from music_agent.genre_affinity import (
            GenreAffinityPolicy,
            SourcedContribution,
            build_genre_affinities,
            infer_track_affinity,
        )
        from music_agent.preference_propagation import (
            PropagationKind,
            propagate_track_preference,
        )
        from music_agent.exploration_selection import (
            apply_exploration_floor,
            fresh_membership_predicate,
            is_catalog_exploration,
            is_fresh_driven,
        )
        from music_agent.fresh_candidate_generation import (
            FRESH_CANDIDATE_SOURCE_PATH,
            generate_fresh_candidates,
        )

        source_system = payload.get("source_system") or DEFAULT_SOURCE_SYSTEM
        # P15 burn-down Issue 1: run time is service-authoritative (trusted
        # completed_dt context seam); there is no payload path for produced_at.
        produced_at = produced_at or datetime.now(timezone.utc)
        target_ids = self._direction_filtered_target_ids(payload)
        if recommendation_scope_ids is not None:
            allowed = frozenset(recommendation_scope_ids)
            target_ids = [target_id for target_id in target_ids if target_id in allowed]
        model = self._canonical.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        # P18-S1.3: album names power the same version/source disambiguation the
        # plain generate response gets (single vs album release display fact).
        album_by_id = {album["id"]: album for album in model["albums"]}
        if similarity_context is not None:
            return self._execute_true_similarity_recommendation(
                payload,
                similarity_context=similarity_context,
                produced_at=produced_at,
                fresh_canonical_ids=fresh_canonical_ids,
                track_by_id=track_by_id,
                artist_by_id=artist_by_id,
                album_by_id=album_by_id,
            )
        catalog_tracks = self._direction_filtered_tracks(
            payload,
            [
                track
                for track in model["tracks"]
                if track["external_ids"].get("apple_music_catalog_id")
                or track["external_ids"].get("itunes_store_id")
            ],
        )
        if recommendation_scope_ids is not None:
            catalog_tracks = [
                track for track in catalog_tracks if track["id"] in allowed
            ]

        direct_states: dict[str, TrackPreferenceState] = {}
        direct_inputs: list[PreferenceInput] = []
        sourced_contributions: list[SourcedContribution] = []
        artist_contributions: list[SourcedArtistContribution] = []
        split_policy = EvenSplitPolicy()
        for target_id in target_ids:
            state = query_track_preference(
                self._preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
                rating_policy=self._rating_policy,
                magnitude_policy=self._magnitude_policy,
                familiarity_policy=self._familiarity_policy,
                source_system=source_system,
            )
            direct_states[target_id] = state
            direct_inputs.append(PreferenceInput.from_direct(state.direct_preference))
            if state.direct_preference.strength.state in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                track = track_by_id.get(target_id)
                artist_ids = track["artist_ids"] if track is not None else []
                genres = track["genres"] if track is not None else []
                for contribution in propagate_track_preference(
                    state.direct_preference,
                    artist_ids=artist_ids,
                    genres=genres,
                    artist_split=split_policy,
                    genre_split=split_policy,
                ):
                    if contribution.kind is PropagationKind.GENRE:
                        sourced_contributions.append(
                            SourcedContribution(source_system, contribution)
                        )
                    elif contribution.kind is PropagationKind.ARTIST:
                        artist_contributions.append(
                            SourcedArtistContribution(source_system, contribution)
                        )
        # P12-02: every canonical track contributes its own directional direct evidence
        # to the affinity reducers, including local-library tracks bound only by
        # apple_music_persistent_id. Only catalog-bound tracks can become catalog
        # candidates; widening the contribution scan never makes a local-only track
        # a candidate.
        for track in model["tracks"]:
            target_id = track["id"]
            if target_id in direct_states:
                continue
            state = query_track_preference(
                self._preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
                rating_policy=self._rating_policy,
                magnitude_policy=self._magnitude_policy,
                familiarity_policy=self._familiarity_policy,
                source_system=source_system,
            )
            if state.direct_preference.strength.state not in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                continue
            artist_ids = track["artist_ids"] if track is not None else []
            genres = track["genres"] if track is not None else []
            for contribution in propagate_track_preference(
                state.direct_preference,
                artist_ids=artist_ids,
                genres=genres,
                artist_split=split_policy,
                genre_split=split_policy,
            ):
                if contribution.kind is PropagationKind.GENRE:
                    sourced_contributions.append(
                        SourcedContribution(source_system, contribution)
                    )
                elif contribution.kind is PropagationKind.ARTIST:
                    artist_contributions.append(
                        SourcedArtistContribution(source_system, contribution)
                    )

        affinities = build_genre_affinities(
            sourced_contributions, GenreAffinityPolicy()
        )
        artist_affinities = build_artist_affinities(
            artist_contributions, GenreAffinityPolicy()
        )
        inferred_inputs: list[PreferenceInput] = []
        inferred_map: dict[str, str] = {}
        for target_id in target_ids:
            state = direct_states[target_id]
            if state.direct_preference.strength.state in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                continue  # directional direct evidence rules the target
            track = track_by_id.get(target_id)
            genres = track["genres"] if track is not None else []
            strength = infer_track_affinity(
                target_id, genres, source_system, affinities, GenreAffinityPolicy()
            )
            if strength is not None:
                inferred_inputs.append(
                    PreferenceInput.from_inferred(
                        InferredAffinity(
                            PreferenceTargetReference(
                                PreferenceTargetKind.TRACK, target_id
                            ),
                            strength,
                        )
                    )
                )
                inferred_map[target_id] = str(strength.state.value)

        # P11.2: unified affinity inputs (GENRE + ARTIST) feed catalog candidates.
        affinity_input_list = affinity_inputs(
            affinities, artist_affinities, GenreAffinityPolicy()
        )
        context_inputs = (
            tuple(direct_inputs) + tuple(inferred_inputs) + affinity_input_list
        )
        context = RecommendationContext(produced_at, context_inputs)
        catalog_candidates = ()
        if catalog_tracks:
            catalog_candidates = generate_catalog_candidates(context, catalog_tracks)
        # P19-T15: inject the persisted recommendation-count tie-break authority
        # for this execution's whole candidate universe (preference targets +
        # catalog pool; fresh candidates are a subset of the catalog pool).
        # Absent rows count as zero ("no memory"), so unknown tracks order as
        # never-recommended within their score tier.
        recommendation_counts = {
            canonical_id: state.recommendation_count
            for canonical_id, state in self._catalog_track_state.load_states(
                list(target_ids) + [track["id"] for track in catalog_tracks]
            ).items()
        }
        # P15-S3-S3E: the additive Fresh candidate channel. Activation is the
        # explicit-intent gate ONLY -- ``min_fresh > 0`` with a non-empty
        # authoritative same-run promoted set; an ordinary run (knob omitted/0,
        # or no executed discovery this run) never enters the channel, so its
        # candidates, scores, ordering, and result keys stay byte-equivalent.
        # Fresh candidates carry the dedicated ``fresh_driven`` source and an
        # EMPTY basis (zero preference claim): the existing dedupe keeps a real
        # preference/catalog candidate for the same target (stronger evidence
        # wins), while Fresh identity stays target-membership -- never source.
        min_fresh = payload.get("min_fresh") or 0
        fresh_ids = frozenset(fresh_canonical_ids)
        fresh_generated: tuple[Candidate, ...] = ()
        fresh_candidate_count = 0
        fresh_negative_rejected_count = 0
        if min_fresh > 0 and fresh_ids:
            # Fail-closed veto completeness: the module vetoes only what the
            # context carries, and the shared propagation surface can mask an
            # explicit dislike on the promoted track itself (its genre may read
            # CONFLICT while the track is disliked). Feed each fresh track's
            # OWN direct state from the preference store into the context the
            # fresh generator sees, so an explicit dislike on the promoted
            # track always vetoes it. The module stays pure (context-driven)
            # and the shared context feeding the frozen catalog layer is never
            # touched.
            fresh_veto_inputs: list[PreferenceInput] = []
            for fresh_id in sorted(fresh_ids):
                if fresh_id not in track_by_id:
                    continue
                state = query_track_preference(
                    self._preference,
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, fresh_id),
                    rating_policy=self._rating_policy,
                    magnitude_policy=self._magnitude_policy,
                    familiarity_policy=self._familiarity_policy,
                    source_system=source_system,
                )
                if state.direct_preference.strength.state is PreferenceState.NEGATIVE:
                    fresh_veto_inputs.append(
                        PreferenceInput.from_direct(state.direct_preference)
                    )
            fresh_context = (
                RecommendationContext(
                    context.now,
                    context.preference_inputs + tuple(fresh_veto_inputs),
                )
                if fresh_veto_inputs
                else context
            )
            fresh_generated = generate_fresh_candidates(
                fresh_context, fresh_ids, catalog_tracks
            )
            fresh_candidate_count = sum(
                1
                for candidate in fresh_generated
                if candidate.eligibility is Eligibility.ELIGIBLE
            )
            fresh_negative_rejected_count = (
                len(fresh_generated) - fresh_candidate_count
            )
        extra_candidates = tuple(catalog_candidates) + fresh_generated

        request = RecommendationRequest(
            context,
            RecommendedItemKind.TRACK,
            payload["limit"],
        )
        # P15-S3-S3C: best-effort exploration floor, FINAL SELECTION ONLY. The floor
        # never touches scoring/eligibility/persistence: it widens the internal rank
        # request to the exact complete-list bound (no heuristic window), then swaps
        # the lowest-ranked Familiar places for the highest-ranked qualified catalog
        # exploration items below the visible limit -- original rank order preserved,
        # ``score.total`` and the ranking comparator untouched. ``min_exploration`` is
        # validated (0 <= v <= limit) at the tool boundary; default 0 reproduces the
        # pre-S3-S3C behavior exactly.
        # P15-S3-S3D: the fresh floor rides the same frozen primitive with a target-
        # membership predicate over the authoritative same-run promoted set, and runs
        # BEFORE the exploration floor. Asking for N fresh items means wanting at
        # least N exploration places, so the exploration floor gets
        # ``effective_min_exploration = max(min_exploration, min_fresh)``. A selected
        # fresh item counts toward the exploration floor (one place, never two
        # quotas) -- the exploration classification below IS the composite
        # "catalog-driven OR fresh-driven OR same-run-fresh", so a fresh
        # preference-driven target (the live Perfect-blue shape:
        # catalog_driven=False) and a zero-basis fresh-driven target alike
        # satisfy both floors from their single place instead of being
        # displaced by a catalog top-up.
        # Both floors are best effort over the complete rank: no score change, no
        # resurrection, original rank order preserved among the selected items.
        min_exploration = payload.get("min_exploration") or 0
        effective_min_exploration = max(min_exploration, min_fresh)
        fresh_pred = fresh_membership_predicate(fresh_ids)
        run_id = self._run_id_factory()
        quality_evidence = self._repeat_quality_evidence(payload)
        rank_request = request
        scan_limit = _exploration_scan_limit(context_inputs, extra_candidates)
        if effective_min_exploration > 0 and scan_limit > request.limit:
            rank_request = RecommendationRequest(
                request.context,
                request.recommended_kind,
                scan_limit,
            )
        outcome = build_recommendation(
            rank_request,
            run_id=run_id,
            produced_at=produced_at,
            quality_evidence=quality_evidence,
            extra_candidates=extra_candidates,
            recommendation_counts=recommendation_counts,
        )
        # P20 Fix07: captured BEFORE the floor re-assembly below -- the
        # floor-composed head discards the unselected tail, which the sibling
        # backfill needs as its residual pool.
        full_rank_items = outcome.result.items
        if (
            effective_min_exploration > 0
            and rank_request.limit > request.limit
            and outcome.result.items
        ):
            floored_items = outcome.result.items
            if min_fresh > 0:
                # P15-S3-S3D: fresh floor first (the more specific user intent),
                # via the SAME frozen selection primitive with the fresh predicate.
                fresh_floored = apply_exploration_floor(
                    floored_items,
                    is_exploration=fresh_pred,
                    floor=min_fresh,
                    limit=request.limit,
                )
                selected_ids = {item.candidate.candidate_id for item in fresh_floored}
                # Composition: fresh head + the untouched remainder in original rank
                # order; the exploration floor then reads it as head/tail and only
                # tops up when the fresh entries did not already satisfy it.
                floored_items = tuple(fresh_floored) + tuple(
                    item
                    for item in floored_items
                    if item.candidate.candidate_id not in selected_ids
                )
            floored = apply_exploration_floor(
                floored_items,
                is_exploration=lambda item: (
                    is_catalog_exploration(item)
                    or is_fresh_driven(item)
                    or fresh_pred(item)
                ),
                floor=effective_min_exploration,
                limit=request.limit,
            )
            # Re-assemble against the ORIGINAL request so the persisted run stays
            # self-describing at the caller-visible limit; quality_report is carried
            # through untouched (the floor performs no filtering).
            outcome = RankingOutcome(
                result=assemble_recommendation_result(
                    request, floored, run_id=run_id, produced_at=produced_at
                ),
                quality_report=outcome.quality_report,
            )
        if not outcome.result.items:
            # P14-R2: same empty-run refusal as generate_recommendation -- an empty
            # batch never reaches history or the active batch pointer.
            # P15-S4-M2-2: same real-funnel diagnostic envelope as the plain tool.
            exclusion_count, excluded_previous_count = _quality_exclusion_counts(
                outcome.quality_report
            )
            positive_count = sum(
                1
                for state in direct_states.values()
                if state.direct_preference.strength.state is PreferenceState.POSITIVE
            )
            negative_count = sum(
                1
                for state in direct_states.values()
                if state.direct_preference.strength.state is PreferenceState.NEGATIVE
            )
            inferred_positive_count = sum(
                1
                for value in inferred_map.values()
                if value == PreferenceState.POSITIVE.value
            )
            # P15-S3-S3A: Known-Catalog supply facts over this execution's own
            # local inputs (pool + durable memory + the frozen P11.2 candidate
            # verdicts). Pure recounting only -- no freshness verdict and no
            # trigger policy live here (that is S3-S3B's prompt/loop territory).
            known_supply = summarize_known_catalog_supply(
                catalog_tracks,
                states_by_track_id=self._catalog_track_state.load_states(
                    [track["id"] for track in catalog_tracks]
                ),
                # P15-S3-S3E: known-supply facts count only the real catalog-
                # driven verdicts -- a fresh-driven zero-basis discovery has no
                # preference-backed eligibility and must never inflate supply.
                candidates=catalog_candidates,
            )
            raise self._empty_recommendation_error_type(
                diagnostics=_empty_generation_diagnostics(
                    input_target_count=len(payload["target_ids"]),
                    after_direction_filter_count=len(target_ids),
                    positive_evidence_count=positive_count,
                    negative_evidence_count=negative_count,
                    inferred_positive_count=inferred_positive_count,
                    catalog_candidate_count=len(extra_candidates),
                    exclusion_count=exclusion_count,
                    excluded_previous_count=excluded_previous_count,
                    is_inferred=True,
                    known_catalog_track_count=known_supply.known_catalog_track_count,
                    known_never_recommended_count=known_supply.known_never_recommended_count,
                    known_eligible_count=known_supply.known_eligible_count,
                    known_rejected_count=known_supply.known_rejected_count,
                    known_never_recommended_eligible_count=(
                        known_supply.known_never_recommended_eligible_count
                    ),
                    fresh_promoted_count=len(fresh_ids) if min_fresh > 0 else None,
                    fresh_candidate_count=(
                        fresh_candidate_count if min_fresh > 0 else None
                    ),
                    fresh_negative_rejected_count=(
                        fresh_negative_rejected_count if min_fresh > 0 else None
                    ),
                )
            )
        # P20 Fix07: same-batch sibling suppression over the delivered head plus
        # this run's own rank residual. Backfill never widens beyond the
        # current-run pool and never triggers a second discovery (sec.10) --
        # an exhausted pool returns fewer items honestly.
        if has_sibling_duplicate(outcome.result.items, track_by_id):
            head_items = outcome.result.items
            if rank_request.limit > request.limit:
                # Floor-composed head: the residual is the unselected tail of
                # the complete rank captured before the floor re-assembly.
                head_ids = {item.candidate.candidate_id for item in head_items}
                residual = tuple(
                    item
                    for item in full_rank_items
                    if item.candidate.candidate_id not in head_ids
                )
            else:
                wide_outcome = build_recommendation(
                    RecommendationRequest(
                        request.context, request.recommended_kind, scan_limit
                    ),
                    run_id=run_id,
                    produced_at=produced_at,
                    quality_evidence=quality_evidence,
                    extra_candidates=extra_candidates,
                    recommendation_counts=recommendation_counts,
                )
                head_ids = {item.candidate.candidate_id for item in head_items}
                residual = tuple(
                    item
                    for item in wide_outcome.result.items
                    if item.candidate.candidate_id not in head_ids
                )
            selected = select_distinct_works(
                tuple(head_items) + residual, track_by_id, request.limit
            )
            outcome = RankingOutcome(
                result=assemble_recommendation_result(
                    request, selected, run_id=run_id, produced_at=produced_at
                ),
                quality_report=outcome.quality_report,
            )
        self._recommendation_history.save_result(outcome.result)
        # P15-S3-S2: same inverse projection as generate_recommendation -- the inferred
        # run's persisted track items update catalog_track_state over the backfill-B口径
        # (a TRACK run's items are track items by contract homogeneity).
        if outcome.result.request.recommended_kind is RecommendedItemKind.TRACK:
            self._catalog_track_state.record_recommendation_items(
                [item.candidate.target.target_id for item in outcome.result.items],
                produced_at=outcome.result.produced_at,
            )
        # P14-C07.3: same boundary as generate_recommendation -- the inferred run
        # replaces the active batch pointer with a clean item cursor.
        self._active_context.note_recommendation_batch(outcome.result.run_id)
        catalog_ids = {track["id"] for track in catalog_tracks}
        # P20-Fix09: same shared evidence projection as the plain channel --
        # resolved from the run context's own preference inputs by the SAME
        # builder the detail reader uses (one fact source for first
        # presentation and follow-up explanation). The legacy per-item
        # ``label``/``explanation``/``direct_state`` view keys stay untouched;
        # ``evidence`` is the Fix03-shaped authority the presentation rules
        # ground every reason in.
        inputs_by_target = self._preference_inputs_by_target(
            outcome.result.request.context.preference_inputs
        )
        items: list[dict[str, Any]] = []
        for item in outcome.result.items:
            target_id = item.candidate.target.target_id
            track = track_by_id.get(target_id)
            state = direct_states.get(target_id)
            direct_value = (
                state.direct_preference.strength.state.value if state is not None else "unknown"
            )
            if direct_value == "positive":
                label = "known_positive"
            elif target_id in inferred_map:
                label = "novel"
            elif target_id in catalog_ids:
                label = "catalog"
            else:
                label = "other"
            if label == "catalog":
                # P15-S3-S3E: a fresh-driven candidate is a zero-basis discovery --
                # it must never borrow a preference-match explanation it does not
                # have. The honest string states the real contract: surfaced only
                # because of the explicit exploration request, no preference
                # evidence yet.
                if (
                    item.candidate.source.source_path == FRESH_CANDIDATE_SOURCE_PATH
                    and not item.candidate.basis_targets
                ):
                    explanation = "本次目录搜索的新发现（暂无偏好匹配证据）"
                else:
                    explanation = self._catalog_basis_explanation(
                        item, source_system, affinities, artist_affinities
                    )
            else:
                explanation = (
                    f"直接证据（正面收藏）" if label == "known_positive" else
                    next(
                        (a.explanation() for a in affinities
                         if a.source_system == source_system and a.genre_key
                         in {str(g).strip() for g in (track["genres"] if track is not None else [])}
                         and abs(a.affinity) >= GenreAffinityPolicy().affinity_threshold),
                        "推断证据（见贡献明细）",
                    )
                )
            # P12: hydrate display names from the canonical Track -> Artist relation so
            # the compact item is self-contained for user-facing rendering.
            artist_names = (
                [artist_by_id[artist_id]["name"] for artist_id in track["artist_ids"] if artist_id in artist_by_id]
                if track is not None else []
            )
            items.append(
                {
                    "target_id": target_id,
                    "name": track["name"] if track is not None else None,
                    "artist_name": ", ".join(artist_names) if artist_names else None,
                    # P18-S1.3: display-only version/source fact (see S1.3 audit).
                    "album": (
                        (album_by_id.get(track["album_id"]) or {}).get("name")
                        if track is not None and track.get("album_id") is not None
                        else None
                    ),
                    "direct_state": direct_value,
                    "explanation": explanation,
                    "label": label,
                    "score_total": item.score.total,
                    "playback": self._playback_annotation(track),
                    # P15-S3-S3D: machine-visible same-run Fresh truth, computed from
                    # the authoritative loop-captured set -- never from labels,
                    # sources, or the model's own claims.
                    "fresh_this_request": target_id in fresh_ids,
                    # P20-Fix09: Fix03-shaped durable evidence block (see the
                    # plain-channel note above).
                    "evidence": self._item_evidence_entry(
                        item, track_by_id, artist_by_id, inputs_by_target
                    ),
                }
            )
        return {
            "run_id": outcome.result.run_id,
            "item_count": len(outcome.result.items),
            # P15-S3-S3D: batch-level Fresh count over the FINAL selected items --
            # never the discover promoted total, never a durable field.

            "fresh_item_count": sum(
                1
                for item in outcome.result.items
                if item.candidate.target.target_id in fresh_ids
            ),
            # P15-S3-S3E: additive Fresh-channel diagnostics -- emitted only when
            # the explicit-intent gate was open (min_fresh > 0), even with an
            # empty authoritative set, so zeros explain "the discover run did not
            # newly promote anything". Ordinary requests stay byte-equivalent.
            **(
                {
                    "fresh_promoted_count": len(fresh_ids),
                    "fresh_candidate_count": fresh_candidate_count,
                    "fresh_negative_rejected_count": fresh_negative_rejected_count,
                }
                if min_fresh > 0
                else {}
            ),
            "source_system": source_system,
            "items": items,
            "encoded_result": encode_recommendation_result(outcome.result),
        }

    @staticmethod
    def _catalog_basis_explanation(item, source_system, affinities, artist_affinities) -> str:
        """Strongest above-threshold affinity explanation for one catalog candidate's basis."""
        from music_agent.genre_affinity import GenreAffinityPolicy

        basis = {ref.target_id for ref in item.candidate.basis_targets}
        relevant = [
            affinity
            for affinity in (*affinities, *artist_affinities)
            if affinity.source_system == source_system
            and (
                getattr(affinity, "genre_key", None)
                or getattr(affinity, "artist_id", None)
            )
            in basis
            and abs(affinity.affinity) >= GenreAffinityPolicy().affinity_threshold
        ]
        if not relevant:
            return "推断证据（见贡献明细）"
        return max(relevant, key=lambda affinity: abs(affinity.affinity)).explanation()
