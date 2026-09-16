"""Provider-layer recommendation semantics and deterministic policy helpers.

This module is intentionally policy-only: it owns recommendation semantics,
normalization, identity gates, and deterministic recovery construction used by
the provider loop. It does not execute provider rounds or agent tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Mapping

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentToolOutcome
from music_agent.intent_router import (
    RecommendationTurnSemantics,
    TurnPlan,
    TurnTaskSurface,
    resolve_turn_plan,
)
from music_agent.provider_contract import ProviderError, ProviderToolCall

# Keep this local policy predicate aligned with the provider loop's generation
# family without introducing a reverse dependency on provider_agent.py.
_GENERATION_TOOL_NAMES: frozenset[str] = frozenset({
    "generate_recommendation",
    "generate_inferred_recommendation",
})

def _enforce_new_recommendation_freshness(
    turn: str | TurnPlan, call: ProviderToolCall
) -> ProviderToolCall:
    """Keep a new recommendation request on the service's recent-run exclusion.

    The provider may relax a failed generation by sending
    ``avoid_previous_runs:false``. That is valid for an explicit historical
    replay workflow, but contradicts a turn classified as a new recommendation:
    it creates a new run containing the old batch. Normalize only that one flag;
    candidate selection and ranking remain service-owned.
    """
    if call.name not in _GENERATION_TOOL_NAMES:
        return call
    turn_plan = turn if isinstance(turn, TurnPlan) else resolve_turn_plan(turn)
    if (
        turn_plan.task_surface is not TurnTaskSurface.RECOMMENDATION
        or turn_plan.fresh_discovery
    ):
        return call
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return call
    if not isinstance(arguments, dict) or arguments.get("avoid_previous_runs") is not False:
        return call
    arguments["avoid_previous_runs"] = True
    return ProviderToolCall(
        call_id=call.call_id,
        name=call.name,
        arguments=json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedRecommendationSemantics:
    mode: str
    target: str | None
    target_kind: str | None
    scene: str | None
    target_ids: tuple[str, ...]
    requested_count: int | None = None
    seed_source: str | None = None


def _entity_name_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s,，、/&·・]+", "", value).casefold()


def _artist_name_matches(value: object, target_key: str) -> bool:
    if not isinstance(value, str):
        return False
    return _entity_name_key(value) == target_key or any(
        _entity_name_key(part) == target_key
        for part in re.split(r"[,，、/&]+", value)
    )


def _resolve_recommendation_semantics(
    client: AgentClient, source: str | RecommendationTurnSemantics | None
) -> _ResolvedRecommendationSemantics | None:
    """Resolve an explicit turn target against the read-only canonical projection."""
    if isinstance(source, RecommendationTurnSemantics):
        semantics = source
    elif isinstance(source, str):
        # Compatibility entry point for direct callers while TurnPlan adoption
        # proceeds. Production run() passes the object from its single plan.
        from music_agent.intent_router import resolve_recommendation_turn_semantics

        semantics = resolve_recommendation_turn_semantics(source)
    else:
        semantics = None
    if semantics is None:
        return None
    if semantics.target is None:
        return _ResolvedRecommendationSemantics(
            mode=semantics.mode,
            target=None,
            target_kind=semantics.target_kind,
            scene=semantics.scene,
            target_ids=(),
            requested_count=semantics.requested_count,
            seed_source=semantics.seed_source,
        )
    try:
        result = client.call(
            "search_library_tracks", {"term": semantics.target, "limit": 200}
        )
    except Exception:
        result = None
    matches = (
        result.payload.get("matches", [])
        if result is not None
        and result.outcome is AgentToolOutcome.OK
        and isinstance(result.payload, Mapping)
        else []
    )
    target_key = _entity_name_key(semantics.target)
    artist_ids: list[str] = []
    track_ids: list[str] = []
    for entry in matches if isinstance(matches, list) else []:
        if not isinstance(entry, Mapping):
            continue
        target_id = entry.get("target_id")
        if not isinstance(target_id, str):
            continue
        if _artist_name_matches(entry.get("artist_name"), target_key):
            artist_ids.append(target_id)
        if _entity_name_key(entry.get("name")) == target_key:
            track_ids.append(target_id)
    if semantics.target_kind == "artist" or artist_ids:
        target_kind = "artist"
        target_ids = artist_ids
    elif track_ids:
        target_kind = "track"
        target_ids = track_ids
    else:
        target_kind = semantics.target_kind
        target_ids = []
    return _ResolvedRecommendationSemantics(
        semantics.mode,
        semantics.target,
        target_kind,
        semantics.scene,
        tuple(dict.fromkeys(target_ids)),
        semantics.requested_count,
        semantics.seed_source,
    )


def _recommendation_semantics_prompt(
    semantics: _ResolvedRecommendationSemantics | None,
) -> str:
    if semantics is None:
        return ""
    ids = ", ".join(semantics.target_ids) if semantics.target_ids else "尚未解析"
    scene = semantics.scene or "none"
    target = semantics.target or "none"
    seed_source = semantics.seed_source or "none"
    requested_count = semantics.requested_count or "unspecified"
    if semantics.mode == "generic":
        rule = (
            "这是无显式 seed 的通用推荐；current playback 只能作为上下文，"
            "不得被解释成用户指定的唯一推荐目标。"
        )
    elif (
        semantics.mode == "similarity_seed"
        and semantics.seed_source == "current_track"
    ):
        rule = (
            "这是当前播放曲目相似度请求；先从运行时上下文解析当前曲目的 canonical_id，"
            "再把它作为 similarity seed。TurnPlan 本身不预造该 identity。"
        )
    elif semantics.mode == "artist_constraint":
        rule = (
            "这是艺人硬约束；生成结果只能属于该艺人，不得用其他艺人凑数量。"
        )
    elif semantics.mode == "similarity_seed":
        rule = (
            "这是显式相似度 seed；它必须覆盖当前播放、最近推荐/搜索与默认上下文。"
        )
    else:
        rule = (
            "这是显式偏好 seed，不是艺人硬约束；允许其他艺人，但生成必须以该 seed 为首要依据。"
        )
    return (
        "\n\n<current_turn_recommendation_semantics>\n"
        f"mode={semantics.mode}; target={target}; seed_source={seed_source}; "
        f"target_kind={semantics.target_kind or 'unresolved'}; "
        f"scene={scene}; requested_count={requested_count}; "
        f"canonical_target_ids={ids}.\n"
        f"{rule} 若 scene 非 none，须与 seed/约束同时保留。"
        "不得用 current playback 替换本句显式 target。\n"
        "</current_turn_recommendation_semantics>"
    )


def _apply_recommendation_semantics(
    call: ProviderToolCall,
    semantics: _ResolvedRecommendationSemantics | None,
) -> ProviderToolCall:
    """Pin the seed without turning scene semantics into a Catalog literal."""
    if semantics is None:
        return call
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return call
    if not isinstance(arguments, dict):
        return call
    normalized_name = call.name
    if call.name in _GENERATION_TOOL_NAMES and semantics.requested_count is not None:
        arguments["limit"] = semantics.requested_count
    if call.name == "search_library_tracks":
        if semantics.target is None:
            return call
        arguments["term"] = semantics.target
    elif call.name == "discover_catalog_tracks":
        # Catalog discovery expands supply by the explicit seed.  Scene is a
        # recommendation semantic, not an iTunes Search field: appending a
        # phrase such as ``适合晚上听的歌`` makes the literal query so narrow
        # that a valid ``IU`` expansion returns no usable supply.  The scene
        # remains in the current-turn envelope for provider understanding and
        # presentation; the generator has no structured scene field today.
        if semantics.target is None:
            return call
        arguments["term"] = semantics.target
    elif call.name in _GENERATION_TOOL_NAMES:
        if semantics.target_ids:
            current_track_similarity = (
                semantics.mode == "similarity_seed"
                and semantics.target_kind == "track"
                and semantics.seed_source == "current_track"
            )
            if current_track_similarity:
                # The current track is evidence for the existing inferred
                # artist/genre candidate source, never candidate supply. Keep
                # any non-seed candidates the provider found through the
                # existing search/discovery tools, but pin the strict seed at
                # the front so it remains the affinity basis. The inferred
                # generator's canonical catalog pool supplies additional
                # non-seed candidates without a new recommendation engine.
                proposed = arguments.get("target_ids")
                proposed_non_seed = [
                    value
                    for value in (proposed if isinstance(proposed, list) else [])
                    if isinstance(value, str)
                    and value
                    and value not in semantics.target_ids
                ]
                arguments["target_ids"] = list(
                    dict.fromkeys((*semantics.target_ids, *proposed_non_seed))
                )
                # A direct generation over a positive singleton seed can only
                # recommend the seed itself. Route this one semantic case to
                # the existing inferred candidate source before execution.
                normalized_name = "generate_inferred_recommendation"
            else:
                arguments["target_ids"] = list(semantics.target_ids)
            # A concrete track used as a similarity seed is context, not a valid
            # answer to its own similarity request.  Enforce self-exclusion in
            # code so the provider cannot accidentally recommend the seed itself.
            # Artist seeds are deliberately NOT expanded into an artist-wide
            # exclusion: similar-to-an-artist may legitimately include that
            # artist, and that product decision is separate from track identity.
            if semantics.mode == "similarity_seed" and semantics.target_kind == "track":
                existing = arguments.get("exclude_target_ids")
                excluded = [
                    value
                    for value in (existing if isinstance(existing, list) else [])
                    if isinstance(value, str) and value
                ]
                arguments["exclude_target_ids"] = list(
                    dict.fromkeys((*excluded, *semantics.target_ids))
                )
                if current_track_similarity:
                    # Supplying an explicit exclude list otherwise disables
                    # the service's default recent-run window. Preserve that
                    # existing policy while making seed exclusion mandatory.
                    arguments["avoid_previous_runs"] = True
        elif semantics.requested_count is None:
            return call
    else:
        return call
    return ProviderToolCall(
        call_id=call.call_id,
        name=normalized_name,
        arguments=json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )


def _cached_current_player_canonical_id(
    results_cache: Mapping[tuple[str, str], str],
) -> str | None:
    """Read the player identity already observed this turn from cached tools."""
    for tool_name in ("get_active_context", "get_now_playing"):
        content = results_cache.get((tool_name, "{}"))
        if not isinstance(content, str):
            continue
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError:
            continue
        payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        if tool_name == "get_active_context":
            player = payload.get("player")
            candidate = (
                player.get("canonical_id") if isinstance(player, Mapping) else None
            )
        else:
            candidate = payload.get("player_canonical_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _cached_strict_current_player_canonical_id(
    results_cache: Mapping[tuple[str, str], str],
) -> str | None:
    """Read only a canonical identity proven by the current-player resolver.

    Similarity cannot use the transient agent-channel bookkeeping or an
    unqualified provider target as its seed. The two read tools expose the
    existing strict resolver's identity together with its proof class; absent
    or unresolved proof fails closed.
    """
    for tool_name in ("get_active_context", "get_now_playing"):
        content = results_cache.get((tool_name, "{}"))
        if not isinstance(content, str):
            continue
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError:
            continue
        payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        if tool_name == "get_active_context":
            player = payload.get("player")
            if not isinstance(player, Mapping):
                continue
            candidate = player.get("canonical_id")
            resolution = player.get("canonical_resolution")
        else:
            candidate = payload.get("player_canonical_id")
            resolution = payload.get("canonical_resolution")
        if (
            isinstance(candidate, str)
            and candidate
            and resolution in {"binding", "playback_equivalent"}
        ):
            return candidate
    return None


def _bind_current_track_similarity_seed(
    semantics: _ResolvedRecommendationSemantics | None,
    results_cache: Mapping[tuple[str, str], str],
) -> _ResolvedRecommendationSemantics | None:
    """Bind the abstract current-track seed to strict readback for this turn."""
    if (
        semantics is None
        or semantics.mode != "similarity_seed"
        or semantics.seed_source != "current_track"
    ):
        return semantics
    # The first strict observation binds this turn's seed. Later cache
    # invalidation (for example Catalog promotion) must not erase or silently
    # retarget that already-proven identity mid-turn.
    if semantics.target_ids:
        return semantics
    seed_id = _cached_strict_current_player_canonical_id(results_cache)
    if seed_id is None:
        return replace(semantics, target_ids=())
    return replace(semantics, target_kind="track", target_ids=(seed_id,))


def _generation_payload_contains_similarity_seed(
    payload: Mapping | None,
    semantics: _ResolvedRecommendationSemantics | None,
) -> bool:
    """Defensive terminal gate: a similarity seed can never be delivered."""
    if (
        not isinstance(payload, Mapping)
        or semantics is None
        or semantics.mode != "similarity_seed"
        or semantics.seed_source != "current_track"
        or semantics.target_kind != "track"
        or not semantics.target_ids
    ):
        return False
    items = payload.get("items")
    if not isinstance(items, list):
        return False
    seeds = frozenset(semantics.target_ids)
    return any(
        isinstance(item, Mapping) and item.get("target_id") in seeds
        for item in items
    )


def _generic_recommendation_targets_only_current_player(
    call: ProviderToolCall,
    semantics: _ResolvedRecommendationSemantics | None,
    results_cache: Mapping[tuple[str, str], str],
) -> bool:
    """Reject the ambient player being promoted into a generic sole seed."""
    if (
        semantics is None
        or semantics.mode != "generic"
        or call.name not in _GENERATION_TOOL_NAMES
    ):
        return False
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return False
    targets = arguments.get("target_ids") if isinstance(arguments, Mapping) else None
    if not isinstance(targets, list):
        return False
    unique_targets = tuple(
        dict.fromkeys(
            target for target in targets if isinstance(target, str) and target
        )
    )
    if len(unique_targets) != 1:
        return False
    current_player_id = _cached_current_player_canonical_id(results_cache)
    return current_player_id is not None and unique_targets[0] == current_player_id


def _generic_recommendation_result_only_current_player(
    payload: Mapping | None,
    semantics: _ResolvedRecommendationSemantics | None,
    current_player_id: str | None,
) -> bool:
    """Reject a final generic batch whose sole item is the strict current player."""
    if (
        not isinstance(payload, Mapping)
        or semantics is None
        or semantics.mode != "generic"
        or current_player_id is None
    ):
        return False
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 1:
        return False
    item = items[0]
    return isinstance(item, Mapping) and item.get("target_id") == current_player_id


def _empty_generation_diagnostics(
    *, error_code: str | None, error_message: str | None
) -> Mapping | None:
    """Parse the service's existing structured empty-generation diagnostics."""
    if error_code != "empty_recommendation" or not isinstance(error_message, str):
        return None
    marker = " diagnostics: "
    if marker not in error_message:
        return None
    raw = error_message.split(marker, 1)[1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _generic_direct_history_exhausted(
    call: ProviderToolCall,
    semantics: _ResolvedRecommendationSemantics | None,
    *,
    error_code: str | None,
    error_message: str | None,
) -> bool:
    """True only for direct generic emptiness caused by recent-run exclusion."""
    if (
        call.name != "generate_recommendation"
        or semantics is None
        or semantics.mode != "generic"
    ):
        return False
    diagnostics = _empty_generation_diagnostics(
        error_code=error_code, error_message=error_message
    )
    if diagnostics is None:
        return False
    return (
        diagnostics.get("reason") == "all_eligible_candidates_excluded"
        and isinstance(diagnostics.get("excluded_previous_count"), int)
        and diagnostics["excluded_previous_count"] > 0
    )


def _deterministic_generic_recovery_call(
    direct_call: ProviderToolCall,
    *,
    round_index: int,
    call_index: int,
) -> ProviderToolCall:
    """Reuse the existing inferred generator for the code-owned generic retry."""
    arguments = json.loads(direct_call.arguments)
    if not isinstance(arguments, dict):  # unreachable after real execution
        raise ProviderError("executed generation arguments must be a JSON object")
    arguments["avoid_previous_runs"] = True
    return ProviderToolCall(
        call_id=f"policy_generic_recovery_{round_index}_{call_index}",
        name="generate_inferred_recommendation",
        arguments=json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _deterministic_preference_fallback_calls(
    direct_call: ProviderToolCall,
    semantics: _ResolvedRecommendationSemantics,
    *,
    round_index: int,
    call_index: int,
) -> tuple[ProviderToolCall, ProviderToolCall]:
    """Build the policy-owned discovery + inferred retry pair.

    The Catalog query is deliberately seed-only.  The inferred retry starts
    from the actual normalized direct-generation arguments, preserving target
    ids, limit, exclusions, genres and every other supported field while
    pinning freshness on for the deterministic retry.
    """
    arguments = json.loads(direct_call.arguments)
    if not isinstance(arguments, dict):  # unreachable after real execution
        raise ProviderError("executed generation arguments must be a JSON object")
    arguments["avoid_previous_runs"] = True
    prefix = f"policy_preference_fallback_{round_index}_{call_index}"
    discovery = ProviderToolCall(
        call_id=f"{prefix}_discover",
        name="discover_catalog_tracks",
        arguments=json.dumps(
            {"limit": 10, "term": semantics.target},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    inferred = ProviderToolCall(
        call_id=f"{prefix}_inferred",
        name="generate_inferred_recommendation",
        arguments=json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return discovery, inferred


def _recommendation_scope_ids(
    semantics: _ResolvedRecommendationSemantics | None,
) -> tuple[str, ...] | None:
    """Only artist ownership is a hard output scope; seeds stay permissive."""
    if semantics is None or semantics.mode != "artist_constraint":
        return None
    return semantics.target_ids
