"""Code-owned authorization and selection for active recommendation actions.

The grant is a pure projection of an already-resolved ``TurnPlan`` and an
authoritative ``RecommendationRun`` payload.  It contains no provider proposal
and no execution lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from music_agent.intent_router import TurnPlan


_ROUTE_TO_TOOL = {
    "library": "play_track",
    "preview_only": "preview_catalog_track",
}


@dataclass(frozen=True, slots=True)
class AuthorizedSelectionItem:
    """One authoritative RecommendationRun item available to this turn."""

    position: int
    canonical_id: str
    playback_route: str | None
    title: str | None = None
    artist: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionGrant:
    """The bounded item/action authority derived for one delegated turn."""

    recommendation_run_id: str
    authorized_items: tuple[AuthorizedSelectionItem, ...]
    selection_mode: str
    allowed_actions: frozenset[str]
    max_audio_actions: int
    verified_canonical_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class DelegatedAudioAction:
    """An authoritative command description, not an execution attempt/result."""

    recommendation_run_id: str
    item_position: int
    canonical_id: str
    playback_route: str
    tool_name: str
    title: str | None = None
    artist: str | None = None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def build_selection_grant(
    turn_plan: TurnPlan,
    recommendation_payload: Mapping | None,
    *,
    verified_canonical_ids: frozenset[str] = frozenset(),
) -> SelectionGrant | None:
    """Project a delegated TurnPlan and exact RecommendationRun into authority.

    Invalid identities are omitted rather than guessed.  Unsupported or
    unresolved routes remain visible on authorized items, but cannot produce
    an executable action.
    """

    action = turn_plan.playback_action
    if (
        action is None
        or not turn_plan.delegated_action_authorized
        or action.source != "active_recommendation"
        or action.selection_mode not in {"agent_choose_one", "choose_another"}
        or not isinstance(recommendation_payload, Mapping)
        or not isinstance(verified_canonical_ids, frozenset)
        or not all(
            isinstance(canonical_id, str) and canonical_id
            for canonical_id in verified_canonical_ids
        )
    ):
        return None

    run_id = recommendation_payload.get("run_id")
    payload_items = recommendation_payload.get("items")
    if not isinstance(run_id, str) or not run_id or not isinstance(payload_items, list):
        return None

    authorized_items: list[AuthorizedSelectionItem] = []
    for ordinal, payload_item in enumerate(payload_items, start=1):
        if not isinstance(payload_item, Mapping):
            continue
        canonical_id = payload_item.get("target_id")
        if not isinstance(canonical_id, str) or not canonical_id:
            continue
        position = payload_item.get("position")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position <= 0
        ):
            position = ordinal
        playback = payload_item.get("playback")
        route = playback.get("route") if isinstance(playback, Mapping) else None
        if not isinstance(route, str) or not route:
            route = None
        authorized_items.append(
            AuthorizedSelectionItem(
                position=position,
                canonical_id=canonical_id,
                playback_route=route,
                title=_optional_text(payload_item.get("name")),
                artist=_optional_text(payload_item.get("artist_name")),
            )
        )

    if not authorized_items:
        return None
    allowed_actions = frozenset(
        _ROUTE_TO_TOOL[item.playback_route]
        for item in authorized_items
        if item.playback_route in _ROUTE_TO_TOOL
    )
    return SelectionGrant(
        recommendation_run_id=run_id,
        authorized_items=tuple(authorized_items),
        selection_mode=action.selection_mode,
        allowed_actions=allowed_actions,
        max_audio_actions=1,
        verified_canonical_ids=verified_canonical_ids,
    )


def select_delegated_audio_action(
    grant: SelectionGrant | None,
) -> DelegatedAudioAction | None:
    """Select the first executable authoritative item in stable run order."""

    if grant is None or grant.selection_mode not in {
        "agent_choose_one",
        "choose_another",
    }:
        return None
    for item in grant.authorized_items:
        if (
            grant.selection_mode == "choose_another"
            and item.canonical_id in grant.verified_canonical_ids
        ):
            continue
        tool_name = _ROUTE_TO_TOOL.get(item.playback_route)
        if tool_name is None:
            continue
        return DelegatedAudioAction(
            recommendation_run_id=grant.recommendation_run_id,
            item_position=item.position,
            canonical_id=item.canonical_id,
            playback_route=item.playback_route,
            tool_name=tool_name,
            title=item.title,
            artist=item.artist,
        )
    return None


def selection_grant_is_exhausted(grant: SelectionGrant | None) -> bool:
    """Whether choose-another consumed every executable item in this run."""
    if grant is None or grant.selection_mode != "choose_another":
        return False
    executable = tuple(
        item
        for item in grant.authorized_items
        if item.playback_route in _ROUTE_TO_TOOL
    )
    return bool(executable) and all(
        item.canonical_id in grant.verified_canonical_ids for item in executable
    )


def delegated_action_is_authorized(
    action: DelegatedAudioAction, grant: SelectionGrant | None
) -> bool:
    """Defensively prove that an action is exactly contained by its grant."""

    if (
        grant is None
        or grant.max_audio_actions < 1
        or action.recommendation_run_id != grant.recommendation_run_id
        or action.tool_name not in grant.allowed_actions
        or _ROUTE_TO_TOOL.get(action.playback_route) != action.tool_name
    ):
        return False
    return any(
        item.position == action.item_position
        and item.canonical_id == action.canonical_id
        and item.playback_route == action.playback_route
        for item in grant.authorized_items
    )
