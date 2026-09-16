"""P20 Quality Fix 05: the deterministic direction-shift executor.

``run_direction_shift(client, line)`` turns a recognized direction request into
one of three honest outcomes without any provider round:

- ``None`` -- the line is not a direction request, or a required durable read
  failed: the caller remits to its ordinary provider loop untouched (fail-safe;
  the model keeps handling near-misses and explicit-but-unmappable words);
- a ``"reply"`` result -- the request is a shift but no different real
  direction exists (or generation itself failed honestly): the caller prints
  exactly this text and must NOT generate anything;
- a ``"generated"`` result -- a real direction shift: one
  ``generate_inferred_recommendation`` call with ``genres=[new direction]``
  AND ``exclude_target_ids=previous batch targets`` (the previous batch stays
  excluded), the new run id and items for the caller to present.

Sources of truth -- never the model's own reading of song names:

- the previous batch's direction: the active batch's DURABLE genre basis
  (``get_active_context`` -> ``get_recommendation_run`` items' evidence);
- the user's real alternative directions AND the candidate scope: the
  service-level ``direction_shift_inputs`` (the sealed P10 genre-affinity
  reducer plus the model-wide positive-track pool, one durable pass) --
  the scope must be the full positive pool, never just the previous batch:
  the direction filter is a hard input-side genre cut, and a direction-pure
  batch scoped to itself would filter itself to zero candidates;
- the replacement pick: the deterministic ``pick_shifted_direction`` policy
  (strongest real alternative first, previous main direction(s) excluded,
  fail-honest when none survives).

The executor never fabricates a direction the user has no positive evidence
for, never reruns generation after a failure, and never touches Fresh /
catalog discovery / playback -- 「换个方向」 is not 「找新歌」.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

from music_agent.direction_shift import (
    DirectionRequest,
    NO_ALTERNATIVE_DIRECTION_REPLY,
    classify_direction_request,
    explicit_direction_note,
    pick_shifted_direction,
    shifted_direction_note,
)
from music_agent.provider_agent import RECOMMENDATION_UNFULFILLED_FALLBACK


def _is_ok_tool_result(result: Any) -> bool:
    """True when ``result`` is a successful agent tool result (defensive)."""
    if not hasattr(result, "outcome") or not hasattr(result, "payload"):
        return False
    try:
        from music_agent.agent_contract import AgentToolOutcome
    except Exception:
        return False
    return result.outcome is AgentToolOutcome.OK and isinstance(result.payload, Mapping)


def _active_batch_run_id(client: Any) -> str | None:
    """The active recommendation batch's run id (register or derived), or None."""
    result = client.call("get_active_context", {})
    if not _is_ok_tool_result(result):
        return None
    batch = result.payload.get("active_batch")
    if not isinstance(batch, Mapping):
        return None
    run_id = batch.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _batch_genre_counts(
    run_payload: Mapping[str, Any],
    genres_by_track: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """(genre basis counts, item target ids) of one run payload.

    Counts come from the run's DURABLE per-item evidence basis -- the same
    frozen primitives the explanation reader (P20 Fix03) resolves:

    * genre-kind basis labels are canonicalized and counted directly;
    * track-kind basis entries expose only display names on the tool surface,
      so the ITEM's own canonical genres (resolved through the service's
      ``canonical_genre_keys`` -- canonical keys, never invented) stand in
      as the direction that item presented. Without a resolver the track-kind
      half is skipped: the counts shrink instead of guessing.

    Items with no representation in the payload fail closed the same way.
    """
    from music_agent.preference_propagation import canonicalize_genre_key

    counts: dict[str, int] = {}
    targets: list[str] = []
    items = run_payload.get("items")
    if not isinstance(items, list):
        return {}, ()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        target_id = item.get("target_id")
        if isinstance(target_id, str) and target_id:
            targets.append(target_id)
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        basis = evidence.get("basis")
        if not isinstance(basis, list):
            continue
        has_track_basis = False
        for entry in basis:
            if not isinstance(entry, Mapping):
                continue
            kind = entry.get("kind")
            if kind == "genre":
                label = entry.get("label")
                if not isinstance(label, str) or not label:
                    continue
                key = canonicalize_genre_key(label)
                if key:
                    counts[key] = counts.get(key, 0) + 1
            elif kind == "track":
                has_track_basis = True
        if has_track_basis and target_id and genres_by_track is not None:
            for key in genres_by_track.get(target_id, ()):
                counts[key] = counts.get(key, 0) + 1
    return counts, tuple(targets)


def _track_basis_item_ids(run_payload: Mapping[str, Any]) -> set[str]:
    """Item target ids whose durable basis contains a track-kind entry."""
    ids: set[str] = set()
    items = run_payload.get("items")
    if not isinstance(items, list):
        return ids
    for item in items:
        if not isinstance(item, Mapping):
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        basis = evidence.get("basis")
        if not isinstance(basis, list):
            continue
        if any(
            isinstance(entry, Mapping) and entry.get("kind") == "track"
            for entry in basis
        ):
            target_id = item.get("target_id")
            if isinstance(target_id, str) and target_id:
                ids.add(target_id)
    return ids


def _genre_resolver(client: Any):
    """The service-level canonical genre reader, or None when unavailable."""
    service = getattr(client, "service", None)
    resolver = getattr(service, "canonical_genre_keys", None)
    return resolver if callable(resolver) else None


def _resolve_track_genres(
    resolver, track_ids: set[str]
) -> dict[str, tuple[str, ...]] | None:
    """Canonical genres for the track-basis items, or None on any failure."""
    if resolver is None or not track_ids:
        return None
    try:
        resolved = resolver(track_ids)
    except Exception:
        logger.exception("[direction-coach] track genre resolution failed")
        return None
    if not isinstance(resolved, Mapping):
        return None
    cleaned: dict[str, tuple[str, ...]] = {}
    for track_id, genres in resolved.items():
        if not isinstance(track_id, str) or not isinstance(genres, tuple):
            continue
        cleaned[track_id] = tuple(
            genre for genre in genres if isinstance(genre, str) and genre
        )
    return cleaned


def _direction_shift_truth(
    client: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The user's real (positive genre keys, positive track scope) from the
    service-level single-pass reader, or None when it is unavailable."""
    service = getattr(client, "service", None)
    reader = getattr(service, "direction_shift_inputs", None)
    if not callable(reader):
        return None
    try:
        truth = reader()
    except Exception:
        logger.exception("[direction-coach] direction truth read failed")
        return None
    if (
        not isinstance(truth, tuple)
        or len(truth) != 2
        or not isinstance(truth[0], tuple)
        or not isinstance(truth[1], tuple)
    ):
        return None
    genres = tuple(
        genre for genre in truth[0] if isinstance(genre, str) and genre
    )
    scope = tuple(
        track_id for track_id in truth[1] if isinstance(track_id, str) and track_id
    )
    return genres, scope


def _generate_shifted_batch(
    client: Any,
    *,
    selected: str,
    scope_targets: tuple[str, ...],
    previous_targets: tuple[str, ...],
    limit: int,
) -> Mapping[str, Any] | None:
    """One inferred generation pinned to the new direction, or None on failure.

    The candidate scope is the user's model-wide real positive tracks (the
    hard genre filter then slices the selected direction out of it); the
    previous batch's targets are the exclusion -- 「换个方向」 keeps excluding
    the previous batch. The tool's default short-term dedupe (recent 5
    batches) stays active. Any failure (empty scope, transport, ...) returns
    None -- the caller answers honestly instead of retrying or silently
    regenerating the old direction.
    """
    if not scope_targets:
        return None
    try:
        result = client.call(
            "generate_inferred_recommendation",
            {
                "target_ids": list(scope_targets),
                "genres": [selected],
                "exclude_target_ids": list(previous_targets),
                "limit": limit,
            },
        )
    except Exception:
        logger.exception("[direction-coach] shifted generation failed")
        return None
    if not _is_ok_tool_result(result):
        return None
    payload = result.payload
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    return payload


def run_direction_shift(client: Any, line: str) -> dict[str, Any] | None:
    """Execute one recognized direction request on the caller's surface.

    Returns ``None`` when nothing deterministic ran (remit to the caller's
    provider loop), else a result dict: ``{"kind": "reply", "text": ...}`` for
    a fail-honest answer, or ``{"kind": "generated", "genre", "note",
    "run_id", "item_count", "items"}`` for a real direction shift. The
    executor is fail-safe by construction: every unreadable/ambiguous state
    remits instead of guessing, and every generation failure collapses to the
    single honest fallback sentence -- it never claims a shift that did not
    happen.
    """
    request: DirectionRequest | None
    try:
        request = classify_direction_request(line)
    except Exception:
        return None
    if request is None:
        return None

    try:
        batch_run_id = _active_batch_run_id(client)
    except Exception:
        logger.exception("[direction-coach] active batch read failed")
        return None
    if batch_run_id is None:
        # No previous batch to shift FROM (or unreadable): the direction
        # request has no deterministic meaning here -- remit.
        return None

    try:
        run_result = client.call("get_recommendation_run", {"run_id": batch_run_id})
    except Exception:
        logger.exception("[direction-coach] recommendation run read failed")
        return None
    if not _is_ok_tool_result(run_result):
        return None
    genres_by_track = _resolve_track_genres(
        _genre_resolver(client), _track_basis_item_ids(run_result.payload)
    )
    batch_counts, previous_targets = _batch_genre_counts(
        run_result.payload, genres_by_track
    )
    if not previous_targets:
        return None
    limit = run_result.payload.get("item_count")
    if not isinstance(limit, int) or limit < 1:
        limit = len(previous_targets)

    truth = _direction_shift_truth(client)
    if truth is None:
        # The one lawful reader is unavailable: nothing deterministic can be
        # claimed, and a fixed "no alternative" answer could be wrong -- remit
        # so the provider loop (and its own rules) owns the turn.
        return None
    positive_genres, positive_scope = truth

    selected: str
    if request.kind == "explicit":
        # §八: the user's named direction wins outright -- no automatic
        # re-selection against the preference ranking. It still needs the real
        # positive scope (the direction filter is a hard genre cut); without
        # it the provider loop's direction-word rule is the honest surface.
        selected = request.genre or ""
        if not selected or not positive_scope:
            return None
    else:
        if not positive_genres:
            return {
                "kind": "reply",
                "text": NO_ALTERNATIVE_DIRECTION_REPLY,
            }
        decision = pick_shifted_direction(batch_counts, positive_genres)
        if decision.selected is None:
            return {
                "kind": "reply",
                "text": NO_ALTERNATIVE_DIRECTION_REPLY,
            }
        selected = decision.selected

    payload = _generate_shifted_batch(
        client,
        selected=selected,
        scope_targets=positive_scope,
        previous_targets=previous_targets,
        limit=limit,
    )
    if payload is None:
        if request.kind == "explicit":
            # §八: an explicit user-named direction keeps its old capable
            # surface -- the provider loop's direction-word rule handles it
            # with the full tool set. Remit; the failed attempt wrote nothing
            # (empty generations are never persisted).
            return None
        return {"kind": "reply", "text": RECOMMENDATION_UNFULFILLED_FALLBACK}
    note = (
        explicit_direction_note(selected)
        if request.kind == "explicit"
        else shifted_direction_note(selected)
    )
    return {
        "kind": "generated",
        "genre": selected,
        "note": note,
        "run_id": payload.get("run_id"),
        "item_count": payload.get("item_count"),
        "items": payload.get("items", []),
    }