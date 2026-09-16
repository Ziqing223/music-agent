"""P15-S3-S3A: known-catalog supply facts for the empty-generation diagnostics.

Pure, mapping-driven projection over facts one inferred-generation execution already holds --
the direction-filtered catalog-bound pool, the durable ``catalog_track_state`` memory for its
tracks, and the catalog candidates that execution generated. The resulting counts are the
"known supply" facts exposed through the existing P15-S4-M2-2 empty-diagnostic envelope
(decision 2 of P15-S3-S3: server-side facts and diagnostics, no new model-visible tool), so
the provider model can distinguish "no known supply left -- fresh discovery is the way" from
"known supply exists but nothing matches the current positive directions" without a second
tool call.

No ranking, no score, no fresh-exploration verdicts, and no persistence: every count is
recomputed per execution from the caller's local facts. This module never reads the database,
never re-derives eligibility (it recounts the frozen P11.2 verdicts already carried by the
candidates), and never decides whether a catalog search should happen -- it only reports
supply. The Known-vs-Fresh trigger policy lives in the prompt and the provider loop
(S3-S3B), not here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from music_agent.catalog_track_state_repository import CatalogTrackState
from music_agent.identity import EntityType, validate_canonical_id
from music_agent.recommendation_contract import Candidate, Eligibility


class KnownCatalogSupplyError(ValueError):
    code = "known_catalog_supply_error"


@dataclass(frozen=True, slots=True)
class KnownCatalogSupply:
    """Known-catalog supply counts of one inferred execution (P15-S3-S3A).

    ``known_catalog_track_count`` is the size of the direction-filtered catalog-bound pool
    (Known pool B of the S3-S3 design). ``known_never_recommended_count`` counts pool tracks
    whose durable memory shows zero recommendation events; a pool track without a memory row
    is *unknown memory*, not "never recommended", and is excluded from the never counts.
    ``known_eligible_count`` / ``known_rejected_count`` recount the eligibility of the catalog
    candidates the execution actually generated (frozen P11.2 verdicts -- never re-derived
    here), and ``known_never_recommended_eligible_count`` is the usable-and-yet-unshown
    intersection: the true never-before-recommended supply available to this execution.
    """

    known_catalog_track_count: int
    known_never_recommended_count: int
    known_eligible_count: int
    known_rejected_count: int
    known_never_recommended_eligible_count: int


def summarize_known_catalog_supply(
    catalog_tracks: object,
    *,
    states_by_track_id: object,
    candidates: object,
) -> KnownCatalogSupply:
    """Project known-catalog supply facts from one execution's local inputs.

    ``catalog_tracks`` is the direction-filtered catalog-bound pool -- a sequence of canonical
    track mappings (the same shape ``generate_catalog_candidates`` consumes). ``states_by_track_id``
    maps canonical Track ID to ``CatalogTrackState`` (rows the caller loaded; a missing entry
    means "no memory", never "never recommended"). ``candidates`` are the catalog candidates
    ``generate_catalog_candidates`` produced from that pool; their eligibility is recounted
    verbatim. Wrong types fail closed with :class:`KnownCatalogSupplyError`.
    """
    pool_ids = _require_track_ids(catalog_tracks)
    states = _require_states(states_by_track_id)
    candidate_values = _require_candidates(candidates)

    never_ids = {
        canonical_id
        for canonical_id in pool_ids
        if _is_never_recommended(states.get(canonical_id))
    }
    eligible_never_count = 0
    eligible_count = 0
    rejected_count = 0
    for candidate in candidate_values:
        if candidate.eligibility is Eligibility.ELIGIBLE:
            eligible_count += 1
            if candidate.target.target_id in never_ids:
                eligible_never_count += 1
        elif candidate.eligibility is Eligibility.REJECTED:
            rejected_count += 1
    return KnownCatalogSupply(
        known_catalog_track_count=len(pool_ids),
        known_never_recommended_count=len(never_ids),
        known_eligible_count=eligible_count,
        known_rejected_count=rejected_count,
        known_never_recommended_eligible_count=eligible_never_count,
    )


def _is_never_recommended(state: CatalogTrackState | None) -> bool:
    """A memory row with zero recorded recommendation events; no row is "no memory"."""
    return state is not None and state.recommendation_count == 0


def _require_track_ids(catalog_tracks: object) -> list[str]:
    if isinstance(catalog_tracks, (str, bytes)) or not isinstance(
        catalog_tracks, (list, tuple)
    ):
        raise KnownCatalogSupplyError(
            "catalog_tracks must be a list or tuple of track mappings"
        )
    pool_ids: list[str] = []
    seen: set[str] = set()
    for track in catalog_tracks:
        if not isinstance(track, Mapping):
            raise KnownCatalogSupplyError("each catalog track must be a mapping")
        canonical_id = _require_canonical_id(track.get("id"))
        if canonical_id in seen:
            raise KnownCatalogSupplyError(
                f"duplicate catalog track {canonical_id!r}"
            )
        seen.add(canonical_id)
        pool_ids.append(canonical_id)
    return pool_ids


def _require_states(states_by_track_id: object) -> dict[str, CatalogTrackState]:
    if not isinstance(states_by_track_id, Mapping):
        raise KnownCatalogSupplyError(
            "states_by_track_id must be a mapping of canonical_id -> CatalogTrackState"
        )
    states: dict[str, CatalogTrackState] = {}
    for canonical_id, state in states_by_track_id.items():
        key = _require_canonical_id(canonical_id)
        if not isinstance(state, CatalogTrackState):
            raise KnownCatalogSupplyError(
                f"state for {key!r} must be a CatalogTrackState"
            )
        states[key] = state
    return states


def _require_candidates(candidates: object) -> list[Candidate]:
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise KnownCatalogSupplyError(
            "candidates must be a sequence of Candidate values"
        )
    values: list[Candidate] = []
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise KnownCatalogSupplyError(
                "each candidates entry must be a Candidate"
            )
        values.append(candidate)
    return values


def _require_canonical_id(canonical_id: object) -> str:
    if not isinstance(canonical_id, str):
        raise KnownCatalogSupplyError("canonical_id must be a non-empty string")
    try:
        validate_canonical_id(EntityType.TRACK, canonical_id)
    except ValueError as error:
        raise KnownCatalogSupplyError(str(error)) from error
    return canonical_id