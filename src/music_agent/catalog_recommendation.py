"""P11.2: Unified preference view for catalog recommendations (orchestration layer).

Two pieces, both built strictly on the sealed P06/P07 machinery:

``EvenSplitPolicy``
    The production split policy for the sealed propagation seam: a multi-artist /
    multi-genre track's direct magnitude is divided evenly across its relationships.
    The seam was always injected policy; this is the first production instance.

``build_affinity_inputs``
    The unified view for external songs. It takes the caller's already-derived direct
    Track preferences (one coherent conclusion per track, already merging Music.app
    evidence and recommendation feedback through the P06 signal heads), propagates each
    directional conclusion to its track's ARTIST and GENRE relationships, and reduces
    both dimensions through the signed affinity reducers (genre: P10.17b; artist: its
    P11.2 twin). Every above-threshold affinity becomes one INFERRED
    :class:`~music_agent.recommendation_contract.PreferenceInput` on an ARTIST or GENRE
    target -- the exact inputs :func:`~music_agent.catalog_candidate_generation.generate_catalog_candidates`
    matches catalog candidates against. No cross-source merging, no fabricated semantic
    dimensions (mood etc. stay unsupported), provenance preserved end to end.

``build_catalog_recommendation``
    One complete run: the caller's ambient context (direct + per-track inferred inputs),
    plus the affinity inputs, plus catalog candidates from the canonical catalog Tracks,
    through the sealed pipeline (candidates -> score -> rank -> result). Catalog
    candidates carry the ``catalog_driven`` source and compete by the same score
    components as preference-driven candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from music_agent.artist_affinity import SourcedArtistContribution, build_artist_affinities
from music_agent.catalog_candidate_generation import generate_catalog_candidates
from music_agent.genre_affinity import (
    GenreAffinityPolicy,
    SourcedContribution,
    build_genre_affinities,
)
from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import PropagationKind, propagate_track_preference
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    PreferenceInput,
    RecommendationContext,
    RecommendationRequest,
    RecommendedItemKind,
)
from music_agent.recommendation_ranking import RankingOutcome, build_recommendation


class CatalogRecommendationError(ValueError):
    code = "catalog_recommendation_error"


@dataclass(frozen=True, slots=True)
class EvenSplitPolicy:
    """Even division of a track's direct magnitude across its relationships."""

    def split_artist_magnitude(self, magnitude: float, artist_count: int) -> float:
        return round(magnitude / artist_count, 6)

    def split_genre_magnitude(self, magnitude: float, genre_count: int) -> float:
        return round(magnitude / genre_count, 6)


def build_affinity_inputs(
    direct_preferences: Iterable[DerivedPreference],
    tracks_by_id: Mapping[str, Mapping[str, Any]],
    source_system: str,
    *,
    policy: GenreAffinityPolicy | None = None,
) -> tuple[PreferenceInput, ...]:
    """Reduce directional direct Track preferences into ARTIST + GENRE inferred inputs.

    ``direct_preferences`` are already-derived direct conclusions (the unified track-level
    view); ``tracks_by_id`` supplies each track's canonical ``artist_ids`` and ``genres``;
    ``source_system`` scopes the reduction exactly like the existing source-scoped tool.
    Only affinities at or above the policy threshold become inputs; the sign survives as
    POSITIVE or NEGATIVE and the bounded magnitude as |affinity|.
    """
    policy = policy or GenreAffinityPolicy()
    if not isinstance(policy, GenreAffinityPolicy):
        raise CatalogRecommendationError("policy must be a GenreAffinityPolicy")
    if not isinstance(tracks_by_id, Mapping):
        raise CatalogRecommendationError("tracks_by_id must be a mapping")
    split = EvenSplitPolicy()
    genre_sourced: list[SourcedContribution] = []
    artist_sourced: list[SourcedArtistContribution] = []
    for preference in direct_preferences:
        if not isinstance(preference, DerivedPreference):
            raise CatalogRecommendationError(
                "each direct preference must be a DerivedPreference"
            )
        if preference.strength.state not in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            continue
        track = tracks_by_id.get(preference.target.target_id)
        if track is None:
            continue  # a preference whose track is gone contributes nothing, silently
        artist_ids = track.get("artist_ids", ())
        genres = track.get("genres", ())
        for contribution in propagate_track_preference(
            preference,
            artist_ids=artist_ids,
            genres=genres,
            artist_split=split,
            genre_split=split,
        ):
            if contribution.kind is PropagationKind.GENRE:
                genre_sourced.append(SourcedContribution(source_system, contribution))
            elif contribution.kind is PropagationKind.ARTIST:
                artist_sourced.append(SourcedArtistContribution(source_system, contribution))
    return affinity_inputs(
        build_genre_affinities(genre_sourced, policy),
        build_artist_affinities(artist_sourced, policy),
        policy,
    )


def affinity_inputs(
    genre_affinities: Iterable[Any],
    artist_affinities: Iterable[Any],
    policy: GenreAffinityPolicy | None = None,
) -> tuple[PreferenceInput, ...]:
    """Project above-threshold genre and artist affinities into inferred preference inputs.

    Shared by the orchestration path and by callers that already hold reduced affinities
    (the agent-service tool propagates and reduces inline for its own source scoping).
    """
    policy = policy or GenreAffinityPolicy()
    if not isinstance(policy, GenreAffinityPolicy):
        raise CatalogRecommendationError("policy must be a GenreAffinityPolicy")
    inputs: list[PreferenceInput] = []
    for affinity in genre_affinities:
        if abs(affinity.affinity) < policy.affinity_threshold:
            continue
        inputs.append(
            PreferenceInput.from_inferred(
                InferredAffinity(
                    PreferenceTargetReference(PreferenceTargetKind.GENRE, affinity.genre_key),
                    _signed_strength(affinity.affinity),
                )
            )
        )
    for affinity in artist_affinities:
        if abs(affinity.affinity) < policy.affinity_threshold:
            continue
        inputs.append(
            PreferenceInput.from_inferred(
                InferredAffinity(
                    PreferenceTargetReference(PreferenceTargetKind.ARTIST, affinity.artist_id),
                    _signed_strength(affinity.affinity),
                )
            )
        )
    return tuple(inputs)


def _signed_strength(affinity: float) -> PreferenceStrength:
    state = PreferenceState.POSITIVE if affinity > 0 else PreferenceState.NEGATIVE
    return PreferenceStrength(state, round(abs(affinity), 4))


def build_catalog_recommendation(
    *,
    run_id: str,
    produced_at: datetime,
    ambient_inputs: Iterable[PreferenceInput],
    affinity_inputs: Iterable[PreferenceInput],
    catalog_tracks: Iterable[Mapping[str, Any]],
    limit: int,
    quality_evidence: object | None = None,
) -> RankingOutcome:
    """Run the sealed pipeline with unified inputs plus catalog-driven candidates.

    The context carries the caller's ambient inputs (direct and per-track inferred) and
    the affinity inputs; catalog candidates are generated from the canonical catalog
    Tracks and merged before scoring. Preference-driven candidates win on duplicate
    targets (the ranking seam dedupes).
    """
    context = RecommendationContext(
        produced_at, tuple(ambient_inputs) + tuple(affinity_inputs)
    )
    request = RecommendationRequest(context, RecommendedItemKind.TRACK, limit)
    return build_recommendation(
        request,
        run_id=run_id,
        produced_at=produced_at,
        quality_evidence=quality_evidence,
        extra_candidates=generate_catalog_candidates(context, tuple(catalog_tracks)),
    )
