"""P11.2: Catalog-driven candidate generation (the external-song twin of P07.2 Track A).

Consumes the same :class:`~music_agent.recommendation_contract.RecommendationContext` as the
preference-driven generator, plus a set of canonical catalog Tracks (promoted through the P11.1
ingestion path, each carrying genres and canonical artist relations), and emits unscored
:class:`~music_agent.recommendation_contract.Candidate` values whose *basis* is the song's own
identity dimensions -- its canonical Artist IDs and genre keys. The basis is never fabricated
semantic features (no mood, no title-derived attributes): only the reliable dimensions the
catalog metadata already supplies.

A catalog candidate is scored by the existing scorer exactly like a preference-driven one: each
basis target is matched against the context's unified preference inputs (the same conclusions
derived from Music.app evidence and recommendation feedback). The frozen directionality rule
carries over: a NEGATIVE operative conclusion on any basis target vetoes the candidate (fail
closed), and a candidate whose basis targets carry no directional claim is omitted -- an
unjudged song never competes by accident.

Determinism: candidates are emitted in canonical-target order; basis targets are sorted by
(kind, target_id); the only nondeterminism is the opaque ``cnd_`` identity.
"""

from __future__ import annotations

from typing import Any, Mapping

from music_agent.preference_attribution import (
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import canonicalize_genre_key
from music_agent.preference_strength import PreferenceState
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    RecommendationContext,
    Rejection,
    generate_candidate_id,
)
from music_agent.recommendation_scoring import (
    is_inferred_fallback_eligible,
)


class CatalogCandidateGenerationError(ValueError):
    code = "catalog_candidate_generation_error"


class CatalogCandidateGenerationValidationError(CatalogCandidateGenerationError):
    code = "validation_error"


# Stable source vocabulary carried by every candidate this module emits.
CATALOG_CANDIDATE_SOURCE_SYSTEM = "music_agent"
CATALOG_CANDIDATE_SOURCE_PATH = "catalog_driven"

# The machine code shared with preference-driven rejection vocabulary.
REJECTION_NEGATIVE_PREFERENCE = "negative_preference"


def generate_catalog_candidates(
    context: RecommendationContext,
    catalog_tracks: object,
) -> tuple[Candidate, ...]:
    """Generate unscored candidates for canonical catalog Tracks against unified inputs.

    ``catalog_tracks`` is an iterable of mappings, each carrying ``"id"`` (a canonical Track
    ID), ``"genres"`` (an iterable of genre strings), and ``"artist_ids"`` (an iterable of
    canonical Artist IDs). For each track, the basis is its genre keys (canonicalized) and its
    artists, restricted to targets that actually have a matching preference input in the
    context. A track with no matched basis, or whose matched basis carries no directional
    claim, yields no candidate; a NEGATIVE claim on any basis target rejects the candidate;
    otherwise the candidate is ELIGIBLE with the matched basis. Wrong argument types fail
    closed with :class:`CatalogCandidateGenerationValidationError`.
    """
    _require_context(context)
    source = CandidateSourceReference(
        CATALOG_CANDIDATE_SOURCE_SYSTEM, CATALOG_CANDIDATE_SOURCE_PATH
    )
    candidates: list[Candidate] = []
    for track in sorted(_require_tracks(catalog_tracks), key=lambda item: item["id"]):
        refs = _track_basis_refs(track)
        matched = [ref for ref in refs if _has_input(context, ref)]
        if not matched:
            continue
        states = {
            _operative_state(context, ref)
            for ref in matched
        }
        if PreferenceState.NEGATIVE in states:
            candidates.append(
                Candidate(
                    candidate_id=generate_candidate_id(),
                    target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track["id"]),
                    source=source,
                    basis_targets=tuple(sorted(matched, key=_ref_sort_key)),
                    eligibility=Eligibility.REJECTED,
                    rejection=Rejection(REJECTION_NEGATIVE_PREFERENCE),
                )
            )
        elif PreferenceState.POSITIVE in states:
            candidates.append(
                Candidate(
                    candidate_id=generate_candidate_id(),
                    target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track["id"]),
                    source=source,
                    basis_targets=tuple(sorted(matched, key=_ref_sort_key)),
                    eligibility=Eligibility.ELIGIBLE,
                )
            )
    return tuple(candidates)


def _require_tracks(catalog_tracks: object) -> list[Mapping[str, Any]]:
    if not isinstance(catalog_tracks, (tuple, list)):
        raise CatalogCandidateGenerationValidationError(
            "catalog_tracks must be a sequence of track mappings"
        )
    tracks: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for track in catalog_tracks:
        if not isinstance(track, Mapping):
            raise CatalogCandidateGenerationValidationError(
                "each catalog track must be a mapping"
            )
        track_id = track.get("id")
        if not isinstance(track_id, str) or not track_id:
            raise CatalogCandidateGenerationValidationError(
                "each catalog track requires a non-empty id"
            )
        try:
            PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id)
        except ValueError as error:
            raise CatalogCandidateGenerationValidationError(
                f"catalog track id {track_id!r} is not a canonical Track ID"
            ) from error
        if track_id in seen:
            raise CatalogCandidateGenerationValidationError(
                f"duplicate catalog track {track_id!r}"
            )
        seen.add(track_id)
        tracks.append(track)
    return tracks


def _track_basis_refs(track: Mapping[str, Any]) -> tuple[PreferenceTargetReference, ...]:
    refs: list[PreferenceTargetReference] = []
    seen: set[PreferenceTargetReference] = set()
    artist_ids = track.get("artist_ids", ())
    genres = track.get("genres", ())
    if not isinstance(artist_ids, (tuple, list)) or not isinstance(genres, (tuple, list)):
        raise CatalogCandidateGenerationValidationError(
            "artist_ids and genres must be sequences"
        )
    for artist_id in artist_ids:
        if not isinstance(artist_id, str):
            raise CatalogCandidateGenerationValidationError("artist_ids must contain strings")
        try:
            ref = PreferenceTargetReference(PreferenceTargetKind.ARTIST, artist_id)
        except ValueError as error:
            raise CatalogCandidateGenerationValidationError(
                f"artist id {artist_id!r} is not a canonical Artist ID"
            ) from error
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    for genre in genres:
        if not isinstance(genre, str):
            raise CatalogCandidateGenerationValidationError("genres must contain strings")
        key = canonicalize_genre_key(genre)
        if not key:
            continue
        ref = PreferenceTargetReference(PreferenceTargetKind.GENRE, key)
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return tuple(refs)


def _has_input(context: RecommendationContext, target: PreferenceTargetReference) -> bool:
    return any(input_.target == target for input_ in context.preference_inputs)


def _operative_state(
    context: RecommendationContext, target: PreferenceTargetReference
) -> PreferenceState:
    """The frozen P06 fallback rule, applied to one target's inputs (direct vs inferred)."""
    inputs = [input_ for input_ in context.preference_inputs if input_.target == target]
    direct = next((input_ for input_ in inputs if input_.provenance is PreferenceProvenance.DIRECT), None)
    inferred = next((input_ for input_ in inputs if input_.provenance is PreferenceProvenance.INFERRED), None)
    if direct is None:
        return inferred.strength.state
    if inferred is None:
        return direct.strength.state
    if is_inferred_fallback_eligible(direct.strength.state):
        return inferred.strength.state
    return direct.strength.state


def _require_context(context: object) -> RecommendationContext:
    if not isinstance(context, RecommendationContext):
        raise CatalogCandidateGenerationValidationError(
            f"context must be a RecommendationContext, not {type(context).__name__}"
        )
    return context


def _ref_sort_key(target: PreferenceTargetReference) -> tuple[str, str]:
    return (target.kind.value, target.target_id)
