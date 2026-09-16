"""P15-S3-S3E: Fresh-driven candidate generation (the explicit Fresh-intent channel).

Consumes the same :class:`~music_agent.recommendation_contract.RecommendationContext`
as the catalog-driven generator, plus the *authoritative same-run promoted set*
(canonical Track IDs captured by the provider loop from genuinely executed
``discover_catalog_tracks`` results) and the direction-filtered catalog-bound
pool, and emits unscored :class:`~music_agent.recommendation_contract.Candidate`
values whose source is the dedicated ``fresh_driven`` vocabulary.

The channel exists for one reason: the frozen preference-driven and
catalog-driven generators demand a POSITIVE directional claim before a song may
compete, so a just-discovered track that carries no preference evidence at all
can never reach the final batch -- and the P15-S3-S3D ``min_fresh`` final-
selection floor has nothing to select. An explicit Fresh request ("新的/没听过
的/库外的") is a deliberate exploration contract between the user and the
system: here, the absence of positive evidence is not a reason to drop the
track, and the floor -- not scoring -- is what admits it.

Semantics (frozen for this module):

* Identity gate: a candidate is considered ONLY for authoritative fresh IDs
  (the caller's run-local promoted capture). Model-supplied target ids, durable
  ``catalog_track_state`` memory, ALREADY_BOUND entries, and arbitrary
  catalog-only tracks never enter through this module -- the caller enforces
  the activation boundary (``min_fresh > 0`` with a non-empty authoritative
  set) and this module only ever iterates the supplied set.
* Eligibility: an eligible fresh candidate carries an EMPTY basis
  (``basis_targets == ()``) -- no preference evidence is claimed, so the
  existing scoring model produces its documented zero-evidence breakdown
  (total ``0``) and the item ranks at the deterministic tail. The contract
  explicitly allows an empty basis for a non-preference-driven source.
* Negative veto (fail closed): a fresh track whose resolvable preference
  surface -- its own Track target, its canonical Artist IDs, or its
  canonicalized genre keys -- carries a NEGATIVE *operative* conclusion in the
  context yields a REJECTED candidate with reason ``negative_preference``.
  The operative rule is the frozen P06 fallback (an inferred conclusion fills a
  direct ``UNKNOWN``/``INSUFFICIENT`` gap; a formed direct conclusion governs),
  read through the public :func:`preference_attribution.is_inferred_fallback_eligible`
  seam. ``UNKNOWN`` / ``INSUFFICIENT`` / ``NEUTRAL`` / ``CONFLICT`` do NOT
  veto -- freshness exploration exists precisely because evidence is absent.
* No fabrication: never a positive claim, never a magnitude, never a basis
  target beyond the veto scan, never a synthetic candidate for a missing track.

Determinism: candidates are emitted in canonical-target order (fresh IDs
sorted); the veto scan sorts the track's refs by (kind, target_id); the only
nondeterminism is the opaque ``cnd_`` identity.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from music_agent.preference_attribution import (
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
    is_inferred_fallback_eligible,
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


class FreshCandidateGenerationError(ValueError):
    code = "fresh_candidate_generation_error"


class FreshCandidateGenerationValidationError(FreshCandidateGenerationError):
    code = "validation_error"


# Stable source vocabulary carried by every candidate this module emits --
# distinct from ``catalog_driven`` so exploration classification can read the
# two sources separately while Fresh identity stays membership-based.
FRESH_CANDIDATE_SOURCE_SYSTEM = "music_agent"
FRESH_CANDIDATE_SOURCE_PATH = "fresh_driven"

# The machine code shared with preference-driven rejection vocabulary.
REJECTION_NEGATIVE_PREFERENCE = "negative_preference"


def generate_fresh_candidates(
    context: RecommendationContext,
    fresh_track_ids: Iterable[str],
    catalog_tracks: Iterable[Mapping[str, Any]],
) -> tuple[Candidate, ...]:
    """Generate zero-basis fresh-driven candidates over the authoritative set.

    ``fresh_track_ids`` is the caller's run-local authoritative promoted set.
    ``catalog_tracks`` is the direction-filtered catalog-bound pool (the same
    shape the catalog-driven generator consumes): a fresh ID is only considered
    while its track is present there (a track filtered out by a hard genre
    direction, or absent from the store, is skipped silently -- the direction
    filter stays a hard filter for fresh too). For each considered track the
    veto scan resolves its Track/Artist/Genre targets against the context; any
    NEGATIVE operative conclusion rejects the candidate, everything else yields
    an ELIGIBLE empty-basis candidate. Wrong argument types fail closed with
    :class:`FreshCandidateGenerationValidationError`.
    """
    _require_context(context)
    fresh_ids = _require_fresh_ids(fresh_track_ids)
    pool = _require_pool(catalog_tracks)
    source = CandidateSourceReference(
        FRESH_CANDIDATE_SOURCE_SYSTEM, FRESH_CANDIDATE_SOURCE_PATH
    )
    candidates: list[Candidate] = []
    for fresh_id in sorted(fresh_ids):
        track = pool.get(fresh_id)
        if track is None:
            # Not in the direction-filtered pool (or no longer in the store):
            # the hard direction filter applies to fresh exactly like it does
            # to every catalog candidate, so this fresh ID contributes nothing.
            continue
        refs = _veto_refs(fresh_id, track)
        states = {_operative_state(context, ref) for ref in refs if _has_input(context, ref)}
        if PreferenceState.NEGATIVE in states:
            candidates.append(
                Candidate(
                    candidate_id=generate_candidate_id(),
                    target=PreferenceTargetReference(
                        PreferenceTargetKind.TRACK, fresh_id
                    ),
                    source=source,
                    basis_targets=(),
                    eligibility=Eligibility.REJECTED,
                    rejection=Rejection(REJECTION_NEGATIVE_PREFERENCE),
                )
            )
        else:
            candidates.append(
                Candidate(
                    candidate_id=generate_candidate_id(),
                    target=PreferenceTargetReference(
                        PreferenceTargetKind.TRACK, fresh_id
                    ),
                    source=source,
                    basis_targets=(),
                    eligibility=Eligibility.ELIGIBLE,
                )
            )
    return tuple(candidates)


def _veto_refs(
    fresh_id: str, track: Mapping[str, Any]
) -> tuple[PreferenceTargetReference, ...]:
    """The preference surface a fresh track can be judged against.

    Its own Track target, then its canonical Artist IDs and canonicalized genre
    keys -- the same identity dimensions the catalog-driven generator uses as
    basis, plus the track itself (a direct dislike on the just-promoted track
    is impossible in practice, but the scan stays uniform and fail-closed).
    """
    refs: list[PreferenceTargetReference] = [PreferenceTargetReference(
        PreferenceTargetKind.TRACK, fresh_id
    )]
    seen: set[PreferenceTargetReference] = set(refs)
    artist_ids = track.get("artist_ids", ())
    genres = track.get("genres", ())
    if not isinstance(artist_ids, (tuple, list)) or not isinstance(genres, (tuple, list)):
        raise FreshCandidateGenerationValidationError(
            "artist_ids and genres must be sequences"
        )
    for artist_id in artist_ids:
        if not isinstance(artist_id, str):
            raise FreshCandidateGenerationValidationError(
                "artist_ids must contain strings"
            )
        try:
            ref = PreferenceTargetReference(PreferenceTargetKind.ARTIST, artist_id)
        except ValueError as error:
            raise FreshCandidateGenerationValidationError(
                f"artist id {artist_id!r} is not a canonical Artist ID"
            ) from error
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    for genre in genres:
        if not isinstance(genre, str):
            raise FreshCandidateGenerationValidationError(
                "genres must contain strings"
            )
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
    """The frozen P06 fallback rule for one target (direct vs inferred inputs).

    Written locally from the public
    :func:`preference_attribution.is_inferred_fallback_eligible` seam so the
    catalog-driven sibling module stays untouched; semantics are identical to
    its private ``_operative_state``.
    """
    inputs = [input_ for input_ in context.preference_inputs if input_.target == target]
    direct = next(
        (input_ for input_ in inputs if input_.provenance is PreferenceProvenance.DIRECT),
        None,
    )
    inferred = next(
        (input_ for input_ in inputs if input_.provenance is PreferenceProvenance.INFERRED),
        None,
    )
    if direct is None:
        return inferred.strength.state
    if inferred is None:
        return direct.strength.state
    if is_inferred_fallback_eligible(direct.strength.state):
        return inferred.strength.state
    return direct.strength.state


def _require_fresh_ids(fresh_track_ids: object) -> list[str]:
    if isinstance(fresh_track_ids, (str, bytes)) or not isinstance(
        fresh_track_ids, Iterable
    ):
        raise FreshCandidateGenerationValidationError(
            "fresh_track_ids must be an iterable of canonical Track IDs"
        )
    ids: set[str] = set()
    for fresh_id in fresh_track_ids:
        if not isinstance(fresh_id, str) or not fresh_id:
            raise FreshCandidateGenerationValidationError(
                "each fresh id must be a non-empty string"
            )
        try:
            PreferenceTargetReference(PreferenceTargetKind.TRACK, fresh_id)
        except ValueError as error:
            raise FreshCandidateGenerationValidationError(
                f"fresh id {fresh_id!r} is not a canonical Track ID"
            ) from error
        ids.add(fresh_id)
    return sorted(ids)


def _require_pool(catalog_tracks: object) -> dict[str, Mapping[str, Any]]:
    if isinstance(catalog_tracks, (str, bytes)) or not isinstance(
        catalog_tracks, Iterable
    ):
        raise FreshCandidateGenerationValidationError(
            "catalog_tracks must be an iterable of track mappings"
        )
    pool: dict[str, Mapping[str, Any]] = {}
    for track in catalog_tracks:
        if not isinstance(track, Mapping):
            raise FreshCandidateGenerationValidationError(
                "each catalog track must be a mapping"
            )
        track_id = track.get("id")
        if not isinstance(track_id, str) or not track_id:
            raise FreshCandidateGenerationValidationError(
                "each catalog track requires a non-empty id"
            )
        if track_id in pool:
            raise FreshCandidateGenerationValidationError(
                f"duplicate catalog track {track_id!r}"
            )
        pool[track_id] = track
    return pool


def _require_context(context: object) -> RecommendationContext:
    if not isinstance(context, RecommendationContext):
        raise FreshCandidateGenerationValidationError(
            f"context must be a RecommendationContext, not {type(context).__name__}"
        )
    return context