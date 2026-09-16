"""Preference-driven candidate generation for recommendations (P07.2, Track A).

This module is the candidate-generation slice of the recommendation system. It consumes a
:class:`~music_agent.recommendation_contract.RecommendationContext` holding read-only P06
preference conclusions, plus the requested
:class:`~music_agent.recommendation_contract.RecommendedItemKind`, and emits the tuple of
unscored :class:`~music_agent.recommendation_contract.Candidate` values the scoring track
consumes. It is a pure domain layer: deterministic, side-effect free, and independent of SQLite
rows, external source payloads, the system clock, and global state. It never scores, ranks, or
persists anything, never reads ``request.limit`` (that is ranking's job), and never mutates P06
-- it only reads ``PreferenceInput`` snapshots from the context. ``context.now`` is accepted as
part of the contract's context but is never consulted: the same inputs produce the same
candidates at any instant.

Frozen generation semantics
---------------------------

Kind matching. A preference input motivates a candidate only when its target kind matches the
requested ``RecommendedItemKind``; the three kinds map 1:1 to ``PreferenceTargetKind`` (TRACK ->
TRACK, ARTIST -> ARTIST, ALBUM -> ALBUM). Genre inputs never motivate a candidate: the contract
forbids genre candidates, and a genre key can never match a requested kind, so genre inputs are
skipped for every requested kind.

Directionality. Only a POSITIVE conclusion supports recommending a target: a target with a
positive conclusion becomes an ELIGIBLE candidate. A NEGATIVE conclusion makes the target a
REJECTED candidate carrying ``Rejection(REJECTION_NEGATIVE_PREFERENCE)``. The four
non-directional states (``UNKNOWN`` / ``INSUFFICIENT`` / ``NEUTRAL`` / ``CONFLICT``) carry no
directional claim, so a target whose inputs are all non-directional produces no candidate at
all -- skipped, not rejected: a rejection asserts an active decision against the target, and the
inputs make no such claim.

One candidate per target. All matching inputs for one target merge into exactly one candidate;
``basis_targets`` names the motivating target exactly once (the contract forbids duplicate basis
targets). When a target carries both POSITIVE and NEGATIVE conclusions (for example a DIRECT
positive and an INFERRED negative), the NEGATIVE claim vetoes the positive one -- evidence
against recommending wins over evidence for it, fail closed -- and the target is emitted as one
REJECTED candidate with reason ``negative_preference``.

Determinism and order. Candidates are emitted in deterministic order: the motivating targets are
sorted by ``target_id`` (with ``kind`` as the tie-breaker, though every target in one output
shares the requested kind). The only nondeterminism is the opaque ``cnd_`` identity, minted per
candidate with the contract's :func:`generate_candidate_id` and never derived from the target.

Source and rejection vocabulary
-------------------------------

Every candidate carries ``CandidateSourceReference(CANDIDATE_SOURCE_SYSTEM, CANDIDATE_SOURCE_PATH)``
= (``"music_agent"``, ``"preference_driven"``). The rejection vocabulary is one machine code,
``REJECTION_NEGATIVE_PREFERENCE`` = ``"negative_preference"``; the contract fixes only the shape
(a non-empty machine code, never prose). A context with no matching inputs yields the empty
tuple, a valid "nothing to recommend" outcome.
"""

from __future__ import annotations

from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    RecommendationContext,
    RecommendedItemKind,
    Rejection,
    generate_candidate_id,
)


class CandidateGenerationError(ValueError):
    code = "candidate_generation_error"


class CandidateGenerationValidationError(CandidateGenerationError):
    code = "validation_error"


# Stable source vocabulary carried by every candidate this module emits.
CANDIDATE_SOURCE_SYSTEM = "music_agent"
CANDIDATE_SOURCE_PATH = "preference_driven"

# The machine code carried by the Rejection of a candidate whose target has a NEGATIVE conclusion.
REJECTION_NEGATIVE_PREFERENCE = "negative_preference"

_RECOMMENDED_TARGET_KIND: dict[RecommendedItemKind, PreferenceTargetKind] = {
    RecommendedItemKind.TRACK: PreferenceTargetKind.TRACK,
    RecommendedItemKind.ARTIST: PreferenceTargetKind.ARTIST,
    RecommendedItemKind.ALBUM: PreferenceTargetKind.ALBUM,
}

# Only a directional claim can motivate a candidate. The non-directional states
# (UNKNOWN / INSUFFICIENT / NEUTRAL / CONFLICT) carry no claim to act on and are skipped.
_DIRECTIONAL_STATES = {PreferenceState.POSITIVE, PreferenceState.NEGATIVE}


def generate_candidates(
    context: RecommendationContext,
    recommended_kind: RecommendedItemKind,
) -> tuple[Candidate, ...]:
    """Generate the unscored candidates one request of ``recommended_kind`` may be built from.

    ``context`` supplies the read-only preference inputs and ``recommended_kind`` selects the
    entity kind to recommend. Only inputs whose target kind matches ``recommended_kind`` motivate
    candidates; genre inputs are always skipped. Each motivating target yields exactly one
    candidate, ordered deterministically by ``target_id``, whose ``basis_targets`` names the
    target once. A target with a POSITIVE conclusion is ELIGIBLE; a target with a NEGATIVE
    conclusion is REJECTED with reason ``negative_preference`` (a negative claim vetoes a
    positive claim for the same target); a target whose inputs are all non-directional is
    omitted. The result is the empty tuple when no matching input exists. ``request.limit`` is
    never applied here; ranking owns the bound. Wrong argument types fail closed with
    :class:`CandidateGenerationValidationError`.
    """
    _require_context(context)
    _require_kind(recommended_kind)
    target_kind = _RECOMMENDED_TARGET_KIND[recommended_kind]

    claims: dict[PreferenceTargetReference, set[PreferenceState]] = {}
    for input_ in context.preference_inputs:
        if input_.target.kind is not target_kind:
            continue
        if input_.strength.state not in _DIRECTIONAL_STATES:
            continue
        claims.setdefault(input_.target, set()).add(input_.strength.state)

    source = CandidateSourceReference(CANDIDATE_SOURCE_SYSTEM, CANDIDATE_SOURCE_PATH)
    candidates: list[Candidate] = []
    for target in sorted(claims, key=_target_sort_key):
        if PreferenceState.NEGATIVE in claims[target]:
            candidates.append(
                _build_candidate(
                    target,
                    source,
                    eligibility=Eligibility.REJECTED,
                    rejection=Rejection(REJECTION_NEGATIVE_PREFERENCE),
                )
            )
        else:
            candidates.append(_build_candidate(target, source, eligibility=Eligibility.ELIGIBLE))
    return tuple(candidates)


def _build_candidate(
    target: PreferenceTargetReference,
    source: CandidateSourceReference,
    *,
    eligibility: Eligibility,
    rejection: Rejection | None = None,
) -> Candidate:
    return Candidate(
        candidate_id=generate_candidate_id(),
        target=target,
        source=source,
        basis_targets=(target,),
        eligibility=eligibility,
        rejection=rejection,
    )


def _require_context(context: object) -> RecommendationContext:
    if not isinstance(context, RecommendationContext):
        raise CandidateGenerationValidationError(
            f"context must be a RecommendationContext, not {type(context).__name__}"
        )
    return context


def _require_kind(recommended_kind: object) -> RecommendedItemKind:
    if not isinstance(recommended_kind, RecommendedItemKind):
        raise CandidateGenerationValidationError(
            f"recommended_kind must be a RecommendedItemKind, not {type(recommended_kind).__name__}"
        )
    return recommended_kind


def _target_sort_key(target: PreferenceTargetReference) -> tuple[str, str]:
    return (target.target_id, target.kind.value)
