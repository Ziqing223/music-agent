"""P10.17c: Recommendation-feedback learning bridge (explicit cross-phase amendment).

Genuine recommendation reactions ("liked"/"disliked") must be able to affect future
preference-based recommendations. Frozen P08 semantics derive ``NO_EFFECT`` for
recommendation-linked observations (``target=None``), so this P10 bridge resolves the
reacted-to Track STRICTLY from the persisted recommendation provenance and hands the
resolved target to the existing effect-derivation boundary as an explicit optional
input. The frozen observation semantics are preserved: the original feedback event
stays immutable and recommendation-linked; nothing is rewritten, deleted, or
duplicated; the application journal records the resolved target while the observation
keeps its original shape.

Resolution rules (fail closed, no inference):

- the recommendation run must exist in the persisted history;
- the candidate must belong to that exact run (exactly one match);
- the candidate's target must be a canonical TRACK reference;
- no title/name/artist matching, no LLM inference, no fuzzy resolution.

Idempotency: one original ``feedback_id`` can produce at most one learning
application (the existing per-feedback_id dedupe in the learning-application
repository remains the authority); replay never creates a second P06 revision.
"""

from __future__ import annotations

from typing import Any

from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference


class RecommendationFeedbackResolutionError(ValueError):
    code = "recommendation_feedback_resolution_error"


def resolve_recommendation_feedback_target(
    history: Any, reference: Any
) -> PreferenceTargetReference | None:
    """Resolve one recommendation reference to its canonical TRACK target.

    ``history`` is the persisted recommendation-history repository; ``reference`` is
    the observation's ``FeedbackRecommendationReference``. Returns the resolved
    TRACK target, or None when the reference is absent. Raises the typed error on
    any missing/ambiguous provenance.
    """
    if reference is None:
        return None
    run_id = getattr(reference, "run_id", None)
    candidate_id = getattr(reference, "candidate_id", None)
    if not isinstance(run_id, str) or not run_id or not isinstance(candidate_id, str) or not candidate_id:
        raise RecommendationFeedbackResolutionError("malformed recommendation reference")
    run = history.get_result(run_id)
    if run is None:
        raise RecommendationFeedbackResolutionError(
            f"recommendation run does not exist: {run_id}"
        )
    matches = [
        item for item in run.items
        if item.candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise RecommendationFeedbackResolutionError(
            f"candidate {candidate_id!r} does not belong to run {run_id!r} exactly once "
            f"(found {len(matches)})"
        )
    target = matches[0].candidate.target
    if not isinstance(target, PreferenceTargetReference):
        raise RecommendationFeedbackResolutionError("candidate target must be a PreferenceTargetReference")
    if target.kind is not PreferenceTargetKind.TRACK:
        raise RecommendationFeedbackResolutionError(
            f"recommendation feedback resolves only to TRACK targets, got {target.kind.value}"
        )
    return target
