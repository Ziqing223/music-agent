"""P07.3: the scoring model -- eligible candidate to ScoreBreakdown (Track B).

This module is the scoring slice of the recommendation system. It consumes one ``ELIGIBLE``
:class:`~music_agent.recommendation_contract.Candidate` plus the
:class:`~music_agent.recommendation_contract.RecommendationContext` that produced it and emits
the candidate's :class:`~music_agent.recommendation_contract.ScoreBreakdown`. It is a pure
domain layer: deterministic, side-effect free, independent of SQLite rows, the clock, the
network, randomness, and global state. It never ranks candidates, never constructs a
:class:`~music_agent.recommendation_contract.RecommendationItem` or
:class:`~music_agent.recommendation_contract.RecommendationResult`, never applies a request
limit, and never mutates P06: it only reads the frozen ``PreferenceInput`` snapshots carried by
the context.

Component vocabulary (owned by this module, fixed)
--------------------------------------------------

Every breakdown this module emits carries exactly the four components below, always in this
order, so a breakdown is comparable across candidates and reproducible by hand. Component
names are opaque machine keys; their semantics are fixed here:

``basis_support``
    The fraction of basis targets whose *operative* conclusion is ``POSITIVE``: how much of
    the basis agrees that the candidate is worth recommending. ``0`` means no basis target
    positively supports the candidate.

``basis_strength``
    The mean magnitude over the positively supporting basis targets (``0`` when there are
    none): how *intensely* the agreeing evidence favors the candidate.

``negative_evidence``
    The fraction of basis targets whose operative conclusion is ``NEGATIVE``: how much of the
    basis explicitly opposes the candidate. A ``NEGATIVE`` conclusion is a categorical fact and
    is never folded into ``basis_support`` or the total; it is carried here.

``provenance_directness``
    The fraction of basis targets whose operative conclusion is ``DIRECT``: how much of the
    basis rests on direct observation rather than inference. Provenance is a categorical fact
    and is never collapsed onto a scalar anywhere else.

Component-to-total mapping
--------------------------

``total = basis_support * basis_strength``

Equivalently, the total is the mean over *all* basis targets of the positive magnitude (``0``
for every non-positive target). A candidate needs both agreement (``basis_support``) and
intensity (``basis_strength``) to score high: one full-strength positive target scores ``1``,
and one positive among three non-positives scores ``1/3`` at full strength. The mapping is
deterministic and reproducible by hand; the contract independently re-validates the ``[0, 1]``
bounds.

Per-target operative conclusion
-------------------------------

Each basis target is matched against the context's ``preference_inputs`` by target identity
(kind + id). The contract guarantees at most one ``DIRECT`` and one ``INFERRED`` input per
target, and the two may coexist. The *operative* conclusion for a target is selected by the
frozen P06 fallback rule (:func:`music_agent.preference_attribution.is_inferred_fallback_eligible`),
which this module reads but never re-derives:

- only a ``DIRECT`` input exists -> the direct conclusion operates.
- only an ``INFERRED`` input exists -> the inferred conclusion operates.
- both exist -> if the direct state is ``UNKNOWN`` or ``INSUFFICIENT`` (the only states for
  which inferred fallback is eligible), the inferred conclusion fills the gap and operates;
  otherwise the direct conclusion governs and the inferred input must not override it, even
  when it disagrees.

A non-``POSITIVE`` operative conclusion contributes no directional evidence: ``NEGATIVE``,
``NEUTRAL``, ``UNKNOWN``, ``INSUFFICIENT``, and ``CONFLICT`` all contribute magnitude ``0`` to
the total. ``NEGATIVE`` is nevertheless visible through ``negative_evidence``. The remaining
non-directional states (``NEUTRAL``, ``UNKNOWN``, ``INSUFFICIENT``, ``CONFLICT``) carry no
directional claim and are deliberately *excluded* from the scalar vocabulary: their categorical
distinction is preserved in the context's ``PreferenceInput`` snapshots, and no bounded scalar
could encode the four-way distinction without collapsing another axis. Their only influence on
the breakdown is absence -- they lower ``basis_support``. Provenance, by contrast, is carried
by ``provenance_directness`` because it qualifies which evidence operated.

Fail-closed behavior
--------------------

Scoring fails closed with :class:`RecommendationScoringValidationError` for: a
non-:class:`~music_agent.recommendation_contract.Candidate` input, a
non-:class:`~music_agent.recommendation_contract.RecommendationContext` input, a candidate that
is not ``ELIGIBLE``, and a basis target with no matching preference input in the context. A
missing input is never guessed at, and a rejected candidate is never scored.

Empty basis
-----------

A candidate with no basis targets carries no preference evidence and scores zero: every
component is ``0`` and ``total`` is ``0``. The contract explicitly allows an empty basis for a
non-preference-driven source, so this is a well-formed input and the zero-evidence breakdown is
the documented, deterministic answer rather than an error.

Determinism
-----------

The breakdown depends only on the candidate's ``basis_targets`` and the context's
``preference_inputs``. It is invariant under the order of the basis targets and the order of
the preference inputs, and inputs for targets outside the basis do not affect it.

Deferred
--------

Ranking, :class:`~music_agent.recommendation_contract.RecommendationItem` /
:class:`~music_agent.recommendation_contract.RecommendationResult` assembly, request limits,
repeat control, diversity, familiarity, and scenario relevance are later slices and are not
implemented here.
"""

from __future__ import annotations

from music_agent.preference_attribution import (
    PreferenceProvenance,
    PreferenceState,
    is_inferred_fallback_eligible,
)
from music_agent.recommendation_contract import (
    Candidate,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    ScoreBreakdown,
    ScoreComponent,
)


class RecommendationScoringError(ValueError):
    """Base error of the scoring model; every scoring failure is one of these."""

    code = "recommendation_scoring_error"


class RecommendationScoringValidationError(RecommendationScoringError):
    """A scoring input or invariant failure; raised by every fail-closed path."""

    code = "validation_error"


# The fixed component vocabulary of this scoring model, in emission order. Every breakdown this
# module produces carries exactly these components in exactly this order.
SCORE_COMPONENT_NAMES = (
    "basis_support",
    "basis_strength",
    "negative_evidence",
    "provenance_directness",
)


def score_candidate(candidate: Candidate, context: RecommendationContext) -> ScoreBreakdown:
    """Score one ELIGIBLE candidate against the context that produced it.

    ``candidate`` must be an ``ELIGIBLE`` :class:`~music_agent.recommendation_contract.Candidate`
    and ``context`` the :class:`~music_agent.recommendation_contract.RecommendationContext` it
    was generated against. Every basis target must have at least one matching preference input
    in the context; anything else fails closed with
    :class:`RecommendationScoringValidationError`. The returned
    :class:`~music_agent.recommendation_contract.ScoreBreakdown` carries the fixed component
    vocabulary documented in the module docstring, with ``total = basis_support *
    basis_strength``. The result is deterministic and depends only on the candidate and the
    context, never on the clock, the database, or global state.
    """
    if not isinstance(candidate, Candidate):
        raise RecommendationScoringValidationError("candidate must be a Candidate")
    if not isinstance(context, RecommendationContext):
        raise RecommendationScoringValidationError("context must be a RecommendationContext")
    if candidate.eligibility is not Eligibility.ELIGIBLE:
        raise RecommendationScoringValidationError(
            "only an ELIGIBLE candidate can be scored"
        )

    basis_targets = candidate.basis_targets
    if not basis_targets:
        # Empty basis: no preference evidence at all, so the zero-evidence breakdown.
        return _breakdown(0.0, 0.0, 0.0, 0.0)

    support_count = 0
    strength_sum = 0.0
    negative_count = 0
    direct_count = 0

    for target in basis_targets:
        inputs = [input_ for input_ in context.preference_inputs if input_.target == target]
        if not inputs:
            raise RecommendationScoringValidationError(
                f"no preference input for basis target {target!r}"
            )
        operative = _select_operative_input(inputs)
        if operative.provenance is PreferenceProvenance.DIRECT:
            direct_count += 1
        if operative.strength.state is PreferenceState.POSITIVE:
            support_count += 1
            strength_sum += operative.strength.magnitude
        elif operative.strength.state is PreferenceState.NEGATIVE:
            negative_count += 1

    count = len(basis_targets)
    support = support_count / count
    strength = strength_sum / support_count if support_count else 0.0
    negative = negative_count / count
    directness = direct_count / count
    return _breakdown(support, strength, negative, directness)


def _select_operative_input(inputs: list[PreferenceInput]) -> PreferenceInput:
    """Select the operative conclusion for one basis target from its matching inputs.

    ``inputs`` is the non-empty list of matching ``PreferenceInput`` values for one target. The
    frozen P06 fallback rule decides when both provenances coexist: an inferred conclusion may
    only fill a gap left by an unusable direct state (``UNKNOWN`` / ``INSUFFICIENT``); a formed
    direct conclusion always governs and an inferred input never overrides it.
    """
    direct = next(
        (input_ for input_ in inputs if input_.provenance is PreferenceProvenance.DIRECT),
        None,
    )
    inferred = next(
        (input_ for input_ in inputs if input_.provenance is PreferenceProvenance.INFERRED),
        None,
    )
    if direct is None:
        return inferred
    if inferred is None:
        return direct
    if is_inferred_fallback_eligible(direct.strength.state):
        return inferred
    return direct


def _breakdown(
    support: float, strength: float, negative: float, directness: float
) -> ScoreBreakdown:
    """Build the fixed-vocabulary breakdown; the total is ``support * strength``."""
    return ScoreBreakdown(
        support * strength,
        (
            ScoreComponent("basis_support", support),
            ScoreComponent("basis_strength", strength),
            ScoreComponent("negative_evidence", negative),
            ScoreComponent("provenance_directness", directness),
        ),
    )
