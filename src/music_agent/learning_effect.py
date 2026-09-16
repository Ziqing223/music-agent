"""Learning-effect contract (P08.4): what learning action an interpretation permits.

This module is the third derived layer of the feedback-learning loop. It sits between the
:class:`~music_agent.feedback_interpretation.FeedbackInterpretation` (P08.3) and the future
learning policy that will eventually change P06 preference state. It answers, in structured
data, "given this interpretation, what preference-learning action is permitted?" and nothing
more. It never answers "by how much should a score or confidence change?" -- no numeric weight,
magnitude, confidence delta, reinforcement multiplier, decay, aggregation, or reranking exists
here. It is a pure domain layer: deterministic, side-effect free, independent of SQLite rows,
external source payloads, the system clock, and global state. It never mutates the P06
preference model -- no preference head, revision, or conclusion is read or written -- and it
never persists anything: a :class:`LearningEffect` is a derived view recomputable from the
immutable observation history plus the versioned interpretation and effect policies.

The pipeline this slice completes structurally
----------------------------------------------

``FeedbackObservation``
    (P08.1, immutable evidence; persisted append-only by P08.2)
``FeedbackInterpretation``
    (P08.3, derived semantic view: what the observation means)
``LearningEffect``
    (this slice, derived categorical action: what learning is permitted)
``Learning policy / Preference update``
    (future P08 slice, deliberately absent here: the actual P06 head mutation)

Every effect references its interpretation in full, so every permitted action remains traceable
to its ``feedback_id``, its interpretation policy and contract versions, and the target and
attribution it concerns.

Frozen mapping semantics (policy v1)
------------------------------------

The only policy this contract ships is :class:`LearningEffectPolicy` version ``1``. Its rules are
frozen in this module; they are not configurable per call. For an interpretation with entity
target ``T`` (the observation's ``target``) and attribution ``A``:

- ``T`` is ``None`` -> ``NO_EFFECT`` (``NO_PREFERENCE_TARGET``). A recommendation-only
  observation names no P06 preference target, so no preference-learning action is permitted;
  naming one would require reading P07 run history, which is not this layer's job.
- interpretation ``POSITIVE`` with ``EXPLICIT`` origin -> ``POSITIVE_EVIDENCE``
  (``EXPLICIT_EVIDENCE``) concerning ``T``.
- interpretation ``POSITIVE`` with ``IMPLICIT`` origin -> ``POSITIVE_EVIDENCE``
  (``IMPLICIT_EVIDENCE``) concerning ``T``. The implicit origin stays visible on the effect, so
  the future learning policy can treat it differently from an explicit statement.
- interpretation ``NEGATIVE`` -> ``NEGATIVE_EVIDENCE`` (``EXPLICIT_EVIDENCE``) concerning
  ``T``. Under interpretation policy v1 negative claims only ever come from explicit statements,
  so there is no implicit-negative branch.
- interpretation ``NONE`` with attribution relation ``EXCLUDED`` -> ``ATTRIBUTION_EXCLUSION``
  (``ATTRIBUTION_EXCLUDED``): the feedback must not be attributed to the excluded aspect. The
  exclusion survives into the learning boundary as its own categorical action; the excluded
  aspect itself rides in the preserved attribution, never as a numeric discount.
- interpretation ``NONE`` otherwise -> ``NO_EFFECT`` (``NO_DIRECTIONAL_CLAIM``): a no-claim
  interpretation never silently becomes a learning effect.

Attribution is preserved exactly on every effect (``attribution`` delegates to the
interpretation's, which P08.3 already guarantees equals the observation's). In particular a
positive or negative evidence effect may carry an ``EXCLUDED`` attribution ("liked the track,
not because of the artist"), and the future learning policy is what decides how to apply that
exclusion -- this layer only guarantees the dimension survives.

PLAYED and aggregation stay deferred
------------------------------------

A ``PLAYED`` observation interprets to no directional claim (P08.3), so it maps to
``NO_EFFECT`` here. Frequency-based evidence ("repeated / frequent recent playback") is an
aggregated interpretation computed by a future slice over history; this contract supports it by
versioning -- that slice would introduce a new interpretation/effect policy version, never by
reclassifying a single ``PLAYED`` observation here.

Identity and version
--------------------

A :class:`LearningEffect` has no independent ID: it is derived, never persisted, and its identity
is ``(feedback_id, interpretation policy_version, effect policy_version, effect
contract_version)``. ``policy_version`` records the :class:`LearningEffectPolicy` version whose
frozen rules produced the effect, ``contract_version`` records
``LEARNING_EFFECT_CONTRACT_VERSION``, and the referenced interpretation carries its own policy
and contract versions. :func:`derive_learning_effect` is the single documented derivation
boundary: it validates both inputs, fails closed on an unknown policy version rather than
guessing, applies the frozen rules, and stamps the current contract version.

Persistence is deliberately absent
----------------------------------

No durable store for learning effects exists and none is introduced: recomputation from the
immutable history plus versioned policies is the preferred provenance, exactly as for P08.3. A
future slice that persists *applied preference updates* makes its own migration decision; that
is not this layer's concern and must not be smuggled in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackDirection,
    FeedbackExplicitness,
    FeedbackObservation,
)
from music_agent.feedback_interpretation import FeedbackInterpretation
from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference


class LearningEffectError(ValueError):
    code = "learning_effect_error"


class LearningEffectValidationError(LearningEffectError):
    code = "validation_error"


# The current learning-effect contract. It scopes the effect shape and semantics under which a
# permitted action is derived. Bump it on any material change to those semantics; it does not
# version the mapping rules (LearningEffectPolicy.version owns those).
LEARNING_EFFECT_CONTRACT_VERSION = 1


class LearningEffectKind(StrEnum):
    """The categorical learning action an interpretation permits.

    Every kind is a permitted action, never a magnitude: the future learning policy decides
    *how much* each kind changes preference state, and this vocabulary gives it nothing numeric
    to smuggle in.
    """

    POSITIVE_EVIDENCE = "positive_evidence"
    NEGATIVE_EVIDENCE = "negative_evidence"
    ATTRIBUTION_EXCLUSION = "attribution_exclusion"
    NO_EFFECT = "no_effect"


class LearningEffectReason(StrEnum):
    """The structured machine reason for the permitted action."""

    EXPLICIT_EVIDENCE = "explicit_evidence"
    IMPLICIT_EVIDENCE = "implicit_evidence"
    ATTRIBUTION_EXCLUDED = "attribution_excluded"
    NO_DIRECTIONAL_CLAIM = "no_directional_claim"
    NO_PREFERENCE_TARGET = "no_preference_target"


@dataclass(frozen=True, slots=True)
class LearningEffectPolicy:
    """The frozen, injected learning-effect policy whose version selects the ruleset.

    Version ``1`` is the only policy this contract ships; its rules are frozen in this module.
    A future version must be implemented as a new frozen ruleset here -- never by mutating the
    v1 mapping -- and the derivation function must fail closed on any version it does not know.
    """

    version: int

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise LearningEffectValidationError("version must be an integer")
        if self.version < 1:
            raise LearningEffectValidationError("version must be >= 1")


# The frozen mapping from effect kind to the directional claim the kind expresses. No other
# kind->direction combination is possible, so a LearningEffect can never disagree with itself.
_DIRECTION_BY_KIND: dict[LearningEffectKind, FeedbackDirection] = {
    LearningEffectKind.POSITIVE_EVIDENCE: FeedbackDirection.POSITIVE,
    LearningEffectKind.NEGATIVE_EVIDENCE: FeedbackDirection.NEGATIVE,
    LearningEffectKind.ATTRIBUTION_EXCLUSION: FeedbackDirection.NONE,
    LearningEffectKind.NO_EFFECT: FeedbackDirection.NONE,
}


@dataclass(frozen=True, slots=True)
class LearningEffect:
    """One derived categorical learning action permitted by one interpretation.

    ``interpretation`` is the derived semantic view the action is based on, carried in full so
    every effect is traceable to its ``feedback_id`` and interpretation policy/contract
    versions. ``kind`` is the permitted categorical action (``POSITIVE_EVIDENCE`` /
    ``NEGATIVE_EVIDENCE`` / ``ATTRIBUTION_EXCLUSION`` / ``NO_EFFECT``) and ``target`` names the
    P06 preference target the action concerns -- ``None`` only for ``NO_EFFECT``. ``reason`` is
    the structured machine reason for the action. ``policy_version`` records the learning-effect
    policy version and ``contract_version`` scopes this contract.

    The record is a permitted action, never a mutation and never a magnitude: it carries no
    weight, delta, multiplier, or confidence, and it does not touch P06 state.
    """

    interpretation: FeedbackInterpretation
    kind: LearningEffectKind
    target: PreferenceTargetReference | None
    reason: LearningEffectReason
    policy_version: int
    contract_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.interpretation, FeedbackInterpretation):
            raise LearningEffectValidationError(
                "interpretation must be a FeedbackInterpretation"
            )
        if not isinstance(self.kind, LearningEffectKind):
            raise LearningEffectValidationError("kind must be a LearningEffectKind")
        if self.target is not None and not isinstance(
            self.target, PreferenceTargetReference
        ):
            raise LearningEffectValidationError(
                "target must be a PreferenceTargetReference or None"
            )
        if self.kind is not LearningEffectKind.NO_EFFECT and self.target is None:
            raise LearningEffectValidationError(
                f"a {self.kind.value} effect requires a preference target"
            )
        if not isinstance(self.reason, LearningEffectReason):
            raise LearningEffectValidationError("reason must be a LearningEffectReason")
        _require_positive_int(self.policy_version, label="policy_version")
        _require_positive_int(self.contract_version, label="contract_version")

    @property
    def feedback_id(self) -> str:
        """The observation identity every permitted action is traceable to."""
        return self.interpretation.feedback_id

    @property
    def observation(self) -> FeedbackObservation:
        """The immutable observation the effect ultimately derives from."""
        return self.interpretation.observation

    @property
    def attribution(self) -> FeedbackAttribution | None:
        """The preserved attribution (``ATTRIBUTED`` / ``EXCLUDED``) of the observation.

        Interpretation policy v1 preserves attribution exactly, so the effect's attribution is
        the interpretation's and the observation's. Attribution exclusion therefore survives
        into the learning boundary unchanged.
        """
        return self.interpretation.attribution

    @property
    def explicitness(self) -> FeedbackExplicitness:
        """The explicit-vs-implicit origin of the evidence, carried through unchanged."""
        return self.interpretation.explicitness

    @property
    def direction(self) -> FeedbackDirection:
        """The directional claim expressed by the effect's kind.

        ``POSITIVE_EVIDENCE`` is ``POSITIVE``, ``NEGATIVE_EVIDENCE`` is ``NEGATIVE``, and
        ``ATTRIBUTION_EXCLUSION`` / ``NO_EFFECT`` carry no directional claim.
        """
        return _DIRECTION_BY_KIND[self.kind]


def derive_learning_effect(
    interpretation: FeedbackInterpretation,
    policy: LearningEffectPolicy,
    *,
    resolved_target: PreferenceTargetReference | None = None,
) -> LearningEffect:
    """Derive the permitted learning action for one interpretation under one policy version.

    ``interpretation`` is the derived semantic view and ``policy`` selects the frozen ruleset;
    both are validated fail-closed -- a non-interpretation, a non-policy, or an unknown policy
    version raise rather than guess an action. The current
    ``LEARNING_EFFECT_CONTRACT_VERSION`` is stamped automatically. This is the single documented
    derivation boundary; it reads neither the clock nor any store, so the effect is
    deterministic for the same inputs.

    P10 cross-phase amendment (explicit, optional): ``resolved_target`` is the strict
    recommendation-provenance resolution produced by the recommendation-feedback bridge
    for observations whose own ``target`` is None. It substitutes ONLY when the
    observation carries no target; supplying it alongside a target-bearing observation
    fails closed (ambiguity is never resolved silently). Omitted -> the frozen v1
    behavior is byte-identical.
    """
    if not isinstance(interpretation, FeedbackInterpretation):
        raise LearningEffectValidationError(
            "interpretation must be a FeedbackInterpretation"
        )
    if not isinstance(policy, LearningEffectPolicy):
        raise LearningEffectValidationError("policy must be a LearningEffectPolicy")
    if policy.version != 1:
        raise LearningEffectValidationError(
            f"unsupported learning-effect policy version {policy.version}"
        )
    if resolved_target is not None:
        if not isinstance(resolved_target, PreferenceTargetReference):
            raise LearningEffectValidationError(
                "resolved_target must be a PreferenceTargetReference"
            )
        if resolved_target.kind is not PreferenceTargetKind.TRACK:
            raise LearningEffectValidationError("resolved_target must reference a TRACK")
        if interpretation.observation.target is not None:
            raise LearningEffectValidationError(
                "resolved_target is only valid when the observation carries no target"
            )
    return _derive_v1(interpretation, policy, resolved_target)


def _derive_v1(
    interpretation: FeedbackInterpretation,
    policy: LearningEffectPolicy,
    resolved_target: PreferenceTargetReference | None = None,
) -> LearningEffect:
    attribution = interpretation.attribution
    target = interpretation.observation.target
    if target is None and resolved_target is not None:
        target = resolved_target  # P10 bridge: strict recommendation-provenance resolution

    if target is None:
        kind = LearningEffectKind.NO_EFFECT
        reason = LearningEffectReason.NO_PREFERENCE_TARGET
    elif interpretation.direction is FeedbackDirection.POSITIVE:
        kind = LearningEffectKind.POSITIVE_EVIDENCE
        reason = (
            LearningEffectReason.EXPLICIT_EVIDENCE
            if interpretation.explicitness is FeedbackExplicitness.EXPLICIT
            else LearningEffectReason.IMPLICIT_EVIDENCE
        )
    elif interpretation.direction is FeedbackDirection.NEGATIVE:
        kind = LearningEffectKind.NEGATIVE_EVIDENCE
        reason = LearningEffectReason.EXPLICIT_EVIDENCE
    elif attribution is not None and attribution.relation is AttributionRelation.EXCLUDED:
        kind = LearningEffectKind.ATTRIBUTION_EXCLUSION
        reason = LearningEffectReason.ATTRIBUTION_EXCLUDED
    else:
        kind = LearningEffectKind.NO_EFFECT
        reason = LearningEffectReason.NO_DIRECTIONAL_CLAIM

    return LearningEffect(
        interpretation=interpretation,
        kind=kind,
        target=target,
        reason=reason,
        policy_version=policy.version,
        contract_version=LEARNING_EFFECT_CONTRACT_VERSION,
    )


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LearningEffectValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise LearningEffectValidationError(f"{label} must be >= 1")
    return value
