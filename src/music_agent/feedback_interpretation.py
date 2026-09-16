"""Derived feedback interpretation (P08.3): what one observation currently means.

This module is the *interpretation* layer of the feedback-learning loop, sitting between the
immutable :class:`~music_agent.feedback_contract.FeedbackObservation` history (P08.1/P08.2) and
the future learning effect on the P06 preference model. It answers, in structured data, "what
does this observation currently mean?" and nothing more. It never answers "how much should the
Preference Model change?" -- no preference mutation, no weights, no reinforcement magnitude, no
confidence, no decay, no aggregation, and no reranking exist here. It is a pure domain layer:
deterministic, side-effect free, and independent of SQLite rows, external source payloads, the
system clock, and global state. It never rewrites, deletes, or persists anything: the
:class:`~music_agent.feedback_contract.FeedbackObservation` remains the immutable, authoritative
evidence, and an interpretation is a derived *view over it*, exactly as P06 S9 explains derived
preferences without re-deriving them.

The boundary this slice enforces
--------------------------------

``FeedbackObservation``
    (P08.1) what happened -- immutable evidence, persisted append-only by P08.2.
``FeedbackInterpretation``
    (this slice) what the observation is believed to mean under a named, versioned policy.
``Learning effect``
    (future P08 slice) how P06 preference state changes -- deliberately absent here.

An interpretation references its observation in full (``observation``), so every claim is
traceable to its evidence through ``observation.feedback_id`` and the preserved provenance;
nothing about the observation is copied loosely and nothing is re-expressed without the source
attached.

Frozen interpretation semantics (policy v1)
-------------------------------------------

The only policy this contract ships is :class:`InterpretationPolicy` version ``1``. Its rules are
frozen in this module and deliberately conservative; they are not configurable per call:

- ``LIKED`` -> ``POSITIVE`` (``EXPLICIT_STATEMENT``): the user stated a positive response.
- ``DISLIKED`` -> ``NEGATIVE`` (``EXPLICIT_STATEMENT``): the user stated a negative response.
- ``DIRECTION_GOOD`` -> ``POSITIVE`` (``EXPLICIT_STATEMENT``): the user stated the direction is
  good; the direction itself is named by the observation's target or recommendation.
- ``CORRECTED`` -> ``NONE`` (``NO_PREFERENCE_CLAIM``): a correction is not preference-valence
  evidence, so no directional claim is derived.
- ``ATTRIBUTION_CORRECTION`` -> ``NONE`` (``NO_PREFERENCE_CLAIM``): same -- it corrects
  attribution, not valence. The ``ATTRIBUTED`` / ``EXCLUDED`` attribution is preserved on the
  interpretation unchanged, so the future learning slice can apply the correction where it
  belongs.
- ``FAVORITED`` -> ``POSITIVE`` (``IMPLICIT_BEHAVIOR``): a deliberate positive action, but
  implicit: the claim is direction-only, carries no strength, and is explicitly marked implicit
  so no later slice can mistake it for a strong explicit statement.
- ``REPLAYED`` -> ``POSITIVE`` (``IMPLICIT_BEHAVIOR``): deliberate re-engagement, same implicit
  status.
- ``SKIPPED`` -> ``NONE`` (``AMBIGUOUS_BEHAVIOR``): a skip does **not** mean dislike under this
  policy. It is ambiguous (could be mood, interruption, repetition), so no directional claim is
  derived and the ambiguity is preserved rather than resolved.
- ``COMPLETED`` -> ``NONE`` (``AMBIGUOUS_BEHAVIOR``): completing playback does **not** mean like
  under this policy. Same treatment as skip.
- ``PLAYED`` -> ``NONE`` (``REQUIRES_AGGREGATION``): one playback occurrence carries no
  directional claim by itself; "repeated / frequent recent playback" is an aggregate meaning
  computed by a future slice over multiple observations, never by this per-observation layer.

``NONE`` is the fail-closed outcome: every kind this policy cannot honestly read as positive or
negative evidence resolves to no directional claim with a structured machine reason, never to a
guessed direction. Explicit and implicit feedback remain structurally distinguishable: the
interpretation's ``explicitness`` is the observation's frozen per-kind explicitness, and an
implicit claim can never be constructed as explicit.

Attribution and provenance preservation
---------------------------------------

The interpretation preserves the observation's attribution exactly: ``attribution`` on the
interpretation must equal ``observation.attribution`` -- policy v1 never invents, drops, or
rewrites attribution. The future learning slice resolves what ``ATTRIBUTED`` / ``EXCLUDED``
means for preference state; this layer only guarantees the dimension survives interpretation
intact.

Identity and version
--------------------

An interpretation has no independent ID: it is derived, never persisted as a distinct entity,
and its identity is ``(observation.feedback_id, policy_version, contract_version)``. The two
versions are recorded on the interpretation so any claim is reproducible:

- ``policy_version`` -- the :class:`InterpretationPolicy` version whose frozen rules produced the
  interpretation. A future policy version must be added as a new frozen ruleset in this module
  (and a constructor/rules branch), never by mutating v1.
- ``contract_version`` -- ``INTERPRETATION_CONTRACT_VERSION``, scoping the interpretation shape
  and semantics; bump on any material change to those.

``interpret_observation`` is the single documented derivation boundary: it validates both inputs,
fails closed on an unknown policy version rather than guessing, applies the frozen rules, and
stamps the current contract version. The constructor itself validates shape only -- like P06 S1,
semantics live in the derivation function, never in free-form construction.

Persistence is deliberately absent
----------------------------------

This layer needs no durable store of its own: interpretations are views recomputed from the
immutable observation history (P08.2) plus a policy version, and persisting them would create a
second, denormalized source of derived state -- the exact kind of mutable derived history the
three-layer boundary exists to prevent. A canonical serialization is likewise not introduced:
there is no persistence or interchange consumer for it yet. If a future slice needs to persist
applied *learning effects*, that is that slice's own migration decision, not this layer's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.feedback_contract import (
    FeedbackAttribution,
    FeedbackDirection,
    FeedbackExplicitness,
    FeedbackKind,
    FeedbackObservation,
)


class FeedbackInterpretationError(ValueError):
    code = "feedback_interpretation_error"


class FeedbackInterpretationValidationError(FeedbackInterpretationError):
    code = "validation_error"


# The current interpretation contract. It scopes the interpretation shape and semantics under
# which a claim about an observation is produced. Bump it on any material change to those
# semantics; it does not version the policy rules (InterpretationPolicy.version owns those).
INTERPRETATION_CONTRACT_VERSION = 1


class InterpretationReason(StrEnum):
    """The structured machine reason for a derived directional claim.

    Every reason says *why* the interpretation derives the claim it does; none of them carries a
    magnitude, weight, or confidence. The vocabulary is frozen and owned by this module.
    """

    EXPLICIT_STATEMENT = "explicit_statement"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    NO_PREFERENCE_CLAIM = "no_preference_claim"
    AMBIGUOUS_BEHAVIOR = "ambiguous_behavior"
    REQUIRES_AGGREGATION = "requires_aggregation"


@dataclass(frozen=True, slots=True)
class InterpretationPolicy:
    """The frozen, injected interpretation policy whose version selects the ruleset.

    Version ``1`` is the only policy this contract ships; its rules are frozen in this module.
    A future version must be implemented as a new frozen ruleset here -- never by mutating the
    v1 mapping -- and the derivation function must fail closed on any version it does not know.
    The policy is a plain immutable value with no global mutable config, like P06 S1's
    ``RatingBandPolicy``.
    """

    version: int

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise FeedbackInterpretationValidationError("version must be an integer")
        if self.version < 1:
            raise FeedbackInterpretationValidationError("version must be >= 1")


# The frozen policy-v1 ruleset: kind -> (direction, reason). Conservative by design: implicit
# behaviors produce direction-only claims explicitly marked implicit, ambiguous behaviors and
# corrections produce no directional claim, and a single playback produces no claim because its
# meaning requires aggregation over history. Do not mutate this mapping; a new policy version is
# a new frozen mapping.
_RULES_BY_KIND: dict[FeedbackKind, tuple[FeedbackDirection, InterpretationReason]] = {
    FeedbackKind.LIKED: (FeedbackDirection.POSITIVE, InterpretationReason.EXPLICIT_STATEMENT),
    FeedbackKind.DISLIKED: (FeedbackDirection.NEGATIVE, InterpretationReason.EXPLICIT_STATEMENT),
    FeedbackKind.DIRECTION_GOOD: (
        FeedbackDirection.POSITIVE,
        InterpretationReason.EXPLICIT_STATEMENT,
    ),
    FeedbackKind.CORRECTED: (FeedbackDirection.NONE, InterpretationReason.NO_PREFERENCE_CLAIM),
    FeedbackKind.ATTRIBUTION_CORRECTION: (
        FeedbackDirection.NONE,
        InterpretationReason.NO_PREFERENCE_CLAIM,
    ),
    FeedbackKind.FAVORITED: (FeedbackDirection.POSITIVE, InterpretationReason.IMPLICIT_BEHAVIOR),
    FeedbackKind.REPLAYED: (FeedbackDirection.POSITIVE, InterpretationReason.IMPLICIT_BEHAVIOR),
    FeedbackKind.SKIPPED: (FeedbackDirection.NONE, InterpretationReason.AMBIGUOUS_BEHAVIOR),
    FeedbackKind.COMPLETED: (FeedbackDirection.NONE, InterpretationReason.AMBIGUOUS_BEHAVIOR),
    FeedbackKind.PLAYED: (FeedbackDirection.NONE, InterpretationReason.REQUIRES_AGGREGATION),
}


@dataclass(frozen=True, slots=True)
class FeedbackInterpretation:
    """One derived view over one immutable feedback observation.

    ``observation`` is the authoritative evidence the interpretation derives from, carried in
    full so every claim is traceable to its provenance through ``feedback_id``. ``direction`` is
    the derived directional claim (``POSITIVE`` / ``NEGATIVE`` / ``NONE`` -- ``NONE`` is the
    fail-closed no-claim outcome). ``reason`` is the structured machine reason for the claim.
    ``attribution`` must equal ``observation.attribution``: policy v1 preserves attribution
    exactly and never invents or drops it. ``policy_version`` names the frozen policy ruleset
    that produced the interpretation and ``contract_version`` scopes this contract.

    The record is a claim *about* the observation, never a rewrite of it and never a preference
    change: no magnitude, weight, confidence, or decay value exists on it by construction.
    """

    observation: FeedbackObservation
    direction: FeedbackDirection
    reason: InterpretationReason
    attribution: FeedbackAttribution | None
    policy_version: int
    contract_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, FeedbackObservation):
            raise FeedbackInterpretationValidationError(
                "observation must be a FeedbackObservation"
            )
        if not isinstance(self.direction, FeedbackDirection):
            raise FeedbackInterpretationValidationError(
                "direction must be a FeedbackDirection"
            )
        if not isinstance(self.reason, InterpretationReason):
            raise FeedbackInterpretationValidationError(
                "reason must be an InterpretationReason"
            )
        if self.attribution is not None and not isinstance(
            self.attribution, FeedbackAttribution
        ):
            raise FeedbackInterpretationValidationError(
                "attribution must be a FeedbackAttribution or None"
            )
        if self.attribution != self.observation.attribution:
            raise FeedbackInterpretationValidationError(
                "attribution must equal observation.attribution: interpretation preserves "
                "attribution exactly and never invents or drops it"
            )
        _require_positive_int(self.policy_version, label="policy_version")
        _require_positive_int(self.contract_version, label="contract_version")

    @property
    def feedback_id(self) -> str:
        """The observation identity every claim is traceable to."""
        return self.observation.feedback_id

    @property
    def explicitness(self) -> FeedbackExplicitness:
        """The observation's frozen explicitness, carried through interpretation unchanged.

        Explicit statements and implicit behaviors stay structurally distinguishable on the
        interpretation; an implicit claim can never be mistaken for an explicit one.
        """
        return self.observation.explicitness


def interpret_observation(
    observation: FeedbackObservation, policy: InterpretationPolicy
) -> FeedbackInterpretation:
    """Derive the interpretation of one observation under one policy version.

    ``observation`` is the immutable evidence and ``policy`` selects the frozen ruleset; both are
    validated fail-closed -- a non-observation, a non-policy, or an unknown policy version raise
    rather than guess a claim. The current ``INTERPRETATION_CONTRACT_VERSION`` is stamped
    automatically. This is the single documented derivation boundary; it reads neither the clock
    nor any store, so the interpretation is deterministic for the same inputs.
    """
    if not isinstance(observation, FeedbackObservation):
        raise FeedbackInterpretationValidationError(
            "observation must be a FeedbackObservation"
        )
    if not isinstance(policy, InterpretationPolicy):
        raise FeedbackInterpretationValidationError(
            "policy must be an InterpretationPolicy"
        )
    rules = _rules_for_policy(policy)
    direction, reason = rules[observation.kind]
    return FeedbackInterpretation(
        observation=observation,
        direction=direction,
        reason=reason,
        attribution=observation.attribution,
        policy_version=policy.version,
        contract_version=INTERPRETATION_CONTRACT_VERSION,
    )


def _rules_for_policy(
    policy: InterpretationPolicy,
) -> dict[FeedbackKind, tuple[FeedbackDirection, InterpretationReason]]:
    if policy.version == 1:
        return _RULES_BY_KIND
    raise FeedbackInterpretationValidationError(
        f"unsupported interpretation policy version {policy.version}"
    )


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeedbackInterpretationValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise FeedbackInterpretationValidationError(f"{label} must be >= 1")
    return value
