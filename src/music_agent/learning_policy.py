"""Conservative learning policy (P08.5, corrected at the P08.6 gate): a bounded, P06-faithful
proposed update, never an applied one.

This module is the first *policy* layer of the feedback-learning loop. It converts the
categorical :class:`~music_agent.learning_effect.LearningEffect` (P08.4) into a
:class:`ProposedPreferenceUpdate` -- a bounded, deterministic proposal of what the P06
preference model *should* learn -- and nothing more. It answers "given this permitted learning
effect, what bounded update should be proposed?" and never "has the user's persisted Preference
Model actually changed?" It is a pure domain layer: deterministic, side-effect free, independent
of SQLite rows, the system clock, and global state, and it **never writes to the P06
persistence** -- no ``record_observation`` call exists here; the application slice (P08.6)
decides when and whether to apply a proposal.

The pipeline this slice completes structurally
----------------------------------------------

``FeedbackObservation`` -> ``FeedbackInterpretation`` -> ``LearningEffect`` ->
``LearningPolicy`` -> ``ProposedPreferenceUpdate`` -> P08.6 application to P06.

P08.6 gate correction: why this contract is evidence-based, not magnitude-based
------------------------------------------------------------------------------

Policy v1 as first shipped proposed numeric ``PreferenceStrength`` magnitudes (explicit ``0.8``,
implicit ``0.5``). The P08.6 hard gate inspected the actual P06 implementation and proved those
magnitudes were **not persistable and not derivable**:

- P06's single write boundary records a three-state *semantic observation* (``VALUE`` /
  ``MISSING`` / ``NULL``) against a stable :class:`~music_agent.preference_persistence.SignalIdentity`;
  it accepts no numeric strength or delta.
- P06 derives :class:`~music_agent.preference_strength.PreferenceStrength` at query time from
  exactly the three frozen signal paths ``favorited`` / ``disliked`` / ``rating`` under one
  ``source_system``, through S1/S3, with one injected per-direction magnitude policy
  (:class:`~music_agent.direct_track_preference.DirectPreferenceMagnitudePolicy`). No evidence
  class, provenance, or per-observation magnitude participates in that derivation.

Silently dropping the proposed magnitude at application was forbidden, so policy v1 is
**superseded by policy v2** (this module's current ruleset; ``LearningPolicy(1)`` fails closed).
Policy v2 proposes exactly what P06 can faithfully consume:

- **Direction** rides P06's frozen signal vocabulary under a dedicated source system
  (``feedback_learning``), so one feedback observation does durably change the derived
  preference direction: ``LIKED`` / ``DIRECTION_GOOD`` / ``FAVORITED`` / ``REPLAYED`` propose a
  ``VALUE(true)`` observation on the ``favorited`` path; ``DISLIKED`` proposes ``VALUE(true)``
  on the ``disliked`` path. Feedback evidence therefore participates in P06 derivation without
  touching the source-of-truth heads (those live under other source systems).
- **Evidence class** (explicit statement vs implicit behavior) is preserved durably as the
  revision provenance label: ``feedback_learning:explicit`` /
  ``feedback_learning:implicit``. P06 stores provenance per evidence revision, so the
  explicit-vs-implicit distinction survives application and readback even though the current
  P06 derivation does not (yet) consume it. The magnitude that ultimately enters recommendation
  scoring is P06's own query-time injected calibration policy -- deliberately not duplicated
  here.

Frozen policy rules (version 2)
-------------------------------

The only policy this contract ships is :class:`LearningPolicy` version ``2``; version ``1`` and
any unknown version fail closed. For a :class:`~music_agent.learning_effect.LearningEffect`
``E``:

- ``NO_EFFECT`` -> ``None``: no proposed mutation of any kind.
- ``ATTRIBUTION_EXCLUSION`` -> a proposal of kind ``ATTRIBUTION_EXCLUSION`` carrying no signal
  identity, no value, no provenance, and no timestamps. The exclusion is a directive to the
  application slice ("do not attribute this feedback to the excluded aspect"), never ordinary
  evidence -- in particular it can never become negative preference evidence, because it
  proposes no observation at all.
- ``POSITIVE_EVIDENCE`` / ``NEGATIVE_EVIDENCE`` -> a proposal of kind ``EVIDENCE_OBSERVATION``:
  one proposed semantic observation ``VALUE(true)`` against a
  :class:`~music_agent.preference_persistence.SignalIdentity`` whose ``source_system`` is
  ``feedback_learning`` and whose ``signal_path`` is the P06-frozen path carrying the direction
  (``favorited`` for positive evidence, ``disliked`` for negative evidence), plus the class
  provenance label and the feedback event times as ISO-8601 strings.

  ``SKIPPED`` and ``COMPLETED`` carry no directional claim (P08.3) and therefore never reach an
  evidence effect, so no proposal exists for them; ``PLAYED`` remains deferred pending
  aggregation. An observation kind outside the frozen signal-path mapping fails closed rather
  than proposing a guessed path.

- Confidence: P06's confidence concept (:class:`~music_agent.confidence.ConfidenceComponents`)
  requires history-derived components (``quantity``, ``freshness``, ``consistency``) that a
  single pure proposal cannot honestly compute, so no confidence contribution is proposed here,
  and the application slice (P08.6) does not compute confidence either -- history-derived
  confidence remains deferred. The explicit-vs-implicit provenance this proposal carries is
  durable, but the current P06 derivation does not consume it for confidence.

Traceability and provenance
---------------------------

Every proposal references its :class:`~music_agent.learning_effect.LearningEffect` in full, so
it remains traceable to ``feedback_id``, the interpretation policy version, the learning-effect
policy version, the target, the attribution, and the explicit-vs-implicit origin. The proposed
``observed_at`` / ``event_at`` are carried as ISO-8601 strings taken directly from the feedback
observation, so an eventual application records the feedback event times, not application time.

Identity and version
--------------------

A :class:`ProposedPreferenceUpdate` has no independent ID: it is derived, never persisted, and
its identity is ``(feedback_id, interpretation policy_version, effect policy_version, policy
version, contract_version)``. ``policy_version`` records the :class:`LearningPolicy` version,
``contract_version`` records ``LEARNING_POLICY_CONTRACT_VERSION``, and the referenced effect
carries the upstream versions. :func:`propose_preference_update` is the single documented
derivation boundary: it validates its inputs, fails closed on unknown or superseded versions
and unmapped kinds, applies the frozen rules, and stamps the current contract version.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.feedback_contract import (
    FeedbackAttribution,
    FeedbackExplicitness,
    FeedbackKind,
)
from music_agent.learning_effect import (
    LearningEffect,
    LearningEffectKind,
    LearningEffectReason,
)
from music_agent.preference_attribution import PreferenceTargetReference
from music_agent.preference_persistence import SignalIdentity
from music_agent.source_observation import ObservedValue, ObservationState


class LearningPolicyError(ValueError):
    code = "learning_policy_error"


class LearningPolicyValidationError(LearningPolicyError):
    code = "validation_error"


# The current learning-policy contract. Version 2 removed the non-persistable numeric magnitude
# proposals of v1 (see the P08.6 gate correction in the module docstring) and introduced the
# evidence-class provenance encoding. Bump again on any material change to the proposal shape or
# semantics; the frozen rules themselves are versioned by LearningPolicy.version.
LEARNING_POLICY_CONTRACT_VERSION = 2

# The feedback-learning namespace: the proposed SignalIdentity source_system. Distinct from
# every source-of-truth source (for example ``apple_music``), so feedback evidence can never be
# confused with observed library state and never overwrites source-of-truth heads.
FEEDBACK_LEARNING_SOURCE_SYSTEM = "feedback_learning"

# The evidence-class provenance labels recorded on the eventual P06 evidence revisions, so the
# explicit-vs-implicit distinction survives application durably even though current P06
# derivation does not consume provenance.
EXPLICIT_FEEDBACK_PROVENANCE = "feedback_learning:explicit"
IMPLICIT_FEEDBACK_PROVENANCE = "feedback_learning:implicit"

# Frozen policy-v2 signal-path mapping: feedback kind -> the P06-frozen signal path carrying the
# direction into P06 derivation. Positive evidence rides ``favorited`` and negative evidence
# rides ``disliked`` under the feedback_learning source system; the finer per-kind distinction
# remains preserved in the durable feedback history (P08.2) and the application ledger (P08.6).
# Kinds that cannot produce evidence effects under the P08.3 / P08.4 frozen semantics are
# deliberately absent; reaching one of them here is a contract violation and fails closed.
_SIGNAL_PATH_BY_KIND = {
    FeedbackKind.LIKED: "favorited",
    FeedbackKind.DISLIKED: "disliked",
    FeedbackKind.DIRECTION_GOOD: "favorited",
    FeedbackKind.FAVORITED: "favorited",
    FeedbackKind.REPLAYED: "favorited",
}


class ProposedUpdateKind(StrEnum):
    """The kind of update a proposal carries."""

    EVIDENCE_OBSERVATION = "evidence_observation"
    ATTRIBUTION_EXCLUSION = "attribution_exclusion"


@dataclass(frozen=True, slots=True)
class LearningPolicy:
    """The frozen, injected learning policy whose version selects the ruleset.

    Version ``2`` is the only supported policy. Version ``1`` proposed numeric magnitudes the
    P08.6 gate proved non-persistable in P06; it is rejected as superseded rather than silently
    reinterpreted. A future version must be implemented as a new frozen ruleset here -- never by
    mutating the v2 constants -- and the proposal function must fail closed on any version it
    does not know.
    """

    version: int

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise LearningPolicyValidationError("version must be an integer")
        if self.version < 1:
            raise LearningPolicyValidationError("version must be >= 1")


@dataclass(frozen=True, slots=True)
class ProposedPreferenceUpdate:
    """One bounded, deterministic proposal derived from one permitted learning effect.

    ``effect`` is the permitted action the proposal is based on, carried in full for
    traceability. ``kind`` is ``EVIDENCE_OBSERVATION`` (one proposed semantic observation with
    its evidence-class provenance) or ``ATTRIBUTION_EXCLUSION`` (a directive never to attribute
    the feedback to the excluded aspect). For ``EVIDENCE_OBSERVATION``, ``signal_identity``
    names the P06 signal head the observation would target, ``proposed_value`` is the proposed
    three-state observation (always a ``VALUE``), and ``provenance`` is the evidence-class
    label that survives application durably. ``observed_at`` / ``event_at`` carry the feedback
    event times as ISO-8601 strings. ``policy_version`` records the :class:`LearningPolicy`
    version and ``contract_version`` scopes this contract.

    The record is a proposal, never an application: nothing here writes to P06, and it carries
    no numeric magnitude -- the magnitude that reaches recommendation scoring is P06's own
    query-time injected calibration, deliberately not duplicated here.
    """

    effect: LearningEffect
    kind: ProposedUpdateKind
    signal_identity: SignalIdentity | None = None
    proposed_value: ObservedValue | None = None
    provenance: str | None = None
    observed_at: str | None = None
    event_at: str | None = None
    policy_version: int = 2
    contract_version: int = LEARNING_POLICY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.effect, LearningEffect):
            raise LearningPolicyValidationError("effect must be a LearningEffect")
        if not isinstance(self.kind, ProposedUpdateKind):
            raise LearningPolicyValidationError("kind must be a ProposedUpdateKind")
        _require_positive_int(self.policy_version, label="policy_version")
        _require_positive_int(self.contract_version, label="contract_version")

        if self.kind is ProposedUpdateKind.EVIDENCE_OBSERVATION:
            if self.effect.kind not in (
                LearningEffectKind.POSITIVE_EVIDENCE,
                LearningEffectKind.NEGATIVE_EVIDENCE,
            ):
                raise LearningPolicyValidationError(
                    "an EVIDENCE_OBSERVATION proposal requires a positive or negative "
                    "evidence effect"
                )
            if not isinstance(self.signal_identity, SignalIdentity):
                raise LearningPolicyValidationError(
                    "an EVIDENCE_OBSERVATION proposal requires a signal_identity"
                )
            if self.signal_identity.target != self.effect.target:
                raise LearningPolicyValidationError(
                    "signal_identity.target must equal effect.target"
                )
            if not isinstance(self.proposed_value, ObservedValue) or (
                self.proposed_value.state is not ObservationState.VALUE
            ):
                raise LearningPolicyValidationError(
                    "an EVIDENCE_OBSERVATION proposal requires a VALUE observed value"
                )
            _require_non_empty_string(self.provenance, "provenance")
            _require_non_empty_string(self.observed_at, "observed_at")
            if self.event_at is not None:
                _require_non_empty_string(self.event_at, "event_at")
        else:
            if self.effect.kind is not LearningEffectKind.ATTRIBUTION_EXCLUSION:
                raise LearningPolicyValidationError(
                    "an ATTRIBUTION_EXCLUSION proposal requires an attribution-exclusion "
                    "effect"
                )
            if (
                self.signal_identity is not None
                or self.proposed_value is not None
                or self.provenance is not None
                or self.observed_at is not None
                or self.event_at is not None
            ):
                raise LearningPolicyValidationError(
                    "an ATTRIBUTION_EXCLUSION proposal must not carry a signal identity, "
                    "value, provenance, or timestamps"
                )

    @property
    def feedback_id(self) -> str:
        """The observation identity the proposal is traceable to."""
        return self.effect.feedback_id

    @property
    def target(self) -> PreferenceTargetReference | None:
        """The P06 preference target the proposal concerns."""
        return self.effect.target

    @property
    def attribution(self) -> FeedbackAttribution | None:
        """The preserved attribution, including any ``EXCLUDED`` aspect."""
        return self.effect.attribution

    @property
    def explicitness(self) -> FeedbackExplicitness:
        """The explicit-vs-implicit origin of the underlying evidence."""
        return self.effect.explicitness

    @property
    def reason(self) -> LearningEffectReason:
        """The learning-effect reason this proposal is based on."""
        return self.effect.reason


def propose_preference_update(
    effect: LearningEffect, policy: LearningPolicy
) -> ProposedPreferenceUpdate | None:
    """Propose the bounded update one permitted learning effect supports, or ``None``.

    ``effect`` is the permitted action and ``policy`` selects the frozen ruleset; both are
    validated fail-closed -- a non-effect, a non-policy, a superseded or unknown policy version,
    or an observation kind outside the frozen signal-path mapping raise rather than guess. A
    ``NO_EFFECT`` effect produces ``None``: no proposed mutation of any kind. The current
    ``LEARNING_POLICY_CONTRACT_VERSION`` is stamped automatically. This is the single documented
    derivation boundary; it reads neither the clock nor any store, so the proposal is
    deterministic for the same inputs.
    """
    if not isinstance(effect, LearningEffect):
        raise LearningPolicyValidationError("effect must be a LearningEffect")
    if not isinstance(policy, LearningPolicy):
        raise LearningPolicyValidationError("policy must be a LearningPolicy")
    if policy.version == 1:
        raise LearningPolicyValidationError(
            "learning policy v1 is superseded: it proposed numeric magnitudes that P06 cannot "
            "persist or derive (see the P08.6 gate correction); use policy v2"
        )
    if policy.version != 2:
        raise LearningPolicyValidationError(
            f"unsupported learning policy version {policy.version}"
        )
    return _propose_v2(effect, policy)


def _propose_v2(
    effect: LearningEffect, policy: LearningPolicy
) -> ProposedPreferenceUpdate | None:
    if effect.kind is LearningEffectKind.NO_EFFECT:
        return None
    if effect.kind is LearningEffectKind.ATTRIBUTION_EXCLUSION:
        return ProposedPreferenceUpdate(
            effect=effect,
            kind=ProposedUpdateKind.ATTRIBUTION_EXCLUSION,
            policy_version=policy.version,
            contract_version=LEARNING_POLICY_CONTRACT_VERSION,
        )

    observation = effect.observation
    signal_path = _SIGNAL_PATH_BY_KIND.get(observation.kind)
    if signal_path is None:
        raise LearningPolicyValidationError(
            f"feedback kind {observation.kind.value!r} has no frozen signal-path mapping"
        )
    provenance = (
        EXPLICIT_FEEDBACK_PROVENANCE
        if effect.explicitness is FeedbackExplicitness.EXPLICIT
        else IMPLICIT_FEEDBACK_PROVENANCE
    )
    return ProposedPreferenceUpdate(
        effect=effect,
        kind=ProposedUpdateKind.EVIDENCE_OBSERVATION,
        signal_identity=SignalIdentity(
            effect.target, FEEDBACK_LEARNING_SOURCE_SYSTEM, signal_path
        ),
        proposed_value=ObservedValue.value(True),
        provenance=provenance,
        observed_at=observation.observed_at.isoformat(),
        event_at=(
            None if observation.event_at is None else observation.event_at.isoformat()
        ),
        policy_version=policy.version,
        contract_version=LEARNING_POLICY_CONTRACT_VERSION,
    )


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LearningPolicyValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise LearningPolicyValidationError(f"{label} must be >= 1")
    return value


def _require_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise LearningPolicyValidationError(f"{field} must be a non-empty string")
