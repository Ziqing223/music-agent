"""Bounded first-version recommendation quality controls (P07.6).

This module is the quality-control slice of the recommendation system: a deterministic,
explainable, testable policy that filters *scored* recommendation items before ranking. It
covers exactly the four remaining P07 roadmap goals, each as one named control:

``repeat_recommendation_control``
    Exclude items whose target was recommended in an earlier run (the injection of
    ``previous_targets`` comes from P07.5 history; :func:`collect_previous_targets` builds it
    from decoded results).

``scenario_relevance``
    Derive the deterministic first-version scenario from the context's reference instant
    (:func:`derive_scenario`, a daypart) and exclude items whose injected
    ``scenario_suitability`` evidence marks them unsuitable for it.

``diversity``
    Cap how many selected items may come from one diversity group (injected
    ``diversity_groups`` evidence); when a group exceeds the cap, the highest-scoring members
    survive and the rest are excluded.

``familiar_new_balance``
    Cap how many selected items may be unfamiliar (injected ``unfamiliar_targets`` evidence);
    when the cap is exceeded, the highest-scoring unfamiliar items survive and the rest are
    excluded.

Bounded-first-version rules
---------------------------

- Every control is pure and deterministic. ``keep-best`` decisions use the same canonical order
  as ranking: ``score.total`` descending, ``candidate_id`` ascending.
- A control applies only when its evidence is provided. Missing evidence never guesses: the
  control is skipped and the report records it as not applied, so absence of evidence is always
  visible and never silently treated as "everything is fine". Orphan parameters (a cap without
  its evidence) fail closed.
- Every exclusion carries a machine-code ``reason``; the report preserves the applied/skipped
  state and exclusions per control in fixed order, so the assembled recommendation stays
  explainable (the surviving items keep their original ``ScoreBreakdown`` -- quality controls
  never rewrite scores).
- ``scenario_suitability`` marks targets *known unsuitable* for scenarios; a target absent from
  the evidence is unrestricted, and a target present maps to the scenarios it is suitable for.
  The evidence's semantics are the caller's responsibility (e.g. derived from a catalog); this
  module never re-derives them.
- No ML, no feedback learning, no P08 behavior, no calibration framework, no frontend. The
  module is a pure domain layer: no SQLite, no clock (``now`` is injected), no randomness, and
  it never mutates P06.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable

from music_agent.preference_attribution import PreferenceTargetReference
from music_agent.recommendation_contract import (
    RecommendationContractValidationError,
    RecommendationItem,
    RecommendationResult,
    validate_candidate_id,
)


class RecommendationQualityError(ValueError):
    code = "recommendation_quality_error"


class RecommendationQualityValidationError(RecommendationQualityError):
    code = "validation_error"


class Scenario(StrEnum):
    """The deterministic first-version scenario vocabulary (time-derived dayparts)."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


#: Machine codes of the four quality controls, in the fixed application order.
CONTROL_REPEAT_RECOMMENDATION = "repeat_recommendation_control"
CONTROL_SCENARIO_RELEVANCE = "scenario_relevance"
CONTROL_DIVERSITY = "diversity"
CONTROL_FAMILIAR_NEW_BALANCE = "familiar_new_balance"
_CONTROL_ORDER = (
    CONTROL_REPEAT_RECOMMENDATION,
    CONTROL_SCENARIO_RELEVANCE,
    CONTROL_DIVERSITY,
    CONTROL_FAMILIAR_NEW_BALANCE,
)

#: Machine codes of the exclusion reasons, one per control.
REASON_PREVIOUSLY_RECOMMENDED = "previously_recommended"
REASON_SCENARIO_UNSUITABLE = "scenario_unsuitable"
REASON_DIVERSITY_GROUP_CAP = "diversity_group_cap"
REASON_UNFAMILIAR_CAP = "unfamiliar_cap"

#: The complete exclusion-reason vocabulary; a QualityExclusion never carries anything else.
_EXCLUSION_REASONS = frozenset(
    {
        REASON_PREVIOUSLY_RECOMMENDED,
        REASON_SCENARIO_UNSUITABLE,
        REASON_DIVERSITY_GROUP_CAP,
        REASON_UNFAMILIAR_CAP,
    }
)


def derive_scenario(now: datetime) -> Scenario:
    """Derive the deterministic daypart scenario from the injected reference instant.

    Boundaries use the aware datetime's own hour: ``[6, 12)`` morning, ``[12, 18)`` afternoon,
    ``[18, 24)`` evening, ``[0, 6)`` night. The result depends only on ``now``, never on the
    system clock. A non-timezone-aware datetime fails closed.
    """
    if not isinstance(now, datetime):
        raise RecommendationQualityValidationError(
            f"now must be a datetime, not {type(now).__name__}"
        )
    if now.tzinfo is None or now.utcoffset() is None:
        raise RecommendationQualityValidationError("now must be timezone-aware")
    hour = now.hour
    if 6 <= hour < 12:
        return Scenario.MORNING
    if 12 <= hour < 18:
        return Scenario.AFTERNOON
    if 18 <= hour < 24:
        return Scenario.EVENING
    return Scenario.NIGHT


@dataclass(frozen=True, slots=True)
class QualityExclusion:
    """One item excluded by one quality control, with a machine-code reason."""

    candidate_id: str
    target: PreferenceTargetReference
    reason: str

    def __post_init__(self) -> None:
        try:
            validate_candidate_id(self.candidate_id)
        except RecommendationContractValidationError as error:
            raise RecommendationQualityValidationError(
                f"candidate_id must use the contract's cnd_ namespace, got {self.candidate_id!r}"
            ) from error
        if not isinstance(self.target, PreferenceTargetReference):
            raise RecommendationQualityValidationError(
                "target must be a PreferenceTargetReference"
            )
        if self.reason not in _EXCLUSION_REASONS:
            raise RecommendationQualityValidationError(
                f"reason must be one of the exclusion-reason machine codes, got {self.reason!r}"
            )


@dataclass(frozen=True, slots=True)
class QualityControlDecision:
    """The outcome of one control: whether it applied and what it excluded."""

    control: str
    applied: bool
    exclusions: tuple[QualityExclusion, ...]

    def __post_init__(self) -> None:
        if self.control not in _CONTROL_ORDER:
            raise RecommendationQualityValidationError(f"unknown quality control {self.control!r}")
        if not isinstance(self.applied, bool):
            raise RecommendationQualityValidationError("applied must be a bool")
        exclusions = _coerce_exclusions(self.exclusions)
        if not self.applied and exclusions:
            raise RecommendationQualityValidationError(
                "a skipped control must not carry exclusions"
            )
        object.__setattr__(self, "exclusions", exclusions)


@dataclass(frozen=True, slots=True)
class QualityControlReport:
    """The explainability record of one quality-policy application."""

    scenario: Scenario
    decisions: tuple[QualityControlDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, Scenario):
            raise RecommendationQualityValidationError("scenario must be a Scenario")
        decisions = tuple(self.decisions)
        if len(decisions) != len(_CONTROL_ORDER) or any(
            decision.control != control
            for decision, control in zip(decisions, _CONTROL_ORDER)
        ):
            raise RecommendationQualityValidationError(
                "decisions must contain one QualityControlDecision per control in fixed order"
            )
        object.__setattr__(self, "decisions", decisions)


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    """Injected, read-only evidence for the quality policy.

    Every field is optional; a ``None`` field means the corresponding control is skipped (see the
    module docstring). ``max_unfamiliar`` is required exactly when ``unfamiliar_targets`` is
    provided, and ``max_per_group`` exactly when ``diversity_groups`` is provided -- orphan
    parameters fail closed.
    """

    previous_targets: frozenset[PreferenceTargetReference] | None = None
    unfamiliar_targets: frozenset[PreferenceTargetReference] | None = None
    max_unfamiliar: int | None = None
    diversity_groups: dict[str, frozenset[PreferenceTargetReference]] | None = None
    max_per_group: int | None = None
    scenario_suitability: dict[PreferenceTargetReference, frozenset[Scenario]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "previous_targets", _coerce_target_set(self.previous_targets, "previous_targets")
        )
        object.__setattr__(
            self, "unfamiliar_targets", _coerce_target_set(self.unfamiliar_targets, "unfamiliar_targets")
        )
        object.__setattr__(
            self,
            "scenario_suitability",
            _coerce_scenario_suitability(self.scenario_suitability),
        )
        object.__setattr__(
            self, "diversity_groups", _coerce_diversity_groups(self.diversity_groups)
        )
        if self.unfamiliar_targets is None:
            if self.max_unfamiliar is not None:
                raise RecommendationQualityValidationError(
                    "max_unfamiliar requires unfamiliar_targets evidence"
                )
        else:
            _require_non_negative_int(self.max_unfamiliar, "max_unfamiliar")
        if self.diversity_groups is None:
            if self.max_per_group is not None:
                raise RecommendationQualityValidationError(
                    "max_per_group requires diversity_groups evidence"
                )
        else:
            _require_positive_int(self.max_per_group, "max_per_group")


@dataclass(frozen=True, slots=True)
class QualityOutcome:
    """The survivors of the quality policy plus its explainability report."""

    selected_items: tuple[RecommendationItem, ...]
    report: QualityControlReport

    def __post_init__(self) -> None:
        items = _coerce_items(self.selected_items)
        if not isinstance(self.report, QualityControlReport):
            raise RecommendationQualityValidationError(
                "report must be a QualityControlReport"
            )
        object.__setattr__(self, "selected_items", items)


def apply_quality_policy(
    items: Iterable[RecommendationItem],
    now: datetime,
    evidence: QualityEvidence | None,
) -> QualityOutcome:
    """Apply the quality policy to scored items in fixed control order.

    ``items`` are the scored eligible candidates (any order; survivors keep their relative input
    order). ``now`` is the injected reference instant for scenario derivation. ``evidence``
    carries the injected per-control evidence; ``None`` skips every control (the report still
    records the derived scenario and the skipped decisions). Controls run in the fixed order
    repeat -> scenario -> diversity -> familiar/new, each seeing only the survivors of the ones
    before it. Wrong types fail closed with
    :class:`RecommendationQualityValidationError`.
    """
    items = _coerce_items(items)
    if evidence is not None and not isinstance(evidence, QualityEvidence):
        raise RecommendationQualityValidationError("evidence must be a QualityEvidence or None")
    scenario = derive_scenario(now)
    decisions: list[QualityControlDecision] = []
    selected: list[RecommendationItem] = list(items)

    if evidence is None or evidence.previous_targets is None:
        decisions.append(QualityControlDecision(CONTROL_REPEAT_RECOMMENDATION, False, ()))
    else:
        selected = _filter_with_decision(
            selected,
            CONTROL_REPEAT_RECOMMENDATION,
            REASON_PREVIOUSLY_RECOMMENDED,
            lambda item: item.candidate.target in evidence.previous_targets,
            decisions,
        )

    if evidence is None or evidence.scenario_suitability is None:
        decisions.append(QualityControlDecision(CONTROL_SCENARIO_RELEVANCE, False, ()))
    else:
        selected = _filter_with_decision(
            selected,
            CONTROL_SCENARIO_RELEVANCE,
            REASON_SCENARIO_UNSUITABLE,
            lambda item: item.candidate.target in evidence.scenario_suitability
            and scenario not in evidence.scenario_suitability[item.candidate.target],
            decisions,
        )

    if evidence is None or evidence.diversity_groups is None:
        decisions.append(QualityControlDecision(CONTROL_DIVERSITY, False, ()))
    else:
        selected = _apply_diversity(selected, evidence.diversity_groups, evidence.max_per_group, decisions)

    if evidence is None or evidence.unfamiliar_targets is None:
        decisions.append(QualityControlDecision(CONTROL_FAMILIAR_NEW_BALANCE, False, ()))
    else:
        selected = _apply_familiar_new_balance(
            selected, evidence.unfamiliar_targets, evidence.max_unfamiliar, decisions
        )

    return QualityOutcome(tuple(selected), QualityControlReport(scenario, tuple(decisions)))


def collect_previous_targets(
    results: Iterable[RecommendationResult],
) -> frozenset[PreferenceTargetReference]:
    """Collect every target recommended by the given decoded results (repeat-control evidence).

    ``results`` is an iterable of decoded ``RecommendationResult`` values -- typically the
    recency slice the caller obtained from ``RecommendationHistoryRepository.list_runs()``
    (the caller owns the window: e.g. the bounded ``avoid_previous_runs`` window), and the
    returned frozenset names every recommended target across them. Wrong types fail closed.
    """
    if isinstance(results, (str, bytes)) or not hasattr(results, "__iter__"):
        raise RecommendationQualityValidationError(
            f"results must be an iterable, not {type(results).__name__}"
        )
    targets: set[PreferenceTargetReference] = set()
    for result in results:
        if not isinstance(result, RecommendationResult):
            raise RecommendationQualityValidationError(
                "each results entry must be a RecommendationResult"
            )
        targets.update(item.candidate.target for item in result.items)
    return frozenset(targets)


# --- control implementations ---------------------------------------------


def _apply_diversity(
    selected: list[RecommendationItem],
    groups: dict[str, frozenset[PreferenceTargetReference]],
    max_per_group: int,
    decisions: list[QualityControlDecision],
) -> list[RecommendationItem]:
    survivors: list[RecommendationItem] = []
    exclusions: list[QualityExclusion] = []
    seen_groups: dict[str, list[RecommendationItem]] = {}
    for item in selected:
        owning = [name for name, members in groups.items() if item.candidate.target in members]
        if len(owning) > 1:
            raise RecommendationQualityValidationError(
                f"target {item.candidate.target!r} belongs to multiple diversity groups: "
                f"{owning}"
            )
        if not owning:
            survivors.append(item)
            continue
        seen_groups.setdefault(owning[0], []).append(item)
    for name, members in seen_groups.items():
        ordered = sorted(members, key=_canonical_order_key)
        survivors.extend(ordered[:max_per_group])
        for item in ordered[max_per_group:]:
            exclusions.append(
                QualityExclusion(
                    item.candidate.candidate_id, item.candidate.target, REASON_DIVERSITY_GROUP_CAP
                )
            )
    survivors.sort(key=lambda item: selected.index(item))
    decisions.append(QualityControlDecision(CONTROL_DIVERSITY, True, tuple(exclusions)))
    return survivors


def _apply_familiar_new_balance(
    selected: list[RecommendationItem],
    unfamiliar_targets: frozenset[PreferenceTargetReference],
    max_unfamiliar: int,
    decisions: list[QualityControlDecision],
) -> list[RecommendationItem]:
    unfamiliar = [item for item in selected if item.candidate.target in unfamiliar_targets]
    if len(unfamiliar) <= max_unfamiliar:
        decisions.append(QualityControlDecision(CONTROL_FAMILIAR_NEW_BALANCE, True, ()))
        return selected
    ordered = sorted(unfamiliar, key=_canonical_order_key)
    kept_ids = {item.candidate.candidate_id for item in ordered[:max_unfamiliar]}
    exclusions = [
        QualityExclusion(item.candidate.candidate_id, item.candidate.target, REASON_UNFAMILIAR_CAP)
        for item in ordered[max_unfamiliar:]
    ]
    decisions.append(
        QualityControlDecision(CONTROL_FAMILIAR_NEW_BALANCE, True, tuple(exclusions))
    )
    return [item for item in selected if item.candidate.candidate_id in kept_ids
            or item.candidate.target not in unfamiliar_targets]


def _filter_with_decision(
    selected: list[RecommendationItem],
    control: str,
    reason: str,
    excluded_predicate,
    decisions: list[QualityControlDecision],
) -> list[RecommendationItem]:
    """Filter ``selected`` by a predicate, recording one decision for the whole control."""
    survivors: list[RecommendationItem] = []
    exclusions: list[QualityExclusion] = []
    for item in selected:
        if excluded_predicate(item):
            exclusions.append(
                QualityExclusion(item.candidate.candidate_id, item.candidate.target, reason)
            )
        else:
            survivors.append(item)
    decisions.append(QualityControlDecision(control, True, tuple(exclusions)))
    return survivors


def _canonical_order_key(item: RecommendationItem) -> tuple[float, str]:
    return (-item.score.total, item.candidate.candidate_id)


# --- validation helpers ----------------------------------------------------


def _coerce_items(value: object) -> tuple[RecommendationItem, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise RecommendationQualityValidationError(
            f"items must be an iterable, not {type(value).__name__}"
        )
    items = tuple(value)
    if any(not isinstance(item, RecommendationItem) for item in items):
        raise RecommendationQualityValidationError(
            "each items entry must be a RecommendationItem"
        )
    return items


def _coerce_exclusions(value: object) -> tuple[QualityExclusion, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise RecommendationQualityValidationError("exclusions must be an iterable")
    exclusions = tuple(value)
    if any(not isinstance(exclusion, QualityExclusion) for exclusion in exclusions):
        raise RecommendationQualityValidationError(
            "each exclusions entry must be a QualityExclusion"
        )
    return exclusions


def _coerce_target_set(
    value: object, label: str
) -> frozenset[PreferenceTargetReference] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise RecommendationQualityValidationError(f"{label} must be a set-like iterable or None")
    targets: set[PreferenceTargetReference] = set()
    for entry in value:
        if not isinstance(entry, PreferenceTargetReference):
            raise RecommendationQualityValidationError(
                f"each {label} entry must be a PreferenceTargetReference"
            )
        try:
            targets.add(entry)
        except TypeError as error:
            raise RecommendationQualityValidationError(
                f"{label} entries must be hashable PreferenceTargetReference values"
            ) from error
    return frozenset(targets)


def _coerce_scenario_suitability(
    value: object,
) -> dict[PreferenceTargetReference, frozenset[Scenario]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RecommendationQualityValidationError(
            "scenario_suitability must be a dict or None"
        )
    coerced: dict[PreferenceTargetReference, frozenset[Scenario]] = {}
    for target, scenarios in value.items():
        if not isinstance(target, PreferenceTargetReference):
            raise RecommendationQualityValidationError(
                "each scenario_suitability key must be a PreferenceTargetReference"
            )
        if isinstance(scenarios, (str, bytes)) or not hasattr(scenarios, "__iter__"):
            raise RecommendationQualityValidationError(
                "each scenario_suitability value must be a set-like iterable"
            )
        scenario_set: set[Scenario] = set()
        for entry in scenarios:
            if not isinstance(entry, Scenario):
                raise RecommendationQualityValidationError(
                    "each scenario_suitability entry must be a Scenario"
                )
            try:
                scenario_set.add(entry)
            except TypeError as error:
                raise RecommendationQualityValidationError(
                    "scenario_suitability entries must be hashable Scenario values"
                ) from error
        coerced[target] = frozenset(scenario_set)
    return coerced


def _coerce_diversity_groups(
    value: object,
) -> dict[str, frozenset[PreferenceTargetReference]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RecommendationQualityValidationError("diversity_groups must be a dict or None")
    coerced: dict[str, frozenset[PreferenceTargetReference]] = {}
    for name, members in value.items():
        if not isinstance(name, str) or name == "":
            raise RecommendationQualityValidationError(
                "each diversity_groups key must be a non-empty string"
            )
        if isinstance(members, (str, bytes)) or not hasattr(members, "__iter__"):
            raise RecommendationQualityValidationError(
                "each diversity_groups value must be a set-like iterable"
            )
        member_set: set[PreferenceTargetReference] = set()
        for entry in members:
            if not isinstance(entry, PreferenceTargetReference):
                raise RecommendationQualityValidationError(
                    "each diversity_groups entry must be a PreferenceTargetReference"
                )
            try:
                member_set.add(entry)
            except TypeError as error:
                raise RecommendationQualityValidationError(
                    "diversity_groups entries must be hashable PreferenceTargetReference values"
                ) from error
        coerced[name] = frozenset(member_set)
    return coerced


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecommendationQualityValidationError(f"{label} must be an integer")
    if value < 1:
        raise RecommendationQualityValidationError(f"{label} must be >= 1")
    return value


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecommendationQualityValidationError(f"{label} must be an integer")
    if value < 0:
        raise RecommendationQualityValidationError(f"{label} must be >= 0")
    return value
