"""Recommendation contract shared by candidate generation, scoring, and persistence (P07.1).

This module is the *contract* layer of the recommendation system. It defines the stable, typed
data structures that connect the three downstream tracks -- candidate generation, scoring, and
persistence -- and nothing more. It is a pure domain layer: deterministic, side-effect free, and
independent of SQLite rows, external source payloads, the system clock, and global state. It never
generates candidates, never computes a score or a ranking, never reads or writes a database, and
never mutates the P06 preference model.

One recommendation, one pipeline
--------------------------------

A single recommendation flows through four immutable stages, each with its own type:

``RecommendationRequest``
    What the caller asked for: a :class:`RecommendationContext` (the read-only ambient inputs),
    the kind of entity to recommend (:class:`RecommendedItemKind`), and the maximum number of
    items to return.

``Candidate``
    An *unscored* item produced by candidate generation. It carries a stable ``candidate_id``,
    the entity it proposes to recommend, the structured source that produced it, the preference
    targets that motivated it, and an explicit eligibility status.

``RecommendationItem``
    A *scored* candidate produced by scoring: a :class:`Candidate` bound to a
    :class:`ScoreBreakdown`.

``RecommendationResult``
    The *ranked run* produced by ranking: a stable ``run_id``, the request it answered, the
    ordered items, and the production instant. This is the unit persistence stores.

The ordered ``items`` tuple of a result is the ranking: tuple index is rank, with no separate rank
field that could disagree with the order.

Preference inputs are read-only references
------------------------------------------

P07 consumes P06 conclusions through :class:`PreferenceInput`, an immutable snapshot of one
already-derived preference conclusion (its target, provenance, and frozen strength). P07 never
re-derives a preference and never mutates P06: it only *reads* a conclusion into a
:class:`PreferenceInput` via :func:`PreferenceInput.from_direct` /
:func:`PreferenceInput.from_inferred`, which copy the fixed provenance from the P06 type. There is
no operation in this module that writes to a preference head, revision, or conclusion.

Score and score breakdown boundary
----------------------------------

A score is *never* an opaque scalar and *never* collapses a categorical fact onto a number. The
contract fixes the value domain -- the ``total`` and every component ``value`` are finite reals in
``[0, 1]`` -- but does **not** fix how components combine into the total: that mapping is the
scoring algorithm's responsibility (Track B). A :class:`ScoreBreakdown` therefore carries a
bounded ``total`` and a non-empty, named component list, so a recommendation reason can always be
expressed as structured components rather than free text. Component names are opaque machine keys;
the scoring algorithm owns the vocabulary.

Eligibility and rejection belong to the candidate
-------------------------------------------------

Eligibility is decided at the candidate layer (Track A), not the score layer. A
:class:`Candidate` is either ``ELIGIBLE`` or ``REJECTED`` with a structured
:class:`Rejection` reason; the two states are never folded together (a rejected candidate carries
no score and never becomes a :class:`RecommendationItem`). Scoring only ever sees eligible
candidates, and the contract refuses to bind a score to a rejected candidate. The rejection
reason vocabulary is owned by candidate generation; the contract only fixes its shape (a non-empty
machine code, not prose).

Identity and version
--------------------

``run_id`` uses the ``rcm_`` namespace and ``candidate_id`` uses the ``cnd_`` namespace. Both are
opaque random UUIDv4 identities outside every existing namespace (canonical ``trk_`` / ``art_`` /
``alb_`` / ``pl_`` / ``pm_`` and the operational ``int_`` / ``att_`` / ``prb_`` / ``rec_``), so a
run or candidate ID can never be confused with a canonical, intent, attempt, probe, or recovery
ID. ``RECOMMENDATION_CONTRACT_VERSION`` scopes the serialization and identity semantics; bump it on
any material change to those semantics. :func:`assemble_recommendation_result` is the single
documented assembly boundary and stamps the current version.

Serialization
-------------

:func:`encode_recommendation_result` / :func:`decode_recommendation_result` give a canonical,
deterministic JSON round-trip for the persisted unit. The encoding is ordered, preserves item
order, and round-trips every field; it is the interchange form persistence (Track C) builds on.
Persistence itself, and any recommendation history table, are deferred and not implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength


class RecommendationContractError(ValueError):
    code = "recommendation_contract_error"


class RecommendationContractValidationError(RecommendationContractError):
    code = "validation_error"


# The current recommendation contract. It scopes the serialization and identity semantics under
# which a recommendation result is produced. Bump it on any material change to those semantics; it
# does not version candidate-generation strategy, scoring weights, or ranking rules.
RECOMMENDATION_CONTRACT_VERSION = 1

_RUN_ID_PREFIX = "rcm_"
_CANDIDATE_ID_PREFIX = "cnd_"


class RecommendedItemKind(StrEnum):
    """The kind of entity a recommendation can return."""

    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"


_RECOMMENDED_TARGET_KIND: dict[RecommendedItemKind, PreferenceTargetKind] = {
    RecommendedItemKind.TRACK: PreferenceTargetKind.TRACK,
    RecommendedItemKind.ARTIST: PreferenceTargetKind.ARTIST,
    RecommendedItemKind.ALBUM: PreferenceTargetKind.ALBUM,
}


class Eligibility(StrEnum):
    """Whether a candidate is eligible to be scored and recommended."""

    ELIGIBLE = "eligible"
    REJECTED = "rejected"


def generate_run_id() -> str:
    """Generate a stable recommendation-run identity.

    The ``rcm_`` prefix is outside ``identity.ENTITY_ID_PREFIX`` and outside the ``int_`` /
    ``att_`` / ``prb_`` / ``rec_`` operational namespaces, so a run ID can never be confused with a
    canonical, intent, attempt, probe, or recovery ID, and it is never derived from a request or a
    candidate.
    """
    return f"{_RUN_ID_PREFIX}{uuid4()}"


def validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.startswith(_RUN_ID_PREFIX):
        raise RecommendationContractValidationError(
            f"run_id must use the {_RUN_ID_PREFIX} namespace"
        )
    _require_uuid_suffix(run_id[len(_RUN_ID_PREFIX) :], label="run_id")


def generate_candidate_id() -> str:
    """Generate a stable candidate identity.

    The ``cnd_`` prefix is outside every canonical and operational namespace, so a candidate ID can
    never be confused with a run, canonical, intent, attempt, probe, or recovery ID, and it is never
    derived from the candidate's target or source.
    """
    return f"{_CANDIDATE_ID_PREFIX}{uuid4()}"


def validate_candidate_id(candidate_id: str) -> None:
    if not isinstance(candidate_id, str) or not candidate_id.startswith(_CANDIDATE_ID_PREFIX):
        raise RecommendationContractValidationError(
            f"candidate_id must use the {_CANDIDATE_ID_PREFIX} namespace"
        )
    _require_uuid_suffix(candidate_id[len(_CANDIDATE_ID_PREFIX) :], label="candidate_id")


@dataclass(frozen=True, slots=True)
class PreferenceInput:
    """A read-only reference to one already-derived P06 preference conclusion.

    ``target`` names the referenced preference target, ``provenance`` records whether the
    conclusion was observed directly or inferred, and ``strength`` is the frozen
    :class:`~music_agent.preference_strength.PreferenceStrength` conclusion. This is a snapshot,
    never a derivation: it copies an already-produced P06 value and offers no path to mutate it.
    """

    target: PreferenceTargetReference
    provenance: PreferenceProvenance
    strength: PreferenceStrength

    def __post_init__(self) -> None:
        if not isinstance(self.target, PreferenceTargetReference):
            raise RecommendationContractValidationError(
                "target must be a PreferenceTargetReference"
            )
        if not isinstance(self.provenance, PreferenceProvenance):
            raise RecommendationContractValidationError(
                "provenance must be a PreferenceProvenance"
            )
        if not isinstance(self.strength, PreferenceStrength):
            raise RecommendationContractValidationError("strength must be a PreferenceStrength")

    @classmethod
    def from_direct(cls, preference: DerivedPreference) -> PreferenceInput:
        """Build a read-only input from a direct preference conclusion."""
        if not isinstance(preference, DerivedPreference):
            raise RecommendationContractValidationError(
                "preference must be a DerivedPreference"
            )
        return cls(preference.target, PreferenceProvenance.DIRECT, preference.strength)

    @classmethod
    def from_inferred(cls, affinity: InferredAffinity) -> PreferenceInput:
        """Build a read-only input from an inferred affinity conclusion."""
        if not isinstance(affinity, InferredAffinity):
            raise RecommendationContractValidationError("affinity must be an InferredAffinity")
        return cls(affinity.target, PreferenceProvenance.INFERRED, affinity.strength)


@dataclass(frozen=True, slots=True)
class CandidateSourceReference:
    """Structured provenance of how a candidate was generated.

    ``source_system`` names the generating system and ``source_path`` names the generating path
    within it (for example the candidate-generation strategy). Both are opaque machine-oriented
    non-empty strings, never free prose: the specific strategy vocabulary is owned by candidate
    generation (Track A), while the contract fixes only the ``(source_system, source_path)`` shape.
    """

    source_system: str
    source_path: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.source_system, "source_system")
        _require_non_empty_string(self.source_path, "source_path")


@dataclass(frozen=True, slots=True)
class Rejection:
    """A structured, machine-oriented reason a candidate was rejected.

    ``reason`` is a non-empty machine code owned by candidate generation, never a free-text
    explanation. The contract fixes the shape; the vocabulary is Track A's responsibility.
    """

    reason: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class Candidate:
    """An unscored, eligibility-marked item produced by candidate generation.

    ``candidate_id`` is a stable ``cnd_`` identity. ``target`` is the entity proposed for
    recommendation and must be a recommendable kind (``TRACK`` / ``ARTIST`` / ``ALBUM``, never a
    genre key). ``source`` records where the candidate came from, ``basis_targets`` names the
    preference targets that motivated it (possibly empty for a non-preference-driven source), and
    ``eligibility`` / ``rejection`` record whether it may be scored. A ``REJECTED`` candidate must
    carry a :class:`Rejection` and an ``ELIGIBLE`` candidate must not.
    """

    candidate_id: str
    target: PreferenceTargetReference
    source: CandidateSourceReference
    basis_targets: tuple[PreferenceTargetReference, ...] = ()
    eligibility: Eligibility = Eligibility.ELIGIBLE
    rejection: Rejection | None = None

    def __post_init__(self) -> None:
        validate_candidate_id(self.candidate_id)
        if not isinstance(self.target, PreferenceTargetReference):
            raise RecommendationContractValidationError("target must be a PreferenceTargetReference")
        if self.target.kind is PreferenceTargetKind.GENRE:
            raise RecommendationContractValidationError(
                "candidate target must be a recommendable TRACK, ARTIST, or ALBUM"
            )
        if not isinstance(self.source, CandidateSourceReference):
            raise RecommendationContractValidationError("source must be a CandidateSourceReference")
        if not isinstance(self.eligibility, Eligibility):
            raise RecommendationContractValidationError("eligibility must be an Eligibility")

        basis = _coerce_references(self.basis_targets, "basis_targets")
        if self.eligibility is Eligibility.REJECTED and self.rejection is None:
            raise RecommendationContractValidationError(
                "a REJECTED candidate requires a rejection"
            )
        if self.eligibility is Eligibility.ELIGIBLE and self.rejection is not None:
            raise RecommendationContractValidationError(
                "an ELIGIBLE candidate must not carry a rejection"
            )
        if self.rejection is not None and not isinstance(self.rejection, Rejection):
            raise RecommendationContractValidationError("rejection must be a Rejection or None")

        object.__setattr__(self, "basis_targets", basis)


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """One named, bounded component of a recommendation score.

    ``name`` is an opaque machine key owned by the scoring algorithm and ``value`` is a finite real
    in ``[0, 1]``. A component is never free prose and never a categorical state folded onto a
    scalar.
    """

    name: str
    value: float

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        _require_unit_interval(self.value, label="value")


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """A bounded total score plus its named components.

    ``total`` is a finite real in ``[0, 1]`` and ``components`` is a non-empty, name-unique tuple of
    :class:`ScoreComponent` values. How the components combine into ``total`` is the scoring
    algorithm's responsibility and is not frozen here; the contract fixes only the value domain and
    the requirement that a score always decompose into named components (never an opaque scalar).
    """

    total: float
    components: tuple[ScoreComponent, ...]

    def __post_init__(self) -> None:
        _require_unit_interval(self.total, label="total")
        components = _coerce_components(self.components)
        if not components:
            raise RecommendationContractValidationError(
                "a score breakdown requires at least one component"
            )
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, slots=True)
class RecommendationItem:
    """A scored candidate: one :class:`Candidate` bound to its :class:`ScoreBreakdown`.

    Only an ``ELIGIBLE`` candidate may be scored; a rejected candidate never becomes an item, so
    rejection information is expressed at the candidate layer and never re-expressed here.
    """

    candidate: Candidate
    score: ScoreBreakdown

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise RecommendationContractValidationError("candidate must be a Candidate")
        if self.candidate.eligibility is not Eligibility.ELIGIBLE:
            raise RecommendationContractValidationError(
                "only an ELIGIBLE candidate can be scored"
            )
        if not isinstance(self.score, ScoreBreakdown):
            raise RecommendationContractValidationError("score must be a ScoreBreakdown")


@dataclass(frozen=True, slots=True)
class RecommendationContext:
    """The read-only ambient inputs and reference instant for one recommendation.

    ``now`` is the injected reference instant (never the system clock) and ``preference_inputs`` is
    the read-only set of P06 preference conclusions the recommendation is computed against. The two
    kinds of preference input (direct and inferred) are both carried here as
    :class:`PreferenceInput` snapshots and never re-derived.
    """

    now: datetime
    preference_inputs: tuple[PreferenceInput, ...]

    def __post_init__(self) -> None:
        _require_aware_datetime(self.now, label="now")
        inputs = _coerce_inputs(self.preference_inputs)
        object.__setattr__(self, "preference_inputs", inputs)


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    """What one recommendation was asked to produce.

    ``context`` carries the read-only inputs, ``recommended_kind`` selects the entity kind to
    recommend, and ``limit`` is the maximum number of items the caller asked for.
    """

    context: RecommendationContext
    recommended_kind: RecommendedItemKind
    limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.context, RecommendationContext):
            raise RecommendationContractValidationError("context must be a RecommendationContext")
        if not isinstance(self.recommended_kind, RecommendedItemKind):
            raise RecommendationContractValidationError(
                "recommended_kind must be a RecommendedItemKind"
            )
        _require_positive_int(self.limit, label="limit")


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    """A ranked recommendation run: the unit persistence stores.

    ``run_id`` is a stable ``rcm_`` identity. ``request`` is the full request the run answered, so a
    run is self-describing and reproducible from its own inputs. ``items`` is the ranked result:
    tuple index is rank, the order is fixed, and every item's target kind matches the request. A
    result may be empty (a valid "nothing to recommend" outcome). ``produced_at`` is the injected
    production instant (never the system clock) and gives runs a durable order for history;
    ``contract_version`` scopes the serialization semantics.
    """

    run_id: str
    request: RecommendationRequest
    items: tuple[RecommendationItem, ...]
    produced_at: datetime
    contract_version: int

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not isinstance(self.request, RecommendationRequest):
            raise RecommendationContractValidationError("request must be a RecommendationRequest")
        _require_aware_datetime(self.produced_at, label="produced_at")
        _require_positive_int(self.contract_version, label="contract_version")

        items = _coerce_items(self.items)
        expected_kind = _RECOMMENDED_TARGET_KIND[self.request.recommended_kind]
        for item in items:
            if item.candidate.target.kind is not expected_kind:
                raise RecommendationContractValidationError(
                    f"item target kind {item.candidate.target.kind.value} does not match "
                    f"requested kind {self.request.recommended_kind.value}"
                )
        object.__setattr__(self, "items", items)


def assemble_recommendation_result(
    request: RecommendationRequest,
    items: object,
    *,
    run_id: str,
    produced_at: datetime,
) -> RecommendationResult:
    """Assemble and validate a ranked recommendation result.

    ``request`` is the answered request and ``items`` is the ranked iterable of
    :class:`RecommendationItem` values (tuple order is rank). ``run_id`` and ``produced_at`` are
    injected by the caller -- this function reads neither the clock nor randomness, so the result
    is deterministic for the same inputs. The current ``RECOMMENDATION_CONTRACT_VERSION`` is stamped
    automatically. This is the single documented boundary the three tracks share for producing a
    result.
    """
    return RecommendationResult(
        run_id=run_id,
        request=request,
        items=items,
        produced_at=produced_at,
        contract_version=RECOMMENDATION_CONTRACT_VERSION,
    )


# --- canonical serialization ---------------------------------------------


def encode_recommendation_result(result: RecommendationResult) -> str:
    """Encode a recommendation result to its canonical JSON text form.

    The encoding is deterministic (sorted keys) and order-preserving (item order is rank). It is the
    interchange form persistence builds on; it does not persist anything itself.
    """
    if not isinstance(result, RecommendationResult):
        raise RecommendationContractValidationError("result must be a RecommendationResult")
    return json.dumps(
        _result_to_dict(result),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def decode_recommendation_result(text: str) -> RecommendationResult:
    """Decode a canonical JSON recommendation result back to a :class:`RecommendationResult`.

    Fails closed on a non-string payload, unparseable JSON, or a decoded structure outside the
    contract, rather than coercing an unknown value.
    """
    if not isinstance(text, str):
        raise RecommendationContractValidationError("encoded result must be a string")
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise RecommendationContractValidationError(
            f"recommendation result is not valid JSON: {error}"
        ) from error
    return _result_from_dict(data)


def _result_to_dict(result: RecommendationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "request": {
            "context": {
                "now": result.request.context.now.isoformat(),
                "preference_inputs": [
                    _input_to_dict(input_) for input_ in result.request.context.preference_inputs
                ],
            },
            "recommended_kind": result.request.recommended_kind.value,
            "limit": result.request.limit,
        },
        "items": [_item_to_dict(item) for item in result.items],
        "produced_at": result.produced_at.isoformat(),
        "contract_version": result.contract_version,
    }


def _input_to_dict(input_: PreferenceInput) -> dict[str, Any]:
    return {
        "target": _target_to_dict(input_.target),
        "provenance": input_.provenance.value,
        "strength": {
            "state": input_.strength.state.value,
            "magnitude": input_.strength.magnitude,
        },
    }


def _target_to_dict(target: PreferenceTargetReference) -> dict[str, str]:
    return {"kind": target.kind.value, "target_id": target.target_id}


def _item_to_dict(item: RecommendationItem) -> dict[str, Any]:
    return {
        "candidate": {
            "candidate_id": item.candidate.candidate_id,
            "target": _target_to_dict(item.candidate.target),
            "source": {
                "source_system": item.candidate.source.source_system,
                "source_path": item.candidate.source.source_path,
            },
            "basis_targets": [_target_to_dict(t) for t in item.candidate.basis_targets],
            "eligibility": item.candidate.eligibility.value,
            "rejection": (
                None if item.candidate.rejection is None else {"reason": item.candidate.rejection.reason}
            ),
        },
        "score": {
            "total": item.score.total,
            "components": [
                {"name": component.name, "value": component.value}
                for component in item.score.components
            ],
        },
    }


def _result_from_dict(data: object) -> RecommendationResult:
    try:
        request = RecommendationRequest(
            context=RecommendationContext(
                now=_decode_datetime(data["request"]["context"]["now"]),
                preference_inputs=tuple(
                    _input_from_dict(input_)
                    for input_ in data["request"]["context"]["preference_inputs"]
                ),
            ),
            recommended_kind=RecommendedItemKind(data["request"]["recommended_kind"]),
            limit=data["request"]["limit"],
        )
        items = tuple(_item_from_dict(item) for item in data["items"])
        return RecommendationResult(
            run_id=data["run_id"],
            request=request,
            items=items,
            produced_at=_decode_datetime(data["produced_at"]),
            contract_version=data["contract_version"],
        )
    except RecommendationContractValidationError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise RecommendationContractValidationError(
            f"malformed recommendation result: {error}"
        ) from error


def _input_from_dict(data: object) -> PreferenceInput:
    return PreferenceInput(
        target=_target_from_dict(data["target"]),
        provenance=PreferenceProvenance(data["provenance"]),
        strength=PreferenceStrength(
            PreferenceState(data["strength"]["state"]), data["strength"]["magnitude"]
        ),
    )


def _target_from_dict(data: object) -> PreferenceTargetReference:
    return PreferenceTargetReference(
        PreferenceTargetKind(data["kind"]), data["target_id"]
    )


def _item_from_dict(data: object) -> RecommendationItem:
    candidate = Candidate(
        candidate_id=data["candidate"]["candidate_id"],
        target=_target_from_dict(data["candidate"]["target"]),
        source=CandidateSourceReference(
            data["candidate"]["source"]["source_system"],
            data["candidate"]["source"]["source_path"],
        ),
        basis_targets=tuple(
            _target_from_dict(t) for t in data["candidate"]["basis_targets"]
        ),
        eligibility=Eligibility(data["candidate"]["eligibility"]),
        rejection=(
            None
            if data["candidate"]["rejection"] is None
            else Rejection(data["candidate"]["rejection"]["reason"])
        ),
    )
    return RecommendationItem(
        candidate=candidate,
        score=ScoreBreakdown(
            total=data["score"]["total"],
            components=tuple(
                ScoreComponent(component["name"], component["value"])
                for component in data["score"]["components"]
            ),
        ),
    )


def _decode_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise RecommendationContractValidationError("timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RecommendationContractValidationError("timestamp must be timezone-aware")
    return parsed


# --- validation helpers ---------------------------------------------------


def _require_uuid_suffix(suffix: object, *, label: str) -> None:
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise RecommendationContractValidationError(
            f"{label} suffix must be a canonical UUID"
        ) from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise RecommendationContractValidationError(f"{label} suffix must be a canonical UUID")


def _require_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise RecommendationContractValidationError(f"{field} must be a non-empty string")


def _require_unit_interval(value: object, *, label: str) -> int | float:
    """Return ``value`` as a finite real in ``[0, 1]``, failing closed otherwise.

    Booleans are rejected explicitly (``bool`` is an ``int`` subclass), as are ``str``, ``None``,
    and any other non-real type. ``float('nan')`` and infinities are rejected as non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecommendationContractValidationError(
            f"{label} must be an int or float, not {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise RecommendationContractValidationError(f"{label} must be finite")
    if not 0 <= value <= 1:
        raise RecommendationContractValidationError(f"{label} must be within [0, 1]")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecommendationContractValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise RecommendationContractValidationError(f"{label} must be >= 1")
    return value


def _require_aware_datetime(value: object, *, label: str) -> datetime:
    """Return ``value`` as a timezone-aware ``datetime``, failing closed otherwise."""
    if not isinstance(value, datetime):
        raise RecommendationContractValidationError(
            f"{label} must be a datetime, not {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise RecommendationContractValidationError(f"{label} must be timezone-aware")
    return value


def _iterable(value: object, label: str):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise RecommendationContractValidationError(
            f"{label} must be an iterable, not {type(value).__name__}"
        )
    return iter(value)


def _coerce_references(
    value: object, label: str
) -> tuple[PreferenceTargetReference, ...]:
    references: list[PreferenceTargetReference] = []
    seen: set[PreferenceTargetReference] = set()
    for item in _iterable(value, label):
        if not isinstance(item, PreferenceTargetReference):
            raise RecommendationContractValidationError(
                f"each {label} entry must be a PreferenceTargetReference"
            )
        if item in seen:
            raise RecommendationContractValidationError(f"duplicate {label} target {item!r}")
        seen.add(item)
        references.append(item)
    return tuple(references)


def _coerce_inputs(value: object) -> tuple[PreferenceInput, ...]:
    inputs: list[PreferenceInput] = []
    seen: set[tuple[PreferenceTargetReference, PreferenceProvenance]] = set()
    for item in _iterable(value, "preference_inputs"):
        if not isinstance(item, PreferenceInput):
            raise RecommendationContractValidationError(
                "each preference_inputs entry must be a PreferenceInput"
            )
        key = (item.target, item.provenance)
        if key in seen:
            raise RecommendationContractValidationError(
                f"duplicate preference input for target {item.target!r}"
            )
        seen.add(key)
        inputs.append(item)
    return tuple(inputs)


def _coerce_components(value: object) -> tuple[ScoreComponent, ...]:
    components: list[ScoreComponent] = []
    seen: set[str] = set()
    for item in _iterable(value, "components"):
        if not isinstance(item, ScoreComponent):
            raise RecommendationContractValidationError(
                "each components entry must be a ScoreComponent"
            )
        if item.name in seen:
            raise RecommendationContractValidationError(
                f"duplicate score component {item.name!r}"
            )
        seen.add(item.name)
        components.append(item)
    return tuple(components)


def _coerce_items(value: object) -> tuple[RecommendationItem, ...]:
    items: list[RecommendationItem] = []
    seen: set[str] = set()
    for item in _iterable(value, "items"):
        if not isinstance(item, RecommendationItem):
            raise RecommendationContractValidationError(
                "each items entry must be a RecommendationItem"
            )
        if item.candidate.candidate_id in seen:
            raise RecommendationContractValidationError(
                f"duplicate candidate {item.candidate.candidate_id!r} in result"
            )
        seen.add(item.candidate.candidate_id)
        items.append(item)
    return tuple(items)
