"""Feedback observation contract (P08.1): what happened, with no claim about what it means.

This module is the *contract* layer of the feedback-learning loop. It defines the stable, typed
record of one user feedback event -- an explicit statement or an implicit behavior observed in
response to the recommendation system -- and nothing more. It is a pure domain layer:
deterministic, side-effect free, and independent of SQLite rows, external source payloads, the
system clock, and global state. It never interprets an observation, never computes a preference
change, never reads or writes a database, and never mutates the P06 preference model or the P07
recommendation history.

The three-layer boundary
------------------------

The P08 phase separates three things and this slice implements only the first:

``OBSERVATION``
    What the user actually did or explicitly said. This is :class:`FeedbackObservation`: an
    immutable, frozen record answering "something happened -- what exactly was observed, where did
    it come from, what object or recommendation did it concern, and what aspect is the feedback
    attributed to?"

``INTERPRETATION``
    What that observation is believed to mean (for example "a skip is a negative signal").
    **Not implemented in this slice.** The structural marker of this boundary is
    :class:`FeedbackDirection`: every *implicit* kind carries ``NONE`` -- no directional claim --
    so no observation of a behavior ever smuggles in a meaning. Interpretation of implicit
    feedback is a later slice's job and must attach to the observation, never rewrite it.

``LEARNING EFFECT``
    How P06 preference state is eventually changed. **Not implemented in this slice.** No type in
    this module writes to a preference head, revision, or conclusion, and none may be added here
    later without changing this contract's boundary.

Identity
--------

``feedback_id`` uses the ``fbk_`` namespace: opaque random UUIDv4 identities outside every
existing namespace (canonical ``trk_`` / ``art_`` / ``alb_`` / ``pl_`` / ``pm_``, operational
``int_`` / ``att_`` / ``prb_`` / ``rec_``, and recommendation ``rcm_`` / ``cnd_``), so a feedback
ID can never be confused with any of them and is never derived from its target, source, or
recommendation. ``FEEDBACK_CONTRACT_VERSION`` scopes the serialization and identity semantics;
bump it on any material change to those semantics. :func:`assemble_feedback_observation` is the
single documented assembly boundary and stamps the current version.

What the feedback is about
--------------------------

Every observation must name at least one of two orthogonal references:

``target``
    The entity the user acted on or spoke about, as a
    :class:`~music_agent.preference_attribution.PreferenceTargetReference` (``TRACK`` / ``ARTIST``
    / ``ALBUM`` / ``GENRE``), reusing the P06 target identity rules unchanged.

``recommendation``
    The P07 recommendation item that surfaced the feedback, as a
    :class:`FeedbackRecommendationReference` carrying the frozen P07 identity pair -- ``run_id``
    (``rcm_``) and ``candidate_id`` (``cnd_``) -- validated by the P07.1 validators. A feedback
    event therefore preserves the exact recommendation identity it refers to; whether the
    referenced item's own target agrees with ``target`` is not checked here, because that
    cross-check requires the persisted run history and belongs to a later slice.

``attribution``
    What *aspect* the feedback is attributed to -- or explicitly disclaimed from -- via
    :class:`FeedbackAttribution`. It is an optional, orthogonal axis: ``target`` says what the
    feedback is about, ``attribution`` says why the user believes it happened
    (``ATTRIBUTED``: "because of X"; ``EXCLUDED``: "not because of X"). An
    ``ATTRIBUTION_CORRECTION`` observation (for example "not because of the artist") requires an
    attribution and carries no directional claim of its own.

Vocabulary, explicitness, and direction
---------------------------------------

:class:`FeedbackKind` is the frozen vocabulary of this contract:

- explicit statements: ``LIKED``, ``DISLIKED``, ``CORRECTED`` (user correction),
  ``DIRECTION_GOOD`` ("this direction is good"), ``ATTRIBUTION_CORRECTION``.
- implicit behaviors: ``FAVORITED``, ``SKIPPED``, ``REPLAYED``, ``COMPLETED``,
  ``PLAYED`` (a single playback occurrence; "repeated / frequent recent playback" is an
  aggregation of these, computed by a later slice, never stored on one observation).

Explicitness and direction are *fixed by the kind* through frozen mappings and exposed as
properties, never as settable fields, so a behavior can never be re-labelled explicit at
construction time:

- ``LIKED``, ``DIRECTION_GOOD`` -> explicit, ``POSITIVE``.
- ``DISLIKED`` -> explicit, ``NEGATIVE``.
- ``CORRECTED``, ``ATTRIBUTION_CORRECTION`` -> explicit, ``NONE`` (they correct attribution or
  fact, not valence).
- every implicit kind -> implicit, ``NONE`` (no directional claim; interpretation deferred).

A feedback event is not a preference signal: P06's ``favorited`` / ``disliked`` / ``rating``
describe source-of-truth library state, while a :class:`FeedbackObservation` records a user
behavior or statement in the feedback loop. The same underlying user action may surface in both
layers; how an observation eventually feeds a P06 signal identity is the learning-effect slice's
decision and is deliberately absent here.

Provenance and time
-------------------

``source`` is a :class:`FeedbackSourceReference` -- the observing system and the path within it
(for example the UI surface or the playback telemetry pipeline), both opaque machine-oriented
non-empty strings, never free prose.

``observed_at`` is the timezone-aware instant at which the system observed the feedback. It is
injected by the caller; this module never reads the system clock, so an observation is
deterministic for the same inputs. ``event_at`` is the optional timezone-aware instant of the
underlying user event when the source exposes it (for example a playback timestamp). Both
serialize as ISO-8601 strings and decode to timezone-aware datetimes only.

Duplicate observations
----------------------

Two observations describe the same user event when their :func:`feedback_duplicate_key` values
are equal. When the source supplies ``source_event_id`` (its own event identity), the key is
``(source_system, source_event_id)`` -- the strongest deduplication anchor. Without it the key is
the full observation fields, which conservatively treats distinct sources, paths, targets, and
instants as distinct events. What a *consumer* does with same-key observations -- dropping the
duplicate, counting it as confirmation -- is policy belonging to the future feedback-history and
learning slices; this contract only defines the identity. Note the identity excludes nothing a
consumer needs: two plays of the same track at different ``observed_at`` values are distinct
events by construction, while two reports of the same skip are recognizably one.

Immutable observation history vs derived learning state
-------------------------------------------------------

:class:`FeedbackObservation` is an immutable frozen record: it is the history. Everything derived
-- direction of implicit feedback, weights, confidence, decay, preference-head updates, playback
frequency -- is learning state computed elsewhere and is never stored on an observation. A
durable SQLite history table for observations is intentionally not introduced in this slice
(it would follow the P07.5 append-only pattern); :func:`encode_feedback_observation` /
:func:`decode_feedback_observation` provide the canonical JSON interchange such a table will build
on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.recommendation_contract import validate_candidate_id, validate_run_id


class FeedbackContractError(ValueError):
    code = "feedback_contract_error"


class FeedbackContractValidationError(FeedbackContractError):
    code = "validation_error"


# The current feedback contract. It scopes the serialization and identity semantics under which a
# feedback observation is produced. Bump it on any material change to those semantics; it does not
# version interpretation rules, learning-effect policy, or the external source.
FEEDBACK_CONTRACT_VERSION = 1

_FEEDBACK_ID_PREFIX = "fbk_"


class FeedbackKind(StrEnum):
    """The frozen vocabulary of feedback events this contract can represent."""

    # explicit statements
    LIKED = "liked"
    DISLIKED = "disliked"
    CORRECTED = "corrected"
    DIRECTION_GOOD = "direction_good"
    ATTRIBUTION_CORRECTION = "attribution_correction"
    # implicit behaviors
    FAVORITED = "favorited"
    SKIPPED = "skipped"
    REPLAYED = "replayed"
    COMPLETED = "completed"
    PLAYED = "played"


class FeedbackExplicitness(StrEnum):
    """Whether the feedback was stated by the user or only observed as a behavior."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class FeedbackDirection(StrEnum):
    """The directional claim an observation carries.

    ``POSITIVE`` and ``NEGATIVE`` only ever come from explicit statements. ``NONE`` marks an
    observation that makes no directional claim: implicit behaviors (interpretation is a later
    slice's job) and explicit corrections (which correct attribution or fact, not valence).
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


class AttributionRelation(StrEnum):
    """How an attribution relates the feedback to an aspect.

    ``ATTRIBUTED`` claims the feedback happened *because of* the aspect (for example "because of
    the artist"). ``EXCLUDED`` explicitly disclaims the aspect as the reason (for example "not
    because of the artist").
    """

    ATTRIBUTED = "attributed"
    EXCLUDED = "excluded"


# Explicitness and direction are fixed properties of the vocabulary, never per-event judgments:
# a kind carries the same explicitness and directional claim everywhere it appears. Implicit
# kinds deliberately carry ``NONE`` so an observation can never smuggle in an interpretation.
_EXPLICITNESS_BY_KIND: dict[FeedbackKind, FeedbackExplicitness] = {
    FeedbackKind.LIKED: FeedbackExplicitness.EXPLICIT,
    FeedbackKind.DISLIKED: FeedbackExplicitness.EXPLICIT,
    FeedbackKind.CORRECTED: FeedbackExplicitness.EXPLICIT,
    FeedbackKind.DIRECTION_GOOD: FeedbackExplicitness.EXPLICIT,
    FeedbackKind.ATTRIBUTION_CORRECTION: FeedbackExplicitness.EXPLICIT,
    FeedbackKind.FAVORITED: FeedbackExplicitness.IMPLICIT,
    FeedbackKind.SKIPPED: FeedbackExplicitness.IMPLICIT,
    FeedbackKind.REPLAYED: FeedbackExplicitness.IMPLICIT,
    FeedbackKind.COMPLETED: FeedbackExplicitness.IMPLICIT,
    FeedbackKind.PLAYED: FeedbackExplicitness.IMPLICIT,
}

_DIRECTION_BY_KIND: dict[FeedbackKind, FeedbackDirection] = {
    FeedbackKind.LIKED: FeedbackDirection.POSITIVE,
    FeedbackKind.DIRECTION_GOOD: FeedbackDirection.POSITIVE,
    FeedbackKind.DISLIKED: FeedbackDirection.NEGATIVE,
    FeedbackKind.CORRECTED: FeedbackDirection.NONE,
    FeedbackKind.ATTRIBUTION_CORRECTION: FeedbackDirection.NONE,
    FeedbackKind.FAVORITED: FeedbackDirection.NONE,
    FeedbackKind.SKIPPED: FeedbackDirection.NONE,
    FeedbackKind.REPLAYED: FeedbackDirection.NONE,
    FeedbackKind.COMPLETED: FeedbackDirection.NONE,
    FeedbackKind.PLAYED: FeedbackDirection.NONE,
}


def generate_feedback_id() -> str:
    """Generate a stable feedback-observation identity.

    The ``fbk_`` prefix is outside every canonical, operational, and recommendation namespace, so
    a feedback ID can never be confused with a canonical, intent, attempt, probe, recovery, run,
    or candidate ID, and it is never derived from the observation's target, source, or
    recommendation.
    """
    return f"{_FEEDBACK_ID_PREFIX}{uuid4()}"


def validate_feedback_id(feedback_id: str) -> None:
    if not isinstance(feedback_id, str) or not feedback_id.startswith(_FEEDBACK_ID_PREFIX):
        raise FeedbackContractValidationError(
            f"feedback_id must use the {_FEEDBACK_ID_PREFIX} namespace"
        )
    _require_uuid_suffix(feedback_id[len(_FEEDBACK_ID_PREFIX) :], label="feedback_id")


@dataclass(frozen=True, slots=True)
class FeedbackSourceReference:
    """Structured provenance of where a feedback observation came from.

    ``source_system`` names the observing system and ``source_path`` names the observing path
    within it (for example the UI surface or the playback telemetry pipeline). Both are opaque
    machine-oriented non-empty strings, never free prose; the path vocabulary is owned by the
    emitting subsystem.
    """

    source_system: str
    source_path: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.source_system, "source_system")
        _require_non_empty_string(self.source_path, "source_path")


@dataclass(frozen=True, slots=True)
class FeedbackRecommendationReference:
    """The P07 recommendation item that surfaced the feedback.

    ``run_id`` (``rcm_``) and ``candidate_id`` (``cnd_``) are the frozen P07.1 identity pair and
    are validated by the P07.1 validators unchanged, so a feedback event preserves the exact
    recommendation identity it refers to. Tuple position (rank) is deliberately not carried: rank
    is derivable from the persisted run and is not part of an item's identity.
    """

    run_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_candidate_id(self.candidate_id)


@dataclass(frozen=True, slots=True)
class FeedbackAttribution:
    """What aspect the feedback is attributed to, or explicitly disclaimed from.

    ``aspect`` is a :class:`~music_agent.preference_attribution.PreferenceTargetReference`
    (``TRACK`` / ``ARTIST`` / ``ALBUM`` / ``GENRE``), reusing the P06 target identity rules.
    ``relation`` is ``ATTRIBUTED`` ("because of X") or ``EXCLUDED`` ("not because of X").
    """

    aspect: PreferenceTargetReference
    relation: AttributionRelation

    def __post_init__(self) -> None:
        if not isinstance(self.aspect, PreferenceTargetReference):
            raise FeedbackContractValidationError(
                "aspect must be a PreferenceTargetReference"
            )
        if not isinstance(self.relation, AttributionRelation):
            raise FeedbackContractValidationError(
                "relation must be an AttributionRelation"
            )


@dataclass(frozen=True, slots=True)
class FeedbackObservation:
    """One immutable record of a user feedback event.

    ``feedback_id`` is a stable ``fbk_`` identity. ``kind`` selects the feedback vocabulary, and
    the ``explicitness`` / ``direction`` properties are fixed by that kind. ``source`` records
    where the observation came from and ``observed_at`` is the injected timezone-aware
    observation instant (never the system clock). ``target`` names the entity the feedback is
    about and ``recommendation`` names the P07 recommendation item that surfaced it -- at least
    one of the two is required. ``attribution`` optionally names the aspect the feedback is
    attributed to (or disclaimed from); an ``ATTRIBUTION_CORRECTION`` observation requires one.
    ``event_at`` is the optional timezone-aware instant of the underlying user event, and
    ``source_event_id`` is the optional source-scoped event identity used as the duplicate key.
    ``contract_version`` scopes the serialization semantics and is stamped by
    :func:`assemble_feedback_observation`.

    The record is an *observation only*: it carries no interpretation, no directional claim for
    implicit behavior, and no learning effect on the P06 preference model.
    """

    feedback_id: str
    kind: FeedbackKind
    source: FeedbackSourceReference
    observed_at: datetime
    target: PreferenceTargetReference | None = None
    recommendation: FeedbackRecommendationReference | None = None
    attribution: FeedbackAttribution | None = None
    event_at: datetime | None = None
    source_event_id: str | None = None
    # Defaults to the current contract constant so direct construction and
    # assemble_feedback_observation can never stamp different versions; the assembly boundary
    # remains the documented production path.
    contract_version: int = FEEDBACK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        validate_feedback_id(self.feedback_id)
        if not isinstance(self.kind, FeedbackKind):
            raise FeedbackContractValidationError("kind must be a FeedbackKind")
        if not isinstance(self.source, FeedbackSourceReference):
            raise FeedbackContractValidationError(
                "source must be a FeedbackSourceReference"
            )
        _require_aware_datetime(self.observed_at, label="observed_at")

        if self.target is None and self.recommendation is None:
            raise FeedbackContractValidationError(
                "a feedback observation requires a target or a recommendation"
            )
        if self.target is not None and not isinstance(self.target, PreferenceTargetReference):
            raise FeedbackContractValidationError(
                "target must be a PreferenceTargetReference or None"
            )
        if self.recommendation is not None and not isinstance(
            self.recommendation, FeedbackRecommendationReference
        ):
            raise FeedbackContractValidationError(
                "recommendation must be a FeedbackRecommendationReference or None"
            )
        if self.attribution is not None and not isinstance(self.attribution, FeedbackAttribution):
            raise FeedbackContractValidationError(
                "attribution must be a FeedbackAttribution or None"
            )
        if self.kind is FeedbackKind.ATTRIBUTION_CORRECTION and self.attribution is None:
            raise FeedbackContractValidationError(
                "an ATTRIBUTION_CORRECTION observation requires an attribution"
            )

        if self.event_at is not None:
            _require_aware_datetime(self.event_at, label="event_at")
        if self.source_event_id is not None:
            _require_non_empty_string(self.source_event_id, "source_event_id")
        _require_positive_int(self.contract_version, label="contract_version")

    @property
    def explicitness(self) -> FeedbackExplicitness:
        """The explicitness fixed by the observation's kind.

        A kind carries the same explicitness everywhere it appears; it is never a per-event
        judgment and never a constructor argument.
        """
        return _EXPLICITNESS_BY_KIND[self.kind]

    @property
    def direction(self) -> FeedbackDirection:
        """The directional claim fixed by the observation's kind.

        Only explicit statements carry ``POSITIVE`` or ``NEGATIVE``; implicit behaviors and
        corrections carry ``NONE``, so an observation never smuggles in an interpretation. The
        meaning of implicit feedback is a later slice's job.
        """
        return _DIRECTION_BY_KIND[self.kind]


def assemble_feedback_observation(
    *,
    feedback_id: str,
    kind: FeedbackKind,
    source: FeedbackSourceReference,
    observed_at: datetime,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
    event_at: datetime | None = None,
    source_event_id: str | None = None,
) -> FeedbackObservation:
    """Assemble and validate a feedback observation.

    ``feedback_id``, ``source``, and ``observed_at`` are injected by the caller -- this function
    reads neither the clock nor randomness, so the observation is deterministic for the same
    inputs. The current ``FEEDBACK_CONTRACT_VERSION`` is stamped automatically. This is the
    single documented boundary for producing an observation.
    """
    return FeedbackObservation(
        feedback_id=feedback_id,
        kind=kind,
        source=source,
        observed_at=observed_at,
        target=target,
        recommendation=recommendation,
        attribution=attribution,
        event_at=event_at,
        source_event_id=source_event_id,
        contract_version=FEEDBACK_CONTRACT_VERSION,
    )


def feedback_duplicate_key(observation: FeedbackObservation) -> tuple:
    """Return the hashable key identifying the user event an observation describes.

    Two observations with equal keys describe the same event. When ``source_event_id`` is
    present the key is ``(source_system, source_event_id)`` -- the source's own event identity,
    the strongest deduplication anchor. Otherwise the key is the full observation fields, which
    conservatively treats distinct sources, paths, kinds, targets, recommendations, attributions,
    and instants as distinct events. What a consumer *does* with same-key observations --
    dropping the duplicate, counting it as confirmation -- is policy belonging to the future
    feedback-history and learning slices; this function only defines the identity.
    """
    if not isinstance(observation, FeedbackObservation):
        raise FeedbackContractValidationError(
            "observation must be a FeedbackObservation"
        )
    if observation.source_event_id is not None:
        return (observation.source.source_system, observation.source_event_id)
    return (
        observation.source.source_system,
        observation.source.source_path,
        observation.kind,
        observation.target,
        observation.recommendation,
        observation.attribution,
        observation.observed_at,
    )


# --- canonical serialization ---------------------------------------------


def encode_feedback_observation(observation: FeedbackObservation) -> str:
    """Encode a feedback observation to its canonical JSON text form.

    The encoding is deterministic (sorted keys) and round-trips every field; derived properties
    (``explicitness`` / ``direction``) are not encoded because they are fixed by ``kind`` by
    construction. It is the interchange form a future feedback-history store builds on; it does
    not persist anything itself.
    """
    if not isinstance(observation, FeedbackObservation):
        raise FeedbackContractValidationError(
            "observation must be a FeedbackObservation"
        )
    return json.dumps(
        _observation_to_dict(observation),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def decode_feedback_observation(text: str) -> FeedbackObservation:
    """Decode a canonical JSON feedback observation back to a :class:`FeedbackObservation`.

    Fails closed on a non-string payload, unparseable JSON, or a decoded structure outside the
    contract, rather than coercing an unknown value.
    """
    if not isinstance(text, str):
        raise FeedbackContractValidationError("encoded observation must be a string")
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise FeedbackContractValidationError(
            f"feedback observation is not valid JSON: {error}"
        ) from error
    return _observation_from_dict(data)


def _observation_to_dict(observation: FeedbackObservation) -> dict[str, Any]:
    return {
        "attribution": (
            None
            if observation.attribution is None
            else {
                "aspect": _target_to_dict(observation.attribution.aspect),
                "relation": observation.attribution.relation.value,
            }
        ),
        "contract_version": observation.contract_version,
        "event_at": None if observation.event_at is None else observation.event_at.isoformat(),
        "feedback_id": observation.feedback_id,
        "kind": observation.kind.value,
        "observed_at": observation.observed_at.isoformat(),
        "recommendation": (
            None
            if observation.recommendation is None
            else {
                "candidate_id": observation.recommendation.candidate_id,
                "run_id": observation.recommendation.run_id,
            }
        ),
        "source": {
            "source_path": observation.source.source_path,
            "source_system": observation.source.source_system,
        },
        "source_event_id": observation.source_event_id,
        "target": None if observation.target is None else _target_to_dict(observation.target),
    }


def _target_to_dict(target: PreferenceTargetReference) -> dict[str, str]:
    return {"kind": target.kind.value, "target_id": target.target_id}


def _target_from_dict(data: object) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind(data["kind"]), data["target_id"])


def _observation_from_dict(data: object) -> FeedbackObservation:
    try:
        return FeedbackObservation(
            feedback_id=data["feedback_id"],
            kind=FeedbackKind(data["kind"]),
            source=FeedbackSourceReference(
                data["source"]["source_system"], data["source"]["source_path"]
            ),
            observed_at=_decode_datetime(data["observed_at"]),
            target=None if data["target"] is None else _target_from_dict(data["target"]),
            recommendation=(
                None
                if data["recommendation"] is None
                else FeedbackRecommendationReference(
                    data["recommendation"]["run_id"], data["recommendation"]["candidate_id"]
                )
            ),
            attribution=(
                None
                if data["attribution"] is None
                else FeedbackAttribution(
                    _target_from_dict(data["attribution"]["aspect"]),
                    AttributionRelation(data["attribution"]["relation"]),
                )
            ),
            event_at=None if data["event_at"] is None else _decode_datetime(data["event_at"]),
            source_event_id=data["source_event_id"],
            contract_version=data["contract_version"],
        )
    except FeedbackContractValidationError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise FeedbackContractValidationError(
            f"malformed feedback observation: {error}"
        ) from error


def _decode_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise FeedbackContractValidationError("timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FeedbackContractValidationError("timestamp must be timezone-aware")
    return parsed


# --- validation helpers ---------------------------------------------------


def _require_uuid_suffix(suffix: object, *, label: str) -> None:
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise FeedbackContractValidationError(
            f"{label} suffix must be a canonical UUID"
        ) from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise FeedbackContractValidationError(f"{label} suffix must be a canonical UUID")


def _require_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise FeedbackContractValidationError(f"{field} must be a non-empty string")


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeedbackContractValidationError(
            f"{label} must be an integer, not {type(value).__name__}"
        )
    if value < 1:
        raise FeedbackContractValidationError(f"{label} must be >= 1")
    return value


def _require_aware_datetime(value: object, *, label: str) -> datetime:
    """Return ``value`` as a timezone-aware ``datetime``, failing closed otherwise."""
    if not isinstance(value, datetime):
        raise FeedbackContractValidationError(
            f"{label} must be a datetime, not {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise FeedbackContractValidationError(f"{label} must be timezone-aware")
    return value
