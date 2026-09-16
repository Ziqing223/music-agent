"""Durable preference persistence domain objects (P06 S10).

This module is the tenth slice of preference modeling. It defines the *durable* records that make
up the preference source-of-truth -- a mutable :class:`SignalHead` per stable
:class:`SignalIdentity`, and an append-only, immutable :class:`EvidenceRevision` history -- and
nothing more. It is a pure domain layer: it validates and round-trips records but performs no
SQLite I/O, no clock reads, and no external-source access, and it never re-derives preference
direction, strength, confidence, recency, or inference.

Frozen boundary
---------------

Only already-resolved *semantic state* is durable. The persistence layer saves the observed value
of a single preference signal (``favorited`` / ``disliked`` / ``rating``) against a stable target
identity; it does not re-interpret that value into a preference direction (S1 / S3 own inference
semantics) and does not materialize any query-time derivation. Recency, evidence influence, recent
preference, current-preference projection, confidence, familiarity, and temporal-decay results are
all re-computed from the durable head + revision history and are never persisted.

Three-state boundary
--------------------

``MISSING``, ``NULL``, and ``VALUE`` remain three distinct states and are never folded together.
Only a ``VALUE`` observation produces (or changes) a semantic value; ``MISSING`` and ``NULL`` are
observational states that update head metadata only and never create an evidence revision.

Revision rules
--------------

- First semantic sighting (no prior ``VALUE`` for the identity) -> one ``BASELINE`` revision at
  sequence ``1``.
- A later ``VALUE`` that differs from the current semantic value -> one ``TRANSITION`` revision at
  the next sequence.
- A later ``VALUE`` equal to the current semantic value -> confirmation only, no new revision.
- ``MISSING`` / ``NULL`` -> observation metadata only, never an evidence revision.

A revision's semantic value is a JSON-safe scalar (``bool``, ``int``, or ``str``). ``float``,
``None``, and structured values are rejected rather than coerced: ``None`` is the ``NULL`` state,
and ``float`` carries a precision ambiguity a scalar semantic state should never admit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from music_agent.preference_attribution import PreferenceTargetReference
from music_agent.source_observation import ObservationState


class PreferencePersistenceError(ValueError):
    code = "preference_persistence_error"


class PreferencePersistenceValidationError(PreferencePersistenceError):
    code = "validation_error"


class RevisionKind(StrEnum):
    """The kind of an evidence revision: the first sighting, or a later value change."""

    BASELINE = "baseline"
    TRANSITION = "transition"


# The current preference-evidence contract. It scopes the *persistence* semantics under which a
# head and its revisions are produced (the semantic-value encoding, the revision rules, and the
# three-state boundary). Bump it on any material change to those semantics; it does not version
# inference or the external source, and it does not make any preference "currently valid".
PREFERENCE_EVIDENCE_CONTRACT_VERSION = 1

# The default provenance recorded on a revision built from a direct source observation. The
# repository accepts an explicit provenance; this constant is the minimal honest label for the
# direct-observation path. Natural-language provenance is not implemented in this slice and is a
# deferred integration requirement.
DIRECT_OBSERVATION_PROVENANCE = "direct_source_observation"


@dataclass(frozen=True, slots=True)
class SignalIdentity:
    """A stable, unique identity for one preference signal head.

    ``target`` names the referenced preference target and carries its kind-specific identity
    validation (``TRACK`` / ``ARTIST`` / ``ALBUM`` are canonical IDs; ``GENRE`` is an independent
    non-empty string key). ``source_system`` names the observing source and ``signal_path`` names
    the signal within that source (for example ``favorited`` or ``rating``). Identity safety is
    not relaxed: canonical target kinds are validated exactly as
    :class:`~music_agent.preference_attribution.PreferenceTargetReference` requires, and the two
    free-form string fields are non-empty.
    """

    target: PreferenceTargetReference
    source_system: str
    signal_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, PreferenceTargetReference):
            raise PreferencePersistenceValidationError(
                "target must be a PreferenceTargetReference"
            )
        _require_non_empty_string(self.source_system, "source_system")
        _require_non_empty_string(self.signal_path, "signal_path")


@dataclass(frozen=True, slots=True)
class SignalHead:
    """The mutable head for one :class:`SignalIdentity`.

    ``current_semantic_value`` is the most recent ``VALUE`` payload (``None`` if no ``VALUE`` has
    ever been observed for this identity). ``last_observed_state`` is the three-state result of the
    most recent observation. ``first_observed_at`` / ``last_observed_at`` are the first and most
    recent observation instants, ``current_revision_sequence`` is the highest revision sequence
    (``0`` before the first ``VALUE``), and ``evidence_contract_version`` scopes the persistence
    semantics under which the head was written.
    """

    identity: SignalIdentity
    current_semantic_value: Any | None
    last_observed_state: ObservationState
    first_observed_at: str
    last_observed_at: str
    current_revision_sequence: int
    evidence_contract_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SignalIdentity):
            raise PreferencePersistenceValidationError("identity must be a SignalIdentity")
        if self.current_semantic_value is not None:
            _require_semantic_value(self.current_semantic_value)
        if self.last_observed_state not in (
            ObservationState.MISSING,
            ObservationState.NULL,
            ObservationState.VALUE,
        ):
            raise PreferencePersistenceValidationError(
                "last_observed_state must be MISSING, NULL, or VALUE"
            )
        if self.last_observed_state is ObservationState.VALUE and self.current_semantic_value is None:
            raise PreferencePersistenceValidationError(
                "a VALUE head requires a current_semantic_value"
            )
        _require_non_empty_string(self.first_observed_at, "first_observed_at")
        _require_non_empty_string(self.last_observed_at, "last_observed_at")
        if _is_bool(self.current_revision_sequence) or not isinstance(
            self.current_revision_sequence, int
        ) or self.current_revision_sequence < 0:
            raise PreferencePersistenceValidationError(
                "current_revision_sequence must be a non-negative int"
            )
        _require_contract_version(self.evidence_contract_version)


@dataclass(frozen=True, slots=True)
class EvidenceRevision:
    """An immutable, append-only evidence revision for one :class:`SignalIdentity`.

    ``revision_sequence`` is the head's monotonic sequence counter; ``revision_kind`` is
    ``BASELINE`` for the first sighting (sequence ``1``) and ``TRANSITION`` for a later value
    change (sequence ``>= 2``). ``semantic_value`` is the observed ``VALUE`` payload,
    ``observed_at`` is when it was observed, and ``event_at`` is the optional time of the
    underlying event (``None`` when the source does not expose it). ``provenance`` records the
    observation provenance and ``evidence_contract_version`` scopes the persistence semantics.
    """

    identity: SignalIdentity
    revision_sequence: int
    revision_kind: RevisionKind
    semantic_value: Any
    observed_at: str
    event_at: str | None
    provenance: str
    evidence_contract_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SignalIdentity):
            raise PreferencePersistenceValidationError("identity must be a SignalIdentity")
        if _is_bool(self.revision_sequence) or not isinstance(self.revision_sequence, int) or self.revision_sequence < 1:
            raise PreferencePersistenceValidationError(
                "revision_sequence must be a positive int"
            )
        if not isinstance(self.revision_kind, RevisionKind):
            raise PreferencePersistenceValidationError("revision_kind must be a RevisionKind")
        _require_semantic_value(self.semantic_value)
        _require_non_empty_string(self.observed_at, "observed_at")
        if self.event_at is not None:
            _require_non_empty_string(self.event_at, "event_at")
        _require_non_empty_string(self.provenance, "provenance")
        _require_contract_version(self.evidence_contract_version)
        if self.revision_kind is RevisionKind.BASELINE and self.revision_sequence != 1:
            raise PreferencePersistenceValidationError(
                "a BASELINE revision must be the first revision (sequence 1)"
            )
        if self.revision_kind is RevisionKind.TRANSITION and self.revision_sequence < 2:
            raise PreferencePersistenceValidationError(
                "a TRANSITION revision requires sequence >= 2"
            )


@dataclass(frozen=True, slots=True)
class RecordObservationOutcome:
    """The result of recording one observation against a signal head.

    ``head`` is the head as it stands after the observation. ``revision`` is the newly appended
    evidence revision when the observation created one (a first sighting ``BASELINE`` or a value
    change ``TRANSITION``), and ``None`` when the observation was a ``MISSING`` / ``NULL``
    metadata-only update or a same-value confirmation.
    """

    head: SignalHead
    revision: EvidenceRevision | None

    def __post_init__(self) -> None:
        if not isinstance(self.head, SignalHead):
            raise PreferencePersistenceValidationError("head must be a SignalHead")
        if self.revision is not None and not isinstance(self.revision, EvidenceRevision):
            raise PreferencePersistenceValidationError(
                "revision must be an EvidenceRevision or None"
            )


def encode_semantic_value(value: object) -> str:
    """Encode a semantic value to its canonical JSON text form.

    The encoded text is the canonical comparison form: ``True`` and ``1`` encode to distinct
    strings (``true`` vs ``1``), so a type change is correctly observed as a value change. Only
    ``bool``, ``int``, and ``str`` are accepted; ``float``, ``None``, and structured values fail
    closed rather than being coerced.
    """
    _require_semantic_value(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def decode_semantic_value(encoded: object) -> Any:
    """Decode a canonical JSON semantic-value text back to its Python scalar.

    Fails closed on a non-string payload, unparseable JSON, or a decoded value outside the
    supported scalar set (``bool`` / ``int`` / ``str``).
    """
    if not isinstance(encoded, str):
        raise PreferencePersistenceValidationError(
            "encoded semantic value must be a string"
        )
    try:
        value = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise PreferencePersistenceValidationError(
            f"semantic value is not valid JSON: {error}"
        ) from error
    _require_semantic_value(value)
    return value


def _require_semantic_value(value: object) -> None:
    if _is_bool(value) or isinstance(value, int) or isinstance(value, str):
        return
    raise PreferencePersistenceValidationError(
        "semantic value must be a bool, int, or str "
        f"(got {type(value).__name__})"
    )


def _require_contract_version(version: object) -> None:
    if _is_bool(version) or not isinstance(version, int) or version < 1:
        raise PreferencePersistenceValidationError(
            "evidence_contract_version must be a positive int"
        )


def _require_non_empty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise PreferencePersistenceValidationError(f"{field} must be a non-empty string")


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)
