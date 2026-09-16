"""Apple Music write capability matrix and pending-intent contract.

This module is a pure domain layer. It defines:

1. the Apple Music write capability matrix, which keeps five independent facts distinct:

   - ``domain_permission``: ``ALLOWED`` / ``DISALLOWED`` / ``UNSPECIFIED``. ``ALLOWED`` means
     the sealed P02 write-boundary table authorizes the Agent to request the write;
     ``DISALLOWED`` means it explicitly forbids it; ``UNSPECIFIED`` means the sealed contract
     does not say. ``UNSPECIFIED`` is neither permission nor prohibition.
   - ``capability_verified``: the write path was actually proven by real P01 verification;
   - ``adapter_implemented``: the current production adapter actually implements the write;
   - ``readback_verified``: P01 actually ran the write-then-readback loop for this write;
   - ``readback_implemented``: the current production readback orchestration implements it.

   These are independent and must not be conflated. A write can be historically verified but
   not currently implemented (the playlist writes), and a verified write can still have
   ``UNSPECIFIED`` domain permission (playlist create / delete).

2. ``PendingIntent``, an operational object with its own ``int_`` identity that is NOT a
   canonical entity, is NOT a canonical ID, and is NOT an external identity. It records the
   requested operation, an explicit set of identity ``requirements`` (each with a ``role``, a
   canonical ID, and an Apple Music external identity), the requested value, and a lifecycle
   state. A scalar field write has one ``TARGET`` requirement; ``add_playlist_membership`` has
   one ``PLAYLIST`` and one ``TRACK`` requirement, so the same contract expresses both.

3. a pure transition / readback evaluation API that fails closed.

Nothing here persists intents, calls Music.app or ``osascript``, mutates the canonical model,
or implements a real write adapter. Canonical state is never updated from a successful command
alone; only readback may confirm an Apple Music-owned canonical change.

P01 write evidence
------------------

The reachable P01 evidence (``PHASE_01_SUMMARY`` / ``P01-C01`` under the AI System vault)
verified a Playlist write closed loop: create playlist, add existing Library tracks by
persistent ID, read the playlist contents back, confirm in the Music.app UI, then delete the
playlist. So ``CREATE_PLAYLIST``, ``ADD_PLAYLIST_MEMBERSHIP``, and ``DELETE_PLAYLIST`` are
``capability_verified=True``. The playlist *contents* readback is verified; the deletion is
recorded only as "no test Playlist remains" with no explicit post-delete readback, so
``DELETE_PLAYLIST`` readback stays unverified (``readback_verified=False``).

P01 only *read* favorited / disliked / rating; it never wrote them, so their write
``capability_verified`` remains ``False`` regardless of ``domain_permission``.

The current production adapter implements two writes: ``add_playlist_membership``
(``music_agent.apple_music_write``) and ``set_favorited`` (``apple_music_favorited_write``), both
``adapter_implemented=True``. ``set_favorited`` also has a desired-state readback
(``readback_implemented=True``), but it stays ``capability_verified=False``; the membership write's
readback cannot safely confirm a new membership occurrence from member enumeration alone, so its
``readback_implemented`` stays ``False``. No write is execution-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping
from uuid import UUID, uuid4

from music_agent.identity import (
    EntityType,
    ExternalIdentityKey,
    validate_canonical_id,
)
from music_agent.source_observation import ObservedValue, ObservationState


_INTENT_ID_PREFIX = "int_"
_APPLE_MUSIC_SOURCE = "apple_music"


class WriteIntentError(ValueError):
    code = "write_intent_error"


class UnsupportedWriteOperation(WriteIntentError):
    code = "unsupported_write_operation"


class WriteNotDomainWritable(WriteIntentError):
    code = "not_domain_writable"


class WriteDomainPermissionUnspecified(WriteIntentError):
    code = "domain_permission_unspecified"


class WriteIntentValidationError(WriteIntentError):
    code = "validation_error"


class WriteTransitionError(WriteIntentError):
    code = "invalid_transition"


class DomainPermission(StrEnum):
    """Whether the sealed domain contract permits a write.

    ``UNSPECIFIED`` is distinct from both ``ALLOWED`` and ``DISALLOWED``: the sealed contract
    simply does not state a permission. It must not be inferred from P01 verification or from
    Apple Music authority, and it never makes a write execution-ready.
    """

    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    UNSPECIFIED = "unspecified"


class WriteOperation(StrEnum):
    """Candidate Apple Music write operations subject to the write-control policy."""

    SET_FAVORITED = "set_favorited"
    SET_DISLIKED = "set_disliked"
    SET_RATING = "set_rating"
    SET_PLAY_COUNT = "set_play_count"
    SET_SKIP_COUNT = "set_skip_count"
    SET_LAST_PLAYED_AT = "set_last_played_at"
    SET_ADDED_TO_LIBRARY_AT = "set_added_to_library_at"
    SET_ARTIST_IDS = "set_artist_ids"
    SET_ALBUM_ID = "set_album_id"
    CREATE_PLAYLIST = "create_playlist"
    DELETE_PLAYLIST = "delete_playlist"
    ADD_PLAYLIST_MEMBERSHIP = "add_playlist_membership"
    REMOVE_PLAYLIST_MEMBERSHIP = "remove_playlist_membership"
    ADD_LIBRARY_SONG = "add_library_song"


class ReadbackStrategy(StrEnum):
    """How a successful write can be confirmed by reading the source back.

    ``READ_FIELD`` reads back the written canonical field; ``READ_PLAYLIST_CONTENTS``
    enumerates a playlist's members; ``READ_PLAYLIST_ABSENCE`` confirms a playlist no longer
    exists; ``READ_CATALOG_RELATIONSHIP`` confirms a library song's exact catalog-id
    relationship (P11.3). ``UNAVAILABLE`` means there is no readback mechanism.
    """

    READ_FIELD = "read_field"
    READ_PLAYLIST_CONTENTS = "read_playlist_contents"
    READ_PLAYLIST_ABSENCE = "read_playlist_absence"
    READ_CATALOG_RELATIONSHIP = "read_catalog_relationship"
    UNAVAILABLE = "unavailable"


class IntentState(StrEnum):
    """PendingIntent lifecycle.

    ``PENDING`` means the intent is formed but the external write has not been confirmed to
    have run. ``command success`` moves to ``AWAITING_READBACK``; only a matching readback
    moves to ``CONFIRMED``. Command failure and readback disagreement are distinct terminal
    states. ``OUTCOME_UNKNOWN`` records an ambiguous command outcome (a dispatched command whose
    side effect cannot be proven absent); it never re-executes the command, but a matching readback
    may still reconcile it to ``CONFIRMED``.
    """

    PENDING = "pending"
    EXECUTION_FAILED = "execution_failed"
    AWAITING_READBACK = "awaiting_readback"
    CONFIRMED = "confirmed"
    READBACK_MISMATCH = "readback_mismatch"
    OUTCOME_UNKNOWN = "outcome_unknown"


class WriteEvent(StrEnum):
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_UNKNOWN = "command_unknown"
    READBACK_MATCHED = "readback_matched"
    READBACK_MISMATCHED = "readback_mismatched"


class ReadbackDecision(StrEnum):
    MATCHED = "matched"
    MISMATCHED = "mismatched"
    UNAVAILABLE = "unavailable"


class RequirementRole(StrEnum):
    """The semantic slot a requirement fills within a write's argument set.

    A role is explicit and never inferred from array position. ``TARGET`` names the single
    entity a scalar field write targets; ``PLAYLIST`` / ``TRACK`` name the two independent
    entities a relation write (``add_playlist_membership``) needs. A relation intent therefore
    never depends on ``requirements[0] == playlist`` / ``requirements[1] == track``.
    """

    TARGET = "target"
    PLAYLIST = "playlist"
    TRACK = "track"


@dataclass(frozen=True, slots=True)
class WriteCapability:
    """One write operation's permission, verification, implementation, and readback contract.

    ``domain_permission`` is the sealed domain contract's three-state answer (ALLOWED /
    DISALLOWED / UNSPECIFIED). ``capability_verified`` records historical P01 verification (a
    fact that does not change), while ``adapter_implemented`` records whether the current
    production adapter performs the write today. ``readback_verified`` / ``readback_implemented``
    split the same way for the write-then-readback loop.
    """

    operation: WriteOperation
    entity_type: EntityType
    field_path: str | None
    domain_permission: DomainPermission
    capability_verified: bool
    adapter_implemented: bool
    readback_verified: bool
    readback_implemented: bool
    readback_strategy: ReadbackStrategy
    readback_field_path: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, WriteOperation):
            raise WriteIntentValidationError("operation must be a WriteOperation")
        if not isinstance(self.entity_type, EntityType):
            raise WriteIntentValidationError("entity_type must be an EntityType")
        if self.field_path is not None and (
            not isinstance(self.field_path, str) or self.field_path == ""
        ):
            raise WriteIntentValidationError("field_path must be a non-empty string or None")
        if not isinstance(self.domain_permission, DomainPermission):
            raise WriteIntentValidationError("domain_permission must be a DomainPermission")
        for name in (
            "capability_verified",
            "adapter_implemented",
            "readback_verified",
            "readback_implemented",
        ):
            if not isinstance(getattr(self, name), bool):
                raise WriteIntentValidationError(f"{name} must be a bool")
        if not isinstance(self.readback_strategy, ReadbackStrategy):
            raise WriteIntentValidationError("readback_strategy must be a ReadbackStrategy")
        if self.readback_strategy is ReadbackStrategy.READ_FIELD:
            if self.readback_field_path is None or self.readback_field_path == "":
                raise WriteIntentValidationError(
                    "READ_FIELD readback requires a readback_field_path"
                )
        elif self.readback_field_path is not None:
            raise WriteIntentValidationError(
                "only READ_FIELD readback carries a readback_field_path"
            )


def _track_field_capability(
    operation: WriteOperation,
    field_path: str,
    *,
    domain_permission: DomainPermission,
    readback_field_path: str | None = None,
    adapter_implemented: bool = False,
    readback_implemented: bool = False,
) -> WriteCapability:
    """Track field writes: P01 read these fields but never wrote them, so neither the write
    nor its readback loop is verified; the read-only adapter implements the read half only.
    ``adapter_implemented`` / ``readback_implemented`` may flip to True only for an operation
    whose production command adapter and readback are actually implemented, which never changes
    the historical ``capability_verified`` fact."""
    strategy = (
        ReadbackStrategy.READ_FIELD if readback_field_path is not None else ReadbackStrategy.UNAVAILABLE
    )
    return WriteCapability(
        operation=operation,
        entity_type=EntityType.TRACK,
        field_path=field_path,
        domain_permission=domain_permission,
        capability_verified=False,
        adapter_implemented=adapter_implemented,
        readback_verified=False,
        readback_implemented=readback_implemented,
        readback_strategy=strategy,
        readback_field_path=readback_field_path,
    )


def _playlist_capability(
    operation: WriteOperation,
    entity_type: EntityType,
    *,
    domain_permission: DomainPermission,
    capability_verified: bool,
    readback_verified: bool,
    readback_strategy: ReadbackStrategy,
    adapter_implemented: bool = False,
) -> WriteCapability:
    return WriteCapability(
        operation=operation,
        entity_type=entity_type,
        field_path=None,
        domain_permission=domain_permission,
        capability_verified=capability_verified,
        adapter_implemented=adapter_implemented,
        readback_verified=readback_verified,
        readback_implemented=False,
        readback_strategy=readback_strategy,
        readback_field_path=None,
    )


WRITE_CAPABILITY_MATRIX: Mapping[WriteOperation, WriteCapability] = MappingProxyType(
    {
        WriteOperation.SET_FAVORITED: _track_field_capability(
            WriteOperation.SET_FAVORITED,
            "library_state.favorited",
            domain_permission=DomainPermission.ALLOWED,
            readback_field_path="library_state.favorited",
            adapter_implemented=True,
            readback_implemented=True,
        ),
        WriteOperation.SET_DISLIKED: _track_field_capability(
            WriteOperation.SET_DISLIKED,
            "library_state.disliked",
            domain_permission=DomainPermission.ALLOWED,
            readback_field_path="library_state.disliked",
        ),
        WriteOperation.SET_RATING: _track_field_capability(
            WriteOperation.SET_RATING,
            "library_state.rating",
            domain_permission=DomainPermission.ALLOWED,
            readback_field_path="library_state.rating",
        ),
        WriteOperation.SET_PLAY_COUNT: _track_field_capability(
            WriteOperation.SET_PLAY_COUNT,
            "library_state.play_count",
            domain_permission=DomainPermission.DISALLOWED,
            readback_field_path="library_state.play_count",
        ),
        WriteOperation.SET_SKIP_COUNT: _track_field_capability(
            WriteOperation.SET_SKIP_COUNT,
            "library_state.skip_count",
            domain_permission=DomainPermission.DISALLOWED,
        ),
        WriteOperation.SET_LAST_PLAYED_AT: _track_field_capability(
            WriteOperation.SET_LAST_PLAYED_AT,
            "library_state.last_played_at",
            domain_permission=DomainPermission.DISALLOWED,
            readback_field_path="library_state.last_played_at",
        ),
        WriteOperation.SET_ADDED_TO_LIBRARY_AT: _track_field_capability(
            WriteOperation.SET_ADDED_TO_LIBRARY_AT,
            "library_state.added_to_library_at",
            domain_permission=DomainPermission.DISALLOWED,
            readback_field_path="library_state.added_to_library_at",
        ),
        WriteOperation.SET_ARTIST_IDS: _track_field_capability(
            WriteOperation.SET_ARTIST_IDS,
            "artist_ids",
            domain_permission=DomainPermission.DISALLOWED,
        ),
        WriteOperation.SET_ALBUM_ID: _track_field_capability(
            WriteOperation.SET_ALBUM_ID,
            "album_id",
            domain_permission=DomainPermission.DISALLOWED,
        ),
        WriteOperation.CREATE_PLAYLIST: _playlist_capability(
            WriteOperation.CREATE_PLAYLIST,
            EntityType.PLAYLIST,
            domain_permission=DomainPermission.UNSPECIFIED,
            capability_verified=True,
            readback_verified=True,
            readback_strategy=ReadbackStrategy.READ_PLAYLIST_CONTENTS,
        ),
        WriteOperation.DELETE_PLAYLIST: _playlist_capability(
            WriteOperation.DELETE_PLAYLIST,
            EntityType.PLAYLIST,
            domain_permission=DomainPermission.UNSPECIFIED,
            capability_verified=True,
            readback_verified=False,
            readback_strategy=ReadbackStrategy.READ_PLAYLIST_ABSENCE,
        ),
        WriteOperation.ADD_PLAYLIST_MEMBERSHIP: _playlist_capability(
            WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            EntityType.PLAYLIST_MEMBERSHIP,
            domain_permission=DomainPermission.ALLOWED,
            capability_verified=True,
            readback_verified=True,
            readback_strategy=ReadbackStrategy.READ_PLAYLIST_CONTENTS,
            adapter_implemented=True,
        ),
        WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP: _playlist_capability(
            WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP,
            EntityType.PLAYLIST_MEMBERSHIP,
            domain_permission=DomainPermission.ALLOWED,
            capability_verified=False,
            readback_verified=False,
            readback_strategy=ReadbackStrategy.READ_PLAYLIST_CONTENTS,
        ),
        # P11.3: add a Catalog Song to the user's library. The transport and the
        # catalog-id readback are implemented and deterministic-tested; the capability
        # stays UNVERIFIED until the live MusicKit gate produces real evidence.
        WriteOperation.ADD_LIBRARY_SONG: WriteCapability(
            operation=WriteOperation.ADD_LIBRARY_SONG,
            entity_type=EntityType.TRACK,
            field_path=None,
            domain_permission=DomainPermission.ALLOWED,
            capability_verified=False,
            adapter_implemented=True,
            readback_verified=False,
            readback_implemented=True,
            readback_strategy=ReadbackStrategy.READ_CATALOG_RELATIONSHIP,
            readback_field_path=None,
        ),
    }
)


def resolve_capability(operation: WriteOperation | str) -> WriteCapability:
    """Return the capability for an operation, failing closed on unknown operations."""
    if isinstance(operation, WriteOperation):
        key = operation
    elif isinstance(operation, str):
        try:
            key = WriteOperation(operation)
        except ValueError as error:
            raise UnsupportedWriteOperation(
                f"unsupported Apple Music write operation: {operation!r}"
            ) from error
    else:
        raise UnsupportedWriteOperation(
            f"operation must be a WriteOperation or its value string, got {operation!r}"
        )
    return WRITE_CAPABILITY_MATRIX[key]


def _require_domain_allowed(capability: WriteCapability) -> None:
    if capability.domain_permission is DomainPermission.DISALLOWED:
        raise WriteNotDomainWritable(
            f"operation {capability.operation.value} is disallowed by the domain contract"
        )
    if capability.domain_permission is DomainPermission.UNSPECIFIED:
        raise WriteDomainPermissionUnspecified(
            f"operation {capability.operation.value} has unspecified domain permission"
        )


def is_execution_ready(capability: WriteCapability) -> bool:
    """True only when a write is domain-ALLOWED, capability-verified, AND fully implemented.

    ``UNSPECIFIED`` and ``DISALLOWED`` operations are never execution-ready. A write must also
    have both its external write command and its readback orchestration implemented: a write
    that cannot be read back can never complete the ``write -> readback -> canonical
    confirmation`` loop, so it must not execute. This gate must precede any external write.
    """
    if not isinstance(capability, WriteCapability):
        raise WriteIntentValidationError("capability must be a WriteCapability")
    return (
        capability.domain_permission is DomainPermission.ALLOWED
        and capability.capability_verified
        and capability.adapter_implemented
        and capability.readback_implemented
    )


def generate_intent_id() -> str:
    """Generate a stable pending-intent identity.

    The ``int_`` prefix is intentionally outside ``identity.ENTITY_ID_PREFIX`` so an intent ID
    can never be confused with a canonical entity ID, and it is never derived from a canonical
    ID or an external persistent ID.
    """
    return f"{_INTENT_ID_PREFIX}{uuid4()}"


def validate_intent_id(intent_id: str) -> None:
    if not isinstance(intent_id, str) or not intent_id.startswith(_INTENT_ID_PREFIX):
        raise WriteIntentValidationError(
            f"intent_id must use the {_INTENT_ID_PREFIX} namespace"
        )
    suffix = intent_id[len(_INTENT_ID_PREFIX) :]
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise WriteIntentValidationError("intent_id suffix must be a canonical UUID") from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise WriteIntentValidationError("intent_id suffix must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class WriteRequirement:
    """One explicit identity requirement of a write.

    A requirement names a single canonical entity and the exact external identity it must be
    addressed by. ``canonical_id`` and ``external_identity`` are both stored because they are not
    the same thing: the requirement says "this canonical entity must correspond to this exact
    source identity", and that correspondence is re-verified against the physical
    ``external_identity_bindings`` authority before execution (never collapsed into one identity,
    and never satisfied from the canonical ``external_ids`` projection).

    ``role`` is the semantic slot the requirement fills (``TARGET`` / ``PLAYLIST`` / ``TRACK``);
    it is explicit and never inferred from position.
    """

    role: RequirementRole
    canonical_id: str
    external_identity: ExternalIdentityKey

    def __post_init__(self) -> None:
        if not isinstance(self.role, RequirementRole):
            raise WriteIntentValidationError("role must be a RequirementRole")
        if not isinstance(self.external_identity, ExternalIdentityKey):
            raise WriteIntentValidationError("external_identity must be an ExternalIdentityKey")
        if self.external_identity.source_system != _APPLE_MUSIC_SOURCE:
            raise WriteIntentValidationError("external identity must target apple_music")
        validate_canonical_id(self.external_identity.entity_type, self.canonical_id)


# Relation writes (add_playlist_membership) carry no scalar field value: the operation payload is
# fully expressed by the requirements. ``requested_value`` is still a required PendingIntent field
# so the contract is shared with scalar writes, but it is inert for relation writes -- relation
# readback is ``READ_PLAYLIST_CONTENTS``, not ``READ_FIELD``, and never consumes ``requested_value``.
RELATION_WRITE_VALUE: ObservedValue = ObservedValue.value(True)


@dataclass(frozen=True, slots=True)
class PendingIntent:
    """An operational write intent, not a canonical entity.

    ``intent_id`` is the intent's own identity and must never be treated as a canonical ID or an
    external identity. ``requirements`` is the ordered-by-role set of identity requirements that
    fully express what the write addresses: a scalar field write carries exactly one ``TARGET``
    requirement, while ``add_playlist_membership`` carries exactly one ``PLAYLIST`` and one
    ``TRACK`` requirement. ``requested_value`` is restricted to ``VALUE`` or ``NULL`` -- a scalar
    write always states what to write, so ``MISSING`` (and the False/0/null conflation it would
    allow) is rejected; relation writes use the inert ``RELATION_WRITE_VALUE``.
    """

    intent_id: str
    operation: WriteOperation
    requirements: tuple[WriteRequirement, ...]
    requested_value: ObservedValue
    state: IntentState = IntentState.PENDING

    def __post_init__(self) -> None:
        validate_intent_id(self.intent_id)
        if not isinstance(self.operation, WriteOperation):
            raise WriteIntentValidationError("operation must be a WriteOperation")
        capability = resolve_capability(self.operation)
        _require_domain_allowed(capability)
        if not isinstance(self.state, IntentState):
            raise WriteIntentValidationError("state must be an IntentState")
        requirements = tuple(self.requirements)
        if any(not isinstance(requirement, WriteRequirement) for requirement in requirements):
            raise WriteIntentValidationError("requirements must be WriteRequirement values")
        # Normalize order by role so requirement order carries no semantic weight and two
        # intents with the same requirement set compare equal regardless of construction order.
        object.__setattr__(
            self, "requirements", tuple(sorted(requirements, key=lambda r: r.role.value))
        )
        _validate_requirements(capability, self.requirements)
        if not isinstance(self.requested_value, ObservedValue):
            raise WriteIntentValidationError("requested_value must be an ObservedValue")
        if self.requested_value.state is ObservationState.MISSING:
            raise WriteIntentValidationError(
                "requested_value cannot be MISSING; a write states VALUE or NULL"
            )


def _validate_requirements(
    capability: WriteCapability, requirements: tuple[WriteRequirement, ...]
) -> None:
    if capability.field_path is not None:
        _validate_scalar_requirements(capability, requirements)
    elif capability.operation is WriteOperation.ADD_PLAYLIST_MEMBERSHIP:
        _validate_add_membership_requirements(requirements)
    else:
        raise WriteIntentValidationError(
            f"relation write payload for {capability.operation.value} is deferred; "
            "no pending intent can be formed yet"
        )


def _validate_scalar_requirements(
    capability: WriteCapability, requirements: tuple[WriteRequirement, ...]
) -> None:
    if len(requirements) != 1:
        raise WriteIntentValidationError(
            "scalar write requires exactly one target requirement"
        )
    requirement = requirements[0]
    if requirement.role is not RequirementRole.TARGET:
        raise WriteIntentValidationError("scalar write requires a target requirement")
    if requirement.external_identity.entity_type is not capability.entity_type:
        raise WriteIntentValidationError(
            "target requirement entity type does not match the operation"
        )


def _validate_add_membership_requirements(
    requirements: tuple[WriteRequirement, ...],
) -> None:
    by_role: dict[RequirementRole, WriteRequirement] = {}
    for requirement in requirements:
        if requirement.role in by_role:
            raise WriteIntentValidationError(
                f"duplicate {requirement.role.value} requirement"
            )
        by_role[requirement.role] = requirement
    playlist = by_role.get(RequirementRole.PLAYLIST)
    if playlist is None:
        raise WriteIntentValidationError(
            "add_playlist_membership requires exactly one playlist requirement"
        )
    track = by_role.get(RequirementRole.TRACK)
    if track is None:
        raise WriteIntentValidationError(
            "add_playlist_membership requires exactly one track requirement"
        )
    unknown = set(by_role) - {RequirementRole.PLAYLIST, RequirementRole.TRACK}
    if unknown:
        raise WriteIntentValidationError(
            "add_playlist_membership has unknown requirements: "
            + ", ".join(sorted(role.value for role in unknown))
        )
    if playlist.external_identity.entity_type is not EntityType.PLAYLIST:
        raise WriteIntentValidationError(
            "playlist requirement must reference a Playlist entity"
        )
    if track.external_identity.entity_type is not EntityType.TRACK:
        raise WriteIntentValidationError("track requirement must reference a Track entity")


def create_pending_intent(
    operation: WriteOperation | str,
    requirements: tuple[WriteRequirement, ...] | list[WriteRequirement],
    requested_value: ObservedValue,
) -> PendingIntent:
    """Build a PENDING intent from an explicit requirement set, failing closed on bad contracts.

    This is the single constructor for both scalar and relation intents: the requirement roles
    (and the operation's own contract) decide whether the set is legal. ``create_playlist`` /
    ``delete_playlist`` remain ``UNSPECIFIED`` and ``remove_playlist_membership`` remains deferred,
    so they still cannot form an intent even though the payload now *could* express them.
    """
    capability = resolve_capability(operation)
    _require_domain_allowed(capability)
    return PendingIntent(
        intent_id=generate_intent_id(),
        operation=capability.operation,
        requirements=tuple(requirements),
        requested_value=requested_value,
        state=IntentState.PENDING,
    )


def create_scalar_pending_intent(
    operation: WriteOperation | str,
    target_canonical_id: str,
    required_external_id: ExternalIdentityKey,
    requested_value: ObservedValue,
) -> PendingIntent:
    """Build a PENDING scalar intent from a single ``TARGET`` requirement.

    This preserves the original scalar write API shape; it is a convenience that wraps
    ``create_pending_intent`` with one ``TARGET`` requirement and does not add a second intent
    contract.
    """
    return create_pending_intent(
        operation,
        (WriteRequirement(RequirementRole.TARGET, target_canonical_id, required_external_id),),
        requested_value,
    )


_TRANSITIONS: Mapping[tuple[IntentState, WriteEvent], IntentState] = MappingProxyType(
    {
        (IntentState.PENDING, WriteEvent.COMMAND_SUCCEEDED): IntentState.AWAITING_READBACK,
        (IntentState.PENDING, WriteEvent.COMMAND_FAILED): IntentState.EXECUTION_FAILED,
        (IntentState.PENDING, WriteEvent.COMMAND_UNKNOWN): IntentState.OUTCOME_UNKNOWN,
        (IntentState.AWAITING_READBACK, WriteEvent.READBACK_MATCHED): IntentState.CONFIRMED,
        (IntentState.AWAITING_READBACK, WriteEvent.READBACK_MISMATCHED): IntentState.READBACK_MISMATCH,
        (IntentState.OUTCOME_UNKNOWN, WriteEvent.READBACK_MATCHED): IntentState.CONFIRMED,
    }
)


def advance_intent(intent: PendingIntent, event: WriteEvent) -> PendingIntent:
    """Return the intent advanced by ``event``, failing closed on illegal transitions.

    A command success never lands on CONFIRMED; only a readback event from AWAITING_READBACK
    can. Command failure, confirmation, and readback mismatch are terminal. An ambiguous command
    outcome lands on OUTCOME_UNKNOWN, from which only a matching readback (never a mismatch and
    never a re-issued command) may reconcile to CONFIRMED.
    """
    if not isinstance(intent, PendingIntent):
        raise WriteIntentValidationError("intent must be a PendingIntent")
    if not isinstance(event, WriteEvent):
        raise WriteIntentValidationError("event must be a WriteEvent")
    try:
        next_state = _TRANSITIONS[(intent.state, event)]
    except KeyError as error:
        raise WriteTransitionError(
            f"illegal transition from {intent.state.value} on {event.value}"
        ) from error
    return replace(intent, state=next_state)


def mark_command_succeeded(intent: PendingIntent) -> PendingIntent:
    return advance_intent(intent, WriteEvent.COMMAND_SUCCEEDED)


def mark_command_failed(intent: PendingIntent) -> PendingIntent:
    return advance_intent(intent, WriteEvent.COMMAND_FAILED)


def evaluate_readback(
    requested: ObservedValue, observed: ObservedValue
) -> ReadbackDecision:
    """Compare a requested write value against a source readback observation.

    ``MISSING`` observations are ``UNAVAILABLE``: no readback value is present, so the write
    cannot be confirmed. False / 0 / null are explicit values and are never mistaken for
    missing. A ``VALUE`` vs ``NULL`` disagreement is a mismatch.
    """
    if not isinstance(requested, ObservedValue) or not isinstance(observed, ObservedValue):
        raise WriteIntentValidationError("requested and observed must be ObservedValue")
    if requested.state is ObservationState.MISSING:
        raise WriteIntentValidationError("a requested write value cannot be MISSING")
    if observed.state is ObservationState.MISSING:
        return ReadbackDecision.UNAVAILABLE
    if requested.state is observed.state:
        if requested.state is ObservationState.NULL:
            return ReadbackDecision.MATCHED
        if requested.payload == observed.payload:
            return ReadbackDecision.MATCHED
    return ReadbackDecision.MISMATCHED


def apply_readback(intent: PendingIntent, observed: ObservedValue) -> PendingIntent:
    """Apply a readback observation to an AWAITING_READBACK intent.

    Matching readback confirms; disagreement is READBACK_MISMATCH (not execution failure). A
    MISSING observation fails closed: no readback means no confirmation.
    """
    decision = evaluate_readback(intent.requested_value, observed)
    if decision is ReadbackDecision.MATCHED:
        return advance_intent(intent, WriteEvent.READBACK_MATCHED)
    if decision is ReadbackDecision.MISMATCHED:
        return advance_intent(intent, WriteEvent.READBACK_MISMATCHED)
    raise WriteTransitionError("readback unavailable; cannot confirm without a readback value")
