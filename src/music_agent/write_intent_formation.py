"""Relation write-intent formation and drift-validated requirement resolution.

A relation intent (``add_playlist_membership``) cannot be formed from canonical IDs alone: its
external write needs two independent source identities (the Playlist and the Track), and each
must be resolved through the physical authority ``external_identity_bindings``. This module owns
that boundary.

Two distinct moments live here:

1. **Formation** -- ``form_add_playlist_membership_intent`` resolves the Playlist and Track
   canonical IDs to durable external identities *before* the intent exists. If either binding is
   missing, formation fails closed and no partial intent is created; the intent is never formed
   in the hope that execution will later fill in a second binding. The result is a
   self-contained ``PendingIntent`` whose requirements record both canonical IDs and their
   captured external identities.

2. **Execution preparation** -- ``resolve_requirements`` re-verifies, before any external command,
   that the current physical binding for each captured requirement still equals the external
   identity the intent recorded. If a binding has changed (drift) or disappeared, execution fails
   closed rather than silently using a new external ID, which would change the meaning of an
   already-persisted intent.

The canonical ``external_ids.apple_music_persistent_id`` projection is never consulted as
authority here: it is a projection, not a physical binding. No name matching and no fuzzy
matching occur anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping, Protocol

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.source_observation import ObservedValue
from music_agent.write_intent import (
    RELATION_WRITE_VALUE,
    PendingIntent,
    RequirementRole,
    WriteOperation,
    WriteRequirement,
    create_pending_intent,
)

_APPLE_MUSIC_SOURCE = "apple_music"


class BindingResolver(Protocol):
    """Resolve a canonical entity to its durable Apple Music persistent ID, or ``None``."""

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None: ...


class WriteIntentFormationError(ValueError):
    code = "write_intent_formation_error"


class MissingExternalBindingError(WriteIntentFormationError):
    code = "missing_external_binding"


class BindingDriftError(WriteIntentFormationError):
    code = "binding_drift"


def _resolve_external_id(
    resolver: BindingResolver, entity_type: EntityType, canonical_id: str
) -> str:
    external_id = resolver.resolve(entity_type, canonical_id)
    if external_id is None:
        raise MissingExternalBindingError(
            f"no durable apple_music binding for {entity_type.value} {canonical_id!r}"
        )
    return external_id


def form_add_playlist_membership_intent(
    playlist_canonical_id: str,
    track_canonical_id: str,
    resolver: BindingResolver,
    requested_value: ObservedValue = RELATION_WRITE_VALUE,
) -> PendingIntent:
    """Form a self-contained ``add_playlist_membership`` intent, failing closed on any gap.

    Both the Playlist and the Track must resolve to durable Apple Music persistent IDs through
    ``resolver``. If either is missing, no intent is formed -- the function never creates a
    partial intent that execution would have to complete later.
    """
    playlist_external_id = _resolve_external_id(resolver, EntityType.PLAYLIST, playlist_canonical_id)
    track_external_id = _resolve_external_id(resolver, EntityType.TRACK, track_canonical_id)
    return create_pending_intent(
        WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
        (
            WriteRequirement(
                RequirementRole.PLAYLIST,
                playlist_canonical_id,
                ExternalIdentityKey(_APPLE_MUSIC_SOURCE, EntityType.PLAYLIST, playlist_external_id),
            ),
            WriteRequirement(
                RequirementRole.TRACK,
                track_canonical_id,
                ExternalIdentityKey(_APPLE_MUSIC_SOURCE, EntityType.TRACK, track_external_id),
            ),
        ),
        requested_value,
    )


def resolve_requirements(
    requirements: Sequence[WriteRequirement], resolver: BindingResolver
) -> Mapping[RequirementRole, str]:
    """Re-verify each requirement against the current binding and return role -> external ID.

    This is the execution-preparation boundary for a persisted relation intent: it re-reads the
    physical binding for every requirement and fails closed if a binding is missing or no longer
    equals the captured external identity (binding drift). The returned external IDs are the
    *captured* IDs (equal to the verified current IDs), so the caller can never silently substitute
    a re-resolved, changed target.
    """
    resolved: dict[RequirementRole, str] = {}
    for requirement in requirements:
        if not isinstance(requirement, WriteRequirement):
            raise WriteIntentFormationError("requirements must be WriteRequirement values")
        current = resolver.resolve(
            requirement.external_identity.entity_type, requirement.canonical_id
        )
        if current is None:
            raise MissingExternalBindingError(
                f"no durable apple_music binding for "
                f"{requirement.external_identity.entity_type.value} {requirement.canonical_id!r}"
            )
        if current != requirement.external_identity.external_id:
            raise BindingDriftError(
                f"binding drift for {requirement.role.value} {requirement.canonical_id!r}: "
                f"intent captured {requirement.external_identity.external_id!r} but the current "
                f"binding is {current!r}"
            )
        resolved[requirement.role] = current
    return resolved
