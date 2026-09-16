"""Apple Music write adapter for ``add_playlist_membership``, and its fail-closed readback.

This module is the production write boundary for the first Apple Music write operation. It is
deliberately narrower than the read-only adapter in ``music_agent.apple_music``: it implements one
command path (add an existing Library track to a user playlist as a new membership occurrence)
and one read path (enumerate a playlist's member Track persistent IDs), and it never performs a
live mutation on its own -- callers decide when to run it.

Two facts are kept strictly separate here, mirroring ``write_intent``:

- **command adapter implemented** -- the ``add_playlist_membership`` command is real and tested.
  It resolves both the canonical Playlist and the canonical Track to durable Apple Music
  persistent IDs and issues a single, unambiguous ``duplicate ... to playlist`` AppleScript. This
  sets ``adapter_implemented = True`` for ``ADD_PLAYLIST_MEMBERSHIP``.

- **readback NOT implemented as confirmation** -- the read path enumerates a playlist's members,
  but member enumeration carries **no occurrence identity**. A PlaylistMembership has an
  independent ``pm_`` identity and ``(playlist_id, track_id)`` is not a membership identity, so
  "the target Track is present in the Playlist" cannot prove that *this* command created a new
  occurrence: the same Track may already be present, once or more, before the command. The
  readback evaluator therefore always returns ``UNAVAILABLE`` and never ``MATCHED``. This keeps
  ``readback_implemented = False`` and the operation not execution-ready.

Identity boundary
-----------------

The adapter resolves external identity through a ``MembershipBindingResolver``, whose production
implementation reads ``external_identity_bindings`` (the physical authority). It never reads the
canonical ``external_ids.apple_music_persistent_id`` projection, never matches by name, and never
does fuzzy matching. A canonical Playlist or Track with no durable binding fails closed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Protocol

from music_agent.identity import EntityType, validate_canonical_id
from music_agent.repository import CanonicalRepository
from music_agent.write_intent import ReadbackDecision, RequirementRole, WriteRequirement
from music_agent.write_intent_formation import resolve_requirements


class AppleMusicWriteError(RuntimeError):
    code = "apple_music_write_failed"


class AppleMusicWriteMappingError(ValueError):
    code = "validation_error"


class MembershipBindingMissingError(AppleMusicWriteError):
    code = "membership_binding_missing"


ADD_MEMBERSHIP_SCRIPT = r'''
on run argv
    if (count of argv) is not 2 then error "playlist and track persistent IDs are required"
    set targetPlaylistID to item 1 of argv
    set targetTrackID to item 2 of argv
    tell application "Music"
        set targetPlaylist to first user playlist whose persistent ID is targetPlaylistID
        set targetTrack to first track of library playlist 1 whose persistent ID is targetTrackID
        duplicate targetTrack to targetPlaylist
    end tell
    return "added"
end run
'''

READ_PLAYLIST_MEMBERS_SCRIPT = r'''
on replaceText(findText, replacementText, sourceText)
    set previousDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to findText
    set textItems to every text item of sourceText
    set AppleScript's text item delimiters to replacementText
    set replacedText to textItems as text
    set AppleScript's text item delimiters to previousDelimiters
    return replacedText
end replaceText

on jsonString(sourceValue)
    set valueText to sourceValue as text
    set valueText to my replaceText("\\", "\\\\", valueText)
    set valueText to my replaceText(quote, "\\\"", valueText)
    return quote & valueText & quote
end jsonString

on run argv
    if (count of argv) is not 1 then error "one playlist persistent ID is required"
    set targetPlaylistID to item 1 of argv
    tell application "Music"
        set targetPlaylist to first user playlist whose persistent ID is targetPlaylistID
        set memberIDs to {}
        repeat with sourceTrack in every track of targetPlaylist
            set end of memberIDs to persistent ID of sourceTrack
        end repeat
    end tell
    set jsonItems to {}
    repeat with memberID in memberIDs
        set end of jsonItems to my jsonString(memberID)
    end repeat
    set previousDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to ","
    set joined to jsonItems as text
    set AppleScript's text item delimiters to previousDelimiters
    return "[" & joined & "]"
end run
'''


class AddMembershipRunner(Protocol):
    """Issue the add-membership AppleScript; replaced by a fake in tests."""

    def run(self, playlist_persistent_id: str, track_persistent_id: str) -> str: ...


class PlaylistMembersRunner(Protocol):
    """Enumerate one playlist's member Track persistent IDs; replaced by a fake in tests."""

    def run(self, playlist_persistent_id: str) -> str: ...


class MembershipBindingResolver(Protocol):
    """Resolve a canonical entity to its durable Apple Music persistent ID, or ``None``."""

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None: ...


class OsascriptAddMembershipRunner:
    """Invoke the add-membership AppleScript with persistent IDs passed as argv, never interpolated."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, playlist_persistent_id: str, track_persistent_id: str) -> str:
        _require_persistent_id(playlist_persistent_id)
        _require_persistent_id(track_persistent_id)
        try:
            completed = subprocess.run(
                ["osascript", "-e", ADD_MEMBERSHIP_SCRIPT, "--",
                 playlist_persistent_id, track_persistent_id],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AppleMusicWriteError(str(error)) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"osascript exited {completed.returncode}"
            raise AppleMusicWriteError(detail)
        return completed.stdout.strip()


class OsascriptPlaylistMembersRunner:
    """Invoke the member-enumeration AppleScript with the playlist persistent ID as argv."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, playlist_persistent_id: str) -> str:
        _require_persistent_id(playlist_persistent_id)
        try:
            completed = subprocess.run(
                ["osascript", "-e", READ_PLAYLIST_MEMBERS_SCRIPT, "--", playlist_persistent_id],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AppleMusicWriteError(str(error)) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"osascript exited {completed.returncode}"
            raise AppleMusicWriteError(detail)
        return completed.stdout.strip()


class RepositoryMembershipBindingResolver:
    """Resolve canonical IDs through ``external_identity_bindings``, the physical authority.

    This deliberately does not consult the canonical ``external_ids`` projection. A canonical
    entity whose persistent ID only appears in ``load_model`` projections but has no durable
    binding resolves to ``None`` and fails closed.
    """

    def __init__(self, repository: CanonicalRepository) -> None:
        self._repository = repository

    def resolve(self, entity_type: EntityType, canonical_id: str) -> str | None:
        validate_canonical_id(entity_type, canonical_id)
        for key, bound_canonical_id in self._repository.list_external_identity_bindings(
            "apple_music", entity_type
        ):
            if bound_canonical_id == canonical_id:
                return key.external_id
        return None


class AppleMusicWriteAdapter:
    """Production add-membership command and member-enumeration read, behind injected runners."""

    def __init__(
        self,
        command_runner: AddMembershipRunner,
        members_runner: PlaylistMembersRunner,
        binding_resolver: MembershipBindingResolver,
    ) -> None:
        self._command_runner = command_runner
        self._members_runner = members_runner
        self._binding_resolver = binding_resolver

    def add_playlist_membership(self, playlist_canonical_id: str, track_canonical_id: str) -> None:
        """Issue the add-membership command for one Playlist and one Track.

        Both canonical references must resolve to durable Apple Music persistent IDs. A missing
        binding fails closed. This method performs the external command only; it never mutates the
        canonical model, never confirms canonical state, and never records a readback.
        """
        playlist_persistent_id = self._resolve_external_id(
            EntityType.PLAYLIST, playlist_canonical_id
        )
        track_persistent_id = self._resolve_external_id(EntityType.TRACK, track_canonical_id)
        self._run_add_membership(playlist_persistent_id, track_persistent_id)

    def add_playlist_membership_requirements(
        self, requirements: Sequence[WriteRequirement]
    ) -> None:
        """Issue the add-membership command from a persisted intent's requirement set.

        This is the single authoritative path for replaying a relation intent: it re-verifies each
        captured requirement against the current physical binding before the command, and fails
        closed on a missing binding or on drift (a binding that changed since the intent was
        formed). It never silently re-resolves a changed canonical reference to a different
        external ID.
        """
        resolved = resolve_requirements(requirements, self._binding_resolver)
        playlist_persistent_id = resolved.get(RequirementRole.PLAYLIST)
        track_persistent_id = resolved.get(RequirementRole.TRACK)
        if playlist_persistent_id is None or track_persistent_id is None:
            raise MembershipBindingMissingError(
                "add_playlist_membership requires a playlist and a track requirement"
            )
        self._run_add_membership(playlist_persistent_id, track_persistent_id)

    def _run_add_membership(
        self, playlist_persistent_id: str, track_persistent_id: str
    ) -> None:
        self._command_runner.run(playlist_persistent_id, track_persistent_id)

    def read_playlist_members(self, playlist_canonical_id: str) -> tuple[str, ...]:
        """Enumerate one Playlist's member Track persistent IDs, in source order.

        The Playlist must resolve to a durable persistent ID. The result is ordered member
        evidence only; it carries no per-occurrence identity and cannot be used to confirm that a
        specific new membership was created.
        """
        playlist_persistent_id = self._resolve_external_id(
            EntityType.PLAYLIST, playlist_canonical_id
        )
        output = self._members_runner.run(playlist_persistent_id)
        return _parse_members(output)

    def _resolve_external_id(self, entity_type: EntityType, canonical_id: str) -> str:
        validate_canonical_id(entity_type, canonical_id)
        external_id = self._binding_resolver.resolve(entity_type, canonical_id)
        if external_id is None:
            raise MembershipBindingMissingError(
                f"no durable apple_music binding for {entity_type.value} {canonical_id!r}"
            )
        return external_id


def evaluate_membership_readback(
    observed_member_persistent_ids: Sequence[str], requested_track_persistent_id: str
) -> ReadbackDecision:
    """Evaluate a membership readback observation, always failing closed to ``UNAVAILABLE``.

    The only source evidence Apple Music can provide for a playlist is an ordered enumeration of
    member Track persistent IDs. That enumeration has no occurrence identity, so it can never
    distinguish the occurrence this command created from a pre-existing occurrence of the same
    Track. Whether the requested Track is present, absent, or present twice, the decision is
    ``UNAVAILABLE``: presence alone is not confirmation, and absence alone is not a safe mismatch
    (a concurrent external edit or a crash window could explain either). This is the mechanism
    that prevents a pre-existing duplicate Track from being falsely confirmed.
    """
    if isinstance(observed_member_persistent_ids, (str, bytes)) or not isinstance(
        observed_member_persistent_ids, Sequence
    ):
        raise AppleMusicWriteMappingError(
            "observed members must be a sequence of persistent IDs"
        )
    for persistent_id in observed_member_persistent_ids:
        if not isinstance(persistent_id, str) or persistent_id == "":
            raise AppleMusicWriteMappingError("observed member persistent IDs must be non-empty strings")
    _require_persistent_id(requested_track_persistent_id)
    return ReadbackDecision.UNAVAILABLE


def _parse_members(output: str) -> tuple[str, ...]:
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise AppleMusicWriteMappingError(f"playlist members output is not JSON: {error}") from error
    if not isinstance(payload, list):
        raise AppleMusicWriteMappingError("playlist members output must be a JSON array")
    members: list[str] = []
    for item in payload:
        if not isinstance(item, str) or item == "":
            raise AppleMusicWriteMappingError("playlist members must be non-empty strings")
        members.append(item)
    return tuple(members)


def _require_persistent_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AppleMusicWriteMappingError("persistent_id must be a non-empty string")
    return value
