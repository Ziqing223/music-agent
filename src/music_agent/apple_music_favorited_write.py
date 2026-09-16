"""Apple Music ``set_favorited`` write adapter and its desired-state readback.

This module is the production write boundary for a single scalar field write: setting a Library
track's ``favorited`` flag. It is deliberately scoped to ``SET_FAVORITED`` and does not implement
``set_disliked`` or ``set_rating``, does not mutate the canonical model, and does not run a live
Music.app write on its own -- callers decide when to run it.

Scalar writes differ fundamentally from the membership write in ``apple_music_write``. A
``PlaylistMembership`` has an independent ``pm_`` identity and ``(playlist_id, track_id)`` is not a
membership identity, so "the track is present in the playlist" cannot prove that a specific new
occurrence was created. A scalar field has no occurrence-identity problem: ``favorited`` is a
single boolean on a single persistent-ID-resolved Track, and the write is an idempotent
desired-state write. ``requested -> command -> read same field -> observed == requested`` is
therefore a valid ``CONFIRMED`` (desired-state confirmation), not causal attribution.

Two facts are kept strictly separate here, mirroring ``write_intent``:

- **adapter + readback implemented** -- the ``set favorited`` command and the
  ``library_state.favorited`` readback are real, tested against fake runners, and live-verified in
  P04 against the real Music.app in both directions (``False -> True`` forward and ``True -> False``
  restore, each read back through the production readback path). This sets
  ``adapter_implemented = True`` and ``readback_implemented = True`` for ``SET_FAVORITED``.

- **capability NOT yet globally verified** -- the P04 live probe covered the ``D0=false`` class
  only: whether setting ``favorited`` on a target whose ``disliked`` is already ``true`` silently
  clears ``disliked`` (the favorited/disliked cross-field side effect) remains unverified, and there
  is no verified ``set_disliked`` restore path. The promoted durable evidence is therefore not
  global verification, ``capability_verified`` stays ``False``, and the operation remains not
  execution-ready even with an implemented adapter and readback.
"""

from __future__ import annotations

import subprocess
from typing import Protocol

from music_agent.apple_music import (
    AppleMusicSourceAdapter,
    SourceReadStatus,
)
from music_agent.apple_music_write import AppleMusicWriteError, AppleMusicWriteMappingError
from music_agent.identity import EntityType
from music_agent.source_observation import ObservedValue, ObservationState
from music_agent.write_execution import AmbiguousCommandOutcomeError, DeterministicCommandError
from music_agent.write_intent import (
    PendingIntent,
    RequirementRole,
    WriteOperation,
    WriteRequirement,
)
from music_agent.write_intent_formation import (
    BindingDriftError,
    BindingResolver,
    MissingExternalBindingError,
    resolve_requirements,
)


SET_FAVORITED_SCRIPT = r'''
on run argv
    if (count of argv) is not 2 then error "track persistent ID and favorited boolean are required"
    set targetTrackID to item 1 of argv
    set targetFavoritedText to item 2 of argv
    tell application "Music"
        set matchingTracks to every track of library playlist 1 whose persistent ID is targetTrackID
        if (count of matchingTracks) is 0 then error "track not found"
        set sourceTrack to item 1 of matchingTracks
        set favorited of sourceTrack to (targetFavoritedText is "true")
    end tell
    return "set"
end run
'''


class FavoritedCommandRunner(Protocol):
    """Issue the set-favorited AppleScript; replaced by a fake in tests."""

    def run(self, track_persistent_id: str, favorited: bool) -> str: ...


class OsascriptFavoritedCommandRunner:
    """Invoke the set-favorited AppleScript with persistent ID and boolean passed as argv.

    A failure before the subprocess is spawned (``OSError``) is a known failure: no side effect
    occurred, so it raises ``DeterministicCommandError``. A ``TimeoutExpired`` or a non-zero exit
    means the command reached ``osascript`` and may or may not have applied, so it raises
    ``AmbiguousCommandOutcomeError`` -- the orchestrator records that as an unknown outcome, never
    as a deterministic failure.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, track_persistent_id: str, favorited: bool) -> str:
        _require_persistent_id(track_persistent_id)
        _require_bool(favorited)
        flag = "true" if favorited else "false"
        try:
            completed = subprocess.run(
                ["osascript", "-e", SET_FAVORITED_SCRIPT, "--", track_persistent_id, flag],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AmbiguousCommandOutcomeError(str(error)) from error
        except OSError as error:
            raise DeterministicCommandError(str(error)) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"osascript exited {completed.returncode}"
            raise AmbiguousCommandOutcomeError(detail)
        return completed.stdout.strip()


class FavoritedWriteAdapter:
    """Production ``set_favorited`` command and readback behind injected runners.

    Implements the ``WriteCommandAdapter`` shape (``command`` / ``readback``) the write
    orchestrator consumes, scoped to the single ``SET_FAVORITED`` operation. Both halves resolve
    the intent's single ``TARGET`` requirement through ``external_identity_bindings`` (via the
    injected ``BindingResolver``) and fail closed on a missing or drifted binding. The command
    never reads the canonical ``external_ids`` projection, never matches by name, and never does
    fuzzy matching.
    """

    def __init__(
        self,
        command_runner: FavoritedCommandRunner,
        read_adapter: AppleMusicSourceAdapter,
        binding_resolver: BindingResolver,
    ) -> None:
        self._command_runner = command_runner
        self._read_adapter = read_adapter
        self._binding_resolver = binding_resolver

    def command(self, intent: PendingIntent) -> None:
        """Issue the set-favorited command for the intent's single TARGET requirement.

        Every failure that provably happens before dispatch -- argument/intent validation, or a
        missing or drifted binding resolved before the command -- is raised as a
        ``DeterministicCommandError``, the only signal the orchestrator maps to a deterministic
        failure. A dispatched command whose outcome is ambiguous stays an
        ``AmbiguousCommandOutcomeError``. Any other exception propagates unchanged and is treated
        by the orchestrator as an unknown outcome.
        """
        try:
            _require_favorited_intent(intent)
            _target_requirement(intent)
            favorited = _requested_favorited(intent)
            track_persistent_id = self._resolve_target(intent)
            self._command_runner.run(track_persistent_id, favorited)
        except (AmbiguousCommandOutcomeError, DeterministicCommandError):
            raise
        except (
            AppleMusicWriteMappingError,
            MissingExternalBindingError,
            BindingDriftError,
        ) as error:
            raise DeterministicCommandError(str(error)) from error

    def readback(self, intent: PendingIntent) -> ObservedValue:
        """Read ``library_state.favorited`` back from the same persistent-ID Track.

        Reuses the verified read adapter. A confirmed-not-found Track yields ``MISSING``
        (unavailable); a transient lookup failure raises so the orchestrator leaves the intent
        ``AWAITING_READBACK`` for a later attempt. Neither ever confirms.
        """
        _require_favorited_intent(intent)
        target = _target_requirement(intent)
        track_persistent_id = self._resolve_target(intent)
        read_result = self._read_adapter.read_track(track_persistent_id)
        if read_result.status is SourceReadStatus.CONFIRMED_NOT_FOUND:
            return ObservedValue.missing()
        if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
            raise AppleMusicWriteError(f"favorited readback lookup failed: {read_result.error}")
        observation = self._read_adapter.build_observation(target.canonical_id, read_result)
        return observation.fields["library_state.favorited"]

    def _resolve_target(self, intent: PendingIntent) -> str:
        resolved = resolve_requirements(intent.requirements, self._binding_resolver)
        track_persistent_id = resolved.get(RequirementRole.TARGET)
        if track_persistent_id is None:
            raise AppleMusicWriteMappingError(
                "set_favorited requires a resolvable TARGET binding"
            )
        return track_persistent_id


def _require_favorited_intent(intent: object) -> None:
    if not isinstance(intent, PendingIntent):
        raise AppleMusicWriteMappingError("intent must be a PendingIntent")
    if intent.operation is not WriteOperation.SET_FAVORITED:
        raise AppleMusicWriteMappingError(
            f"favorited adapter supports only set_favorited, got {intent.operation.value}"
        )


def _target_requirement(intent: PendingIntent) -> WriteRequirement:
    targets = [r for r in intent.requirements if r.role is RequirementRole.TARGET]
    if len(targets) != 1:
        raise AppleMusicWriteMappingError(
            "set_favorited requires exactly one TARGET requirement"
        )
    target = targets[0]
    if target.external_identity.entity_type is not EntityType.TRACK:
        raise AppleMusicWriteMappingError(
            "set_favorited TARGET requirement must reference a Track entity"
        )
    return target


def _requested_favorited(intent: PendingIntent) -> bool:
    requested = intent.requested_value
    if requested.state is not ObservationState.VALUE:
        raise AppleMusicWriteMappingError(
            "set_favorited requested value must be a VALUE, not MISSING or NULL"
        )
    if not isinstance(requested.payload, bool):
        raise AppleMusicWriteMappingError("set_favorited requested value must be a strict bool")
    return requested.payload


def _require_persistent_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AppleMusicWriteMappingError("persistent_id must be a non-empty string")
    return value


def _require_bool(value: object) -> None:
    if not isinstance(value, bool):
        raise AppleMusicWriteMappingError("favorited must be a strict bool")
