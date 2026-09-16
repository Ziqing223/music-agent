"""Production Apple Music glue for the capability-probe adapter boundary.

``CapabilityProbeOrchestrator`` sequences a reversible ``set_favorited`` probe against an injected
``CapabilityProbeAdapter``. This module provides the production implementation of that boundary,
wired from the same primitives the ordinary ``set_favorited`` write path already uses -- the
``FavoritedCommandRunner`` command from ``apple_music_favorited_write`` and the
``AppleMusicSourceAdapter`` read from ``apple_music`` -- so a real probe drives the real Music.app
with no new osascript. It never issues a live command on its own; the orchestrator decides when to
call it.

The adapter is identity-narrow by contract. It accepts a single already-selected
``target_persistent_id`` and performs no target selection, no canonical-id -> binding resolution,
and no name/fuzzy lookup. Binding, presence, and target-eligibility preflight belong to the E2
preflight stage, not here. It never creates a ``PendingIntent`` or an ``ExecutionAttempt``: a
capability probe is not an ordinary user write, and it must not be dressed up as one.

Two halves:

- ``command(target_persistent_id, requested_favorited)`` delegates to the existing
  ``FavoritedCommandRunner`` primitive. The adapter does no outcome translation itself: it returns
  ``None`` on a clean run and lets exceptions propagate, and the orchestrator owns the exception ->
  ``CommandOutcome`` mapping. No exception type proves the side effect did not happen --
  ``AppleMusicWriteError`` is also raised for a subprocess timeout, after which the command may or
  may not have applied -- so the orchestrator treats every exception as an ambiguous ``UNKNOWN``.
  The adapter never invents a ``UNKNOWN`` or a ``FAILED`` the primitive cannot express.

- ``readback(target_persistent_id)`` reuses ``AppleMusicSourceAdapter.read_track`` on the same
  persistent ID and returns a ``ProbeReadback`` carrying *both* ``library_state.favorited`` and
  ``library_state.disliked``. The cross-field ``disliked`` observation is required probe safety
  evidence, not optional. A not-found or unavailable Track maps both fields to ``MISSING``
  (unavailable), never to a guessed value; a present non-boolean source value fails closed rather
  than being coerced.
"""

from __future__ import annotations

from typing import Any, Mapping

from music_agent.apple_music import (
    AppleMusicMappingError,
    AppleMusicSourceAdapter,
    SourceReadStatus,
)
from music_agent.apple_music_favorited_write import FavoritedCommandRunner
from music_agent.apple_music_write import AppleMusicWriteMappingError
from music_agent.capability_probe_orchestrator import ProbeReadback
from music_agent.source_observation import ObservedValue


class AppleMusicCapabilityProbeAdapter:
    """Production ``CapabilityProbeAdapter`` over the existing favorited command and read primitives.

    The command and read halves share no canonical identity context: both are keyed solely on the
    already-selected ``target_persistent_id`` the caller passes in. This adapter can therefore be
    injected with a fake command runner and a fake read runner (wrapped in the real
    ``AppleMusicSourceAdapter``) so unit and integration tests never touch the real Music.app.
    """

    def __init__(
        self,
        command_runner: FavoritedCommandRunner,
        read_adapter: AppleMusicSourceAdapter,
    ) -> None:
        self._command_runner = command_runner
        self._read_adapter = read_adapter

    def command(self, target_persistent_id: str, requested_favorited: bool) -> None:
        """Apply ``requested_favorited`` to the persistent-ID Track via the existing command runner.

        ``requested_favorited`` must be a strict ``bool``; ``False`` is a real requested value and
        is never mistaken for missing. The command outcome is expressed by exception, not return
        value: a clean run returns ``None``, and any exception propagates as an ambiguous outcome
        for the orchestrator to map (``AppleMusicWriteError`` included -- no exception proves the
        side effect did not happen).
        """
        _require_persistent_id(target_persistent_id)
        _require_bool(requested_favorited)
        self._command_runner.run(target_persistent_id, requested_favorited)

    def readback(self, target_persistent_id: str) -> ProbeReadback:
        """Read ``favorited`` and ``disliked`` back from the same persistent-ID Track.

        Both fields come from one ``read_track`` observation. ``MISSING`` (unavailable) is preserved
        distinctly from ``False``: an absent or ``null`` source field is ``MISSING``, a boolean is a
        ``VALUE``, and a not-found or unavailable Track yields ``MISSING`` for both fields rather
        than a guessed value.
        """
        _require_persistent_id(target_persistent_id)
        read_result = self._read_adapter.read_track(target_persistent_id)
        if read_result.status is SourceReadStatus.FOUND and read_result.record is not None:
            return _probe_readback(read_result.record.fields)
        return ProbeReadback(ObservedValue.missing(), ObservedValue.missing())


def _probe_readback(fields: Mapping[str, Any]) -> ProbeReadback:
    return ProbeReadback(
        _bool_observation(fields, "favorited"),
        _bool_observation(fields, "disliked"),
    )


def _bool_observation(fields: Mapping[str, Any], source_property: str) -> ObservedValue:
    if source_property not in fields or fields[source_property] is None:
        return ObservedValue.missing()
    value = fields[source_property]
    if not isinstance(value, bool):
        raise AppleMusicMappingError(
            f"source {source_property} must be a bool, got {type(value).__name__}"
        )
    return ObservedValue.value(value)


def _require_persistent_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AppleMusicWriteMappingError("persistent_id must be a non-empty string")
    return value


def _require_bool(value: object) -> None:
    if not isinstance(value, bool):
        raise AppleMusicWriteMappingError("requested_favorited must be a strict bool")
