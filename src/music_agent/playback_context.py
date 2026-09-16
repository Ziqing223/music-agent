"""P15-PC: Playback Context -- the service-level playback-continuity coordination layer.

P14 froze the action-log register (``ActiveMusicContext``: what the service commanded)
and engine truth is read on demand (Music.app snapshot, afplay ``is_preview_active``).
This module adds the coordination facts playback continuity needs without touching
either: the suspension record (what a preview interrupted) and -- from P15-S1 -- the
continuous-preview session.

Deliberate boundaries (extending the ActiveMusicContext constitution):

- **Runtime only.** Nothing here is persisted, migrated, or replayed; a fresh service
  instance starts with no suspension and no session.
- **Single writer.** Only ``SharedAgentService`` writes; readers get one read at a time
  under the lock, never a live mutable view.
- **suspend-don't-overlap (P2).** A suspension entry exists only when Music.app was
  actually playing when a preview started. Any other player state pauses nothing and
  records nothing (fail closed, never guessed).
- **restore-by-intent (P3).** The suspension never resumes anything on its own.
  It only feeds honest user-facing messaging; restore is the user's explicit
  继续播放/play command through the existing playback boundary.
- **natural-end restore arm (P17 acceptance).** One narrowly-scoped exception to
  P3: the preview entry points may *arm* the suspension they just recorded; the
  single natural end of that preview (its last clip, or its session-less clip)
  may then restore formal playback -- fail closed (see
  ``take_restore_arm``). Explicit stop, failure, advance-to-completion, and any
  user playback command disarm it, so a stale or user-overridden interruption
  can never auto-restore. The arm is per-preview-start: a preview that records
  no suspension re-arms to ``None``.
- **Thread safety.** Preview lifecycle events arrive on the reaper thread (natural-end
  hook) while commands arrive on the agent-loop thread; every mutation runs under a lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from music_agent.preview_session import PreviewSessionRegistry


@dataclass(frozen=True, slots=True)
class SuspendedPlaybackEntry:
    """One recorded interruption of formal playback.

    Created only when Music.app read as playing before a preview started.
    ``pause_ok`` is True only when the pause command succeeded; False keeps the
    interruption fact honest when the pause itself failed (the music may still be
    sounding -- recovery messaging must not claim it was paused).
    """

    player_state: str
    persistent_id: str | None
    name: str | None
    pause_ok: bool


class AudioSuspension:
    """Service-level register of the most recent preview-triggered suspension.

    One entry at a time: a later record replaces the earlier one; ``clear`` empties.
    Terminal preview states deliberately do NOT clear it -- the entry keeps serving
    honest messaging on later messages (Music.app's own paused state is the actual
    restore basis; the entry only describes what was interrupted).

    P17 acceptance: the register also holds the *restore arm* -- which recorded
    interruption (if any) may auto-restore on a preview's natural end. The arm is
    armed by the preview entry points (``arm_restore`` with their own entry, or
    ``None`` when nothing was playing) and consumed one-shot by the natural-end
    path (``take_restore_arm``); explicit stops, failures, and user playback
    commands disarm it. Entry and arm are independent: disarming never drops the
    messaging entry, and clearing the entry never implicitly disarms (callers
    that mean both do both).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._entry: SuspendedPlaybackEntry | None = None
        self._restore_arm: SuspendedPlaybackEntry | None = None

    def record(self, entry: SuspendedPlaybackEntry) -> None:
        if not isinstance(entry, SuspendedPlaybackEntry):
            raise ValueError("entry must be a SuspendedPlaybackEntry")
        with self._lock:
            self._entry = entry

    def clear(self) -> None:
        with self._lock:
            self._entry = None

    def arm_restore(self, entry: SuspendedPlaybackEntry | None) -> None:
        """Arm (``entry``) or disarm (``None``) the natural-end auto-restore.

        Always the specific entry THIS preview recorded -- never an older one --
        so a preview that interrupts nothing re-arms to ``None`` and an earlier
        preview's arm cannot leak forward.
        """
        if entry is not None and not isinstance(entry, SuspendedPlaybackEntry):
            raise ValueError("restore arm must be a SuspendedPlaybackEntry or None")
        with self._lock:
            self._restore_arm = entry

    def disarm_restore(self) -> None:
        """Cancel auto-restore eligibility (explicit stop / failure / user command)."""
        with self._lock:
            self._restore_arm = None

    def take_restore_arm(self) -> SuspendedPlaybackEntry | None:
        """Consume the arm one-shot: a natural end takes it, eligible or not.

        The caller still applies the readback guard before acting: only a
        pause that succeeded, on the same persistent id, still paused, restores.
        """
        with self._lock:
            arm = self._restore_arm
            self._restore_arm = None
            return arm

    @property
    def value(self) -> SuspendedPlaybackEntry | None:
        """The recorded entry, or ``None`` when nothing was suspended."""
        with self._lock:
            return self._entry

    @property
    def pending_restore(self) -> SuspendedPlaybackEntry | None:
        """The active preview's unfulfilled restore obligation, if any.

        Unlike ``value`` (the durable-in-process restore-by-intent memo), this
        is lifecycle state: natural completion consumes it and explicit stop,
        failure, or a user playback command disarms it. Presentation reads may
        inspect it, but only ``take_restore_arm`` may consume it.
        """
        with self._lock:
            return self._restore_arm


class PlaybackContext:
    """Coordination holder owned by ``SharedAgentService`` (one per service instance).

    Composes the two P15 coordination registers: the suspension record (what a
    preview interrupted) and -- since P15-S1 -- the continuous-preview session
    (queue + cursor + state). Readers compose them at observation time; neither is
    derived from the other.
    """

    def __init__(self) -> None:
        self._suspension = AudioSuspension()
        self._sessions = PreviewSessionRegistry()

    @property
    def suspension(self) -> AudioSuspension:
        return self._suspension

    @property
    def sessions(self) -> PreviewSessionRegistry:
        return self._sessions
