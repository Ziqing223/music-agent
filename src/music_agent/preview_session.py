"""P15-S1: Preview Session -- the service-level continuous-preview register.

Where ``catalog_preview`` plays one 30-second clip and ``ActiveMusicContext`` records
audio actions, this module records the *continuous* task: the queue of a「都放一遍」
run, where it currently is, and whether it is running, completed, or cancelled.

Constitution (mirrors ``active_music_context``):

- **Runtime only.** Nothing is persisted or migrated; a fresh service has no session.
- **Single writer.** Only ``SharedAgentService`` writes; readers get one point-in-time
  ``PreviewSessionSnapshot`` under the lock, never a live mutable view.
- **Terminal states are one-shot.** COMPLETED (every item ended naturally, skips
  included), CANCELLED (user stop / ``暂停`` / pre-empted by a single preview), and
  FAILED (a systemic clip-start error, reason exposed) all empty the registry; a
  late reaper callback then finds nothing and does nothing.
- **The session never owns the suspension.** Pausing formal playback before the first
  clip is the service-level ``AudioSuspension`` ([[playback_context]]) concern; this
  register carries queue + cursor only, and readers compose the two.
- **Thread safety.** The runner's natural-finish hook arrives on the reaper thread
  while commands arrive on the agent-loop thread; every transition runs under a lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any


class PreviewSessionState(StrEnum):
    """The four states a continuous-preview session may be observed in.

    Deliberately no PAUSED state in this MVP: afplay has no mid-clip resume, so a
    paused-试听 notion would be fiction. ``暂停`` during a session is cancelled by
    the routing layer instead.

    FAILED (P15 真机修复) is the third one-shot terminal: a *systemic* clip-start
    failure (infrastructure, not one track being unavailable) ends the whole run
    with its reason exposed -- the per-item ``skipped`` path is for item-level
    facts only, never a masquerade for a broken pipeline.
    """

    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PreviewSessionItem:
    """One queue entry, assembled service-side from the active recommendation batch.

    ``route`` is the ``_playback_annotation`` projection (``library`` /
    ``preview_only`` -- ``unavailable`` entries are filtered out at assembly).

    P15 真机修复: ``run_id`` / ``itunes_id`` / ``batch_item_index`` are the
    *assembly-time* facts resolved on the agent-loop thread, so the reaper-thread
    auto-advance never touches a repository (SQLite stays single-threaded by
    construction).
    """

    canonical_id: str
    name: str | None
    artist_name: str | None
    route: str
    run_id: str | None = None
    itunes_id: str | None = None
    batch_item_index: int | None = None


@dataclass(frozen=True, slots=True)
class PreviewSessionSnapshot:
    """Immutable point-in-time export: the tool/messaging shape, never a view."""

    state: str
    total: int
    position: int  # 1-based: the clip currently (or last) sounded; next = position + 1
    current_canonical_id: str | None
    current_name: str | None  # display name of the position above (presentation, models)
    skipped: tuple[dict[str, Any], ...]  # {canonical_id, name, reason}
    failure_reason: str | None = None  # set only when state == "failed"


class PreviewSession:
    """One continuous-preview run: ranked queue + 1-based cursor + terminal state.

    ``advance`` returns the next item to start (``None`` once the queue is exhausted,
    flipping the state to COMPLETED then and only then) -- or, with a ``skipped``
    record for the item that failed to start, moves past it and keeps going. A
    systemic failure ends the run outright through ``fail`` (state FAILED, reason
    exposed). A session whose state grew terminal answers every later call with a
    no-op, so a stop racing a natural finish can never resurrect it.
    """

    def __init__(
        self,
        items: list[PreviewSessionItem] | tuple[PreviewSessionItem, ...],
        skipped: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> None:
        if not items:
            raise ValueError("a preview session requires at least one item")
        if any(not isinstance(item, PreviewSessionItem) for item in items):
            raise ValueError("every session item must be a PreviewSessionItem")
        self._lock = Lock()
        self._items: tuple[PreviewSessionItem, ...] = tuple(items)
        self._skipped: list[dict[str, Any]] = [dict(record) for record in skipped]
        self._position = 1
        self._state = PreviewSessionState.RUNNING
        self._failure_reason: str | None = None

    def advance(self, skipped: dict[str, Any] | None = None) -> PreviewSessionItem | None:
        """Consume the current clip's end and hand out the next item to start.

        ``skipped`` records the item due at ``position`` as failed-to-start (with the
        caller's ``reason``) before moving on -- the honest "skip and continue" path
        for item-level unavailability. Returns ``None`` when the queue is exhausted
        (state becomes COMPLETED) or when the session is already terminal (the
        caller treats that as end-of-line).
        """
        if skipped is not None and not isinstance(skipped, dict):
            raise ValueError("skipped must be a dict")
        with self._lock:
            if self._state is not PreviewSessionState.RUNNING:
                return None
            if skipped is not None:
                self._skipped.append(dict(skipped))
            if self._position >= len(self._items):
                self._state = PreviewSessionState.COMPLETED
                return None
            self._position += 1
            return self._items[self._position - 1]

    def cancel(self) -> bool:
        """Terminate a RUNNING session; False when it was already terminal (idempotent)."""
        with self._lock:
            if self._state is not PreviewSessionState.RUNNING:
                return False
            self._state = PreviewSessionState.CANCELLED
            return True

    def fail(self, reason: str) -> bool:
        """P15 真机修复: end a RUNNING session as FAILED with the reason exposed.

        One-shot terminal exactly like ``cancel``/natural completion: a later
        advance, cancel or fail finds the session terminal and no-ops.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fail requires a non-empty reason")
        with self._lock:
            if self._state is not PreviewSessionState.RUNNING:
                return False
            self._state = PreviewSessionState.FAILED
            self._failure_reason = reason
            return True

    def current_item(self) -> PreviewSessionItem | None:
        """The clip expected to be sounding now -- None once the session went terminal."""
        with self._lock:
            if self._state is not PreviewSessionState.RUNNING:
                return None
            return self._items[self._position - 1]

    def snapshot(self) -> PreviewSessionSnapshot:
        """Point-in-time export; terminal states keep the last position honestly."""
        with self._lock:
            index = min(self._position, len(self._items)) - 1
            return PreviewSessionSnapshot(
                state=self._state.value,
                total=len(self._items),
                position=self._position,
                current_canonical_id=self._items[index].canonical_id,
                current_name=self._items[index].name,
                skipped=tuple(self._skipped),
                failure_reason=(
                    self._failure_reason
                    if self._state is PreviewSessionState.FAILED
                    else None
                ),
            )


class PreviewSessionRegistry:
    """Single-writer holder for the one active session (at most one ever exists)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._session: PreviewSession | None = None

    def start(
        self,
        items: list[PreviewSessionItem] | tuple[PreviewSessionItem, ...],
        skipped: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> tuple[PreviewSession, PreviewSession | None]:
        """Start (or replace) the session; returns it plus the session it replaced.

        The replaced session is returned so the caller can close it out honestly
        (a cancellation notice, since its clips will never finish) -- the registry
        itself drops it here.
        """
        with self._lock:
            replaced = self._session
            session = PreviewSession(items, skipped)
            self._session = session
            return session, replaced

    def current(self) -> PreviewSession | None:
        """The live session, or None (never a terminal session: those are cleared)."""
        with self._lock:
            return self._session

    def snapshot(self) -> PreviewSessionSnapshot | None:
        """Point-in-time snapshot of the live session, or None when none is running."""
        with self._lock:
            session = self._session
            return session.snapshot() if session is not None else None

    def cancel(self) -> PreviewSessionSnapshot | None:
        """Cancel the live session and clear the registry; returns its terminal snapshot.

        None when no session is live. Idempotent: a second call finds nothing.
        """
        with self._lock:
            session = self._session
            if session is None:
                return None
            session.cancel()
            snapshot = session.snapshot()
            self._session = None
            return snapshot

    def finish_completed(self) -> PreviewSessionSnapshot | None:
        """Clear the registry when the live session exhausted itself naturally.

        Returns its COMPLETED snapshot exactly once; a cancelled (or none) session
        yields None and leaves the registry alone.
        """
        with self._lock:
            session = self._session
            if session is None:
                return None
            snapshot = session.snapshot()
            if snapshot.state != PreviewSessionState.COMPLETED.value:
                return None
            self._session = None
            return snapshot

    def fail(self, reason: str) -> PreviewSessionSnapshot | None:
        """P15 真机修复: end the live session as FAILED and clear the registry.

        Returns its terminal snapshot exactly once; a none (or already-terminal)
        session yields None. One-shot like ``cancel`` and ``finish_completed``.
        """
        with self._lock:
            session = self._session
            if session is None:
                return None
            session.fail(reason)
            snapshot = session.snapshot()
            self._session = None
            return snapshot