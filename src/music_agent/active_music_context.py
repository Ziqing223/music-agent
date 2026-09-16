"""P14-C06.1: runtime-only ActiveMusicContext register.

One in-memory, single-writer value holder for the agent's transient music-interaction
state, owned exclusively by :class:`~music_agent.agent_service.SharedAgentService`. It
replaces the service's former ad-hoc playback-state dict without changing any observable
behavior: the same three audio-action facts are held under the same names and mutated at
the same boundaries (``play_track`` -> library channel + anchor, preview -> preview
channel, ``stop_preview`` / next / previous -> channel cleared while the anchor survives).

P14-C07.3 adds two recommendation-batch facts under the same constitution:
``active_run_id`` names the recommendation run this service last delivered (recorded at
the generate boundary, only after the run is durably saved -- a pointer is never left
naming a run that does not exist), and ``active_item_index`` is the 0-based rank of the
last batch item the service acted on via ``play_track`` / ``preview_catalog_track``.
The pointer is a reference, never a copy: candidate content, scores, playback routes and
target lists always stay in recommendation history.

P21 Slice 3 adds a separate, runtime-only verified-selection projection scoped to that
active run. It records only actions whose ``ActionAttempt`` reached ``COMPLETED``; the
older ``active_item_index`` remains an action-dispatch cursor and is deliberately not
repurposed as verified truth.

P19-T14-F-R4 adds one session-local referent fact: ``referent_canonical_id`` is the
last successfully resolved explicit track target -- the track the user most recently
referred to in conversation (试听 X / 播放 X / an explicit batch-item interaction such
as 第N首, all of which resolve through the same two audio mutators). It deliberately
decouples three facts that were previously conflated into one field:

- ``channel`` -- the service's latest audio *action* (an action log; cleared on
  stop_preview / queue navigation exactly as before);
- ``persistent_id`` anchor / the Music.app snapshot -- formal *playback* truth;
- ``referent_canonical_id`` -- the conversational *target* the user just referred to.

A new explicit target replaces the previous referent. The referent survives every
channel clear by design (the user may stop a preview and still say 试听他), is never
touched by queue movement, pause/resume, or prose, and is never written by a failed or
ambiguous request -- the two mutators below are only called after the underlying audio
action has succeeded. A fresh service instance starts with no referent.

Deliberate boundaries:

- **Runtime only.** Nothing here is persisted, migrated, or replayed; there is no
  repository and no schema. A fresh service instance always starts at ``channel="none"``.
- **Single writer.** Only the service writes, through the three mutators below. There is
  no public setter and no update-dict API, so a channel transition can never be invented
  by a reader.
- **A channel is an action log, not a status snapshot.** ``library`` / ``preview``
  records the service's last audio action, exactly as before; it never claims that audio
  is currently sounding.
- **Preview-active truth is deliberately NOT mirrored here.** The sounding afplay process
  belongs to the preview runner and only the runner can report it (today through the
  ``stop_preview`` return value). Storing a copy would recreate the staleness bug this
  layer is meant to prevent; consumers that need "is a preview sounding right now" must
  ask the runner.
- **Music.app state is never stored here.** The player is read on demand through the
  playback adapter snapshot; the register records only what the service commanded.
- **Recommendation content stays in history.** The register holds the run pointer,
  action-dispatch cursor and a bounded tuple of verified item identities for that run;
  it never copies candidate content, scores or ranking state. An item index without a
  run is a forbidden state (``note_batch_item`` refuses), and a missing pointer means
  "derive from history at read time".
"""

from __future__ import annotations

from dataclasses import dataclass

CHANNEL_NONE = "none"
CHANNEL_LIBRARY = "library"
CHANNEL_PREVIEW = "preview"


class ActiveMusicContextError(ValueError):
    code = "active_music_context_error"


class ActiveMusicContextValidationError(ActiveMusicContextError):
    code = "active_music_context_validation_error"


@dataclass(frozen=True)
class VerifiedSelection:
    """One completed action fact scoped to an active RecommendationRun."""

    canonical_id: str
    item_position: int
    action_kind: str
    playback_route: str


@dataclass(frozen=True)
class ActiveMusicContextSnapshot:
    """Immutable point-in-time export of the register (P14-C06.2 observation endpoint).

    A copy, not a view: mutating the register afterwards never invalidates or changes a
    snapshot that was already taken. Readers build output strictly from this, so an
    observation is always internally consistent.
    """

    channel: str
    canonical_id: str | None
    persistent_id: str | None
    active_run_id: str | None
    active_item_index: int | None
    verified_selection_run_id: str | None
    verified_selections: tuple[VerifiedSelection, ...]
    referent_canonical_id: str | None


class ActiveMusicContext:
    """Service-owned register of transient music-interaction state (in-memory only)."""

    def __init__(self) -> None:
        self._channel: str = CHANNEL_NONE
        self._canonical_id: str | None = None
        self._persistent_id: str | None = None
        self._active_run_id: str | None = None
        self._active_item_index: int | None = None
        self._verified_selection_run_id: str | None = None
        self._verified_selections: tuple[VerifiedSelection, ...] = ()
        self._referent_canonical_id: str | None = None

    @property
    def channel(self) -> str:
        """The service's last audio action: ``"none"`` / ``"library"`` / ``"preview"``."""
        return self._channel

    @property
    def canonical_id(self) -> str | None:
        """Canonical Track of the last audio action, or ``None`` when the channel closed."""
        return self._canonical_id

    @property
    def persistent_id(self) -> str | None:
        """Last library persistent ID the service commanded via ``play_track``.

        The ownership anchor: queue navigation and preview stops clear the channel but
        deliberately keep this value, so ``get_now_playing`` can keep judging whether the
        Music.app current track is the one the agent last commanded.
        """
        return self._persistent_id

    @property
    def active_run_id(self) -> str | None:
        """The recommendation run this service last delivered, or ``None`` when none has been.

        Recorded only at the generate boundary, after the run is durably saved; batch
        content is never mirrored here (membership is resolved against recommendation
        history at read/write time).
        """
        return self._active_run_id

    @property
    def active_item_index(self) -> int | None:
        """0-based rank of the last batch item acted on (``play_track`` /
        ``preview_catalog_track``), or ``None`` when no action has hit the active run.

        Meaningful only while ``active_run_id`` is set; an index without a run is a
        forbidden state the mutators refuse to produce.
        """
        return self._active_item_index

    @property
    def verified_selection_run_id(self) -> str | None:
        """Run owning ``verified_selections``, or ``None`` in a fresh session."""
        return self._verified_selection_run_id

    @property
    def verified_selections(self) -> tuple[VerifiedSelection, ...]:
        """Immutable completed-action facts for the current run, in write order."""
        return self._verified_selections

    def verified_selections_for_run(
        self, run_id: str
    ) -> tuple[VerifiedSelection, ...]:
        """Return verified facts only when the requested run owns the projection."""
        _require_non_empty_string(run_id, "run_id")
        if self._verified_selection_run_id != run_id:
            return ()
        return self._verified_selections

    @property
    def referent_canonical_id(self) -> str | None:
        """Last successfully resolved explicit track target (P19-T14-F-R4).

        The track the user most recently referred to in conversation -- the pronoun
        referent for 试听它/播放它. Written only by ``note_library_playback`` /
        ``note_preview`` (whose call sites fire strictly after a successful audio
        action), replaced whole by each new explicit target, and deliberately *not*
        cleared by ``clear_channel`` (a preview stop keeps the target conversational).
        Runtime-only: a fresh service instance (session reset) starts with ``None``.
        """
        return self._referent_canonical_id

    def note_library_playback(self, canonical_id: str, persistent_id: str) -> None:
        """Record a completed ``play_track``: library channel plus the ownership anchor
        plus the conversational referent (an explicit play IS an explicit target)."""
        _require_non_empty_string(canonical_id, "canonical_id")
        _require_non_empty_string(persistent_id, "persistent_id")
        self._channel = CHANNEL_LIBRARY
        self._canonical_id = canonical_id
        self._persistent_id = persistent_id
        self._referent_canonical_id = canonical_id

    def note_preview(self, canonical_id: str) -> None:
        """Record a started preview: preview channel for ``canonical_id`` plus the
        conversational referent (an explicit preview IS an explicit target); anchor
        untouched."""
        _require_non_empty_string(canonical_id, "canonical_id")
        self._channel = CHANNEL_PREVIEW
        self._canonical_id = canonical_id
        self._referent_canonical_id = canonical_id

    def clear_channel(self) -> None:
        """Close the channel (stop_preview / queue navigation); the anchor and the
        conversational referent survive."""
        self._channel = CHANNEL_NONE
        self._canonical_id = None

    def note_recommendation_batch(self, run_id: str) -> None:
        """Record a delivered recommendation run: the batch pointer replaces any prior
        batch and the item index restarts unset. Callers record this only after the run
        has been durably saved, so the pointer never names a run that does not exist."""
        _require_non_empty_string(run_id, "run_id")
        if self._verified_selection_run_id != run_id:
            self._verified_selection_run_id = run_id
            self._verified_selections = ()
        self._active_run_id = run_id
        self._active_item_index = None

    def note_verified_selection(
        self,
        *,
        run_id: str,
        canonical_id: str,
        item_position: int,
        action_kind: str,
        playback_route: str,
    ) -> VerifiedSelection:
        """Record one strictly completed action under its authoritative run.

        A fresh/derived session may bind its current run from the already-authoritative
        SelectionGrant. A different active run is never overwritten: that would leak
        verified facts across run boundaries. Re-recording the same completed fact is
        idempotent; a conflicting fact for one canonical id is rejected.
        """
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(canonical_id, "canonical_id")
        _require_positive_int(item_position, "item_position")
        _require_non_empty_string(action_kind, "action_kind")
        _require_non_empty_string(playback_route, "playback_route")
        if self._active_run_id is None:
            self._active_run_id = run_id
            self._verified_selection_run_id = run_id
            self._verified_selections = ()
        elif self._active_run_id != run_id:
            raise ActiveMusicContextError(
                "verified selection run must match the active recommendation run"
            )
        if self._verified_selection_run_id != run_id:
            raise ActiveMusicContextError(
                "verified selection projection must match the active recommendation run"
            )
        selection = VerifiedSelection(
            canonical_id=canonical_id,
            item_position=item_position,
            action_kind=action_kind,
            playback_route=playback_route,
        )
        existing = next(
            (
                item
                for item in self._verified_selections
                if item.canonical_id == canonical_id
            ),
            None,
        )
        if existing is not None:
            if existing != selection:
                raise ActiveMusicContextError(
                    "canonical id already has a different verified selection fact"
                )
            return existing
        self._verified_selections = self._verified_selections + (selection,)
        return selection

    def note_batch_item(self, item_index: int) -> None:
        """Record the batch item the service last acted on; refuses without an active run.

        ``item_index`` is the 0-based rank of the item inside the run named by
        ``active_run_id`` (tuple index is rank in the recommendation contract). An index
        without a run is a forbidden state, so the mutator fails closed instead of ever
        producing ``(None, index)``.
        """
        _require_non_negative_int(item_index, "item_index")
        if self._active_run_id is None:
            raise ActiveMusicContextError(
                "an item index requires an active recommendation run"
            )
        self._active_item_index = item_index

    def clear_item_index(self) -> None:
        """Drop the batch-item position without dropping the batch pointer.

        Used when the last action did not hit the active run (``play_track`` /
        ``preview_catalog_track`` of a track outside it, or an unresolvable pointer);
        idempotent and valid in every state -- ``(None, None)`` and ``(run, None)`` are
        both legal.
        """
        self._active_item_index = None

    def snapshot(self) -> ActiveMusicContextSnapshot:
        """Return an immutable copy of the current register state (see
        :class:`ActiveMusicContextSnapshot`)."""
        return ActiveMusicContextSnapshot(
            channel=self._channel,
            canonical_id=self._canonical_id,
            persistent_id=self._persistent_id,
            active_run_id=self._active_run_id,
            active_item_index=self._active_item_index,
            verified_selection_run_id=self._verified_selection_run_id,
            verified_selections=self._verified_selections,
            referent_canonical_id=self._referent_canonical_id,
        )


def _require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ActiveMusicContextValidationError(f"{label} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActiveMusicContextValidationError(f"{label} must be a non-negative int")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActiveMusicContextValidationError(f"{label} must be a positive int")
    return value
