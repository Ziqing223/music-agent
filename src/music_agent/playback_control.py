"""P10.5: Transient Music.app playback control -- player-state read and pause.

This is a deliberately SEPARATE surface from the sealed library-write capability matrix
(``write_intent.py``): pause is a transient, reversible playback mutation, not a durable
library write, and it carries its own fail-closed readiness rules:

- ``pause`` is issued only by the audio-safety monitor (P10.5), and only after that monitor
  proved its preconditions (an observed private-output -> built-in-speaker transition while
  the player state reads ``playing``). Nothing in this module decides WHEN to pause.
- A failed or ambiguous player-state read surfaces as ``PlayerState.UNKNOWN``; a pause is
  never issued from an unknown state (the caller's decision).
- No resume capability exists anywhere in this module (no auto-resume, by design).
"""

from __future__ import annotations

import subprocess
import time
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class PlaybackControlError(RuntimeError):
    code = "playback_control_error"


class PlaybackControlUnavailableError(PlaybackControlError):
    """Music.app could not be reached; no playback control is possible."""

    code = "playback_control_unavailable"


class PlayerState(StrEnum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@runtime_checkable
class PlaybackCommandRunner(Protocol):
    """Injected boundary: transient Music.app playback commands (production: osascript).

    Transient playback control only -- never favorites/ratings/playlists/library
    membership. ``pause`` doubles as the audio-safety monitor's fail-safe command.
    """

    def read_player_state(self) -> str: ...

    def read_now_playing(self) -> str: ...

    def pause(self) -> None: ...

    def play(self) -> None: ...

    def next_track(self) -> None: ...

    def previous_track(self) -> None: ...

    def play_track(self, persistent_id: str) -> None: ...


PLAYER_STATE_SCRIPT = (
    'tell application "Music"\n'
    "    return player state as text\n"
    "end tell\n"
)

NOW_PLAYING_SCRIPT = (
    'tell application "Music"\n'
    "    try\n"
    "        set currentTrack to current track\n"
    "        set AppleScript's text item delimiters to tab\n"
    "        return ((persistent ID of currentTrack as text) & tab & "
    "(name of currentTrack as text) & tab & (artist of currentTrack as text) & tab & "
    "(album of currentTrack as text) & tab & (player state as text))\n"
    "    on error\n"
    "        return (player state as text)\n"
    "    end try\n"
    "end tell\n"
)

PAUSE_SCRIPT = 'tell application "Music"\n    pause\nend tell\n'

PLAY_SCRIPT = 'tell application "Music"\n    play\nend tell\n'

NEXT_TRACK_SCRIPT = 'tell application "Music"\n    next track\nend tell\n'

PREVIOUS_TRACK_SCRIPT = 'tell application "Music"\n    previous track\nend tell\n'

PLAY_TRACK_SCRIPT = (
    'tell application "Music"\n'
    '    set matchingTracks to every track of library playlist 1 '
    "whose persistent ID is targetTrackID\n"
    '    if (count of matchingTracks) is 0 then error "track not found"\n'
    "    play (first item of matchingTracks)\n"
    "end tell\n"
)

LIBRARY_RESOLVE_SCRIPT = (
    'on run argv\n'
    '    if (count of argv) is not 1 then error "one target name is required"\n'
    '    set targetName to item 1 of argv\n'
    '    set out to ""\n'
    '    tell application "Music"\n'
    '        set matches to every track of library playlist 1 whose name is targetName\n'
    '        repeat with t in matches\n'
    '            set tname to ""\n'
    '            set tartist to ""\n'
    '            set talbum to ""\n'
    '            set tduration to ""\n'
    '            try\n'
    '                set tname to name of t as text\n'
    '            end try\n'
    '            try\n'
    '                set tartist to artist of t as text\n'
    '            end try\n'
    '            try\n'
    '                set talbum to album of t as text\n'
    '            end try\n'
    '            try\n'
    '                set tduration to duration of t as text\n'
    '            end try\n'
    '            try\n'
    '                set out to out & ((persistent ID of t as text) & tab & tname & tab & '
    "tartist & tab & talbum & tab & tduration & linefeed)\n"
    '            end try\n'
    '        end repeat\n'
    '    end tell\n'
    '    return out\n'
    'end run\n'
)


class OsascriptPlaybackRunner:
    """Production playback runner: one osascript subprocess per command, bounded timeout."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise PlaybackControlError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def read_player_state(self) -> str:
        return self._run(PLAYER_STATE_SCRIPT).strip()

    def read_now_playing(self) -> str:
        return self._run(NOW_PLAYING_SCRIPT).strip()

    def pause(self) -> None:
        self._run(PAUSE_SCRIPT)

    def play(self) -> None:
        self._run(PLAY_SCRIPT)

    def next_track(self) -> None:
        self._run(NEXT_TRACK_SCRIPT)

    def previous_track(self) -> None:
        self._run(PREVIOUS_TRACK_SCRIPT)

    def play_track(self, persistent_id: str) -> None:
        if not isinstance(persistent_id, str) or persistent_id == "":
            raise PlaybackControlError("persistent_id must be a non-empty string")
        script = PLAY_TRACK_SCRIPT.replace(
            "targetTrackID", f'"{persistent_id}"', 1
        )
        self._run(script)

    def _run(self, script: str) -> str:
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PlaybackControlUnavailableError(str(error)) from error
        if completed.returncode != 0:
            raise PlaybackControlUnavailableError(completed.stderr.strip() or "osascript failed")
        return completed.stdout


@dataclass(frozen=True, slots=True)
class NowPlaying:
    """One read-only snapshot of the current playback context.

    ``persistent_id`` is the identity authority; name/artist/album are informational
    metadata only (never used for identity inference). All optional fields are None
    when no current track exists.
    """

    state: PlayerState
    persistent_id: str | None = None
    name: str | None = None
    artist: str | None = None
    album: str | None = None


@runtime_checkable
class LibraryResolveRunner(Protocol):
    """Injected boundary: one read-only library candidate query (production: osascript)."""

    def resolve_candidates(self, name: str) -> str: ...


class OsascriptLibraryResolveRunner:
    """Production resolver runner: one osascript subprocess, bounded timeout.

    The target name rides ``--`` argv, never the script text (no shell interpolation).
    """

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise PlaybackControlError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def resolve_candidates(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise PlaybackControlError("name must be a non-empty string")
        try:
            completed = subprocess.run(
                ["osascript", "-e", LIBRARY_RESOLVE_SCRIPT, "--", name],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PlaybackControlUnavailableError(str(error)) from error
        if completed.returncode != 0:
            raise PlaybackControlUnavailableError(completed.stderr.strip() or "osascript failed")
        return completed.stdout


@dataclass(frozen=True, slots=True)
class LibraryTrackCandidate:
    """One read-only Music.app library candidate (presentation + identity fields)."""

    persistent_id: str
    name: str
    artist: str
    album: str | None
    duration_seconds: float | None


def _normalize_text(value: str) -> str:
    """Unicode NFC, trimmed, whitespace-collapsed, casefolded comparison form."""
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


def _nonempty(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_library_candidates(raw: str) -> tuple[LibraryTrackCandidate, ...]:
    """Parse the resolver script output; malformed lines fail closed, never guessed.

    A trailing empty duration field (Music.app returned nothing) must survive as
    ``None`` -- the evidence gate excludes the candidate -- not be misread as a
    malformed row. Lines with fewer than five fields are malformed.
    """
    candidates: list[LibraryTrackCandidate] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise PlaybackControlUnavailableError(
                f"malformed library resolve output ({len(fields)} fields)"
            )
        persistent_id, name, artist, album, duration = fields
        if not persistent_id:
            raise PlaybackControlUnavailableError("library resolve output lacks a persistent ID")
        try:
            seconds = float(duration.strip()) if duration.strip() else None
        except ValueError:
            seconds = None
        candidates.append(
            LibraryTrackCandidate(persistent_id, name, artist, album or None, seconds)
        )
    return tuple(candidates)


class MusicLibraryResolver:
    """Read-only playback-equivalent resolution (P12).

    Primary evidence gate: exact normalized title, exact normalized artist, duration
    within 1.0 s. Album is release/container metadata, never a gate: exactly one primary
    match resolves regardless of album; with several primary matches, a non-empty target
    album acts as tie-break (exact normalized equality) and exactly one album match wins.
    Anything else (zero, multiple with no unique album, missing target evidence,
    malformed or unavailable Music.app output) fails closed. Playback resolution only:
    nothing here persists, binds, or creates an identity fact.
    """

    def __init__(self, runner: LibraryResolveRunner) -> None:
        if not callable(getattr(runner, "resolve_candidates", None)):
            raise PlaybackControlError("runner must implement LibraryResolveRunner")
        self.runner = runner

    def resolve_playback_track(
        self,
        *,
        name: str | None,
        artist: str | None,
        album: str | None,
        duration_ms: int | None,
    ) -> str | None:
        if not _nonempty(name) or not _nonempty(artist) or duration_ms is None:
            return None
        candidates = _parse_library_candidates(self.runner.resolve_candidates(name))
        target_name = _normalize_text(name)
        target_artist = _normalize_text(artist)
        target_album = _normalize_text(album) if _nonempty(album) else None
        target_seconds = duration_ms / 1000.0
        matches = [
            candidate
            for candidate in candidates
            if _is_playback_equivalent(candidate, target_name, target_artist, target_seconds)
        ]
        if len(matches) == 1:
            return matches[0].persistent_id
        if len(matches) > 1 and target_album is not None:
            album_matches = [
                candidate
                for candidate in matches
                if _nonempty(candidate.album)
                and _normalize_text(candidate.album) == target_album
            ]
            if len(album_matches) == 1:
                return album_matches[0].persistent_id
        return None


def _is_playback_equivalent(
    candidate: LibraryTrackCandidate,
    target_name: str,
    target_artist: str,
    target_seconds: float,
) -> bool:
    if _normalize_text(candidate.name) != target_name:
        return False
    if _normalize_text(candidate.artist) != target_artist:
        return False
    if candidate.duration_seconds is None:
        return False
    if abs(candidate.duration_seconds - target_seconds) > 1.0:
        return False
    return True


def _require_known_state(raw: str) -> PlayerState:
    name = raw.strip().lower()
    if name in MusicPlaybackAdapter._STATE_NAMES:
        return PlayerState(name)
    raise PlaybackControlUnavailableError(f"unknown player state: {raw!r}")


class MusicPlaybackAdapter:
    """Maps raw runner output to the typed player-state vocabulary, fail closed.

    Latency instrumentation (P10 daily-use performance item): every command records
    its wall-clock elapsed milliseconds in ``last_command_elapsed_ms`` (monotonic
    clock; observational only, never changes semantics).
    """

    _STATE_NAMES = frozenset({"playing", "paused", "stopped"})

    def __init__(self, runner: PlaybackCommandRunner) -> None:
        if not isinstance(runner, PlaybackCommandRunner):
            raise PlaybackControlError("runner must implement PlaybackCommandRunner")
        self.runner = runner
        self.last_command_elapsed_ms: float | None = None

    def read_player_state(self) -> PlayerState:
        """Read the current player state; any failure maps to UNKNOWN (never guessed)."""
        try:
            raw = self._timed(lambda: self.runner.read_player_state())
        except PlaybackControlError as error:
            raise PlaybackControlUnavailableError(str(error)) from error
        if not isinstance(raw, str):
            return PlayerState.UNKNOWN
        name = raw.strip().lower()
        if name not in self._STATE_NAMES:
            return PlayerState.UNKNOWN
        return PlayerState(name)

    def read_now_playing(self) -> NowPlaying:
        """Typed now-playing context, parsed strictly from the runner's raw text.

        The production script returns tab-delimited fields (persistent ID, name,
        artist, album, state) or just the state when nothing is playing. Any other
        shape or an unrecognized state fails closed -- never guessed.
        """
        try:
            raw = self._timed(lambda: self.runner.read_now_playing())
        except PlaybackControlError as error:
            raise PlaybackControlUnavailableError(str(error)) from error
        if not isinstance(raw, str):
            raise PlaybackControlUnavailableError("now-playing output must be a string")
        fields = raw.strip().split("\t")
        if len(fields) == 1:
            return NowPlaying(state=_require_known_state(fields[0]))
        if len(fields) == 5:
            return NowPlaying(
                state=_require_known_state(fields[4]),
                persistent_id=fields[0].strip() or None,
                name=fields[1].strip() or None,
                artist=fields[2].strip() or None,
                album=fields[3].strip() or None,
            )
        raise PlaybackControlUnavailableError(
            f"unexpected now-playing output shape ({len(fields)} fields)"
        )

    def pause(self) -> None:
        """Issue pause; any failure raises (the caller owns retry/bounding)."""
        self._timed(self.runner.pause)

    def play(self) -> None:
        """Start/resume playback. User-triggered only: no automatic path calls this."""
        self._timed(self.runner.play)

    def next_track(self) -> None:
        self._timed(self.runner.next_track)

    def previous_track(self) -> None:
        self._timed(self.runner.previous_track)

    def play_track(self, persistent_id: str) -> None:
        """Play one specific track by its Apple Music persistent ID (never by name)."""
        self._timed(lambda: self.runner.play_track(persistent_id))

    def _timed(self, invoke):
        started = time.monotonic()
        try:
            return invoke()
        finally:
            self.last_command_elapsed_ms = (time.monotonic() - started) * 1000.0
