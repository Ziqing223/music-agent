"""P10.11: Read-only Apple Music library discovery boundary (persistent IDs only).

Enumerates the REAL Music.app library's track persistent IDs -- the only external
identity authority -- and nothing else. Track metadata is deliberately NOT read here:
the existing production per-track adapter (:class:`~music_agent.apple_music.AppleMusicSourceAdapter`)
is the single metadata interpretation path. This module cannot modify Music.app: the
AppleScript only lists IDs; no ratings/favorites/playlists/playback are touched.

Safety: entries that are not usable persistent IDs (empty/whitespace/duplicate) are
skipped, never guessed; a runner failure raises so the caller fails closed (no scan, no
inference).
"""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


class LibraryDiscoveryError(RuntimeError):
    code = "library_discovery_error"


LIST_TRACK_IDS_SCRIPT = (
    'tell application "Music"\n'
    "    get persistent ID of every track of library playlist 1\n"
    "end tell\n"
)


@runtime_checkable
class LibraryTrackIdsRunner(Protocol):
    """Injected boundary: one read-only enumeration returning the raw osascript output."""

    def list_persistent_ids(self) -> str: ...


class OsascriptLibraryTrackIdsRunner:
    """Production runner: one osascript subprocess per enumeration, bounded timeout."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise LibraryDiscoveryError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def list_persistent_ids(self) -> str:
        try:
            completed = subprocess.run(
                ["osascript", "-e", LIST_TRACK_IDS_SCRIPT],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LibraryDiscoveryError(str(error)) from error
        if completed.returncode != 0:
            raise LibraryDiscoveryError(
                completed.stderr.strip() or "library enumeration failed"
            )
        return completed.stdout


class AppleMusicLibraryDiscoveryAdapter:
    """Parses the enumeration output into validated, deduplicated persistent IDs."""

    def __init__(self, runner: LibraryTrackIdsRunner) -> None:
        if not isinstance(runner, LibraryTrackIdsRunner):
            raise LibraryDiscoveryError("runner must implement LibraryTrackIdsRunner")
        self.runner = runner

    def list_persistent_ids(self) -> tuple[str, ...]:
        """Enumerate the library's persistent IDs (ordered, deduplicated, validated)."""
        raw = self.runner.list_persistent_ids()
        if not isinstance(raw, str):
            raise LibraryDiscoveryError("enumeration output must be a string")
        seen: set[str] = set()
        identifiers: list[str] = []
        for entry in raw.split(","):
            persistent_id = entry.strip()
            if not persistent_id or persistent_id in seen:
                continue  # unusable/duplicate entries are skipped, never guessed
            seen.add(persistent_id)
            identifiers.append(persistent_id)
        return tuple(identifiers)
