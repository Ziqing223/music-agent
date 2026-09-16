"""P10.17a: Exact Apple Music genre read (feature enrichment only, no identity).

Reads ONE track's ``genre`` as an exact source string through a bounded read-only
AppleScript. Genre is a category feature: same string = same category by definition, so
exact-key grouping is identity-free. No Artist/Album entities, no name-derived identity,
no fuzzy matching -- the persistent ID selects the track, the genre string is the fact.
"""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


class GenreReadError(RuntimeError):
    code = "genre_read_error"


GENRE_READ_SCRIPT = (
    'tell application "Music"\n'
    '    set matchingTracks to every track of library playlist 1 '
    "whose persistent ID is targetTrackID\n"
    '    if (count of matchingTracks) is 0 then error "track not found"\n'
    "    return genre of (first item of matchingTracks) as text\n"
    "end tell\n"
)


@runtime_checkable
class GenreReadRunner(Protocol):
    """Injected boundary: one read-only genre read for one persistent ID."""

    def read_genre(self, persistent_id: str) -> str: ...


class OsascriptGenreReadRunner:
    """Production runner: one osascript subprocess per genre read, bounded timeout."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise GenreReadError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def read_genre(self, persistent_id: str) -> str:
        if not isinstance(persistent_id, str) or persistent_id == "":
            raise GenreReadError("persistent_id must be a non-empty string")
        script = GENRE_READ_SCRIPT.replace("targetTrackID", f'"{persistent_id}"', 1)
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GenreReadError(str(error)) from error
        if completed.returncode != 0:
            raise GenreReadError(completed.stderr.strip() or "genre read failed")
        return completed.stdout


class AppleMusicGenreReadAdapter:
    """Parses the genre read into an exact source string (empty -> None)."""

    def __init__(self, runner: GenreReadRunner) -> None:
        if not isinstance(runner, GenreReadRunner):
            raise GenreReadError("runner must implement GenreReadRunner")
        self.runner = runner

    def read_genre(self, persistent_id: str) -> str | None:
        raw = self.runner.read_genre(persistent_id)
        if not isinstance(raw, str):
            raise GenreReadError("genre output must be a string")
        genre = raw.strip()
        return genre if genre else None
