"""Open one canonical Track in the macOS Music.app client, fail closed.

The durable identity stays the track's ``itunes_store`` binding. The service still
re-resolves Apple's official ``trackViewUrl`` at action time and the model never
constructs any URL. This boundary validates that resolved Apple URL, derives only a
client handoff URL for the same numeric track id and storefront, and asks LaunchServices
to hand it to Music.app explicitly.

A successful return means the OS accepted the handoff request. It does *not* claim that
Music.app kept a row highlighted, that a page remained selected, or that any playback
started. Nothing here touches the library, playback, or repository state.
"""

from __future__ import annotations

import re
import subprocess
import urllib.parse

APPLE_MUSIC_HOST_SUFFIXES = ("music.apple.com", "itunes.apple.com")
_STOREFRONT_RE = re.compile(r"^[A-Za-z]{2}$")
_ITUNES_TRACK_ID_RE = re.compile(r"^[0-9]+$")


class AppleMusicOpenError(ValueError):
    code = "apple_music_open_failed"


class AppleMusicOpenUnavailableError(AppleMusicOpenError):
    code = "apple_music_open_unavailable"


def _validated_apple_url(url: str) -> urllib.parse.SplitResult:
    if not isinstance(url, str) or not url.strip():
        raise AppleMusicOpenError("url must be a non-empty string")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise AppleMusicOpenError(
            f"refusing to open a non-http(s) URL: {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if not any(
        host == suffix or host.endswith("." + suffix)
        for suffix in APPLE_MUSIC_HOST_SUFFIXES
    ):
        raise AppleMusicOpenError(
            f"refusing to open a non-Apple Music URL: {parsed.hostname!r}"
        )
    return parsed


def client_song_url(track_view_url: str, itunes_track_id: str) -> str:
    """Derive a song-level Music.app handoff URL from Apple-owned identity facts."""
    parsed = _validated_apple_url(track_view_url)
    if not isinstance(itunes_track_id, str) or not _ITUNES_TRACK_ID_RE.fullmatch(
        itunes_track_id
    ):
        raise AppleMusicOpenError("itunes track id must be a non-empty numeric string")
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts or not _STOREFRONT_RE.fullmatch(path_parts[0]):
        raise AppleMusicOpenError(
            "Apple Music trackViewUrl does not expose a two-letter storefront"
        )
    storefront = path_parts[0].lower()
    return f"https://music.apple.com/{storefront}/song/{itunes_track_id}"


def open_music_app(track_view_url: str, itunes_track_id: str) -> str:
    """Ask macOS to hand one concrete Apple Music song to Music.app.

    Returns the validated song-level handoff URL after ``/usr/bin/open`` accepts the
    request. This is command acceptance, not UI/readback verification inside Music.app.
    """
    client_url = client_song_url(track_view_url, itunes_track_id)
    try:
        subprocess.run(
            ["/usr/bin/open", "-a", "Music", client_url],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise AppleMusicOpenError(
            f"Music.app handoff request failed: {error}"
        ) from error
    return client_url
