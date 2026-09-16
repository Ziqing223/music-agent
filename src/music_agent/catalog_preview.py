"""P11-T4: real 30-second audio preview for catalog tracks, never the library.

Preview plays one iTunes ``previewUrl`` through the smallest macOS-native playback boundary:
download the signed ~30-second clip to a temporary file, play it with ``afplay``, and delete
the file. The temp-download step is technically necessary (the iTunes m4a preview container is
not streamable by ``afplay`` -- live-verified P11-T4); the clip is never kept.

Preview is deliberately its own capability, distinct from playback (the P10.12 boundary, which
requires a Music.app persistent ID binding) and from library mutation (``catalog_library``,
which requires user-scoped authorization). Nothing here touches Music.app and no library state
changes anywhere. The preview URL is resolved by the service layer from the track's durable
``itunes_store`` binding (``itunes_search.lookup_preview_url``); this module only plays.

P3B batch 2: playback is non-blocking. ``afplay`` is spawned with ``Popen`` and
``start_audio`` returns as soon as audio starts -- the agent loop is released while the clip
sounds. One preview sounds at a time: starting a new preview stops the previous one, and
``stop_preview`` stops the active one. A daemon reaper thread removes the temp file when the
process exits (naturally or after a stop), preserving the no-media-left-behind guarantee.

P15-S1: the reaper now distinguishes the two exits (``AfplayPlayback._stopped``) and fires
the optional ``on_natural_finish`` hook only for natural ends -- the continuous-preview
auto-advance signal. Listener failures are swallowed so cleanup never breaks.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import tempfile
import threading
import urllib.parse
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class CatalogPreviewError(ValueError):
    code = "catalog_preview_failed"


class CatalogPreviewUnavailableError(CatalogPreviewError):
    code = "catalog_preview_unavailable"


class AudioPreviewRunner(Protocol):
    """Injected boundary: start one audio preview URL (production: non-blocking afplay)."""

    def start_audio(self, url: str) -> None: ...

    def stop_preview(self) -> bool: ...

    def is_preview_active(self) -> bool: ...


class AfplayPlayback:
    """One running preview playback: the afplay process plus its temp file.

    ``stop`` terminates the process (kill fallback) and removes the temp file; the reaper
    thread removes it on natural exit. Both are idempotent, so stop + natural exit racing
    is safe.
    """

    _STOP_WAIT_SECONDS = 5.0

    def __init__(self, process: subprocess.Popen, temp_path: str) -> None:
        self.process = process
        self.temp_path = temp_path
        self._stopped = False

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        process = self.process
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=self._STOP_WAIT_SECONDS)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=self._STOP_WAIT_SECONDS)
                except Exception:
                    pass
        _unlink_quietly(self.temp_path)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class AfplayPreviewRunner:
    """Production runner: temp download (bounded) -> non-blocking ``afplay`` -> reaped delete.

    The download stage stays synchronous and time-bounded (the clip must exist before audio
    starts); the playback stage is spawned and returned immediately. ``timeout_seconds``
    bounds the download and the stop-wait; the temp file is removed on every outcome
    (failure, stop, or natural exit), so no media is ever kept on disk.
    """

    _TEMP_PREFIX = "music_agent_preview_"

    def __init__(
        self, timeout_seconds: float = 60.0, *, on_natural_finish: Callable[[], None] | None = None
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise CatalogPreviewError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._current: AfplayPlayback | None = None
        self._on_natural_finish: Callable[[], None] | None = None
        self.on_natural_finish = on_natural_finish
        # Bounded safety net: a killed agent process must not leave afplay sounding or the
        # temp clip on disk. Best effort only -- the reaper owns normal cleanup.
        atexit.register(self._cleanup_at_exit)

    @property
    def on_natural_finish(self) -> Callable[[], None] | None:
        """P15-S1: the hook fired once per clip that ends *naturally* (never on stop).

        Injected by the service so a continuous-preview session can auto-advance. A
        clip ended by ``stop_preview`` / replacement never fires it -- ``_stopped``
        distinguishes the two exits.
        """
        return self._on_natural_finish

    @on_natural_finish.setter
    def on_natural_finish(self, callback: Callable[[], None] | None) -> None:
        if callback is not None and not callable(callback):
            raise CatalogPreviewError("on_natural_finish must be callable or None")
        self._on_natural_finish = callback

    def start_audio(self, url: str) -> None:
        if not isinstance(url, str) or not url:
            raise CatalogPreviewError("url must be a non-empty string")
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise CatalogPreviewError("preview url must be http(s); refusing other schemes")
        # Single audio channel: a new preview replaces the sounding one, never overlaps it.
        self.stop_preview()
        temp = tempfile.NamedTemporaryFile(
            suffix=".m4a", prefix=self._TEMP_PREFIX, delete=False
        )
        temp.close()
        process = None
        try:
            self._run(["curl", "-fsSL", "--max-time", str(self.timeout_seconds), "-o", temp.name, url])
            try:
                process = subprocess.Popen(["afplay", temp.name])
            except OSError as error:
                raise CatalogPreviewError(f"preview playback failed to start: {error}") from error
        except BaseException:
            _unlink_quietly(temp.name)
            raise
        playback = AfplayPlayback(process, temp.name)
        self._current = playback
        threading.Thread(target=self._reap, args=(playback,), daemon=True).start()

    def stop_preview(self) -> bool:
        """Stop one sounding preview; ``False`` when none was active (idempotent)."""
        playback = self._current
        if playback is None:
            return False
        self._current = None
        playback.stop()
        return True

    def is_preview_active(self) -> bool:
        """P14-C06.3b: read-only truth -- True while a preview is (still) sounding.

        Never starts, stops, or mutates anything; it only reports. The reaper
        thread owns clearing ``_current`` on natural exit, so a died-but-unreaped
        process already reports False here (``poll()`` has returned) -- the truth
        diverges from the action-log channel register exactly as intended.
        """
        playback = self._current
        return playback is not None and playback.process.poll() is None

    def _reap(self, playback: AfplayPlayback) -> None:
        """Natural-exit cleanup: wait for afplay to finish, then remove the temp file.

        P15-S1: a naturally-ended clip fires ``on_natural_finish`` *first* -- while
        ``_current`` still names it -- so the listener can start the next clip right
        away (its ``start_audio`` atomically replaces ``_current``). A stop-terminated
        clip skips the hook entirely: only natural ends advance a session.

        P17 acceptance: "natural" additionally requires a clean exit (code 0). An
        afplay that dies nonzero mid-clip is a *failure* -- the hook stays silent,
        so no session advances and no formal-playback auto-restore fires on a clip
        the user never actually heard.
        """
        exit_code = playback.process.wait()
        if not playback._stopped and exit_code == 0:
            self._notify_natural_finish()
        elif not playback._stopped:
            logger.info(
                "afplay exited with code %s mid-preview; treating as failure "
                "(no advance, no auto-restore)",
                exit_code,
            )
        if self._current is playback:
            self._current = None
        _unlink_quietly(playback.temp_path)

    def _notify_natural_finish(self) -> None:
        """Fire the natural-finish hook on this reaper thread, never letting a
        listener's failure interrupt cleanup -- the no-media-left-behind guarantee
        outranks the observer.
        """
        callback = self._on_natural_finish
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.exception("preview natural-finish listener raised; cleanup continues")

    def _cleanup_at_exit(self) -> None:
        try:
            playback = self._current
            if playback is not None:
                self._current = None
                playback.stop()
        except Exception:
            pass

    def _run(self, command: list[str]) -> None:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CatalogPreviewError(f"preview command failed: {error}") from error
        if completed.returncode != 0:
            raise CatalogPreviewError(
                completed.stderr.strip() or f"{command[0]} exited {completed.returncode}"
            )