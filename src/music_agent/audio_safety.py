"""P15-S2 r3: the audio-output observer -- the safety authority's event pump.

History: P10.5 shipped ``AudioSafetyMonitor``, a poller that *detected* the
default output falling back to the built-in speaker, *decided* the hazard with
a transport-based heuristic, and *paused* Music itself (with its own retry
state and its own playback decisions). Round 3 converts that monitor into the
thin observer this module now contains and reclaims detection as its only job.

The single-authority constitution (r3):

- **Detection only.** The observer reads the default output (system device id,
  transport code, uid via ctypes -- passive metadata queries, no capture, no
  microphones), turns the first read into a ``baseline`` snapshot and every
  later change into a (before, after) pair of ``default_device_changed``
  snapshots, and forwards the pair to the one decision point:
  ``agent_service.handle_default_output_transition`` (P15-S2 round 2). The
  union returned is echoed into the observer's event and never interpreted
  here.
- **No playback surface.** The observer holds no playback adapter, issues no
  pause, reads no player state, and keeps no pending-pause/retry machinery.
  Playback remains the service's exclusive surface. The P10.5 transport-based
  builtin judgement (``is_built_in_speaker``) is deleted: identity is uid-only,
  decided by the authority -- never a transport/name shortcut in the pump.
- **Fail closed, survive the authority.** An unreadable default output is an
  ``unavailable`` event and the last known good state stays the reference, so
  the next successful read still yields the transition. A failing or crashing
  authority call is recorded as ``forward_failed`` -- the observer thread never
  dies on it, and a later distinct transition forwards again. Steady polls keep
  the reference and forward nothing, so repeated Core Audio notifications of
  one change produce one forward.
- **No resume, no listener.** Nothing here resumes playback (restore-by-intent,
  untouched), and nothing here is event-driven: detection is polling at the
  runtime-configured interval, so worst-case detection latency is one poll
  interval. Immediate (sub-interval) delivery is explicitly *not* claimed -- a
  native Core Audio listener is a separate future slice.

Lifecycle: ``start()`` spawns the polling thread, ``close()`` stops and joins
it. ``poll_once()`` is the deterministic unit (tests drive it directly); the
runtime component contract (start/tick/close) is preserved unchanged.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from music_agent.device_context import AudioOutputEventType, AudioOutputSnapshot

logger = logging.getLogger("music_agent.audio_safety")


def device_safety_trace_enabled() -> bool:
    """Whether the env-gated P15-S2 device-safety diagnosis channel is on.

    The flag gates *observability only* -- including extra read-only calls
    (like the post-pause readback) that must never run on normal paths.
    """
    return os.environ.get("MUSIC_AGENT_DEVICE_SAFETY_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def device_safety_trace(message: str) -> None:
    """One line to stderr prefixed ``[device-safety]`` -- the env-gated P15-S2
    diagnosis channel for the production path.

    Prints only when the ``MUSIC_AGENT_DEVICE_SAFETY_TRACE`` environment
    variable holds a truthy value; with it unset (every normal run) this is a
    no-op. The flag changes observability only -- no control flow, no behavior,
    and nothing is ever printed to the user-facing surface.
    """
    if device_safety_trace_enabled():
        print(f"[device-safety] {message}", file=sys.stderr)

_COREAUDIO_PATH = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
_COREFOUNDATION_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"

# CoreAudio four-character selectors (network byte order, i.e. big-endian char constants).
_kSystemObject = 1
_kGlobalScope = 0x676C6F62  # 'glob'
_kDefaultOutputDevice = 0x644F7574  # 'dOut'
_kTransportType = 0x7472616E  # 'tran'
_kDeviceUID = 0x75696420  # 'uid '
_kNoError = 0

_kCFStringEncodingUTF8 = 0x08000100


class AudioSafetyError(RuntimeError):
    code = "audio_safety_error"


class AudioSafetyUnavailableError(AudioSafetyError):
    """The CoreAudio boundary could not be loaded/queried; no observation is possible."""

    code = "audio_safety_unavailable"


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class OutputDeviceState:
    """One observed default-output device snapshot (raw system facts only)."""

    device_id: int
    transport_type: str
    uid: str


class CoreAudioDefaultOutputReader:
    """Reads the system default output device through the CoreAudio C API (ctypes)."""

    def __init__(self) -> None:
        try:
            self._coreaudio = ctypes.cdll.LoadLibrary(_COREAUDIO_PATH)
            self._corefoundation = ctypes.cdll.LoadLibrary(_COREFOUNDATION_PATH)
        except OSError as error:
            raise AudioSafetyUnavailableError(f"cannot load CoreAudio: {error}") from error
        self._corefoundation.CFStringGetCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
        ]

    def read_default_output(self) -> OutputDeviceState | None:
        """Snapshot the default output device; ``None`` when it cannot be determined."""
        device_id = self._read_device_id()
        if device_id is None:
            return None
        transport = self._read_transport(device_id)
        if transport is None:
            return None
        uid = self._read_uid(device_id)
        if uid is None:
            return None
        return OutputDeviceState(device_id=device_id, transport_type=transport, uid=uid)

    def _read_device_id(self) -> int | None:
        device_id = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(device_id))
        address = _PropertyAddress(_kDefaultOutputDevice, _kGlobalScope, 0)
        result = self._coreaudio.AudioObjectGetPropertyData(
            _kSystemObject, ctypes.byref(address), 0, None, ctypes.byref(size),
            ctypes.byref(device_id),
        )
        if result != _kNoError or device_id.value == 0:
            return None
        return device_id.value

    def _read_transport(self, device_id: int) -> str | None:
        value = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(value))
        address = _PropertyAddress(_kTransportType, _kGlobalScope, 0)
        result = self._coreaudio.AudioObjectGetPropertyData(
            device_id, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)
        )
        if result != _kNoError:
            return None
        tag = _fourcc(value.value)
        if not tag:
            return None
        return tag

    def _read_uid(self, device_id: int) -> str | None:
        uid_ref = ctypes.c_void_p(0)
        size = ctypes.c_uint32(ctypes.sizeof(uid_ref))
        address = _PropertyAddress(_kDeviceUID, _kGlobalScope, 0)
        result = self._coreaudio.AudioObjectGetPropertyData(
            device_id, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(uid_ref)
        )
        if result != _kNoError or not uid_ref.value:
            return None
        buffer = ctypes.create_string_buffer(1024)
        try:
            if not self._corefoundation.CFStringGetCString(
                uid_ref.value, buffer, 1024, _kCFStringEncodingUTF8
            ):
                return None
            return buffer.value.decode("utf-8")
        except UnicodeDecodeError:
            return None
        finally:
            self._corefoundation.CFRelease(ctypes.c_void_p(uid_ref.value))


def _fourcc(value: int) -> str:
    """Decode a UInt32 FourCC tag to its 4-character string (empty on undecodable)."""
    raw = bytes(
        ((value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
    )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return ""
    return text if all(c.isprintable() for c in text) else ""


# The device facts this ctypes boundary does not read. Marked explicitly on every
# snapshot (per the device_context constitution: unobserved is recorded, never
# guessed). transport/device_id are read; uid is read; names and data sources
# are not.
_UNREADABLE_FACTS: frozenset[str] = frozenset(
    {"device_name", "data_source_id", "data_source_name", "device_alive"}
)


def _snapshot_from_state(
    state: OutputDeviceState,
    event_type: AudioOutputEventType,
    timestamp: str,
) -> AudioOutputSnapshot:
    """Translate one raw system read into the policy-free domain snapshot."""
    return AudioOutputSnapshot(
        event_type=event_type,
        timestamp=timestamp,
        device_id=str(state.device_id),
        device_uid=state.uid,
        transport_type=state.transport_type,
        unavailable_fields=_UNREADABLE_FACTS,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AudioObserverEventKind(StrEnum):
    BASELINE = "baseline"
    FORWARDED = "forwarded"
    NO_CHANGE = "no_change"
    UNAVAILABLE = "unavailable"
    FORWARD_FAILED = "forward_failed"


@dataclass(frozen=True, slots=True)
class AudioObserverEvent:
    """The observable outcome of one observer poll (status surface).

    ``action`` is the authority's decision on a forwarded pair, echoed verbatim
    -- the observer never interprets it. ``error`` is set when the authority
    raised; reading a device as unavailable is recorded as ``kind`` alone.
    """

    kind: AudioObserverEventKind
    snapshot: AudioOutputSnapshot | None
    action: object | None = None
    error: str | None = None


class AudioOutputObserver:
    """Polls the default output and forwards every change to the safety authority.

    Detection only (see the module constitution): the observer holds the reader
    and the service entry, nothing else. Lifecycle mirrors the former P10.5
    monitor so the runtime component contract is untouched: ``start()`` spawns
    the polling thread, ``close()`` stops and joins it, ``poll_once()`` is the
    deterministic unit driven directly by tests.
    """

    def __init__(
        self,
        device_reader: CoreAudioDefaultOutputReader,
        service: object,
        *,
        poll_interval_seconds: float = 2.0,
        clock: object = None,
    ) -> None:
        if not isinstance(device_reader, CoreAudioDefaultOutputReader):
            raise AudioSafetyError("device_reader must be a CoreAudioDefaultOutputReader")
        if not callable(getattr(service, "handle_default_output_transition", None)):
            raise AudioSafetyError(
                "service must expose handle_default_output_transition(before, after)"
            )
        if not isinstance(poll_interval_seconds, (int, float)) or poll_interval_seconds <= 0:
            raise AudioSafetyError("poll_interval_seconds must be positive")
        if clock is not None and not callable(clock):
            raise AudioSafetyError("clock must be callable")
        self._device_reader = device_reader
        self._service = service
        self.poll_interval_seconds = poll_interval_seconds
        self._clock = clock if clock is not None else _utc_now_iso
        self._previous_state: OutputDeviceState | None = None
        self._last_event: AudioObserverEvent | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def last_event(self) -> AudioObserverEvent | None:
        return self._last_event

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="audio-output-observer", daemon=True
        )
        self._thread.start()
        device_safety_trace(
            f"observer: poll thread started (interval {self.poll_interval_seconds:.1f}s)"
        )

    def tick(self) -> None:
        """Runtime component hook: the observer runs on its own thread, not the tick loop."""

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval_seconds + 2.0)
        self._thread = None
        device_safety_trace("observer: poll thread stopped")

    def _run_loop(self) -> None:
        logger.info(
            "audio output observer started (poll interval %.1fs)",
            self.poll_interval_seconds,
        )
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("audio output observer poll failed")
            self._stop_event.wait(timeout=self.poll_interval_seconds)
        logger.info("audio output observer stopped")

    def poll_once(self) -> AudioObserverEvent:
        """One deterministic poll: system read -> transition detection -> forward."""
        device_state = self._device_reader.read_default_output()
        if device_state is None:
            device_safety_trace(
                "observer: default output unreadable (reader returned None); "
                "previous state kept as the reference"
            )
            event = AudioObserverEvent(AudioObserverEventKind.UNAVAILABLE, None)
            self._last_event = event
            return event

        if self._previous_state is None:
            self._previous_state = device_state
            event = AudioObserverEvent(
                AudioObserverEventKind.BASELINE,
                _snapshot_from_state(
                    device_state, AudioOutputEventType.BASELINE, self._clock()
                ),
            )
            self._last_event = event
            device_safety_trace(
                f"observer: baseline uid={device_state.uid!r} "
                f"(device_id={device_state.device_id} transport={device_state.transport_type!r})"
            )
            return event

        if device_state == self._previous_state:
            event = AudioObserverEvent(
                AudioObserverEventKind.NO_CHANGE,
                _snapshot_from_state(
                    device_state, AudioOutputEventType.BASELINE, self._clock()
                ),
            )
            self._last_event = event
            return event

        now = self._clock()
        before = _snapshot_from_state(
            self._previous_state, AudioOutputEventType.DEFAULT_DEVICE_CHANGED, now
        )
        self._previous_state = device_state
        after = _snapshot_from_state(
            device_state, AudioOutputEventType.DEFAULT_DEVICE_CHANGED, now
        )
        try:
            action = self._service.handle_default_output_transition(before, after)
        except Exception as error:
            logger.exception(
                "audio output observer: forwarding to the safety authority failed"
            )
            device_safety_trace(
                f"observer: forwarding to the safety authority raised: {error!r}"
            )
            event = AudioObserverEvent(
                AudioObserverEventKind.FORWARD_FAILED, after, error=str(error)
            )
        else:
            logger.info(
                "audio output observer: default output changed "
                "(device %s -> %s); forwarded to the safety authority",
                before.device_id,
                after.device_id,
            )
            device_safety_trace(
                f"observer: transition {before.device_uid!r} -> {after.device_uid!r} "
                "forwarded to the service entry"
            )
            event = AudioObserverEvent(AudioObserverEventKind.FORWARDED, after, action)
        self._last_event = event
        return event