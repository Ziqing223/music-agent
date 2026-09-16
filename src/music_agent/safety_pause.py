"""P15-S2 round 2: Transition Classifier + Safety Pause Policy.

Round 1 delivered the observation surface (``device_context``) and the
read-only Core Audio probe. This module adds the first interpretation layer:
*one* safety classification and the deterministic pause decision built on it.
Deliberately NOT here: a device classification system and any listener
lifecycle (the runtime's r3 polling observer is the event pump; a native
listener would be a separate slice). As of r3 this service entry is the single
safety authority: the former P10.5 ``AudioSafetyMonitor`` (own transport
heuristic + own pause path) has been converted into that pump-level observer.

Constitution (the round-1 domain boundaries are inherited, not widened):

- **One claim only.** The classifier recognizes exactly
  ``OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER``: a known-non-builtin default output
  becoming the built-in speaker. Everything else -- builtin -> external,
  builtin -> builtin, external A -> external B, same device -- classifies to
  nothing. No proper device taxonomy exists here and none is invented.
- **Stable uid facts decide.** Builtin-speaker identity is the durable,
  well-known device uid (``BuiltInSpeakerDevice``, confirmed on the target
  Mac by the round-1 probe); names and transport codes are never consulted.
  Bluetooth transports and AirPods-shaped names stay raw facts ([[p15-s2-round1]]).
- **Unknown fails closed.** A transition with an unidentifiable before or
  after never classifies and never triggers behavior -- including reads that
  fail mid-transition. Only the two authoritative default-output event kinds
  (``baseline`` / ``default_device_changed``) may classify; ``device_list`` /
  ``alive`` events are fact updates and diagnostics, never decisions.
- **Deterministic, LLM-free, resume-free.** The policy is pure logic. It
  never consults a model, never resumes anything, and has no restore surface;
  restore remains the user's 继续播放 via the existing suspension record
  (restore-by-intent, untouched).
- **One effect per transition.** Duplicate Core Audio notifications of the
  same fallback produce one pause request; a later fallback after the output
  left the speaker is a new hazard and acts again. Already-paused state is
  idempotent by construction (deciding "nothing is sounding" is a no-op).
- **No second playback state.** The policy keeps only its dedup key; every
  actual pause executes through the P15-S1 unified control semantics
  (``_suspend_music_for_preview`` for formal playback, the shared
  ``_stop_sounding_preview`` for preview sessions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from music_agent.device_context import (
    AudioOutputEventType,
    AudioOutputSnapshot,
)

# The durable well-known uid of the Mac built-in speaker observed by the
# round-1 probe (device id 70, uid "BuiltInSpeakerDevice"). Identity is uid
# equality only -- no name matching, no transport guessing, no family logic.
BUILTIN_SPEAKER_DEVICE_UID = "BuiltInSpeakerDevice"

# The only snapshot kinds the classifier accepts: authoritative default-output
# facts. ``device_list_changed`` / ``alive_changed`` observations update facts
# and feed diagnostics, never the safety decision.
_DEFAULT_FACT_EVENT_KINDS: frozenset[AudioOutputEventType] = frozenset(
    {
        AudioOutputEventType.BASELINE,
        AudioOutputEventType.DEFAULT_DEVICE_CHANGED,
    }
)


class AudioTransitionClassification(StrEnum):
    """The closed V1 vocabulary of actionable output transitions.

    Exactly one member -- the safety fallback. Everything else classifies to
    ``None`` (no claim), which is the fail-closed stance of the classifier.
    """

    OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER = "output_fallback_to_builtin_speaker"


@dataclass(frozen=True, slots=True)
class SafetyPauseAction:
    """One decided pause request: which controlled path is sounding, each named.

    Both flags False is never a valid action (``decide`` returns ``None``
    instead) -- an action always pauses at least one sounding path.
    """

    classification: AudioTransitionClassification
    pause_formal_playback: bool
    pause_preview: bool


def classify_output_fallback(
    before: AudioOutputSnapshot,
    after: AudioOutputSnapshot,
) -> AudioTransitionClassification | None:
    """V1 classification: known-non-builtin default output -> built-in speaker.

    Physics of the claim: the Mac's default output falls back to the built-in
    speaker when a private output disappears (the round-1 real-machine facts:
    AirPods disconnect and wired unplug both end here). The classifier does
    not care *how* it got there -- no alive/devicelist evidence is required --
    and never inspects names or transports.

    Fail closed on every uncertain side: non-authoritative event kinds,
    unidentifiable before/after uids, and same-device pairs all return
    ``None``; a ``None`` never triggers behavior.
    """
    if (
        before.event_type not in _DEFAULT_FACT_EVENT_KINDS
        or after.event_type not in _DEFAULT_FACT_EVENT_KINDS
    ):
        return None
    previous_uid = before.device_uid
    current_uid = after.device_uid
    if previous_uid is None or current_uid is None:
        return None
    if previous_uid == current_uid:
        return None
    if current_uid != BUILTIN_SPEAKER_DEVICE_UID:
        return None
    return AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER


class SafetyPausePolicy:
    """Deterministic V1 policy: classify, dedup, decide. Never resume.

    State is exactly one dedup key (the last acted transition), so repeated
    Core Audio notifications of the same fallback produce a single effect,
    while leaving the speaker (an observed known-non-speaker default) re-opens
    the way for the next genuine fallback. An unreadable ``after`` never
    re-arms on its own and never blocks a *distinct* later fallback -- the
    fail-safe direction prefers an extra idempotent pause request over a
    missed one.
    """

    def __init__(self) -> None:
        self._last_acted: tuple[str, str] | None = None

    def on_default_output_change(
        self,
        before: AudioOutputSnapshot,
        after: AudioOutputSnapshot,
    ) -> AudioTransitionClassification | None:
        """Feed one authoritative default-output transition; returns the
        classification when it is actionable, ``None`` otherwise (including
        suppressed duplicates). Also maintains the re-arm facts."""
        classification = classify_output_fallback(before, after)
        if classification is None:
            if (
                after.device_uid is not None
                and after.device_uid != BUILTIN_SPEAKER_DEVICE_UID
            ):
                self._last_acted = None
            return None
        key = (before.device_uid, after.device_uid)
        if key == self._last_acted:
            return None
        self._last_acted = key
        return classification

    def decide(
        self,
        classification: AudioTransitionClassification | None,
        *,
        formal_sounding: bool,
        preview_active: bool,
    ) -> SafetyPauseAction | None:
        """The pause decision: which sounding controlled path to pause, or none.

        Fail closed: any classification other than the V1 fallback is no
        action. With no sounding controlled playback the fallback is a no-op
        (an already-paused state is idempotent by construction). This method
        never executes anything -- execution is the caller's, exclusively
        through the existing P15-S1 control semantics.
        """
        if classification is not AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER:
            return None
        if not formal_sounding and not preview_active:
            return None
        return SafetyPauseAction(
            classification=classification,
            pause_formal_playback=formal_sounding,
            pause_preview=preview_active,
        )