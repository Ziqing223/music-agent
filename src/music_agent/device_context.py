"""P15-S2: Device Context -- the policy-free domain for observed audio output facts.

Round 1 delivers the skeleton only: what a snapshot *is* and how a transition
*between two snapshots* reads. The live probe (``tools/diagnostics``) produces
JSONL events; this module stays a pure Python structure with no dependency on
that tool and no import of the playback machinery.

Deliberate boundaries (this constitution may only grow with P15-S2 rounds):

- **Observation only.** Snapshots record raw facts (ids, names, transport codes,
  data- source ids, alive flags). Nothing here pauses, switches, resumes, or
  otherwise mutates audio state -- and nothing here is persisted.
- **No interpretation.** A transport type of ``bltn``/``blue`` or a device name
  like "AirPods Pro" is a hardware fact, not a conclusion: the skeleton has no
  notion of headphones, personal devices, or Bluetooth categories, and must
  never grow one here (raw facts only; interpretation, if it ever exists,
  belongs to a later policy layer).
- **Unsupported is explicit.** A property a device does not support or a fact
  that could not be read is a recorded observation: the value stays ``None``
  and the field name is listed in ``unavailable_fields``. It is never a
  system failure and never guessed.
- **No policy, no manager.** Safety Pause, automatic resume, reconnection
  behavior, device preference -- all out of scope. This module defines data
  shapes, which is why it has no registry, no listeners, and no state at all.
- **P15-S1 untouched.** Suspension/preview semantics (restore-by-intent,
  suspend-don't-overlap) live in ``playback_context`` / ``agent_service`` and
  are in no way redefined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AudioOutputEventType(StrEnum):
    """The closed vocabulary of observed audio-output events (probe-emitted)."""

    BASELINE = "baseline"
    DEFAULT_DEVICE_CHANGED = "default_device_changed"
    DEVICE_LIST_CHANGED = "device_list_changed"
    DATA_SOURCE_CHANGED = "data_source_changed"
    ALIVE_CHANGED = "alive_changed"


# The optional facts one snapshot may carry. ``unavailable_fields`` may only
# name these -- anything else is a contract typo and fails closed.
_SNAPSHOT_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "device_uid",
        "device_name",
        "transport_type",
        "data_source_id",
        "data_source_name",
        "device_alive",
    }
)


@dataclass(frozen=True, slots=True)
class AudioOutputSnapshot:
    """One point-in-time observation of the real audio output state.

    Only ``event_type`` and ``timestamp`` are required; every device fact is
    optional and ``None`` until observed. ``device_id`` is the decimal
    serialization of the system AudioDeviceID, kept a string so no boundary
    presumes integer precision. A field in ``unavailable_fields`` marks a fact
    the device does not support (or the probe could not read) -- explicitly
    recorded, never a failure, never inferred.
    """

    event_type: AudioOutputEventType
    timestamp: str
    device_id: str | None = None
    device_uid: str | None = None
    device_name: str | None = None
    transport_type: str | None = None
    data_source_id: str | None = None
    data_source_name: str | None = None
    device_alive: bool | None = None
    unavailable_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.event_type, str):
            object.__setattr__(
                self, "event_type", AudioOutputEventType(self.event_type)
            )
        unknown = set(self.unavailable_fields) - _SNAPSHOT_OPTIONAL_FIELDS
        if unknown:
            raise ValueError(
                f"unavailable_fields names unknown facts: {sorted(unknown)}"
            )

    def as_dict(self) -> dict[str, Any]:
        """The raw observed surface, verbatim -- no derived fields are exposed."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "device_uid": self.device_uid,
            "device_name": self.device_name,
            "transport_type": self.transport_type,
            "data_source_id": self.data_source_id,
            "data_source_name": self.data_source_name,
            "device_alive": self.device_alive,
            "unavailable_fields": sorted(self.unavailable_fields),
        }


@dataclass(frozen=True, slots=True)
class AudioOutputTransition:
    """One observed change: the snapshot before, the snapshot after, and why.

    The device identity is compared on ``device_id`` alone (the skeleton keeps
    no identity registry). ``same_device`` demands *known and equal* ids --
    unknown ids never claim sameness. ``changed_device`` is the strict
    inequality (known → unknown counts as a change; unknown → unknown names
    neither). No further judgment -- reason is carried verbatim.
    """

    before: AudioOutputSnapshot
    after: AudioOutputSnapshot
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must name the observed event, non-empty")

    @property
    def same_device(self) -> bool:
        """True only when both sides know the same device id."""
        return (
            self.before.device_id is not None
            and self.before.device_id == self.after.device_id
        )

    @property
    def changed_device(self) -> bool:
        """True when the ids differ or either side goes unknown."""
        return self.before.device_id != self.after.device_id