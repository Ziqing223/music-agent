"""P10.6: Ephemeral running-process state for the runtime observability surface.

One small JSON file next to the store (``<store>.runtime.json``) records the PID and
startup instant of the currently running runtime process. This is transient process
state -- deliberately NOT durable user state, so it lives outside SQLite and outside
the migration registry. Semantics:

- ``run`` writes the file at startup (after the store opens) and removes it on clean
  shutdown. A crashed process leaves a stale file behind; the next ``run`` overwrites
  it, and ``status`` reports the stale state as ``stale`` (not running).
- ``status`` derives running/stopped from the file plus a liveness probe of the PID
  (``kill(pid, 0)``). PID reuse after a crash is possible in principle; the staleness
  path (liveness probe) makes the common crash case honest, and a reused PID reading
  ``running`` is the accepted, documented approximation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class RuntimeStateError(ValueError):
    code = "runtime_state_error"


@dataclass(frozen=True, slots=True)
class RuntimeProcessState:
    """One observed runtime-process snapshot (from the state file)."""

    pid: int
    started_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.pid, int) or self.pid <= 0:
            raise RuntimeStateError("pid must be a positive integer")
        if not isinstance(self.started_at, str) or self.started_at == "":
            raise RuntimeStateError("started_at must be a non-empty string")


class RuntimePidFile:
    """Reader/writer for the runtime process-state file (``<store>.runtime.json``)."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(f"{Path(database_path)}.runtime.json")

    def write(self, pid: int, started_at: str) -> None:
        """Record the running process state (atomic replace)."""
        if not isinstance(pid, int) or pid <= 0:
            raise RuntimeStateError("pid must be a positive integer")
        if not isinstance(started_at, str) or started_at == "":
            raise RuntimeStateError("started_at must be a non-empty string")
        payload = json.dumps({"pid": pid, "started_at": started_at}, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    def read(self) -> RuntimeProcessState | None:
        """Read the recorded state, or None when no file exists or it is malformed."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            state = RuntimeProcessState(pid=int(payload["pid"]), started_at=str(payload["started_at"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return state

    def clear(self) -> None:
        """Remove the state file (clean shutdown). Missing file is not an error."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def process_alive(self, pid: int) -> bool:
        """Liveness probe: True when the PID exists in the current process namespace."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def running_state(self) -> RuntimeProcessState | None:
        """The running process state, or None when no live runtime process is recorded."""
        state = self.read()
        if state is None or not self.process_alive(state.pid):
            return None
        return state

    def stale_state(self) -> RuntimeProcessState | None:
        """A recorded state whose process is gone (crashed runtime), or None."""
        state = self.read()
        if state is None or self.process_alive(state.pid):
            return None
        return state
