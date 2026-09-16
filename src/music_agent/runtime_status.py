"""P10.6: Offline runtime status builder -- durable truth only.

The ``status`` CLI runs in its own process, so it reports exactly what the durable store
and the process-state file can prove: schema, task-run history, capability projection, and
running/stopped/stale process state. In-memory facts (e.g. the audio-safety monitor's last
event) live only in the running process and are reported by ``Runtime.status_snapshot()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from music_agent.agent_permission import write_capability_summary
from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.runtime_state import RuntimePidFile, RuntimeStateError
from music_agent.runtime_task_run_repository import RuntimeTaskRunRepository


class StoreStatusError(ValueError):
    code = "store_status_error"


def build_store_status(database_path: str | Path) -> dict[str, Any]:
    """Build the durable status snapshot for one store (opens repositories read-mostly)."""
    database_path = Path(database_path)
    try:
        with CanonicalRepository(database_path) as canonical:
            schema_version = canonical.schema_version
    except Exception as error:
        raise StoreStatusError(f"could not open store: {error}") from error

    tasks: dict[str, Any] = {}
    try:
        with RuntimeTaskRunRepository(database_path) as runs:
            for task_name in ("music_refresh", "capability_status", "library_discovery"):
                latest = runs.latest_run(task_name)
                if latest is None:
                    tasks[task_name] = {"runs": 0, "last_run": None}
                    continue
                tasks[task_name] = {
                    "runs": len(runs.list_runs(task_name, limit=1000)),
                    "last_run": {
                        "run_id": latest.run_id,
                        "status": latest.status.value,
                        "error": latest.error,
                        "detail": dict(latest.detail),
                        "started_at": latest.started_at,
                        "finished_at": latest.finished_at,
                    },
                }
    except Exception as error:
        raise StoreStatusError(f"could not read task-run journal: {error}") from error

    pid_file = RuntimePidFile(database_path)
    running = pid_file.running_state()
    stale = pid_file.stale_state()
    if running is not None:
        process = {"state": "running", "pid": running.pid, "started_at": running.started_at}
    elif stale is not None:
        process = {"state": "stale", "pid": stale.pid, "started_at": stale.started_at}
    else:
        process = {"state": "stopped"}

    summary = write_capability_summary()
    return {
        "store": str(database_path),
        "schema_version": schema_version,
        "expected_schema_version": CURRENT_SCHEMA_VERSION,
        "process": process,
        "tasks": tasks,
        "write_capabilities": {
            "operations": len(summary),
            "execution_ready": sum(1 for entry in summary if entry["execution_ready"]),
        },
    }


def format_store_status(status: dict[str, Any]) -> str:
    """Render the status snapshot as stable human-readable text."""
    lines = [
        f"store: {status['store']}",
        f"schema_version: {status['schema_version']} "
        f"(expected {status['expected_schema_version']})",
    ]
    process = status["process"]
    if process["state"] == "running":
        lines.append(f"running: yes (pid {process['pid']}, started {process['started_at']})")
    elif process["state"] == "stale":
        lines.append(
            f"running: no (stale state file: pid {process['pid']} is gone; "
            f"last started {process['started_at']})"
        )
    else:
        lines.append("running: no")
    for task_name, task in status["tasks"].items():
        if task["last_run"] is None:
            lines.append(f"task {task_name}: no runs")
            continue
        last = task["last_run"]
        detail = json.dumps(last["detail"], sort_keys=True)
        if last["error"]:
            lines.append(
                f"task {task_name}: {last['status']} at {last['finished_at']} "
                f"({task['runs']} runs) error={last['error']}"
            )
        else:
            lines.append(
                f"task {task_name}: {last['status']} at {last['finished_at']} "
                f"({task['runs']} runs) detail={detail}"
            )
    capabilities = status["write_capabilities"]
    lines.append(
        f"write_capabilities: {capabilities['execution_ready']}/"
        f"{capabilities['operations']} execution_ready"
    )
    return "\n".join(lines)
