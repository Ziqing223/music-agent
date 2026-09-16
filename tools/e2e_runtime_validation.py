"""P10.7: End-to-end runtime validation on the real Mac.

Runs the full integrated-runtime loop with production code over a fresh temporary store:
startup, store open + migration, agent hosting, automation ticks, status surfaces,
cross-process observability, clean shutdown, restart and durable recovery, plus the live
CoreAudio default-output read. This tier performs NO live Music.app interaction (no
osascript): real Music.app reads/pause belong to ``tools/validate_live.py``, run by the
user. Every step prints PASS/FAIL and the process exits nonzero on any failure.

Run:  PYTHONPATH=src .venv/bin/python tools/e2e_runtime_validation.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.runtime import Runtime, RuntimeConfig
from music_agent.runtime_state import RuntimePidFile
from music_agent.runtime_status import build_store_status
from music_agent.runtime_task_run_repository import RuntimeTaskRunRepository

CLIENT_ID = "agt_e2e00000-0000-4000-8000-000000000000"


def empty_model() -> dict:
    return {
        "tracks": [],
        "artists": [],
        "albums": [],
        "playlists": [],
        "playlist_memberships": [],
    }


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def run() -> int:
    try:
        return _run()
    except Exception as error:
        check("validation completed without exceptions", False, f"{type(error).__name__}: {error}")
        raise


def _run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        database_path = Path(tmp) / "music_agent_e2e.db"
        print(f"validation store: {database_path}")

        # 1. Fresh process path: store open + migrations on first touch.
        with CanonicalRepository(database_path) as repository:
            repository.save_model(empty_model())
            check("store init + migrations", repository.schema_version == CURRENT_SCHEMA_VERSION,
                  f"schema v{repository.schema_version}")

        # 2. Runtime startup: full production composition (agent service, automation,
        #    audio-safety monitor with the REAL CoreAudio reader).
        config = RuntimeConfig(
            database_path=database_path,
            agent_clients={CLIENT_ID: "full"},
            refresh_interval_seconds=60,
            capability_status_interval_seconds=120,
            audio_safety_enabled=True,
            audio_safety_poll_interval_seconds=1.0,
        )
        runtime = Runtime(config)
        runtime.start()
        check("runtime startup", runtime.started,
              f"store={database_path.name} schema=v{runtime.schema_version}")
        check("agent service hosted", runtime.agent_service is not None)
        check("automation wired", runtime.automation.task_names == ("music_refresh", "capability_status"))
        check("audio safety wired (real CoreAudio)", runtime.audio_monitor is not None)

        # 3. Shared agent reads through the hosted service (P09 surface).
        client = AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="e2e-validation", label="p10-e2e"),
            runtime.agent_service,
        )
        capabilities = client.call("get_agent_capabilities", {})
        check("agent capabilities read",
              capabilities.outcome.value == "ok"
              and capabilities.payload["schema_version"] == CURRENT_SCHEMA_VERSION,
              f"outcome={capabilities.outcome.value} "
              f"schema=v{capabilities.payload['schema_version']} "
              f"tools={len(capabilities.payload['tools'])} "
              f"writes_execution_ready={sum(1 for w in capabilities.payload['writes'] if w['execution_ready'])}")

        # 4. Automation runs without manual orchestration and records durable runs.
        reports = runtime.automation.run_all_now()
        check("automation tasks run",
              all(r.status.value == "completed" for r in reports),
              ", ".join(f"{r.task_name}={r.status.value}" for r in reports))

        # 5. In-process status snapshot.
        snapshot = runtime.status_snapshot()
        check("status snapshot",
              snapshot["schema_version"] == CURRENT_SCHEMA_VERSION
              and snapshot["tasks"]["music_refresh"]["status"] == "completed",
              f"schema=v{snapshot['schema_version']} "
              f"audio_safety_enabled={snapshot['audio_safety']['enabled']}")

        # 6. Cross-process observability: a separate process sees 'running'.
        env = dict(os.environ, PYTHONPATH="src")
        status_proc = subprocess.run(
            [sys.executable, "-m", "music_agent", "status", "--db", str(database_path)],
            capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1],
        )
        status_text = status_proc.stdout
        check("status CLI (running process)",
              status_proc.returncode == 0
              and f"running: yes (pid {os.getpid()}" in status_text,
              status_text.splitlines()[2] if len(status_text.splitlines()) > 2 else status_text)

        # 7. Live CoreAudio read through the production reader (this Mac).
        from music_agent.audio_safety import CoreAudioDefaultOutputReader

        device_state = CoreAudioDefaultOutputReader().read_default_output()
        check("live CoreAudio default output read", device_state is not None,
              f"{device_state}")
        if device_state is not None:
            monitor_event = runtime.audio_monitor.poll_once()
            check("audio monitor poll (real device)", monitor_event.kind.value in ("no_action", "unavailable"),
                  f"kind={monitor_event.kind.value}")

        # 8. Clean shutdown clears process state.
        runtime.close()
        check("clean shutdown", not runtime.started and runtime.closed)
        check("pid file cleared on shutdown", RuntimePidFile(database_path).read() is None)

        status_proc = subprocess.run(
            [sys.executable, "-m", "music_agent", "status", "--db", str(database_path)],
            capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1],
        )
        check("status CLI (stopped process)",
              status_proc.returncode == 0 and "running: no" in status_proc.stdout,
              "running: no" if "running: no" in status_proc.stdout else status_proc.stdout)

        # 9. Restart recovers the same durable state.
        second = Runtime(config)
        second.start()
        try:
            with RuntimeTaskRunRepository(database_path) as runs:
                recovered = len(runs.list_runs("music_refresh"))
            check("restart recovers durable task history", recovered >= 1,
                  f"{recovered} music_refresh run(s) in the journal")
            recovered_status = build_store_status(database_path)
            check("restart recovers store status",
                  recovered_status["tasks"]["music_refresh"]["last_run"]["status"] == "completed")
            second.automation.run_all_now()
            with RuntimeTaskRunRepository(database_path) as runs:
                after = len(runs.list_runs("music_refresh"))
            check("restarted runtime appends (no duplicate application)", after == recovered + 1,
                  f"runs {recovered} -> {after}")
        finally:
            second.close()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\nE2E validation: {len(RESULTS) - len(failed)}/{len(RESULTS)} steps passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
