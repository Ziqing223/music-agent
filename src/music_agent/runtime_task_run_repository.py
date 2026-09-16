"""P10.4: Append-only durable journal for automation task runs (schema v16).

One immutable row per completed or failed automation run, keyed by an opaque ``rn_`` run
identity. This is runtime observability only: last-successful/last-failed refresh, per-task
status history -- never domain user state. Immutability is a hard SQLite guarantee
(trigger-enforced), matching the established repository conventions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from music_agent.repository import _open_store_connection


def generate_task_run_id() -> str:
    """Generate a stable task-run identity (``rn_`` namespace, never derived from content)."""
    return f"rn_{uuid4()}"


class TaskRunRepositoryError(ValueError):
    code = "task_run_repository_error"


class TaskRunRepositoryValidationError(TaskRunRepositoryError):
    code = "validation_error"


class TaskRunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TaskRunRecord:
    """One durable task-run row (exactly the migration 0016 shape)."""

    run_id: str
    task_name: str
    status: TaskRunStatus
    error: str | None
    detail: Mapping[str, object]
    started_at: str
    finished_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.startswith("rn_"):
            raise TaskRunRepositoryValidationError("run_id must be an rn_ identity")
        if not isinstance(self.task_name, str) or self.task_name == "":
            raise TaskRunRepositoryValidationError("task_name must be a non-empty string")
        if not isinstance(self.status, TaskRunStatus):
            raise TaskRunRepositoryValidationError("status must be a TaskRunStatus")
        if self.status is TaskRunStatus.FAILED and (
            not isinstance(self.error, str) or self.error == ""
        ):
            raise TaskRunRepositoryValidationError("a failed run must carry a non-empty error")
        if self.status is TaskRunStatus.COMPLETED and self.error is not None:
            raise TaskRunRepositoryValidationError("a completed run must not carry an error")
        if not isinstance(self.detail, Mapping):
            raise TaskRunRepositoryValidationError("detail must be a mapping")
        if not isinstance(self.started_at, str) or self.started_at == "":
            raise TaskRunRepositoryValidationError("started_at must be a non-empty string")
        if not isinstance(self.finished_at, str) or self.finished_at == "":
            raise TaskRunRepositoryValidationError("finished_at must be a non-empty string")
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


class RuntimeTaskRunRepository:
    """Append-only persistence for automation task-run records."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> RuntimeTaskRunRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def record_run(self, record: TaskRunRecord) -> None:
        """Append one immutable task-run row (duplicate run_id fails closed)."""
        if not isinstance(record, TaskRunRecord):
            raise TaskRunRepositoryValidationError("record must be a TaskRunRecord")
        self._connection.execute(
            "INSERT INTO runtime_task_runs "
            "(run_id, task_name, status, error, detail_json, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.task_name,
                record.status.value,
                record.error,
                json.dumps(dict(record.detail), sort_keys=True),
                record.started_at,
                record.finished_at,
            ),
        )

    def list_runs(self, task_name: str, *, limit: int = 100) -> tuple[TaskRunRecord, ...]:
        """Return the most recent runs for one task, oldest first."""
        if not isinstance(task_name, str) or task_name == "":
            raise TaskRunRepositoryValidationError("task_name must be a non-empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise TaskRunRepositoryValidationError("limit must be a positive integer")
        rows = self._connection.execute(
            "SELECT run_id, task_name, status, error, detail_json, started_at, finished_at "
            "FROM runtime_task_runs WHERE task_name = ? "
            "ORDER BY started_at DESC, run_id DESC LIMIT ?",
            (task_name, limit),
        ).fetchall()
        return tuple(_decode_row(row) for row in reversed(rows))

    def latest_run(self, task_name: str) -> TaskRunRecord | None:
        """The most recent run for one task, or None when the task has never run."""
        runs = self.list_runs(task_name, limit=1)
        return runs[0] if runs else None


def _decode_row(row: sqlite3.Row) -> TaskRunRecord:
    detail = json.loads(row["detail_json"])
    if not isinstance(detail, dict):
        raise TaskRunRepositoryError("stored detail_json must be a JSON object")
    return TaskRunRecord(
        run_id=row["run_id"],
        task_name=row["task_name"],
        status=TaskRunStatus(row["status"]),
        error=row["error"],
        detail=detail,
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
