"""P10.15: Read-only storage report -- table row counts and store file size.

Retention policy (documented, deliberately non-destructive): the sealed append-only
tables (agent_requests, runtime_task_runs, feedback/recommendation history) carry
immutability triggers, so retention does NOT delete rows -- file-level backup
rotation (P10.16 ``music-agent backup``) is the supported retention mechanism. This
module only OBSERVES growth so the operator can decide when to archive.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from music_agent.repository import _open_store_connection


@dataclass(frozen=True, slots=True)
class StorageReport:
    store_path: Path
    size_bytes: int
    table_rows: dict[str, int]

    def summary(self) -> str:
        lines = [
            f"store: {self.store_path}",
            f"size_bytes: {self.size_bytes}",
        ]
        for table, rows in sorted(self.table_rows.items()):
            lines.append(f"table {table}: {rows} rows")
        return "\n".join(lines)


def build_storage_report(database_path: str | Path) -> StorageReport:
    database_path = Path(database_path)
    try:
        size_bytes = database_path.stat().st_size
    except OSError as error:
        raise ValueError(f"could not stat the store: {error}") from error
    connection = _open_store_connection(database_path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        table_rows: dict[str, int] = {}
        for table in tables:
            try:
                count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            except sqlite3.Error:
                continue
            table_rows[table] = count
    finally:
        connection.close()
    return StorageReport(database_path, size_bytes, table_rows)
