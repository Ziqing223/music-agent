"""P15 burn-down Issue 1 -- one-shot historical repair of model-controlled
``produced_at`` timestamps in durable recommendation history.

Before Issue 1's authority fix, the model could name ``produced_at`` in a
generation payload and the service adopted that value wholesale. The fabricated
instant then flowed into:

* ``recommendation_runs.produced_at`` (the durable column),
* ``produced_at`` and ``request.context.now`` inside ``encoded_result``
  (both carry the same string), and
* ``catalog_track_state`` first/last_recommended_at (min/max projection
  over the run corpus).

The P09 journal is immune: ``agent_requests.completed_at`` is the service's
own execution instant and the rows are append-only/immutable, so it is the
unambiguous reconstruction authority.

This module repairs history ONLY where it can be rebuilt unambiguously:

* the run maps to EXACTLY one ok ``generate_*`` journal request (a replay row
  naming the same run is only acceptable when its completed_at agrees);
* the drift between the run's stored produced_at and the journal instant
  exceeds the tolerance; and
* the wrong produced_at string appears verbatim inside ``encoded_result`` --
  the byte-scoped rewrite then provably covers every field that carried it.

Wrong timestamps that cannot be proven this way are left untouched and
reported, never guessed (no fabricated history).

``recommendation_runs`` carries immutable UPDATE/DELETE triggers. The repair
is a maintenance operator, not the service: it captures the triggers' own SQL,
drops them for the duration of ONE exclusive transaction, rewrites only the
rows the plan names (re-verified against live state inside the transaction),
recreates the triggers before commit, and refuses to run without a successful
sqlite backup written first. ``catalog_track_state`` has no immutability
trigger; its first/last_recommended_at recompute (min/max over the corrected
run corpus) uses ordinary UPDATEs and only touches rows whose recomputed
values differ from their stored values.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_RUN_UPDATE_TRIGGER = "trg_recommendation_runs_immutable_update"
_RUN_DELETE_TRIGGER = "trg_recommendation_runs_immutable_delete"
_RUN_MUTATION_TRIGGERS = (_RUN_UPDATE_TRIGGER, _RUN_DELETE_TRIGGER)
_GENERATE_TOOL_PREFIX = "generate_"


class RepairError(Exception):
    """The repair refused to proceed; state is unchanged."""


@dataclass(frozen=True)
class JournalRef:
    """One ok generate-* journal row that names a run in its result envelope."""

    request_id: str
    completed_at: str


@dataclass(frozen=True)
class RepairEntry:
    """One recommendation run whose fabricated produced_at is provably
    rebuildable from a single journal instant."""

    run_id: str
    old_produced_at: str
    new_produced_at: str
    journal_request_id: str


@dataclass(frozen=True)
class BlockedEntry:
    """A run with a suspicious time that cannot be rebuilt unambiguously."""

    run_id: str
    old_produced_at: str
    reason: str
    journal_ref_count: int


@dataclass(frozen=True)
class CatalogReset:
    """A catalog_track_state row whose recommendation-time projection is
    recomputed purely from the corrected run corpus."""

    canonical_id: str
    old_first: str | None
    old_last: str | None
    new_first: str
    new_last: str


@dataclass(frozen=True)
class RepairPlan:
    database_path: str
    tolerance_hours: float
    total_runs: int
    within_tolerance: tuple[str, ...]
    entries: tuple[RepairEntry, ...]
    blocked: tuple[BlockedEntry, ...]
    catalog_resets: tuple[CatalogReset, ...]
    catalog_skipped: tuple[tuple[str, str, tuple[str, ...]], ...]

    @property
    def has_work(self) -> bool:
        return bool(self.entries or self.catalog_resets)


def _require_aware(value: str, owner: str) -> datetime:
    """Parse one ISO timestamp and fail closed unless it is timezone-aware."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RepairError(f"refusing naive timestamp for {owner}: {value!r}")
    return parsed


def _collect_run_ids(value: Any) -> set[str]:
    """Every rcm_ id named anywhere in a journal result envelope (each run is
    created by exactly one generate request; the envelope names it)."""
    found: set[str] = set()
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "run_id" and isinstance(child, str) and child.startswith("rcm_"):
                    found.add(child)
                else:
                    stack.append(child)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _journal_run_refs(conn: sqlite3.Connection) -> dict[str, list[JournalRef]]:
    refs: dict[str, list[JournalRef]] = {}
    rows = conn.execute(
        "SELECT request_id, completed_at, result_text FROM agent_requests "
        "WHERE tool_name LIKE ? AND outcome = 'ok' ORDER BY created_at, request_id",
        (_GENERATE_TOOL_PREFIX + "%",),
    ).fetchall()
    for request_id, completed_at, result_text in rows:
        try:
            envelope = json.loads(result_text)
        except json.JSONDecodeError:
            continue  # not an envelope we can interpret -- never guess
        for run_id in _collect_run_ids(envelope):
            refs.setdefault(run_id, []).append(
                JournalRef(request_id, completed_at)
            )
    return refs


def _resolve_ref(
    refs: list[JournalRef],
) -> tuple[JournalRef | None, str | None]:
    """A run must map to exactly one journal instant. Multiple refs (a replay
    row naming the same run) are acceptable only when they all agree."""
    if not refs:
        return None, "no_ok_generate_journal_reference"
    instants = {ref.completed_at for ref in refs}
    if len(instants) != 1:
        return None, "ambiguous_journal_reference"
    return refs[0], None


def _item_canonical_ids(encoded_result: str) -> frozenset[str]:
    """Canonical ids of every persisted item of a run (contract-decoded)."""
    from music_agent.recommendation_contract import decode_recommendation_result

    result = decode_recommendation_result(encoded_result)
    return frozenset(
        item.candidate.target.target_id for item in result.items
    )


def _snapshot_catalog_rows(
    conn: sqlite3.Connection,
) -> list[tuple[str, str | None, str | None]]:
    return conn.execute(
        "SELECT canonical_id, first_recommended_at, last_recommended_at "
        "FROM catalog_track_state ORDER BY canonical_id"
    ).fetchall()


def build_repair_plan(database_path: str | Path, *, tolerance_hours: float = 0.25) -> RepairPlan:
    """Read-only audit + plan. Never mutates."""
    path = str(database_path)
    tolerdelta = timedelta(hours=tolerance_hours)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        journal = _journal_run_refs(conn)
        run_rows = conn.execute(
            "SELECT run_id, produced_at, encoded_result FROM recommendation_runs "
            "ORDER BY created_at, run_id"
        ).fetchall()
        items_by_run = {
            run_id: _item_canonical_ids(encoded)
            for run_id, _, encoded in run_rows
        }
        corrected: dict[str, str] = {}
        unmapped: set[str] = set()
        within_tolerance: list[str] = []
        entries: list[RepairEntry] = []
        blocked: list[BlockedEntry] = []
        for run_id, produced_at, encoded in run_rows:
            ref, reason = _resolve_ref(journal.get(run_id, []))
            if ref is None:
                blocked.append(
                    BlockedEntry(
                        run_id,
                        produced_at,
                        reason or "unknown",
                        len(journal.get(run_id, [])),
                    )
                )
                unmapped.add(run_id)
                continue
            old_dt = _require_aware(produced_at, run_id)
            new_dt = _require_aware(ref.completed_at, f"journal {ref.request_id}")
            if abs(new_dt - old_dt) <= tolerdelta:
                within_tolerance.append(run_id)
                corrected[run_id] = produced_at
                continue
            new_iso = new_dt.isoformat()
            if produced_at not in encoded:
                # The fabricated string must be provably present so the
                # byte-scoped rewrite covers every field that carried it.
                blocked.append(
                    BlockedEntry(
                        run_id,
                        produced_at,
                        "drifted_timestamp_not_verbatim_in_encoded_result",
                        len(journal.get(run_id, [])),
                    )
                )
                unmapped.add(run_id)
                continue
            entries.append(
                RepairEntry(run_id, produced_at, new_iso, ref.request_id)
            )
            corrected[run_id] = new_iso

        dt_by_run = {
            run_id: _require_aware(corrected[run_id], run_id)
            for run_id in corrected
        }
        repaired_ids = {entry.run_id for entry in entries}
        # Per-catalog-row run-time contributions from the corrected corpus.
        cat_times: dict[str, list[datetime]] = {}
        cat_dirty: dict[str, set[str]] = {}
        cat_driven_by_repaired: dict[str, bool] = {}
        for run_id, ids in items_by_run.items():
            if run_id in unmapped:
                for can_id in ids:
                    cat_dirty.setdefault(can_id, set()).add(run_id)
            elif run_id in dt_by_run:
                for can_id in ids:
                    cat_times.setdefault(can_id, []).append(dt_by_run[run_id])
                    if run_id in repaired_ids:
                        cat_driven_by_repaired[can_id] = True

        catalog_resets: list[CatalogReset] = []
        catalog_skipped: list[tuple[str, str, tuple[str, ...]]] = []
        for can_id, first, last in _snapshot_catalog_rows(conn):
            if can_id in cat_dirty:
                catalog_skipped.append(
                    (
                        can_id,
                        "recommendation_times_refer_to_unrepaired_runs",
                        tuple(sorted(cat_dirty[can_id])),
                    )
                )
                continue
            times = sorted(cat_times.get(can_id, []))
            if not times:
                if first is None and last is None:
                    continue  # never-recommended row: consistent, untouched
                # Recommendation stamps exist but no run in the corpus can be
                # traced to this row -- a consistency question outside
                # produced_at scope: untouched.
                catalog_skipped.append(
                    (can_id, "no_run_corpus_contribution", ())
                )
                continue
            new_first, new_last = times[0].isoformat(), times[-1].isoformat()
            if new_first == first and new_last == last:
                continue  # projection already consistent: nothing to do
            if not cat_driven_by_repaired.get(can_id, False):
                # The recompute differs but no contributing run was repaired:
                # resetting would propagate unrepaired drift into the
                # projection -- an inconsistency outside produced_at authority.
                catalog_skipped.append(
                    (can_id, "recompute_differs_but_no_repaired_run_contributes", ())
                )
                continue
            catalog_resets.append(
                CatalogReset(can_id, first, last, new_first, new_last)
            )

        return RepairPlan(
            database_path=path,
            tolerance_hours=tolerance_hours,
            total_runs=len(run_rows),
            within_tolerance=tuple(within_tolerance),
            entries=tuple(entries),
            blocked=tuple(blocked),
            catalog_resets=tuple(catalog_resets),
            catalog_skipped=tuple(catalog_skipped),
        )
    finally:
        conn.close()


def _trigger_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (name,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _restore_missing_triggers(
    conn: sqlite3.Connection, captured: dict[str, str]
) -> None:
    for name, sql in captured.items():
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (name,),
        ).fetchone()
        if not present:
            conn.execute(sql)


def apply_repair(database_path: str | Path, plan: RepairPlan) -> dict:
    """Apply a plan built by :func:`build_repair_plan`.

    Refuses to run when the plan has no work. Writes a sqlite backup next to
    the database first (never overwriting an existing backup). The two
    recommendation_runs immutability triggers are suspended for the duration
    of one exclusive transaction with live re-verification of every planned
    row -- any drift raises and rolls back (which also restores the triggers).
    """
    if not plan.has_work:
        return {
            "backup": None,
            "runs_repaired": 0,
            "catalog_resets": 0,
            "unchanged": True,
        }
    db_path = Path(database_path)
    backup_path = db_path.with_suffix(".pre-produced-at-repair.bak")
    if backup_path.exists():
        raise RepairError(
            f"refusing to overwrite existing backup {backup_path}; "
            "move it aside and retry"
        )
    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 30000")
    captured = {
        name: sql
        for name in _RUN_MUTATION_TRIGGERS
        if (sql := _trigger_sql(conn, name)) is not None
    }
    if set(captured) != set(_RUN_MUTATION_TRIGGERS):
        conn.close()
        raise RepairError(
            "expected both recommendation_runs immutability triggers; "
            f"found {sorted(captured)}"
        )

    try:
        conn.execute("BEGIN IMMEDIATE")
        # Live re-verification: every planned row must still be exactly as the
        # RO plan saw it, otherwise nothing in the plan is trustworthy.
        rewrites: dict[str, str] = {}
        for entry in plan.entries:
            row = conn.execute(
                "SELECT produced_at, encoded_result FROM recommendation_runs "
                "WHERE run_id = ?",
                (entry.run_id,),
            ).fetchone()
            if row is None or row[0] != entry.old_produced_at:
                raise RepairError(
                    f"run {entry.run_id} drifted since the plan was built"
                )
            if row[0] not in row[1]:
                raise RepairError(
                    f"run {entry.run_id}: wrong timestamp no longer verbatim "
                    "in encoded_result"
                )
            rewrites[entry.run_id] = row[1].replace(
                row[0], entry.new_produced_at
            )
        for reset in plan.catalog_resets:
            row = conn.execute(
                "SELECT first_recommended_at, last_recommended_at "
                "FROM catalog_track_state WHERE canonical_id = ?",
                (reset.canonical_id,),
            ).fetchone()
            if row is None or (row[0], row[1]) != (reset.old_first, reset.old_last):
                raise RepairError(
                    f"catalog row {reset.canonical_id} drifted since the plan "
                    "was built"
                )

        conn.execute(f"DROP TRIGGER {_RUN_UPDATE_TRIGGER}")
        conn.execute(f"DROP TRIGGER {_RUN_DELETE_TRIGGER}")
        for entry in plan.entries:
            cur = conn.execute(
                "UPDATE recommendation_runs "
                "SET produced_at = ?, encoded_result = ? WHERE run_id = ?",
                (
                    entry.new_produced_at,
                    rewrites[entry.run_id],
                    entry.run_id,
                ),
            )
            if cur.rowcount != 1:
                raise RepairError(f"run {entry.run_id} update affected {cur.rowcount} rows")
        for reset in plan.catalog_resets:
            cur = conn.execute(
                "UPDATE catalog_track_state SET first_recommended_at = ?, "
                "last_recommended_at = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE canonical_id = ?",
                (reset.new_first, reset.new_last, reset.canonical_id),
            )
            if cur.rowcount != 1:
                raise RepairError(
                    f"catalog row {reset.canonical_id} update affected {cur.rowcount} rows"
                )
        for name, sql in captured.items():
            conn.execute(sql)
        conn.execute("COMMIT")
        return {
            "backup": str(backup_path),
            "runs_repaired": len(plan.entries),
            "catalog_resets": len(plan.catalog_resets),
            "unchanged": False,
        }
    except BaseException:
        try:
            conn.execute("ROLLBACK")  # DDL is transactional: un-drops triggers
        except sqlite3.Error:
            pass
        _restore_missing_triggers(conn, captured)
        raise
    finally:
        conn.close()