"""P16-S1 -- one-shot historical repair of model-controlled feedback/learning
timestamps in durable history.

Before S1's authority fix, the model could name ``observed_at`` in a
``record_feedback`` payload (and ``applied_at`` in an ``apply_learning``
payload) and the service adopted the fabricated instant wholesale. The value
then flowed into:

* ``feedback_observations.observed_at`` and the same string inside
  ``encoded_observation`` (the canonical unit; the column mirrors it), and
* ``learning_applications.applied_at``.

``feedback_observations.duplicate_key`` also derives from ``observed_at``
(identity of the observed event), so a repaired observation needs its key
recomputed through the canonical :func:`encode_duplicate_key` interchange --
the rewrite is decode -> re-stamp -> re-encode, never a hand-rolled patch.

The P09 journal is immune: ``agent_requests.completed_at`` is the service's
own execution instant and the rows are append-only/immutable, so it is the
unambiguous reconstruction authority.

This module repairs history ONLY where it can be rebuilt unambiguously:

* a feedback observation maps to EXACTLY one ok ``record_feedback`` journal
  request (by the payload's explicit ``feedback_id``, else its
  ``run_id``/``candidate_id`` pair, else its ``kind`` + ``target_id``);
* a learning application maps to EXACTLY one ``apply_learning`` journal
  request whose result payload has ``applied: true`` (no_proposal rows wrote
  nothing durable and are not authorities);
* the drift between the stored time and the journal instant exceeds the
  tolerance; and
* the wrong ``observed_at`` string appears verbatim inside
  ``encoded_observation`` -- the byte-scoped rewrite then provably covers
  every field that carried it.

Wrong timestamps that cannot be proven this way are left untouched and
reported, never guessed (no fabricated history).

``feedback_observations`` and ``learning_applications`` both carry immutable
UPDATE/DELETE triggers. The repair is a maintenance operator, not the service:
it captures the triggers' own SQL, drops them for the duration of ONE
exclusive transaction, rewrites only the rows the plan names (re-verified
against live state inside the transaction), recreates the triggers before
commit, and refuses to run without a successful sqlite backup written first.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from music_agent.feedback_contract import (
    FeedbackObservation,
    decode_feedback_observation,
    encode_feedback_observation,
)
from music_agent.feedback_history_repository import encode_duplicate_key

_FEEDBACK_UPDATE_TRIGGER = "trg_feedback_observations_immutable_update"
_FEEDBACK_DELETE_TRIGGER = "trg_feedback_observations_immutable_delete"
_LEARNING_UPDATE_TRIGGER = "trg_learning_applications_immutable_update"
_LEARNING_DELETE_TRIGGER = "trg_learning_applications_immutable_delete"
_MUTATION_TRIGGERS = (
    _FEEDBACK_UPDATE_TRIGGER,
    _FEEDBACK_DELETE_TRIGGER,
    _LEARNING_UPDATE_TRIGGER,
    _LEARNING_DELETE_TRIGGER,
)


class RepairError(Exception):
    """The repair refused to proceed; state is unchanged."""


@dataclass(frozen=True)
class JournalFeedbackRef:
    """One ok record_feedback journal row (the request that wrote one observation)."""

    request_id: str
    completed_at: str


@dataclass(frozen=True)
class JournalApplyRef:
    """One apply_learning journal row whose result payload has ``applied: true``."""

    request_id: str
    completed_at: str


@dataclass(frozen=True)
class FeedbackRepairEntry:
    """One observation whose fabricated observed_at is provably rebuildable
    from a single journal instant."""

    feedback_id: str
    old_observed_at: str
    new_observed_at: str
    journal_request_id: str


@dataclass(frozen=True)
class ApplicationRepairEntry:
    """One learning application whose fabricated applied_at is provably
    rebuildable from a single journal instant."""

    feedback_id: str
    old_applied_at: str
    new_applied_at: str
    journal_request_id: str


@dataclass(frozen=True)
class BlockedEntry:
    """A row with a suspicious time that cannot be rebuilt unambiguously."""

    owner: str  # "feedback" | "application"
    feedback_id: str
    old_value: str
    reason: str
    journal_ref_count: int


@dataclass(frozen=True)
class RepairPlan:
    database_path: str
    tolerance_hours: float
    total_observations: int
    total_applications: int
    observations_within_tolerance: tuple[str, ...]
    applications_within_tolerance: tuple[str, ...]
    feedback_entries: tuple[FeedbackRepairEntry, ...]
    application_entries: tuple[ApplicationRepairEntry, ...]
    blocked: tuple[BlockedEntry, ...]

    @property
    def has_work(self) -> bool:
        return bool(self.feedback_entries or self.application_entries)


def _require_aware(value: str, owner: str) -> datetime:
    """Parse one ISO timestamp and fail closed unless it is timezone-aware."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RepairError(f"refusing naive timestamp for {owner}: {value!r}")
    return parsed


def _journal_feedback_refs(conn: sqlite3.Connection) -> dict[tuple[str, ...], list[JournalFeedbackRef]]:
    """Index every ok record_feedback journal request by match key.

    Match keys mirror the two payload forms: an explicit ``feedback_id`` wins,
    else the ``run_id``/``candidate_id`` pair, else ``kind`` + ``target_id``.
    A row that is not a dict or carries no identifying keys is skipped -- the
    repair never guesses.
    """
    refs: dict[tuple[str, ...], list[JournalFeedbackRef]] = {}
    rows = conn.execute(
        "SELECT request_id, completed_at, payload_text FROM agent_requests "
        "WHERE tool_name = 'record_feedback' AND outcome = 'ok' "
        "ORDER BY created_at, request_id"
    ).fetchall()
    for request_id, completed_at, payload_text in rows:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        key: tuple[str, ...] | None = None
        if isinstance(payload.get("feedback_id"), str) and payload["feedback_id"]:
            key = ("id", payload["feedback_id"])
        elif isinstance(payload.get("run_id"), str) and payload["run_id"]:
            key = ("rec", payload["run_id"], str(payload.get("candidate_id", "")))
        elif isinstance(payload.get("target_id"), str) and payload["target_id"]:
            key = ("target", str(payload.get("kind", "")), payload["target_id"])
        if key is not None:
            refs.setdefault(key, []).append(JournalFeedbackRef(request_id, completed_at))
    return refs


def _journal_apply_refs(conn: sqlite3.Connection) -> dict[str, list[JournalApplyRef]]:
    """Index apply_learning journal requests that actually wrote durable state.

    Only a result whose payload has ``applied: true`` wrote a
    ``learning_applications`` row; ``applied: false`` (no_proposal) and
    refused requests are not reconstruction authorities.
    """
    refs: dict[str, list[JournalApplyRef]] = {}
    rows = conn.execute(
        "SELECT request_id, completed_at, payload_text, result_text FROM agent_requests "
        "WHERE tool_name = 'apply_learning' AND outcome = 'ok' "
        "ORDER BY created_at, request_id"
    ).fetchall()
    for request_id, completed_at, payload_text, result_text in rows:
        try:
            payload = json.loads(payload_text)
            result = json.loads(result_text)
        except json.JSONDecodeError:
            continue
        feedback_id = payload.get("feedback_id") if isinstance(payload, dict) else None
        if not isinstance(feedback_id, str) or not feedback_id:
            continue
        applied = (result.get("payload") or {}).get("applied") if isinstance(result, dict) else None
        if applied is not True:
            continue
        refs.setdefault(feedback_id, []).append(JournalApplyRef(request_id, completed_at))
    return refs


def _resolve_refs(refs: list, owner: str) -> tuple[object | None, str | None]:
    """One owner row must map to exactly one journal instant. Multiple refs are
    acceptable only when they all agree."""
    if not refs:
        return None, "no_journal_reference"
    instants = {ref.completed_at for ref in refs}
    if len(instants) != 1:
        return None, "ambiguous_journal_reference"
    return refs[0], None


def _observation_match_key(observation: FeedbackObservation) -> tuple[str, ...]:
    """The same payload forms as ``_journal_feedback_refs``, from the durable row:
    recommendation pair when present, else kind + target."""
    if observation.recommendation is not None:
        return ("rec", observation.recommendation.run_id, observation.recommendation.candidate_id)
    return ("target", observation.kind.value, observation.target.target_id)


def build_repair_plan(database_path: str | Path, *, tolerance_hours: float = 0.25) -> RepairPlan:
    """Read-only audit + plan. Never mutates."""
    path = str(database_path)
    tolerance = timedelta(hours=tolerance_hours)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        feedback_refs = _journal_feedback_refs(conn)
        apply_refs = _journal_apply_refs(conn)

        observation_rows = conn.execute(
            "SELECT feedback_id, encoded_observation, observed_at "
            "FROM feedback_observations ORDER BY created_at, feedback_id"
        ).fetchall()
        application_rows = conn.execute(
            "SELECT feedback_id, applied_at FROM learning_applications "
            "ORDER BY created_at, feedback_id"
        ).fetchall()

        observations: dict[str, tuple[FeedbackObservation, str]] = {}
        for feedback_id, encoded, observed_at in observation_rows:
            observations[feedback_id] = (
                decode_feedback_observation(encoded),
                observed_at,
            )

        observations_within: list[str] = []
        feedback_entries: list[FeedbackRepairEntry] = []
        blocked: list[BlockedEntry] = []
        for feedback_id, (observation, stored_at) in observations.items():
            key = _observation_match_key(observation)
            ref, reason = _resolve_refs(feedback_refs.get(key, []), f"observation {feedback_id}")
            if ref is None:
                blocked.append(
                    BlockedEntry(
                        "feedback",
                        feedback_id,
                        stored_at,
                        reason or "unknown",
                        len(feedback_refs.get(key, [])),
                    )
                )
                continue
            old_dt = _require_aware(stored_at, feedback_id)
            new_dt = _require_aware(ref.completed_at, f"journal {ref.request_id}")
            if abs(new_dt - old_dt) <= tolerance:
                observations_within.append(feedback_id)
                continue
            new_iso = new_dt.isoformat()
            encoded = _encoded_for(conn, feedback_id)
            if stored_at not in encoded:
                blocked.append(
                    BlockedEntry(
                        "feedback",
                        feedback_id,
                        stored_at,
                        "drifted_timestamp_not_verbatim_in_encoded_observation",
                        len(feedback_refs.get(key, [])),
                    )
                )
                continue
            feedback_entries.append(
                FeedbackRepairEntry(feedback_id, stored_at, new_iso, ref.request_id)
            )

        applications_within: list[str] = []
        application_entries: list[ApplicationRepairEntry] = []
        for feedback_id, applied_at in application_rows:
            ref, reason = _resolve_refs(apply_refs.get(feedback_id, []), f"application {feedback_id}")
            if ref is None:
                blocked.append(
                    BlockedEntry(
                        "application",
                        feedback_id,
                        applied_at,
                        reason or "unknown",
                        len(apply_refs.get(feedback_id, [])),
                    )
                )
                continue
            old_dt = _require_aware(applied_at, feedback_id)
            new_dt = _require_aware(ref.completed_at, f"journal {ref.request_id}")
            if abs(new_dt - old_dt) <= tolerance:
                applications_within.append(feedback_id)
                continue
            application_entries.append(
                ApplicationRepairEntry(
                    feedback_id, applied_at, new_dt.isoformat(), ref.request_id
                )
            )

        return RepairPlan(
            database_path=path,
            tolerance_hours=tolerance_hours,
            total_observations=len(observation_rows),
            total_applications=len(application_rows),
            observations_within_tolerance=tuple(observations_within),
            applications_within_tolerance=tuple(applications_within),
            feedback_entries=tuple(feedback_entries),
            application_entries=tuple(application_entries),
            blocked=tuple(blocked),
        )
    finally:
        conn.close()


def _encoded_for(conn: sqlite3.Connection, feedback_id: str) -> str:
    row = conn.execute(
        "SELECT encoded_observation FROM feedback_observations WHERE feedback_id = ?",
        (feedback_id,),
    ).fetchone()
    return str(row[0])


def _trigger_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (name,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _restore_missing_triggers(conn: sqlite3.Connection, captured: dict[str, str]) -> None:
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
    immutability-trigger pairs are suspended for the duration of one exclusive
    transaction with live re-verification of every planned row -- any drift
    raises and rolls back (which also restores the triggers). Feedback rows
    are re-encoded through the canonical interchange so the mirrored
    ``observed_at`` and the recomputed ``duplicate_key`` always agree with the
    repaired payload.
    """
    if not plan.has_work:
        return {
            "backup": None,
            "observations_repaired": 0,
            "applications_repaired": 0,
            "unchanged": True,
        }
    db_path = Path(database_path)
    backup_path = db_path.with_suffix(".pre-feedback-time-repair.bak")
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
        for name in _MUTATION_TRIGGERS
        if (sql := _trigger_sql(conn, name)) is not None
    }
    if set(captured) != set(_MUTATION_TRIGGERS):
        conn.close()
        raise RepairError(
            "expected the feedback/learning immutability triggers; "
            f"found {sorted(captured)}"
        )

    try:
        conn.execute("BEGIN IMMEDIATE")
        # Live re-verification: every planned row must still be exactly as the
        # RO plan saw it, otherwise nothing in the plan is trustworthy.
        rewrites: dict[str, tuple[str, str, str]] = {}
        for entry in plan.feedback_entries:
            row = conn.execute(
                "SELECT observed_at, encoded_observation FROM feedback_observations "
                "WHERE feedback_id = ?",
                (entry.feedback_id,),
            ).fetchone()
            if row is None or row[0] != entry.old_observed_at:
                raise RepairError(
                    f"observation {entry.feedback_id} drifted since the plan was built"
                )
            if row[0] not in row[1]:
                raise RepairError(
                    f"observation {entry.feedback_id}: wrong timestamp no longer "
                    "verbatim in encoded_observation"
                )
            repaired = decode_feedback_observation(row[1])
            repaired = FeedbackObservation(
                feedback_id=repaired.feedback_id,
                kind=repaired.kind,
                source=repaired.source,
                observed_at=_require_aware(entry.new_observed_at, entry.feedback_id),
                target=repaired.target,
                recommendation=repaired.recommendation,
                attribution=repaired.attribution,
                event_at=repaired.event_at,
                source_event_id=repaired.source_event_id,
            )
            rewrites[entry.feedback_id] = (
                encode_feedback_observation(repaired),
                entry.new_observed_at,
                encode_duplicate_key(repaired),
            )
        for entry in plan.application_entries:
            row = conn.execute(
                "SELECT applied_at FROM learning_applications WHERE feedback_id = ?",
                (entry.feedback_id,),
            ).fetchone()
            if row is None or row[0] != entry.old_applied_at:
                raise RepairError(
                    f"application {entry.feedback_id} drifted since the plan was built"
                )

        conn.execute(f"DROP TRIGGER {_FEEDBACK_UPDATE_TRIGGER}")
        conn.execute(f"DROP TRIGGER {_FEEDBACK_DELETE_TRIGGER}")
        conn.execute(f"DROP TRIGGER {_LEARNING_UPDATE_TRIGGER}")
        conn.execute(f"DROP TRIGGER {_LEARNING_DELETE_TRIGGER}")
        for entry in plan.feedback_entries:
            encoded, new_at, new_key = rewrites[entry.feedback_id]
            cur = conn.execute(
                "UPDATE feedback_observations SET encoded_observation = ?, "
                "observed_at = ?, duplicate_key = ? WHERE feedback_id = ?",
                (encoded, new_at, new_key, entry.feedback_id),
            )
            if cur.rowcount != 1:
                raise RepairError(
                    f"observation {entry.feedback_id} update affected {cur.rowcount} rows"
                )
        for entry in plan.application_entries:
            cur = conn.execute(
                "UPDATE learning_applications SET applied_at = ? WHERE feedback_id = ?",
                (entry.new_applied_at, entry.feedback_id),
            )
            if cur.rowcount != 1:
                raise RepairError(
                    f"application {entry.feedback_id} update affected {cur.rowcount} rows"
                )
        for name, sql in captured.items():
            conn.execute(sql)
        conn.execute("COMMIT")
        return {
            "backup": str(backup_path),
            "observations_repaired": len(plan.feedback_entries),
            "applications_repaired": len(plan.application_entries),
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