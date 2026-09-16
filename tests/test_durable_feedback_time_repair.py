"""P16-S1 -- historical feedback/learning time repair, plan + apply.

The fixture simulates the pre-fix world: the model named ``observed_at`` /
``applied_at`` in tool payloads and the service adopted the fabricated instant
wholesale, so the durable columns (plus the canonical ``encoded_observation``
text and its derived ``duplicate_key``) carried fabricated times. The
journal's completed_at (the service's own execution instant) is the only
reconstruction authority, and the repair must refuse anything it cannot prove
-- blocked rows stay untouched, drifted rows abort the transaction, and the
feedback/learning immutability triggers are suspended only for the apply
transaction's duration and are restored (and re-verified active) afterwards.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.durable_feedback_time_repair import (
    RepairError,
    _restore_missing_triggers,
    _trigger_sql,
    apply_repair,
    build_repair_plan,
)
from music_agent.feedback_contract import (
    FeedbackKind,
    FeedbackObservation,
    FeedbackSourceReference,
    assemble_feedback_observation,
    decode_feedback_observation,
    encode_feedback_observation,
)
from music_agent.feedback_history_repository import encode_duplicate_key
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.repository import CanonicalRepository
from music_agent.validation import validate_fixture

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_33333333-3333-4333-8333-333333333333"
CLIENT_ID = "agt_44444444-4444-4444-8444-444444444444"

EXECUTION_AT = "2026-08-16T07:40:00+00:00"      # journal authority instant
WRONG_AT = "2026-08-16T01:10:00+08:00"          # fabricated model instant (~15h drift)
WRONG_AT_D = "2026-08-16T02:10:00+08:00"        # another fabricated instant (distinct event key)
CLOSE_AT = "2026-08-16T07:45:00+00:00"          # 5min drift: within tolerance

FBK_A = "fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
FBK_B = "fbk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
FBK_C = "fbk_cccccccc-cccc-4ccc-8ccc-cccccccccccc"  # no journal reference
FBK_D = "fbk_dddddddd-dddd-4ddd-8ddd-dddddddddddd"  # ambiguous journal reference

_MUTATION_TRIGGERS = (
    "trg_feedback_observations_immutable_update",
    "trg_feedback_observations_immutable_delete",
    "trg_learning_applications_immutable_update",
    "trg_learning_applications_immutable_delete",
)


class DurableFeedbackTimeRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        validate_fixture(fixture)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    # --- fixture helpers ------------------------------------------------------

    def legacy_feedback_row(
        self,
        feedback_id: str,
        observed_at: str,
        *,
        track_id: str = TRACK_A,
        kind: FeedbackKind = FeedbackKind.LIKED,
    ) -> None:
        """INSERT the pre-fix durable row exactly as the old service wrote it:
        fabricated instant inside the canonical encoding, mirrored column, and
        the derived duplicate key."""
        observation = assemble_feedback_observation(
            feedback_id=feedback_id,
            kind=kind,
            source=FeedbackSourceReference("apple_music", "user-explicit-statement"),
            observed_at=datetime.fromisoformat(observed_at),
            target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
        )
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute(
                "INSERT INTO feedback_observations"
                "(feedback_id, encoded_observation, kind, contract_version, "
                "observed_at, duplicate_key) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    encode_feedback_observation(observation),
                    kind.value,
                    observation.contract_version,
                    observation.observed_at.isoformat(),
                    encode_duplicate_key(observation),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def legacy_application_row(self, feedback_id: str, applied_at: str) -> None:
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute(
                "INSERT INTO learning_applications"
                "(feedback_id, proposal_kind, target_kind, target_id, provenance, "
                "interpretation_policy_version, effect_policy_version, "
                "learning_policy_version, learning_policy_contract_version, applied_at) "
                "VALUES (?, 'evidence_observation', 'track', ?, "
                "'feedback_learning:explicit', 1, 1, 2, 2, ?)",
                (feedback_id, TRACK_A, applied_at),
            )
            conn.commit()
        finally:
            conn.close()

    def journal_row(
        self,
        request_id: str,
        tool_name: str,
        payload: dict,
        result_payload: dict,
        completed_at: str,
        *,
        outcome: str = "ok",
    ) -> None:
        """INSERT one journal row (INSERT is allowed; immutability only guards
        UPDATE/DELETE)."""
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute(
                "INSERT INTO agent_requests"
                "(request_id, client_id, model_id, tool_name, contract_version, "
                "request_text, payload_text, outcome, result_text, issued_at, completed_at) "
                "VALUES (?, ?, 'codex', ?, 1, '{}', ?, ?, ?, ?, ?)",
                (
                    request_id,
                    CLIENT_ID,
                    tool_name,
                    json.dumps(payload, ensure_ascii=False),
                    outcome,
                    json.dumps(
                        {
                            "outcome": outcome,
                            "payload": result_payload,
                            "completed_at": completed_at,
                        },
                        ensure_ascii=False,
                    ),
                    completed_at,
                    completed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_journal(
        self,
        request_id: str,
        feedback_id: str,
        completed_at: str,
        *,
        target_id: str = TRACK_A,
        kind: str = "liked",
    ) -> None:
        self.journal_row(
            request_id,
            "record_feedback",
            {
                "kind": kind,
                "source_system": "apple_music",
                "source_path": "user-explicit-statement",
                "observed_at": WRONG_AT,  # what the pre-fix model actually sent
                "target_id": target_id,
            },
            {"feedback_id": feedback_id, "kind": kind},
            completed_at,
        )

    def apply_journal(
        self, request_id: str, feedback_id: str, completed_at: str
    ) -> None:
        self.journal_row(
            request_id,
            "apply_learning",
            {"feedback_id": feedback_id},
            {"applied": True, "feedback_id": feedback_id},
            completed_at,
        )

    def corrupted_row(self, feedback_id: str) -> tuple[str, str] | None:
        conn = sqlite3.connect(self.database_path)
        try:
            return conn.execute(
                "SELECT observed_at, encoded_observation FROM feedback_observations "
                "WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
        finally:
            conn.close()

    def trigger_names(self) -> set[str]:
        conn = sqlite3.connect(self.database_path)
        try:
            return {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
        finally:
            conn.close()

    # --- tests ----------------------------------------------------------------

    def test_plan_rebuilds_fabricated_rows_and_keeps_close_ones(self) -> None:
        # A: hours of fabricated drift on both durable tables.
        self.legacy_feedback_row(FBK_A, WRONG_AT)
        self.legacy_application_row(FBK_A, WRONG_AT)
        self.record_journal("req_c0000000-0000-4000-8000-0000000000a1", FBK_A, EXECUTION_AT)
        self.apply_journal("req_c0000000-0000-4000-8000-0000000000a2", FBK_A, EXECUTION_AT)
        # B: near-correct times stay untouched.
        self.legacy_feedback_row(FBK_B, CLOSE_AT)
        self.legacy_application_row(FBK_B, CLOSE_AT)
        self.record_journal("req_c0000000-0000-4000-8000-0000000000b1", FBK_B, EXECUTION_AT)
        self.apply_journal("req_c0000000-0000-4000-8000-0000000000b2", FBK_B, EXECUTION_AT)

        plan = build_repair_plan(self.database_path)
        self.assertEqual(plan.total_observations, 2)
        self.assertEqual(plan.total_applications, 2)
        self.assertEqual(plan.blocked, ())
        self.assertEqual(plan.observations_within_tolerance, (FBK_B,))
        self.assertEqual(plan.applications_within_tolerance, (FBK_B,))
        self.assertEqual(
            [(e.feedback_id, e.old_observed_at, e.new_observed_at) for e in plan.feedback_entries],
            [(FBK_A, WRONG_AT, EXECUTION_AT)],
        )
        self.assertEqual(
            [(e.feedback_id, e.old_applied_at, e.new_applied_at) for e in plan.application_entries],
            [(FBK_A, WRONG_AT, EXECUTION_AT)],
        )
        self.assertTrue(plan.has_work)

    def test_apply_rewrites_consistently_and_restores_triggers(self) -> None:
        self.legacy_feedback_row(FBK_A, WRONG_AT)
        self.legacy_application_row(FBK_A, WRONG_AT)
        self.record_journal("req_d0000000-0000-4000-8000-0000000000a1", FBK_A, EXECUTION_AT)
        self.apply_journal("req_d0000000-0000-4000-8000-0000000000a2", FBK_A, EXECUTION_AT)
        old_at, old_encoded = self.corrupted_row(FBK_A)

        report = apply_repair(self.database_path, build_repair_plan(self.database_path))
        self.assertEqual(
            report,
            {
                "backup": str(self.database_path.with_suffix(".pre-feedback-time-repair.bak")),
                "observations_repaired": 1,
                "applications_repaired": 1,
                "unchanged": False,
            },
        )
        self.assertTrue(self.database_path.with_suffix(".pre-feedback-time-repair.bak").exists())

        conn = sqlite3.connect(self.database_path)
        try:
            row = conn.execute(
                "SELECT observed_at, encoded_observation, duplicate_key "
                "FROM feedback_observations WHERE feedback_id = ?",
                (FBK_A,),
            ).fetchone()
            self.assertEqual(row[0], EXECUTION_AT)
            self.assertIn(EXECUTION_AT, row[1])
            self.assertNotIn(WRONG_AT, row[1])
            repaired = decode_feedback_observation(row[1])
            self.assertEqual(repaired.observed_at.isoformat(), EXECUTION_AT)
            self.assertEqual(row[2], encode_duplicate_key(repaired))
            self.assertNotEqual(row[2], old_encoded)  # key is recomputed, never copied
            applied_at = conn.execute(
                "SELECT applied_at FROM learning_applications WHERE feedback_id = ?",
                (FBK_A,),
            ).fetchone()[0]
            self.assertEqual(applied_at, EXECUTION_AT)
            # The journal is the authority: it is never rewritten.
            journal_times = conn.execute(
                "SELECT completed_at FROM agent_requests ORDER BY request_id"
            ).fetchall()
            self.assertEqual([row_i[0] for row_i in journal_times], [EXECUTION_AT, EXECUTION_AT])
        finally:
            conn.close()

        # Triggers are re-created and enforced again.
        self.assertEqual(set(_MUTATION_TRIGGERS), set(_MUTATION_TRIGGERS) & self.trigger_names())
        conn = sqlite3.connect(self.database_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE feedback_observations SET observed_at = '2026-01-01T00:00:00+00:00'"
                )
        finally:
            conn.close()

        # Nothing is left to repair, and a second apply reports unchanged.
        plan_after = build_repair_plan(self.database_path)
        self.assertFalse(plan_after.has_work)
        report_after = apply_repair(self.database_path, plan_after)
        self.assertEqual(report_after["unchanged"], True)

    def test_plan_blocks_rows_without_unique_journal_authority(self) -> None:
        # C: no journal request ever recorded it.
        self.legacy_feedback_row(FBK_C, WRONG_AT)
        self.legacy_application_row(FBK_C, WRONG_AT)
        # D: two disagreeing journal rows name the same target.
        self.legacy_feedback_row(FBK_D, WRONG_AT_D, track_id=TRACK_A)
        self.record_journal("req_d0000000-0000-4000-8000-0000000000d1", FBK_D, EXECUTION_AT)
        self.record_journal(
            "req_d0000000-0000-4000-8000-0000000000d2", FBK_D, CLOSE_AT
        )

        plan = build_repair_plan(self.database_path)
        self.assertFalse(plan.has_work)
        reasons = {b.feedback_id: b.reason for b in plan.blocked}
        self.assertEqual(reasons[FBK_C], "no_journal_reference")
        self.assertEqual(reasons[FBK_D], "ambiguous_journal_reference")
        # apply is a no-op on the blocked rows; their stored values survive.
        apply_repair(self.database_path, plan)
        self.assertEqual(self.corrupted_row(FBK_C)[0], WRONG_AT)
        self.assertEqual(self.corrupted_row(FBK_D)[0], WRONG_AT_D)

    def test_apply_refuses_existing_backup(self) -> None:
        self.legacy_feedback_row(FBK_A, WRONG_AT)
        self.record_journal("req_e0000000-0000-4000-8000-0000000000a1", FBK_A, EXECUTION_AT)
        backup_path = self.database_path.with_suffix(".pre-feedback-time-repair.bak")
        backup_path.write_text("occupied", encoding="utf-8")
        with self.assertRaises(RepairError):
            apply_repair(self.database_path, build_repair_plan(self.database_path))

    def test_apply_aborts_on_live_drift_and_restores_triggers(self) -> None:
        self.legacy_feedback_row(FBK_A, WRONG_AT)
        self.record_journal("req_e0000000-0000-4000-8000-0000000000a1", FBK_A, EXECUTION_AT)
        plan = build_repair_plan(self.database_path)

        # Someone alters the durable row between plan and apply.
        conn = sqlite3.connect(self.database_path)
        captured = {
            name: sql
            for name in _MUTATION_TRIGGERS
            if (sql := _trigger_sql(conn, name)) is not None
        }
        self.assertEqual(set(captured), set(_MUTATION_TRIGGERS))
        conn.execute(f"DROP TRIGGER {_MUTATION_TRIGGERS[0]}")
        conn.execute(
            "UPDATE feedback_observations SET observed_at = ? WHERE feedback_id = ?",
            ("2026-08-17T00:00:00+00:00", FBK_A),
        )
        _restore_missing_triggers(conn, captured)
        conn.commit()
        conn.close()

        with self.assertRaises(RepairError):
            apply_repair(self.database_path, plan)
        # The transaction rolled back; the triggers are enforced again.
        self.assertEqual(set(_MUTATION_TRIGGERS), set(_MUTATION_TRIGGERS) & self.trigger_names())
        self.assertEqual(self.corrupted_row(FBK_A)[0], "2026-08-17T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()