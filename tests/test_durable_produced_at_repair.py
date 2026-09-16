"""P15 burn-down Issue 1 -- historical produced_at repair, plan + apply.

The fixture simulates the pre-fix world: the model named ``produced_at`` in a
generation payload and the service adopted it wholesale, so the durable run
column, the encoded-result fields, and the catalog_track_state projection all
carried the fabricated instant. The journal's completed_at (the service's own
execution instant) is the only reconstruction authority, and the repair must
refuse anything it cannot prove -- blocked rows stay untouched, drifted rows
abort the transaction, and the recommendation_runs immutability triggers are
suspended only for the apply transaction's duration and are restored (and
re-verified active) afterwards.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.catalog_track_state_repository import CatalogTrackStateRepository
from music_agent.durable_produced_at_repair import (
    RepairError,
    _restore_missing_triggers,
    _trigger_sql,
    apply_repair,
    build_repair_plan,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    SignalIdentity,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue
from music_agent.validation import validate_fixture

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_33333333-3333-4333-8333-333333333333"  # fixture's third track
CLIENT_ID = "agt_44444444-4444-4444-8444-444444444444"
BLOCKED_RUN_ID = "rcm_b1000000-0000-4000-8000-0000000000b1"

EXECUTION_AT = "2026-08-16T07:40:00+00:00"
WRONG_AT = "2026-08-16T01:10:00+08:00"  # fabricated model instant (~14.5h drift)
CLOSE_WRONG_AT = "2026-08-16T07:45:00+00:00"  # 5min drift: within tolerance

_RUN_TRIGGERS = (
    "trg_recommendation_runs_immutable_update",
    "trg_recommendation_runs_immutable_delete",
)


class DurableProducedAtRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for track in fixture["tracks"]:
            if track["id"] == TRACK_B:
                track["genres"] = ["Synthetic Ambient"]
                track["external_ids"]["itunes_store_id"] = "STORE-B"
        validate_fixture(fixture)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
        )
        self.addCleanup(self.service.close)

    # --- fixture helpers ------------------------------------------------------

    def seed_positive(self, track_id: str) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                    "apple_music",
                    "favorited",
                ),
                ObservedValue.value(True),
                observed_at="2026-08-10T00:00:00+00:00",
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )

    def generate_run(self, target_ids: list[str]) -> str:
        result = self.service.execute(
            AgentRequest(
                f"req_{uuid.uuid4()}",
                AgentClientIdentity(CLIENT_ID, "codex"),
                "generate_recommendation",
                {"target_ids": target_ids, "limit": 5},
                datetime.now(timezone.utc),
                contract_version=AGENT_CONTRACT_VERSION,
            ),
            completed_at=EXECUTION_AT,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_message)
        return result.payload["run_id"]

    def corrupt(self, run_id: str, wrong: str, *, catalog_ids: tuple[str, ...] = ()) -> None:
        """Simulate the pre-fix bug: overwrite a run's produced_at (and the
        same string inside its encoded result) plus the catalog projection,
        suspending the immutability triggers exactly like a manipulation of
        the pre-fix-era data would have required back then."""
        conn = sqlite3.connect(self.database_path)
        captured = {
            name: sql
            for name in _RUN_TRIGGERS
            if (sql := _trigger_sql(conn, name)) is not None
        }
        self.assertEqual(set(captured), set(_RUN_TRIGGERS))
        try:
            conn.execute(f"DROP TRIGGER {_RUN_TRIGGERS[0]}")
            conn.execute(f"DROP TRIGGER {_RUN_TRIGGERS[1]}")
            produced_at, encoded = conn.execute(
                "SELECT produced_at, encoded_result FROM recommendation_runs "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            self.assertIn(produced_at, encoded)
            conn.execute(
                "UPDATE recommendation_runs SET produced_at = ?, "
                "encoded_result = ? WHERE run_id = ?",
                (wrong, encoded.replace(produced_at, wrong), run_id),
            )
            for can_id in catalog_ids:
                conn.execute(
                    "UPDATE catalog_track_state SET first_recommended_at = ?, "
                    "last_recommended_at = ? WHERE canonical_id = ?",
                    (wrong, wrong, can_id),
                )
            _restore_missing_triggers(conn, captured)
            conn.commit()
        except BaseException:
            conn.rollback()
            _restore_missing_triggers(conn, captured)
            raise
        finally:
            conn.close()

    def insert_unjournaled_run(self, *,
                               template_run_id: str,
                               produced_at: str) -> None:
        """A run no journal request ever named (e.g. hypothetical pre-journal
        history): INSERT is allowed on runs, the immutable triggers only guard
        UPDATE/DELETE."""
        template = self.service.execute(
            AgentRequest(
                f"req_{uuid.uuid4()}",
                AgentClientIdentity(CLIENT_ID, "codex"),
                "get_recommendation_run",
                {"run_id": template_run_id},
                datetime.now(timezone.utc),
                contract_version=AGENT_CONTRACT_VERSION,
            ),
            completed_at=EXECUTION_AT,
        ).payload["encoded_result"]
        conn = sqlite3.connect(self.database_path)
        try:
            contract_version = conn.execute(
                "SELECT contract_version FROM recommendation_runs WHERE run_id = ?",
                (template_run_id,),
            ).fetchone()[0]
            # Craft a period-consistent row: the run names itself everywhere the
            # template did and carries the same fabricated instant end to end.
            crafted_env = json.loads(template)
            crafted_env["run_id"] = BLOCKED_RUN_ID
            crafted_env["produced_at"] = produced_at
            crafted_env["request"]["context"]["now"] = produced_at
            crafted = json.dumps(crafted_env, ensure_ascii=False)
            conn.execute(
                "INSERT INTO recommendation_runs "
                "(run_id, encoded_result, contract_version, produced_at) "
                "VALUES (?, ?, ?, ?)",
                (BLOCKED_RUN_ID, crafted, contract_version, produced_at),
            )
            conn.commit()
        finally:
            conn.close()

    def query_run(self, run_id: str) -> tuple[str, str]:
        conn = sqlite3.connect(self.database_path)
        try:
            return conn.execute(
                "SELECT produced_at, encoded_result FROM recommendation_runs "
                "WHERE run_id = ?",
                (run_id,),
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

    def test_plan_rebuilds_drifted_run_from_journal_and_resets_catalog(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertTrue(state.ensure_state(TRACK_B, source_system="itunes_store"))
        run_id = self.generate_run([TRACK_A, TRACK_B])
        self.corrupt(run_id, WRONG_AT, catalog_ids=(TRACK_B,))

        plan = build_repair_plan(self.database_path)
        self.assertEqual(plan.total_runs, 1)
        self.assertEqual(plan.within_tolerance, ())
        self.assertEqual(plan.blocked, ())
        self.assertEqual(
            [(e.run_id, e.old_produced_at, e.new_produced_at) for e in plan.entries],
            [(run_id, WRONG_AT, EXECUTION_AT)],
        )
        self.assertEqual(
            [(r.canonical_id, r.old_first, r.old_last, r.new_first, r.new_last)
             for r in plan.catalog_resets],
            [(TRACK_B, WRONG_AT, WRONG_AT, EXECUTION_AT, EXECUTION_AT)],
        )
        self.assertTrue(plan.has_work)

    def test_unjournaled_run_is_blocked_never_guessed(self) -> None:
        self.seed_positive(TRACK_A)
        run_id = self.generate_run([TRACK_A])
        self.corrupt(run_id, WRONG_AT)
        self.insert_unjournaled_run(
            template_run_id=run_id, produced_at="2025-01-01T00:00:00+00:00"
        )

        plan = build_repair_plan(self.database_path)
        self.assertEqual(len(plan.entries), 1)
        self.assertEqual(
            [(b.run_id, b.reason) for b in plan.blocked],
            [(BLOCKED_RUN_ID, "no_ok_generate_journal_reference")],
        )
        # Applying repairs the provable row; the blocked row stays untouched.
        report = apply_repair(self.database_path, plan)
        self.assertEqual(report["runs_repaired"], 1)
        self.assertEqual(self.query_run(BLOCKED_RUN_ID)[0], "2025-01-01T00:00:00+00:00")
        self.assertEqual(self.query_run(run_id)[0], EXECUTION_AT)
        self.assertIn("2025-01-01T00:00:00+00:00", self.query_run(BLOCKED_RUN_ID)[1])

    def test_apply_restores_triggers_and_verifies_clean(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertTrue(state.ensure_state(TRACK_B, source_system="itunes_store"))
        run_id = self.generate_run([TRACK_A, TRACK_B])
        self.corrupt(run_id, WRONG_AT, catalog_ids=(TRACK_B,))

        plan = build_repair_plan(self.database_path)
        report = apply_repair(self.database_path, plan)
        self.assertEqual(report["runs_repaired"], 1)
        self.assertEqual(report["catalog_resets"], 1)
        self.assertFalse(report["unchanged"])
        backup = Path(report["backup"])
        self.assertTrue(backup.exists())

        produced_at, encoded = self.query_run(run_id)
        self.assertEqual(produced_at, EXECUTION_AT)
        self.assertNotIn(WRONG_AT, encoded)
        envelope = json.loads(encoded)
        self.assertEqual(envelope["produced_at"], EXECUTION_AT)
        self.assertEqual(envelope["request"]["context"]["now"], EXECUTION_AT)
        conn = sqlite3.connect(self.database_path)
        try:
            row = conn.execute(
                "SELECT first_recommended_at, last_recommended_at "
                "FROM catalog_track_state WHERE canonical_id = ?",
                (TRACK_B,),
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, (EXECUTION_AT, EXECUTION_AT))

        # The immutability boundary is back: the service can never rewrite runs.
        self.assertGreaterEqual(
            len(self.trigger_names() & set(_RUN_TRIGGERS)), 2
        )
        conn = sqlite3.connect(self.database_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE recommendation_runs SET produced_at = ? WHERE run_id = ?",
                    (WRONG_AT, run_id),
                )
        finally:
            conn.close()

        self.assertEqual(build_repair_plan(self.database_path).entries, ())

    def test_apply_refuses_when_a_row_drifted_since_the_plan(self) -> None:
        self.seed_positive(TRACK_A)
        run_id = self.generate_run([TRACK_A])
        self.corrupt(run_id, WRONG_AT)
        plan = build_repair_plan(self.database_path)
        self.corrupt(run_id, CLOSE_WRONG_AT)  # drift after plan was built

        with self.assertRaises(RepairError):
            apply_repair(self.database_path, plan)
        self.assertEqual(self.query_run(run_id)[0], CLOSE_WRONG_AT)
        self.assertGreaterEqual(len(self.trigger_names() & set(_RUN_TRIGGERS)), 2)

    def test_within_tolerance_drift_is_left_alone(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertTrue(state.ensure_state(TRACK_B, source_system="itunes_store"))
        run_id = self.generate_run([TRACK_A, TRACK_B])
        # Faithful simulation: the pre-fix world corrupted run AND projection
        # together, so the corpus and its min/max stay mutually consistent.
        self.corrupt(run_id, CLOSE_WRONG_AT, catalog_ids=(TRACK_B,))

        plan = build_repair_plan(self.database_path)
        self.assertEqual(plan.entries, ())
        self.assertEqual(plan.catalog_resets, ())
        self.assertIn(run_id, plan.within_tolerance)
        self.assertFalse(plan.has_work)  # nothing drifted beyond tolerance
        report = apply_repair(self.database_path, plan)
        self.assertTrue(report["unchanged"])
        self.assertEqual(report["runs_repaired"], 0)
        # An unchanged apply writes no backup and touches nothing.
        self.assertFalse(
            self.database_path.with_suffix(".pre-produced-at-repair.bak").exists()
        )
        self.assertEqual(self.query_run(run_id)[0], CLOSE_WRONG_AT)

    def test_existing_backup_is_never_overwritten(self) -> None:
        self.seed_positive(TRACK_A)
        run_id = self.generate_run([TRACK_A])
        self.corrupt(run_id, WRONG_AT)
        plan = build_repair_plan(self.database_path)
        apply_repair(self.database_path, plan)
        backup_path = self.database_path.with_suffix(".pre-produced-at-repair.bak")
        first_size = backup_path.stat().st_size

        # A second drifted run produces work, but the backup already exists.
        self.corrupt(run_id, CLOSE_WRONG_AT)
        second_plan = build_repair_plan(
            self.database_path, tolerance_hours=0.0
        )
        self.assertTrue(second_plan.has_work)
        with self.assertRaises(RepairError) as raised:
            apply_repair(self.database_path, second_plan)
        self.assertIn("refusing to overwrite existing backup", str(raised.exception))
        self.assertEqual(backup_path.stat().st_size, first_size)

    def test_empty_store_plan_is_a_clean_no_work_report(self) -> None:
        plan = build_repair_plan(self.database_path)
        self.assertEqual(plan.total_runs, 0)
        self.assertFalse(plan.has_work)
        report = apply_repair(self.database_path, plan)
        self.assertTrue(report["unchanged"])
        self.assertEqual(report["runs_repaired"], 0)


if __name__ == "__main__":
    unittest.main()