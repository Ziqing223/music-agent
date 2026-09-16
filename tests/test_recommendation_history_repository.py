"""P07.5: durable recommendation persistence and history.

These tests prove the append-only recommendation-run history: one immutable ``recommendation_runs``
row per :class:`RecommendationResult`, persisted through the canonical
``encode_recommendation_result`` interchange with ``run_id`` / ``contract_version`` /
``produced_at`` mirrored as columns. They cover migration, the save/get round-trip (including
empty and multi-item ranked results), deterministic history ordering with a documented tie-break,
duplicate-run_id rejection, missing-run lookup, fail-closed decoding of corrupted rows and
disagreeing mirrors, hard immutability via the SQL triggers, the absence of any update/delete
path, and the isolation of recommendation history from the P06 preference tables.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
    decode_recommendation_result,
    encode_recommendation_result,
)
from music_agent.recommendation_history_repository import (
    CorruptRecommendationHistoryError,
    DuplicateRecommendationRunError,
    RecommendationHistoryRepository,
    RecommendationHistoryRepositoryError,
)
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS
from music_agent.source_observation import ObservedValue

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
RUN_ID_2 = "rcm_33333333-3333-4333-8333-333333333333"
RUN_ID_3 = "rcm_44444444-4444-4444-8444-444444444444"
RUN_ID_4 = "rcm_55555555-5555-4555-8555-555555555555"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
CANDIDATE_ID_2 = "cnd_66666666-6666-4666-8666-666666666666"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def positive_strength(magnitude: float = 0.9) -> PreferenceStrength:
    return PreferenceStrength(PreferenceState.POSITIVE, magnitude)


def direct_input() -> PreferenceInput:
    return PreferenceInput.from_direct(DerivedPreference(track_target(), positive_strength()))


def source() -> CandidateSourceReference:
    return CandidateSourceReference("candidate_gen", "preference_match")


def candidate(*, candidate_id: str = CANDIDATE_ID) -> Candidate:
    return Candidate(candidate_id, track_target(), source())


def score() -> ScoreBreakdown:
    return ScoreBreakdown(0.9, (ScoreComponent("preference_match", 0.9),))


def item(cand: Candidate | None = None) -> RecommendationItem:
    return RecommendationItem(cand or candidate(), score())


def context() -> RecommendationContext:
    return RecommendationContext(NOW, (direct_input(),))


def request() -> RecommendationRequest:
    return RecommendationRequest(context(), RecommendedItemKind.TRACK, 5)


def result(
    *,
    run_id: str = RUN_ID,
    produced_at: datetime | None = None,
    items: tuple[RecommendationItem, ...] | None = None,
) -> RecommendationResult:
    return assemble_recommendation_result(
        request(),
        (items if items is not None else (item(),)),
        run_id=run_id,
        produced_at=produced_at if produced_at is not None else NOW,
    )


def at(hour: int) -> datetime:
    return datetime(2026, 8, 16, hour, 0, 0, tzinfo=timezone.utc)


class RecommendationHistoryRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_rows(self) -> list[sqlite3.Row]:
        with RecommendationHistoryRepository(self.database_path) as repository:
            return list(
                repository._connection.execute(
                    "SELECT * FROM recommendation_runs ORDER BY run_id"
                )
            )

    def insert_history_rows(self, repository, entries) -> None:
        """Raw-insert history rows with explicit created_at values so tests control
        the newest-first ordering deterministically (save_result stamps real time)."""
        for run, created_at in entries:
            repository._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    encode_recommendation_result(run),
                    run.contract_version,
                    run.produced_at.isoformat(),
                    created_at,
                ),
            )

    # --- schema / migration ------------------------------------------------

    def test_fresh_database_reaches_current_version_with_history_table(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("recommendation_runs", tables)
            indexes = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            self.assertIn("ix_recommendation_runs_produced_at", indexes)

    def test_recommendation_runs_table_has_only_the_frozen_columns(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(recommendation_runs)"
                )
            }
        self.assertEqual(
            columns,
            {"run_id", "encoded_result", "contract_version", "produced_at", "created_at"},
        )

    def test_v11_store_upgrades_to_current_without_changing_prior_state(self) -> None:
        identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:11]):
            with PreferencePersistenceRepository(self.database_path) as repository:
                repository.record_observation(
                    identity, ObservedValue.value(True), observed_at=NOW.isoformat()
                )
                self.assertEqual(repository.schema_version, 11)

        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            repository.save_result(result())

        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertIs(head.current_semantic_value, True)

    # --- save / get round-trip --------------------------------------------

    def test_save_get_round_trip_equals_original(self) -> None:
        ranked = result(
            items=(item(), item(candidate(candidate_id=CANDIDATE_ID_2)))
        )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(ranked)
            loaded = repository.get_result(ranked.run_id)
        self.assertEqual(loaded, ranked)
        self.assertEqual(
            [i.candidate.candidate_id for i in loaded.items],
            [CANDIDATE_ID, CANDIDATE_ID_2],
        )

    def test_empty_items_result_round_trips(self) -> None:
        empty = result(items=())
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(empty)
            loaded = repository.get_result(empty.run_id)
        self.assertEqual(loaded, empty)
        self.assertEqual(loaded.items, ())

    def test_runs_reload_after_reopen(self) -> None:
        run = result(run_id=RUN_ID_2, produced_at=at(1))
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(run)
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.get_result(run.run_id), run)
            self.assertEqual(list(repository.list_runs()), [run])

    # --- ordering ----------------------------------------------------------

    def test_list_runs_orders_by_created_at_newest_first(self) -> None:
        # P12 latency: produced_at is model-supplied and can contradict real insertion
        # recency (e.g. a run stamped "12:00+08:00" created before a run stamped
        # "00:00+08:00"). created_at -- the insertion instant -- is the ordering key.
        future_stamped = result(
            run_id=RUN_ID, produced_at=datetime(2026, 8, 30, 0, 0, tzinfo=timezone.utc)
        )
        older_stamped = result(
            run_id=RUN_ID_2, produced_at=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    future_stamped.run_id,
                    encode_recommendation_result(future_stamped),
                    future_stamped.contract_version,
                    future_stamped.produced_at.isoformat(),
                    "2026-08-17 06:00:00",
                ),
            )
            repository._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    older_stamped.run_id,
                    encode_recommendation_result(older_stamped),
                    older_stamped.contract_version,
                    older_stamped.produced_at.isoformat(),
                    "2026-08-17 06:05:00",
                ),
            )
            runs = repository.list_runs()
        self.assertIsInstance(runs, tuple)
        # The later-created run is first even though its produced_at is earlier.
        self.assertEqual([r.run_id for r in runs], [RUN_ID_2, RUN_ID])

    def test_list_runs_ties_break_deterministically_by_run_id(self) -> None:
        same_instant = at(12)
        later_run_id = result(run_id=RUN_ID_2, produced_at=same_instant)
        earlier_run_id = result(run_id=RUN_ID, produced_at=same_instant)
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(later_run_id)
            repository.save_result(earlier_run_id)
            runs = repository.list_runs()
        # Both saves land in the same second: equal created_at, run_id descending is
        # the documented tie-break.
        self.assertEqual([r.run_id for r in runs], [RUN_ID_2, RUN_ID])

    # --- insertion-chronology identity (P19-T14-B) --------------------------

    def test_newest_run_id_follows_real_insertion_order(self) -> None:
        # The reply door needs true insertion chronology. Two runs written in
        # the same second share one created_at, so the frozen list_runs
        # tie-break (run_id DESC) puts the OLDER (lexically larger) run
        # first; the rowid read must still identify the row inserted most
        # recently.
        older_stamped = result(run_id=RUN_ID_2, produced_at=at(1))
        newer_stamped = result(run_id=RUN_ID, produced_at=at(2))
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(
                repository,
                (
                    (older_stamped, "2026-08-17 06:00:00"),
                    (newer_stamped, "2026-08-17 06:00:00"),
                ),
            )
            self.assertEqual(repository.newest_run_id(), RUN_ID)
            # the frozen display order is unchanged -- independent of identity
            self.assertEqual(
                [run.run_id for run in repository.list_runs()], [RUN_ID_2, RUN_ID]
            )

    def test_newest_run_id_matches_single_run_and_empty_history(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertIsNone(repository.newest_run_id())
            repository.save_result(result(run_id=RUN_ID))
            self.assertEqual(repository.newest_run_id(), RUN_ID)

    def test_newest_run_id_ignores_produced_at_contradiction(self) -> None:
        # Insertion chronology wins over produced_at stamps: a row inserted
        # later with an EARLIER produced_at is still the newest run (rowid is
        # pure insertion order; produced_at never plays a role).
        future_stamped = result(run_id=RUN_ID, produced_at=at(9))
        past_stamped = result(run_id=RUN_ID_2, produced_at=at(1))
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(
                repository,
                (
                    (future_stamped, "2026-08-17 06:00:00"),
                    (past_stamped, "2026-08-17 06:00:05"),
                ),
            )
            self.assertEqual(repository.newest_run_id(), RUN_ID_2)
            # created_at is the display order key: the later-created row is first
            self.assertEqual(
                [run.run_id for run in repository.list_runs()], [RUN_ID_2, RUN_ID]
            )

    # --- bounded window (P14-R4.2) ------------------------------------------

    def test_list_runs_limit_returns_only_the_most_recent_runs(self) -> None:
        entries = [
            result(run_id=RUN_ID_2, produced_at=at(6)),
            result(run_id=RUN_ID_3, produced_at=at(7)),
            result(run_id=RUN_ID_4, produced_at=at(8)),
        ]
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(
                repository,
                (
                    (entries[0], "2026-08-17 06:00:00"),
                    (entries[1], "2026-08-17 06:01:00"),
                    (entries[2], "2026-08-17 06:02:00"),
                ),
            )
            window = repository.list_runs(limit=2)
        # The newest two, in newest-first order -- the window never starts elsewhere.
        self.assertEqual([run.run_id for run in window], [RUN_ID_4, RUN_ID_3])

    def test_list_runs_limit_beyond_history_returns_everything(self) -> None:
        entries = [
            result(run_id=RUN_ID_2, produced_at=at(6)),
            result(run_id=RUN_ID_3, produced_at=at(7)),
        ]
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(
                repository,
                (
                    (entries[0], "2026-08-17 06:00:00"),
                    (entries[1], "2026-08-17 06:01:00"),
                ),
            )
            window = repository.list_runs(limit=10)
        self.assertEqual([run.run_id for run in window], [RUN_ID_3, RUN_ID_2])

    def test_list_runs_default_and_none_limit_scan_identically(self) -> None:
        older = result(run_id=RUN_ID, produced_at=at(6))
        newer = result(run_id=RUN_ID_2, produced_at=at(7))
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(
                repository,
                (
                    (older, "2026-08-17 06:00:00"),
                    (newer, "2026-08-17 06:01:00"),
                ),
            )
            default_scan = repository.list_runs()
            none_scan = repository.list_runs(limit=None)
        # Unbounded default behaviour is byte-identical to the pre-R4.2 contract.
        self.assertEqual([run.run_id for run in default_scan], [RUN_ID_2, RUN_ID])
        self.assertEqual(none_scan, default_scan)

    def test_list_runs_limit_zero_returns_no_runs(self) -> None:
        entry = result()
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.insert_history_rows(repository, ((entry, "2026-08-17 06:00:00"),))
            self.assertEqual(repository.list_runs(limit=0), ())

    # --- duplicates / missing / corruption ---------------------------------

    def test_duplicate_run_id_raises_specific_error_and_stores_nothing_extra(self) -> None:
        first = result()
        second = result(produced_at=at(1))
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(first)
            with self.assertRaises(DuplicateRecommendationRunError):
                repository.save_result(second)
            self.assertEqual(len(repository.list_runs()), 1)
            self.assertEqual(repository.get_result(RUN_ID), first)

    def test_get_result_for_missing_run_returns_none(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_result(RUN_ID_4))

    def test_corrupted_encoded_result_fails_closed(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at
                ) VALUES (?, ?, ?, ?)""",
                (RUN_ID, "not canonical json", 1, NOW.isoformat()),
            )
            with self.assertRaises(CorruptRecommendationHistoryError):
                repository.get_result(RUN_ID)
            with self.assertRaises(CorruptRecommendationHistoryError):
                repository.list_runs()

    def test_mirror_mismatch_fails_closed(self) -> None:
        run = result()
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at
                ) VALUES (?, ?, ?, ?)""",
                (run.run_id, encode_recommendation_result(run), 99, run.produced_at.isoformat()),
            )
            with self.assertRaises(CorruptRecommendationHistoryError):
                repository.get_result(run.run_id)

    # --- immutability ------------------------------------------------------

    def test_raw_sql_update_and_delete_raise(self) -> None:
        run = result()
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(run)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE recommendation_runs SET produced_at=? WHERE run_id=?",
                    ("2026-08-17T00:00:00+00:00", run.run_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute("DELETE FROM recommendation_runs")
        self.assertEqual(len(self.run_rows()), 1)

    def test_repository_exposes_no_update_or_delete_path(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertFalse(hasattr(repository, "update_result"))
            self.assertFalse(hasattr(repository, "delete_result"))
            self.assertFalse(hasattr(repository, "update_run"))
            self.assertFalse(hasattr(repository, "delete_run"))

    # --- mirrored columns --------------------------------------------------

    def test_mirrored_columns_match_result_fields_for_every_run(self) -> None:
        runs = [
            result(),
            result(
                run_id=RUN_ID_2,
                produced_at=at(1),
                items=(item(candidate(candidate_id=CANDIDATE_ID_2)),),
            ),
        ]
        with RecommendationHistoryRepository(self.database_path) as repository:
            for run in runs:
                repository.save_result(run)
            rows = list(
                repository._connection.execute(
                    "SELECT * FROM recommendation_runs ORDER BY run_id"
                )
            )
        for run, row in zip(sorted(runs, key=lambda r: r.run_id), rows):
            with self.subTest(run_id=run.run_id):
                self.assertEqual(row["run_id"], run.run_id)
                self.assertEqual(row["contract_version"], run.contract_version)
                self.assertEqual(row["produced_at"], run.produced_at.isoformat())
                self.assertEqual(decode_recommendation_result(row["encoded_result"]), run)

    # --- input validation --------------------------------------------------

    def test_save_result_rejects_non_result_input(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            for bad in ("not-a-result", None, {"run_id": RUN_ID}, 42, [item()]):
                with self.subTest(bad=bad):
                    with self.assertRaises(RecommendationHistoryRepositoryError):
                        repository.save_result(bad)  # type: ignore[arg-type]
            self.assertEqual(len(repository.list_runs()), 0)

    # --- isolation from P06 ------------------------------------------------

    def test_recommendation_history_leaves_preference_tables_untouched(self) -> None:
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(result())
            for table in ("preference_signal_heads", "preference_evidence_revisions"):
                count = repository._connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                self.assertEqual(count, 0, table)
            canonical_count = repository._connection.execute(
                "SELECT COUNT(*) FROM canonical_entities"
            ).fetchone()[0]
            self.assertEqual(canonical_count, 0)

    def test_preference_persistence_leaves_recommendation_history_untouched(self) -> None:
        identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                identity, ObservedValue.value(True), observed_at=NOW.isoformat()
            )
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_runs()), 0)
            count = repository._connection.execute(
                "SELECT COUNT(*) FROM recommendation_runs"
            ).fetchone()[0]
            self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
