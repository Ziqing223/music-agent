"""P15-S3-S1: ``catalog_track_state`` migration, historical backfill, and discovery write hook.

The table (migration 0019) is derived, re-buildable memory per canonical Catalog track. These
tests pin, in order: the fresh-DB schema, the historical-upgrade backfill (catalog-row seeding
with honestly NULL discovery history + recommendation projection from persisted successful items
only), the idempotency of the backfill phase, the hard FK/PK/CHECK constraints, the repository
recording semantics (first/last/count, bounded term merge, normalization, validation), and the
authoritative discovery write point in the catalog ingestion orchestrator (promoted /
already-bound occurrences record, blocked/library-known/staged events never do), and -- from
  P15-S3-S2 -- the runtime inverse of migration B (``record_recommendation_items`` projection
  with the backfill-identical ordering口径, the incremental≡rebuild pin, and the
  ``find_states_by_term`` facts-only memory lookup).
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from unittest.mock import patch

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.apple_music_catalog import CatalogTrack
from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.catalog_ingestion import CatalogIngestionOrchestrator, CatalogIngestStatus
from music_agent.catalog_track_state_repository import (
    CatalogTrackStateError,
    CatalogTrackStateRepository,
    normalize_discovery_term,
)
from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS, CanonicalRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"

ITUNES = "itunes_store"

TRK_CAT_1 = "trk_cccccccc-1111-4111-8111-111111111111"
TRK_CAT_2 = "trk_cccccccc-2222-4222-8222-222222222222"
TRK_LIB_1 = "trk_11111111-3333-4333-8333-333333333333"

REF = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def seed_canonical(repository: CanonicalRepository, canonical_id: str) -> None:
    repository._connection.execute(
        "INSERT INTO canonical_entities(id, entity_type) VALUES (?, 'track')",
        (canonical_id,),
    )


def seed_binding(
    repository: CanonicalRepository,
    canonical_id: str,
    source_system: str,
    external_id: str,
) -> None:
    repository._connection.execute(
        """INSERT INTO external_identity_bindings(
            source_system, entity_type, external_id, canonical_id
        ) VALUES (?, 'track', ?, ?)""",
        (source_system, external_id, canonical_id),
    )


def seed_catalog_binding(repository: CanonicalRepository, canonical_id: str) -> None:
    seed_binding(repository, canonical_id, ITUNES, f"ext-{canonical_id}")


def seed_library_binding(repository: CanonicalRepository, canonical_id: str) -> None:
    seed_binding(repository, canonical_id, "apple_music", f"persist-{canonical_id}")


def seed_catalog_presence(repository: CanonicalRepository, canonical_id: str) -> None:
    repository._connection.execute(
        """INSERT INTO source_entity_presence(
            source_system, entity_type, canonical_id, scope_key, presence
        ) VALUES (?, 'track', ?, 'catalog', 'present')""",
        (ITUNES, canonical_id),
    )


def itunes_track(
    *,
    track_id: str = "1258917044",
    name: str = "Delicate",
    isrc: str | None = None,
    with_album: bool = True,
) -> CatalogTrack:
    return CatalogTrack(
        catalog_id=track_id,
        name=name,
        artist_names=("Taylor Swift",),
        album_name="reputation" if with_album else None,
        genres=("Pop",),
        isrc=isrc,
        duration_ms=232861,
        release_date="2017-11-10",
        url=None,
        artist_catalog_ids=("148607010",),
        album_catalog_id="1258917041" if with_album else None,
        source_system=ITUNES,
        preview_url="https://audio-ssl.itunes.apple.com/preview.m4a",
    )


def track_target(target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id)


def direct_input(target_id: str) -> PreferenceInput:
    return PreferenceInput.from_direct(
        DerivedPreference(track_target(target_id), PreferenceStrength(PreferenceState.POSITIVE, 0.9))
    )


def candidate_for(sequence: int, target_id: str) -> Candidate:
    return Candidate(
        f"cnd_{sequence:08x}-1111-4aaa-8aaa-111111111111",
        track_target(target_id),
        CandidateSourceReference("candidate_gen", "preference_match"),
    )


def item_for(sequence: int, target_id: str) -> RecommendationItem:
    return RecommendationItem(
        candidate_for(sequence, target_id),
        ScoreBreakdown(0.9, (ScoreComponent("preference_match", 0.9),)),
    )


def result_with(
    run_id: str,
    produced_at: datetime,
    item_targets: tuple[str, ...],
    input_targets: tuple[str, ...],
):
    request = RecommendationRequest(
        RecommendationContext(REF, tuple(direct_input(t) for t in input_targets)),
        RecommendedItemKind.TRACK,
        5,
    )
    return assemble_recommendation_result(
        request,
        tuple(item_for(i + 1, t) for i, t in enumerate(item_targets)),
        run_id=run_id,
        produced_at=produced_at,
    )


def migration_sql() -> str:
    return resources.files("music_agent.migrations").joinpath(
        "0019_catalog_track_state.sql"
    ).read_text(encoding="utf-8")


class CatalogTrackStateMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def seed_catalog_history_v18(self) -> None:
        """Build a v18 store carrying catalog tracks and historical recommendation runs."""
        with patch("music_agent.repository.MIGRATIONS", tuple(MIGRATIONS[:18])):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 18)
                seed_canonical(repository, TRK_CAT_1)
                seed_canonical(repository, TRK_CAT_2)
                seed_canonical(repository, TRK_LIB_1)
                seed_catalog_binding(repository, TRK_CAT_1)
                seed_catalog_binding(repository, TRK_CAT_2)
                seed_library_binding(repository, TRK_LIB_1)
                seed_catalog_presence(repository, TRK_CAT_1)
            with RecommendationHistoryRepository(self.database_path) as history:
                t1 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
                t2 = t1 + timedelta(days=1)
                # Run 1: CAT_1 and LIB_1 are real items; CAT_2 (and CAT_1) appear as request
                # inputs. Run 2: CAT_1 again. Run 3: an empty run that must contribute nothing.
                history.save_result(
                    result_with(
                        "rcm_11111111-1111-4111-8111-111111111111",
                        t1,
                        (TRK_CAT_1, TRK_LIB_1),
                        (TRK_CAT_2, TRK_CAT_1),
                    )
                )
                history.save_result(
                    result_with(
                        "rcm_22222222-2222-4222-8222-222222222222",
                        t2,
                        (TRK_CAT_1,),
                        (TRK_CAT_2,),
                    )
                )
                history.save_result(
                    result_with(
                        "rcm_33333333-3333-4333-8333-333333333333",
                        t2,
                        (),
                        (TRK_CAT_1,),
                    )
                )

    def test_fresh_database_creates_table_and_reaches_v19(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("catalog_track_state", tables)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.schema_version, 19)
            self.assertEqual(state.count(), 0)

    def test_historical_upgrade_seeds_rows_without_fabricated_discovery_history(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", tuple(MIGRATIONS[:18])):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 18)
                seed_canonical(repository, TRK_CAT_1)
                seed_canonical(repository, TRK_LIB_1)
                seed_catalog_binding(repository, TRK_CAT_1)
                seed_library_binding(repository, TRK_LIB_1)
                seed_catalog_presence(repository, TRK_CAT_1)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.schema_version, 19)
            row = state.get_state(TRK_CAT_1)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.source_system, ITUNES)
            # Honest empty history: NULL instants, 0 counts, no fabricated terms.
            self.assertIsNone(row.first_discovered_at)
            self.assertIsNone(row.last_discovered_at)
            self.assertEqual(row.discovery_count, 0)
            self.assertEqual(row.discovery_terms, {})
            self.assertIsNone(row.first_recommended_at)
            self.assertIsNone(row.last_recommended_at)
            self.assertEqual(row.recommendation_count, 0)
        # Library tracks are never catalog tracks: no row, even with a binding.
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertIsNone(state.get_state(TRK_LIB_1))

    def test_recommendation_backfill_projects_only_real_items(self) -> None:
        self.seed_catalog_history_v18()
        t1 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(days=1)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.schema_version, 19)
            cat1 = state.get_state(TRK_CAT_1)
            self.assertIsNotNone(cat1)
            assert cat1 is not None
            # Two item occurrences across two runs: count is occurrence-based and re-buildable.
            self.assertEqual(cat1.recommendation_count, 2)
            self.assertEqual(cat1.first_recommended_at, t1.isoformat())
            self.assertEqual(cat1.last_recommended_at, t2.isoformat())
            # Discovery-side facts stay untouched by the recommendation projection.
            self.assertIsNone(cat1.first_discovered_at)
            self.assertEqual(cat1.discovery_count, 0)
            # Request inputs (preference inputs) never count as recommendations.
            cat2 = state.get_state(TRK_CAT_2)
            self.assertIsNotNone(cat2)
            assert cat2 is not None
            self.assertEqual(cat2.recommendation_count, 0)
            self.assertEqual(cat2.recommendation_count, 0)
            # Library tracks appeared as items but have no catalog state row at all.
            self.assertIsNone(state.get_state(TRK_LIB_1))

    def test_projection_orders_first_last_by_true_instant_across_mixed_offsets(self) -> None:
        # Real runs carry mixed UTC offsets (+08:00 and +00:00). Lexically
        # "2026-08-16T23:00:00+00:00" sorts before "2026-08-17T00:10:00+08:00", but as instants
        # the +08:00 one (16:10Z) is earlier. The projection must pick by true instant.
        later_lexically = datetime(2026, 8, 16, 23, 0, 0, tzinfo=timezone.utc)
        earlier_chronologically = datetime(
            2026, 8, 17, 0, 10, 0, tzinfo=timezone(timedelta(hours=8))
        )
        self.assertLess(earlier_chronologically, later_lexically)
        with patch("music_agent.repository.MIGRATIONS", tuple(MIGRATIONS[:18])):
            with CanonicalRepository(self.database_path) as repository:
                seed_canonical(repository, TRK_CAT_1)
                seed_catalog_binding(repository, TRK_CAT_1)
            with RecommendationHistoryRepository(self.database_path) as history:
                history.save_result(
                    result_with(
                        "rcm_11111111-1111-4111-8111-111111111111",
                        later_lexically,
                        (TRK_CAT_1,),
                        (TRK_CAT_1,),
                    )
                )
                history.save_result(
                    result_with(
                        "rcm_22222222-2222-4222-8222-222222222222",
                        earlier_chronologically,
                        (TRK_CAT_1,),
                        (TRK_CAT_1,),
                    )
                )
        with CatalogTrackStateRepository(self.database_path) as state:
            row = state.get_state(TRK_CAT_1)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.first_recommended_at, earlier_chronologically.isoformat())
        self.assertEqual(row.last_recommended_at, later_lexically.isoformat())
        self.assertEqual(row.recommendation_count, 2)

    def test_backfill_phase_rerun_does_not_accumulate(self) -> None:
        self.seed_catalog_history_v18()
        with CatalogTrackStateRepository(self.database_path) as state:
            before = {
                key: state.get_state(key)
                for key in (TRK_CAT_1, TRK_CAT_2)
            }
            rows_before = state.count()
        # Re-execute exactly the backfill phase (everything after the table/index DDL): OR
        # IGNORE seeding plus recomputed UPDATE must repeat the same result, never accumulate.
        backfill = "REPLACED-MARKER" + migration_sql().split("-- A. One row per already-bound", 1)[1]
        backfill = backfill.replace("REPLACED-MARKER", "-- A. One row per already-bound")
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(f"BEGIN IMMEDIATE;\n{backfill}\nCOMMIT;")
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.count(), rows_before)
            for key, snapshot in before.items():
                after = state.get_state(key)
                self.assertIsNotNone(after)
                assert after is not None and snapshot is not None
                self.assertEqual(after.discovery_count, snapshot.discovery_count)
                self.assertEqual(after.recommendation_count, snapshot.recommendation_count)
                self.assertEqual(after.first_recommended_at, snapshot.first_recommended_at)
                self.assertEqual(after.last_recommended_at, snapshot.last_recommended_at)

    def test_pk_fk_and_check_constraints_hold(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            seed_canonical(repository, TRK_CAT_1)
            connection = repository._connection
            with self.assertRaises(sqlite3.IntegrityError):
                # Wrong id namespace.
                connection.execute(
                    """INSERT INTO catalog_track_state(canonical_id, entity_type, source_system)
                    VALUES ('alb_badbadbad-1111-4111-8111-111111111111', 'track', 'itunes_store')"""
                )
            with self.assertRaises(sqlite3.IntegrityError):
                # Valid prefix, unknown canonical -> foreign key.
                connection.execute(
                    """INSERT INTO catalog_track_state(canonical_id, entity_type, source_system)
                    VALUES ('trk_99999999-9999-4999-8999-999999999999', 'track', 'itunes_store')"""
                )
            with self.assertRaises(sqlite3.IntegrityError):
                # Non-catalog source system.
                connection.execute(
                    """INSERT INTO catalog_track_state(canonical_id, entity_type, source_system)
                    VALUES (?, 'track', 'apple_music')""",
                    (TRK_CAT_1,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO catalog_track_state(
                        canonical_id, entity_type, source_system, discovery_count
                    ) VALUES (?, 'track', 'itunes_store', -1)""",
                    (TRK_CAT_1,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO catalog_track_state(
                        canonical_id, entity_type, source_system, recommendation_count
                    ) VALUES (?, 'track', 'itunes_store', -1)""",
                    (TRK_CAT_1,),
                )
            connection.execute(
                """INSERT INTO catalog_track_state(canonical_id, entity_type, source_system)
                VALUES (?, 'track', 'itunes_store')""",
                (TRK_CAT_1,),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                # Duplicate primary key.
                connection.execute(
                    """INSERT INTO catalog_track_state(canonical_id, entity_type, source_system)
                    VALUES (?, 'track', 'itunes_store')""",
                    (TRK_CAT_1,),
                )


class CatalogTrackStateRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        fixture = load_fixture()
        self.track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def state_repo(self) -> CatalogTrackStateRepository:
        return CatalogTrackStateRepository(self.database_path)

    def test_first_occurrence_sets_first_equal_last_and_count_one(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="J-Pop", now=fixed
            )
            row = state.get_state(self.track_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.first_discovered_at, fixed.isoformat())
        self.assertEqual(row.last_discovered_at, fixed.isoformat())
        self.assertEqual(row.discovery_count, 1)
        self.assertEqual(row.discovery_terms, {"j-pop": 1})

    def test_repeated_occurrence_keeps_first_bumps_last_and_count(self) -> None:
        first = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        second = first + timedelta(minutes=5)
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="J-Pop", now=first
            )
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="Yorushika", now=second
            )
            row = state.get_state(self.track_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.first_discovered_at, first.isoformat())
        self.assertEqual(row.last_discovered_at, second.isoformat())
        self.assertEqual(row.discovery_count, 2)
        self.assertEqual(row.discovery_terms, {"j-pop": 1, "yorushika": 1})

    def test_same_normalized_term_accumulates_without_new_entries(self) -> None:
        with self.state_repo() as state:
            for raw in ("  DELICATE ", "delicate", "Delicate"):
                state.record_discovery_occurrence(
                    self.track_id,
                    source_system=ITUNES,
                    term=raw,
                    now=datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc),
                )
            row = state.get_state(self.track_id)
        assert row is not None
        self.assertEqual(row.discovery_count, 3)
        self.assertEqual(row.discovery_terms, {"delicate": 3})

    def test_bounded_term_merge_evicts_oldest_first(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            for index in range(1, 10):
                state.record_discovery_occurrence(
                    self.track_id, source_system=ITUNES, term=f"term-{index}", now=fixed
                )
            row = state.get_state(self.track_id)
        assert row is not None
        # Nine occurrences, eight distinct terms: the oldest (term-1) was evicted.
        self.assertEqual(row.discovery_count, 9)
        self.assertEqual(set(row.discovery_terms), {f"term-{i}" for i in range(2, 10)})
        self.assertEqual(row.discovery_terms["term-2"], 1)
        # Re-recording an existing term moves it to the newest position...
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="term-2", now=fixed
            )
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="term-10", now=fixed
            )
            row = state.get_state(self.track_id)
        assert row is not None
        # ...so the next eviction drops term-3, not term-2.
        self.assertNotIn("term-3", row.discovery_terms)
        self.assertEqual(row.discovery_terms["term-2"], 2)
        self.assertEqual(set(row.discovery_terms), {f"term-{i}" for i in range(4, 11)} | {"term-2"})

    def test_blank_or_missing_term_records_no_terms(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term=None, now=fixed
            )
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="   \n  ", now=fixed
            )
            row = state.get_state(self.track_id)
        assert row is not None
        self.assertEqual(row.discovery_count, 2)
        self.assertEqual(row.discovery_terms, {})

    def test_overlong_term_is_truncated_to_the_bound(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        long_term = "x" * 500
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term=long_term, now=fixed
            )
            row = state.get_state(self.track_id)
        assert row is not None
        (stored,) = row.discovery_terms
        self.assertLessEqual(len(stored), 200)
        self.assertEqual(row.discovery_terms[stored], 1)

    def test_ensure_state_creates_zeroed_row_exactly_once(self) -> None:
        with self.state_repo() as state:
            self.assertTrue(state.ensure_state(self.track_id, source_system=ITUNES))
            self.assertFalse(state.ensure_state(self.track_id, source_system="apple_music_catalog"))
            row = state.get_state(self.track_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.source_system, ITUNES)
        self.assertIsNone(row.first_discovered_at)
        self.assertEqual(row.discovery_count, 0)
        self.assertEqual(row.recommendation_count, 0)

    def test_fk_violation_is_never_swallowed_as_an_ignore(self) -> None:
        unknown = "trk_99999999-9999-4999-8999-999999999999"
        with self.state_repo() as state:
            with self.assertRaises(sqlite3.IntegrityError):
                state.record_discovery_occurrence(
                    unknown,
                    source_system=ITUNES,
                    now=datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc),
                )

    def test_record_validation_fails_closed(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            with self.assertRaises(CatalogTrackStateError):
                state.record_discovery_occurrence(
                    "alb_badbadbad-1111-4111-8111-111111111111",
                    source_system=ITUNES,
                    now=fixed,
                )
            with self.assertRaises(CatalogTrackStateError):
                state.record_discovery_occurrence(
                    self.track_id, source_system="apple_music", now=fixed
                )
            with self.assertRaises(CatalogTrackStateError):
                state.record_discovery_occurrence(
                    self.track_id,
                    source_system=ITUNES,
                    now=datetime(2026, 8, 20, 10, 0, 0),  # naive
                )

    def test_get_state_unknown_returns_none(self) -> None:
        with self.state_repo() as state:
            self.assertIsNone(state.get_state(TRK_CAT_2))

    # --- P15-S3-S3A: bulk supply-facts read ---------------------------------

    def test_load_states_returns_only_existing_rows(self) -> None:
        """Unknown ids are dropped, not fabricated -- absence is "no memory"."""
        second_id = load_fixture()["tracks"][1]["id"]
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="J-Pop", now=REF
            )
            loaded = state.load_states([self.track_id, second_id, TRK_CAT_1])
        self.assertEqual(set(loaded), {self.track_id})
        row = loaded[self.track_id]
        self.assertEqual(row.discovery_count, 1)
        self.assertEqual(row.discovery_terms, {"j-pop": 1})

    def test_load_states_dedupes_ids_and_accepts_empty_input(self) -> None:
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="J-Pop", now=REF
            )
            loaded = state.load_states([self.track_id, self.track_id])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[self.track_id].discovery_count, 1)
            self.assertEqual(state.load_states([]), {})

    def test_load_states_reads_many_rows_in_one_batch(self) -> None:
        """Multiple seeded rows come back together (order-free dict, real counts)."""
        second_id = load_fixture()["tracks"][1]["id"]
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                self.track_id, source_system=ITUNES, term="Alpha", now=REF
            )
            state.record_discovery_occurrence(
                second_id, source_system=ITUNES, term="Beta", now=REF
            )
            loaded = state.load_states([second_id, self.track_id])
        self.assertEqual(set(loaded), {self.track_id, second_id})
        self.assertEqual(loaded[self.track_id].discovery_terms, {"alpha": 1})
        self.assertEqual(loaded[second_id].discovery_terms, {"beta": 1})

    def test_load_states_validation_fails_closed(self) -> None:
        with self.state_repo() as state:
            for bad in (
                self.track_id,  # bare string, not a sequence
                ["alb_aaaaaaaa-1111-4111-8111-111111111111"],
                [7],
                ["not-an-id"],
            ):
                with self.subTest(bad=bad):
                    with self.assertRaises(CatalogTrackStateError):
                        state.load_states(bad)


class CatalogDiscoveryRecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        fixture = load_fixture()
        self.library_track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def client_for(self, source) -> AgentClient:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.FULL}),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_FULL, model_id="test", label="tests"),
            service,
        )

    def discover(self, tracks: tuple[CatalogTrack, ...], term: str = "delicate") -> dict:
        class FakeCatalogSource:
            def __init__(self, tracks: tuple[CatalogTrack, ...]) -> None:
                self.tracks = tracks

            def search(self, term: str, limit: int) -> tuple[CatalogTrack, ...]:
                return self.tracks

        result = self.client_for(FakeCatalogSource(tracks)).call(
            "discover_catalog_tracks", {"term": term}
        )
        self.assertEqual(result.outcome.value, "ok")
        return result.payload

    def state_of(self, canonical_id: str):
        with CatalogTrackStateRepository(self.database_path) as state:
            return state.get_state(canonical_id)

    def test_promoted_discovery_records_first_occurrence_with_term(self) -> None:
        payload = self.discover((itunes_track(),), "delicate")
        self.assertEqual(payload["promoted_count"], 1)
        canonical_id = payload["promoted"][0]["canonical_id"]
        row = self.state_of(canonical_id)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.source_system, ITUNES)
        self.assertEqual(row.discovery_count, 1)
        self.assertIsNotNone(row.first_discovered_at)
        self.assertEqual(row.first_discovered_at, row.last_discovered_at)
        self.assertEqual(row.discovery_terms, {"delicate": 1})

    def test_already_bound_records_a_second_occurrence_with_its_own_term(self) -> None:
        first = self.discover((itunes_track(),), "delicate")["promoted"][0]["canonical_id"]
        first_row = self.state_of(first)
        assert first_row is not None
        second = self.discover((itunes_track(),), "taylor swift")
        self.assertEqual(second["promoted_count"], 0)
        self.assertEqual(second["skipped"][0]["status"], "already_bound")
        row = self.state_of(first)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.discovery_count, 2)
        self.assertEqual(row.first_discovered_at, first_row.first_discovered_at)
        self.assertEqual(row.discovery_terms, {"delicate": 1, "taylor swift": 1})

    def test_blocked_discovery_records_nothing(self) -> None:
        payload = self.discover((itunes_track(with_album=False),), "broken")
        self.assertEqual(payload["staged_count"], 1)
        self.assertEqual(payload["staged"][0]["status"], "staged_blocked")
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.count(), 0)

    def test_library_known_hit_records_nothing(self) -> None:
        # The catalog hit's ISRC resolves to an already-library canonical: proven known music,
        # never a catalog discovery of new music.
        with CanonicalRepository(self.database_path) as repository:
            seed_binding(repository, self.library_track_id, "isrc", "ISO-00000001")
            repository._connection.execute(
                """INSERT INTO source_entity_presence(
                    source_system, entity_type, canonical_id, scope_key, presence
                ) VALUES ('apple_music', 'track', ?, 'library_tracks', 'present')""",
                (self.library_track_id,),
            )
        payload = self.discover((itunes_track(isrc="ISO-00000001"),), "known song")
        self.assertEqual(payload["skipped"][0]["status"], "library_known")
        self.assertIsNone(self.state_of(self.library_track_id))
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.count(), 0)

    def test_fixed_clock_yields_deterministic_discovery_instants(self) -> None:
        fixed = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        orchestrator = CatalogIngestionOrchestrator(
            CanonicalRepository(self.database_path),
            CandidateStagingRepository(self.database_path),
            None,
            track_state=CatalogTrackStateRepository(self.database_path),
            now_fn=lambda: fixed,
        )
        outcome = orchestrator.ingest((itunes_track(),), term="j-pop")[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.PROMOTED)
        self.assertIsNotNone(outcome.canonical_id)
        row = self.state_of(outcome.canonical_id)
        assert row is not None
        self.assertEqual(row.first_discovered_at, fixed.isoformat())
        self.assertEqual(row.discovery_terms, {"j-pop": 1})


class DiscoveryTermNormalizationTest(unittest.TestCase):
    def test_normalization_rules(self) -> None:
        self.assertIsNone(normalize_discovery_term(None))
        self.assertIsNone(normalize_discovery_term("   "))
        self.assertIsNone(normalize_discovery_term("\n\t"))
        self.assertEqual(normalize_discovery_term("  J-Pop  "), "j-pop")
        self.assertEqual(normalize_discovery_term("DELICATE"), "delicate")
        self.assertEqual(normalize_discovery_term("x" * 500), "x" * 200)


class CatalogRecommendationProjectionTest(unittest.TestCase):
    """P15-S3-S2: the runtime inverse of migration B.

    Every persisted run's track items are projected through ``record_recommendation_items``
    at the two save_result write points. These tests pin the backfill-identical口径:
    occurrence counts, first/last ordered by true aware instant with the verbatim lexical
    tie-break, existing-rows-only (library tracks never materialize a row), fail-closed
    validation, and the rebuild invariant -- incremental projection must equal re-running
    migration B over the same history, byte for byte.
    """

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            seed_canonical(repository, TRK_CAT_1)
            seed_canonical(repository, TRK_CAT_2)
            seed_canonical(repository, TRK_LIB_1)
            seed_catalog_binding(repository, TRK_CAT_1)
            seed_catalog_binding(repository, TRK_CAT_2)
            seed_library_binding(repository, TRK_LIB_1)
        with self.state_repo() as state:
            self.assertTrue(state.ensure_state(TRK_CAT_1, source_system=ITUNES))
            self.assertTrue(state.ensure_state(TRK_CAT_2, source_system=ITUNES))

    def state_repo(self) -> CatalogTrackStateRepository:
        return CatalogTrackStateRepository(self.database_path)

    def test_first_projection_sets_first_equal_last_and_occurrence_count(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            projected = state.record_recommendation_items(
                [TRK_CAT_1, TRK_CAT_1, TRK_CAT_2], produced_at=fixed
            )
            cat1 = state.get_state(TRK_CAT_1)
            cat2 = state.get_state(TRK_CAT_2)
        # Two of the three appearances belong to ids with rows (the library skip is pinned
        # separately); CAT_1's double appearance still counts twice.
        self.assertEqual(projected, 2)
        assert cat1 is not None and cat2 is not None
        self.assertEqual(cat1.first_recommended_at, fixed.isoformat())
        self.assertEqual(cat1.last_recommended_at, fixed.isoformat())
        self.assertEqual(cat1.recommendation_count, 2)
        # Discovery-side facts stay untouched by the recommendation projection.
        self.assertIsNone(cat1.first_discovered_at)
        self.assertEqual(cat1.discovery_count, 0)
        self.assertEqual(cat1.updated_at, fixed.isoformat())
        self.assertEqual(cat2.recommendation_count, 1)
        self.assertEqual(cat2.first_recommended_at, cat2.last_recommended_at)

    def test_second_projection_keeps_first_moves_last_and_accumulates(self) -> None:
        first = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        second = first + timedelta(hours=2)
        with self.state_repo() as state:
            state.record_recommendation_items([TRK_CAT_1], produced_at=first)
            state.record_recommendation_items(
                [TRK_CAT_1, TRK_CAT_2], produced_at=second
            )
            cat1 = state.get_state(TRK_CAT_1)
            cat2 = state.get_state(TRK_CAT_2)
        assert cat1 is not None and cat2 is not None
        self.assertEqual(cat1.first_recommended_at, first.isoformat())
        self.assertEqual(cat1.last_recommended_at, second.isoformat())
        self.assertEqual(cat1.recommendation_count, 2)
        self.assertEqual(cat1.updated_at, second.isoformat())
        self.assertEqual(cat2.first_recommended_at, second.isoformat())
        self.assertEqual(cat2.recommendation_count, 1)

    def test_projection_orders_by_true_instant_across_mixed_offsets(self) -> None:
        # Runtime projection order can disagree with chronology (mixed UTC offsets):
        # lexically-later text can name an earlier instant. first/last must follow the
        # migration-B comparator -- true instant, never insertion order or lexical text.
        later_lexically = datetime(2026, 8, 16, 23, 0, 0, tzinfo=timezone.utc)
        earlier_chronologically = datetime(
            2026, 8, 17, 0, 10, 0, tzinfo=timezone(timedelta(hours=8))
        )
        self.assertLess(earlier_chronologically, later_lexically)
        with self.state_repo() as state:
            projected = state.record_recommendation_items(
                [TRK_CAT_1], produced_at=later_lexically
            )
            self.assertEqual(projected, 1)
            state.record_recommendation_items(
                [TRK_CAT_1], produced_at=earlier_chronologically
            )
            row = state.get_state(TRK_CAT_1)
        assert row is not None
        self.assertEqual(row.first_recommended_at, earlier_chronologically.isoformat())
        self.assertEqual(row.last_recommended_at, later_lexically.isoformat())
        self.assertEqual(row.recommendation_count, 2)

    def test_same_instant_tie_break_converges_regardless_of_arrival_order(self) -> None:
        # The B secondary order is the verbatim produced_at text; two instants that are the
        # same instant with different offsets must converge to the lexically smallest text
        # as first and the largest as last, whichever arrived first.
        utc_text = datetime(2026, 8, 17, 0, 0, 0, tzinfo=timezone.utc)
        plus_eight = datetime(
            2026, 8, 17, 8, 0, 0, tzinfo=timezone(timedelta(hours=8))
        )
        self.assertEqual(utc_text, plus_eight)
        self.assertLess(utc_text.isoformat(), plus_eight.isoformat())
        with self.state_repo() as state:
            state.record_recommendation_items([TRK_CAT_1], produced_at=plus_eight)
            state.record_recommendation_items([TRK_CAT_1], produced_at=utc_text)
            state.record_recommendation_items([TRK_CAT_2], produced_at=utc_text)
            state.record_recommendation_items([TRK_CAT_2], produced_at=plus_eight)
            cat1 = state.get_state(TRK_CAT_1)
            cat2 = state.get_state(TRK_CAT_2)
        assert cat1 is not None and cat2 is not None
        for row in (cat1, cat2):
            self.assertEqual(row.first_recommended_at, utc_text.isoformat())
            self.assertEqual(row.last_recommended_at, plus_eight.isoformat())

    def test_missing_row_is_skipped_never_created(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        unknown = "trk_99999999-9999-4999-8999-999999999999"
        with self.state_repo() as state:
            projected = state.record_recommendation_items(
                [TRK_CAT_1, TRK_LIB_1, unknown, TRK_CAT_2], produced_at=fixed
            )
            self.assertEqual(projected, 2)
            # Library target: no row, and none was created (table stays catalog rows only).
            self.assertIsNone(state.get_state(TRK_LIB_1))
            # Identity without a canonical entity: no row to update, skipped like B.
            self.assertIsNone(state.get_state(unknown))
            self.assertEqual(state.count(), 2)

    def test_empty_batch_is_a_noop(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            self.assertEqual(state.record_recommendation_items([], produced_at=fixed), 0)
            self.assertEqual(state.count(), 2)

    def test_record_validation_fails_closed(self) -> None:
        fixed = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
        naive = datetime(2026, 8, 20, 10, 0, 0)
        with self.state_repo() as state:
            with self.assertRaises(CatalogTrackStateError):
                state.record_recommendation_items(TRK_CAT_1, produced_at=fixed)
            with self.assertRaises(CatalogTrackStateError):
                state.record_recommendation_items(
                    ["alb_badbadbad-1111-4111-8111-111111111111"], produced_at=fixed
                )
            with self.assertRaises(CatalogTrackStateError):
                state.record_recommendation_items(
                    ["not-a-track-id"], produced_at=fixed
                )
            with self.assertRaises(CatalogTrackStateError):
                state.record_recommendation_items([TRK_CAT_1], produced_at=naive)
            # No partial writes from any rejected batch.
            cat1 = state.get_state(TRK_CAT_1)
        assert cat1 is not None
        self.assertEqual(cat1.recommendation_count, 0)
        self.assertIsNone(cat1.first_recommended_at)

    def test_incremental_projection_matches_backfill_b_recompute(self) -> None:
        """The rebuild invariant: incremental additive projection ≡ migration B rerun.

        Three persisted runs carry a library item (never counts), request-only inputs
        (never count), a repeated item in one run (occurrence-based count), and mixed UTC
        offsets (true-instant ordering). After projecting each run incrementally, re-running
        B's own UPDATE over the same history must reproduce every byte.
        """
        t1 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(days=1)
        t3 = datetime(2026, 8, 1, 16, 0, 0, tzinfo=timezone(timedelta(hours=8)))  # 08:00Z
        runs = [
            result_with(
                "rcm_11111111-1111-4111-8111-111111111111",
                t1,
                (TRK_CAT_1, TRK_LIB_1),
                (TRK_CAT_2, TRK_CAT_1),
            ),
            result_with(
                "rcm_22222222-2222-4222-8222-222222222222",
                t2,
                (TRK_CAT_1,),
                (TRK_CAT_2,),
            ),
            result_with(
                "rcm_33333333-3333-4333-8333-333333333333",
                t3,
                (TRK_CAT_1, TRK_CAT_2, TRK_CAT_1),
                (TRK_LIB_1,),
            ),
        ]
        with RecommendationHistoryRepository(self.database_path) as history:
            for run in runs:
                history.save_result(run)
        with self.state_repo() as state:
            # The runtime inverse: project each persisted run at its save point. The
            # contract enforces target-kind homogeneity, so a TRACK run's items are all
            # track items (the service hook gates on the same recommended_kind fact).
            for run in runs:
                state.record_recommendation_items(
                    [item.candidate.target.target_id for item in run.items],
                    produced_at=run.produced_at,
                )
            incremental_cat1 = state.get_state(TRK_CAT_1)
            incremental_cat2 = state.get_state(TRK_CAT_2)
            self.assertIsNone(state.get_state(TRK_LIB_1))
        assert incremental_cat1 is not None and incremental_cat2 is not None
        # Expected: CAT_1 occurs once in run 1, once in run 2, twice in run 3; t3 < t1 < t2.
        self.assertEqual(incremental_cat1.recommendation_count, 4)
        self.assertEqual(incremental_cat1.first_recommended_at, t3.isoformat())
        self.assertEqual(incremental_cat1.last_recommended_at, t2.isoformat())
        self.assertEqual(incremental_cat2.first_recommended_at, t3.isoformat())
        self.assertEqual(incremental_cat2.last_recommended_at, t3.isoformat())
        self.assertEqual(incremental_cat2.recommendation_count, 1)
        # Re-run B over the same history: zero drift against the incremental result.
        marker = "-- B. Recommendation-history projection"
        backfill_b = marker + migration_sql().split(marker, 1)[1]
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(f"BEGIN IMMEDIATE;\n{backfill_b}\nCOMMIT;")
        with self.state_repo() as state:
            for incremental, canonical_id in (
                (incremental_cat1, TRK_CAT_1),
                (incremental_cat2, TRK_CAT_2),
            ):
                recomputed = state.get_state(canonical_id)
                assert recomputed is not None
                self.assertEqual(
                    recomputed.first_recommended_at, incremental.first_recommended_at
                )
                self.assertEqual(
                    recomputed.last_recommended_at, incremental.last_recommended_at
                )
                self.assertEqual(
                    recomputed.recommendation_count, incremental.recommendation_count
                )


class CatalogTermMemoryLookupTest(unittest.TestCase):
    """P15-S3-S2: ``find_states_by_term`` is the facts-only memory view behind the query
    tool -- "which remembered tracks did this normalized term yield", never a freshness
    verdict on the live catalog."""

    FRESH = "trk_cccccccc-4444-4444-8444-444444444444"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            seed_canonical(repository, TRK_CAT_1)
            seed_canonical(repository, TRK_CAT_2)
            seed_catalog_binding(repository, TRK_CAT_1)
            seed_catalog_binding(repository, TRK_CAT_2)
        with self.state_repo() as state:
            state.ensure_state(TRK_CAT_1, source_system=ITUNES)
            state.ensure_state(TRK_CAT_2, source_system=ITUNES)

    def state_repo(self) -> CatalogTrackStateRepository:
        return CatalogTrackStateRepository(self.database_path)

    def test_matched_rows_carry_term_counts_in_recency_order(self) -> None:
        t1 = datetime(2026, 8, 20, 9, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                TRK_CAT_1, source_system=ITUNES, term="J-Pop", now=t1
            )
            state.record_discovery_occurrence(
                TRK_CAT_1, source_system=ITUNES, term="j-pop", now=t1 + timedelta(minutes=1)
            )
            state.record_discovery_occurrence(
                TRK_CAT_1, source_system=ITUNES, term="Yorushika", now=t1 + timedelta(minutes=2)
            )
            state.record_discovery_occurrence(
                TRK_CAT_2, source_system=ITUNES, term="J-Pop", now=t1 + timedelta(minutes=30)
            )
            matches = state.find_states_by_term("  J-POP ")
        # Most recently discovered first: CAT_2 (t1+30m) beats CAT_1 (t1+2m).
        self.assertEqual([m.canonical_id for m in matches], [TRK_CAT_2, TRK_CAT_1])
        self.assertEqual(matches[0].discovery_terms["j-pop"], 1)
        self.assertEqual(matches[1].discovery_terms["j-pop"], 2)
        # Normalized exact-key matching only: no substring hits, blank yields nothing.
        with self.state_repo() as state:
            self.assertEqual(state.find_states_by_term("pop"), ())
            self.assertEqual(
                [m.canonical_id for m in state.find_states_by_term("yorushika")],
                [TRK_CAT_1],
            )
            self.assertEqual(state.find_states_by_term(None), ())
            self.assertEqual(state.find_states_by_term("   "), ())

    def test_same_last_discovered_ties_break_on_canonical_id_desc(self) -> None:
        fixed = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                TRK_CAT_1, source_system=ITUNES, term="refresh", now=fixed
            )
            state.record_discovery_occurrence(
                TRK_CAT_2, source_system=ITUNES, term="refresh", now=fixed
            )
            matches = state.find_states_by_term("refresh")
        # Identical discovery instants: the canonical-id tie-break decides deterministically.
        self.assertEqual([m.canonical_id for m in matches], [TRK_CAT_2, TRK_CAT_1])

    def test_json_special_char_terms_match_and_backfilled_rows_never_do(self) -> None:
        fixed = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
        special = 'say "hi" \\ once'
        with self.state_repo() as state:
            state.record_discovery_occurrence(
                TRK_CAT_1, source_system=ITUNES, term=special, now=fixed
            )
            self.assertEqual(
                [m.canonical_id for m in state.find_states_by_term(special)],
                [TRK_CAT_1],
            )
        # A row with honestly empty term memory (backfilled shape) never matches anything.
        with CanonicalRepository(self.database_path) as repository:
            seed_canonical(repository, self.FRESH)
            seed_catalog_binding(repository, self.FRESH)
        with self.state_repo() as state:
            state.ensure_state(self.FRESH, source_system=ITUNES)
            self.assertEqual(state.find_states_by_term("j-pop"), ())
            self.assertEqual(
                [m.canonical_id for m in state.find_states_by_term(special)],
                [TRK_CAT_1],
            )