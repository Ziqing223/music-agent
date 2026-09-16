"""P08.2: durable feedback observation persistence and history.

These tests prove the append-only feedback-observation history: one immutable
``feedback_observations`` row per :class:`FeedbackObservation`, persisted through the canonical
``encode_feedback_observation`` interchange with ``feedback_id`` / ``kind`` / ``contract_version``
/ ``observed_at`` / ``duplicate_key`` mirrored as columns. They cover migration, the save/get
round-trip (including target-only, recommendation-only, and fully-attributed observations),
deterministic chronological history ordering with a documented tie-break, duplicate rejection at
the persistence boundary (reused feedback_id and same-event duplicate key, including across
distinct feedback_ids), missing-observation lookup, fail-closed decoding of corrupted rows and
disagreeing mirrors, hard immutability via the SQL triggers, the absence of any update/delete
path, and the isolation of feedback history from the P06 preference tables and the P07
recommendation history.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackKind,
    FeedbackObservation,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
    encode_feedback_observation,
)
from music_agent.feedback_history_repository import (
    CorruptFeedbackHistoryError,
    DuplicateFeedbackObservationError,
    FeedbackHistoryRepository,
    FeedbackHistoryRepositoryError,
    encode_duplicate_key,
)
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
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS
from music_agent.source_observation import ObservedValue

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
FEEDBACK_ID = "fbk_44444444-4444-4444-8444-444444444444"
FEEDBACK_ID_2 = "fbk_55555555-5555-4555-8555-555555555555"
FEEDBACK_ID_3 = "fbk_66666666-6666-4666-8666-666666666666"
FEEDBACK_ID_4 = "fbk_77777777-7777-4777-8777-777777777777"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def source(
    system: str = "recommendation_ui", path: str = "card_actions"
) -> FeedbackSourceReference:
    return FeedbackSourceReference(system, path)


def recommendation_ref() -> FeedbackRecommendationReference:
    return FeedbackRecommendationReference(RUN_ID, CANDIDATE_ID)


def observation(
    *,
    feedback_id: str = FEEDBACK_ID,
    kind: FeedbackKind = FeedbackKind.LIKED,
    observed_at: datetime = NOW,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
    event_at: datetime | None = None,
    source_event_id: str | None = None,
) -> FeedbackObservation:
    if target is None and recommendation is None:
        target = track_target()
    return assemble_feedback_observation(
        feedback_id=feedback_id,
        kind=kind,
        source=source(),
        observed_at=observed_at,
        target=target,
        recommendation=recommendation,
        attribution=attribution,
        event_at=event_at,
        source_event_id=source_event_id,
    )


def at(hour: int) -> datetime:
    return datetime(2026, 8, 16, hour, 0, 0, tzinfo=timezone.utc)


# P07 fixtures reused by the historical-upgrade and isolation tests.


def direct_input() -> PreferenceInput:
    strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
    return PreferenceInput.from_direct(DerivedPreference(track_target(), strength))


def result() -> object:
    context = RecommendationContext(NOW, (direct_input(),))
    request = RecommendationRequest(context, RecommendedItemKind.TRACK, 5)
    candidate = Candidate(
        CANDIDATE_ID,
        track_target(),
        CandidateSourceReference("candidate_gen", "preference_match"),
    )
    item = RecommendationItem(
        candidate, ScoreBreakdown(0.9, (ScoreComponent("preference_match", 0.9),))
    )
    return assemble_recommendation_result(
        request, (item,), run_id=RUN_ID, produced_at=NOW
    )


class FeedbackHistoryRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_rows(self) -> list[sqlite3.Row]:
        with FeedbackHistoryRepository(self.database_path) as repository:
            return list(
                repository._connection.execute(
                    "SELECT * FROM feedback_observations ORDER BY feedback_id"
                )
            )

    # --- schema / migration ------------------------------------------------

    def test_fresh_database_reaches_current_version_with_history_table(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("feedback_observations", tables)
            indexes = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            self.assertIn("ix_feedback_observations_observed_at", indexes)
            self.assertIn("ux_feedback_observations_duplicate_key", indexes)

    def test_feedback_observations_table_has_only_the_frozen_columns(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(feedback_observations)"
                )
            }
        self.assertEqual(
            columns,
            {
                "feedback_id",
                "encoded_observation",
                "kind",
                "contract_version",
                "observed_at",
                "duplicate_key",
                "created_at",
            },
        )

    def test_immutability_triggers_exist(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            triggers = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
        self.assertIn("trg_feedback_observations_immutable_update", triggers)
        self.assertIn("trg_feedback_observations_immutable_delete", triggers)

    def test_v12_store_upgrades_to_current_without_changing_prior_state(self) -> None:
        identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:12]):
            with PreferencePersistenceRepository(self.database_path) as repository:
                repository.record_observation(
                    identity, ObservedValue.value(True), observed_at=NOW.isoformat()
                )
                self.assertEqual(repository.schema_version, 12)
            with RecommendationHistoryRepository(self.database_path) as repository:
                repository.save_result(result())
                self.assertEqual(repository.schema_version, 12)

        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            repository.save_observation(observation())
            self.assertEqual(
                repository.get_observation(FEEDBACK_ID), observation()
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertIs(head.current_semantic_value, True)
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.get_result(RUN_ID), result())

    # --- save / get round-trip --------------------------------------------

    def test_save_get_round_trip_equals_original(self) -> None:
        fully_attributed = observation(
            kind=FeedbackKind.LIKED,
            target=track_target(),
            recommendation=recommendation_ref(),
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.ATTRIBUTED),
            event_at=at(23),
            source_event_id="ui_evt_9",
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(fully_attributed)
            loaded = repository.get_observation(fully_attributed.feedback_id)
        self.assertEqual(loaded, fully_attributed)

    def test_target_only_observation_round_trips(self) -> None:
        skipped = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(skipped)
            loaded = repository.get_observation(skipped.feedback_id)
        self.assertEqual(loaded, skipped)
        self.assertIsNone(loaded.recommendation)
        self.assertIsNone(loaded.attribution)

    def test_recommendation_only_observation_round_trips(self) -> None:
        direction = observation(
            kind=FeedbackKind.DIRECTION_GOOD, recommendation=recommendation_ref()
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(direction)
            loaded = repository.get_observation(direction.feedback_id)
        self.assertEqual(loaded, direction)
        self.assertIsNone(loaded.target)

    def test_observations_reload_after_reopen(self) -> None:
        skipped = observation(
            feedback_id=FEEDBACK_ID_2, kind=FeedbackKind.SKIPPED, target=track_target(), observed_at=at(1)
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(skipped)
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertEqual(repository.get_observation(skipped.feedback_id), skipped)
            self.assertEqual(list(repository.list_observations()), [skipped])

    # --- ordering ----------------------------------------------------------

    def test_list_observations_orders_by_observed_at_ascending(self) -> None:
        early = observation(
            feedback_id=FEEDBACK_ID_2, observed_at=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        )
        middle = observation(
            feedback_id=FEEDBACK_ID, observed_at=datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)
        )
        late = observation(
            feedback_id=FEEDBACK_ID_3, observed_at=datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc)
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(middle)
            repository.save_observation(late)
            repository.save_observation(early)
            observations = repository.list_observations()
        self.assertIsInstance(observations, tuple)
        self.assertEqual(
            [o.feedback_id for o in observations],
            [FEEDBACK_ID_2, FEEDBACK_ID, FEEDBACK_ID_3],
        )

    def test_list_observations_ties_break_deterministically_by_feedback_id(self) -> None:
        same_instant = at(12)
        later_id = observation(
            feedback_id=FEEDBACK_ID_2,
            observed_at=same_instant,
            source_event_id="ui_evt_b",
        )
        earlier_id = observation(
            feedback_id=FEEDBACK_ID,
            observed_at=same_instant,
            source_event_id="ui_evt_a",
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(later_id)
            repository.save_observation(earlier_id)
            observations = repository.list_observations()
        # Equal observed_at: feedback_id ascending is the documented tie-break.
        self.assertEqual(
            [o.feedback_id for o in observations], [FEEDBACK_ID, FEEDBACK_ID_2]
        )

    def test_list_observations_orders_chronologically_across_utc_offsets(self) -> None:
        # Stored TEXT order contradicts chronological order here: the stored strings
        # "2026-08-16T10:00:00-05:00" and "2026-08-16T12:00:00+14:00" sort lexicographically with
        # -05:00 first, but the +14:00 observation is the earlier instant. The tuple must follow
        # the datetimes, not the TEXT.
        plus_fourteen = observation(
            feedback_id=FEEDBACK_ID,
            observed_at=datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone(timedelta(hours=14))),
        )
        minus_five = observation(
            feedback_id=FEEDBACK_ID_2,
            observed_at=datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone(timedelta(hours=-5))),
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(minus_five)
            repository.save_observation(plus_fourteen)
            observations = repository.list_observations()
        self.assertEqual(
            [o.feedback_id for o in observations], [FEEDBACK_ID, FEEDBACK_ID_2]
        )

    # --- duplicates --------------------------------------------------------

    def test_reusing_the_same_feedback_id_raises_specific_error(self) -> None:
        first = observation()
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            with self.assertRaises(DuplicateFeedbackObservationError):
                repository.save_observation(first)
            self.assertEqual(len(repository.list_observations()), 1)

    def test_same_event_under_distinct_feedback_ids_is_rejected_by_duplicate_key(self) -> None:
        first = observation(
            kind=FeedbackKind.SKIPPED, target=track_target(), source_event_id="playback_evt_7"
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            with self.assertRaises(DuplicateFeedbackObservationError):
                repository.save_observation(
                    observation(
                        feedback_id=FEEDBACK_ID_2,
                        kind=FeedbackKind.SKIPPED,
                        target=track_target(),
                        source_event_id="playback_evt_7",
                    )
                )
            self.assertEqual(len(repository.list_observations()), 1)
            self.assertEqual(
                repository.get_observation(FEEDBACK_ID),
                first,
            )

    def test_same_event_without_source_event_id_is_rejected_by_full_fields_key(self) -> None:
        first = observation(kind=FeedbackKind.COMPLETED, target=track_target())
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            with self.assertRaises(DuplicateFeedbackObservationError):
                repository.save_observation(
                    observation(
                        feedback_id=FEEDBACK_ID_2,
                        kind=FeedbackKind.COMPLETED,
                        target=track_target(),
                    )
                )
            self.assertEqual(len(repository.list_observations()), 1)

    def test_reused_feedback_id_with_different_content_is_rejected(self) -> None:
        first = observation()
        second = observation(
            feedback_id=FEEDBACK_ID, kind=FeedbackKind.SKIPPED, target=track_target(), observed_at=at(1)
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            with self.assertRaises(DuplicateFeedbackObservationError):
                repository.save_observation(second)

    def test_distinct_events_without_source_event_id_are_stored_separately(self) -> None:
        first = observation(kind=FeedbackKind.PLAYED, target=track_target())
        later = observation(
            feedback_id=FEEDBACK_ID_2,
            kind=FeedbackKind.PLAYED,
            target=track_target(),
            observed_at=at(1),
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            repository.save_observation(later)
            self.assertEqual(len(repository.list_observations()), 2)

    def test_same_source_event_id_from_different_systems_is_not_a_duplicate(self) -> None:
        first = observation(
            kind=FeedbackKind.SKIPPED, target=track_target(), source_event_id="playback_evt_7"
        )
        other_system = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID_2,
            kind=FeedbackKind.SKIPPED,
            source=FeedbackSourceReference("watch_app", "card_actions"),
            observed_at=NOW,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            repository.save_observation(other_system)
            self.assertEqual(len(repository.list_observations()), 2)

    def test_different_event_at_does_not_distinguish_the_same_event(self) -> None:
        first = observation(
            kind=FeedbackKind.SKIPPED, target=track_target(), source_event_id="playback_evt_7"
        )
        second = observation(
            feedback_id=FEEDBACK_ID_2,
            kind=FeedbackKind.SKIPPED,
            target=track_target(),
            source_event_id="playback_evt_7",
            event_at=at(3),
        )
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(first)
            with self.assertRaises(DuplicateFeedbackObservationError):
                repository.save_observation(second)

    # --- missing / corruption ----------------------------------------------

    def test_get_observation_for_missing_feedback_returns_none(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_observation(FEEDBACK_ID_4))

    def test_corrupted_encoded_observation_fails_closed(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO feedback_observations(
                    feedback_id, encoded_observation, kind, contract_version,
                    observed_at, duplicate_key
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (FEEDBACK_ID, "not canonical json", "liked", 1, NOW.isoformat(), "corrupt-key"),
            )
            with self.assertRaises(CorruptFeedbackHistoryError):
                repository.get_observation(FEEDBACK_ID)
            with self.assertRaises(CorruptFeedbackHistoryError):
                repository.list_observations()

    def test_kind_mirror_mismatch_fails_closed(self) -> None:
        liked = observation()
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO feedback_observations(
                    feedback_id, encoded_observation, kind, contract_version,
                    observed_at, duplicate_key
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    liked.feedback_id,
                    _encoded(liked),
                    "skipped",
                    liked.contract_version,
                    liked.observed_at.isoformat(),
                    encode_duplicate_key(liked),
                ),
            )
            with self.assertRaises(CorruptFeedbackHistoryError):
                repository.get_observation(liked.feedback_id)

    def test_duplicate_key_mirror_mismatch_fails_closed(self) -> None:
        liked = observation()
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO feedback_observations(
                    feedback_id, encoded_observation, kind, contract_version,
                    observed_at, duplicate_key
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    liked.feedback_id,
                    _encoded(liked),
                    liked.kind.value,
                    liked.contract_version,
                    liked.observed_at.isoformat(),
                    "tampered-key",
                ),
            )
            with self.assertRaises(CorruptFeedbackHistoryError):
                repository.get_observation(liked.feedback_id)

    # --- immutability ------------------------------------------------------

    def test_raw_sql_update_and_delete_raise(self) -> None:
        liked = observation()
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(liked)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE feedback_observations SET observed_at=? WHERE feedback_id=?",
                    ("2026-08-17T00:00:00+00:00", liked.feedback_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute("DELETE FROM feedback_observations")
        self.assertEqual(len(self.run_rows()), 1)

    def test_repository_exposes_no_update_or_delete_path(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertFalse(hasattr(repository, "update_observation"))
            self.assertFalse(hasattr(repository, "delete_observation"))

    # --- mirrored columns --------------------------------------------------

    def test_mirrored_columns_match_observation_fields_for_every_row(self) -> None:
        observations = [
            observation(),
            observation(
                feedback_id=FEEDBACK_ID_2,
                kind=FeedbackKind.SKIPPED,
                target=track_target(),
                recommendation=recommendation_ref(),
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
                source_event_id="playback_evt_8",
                observed_at=at(1),
            ),
        ]
        with FeedbackHistoryRepository(self.database_path) as repository:
            for obs in observations:
                repository.save_observation(obs)
            rows = list(
                repository._connection.execute(
                    "SELECT * FROM feedback_observations ORDER BY feedback_id"
                )
            )
        for obs, row in zip(sorted(observations, key=lambda o: o.feedback_id), rows):
            with self.subTest(feedback_id=obs.feedback_id):
                self.assertEqual(row["feedback_id"], obs.feedback_id)
                self.assertEqual(row["kind"], obs.kind.value)
                self.assertEqual(row["contract_version"], obs.contract_version)
                self.assertEqual(row["observed_at"], obs.observed_at.isoformat())
                self.assertEqual(row["duplicate_key"], encode_duplicate_key(obs))

    # --- input validation --------------------------------------------------

    def test_save_observation_rejects_non_observation_input(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            for bad in ("not-an-observation", None, {"feedback_id": FEEDBACK_ID}, 42):
                with self.subTest(bad=bad):
                    with self.assertRaises(FeedbackHistoryRepositoryError):
                        repository.save_observation(bad)  # type: ignore[arg-type]
            self.assertEqual(len(repository.list_observations()), 0)

    # --- isolation from P06 / P07 ------------------------------------------

    def test_feedback_history_leaves_preference_and_recommendation_tables_untouched(self) -> None:
        with FeedbackHistoryRepository(self.database_path) as repository:
            repository.save_observation(observation())
            for table in (
                "preference_signal_heads",
                "preference_evidence_revisions",
                "recommendation_runs",
                "canonical_entities",
            ):
                count = repository._connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                self.assertEqual(count, 0, table)

    def test_preference_and_recommendation_writes_leave_feedback_history_untouched(self) -> None:
        identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                identity, ObservedValue.value(True), observed_at=NOW.isoformat()
            )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(result())
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertEqual(len(repository.list_observations()), 0)
            count = repository._connection.execute(
                "SELECT COUNT(*) FROM feedback_observations"
            ).fetchone()[0]
            self.assertEqual(count, 0)


class EncodeDuplicateKeyTest(unittest.TestCase):
    def test_source_event_id_form_is_scoped_to_the_source_system(self) -> None:
        first = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID,
            kind=FeedbackKind.SKIPPED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=NOW,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        self.assertEqual(
            encode_duplicate_key(first),
            '["recommendation_ui","playback_evt_7"]',
        )

    def test_full_fields_form_includes_all_key_fields(self) -> None:
        obs = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID,
            kind=FeedbackKind.COMPLETED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=NOW,
            target=track_target(),
        )
        self.assertEqual(
            encode_duplicate_key(obs),
            '["recommendation_ui","card_actions","completed",'
            '{"kind":"track","target_id":"trk_11111111-1111-4111-8111-111111111111"},'
            "null,null,"
            '"2026-08-16T00:00:00+00:00"]',
        )

    def test_equal_instants_in_different_offsets_encode_identically(self) -> None:
        plus_fourteen = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID,
            kind=FeedbackKind.PLAYED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone(timedelta(hours=14))),
            target=track_target(),
        )
        utc = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID_2,
            kind=FeedbackKind.PLAYED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=datetime(2026, 8, 15, 22, 0, 0, tzinfo=timezone.utc),
            target=track_target(),
        )
        self.assertEqual(encode_duplicate_key(plus_fourteen), encode_duplicate_key(utc))

    def test_distinct_kinds_encode_distinct_keys(self) -> None:
        skipped = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID,
            kind=FeedbackKind.SKIPPED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=NOW,
            target=track_target(),
        )
        completed = assemble_feedback_observation(
            feedback_id=FEEDBACK_ID_2,
            kind=FeedbackKind.COMPLETED,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=NOW,
            target=track_target(),
        )
        self.assertNotEqual(
            encode_duplicate_key(skipped), encode_duplicate_key(completed)
        )

    def test_rejects_non_observation(self) -> None:
        with self.assertRaises(FeedbackHistoryRepositoryError):
            encode_duplicate_key("not an observation")  # type: ignore[arg-type]


def _encoded(obs: FeedbackObservation) -> str:
    return encode_feedback_observation(obs)


if __name__ == "__main__":
    unittest.main()
