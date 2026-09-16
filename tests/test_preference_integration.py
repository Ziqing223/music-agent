"""P06 end-to-end integration: source observation -> persistence -> query-time reconstruction.

These tests exercise the full P06 loop through the real domain APIs: a ``SourceObservation`` is
ingested into the durable preference store, the repository is closed and reopened (restart), and the
query read model reconstructs the Track-level direct preference, familiarity, and structured
explanation. They also prove the write pipeline's idempotency, the three-state MISSING / NULL /
VALUE boundary, restart safety, and the isolation of preference state from canonical library state.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.direct_track_preference import DirectPreferenceMagnitudePolicy
from music_agent.familiarity import FamiliarityLevel
from music_agent.identity import EntityType
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_explainability import DerivationKind
from music_agent.preference_ingestion import (
    PreferenceIngestionError,
    ingest_track_observation,
)
from music_agent.preference_persistence import RevisionKind, SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_query import query_track_preference
from music_agent.preference_signal import RatingBandPolicy
from music_agent.preference_strength import PreferenceState
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import (
    ObservationState,
    ObservedValue,
    SourceObservation,
    SourcePresence,
)
from music_agent.temporal_evolution import TemporalRelation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
OBSERVED_AT = "2026-08-16T00:00:00+00:00"

class _CappedFamiliarity:
    """Deterministic play-count -> familiarity magnitude normalization."""

    def __init__(self, divisor: float = 10.0) -> None:
        self._divisor = divisor

    def normalize(self, play_count: int) -> float:
        return min(1.0, play_count / self._divisor)

class _FixedGapScope:
    """Contemporaneous iff two event times fall within a fixed gap of one another."""

    def __init__(self, gap_days: float = 7.0) -> None:
        self._gap = timedelta(days=gap_days)

    def contemporaneous(self, first: datetime, second: datetime, *, now: datetime) -> bool:
        return abs((first - second).total_seconds()) < self._gap.total_seconds()

def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)

def rating_policy() -> RatingBandPolicy:
    return RatingBandPolicy(positive_threshold=70, negative_threshold=30)

def magnitude_policy() -> DirectPreferenceMagnitudePolicy:
    return DirectPreferenceMagnitudePolicy(0.9, 0.8)

def familiarity_policy() -> _CappedFamiliarity:
    return _CappedFamiliarity(10.0)

def observation(**fields: ObservedValue) -> SourceObservation:
    return SourceObservation(EntityType.TRACK, TRACK_ID, "apple_music", fields, SourcePresence.PRESENT)

def full_observation() -> SourceObservation:
    return observation(
        **{
            "library_state.favorited": ObservedValue.value(True),
            "library_state.disliked": ObservedValue.value(False),
            "library_state.rating": ObservedValue.value(80),
            "library_state.play_count": ObservedValue.value(12),
        }
    )

class EndToEndIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def query(self, *, now: datetime | None = None) -> object:
        with PreferencePersistenceRepository(self.database_path) as repository:
            return query_track_preference(
                repository,
                track_target(),
                rating_policy=rating_policy(),
                magnitude_policy=magnitude_policy(),
                familiarity_policy=familiarity_policy(),
                source_system="apple_music",
                scope_policy=_FixedGapScope(),
                now=now,
            )

    # --- end-to-end reconstruction -----------------------------------------

    def test_source_observation_to_reloaded_query_state(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)

        state = self.query()
        self.assertEqual(state.direct_preference.strength.state, PreferenceState.POSITIVE)
        self.assertEqual(state.direct_preference.strength.magnitude, 0.9)
        self.assertEqual(state.direct_preference.target, track_target())

        self.assertIs(state.familiarity.level, FamiliarityLevel.KNOWN)
        self.assertEqual(state.familiarity.magnitude, 1.0)

        self.assertIs(state.explanation.derivation, DerivationKind.DIRECT)
        self.assertEqual(state.explanation.direct_preference.target, track_target())
        self.assertEqual(state.explanation.result, state.direct_preference.strength)
        self.assertEqual(len(state.explanation.signals), 3)
        self.assertEqual(state.explanation.inferred_contributions, ())

    def test_repeated_identical_observation_does_not_duplicate_evidence(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)
            second = ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)
            self.assertTrue(all(outcome.revision is None for outcome in second))

        # Re-open to count revisions directly.
        with PreferencePersistenceRepository(self.database_path) as repository:
            favorited = SignalIdentity(track_target(), "apple_music", "favorited")
            self.assertEqual(len(repository.list_revisions(favorited)), 1)
            self.assertEqual(repository.get_head(favorited).current_revision_sequence, 1)

    def test_transition_survives_restart(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)
            ingest_track_observation(
                repository,
                observation(**{"library_state.favorited": ObservedValue.value(False)}),
                observed_at="2026-08-16T00:00:01+00:00",
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            favorited = SignalIdentity(track_target(), "apple_music", "favorited")
            revisions = repository.list_revisions(favorited)
            self.assertEqual(
                [r.semantic_value for r in revisions], [True, False]
            )
            self.assertEqual(
                [r.revision_kind for r in revisions],
                [RevisionKind.BASELINE, RevisionKind.TRANSITION],
            )
            self.assertEqual(repository.get_head(favorited).current_revision_sequence, 2)

        # favorited=false resolves to NO_CLAIM, so the direct preference falls back to rating=80
        # (POSITIVE). Either way, the transition must be visible through the reloaded store.
        state = self.query()
        self.assertIs(state.direct_preference.strength.state, PreferenceState.POSITIVE)

    def test_missing_does_not_erase_prior_semantic_value(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)
            ingest_track_observation(
                repository,
                observation(**{"library_state.favorited": ObservedValue.missing()}),
                observed_at="2026-08-16T00:00:01+00:00",
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            favorited = SignalIdentity(track_target(), "apple_music", "favorited")
            head = repository.get_head(favorited)
            self.assertIs(head.current_semantic_value, True)
            self.assertIs(head.last_observed_state, ObservationState.MISSING)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertEqual(len(repository.list_revisions(favorited)), 1)

        # The prior semantic value True must still drive the preference despite the MISSING read.
        state = self.query()
        self.assertIs(state.direct_preference.strength.state, PreferenceState.POSITIVE)

    def test_null_preserves_prior_value_and_creates_no_revision(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)
            ingest_track_observation(
                repository,
                observation(**{"library_state.rating": ObservedValue.null()}),
                observed_at="2026-08-16T00:00:01+00:00",
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            rating = SignalIdentity(track_target(), "apple_music", "rating")
            head = repository.get_head(rating)
            self.assertEqual(head.current_semantic_value, 80)
            self.assertIs(head.last_observed_state, ObservationState.NULL)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertEqual(len(repository.list_revisions(rating)), 1)

        # The preserved rating 80 still resolves POSITIVE.
        state = self.query()
        self.assertIs(state.direct_preference.strength.state, PreferenceState.POSITIVE)

    def test_ingestion_does_not_pollute_canonical_state(self) -> None:
        fixture = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            before_counts = repository.counts()
            before_model = repository.load_model()

        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)
            self.assertEqual(repository.load_model(), before_model)

    def test_unsupported_signals_remain_absent(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(
                repository,
                observation(
                    **{
                        "library_state.skip_count": ObservedValue.value(3),
                        "library_state.last_played_at": ObservedValue.value(
                            "2026-08-16T00:00:00+00:00"
                        ),
                        "library_state.added_to_library_at": ObservedValue.value(
                            "2026-08-16T00:00:00+00:00"
                        ),
                    }
                ),
                observed_at=OBSERVED_AT,
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            for path in ("skip_count", "last_played_at", "added_to_library_at"):
                identity = SignalIdentity(track_target(), "apple_music", path)
                self.assertIsNone(repository.get_head(identity), path)

    # --- temporal interpretation -------------------------------------------

    def test_temporal_evolution_from_event_bearing_revisions(self) -> None:

        rating = SignalIdentity(track_target(), "apple_music", "rating")
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                rating,
                ObservedValue.value(80),
                observed_at="2026-01-01T00:00:00+00:00",
                event_at="2026-01-01T00:00:00+00:00",
            )
            repository.record_observation(
                rating,
                ObservedValue.value(20),
                observed_at="2026-06-01T00:00:00+00:00",
                event_at="2026-06-01T00:00:00+00:00",
            )

        now = datetime(2026, 8, 16, tzinfo=timezone.utc)
        state = self.query(now=now)
        self.assertIsNotNone(state.temporal_interpretation)
        self.assertIs(
            state.temporal_interpretation.relation, TemporalRelation.TEMPORAL_EVOLUTION
        )
        self.assertIs(state.direct_preference.strength.state, PreferenceState.NEGATIVE)

    def test_no_event_times_yields_no_temporal_interpretation(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(repository, full_observation(), observed_at=OBSERVED_AT)

        state = self.query(now=datetime(2026, 8, 16, tzinfo=timezone.utc))
        self.assertIsNone(state.temporal_interpretation)

    # --- conflict visibility -----------------------------------------------

    def test_direct_categorical_conflict_is_surfaced(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(
                repository,
                observation(
                    **{
                        "library_state.favorited": ObservedValue.value(True),
                        "library_state.disliked": ObservedValue.value(True),
                    }
                ),
                observed_at=OBSERVED_AT,
            )

        state = self.query()
        self.assertIs(state.direct_preference.strength.state, PreferenceState.CONFLICT)
        self.assertEqual(len(state.explanation.conflicts), 1)

class WritePipelineValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_non_track_observation_fails_closed(self) -> None:

        artist_observation = SourceObservation(
            EntityType.ARTIST, "art_11111111-1111-4111-8111-111111111111", "apple_music", {}
        )
        with PreferencePersistenceRepository(self.database_path) as repository:
            with self.assertRaises(PreferenceIngestionError):
                ingest_track_observation(repository, artist_observation, observed_at=OBSERVED_AT)

if __name__ == "__main__":
    unittest.main()
