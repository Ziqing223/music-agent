"""P13-C02: read-side confidence derivation for directional track preferences.

These tests exercise the full reliability loop through the real domain APIs: durable
observations are recorded, the confidence claim is derived read-only from the heads and
revisions, and the production aggregation folds it into a bounded reliability score. They prove
the four required scenarios -- explicit liking -> high reliability, weak evidence -> low
reliability, conflicting evidence -> reduced reliability, and insufficient evidence -> no claim
-- plus the guardrails: freshness stays descriptive (no temporal decay), same-value confirmations
never inflate the evidence count, and absent claims map to ``None``.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.confidence import (
    ClaimScope,
    ConfidenceAggregationPolicy,
    ConfidenceValidationError,
    Contradiction,
)
from music_agent.confidence_derivation import (
    ConfidenceDerivationError,
    ConfidenceDerivationPolicy,
    ConservativeConfidencePolicy,
    derive_confidence_claim,
)
from music_agent.direct_track_preference import DirectPreferenceMagnitudePolicy
from music_agent.identity import EntityType
from music_agent.learning_policy import (
    FEEDBACK_LEARNING_SOURCE_SYSTEM,
    IMPLICIT_FEEDBACK_PROVENANCE,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_ingestion import ingest_track_observation
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    SignalIdentity,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_query import (
    PreferenceQueryError,
    query_track_preference,
)
from music_agent.preference_signal import RatingBandPolicy
from music_agent.source_observation import (
    ObservedValue,
    SourceObservation,
    SourcePresence,
)

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
APPLE_SOURCE = "apple_music"
OBSERVED_AT = "2026-08-16T00:00:00+00:00"


class _CappedFamiliarity:
    """Deterministic play-count -> familiarity magnitude normalization."""

    def __init__(self, divisor: float = 10.0) -> None:
        self._divisor = divisor

    def normalize(self, play_count: int) -> float:
        return min(1.0, play_count / self._divisor)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def derivation_policy() -> ConfidenceDerivationPolicy:
    return ConfidenceDerivationPolicy()


def aggregation_policy() -> ConservativeConfidencePolicy:
    return ConservativeConfidencePolicy()


def score_of(repository: PreferencePersistenceRepository, *, source_system: str = APPLE_SOURCE,
             now: datetime | None = None) -> float | None:
    state = query(
        repository,
        source_system=source_system,
        now=now,
    )
    claim = state.confidence_claim
    return aggregation_policy().aggregate(claim)


def query(repository: PreferencePersistenceRepository, *, source_system: str = APPLE_SOURCE,
          confidence_policy: ConfidenceDerivationPolicy | None = None,
          now: datetime | None = None):
    if confidence_policy is None:
        confidence_policy = derivation_policy()
    return query_track_preference(
        repository,
        track_target(),
        rating_policy=RatingBandPolicy(positive_threshold=70, negative_threshold=30),
        magnitude_policy=DirectPreferenceMagnitudePolicy(0.9, 0.8),
        familiarity_policy=_CappedFamiliarity(10.0),
        source_system=source_system,
        now=now,
        confidence_policy=confidence_policy,
    )


def observation(**fields: ObservedValue) -> SourceObservation:
    return SourceObservation(EntityType.TRACK, TRACK_ID, APPLE_SOURCE, fields, SourcePresence.PRESENT)


def record(repository: PreferencePersistenceRepository, signal_path: str, value: object,
           *, source_system: str = APPLE_SOURCE,
           provenance: str = DIRECT_OBSERVATION_PROVENANCE,
           observed_at: str = OBSERVED_AT) -> None:
    repository.record_observation(
        SignalIdentity(track_target(), source_system, signal_path),
        ObservedValue.value(value),
        observed_at=observed_at,
        provenance=provenance,
    )


class HighReliabilityTest(unittest.TestCase):
    """明确的喜欢行为 -> 高可靠."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_explicit_apple_favorite_carries_high_reliability(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            ingest_track_observation(
                repository,
                observation(
                    **{
                        "library_state.favorited": ObservedValue.value(True),
                        "library_state.disliked": ObservedValue.value(False),
                        "library_state.rating": ObservedValue.value(85),
                        "library_state.play_count": ObservedValue.value(12),
                    }
                ),
                observed_at=OBSERVED_AT,
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query(repository)
            claim = state.confidence_claim
            self.assertIsNotNone(claim)
            self.assertIs(claim.scope, ClaimScope.CURRENT_PREFERENCE)
            components = claim.components
            self.assertEqual(components.quantity, 2)  # favorited + rating, both directional
            self.assertIs(components.contradiction, Contradiction.NONE)
            self.assertEqual(components.consistency, 1.0)
            self.assertEqual(components.source_reliability, 1.0)
            self.assertEqual(components.inference_distance, 0.0)
            score = aggregation_policy().aggregate(claim)
            self.assertGreaterEqual(score, 0.9)

    def test_claim_is_attached_to_state_and_explanation(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "favorited", True)

        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query(repository)
            self.assertIsNotNone(state.confidence_claim)
            self.assertEqual(state.explanation.confidence_claims, (state.confidence_claim,))


class WeakEvidenceTest(unittest.TestCase):
    """弱证据 -> 低可靠."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_single_implicit_feedback_scores_low(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(
                repository,
                "favorited",
                True,
                source_system=FEEDBACK_LEARNING_SOURCE_SYSTEM,
                provenance=IMPLICIT_FEEDBACK_PROVENANCE,
            )

        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query(repository, source_system=FEEDBACK_LEARNING_SOURCE_SYSTEM)
            claim = state.confidence_claim
            self.assertIsNotNone(claim)
            self.assertEqual(claim.components.quantity, 1)
            self.assertEqual(claim.components.quality, 0.7)
            score = aggregation_policy().aggregate(claim)
            self.assertLess(score, 0.5)
            self.assertLess(score, 0.9)

    def test_weak_evidence_scores_below_strong_explicit_evidence(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            # same repository, two sources: explicit Apple favorite vs implicit feedback for
            # a second identity. The explicit, multi-signal track must score strictly higher.
            ingest_track_observation(
                repository,
                observation(
                    **{
                        "library_state.favorited": ObservedValue.value(True),
                        "library_state.disliked": ObservedValue.value(False),
                        "library_state.rating": ObservedValue.value(85),
                    }
                ),
                observed_at=OBSERVED_AT,
            )
            record(
                repository,
                "favorited",
                True,
                source_system=FEEDBACK_LEARNING_SOURCE_SYSTEM,
                provenance=IMPLICIT_FEEDBACK_PROVENANCE,
            )
            strong = score_of(repository)
            weak = score_of(repository, source_system=FEEDBACK_LEARNING_SOURCE_SYSTEM)
        self.assertGreater(strong, weak)


class ConflictEvidenceTest(unittest.TestCase):
    """冲突证据 -> 降低可靠（而不是抹除证据）."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_conflicting_evidence_marks_contradiction_and_reduces_reliability(self) -> None:
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

        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query(repository)
            claim = state.confidence_claim
            self.assertIsNotNone(claim)  # a conflict is evidence, never dropped
            self.assertIs(claim.components.contradiction, Contradiction.PRESENT)
            self.assertEqual(claim.components.quantity, 2)
            conflicted = aggregation_policy().aggregate(claim)
        # the same single signal without the opposing evidence scores higher
        single_claim = self._single_favorite_claim()
        clean = aggregation_policy().aggregate(single_claim)
        self.assertIsNotNone(conflicted)
        self.assertIsNotNone(clean)
        self.assertLess(conflicted, clean)

    def _single_favorite_claim(self):
        """The same favorited signal without opposing evidence, in an isolated store."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.sqlite3"
            with PreferencePersistenceRepository(path) as repository:
                record(repository, "favorited", True)
            with PreferencePersistenceRepository(path) as repository:
                return query(repository).confidence_claim


class InsufficientEvidenceTest(unittest.TestCase):
    """无足够证据 -> 不生成可靠程度."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_empty_store_derives_no_claim(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            claim = query(repository).confidence_claim
            self.assertIsNone(claim)
            self.assertIsNone(aggregation_policy().aggregate(claim))

    def test_play_count_only_derives_no_claim(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "play_count", 12)
        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query(repository)
            self.assertIsNone(state.confidence_claim)
            self.assertEqual(state.explanation.confidence_claims, ())

    def test_directionless_rating_derives_no_claim(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "rating", 50)
        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertIsNone(query(repository).confidence_claim)

    def test_default_query_without_policy_carries_no_claim(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "favorited", True)
        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query_track_preference(
                repository,
                track_target(),
                rating_policy=RatingBandPolicy(positive_threshold=70, negative_threshold=30),
                magnitude_policy=DirectPreferenceMagnitudePolicy(0.9, 0.8),
                familiarity_policy=_CappedFamiliarity(10.0),
            )
            self.assertIsNone(state.confidence_claim)
            self.assertEqual(state.explanation.confidence_claims, ())


class FreshnessDescriptionTest(unittest.TestCase):
    """时间信息仅作描述：新鲜度不衰减可靠度分数."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_freshness_describes_but_never_decays_the_score(self) -> None:
        observed_at = "2026-01-01T00:00:00+00:00"
        policy = ConfidenceDerivationPolicy(freshness_window=timedelta(days=30))
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "favorited", True, observed_at=observed_at)
        with PreferencePersistenceRepository(self.database_path) as repository:
            fresh = query(
                repository,
                confidence_policy=policy,
                now=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            stale = query(
                repository,
                confidence_policy=policy,
                now=datetime(2026, 12, 31, tzinfo=timezone.utc),
            )
        self.assertLess(stale.confidence_claim.components.freshness,
                        fresh.confidence_claim.components.freshness)
        self.assertGreater(fresh.confidence_claim.components.freshness, 0.9)
        self.assertEqual(
            aggregation_policy().aggregate(stale.confidence_claim),
            aggregation_policy().aggregate(fresh.confidence_claim),
        )


class DerivationDetailsTest(unittest.TestCase):
    """组件推导的边界细节."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_same_value_confirmation_does_not_inflate_quantity(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "favorited", True, observed_at=OBSERVED_AT)
            record(repository, "favorited", True, observed_at="2026-08-17T00:00:00+00:00")
            claim = derive_confidence_claim(
                repository,
                track_target(),
                query(repository).direct_preference.strength,
                rating_policy=RatingBandPolicy(positive_threshold=70, negative_threshold=30),
                policy=derivation_policy(),
                source_system=APPLE_SOURCE,
            )
            self.assertEqual(claim.components.quantity, 1)

    def test_unknown_provenance_uses_default_quality(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            record(repository, "favorited", True, provenance="unknown_label")
            claim = query(repository).confidence_claim
            self.assertEqual(claim.components.quality, 0.5)

    def test_aggregate_maps_absent_claim_to_none(self) -> None:
        self.assertIsNone(aggregation_policy().aggregate(None))

    def test_production_aggregation_implements_the_s5_seam(self) -> None:
        self.assertIsInstance(aggregation_policy(), ConfidenceAggregationPolicy)

    def test_derivation_rejects_invalid_inputs(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            for bad in (None, "repository", object(), derivation_policy()):
                with self.subTest(bad=bad):
                    with self.assertRaises(ConfidenceDerivationError):
                        derive_confidence_claim(
                            bad,
                            track_target(),
                            query(repository).direct_preference.strength,
                            rating_policy=RatingBandPolicy(70, 30),
                            policy=derivation_policy(),
                            source_system=APPLE_SOURCE,
                        )

    def test_query_rejects_a_malformed_confidence_policy(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            with self.assertRaises(PreferenceQueryError):
                query_track_preference(
                    repository,
                    track_target(),
                    rating_policy=RatingBandPolicy(70, 30),
                    magnitude_policy=DirectPreferenceMagnitudePolicy(0.9, 0.8),
                    familiarity_policy=_CappedFamiliarity(),
                    confidence_policy=object(),
                )


class PolicyValidationTest(unittest.TestCase):
    """标定策略 fail-closed 校验."""

    def test_derivation_policy_rejects_out_of_range_quality(self) -> None:
        for bad in (1.5, -0.1, True, "high"):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    ConfidenceDerivationPolicy(quality_by_provenance={"a": bad})

    def test_derivation_policy_rejects_non_table(self) -> None:
        with self.assertRaises(ConfidenceValidationError):
            ConfidenceDerivationPolicy(quality_by_provenance="direct")

    def test_derivation_policy_rejects_bad_window(self) -> None:
        for bad in (timedelta(0), timedelta(days=-1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfidenceValidationError):
                    ConfidenceDerivationPolicy(freshness_window=bad)

    def test_aggregation_policy_rejects_bad_penalty_and_weight(self) -> None:
        with self.assertRaises(ConfidenceValidationError):
            ConservativeConfidencePolicy(conflict_penalty=1.5)
        with self.assertRaises(ConfidenceValidationError):
            ConservativeConfidencePolicy(freshness_weight=-0.1)
        with self.assertRaises(ConfidenceValidationError):
            ConservativeConfidencePolicy(quantity_full=0)


if __name__ == "__main__":
    unittest.main()