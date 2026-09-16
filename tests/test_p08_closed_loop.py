"""P08.7: closed-loop learning validation over the real P06 + P07 + P08 implementation.

These tests prove the actual product-learning loop end to end on one shared SQLite store:
FeedbackObservation (P08.1, persisted P08.2) -> FeedbackInterpretation (P08.3) ->
LearningEffect (P08.4) -> LearningPolicy (P08.5, corrected policy v2) -> LearningApplication
(P08.6) -> durable P06 evidence -> P06 query derivation -> P07 recommendation. Nothing in the
learning boundary is mocked; fixtures are deterministic and controlled so recommendation changes
can be attributed to feedback learning. Recommendation comparisons use a deterministic
per-target score signature (target_id -> total), because tie ordering uses randomly generated
candidate ids by design (P07.2); set/signature membership is the controlled observable.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.direct_track_preference import DirectPreferenceMagnitudePolicy
from music_agent.familiarity import FamiliarityNormalizationPolicy
from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackKind,
    FeedbackObservation,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
)
from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.feedback_interpretation import InterpretationPolicy, interpret_observation
from music_agent.learning_application import (
    DuplicateLearningApplicationError,
    LearningApplicationRepository,
)
from music_agent.learning_effect import LearningEffectPolicy, derive_learning_effect
from music_agent.learning_policy import (
    EXPLICIT_FEEDBACK_PROVENANCE,
    IMPLICIT_FEEDBACK_PROVENANCE,
    LearningPolicy,
    ProposedPreferenceUpdate,
    ProposedUpdateKind,
    propose_preference_update,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_query import query_track_preference
from music_agent.preference_signal import RatingBandPolicy
from music_agent.recommendation_contract import (
    PreferenceInput,
    RecommendationContext,
    RecommendationRequest,
    RecommendedItemKind,
    generate_run_id,
)
from music_agent.recommendation_ranking import build_recommendation
from music_agent.source_observation import ObservedValue

TRACK_A = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TRACK_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

POSITIVE_MAGNITUDE = 0.9
SEED_PROVENANCE = "fixture_seed"


def target(track_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def rating_policy() -> RatingBandPolicy:
    return RatingBandPolicy(positive_threshold=70, negative_threshold=30)


def magnitude_policy() -> DirectPreferenceMagnitudePolicy:
    return DirectPreferenceMagnitudePolicy(POSITIVE_MAGNITUDE, 0.8)


@dataclass(frozen=True, slots=True)
class _CappedFamiliarity:
    cap: float

    def normalize(self, play_count: int) -> float:
        return min(1.0, play_count / self.cap)


def familiarity_policy() -> FamiliarityNormalizationPolicy:
    return _CappedFamiliarity(10.0)


class ClosedLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    # --- helpers ------------------------------------------------------------

    def seed_positive(self, track_id: str) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(track_id), "feedback_learning", "favorited"),
                ObservedValue.value(True),
                observed_at="2026-08-01T00:00:00+00:00",
                provenance=SEED_PROVENANCE,
            )

    def preference_input(self, track_id: str) -> PreferenceInput:
        with PreferencePersistenceRepository(self.database_path) as repository:
            state = query_track_preference(
                repository,
                target(track_id),
                rating_policy=rating_policy(),
                magnitude_policy=magnitude_policy(),
                familiarity_policy=familiarity_policy(),
                source_system="feedback_learning",
            )
        return PreferenceInput.from_direct(state.direct_preference)

    def recommend(self, track_ids: tuple[str, ...]) -> dict[str, float]:
        """Run the real P07 chain and return the deterministic per-target score signature."""
        context = RecommendationContext(
            NOW, tuple(self.preference_input(track_id) for track_id in track_ids)
        )
        request = RecommendationRequest(context, RecommendedItemKind.TRACK, 5)
        outcome = build_recommendation(
            request, run_id=generate_run_id(), produced_at=NOW
        )
        return {
            item.candidate.target.target_id: item.score.total
            for item in outcome.result.items
        }

    def run_full_pipeline(
        self, observation: FeedbackObservation
    ) -> ProposedPreferenceUpdate | None:
        """Persist, interpret, effect, and propose for one observation (the real P08 layers)."""
        with FeedbackHistoryRepository(self.database_path) as history:
            history.save_observation(observation)
        interpretation = interpret_observation(observation, InterpretationPolicy(1))
        effect = derive_learning_effect(interpretation, LearningEffectPolicy(1))
        return propose_preference_update(effect, LearningPolicy(2))

    def apply(self, proposal: ProposedPreferenceUpdate) -> object:
        with LearningApplicationRepository(self.database_path) as application:
            return application.apply(proposal, applied_at="2026-08-16T02:00:00+00:00")

    def feedback(
        self,
        kind: FeedbackKind,
        *,
        track_id: str = TRACK_A,
        feedback_id: str = "fbk_cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        attribution: FeedbackAttribution | None = None,
        recommendation_only: bool = False,
    ) -> FeedbackObservation:
        return assemble_feedback_observation(
            feedback_id=feedback_id,
            kind=kind,
            source=FeedbackSourceReference("recommendation_ui", "card_actions"),
            observed_at=datetime(2026, 8, 16, 1, 0, 0, tzinfo=timezone.utc),
            target=None if recommendation_only else target(track_id),
            recommendation=(
                FeedbackRecommendationReference(RUN_ID, CANDIDATE_ID)
                if recommendation_only
                else None
            ),
            attribution=attribution,
        )

    def revisions(self, track_id: str, signal_path: str) -> tuple:
        with PreferencePersistenceRepository(self.database_path) as repository:
            return tuple(
                repository.list_revisions(
                    SignalIdentity(target(track_id), "feedback_learning", signal_path)
                )
            )

    def applied_rows(self) -> tuple:
        with LearningApplicationRepository(self.database_path) as application:
            return application.list_applications()

    # --- required scenarios -------------------------------------------------

    def test_explicit_positive_feedback_moves_the_recommendation(self) -> None:
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))
        self.assertNotIn(TRACK_A, before)
        self.assertIn(TRACK_B, before)

        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.LIKED))
        self.apply(proposal)

        after = self.recommend((TRACK_A, TRACK_B))
        self.assertIn(TRACK_A, after)
        self.assertIn(TRACK_B, after)
        # The learned track's score is exactly basis_support(1) x the query-time magnitude.
        self.assertEqual(after[TRACK_A], POSITIVE_MAGNITUDE)
        # Baseline track unchanged.
        self.assertEqual(after[TRACK_B], before[TRACK_B])

    def test_explicit_negative_feedback_removes_the_track_from_recommendations(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))
        self.assertIn(TRACK_A, before)
        self.assertIn(TRACK_B, before)

        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.DISLIKED))
        self.apply(proposal)

        after = self.recommend((TRACK_A, TRACK_B))
        # P06 S3 frozen rule: favorited=true + disliked=true is a hard CONFLICT, so the track
        # loses its directional claim and its candidate disappears entirely.
        self.assertNotIn(TRACK_A, after)
        self.assertIn(TRACK_B, after)

    def test_implicit_positive_feedback_applies_with_class_provenance(self) -> None:
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))
        self.assertNotIn(TRACK_A, before)

        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.FAVORITED))
        self.apply(proposal)

        after = self.recommend((TRACK_A, TRACK_B))
        self.assertIn(TRACK_A, after)
        revisions = self.revisions(TRACK_A, "favorited")
        self.assertEqual(revisions[0].provenance, IMPLICIT_FEEDBACK_PROVENANCE)

    def test_explicit_and_implicit_class_distinction_survives_application(self) -> None:
        self.apply(self.run_full_pipeline(self.feedback(FeedbackKind.LIKED)))
        explicit_revisions = self.revisions(TRACK_A, "favorited")
        self.assertEqual(explicit_revisions[0].provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        # The corrected P08.5/P08.6 semantics claim no numeric magnitude distinction: both
        # classes produce the same direction through P06's frozen paths, and the class itself is
        # the durable distinguishing evidence (revision provenance).
        implicit_feedback = self.feedback(
            FeedbackKind.FAVORITED,
            track_id=TRACK_B,
            feedback_id="fbk_dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        )
        self.apply(self.run_full_pipeline(implicit_feedback))
        implicit_revisions = self.revisions(TRACK_B, "favorited")
        self.assertEqual(implicit_revisions[0].provenance, IMPLICIT_FEEDBACK_PROVENANCE)
        self.assertNotEqual(
            explicit_revisions[0].provenance, implicit_revisions[0].provenance
        )

    def test_skipped_remains_ambiguous_and_changes_nothing(self) -> None:
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))

        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.SKIPPED))
        self.assertIsNone(proposal)

        after = self.recommend((TRACK_A, TRACK_B))
        self.assertEqual(after, before)
        self.assertNotIn(TRACK_A, after)
        self.assertEqual(self.revisions(TRACK_A, "favorited"), ())
        self.assertEqual(self.revisions(TRACK_A, "disliked"), ())
        self.assertEqual(self.applied_rows(), ())

    def test_completed_remains_ambiguous_and_changes_nothing(self) -> None:
        self.seed_positive(TRACK_A)
        before = self.recommend((TRACK_A, TRACK_B))

        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.COMPLETED))
        self.assertIsNone(proposal)

        after = self.recommend((TRACK_A, TRACK_B))
        self.assertEqual(after, before)
        self.assertEqual(self.applied_rows(), ())

    def test_attribution_exclusion_is_journaled_without_evidence_change(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))

        proposal = self.run_full_pipeline(
            self.feedback(
                FeedbackKind.ATTRIBUTION_CORRECTION,
                attribution=FeedbackAttribution(
                    artist_target(), AttributionRelation.EXCLUDED
                ),
            )
        )
        record = self.apply(proposal)
        self.assertEqual(record.proposal_kind, ProposedUpdateKind.ATTRIBUTION_EXCLUSION)
        self.assertEqual(record.attribution.relation, AttributionRelation.EXCLUDED)
        self.assertEqual(record.attribution.aspect, artist_target())

        after = self.recommend((TRACK_A, TRACK_B))
        self.assertEqual(after, before)
        # No feedback_learning evidence was written for the excluded feedback.
        self.assertEqual(self.revisions(TRACK_A, "favorited")[0].provenance, SEED_PROVENANCE)
        self.assertEqual(self.revisions(TRACK_A, "disliked"), ())

    def test_duplicate_feedback_never_double_applies(self) -> None:
        self.seed_positive(TRACK_B)
        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.LIKED))
        self.apply(proposal)
        after_single = self.recommend((TRACK_A, TRACK_B))
        revisions_after_single = self.revisions(TRACK_A, "favorited")

        with LearningApplicationRepository(self.database_path) as application:
            with self.assertRaises(DuplicateLearningApplicationError):
                application.apply(
                    proposal, applied_at="2026-08-16T03:00:00+00:00"
                )

        self.assertEqual(self.revisions(TRACK_A, "favorited"), revisions_after_single)
        self.assertEqual(self.recommend((TRACK_A, TRACK_B)), after_single)
        self.assertEqual(len(self.applied_rows()), 1)

    def test_target_less_no_effect_mutates_nothing(self) -> None:
        self.seed_positive(TRACK_B)
        before = self.recommend((TRACK_A, TRACK_B))
        proposal = self.run_full_pipeline(
            self.feedback(FeedbackKind.DIRECTION_GOOD, recommendation_only=True)
        )
        self.assertIsNone(proposal)
        self.assertEqual(self.recommend((TRACK_A, TRACK_B)), before)
        self.assertEqual(self.applied_rows(), ())

    def test_provenance_traces_through_every_pipeline_layer(self) -> None:
        observation = self.feedback(FeedbackKind.LIKED)
        proposal = self.run_full_pipeline(observation)
        record = self.apply(proposal)

        with FeedbackHistoryRepository(self.database_path) as history:
            stored = history.get_observation(observation.feedback_id)
            self.assertEqual(stored, observation)

        self.assertEqual(record.feedback_id, observation.feedback_id)
        self.assertEqual(record.interpretation_policy_version, 1)
        self.assertEqual(record.effect_policy_version, 1)
        self.assertEqual(record.learning_policy_version, 2)
        self.assertEqual(record.signal_source_system, "feedback_learning")
        self.assertEqual(record.signal_path, "favorited")
        self.assertEqual(record.provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        revisions = self.revisions(TRACK_A, "favorited")
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0].provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        self.assertEqual(revisions[0].observed_at, observation.observed_at.isoformat())

    def test_learned_state_survives_restart_and_reads_back_identically(self) -> None:
        self.seed_positive(TRACK_B)
        proposal = self.run_full_pipeline(self.feedback(FeedbackKind.LIKED))
        self.apply(proposal)
        after_apply = self.recommend((TRACK_A, TRACK_B))

        # Reopen everything from durable state only.
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(
                SignalIdentity(target(TRACK_A), "feedback_learning", "favorited")
            )
            self.assertIs(head.current_semantic_value, True)
        with LearningApplicationRepository(self.database_path) as application:
            loaded = application.get_application(proposal.feedback_id)
            self.assertEqual(loaded.feedback_id, proposal.feedback_id)
            self.assertEqual(loaded.provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        self.assertEqual(self.recommend((TRACK_A, TRACK_B)), after_apply)


if __name__ == "__main__":
    unittest.main()
