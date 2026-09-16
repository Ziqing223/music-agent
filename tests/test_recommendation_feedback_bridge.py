"""P10.17c: Recommendation-feedback learning bridge tests (deterministic stores)."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.feedback_contract import (
    FeedbackKind,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
    generate_feedback_id,
)
from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.learning_application import LearningApplicationRepository
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.recommendation_contract import (
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    assemble_recommendation_result,
    generate_run_id,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue

CLIENT_ID = "agt_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
TARGET = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OBSERVED_AT = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)


def empty_model() -> dict:
    return {"tracks": [], "artists": [], "albums": [], "playlists": [], "playlist_memberships": []}


def _save_run(database_path: Path, candidate_id: str = "cnd_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb") -> str:
    from music_agent.recommendation_contract import (
        Candidate,
        CandidateSourceReference,
        Eligibility,
    )

    candidate = Candidate(
        candidate_id=candidate_id,
        target=PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
        source=CandidateSourceReference("music_agent", "preference_driven"),
        basis_targets=(PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),),
        eligibility=Eligibility.ELIGIBLE,
        rejection=None,
    )
    item = RecommendationItem(candidate, _score())
    request = RecommendationRequest(
        RecommendationContext(OBSERVED_AT, ()), RecommendedItemKind.TRACK, 8
    )
    result = assemble_recommendation_result(
        request, (item,), run_id=generate_run_id(), produced_at=OBSERVED_AT
    )
    with RecommendationHistoryRepository(database_path) as history:
        history.save_result(result)
    return result.run_id


def _score():
    from music_agent.recommendation_contract import ScoreBreakdown, ScoreComponent

    return ScoreBreakdown(0.9, (ScoreComponent("basis_support", 0.9),))


class RecommendationFeedbackBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(empty_model())
        self.run_id = _save_run(self.database_path)
        self.service = SharedAgentService(
            self.database_path, clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL})
        )
        self.addCleanup(self.service.close)
        self.client = AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            self.service,
        )

    def _record_recommendation_feedback(self, kind, run_id=None, candidate_id=None) -> str:
        observation = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=kind,
            source=FeedbackSourceReference("apple_music", "recommendation_feedback"),
            observed_at=OBSERVED_AT,
            recommendation=FeedbackRecommendationReference(
                run_id or self.run_id, candidate_id or "cnd_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
            ),
        )
        with FeedbackHistoryRepository(self.database_path) as history:
            history.save_observation(observation)
        return observation.feedback_id

    def _apply(self, feedback_id: str):
        return self.client.call("apply_learning", {"feedback_id": feedback_id})

    def test_valid_liked_recommendation_feedback_applies_positive(self) -> None:
        feedback_id = self._record_recommendation_feedback(FeedbackKind.LIKED)
        result = self._apply(feedback_id)
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertTrue(result.payload["applied"])
        with PreferencePersistenceRepository(self.database_path) as preference:
            head = preference.get_head(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
                    "feedback_learning",
                    "favorited",
                )
            )
            self.assertIs(head.current_semantic_value, True)
        with LearningApplicationRepository(self.database_path) as la:
            record = la.get_application(feedback_id)
            self.assertEqual(record.target.target_id, TARGET)
        # The original observation stays recommendation-linked (never rewritten).
        with FeedbackHistoryRepository(self.database_path) as history:
            self.assertIsNone(history.get_observation(feedback_id).target)

    def test_valid_disliked_applies_negative(self) -> None:
        feedback_id = self._record_recommendation_feedback(FeedbackKind.DISLIKED)
        result = self._apply(feedback_id)
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertTrue(result.payload["applied"])
        with PreferencePersistenceRepository(self.database_path) as preference:
            head = preference.get_head(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
                    "feedback_learning",
                    "disliked",
                )
            )
            self.assertIs(head.current_semantic_value, True)

    def test_run_missing_fails_closed(self) -> None:
        feedback_id = self._record_recommendation_feedback(
            FeedbackKind.LIKED, run_id="rcm_00000000-0000-4000-8000-000000000000"
        )
        result = self._apply(feedback_id)
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "recommendation_feedback_resolution_error")
        with LearningApplicationRepository(self.database_path) as la:
            self.assertIsNone(la.get_application(feedback_id))

    def test_candidate_missing_fails_closed(self) -> None:
        feedback_id = self._record_recommendation_feedback(
            FeedbackKind.LIKED, candidate_id="cnd_ffffffff-ffff-4fff-8fff-ffffffffffff"
        )
        result = self._apply(feedback_id)
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "recommendation_feedback_resolution_error")

    def test_candidate_run_mismatch_fails_closed(self) -> None:
        other_run = _save_run(self.database_path, candidate_id="cnd_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
        # Reference a candidate from run A against run B's id.
        feedback_id = self._record_recommendation_feedback(
            FeedbackKind.LIKED,
            run_id=other_run,
            candidate_id="cnd_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        )
        result = self._apply(feedback_id)
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "recommendation_feedback_resolution_error")

    def test_replay_is_idempotent_no_duplicate_revision(self) -> None:
        feedback_id = self._record_recommendation_feedback(FeedbackKind.LIKED)
        first = self._apply(feedback_id)
        self.assertEqual(first.outcome.value, "ok")
        second = self._apply(feedback_id)  # duplicate feedback_id fails closed
        self.assertEqual(second.outcome.value, "execution_error")
        with PreferencePersistenceRepository(self.database_path) as preference:
            identity = SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
                "feedback_learning",
                "favorited",
            )
            self.assertEqual(len(preference.list_revisions(identity)), 1)

    def test_existing_target_linked_path_unchanged(self) -> None:
        observation = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=FeedbackKind.LIKED,
            source=FeedbackSourceReference("apple_music", "user-explicit-statement"),
            observed_at=OBSERVED_AT,
            target=PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
        )
        with FeedbackHistoryRepository(self.database_path) as history:
            history.save_observation(observation)
        result = self._apply(observation.feedback_id)
        self.assertEqual(result.outcome.value, "ok")
        self.assertTrue(result.payload["applied"])
        with LearningApplicationRepository(self.database_path) as la:
            self.assertEqual(
                la.get_application(observation.feedback_id).target.target_id, TARGET
            )

    def test_resolved_target_with_targeted_observation_fails_closed(self) -> None:
        from music_agent.learning_effect import LearningEffectValidationError, derive_learning_effect, LearningEffectPolicy
        from music_agent.feedback_interpretation import InterpretationPolicy, interpret_observation

        observation = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=FeedbackKind.LIKED,
            source=FeedbackSourceReference("apple_music", "x"),
            observed_at=OBSERVED_AT,
            target=PreferenceTargetReference(PreferenceTargetKind.TRACK, TARGET),
        )
        interpretation = interpret_observation(observation, InterpretationPolicy(1))
        with self.assertRaises(LearningEffectValidationError):
            derive_learning_effect(
                interpretation,
                LearningEffectPolicy(1),
                resolved_target=PreferenceTargetReference(
                    PreferenceTargetKind.TRACK, TARGET
                ),
            )


if __name__ == "__main__":
    unittest.main()
