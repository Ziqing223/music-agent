"""P08.6: applying proposed preference updates to durable P06 state.

These tests prove the application layer: one ``EVIDENCE_OBSERVATION`` proposal becomes exactly
one durable P06 semantic observation on the feedback_learning head (direction via the frozen
P06 path, evidence class via the revision provenance, event times preserved), an
``ATTRIBUTION_EXCLUSION`` proposal journals the exclusion without writing any P06 evidence,
every application is journaled with full provenance (feedback_id + all upstream policy/contract
versions), replay is blocked by the feedback_id primary key, an interrupted application (evidence
durable but journal absent) is recovered by retry exactly once without duplicating evidence,
NO_EFFECT never reaches this layer, unsupported policy/contract versions fail closed, the
journal is immutable, and P06 heads under other source systems are never touched.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackKind,
)
from music_agent.feedback_interpretation import InterpretationPolicy, interpret_observation
from music_agent.learning_application import (
    DuplicateLearningApplicationError,
    LearningApplicationRepository,
    LearningApplicationRepositoryError,
)
from music_agent.learning_effect import LearningEffectPolicy, derive_learning_effect
from music_agent.learning_policy import (
    EXPLICIT_FEEDBACK_PROVENANCE,
    IMPLICIT_FEEDBACK_PROVENANCE,
    LEARNING_POLICY_CONTRACT_VERSION,
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
from music_agent.source_observation import ObservedValue

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
APPLIED_AT = "2026-08-16T01:00:00+00:00"


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def proposal(
    kind: FeedbackKind,
    *,
    attribution: FeedbackAttribution | None = None,
) -> ProposedPreferenceUpdate | None:
    from music_agent.feedback_contract import (
        FeedbackSourceReference,
        assemble_feedback_observation,
    )

    observation = assemble_feedback_observation(
        feedback_id="fbk_44444444-4444-4444-8444-444444444444",
        kind=kind,
        source=FeedbackSourceReference("recommendation_ui", "card_actions"),
        observed_at=NOW,
        target=track_target(),
        attribution=attribution,
    )
    effect = derive_learning_effect(
        interpret_observation(observation, InterpretationPolicy(1)),
        LearningEffectPolicy(1),
    )
    return propose_preference_update(effect, LearningPolicy(2))


class LearningApplicationRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    # --- schema / migration ------------------------------------------------

    def test_fresh_database_reaches_v14_with_application_table(self) -> None:
        with LearningApplicationRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("learning_applications", tables)

    # --- explicit evidence application -------------------------------------

    def test_explicit_positive_evidence_becomes_durable_p06_state(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            record = repository.apply(liked, applied_at=APPLIED_AT)
        self.assertEqual(record.feedback_id, liked.feedback_id)
        self.assertEqual(record.proposal_kind, ProposedUpdateKind.EVIDENCE_OBSERVATION)
        self.assertEqual(record.signal_source_system, "feedback_learning")
        self.assertEqual(record.signal_path, "favorited")
        self.assertEqual(record.provenance, EXPLICIT_FEEDBACK_PROVENANCE)
        self.assertEqual(record.target, track_target())

        identity = SignalIdentity(track_target(), "feedback_learning", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertIs(head.current_semantic_value, True)
            self.assertEqual(head.current_revision_sequence, 1)
            revisions = repository.list_revisions(identity)
            self.assertEqual(len(revisions), 1)
            self.assertEqual(revisions[0].provenance, EXPLICIT_FEEDBACK_PROVENANCE)
            self.assertEqual(revisions[0].observed_at, NOW.isoformat())

    def test_implicit_evidence_carries_implicit_provenance(self) -> None:
        favorited = proposal(FeedbackKind.FAVORITED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(favorited, applied_at=APPLIED_AT)
        identity = SignalIdentity(track_target(), "feedback_learning", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertIs(head.current_semantic_value, True)
            revisions = repository.list_revisions(identity)
            self.assertEqual(revisions[0].provenance, IMPLICIT_FEEDBACK_PROVENANCE)

    def test_negative_evidence_becomes_durable_disliked_state(self) -> None:
        disliked = proposal(FeedbackKind.DISLIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(disliked, applied_at=APPLIED_AT)
        identity = SignalIdentity(track_target(), "feedback_learning", "disliked")
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertIs(head.current_semantic_value, True)

    def test_application_never_touches_source_of_truth_heads(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(liked, applied_at=APPLIED_AT)
        apple_identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_head(apple_identity))

    # --- exclusion ----------------------------------------------------------

    def test_attribution_exclusion_journals_without_p06_evidence(self) -> None:
        exclusion = proposal(
            FeedbackKind.ATTRIBUTION_CORRECTION,
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED),
        )
        with LearningApplicationRepository(self.database_path) as repository:
            record = repository.apply(exclusion, applied_at=APPLIED_AT)
        self.assertEqual(record.proposal_kind, ProposedUpdateKind.ATTRIBUTION_EXCLUSION)
        self.assertIsNone(record.signal_source_system)
        self.assertIsNone(record.signal_path)
        self.assertIsNone(record.provenance)
        self.assertEqual(record.attribution.aspect, artist_target())
        self.assertEqual(record.attribution.relation, AttributionRelation.EXCLUDED)

        with PreferencePersistenceRepository(self.database_path) as repository:
            for path in ("favorited", "disliked", "rating"):
                identity = SignalIdentity(track_target(), "feedback_learning", path)
                self.assertIsNone(repository.get_head(identity), path)

    def test_exclusion_is_never_negative_evidence(self) -> None:
        exclusion = proposal(
            FeedbackKind.ATTRIBUTION_CORRECTION,
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED),
        )
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(exclusion, applied_at=APPLIED_AT)
            disliked_identity = SignalIdentity(track_target(), "feedback_learning", "disliked")
            self.assertIsNone(
                repository._preference_repository.get_head(disliked_identity)
            )

    # --- replay / duplicates ------------------------------------------------

    def test_same_feedback_never_applies_twice(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(liked, applied_at=APPLIED_AT)
            with self.assertRaises(DuplicateLearningApplicationError):
                repository.apply(liked, applied_at="2026-08-16T02:00:00+00:00")
            self.assertEqual(len(repository.list_applications()), 1)
        identity = SignalIdentity(track_target(), "feedback_learning", "favorited")
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)

    def test_partial_write_recovery_completes_journal_exactly_once(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        identity = SignalIdentity(track_target(), "feedback_learning", "favorited")

        # Interrupted first attempt: the P06 evidence write commits durably through the real
        # persistence boundary (the same call the application layer makes internally), then
        # execution stops before the journal row is written.
        with PreferencePersistenceRepository(self.database_path) as repository:
            outcome = repository.record_observation(
                liked.signal_identity,
                liked.proposed_value,
                observed_at=liked.observed_at,
                event_at=liked.event_at,
                provenance=liked.provenance,
            )
        self.assertIsNotNone(outcome.revision)
        # Intermediate durable state: evidence present, application journal absent.
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertIs(head.current_semantic_value, True)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertEqual(len(repository.list_revisions(identity)), 1)
        with LearningApplicationRepository(self.database_path) as application:
            self.assertIsNone(application.get_application(liked.feedback_id))
            self.assertEqual(application.list_applications(), ())

        # Retry the same feedback through the real application boundary.
        with LearningApplicationRepository(self.database_path) as application:
            record = application.apply(liked, applied_at=APPLIED_AT)

        # P06 same-value idempotency turned the retry's evidence write into a confirmation:
        # no duplicate semantic revision, and the journal was completed exactly once.
        self.assertEqual(record.feedback_id, liked.feedback_id)
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertIs(head.current_semantic_value, True)
            self.assertEqual(head.current_revision_sequence, 1)
            revisions = repository.list_revisions(identity)
            self.assertEqual(len(revisions), 1)
            self.assertEqual(revisions[0].provenance, EXPLICIT_FEEDBACK_PROVENANCE)
            self.assertEqual(revisions[0].observed_at, NOW.isoformat())
        with LearningApplicationRepository(self.database_path) as application:
            applications = application.list_applications()
            self.assertEqual(len(applications), 1)
            self.assertEqual(applications[0].feedback_id, liked.feedback_id)
            self.assertEqual(
                application.get_application(liked.feedback_id), applications[0]
            )

    # --- journal readback ---------------------------------------------------

    def test_journal_records_full_provenance_and_versions(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(liked, applied_at=APPLIED_AT)
            loaded = repository.get_application(liked.feedback_id)
            self.assertEqual(loaded, repository.list_applications()[0])
        self.assertEqual(loaded.interpretation_policy_version, 1)
        self.assertEqual(loaded.effect_policy_version, 1)
        self.assertEqual(loaded.learning_policy_version, 2)
        self.assertEqual(
            loaded.learning_policy_contract_version, LEARNING_POLICY_CONTRACT_VERSION
        )
        self.assertEqual(loaded.applied_at, APPLIED_AT)

    def test_applications_survive_reopen(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(liked, applied_at=APPLIED_AT)
        with LearningApplicationRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            loaded = repository.get_application(liked.feedback_id)
            self.assertEqual(loaded.feedback_id, liked.feedback_id)
            self.assertEqual(
                [record.feedback_id for record in repository.list_applications()],
                [liked.feedback_id],
            )

    def test_get_application_for_missing_feedback_returns_none(self) -> None:
        with LearningApplicationRepository(self.database_path) as repository:
            self.assertIsNone(
                repository.get_application("fbk_99999999-9999-4999-8999-999999999999")
            )

    # --- fail-closed --------------------------------------------------------

    def test_rejects_non_proposal_input(self) -> None:
        with LearningApplicationRepository(self.database_path) as repository:
            for bad in ("not-a-proposal", None, {"kind": "evidence_observation"}, 42):
                with self.subTest(bad=bad):
                    with self.assertRaises(LearningApplicationRepositoryError):
                        repository.apply(bad)  # type: ignore[arg-type]

    def test_rejects_wrong_learning_policy_contract_version(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        forged = ProposedPreferenceUpdate(
            effect=liked.effect,
            kind=liked.kind,
            signal_identity=liked.signal_identity,
            proposed_value=liked.proposed_value,
            provenance=liked.provenance,
            observed_at=liked.observed_at,
            policy_version=2,
            contract_version=1,
        )
        with LearningApplicationRepository(self.database_path) as repository:
            with self.assertRaises(LearningApplicationRepositoryError):
                repository.apply(forged, applied_at=APPLIED_AT)
            self.assertEqual(len(repository.list_applications()), 0)

    # --- immutability -------------------------------------------------------

    def test_journal_rows_are_immutable(self) -> None:
        liked = proposal(FeedbackKind.LIKED)
        with LearningApplicationRepository(self.database_path) as repository:
            repository.apply(liked, applied_at=APPLIED_AT)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE learning_applications SET provenance='tampered' "
                    "WHERE feedback_id=?",
                    (liked.feedback_id,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute("DELETE FROM learning_applications")
        with LearningApplicationRepository(self.database_path) as repository:
            self.assertEqual(
                repository.get_application(liked.feedback_id).provenance,
                EXPLICIT_FEEDBACK_PROVENANCE,
            )


if __name__ == "__main__":
    unittest.main()
