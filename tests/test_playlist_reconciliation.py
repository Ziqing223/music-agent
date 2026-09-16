import copy
import json
import unittest
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.playlist_reconciliation import (
    MembershipOccurrenceEvidence,
    MembershipOccurrenceIdentity,
    MembershipReconciliationResult,
    PlaylistIdentityEvidence,
    PlaylistIdentityResult,
    PlaylistReconciliationOutcome,
    PlaylistReconciliationReason,
    evaluate_membership_occurrence,
    evaluate_membership_occurrences,
    evaluate_playlist_identity,
)
from music_agent.reconciliation import (
    BoundExternalIdentityEvidence,
    ReconciliationOutcome,
    ReconciliationValidationError,
)
from music_agent.source_observation import ObservedValue


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def playlist_key(external_id: str) -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.PLAYLIST, external_id)


def track_key(external_id: str) -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


class PlaylistIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_fixture()
        self.playlist = self.model["playlists"][0]  # SYNTH-PLAYLIST-1

    def test_caller_bound_playlist_external_identity_resolves(self) -> None:
        result = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("SYNTH-PLAYLIST-1"), self.playlist["id"]
                )
            ),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.EXACT_EXTERNAL_IDENTITY)
        self.assertEqual(result.canonical_id, self.playlist["id"])

    def test_explicit_canonical_playlist_decision_resolves(self) -> None:
        result = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(explicit_canonical_id=self.playlist["id"]),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.EXPLICIT_CANONICAL_DECISION)
        self.assertEqual(result.canonical_id, self.playlist["id"])

    def test_display_name_never_resolves_playlist_identity(self) -> None:
        result = evaluate_playlist_identity(
            self.model, PlaylistIdentityEvidence(display_name="Repeated Membership")
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.DISPLAY_NAME_INSUFFICIENT)
        self.assertIsNone(result.canonical_id)

    def test_missing_playlist_evidence_remains_unresolved(self) -> None:
        result = evaluate_playlist_identity(self.model, PlaylistIdentityEvidence())
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.MISSING_IDENTITY_EVIDENCE)

    def test_unbound_unknown_playlist_identity_remains_unresolved(self) -> None:
        result = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(unbound_identity=playlist_key("NO-SUCH-PLAYLIST")),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY)

    def test_wrong_entity_type_fails_closed(self) -> None:
        bound = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    track_key("SYNTH-TRACK-001"), self.model["tracks"][0]["id"]
                )
            ),
        )
        self.assertEqual(bound.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(bound.reason, PlaylistReconciliationReason.WRONG_ENTITY_TYPE)

        unbound = evaluate_playlist_identity(
            self.model, PlaylistIdentityEvidence(unbound_identity=track_key("SYNTH-TRACK-001"))
        )
        self.assertEqual(unbound.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(unbound.reason, PlaylistReconciliationReason.WRONG_ENTITY_TYPE)

    def test_dangling_canonical_target_fails_closed(self) -> None:
        dangling = "pl_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        bound = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("SYNTH-PLAYLIST-1"), dangling
                )
            ),
        )
        self.assertEqual(bound.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(bound.reason, PlaylistReconciliationReason.DANGLING_CANONICAL_REFERENCE)

        explicit = evaluate_playlist_identity(
            self.model, PlaylistIdentityEvidence(explicit_canonical_id=dangling)
        )
        self.assertEqual(explicit.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(explicit.reason, PlaylistReconciliationReason.DANGLING_CANONICAL_REFERENCE)

    def test_bound_identity_and_explicit_decision_conflict_fails_closed(self) -> None:
        other = self.model["playlists"][1]["id"]
        result = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("SYNTH-PLAYLIST-1"), self.playlist["id"]
                ),
                explicit_canonical_id=other,
            ),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(result.reason, PlaylistReconciliationReason.CONFLICTING_STRONG_EVIDENCE)

    def test_agreeing_bound_and_explicit_resolve(self) -> None:
        result = evaluate_playlist_identity(
            self.model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("SYNTH-PLAYLIST-1"), self.playlist["id"]
                ),
                explicit_canonical_id=self.playlist["id"],
            ),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.EXPLICIT_CANONICAL_DECISION)


class MembershipOccurrenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_fixture()
        self.playlist = self.model["playlists"][0]
        self.track = self.model["tracks"][0]

    def playlist_evidence(self) -> PlaylistIdentityEvidence:
        return PlaylistIdentityEvidence(
            bound_identity=BoundExternalIdentityEvidence(
                playlist_key("SYNTH-PLAYLIST-1"), self.playlist["id"]
            )
        )

    def track_bound(self) -> BoundExternalIdentityEvidence:
        return BoundExternalIdentityEvidence(track_key("SYNTH-TRACK-001"), self.track["id"])

    def occurrence_evidence(
        self,
        *,
        occurrence_identity: MembershipOccurrenceIdentity | None = None,
        position: ObservedValue | None = None,
    ) -> MembershipOccurrenceEvidence:
        return MembershipOccurrenceEvidence(
            playlist_evidence=self.playlist_evidence(),
            track_bound=self.track_bound(),
            occurrence_identity=occurrence_identity,
            position=position,
        )

    def test_occurrence_with_identity_is_identified_not_canonically_resolved(self) -> None:
        result = evaluate_membership_occurrence(
            self.model,
            self.occurrence_evidence(
                occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
            ),
        )
        self.assertEqual(result.outcome, PlaylistReconciliationOutcome.IDENTIFIED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.EXACT_EXTERNAL_IDENTITY)
        self.assertEqual(result.playlist_canonical_id, self.playlist["id"])
        self.assertEqual(result.track_canonical_id, self.track["id"])
        self.assertEqual(
            result.occurrence_identity, MembershipOccurrenceIdentity("apple_music", "MEM-1")
        )

    def test_same_playlist_track_with_two_occurrence_ids_identifies_two_occurrences(self) -> None:
        results = evaluate_membership_occurrences(
            self.model,
            (
                self.occurrence_evidence(
                    occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
                ),
                self.occurrence_evidence(
                    occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-2")
                ),
            ),
        )
        self.assertEqual(
            [r.outcome for r in results],
            [PlaylistReconciliationOutcome.IDENTIFIED, PlaylistReconciliationOutcome.IDENTIFIED],
        )
        self.assertNotEqual(results[0].occurrence_identity, results[1].occurrence_identity)
        self.assertEqual(results[0].playlist_canonical_id, results[1].playlist_canonical_id)
        self.assertEqual(results[0].track_canonical_id, results[1].track_canonical_id)

    def test_ambiguous_occurrence_without_identity_fails_closed(self) -> None:
        result = evaluate_membership_occurrence(
            self.model,
            self.occurrence_evidence(
                position=ObservedValue.value(3),  # position is not identity
            ),
        )
        self.assertEqual(result.outcome, PlaylistReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.AMBIGUOUS_OCCURRENCE_IDENTITY)
        self.assertIsNone(result.playlist_canonical_id)
        self.assertIsNone(result.track_canonical_id)

    def test_duplicate_occurrence_identity_fails_closed(self) -> None:
        results = evaluate_membership_occurrences(
            self.model,
            (
                self.occurrence_evidence(
                    occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
                ),
                self.occurrence_evidence(
                    occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
                ),
            ),
        )
        self.assertEqual(
            [r.outcome for r in results],
            [PlaylistReconciliationOutcome.CONFLICT, PlaylistReconciliationOutcome.CONFLICT],
        )
        self.assertEqual(
            [r.reason for r in results],
            [PlaylistReconciliationReason.DUPLICATE_OCCURRENCE_IDENTITY] * 2,
        )

    def test_unresolved_playlist_blocks_membership(self) -> None:
        result = evaluate_membership_occurrence(
            self.model,
            MembershipOccurrenceEvidence(
                playlist_evidence=PlaylistIdentityEvidence(display_name="Repeated Membership"),
                track_bound=self.track_bound(),
                occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1"),
            ),
        )
        self.assertEqual(result.outcome, PlaylistReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.DISPLAY_NAME_INSUFFICIENT)

    def test_unbound_track_identity_remains_unresolved(self) -> None:
        result = evaluate_membership_occurrence(
            self.model,
            MembershipOccurrenceEvidence(
                playlist_evidence=self.playlist_evidence(),
                track_unbound=track_key("NO-SUCH-TRACK"),
                occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1"),
            ),
        )
        self.assertEqual(result.outcome, PlaylistReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.UNKNOWN_TRACK_IDENTITY)

    def test_missing_track_identity_remains_unresolved(self) -> None:
        result = evaluate_membership_occurrence(
            self.model,
            MembershipOccurrenceEvidence(
                playlist_evidence=self.playlist_evidence(),
                occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1"),
            ),
        )
        self.assertEqual(result.outcome, PlaylistReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, PlaylistReconciliationReason.MISSING_TRACK_IDENTITY)

    def test_evaluation_does_not_mutate_model_or_evidence(self) -> None:
        model = load_fixture()
        original = copy.deepcopy(model)
        evidence = self.occurrence_evidence(
            occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
        )
        evaluate_membership_occurrence(model, evidence)
        evaluate_playlist_identity(
            model,
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("SYNTH-PLAYLIST-1"), self.playlist["id"]
                )
            ),
        )
        self.assertEqual(model, original)

    def test_invalid_evidence_rejected(self) -> None:
        with self.assertRaises(ReconciliationValidationError):
            MembershipOccurrenceIdentity("", "X")
        with self.assertRaises(ReconciliationValidationError):
            MembershipOccurrenceIdentity("apple_music", "")
        with self.assertRaises(ReconciliationValidationError):
            PlaylistIdentityEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    playlist_key("X"), self.playlist["id"]
                ),
                unbound_identity=playlist_key("Y"),
            )
        with self.assertRaises(ReconciliationValidationError):
            PlaylistIdentityEvidence(explicit_canonical_id="")
        with self.assertRaises(ReconciliationValidationError):
            MembershipOccurrenceEvidence(
                playlist_evidence=self.playlist_evidence(),
                track_bound=self.track_bound(),
                track_unbound=track_key("X"),
            )
        with self.assertRaises(ReconciliationValidationError):
            MembershipOccurrenceEvidence(
                playlist_evidence=self.playlist_evidence(),
                position="not-an-observed-value",  # type: ignore[arg-type]
            )
        with self.assertRaises(ReconciliationValidationError):
            evaluate_membership_occurrences(self.model, "not-a-sequence")  # type: ignore[arg-type]
        with self.assertRaises(ReconciliationValidationError):
            evaluate_playlist_identity({"playlists": "not-a-list"}, PlaylistIdentityEvidence())  # type: ignore[arg-type]

    def test_result_type_guards(self) -> None:
        resolved = evaluate_membership_occurrence(
            self.model,
            self.occurrence_evidence(
                occurrence_identity=MembershipOccurrenceIdentity("apple_music", "MEM-1")
            ),
        )
        self.assertIsInstance(resolved, MembershipReconciliationResult)
        self.assertIsInstance(
            evaluate_playlist_identity(self.model, PlaylistIdentityEvidence()),
            PlaylistIdentityResult,
        )


if __name__ == "__main__":
    unittest.main()
