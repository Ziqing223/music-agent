import copy
import json
import tempfile
import unittest
from pathlib import Path

from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    AlbumRelationState,
    ArtistRelationResolution,
    ArtistRelationState,
    IngestionCandidate,
)
from music_agent.promotion import PromotionStatus, promote_staged_track
from music_agent.reconciliation import (
    ReconciliationOutcome,
    ReconciliationReason,
)
from music_agent.reconciliation_workflow import (
    ReconciliationWorkflowError,
    ReconciliationWorkflowStatus,
    reconcile_staged_track_relations,
)
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = "WORKFLOW-TRACK-1") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


class ReconciliationWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.artist_a = self.model["artists"][0]["id"]
        self.artist_b = self.model["artists"][1]["id"]
        self.artist_c = self.model["artists"][2]["id"]
        self.album_e = self.model["albums"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def promotable_facts(self) -> dict:
        return {
            "name": ObservedValue.value("Workflow Track"),
            "duration_ms": ObservedValue.value(123000),
            "genres": ObservedValue.value(["Workflow"]),
            "composer": ObservedValue.null(),
            "library_state.favorited": ObservedValue.value(False),
            "library_state.rating": ObservedValue.value(0),
            "library_state.play_count": ObservedValue.value(0),
        }

    def unresolved_candidate(self, external_id: str = "WORKFLOW-TRACK-1") -> IngestionCandidate:
        return IngestionCandidate(track_key(external_id), self.promotable_facts())

    def stage(self, candidate: IngestionCandidate, *scopes: str) -> None:
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(candidate)
            for scope_key in scopes:
                staging.stage_candidate(candidate, scope_key)

    def load_candidate(self, key: ExternalIdentityKey) -> IngestionCandidate:
        with CandidateStagingRepository(self.database_path) as staging:
            candidate = staging.get_candidate(key)
        self.assertIsNotNone(candidate)
        return candidate  # type: ignore[return-value]

    # --- durable resolution ---

    def test_explicit_single_artist_resolves_to_artists(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), artist_canonical_ids=[self.artist_a]
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        self.assertEqual(
            self.load_candidate(track_key()).artist_relation,
            ArtistRelationResolution.resolved_to_artists([self.artist_a]),
        )

    def test_explicit_multi_artist_resolves_ordered_ids(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), artist_canonical_ids=[self.artist_b, self.artist_a]
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        self.assertEqual(
            self.load_candidate(track_key()).artist_relation,
            ArtistRelationResolution.resolved_to_artists([self.artist_b, self.artist_a]),
        )

    def test_explicit_album_resolves_to_album(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), album_canonical_id=self.album_e
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        self.assertEqual(
            self.load_candidate(track_key()).album_relation,
            AlbumRelationResolution.resolved_to_album(self.album_e),
        )

    def test_explicit_album_absence_resolves_absent(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), album_absent=True
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        self.assertEqual(
            self.load_candidate(track_key()).album_relation,
            AlbumRelationResolution.resolved_absent(),
        )

    def test_atomic_artist_and_album_update(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a, self.artist_c],
            album_canonical_id=self.album_e,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        candidate = self.load_candidate(track_key())
        self.assertEqual(
            candidate.artist_relation,
            ArtistRelationResolution.resolved_to_artists([self.artist_a, self.artist_c]),
        )
        self.assertEqual(
            candidate.album_relation,
            AlbumRelationResolution.resolved_to_album(self.album_e),
        )

    # --- idempotency and preservation ---

    def test_repeated_identical_decision_is_idempotent(self) -> None:
        self.stage(self.unresolved_candidate(), "scope_x")
        first = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a],
            album_canonical_id=self.album_e,
        )
        self.assertIs(first.status, ReconciliationWorkflowStatus.UPDATED)

        with CandidateStagingRepository(self.database_path) as staging:
            before = staging.get_candidate(track_key())
            scopes_before = staging.list_candidate_scopes(track_key())

        second = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a],
            album_canonical_id=self.album_e,
        )
        self.assertIs(second.status, ReconciliationWorkflowStatus.UNCHANGED)

        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(track_key()), before)
            self.assertEqual(staging.list_candidate_scopes(track_key()), scopes_before)

    def test_existing_resolution_preserved_when_decision_omitted(self) -> None:
        candidate = IngestionCandidate(
            track_key(),
            self.promotable_facts(),
            ArtistRelationResolution.resolved_to_artists([self.artist_a]),
            AlbumRelationResolution.resolved_to_album(self.album_e),
        )
        self.stage(candidate)
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), artist_canonical_ids=[self.artist_b]
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        loaded = self.load_candidate(track_key())
        self.assertEqual(
            loaded.artist_relation,
            ArtistRelationResolution.resolved_to_artists([self.artist_b]),
        )
        self.assertEqual(
            loaded.album_relation,
            AlbumRelationResolution.resolved_to_album(self.album_e),
        )

    def test_existing_album_resolution_preserved_when_album_not_requested(self) -> None:
        candidate = IngestionCandidate(
            track_key(),
            self.promotable_facts(),
            ArtistRelationResolution.resolved_to_artists([self.artist_a]),
            AlbumRelationResolution.resolved_to_album(self.album_e),
        )
        self.stage(candidate)
        result = reconcile_staged_track_relations(
            self.database_path, track_key(), album_absent=True
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        loaded = self.load_candidate(track_key())
        self.assertEqual(loaded.album_relation, AlbumRelationResolution.resolved_absent())
        self.assertEqual(
            loaded.artist_relation,
            ArtistRelationResolution.resolved_to_artists([self.artist_a]),
        )

    # --- fail-closed behavior ---

    def test_invalid_artist_target_does_not_update(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=["art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"],
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertIs(result.artist_result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(
            result.artist_result.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE
        )
        self.assertEqual(
            self.load_candidate(track_key()).artist_relation,
            ArtistRelationResolution.unresolved(),
        )

    def test_invalid_album_target_does_not_update(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            album_canonical_id="alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertIs(result.album_result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(
            result.album_result.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE
        )
        self.assertEqual(
            self.load_candidate(track_key()).album_relation,
            AlbumRelationResolution.unresolved(),
        )

    def test_artist_valid_and_album_invalid_updates_neither(self) -> None:
        self.stage(self.unresolved_candidate(), "scope_a")
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a],
            album_canonical_id="alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertIs(result.artist_result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertIs(result.album_result.outcome, ReconciliationOutcome.CONFLICT)
        loaded = self.load_candidate(track_key())
        self.assertEqual(loaded.artist_relation, ArtistRelationResolution.unresolved())
        self.assertEqual(loaded.album_relation, AlbumRelationResolution.unresolved())

    def test_artist_invalid_and_album_valid_updates_neither(self) -> None:
        self.stage(self.unresolved_candidate(), "scope_a")
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a, self.artist_a],
            album_canonical_id=self.album_e,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertIs(result.artist_result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(
            result.artist_result.reason, ReconciliationReason.DUPLICATE_CANONICAL_ID
        )
        self.assertIs(result.album_result.outcome, ReconciliationOutcome.RESOLVED)
        loaded = self.load_candidate(track_key())
        self.assertEqual(loaded.artist_relation, ArtistRelationResolution.unresolved())
        self.assertEqual(loaded.album_relation, AlbumRelationResolution.unresolved())

    def test_duplicate_artist_ids_fail_closed(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a, self.artist_a],
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertEqual(
            result.artist_result.reason, ReconciliationReason.DUPLICATE_CANONICAL_ID
        )
        self.assertEqual(
            self.load_candidate(track_key()).artist_relation,
            ArtistRelationResolution.unresolved(),
        )

    def test_wrong_entity_type_fails_closed(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.album_e],  # album id passed as artist id
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertEqual(result.artist_result.reason, ReconciliationReason.WRONG_ENTITY_TYPE)
        self.assertEqual(
            self.load_candidate(track_key()).artist_relation,
            ArtistRelationResolution.unresolved(),
        )

    def test_both_album_inputs_are_conflicting_request(self) -> None:
        self.stage(self.unresolved_candidate())
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            album_canonical_id=self.album_e,
            album_absent=True,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.BLOCKED)
        self.assertIs(result.album_result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(
            result.album_result.reason, ReconciliationReason.CONFLICTING_STRONG_EVIDENCE
        )
        self.assertEqual(
            self.load_candidate(track_key()).album_relation,
            AlbumRelationResolution.unresolved(),
        )

    def test_candidate_not_found_is_typed(self) -> None:
        result = reconcile_staged_track_relations(
            self.database_path, track_key("NEVER-STAGED"), artist_canonical_ids=[self.artist_a]
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.CANDIDATE_NOT_FOUND)

    # --- isolation ---

    def test_success_preserves_source_facts_and_scopes(self) -> None:
        candidate = self.unresolved_candidate()
        self.stage(candidate, "scope_a", "scope_b")
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a],
            album_canonical_id=self.album_e,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        with CandidateStagingRepository(self.database_path) as staging:
            loaded = staging.get_candidate(track_key())
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.source_facts, candidate.source_facts)
            self.assertEqual(loaded.external_identity, candidate.external_identity)
            self.assertEqual(staging.list_candidate_scopes(track_key()), ("scope_a", "scope_b"))

    def test_success_does_not_touch_canonical_bindings_or_presence(self) -> None:
        self.stage(self.unresolved_candidate(), "candidate_scope")
        with CanonicalRepository(self.database_path) as repository:
            before_model = repository.load_model()
            before_bindings = repository.list_external_identity_bindings("apple_music", EntityType.TRACK)
            before_presence = repository._connection.execute(
                "SELECT COUNT(*) FROM source_entity_presence"
            ).fetchone()[0]

        result = reconcile_staged_track_relations(
            self.database_path,
            track_key(),
            artist_canonical_ids=[self.artist_a],
            album_canonical_id=self.album_e,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), before_model)
            self.assertEqual(
                repository.list_external_identity_bindings("apple_music", EntityType.TRACK),
                before_bindings,
            )
            self.assertEqual(
                repository._connection.execute(
                    "SELECT COUNT(*) FROM source_entity_presence"
                ).fetchone()[0],
                before_presence,
            )

    # --- promotion integration ---

    def test_workflow_reconciles_then_promotion_succeeds_with_exact_relations(self) -> None:
        self.stage(self.unresolved_candidate("WORKFLOW-PROMOTE"), "promo_scope")
        result = reconcile_staged_track_relations(
            self.database_path,
            track_key("WORKFLOW-PROMOTE"),
            artist_canonical_ids=[self.artist_a, self.artist_c],
            album_canonical_id=self.album_e,
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)

        promoted = promote_staged_track(self.database_path, track_key("WORKFLOW-PROMOTE"))
        self.assertIs(promoted.status, PromotionStatus.PROMOTED)

        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            track = next(t for t in model["tracks"] if t["id"] == promoted.canonical_id)
            self.assertEqual(track["artist_ids"], [self.artist_a, self.artist_c])
            self.assertEqual(track["album_id"], self.album_e)

    def test_explicit_absence_promotes_to_null_album(self) -> None:
        self.stage(self.unresolved_candidate("WORKFLOW-ABSENT"), "absent_scope")
        result = reconcile_staged_track_relations(
            self.database_path, track_key("WORKFLOW-ABSENT"), album_absent=True
        )
        self.assertIs(result.status, ReconciliationWorkflowStatus.UPDATED)
        self.assertEqual(
            self.load_candidate(track_key("WORKFLOW-ABSENT")).album_relation,
            AlbumRelationResolution.resolved_absent(),
        )

        # Absent album plus resolved artist satisfies the gate; album_id becomes null via
        # explicit decision, not via MISSING inference.
        result2 = reconcile_staged_track_relations(
            self.database_path, track_key("WORKFLOW-ABSENT"), artist_canonical_ids=[self.artist_a]
        )
        self.assertIs(result2.status, ReconciliationWorkflowStatus.UPDATED)

        promoted = promote_staged_track(self.database_path, track_key("WORKFLOW-ABSENT"))
        self.assertIs(promoted.status, PromotionStatus.PROMOTED)
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            track = next(t for t in model["tracks"] if t["id"] == promoted.canonical_id)
            self.assertEqual(track["artist_ids"], [self.artist_a])
            self.assertIsNone(track["album_id"])

    # --- argument validation ---

    def test_no_requested_decision_is_rejected(self) -> None:
        with self.assertRaises(ReconciliationWorkflowError):
            reconcile_staged_track_relations(self.database_path, track_key())

    def test_non_track_candidate_identity_is_rejected(self) -> None:
        with self.assertRaises(ReconciliationWorkflowError):
            reconcile_staged_track_relations(
                self.database_path,
                ExternalIdentityKey("apple_music", EntityType.ALBUM, "ALBUM-PID"),
                artist_canonical_ids=[self.artist_a],
            )


if __name__ == "__main__":
    unittest.main()
