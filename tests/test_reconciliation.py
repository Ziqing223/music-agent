import copy
import json
import unittest
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    AlbumRelationState,
    ArtistRelationResolution,
    ArtistRelationState,
)
from music_agent.reconciliation import (
    AlbumEvidence,
    AlbumReconciliationResult,
    ArtistEvidence,
    ArtistReconciliationResult,
    BoundExternalIdentityEvidence,
    ReconciliationOutcome,
    ReconciliationReason,
    ReconciliationValidationError,
    evaluate_album_reconciliation,
    evaluate_artist_reconciliation,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class ReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_fixture()
        self.artist_a = self.model["artists"][0]  # SYNTH-ARTIST-A
        self.artist_c = self.model["artists"][2]  # SYNTH-ARTIST-C
        self.album_e = self.model["albums"][0]    # SYNTH-ALBUM-E

    def artist_external_key(self, external_id: str) -> ExternalIdentityKey:
        return ExternalIdentityKey("apple_music", EntityType.ARTIST, external_id)

    def album_external_key(self, external_id: str) -> ExternalIdentityKey:
        return ExternalIdentityKey("apple_music", EntityType.ALBUM, external_id)

    def test_caller_bound_artist_external_identity_resolves_to_artists(self) -> None:
        result = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.artist_external_key("SYNTH-ARTIST-A"), self.artist_a["id"]
                )
            ),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.EXACT_EXTERNAL_IDENTITY)
        self.assertEqual(
            result.resolution,
            ArtistRelationResolution.resolved_to_artists([self.artist_a["id"]]),
        )

    def test_caller_bound_album_external_identity_resolves_to_album(self) -> None:
        result = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.album_external_key("SYNTH-ALBUM-E"), self.album_e["id"]
                )
            ),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.EXACT_EXTERNAL_IDENTITY)
        self.assertEqual(
            result.resolution,
            AlbumRelationResolution.resolved_to_album(self.album_e["id"]),
        )

    def test_unbound_identity_does_not_resolve_via_canonical_projection(self) -> None:
        # The canonical model projects SYNTH-ARTIST-A onto art_aaaaaaaa, but an unbound key
        # must never resolve through that projection.
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(unbound_identity=self.artist_external_key("SYNTH-ARTIST-A")),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(artist.reason, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY)
        self.assertIsNone(artist.resolution)

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(unbound_identity=self.album_external_key("SYNTH-ALBUM-E")),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(album.reason, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY)
        self.assertIsNone(album.resolution)

    def test_unbound_unknown_identity_remains_unresolved(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(unbound_identity=self.artist_external_key("NO-SUCH-ARTIST")),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(artist.reason, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY)

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(unbound_identity=self.album_external_key("NO-SUCH-ALBUM")),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(album.reason, ReconciliationReason.UNKNOWN_EXTERNAL_IDENTITY)

    def test_bound_identity_wrong_entity_type_fails_closed(self) -> None:
        track_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(bound_identity=BoundExternalIdentityEvidence(
                track_key, self.model["tracks"][0]["id"]
            )),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(artist.reason, ReconciliationReason.WRONG_ENTITY_TYPE)

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(bound_identity=BoundExternalIdentityEvidence(
                track_key, self.model["tracks"][0]["id"]
            )),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(album.reason, ReconciliationReason.WRONG_ENTITY_TYPE)

    def test_unbound_identity_wrong_entity_type_fails_closed(self) -> None:
        track_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        artist = evaluate_artist_reconciliation(
            self.model, ArtistEvidence(unbound_identity=track_key)
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(artist.reason, ReconciliationReason.WRONG_ENTITY_TYPE)

        album = evaluate_album_reconciliation(
            self.model, AlbumEvidence(unbound_identity=track_key)
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(album.reason, ReconciliationReason.WRONG_ENTITY_TYPE)

    def test_bound_identity_dangling_target_fails_closed(self) -> None:
        dangling_artist = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(bound_identity=BoundExternalIdentityEvidence(
                self.artist_external_key("SYNTH-ARTIST-A"), dangling_artist
            )),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(artist.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE)

        dangling_album = "alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(bound_identity=BoundExternalIdentityEvidence(
                self.album_external_key("SYNTH-ALBUM-E"), dangling_album
            )),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(album.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE)

    def test_explicit_dangling_canonical_target_fails_closed(self) -> None:
        dangling_artist = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        dangling_album = "alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        artist = evaluate_artist_reconciliation(
            self.model, ArtistEvidence(explicit_canonical_ids=[dangling_artist])
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(artist.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE)

        album = evaluate_album_reconciliation(
            self.model, AlbumEvidence(explicit_canonical_id=dangling_album)
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(album.reason, ReconciliationReason.DANGLING_CANONICAL_REFERENCE)

    def test_artist_display_name_only_remains_unresolved(self) -> None:
        result = evaluate_artist_reconciliation(
            self.model, ArtistEvidence(display_name="Artist Alpha")
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT)
        self.assertIsNone(result.resolution)

    def test_album_display_name_only_remains_unresolved(self) -> None:
        result = evaluate_album_reconciliation(
            self.model, AlbumEvidence(display_name="Synthetic Collection")
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT)
        self.assertIsNone(result.resolution)

    def test_exact_unique_display_name_still_remains_unresolved(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model, ArtistEvidence(display_name="Artist Gamma")
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(artist.reason, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT)

        album = evaluate_album_reconciliation(
            self.model, AlbumEvidence(display_name="Synthetic Sessions")
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(album.reason, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT)

    def test_multi_artist_display_string_remains_unresolved(self) -> None:
        result = evaluate_artist_reconciliation(
            self.model, ArtistEvidence(display_name="Artist Alpha & Artist Beta")
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.DISPLAY_NAME_INSUFFICIENT)
        self.assertIsNone(result.resolution)

    def test_missing_album_evidence_does_not_produce_absence(self) -> None:
        result = evaluate_album_reconciliation(self.model, AlbumEvidence())
        self.assertEqual(result.outcome, ReconciliationOutcome.UNRESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.MISSING_RELATION_EVIDENCE)
        self.assertIsNone(result.resolution)

    def test_explicit_reliable_album_absence_resolves_absent(self) -> None:
        result = evaluate_album_reconciliation(
            self.model, AlbumEvidence(explicit_absence=True)
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(result.reason, ReconciliationReason.EXPLICIT_ABSENCE)
        self.assertEqual(result.resolution, AlbumRelationResolution.resolved_absent())

    def test_binding_evidence_a_and_explicit_decision_b_conflict(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.artist_external_key("SYNTH-ARTIST-A"), self.artist_a["id"]
                ),
                explicit_canonical_ids=[self.artist_c["id"]],
            ),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(artist.reason, ReconciliationReason.CONFLICTING_STRONG_EVIDENCE)

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.album_external_key("SYNTH-ALBUM-E"), self.album_e["id"]
                ),
                explicit_absence=True,
            ),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(album.reason, ReconciliationReason.CONFLICTING_STRONG_EVIDENCE)

    def test_binding_evidence_a_and_explicit_decision_a_resolve(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.artist_external_key("SYNTH-ARTIST-A"), self.artist_a["id"]
                ),
                explicit_canonical_ids=[self.artist_a["id"]],
            ),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(artist.reason, ReconciliationReason.EXPLICIT_CANONICAL_DECISION)
        self.assertEqual(
            artist.resolution,
            ArtistRelationResolution.resolved_to_artists([self.artist_a["id"]]),
        )

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.album_external_key("SYNTH-ALBUM-E"), self.album_e["id"]
                ),
                explicit_canonical_id=self.album_e["id"],
            ),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(album.reason, ReconciliationReason.EXPLICIT_CANONICAL_DECISION)
        self.assertEqual(
            album.resolution,
            AlbumRelationResolution.resolved_to_album(self.album_e["id"]),
        )

    def test_explicit_valid_canonical_decision_resolves(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(explicit_canonical_ids=[self.artist_a["id"], self.artist_c["id"]]),
        )
        self.assertEqual(artist.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(artist.reason, ReconciliationReason.EXPLICIT_CANONICAL_DECISION)
        self.assertEqual(
            artist.resolution,
            ArtistRelationResolution.resolved_to_artists(
                [self.artist_a["id"], self.artist_c["id"]]
            ),
        )

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(explicit_canonical_id=self.album_e["id"]),
        )
        self.assertEqual(album.outcome, ReconciliationOutcome.RESOLVED)
        self.assertEqual(album.reason, ReconciliationReason.EXPLICIT_CANONICAL_DECISION)
        self.assertEqual(
            album.resolution,
            AlbumRelationResolution.resolved_to_album(self.album_e["id"]),
        )

    def test_duplicate_artist_canonical_ids_fail_closed(self) -> None:
        result = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(explicit_canonical_ids=[self.artist_a["id"], self.artist_a["id"]]),
        )
        self.assertEqual(result.outcome, ReconciliationOutcome.CONFLICT)
        self.assertEqual(result.reason, ReconciliationReason.DUPLICATE_CANONICAL_ID)

    def test_evaluation_does_not_mutate_model_or_evidence(self) -> None:
        model = load_fixture()
        original = copy.deepcopy(model)
        evidence = ArtistEvidence(
            bound_identity=BoundExternalIdentityEvidence(
                self.artist_external_key("SYNTH-ARTIST-A"), self.artist_a["id"]
            ),
            explicit_canonical_ids=[self.artist_a["id"]],
            display_name="Artist Alpha",
        )
        evaluate_artist_reconciliation(model, evidence)
        evaluate_album_reconciliation(
            model,
            AlbumEvidence(bound_identity=BoundExternalIdentityEvidence(
                self.album_external_key("SYNTH-ALBUM-E"), self.album_e["id"]
            )),
        )
        self.assertEqual(model, original)

    def test_invalid_evidence_rejected(self) -> None:
        with self.assertRaises(ReconciliationValidationError):
            BoundExternalIdentityEvidence(self.artist_external_key("X"), 42)  # type: ignore[arg-type]
        with self.assertRaises(ReconciliationValidationError):
            BoundExternalIdentityEvidence("not-a-key", self.artist_a["id"])  # type: ignore[arg-type]
        with self.assertRaises(ReconciliationValidationError):
            ArtistEvidence(explicit_canonical_ids=[42])  # type: ignore[list-item]
        with self.assertRaises(ReconciliationValidationError):
            ArtistEvidence(display_name="")
        with self.assertRaises(ReconciliationValidationError):
            ArtistEvidence(
                bound_identity=BoundExternalIdentityEvidence(
                    self.artist_external_key("X"), self.artist_a["id"]
                ),
                unbound_identity=self.artist_external_key("Y"),
            )
        with self.assertRaises(ReconciliationValidationError):
            AlbumEvidence(explicit_canonical_id="")
        with self.assertRaises(ReconciliationValidationError):
            AlbumEvidence(explicit_absence="yes")  # type: ignore[arg-type]
        with self.assertRaises(ReconciliationValidationError):
            evaluate_artist_reconciliation({"artists": "not-a-list"}, ArtistEvidence())  # type: ignore[arg-type]

    def test_resolved_results_reuse_existing_relation_types(self) -> None:
        artist = evaluate_artist_reconciliation(
            self.model,
            ArtistEvidence(explicit_canonical_ids=[self.artist_a["id"]]),
        )
        self.assertIsNotNone(artist.resolution)
        self.assertIs(artist.resolution.state, ArtistRelationState.RESOLVED_TO_ARTISTS)

        album = evaluate_album_reconciliation(
            self.model,
            AlbumEvidence(explicit_absence=True),
        )
        self.assertIsNotNone(album.resolution)
        self.assertIs(album.resolution.state, AlbumRelationState.RESOLVED_ABSENT)


if __name__ == "__main__":
    unittest.main()
