import copy
import json
import unittest
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey, IdentityValidationError
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    ArtistRelationResolution,
    CandidateValidationError,
    IngestionCandidate,
    PromotionBlockedError,
    PromotionBlockerCode,
    build_promotable_track,
    evaluate_track_promotion,
)
from music_agent.snapshot import SnapshotRecord
from music_agent.source_observation import ObservedValue
from music_agent.validation import validate_fixture


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
NEW_TRACK_ID = "trk_99999999-9999-4999-8999-999999999999"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def apple_track_key(external_id: str = "Opaque-New-PID") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


class IngestionCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_fixture()
        self.artist_id = self.model["artists"][0]["id"]
        self.album_id = self.model["albums"][0]["id"]

    def candidate(
        self,
        facts: dict | None = None,
        *,
        artist: ArtistRelationResolution | None = None,
        album: AlbumRelationResolution | None = None,
        key: ExternalIdentityKey | None = None,
    ) -> IngestionCandidate:
        source_facts = {
            "name": ObservedValue.value("Candidate Track"),
            "genres": ObservedValue.value([]),
        }
        if facts:
            source_facts.update(facts)
        return IngestionCandidate(
            key or apple_track_key(),
            source_facts,
            artist or ArtistRelationResolution.resolved_to_artists([self.artist_id]),
            album or AlbumRelationResolution.resolved_to_album(self.album_id),
        )

    def blocker_codes(self, candidate: IngestionCandidate) -> set[PromotionBlockerCode]:
        return {
            blocker.code for blocker in evaluate_track_promotion(candidate, self.model).blockers
        }

    def test_candidate_identity_is_external_key_and_has_no_canonical_id(self) -> None:
        candidate = self.candidate(key=apple_track_key("Case-Sensitive-PID"))
        self.assertEqual(candidate.external_identity.external_id, "Case-Sensitive-PID")
        self.assertFalse(hasattr(candidate, "canonical_id"))
        self.assertTrue(evaluate_track_promotion(candidate, self.model).is_promotable)

        with self.assertRaises(IdentityValidationError):
            apple_track_key("")
        with self.assertRaises(CandidateValidationError):
            self.candidate(key=ExternalIdentityKey("other", EntityType.TRACK, "PID"))
        with self.assertRaises(CandidateValidationError):
            self.candidate(key=ExternalIdentityKey("apple_music", EntityType.ARTIST, "PID"))

    def test_candidate_source_facts_exclude_relations_and_shared_tags(self) -> None:
        for path in ("artist_ids", "album_id", "agent_metadata.tags"):
            with self.subTest(path=path):
                with self.assertRaises(CandidateValidationError):
                    self.candidate({path: ObservedValue.value([])})

    def test_name_gate_requires_nonempty_value(self) -> None:
        cases = (
            ObservedValue.missing(),
            ObservedValue.null(),
            ObservedValue.value(""),
            ObservedValue.value("   "),
        )
        for observed in cases:
            with self.subTest(state=observed.state, payload=observed.payload):
                candidate = self.candidate({"name": observed})
                self.assertIn(PromotionBlockerCode.MISSING_NAME, self.blocker_codes(candidate))
                with self.assertRaises(PromotionBlockedError):
                    build_promotable_track(candidate, NEW_TRACK_ID, self.model)

    def test_genres_missing_is_not_known_empty_and_null_is_invalid(self) -> None:
        missing = self.candidate({"genres": ObservedValue.missing()})
        self.assertIn(PromotionBlockerCode.GENRES_UNKNOWN, self.blocker_codes(missing))
        with self.assertRaises(PromotionBlockedError):
            build_promotable_track(missing, NEW_TRACK_ID, self.model)

        null = self.candidate({"genres": ObservedValue.null()})
        self.assertIn(PromotionBlockerCode.INVALID_SOURCE_FACT, self.blocker_codes(null))

        known_empty = self.candidate({"genres": ObservedValue.value([])})
        self.assertTrue(evaluate_track_promotion(known_empty, self.model).is_promotable)
        self.assertEqual(
            build_promotable_track(known_empty, NEW_TRACK_ID, self.model)["genres"], []
        )

    def test_artist_relation_must_resolve_to_existing_nonempty_ids(self) -> None:
        unresolved = self.candidate(artist=ArtistRelationResolution.unresolved())
        self.assertIn(
            PromotionBlockerCode.ARTIST_RELATION_UNRESOLVED,
            self.blocker_codes(unresolved),
        )
        with self.assertRaises(PromotionBlockedError):
            build_promotable_track(unresolved, NEW_TRACK_ID, self.model)

        invalid_relations = (
            ArtistRelationResolution.resolved_to_artists([]),
            ArtistRelationResolution.resolved_to_artists([
                "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
            ]),
            ArtistRelationResolution.resolved_to_artists(["not-an-artist-id"]),
        )
        for relation in invalid_relations:
            with self.subTest(relation=relation):
                candidate = self.candidate(artist=relation)
                self.assertIn(
                    PromotionBlockerCode.ARTIST_RELATION_INVALID,
                    self.blocker_codes(candidate),
                )

        resolved = self.candidate(
            artist=ArtistRelationResolution.resolved_to_artists([self.artist_id])
        )
        track = build_promotable_track(resolved, NEW_TRACK_ID, self.model)
        self.assertEqual(track["artist_ids"], [self.artist_id])

    def test_album_relation_unresolved_never_becomes_null(self) -> None:
        unresolved = self.candidate(album=AlbumRelationResolution.unresolved())
        self.assertIn(
            PromotionBlockerCode.ALBUM_RELATION_UNRESOLVED,
            self.blocker_codes(unresolved),
        )
        with self.assertRaises(PromotionBlockedError):
            build_promotable_track(unresolved, NEW_TRACK_ID, self.model)

        dangling = self.candidate(album=AlbumRelationResolution.resolved_to_album(
            "alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        ))
        self.assertIn(
            PromotionBlockerCode.ALBUM_RELATION_INVALID,
            self.blocker_codes(dangling),
        )

        resolved = self.candidate(
            album=AlbumRelationResolution.resolved_to_album(self.album_id)
        )
        self.assertEqual(
            build_promotable_track(resolved, NEW_TRACK_ID, self.model)["album_id"],
            self.album_id,
        )
        absent = self.candidate(album=AlbumRelationResolution.resolved_absent())
        self.assertIsNone(build_promotable_track(absent, NEW_TRACK_ID, self.model)["album_id"])

    def test_invalid_source_facts_produce_typed_validation_blocker(self) -> None:
        cases = (
            ("name", ObservedValue.value(7)),
            ("duration_ms", ObservedValue.value(-1)),
            ("genres", ObservedValue.value("not-an-array")),
            ("track_number", ObservedValue.value(0)),
            ("release_date", ObservedValue.value("2025-02-29")),
            ("library_state.favorited", ObservedValue.value(0)),
            ("library_state.rating", ObservedValue.value(101)),
            ("library_state.play_count", ObservedValue.value(-1)),
            (
                "library_state.last_played_at",
                ObservedValue.value("2025-01-01T12:00:00"),
            ),
        )
        for path, observed in cases:
            with self.subTest(path=path):
                candidate = self.candidate({path: observed})
                self.assertIn(
                    PromotionBlockerCode.INVALID_SOURCE_FACT,
                    self.blocker_codes(candidate),
                )
                with self.assertRaises(PromotionBlockedError):
                    build_promotable_track(candidate, NEW_TRACK_ID, self.model)

    def test_missing_nullable_and_library_facts_initialize_null_without_faking_values(self) -> None:
        candidate = self.candidate()
        track = build_promotable_track(candidate, NEW_TRACK_ID, self.model)
        for path in (
            "duration_ms", "track_number", "disc_number", "release_date", "composer"
        ):
            with self.subTest(path=path):
                self.assertIsNone(track[path])
        self.assertEqual(
            track["library_state"],
            {
                "favorited": None,
                "disliked": None,
                "rating": None,
                "play_count": None,
                "skip_count": None,
                "added_to_library_at": None,
                "last_played_at": None,
            },
        )
        self.assertEqual(track["agent_metadata"]["tags"], [])

    def test_false_zero_and_exact_external_id_survive_canonical_construction(self) -> None:
        candidate = self.candidate({
            "library_state.favorited": ObservedValue.value(False),
            "library_state.disliked": ObservedValue.value(False),
            "library_state.rating": ObservedValue.value(0),
            "library_state.play_count": ObservedValue.value(0),
            "library_state.skip_count": ObservedValue.value(0),
        }, key=apple_track_key("Opaque-Pid-aA"))
        track = build_promotable_track(candidate, NEW_TRACK_ID, self.model)
        self.assertIs(track["library_state"]["favorited"], False)
        self.assertEqual(track["library_state"]["rating"], 0)
        self.assertEqual(track["library_state"]["play_count"], 0)
        self.assertEqual(
            track["external_ids"]["apple_music_persistent_id"], "Opaque-Pid-aA"
        )

    def test_promotable_candidate_builds_structurally_and_graph_valid_track(self) -> None:
        original = copy.deepcopy(self.model)
        candidate = self.candidate({
            "duration_ms": ObservedValue.value(123000),
            "genres": ObservedValue.value(["Synthetic Genre"]),
            "track_number": ObservedValue.value(1),
            "disc_number": ObservedValue.value(1),
            "release_date": ObservedValue.value("2024-02-29"),
            "composer": ObservedValue.null(),
            "library_state.added_to_library_at": ObservedValue.value(
                "2025-01-02T03:04:05+08:00"
            ),
        })
        evaluation = evaluate_track_promotion(candidate, self.model)
        self.assertTrue(evaluation.is_promotable)
        self.assertEqual(evaluation.blockers, ())
        track = build_promotable_track(candidate, NEW_TRACK_ID, self.model)
        candidate_model = copy.deepcopy(self.model)
        candidate_model["tracks"].append(track)
        validate_fixture(candidate_model)
        self.assertEqual(self.model, original)

    def test_snapshot_record_requires_explicit_candidate_construction(self) -> None:
        record = SnapshotRecord(apple_track_key(), {
            "name": ObservedValue.value("Snapshot Candidate"),
            "genres": ObservedValue.value([]),
        })
        candidate = IngestionCandidate(record.external_identity, record.fields)
        self.assertEqual(candidate.external_identity, record.external_identity)
        codes = self.blocker_codes(candidate)
        self.assertIn(PromotionBlockerCode.ARTIST_RELATION_UNRESOLVED, codes)
        self.assertIn(PromotionBlockerCode.ALBUM_RELATION_UNRESOLVED, codes)


if __name__ == "__main__":
    unittest.main()
