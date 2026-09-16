import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.identity import (
    EntityType,
    ExternalIdentityKey,
    generate_canonical_id,
    validate_canonical_id,
)
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    ArtistRelationResolution,
    CandidateValidationError,
    IngestionCandidate,
    PromotionBlockerCode,
)
from music_agent.promotion import (
    PromotionStatus,
    StagedCandidateNotFoundError,
    promote_staged_track,
)
from music_agent.repository import CanonicalRepository
from music_agent.snapshot import (
    SnapshotCompleteness,
    SnapshotRecord,
    SourceSnapshot,
    SourceSnapshotScope,
    apply_snapshot,
)
from music_agent.source_observation import ObservedValue, SourcePresence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = "PROMOTION-TRACK-1") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


class AtomicTrackPromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.artist_id = self.model["artists"][0]["id"]
        self.album_id = self.model["albums"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def candidate(
        self,
        external_id: str = "PROMOTION-TRACK-1",
        *,
        genres: ObservedValue | None = None,
        artist: ArtistRelationResolution | None = None,
        album: AlbumRelationResolution | None = None,
    ) -> IngestionCandidate:
        return IngestionCandidate(
            track_key(external_id),
            {
                "name": ObservedValue.value("Promoted Track"),
                "duration_ms": ObservedValue.missing(),
                "genres": genres or ObservedValue.value([]),
                "composer": ObservedValue.null(),
                "library_state.favorited": ObservedValue.value(False),
                "library_state.rating": ObservedValue.value(0),
                "library_state.play_count": ObservedValue.value(0),
            },
            artist or ArtistRelationResolution.resolved_to_artists([self.artist_id]),
            album or AlbumRelationResolution.resolved_to_album(self.album_id),
        )

    def stage(
        self,
        candidate: IngestionCandidate,
        *scopes: str,
    ) -> None:
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(candidate)
            for scope_key in scopes:
                staging.stage_candidate(candidate, scope_key)

    def test_success_is_atomic_restart_safe_and_repeated_promotion_is_idempotent(self) -> None:
        candidate = self.candidate()
        other = self.candidate("PROMOTION-OTHER")
        self.stage(candidate, "scope_b", "scope_a")
        self.stage(other, "scope_c")

        with patch(
            "music_agent.promotion.generate_canonical_id",
            wraps=generate_canonical_id,
        ) as generate:
            result = promote_staged_track(self.database_path, candidate.external_identity)
        self.assertIs(result.status, PromotionStatus.PROMOTED)
        self.assertEqual(result.external_identity, candidate.external_identity)
        self.assertEqual(result.transferred_scopes, ("scope_a", "scope_b"))
        self.assertIsNotNone(result.canonical_id)
        validate_canonical_id(EntityType.TRACK, result.canonical_id)
        generate.assert_called_once_with(EntityType.TRACK)

        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(candidate.external_identity))
            self.assertEqual(staging.list_candidate_scopes(candidate.external_identity), ())
            self.assertEqual(staging.get_candidate(other.external_identity), other)
            self.assertEqual(
                staging.list_candidate_scopes(other.external_identity), ("scope_c",)
            )
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
            promoted = next(
                track for track in loaded["tracks"] if track["id"] == result.canonical_id
            )
            self.assertEqual(
                repository.lookup_external_identity(candidate.external_identity),
                result.canonical_id,
            )
            for scope_key in ("scope_a", "scope_b"):
                self.assertIs(
                    repository.get_source_presence(
                        "apple_music", EntityType.TRACK, result.canonical_id, scope_key
                    ),
                    SourcePresence.PRESENT,
                )
            self.assertIsNone(repository.get_source_presence(
                "apple_music", EntityType.TRACK, result.canonical_id, "scope_c"
            ))
            self.assertEqual(len(loaded["tracks"]), len(self.model["tracks"]) + 1)
            self.assertEqual(len(loaded["artists"]), len(self.model["artists"]))
            self.assertEqual(len(loaded["albums"]), len(self.model["albums"]))
            self.assertEqual(promoted["agent_metadata"]["tags"], [])
            self.assertIs(promoted["library_state"]["favorited"], False)
            self.assertEqual(promoted["library_state"]["rating"], 0)
            self.assertEqual(promoted["library_state"]["play_count"], 0)
            self.assertEqual(promoted["genres"], [])
            self.assertIsNone(promoted["duration_ms"])
            self.assertIsNone(promoted["composer"])

        repeated = promote_staged_track(self.database_path, candidate.external_identity)
        self.assertIs(repeated.status, PromotionStatus.ALREADY_BOUND)
        self.assertEqual(repeated.canonical_id, result.canonical_id)
        self.assertEqual(repeated.transferred_scopes, ())
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                [track["id"] for track in repository.load_model()["tracks"]].count(
                    result.canonical_id
                ),
                1,
            )

    def test_candidate_without_scope_does_not_invent_canonical_presence(self) -> None:
        candidate = self.candidate("PROMOTION-NO-SCOPE")
        self.stage(candidate)
        result = promote_staged_track(self.database_path, candidate.external_identity)
        self.assertIs(result.status, PromotionStatus.PROMOTED)
        self.assertEqual(result.transferred_scopes, ())
        with CanonicalRepository(self.database_path) as repository:
            count = repository._connection.execute(
                "SELECT COUNT(*) FROM source_entity_presence WHERE canonical_id=?",
                (result.canonical_id,),
            ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_execution_time_gate_blocks_without_generating_or_persisting_identity(self) -> None:
        dangling_artist = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        cases = (
            (
                "artist-unresolved",
                self.candidate(
                    "PROMOTION-BLOCK-ARTIST",
                    artist=ArtistRelationResolution.unresolved(),
                ),
                PromotionBlockerCode.ARTIST_RELATION_UNRESOLVED,
            ),
            (
                "album-unresolved",
                self.candidate(
                    "PROMOTION-BLOCK-ALBUM",
                    album=AlbumRelationResolution.unresolved(),
                ),
                PromotionBlockerCode.ALBUM_RELATION_UNRESOLVED,
            ),
            (
                "genres-unknown",
                self.candidate(
                    "PROMOTION-BLOCK-GENRES",
                    genres=ObservedValue.missing(),
                ),
                PromotionBlockerCode.GENRES_UNKNOWN,
            ),
            (
                "dangling-relation",
                self.candidate(
                    "PROMOTION-BLOCK-DANGLING",
                    artist=ArtistRelationResolution.resolved_to_artists([
                        dangling_artist
                    ]),
                ),
                PromotionBlockerCode.ARTIST_RELATION_INVALID,
            ),
        )
        for label, candidate, expected_code in cases:
            with self.subTest(label=label):
                self.stage(candidate, "blocked_scope")
                with patch("music_agent.promotion.generate_canonical_id") as generate:
                    result = promote_staged_track(
                        self.database_path, candidate.external_identity
                    )
                self.assertIs(result.status, PromotionStatus.BLOCKED)
                self.assertIn(expected_code, {blocker.code for blocker in result.blockers})
                generate.assert_not_called()
                with CanonicalRepository(self.database_path) as repository:
                    self.assertEqual(repository.load_model(), self.model)
                    self.assertIsNone(
                        repository.lookup_external_identity(candidate.external_identity)
                    )
                with CandidateStagingRepository(self.database_path) as staging:
                    self.assertEqual(
                        staging.get_candidate(candidate.external_identity), candidate
                    )
                    self.assertEqual(
                        staging.list_candidate_scopes(candidate.external_identity),
                        ("blocked_scope",),
                    )

    def test_stale_already_bound_candidate_is_retained_without_duplicate_track(self) -> None:
        existing_key = track_key("SYNTH-TRACK-001")
        candidate = self.candidate(existing_key.external_id)
        self.stage(candidate, "stale_scope")
        with CanonicalRepository(self.database_path) as repository:
            existing_id = repository.lookup_external_identity(existing_key)
            before = repository.load_model()
        result = promote_staged_track(self.database_path, existing_key)
        self.assertIs(result.status, PromotionStatus.ALREADY_BOUND)
        self.assertEqual(result.canonical_id, existing_id)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), before)
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(existing_key), candidate)
            self.assertEqual(staging.list_candidate_scopes(existing_key), ("stale_scope",))

    def test_transaction_failures_roll_back_model_binding_presence_and_retirement(self) -> None:
        failure_cases = ("binding", "presence", "cleanup")
        for failure in failure_cases:
            with self.subTest(failure=failure):
                database_path = (
                    Path(self.temporary_directory.name) / f"failure-{failure}.sqlite3"
                )
                with CanonicalRepository(database_path) as repository:
                    repository.save_model(self.model)
                candidate = self.candidate(f"PROMOTION-FAIL-{failure}")
                with CandidateStagingRepository(database_path) as staging:
                    staging.stage_candidate(candidate, "failure_scope")
                    if failure == "binding":
                        staging._connection.execute(
                            """CREATE TRIGGER fail_promotion_binding
                            BEFORE INSERT ON external_identity_bindings
                            WHEN NEW.external_id='PROMOTION-FAIL-binding'
                            BEGIN SELECT RAISE(ABORT, 'binding failure'); END"""
                        )
                    elif failure == "presence":
                        staging._connection.execute(
                            """CREATE TRIGGER fail_promotion_presence
                            BEFORE INSERT ON source_entity_presence
                            WHEN NEW.scope_key='failure_scope'
                            BEGIN SELECT RAISE(ABORT, 'presence failure'); END"""
                        )
                    else:
                        staging._connection.execute(
                            """CREATE TRIGGER fail_promotion_cleanup
                            BEFORE DELETE ON ingestion_candidate_scopes
                            WHEN OLD.external_id='PROMOTION-FAIL-cleanup'
                            BEGIN SELECT RAISE(ABORT, 'cleanup failure'); END"""
                        )

                with self.assertRaises(sqlite3.IntegrityError):
                    promote_staged_track(database_path, candidate.external_identity)
                with CanonicalRepository(database_path) as repository:
                    self.assertEqual(repository.load_model(), self.model)
                    self.assertIsNone(
                        repository.lookup_external_identity(candidate.external_identity)
                    )
                    self.assertEqual(
                        repository._connection.execute(
                            "SELECT COUNT(*) FROM source_entity_presence"
                        ).fetchone()[0],
                        0,
                    )
                with CandidateStagingRepository(database_path) as staging:
                    self.assertEqual(
                        staging.get_candidate(candidate.external_identity), candidate
                    )
                    self.assertEqual(
                        staging.list_candidate_scopes(candidate.external_identity),
                        ("failure_scope",),
                    )

    def test_generated_id_collision_leaves_candidate_and_store_unchanged(self) -> None:
        candidate = self.candidate("PROMOTION-ID-COLLISION")
        self.stage(candidate, "collision_scope")
        existing_id = self.model["tracks"][0]["id"]
        with patch("music_agent.promotion.generate_canonical_id", return_value=existing_id):
            with self.assertRaises(CandidateValidationError):
                promote_staged_track(self.database_path, candidate.external_identity)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), self.model)
            self.assertIsNone(repository.lookup_external_identity(candidate.external_identity))
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(candidate.external_identity), candidate)
            self.assertEqual(
                staging.list_candidate_scopes(candidate.external_identity),
                ("collision_scope",),
            )

    def test_snapshot_stage_resolve_promote_then_snapshot_resolves_same_identity(self) -> None:
        key = track_key("PROMOTION-SNAPSHOT")
        fields = {
            "name": ObservedValue.value("Snapshot Promotion"),
            "genres": ObservedValue.value([]),
            "library_state.play_count": ObservedValue.value(0),
        }
        snapshot = SourceSnapshot(
            SourceSnapshotScope(
                "apple_music",
                EntityType.TRACK,
                "snapshot_scope",
                SnapshotCompleteness.COMPLETE,
                True,
            ),
            (SnapshotRecord(key, fields),),
        )
        with CanonicalRepository(self.database_path) as repository:
            first = apply_snapshot(repository, snapshot)
        self.assertEqual(first.unresolved_source_records, (key,))

        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(
                IngestionCandidate(key, fields),
                "snapshot_scope",
            )
            staging.update_relation_resolution(
                key,
                artist_resolution=ArtistRelationResolution.resolved_to_artists([
                    self.artist_id
                ]),
                album_resolution=AlbumRelationResolution.resolved_to_album(self.album_id),
            )
        promoted = promote_staged_track(self.database_path, key)
        self.assertIs(promoted.status, PromotionStatus.PROMOTED)

        with CanonicalRepository(self.database_path) as repository:
            second = apply_snapshot(repository, snapshot)
            self.assertEqual(second.unresolved_source_records, ())
            self.assertEqual(repository.lookup_external_identity(key), promoted.canonical_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, promoted.canonical_id, "snapshot_scope"
                ),
                SourcePresence.PRESENT,
            )
            self.assertEqual(
                [track["id"] for track in repository.load_model()["tracks"]].count(
                    promoted.canonical_id
                ),
                1,
            )
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    def test_missing_candidate_without_binding_is_explicit(self) -> None:
        with self.assertRaises(StagedCandidateNotFoundError):
            promote_staged_track(self.database_path, track_key("NEVER-STAGED"))


class CatalogPromotionTest(unittest.TestCase):
    """P11.1: apple_music_catalog candidates promote to canonical Tracks with distinct identity."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.artist_id = self.model["artists"][0]["id"]
        self.album_id = self.model["albums"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def catalog_key(self, external_id: str = "CATALOG-SONG-1") -> ExternalIdentityKey:
        return ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, external_id)

    def catalog_candidate(
        self,
        external_id: str = "CATALOG-SONG-1",
        *,
        isrc: str | None = "USSYN2400001",
    ) -> IngestionCandidate:
        secondaries = ()
        if isrc is not None:
            secondaries = (ExternalIdentityKey("isrc", EntityType.TRACK, isrc),)
        return IngestionCandidate(
            self.catalog_key(external_id),
            {
                "name": ObservedValue.value("Catalog Song"),
                "duration_ms": ObservedValue.value(201000),
                "genres": ObservedValue.value(["Synthetic"]),
                "composer": ObservedValue.null(),
                "release_date": ObservedValue.value("2024-01-15"),
                "library_state.favorited": ObservedValue.value(False),
                "library_state.rating": ObservedValue.value(0),
                "library_state.play_count": ObservedValue.value(0),
            },
            ArtistRelationResolution.resolved_to_artists([self.artist_id]),
            AlbumRelationResolution.resolved_to_album(self.album_id),
            secondaries,
        )

    def test_catalog_candidate_promotes_with_catalog_and_isrc_bindings(self) -> None:
        key = self.catalog_key()
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(self.catalog_candidate(), "catalog")
        result = promote_staged_track(self.database_path, key)
        self.assertEqual(result.status, PromotionStatus.PROMOTED)
        canonical_id = result.canonical_id
        self.assertIsNotNone(canonical_id)
        with CanonicalRepository(self.database_path) as repository:
            track = next(
                track for track in repository.load_model()["tracks"]
                if track["id"] == canonical_id
            )
            self.assertEqual(
                track["external_ids"]["apple_music_catalog_id"], "CATALOG-SONG-1"
            )
            self.assertEqual(track["external_ids"]["isrc"], "USSYN2400001")
            self.assertIsNone(track["external_ids"]["apple_music_persistent_id"])
            self.assertEqual(
                repository.get_source_presence(
                    "apple_music_catalog", EntityType.TRACK, canonical_id, "catalog"
                ),
                SourcePresence.PRESENT,
            )
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    def test_repeated_catalog_discovery_converges_on_one_canonical_track(self) -> None:
        key = self.catalog_key()
        first = None
        for _ in range(2):
            with CandidateStagingRepository(self.database_path) as staging:
                staging.stage_candidate(self.catalog_candidate(), "catalog")
            result = promote_staged_track(self.database_path, key)
            if first is None:
                self.assertEqual(result.status, PromotionStatus.PROMOTED)
                first = result.canonical_id
            else:
                # Second discovery resolves the already-bound canonical: no second Track.
                self.assertEqual(result.status, PromotionStatus.ALREADY_BOUND)
                self.assertEqual(result.canonical_id, first)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                sum(
                    1
                    for track in repository.load_model()["tracks"]
                    if track["external_ids"].get("apple_music_catalog_id") == "CATALOG-SONG-1"
                ),
                1,
            )

    def test_catalog_id_already_bound_elsewhere_never_creates_second_track(self) -> None:
        key = self.catalog_key()
        other_canonical = self.model["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(key, other_canonical)
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(self.catalog_candidate(), "catalog")
        result = promote_staged_track(self.database_path, key)
        self.assertEqual(result.status, PromotionStatus.ALREADY_BOUND)
        self.assertEqual(result.canonical_id, other_canonical)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(len(repository.load_model()["tracks"]), len(self.model["tracks"]))

    def test_candidate_rejects_two_isrc_secondaries(self) -> None:
        with self.assertRaises(CandidateValidationError):
            IngestionCandidate(
                self.catalog_key(),
                self.catalog_candidate().source_facts,
                ArtistRelationResolution.resolved_to_artists([self.artist_id]),
                AlbumRelationResolution.resolved_to_album(self.album_id),
                (
                    ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001"),
                    ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400002"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
