import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.candidate_staging import (
    CandidateStagingError,
    CandidateStagingRepository,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.ingestion_candidate import (
    AlbumRelationResolution,
    ArtistRelationResolution,
    IngestionCandidate,
    evaluate_track_promotion,
)
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.snapshot import SnapshotRecord
from music_agent.source_observation import ObservedValue, SourcePresence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
V2_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def key(external_id: str = "STAGING-TRACK-1") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


class CandidateStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.artist_id = self.model["artists"][0]["id"]
        self.album_id = self.model["albums"][0]["id"]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def unresolved_candidate(
        self, external_id: str = "STAGING-TRACK-1", name: str = "Staged Candidate"
    ) -> IngestionCandidate:
        return IngestionCandidate(key(external_id), {
            "name": ObservedValue.value(name),
            "duration_ms": ObservedValue.missing(),
            "genres": ObservedValue.value([]),
            "composer": ObservedValue.null(),
            "library_state.favorited": ObservedValue.value(False),
            "library_state.rating": ObservedValue.value(0),
            "library_state.play_count": ObservedValue.value(0),
        })

    def resolved_candidate(
        self, external_id: str = "STAGING-TRACK-1", name: str = "Resolved Candidate"
    ) -> IngestionCandidate:
        return IngestionCandidate(
            key(external_id),
            {
                "name": ObservedValue.value(name),
                "duration_ms": ObservedValue.value(123000),
                "genres": ObservedValue.value(["Synthetic", "Staging"]),
                "composer": ObservedValue.null(),
                "library_state.favorited": ObservedValue.value(False),
                "library_state.rating": ObservedValue.value(0),
                "library_state.play_count": ObservedValue.value(0),
            },
            ArtistRelationResolution.resolved_to_artists([self.artist_id]),
            AlbumRelationResolution.resolved_to_album(self.album_id),
        )

    def test_fresh_database_reaches_current_version_with_separate_staging_tables(self) -> None:
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0] for row in staging._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("ingestion_candidates", tables)
            self.assertIn("ingestion_candidate_scopes", tables)
            columns = {
                row[1] for row in staging._connection.execute(
                    "PRAGMA table_info(ingestion_candidates)"
                )
            }
            self.assertEqual(
                columns, {"source_system", "entity_type", "external_id", "payload"}
            )

    def test_real_v2_store_upgrades_to_current_without_changing_canonical_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey(
            "apple_music", EntityType.TRACK, "SYNTH-TRACK-001"
        )
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with patch("music_agent.repository.MIGRATIONS", V2_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 2)

        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.schema_version, CURRENT_SCHEMA_VERSION)
            staging.stage_candidate(self.unresolved_candidate(), "library_tracks")

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_v3_migration_failure_rolls_back_without_partial_schema(self) -> None:
        with patch("music_agent.repository.MIGRATIONS", V2_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                self.assertEqual(repository.schema_version, 2)

        class InvalidMigration:
            def joinpath(self, _: str) -> "InvalidMigration":
                return self

            def read_text(self, **_: str) -> str:
                return "CREATE TABLE partial_v3(id INTEGER); INVALID SQL;"

        with patch("music_agent.repository.resources.files", return_value=InvalidMigration()):
            with self.assertRaises(sqlite3.OperationalError):
                CandidateStagingRepository(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0],
                2,
            )
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_v3'"
            ).fetchone())

    def test_tri_state_and_relation_states_round_trip_after_restart(self) -> None:
        unresolved = self.unresolved_candidate()
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(unresolved, "scope_b")
            staging.stage_candidate(unresolved, "scope_a")
            staging.stage_candidate(unresolved, "scope_a")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(unresolved.external_identity), unresolved)
            self.assertEqual(
                staging.list_candidate_scopes(unresolved.external_identity),
                ("scope_a", "scope_b"),
            )

        resolved = self.resolved_candidate("STAGING-TRACK-2")
        absent = IngestionCandidate(
            key("STAGING-TRACK-3"),
            {"name": ObservedValue.value("Albumless"), "genres": ObservedValue.value([])},
            ArtistRelationResolution.resolved_to_artists([self.artist_id]),
            AlbumRelationResolution.resolved_absent(),
        )
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(resolved)
            staging.stage_candidate(absent)
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(resolved.external_identity), resolved)
            self.assertEqual(staging.get_candidate(absent.external_identity), absent)
            self.assertEqual(staging.list_candidate_scopes(resolved.external_identity), ())

    def test_rediscovery_updates_source_facts_but_preserves_resolved_relations_and_scopes(self) -> None:
        original = self.unresolved_candidate()
        artist_resolution = ArtistRelationResolution.resolved_to_artists([self.artist_id])
        album_resolution = AlbumRelationResolution.resolved_to_album(self.album_id)
        rediscovered = self.unresolved_candidate(name="Updated Source Facts")
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(original, "scope_a")
            staging.update_relation_resolution(
                key(),
                artist_resolution=artist_resolution,
                album_resolution=album_resolution,
            )
            before_rediscovery = staging.get_candidate(key())
            self.assertIsNotNone(before_rediscovery)
            self.assertTrue(
                evaluate_track_promotion(before_rediscovery, self.model).is_promotable
            )
            staging.stage_candidate(rediscovered, "scope_b")
            staging.stage_candidate(rediscovered)
            staging.stage_candidate(rediscovered, "scope_b")
            expected = IngestionCandidate(
                key(), rediscovered.source_facts, artist_resolution, album_resolution
            )
            self.assertEqual(staging.get_candidate(key()), expected)
            self.assertTrue(
                evaluate_track_promotion(staging.get_candidate(key()), self.model).is_promotable
            )
            self.assertEqual(staging.list_candidate_scopes(key()), ("scope_a", "scope_b"))
            self.assertEqual(
                staging._connection.execute(
                    "SELECT COUNT(*) FROM ingestion_candidates"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                staging._connection.execute(
                    "SELECT COUNT(*) FROM ingestion_candidate_scopes"
                ).fetchone()[0],
                2,
            )
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.get_candidate(key()), expected)
            staging.stage_candidate(rediscovered, "scope_a")
            self.assertEqual(
                staging._connection.execute(
                    "SELECT COUNT(*) FROM ingestion_candidates"
                ).fetchone()[0],
                1,
            )

    def test_rediscovery_does_not_clear_explicit_resolved_absent_album(self) -> None:
        original = self.unresolved_candidate()
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(original)
            staging.update_relation_resolution(
                key(),
                artist_resolution=ArtistRelationResolution.resolved_to_artists([
                    self.artist_id
                ]),
                album_resolution=AlbumRelationResolution.resolved_absent(),
            )
            staging.stage_candidate(self.unresolved_candidate(name="Rediscovered"))
            loaded = staging.get_candidate(key())
            self.assertEqual(
                loaded.album_relation,
                AlbumRelationResolution.resolved_absent(),
            )
            self.assertEqual(loaded.source_facts["name"].payload, "Rediscovered")

    def test_invalid_atomic_relation_update_preserves_original_candidate(self) -> None:
        original = self.unresolved_candidate()
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(original)
            with self.assertRaises(CandidateStagingError):
                staging.update_relation_resolution(
                    key(),
                    artist_resolution=ArtistRelationResolution.resolved_to_artists([
                        self.artist_id
                    ]),
                    album_resolution="invalid",  # type: ignore[arg-type]
                )
            self.assertEqual(staging.get_candidate(key()), original)

    def test_staging_does_not_mutate_canonical_tables_bindings_or_presence(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()
        candidate = self.unresolved_candidate()
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(candidate, "candidate_scope")
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)
            self.assertEqual(repository.load_model(), fixture)
            self.assertIsNone(repository.lookup_external_identity(candidate.external_identity))
            self.assertIsNone(repository.get_source_presence(
                "apple_music", EntityType.TRACK, track_id, "candidate_scope"
            ))
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )

    def test_promotion_evaluation_is_identical_before_and_after_restart(self) -> None:
        candidates = (
            self.unresolved_candidate("STAGING-TRACK-1"),
            self.resolved_candidate("STAGING-TRACK-2"),
        )
        before = {
            candidate.external_identity: evaluate_track_promotion(candidate, self.model)
            for candidate in candidates
        }
        with CandidateStagingRepository(self.database_path) as staging:
            for candidate in candidates:
                staging.stage_candidate(candidate)
        with CandidateStagingRepository(self.database_path) as staging:
            for candidate in candidates:
                loaded = staging.get_candidate(candidate.external_identity)
                self.assertIsNotNone(loaded)
                self.assertEqual(evaluate_track_promotion(loaded, self.model), before[candidate.external_identity])
        self.assertFalse(before[candidates[0].external_identity].is_promotable)
        self.assertTrue(before[candidates[1].external_identity].is_promotable)

    def test_corrupt_payload_fails_closed(self) -> None:
        candidate = self.unresolved_candidate()
        corrupt_payloads = (
            "not-json",
            json.dumps({
                "source_facts": {"name": {"state": "unexpected"}},
                "artist_relation": {"state": "unresolved"},
                "album_relation": {"state": "unresolved"},
            }),
            json.dumps({
                "source_facts": {},
                "artist_relation": {
                    "state": "resolved_to_artists",
                    "canonical_ids": "not-an-array",
                },
                "album_relation": {"state": "unresolved"},
            }),
        )
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(candidate)
            for payload in corrupt_payloads:
                with self.subTest(payload=payload):
                    staging._connection.execute(
                        "UPDATE ingestion_candidates SET payload=?", (payload,)
                    )
                    with self.assertRaises(CandidateStagingError):
                        staging.get_candidate(candidate.external_identity)

    def test_candidate_update_and_scope_insert_are_atomic(self) -> None:
        original = self.unresolved_candidate(name="Original")
        updated = self.unresolved_candidate(name="Must Roll Back")
        with CandidateStagingRepository(self.database_path) as staging:
            staging.stage_candidate(original, "existing_scope")
            staging._connection.execute(
                """CREATE TRIGGER fail_candidate_scope
                BEFORE INSERT ON ingestion_candidate_scopes
                WHEN NEW.scope_key='failing_scope'
                BEGIN SELECT RAISE(ABORT, 'scope failure'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                staging.stage_candidate(updated, "failing_scope")
            self.assertEqual(staging.get_candidate(key()), original)
            self.assertEqual(staging.list_candidate_scopes(key()), ("existing_scope",))

    def test_snapshot_record_is_staged_only_by_explicit_caller_action(self) -> None:
        record = SnapshotRecord(key(), {
            "name": ObservedValue.value("Explicit Snapshot Candidate"),
            "genres": ObservedValue.value([]),
        })
        candidate = IngestionCandidate(record.external_identity, record.fields)
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(record.external_identity))
            staging.stage_candidate(candidate, "snapshot_scope")
            self.assertEqual(staging.get_candidate(record.external_identity), candidate)
            self.assertEqual(
                staging.list_candidate_scopes(record.external_identity),
                ("snapshot_scope",),
            )
            self.assertFalse(hasattr(staging, "delete_candidate"))


if __name__ == "__main__":
    unittest.main()
