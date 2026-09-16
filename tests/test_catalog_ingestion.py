"""P11.1/P11-T2: catalog discovery -> staging -> relation resolution -> automatic promotion.

Covers the orchestration surface: dedupe / library exclusion, authoritative Artist / Album
identity resolution (reuse by bound Catalog ID, never by name), automatic promotion through
the sealed path, convergence on repeated discovery, fail-closed blocking on missing identity
evidence, and visibility of a promoted Track to the catalog candidate recommendation path.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.apple_music_catalog import CatalogTrack
from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.catalog_candidate_generation import generate_catalog_candidates
from music_agent.catalog_ingestion import (
    CATALOG_SCOPE,
    CatalogIngestionOrchestrator,
    CatalogIngestStatus,
    build_catalog_candidate,
)
from music_agent.catalog_track_state_repository import CatalogTrackStateRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.preference_attribution import (
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.promotion import PromotionStatus, promote_staged_track
from music_agent.recommendation_contract import (
    Eligibility,
    PreferenceInput,
    RecommendationContext,
)
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import SourcePresence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"

ARTIST_KEY = "CATALOG-ARTIST-1"
ALBUM_KEY = "CATALOG-ALBUM-1"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def catalog_track(
    catalog_id: str = "CATALOG-SONG-1",
    *,
    isrc: str | None = "USSYN2400001",
    genres: tuple[str, ...] = ("Synthetic",),
    artist_catalog_ids: tuple[str, ...] = (ARTIST_KEY,),
    album_catalog_id: str | None = ALBUM_KEY,
    artist_names: tuple[str, ...] = ("Artist Alpha",),
    album_name: str | None = "Catalog Album",
) -> CatalogTrack:
    return CatalogTrack(
        catalog_id=catalog_id,
        name=f"Catalog Song {catalog_id}",
        artist_names=artist_names,
        album_name=album_name,
        genres=genres,
        isrc=isrc,
        duration_ms=201000,
        release_date="2024-01-15",
        url=None,
        artist_catalog_ids=artist_catalog_ids,
        album_catalog_id=album_catalog_id,
    )


class FakeCatalogSource:
    def __init__(self, tracks: tuple[CatalogTrack, ...]) -> None:
        self.tracks = tracks
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> tuple[CatalogTrack, ...]:
        self.calls.append((term, limit))
        return self.tracks


class CatalogIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def orchestrator(self, source=None) -> CatalogIngestionOrchestrator:
        return CatalogIngestionOrchestrator(
            CanonicalRepository(self.database_path),
            CandidateStagingRepository(self.database_path),
            source,
            track_state=CatalogTrackStateRepository(self.database_path),
        )

    def model(self) -> dict:
        with CanonicalRepository(self.database_path) as repository:
            return repository.load_model()

    def test_fresh_discovery_resolves_relations_and_promotes_automatically(self) -> None:
        outcome = self.orchestrator().ingest((catalog_track(),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.PROMOTED)
        self.assertIsNotNone(outcome.canonical_id)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        # The staged candidate is retired by promotion; its scope transferred to presence.
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))
        with CanonicalRepository(self.database_path) as repository:
            track = next(
                t for t in repository.load_model()["tracks"] if t["id"] == outcome.canonical_id
            )
            self.assertEqual(
                track["external_ids"]["apple_music_catalog_id"], "CATALOG-SONG-1"
            )
            self.assertEqual(track["external_ids"]["isrc"], "USSYN2400001")
            self.assertEqual(track["album_id"], self.catalog_album_id())
            self.assertEqual(track["artist_ids"], [self.catalog_artist_id()])
            self.assertEqual(
                repository.get_source_presence(
                    "apple_music_catalog", EntityType.TRACK, outcome.canonical_id, CATALOG_SCOPE
                ),
                SourcePresence.PRESENT,
            )

    def test_repeated_discovery_converges_on_one_canonical_track(self) -> None:
        orchestrator = self.orchestrator()
        first = orchestrator.ingest((catalog_track(),))[0]
        second = orchestrator.ingest((catalog_track(),))[0]
        self.assertEqual(first.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(second.status, CatalogIngestStatus.ALREADY_BOUND)
        self.assertEqual(second.canonical_id, first.canonical_id)
        model = self.model()
        self.assertEqual(
            sum(
                1
                for track in model["tracks"]
                if track["external_ids"].get("apple_music_catalog_id") == "CATALOG-SONG-1"
            ),
            1,
        )
        self.assertEqual(len(model["tracks"]), len(load_fixture()["tracks"]) + 1)

    def test_isrc_bound_to_library_track_excludes_catalog_hit(self) -> None:
        fixture = load_fixture()
        library_track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(
                ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001"), library_track_id
            )
            repository.save_model_with_source_presence(
                fixture,
                [
                    SourcePresenceRecord(
                        "apple_music",
                        EntityType.TRACK,
                        library_track_id,
                        "library_tracks",
                        SourcePresence.PRESENT,
                    )
                ],
            )
        outcome = self.orchestrator().ingest((catalog_track(),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.LIBRARY_KNOWN)
        self.assertEqual(outcome.canonical_id, library_track_id)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    def test_isrc_bound_without_library_presence_resolves_same_canonical_without_staging(self) -> None:
        fixture = load_fixture()
        known_canonical = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(
                ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001"), known_canonical
            )
        outcome = self.orchestrator().ingest((catalog_track(),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.ALREADY_BOUND)
        self.assertEqual(outcome.canonical_id, known_canonical)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    def test_hit_without_isrc_promotes_without_secondary_identities(self) -> None:
        outcome = self.orchestrator().ingest((catalog_track(isrc=None),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.PROMOTED)
        track = next(t for t in self.model()["tracks"] if t["id"] == outcome.canonical_id)
        self.assertIsNone(track["external_ids"].get("isrc"))

    def test_discover_and_ingest_wires_catalog_source(self) -> None:
        source = FakeCatalogSource((catalog_track(),))
        outcomes = self.orchestrator(source).discover_and_ingest("catalog song", 5)
        self.assertEqual(source.calls, [("catalog song", 5)])
        self.assertEqual([outcome.status for outcome in outcomes], [CatalogIngestStatus.PROMOTED])

    def test_build_catalog_candidate_maps_metadata_facts(self) -> None:
        candidate = build_catalog_candidate(catalog_track())
        self.assertEqual(candidate.external_identity.external_id, "CATALOG-SONG-1")
        self.assertEqual(candidate.external_identity.source_system, "apple_music_catalog")
        self.assertEqual(candidate.source_facts["name"].payload, "Catalog Song CATALOG-SONG-1")
        self.assertEqual(candidate.source_facts["genres"].payload, ["Synthetic"])
        self.assertEqual(candidate.source_facts["duration_ms"].payload, 201000)
        self.assertNotIn("library_state.favorited", candidate.source_facts)

    # --- T2: authoritative relation resolution ---------------------------------

    def catalog_artist_id(self) -> str:
        return self.catalog_artists(ARTIST_KEY)[0]["id"]

    def catalog_album_id(self) -> str:
        return self.catalog_albums(ALBUM_KEY)[0]["id"]

    def catalog_artists(self, catalog_id: str) -> list[dict]:
        return [
            artist
            for artist in self.model()["artists"]
            if artist["external_ids"].get("apple_music_catalog_id") == catalog_id
        ]

    def catalog_albums(self, catalog_id: str) -> list[dict]:
        return [
            album
            for album in self.model()["albums"]
            if album["external_ids"].get("apple_music_catalog_id") == catalog_id
        ]

    def test_new_catalog_artist_id_creates_one_canonical_artist_with_binding(self) -> None:
        self.orchestrator().ingest((catalog_track(),))
        artists = self.catalog_artists(ARTIST_KEY)
        self.assertEqual(len(artists), 1)
        self.assertEqual(artists[0]["name"], "Artist Alpha")
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.ARTIST, ARTIST_KEY)
                ),
                artists[0]["id"],
            )

    def test_repeated_catalog_artist_id_reuses_same_canonical_artist(self) -> None:
        orchestrator = self.orchestrator()
        first = orchestrator.ingest((catalog_track("CATALOG-SONG-1"),))[0]
        second = orchestrator.ingest(
            (catalog_track("CATALOG-SONG-2", isrc="USSYN2400002", album_catalog_id="CATALOG-ALBUM-2"),)
        )[0]
        self.assertEqual(first.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(second.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(len(self.catalog_artists(ARTIST_KEY)), 1)
        for outcome in (first, second):
            track = next(t for t in self.model()["tracks"] if t["id"] == outcome.canonical_id)
            self.assertEqual(track["artist_ids"], [self.catalog_artist_id()])

    def test_catalog_album_id_creates_then_reuses_one_canonical_album(self) -> None:
        orchestrator = self.orchestrator()
        first = orchestrator.ingest((catalog_track("CATALOG-SONG-1"),))[0]
        second = orchestrator.ingest(
            (catalog_track("CATALOG-SONG-2", isrc="USSYN2400002"),)
        )[0]
        self.assertEqual(first.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(second.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(len(self.catalog_albums(ALBUM_KEY)), 1)
        for outcome in (first, second):
            track = next(t for t in self.model()["tracks"] if t["id"] == outcome.canonical_id)
            self.assertEqual(track["album_id"], self.catalog_album_id())

    def test_name_equality_without_matching_catalog_id_does_not_merge(self) -> None:
        orchestrator = self.orchestrator()
        first = orchestrator.ingest((catalog_track("CATALOG-SONG-1"),))[0]
        second = orchestrator.ingest(
            (
                catalog_track(
                    "CATALOG-SONG-2",
                    isrc="USSYN2400002",
                    artist_catalog_ids=("CATALOG-ARTIST-2",),
                    album_catalog_id="CATALOG-ALBUM-2",
                ),
            )
        )[0]
        self.assertEqual(first.status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(second.status, CatalogIngestStatus.PROMOTED)
        # Identical display names, different authoritative IDs: two of each entity.
        self.assertEqual(len(self.catalog_artists(ARTIST_KEY)), 1)
        self.assertEqual(len(self.catalog_artists("CATALOG-ARTIST-2")), 1)
        self.assertEqual(len(self.catalog_albums(ALBUM_KEY)), 1)
        self.assertEqual(len(self.catalog_albums("CATALOG-ALBUM-2")), 1)
        first_track = next(t for t in self.model()["tracks"] if t["id"] == first.canonical_id)
        second_track = next(t for t in self.model()["tracks"] if t["id"] == second.canonical_id)
        self.assertNotEqual(first_track["artist_ids"], second_track["artist_ids"])
        self.assertNotEqual(first_track["album_id"], second_track["album_id"])

    def test_missing_artist_identity_evidence_blocks_promotion_fail_closed(self) -> None:
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        outcome = self.orchestrator().ingest((catalog_track(artist_catalog_ids=()),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.STAGED_BLOCKED)
        self.assertEqual(outcome.blocker_codes, ("artist_relation_unresolved",))
        self.assertIn("artist identity", outcome.error or "")
        # Still durably staged, and promotion stays blocked: no silent name resolution.
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNotNone(staging.get_candidate(key))
        self.assertEqual(
            promote_staged_track(self.database_path, key).status, PromotionStatus.BLOCKED
        )
        self.assertEqual(len(self.model()["tracks"]), len(load_fixture()["tracks"]))

    def test_missing_album_identity_evidence_blocks_promotion_fail_closed(self) -> None:
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        outcome = self.orchestrator().ingest((catalog_track(album_catalog_id=None),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.STAGED_BLOCKED)
        self.assertEqual(outcome.blocker_codes, ("album_relation_unresolved",))
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNotNone(staging.get_candidate(key))
        self.assertEqual(
            promote_staged_track(self.database_path, key).status, PromotionStatus.BLOCKED
        )

    def test_promoted_track_reaches_catalog_candidate_recommendation_path(self) -> None:
        outcome = self.orchestrator().ingest((catalog_track(),))[0]
        self.assertEqual(outcome.status, CatalogIngestStatus.PROMOTED)
        track = next(t for t in self.model()["tracks"] if t["id"] == outcome.canonical_id)
        # The existing selection rule: catalog-bound canonical tracks become catalog candidates.
        self.assertEqual(track["external_ids"].get("apple_music_catalog_id"), "CATALOG-SONG-1")
        self.assertTrue(track["artist_ids"])
        # A positive genre input on the track's own genre must yield an ELIGIBLE candidate.
        genre = PreferenceTargetReference(PreferenceTargetKind.GENRE, track["genres"][0])
        context = RecommendationContext(
            datetime.now(timezone.utc),
            (
                PreferenceInput.from_inferred(
                    InferredAffinity(genre, PreferenceStrength(PreferenceState.POSITIVE, 0.6))
                ),
            ),
        )
        candidates = generate_catalog_candidates(context, [track])
        matched = [
            candidate
            for candidate in candidates
            if candidate.target.target_id == track["id"]
        ]
        self.assertEqual(len(matched), 1)
        self.assertIs(matched[0].eligibility, Eligibility.ELIGIBLE)


class CountingRepository(CanonicalRepository):
    """Test-level instrumentation: counts canonical model I/O calls (P20 Performance Fix 01).

    ``commit_catalog_discovery`` is the batch counterpart of the old per-track saves, so the
    linear-growth invariant is asserted on load calls plus batch commits plus save calls.
    """

    def __init__(self, database_path) -> None:
        super().__init__(database_path)
        self.load_calls = 0
        self.save_calls = 0
        self.batch_commit_calls = 0

    def load_model(self):
        self.load_calls += 1
        return super().load_model()

    def save_model(self, model):
        self.save_calls += 1
        return super().save_model(model)

    def commit_catalog_discovery(self, model, promotions):
        self.batch_commit_calls += 1
        return super().commit_catalog_discovery(model, promotions)


class FailingRelationsOrchestrator(CatalogIngestionOrchestrator):
    """Deterministic mid-batch failure: relation resolution blows up for one catalog id."""

    def _resolve_relations(self, track, batch):
        if track.catalog_id == "CATALOG-FAIL":
            raise RuntimeError("simulated resolver failure")
        return super()._resolve_relations(track, batch)


class FailingCommitRepository(CountingRepository):
    """Deterministic persistence failure at the single batch commit boundary."""

    def commit_catalog_discovery(self, model, promotions):
        self.batch_commit_calls += 1
        raise RuntimeError("simulated disk full")


def unique_track(
    index: int, *, artist: str = ARTIST_KEY, album: str | None = ALBUM_KEY
) -> CatalogTrack:
    return catalog_track(
        f"CATALOG-SONG-{index}",
        isrc=f"USSYN240{index:04d}",
        artist_catalog_ids=(artist,),
        album_catalog_id=album,
    )


class BatchDiscoveryPersistenceTest(unittest.TestCase):
    """P20 Performance Fix 01: batched catalog discovery must keep old durable semantics
    while amortizing canonical model I/O across the batch."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def fixture_counts(self) -> dict:
        fixture = load_fixture()
        return {
            "tracks": len(fixture["tracks"]),
            "artists": len(fixture["artists"]),
            "albums": len(fixture["albums"]),
        }

    def orchestrator(self, repository=None, source=None) -> CatalogIngestionOrchestrator:
        return CatalogIngestionOrchestrator(
            repository or CanonicalRepository(self.database_path),
            CandidateStagingRepository(self.database_path),
            source,
            track_state=CatalogTrackStateRepository(self.database_path),
        )

    def model(self) -> dict:
        with CanonicalRepository(self.database_path) as repository:
            return repository.load_model()

    def catalog_artists(self, catalog_id: str) -> list[dict]:
        return [
            artist
            for artist in self.model()["artists"]
            if artist["external_ids"].get("apple_music_catalog_id") == catalog_id
        ]

    def catalog_albums(self, catalog_id: str) -> list[dict]:
        return [
            album
            for album in self.model()["albums"]
            if album["external_ids"].get("apple_music_catalog_id") == catalog_id
        ]

    # --- constant-I/O invariants -------------------------------------------------

    def test_multi_result_batch_keeps_model_io_constant_regardless_of_batch_size(self) -> None:
        repository = CountingRepository(self.database_path)
        outcomes = self.orchestrator(repository).ingest(
            tuple(unique_track(i, artist=f"ART-{i}", album=f"ALB-{i}") for i in range(48))
        )
        self.assertEqual(len(outcomes), 48)
        self.assertTrue(all(o.status is CatalogIngestStatus.PROMOTED for o in outcomes))
        # One load, one batch commit, zero per-track saves -- no matter how many results.
        self.assertEqual(repository.load_calls, 1)
        self.assertEqual(repository.batch_commit_calls, 1)
        self.assertEqual(repository.save_calls, 0)
        counts = self.fixture_counts()
        model = self.model()
        self.assertEqual(len(model["tracks"]), counts["tracks"] + 48)
        self.assertEqual(len(model["artists"]), counts["artists"] + 48)
        self.assertEqual(len(model["albums"]), counts["albums"] + 48)

    def test_already_bound_batch_performs_no_model_io(self) -> None:
        repository = CountingRepository(self.database_path)
        orchestrator = self.orchestrator(repository)
        first = orchestrator.ingest((unique_track(1), unique_track(2)))
        self.assertEqual(repository.load_calls, 1)
        self.assertEqual(repository.batch_commit_calls, 1)
        repository.load_calls = 0
        repository.batch_commit_calls = 0
        second = orchestrator.ingest((unique_track(1), unique_track(2)))
        self.assertEqual(
            [o.status for o in second],
            [CatalogIngestStatus.ALREADY_BOUND, CatalogIngestStatus.ALREADY_BOUND],
        )
        self.assertEqual(second[0].canonical_id, first[0].canonical_id)
        self.assertEqual(second[1].canonical_id, first[1].canonical_id)
        self.assertEqual(repository.load_calls, 0)
        self.assertEqual(repository.batch_commit_calls, 0)

    # --- in-batch identity convergence ------------------------------------------

    def test_batch_reuses_shared_artist_and_album_entities(self) -> None:
        outcomes = self.orchestrator().ingest(
            tuple(unique_track(i, artist="ART-SHARED", album="ALB-SHARED") for i in range(10))
        )
        self.assertTrue(all(o.status is CatalogIngestStatus.PROMOTED for o in outcomes))
        artists = self.catalog_artists("ART-SHARED")
        albums = self.catalog_albums("ALB-SHARED")
        self.assertEqual(len(artists), 1)
        self.assertEqual(len(albums), 1)
        model = self.model()
        for outcome in outcomes:
            track = next(t for t in model["tracks"] if t["id"] == outcome.canonical_id)
            self.assertEqual(track["artist_ids"], [artists[0]["id"]])
            self.assertEqual(track["album_id"], albums[0]["id"])

    def test_duplicate_catalog_id_within_batch_converges_on_one_canonical_track(self) -> None:
        outcomes = self.orchestrator().ingest((unique_track(1), unique_track(1)))
        self.assertEqual(outcomes[0].status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(outcomes[1].status, CatalogIngestStatus.ALREADY_BOUND)
        self.assertEqual(outcomes[1].canonical_id, outcomes[0].canonical_id)
        counts = self.fixture_counts()
        self.assertEqual(len(self.model()["tracks"]), counts["tracks"] + 1)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    def test_shared_isrc_within_batch_resolves_to_the_same_canonical_track(self) -> None:
        first = unique_track(1, artist="ART-1", album="ALB-1")
        second = catalog_track(
            "CATALOG-SONG-2",
            isrc="USSYN2400001",
            artist_catalog_ids=("ART-2",),
            album_catalog_id="ALB-2",
        )
        outcomes = self.orchestrator().ingest((first, second))
        self.assertEqual(outcomes[0].status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(outcomes[1].status, CatalogIngestStatus.ALREADY_BOUND)
        self.assertEqual(outcomes[1].canonical_id, outcomes[0].canonical_id)
        counts = self.fixture_counts()
        self.assertEqual(len(self.model()["tracks"]), counts["tracks"] + 1)
        # Sequential-flow semantics: the second hit never staged, so its catalog id stays
        # unbound onto the already-canonical track (migration-0017 namespaces are distinct).
        with CanonicalRepository(self.database_path) as repository:
            self.assertIsNone(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-2")
                )
            )

    # --- result equivalence with the old per-track flow --------------------------

    def test_batch_ingestion_matches_sequential_single_track_ingestion(self) -> None:
        tracks = (
            unique_track(1, artist="ART-1", album="ALB-1"),
            unique_track(2, artist="ART-1", album="ALB-2"),
            unique_track(3, artist="ART-2", album="ALB-1"),
            unique_track(1, artist="ART-1", album="ALB-1"),  # duplicate within batch
            catalog_track("CATALOG-SONG-4", isrc=None, artist_catalog_ids=("ART-2",), album_catalog_id="ALB-2"),
            catalog_track("CATALOG-SONG-5", isrc=None, artist_catalog_ids=(), album_catalog_id="ALB-3"),
            catalog_track("CATALOG-SONG-6", isrc=None, artist_catalog_ids=("ART-3",), album_catalog_id=None),
        )
        expected_statuses = [
            CatalogIngestStatus.PROMOTED,
            CatalogIngestStatus.PROMOTED,
            CatalogIngestStatus.PROMOTED,
            CatalogIngestStatus.ALREADY_BOUND,
            CatalogIngestStatus.PROMOTED,
            CatalogIngestStatus.STAGED_BLOCKED,
            CatalogIngestStatus.STAGED_BLOCKED,
        ]
        # The canonical-ID-free projection of the durable identity structure: each hit sees
        # exactly the artist/album Catalog identities the old sequential flow bound for it.
        expected_structure = [
            (("ART-1",), "ALB-1"),
            (("ART-1",), "ALB-2"),
            (("ART-2",), "ALB-1"),
            (("ART-1",), "ALB-1"),
            (("ART-2",), "ALB-2"),
            None,
            None,
        ]
        for run in (self._ingest_fresh_db(tracks, batch=True), self._ingest_fresh_db(tracks, batch=False)):
            repository, outcomes = run
            model = repository.load_model()
            counts = self.fixture_counts()
            self.assertEqual(len(model["tracks"]), counts["tracks"] + 4)
            self.assertEqual(len(model["artists"]), counts["artists"] + 2)
            self.assertEqual(len(model["albums"]), counts["albums"] + 2)
            self.assertEqual([o.status for o in outcomes], expected_statuses)
            self.assertEqual(
                [self._structure_label(model, o) for o in outcomes], expected_structure
            )

    def _ingest_fresh_db(self, tracks, *, batch: bool):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "canonical.sqlite3"
        with CanonicalRepository(path) as repository:
            repository.save_model(load_fixture())
        orchestrator = CatalogIngestionOrchestrator(
            CanonicalRepository(path),
            CandidateStagingRepository(path),
            None,
            track_state=CatalogTrackStateRepository(path),
        )
        if batch:
            outcomes = orchestrator.ingest(tracks)
        else:
            outcomes = tuple(
                outcome for track in tracks for outcome in orchestrator.ingest((track,))
            )
        repository = CanonicalRepository(path)
        self.addCleanup(repository.close)
        return repository, outcomes

    def _structure_label(self, model, outcome):
        if outcome.canonical_id is None:
            return None
        track = next(t for t in model["tracks"] if t["id"] == outcome.canonical_id)
        artist_labels = tuple(
            sorted(
                artist["external_ids"].get("apple_music_catalog_id")
                for artist in model["artists"]
                if artist["id"] in set(track["artist_ids"])
            )
        )
        album_label = None
        if track["album_id"] is not None:
            album = next(a for a in model["albums"] if a["id"] == track["album_id"])
            album_label = album["external_ids"].get("apple_music_catalog_id")
        return (artist_labels, album_label)

    # --- classification and exclusion semantics within a batch --------------------

    def test_library_known_classification_inside_a_batch(self) -> None:
        fixture = load_fixture()
        library_track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(
                ExternalIdentityKey("isrc", EntityType.TRACK, "USSYN2400001"), library_track_id
            )
            repository.save_model_with_source_presence(
                fixture,
                [
                    SourcePresenceRecord(
                        "apple_music",
                        EntityType.TRACK,
                        library_track_id,
                        "library_tracks",
                        SourcePresence.PRESENT,
                    )
                ],
            )
        outcomes = self.orchestrator().ingest((unique_track(2), unique_track(1)))
        self.assertEqual(outcomes[0].status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(outcomes[1].status, CatalogIngestStatus.LIBRARY_KNOWN)
        self.assertEqual(outcomes[1].canonical_id, library_track_id)
        counts = self.fixture_counts()
        # The library-known hit never staged or resolved: only CATALOG-SONG-2 persisted.
        self.assertEqual(len(self.model()["tracks"]), counts["tracks"] + 1)
        self.assertEqual(len(self.catalog_artists(ARTIST_KEY)), 1)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))

    # --- failure isolation ------------------------------------------------------

    def test_mid_batch_failure_isolates_to_that_track_only(self) -> None:
        orchestrator = FailingRelationsOrchestrator(
            CanonicalRepository(self.database_path),
            CandidateStagingRepository(self.database_path),
            None,
            track_state=CatalogTrackStateRepository(self.database_path),
        )
        outcomes = orchestrator.ingest(
            (
                unique_track(1),
                catalog_track("CATALOG-FAIL", isrc="USSYN2419999"),
                unique_track(3),
            )
        )
        self.assertEqual(
            [o.status for o in outcomes],
            [
                CatalogIngestStatus.PROMOTED,
                CatalogIngestStatus.STAGING_FAILED,
                CatalogIngestStatus.PROMOTED,
            ],
        )
        self.assertIn("simulated resolver failure", outcomes[1].error)
        counts = self.fixture_counts()
        model = self.model()
        self.assertEqual(len(model["tracks"]), counts["tracks"] + 2)
        self.assertEqual(len(model["artists"]), counts["artists"] + 1)
        self.assertEqual(len(model["albums"]), counts["albums"] + 1)
        # The failed hit stays staged (fail closed) and never corrupted its neighbors.
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-FAIL")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNotNone(staging.get_candidate(key))

    def test_blocked_track_relations_persist_within_a_batch(self) -> None:
        # Relations resolve before evaluation; an evaluation-blocked hit must leave its
        # Artist / Album entities durable, exactly like the old per-track flow.
        empty_name = CatalogTrack(
            catalog_id="CATALOG-BLOCKED",
            name="",
            artist_names=("Art Blk",),
            album_name="Alb Blk",
            genres=("Synthetic",),
            isrc="USSYN2412345",
            duration_ms=201000,
            release_date="2024-01-15",
            url=None,
            artist_catalog_ids=("ART-BLK",),
            album_catalog_id="ALB-BLK",
        )
        outcomes = self.orchestrator().ingest((unique_track(2), empty_name))
        self.assertEqual(outcomes[0].status, CatalogIngestStatus.PROMOTED)
        self.assertEqual(outcomes[1].status, CatalogIngestStatus.STAGED_BLOCKED)
        self.assertEqual(outcomes[1].blocker_codes, ("missing_name",))
        counts = self.fixture_counts()
        model = self.model()
        self.assertEqual(len(model["tracks"]), counts["tracks"] + 1)
        self.assertEqual(len(self.catalog_artists("ART-BLK")), 1)
        self.assertEqual(len(self.catalog_albums("ALB-BLK")), 1)
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-BLOCKED")
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNotNone(staging.get_candidate(key))

    def test_commit_failure_rolls_back_the_whole_batch_without_half_writes(self) -> None:
        repository = FailingCommitRepository(self.database_path)
        tracks = (unique_track(1), unique_track(2), unique_track(3))
        outcomes = self.orchestrator(repository).ingest(tracks)
        self.assertTrue(all(o.status is CatalogIngestStatus.STAGING_FAILED for o in outcomes))
        self.assertTrue(
            all("batch persistence failed" in (o.error or "") for o in outcomes)
        )
        counts = self.fixture_counts()
        model = self.model()
        self.assertEqual(len(model["tracks"]), counts["tracks"])
        self.assertEqual(len(model["artists"]), counts["artists"])
        self.assertEqual(len(model["albums"]), counts["albums"])
        # Fail closed: every candidate stayed staged and no discovery event was recorded.
        with CandidateStagingRepository(self.database_path) as staging:
            for track in tracks:
                key = ExternalIdentityKey(
                    "apple_music_catalog", EntityType.TRACK, track.catalog_id
                )
                self.assertIsNotNone(staging.get_candidate(key))
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.count(), 0)

    # --- idempotence and downstream reach ---------------------------------------

    def test_repeated_batch_ingestion_is_idempotent(self) -> None:
        tracks = tuple(unique_track(i) for i in range(5))
        orchestrator = self.orchestrator()
        first = orchestrator.ingest(tracks)
        second = orchestrator.ingest(tracks)
        self.assertTrue(all(o.status is CatalogIngestStatus.ALREADY_BOUND for o in second))
        self.assertEqual(
            tuple(o.canonical_id for o in second), tuple(o.canonical_id for o in first)
        )
        counts = self.fixture_counts()
        model = self.model()
        self.assertEqual(len(model["tracks"]), counts["tracks"] + 5)
        self.assertEqual(len(model["artists"]), counts["artists"] + 1)
        self.assertEqual(len(model["albums"]), counts["albums"] + 1)
        with CatalogTrackStateRepository(self.database_path) as state:
            for outcome in first:
                self.assertEqual(state.get_state(outcome.canonical_id).discovery_count, 2)

    def test_batch_promotions_reach_catalog_candidate_recommendation_path(self) -> None:
        outcomes = self.orchestrator().ingest(
            (
                unique_track(1, artist="ART-1", album="ALB-1"),
                unique_track(2, artist="ART-2", album="ALB-2"),
                unique_track(3, artist="ART-3", album="ALB-3"),
            )
        )
        self.assertTrue(all(o.status is CatalogIngestStatus.PROMOTED for o in outcomes))
        model = self.model()
        promoted = [
            next(t for t in model["tracks"] if t["id"] == outcome.canonical_id)
            for outcome in outcomes
        ]
        genre = PreferenceTargetReference(PreferenceTargetKind.GENRE, "Synthetic")
        context = RecommendationContext(
            datetime.now(timezone.utc),
            (
                PreferenceInput.from_inferred(
                    InferredAffinity(genre, PreferenceStrength(PreferenceState.POSITIVE, 0.6))
                ),
            ),
        )
        candidates = generate_catalog_candidates(context, promoted)
        matched = {candidate.target.target_id: candidate for candidate in candidates}
        for track in promoted:
            self.assertIs(matched[track["id"]].eligibility, Eligibility.ELIGIBLE)


if __name__ == "__main__":
    unittest.main()
