"""P11-T3: iTunes Store discovery end-to-end -- the free runtime path.

Proves the whole credential-free discovery path over a fake catalog source returning
``itunes_store`` CatalogTracks: staging under the new namespace (migration 0018), identity
convergence, no name-based merging, authoritative relation resolution under
``itunes_store``, automatic promotion, recommendation visibility, and ``previewUrl``
survival through the provider/domain path.
"""

import json
import tempfile
import unittest
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.apple_music_catalog import CatalogTrack
from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.catalog_ingestion import CATALOG_SCOPE
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.itunes_search import ITUNES_SOURCE_SYSTEM
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue, SourcePresence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"

TRACK_ID = "1258917044"
ARTIST_ID = "148607010"
COLLECTION_ID = "1258917041"
PREVIEW_URL = "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview125/v4/delicate.m4a"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def itunes_track(
    *,
    track_id: str = TRACK_ID,
    name: str = "Delicate",
    artist_id: str = ARTIST_ID,
    artist_name: str = "Taylor Swift",
    collection_id: str | None = COLLECTION_ID,
    album_name: str | None = "reputation",
    genre: str = "Pop",
) -> CatalogTrack:
    return CatalogTrack(
        catalog_id=track_id,
        name=name,
        artist_names=(artist_name,),
        album_name=album_name,
        genres=(genre,),
        isrc=None,
        duration_ms=232861,
        release_date="2017-11-10",
        url=None,
        artist_catalog_ids=(artist_id,),
        album_catalog_id=collection_id,
        source_system=ITUNES_SOURCE_SYSTEM,
        preview_url=PREVIEW_URL,
    )


class FakeCatalogSource:
    def __init__(self, tracks: tuple[CatalogTrack, ...]) -> None:
        self.tracks = tracks
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> tuple[CatalogTrack, ...]:
        self.calls.append((term, limit))
        return self.tracks


class ITunesIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())

    def client_for(self, source: FakeCatalogSource) -> AgentClient:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.FULL}),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_FULL, model_id="test", label="tests"),
            service,
        )

    def discover(self, tracks: tuple[CatalogTrack, ...], term: str = "delicate") -> dict:
        source = FakeCatalogSource(tracks)
        result = self.client_for(source).call("discover_catalog_tracks", {"term": term})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(source.calls, [(term, 25)])
        return result.payload

    def load_model(self) -> dict:
        with CanonicalRepository(self.database_path) as repository:
            return repository.load_model()

    def test_discovery_promotes_with_itunes_store_identities(self) -> None:
        payload = self.discover((itunes_track(),))
        self.assertEqual(payload["promoted_count"], 1)
        promoted = payload["promoted"][0]
        self.assertEqual(promoted["catalog_id"], TRACK_ID)
        self.assertEqual(promoted["status"], "promoted")
        self.assertEqual(promoted["preview_url"], PREVIEW_URL)

        model = self.load_model()
        track = next(t for t in model["tracks"] if t["id"] == promoted["canonical_id"])
        # iTunes Store IDs, never Apple Music Catalog IDs.
        self.assertEqual(track["external_ids"]["itunes_store_id"], TRACK_ID)
        self.assertNotIn("apple_music_catalog_id", track["external_ids"])
        artist = next(a for a in model["artists"] if a["id"] == track["artist_ids"][0])
        self.assertEqual(artist["external_ids"]["itunes_store_id"], ARTIST_ID)
        self.assertNotIn("apple_music_catalog_id", artist["external_ids"])
        album = next(a for a in model["albums"] if a["id"] == track["album_id"])
        self.assertEqual(album["external_ids"]["itunes_store_id"], COLLECTION_ID)
        self.assertNotIn("apple_music_catalog_id", album["external_ids"])

        # Candidate retired; presence recorded under the itunes_store source system.
        key = ExternalIdentityKey(ITUNES_SOURCE_SYSTEM, EntityType.TRACK, TRACK_ID)
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                repository.get_source_presence(
                    ITUNES_SOURCE_SYSTEM, EntityType.TRACK, promoted["canonical_id"], CATALOG_SCOPE
                ),
                SourcePresence.PRESENT,
            )

    def test_repeated_discovery_converges_on_one_canonical_track(self) -> None:
        first = self.discover((itunes_track(),))
        second = self.discover((itunes_track(),))
        self.assertEqual(second["promoted_count"], 0)
        self.assertEqual(second["skipped_count"], 1)
        self.assertEqual(second["skipped"][0]["status"], "already_bound")
        self.assertEqual(second["skipped"][0]["canonical_id"], first["promoted"][0]["canonical_id"])
        model = self.load_model()
        matches = [
            t for t in model["tracks"]
            if t["external_ids"].get("itunes_store_id") == TRACK_ID
        ]
        self.assertEqual(len(matches), 1)

    def test_same_names_with_different_ids_do_not_merge(self) -> None:
        payload = self.discover(
            (itunes_track(track_id="1000000001"), itunes_track(track_id="2000000002"))
        )
        self.assertEqual(payload["promoted_count"], 2)
        first, second = payload["promoted"]
        self.assertNotEqual(first["canonical_id"], second["canonical_id"])
        model = self.load_model()
        delicate_tracks = [
            t for t in model["tracks"] if t["name"] == "Delicate"
        ]
        self.assertEqual(
            {t["external_ids"].get("itunes_store_id") for t in delicate_tracks},
            {"1000000001", "2000000002"},
        )
        # Same artistId converges on one canonical Artist, even across two promotions.
        self.assertEqual(
            len({t["artist_ids"][0] for t in delicate_tracks}), 1
        )

    def test_missing_album_identity_stays_staged_with_preview_url_intact(self) -> None:
        payload = self.discover((itunes_track(collection_id=None, album_name=None),))
        self.assertEqual(payload["promoted_count"], 0)
        self.assertEqual(payload["staged_count"], 1)
        staged = payload["staged"][0]
        self.assertEqual(staged["status"], "staged_blocked")
        self.assertEqual(staged["blocker_codes"], ["album_relation_unresolved"])
        self.assertEqual(staged["preview_url"], PREVIEW_URL)
        # Durable staging under the itunes_store namespace carries the preview fact for T4.
        key = ExternalIdentityKey(ITUNES_SOURCE_SYSTEM, EntityType.TRACK, TRACK_ID)
        with CandidateStagingRepository(self.database_path) as staging:
            candidate = staging.get_candidate(key)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(
            candidate.source_facts["preview_url"].payload, PREVIEW_URL
        )

    def test_promoted_track_is_recommendation_visible(self) -> None:
        # One library track with a Pop preference: the promoted Pop iTunes track must rank.
        fixture = load_fixture()
        target_id = fixture["tracks"][0]["id"]
        fixture["tracks"][0]["genres"] = ["Pop"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        with PreferencePersistenceRepository(self.database_path) as preference:
            preference.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
                    "apple_music",
                    "favorited",
                ),
                ObservedValue.value(True),
            )

        discovered = self.discover((itunes_track(),))
        promoted_id = discovered["promoted"][0]["canonical_id"]

        client = self.client_for(FakeCatalogSource(()))
        result = client.call(
            "generate_inferred_recommendation",
            {"target_ids": [target_id], "limit": 10, "source_system": "apple_music"},
        )
        self.assertEqual(result.outcome.value, "ok")
        items = {item["target_id"]: item for item in result.payload["items"]}
        self.assertIn(promoted_id, items)
        self.assertEqual(items[promoted_id]["label"], "catalog")
        self.assertGreater(items[promoted_id]["score_total"], 0)


if __name__ == "__main__":
    unittest.main()
