"""P11-T1/T2: discover_catalog_tracks -- runtime path from agent tool to automatic promotion.

Proves the whole tool surface: registry/permission behavior, dependency injection, a
successful fake-transport discovery -> durable staging -> authoritative relation resolution
-> automatic promotion with reported status, duplicate/known-hit skipping through the
existing P11 ingestion semantics, fail-closed staged_blocked reporting when authoritative
identity evidence is missing, and fail-closed typed errors for missing credentials /
transport failures / an unwired catalog source.
"""

import json
import tempfile
import unittest
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import (
    SharedAgentService,
    SharedAgentServiceValidationError,
)
from music_agent.apple_music_catalog import (
    CatalogCredentialsError,
    CatalogTrack,
    CatalogTransportError,
)
from music_agent.candidate_staging import CandidateStagingRepository
from music_agent.catalog_ingestion import CATALOG_SCOPE
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.promotion import PromotionStatus, promote_staged_track
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import SourcePresence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"
CLIENT_READ_ONLY = "agt_22222222-2222-4222-8222-222222222222"
ISRC = "USSYN2400001"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def catalog_track(
    catalog_id: str,
    *,
    isrc: str | None = ISRC,
    artist_catalog_ids: tuple[str, ...] = ("CATALOG-ARTIST-1",),
    album_catalog_id: str | None = "CATALOG-ALBUM-1",
) -> CatalogTrack:
    return CatalogTrack(
        catalog_id=catalog_id,
        name=f"Catalog Song {catalog_id}",
        artist_names=("Artist Alpha",),
        album_name="Catalog Album",
        genres=("Synthetic",),
        isrc=isrc,
        duration_ms=201000,
        release_date="2024-01-15",
        url=None,
        artist_catalog_ids=artist_catalog_ids,
        album_catalog_id=album_catalog_id,
    )


class FakeCatalogSource:
    def __init__(self, tracks: tuple[CatalogTrack, ...] = (), error: Exception | None = None) -> None:
        self.tracks = tracks
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> tuple[CatalogTrack, ...]:
        self.calls.append((term, limit))
        if self.error is not None:
            raise self.error
        return self.tracks


_UNWIRED = object()


class CatalogDiscoveryToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def service(self, source=_UNWIRED, policy: AgentClientPolicy = AgentClientPolicy.FULL) -> SharedAgentService:
        clients = AgentClientRegistry(
            {
                CLIENT_FULL: policy,
                CLIENT_READ_ONLY: AgentClientPolicy.READ_ONLY,
            }
        )
        kwargs = {} if source is _UNWIRED else {"catalog_search_source": source}
        service = SharedAgentService(self.database_path, clients=clients, **kwargs)
        self.addCleanup(service.close)
        return service

    def client(self, service: SharedAgentService, client_id: str = CLIENT_FULL) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=client_id, model_id="test", label="tests"),
            service,
        )

    def test_discovery_resolves_relations_and_promotes_automatically(self) -> None:
        source = FakeCatalogSource((catalog_track("CATALOG-SONG-1"),))
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(source.calls, [("catalog song", 25)])
        payload = result.payload
        self.assertEqual(payload["discovered_count"], 1)
        self.assertEqual(payload["promoted_count"], 1)
        self.assertEqual(payload["staged_count"], 0)
        self.assertEqual(payload["skipped_count"], 0)
        promoted = payload["promoted"][0]
        self.assertEqual(promoted["catalog_id"], "CATALOG-SONG-1")
        self.assertEqual(promoted["name"], "Catalog Song CATALOG-SONG-1")
        self.assertEqual(promoted["artist_name"], "Artist Alpha")
        self.assertEqual(promoted["status"], "promoted")
        self.assertIsNotNone(promoted["canonical_id"])
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        # Promotion retired the staged candidate and created the canonical Track + binding.
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertIsNone(staging.get_candidate(key))
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            track = next(
                track for track in model["tracks"] if track["id"] == promoted["canonical_id"]
            )
            self.assertEqual(
                track["external_ids"]["apple_music_catalog_id"], "CATALOG-SONG-1"
            )
            self.assertTrue(track["artist_ids"])
            self.assertIsNotNone(track["album_id"])
            artist = next(
                artist for artist in model["artists"] if artist["id"] == track["artist_ids"][0]
            )
            self.assertEqual(
                artist["external_ids"]["apple_music_catalog_id"], "CATALOG-ARTIST-1"
            )
            self.assertEqual(
                repository.get_source_presence(
                    "apple_music_catalog",
                    EntityType.TRACK,
                    promoted["canonical_id"],
                    CATALOG_SCOPE,
                ),
                SourcePresence.PRESENT,
            )

    def test_missing_identity_evidence_reports_staged_blocked(self) -> None:
        source = FakeCatalogSource((catalog_track("CATALOG-SONG-1", artist_catalog_ids=()),))
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "ok")
        payload = result.payload
        self.assertEqual(payload["promoted_count"], 0)
        self.assertEqual(payload["staged_count"], 1)
        self.assertEqual(payload["staged"][0]["status"], "staged_blocked")
        self.assertEqual(
            payload["staged"][0]["blocker_codes"], ["artist_relation_unresolved"]
        )
        self.assertIn("artist identity", payload["staged"][0]["error"] or "")
        key = ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, "CATALOG-SONG-1")
        # Still durably staged, promotion stays blocked: names never resolve a relation.
        with CandidateStagingRepository(self.database_path) as staging:
            self.assertEqual(staging.list_candidate_scopes(key), (CATALOG_SCOPE,))
        blocked = promote_staged_track(self.database_path, key)
        self.assertEqual(blocked.status, PromotionStatus.BLOCKED)

    def test_known_isrc_hit_is_skipped_and_reported(self) -> None:
        fixture = load_fixture()
        library_track_id = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(
                ExternalIdentityKey("isrc", EntityType.TRACK, ISRC), library_track_id
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
        source = FakeCatalogSource((catalog_track("CATALOG-SONG-1"),))
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "ok")
        payload = result.payload
        self.assertEqual(payload["staged_count"], 0)
        self.assertEqual(payload["skipped_count"], 1)
        self.assertEqual(payload["skipped"][0]["status"], "library_known")
        self.assertEqual(payload["skipped"][0]["canonical_id"], library_track_id)

    def test_missing_credentials_fail_closed_with_actionable_result(self) -> None:
        source = FakeCatalogSource(
            error=CatalogCredentialsError(
                "Apple Music developer token is required; "
                "set MUSIC_AGENT_APPLE_MUSIC_DEVELOPER_TOKEN"
            )
        )
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_credentials_missing")
        self.assertIn("MUSIC_AGENT_APPLE_MUSIC_DEVELOPER_TOKEN", result.error_message)

    def test_transport_failure_maps_to_typed_error(self) -> None:
        source = FakeCatalogSource(
            error=CatalogTransportError("catalog search failed with HTTP 500: boom")
        )
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_transport_failed")
        self.assertIn("HTTP 500", result.error_message)

    def test_unwired_service_fails_closed(self) -> None:
        result = self.client(self.service()).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_discovery_unavailable")

    def test_constructor_rejects_bad_catalog_source(self) -> None:
        with self.assertRaises(SharedAgentServiceValidationError):
            SharedAgentService(
                self.database_path,
                clients=AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.FULL}),
                catalog_search_source=object(),
            )

    def test_read_only_client_is_denied_before_any_search(self) -> None:
        source = FakeCatalogSource((catalog_track("CATALOG-SONG-1"),))
        result = self.client(self.service(source), client_id=CLIENT_READ_ONLY).call(
            "discover_catalog_tracks", {"term": "catalog song"}
        )
        self.assertEqual(result.outcome.value, "permission_denied")
        self.assertEqual(source.calls, [])

    def test_limit_reaches_the_injected_source(self) -> None:
        source = FakeCatalogSource((catalog_track("CATALOG-SONG-1"),))
        result = self.client(self.service(source)).call(
            "discover_catalog_tracks", {"term": "catalog song", "limit": 3}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(source.calls, [("catalog song", 3)])
        self.assertEqual(result.payload["limit"], 3)


    def test_multi_result_discovery_reports_aggregate_counts_and_converges(self) -> None:
            # P20 Performance Fix 01: one search of 10 results -- 8 fresh + 1 duplicate within
            # the result set + 1 missing evidence -- promoted/skipped/staged via one atomic
            # batch persistence pass, then converges on a repeat call.
            fresh = tuple(
                catalog_track(
                    f"CATALOG-SONG-{index}",
                    isrc=f"USSYN240{index:04d}",
                    artist_catalog_ids=(f"CATALOG-ARTIST-{index}",),
                    album_catalog_id=f"CATALOG-ALBUM-{index}",
                )
                for index in range(1, 9)
            )
            duplicate = catalog_track("CATALOG-SONG-1", isrc="USSYN2400001")
            blocked = catalog_track("CATALOG-BLOCKED", isrc=None, artist_catalog_ids=())
            source = FakeCatalogSource(fresh + (duplicate, blocked))
            service = self.service(source)

            first = self.client(service).call(
                "discover_catalog_tracks", {"term": "catalog song"}
            )
            self.assertEqual(first.outcome.value, "ok")
            payload = first.payload
            self.assertEqual(payload["discovered_count"], 10)
            self.assertEqual(payload["promoted_count"], 8)
            self.assertEqual(payload["staged_count"], 1)
            self.assertEqual(payload["skipped_count"], 1)
            self.assertEqual(payload["skipped"][0]["status"], "already_bound")
            self.assertEqual(payload["staged"][0]["status"], "staged_blocked")
            promoted_ids = {item["canonical_id"] for item in payload["promoted"]}
            self.assertEqual(len(promoted_ids), 8)
            self.assertEqual(
                payload["skipped"][0]["canonical_id"], payload["promoted"][0]["canonical_id"]
            )

            # A repeat search converges: all 8 fresh hits are now already bound, the blocked
            # candidate is still staged, and nothing new is created.
            repeated = self.client(service).call(
                "discover_catalog_tracks", {"term": "catalog song"}
            )
            self.assertEqual(repeated.payload["promoted_count"], 0)
            self.assertEqual(repeated.payload["staged_count"], 1)
            self.assertEqual(repeated.payload["skipped_count"], 9)
            with CanonicalRepository(self.database_path) as repository:
                model = repository.load_model()
            fixture_tracks = len(load_fixture()["tracks"])
            self.assertEqual(len(model["tracks"]), fixture_tracks + 8)
            self.assertEqual(
                sum(
                    1
                    for artist in model["artists"]
                    if artist["external_ids"].get("apple_music_catalog_id", "").startswith(
                        "CATALOG-ARTIST-"
                    )
                ),
                8,
            )


if __name__ == "__main__":
    unittest.main()
