"""P11.3: catalog -> library add, readback, reconciliation, and preview capability tests."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from music_agent.agent_client import AgentClient, AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.catalog_library import (
    CatalogLibraryCredentialsError,
    CatalogLibraryTransportError,
    LibraryAddOutcome,
    MusicKitLibraryTransport,
    add_catalog_song_to_library,
    parse_add_response,
    parse_library_search,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.library_sync import LibrarySyncOrchestrator
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import SourcePresence
from music_agent.write_intent import WRITE_CAPABILITY_MATRIX, WriteOperation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CATALOG_ID = "CATALOG-SONG-1"
ISRC = "USSYN2400001"
PERSISTENT_ID = "PERSIST-001"
ITUNES_ID = "1258917044"
PREVIEW_URL = "https://audio-ssl.itunes.apple.com/itunes-assets/preview.m4a"
CLIENT_ID = "agt_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def library_entry(catalog_id: str = CATALOG_ID, library_id: str = PERSISTENT_ID) -> dict:
    return {
        "id": library_id,
        "type": "library-songs",
        "attributes": {
            "name": "Catalog Song",
            "playParams": {"catalogId": catalog_id},
            "isrc": ISRC,
        },
    }


def add_payload(**overrides) -> str:
    return json.dumps({"data": [library_entry(**overrides)]})


def search_payload(entries: list | None) -> str:
    return json.dumps({"results": {"library-songs": {"data": entries if entries is not None else []}}})


class FakeLibraryTransport:
    def __init__(self, add_output: str, search_output: str) -> None:
        self.add_output = add_output
        self.search_output = search_output
        self.add_calls: list[str] = []
        self.search_calls: list[tuple[str, int]] = []

    def add_song(self, catalog_id: str) -> str:
        self.add_calls.append(catalog_id)
        return self.add_output

    def search_library_songs(self, term: str, limit: int = 25) -> str:
        self.search_calls.append((term, limit))
        return self.search_output


class FakePreviewRunner:
    def __init__(self) -> None:
        self.audios: list[str] = []
        self.stop_calls = 0
        self.active = False

    def start_audio(self, url: str) -> None:
        self.audios.append(url)
        self.active = True

    def stop_preview(self) -> bool:
        self.stop_calls += 1
        self.active = False
        return bool(self.audios)

    def is_preview_active(self) -> bool:
        # P14-C06.3b: read-only truth, kept independent of the audios history.
        return self.active


class FakeItunesLookupSource:
    """Deterministic stand-in for the iTunes catalog source: search is unused by preview."""

    def __init__(self, preview_url: str | None = PREVIEW_URL) -> None:
        self.preview_url = preview_url
        self.lookup_calls: list[str] = []

    def search(self, term: str, limit: int = 25) -> tuple:
        raise AssertionError("search must not be invoked by preview tests")

    def lookup_preview_url(self, itunes_id: str) -> str | None:
        self.lookup_calls.append(itunes_id)
        return self.preview_url


class CatalogLibraryParsingTest(unittest.TestCase):
    def test_add_response_with_exact_catalog_match_is_added(self) -> None:
        outcome = parse_add_response(CATALOG_ID, add_payload())
        self.assertEqual(outcome.status, "added")
        self.assertEqual(outcome.evidence.library_song_id, PERSISTENT_ID)
        self.assertEqual(outcome.evidence.isrc, ISRC)

    def test_add_response_catalog_mismatch_is_ambiguous(self) -> None:
        outcome = parse_add_response(CATALOG_ID, add_payload(catalog_id="OTHER-ID"))
        self.assertEqual(outcome.status, "ambiguous")

    def test_add_response_with_zero_or_multiple_entries_is_ambiguous(self) -> None:
        self.assertEqual(parse_add_response(CATALOG_ID, '{"data": []}').status, "ambiguous")
        payload = json.dumps({"data": [library_entry(), library_entry()]})
        self.assertEqual(parse_add_response(CATALOG_ID, payload).status, "ambiguous")

    def test_add_response_without_id_is_ambiguous(self) -> None:
        payload = json.dumps({"data": [{"type": "library-songs", "attributes": {"playParams": {"catalogId": CATALOG_ID}}}]})
        self.assertEqual(parse_add_response(CATALOG_ID, payload).status, "ambiguous")

    def test_library_search_match_returns_evidence(self) -> None:
        outcome = parse_library_search(CATALOG_ID, "term", search_payload([library_entry()]))
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.status, "added")
        self.assertEqual(outcome.evidence.library_song_id, PERSISTENT_ID)

    def test_library_search_no_match_returns_none(self) -> None:
        self.assertIsNone(parse_library_search(CATALOG_ID, "term", search_payload([])))
        self.assertIsNone(
            parse_library_search(CATALOG_ID, "term", search_payload([library_entry(catalog_id="OTHER")]))
        )

    def test_library_search_multiple_matches_is_ambiguous(self) -> None:
        outcome = parse_library_search(
            CATALOG_ID, "term", search_payload([library_entry(), library_entry()])
        )
        self.assertEqual(outcome.status, "ambiguous")


class CatalogLibraryReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        fixture = load_fixture()
        self.canonical_id = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
        fixture["tracks"].append(
            {
                "id": self.canonical_id,
                "external_ids": {
                    "apple_music_persistent_id": None,
                    "apple_music_catalog_id": CATALOG_ID,
                },
                "name": "Catalog Song",
                "artist_ids": [fixture["artists"][0]["id"]],
                "album_id": fixture["albums"][0]["id"],
                "duration_ms": 201000,
                "genres": ["Synthetic"],
                "track_number": None,
                "disc_number": None,
                "release_date": "2024-01-15",
                "composer": None,
                "library_state": {
                    "favorited": None, "disliked": None, "rating": None,
                    "play_count": None, "skip_count": None,
                    "added_to_library_at": None, "last_played_at": None,
                },
                "agent_metadata": {"tags": []},
            }
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def transport(self, add: str, search: str) -> FakeLibraryTransport:
        return FakeLibraryTransport(add, search)

    def test_add_readback_reconcile_binds_same_canonical_track(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            result = add_catalog_song_to_library(
                repository,
                self.transport(add_payload(), search_payload([library_entry()])),
                CATALOG_ID,
                self.canonical_id,
            )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.bound_persistent_id, PERSISTENT_ID)
        self.assertEqual(result.bound_isrc, ISRC)
        with CanonicalRepository(self.database_path) as repository:
            # The SAME canonical Track now carries all three identities.
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music_catalog", EntityType.TRACK, CATALOG_ID)
                ),
                self.canonical_id,
            )
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, PERSISTENT_ID)
                ),
                self.canonical_id,
            )
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("isrc", EntityType.TRACK, ISRC)
                ),
                self.canonical_id,
            )
            # No second canonical Track was ever created.
            catalog_bound = [
                track for track in repository.load_model()["tracks"]
                if track["external_ids"].get("apple_music_catalog_id") == CATALOG_ID
            ]
            self.assertEqual(len(catalog_bound), 1)

    def test_readback_absent_after_add_is_not_success(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            result = add_catalog_song_to_library(
                repository,
                self.transport(add_payload(), search_payload([])),
                CATALOG_ID,
                self.canonical_id,
            )
        self.assertFalse(result.succeeded)
        self.assertEqual(result.readback_status, "absent")
        with CanonicalRepository(self.database_path) as repository:
            self.assertIsNone(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, PERSISTENT_ID)
                )
            )

    def test_isrc_conflict_rolls_back_all_bindings(self) -> None:
        fixture = load_fixture()
        other_canonical = fixture["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.bind_external_identity(
                ExternalIdentityKey("isrc", EntityType.TRACK, ISRC), other_canonical
            )
            result = add_catalog_song_to_library(
                repository,
                self.transport(add_payload(), search_payload([library_entry()])),
                CATALOG_ID,
                self.canonical_id,
            )
        self.assertFalse(result.succeeded)
        self.assertEqual(result.readback_status, "ambiguous")
        with CanonicalRepository(self.database_path) as repository:
            # The persistent ID was NOT half-bound: the atomic save rolled back.
            self.assertIsNone(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, PERSISTENT_ID)
                )
            )

    def test_add_failure_is_reported_not_guessed(self) -> None:
        from music_agent.catalog_library import CatalogLibraryTransportError

        class FailingTransport(FakeLibraryTransport):
            def add_song(self, catalog_id: str) -> str:
                raise CatalogLibraryTransportError("HTTP 401: Unauthorized")

        with CanonicalRepository(self.database_path) as repository:
            result = add_catalog_song_to_library(
                repository,
                FailingTransport("", ""),
                CATALOG_ID,
                self.canonical_id,
            )
        self.assertEqual(result.add_status, "failed")

    def test_unbound_canonical_fails_closed(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            with self.assertRaises(ValueError):
                add_catalog_song_to_library(
                    repository,
                    self.transport("", ""),
                    "OTHER-CATALOG",
                    self.canonical_id,
                )

    def test_library_sync_after_reconcile_converges_on_same_canonical_track(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            result = add_catalog_song_to_library(
                repository,
                self.transport(add_payload(), search_payload([library_entry()])),
                CATALOG_ID,
                self.canonical_id,
            )
        self.assertTrue(result.succeeded)
        before = None
        with CanonicalRepository(self.database_path) as repository:
            before = len(repository.load_model()["tracks"])

        # A later full library sync enumerates the new persistent ID: the identity
        # binding converges on the SAME canonical Track instead of ingesting a duplicate.
        class DiscoveryRunner:
            def list_persistent_ids(self) -> tuple[str, ...]:
                return (PERSISTENT_ID,)

        class PerTrackAdapter:
            def read_track(self, persistent_id: str):
                from music_agent.apple_music import RawTrackRecord, SourceReadResult, SourceReadStatus

                payload = json.loads(
                    json.dumps({
                        "status": "found",
                        "fields": {"name": "Catalog Song", "played_count": 0,
                                   "favorited": False, "disliked": False, "rating": 0},
                    })
                )
                return SourceReadResult(
                    SourceReadStatus.FOUND,
                    RawTrackRecord(persistent_id, payload["fields"]),
                )

            def build_observation(self, canonical_id: str, read_result):
                return AppleMusicSourceAdapter.build_observation(self, canonical_id, read_result)

        with CanonicalRepository(self.database_path) as repository:
            orchestrator = LibrarySyncOrchestrator(
                repository, PerTrackAdapter(), DiscoveryRunner(), clock=lambda: NOW
            )
            report = orchestrator.run_cycle()
            self.assertFalse(report.enumeration_failed)
            self.assertEqual(len(repository.load_model()["tracks"]), before)  # no T2
            track = next(
                track for track in repository.load_model()["tracks"]
                if track["external_ids"].get("apple_music_catalog_id") == CATALOG_ID
            )
            self.assertEqual(track["id"], self.canonical_id)
            self.assertEqual(
                track["external_ids"]["apple_music_persistent_id"], PERSISTENT_ID
            )
            self.assertEqual(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, self.canonical_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )


class MusicKitLibraryTransportTest(unittest.TestCase):
    def test_missing_user_token_raises_actionable_error(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            transport = MusicKitLibraryTransport(developer_token="dev")
            with self.assertRaises(CatalogLibraryCredentialsError) as context:
                transport.add_song(CATALOG_ID)
        self.assertIn("MUSIC_AGENT_APPLE_MUSIC_USER_TOKEN", str(context.exception))

    def test_add_posts_form_encoded_catalog_id_with_user_token(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def read(self) -> bytes:
                return add_payload().encode("utf-8")

        with mock.patch("music_agent.catalog_library.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse()
            transport = MusicKitLibraryTransport(developer_token="dev", user_token="user")
            text = transport.add_song(CATALOG_ID)
            self.assertIn(CATALOG_ID, text)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.data, b"ids%5Bsongs%5D=CATALOG-SONG-1")
            self.assertEqual(request.headers["Music-user-token"], "user")


class CatalogUsageToolTest(unittest.TestCase):
    def track_payload(self, canonical_id: str, external_ids: dict) -> dict:
        fixture = load_fixture()
        return {
            "id": canonical_id,
            "external_ids": external_ids,
            "name": "Catalog Song",
            "artist_ids": [fixture["artists"][0]["id"]],
            "album_id": fixture["albums"][0]["id"],
            "duration_ms": 201000,
            "genres": ["Synthetic"],
            "track_number": None,
            "disc_number": None,
            "release_date": "2024-01-15",
            "composer": None,
            "library_state": {
                "favorited": None, "disliked": None, "rating": None,
                "play_count": None, "skip_count": None,
                "added_to_library_at": None, "last_played_at": None,
            },
            "agent_metadata": {"tags": []},
        }

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = load_fixture()
        self.canonical_id = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
        fixture["tracks"].append(
            self.track_payload(
                self.canonical_id,
                {
                    "apple_music_persistent_id": None,
                    "apple_music_catalog_id": CATALOG_ID,
                    "itunes_store_id": ITUNES_ID,
                },
            )
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.preview_runner = FakePreviewRunner()
        self.transport = FakeLibraryTransport(add_payload(), search_payload([library_entry()]))

    def service(
        self,
        policy: AgentClientPolicy,
        *,
        verified: bool = False,
        lookup_url: str | None = PREVIEW_URL,
    ) -> SharedAgentService:
        matrix = None
        if verified:
            capability = WRITE_CAPABILITY_MATRIX[WriteOperation.ADD_LIBRARY_SONG]
            matrix = {
                WriteOperation.ADD_LIBRARY_SONG: replace(
                    capability, capability_verified=True, readback_verified=True
                )
            }
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: policy}),
            catalog_library_transport=self.transport,
            preview_runner=self.preview_runner,
            catalog_search_source=FakeItunesLookupSource(lookup_url),
            capability_matrix=matrix,
        )
        self.addCleanup(service.close)
        return service

    def client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def test_preview_plays_resolved_itunes_preview(self) -> None:
        result = self.client(self.service(AgentClientPolicy.FULL)).call(
            "preview_catalog_track", {"canonical_id": self.canonical_id}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(result.payload["preview_url"], PREVIEW_URL)
        self.assertEqual(self.preview_runner.audios, [PREVIEW_URL])

    def test_preview_mutates_no_canonical_state(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            before = repository.load_model()
        self.client(self.service(AgentClientPolicy.FULL)).call(
            "preview_catalog_track", {"canonical_id": self.canonical_id}
        )
        with CanonicalRepository(self.database_path) as repository:
            after = repository.load_model()
        self.assertEqual(after, before)

    def test_preview_fails_closed_without_itunes_binding(self) -> None:
        fixture = load_fixture()
        no_itunes = "trk_d2d2d2d2-d2d2-4d2d-8d2d-d2d2d2d2d2d2"
        fixture["tracks"].append(
            self.track_payload(
                no_itunes,
                {
                    "apple_music_persistent_id": None,
                    "apple_music_catalog_id": "CATALOG-SONG-2",
                },
            )
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        result = self.client(self.service(AgentClientPolicy.FULL)).call(
            "preview_catalog_track", {"canonical_id": no_itunes}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_preview_unavailable")
        self.assertEqual(self.preview_runner.audios, [])

    def test_preview_fails_closed_when_lookup_resolves_no_url(self) -> None:
        result = self.client(self.service(AgentClientPolicy.FULL, lookup_url=None)).call(
            "preview_catalog_track", {"canonical_id": self.canonical_id}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_preview_unavailable")
        self.assertEqual(self.preview_runner.audios, [])

    def test_preview_fails_closed_when_source_lacks_lookup(self) -> None:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            preview_runner=self.preview_runner,
        )
        self.addCleanup(service.close)
        result = AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        ).call("preview_catalog_track", {"canonical_id": self.canonical_id})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_preview_unavailable")
        self.assertEqual(self.preview_runner.audios, [])

    def test_add_tool_denied_until_capability_verified(self) -> None:
        result = self.client(self.service(AgentClientPolicy.FULL)).call(
            "add_catalog_to_library", {"canonical_id": self.canonical_id}
        )
        self.assertEqual(result.outcome.value, "not_execution_ready")
        self.assertEqual(self.transport.add_calls, [])

    def test_add_tool_runs_when_capability_verified(self) -> None:
        result = self.client(self.service(AgentClientPolicy.FULL, verified=True)).call(
            "add_catalog_to_library", {"canonical_id": self.canonical_id}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(result.payload["add_status"], "added")
        self.assertEqual(result.payload["readback_status"], "matched")
        self.assertEqual(result.payload["bound_persistent_id"], PERSISTENT_ID)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, PERSISTENT_ID)
                ),
                self.canonical_id,
            )

    def test_read_only_client_cannot_add_or_preview(self) -> None:
        service = self.service(AgentClientPolicy.READ_ONLY, verified=True)
        client = self.client(service)
        add = client.call("add_catalog_to_library", {"canonical_id": self.canonical_id})
        self.assertEqual(add.outcome.value, "permission_denied")
        preview = client.call("preview_catalog_track", {"canonical_id": self.canonical_id})
        self.assertEqual(preview.outcome.value, "permission_denied")


if __name__ == "__main__":
    unittest.main()
