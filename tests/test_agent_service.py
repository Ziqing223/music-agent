"""P09.5: SharedAgentService -- the shared agent layer composition root.

These tests prove the full fail-closed execution flow over one shared SQLite store: unknown
clients/tools/contract versions refuse with stable codes, malformed payloads refuse before any
execution, read-only client policy gates mutations while reads pass, replayed request ids return
the journaled outcome (and mismatched replays refuse), every executed/refused request is
journaled, each registered tool executes through the real P06/P07/P08 repositories (canonical
entity lookup, preference query with the service's production calibration, recommendation
generation persisted to history, feedback recording/interpretation, learning application with
durable P06 evidence, capability projection), live writes always fail closed at the capability
gate without reaching the write orchestrator, and state survives a full close/reopen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from music_agent.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentClientIdentity,
    AgentRequest,
    AgentToolOutcome,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import (
    EmptyRecommendationError,
    SharedAgentService,
    SharedAgentServiceValidationError,
)
from music_agent.agent_tools import AgentToolName
from music_agent.recommendation_execution_service import RecommendationExecutionService
from music_agent.catalog_track_state_repository import CatalogTrackStateRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    SignalIdentity,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_signal import (
    PreferenceSignal,
    SignalDirection,
    SignalExplicitness,
    SignalReason,
)
from music_agent.recommendation_contract import decode_recommendation_result
from music_agent.source_observation import ObservedValue
from music_agent.validation import validate_fixture
from music_agent.write_intent import (
    PendingIntent,
    RequirementRole,
    WriteOperation,
    WriteRequirement,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"

CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"
CLIENT_READ_ONLY = "agt_22222222-2222-4222-8222-222222222222"
CLIENT_UNREGISTERED = "agt_99999999-9999-4999-8999-999999999999"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_C = "trk_33333333-3333-4333-8333-333333333333"
INTENT_ID = "int_55555555-5555-4555-8555-555555555555"
ISO = "2026-08-16T00:00:00+00:00"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


class _NeverCalledWriteAdapter:
    def command(self, intent: object) -> None:  # pragma: no cover
        raise AssertionError("write adapter must never be reached while nothing is ready")

    def readback(self, intent: object) -> ObservedValue:  # pragma: no cover
        raise AssertionError("write adapter must never be reached while nothing is ready")


def registry() -> AgentClientRegistry:
    return AgentClientRegistry(
        {
            CLIENT_FULL: AgentClientPolicy.FULL,
            CLIENT_READ_ONLY: AgentClientPolicy.READ_ONLY,
        }
    )


def make_request(
    client_id: str = CLIENT_FULL,
    tool: str = "get_agent_capabilities",
    payload: dict | None = None,
    request_id: str | None = None,
    contract_version: int = AGENT_CONTRACT_VERSION,
) -> AgentRequest:
    from music_agent.agent_contract import generate_request_id

    return AgentRequest(
        request_id or generate_request_id(),
        AgentClientIdentity(client_id, "codex"),
        tool,
        {} if payload is None else payload,
        NOW,
        contract_version=contract_version,
    )


def record_feedback_payload(
    *,
    kind: str = "liked",
    track_id: str = TRACK_A,
    feedback_id: str | None = None,
    source_event_id: str | None = None,
) -> dict:
    # P16-S1: observed_at is service-authoritative; it is not part of the model
    # payload (a payload carrying it is rejected as an unknown key).
    payload = {
        "kind": kind,
        "source_system": "recommendation_ui",
        "source_path": "card_actions",
        "target_id": track_id,
    }
    if feedback_id is not None:
        payload["feedback_id"] = feedback_id
    if source_event_id is not None:
        payload["source_event_id"] = source_event_id
    return payload


def search_track(
    track_id: str,
    name: str,
    artist_id: str | None,
    *,
    persistent_id: str | None = None,
    itunes_store_id: str | None = None,
) -> dict:
    """Minimal canonical track for search_library_tracks ordering tests (P14-R3.2)."""
    return {
        "id": track_id,
        "external_ids": {
            "apple_music_persistent_id": persistent_id,
            "itunes_store_id": itunes_store_id,
        },
        "name": name,
        "artist_ids": [artist_id] if artist_id is not None else [],
        "album_id": None,
        "duration_ms": None,
        "genres": [],
        "track_number": None,
        "disc_number": None,
        "release_date": None,
        "composer": None,
        "library_state": {
            "favorited": None,
            "disliked": None,
            "rating": None,
            "play_count": None,
            "skip_count": None,
            "added_to_library_at": None,
            "last_played_at": None,
        },
        "agent_metadata": {"tags": []},
    }


class SharedAgentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        self.service = SharedAgentService(
            self.database_path, clients=registry(), write_adapter=_NeverCalledWriteAdapter()
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary_directory.cleanup()

    def seed_fixture(self) -> None:
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def seed_positive(self, track_id: str) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(track_id), "apple_music", "favorited"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance="fixture_seed",
            )

    def seed_negative(self, track_id: str) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(track_id), "apple_music", "disliked"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance="fixture_seed",
            )

    def seed_annotated_fixture(self) -> dict[str, dict]:
        """Deterministic playback bindings: A=library (persistent id), B=nothing,
        C=iTunes preview only. Returns the modified track map for name assertions."""
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        by_id = {track["id"]: track for track in fixture["tracks"]}
        for track_id, values in (
            (TRACK_A, {"apple_music_persistent_id": "PERSIST-A", "itunes_store_id": None}),
            (TRACK_B, {"apple_music_persistent_id": None, "itunes_store_id": None}),
            (TRACK_C, {"apple_music_persistent_id": None, "itunes_store_id": "STORE-C"}),
        ):
            track = by_id.get(track_id)
            if track is not None:
                track["external_ids"].update(values)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        return by_id

    def seed_search_fixture(
        self, extra_tracks: list[dict], extra_artists: list[dict] | None = None
    ) -> None:
        """Base canonical fixture plus extra artists/tracks (P14-R3.2 ordering tests)."""
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        if extra_artists:
            fixture["artists"].extend(extra_artists)
        fixture["tracks"].extend(extra_tracks)
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    # --- deterministic catalog supply staging (P15-S3-S3C / S3-S3D) ----------

    EXTRA_1 = "trk_ffffffff-ffff-4fff-8fff-000000000001"
    EXTRA_2 = "trk_ffffffff-ffff-4fff-8fff-000000000002"
    EXTRA_3 = "trk_ffffffff-ffff-4fff-8fff-000000000003"

    def seed_catalog_supply(self, extra_count: int = 3) -> None:
        """Base fixture + positives on A/B + ``extra_count`` catalog-bound tracks
        sharing A's Synthetic Pop genre and Alpha artist (the P11.2 catalog
        candidate layer picks them up at 0.675 vs the 0.9 familiar magnitudes)."""
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        for index in range(1, extra_count + 1):
            fixture["tracks"].append(
                {
                    "id": f"trk_ffffffff-ffff-4fff-8fff-00000000000{index}",
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "itunes_store_id": f"EXTR-{index}",
                    },
                    "name": f"Catalog Pop {index}",
                    "artist_ids": ["art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"],
                    "album_id": None,
                    "duration_ms": None,
                    "genres": ["Synthetic Pop"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": None,
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)

    # --- search_library_tracks (P14-R3.1) ------------------------------------

    def _search(self, term: str, limit: int | None = None) -> dict:
        payload: dict = {"term": term}
        if limit is not None:
            payload["limit"] = limit
        result = self.service.execute(
            make_request(tool="search_library_tracks", payload=payload),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        return result.payload

    def test_search_library_tracks_exact_name_and_playback_annotation(self) -> None:
        self.seed_annotated_fixture()
        hit = self._search("Synthetic Duet")
        # P14-R3.2: token-level matching also brings the "synthetic" token hit of
        # Synthetic Solo; the exact-title entry still ranks first.
        self.assertEqual(hit["matched_count"], 2)
        self.assertEqual(
            hit["matches"][0],
            {
                "target_id": TRACK_A,
                "name": "Synthetic Duet",
                "artist_name": "Artist Alpha, Artist Beta",
                "playback": {"route": "library", "label": "可正式播放"},
                "provenance": {
                    "kind": "apple_music_library",
                    "label": "Apple Music Library",
                    "source_systems": ["apple_music"],
                },
                "bindings": {
                    "apple_music_persistent_id": "PERSIST-A",
                    "apple_music_catalog_id": None,
                    "itunes_store_id": None,
                },
            },
        )
        # iTunes-only binding projects preview_only through the same view.
        preview = self._search("Albumless Study")
        self.assertEqual(preview["matches"][0]["playback"]["route"], "preview_only")
        self.assertEqual(preview["matches"][0]["provenance"]["kind"], "catalog")
        self.assertEqual(preview["matches"][0]["bindings"]["itunes_store_id"], "STORE-C")

    def test_search_library_tracks_unbound_track_annotates_unavailable(self) -> None:
        self.seed_fixture()  # base fixture: trk_3333 has no binding of any kind
        hit = self._search("Albumless Study")
        self.assertEqual(hit["matched_count"], 1)
        self.assertEqual(hit["matches"][0]["playback"], {"route": "unavailable", "label": "不可用"})

    def test_search_library_tracks_normalization_covers_case_whitespace_fullwidth(self) -> None:
        self.seed_fixture()
        for term in ("synthetic duet", "SYNTHETIC DUET", "Synthetic  Duet", "Ｓｙｎｔｈｅｔｉｃ　ｄｕｅｔ"):
            with self.subTest(term=term):
                hit = self._search(term)
                # P14-R3.2: the "synthetic" token also matches Synthetic Solo; the
                # Level-1 exact hit stays first regardless of case/space/fullwidth form.
                self.assertEqual(hit["matched_count"], 2, term)
                self.assertEqual(hit["matches"][0]["target_id"], TRACK_A)
        # NFKC folds the ideographic fullwidth variant to the same Level-1 exact hit.
        self.assertEqual(self._search("Synthetic Duet")["matches"], self._search("Ｓｙｎｔｈｅｔｉｃ　ｄｕｅｔ")["matches"])

    def test_search_library_tracks_artist_substring_hit(self) -> None:
        self.seed_fixture()
        hit = self._search("Beta")
        self.assertEqual(hit["matched_count"], 2)
        self.assertEqual(
            [match["name"] for match in hit["matches"]],
            ["Synthetic Duet", "Year Precision"],
        )
        self.assertEqual(hit["matches"][0]["artist_name"], "Artist Alpha, Artist Beta")
        single = self._search("gamma")
        self.assertEqual([match["name"] for match in single["matches"]], ["Albumless Study"])

    def test_search_library_tracks_no_match_returns_empty(self) -> None:
        self.seed_fixture()
        hit = self._search("从未存在的歌")
        self.assertEqual(hit["matched_count"], 0)
        self.assertEqual(hit["matches"], [])

    def test_search_library_tracks_exact_ranks_before_partial_and_limit_caps(self) -> None:
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        extra = dict(fixture["tracks"][0])
        # P14-R3.2: keep the extra track playable (library binding) so the
        # exact-before-partial assertion stays within one capability tier; the
        # capability-precedes-exactness cross-tier case is covered separately below.
        extra.update(
            {
                "id": "trk_99999999-9999-4999-8999-999999999999",
                "name": "Synthetic",
                "artist_ids": ["art_cccccccc-cccc-4ccc-8ccc-cccccccccccc"],
                "external_ids": {"apple_music_persistent_id": "SYNTH-TRACK-EXTRA"},
            }
        )
        fixture["tracks"].append(extra)
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        all_hits = self._search("Synthetic")
        self.assertEqual(all_hits["matched_count"], 3)
        self.assertEqual(
            [match["name"] for match in all_hits["matches"]],
            ["Synthetic", "Synthetic Duet", "Synthetic Solo"],
        )
        capped = self._search("Synthetic", limit=1)
        self.assertEqual(capped["matched_count"], 3)
        self.assertEqual(len(capped["matches"]), 1)
        self.assertEqual(capped["matches"][0]["name"], "Synthetic")

    def test_search_library_tracks_title_relevance_precedes_partial_library_hit(self) -> None:
        """P20 Slice 1: exact catalog title beats unrelated partial Library titles."""
        self.seed_search_fixture(
            extra_tracks=[
                search_track(
                    "trk_99999999-9999-4999-8999-999999999999",
                    "Synthetic",
                    "art_cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    itunes_store_id="STORE-EXTRA",
                ),
            ],
        )
        hit = self._search("Synthetic")
        self.assertEqual(
            [match["name"] for match in hit["matches"]],
            ["Synthetic", "Synthetic Duet", "Synthetic Solo"],
        )

    def test_search_library_tracks_keeps_sparse_exact_library_siblings(self) -> None:
        """Title + artist queries cannot truncate sparse persistent-bound siblings."""
        yorushika = "art_99999999-9999-4999-8999-999999999999"
        venus_thief = "art_88888888-9999-4999-8999-999999999999"
        library_ids = [
            "trk_10000000-0000-4000-8000-000000000001",
            "trk_10000000-0000-4000-8000-000000000002",
        ]
        library_tracks = [
            search_track(library_ids[0], "Spring Thief", None, persistent_id="PID-ONE"),
            search_track(library_ids[1], "Spring Thief", None, persistent_id="PID-TWO"),
        ]
        catalog_tracks = [
            search_track(
                f"trk_20000000-0000-4000-8000-{index:012d}",
                "Spring Thief",
                yorushika,
                itunes_store_id=f"STORE-{index}",
            )
            for index in range(1, 13)
        ]
        self.seed_search_fixture(
            extra_artists=[
                {
                    "id": yorushika,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Yorushika",
                },
                {
                    "id": venus_thief,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Venus Thief",
                },
            ],
            extra_tracks=[
                *library_tracks,
                *catalog_tracks,
                search_track(
                    "trk_00000000-0000-4000-8000-000000000001",
                    "Spring",
                    venus_thief,
                    itunes_store_id="STORE-PARTIAL",
                ),
            ],
        )

        with_artist = self._search("Spring Thief Yorushika")
        self.assertEqual(
            [match["target_id"] for match in with_artist["matches"][:2]], library_ids
        )
        self.assertEqual(
            [match["playback"]["route"] for match in with_artist["matches"][:2]],
            ["library", "library"],
        )
        self.assertEqual(
            [match["bindings"]["apple_music_persistent_id"] for match in with_artist["matches"][:2]],
            ["PID-ONE", "PID-TWO"],
        )
        self.assertTrue(
            all(match["provenance"]["kind"] == "apple_music_library"
                for match in with_artist["matches"][:2])
        )
        self.assertNotIn("Spring", [match["name"] for match in with_artist["matches"]])

        title_only = self._search("Spring Thief")
        self.assertEqual(
            [match["target_id"] for match in title_only["matches"][:2]], library_ids
        )

    def test_search_library_tracks_artist_only_finds_bound_yorushika_tracks(self) -> None:
        yorushika = "art_99999999-9999-4999-8999-999999999999"
        library_ids = (
            "trk_30000000-0000-4000-8000-000000000001",
            "trk_30000000-0000-4000-8000-000000000002",
        )
        self.seed_search_fixture(
            extra_artists=[
                {
                    "id": yorushika,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Yorushika",
                }
            ],
            extra_tracks=[
                search_track(library_ids[0], "Spring Thief", yorushika,
                             persistent_id="PID-ONE"),
                search_track(library_ids[1], "Spring Thief", yorushika,
                             persistent_id="PID-TWO"),
                search_track(
                    "trk_30000000-0000-4000-8000-000000000003",
                    "Catalog Song", yorushika, itunes_store_id="STORE-ONLY",
                ),
            ],
        )
        result = self._search("Yorushika")
        bound = [
            match for match in result["matches"]
            if match["provenance"]["kind"] == "apple_music_library"
        ]
        self.assertEqual([match["target_id"] for match in bound], list(library_ids))
        self.assertTrue(all(match["artist_name"] == "Yorushika" for match in bound))
        self.assertTrue(all(match["playback"]["route"] == "library" for match in bound))

    def test_search_library_tracks_artist_token_tier_precedes_capability(self) -> None:
        """P14-R3.2: an artist-name token hit (周深) outranks higher playback capability."""
        self.seed_search_fixture(
            extra_artists=[
                {"id": "art_88888888-8888-4888-8888-888888888888", "external_ids": {"apple_music_persistent_id": None}, "name": "周深"},
                {"id": "art_77777777-7777-4777-8777-777777777777", "external_ids": {"apple_music_persistent_id": None}, "name": "柳如烟"},
            ],
            extra_tracks=[
                search_track(
                    "trk_aaaa0001-aaaa-4aaa-8aaa-000000000001",
                    "起风了",
                    "art_88888888-8888-4888-8888-888888888888",
                    itunes_store_id="STORE-ZHOU",
                ),
                search_track(
                    "trk_aaaa0002-aaaa-4aaa-8aaa-000000000002",
                    "起风了",
                    "art_77777777-7777-4777-8777-777777777777",
                    persistent_id="PERSIST-YAN",
                ),
            ],
        )
        hit = self._search("周深 起风了")
        self.assertEqual(hit["matched_count"], 2)
        self.assertEqual(
            [match["target_id"] for match in hit["matches"]],
            [
                "trk_aaaa0001-aaaa-4aaa-8aaa-000000000001",
                "trk_aaaa0002-aaaa-4aaa-8aaa-000000000002",
            ],
        )

    def test_search_library_tracks_multi_token_term_matches_name_variant(self) -> None:
        """P14-R3.2: any term token can hit the name, not only the whole-term substring."""
        self.seed_search_fixture(
            extra_artists=[
                {"id": "art_77777777-7777-4777-8777-777777777777", "external_ids": {"apple_music_persistent_id": None}, "name": "柳如烟"},
            ],
            extra_tracks=[
                search_track(
                    "trk_aaaa0003-aaaa-4aaa-8aaa-000000000003",
                    "起风了 (旧版)",
                    "art_77777777-7777-4777-8777-777777777777",
                    persistent_id="PERSIST-YAN",
                ),
            ],
        )
        hit = self._search("起风了 旧版")
        self.assertEqual(hit["matched_count"], 1)
        self.assertEqual(hit["matches"][0]["target_id"], "trk_aaaa0003-aaaa-4aaa-8aaa-000000000003")

    def test_search_library_tracks_tie_breaks_by_target_id_not_store_order(self) -> None:
        """P14-R3.2: equal (artist tier, capability, level) tracks order by target_id."""
        self.seed_search_fixture(
            extra_tracks=[
                search_track(
                    "trk_aaaa0005-aaaa-4aaa-8aaa-000000000005",
                    "Echo Track A",
                    "art_cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    persistent_id="PERSIST-EA",
                ),
                search_track(
                    "trk_aaaa0004-aaaa-4aaa-8aaa-000000000004",
                    "Echo Track B",
                    "art_cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    persistent_id="PERSIST-EB",
                ),
            ],
        )
        hit = self._search("Echo Track")
        self.assertEqual(
            [match["name"] for match in hit["matches"]],
            ["Echo Track B", "Echo Track A"],
        )

    def test_search_library_tracks_is_readable_by_read_only_client(self) -> None:
        self.seed_fixture()
        result = self.service.execute(
            make_request(
                client_id=CLIENT_READ_ONLY,
                tool="search_library_tracks",
                payload={"term": "Synthetic Duet"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        # P14-R3.2: token-level match adds Synthetic Solo; exact entry stays first.
        self.assertEqual(result.payload["matched_count"], 2)
        self.assertEqual(result.payload["matches"][0]["target_id"], TRACK_A)

    def test_search_library_tracks_rejects_malformed_payload(self) -> None:
        for payload in ({}, {"term": ""}, {"term": "x", "limit": 0}):
            with self.subTest(payload=payload):
                result = self.service.execute(
                    make_request(tool="search_library_tracks", payload=payload),
                    completed_at=ISO,
                )
                self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)

    # --- flow refusals -------------------------------------------------------

    def test_constructor_requires_a_client_registry(self) -> None:
        with self.assertRaises(SharedAgentServiceValidationError):
            SharedAgentService(self.database_path, clients="not-a-registry")  # type: ignore[arg-type]

    def test_unregistered_client_fails_closed(self) -> None:
        result = self.service.execute(
            make_request(client_id=CLIENT_UNREGISTERED), completed_at=ISO
        )
        self.assertEqual(result.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        self.assertEqual(result.error_code, "unknown_client")

    def test_unknown_tool_fails_closed(self) -> None:
        result = self.service.execute(make_request(tool="invented_tool"), completed_at=ISO)
        self.assertEqual(result.outcome, AgentToolOutcome.TOOL_NOT_SUPPORTED)
        self.assertEqual(result.error_code, "tool_not_supported")

    def test_unsupported_contract_version_fails_closed(self) -> None:
        result = self.service.execute(
            make_request(contract_version=99), completed_at=ISO
        )
        self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(result.error_code, "unsupported_contract_version")

    def test_malformed_payload_fails_closed_before_execution(self) -> None:
        result = self.service.execute(
            make_request(tool="query_track_preference", payload={"target_id": "rcm_1"}),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)

    def test_read_only_client_can_read_but_not_mutate(self) -> None:
        read = self.service.execute(
            make_request(client_id=CLIENT_READ_ONLY), completed_at=ISO
        )
        self.assertEqual(read.outcome, AgentToolOutcome.OK)
        mutate = self.service.execute(
            make_request(
                client_id=CLIENT_READ_ONLY,
                tool="record_feedback",
                payload=record_feedback_payload(),
            ),
            completed_at=ISO,
        )
        self.assertEqual(mutate.outcome, AgentToolOutcome.PERMISSION_DENIED)
        self.assertEqual(mutate.error_code, "permission_denied")

    def test_every_request_is_journaled_with_its_outcome(self) -> None:
        self.service.execute(make_request(tool="invented_tool"), completed_at=ISO)
        self.service.execute(make_request(tool="get_agent_capabilities"), completed_at=ISO)
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            records = journal.list()
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record.result.outcome for record in records},
            {AgentToolOutcome.TOOL_NOT_SUPPORTED, AgentToolOutcome.OK},
        )

    # --- replay safety --------------------------------------------------------

    def test_replayed_request_returns_the_journaled_result(self) -> None:
        request = make_request(tool="get_agent_capabilities")
        first = self.service.execute(request, completed_at=ISO)
        self.assertFalse(first.replayed)
        replayed = self.service.execute(request, completed_at=ISO)
        self.assertTrue(replayed.replayed)
        self.assertEqual(replayed.outcome, AgentToolOutcome.OK)
        self.assertEqual(replayed.payload, first.payload)
        self.assertEqual(replayed.completed_at, first.completed_at)

    def test_replay_with_a_different_payload_fails_closed(self) -> None:
        request = make_request(tool="get_agent_capabilities")
        self.service.execute(request, completed_at=ISO)
        from music_agent.agent_contract import AgentRequest

        changed = AgentRequest(
            request.request_id,
            request.client,
            request.tool,
            {"canonical_id": TRACK_A},
            NOW,
        )
        result = self.service.execute(changed, completed_at=ISO)
        self.assertEqual(result.outcome, AgentToolOutcome.REPLAY_CONFLICT)
        self.assertEqual(result.error_code, "replay_conflict")

    # --- read tools ----------------------------------------------------------

    def test_get_canonical_entity_round_trip_and_missing(self) -> None:
        self.seed_fixture()
        result = self.service.execute(
            make_request(tool="get_canonical_entity", payload={"canonical_id": TRACK_A}),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["entity_type"], "track")
        self.assertEqual(result.payload["entity"]["id"], TRACK_A)

        missing = self.service.execute(
            make_request(
                tool="get_canonical_entity",
                payload={"canonical_id": "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(missing.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(missing.error_code, "canonical_entity_not_found")

    def test_query_track_preference_uses_service_calibration(self) -> None:
        self.seed_positive(TRACK_A)
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["preference_state"], "positive")
        self.assertEqual(result.payload["magnitude"], 0.9)

    def test_query_track_preference_carries_confidence(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(TRACK_A), "apple_music", "favorited"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )
            repository.record_observation(
                SignalIdentity(target(TRACK_A), "apple_music", "rating"),
                ObservedValue.value(85),
                observed_at=ISO,
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        confidence = result.payload["confidence"]
        self.assertIsNotNone(confidence)
        self.assertEqual(confidence["quantity"], 2)
        self.assertEqual(confidence["quality"], 1.0)
        self.assertEqual(confidence["contradiction"], "none")
        self.assertEqual(confidence["source_reliability"], 1.0)
        self.assertEqual(confidence["inference_distance"], 0.0)
        self.assertGreaterEqual(confidence["score"], 0.9)

    def test_query_track_preference_confidence_is_none_without_evidence(self) -> None:
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["preference_state"], "unknown")
        self.assertIsNone(result.payload["confidence"])

    def test_query_track_preference_carries_explanation(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(TRACK_A), "apple_music", "favorited"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        explanation = result.payload["explanation"]
        self.assertEqual(explanation["derivation"], "direct")
        signals = explanation["signals"]
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal"], PreferenceSignal.FAVORITED.value)
        self.assertEqual(signals[0]["direction"], SignalDirection.POSITIVE.value)
        self.assertEqual(signals[0]["reason"], SignalReason.EXPLICIT_SIGNAL.value)
        self.assertEqual(signals[0]["explicitness"], SignalExplicitness.EXPLICIT.value)
        self.assertEqual(explanation["conflicts"], [])

    def test_query_track_preference_explanation_surfaces_conflict(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(target(TRACK_A), "apple_music", "favorited"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )
            repository.record_observation(
                SignalIdentity(target(TRACK_A), "apple_music", "disliked"),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance=DIRECT_OBSERVATION_PROVENANCE,
            )
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        explanation = result.payload["explanation"]
        self.assertEqual(explanation["derivation"], "direct")
        self.assertEqual(len(explanation["signals"]), 2)
        conflicts = explanation["conflicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["kind"], "direct_categorical")
        self.assertEqual(
            {conflicts[0]["first_direction"], conflicts[0]["second_direction"]},
            {SignalDirection.POSITIVE.value, SignalDirection.NEGATIVE.value},
        )

    def test_query_track_preference_explanation_is_empty_without_evidence(self) -> None:
        result = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "apple_music"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        explanation = result.payload["explanation"]
        self.assertEqual(explanation["derivation"], "direct")
        self.assertEqual(explanation["signals"], [])
        self.assertEqual(explanation["conflicts"], [])

    def test_generate_recommendation_persists_and_reads_back(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        generated = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                    "source_system": "apple_music",
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        run_id = generated.payload["run_id"]
        self.assertEqual(generated.payload["item_count"], 2)
        decode_recommendation_result(generated.payload["encoded_result"])

        listed = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        )
        self.assertEqual([run["run_id"] for run in listed.payload["runs"]], [run_id])
        fetched = self.service.execute(
            make_request(tool="get_recommendation_run", payload={"run_id": run_id}),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        self.assertEqual(fetched.payload["run_id"], run_id)

    def test_generate_durable_produced_at_is_the_service_execution_instant(self) -> None:
        """P15 burn-down Issue 1: produced_at authority. The trusted completed_at
        execution context (what production derives from the runtime's own clock)
        is the EXACT durable run time and result envelope time -- the model has
        no payload path to influence it."""
        self.seed_positive(TRACK_A)
        execution_at = "2026-08-16T07:15:30+00:00"
        generated = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={"target_ids": [TRACK_A], "limit": 5},
            ),
            completed_at=execution_at,
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(generated.payload["encoded_result"])
        self.assertEqual(decoded.produced_at.isoformat(), execution_at)
        listed = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(listed[0]["produced_at"], execution_at)

    def test_generate_without_completed_at_stamps_real_service_clock(self) -> None:
        """No trusted override on the execute side -> the service stamps its own
        execution instant, which lands within the real clock window bracketing
        the call."""
        self.seed_positive(TRACK_A)
        before = datetime.now(timezone.utc)
        generated = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={"target_ids": [TRACK_A], "limit": 5},
            )
        )
        after = datetime.now(timezone.utc)
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        produced_at = decode_recommendation_result(
            generated.payload["encoded_result"]
        ).produced_at
        self.assertGreaterEqual(produced_at, before)
        self.assertLessEqual(produced_at, after)

    def test_generate_payload_produced_at_fails_closed_with_no_history(self) -> None:
        """A model that sends ``produced_at`` in the payload (a hallucinated
        envelope key) is rejected before execution: validation_error, no run
        persisted, nothing reaches a generation handler."""
        self.seed_positive(TRACK_A)
        refused = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 5,
                    "produced_at": "2026-08-16T01:10:00+08:00",
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(refused.error_code, "validation_error")
        self.assertEqual(
            self.service.execute(
                make_request(tool="list_recommendation_runs"), completed_at=ISO
            ).payload["runs"],
            [],
        )

    def test_generate_recommendation_records_active_batch_pointer(self) -> None:
        """P14-C07.3: a durably saved run becomes the active batch pointer at the
        generate boundary -- get_active_context then reports it as register-sourced,
        with identity facts only."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        generated = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                    "source_system": "apple_music",
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        run_id = generated.payload["run_id"]
        self.assertEqual(self.service._active_context.active_run_id, run_id)
        self.assertIsNone(self.service._active_context.active_item_index)

        observed = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(observed.outcome, AgentToolOutcome.OK)
        batch = observed.payload["active_batch"]
        self.assertIsNotNone(batch)
        self.assertEqual(batch["run_id"], run_id)
        self.assertEqual(batch["source"], "register")
        self.assertEqual(batch["item_count"], 2)
        self.assertEqual(set(batch), {"run_id", "source", "produced_at", "item_count"})

    def test_second_recommendation_replaces_active_batch_pointer(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)

        def generate(produced_at: str) -> dict:
            result = self.service.execute(
                make_request(
                    tool="generate_recommendation",
                    payload={
                        "target_ids": [TRACK_A, TRACK_B],
                        "limit": 5,
                        # P19-T15: explicit opt-out -- this test is about the
                        # pointer replacement semantics, not recent-run dedup.
                        "avoid_previous_runs": False,
                    },
                ),
                completed_at=produced_at,
            )
            self.assertEqual(result.outcome, AgentToolOutcome.OK)
            return result.payload

        generate("2026-08-16T09:00:00+00:00")
        second = generate("2026-08-16T09:05:00+00:00")
        # The newest generate owns the pointer, never the first one.
        self.assertEqual(
            self.service._active_context.active_run_id, second["run_id"]
        )
        self.assertIsNone(self.service._active_context.active_item_index)
        observed = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(observed.payload["active_batch"]["run_id"], second["run_id"])
        self.assertEqual(observed.payload["active_batch"]["source"], "register")

    def test_inferred_recommendation_records_active_batch_pointer(self) -> None:
        self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        inferred = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(inferred.outcome, AgentToolOutcome.OK)
        run_id = inferred.payload["run_id"]
        self.assertEqual(self.service._active_context.active_run_id, run_id)
        self.assertIsNone(self.service._active_context.active_item_index)
        observed = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(observed.payload["active_batch"]["run_id"], run_id)
        self.assertEqual(observed.payload["active_batch"]["source"], "register")

    def test_list_recommendation_runs_limits_to_recent_with_total(self) -> None:
        self.seed_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        for hour in range(6):
            generated = self.service.execute(
                make_request(
                    tool="generate_recommendation",
                    payload={
                        "target_ids": [TRACK_A, TRACK_B],
                        "limit": 5,
                        "source_system": "apple_music",
                        # P19-T15: this test fabricates listing history;
                        # repeat-side novelty is not what it exercises.
                        "avoid_previous_runs": False,
                    },
                ),
                completed_at=f"2026-08-16T{10 + hour:02d}:00:00+00:00",
            )
            self.assertEqual(generated.outcome, AgentToolOutcome.OK)

        full = self.service.execute(
            make_request(
                tool="list_recommendation_runs", payload={"limit": 20}
            ),
            completed_at=ISO,
        )
        self.assertEqual(full.outcome, AgentToolOutcome.OK)
        full_ids = [run["run_id"] for run in full.payload["runs"]]
        self.assertEqual(full.payload["runs_total"], 6)
        self.assertEqual(len(full_ids), 6)

        # Default limit: only the most recent 5 come back; the total is still reported
        # so the model knows more history exists without fetching it.
        default = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        )
        default_ids = [run["run_id"] for run in default.payload["runs"]]
        self.assertEqual(len(default_ids), 5)
        self.assertEqual(default.payload["runs_total"], 6)

        limited = self.service.execute(
            make_request(
                tool="list_recommendation_runs", payload={"limit": 2}
            ),
            completed_at=ISO,
        )
        limited_ids = [run["run_id"] for run in limited.payload["runs"]]
        self.assertEqual(len(limited_ids), 2)
        self.assertEqual(limited.payload["runs_total"], 6)

        # Limited views are prefixes of the newest-first full list.
        self.assertEqual(default_ids, full_ids[:5])
        self.assertEqual(limited_ids, full_ids[:2])

    def test_list_recommendation_runs_rejects_invalid_limits(self) -> None:
        for bad_limit in (0, -1, 21, True, "5"):
            with self.subTest(limit=bad_limit):
                result = self.service.execute(
                    make_request(
                        tool="list_recommendation_runs",
                        payload={"limit": bad_limit},
                    ),
                    completed_at=ISO,
                )
                self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)
                self.assertEqual(result.error_code, "validation_error")

    # --- recommendation experience controls --------------------------------

    def _generate(self, **payload: object) -> object:
        # P15 burn-down Issue 1: ``produced_at`` rides the trusted completed_at
        # execution seam, exactly as production passes the service's own
        # execution instant (never a model-controlled payload key).
        produced_at = payload.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 5, **payload},
            ),
            completed_at=produced_at or ISO,
        )

    def test_exclude_target_ids_drops_listed_targets(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(first.payload["encoded_result"])
        self.assertEqual(
            {item.candidate.target.target_id for item in decoded.items},
            {TRACK_A, TRACK_B},
        )
        refused = self._generate(
            produced_at="2026-08-16T10:01:00+00:00",
            exclude_target_ids=[TRACK_A, TRACK_B],
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")
        self.assertIsNone(refused.payload)
        # P14-R2: the refusal never enters history or replaces the batch pointer.
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], first.payload["run_id"])
        partial = self._generate(
            produced_at="2026-08-16T10:02:00+00:00",
            exclude_target_ids=[TRACK_A],
        )
        self.assertEqual(partial.payload["item_count"], 1)
        partial_decoded = decode_recommendation_result(partial.payload["encoded_result"])
        self.assertEqual(
            [item.candidate.target.target_id for item in partial_decoded.items],
            [TRACK_B],
        )

    def test_avoid_previous_runs_excludes_everything_recommended_before(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(first.payload["item_count"], 2)
        fresh = self._generate(
            produced_at="2026-08-16T10:01:00+00:00",
            avoid_previous_runs=True,
        )
        self.assertEqual(fresh.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(fresh.error_code, "empty_recommendation")
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], first.payload["run_id"])

    def _insert_history_run(
        self, run_id: str, target_ids: tuple[str, ...], created_at: str
    ) -> None:
        """Raw-insert one history row with an explicit created_at, so window
        membership in _AVOID_PREVIOUS_RUNS_WINDOW is deterministic (real
        save_result stamps the current second)."""
        from datetime import datetime

        from music_agent.preference_attribution import DerivedPreference
        from music_agent.preference_strength import (
            PreferenceState,
            PreferenceStrength,
        )
        from music_agent.recommendation_contract import (
            Candidate,
            CandidateSourceReference,
            PreferenceInput,
            RecommendationContext,
            RecommendationItem,
            RecommendationRequest,
            RecommendedItemKind,
            ScoreBreakdown,
            ScoreComponent,
            assemble_recommendation_result,
            encode_recommendation_result,
        )

        now = datetime.fromisoformat(ISO)
        history_context = RecommendationContext(
            now,
            (
                PreferenceInput.from_direct(
                    DerivedPreference(
                        target(target_id),
                        PreferenceStrength(PreferenceState.POSITIVE, 0.9),
                    )
                )
                for target_id in target_ids
            ),
        )
        history_request = RecommendationRequest(
            history_context, RecommendedItemKind.TRACK, len(target_ids)
        )
        items = tuple(
            RecommendationItem(
                Candidate(
                    candidate_id="cnd_00000000-0000-4000-8000-00000000000%d"
                    % (index + 1),
                    target=target(target_id),
                    source=CandidateSourceReference(
                        "candidate_gen", "preference_match"
                    ),
                ),
                ScoreBreakdown(0.9, (ScoreComponent("preference_match", 0.9),)),
            )
            for index, target_id in enumerate(target_ids)
        )
        entry = assemble_recommendation_result(
            history_request, items, run_id=run_id, produced_at=now
        )
        self.service._recommendation_history._connection.execute(
            """INSERT INTO recommendation_runs(
                run_id, encoded_result, contract_version, produced_at, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                entry.run_id,
                encode_recommendation_result(entry),
                entry.contract_version,
                entry.produced_at.isoformat(),
                created_at,
            ),
        )

    def test_avoid_previous_runs_window_excludes_recent_but_readmits_older(self) -> None:
        """P14-R4.2: avoid_previous_runs folds in only the newest
        _AVOID_PREVIOUS_RUNS_WINDOW runs -- a target from the newest 5 stays
        excluded while one last recommended before the window re-enters."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        self._insert_history_run(
            "rcm_00000000-0000-4000-8000-000000000001",
            (TRACK_B,),
            "2026-08-17 06:00:00",
        )
        for index in range(5):
            self._insert_history_run(
                "rcm_00000000-0000-4000-8001-%012d" % index,
                (TRACK_A,),
                "2026-08-17 06:0%d:00" % (index + 1),
            )
        fresh = self._generate(
            produced_at="2026-08-16T10:00:00+00:00",
            avoid_previous_runs=True,
        )
        self.assertEqual(fresh.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(fresh.payload["encoded_result"])
        self.assertEqual(
            [item.candidate.target.target_id for item in decoded.items],
            [TRACK_B],
        )

    def test_avoid_previous_runs_with_empty_history_is_a_noop(self) -> None:
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        fresh = self._generate(
            produced_at="2026-08-16T10:00:00+00:00",
            avoid_previous_runs=True,
        )
        self.assertEqual(fresh.outcome, AgentToolOutcome.OK)
        self.assertEqual(fresh.payload["item_count"], 2)

    # --- P19-T15: server-default recent-run exclusion ------------------------

    def test_unqualified_generate_defaults_to_recent_run_exclusion(self) -> None:
        """P19-T15 proposal 1: with neither exclusion key present the service
        folds in the newest-5 successful runs on its own, so the immediate
        repeat of an unqualified batch is refused through the existing honest
        empty path -- no default-injected payload keys, no history mutation,
        no phantom batch."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        self.assertEqual(first.payload["item_count"], 2)

        second = self._generate(produced_at="2026-08-16T10:01:00+00:00")
        self.assertEqual(second.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(second.error_code, "empty_recommendation")
        self.assertIsNone(second.payload)
        # P19-T15 proposal 3: the refusal is the existing empty/concise path
        # with truthful exclusion diagnostics -- nothing recycled to fill 5.
        diagnostics = self._empty_diagnostics(second)
        self.assertEqual(diagnostics["excluded_previous_count"], 2)
        self.assertEqual(diagnostics["reason"], "all_eligible_candidates_excluded")
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], first.payload["run_id"])
        context = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(
            context.payload["active_batch"]["run_id"], first.payload["run_id"]
        )

    def test_explicit_avoid_previous_runs_false_preserves_repeat_allowed(self) -> None:
        """P19-T15 proposal 1: the default is an opt-out, not a wall -- the
        caller's explicit avoid_previous_runs=false restores the exact pre-T15
        repeat-allowed behavior for the identical payload."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(
            produced_at="2026-08-16T10:00:00+00:00", avoid_previous_runs=False
        )
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        second = self._generate(
            produced_at="2026-08-16T10:01:00+00:00", avoid_previous_runs=False
        )
        self.assertEqual(second.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(second.payload["encoded_result"])
        self.assertEqual(
            {item.candidate.target.target_id for item in decoded.items},
            {TRACK_A, TRACK_B},
        )

    def test_default_exclusion_window_does_not_exceed_five_runs(self) -> None:
        """P19-T15: the default folds the SAME window=5 machinery the flagged
        opt-in uses -- a target last recommended six runs ago re-enters, while
        a target inside the five newest stays excluded."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        self._insert_history_run(
            "rcm_00000000-0000-4000-8000-000000000001",
            (TRACK_B,),
            "2026-08-17 06:00:00",
        )
        for index in range(5):
            self._insert_history_run(
                "rcm_00000000-0000-4000-8001-%012d" % index,
                (TRACK_A,),
                "2026-08-17 06:0%d:00" % (index + 1),
            )
        fresh = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(fresh.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(fresh.payload["encoded_result"])
        self.assertEqual(
            [item.candidate.target.target_id for item in decoded.items],
            [TRACK_B],
        )

    def test_empty_generate_is_refused_and_never_persisted(self) -> None:
        """P14-R2: a zero-item generation is a tool error -- nothing is written to
        recommendation history and the active batch pointer keeps the last valid run."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(first.outcome, AgentToolOutcome.OK)

        refused = self._generate(
            produced_at="2026-08-16T10:01:00+00:00",
            exclude_target_ids=[TRACK_A, TRACK_B],
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")
        self.assertIsNone(refused.payload)

        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], first.payload["run_id"])

        observed = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(observed.outcome, AgentToolOutcome.OK)
        self.assertEqual(
            observed.payload["active_batch"]["run_id"], first.payload["run_id"]
        )

        # The next valid generation proceeds normally and replaces the pointer.
        retry = self._generate(
            produced_at="2026-08-16T10:02:00+00:00",
            exclude_target_ids=[TRACK_A],
        )
        self.assertEqual(retry.outcome, AgentToolOutcome.OK)
        self.assertEqual(retry.payload["item_count"], 1)

    def test_empty_inferred_generate_is_refused_without_history_pollution(self) -> None:
        """P14-R2, inferred boundary: same empty-run refusal semantics as direct
        generation -- no history pollution, no phantom active batch."""
        self.seed_positive(TRACK_A)
        refused = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 5,
                    "genres": ["Classical"],
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")
        self.assertIsNone(refused.payload)
        self.assertEqual(
            self.service.execute(
                make_request(tool="list_recommendation_runs"), completed_at=ISO
            ).payload["runs"],
            [],
        )
        context = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertIsNone(context.payload["active_batch"])

        # An unconstrained inference succeeds normally and records a valid batch.
        valid = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(valid.outcome, AgentToolOutcome.OK)
        self.assertEqual(valid.payload["item_count"], 1)

    # --- P15-S4-M2-2: empty-generation diagnostic envelope ------------------

    def _empty_diagnostics(self, result: object) -> dict:
        """Parse the P15-S4-M2-2 diagnostic JSON out of an empty refusal. The
        refusal keeps its payload=None / error_code="empty_recommendation"
        contract; only the message gains the structured suffix."""
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "empty_recommendation")
        self.assertIsNone(result.payload)
        message = result.error_message
        self.assertTrue(
            message.startswith(
                "generation produced zero items; nothing was written to "
                "recommendation history.",
            ),
            message,
        )
        marker = " diagnostics: "
        self.assertIn(marker, message)
        parsed = json.loads(message.split(marker, 1)[1])
        self.assertIsInstance(parsed, dict)
        return parsed

    def test_empty_plain_fallback_reports_the_inferred_envelope(self) -> None:
        """No signal heads -> direct UNKNOWN -> skipped. P16-S2: the plain tool
        deterministically routes to the inferred channel when no target carries
        positive direct evidence; with nothing in the inferred channel either
        (no catalog supply, no affinities), the refusal surfaces with the
        INFFERED envelope -- the honest funnel of the channel that ran."""
        self.seed_fixture()
        refused = self._generate(
            target_ids=[TRACK_A, TRACK_C],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 2)
        self.assertEqual(diagnostics["after_direction_filter_count"], 2)
        self.assertEqual(diagnostics["direct_evidence_count"], 0)
        self.assertEqual(diagnostics["positive_evidence_count"], 0)
        self.assertEqual(diagnostics["negative_evidence_count"], 0)
        self.assertEqual(diagnostics["candidate_count"], 0)
        self.assertEqual(diagnostics["excluded_previous_count"], 0)
        self.assertEqual(diagnostics["reason"], "no_candidates_in_pool")
        # The executed channel was the inferred one (the deterministic fallback):
        # its keys are present and the supply facts are real (exhausted here).
        self.assertEqual(diagnostics["inferred_positive_count"], 0)
        self.assertEqual(diagnostics["catalog_candidate_count"], 0)
        self.assertEqual(diagnostics["known_catalog_track_count"], 0)
        self.assertIn("耗尽", diagnostics["recommended_next_action"])

    def test_empty_generate_diagnostics_report_direction_filter_removal(self) -> None:
        """The direction filter hard-filters every target before any preference
        work; the remedy is about genres, not about switching tools."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        refused = self._generate(
            genres=["Classical"],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 2)
        self.assertEqual(diagnostics["after_direction_filter_count"], 0)
        self.assertEqual(diagnostics["reason"], "direction_filtered_all_targets")
        self.assertIn("genres", diagnostics["recommended_next_action"])
        self.assertNotIn(
            "generate_inferred_recommendation",
            diagnostics["recommended_next_action"],
        )
        self.assertNotIn("discover", diagnostics["recommended_next_action"])

    def test_empty_generate_diagnostics_report_previous_run_exclusions(self) -> None:
        """Repeat-suppression excluded every pooled candidate; the counts are the
        real exclusion numbers and the remedy is about exclusions, not evidence."""
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        first = self._generate(produced_at="2026-08-16T10:00:00+00:00")
        self.assertEqual(first.payload["item_count"], 2)
        refused = self._generate(
            produced_at="2026-08-16T10:01:00+00:00",
            avoid_previous_runs=True,
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 2)
        self.assertEqual(diagnostics["after_direction_filter_count"], 2)
        self.assertEqual(diagnostics["direct_evidence_count"], 2)
        self.assertEqual(diagnostics["positive_evidence_count"], 2)
        self.assertEqual(diagnostics["negative_evidence_count"], 0)
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["excluded_previous_count"], 2)
        self.assertEqual(diagnostics["reason"], "all_eligible_candidates_excluded")
        self.assertNotIn(
            "generate_inferred_recommendation",
            diagnostics["recommended_next_action"],
        )
        self.assertNotIn("discover", diagnostics["recommended_next_action"])

    def test_empty_generate_diagnostics_mixed_counts_are_truthful(self) -> None:
        """One positive target excluded + one evidence-less target skipped: both
        realities appear in the counts, nothing conflated."""
        self.seed_positive(TRACK_A)
        refused = self._generate(
            target_ids=[TRACK_A, TRACK_C],
            exclude_target_ids=[TRACK_A],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 2)
        self.assertEqual(diagnostics["after_direction_filter_count"], 2)
        self.assertEqual(diagnostics["direct_evidence_count"], 1)
        self.assertEqual(diagnostics["positive_evidence_count"], 1)
        self.assertEqual(diagnostics["negative_evidence_count"], 0)
        self.assertEqual(diagnostics["candidate_count"], 1)
        self.assertEqual(diagnostics["excluded_previous_count"], 1)
        self.assertEqual(diagnostics["reason"], "all_eligible_candidates_excluded")

    def test_empty_generate_diagnostics_report_all_negative_evidence(self) -> None:
        """Negative-only conclusions reject candidates and never enter the pool.
        P16-S2: with zero positive direct states the service deterministically
        routes into the inferred channel, so the envelope that surfaces is the
        inferred funnel's all-negative remedy -- the pointer to the inferred
        tool is obsolete because that channel already ran."""
        self.seed_negative(TRACK_A)
        refused = self._generate(
            target_ids=[TRACK_A],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 1)
        self.assertEqual(diagnostics["after_direction_filter_count"], 1)
        self.assertEqual(diagnostics["direct_evidence_count"], 1)
        self.assertEqual(diagnostics["positive_evidence_count"], 0)
        self.assertEqual(diagnostics["negative_evidence_count"], 1)
        self.assertEqual(diagnostics["candidate_count"], 0)
        self.assertEqual(diagnostics["reason"], "all_targets_negative_evidence")
        self.assertIn("提供带正向证据", diagnostics["recommended_next_action"])

    def test_empty_inferred_diagnostics_report_direction_filter_removal(self) -> None:
        """Same hard direction filter in the inferred funnel, with its extra pool
        fields present and truthful."""
        self.seed_positive(TRACK_A)
        refused = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 5,
                    "genres": ["Classical"],
                },
            ),
            completed_at=ISO,
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 1)
        self.assertEqual(diagnostics["after_direction_filter_count"], 0)
        self.assertEqual(diagnostics["reason"], "direction_filtered_all_targets")
        self.assertEqual(diagnostics["inferred_positive_count"], 0)
        self.assertEqual(diagnostics["catalog_candidate_count"], 0)
        self.assertIn("genres", diagnostics["recommended_next_action"])
        self.assertNotIn(
            "generate_inferred_recommendation",
            diagnostics["recommended_next_action"],
        )

    def test_empty_inferred_diagnostics_report_empty_candidate_pool(self) -> None:
        """No evidence anywhere -- direct, inferred and catalog layers are all
        empty; discovery advice appears only for this truly-empty pool. With no
        catalog bindings in the fixture the P15-S3-S3A supply facts report the
        zero pool (the "supply exhausted" branch of the remedy)."""
        self.seed_fixture()
        refused = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["input_target_count"], 1)
        self.assertEqual(diagnostics["after_direction_filter_count"], 1)
        self.assertEqual(diagnostics["direct_evidence_count"], 0)
        self.assertEqual(diagnostics["positive_evidence_count"], 0)
        self.assertEqual(diagnostics["negative_evidence_count"], 0)
        self.assertEqual(diagnostics["candidate_count"], 0)
        self.assertEqual(diagnostics["inferred_positive_count"], 0)
        self.assertEqual(diagnostics["catalog_candidate_count"], 0)
        self.assertEqual(diagnostics["reason"], "no_candidates_in_pool")
        self.assertNotIn(
            "generate_inferred_recommendation",
            diagnostics["recommended_next_action"],
        )
        self.assertIn(
            "discover_catalog_tracks",
            diagnostics["recommended_next_action"],
        )
        # P15-S3-S3A: zero known supply is reported as such, never invented.
        self.assertEqual(diagnostics["known_catalog_track_count"], 0)
        self.assertEqual(diagnostics["known_never_recommended_count"], 0)
        self.assertEqual(diagnostics["known_eligible_count"], 0)
        self.assertEqual(diagnostics["known_rejected_count"], 0)
        self.assertEqual(diagnostics["known_never_recommended_eligible_count"], 0)
        self.assertIn("已知目录候选供给已耗尽", diagnostics["recommended_next_action"])

    def test_empty_inferred_diagnostics_expose_known_supply_facts(self) -> None:
        """One catalog-bound known track exists (never recommended) but matches
        no positive direction: the supply facts prove the pool is there while
        the remedy explains why it did not become candidates."""
        self.seed_annotated_fixture()
        self.assertTrue(
            self.service._catalog_track_state.ensure_state(
                TRACK_C, source_system="itunes_store"
            )
        )
        self.seed_positive(TRACK_A)
        refused = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_B],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["reason"], "no_candidates_in_pool")
        self.assertEqual(diagnostics["known_catalog_track_count"], 1)
        self.assertEqual(diagnostics["known_never_recommended_count"], 1)
        self.assertEqual(diagnostics["known_eligible_count"], 0)
        self.assertEqual(diagnostics["known_rejected_count"], 0)
        self.assertEqual(diagnostics["known_never_recommended_eligible_count"], 0)
        self.assertIn("已知目录供给为 1 首", diagnostics["recommended_next_action"])
        self.assertIn("从未推荐过", diagnostics["recommended_next_action"])
        self.assertIn("discover_catalog_tracks", diagnostics["recommended_next_action"])

    def test_empty_inferred_diagnostics_report_qualified_known_supply(self) -> None:
        """The known track WAS eligible this run but the repeat-suppression
        excluded it: the supply facts report the usable candidate (eligible=1)
        and stop counting it as never-recommended after its first delivery."""
        import json as _json
        from music_agent.repository import CanonicalRepository

        fixture = _json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for track in fixture["tracks"]:
            if track["id"] == TRACK_C:
                track["external_ids"] = {
                    "apple_music_persistent_id": None,
                    "itunes_store_id": "STORE-C",
                }
                track["genres"] = ["Synthetic Pop"]
        validate_fixture(fixture)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.assertTrue(
            self.service._catalog_track_state.ensure_state(
                TRACK_C, source_system="itunes_store"
            )
        )
        self.seed_positive(TRACK_A)
        generated = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_B],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        self.assertEqual(generated.payload["item_count"], 1)
        refused = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_B],
                    "limit": 5,
                    "avoid_previous_runs": True,
                },
            ),
            completed_at=ISO,
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["reason"], "all_eligible_candidates_excluded")
        self.assertEqual(diagnostics["catalog_candidate_count"], 1)
        self.assertEqual(diagnostics["known_catalog_track_count"], 1)
        self.assertEqual(diagnostics["known_eligible_count"], 1)
        self.assertEqual(diagnostics["known_rejected_count"], 0)
        # Already delivered once this run's history: no longer never-recommended.
        self.assertEqual(diagnostics["known_never_recommended_count"], 0)
        self.assertEqual(diagnostics["known_never_recommended_eligible_count"], 0)

    def test_empty_plain_diagnostics_carry_no_known_supply_keys(self) -> None:
        """The plain envelope survives wherever the deterministic fallback must
        not fire -- a fresh run (same-run promoted provenance present). It
        stays byte-compatible with the M2-2 shape: no Known-Catalog supply
        concept, no fresh keys."""
        self.seed_annotated_fixture()
        self.assertTrue(
            self.service._catalog_track_state.ensure_state(
                TRACK_C, source_system="itunes_store"
            )
        )
        with self.assertRaises(EmptyRecommendationError) as raised:
            self.service._execute_generate_recommendation(
                {"target_ids": [TRACK_A, TRACK_C], "limit": 5},
                fresh_canonical_ids=(TRACK_C,),
                produced_at=datetime.fromisoformat(ISO),
            )
        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["reason"], "no_direct_evidence")
        self.assertEqual(diagnostics["input_target_count"], 2)
        self.assertEqual(diagnostics["after_direction_filter_count"], 2)
        for key in (
            "known_catalog_track_count",
            "known_never_recommended_count",
            "known_eligible_count",
            "known_rejected_count",
            "known_never_recommended_eligible_count",
            "inferred_positive_count",
            "catalog_candidate_count",
            "fresh_promoted_count",
            "fresh_candidate_count",
            "fresh_negative_rejected_count",
        ):
            self.assertNotIn(key, diagnostics)
        self.assertIn(
            "generate_inferred_recommendation",
            diagnostics["recommended_next_action"],
        )

    # --- P16-S2: deterministic plain -> inferred fallback ---------------------

    def test_plain_falls_back_to_inferred_when_no_positive_direct_evidence(self) -> None:
        """Zero positive direct states makes the plain funnel provably empty, so
        the service routes to the inferred channel inside the same tool call.
        The delivered batch is what generate_inferred_recommendation returns
        for the same payload, annotated with channel=inferred_fallback, and is
        a normal persisted run (history + active batch pointer)."""
        self.seed_catalog_supply()
        result = self._generate(
            target_ids=[self.EXTRA_1],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["channel"], "inferred_fallback")
        targets = {
            item.candidate.target.target_id
            for item in decode_recommendation_result(result.payload["encoded_result"]).items
        }
        self.assertIn(self.EXTRA_1, targets)
        self.assertEqual(result.payload["fresh_item_count"], 0)
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], result.payload["run_id"])
        context = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(
            context.payload["active_batch"]["run_id"], result.payload["run_id"]
        )

    def test_plain_fallback_result_matches_inferred_for_the_same_payload(self) -> None:
        """Contract tightening, not a new recommender: the fallback returns
        exactly the inferred tool's items/scores/order for the same payload."""
        self.seed_catalog_supply()
        # P19-T15: both sides opt out of recent-run dedup explicitly, keeping
        # "same payload" literally true (the equality under test is the
        # fallback-vs-inferred contract, not the exclusion default).
        fallback = self._generate(
            target_ids=[self.EXTRA_1],
            produced_at="2026-08-16T10:00:00+00:00",
            avoid_previous_runs=False,
        )
        inferred = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [self.EXTRA_1],
                    "limit": 5,
                    "avoid_previous_runs": False,
                },
            ),
            completed_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(inferred.outcome, AgentToolOutcome.OK)

        def targets(result: object) -> list[str]:
            return [
                item.candidate.target.target_id
                for item in decode_recommendation_result(result.payload["encoded_result"]).items
            ]

        self.assertEqual(targets(fallback), targets(inferred))
        self.assertEqual(fallback.payload["item_count"], inferred.payload["item_count"])
        self.assertEqual(
            fallback.payload["fresh_item_count"], inferred.payload["fresh_item_count"]
        )

    def test_plain_fallback_passes_request_level_exclusions_through(self) -> None:
        """exclude_target_ids / avoid_previous_runs are two of the request's
        quality controls: the fallback must honor them exactly like the
        inferred tool does, or the tightening would silently widen results."""
        self.seed_catalog_supply(extra_count=3)
        # The known catalog track is excluded; the fallback batch must not
        # contain it even though it is the highest-ranked catalog candidate.
        excluded = self._generate(
            target_ids=[self.EXTRA_1, self.EXTRA_2, self.EXTRA_3],
            exclude_target_ids=[self.EXTRA_3],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(excluded.outcome, AgentToolOutcome.OK)
        self.assertEqual(excluded.payload["channel"], "inferred_fallback")
        targets = {
            item.candidate.target.target_id
            for item in decode_recommendation_result(excluded.payload["encoded_result"]).items
        }
        self.assertNotIn(self.EXTRA_3, targets)

    def test_plain_fallback_honors_the_direction_filter(self) -> None:
        """genres is a hard input-side filter in BOTH channels: the fallback
        applies the caller's direction to the inferred funnel instead of
        widening it, and a non-matching direction still refuses (it empties
        even the fallback, so the plain envelope's direction remedy keeps
        being the right answer)."""
        self.seed_catalog_supply()
        matched = self._generate(
            target_ids=[self.EXTRA_1],
            genres=["Synthetic Pop"],
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(matched.outcome, AgentToolOutcome.OK)
        self.assertEqual(matched.payload["channel"], "inferred_fallback")
        self.assertIn(
            self.EXTRA_1,
            {
                item.candidate.target.target_id
                for item in decode_recommendation_result(
                    matched.payload["encoded_result"]
                ).items
            },
        )
        refused = self._generate(
            target_ids=[self.EXTRA_1],
            genres=["Classical"],
            produced_at="2026-08-16T10:01:00+00:00",
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")

    def test_generate_tools_reject_invalid_exclusion_and_direction_args(self) -> None:
        for bad in (
            {"exclude_target_ids": []},
            {"exclude_target_ids": ["not-a-track-id"]},
            {"avoid_previous_runs": "true"},
            {"genres": []},
            {"genres": ["  "]},
            {"genres": [7]},
        ):
            with self.subTest(bad=bad):
                result = self.service.execute(
                    make_request(
                        tool="generate_recommendation",
                        payload={"target_ids": [TRACK_A], "limit": 3, **bad},
                    ),
                    completed_at=ISO,
                )
                self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)
                self.assertEqual(result.error_code, "validation_error")

    def test_generate_and_history_annotate_playback_capability(self) -> None:
        by_id = self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        self.seed_positive(TRACK_C)
        expected = {
            TRACK_A: {"route": "library", "label": "可正式播放"},
            TRACK_B: {"route": "unavailable", "label": "不可用"},
            TRACK_C: {"route": "preview_only", "label": "只能试听"},
        }

        generated = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B, TRACK_C],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(generated.outcome, AgentToolOutcome.OK)
        annotations = {
            item["target_id"]: item["playback"] for item in generated.payload["items"]
        }
        self.assertEqual(annotations, expected)
        names = {item["target_id"]: item["name"] for item in generated.payload["items"]}
        self.assertEqual(names[TRACK_A], by_id[TRACK_A]["name"])
        # The persisted/ranked contract is untouched by the view-layer annotation.
        decoded = decode_recommendation_result(generated.payload["encoded_result"])
        self.assertEqual(
            {item.candidate.target.target_id for item in decoded.items},
            {TRACK_A, TRACK_B, TRACK_C},
        )

        history = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        )
        history_annotations = {
            entry["target_id"]: entry["playback"]
            for entry in history.payload["runs"][0]["items"]
        }
        self.assertEqual(history_annotations, expected)

    def test_inferred_generate_items_carry_playback_annotation(self) -> None:
        self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        self.seed_positive(TRACK_C)
        inferred = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(inferred.outcome, AgentToolOutcome.OK)
        annotations = {
            item["target_id"]: item["playback"] for item in inferred.payload["items"]
        }
        self.assertEqual(
            annotations[TRACK_A], {"route": "library", "label": "可正式播放"}
        )
        self.assertEqual(
            annotations[TRACK_B], {"route": "unavailable", "label": "不可用"}
        )
        # The iTunes-bound catalog candidate surfaces with its preview-only route.
        self.assertIn(TRACK_C, annotations)
        self.assertEqual(
            annotations[TRACK_C], {"route": "preview_only", "label": "只能试听"}
        )

    def test_inferred_genres_direction_filters_targets_and_catalog(self) -> None:
        # Custom model: TRACK_A (Synthetic Pop) and TRACK_C (Synthetic Ambient) both
        # catalog-bound so the catalog candidate layer is active; TRACK_B stays plain.
        # P19-T15: every probe here opts out of recent-run dedup -- the direction
        # filter is the only control under test.
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        for track in fixture["tracks"]:
            if track["id"] == TRACK_A:
                track["external_ids"]["itunes_store_id"] = "STORE-A"
            elif track["id"] == TRACK_C:
                track["genres"] = ["Synthetic Ambient"]
                track["external_ids"]["itunes_store_id"] = "STORE-C"
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        self.seed_positive(TRACK_C)

        def inferred(**extra: object) -> object:
            produced_at = extra.pop("produced_at", None)
            return self.service.execute(
                make_request(
                    tool="generate_inferred_recommendation",
                    payload={
                        "target_ids": [TRACK_A, TRACK_B],
                        "limit": 5,
                        "avoid_previous_runs": False,
                        **extra,
                    },
                ),
                completed_at=produced_at or ISO,
            )

        def item_targets(result: object) -> set[str]:
            return {
                item.candidate.target.target_id
                for item in decode_recommendation_result(result.payload["encoded_result"]).items
            }

        # Undirected: both preference-driven targets plus the catalog candidate C.
        self.assertEqual(
            item_targets(inferred(produced_at="2026-08-16T11:00:00+00:00")),
            {TRACK_A, TRACK_B, TRACK_C},
        )
        # Directed to Synthetic Pop: non-matching targets dropped from every layer.
        self.assertEqual(
            item_targets(inferred(produced_at="2026-08-16T11:01:00+00:00", genres=["Synthetic Pop"])),
            {TRACK_A},
        )
        # Directed to Synthetic Ambient: only the catalog candidate survives.
        self.assertEqual(
            item_targets(inferred(produced_at="2026-08-16T11:02:00+00:00", genres=["Synthetic Ambient"])),
            {TRACK_C},
        )
        # A direction the store has nothing for is refused without persistence
        # (P14-R2): empty runs never enter recommendation history.
        refused = inferred(produced_at="2026-08-16T11:03:00+00:00", genres=["Classical"])
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")
        self.assertIsNone(refused.payload)

    # --- feedback / learning tools ---------------------------------------------

    def test_record_feedback_then_read_and_interpret(self) -> None:
        recorded = self.service.execute(
            make_request(
                tool="record_feedback",
                payload=record_feedback_payload(feedback_id="fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            ),
            completed_at=ISO,
        )
        self.assertEqual(recorded.outcome, AgentToolOutcome.OK)
        feedback_id = recorded.payload["feedback_id"]

        listed = self.service.execute(
            make_request(tool="list_feedback_observations"), completed_at=ISO
        )
        self.assertEqual(listed.payload["count"], 1)
        # P16-S1: the durable observed_at is the service execution instant
        # (trusted completed_at seam), never model input.
        self.assertEqual(
            json.loads(listed.payload["observations"][0])["observed_at"], ISO
        )
        fetched = self.service.execute(
            make_request(
                tool="get_feedback_observation", payload={"feedback_id": feedback_id}
            ),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)

        interpreted = self.service.execute(
            make_request(tool="interpret_feedback", payload={"feedback_id": feedback_id}),
            completed_at=ISO,
        )
        self.assertEqual(interpreted.outcome, AgentToolOutcome.OK)
        self.assertEqual(interpreted.payload["direction"], "positive")
        self.assertEqual(interpreted.payload["reason"], "explicit_statement")
        self.assertEqual(interpreted.payload["explicitness"], "explicit")

    def test_record_feedback_observed_at_is_service_authoritative(self) -> None:
        """P16-S1: the durable observation time is the service's own execution
        instant, even when the same payload executes at different instants."""
        first = self.service.execute(
            make_request(tool="record_feedback", payload=record_feedback_payload()),
            completed_at="2026-08-16T03:30:00+00:00",
        )
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        listed = self.service.execute(
            make_request(tool="list_feedback_observations"), completed_at=ISO
        )
        self.assertEqual(
            json.loads(listed.payload["observations"][0])["observed_at"],
            "2026-08-16T03:30:00+00:00",
        )

    def test_model_cannot_forge_feedback_observed_at(self) -> None:
        """P16-S1: a model-provided observed_at (past or future) is an unknown
        payload key -- rejected before any history is written."""
        for forgery in (
            "2000-01-01T00:00:00+00:00",
            "2099-01-01T00:00:00+00:00",
        ):
            with self.subTest(forgery=forgery):
                result = self.service.execute(
                    make_request(
                        tool="record_feedback",
                        payload={**record_feedback_payload(), "observed_at": forgery},
                    ),
                    completed_at=ISO,
                )
                self.assertEqual(result.outcome, AgentToolOutcome.INVALID_REQUEST)
        listed = self.service.execute(
            make_request(tool="list_feedback_observations"), completed_at=ISO
        )
        self.assertEqual(listed.payload["count"], 0)

    def test_model_cannot_forge_learning_applied_at(self) -> None:
        """P16-S1: a model-provided applied_at is an unknown payload key --
        rejected before any application journal entry exists."""
        self.service.execute(
            make_request(tool="record_feedback", payload=record_feedback_payload()),
            completed_at=ISO,
        )
        forged = self.service.execute(
            make_request(
                tool="apply_learning",
                payload={
                    "feedback_id": "fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "applied_at": "2099-01-01T00:00:00+00:00",
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(forged.outcome, AgentToolOutcome.INVALID_REQUEST)
        listed = self.service.execute(
            make_request(tool="list_learning_applications"), completed_at=ISO
        )
        self.assertEqual(len(listed.payload["applications"]), 0)

    def test_duplicate_feedback_fails_closed_through_the_service(self) -> None:
        payload = record_feedback_payload(
            feedback_id="fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        )
        self.assertEqual(
            self.service.execute(
                make_request(tool="record_feedback", payload=payload), completed_at=ISO
            ).outcome,
            AgentToolOutcome.OK,
        )
        duplicate = self.service.execute(
            make_request(tool="record_feedback", payload=payload), completed_at=ISO
        )
        self.assertEqual(duplicate.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(duplicate.error_code, "duplicate_feedback_observation")

    def test_apply_learning_writes_durable_evidence_and_reads_back(self) -> None:
        recorded = self.service.execute(
            make_request(
                tool="record_feedback",
                payload=record_feedback_payload(
                    track_id=TRACK_A,
                    feedback_id="fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                ),
            ),
            completed_at=ISO,
        )
        applied = self.service.execute(
            make_request(
                tool="apply_learning",
                payload={
                    "feedback_id": recorded.payload["feedback_id"],
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(applied.outcome, AgentToolOutcome.OK)
        self.assertTrue(applied.payload["applied"])
        self.assertEqual(applied.payload["proposal_kind"], "evidence_observation")

        listed = self.service.execute(
            make_request(tool="list_learning_applications"), completed_at=ISO
        )
        self.assertEqual(len(listed.payload["applications"]), 1)
        self.assertEqual(
            listed.payload["applications"][0]["feedback_id"],
            recorded.payload["feedback_id"],
        )
        # P16-S1: applied_at is the service execution instant (the trusted
        # completed_at seam), never model input.
        self.assertEqual(listed.payload["applications"][0]["applied_at"], ISO)

        # The applied evidence is durable P06 state visible through the preference tool.
        queried = self.service.execute(
            make_request(
                tool="query_track_preference",
                payload={"target_id": TRACK_A, "source_system": "feedback_learning"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(queried.payload["preference_state"], "positive")

    def test_apply_learning_without_proposal_applies_nothing(self) -> None:
        recorded = self.service.execute(
            make_request(
                tool="record_feedback",
                payload=record_feedback_payload(
                    kind="skipped",
                    feedback_id="fbk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                ),
            ),
            completed_at=ISO,
        )
        applied = self.service.execute(
            make_request(
                tool="apply_learning",
                payload={"feedback_id": recorded.payload["feedback_id"]},
            ),
            completed_at=ISO,
        )
        self.assertEqual(applied.outcome, AgentToolOutcome.OK)
        self.assertFalse(applied.payload["applied"])
        self.assertEqual(applied.payload["reason"], "no_proposal")

    def test_learning_application_duplicate_fails_closed(self) -> None:
        feedback_id = "fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        self.service.execute(
            make_request(tool="record_feedback", payload=record_feedback_payload(feedback_id=feedback_id)),
            completed_at=ISO,
        )
        apply_payload = {"feedback_id": feedback_id}
        self.assertEqual(
            self.service.execute(
                make_request(tool="apply_learning", payload=apply_payload), completed_at=ISO
            ).outcome,
            AgentToolOutcome.OK,
        )
        duplicate = self.service.execute(
            make_request(tool="apply_learning", payload=apply_payload), completed_at=ISO
        )
        self.assertEqual(duplicate.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(duplicate.error_code, "duplicate_learning_application")

    def test_missing_feedback_fails_closed(self) -> None:
        for tool in ("get_feedback_observation", "interpret_feedback", "apply_learning"):
            result = self.service.execute(
                make_request(
                    tool=tool,
                    payload={"feedback_id": "fbk_cccccccc-cccc-4ccc-8ccc-cccccccccccc"},
                ),
                completed_at=ISO,
            )
            self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR, tool)
            self.assertEqual(result.error_code, "feedback_not_found", tool)

    # --- capability surface -----------------------------------------------------

    def test_get_agent_capabilities_projects_the_gate(self) -> None:
        result = self.service.execute(
            make_request(tool="get_agent_capabilities"), completed_at=ISO
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(result.payload["agent_contract_version"], AGENT_CONTRACT_VERSION)
        self.assertEqual(result.payload["schema_version"], 19)
        self.assertTrue(len(result.payload["tools"]) >= 14)
        self.assertTrue(len(result.payload["writes"]) > 0)
        for write in result.payload["writes"]:
            self.assertFalse(write["execution_ready"])

    def test_execute_write_intent_fails_closed_at_the_capability_gate(self) -> None:
        from music_agent.intent_repository import PendingIntentRepository

        intent = PendingIntent(
            intent_id=INTENT_ID,
            operation=WriteOperation.SET_FAVORITED,
            requirements=(
                WriteRequirement(
                    RequirementRole.TARGET,
                    TRACK_A,
                    ExternalIdentityKey("apple_music", EntityType.TRACK, "persist-001"),
                ),
            ),
            requested_value=ObservedValue.value(True),
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)

        result = self.service.execute(
            make_request(tool="execute_write_intent", payload={"intent_id": INTENT_ID}),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.NOT_EXECUTION_READY)
        self.assertEqual(result.error_code, "not_execution_ready")

    def test_execute_write_intent_missing_intent_fails_closed(self) -> None:
        result = self.service.execute(
            make_request(
                tool="execute_write_intent",
                payload={"intent_id": "int_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "write_intent_not_found")

    # --- restart durability -------------------------------------------------------

    def test_state_survives_close_and_reopen(self) -> None:
        self.seed_positive(TRACK_A)
        self.service.execute(
            make_request(
                tool="record_feedback",
                payload=record_feedback_payload(
                    feedback_id="fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
                ),
            ),
            completed_at=ISO,
        )
        self.service.execute(
            make_request(tool="get_agent_capabilities"), completed_at=ISO
        )
        self.service.close()

        reopened = SharedAgentService(self.database_path, clients=registry())
        try:
            queried = reopened.execute(
                make_request(
                    tool="query_track_preference",
                    payload={"target_id": TRACK_A, "source_system": "apple_music"},
                ),
                completed_at=ISO,
            )
            self.assertEqual(queried.payload["preference_state"], "positive")
            listed = reopened.execute(
                make_request(tool="list_feedback_observations"), completed_at=ISO
            )
            self.assertEqual(listed.payload["count"], 1)
            from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

            with AgentRequestJournalRepository(self.database_path) as journal:
                # Two requests before the reopen, two after: all four survived.
                self.assertEqual(len(journal.list()), 4)
        finally:
            reopened.close()

    # --- query_catalog_discovery_state + save_result projection (P15-S3-S2) -----

    def test_generate_projection_updates_catalog_row_and_skips_library(self) -> None:
        """A persisted track run updates the catalog row of its catalog-bound item and
        leaves library targets exactly as they were: no row, never created."""
        self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_C)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertTrue(state.ensure_state(TRACK_C, source_system="itunes_store"))

        first = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_C],
                    "limit": 5,
                },
            ),
            completed_at="2026-08-16T09:00:00+00:00",
        )
        self.assertEqual(first.outcome, AgentToolOutcome.OK)
        decoded = decode_recommendation_result(first.payload["encoded_result"])
        self.assertEqual(
            {item.candidate.target.target_id for item in decoded.items},
            {TRACK_A, TRACK_C},
        )
        with CatalogTrackStateRepository(self.database_path) as state:
            row = state.get_state(TRACK_C)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.recommendation_count, 1)
            self.assertEqual(row.first_recommended_at, "2026-08-16T09:00:00+00:00")
            self.assertEqual(row.last_recommended_at, "2026-08-16T09:00:00+00:00")
            self.assertEqual(row.source_system, "itunes_store")
            # Library tracks never materialize a catalog row, even as persisted items.
            self.assertIsNone(state.get_state(TRACK_A))
            self.assertEqual(state.count(), 1)

    def test_projection_skips_excluded_and_refused_runs(self) -> None:
        """Request-level exclusions and refused (empty) runs contribute nothing to the
        projection: neither the excluded catalog target nor the refused run's targets
        move any catalog row."""
        self.seed_annotated_fixture()
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_C)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertTrue(state.ensure_state(TRACK_C, source_system="itunes_store"))

        def generate(**payload: object) -> object:
            produced_at = payload.pop("produced_at", None)
            return self.service.execute(
                make_request(
                    tool="generate_recommendation",
                    payload=dict(
                        {"target_ids": [TRACK_A, TRACK_C], "limit": 5}, **payload
                    ),
                ),
                completed_at=produced_at or ISO,
            )

        excluded = generate(
            exclude_target_ids=[TRACK_C], produced_at="2026-08-16T09:00:00+00:00"
        )
        self.assertEqual(excluded.outcome, AgentToolOutcome.OK)
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.get_state(TRACK_C).recommendation_count, 0)

        refused = generate(
            exclude_target_ids=[TRACK_A, TRACK_C],
            produced_at="2026-08-16T09:01:00+00:00",
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(refused.error_code, "empty_recommendation")
        with CatalogTrackStateRepository(self.database_path) as state:
            self.assertEqual(state.get_state(TRACK_C).recommendation_count, 0)
            self.assertIsNone(state.get_state(TRACK_C).first_recommended_at)

        fresh = generate(produced_at="2026-08-16T09:02:00+00:00")
        self.assertEqual(fresh.outcome, AgentToolOutcome.OK)
        with CatalogTrackStateRepository(self.database_path) as state:
            row = state.get_state(TRACK_C)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.recommendation_count, 1)
            self.assertEqual(row.first_recommended_at, "2026-08-16T09:02:00+00:00")
            self.assertEqual(row.last_recommended_at, row.first_recommended_at)

    def test_query_discovery_state_canonical_id_mode(self) -> None:
        self.seed_fixture()
        with CatalogTrackStateRepository(self.database_path) as state:
            state.ensure_state(TRACK_C, source_system="itunes_store")
            state.record_discovery_occurrence(
                TRACK_C,
                source_system="itunes_store",
                term="Midnight",
                now=datetime(2026, 8, 16, 8, 0, 0, tzinfo=timezone.utc),
            )
            state.record_recommendation_items(
                [TRACK_C],
                produced_at=datetime(2026, 8, 16, 9, 0, 0, tzinfo=timezone.utc),
            )

        queried = self.service.execute(
            make_request(
                tool="query_catalog_discovery_state",
                payload={"canonical_id": TRACK_C},
            ),
            completed_at=ISO,
        )
        self.assertEqual(queried.outcome, AgentToolOutcome.OK)
        self.assertTrue(queried.payload["found"])
        entry = queried.payload["state"]
        self.assertEqual(entry["canonical_id"], TRACK_C)
        self.assertEqual(entry["source_system"], "itunes_store")
        self.assertEqual(entry["discovery_count"], 1)
        self.assertEqual(entry["discovery_terms"], {"midnight": 1})
        self.assertEqual(entry["recommendation_count"], 1)
        self.assertFalse(entry["never_recommended"])
        self.assertTrue(entry["previously_recommended"])
        # Deterministic display hydration from the canonical model.
        self.assertIsNotNone(entry["name"])
        # Facts, never ranking/score/eligibility.
        for key in ("score", "ranking", "eligible_for_exploration"):
            self.assertNotIn(key, entry)

        missing = self.service.execute(
            make_request(
                tool="query_catalog_discovery_state",
                payload={"canonical_id": TRACK_B},
            ),
            completed_at=ISO,
        )
        self.assertEqual(missing.outcome, AgentToolOutcome.OK)
        self.assertFalse(missing.payload["found"])
        self.assertNotIn("state", missing.payload)

    def test_query_discovery_state_term_mode_counts_and_caps(self) -> None:
        self.seed_fixture()
        # TRACK_B is not part of the canonical model fixture: give it a canonical entity
        # so its memory row passes the honest FK (display hydration stays empty for it).
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository._connection.execute(
                "INSERT INTO canonical_entities(id, entity_type) VALUES (?, 'track')",
                (TRACK_B,),
            )
        base = datetime(2026, 8, 16, 8, 0, 0, tzinfo=timezone.utc)
        with CatalogTrackStateRepository(self.database_path) as state:
            state.ensure_state(TRACK_C, source_system="itunes_store")
            state.ensure_state(TRACK_B, source_system="itunes_store")
            state.record_discovery_occurrence(
                TRACK_C, source_system="itunes_store", term="J-Pop", now=base
            )
            state.record_discovery_occurrence(
                TRACK_C,
                source_system="itunes_store",
                term="j-pop",
                now=base + timedelta(minutes=1),
            )
            state.record_discovery_occurrence(
                TRACK_B, source_system="itunes_store", term="J-Pop", now=base + timedelta(minutes=30)
            )

        queried = self.service.execute(
            make_request(
                tool="query_catalog_discovery_state",
                payload={"term": "  J-POP "},
            ),
            completed_at=ISO,
        )
        self.assertEqual(queried.outcome, AgentToolOutcome.OK)
        self.assertEqual(queried.payload["term"], "  J-POP ")
        self.assertEqual(queried.payload["match_count"], 2)
        self.assertEqual(
            [entry["canonical_id"] for entry in queried.payload["matches"]],
            [TRACK_B, TRACK_C],
        )
        self.assertEqual(queried.payload["matches"][0]["term_count"], 1)
        self.assertEqual(queried.payload["matches"][1]["term_count"], 2)

        capped = self.service.execute(
            make_request(
                tool="query_catalog_discovery_state",
                payload={"term": "j-pop", "limit": 1},
            ),
            completed_at=ISO,
        )
        self.assertEqual(capped.payload["match_count"], 2)
        self.assertEqual(len(capped.payload["matches"]), 1)

        unknown = self.service.execute(
            make_request(
                tool="query_catalog_discovery_state",
                payload={"term": "never-searched"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(unknown.payload["match_count"], 0)
        self.assertEqual(unknown.payload["matches"], [])

    def test_query_discovery_state_refuses_malformed_payloads(self) -> None:
        for payload in ({}, {"canonical_id": TRACK_A, "term": "x"}):
            with self.subTest(payload=payload):
                refused = self.service.execute(
                    make_request(
                        tool="query_catalog_discovery_state", payload=payload
                    ),
                    completed_at=ISO,
                )
                self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
                self.assertEqual(refused.error_code, "validation_error")


def target(track_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id)


class TenRunNoveltyProofTest(SharedAgentServiceTest):
    """P19-T15 local proof (single run -- not inherited by the floor classes).

    Ten consecutive UNQUALIFIED inferred runs over a seeded 60-track pool (2
    preference favorites + 58 catalog-bound extras) must produce ten mutually
    distinct limit-5 batches with zero adjacent overlap: the deterministic
    cross-run repetition the owner saw is gone without any payload opt-in.
    The fixture stays local (this test's own TemporaryDirectory) -- the
    owner's real recommendation history is never touched.
    """

    def test_ten_consecutive_unqualified_runs_stay_novel_and_never_recycle(self) -> None:
        """Discovery-style state rows give the count tie-break its persisted
        authority; a sleep after each run pushes the next run's
        recommendation-history created_at (DB CURRENT_TIMESTAMP, second
        resolution, the exclusion window's ordering key) onto a distinct
        second so the newest-5 window is deterministic."""
        import time

        from music_agent.catalog_track_state_repository import CatalogTrackStateRepository

        extras = 58

        def extras_id(index: int) -> str:
            return f"trk_ffffffff-ffff-4fff-8fff-{index:012x}"

        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        for index in range(1, extras + 1):
            fixture["tracks"].append(
                {
                    "id": extras_id(index),
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "itunes_store_id": f"PROOF-{index}",
                    },
                    "name": f"Novelty Pop {index}",
                    "artist_ids": ["art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"],
                    "album_id": None,
                    "duration_ms": None,
                    "genres": ["Synthetic Pop"],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": None,
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)
        with CatalogTrackStateRepository(self.database_path) as states:
            for index in range(1, extras + 1):
                self.assertTrue(
                    states.ensure_state(
                        extras_id(index), source_system="apple_music_catalog"
                    )
                )

        batches: list[list[str]] = []
        for run_index in range(10):
            result = self.service.execute(
                make_request(
                    tool="generate_inferred_recommendation",
                    payload={"target_ids": [TRACK_A, TRACK_B], "limit": 5},
                ),
                completed_at=f"2026-08-16T10:{run_index:02d}:00+00:00",
            )
            self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
            decoded = decode_recommendation_result(result.payload["encoded_result"])
            batch = [item.candidate.target.target_id for item in decoded.items]
            self.assertEqual(len(batch), 5)
            self.assertEqual(result.payload["item_count"], 5)
            batches.append(batch)
            if run_index < 9:
                time.sleep(1.01)

        # Zero adjacent overlap: a target from run N never resurfaces in N+1.
        for previous, following in zip(batches, batches[1:]):
            self.assertFalse(set(previous) & set(following))
        # All ten batches mutually distinct -- no repeated full batch (no AAAA).
        self.assertEqual(len({tuple(batch) for batch in batches}), 10)
        # First batch is the pre-T15 head (relevance primary, then id order).
        self.assertEqual(
            batches[0],
            [TRACK_A, TRACK_B, extras_id(1), extras_id(2), extras_id(3)],
        )
        # Window expiry readmits earlier history (A/B, run 1 slid out of the
        # newest-5), and the count tie-break surfaces the never-recommended
        # tail (X29+) ahead of the once-recommended X1-X3 at the same score.
        self.assertEqual(
            batches[6],
            [TRACK_A, TRACK_B, extras_id(29), extras_id(30), extras_id(31)],
        )
        # 48 distinct tracks across 50 slots -- the pool is drawn down, not
        # recycled.
        union = {target for batch in batches for target in batch}
        self.assertEqual(len(union), 48)


class ExplorationFloorIntegrationTest(SharedAgentServiceTest):
    """P15-S3-S3C: the inferred-only ``min_exploration`` floor, service-level.

    Catalog supply is staged deterministically: three Synthetic Pop / artist
    Alpha catalog-bound extras rank at 0.675 below the 0.9 familiar positives
    A and B, giving the exact owner-example shape (all-familiar head, catalog
    tail) with zero randomness.
    """

    def _inferred(self, **extra: object) -> object:
        # P15 burn-down Issue 1: produced_at goes through the trusted
        # completed_at execution seam, never the payload.
        produced_at = extra.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 2, **extra},
            ),
            completed_at=produced_at or ISO,
        )

    def _decoded(self, result: object) -> list:
        return list(
            decode_recommendation_result(result.payload["encoded_result"]).items
        )

    def _targets(self, result: object) -> list[str]:
        return [
            item.candidate.target.target_id for item in self._decoded(result)
        ]

    # --- backward compatibility (items 9-12) ---------------------------------

    def test_floor_omission_and_zero_match_plain_behavior(self) -> None:
        """The default-0 contract: omitting min_exploration (or passing 0) yields
        the exact current items -- byte-for-byte in ids, order, and scores.
        P19-T15: both probes opt out of recent-run dedup so the run-to-run
        comparison is untainted by the exclusion default."""
        self.seed_catalog_supply()
        plain = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", avoid_previous_runs=False
        )
        zero = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00",
            min_exploration=0,
            avoid_previous_runs=False,
        )
        for result in (plain, zero):
            self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
            self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
            self.assertEqual(result.payload["item_count"], 2)
        plain_rows = [
            (item.candidate.target.target_id, item.score.total)
            for item in self._decoded(plain)
        ]
        zero_rows = [
            (item.candidate.target.target_id, item.score.total)
            for item in self._decoded(zero)
        ]
        self.assertEqual(zero_rows, plain_rows)

    def test_plain_generate_rejects_min_exploration_at_service_level(self) -> None:
        self.seed_catalog_supply()
        refused = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A],
                    "limit": 2,
                    "min_exploration": 1,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(refused.error_code, "validation_error")

    def test_floor_above_limit_fails_closed_at_service_level(self) -> None:
        self.seed_catalog_supply()
        refused = self._inferred(min_exploration=3)  # limit is 2
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(refused.error_code, "validation_error")

    # --- selection semantics (items 13-22) ------------------------------------

    def test_floor_one_swaps_lowest_familiar_for_first_catalog(self) -> None:
        """The owner's worked example, service-level: catalog gets NO score bonus,
        it only replaces the lowest-ranked Familiar place; original order kept."""
        self.seed_catalog_supply()
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=1
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["item_count"], 2)
        by_id = {
            item.candidate.target.target_id: item for item in self._decoded(result)
        }
        self.assertEqual(by_id[TRACK_A].score.total, 0.9)
        self.assertEqual(by_id[self.EXTRA_1].score.total, 0.675)
        self.assertEqual(
            by_id[TRACK_A].candidate.source.source_path, "preference_driven"
        )
        self.assertEqual(
            by_id[self.EXTRA_1].candidate.source.source_path, "catalog_driven"
        )
        self.assertEqual(
            [row["label"] for row in result.payload["items"]],
            ["known_positive", "catalog"],
        )

    def test_floor_two_permits_full_catalog_batch(self) -> None:
        self.seed_catalog_supply()
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=2
        )
        self.assertEqual(self._targets(result), [self.EXTRA_1, self.EXTRA_2])

    def test_floor_already_met_in_wider_head_is_unchanged(self) -> None:
        self.seed_catalog_supply()
        # P19-T15: both runs opt out of recent-run dedup -- the invariance
        # under test is the floor, not cross-run novelty.
        run = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 3,
                    "avoid_previous_runs": False,
                },
            ),
            completed_at=ISO,
        )
        floored = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 3,
                    "min_exploration": 1,
                    "avoid_previous_runs": False,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(
            self._targets(floored), self._targets(run)
        )
        self.assertEqual(
            self._targets(floored), [TRACK_A, TRACK_B, self.EXTRA_1]
        )

    def test_zero_qualified_supply_is_silent_best_effort(self) -> None:
        """No catalog supply at all: the floor never fabricates and never fails."""
        self.seed_catalog_supply(extra_count=0)
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=1
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertEqual(result.payload["item_count"], 2)

    def test_floor_above_supply_injects_all_available(self) -> None:
        self.seed_catalog_supply(extra_count=1)
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=2
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])

    def test_fewer_scored_items_than_limit_is_best_effort(self) -> None:
        self.seed_catalog_supply(extra_count=1)
        result = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                    "min_exploration": 1,
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(
            self._targets(result), [TRACK_A, TRACK_B, self.EXTRA_1]
        )
        self.assertEqual(result.payload["item_count"], 3)

    # --- eligibility safety: the floor never resurrects (items 23-26) ---------

    def test_negative_filtered_catalog_is_never_resurrected(self) -> None:
        self.seed_catalog_supply()
        # The dislike on EXTRA_1 directionally poisons its Synthetic Pop genre
        # basis, which the P11.2 candidate layer uses to refuse the whole shared
        # catalog supply -- before ranking. The floor must not bring any of it
        # back: best effort, honest familiar-only batch.
        self.seed_negative(self.EXTRA_1)
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=2
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        for extra in (self.EXTRA_1, self.EXTRA_2, self.EXTRA_3):
            self.assertNotIn(extra, self._targets(result))

    def test_repeat_filtered_catalog_is_never_resurrected(self) -> None:
        self.seed_catalog_supply()
        self._insert_history_run(
            "rcm_aaa00000-0000-4000-8000-0000000000aa",
            (self.EXTRA_1,),
            "2026-08-16T09:00:00+00:00",
        )
        result = self._inferred(
            produced_at="2026-08-16T10:05:00+00:00",
            min_exploration=2,
            avoid_previous_runs=True,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        # EXTRA_1 is repeat-excluded upstream; the floor fills from the tail of
        # the surviving eligible list instead.
        self.assertEqual(self._targets(result), [self.EXTRA_2, self.EXTRA_3])

    def test_floor_never_rescores_items(self) -> None:
        self.seed_catalog_supply()
        # Score oracle: a plain widened run sees the complete ranked list, so its
        # totals are the no-floor reference for every item the floor may surface.
        # P19-T15: both runs opt out of recent-run dedup -- the reference and the
        # floored batch must see identical pools.
        wide = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 3,
                    "avoid_previous_runs": False,
                },
            ),
            completed_at=ISO,
        )
        wide_scores = {
            item.candidate.target.target_id: item.score.total
            for item in self._decoded(wide)
        }
        self.assertEqual(set(wide_scores), {TRACK_A, TRACK_B, self.EXTRA_1})
        floored = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00",
            min_exploration=1,
            avoid_previous_runs=False,
        )
        for item in self._decoded(floored):
            self.assertEqual(
                item.score.total, wide_scores[item.candidate.target.target_id]
            )

    # --- identity boundary + persistence (items 27-29) ------------------------

    def test_response_shape_and_item_keys_are_unchanged(self) -> None:
        """No other new envelope fields: the floor surfaces only through the
        existing item list (silent best-effort per DESIGN D2). P15-S3-S3D adds
        EXACTLY the two fresh provenance keys -- ``fresh_this_request`` per item
        and ``fresh_item_count`` per batch -- purely additively. P18-S1.3 adds
        one further display key (``album``, the version/source disambiguator);
        P20-Fix09 adds one further per-item key (``evidence``, the Fix03-shaped
        durable evidence block); envelope keys are untouched."""
        self.seed_catalog_supply()
        result = self._inferred(
            produced_at="2026-08-16T10:00:00+00:00", min_exploration=1
        )
        self.assertEqual(
            set(result.payload.keys()),
            {
                "run_id",
                "item_count",
                "source_system",
                "items",
                "encoded_result",
                "fresh_item_count",
            },
        )
        self.assertEqual(
            set(result.payload["items"][0].keys()),
            {
                "target_id",
                "name",
                "artist_name",
                "album",
                "direct_state",
                "explanation",
                "label",
                "score_total",
                "playback",
                "fresh_this_request",
                "evidence",
            },
        )

    def test_floored_batch_persists_through_existing_projection(self) -> None:
        self.seed_catalog_supply()
        floored = self._inferred(
            produced_at="2026-08-16T10:15:00+00:00", min_exploration=1
        )
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], floored.payload["run_id"])
        self.assertEqual(
            [item["target_id"] for item in runs[0]["items"]],
            [TRACK_A, self.EXTRA_1],
        )

class FreshFloorIntegrationTest(SharedAgentServiceTest):
    """P15-S3-S3D: the inferred-only best-effort ``min_fresh`` floor plus the
    same-run Fresh truth metadata, service-level, over the S3-S3C staging.

    Fresh identity reaches the handlers ONLY through the internal
    ``fresh_canonical_ids`` execution kwarg (the provider loop's capture from
    genuinely executed discoveries). The model payload has no key for it and
    validation would reject one -- every test here supplies the kwarg on the
    ``execute`` call side, exactly like the loop does.
    """

    def _inferred(self, fresh_ids=(), **extra: object) -> object:
        # P15 burn-down Issue 1: produced_at goes through the trusted
        # completed_at execution seam, never the payload.
        produced_at = extra.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 2, **extra},
            ),
            completed_at=produced_at or ISO,
            fresh_canonical_ids=tuple(fresh_ids),
        )

    def _decoded(self, result: object) -> list:
        return list(
            decode_recommendation_result(result.payload["encoded_result"]).items
        )

    def _targets(self, result: object) -> list[str]:
        return [item.candidate.target.target_id for item in self._decoded(result)]

    def _flags(self, result: object) -> dict[str, bool]:
        return {
            item["target_id"]: item["fresh_this_request"]
            for item in result.payload["items"]
        }

    # --- floor composition (Request 2 shape, service-level) -------------------

    def test_min_fresh_one_exchanges_lowest_familiar_for_first_fresh(self) -> None:
        """The Request-2 exchange at service level: head [A, B] carries no fresh
        item; the promoted EXTRA_1 (rank 3 of the complete rank) is the highest-
        ranked eligible fresh -- min_fresh=1 replaces the lowest head place with
        it, reports 1/2 fresh, and the injected item's catalog source satisfies
        the exploration floor too (one slot, never two quotas)."""
        self.seed_catalog_supply()
        result = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(
            self._flags(result), {TRACK_A: False, self.EXTRA_1: True}
        )
        self.assertEqual(result.payload["fresh_item_count"], 1)
        self.assertEqual(result.payload["item_count"], 2)
        # P07 untouched: the floor performs membership exchange only -- the
        # familiar and the fresh entrant keep their pipeline scores exactly.
        decoded = {
            item.candidate.target.target_id: item for item in self._decoded(result)
        }
        self.assertEqual(decoded[TRACK_A].score.total, 0.9)
        self.assertEqual(decoded[self.EXTRA_1].score.total, 0.675)

    def test_min_fresh_two_replaces_both_head_places(self) -> None:
        self.seed_catalog_supply()
        result = self._inferred(
            (self.EXTRA_1, self.EXTRA_2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=2,
        )
        self.assertEqual(self._targets(result), [self.EXTRA_1, self.EXTRA_2])
        self.assertEqual(result.payload["fresh_item_count"], 2)
        self.assertEqual(
            self._flags(result), {self.EXTRA_1: True, self.EXTRA_2: True}
        )

    def test_min_fresh_picks_the_highest_ranked_fresh_from_the_tail(self) -> None:
        """Only EXTRA_3 is fresh: the floor walks past the higher-ranked
        non-fresh catalog items (EXTRA_1/2) and injects the fresh one -- order
        stays the original rank order among the selected."""
        self.seed_catalog_supply()
        result = self._inferred(
            (self.EXTRA_3,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_3])
        self.assertEqual(result.payload["fresh_item_count"], 1)

    def test_min_fresh_without_provenance_degrades_to_exploration(self) -> None:
        """Option B (normalization, not refusal): a fresh ask with an empty
        authoritative set surfaces the exploration floor (effective minimum =
        max of the two knobs) and the truth layer reports zero fresh -- the
        model's zero-fresh honesty rule then applies."""
        self.seed_catalog_supply()
        result = self._inferred(
            (),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["fresh_item_count"], 0)
        self.assertEqual(
            self._flags(result), {TRACK_A: False, self.EXTRA_1: False}
        )

    def test_min_fresh_zero_qualified_is_silent_best_effort(self) -> None:
        """No catalog supply at all (so no ranked candidate can be fresh and the
        rank cannot even widen): the floor never fabricates and never fails."""
        self.seed_catalog_supply(extra_count=0)
        result = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertEqual(result.payload["fresh_item_count"], 0)

    def test_fresh_head_item_counts_toward_exploration_no_double_quota(self) -> None:
        """A promoted preference-driven target (fresh but not catalog -- the
        live Perfect-blue shape) satisfies BOTH floors from ONE place: the
        exploration classification is catalog OR same-run-fresh, so no catalog
        top-up displaces it and the batch stays head-unchanged."""
        self.seed_catalog_supply()
        result = self._inferred(
            (TRACK_B,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
            min_exploration=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertEqual(result.payload["fresh_item_count"], 1)
        self.assertEqual(
            self._flags(result), {TRACK_A: False, TRACK_B: True}
        )

    def test_exploration_topup_never_displaces_the_fresh_place(self) -> None:
        """When the exploration floor still needs a catalog item beyond the
        counted fresh place, it evicts the lowest-ranked non-fresh non-
        exploration place -- the fresh item itself is protected (两层互不推翻)."""
        self.seed_catalog_supply()
        result = self._inferred(
            (TRACK_B,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
            min_exploration=2,
        )
        self.assertEqual(self._targets(result), [TRACK_B, self.EXTRA_1])
        self.assertEqual(result.payload["fresh_item_count"], 1)
        self.assertEqual(
            self._flags(result), {TRACK_B: True, self.EXTRA_1: False}
        )

    def test_fresh_floor_never_resurrects_filtered_fresh(self) -> None:
        """A directionally poisoned catalog basis refuses the whole shared
        supply before ranking; the fresh floor must not bring any of it back."""
        self.seed_catalog_supply()
        self.seed_negative(self.EXTRA_1)
        result = self._inferred(
            (self.EXTRA_1, self.EXTRA_2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=2,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertEqual(result.payload["fresh_item_count"], 0)

    def test_fresh_item_count_counts_final_selection_not_promoted_total(self) -> None:
        """Both EXTRA_1 and EXTRA_2 are authoritative Fresh; only EXTRA_1 lands in
        the limit-2 batch. fresh_item_count is computed over the FINAL selected
        items -- never the discover promoted total."""
        self.seed_catalog_supply()
        result = self._inferred(
            (self.EXTRA_1, self.EXTRA_2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["fresh_item_count"], 1)

    # --- provenance truth (plain tool + shape) ---------------------------------

    def test_plain_generate_reports_provenance_but_rejects_min_fresh(self) -> None:
        # The plain tool never floors (min_fresh is an unknown key there) but
        # still reports the machine truth when the loop supplies the kwarg.
        refused = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={"target_ids": [TRACK_A], "limit": 2, "min_fresh": 1},
            ),
            completed_at=ISO,
        )
        self.assertEqual(refused.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(refused.error_code, "validation_error")
        self.seed_catalog_supply()
        plain = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B],
                    "limit": 5,
                },
            ),
            completed_at=ISO,
            fresh_canonical_ids=(TRACK_A,),
        )
        self.assertEqual(plain.outcome, AgentToolOutcome.OK, plain.error_code)
        self.assertEqual(plain.payload["fresh_item_count"], 1)
        flags = {
            item["target_id"]: item["fresh_this_request"]
            for item in plain.payload["items"]
        }
        self.assertEqual(flags, {TRACK_A: True, TRACK_B: False})
        self.assertEqual(
            set(plain.payload.keys()),
            {
                "run_id",
                "item_count",
                "fresh_item_count",
                "items",
                "encoded_result",
            },
        )

    def test_absent_kwarg_means_no_fresh_for_any_call(self) -> None:
        self.seed_catalog_supply()
        result = self._inferred(
            (),
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(result.payload["fresh_item_count"], 0)
        self.assertTrue(
            all(not flag for flag in self._flags(result).values())
        )

    def test_fresh_kwarg_is_ignored_for_non_generation_tools(self) -> None:
        self.seed_catalog_supply()
        listed = self.service.execute(
            make_request(tool="list_recommendation_runs"),
            completed_at=ISO,
            fresh_canonical_ids=(TRACK_A,),
        )
        self.assertEqual(listed.outcome, AgentToolOutcome.OK)
        self.assertEqual(set(listed.payload.keys()), {"runs", "runs_total"})

    def test_encoded_result_carries_no_fresh_identity(self) -> None:
        """Persistence is untouched: the fresh identity exists only in the
        response envelope (a provenance snapshot of the run), never inside the
        encoded recommendation result nor any durable track state."""
        self.seed_catalog_supply()
        result = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        for item in self._decoded(result):
            self.assertFalse(hasattr(item, "fresh_this_request"))
        self.assertFalse(
            hasattr(self._decoded(result)[0].candidate, "fresh_this_request")
        )
        # The floored batch persists through the existing S3-S2 projection with
        # exactly the visible items -- fresh identity is not part of the write.
        runs = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        ).payload["runs"]
        self.assertEqual(
            [item["target_id"] for item in runs[0]["items"]],
            [TRACK_A, self.EXTRA_1],
        )


class FreshDrivenChannelIntegrationTest(SharedAgentServiceTest):
    """P15-S3-S3E: the additive Fresh candidate channel, service-level.

    Zero-affinity supply (unique genre + artist, no preference seeding) is the
    exact live-failure shape: the frozen catalog layer omits such tracks
    ("no directional claim => omitted"), so without the channel the fresh
    floor has nothing to select. The channel's zero-basis candidates are the
    only route in, and their honesty contract (score 0, empty basis, no
    preference-match explanation) is asserted end to end. Activation is the
    explicit-intent gate ONLY: ``min_fresh > 0`` with a non-empty
    authoritative same-run promoted set; ordinary runs stay byte-equivalent.
    """

    Z1 = "trk_ffffffff-ffff-4fff-8fff-000000000004"
    Z2 = "trk_ffffffff-ffff-4fff-8fff-000000000005"
    Z_ARTIST = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    UNKNOWN_FRESH = "trk_ffffffff-ffff-4fff-8fff-000000000009"

    @staticmethod
    def _catalog_track(suffix: int, name: str, genres: list, artist_id: str) -> dict:
        return {
            "id": f"trk_ffffffff-ffff-4fff-8fff-00000000000{suffix}",
            "external_ids": {
                "apple_music_persistent_id": None,
                "itunes_store_id": f"S3E-{suffix}",
            },
            "name": name,
            "artist_ids": [artist_id],
            "album_id": None,
            "duration_ms": None,
            "genres": genres,
            "track_number": None,
            "disc_number": None,
            "release_date": None,
            "composer": None,
            "library_state": {
                "favorited": None,
                "disliked": None,
                "rating": None,
                "play_count": None,
                "skip_count": None,
                "added_to_library_at": None,
                "last_played_at": None,
            },
            "agent_metadata": {"tags": []},
        }

    def _seed_supply(self, extras: int, zeros: int) -> None:
        """Base fixture + A/B positives + ``extras`` Synthetic-Pop catalog-bound
        tracks (affinity overlap, the S3-S3C shape) + ``zeros`` zero-affinity
        tracks on a unique genre/artist (the live-failure shape)."""
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        if self.Z_ARTIST not in {artist["id"] for artist in fixture["artists"]}:
            fixture["artists"].append(
                {
                    "id": self.Z_ARTIST,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Deep Field Ensemble",
                }
            )
        for index in range(1, extras + 1):
            fixture["tracks"].append(
                self._catalog_track(
                    index,
                    f"Catalog Pop {index}",
                    ["Synthetic Pop"],
                    "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                )
            )
        for index in range(1, zeros + 1):
            fixture["tracks"].append(
                self._catalog_track(
                    index + 3,
                    f"Deep Field {index}",
                    ["Dark Ambient"],
                    self.Z_ARTIST,
                )
            )
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)

    def _inferred(self, fresh_ids=(), **extra: object) -> object:
        # P15 burn-down Issue 1: produced_at goes through the trusted
        # completed_at execution seam, never the payload.
        produced_at = extra.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 2, **extra},
            ),
            completed_at=produced_at or ISO,
            fresh_canonical_ids=tuple(fresh_ids),
        )

    def _decoded(self, result: object) -> list:
        return list(
            decode_recommendation_result(result.payload["encoded_result"]).items
        )

    def _targets(self, result: object) -> list[str]:
        return [item.candidate.target.target_id for item in self._decoded(result)]

    def _rows(self, result: object) -> list[tuple]:
        return [
            (item.candidate.target.target_id, item.score.total,
             item.candidate.source.source_path)
            for item in self._decoded(result)
        ]

    # --- the zero-affinity live-failure regression (regression B) -------------

    def test_zero_affinity_fresh_reaches_batch_via_channel(self) -> None:
        """The 22-promoted -> 0-fresh live failure, deterministically: promoted
        tracks with NO preference footprint anywhere. The frozen catalog layer
        omits them; the channel admits them as zero-basis discoveries and the
        floor delivers one, honestly (score 0, empty basis, no preference
        explanation)."""
        self._seed_supply(extras=0, zeros=2)
        result = self._inferred(
            (self.Z1, self.Z2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, self.Z1])
        self.assertEqual(result.payload["fresh_item_count"], 1)
        # Diagnostics: 2 promoted -> 2 fresh candidates emitted, 0 vetoed.
        self.assertEqual(result.payload["fresh_promoted_count"], 2)
        self.assertEqual(result.payload["fresh_candidate_count"], 2)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 0)
        decoded = {
            item.candidate.target.target_id: item for item in self._decoded(result)
        }
        fresh_item = decoded[self.Z1]
        self.assertEqual(fresh_item.score.total, 0.0)
        self.assertEqual(fresh_item.candidate.basis_targets, ())
        self.assertEqual(
            fresh_item.candidate.source.source_path, "fresh_driven"
        )
        row = {
            item["target_id"]: item for item in result.payload["items"]
        }[self.Z1]
        self.assertEqual(row["label"], "catalog")
        self.assertEqual(row["score_total"], 0.0)
        self.assertEqual(
            row["explanation"], "本次目录搜索的新发现（暂无偏好匹配证据）"
        )
        self.assertTrue(row["fresh_this_request"])

    def test_min_fresh_two_delivers_both_zero_basis_discoveries(self) -> None:
        self._seed_supply(extras=0, zeros=2)
        result = self._inferred(
            (self.Z1, self.Z2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=2,
        )
        self.assertEqual(self._targets(result), [self.Z1, self.Z2])
        self.assertEqual(result.payload["fresh_item_count"], 2)
        for item in self._decoded(result):
            self.assertEqual(item.score.total, 0.0)
            self.assertEqual(item.candidate.basis_targets, ())
            self.assertEqual(item.candidate.source.source_path, "fresh_driven")

    # --- negative veto, fail closed at every granularity ----------------------

    def test_negative_veto_rejects_fresh_and_splits_diagnostics(self) -> None:
        """An explicit dislike propagates a NEGATIVE genre/artist conclusion in
        the context; the channel fails closed for the whole zero-affinity set,
        reports the 2 rejections, and delivers no fresh item."""
        self._seed_supply(extras=0, zeros=2)
        self.seed_negative(self.Z1)
        result = self._inferred(
            (self.Z1, self.Z2),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertEqual(result.payload["fresh_item_count"], 0)
        self.assertEqual(result.payload["fresh_promoted_count"], 2)
        self.assertEqual(result.payload["fresh_candidate_count"], 0)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 2)

    def test_explicit_dislike_on_promoted_track_never_delivered(self) -> None:
        """The masked case: an explicit dislike on the promoted track whose
        genre is SHARED with positive supply (its genre reads ambiguous, not
        plainly negative). The fresh channel feeds the track's own direct
        state into the veto scan, so the disliked track is still rejected --
        fail closed even when the propagation surface cannot express it."""
        self._seed_supply(extras=1, zeros=0)
        self.seed_negative(self.EXTRA_1)
        result = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(self._targets(result), [TRACK_A, TRACK_B])
        self.assertNotIn(self.EXTRA_1, self._targets(result))
        self.assertEqual(result.payload["fresh_item_count"], 0)
        self.assertEqual(result.payload["fresh_promoted_count"], 1)
        self.assertEqual(result.payload["fresh_candidate_count"], 0)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 1)

    # --- the Request-2 positive-overlap regression (regression A) -------------

    def test_positive_overlap_keeps_stronger_catalog_candidate(self) -> None:
        """Promoted EXTRA_1 already carries a real 0.675 catalog candidate
        (positive affinity overlap). The channel emits a zero-basis twin; the
        existing dedupe keeps the stronger-evidence catalog candidate -- one
        EXTRA_1 in the batch, its catalog source and score intact, and Fresh
        identity stays membership-based (fresh_this_request=true) regardless
        of which source won."""
        self._seed_supply(extras=1, zeros=0)
        result = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["item_count"], 2)
        self.assertEqual(result.payload["fresh_item_count"], 1)
        self.assertEqual(result.payload["fresh_promoted_count"], 1)
        self.assertEqual(result.payload["fresh_candidate_count"], 1)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 0)
        decoded = {
            item.candidate.target.target_id: item for item in self._decoded(result)
        }
        kept = decoded[self.EXTRA_1]
        self.assertEqual(kept.score.total, 0.675)
        self.assertEqual(kept.candidate.source.source_path, "catalog_driven")
        self.assertTrue(kept.candidate.basis_targets)  # real evidence basis
        row = {
            item["target_id"]: item for item in result.payload["items"]
        }[self.EXTRA_1]
        self.assertTrue(row["fresh_this_request"])
        self.assertNotEqual(
            row["explanation"], "本次目录搜索的新发现（暂无偏好匹配证据）"
        )
        # No duplicate: exactly one EXTRA_1 across the batch.
        self.assertEqual(
            sum(
                1
                for item in result.payload["items"]
                if item["target_id"] == self.EXTRA_1
            ),
            1,
        )

    # --- explicit-intent activation gate (ordinary-run compatibility) ---------

    def test_kwarg_without_min_fresh_stays_byte_equivalent(self) -> None:
        """Hard acceptance item: the loop may supply fresh_canonical_ids on an
        ordinary run. Without min_fresh the channel must not activate --
        candidates, scores, ordering and result keys are identical."""
        self._seed_supply(extras=1, zeros=0)
        # P19-T15: both probes opt out of recent-run dedup -- byte-equivalence
        # here is about the channel gate, not cross-run novelty.
        plain = self._inferred(
            (),
            produced_at="2026-08-16T10:00:00+00:00",
            avoid_previous_runs=False,
        )
        with_kwarg = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            avoid_previous_runs=False,
        )
        self.assertEqual(plain.outcome, AgentToolOutcome.OK, plain.error_code)
        self.assertEqual(with_kwarg.outcome, AgentToolOutcome.OK, with_kwarg.error_code)
        self.assertEqual(self._rows(with_kwarg), self._rows(plain))
        self.assertEqual(
            set(with_kwarg.payload.keys()),
            {
                "run_id",
                "item_count",
                "fresh_item_count",
                "source_system",
                "items",
                "encoded_result",
            },
        )
        self.assertEqual(with_kwarg.payload["fresh_item_count"], 0)

    def test_explicit_zero_min_fresh_keeps_channel_off(self) -> None:
        self._seed_supply(extras=1, zeros=0)
        zero = self._inferred(
            (self.EXTRA_1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=0,
        )
        self.assertEqual(self._targets(zero), [TRACK_A, TRACK_B])
        keys = set(zero.payload.keys())
        self.assertNotIn("fresh_promoted_count", keys)
        self.assertNotIn("fresh_candidate_count", keys)
        self.assertNotIn("fresh_negative_rejected_count", keys)

    def test_min_fresh_with_empty_fresh_set_reports_zero_diagnostics(self) -> None:
        """Gate open but no authoritative promoted set: channel off, the S3D
        degrade-to-exploration semantics apply, and the additive diagnostics
        explain zeros ("this discover run did not newly promote anything")."""
        self._seed_supply(extras=1, zeros=0)
        result = self._inferred(
            (),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["fresh_item_count"], 0)
        self.assertEqual(result.payload["fresh_promoted_count"], 0)
        self.assertEqual(result.payload["fresh_candidate_count"], 0)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 0)

    def test_min_fresh_with_unknown_fresh_id_reports_zero_candidates(self) -> None:
        """A promoted id absent from the store contributes nothing (no
        fabrication) and the diagnostics show promoted=1 / candidates=0."""
        self._seed_supply(extras=1, zeros=0)
        result = self._inferred(
            (self.UNKNOWN_FRESH,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.EXTRA_1])
        self.assertEqual(result.payload["fresh_promoted_count"], 1)
        self.assertEqual(result.payload["fresh_candidate_count"], 0)
        self.assertEqual(result.payload["fresh_negative_rejected_count"], 0)

    # --- composition with the frozen floors -----------------------------------

    def test_fresh_driven_item_satisfies_fresh_and_exploration_from_one_place(
        self,
    ) -> None:
        """The S3-S3D composition rule extends to the new source: one zero-
        basis fresh entry satisfies BOTH the fresh floor and the exploration
        floor from its single place -- no catalog top-up displaces it."""
        self._seed_supply(extras=0, zeros=2)
        # P19-T15: both probes opt out of recent-run dedup -- the composition
        # rule under test is floor interplay, not cross-run novelty.
        result = self._inferred(
            (self.Z1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
            min_exploration=1,
            avoid_previous_runs=False,
        )
        self.assertEqual(self._targets(result), [TRACK_A, self.Z1])
        self.assertEqual(result.payload["fresh_item_count"], 1)
        # Above supply for the remaining floor: best effort, still one place.
        widened = self._inferred(
            (self.Z1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
            min_exploration=2,
            avoid_previous_runs=False,
        )
        self.assertEqual(self._targets(widened), [TRACK_A, self.Z1])

    # --- empty-envelope diagnostics --------------------------------------------

    def _empty_diagnostics(self, result: object) -> dict:
        self.assertEqual(result.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(result.error_code, "empty_recommendation")
        message = result.error_message
        marker = " diagnostics: "
        self.assertIn(marker, message)
        return json.loads(message.split(marker, 1)[1])

    def test_empty_envelope_carries_fresh_zeros_when_gate_open(self) -> None:
        """The direction filter empties every input; the envelope explains the
        open Fresh gate with truthful zeros (promoted=1 from the loop capture,
        candidates=0 because the pool is empty)."""
        self._seed_supply(extras=0, zeros=2)
        refused = self._inferred(
            (self.Z1,),
            produced_at="2026-08-16T10:00:00+00:00",
            min_fresh=1,
            genres=["Jazz"],
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertEqual(diagnostics["fresh_promoted_count"], 1)
        self.assertEqual(diagnostics["fresh_candidate_count"], 0)
        self.assertEqual(diagnostics["fresh_negative_rejected_count"], 0)

    def test_empty_envelope_without_min_fresh_omits_fresh_keys(self) -> None:
        """Byte-compatible envelope: without the explicit-intent gate the
        M2-2/S3A keys are exactly the pre-S3E set."""
        self._seed_supply(extras=0, zeros=2)
        refused = self._inferred(
            (self.Z1,),
            produced_at="2026-08-16T10:00:00+00:00",
            genres=["Jazz"],
        )
        diagnostics = self._empty_diagnostics(refused)
        self.assertNotIn("fresh_promoted_count", diagnostics)
        self.assertNotIn("fresh_candidate_count", diagnostics)
        self.assertNotIn("fresh_negative_rejected_count", diagnostics)


class P20Fix03ExplanationEvidenceProjectionTest(SharedAgentServiceTest):
    """P20-Fix03: get_recommendation_run delivers each item's DURABLE evidence
    (mechanism / resolved basis with provenance / honest weak-evidence note)
    so explanation turns can be grounded in what the run actually recorded.

    The projection only ever exposes the persisted primitives: candidate
    source_path through the frozen vocabulary, basis_targets resolved to
    display labels, and the run's own preference_inputs for provenance. It is
    pinned here for: complete coverage of a mixed-evidence 5-item batch,
    artist identity (verbatim canonical names, never LLM re-translation),
    the zero-basis honest note, mixed-direction summary data, scores staying
    machine fields, compactness relative to the raw encode, and fail-closed
    resolution of unresolvable basis references."""

    ARTIST_EEEE = "art_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    ARTIST_DEAD = "art_deaddead-dead-4ead-8ead-deaddeaddead"
    TRACK_EXTRA = "trk_aaaa0006-aaaa-4aaa-8aaa-000000000006"
    RUN_ID = "rcm_55555555-5555-4555-8555-555555555555"

    def _craft_mixed_run(self) -> None:
        from music_agent.preference_attribution import (
            DerivedPreference,
            InferredAffinity,
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.preference_strength import PreferenceState, PreferenceStrength
        from music_agent.recommendation_contract import (
            Candidate,
            CandidateSourceReference,
            PreferenceInput,
            RecommendationContext,
            RecommendationItem,
            RecommendationRequest,
            RecommendedItemKind,
            ScoreBreakdown,
            ScoreComponent,
            assemble_recommendation_result,
            encode_recommendation_result,
        )

        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.8)

        def input_(kind: str, tid: str, direct: bool) -> PreferenceInput:
            reference = PreferenceTargetReference(PreferenceTargetKind(kind), tid)
            if direct:
                return PreferenceInput.from_direct(DerivedPreference(reference, strength))
            return PreferenceInput.from_inferred(InferredAffinity(reference, strength))
        context = RecommendationContext(
            NOW,
            (
                input_("genre", "Synthetic Pop", False),
                input_("genre", "Rock", False),
                input_("genre", "J-Pop", False),
                input_("genre", "Indie Pop", False),
                input_("artist", "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", False),
                input_("artist", self.ARTIST_EEEE, True),
                input_("artist", self.ARTIST_EEEE, False),
                input_("track", TRACK_C, True),
            ),
        )
        request = RecommendationRequest(context, RecommendedItemKind.TRACK, 5)
        result = assemble_recommendation_result(
            request,
            (
                RecommendationItem(
                    Candidate(
                        candidate_id="cnd_00000000-0000-4000-8000-00000000000%d"
                        % (index + 1),
                        target=PreferenceTargetReference(
                            PreferenceTargetKind.TRACK, target_id
                        ),
                        source=CandidateSourceReference(
                            "recommendation_engine", source_path
                        ),
                        basis_targets=tuple(
                            PreferenceTargetReference(
                                PreferenceTargetKind(kind), ref_id
                            )
                            for kind, ref_id in basis
                        ),
                    ),
                    ScoreBreakdown(1.0, (ScoreComponent("base", 1.0),)),
                )
                for index, (target_id, source_path, basis) in enumerate(
                    (
                        # 1: catalog-driven on shared genre + artist refs.
                        (
                            TRACK_A,
                            "catalog_driven",
                            [
                                ("genre", "Synthetic Pop"),
                                ("artist", "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
                            ],
                        ),
                        # 2: catalog-driven on two inferred genres (J-Pop +
                        # Rock -- mixed even within one item, like the real
                        # rcm_e0a5e247 batch).
                        (
                            # TRACK_B (trk_bbbbbbbb) is the virtual P06-only
                            # id with no canonical track; the real Synthetic
                            # Solo lives under its fixture id.
                            "trk_22222222-2222-4222-8222-222222222222",
                            "catalog_driven",
                            [("genre", "J-Pop"), ("genre", "Rock")],
                        ),
                        # 3: preference-driven, direct track basis.
                        (TRACK_C, "preference_driven", [("track", TRACK_C)]),
                        # 4: catalog-driven on the Eileen Yo artist identity
                        # (direct takes precedence over the co-recorded
                        # inferred input) + an inferred genre.
                        (
                            self.TRACK_EXTRA,
                            "catalog_driven",
                            [("genre", "Indie Pop"), ("artist", self.ARTIST_EEEE)],
                        ),
                        # 5: fresh-driven, zero basis -- honest note only.
                        (
                            "trk_44444444-4444-4444-8444-444444444444",
                            "fresh_driven",
                            [],
                        ),
                    )
                )
            ),
            run_id=self.RUN_ID,
            produced_at=NOW,
        )
        self.service._recommendation_history._connection.execute(
            """INSERT INTO recommendation_runs(
                run_id, encoded_result, contract_version, produced_at, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                result.run_id,
                encode_recommendation_result(result),
                result.contract_version,
                result.produced_at.isoformat(),
                ISO,
            ),
        )

    def _detail(self) -> dict:
        fetched = self.service.execute(
            make_request(
                tool="get_recommendation_run",
                payload={"run_id": self.RUN_ID},
            ),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        return fetched.payload

    def _seed_extra_identity(self) -> None:
        extra_track = search_track(
            self.TRACK_EXTRA,
            "Sleepless Wander",
            self.ARTIST_EEEE,
        )
        extra_track["genres"] = ["Indie Pop"]
        self.seed_search_fixture(
            extra_artists=[
                {
                    "id": self.ARTIST_EEEE,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Eileen Yo",
                }
            ],
            extra_tracks=[extra_track],
        )

    def test_mixed_evidence_five_item_batch_projects_completely(self) -> None:
        """Group 1: a mixed (catalog/preference/fresh, genre/artist/track basis,
        direct/inferred provenance) five-item batch projects ALL five items in
        order, each with its exact mechanism, resolved basis and provenance."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        self.assertEqual(payload["item_count"], 5)
        items = payload["items"]
        self.assertEqual([item["position"] for item in items], [1, 2, 3, 4, 5])
        by_position = {item["position"]: item for item in items}
        first = by_position[1]
        self.assertEqual(first["evidence"]["mechanism"], "目录推断")
        self.assertEqual(
            first["evidence"]["basis"],
            [
                {"kind": "genre", "label": "Synthetic Pop", "provenance": "推断"},
                {"kind": "artist", "label": "Artist Alpha", "provenance": "推断"},
            ],
        )
        second = by_position[2]["evidence"]
        self.assertEqual(second["mechanism"], "目录推断")
        self.assertEqual(
            second["basis"],
            [
                {"kind": "genre", "label": "J-Pop", "provenance": "推断"},
                {"kind": "genre", "label": "Rock", "provenance": "推断"},
            ],
        )
        third = by_position[3]["evidence"]
        self.assertEqual(third["mechanism"], "直接偏好")
        self.assertEqual(
            third["basis"],
            [{"kind": "track", "label": "Albumless Study", "provenance": "直接"}],
        )
        fourth = by_position[4]["evidence"]
        self.assertEqual(fourth["mechanism"], "目录推断")
        self.assertEqual(
            fourth["basis"],
            [
                {"kind": "genre", "label": "Indie Pop", "provenance": "推断"},
                # Direct + inferred inputs co-recorded: direct wins.
                {"kind": "artist", "label": "Eileen Yo", "provenance": "直接"},
            ],
        )
        fifth = by_position[5]
        self.assertEqual(fifth["evidence"]["mechanism"], "探索性新发现")
        self.assertEqual(fifth["evidence"]["basis"], [])
        self.assertEqual(
            fifth["evidence"]["note"], "本次目录搜索的新发现（暂无偏好匹配证据）"
        )
        # Complete coverage: every item carries its evidence block.
        for item in items:
            self.assertIn("evidence", item)

    def test_artist_identity_never_translated(self) -> None:
        """Group 2: the artist label comes from the canonical entity verbatim
        (Eileen Yo stays Eileen Yo) and no re-translated identity materializes
        anywhere in the projection."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        text = json.dumps(dict(payload), ensure_ascii=False)
        self.assertIn("Eileen Yo", text)
        self.assertNotIn("游鸿明", text)
        # Verbatim artist_name on the item and as the basis label.
        items = {item["position"]: item for item in payload["items"]}
        self.assertEqual(items[4]["artist_name"], "Eileen Yo")
        self.assertEqual(
            [row["label"] for row in items[4]["evidence"]["basis"] if row["kind"] == "artist"],
            ["Eileen Yo"],
        )

    def test_zero_basis_item_gets_the_honest_note_only(self) -> None:
        """Group 3: an item with no surviving basis -- fresh-driven zero-basis
        here, and (below) basis dropped fail-closed -- projects NO evidence
        claims beyond the honest 暂无偏好匹配证据 wording: mechanism, empty
        basis list, and the exact note. No invented style/mood/background."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        fifth = {item["position"]: item for item in payload["items"]}[5]
        self.assertEqual(
            fifth["evidence"],
            {
                "mechanism": "探索性新发现",
                "basis": [],
                "note": "本次目录搜索的新发现（暂无偏好匹配证据）",
            },
        )
        self.assertNotIn("据你对", json.dumps(fifth, ensure_ascii=False))

    def test_mixed_direction_batch_exposes_both_directions(self) -> None:
        """Group 4: the projection keeps per-item direction data so a batch
        summary can honestly list mixed directions (J-Pop 与 Rock -- never a
        forced single direction) and can never be written as 全部 J-Pop."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        text = json.dumps(dict(payload), ensure_ascii=False)
        self.assertIn("J-Pop", text)
        self.assertIn("Rock", text)
        # The three candidate mechanisms all appear: a summary over 共同证据
        # is structurally impossible; mixed listing is the only honest form.
        self.assertEqual(
            {item["evidence"]["mechanism"] for item in payload["items"]},
            {"目录推断", "直接偏好", "探索性新发现"},
        )

    def test_scores_stay_machine_fields_never_reasons(self) -> None:
        """Group 5: score_total survives as a numeric ranking field, and the
        evidence block carries NO score-derived suitability claim: no 满分 /
        100% / 高度吻合 wording can come out of this projection."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        text = json.dumps(dict(payload), ensure_ascii=False)
        for item in payload["items"]:
            self.assertEqual(item["score_total"], 1.0)
            self.assertTrue(
                set(item["evidence"].keys()) <= {"mechanism", "basis", "note"}
            )
        for banned in ("满分", "100% ", "高度吻合", "契合度"):
            self.assertNotIn(banned, text)

    def test_projection_is_compact_but_complete(self) -> None:
        """Group 6: the whole detail projection (all five items plus
        evidence) is smaller than the raw encoded_result it summarizes,
        while every item name survives."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        # The MODEL-VISIBLE head (items + evidence, encoded_result tail
        # excluded -- the provider already drops that tail, and re-escaping
        # it would double-count) is smaller than the raw durable encode.
        head = {key: value for key, value in dict(payload).items() if key != "encoded_result"}
        projected = json.dumps(head, ensure_ascii=False)
        self.assertLess(len(projected), len(payload["encoded_result"]))
        for name in ("Synthetic Duet", "Synthetic Solo", "Albumless Study",
                     "Sleepless Wander", "Year Precision"):
            self.assertIn(name, projected)

    def test_encoded_result_roundtrips_and_list_reader_stays_identity_only(self) -> None:
        """The detail reader keeps the durable encode decodable (contract
        interchange untouched) while list_recommendation_runs stays the
        identity-only entry: no evidence and no position keys leak into the
        token-cheap list surface."""
        self._seed_extra_identity()
        self._craft_mixed_run()
        payload = self._detail()
        decoded = decode_recommendation_result(payload["encoded_result"])
        self.assertEqual(decoded.run_id, self.RUN_ID)
        self.assertEqual(len(decoded.items), 5)
        listed = self.service.execute(
            make_request(tool="list_recommendation_runs"), completed_at=ISO
        )
        self.assertEqual(listed.outcome, AgentToolOutcome.OK)
        self.assertEqual(len(listed.payload["runs"]), 1)
        list_item = listed.payload["runs"][0]["items"][0]
        self.assertNotIn("evidence", list_item)
        self.assertNotIn("position", list_item)
        self.assertEqual(
            set(list_item.keys()),
            {
                "candidate_id", "target_kind", "target_id", "name",
                "artist_name", "score_total", "playback",
            },
        )

    def test_unresolvable_basis_is_dropped_fail_closed(self) -> None:
        """An item whose basis references have no matching preference input or
        no canonical display entity projects an EMPTY basis with the honest
        note -- evidence shrinks, it never guesses, and the dead references
        never leak into the output."""
        from music_agent.preference_attribution import (
            DerivedPreference,
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.preference_strength import PreferenceState, PreferenceStrength
        from music_agent.recommendation_contract import (
            Candidate,
            CandidateSourceReference,
            PreferenceInput,
            RecommendationContext,
            RecommendationItem,
            RecommendationRequest,
            RecommendedItemKind,
            ScoreBreakdown,
            ScoreComponent,
            assemble_recommendation_result,
            encode_recommendation_result,
        )

        strength = PreferenceStrength(PreferenceState.POSITIVE, 0.8)
        # One input for a target NO basis row references: every basis row below
        # has no matching preference input or canonical entity and must die.
        context = RecommendationContext(
            NOW,
            (
                PreferenceInput.from_direct(
                    DerivedPreference(
                        PreferenceTargetReference(
                            PreferenceTargetKind.GENRE, "Unrelated"
                        ),
                        strength,
                    )
                ),
            ),
        )
        request = RecommendationRequest(context, RecommendedItemKind.TRACK, 1)
        result = assemble_recommendation_result(
            request,
            (
                RecommendationItem(
                    Candidate(
                        candidate_id="cnd_00000000-0000-4000-8000-000000000099",
                        target=PreferenceTargetReference(
                            PreferenceTargetKind.TRACK, TRACK_A
                        ),
                        source=CandidateSourceReference(
                            "recommendation_engine", "catalog_driven"
                        ),
                        basis_targets=(
                            PreferenceTargetReference(
                                PreferenceTargetKind.GENRE, "Gone"
                            ),
                            PreferenceTargetReference(
                                PreferenceTargetKind.GENRE, "Indie Pop"
                            ),
                            PreferenceTargetReference(
                                PreferenceTargetKind.ARTIST, self.ARTIST_DEAD
                            ),
                            PreferenceTargetReference(
                                PreferenceTargetKind.TRACK, "trk_deaddead-dead-4ead-8ead-deaddeaddead"
                            ),
                        ),
                    ),
                    ScoreBreakdown(0.9, (ScoreComponent("base", 0.9),)),
                ),
            ),
            run_id="rcm_55555555-5555-4555-8555-999999999999",
            produced_at=NOW,
        )
        self.service._recommendation_history._connection.execute(
            """INSERT INTO recommendation_runs(
                run_id, encoded_result, contract_version, produced_at, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                result.run_id,
                encode_recommendation_result(result),
                result.contract_version,
                result.produced_at.isoformat(),
                ISO,
            ),
        )
        fetched = self.service.execute(
            make_request(
                tool="get_recommendation_run",
                payload={"run_id": "rcm_55555555-5555-4555-8555-999999999999"},
            ),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        item = fetched.payload["items"][0]
        # All four basis rows died: no-input genre, no-input genre, unknown
        # artist id, unknown track id. Zero claims, honest note.
        self.assertEqual(item["evidence"]["basis"], [])
        self.assertEqual(
            item["evidence"]["note"], "本次目录搜索的新发现（暂无偏好匹配证据）"
        )
        self.assertEqual(item["evidence"]["mechanism"], "目录推断")
        # The no-leak claim is about the EVIDENCE projection: the durable
        # encoded_result legitimately records what was submitted (it is the
        # record, never censored), so the assertion covers the model-visible
        # head -- items + evidence -- where a dead reference must never
        # surface.
        head = {key: value for key, value in dict(fetched.payload).items() if key != "encoded_result"}
        text = json.dumps(head, ensure_ascii=False)
        self.assertNotIn(self.ARTIST_DEAD, text)
        self.assertNotIn("trk_deaddead", text)
        self.assertNotIn("Indie Pop", text)


class P20Fix09GenerationEvidenceTest(SharedAgentServiceTest):
    """P20-Fix09: the FIRST presentation of a fresh batch must be grounded in
    the same durable evidence the follow-up explanation reads. Both generation
    channels now project each item's Fix03-shaped evidence block through the
    SHARED builder the run detail reader uses, resolved from the same
    primitives -- so first view and "为什么推荐这些？" may differ in detail but
    never in fact level. Locks here: direct items carry direct provenance only;
    catalog/inferred items carry inferred provenance only (the Soundtrack UAT
    shape: no invented 直接 claim); a novel item (direct state unknown) whose
    directional evidence is a track-level inferred positive projects
    「推断偏好」 -- never 直接偏好, and the context's non-directional direct
    snapshot never licenses 直接 provenance; zero-basis fresh items carry the
    honest note; a mixed batch resolves per item; and for every channel the
    generation-time evidence equals the persisted-run reader projection
    item-for-item."""

    Z1 = "trk_ffffffff-ffff-4fff-8fff-000000000004"
    Z2 = "trk_ffffffff-ffff-4fff-8fff-000000000005"
    Z_ARTIST = "art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"

    @staticmethod
    def _deep_field_track(
        track_id: str, name: str, genres: list, store_id: str
    ) -> dict:
        """A catalog-bound Deep Field track (own artist/genres, no direct
        preference state of its own), mirroring the live discovered-promoted
        shape."""
        return {
            "id": track_id,
            "external_ids": {
                "apple_music_persistent_id": None,
                "itunes_store_id": store_id,
            },
            "name": name,
            "artist_ids": [P20Fix09GenerationEvidenceTest.Z_ARTIST],
            "album_id": None,
            "duration_ms": None,
            "genres": genres,
            "track_number": None,
            "disc_number": None,
            "release_date": None,
            "composer": None,
            "library_state": {
                "favorited": None,
                "disliked": None,
                "rating": None,
                "play_count": None,
                "skip_count": None,
                "added_to_library_at": None,
                "last_played_at": None,
            },
            "agent_metadata": {"tags": []},
        }

    def _seed_extra_tracks(self, extra_tracks: list) -> None:
        """Base catalog supply plus caller-built Deep Field tracks."""
        self.seed_catalog_supply()
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        if self.Z_ARTIST not in {artist["id"] for artist in fixture["artists"]}:
            fixture["artists"].append(
                {
                    "id": self.Z_ARTIST,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": "Deep Field Ensemble",
                }
            )
        fixture["tracks"].extend(extra_tracks)
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def _seed_zero_affinity(self) -> None:
        """Base catalog supply (positives on A/B + Synthetic-Pop extras) plus
        one zero-affinity catalog track (unique genre/artist) -- the live
        shape behind a fresh-driven zero-basis item."""
        self._seed_extra_tracks(
            [
                self._deep_field_track(
                    self.Z1, "Deep Field 1", ["Dark Ambient"], "FIX09-Z1"
                )
            ]
        )

    def _seed_affinity_novel(self) -> None:
        """Base catalog supply plus one affinity-bearing novel target (a genre
        shared with the direct positives, but no direct state of its own) --
        the live Fix09-UAT shape behind an inferred track-level item whose
        only directional evidence is the inferred positive."""
        self._seed_extra_tracks(
            [
                self._deep_field_track(
                    self.Z2, "Affinity Probe", ["Synthetic Pop"], "FIX09-Z2"
                )
            ]
        )

    def _plain(self, **extra: object) -> object:
        produced_at = extra.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 2, **extra},
            ),
            completed_at=produced_at or ISO,
        )

    def _inferred(self, fresh_ids=(), **extra: object) -> object:
        produced_at = extra.pop("produced_at", None)
        return self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={"target_ids": [TRACK_A, TRACK_B], "limit": 2, **extra},
            ),
            completed_at=produced_at or ISO,
            fresh_canonical_ids=tuple(fresh_ids),
        )

    def _reader_evidence(self, run_id: str) -> list:
        fetched = self.service.execute(
            make_request(tool="get_recommendation_run", payload={"run_id": run_id}),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        return [item["evidence"] for item in fetched.payload["items"]]

    def test_plain_direct_items_carry_direct_evidence(self) -> None:
        """Plain items are the positive direct states themselves: mechanism
        「直接偏好」 with a direct track basis -- the only case entitled to
        「直接命中你收藏的曲目」一类表述 -- and the plain item still exposes no
        score (score stays out of the first presentation entirely)."""
        self.seed_catalog_supply()
        result = self._plain(avoid_previous_runs=False)
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        items = result.payload["items"]
        self.assertEqual([item["target_id"] for item in items], [TRACK_A, TRACK_B])
        first = items[0]
        self.assertEqual(
            first["evidence"],
            {
                "mechanism": "直接偏好",
                "basis": [
                    {"kind": "track", "label": "Synthetic Duet", "provenance": "直接"}
                ],
            },
        )
        self.assertNotIn("note", first["evidence"])
        self.assertNotIn("score_total", first)

    def test_inferred_catalog_item_evidence_is_inferred_only(self) -> None:
        """The Soundtrack-shaped claim boundary: a catalog candidate's basis is
        recorded inference (genre/artist affinity) -- every provenance reads
        「推断」, so the first view may only say 「按 X 方向推断出来的」, never a
        直接/收藏命中 claim."""
        self.seed_catalog_supply()
        result = self._inferred(
            min_exploration=1, avoid_previous_runs=False,
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        items = result.payload["items"]
        self.assertEqual([item["target_id"] for item in items], [TRACK_A, self.EXTRA_1])
        catalog_evidence = items[1]["evidence"]
        self.assertEqual(catalog_evidence["mechanism"], "目录推断")
        self.assertTrue(catalog_evidence["basis"])
        self.assertEqual(
            catalog_evidence["basis"],
            [
                {"kind": "artist", "label": "Artist Alpha", "provenance": "推断"},
                {"kind": "genre", "label": "Synthetic Pop", "provenance": "推断"},
            ],
        )
        # The fixture dict pins it; the loop guards against future drift.
        for basis in catalog_evidence["basis"]:
            self.assertNotEqual(basis["provenance"], "直接")

    def test_inferred_similarity_seed_exclusion_returns_only_catalog_candidates(
        self,
    ) -> None:
        """The seed may provide affinity but never occupy a slot, even on repeat."""
        self.seed_catalog_supply()
        payload = {
            "target_ids": [TRACK_A],
            "exclude_target_ids": [TRACK_A],
            "avoid_previous_runs": True,
            "limit": 5,
        }
        result = self.service.execute(
            make_request(tool="generate_inferred_recommendation", payload=payload),
            completed_at=ISO,
        )

        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_message)
        targets = [item["target_id"] for item in result.payload["items"]]
        self.assertTrue(targets)
        self.assertNotIn(TRACK_A, targets)
        self.assertTrue(set(targets).issubset({self.EXTRA_1, self.EXTRA_2, self.EXTRA_3}))
        encoded = json.loads(result.payload["encoded_result"])
        self.assertEqual(encoded["request"]["limit"], 5)

        repeated = self.service.execute(
            make_request(tool="generate_inferred_recommendation", payload=payload),
            completed_at="2026-08-16T00:01:00+00:00",
        )
        self.assertEqual(repeated.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(repeated.error_code, "empty_recommendation")
        history = self.service.execute(
            make_request(tool="list_recommendation_runs", payload={}),
            completed_at="2026-08-16T00:01:01+00:00",
        )
        self.assertEqual(len(history.payload["runs"]), 1)
        self.assertNotIn(
            TRACK_A,
            [item["target_id"] for item in history.payload["runs"][0]["items"]],
        )

    def test_fresh_zero_basis_item_carries_honest_note(self) -> None:
        """Zero-affinity fresh discovery: mechanism 「探索性新发现」, empty
        basis, honest note -- the only permitted first-view wording."""
        self._seed_zero_affinity()
        result = self._inferred(
            (self.Z1,),
            min_fresh=1,
            avoid_previous_runs=False,
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        fresh_items = [
            item for item in result.payload["items"] if item["fresh_this_request"]
        ]
        self.assertEqual(len(fresh_items), 1)
        self.assertEqual(
            fresh_items[0]["evidence"],
            {
                "mechanism": "探索性新发现",
                "basis": [],
                "note": "本次目录搜索的新发现（暂无偏好匹配证据）",
            },
        )

    def test_mixed_batch_resolves_evidence_per_item(self) -> None:
        """One batch, three fact levels: direct (A), fresh zero-basis note
        (Z1). Each item's block must match its own recorded facts -- no
        batch-wide borrow."""
        self._seed_zero_affinity()
        result = self._inferred(
            (self.Z1,),
            min_fresh=1,
            avoid_previous_runs=False,
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        by_target = {item["target_id"]: item for item in result.payload["items"]}
        self.assertEqual(
            by_target[TRACK_A]["evidence"]["mechanism"], "直接偏好"
        )
        self.assertEqual(by_target[TRACK_A]["evidence"]["basis"][0]["provenance"], "直接")
        self.assertEqual(
            by_target[self.Z1]["evidence"],
            {
                "mechanism": "探索性新发现",
                "basis": [],
                "note": "本次目录搜索的新发现（暂无偏好匹配证据）",
            },
        )

    def test_novel_track_level_inference_never_projects_direct(self) -> None:
        """THE Fix09 live-UAT correction: a novel item (direct state unknown,
        track-level inferred positive via shared genre affinity) is admitted
        by the preference-driven generator, but its directional evidence is
        INFERRED only -- the context's non-directional direct snapshot must
        never license 「直接」 provenance or a 「直接偏好」 mechanism. The
        same batch's direct item keeps its entitlement, and generation equals
        the reader projection item-for-item."""
        self._seed_affinity_novel()
        result = self.service.execute(
            make_request(
                tool="generate_inferred_recommendation",
                payload={
                    "target_ids": [TRACK_A, TRACK_B, self.Z2],
                    "limit": 4,
                    "avoid_previous_runs": False,
                },
            ),
            completed_at="2026-08-16T10:30:00+00:00",
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        items = {item["target_id"]: item for item in result.payload["items"]}
        self.assertIn(self.Z2, items)
        novel = items[self.Z2]
        self.assertEqual(novel["label"], "novel")
        self.assertEqual(novel["direct_state"], "unknown")
        self.assertEqual(
            novel["evidence"],
            {
                "mechanism": "推断偏好",
                "basis": [
                    {"kind": "track", "label": "Affinity Probe", "provenance": "推断"}
                ],
            },
        )
        # No direct claim may survive anywhere on the novel item's surface.
        self.assertNotIn(
            "直接", json.dumps(novel["evidence"], ensure_ascii=False)
        )
        # The known-positive item in the same batch stays entitled to direct
        # wording: the correction is per-evidence, not a batch-wide downgrade.
        delivered = {item["target_id"] for item in result.payload["items"]}
        self.assertIn(TRACK_A, delivered)
        self.assertEqual(
            items[TRACK_A]["evidence"]["basis"][0]["provenance"], "直接"
        )
        self.assertEqual(
            [item["evidence"] for item in result.payload["items"]],
            self._reader_evidence(result.payload["run_id"]),
        )

    def test_generation_evidence_equals_reader_evidence_on_same_run(self) -> None:
        """THE Fix09 contract: for every channel, the generation result's
        per-item evidence is IDENTICAL (deep-equal, in order) to what
        get_recommendation_run would later project from the persisted run --
        first view and follow-up explanation read one fact source."""
        self._seed_zero_affinity()
        plain = self._plain(avoid_previous_runs=False)
        self.assertEqual(
            [item["evidence"] for item in plain.payload["items"]],
            self._reader_evidence(plain.payload["run_id"]),
        )
        inferred = self._inferred(
            avoid_previous_runs=False,
            produced_at="2026-08-16T10:00:00+00:00",
        )
        self.assertEqual(
            [item["evidence"] for item in inferred.payload["items"]],
            self._reader_evidence(inferred.payload["run_id"]),
        )
        fresh = self._inferred(
            (self.Z1,),
            min_fresh=1,
            avoid_previous_runs=False,
            produced_at="2026-08-16T10:15:00+00:00",
        )
        self.assertEqual(
            [item["evidence"] for item in fresh.payload["items"]],
            self._reader_evidence(fresh.payload["run_id"]),
        )

    def test_inferred_legacy_view_keys_stay_untouched(self) -> None:
        """Fix09 adds ``evidence`` additively; the S3D/S3E legacy view keys
        (label / explanation / direct_state / score_total) are not disturbed."""
        self._seed_zero_affinity()
        result = self._inferred(
            (self.Z1,),
            min_fresh=1,
            avoid_previous_runs=False,
            produced_at="2026-08-16T10:00:00+00:00",
        )
        for item in result.payload["items"]:
            self.assertIn("label", item)
            self.assertIn("explanation", item)
            self.assertIn("direct_state", item)
            self.assertIn("score_total", item)


class SiblingSuppressionIntegrationTest(SharedAgentServiceTest):
    """P20 Fix07 service-level: same-batch sibling suppression at the final
    delivery layer of BOTH generation channels (sec.8-12).

    Seeds the UAT duplicate shape into the canonical world: two catalog-bound
    tracks with the verbatim title "Sibling Song", one canonical artist, and
    only the album container differing (album vs single) -- the field shape of
    the Ano yume wo nazotte duplicate pair. Selection semantics, backfill,
    durable run records, Fresh truth, exclusions, and direction filtering are
    exercised end-to-end over the real handler paths.
    """

    ALPHA = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    SIB_ALBUM = "alb_00000000-0000-4000-8000-aaaaaaaaaaa1"
    SIB_SINGLE = "alb_00000000-0000-4000-8000-aaaaaaaaaaa2"
    SIB_A = "trk_eeeeeeee-eeee-4eee-8eee-000000000001"
    SIB_B = "trk_eeeeeeee-eeee-4eee-8eee-000000000002"
    RESERVE_1 = "trk_eeeeeeee-eeee-4eee-8eee-000000000003"
    RESERVE_2 = "trk_eeeeeeee-eeee-4eee-8eee-000000000004"

    def _extra_track(
        self, track_id: str, name: str, album_id: str | None, store_id: str
    ) -> dict:
        return {
            "id": track_id,
            "external_ids": {
                "apple_music_persistent_id": None,
                "itunes_store_id": store_id,
            },
            "name": name,
            "artist_ids": [self.ALPHA],
            "album_id": album_id,
            "duration_ms": None,
            "genres": ["Synthetic Pop"],
            "track_number": None,
            "disc_number": None,
            "release_date": None,
            "composer": None,
            "library_state": {
                "favorited": None,
                "disliked": None,
                "rating": None,
                "play_count": None,
                "skip_count": None,
                "added_to_library_at": None,
                "last_played_at": None,
            },
            "agent_metadata": {"tags": []},
        }

    def seed_sibling_supply(self, reserves: int = 2) -> None:
        """Base fixture + positives on A/B + the sibling pair (+ ``reserves``
        distinct catalog-bound tracks) over Alpha/Synthetic Pop affinity."""
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        fixture["albums"].extend(
            [
                {
                    "id": self.SIB_ALBUM,
                    "external_ids": {"apple_music_persistent_id": "SIB-ALBUM"},
                    "name": "Sibling Album",
                    "artist_ids": [self.ALPHA],
                    "release_date": None,
                },
                {
                    "id": self.SIB_SINGLE,
                    "external_ids": {"apple_music_persistent_id": "SIB-SINGLE"},
                    "name": "Sibling Song - Single",
                    "artist_ids": [self.ALPHA],
                    "release_date": None,
                },
            ]
        )
        extras = [
            self._extra_track(self.SIB_A, "Sibling Song", self.SIB_ALBUM, "SIB-A"),
            self._extra_track(self.SIB_B, "Sibling Song", self.SIB_SINGLE, "SIB-B"),
        ]
        if reserves >= 1:
            extras.append(
                self._extra_track(self.RESERVE_1, "Reserve Pop 1", None, "SIB-R1")
            )
        if reserves >= 2:
            extras.append(
                self._extra_track(self.RESERVE_2, "Reserve Pop 2", None, "SIB-R2")
            )
        fixture["tracks"].extend(extras)
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.seed_positive(TRACK_A)
        self.seed_positive(TRACK_B)

    def _inferred(self, *, limit: int, fresh_ids=(), **extra: object) -> object:
        payload: dict = {"target_ids": [TRACK_A, TRACK_B], "limit": limit, **extra}
        return self.service.execute(
            make_request(tool="generate_inferred_recommendation", payload=payload),
            completed_at="2026-08-16T10:00:00+00:00",
            fresh_canonical_ids=tuple(fresh_ids),
        )

    def _targets(self, result: object) -> list[str]:
        return [item["target_id"] for item in result.payload["items"]]

    def test_inferred_batch_suppresses_sibling_and_backfills_to_limit(self) -> None:
        self.seed_sibling_supply()
        result = self._inferred(limit=5)
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        # Rank [A, B, SIB_A, SIB_B, R1] with the UAT pair adjacent; the later
        # sibling is skipped and the reserves backfill the batch to the limit.
        self.assertEqual(
            self._targets(result),
            [TRACK_A, TRACK_B, self.SIB_A, self.RESERVE_1, self.RESERVE_2],
        )
        self.assertEqual(result.payload["item_count"], 5)
        self.assertEqual(len(set(self._targets(result))), 5)

    def test_exhausted_pool_returns_fewer_items_honestly(self) -> None:
        # No reserves: after suppression the pool holds only three distinct
        # works, so the batch honestly returns 3 < limit 5 (sec.10 -- never a
        # second supply for backfill).
        self.seed_sibling_supply(reserves=0)
        result = self._inferred(limit=5)
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(
            self._targets(result), [TRACK_A, TRACK_B, self.SIB_A]
        )
        self.assertNotIn(self.SIB_B, self._targets(result))
        self.assertEqual(result.payload["item_count"], 3)

    def test_plain_channel_suppresses_sibling_and_backfills(self) -> None:
        # The direct generate_recommendation channel dedupes the same way: all
        # three inputs carry positive evidence; the sibling pair collapses and
        # the reserve backfills the second place.
        self.seed_sibling_supply(reserves=1)
        self.seed_positive(self.SIB_A)
        self.seed_positive(self.SIB_B)
        self.seed_positive(self.RESERVE_1)
        result = self.service.execute(
            make_request(
                tool="generate_recommendation",
                payload={
                    "target_ids": [self.SIB_A, self.SIB_B, self.RESERVE_1],
                    "limit": 2,
                    "source_system": "apple_music",
                },
            ),
            completed_at=ISO,
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        # Equal evidence everywhere -> target-id tie-break ranks SIB_A before
        # SIB_B; the first occurrence survives, the sibling is skipped, and the
        # reserve backfills the freed place.
        self.assertEqual(self._targets(result), [self.SIB_A, self.RESERVE_1])
        self.assertEqual(result.payload["item_count"], 2)
        self.assertEqual(result.payload["fresh_item_count"], 0)

    def test_durable_run_records_only_the_selected_items(self) -> None:
        # The suppressed sibling never reaches history, the encoded result, or
        # the active batch pointer (sec.12: only actually-delivered items are
        # durably recorded).
        self.seed_sibling_supply()
        result = self._inferred(limit=5)
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        run_id = result.payload["run_id"]
        fetched = self.service.execute(
            make_request(
                tool="get_recommendation_run", payload={"run_id": run_id}
            ),
            completed_at=ISO,
        )
        self.assertEqual(fetched.outcome, AgentToolOutcome.OK)
        durable_ids = [item["target_id"] for item in fetched.payload["items"]]
        self.assertEqual(durable_ids, self._targets(result))
        self.assertNotIn(self.SIB_B, durable_ids)
        decoded = decode_recommendation_result(result.payload["encoded_result"])
        self.assertEqual(
            [item.candidate.target.target_id for item in decoded.items],
            self._targets(result),
        )
        pointer = self.service.execute(
            make_request(tool="get_active_context"), completed_at=ISO
        )
        self.assertEqual(pointer.payload["active_batch"]["run_id"], run_id)

    def test_fresh_sibling_pair_stays_distinct_and_honest(self) -> None:
        # Both siblings are this-turn Fresh (sec.10): the floor places them,
        # suppression keeps the first, and the batch backfills from the same
        # pool -- fresh_this_request reports only what is truly delivered.
        self.seed_sibling_supply()
        result = self._inferred(
            limit=5, min_fresh=2, fresh_ids=(self.SIB_A, self.SIB_B)
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        targets = self._targets(result)
        self.assertNotIn(self.SIB_B, targets)
        self.assertIn(self.SIB_A, targets)
        self.assertEqual(len(targets), 5)
        self.assertEqual(len(set(targets)), 5)
        flags = {
            item["target_id"]: item["fresh_this_request"]
            for item in result.payload["items"]
        }
        self.assertTrue(flags[self.SIB_A])
        self.assertEqual(result.payload["fresh_item_count"], 1)
        # sec.10 boundary: the sibling backfill replenished the batch from the
        # turn's own pool -- the P09 journal proves no discovery (or any other
        # supply) tool call was involved in this delivery.
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            entries = journal.list()
        self.assertEqual(
            [entry.request.tool for entry in entries],
            ["generate_inferred_recommendation"],
        )

    def test_exclusions_coexist_with_sibling_suppression(self) -> None:
        # Excluding one sibling leaves the other as a single work -- the
        # suppression layer must not fabricate a duplicate out of thin air,
        # and the canonical-id exclusion semantics stay intact.
        self.seed_sibling_supply()
        result = self._inferred(limit=5, exclude_target_ids=[self.SIB_A])
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(
            self._targets(result),
            [TRACK_A, TRACK_B, self.SIB_B, self.RESERVE_1, self.RESERVE_2],
        )
        self.assertNotIn(self.SIB_A, self._targets(result))
        self.assertEqual(len(set(self._targets(result))), 5)

    def test_direction_filter_coexists_with_sibling_suppression(self) -> None:
        # The direction filter drops B (no Synthetic Pop genre) before ranking;
        # suppression then collapses the sibling pair and the reserves fill the
        # batch -- both controls compose without touching each other.
        self.seed_sibling_supply()
        result = self._inferred(limit=5, genres=["Synthetic Pop"])
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_code)
        self.assertEqual(
            self._targets(result),
            [TRACK_A, self.SIB_A, self.RESERVE_1, self.RESERVE_2],
        )
        self.assertNotIn(self.SIB_B, self._targets(result))
        self.assertNotIn(TRACK_B, self._targets(result))


if __name__ == "__main__":
    unittest.main()

class RecommendationExecutionExtractionOwnershipTest(unittest.TestCase):
    def test_shared_service_keeps_dispatch_authority_and_delegates_recommendation_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "music-agent.db"
            with SharedAgentService(database_path, clients=registry()) as service:
                self.assertIsInstance(
                    service._recommendation_execution, RecommendationExecutionService
                )
                handler = service._handlers[AgentToolName.GENERATE_RECOMMENDATION]
                self.assertIs(handler.__self__, service)
                self.assertIs(
                    service._recommendation_execution._recommendation_history,
                    service._recommendation_history,
                )
