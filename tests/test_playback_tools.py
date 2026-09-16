"""P10.12: Playback-control tool tests (deterministic fakes; no real Music.app)."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.playback_control import (
    MusicPlaybackAdapter,
    OsascriptPlaybackRunner,
    PAUSE_SCRIPT,
    PLAY_SCRIPT,
    NEXT_TRACK_SCRIPT,
    PREVIOUS_TRACK_SCRIPT,
    PLAY_TRACK_SCRIPT,
)
from music_agent.repository import CanonicalRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_99999999-9999-4999-8999-999999999999"
BOUND_TRACK = "trk_11111111-1111-4111-8111-111111111111"  # SYNTH-TRACK-001
UNBOUND_TRACK = "trk_33333333-3333-4333-8333-333333333333"  # no persistent id


class FakePlaybackResolver:
    def __init__(self, result: str | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def resolve_playback_track(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakePlaybackAdapter:
    def __init__(
        self,
        failures: dict[str, Exception] | None = None,
        now_pid: str | None = "REAL-PID-1",
    ) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.failures = failures or {}
        self.now_pid = now_pid

    def read_player_state(self) -> str:
        return "stopped"

    def read_now_playing(self):
        from music_agent.playback_control import NowPlaying, PlayerState

        return NowPlaying(
            state=PlayerState.PLAYING if self.now_pid is not None else PlayerState.STOPPED,
            persistent_id=self.now_pid,
            name="起风了 (旧版)" if self.now_pid is not None else None,
            artist="某艺人" if self.now_pid is not None else None,
            album="某专辑" if self.now_pid is not None else None,
        )

    def play(self) -> None:
        self.calls.append(("play", ()))
        self._maybe_fail("play")

    def pause(self) -> None:
        self.calls.append(("pause", ()))
        self._maybe_fail("pause")

    def next_track(self) -> None:
        self.calls.append(("next_track", ()))
        self._maybe_fail("next_track")

    def previous_track(self) -> None:
        self.calls.append(("previous_track", ()))
        self._maybe_fail("previous_track")

    def play_track(self, persistent_id: str) -> None:
        self.calls.append(("play_track", (persistent_id,)))
        self._maybe_fail("play_track")

    def _maybe_fail(self, name: str) -> None:
        failure = self.failures.get(name)
        if failure is not None:
            raise failure


class FakePreviewRunner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stop_calls = 0
        self.active = False
        # P15-S1: the natural-finish hook surface (plain attribute; the production
        # runner exposes a validating property setter with the same name).
        self.on_natural_finish = None

    def start_audio(self, url: str) -> None:
        self.started.append(url)
        self.active = True

    def stop_preview(self) -> bool:
        self.stop_calls += 1
        was_active = self.active
        self.active = False
        return was_active

    def is_preview_active(self) -> bool:
        # P14-C06.3b: the runner's own read-only truth, like the production runner.
        return self.active

    def trigger_natural_finish(self) -> None:
        """P15-S1 test lever: mirror the reaper's natural exit (clip stops, hook fires)."""
        self.active = False
        callback = self.on_natural_finish
        if callback is not None:
            callback()


class FakePreviewSearchSource:
    def __init__(self, url: str) -> None:
        self.url = url

    def search(self, term: str, limit: int):
        return []

    def lookup_preview_url(self, itunes_id: str) -> str:
        return self.url


class MapPreviewSearchSource:
    """P15-S1: per-itunes-id preview URLs; a None entry fails the lookup (no URL)."""

    def __init__(self, urls: dict[str, str | None]) -> None:
        self.urls = urls

    def search(self, term: str, limit: int):
        return []

    def lookup_preview_url(self, itunes_id: str):
        return self.urls.get(itunes_id)


class FakeOpenSearchSource:
    """P16-S4: per-itunes-id official trackViewUrl values; a None entry means no URL."""

    def __init__(self, urls: dict[str, str | None]) -> None:
        self.urls = urls
        self.lookup_calls: list[str] = []

    def search(self, term: str, limit: int):
        return []

    def lookup_track_view_url(self, itunes_id: str):
        self.lookup_calls.append(itunes_id)
        return self.urls.get(itunes_id)


class FailingPreviewRunner(FakePreviewRunner):
    """P15-S1 真机修复 lever: raise a chosen exception starting with the Nth clip.

    System-class failures (the cross-thread ``ProgrammingError`` of the live bug,
    or a generic transport failure) must fail the whole session -- never cascade
    into per-item skips.
    """

    def __init__(self, failure: Exception, *, from_call: int = 2) -> None:
        super().__init__()
        self.failure = failure
        self.failure_from_call = from_call
        self.start_calls = 0

    def start_audio(self, url: str) -> None:
        self.start_calls += 1
        if self.start_calls >= self.failure_from_call:
            raise self.failure
        super().start_audio(url)


class PlaybackToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _service(
        self,
        policy: AgentClientPolicy,
        adapter=None,
        resolver=None,
        preview_runner=None,
        catalog_search_source=None,
    ) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: policy}),
            playback_adapter=adapter,
            playback_resolver=resolver,
            preview_runner=preview_runner,
            catalog_search_source=catalog_search_source,
        )
        self.addCleanup(service.close)
        return service

    def _seed_run(self, track_id: str) -> str:
        """Persist one minimal recommendation run whose only item targets ``track_id``.
        Direct repository write (P3B context-judgment seam), not a generation."""
        from datetime import datetime, timezone

        from music_agent.preference_attribution import (
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.recommendation_contract import (
            RECOMMENDATION_CONTRACT_VERSION,
            Candidate,
            CandidateSourceReference,
            Eligibility,
            RecommendationContext,
            RecommendationItem,
            RecommendationRequest,
            RecommendationResult,
            RecommendedItemKind,
            ScoreBreakdown,
            ScoreComponent,
            generate_run_id,
        )
        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )

        run = RecommendationResult(
            run_id=generate_run_id(),
            request=RecommendationRequest(
                context=RecommendationContext(datetime.now(timezone.utc), ()),
                recommended_kind=RecommendedItemKind.TRACK,
                limit=1,
            ),
            items=(
                RecommendationItem(
                    candidate=Candidate(
                        candidate_id="cnd_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                        target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                        source=CandidateSourceReference("test_seed", "context_judgment"),
                        eligibility=Eligibility.ELIGIBLE,
                    ),
                    score=ScoreBreakdown(0.9, (ScoreComponent("test", 0.9),)),
                ),
            ),
            produced_at=datetime.now(timezone.utc),
            contract_version=RECOMMENDATION_CONTRACT_VERSION,
        )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(run)
        return run.run_id

    def _seed_itunes_binding(self, track_id: str, itunes_id: str) -> None:
        """Add an itunes_store binding to one fixture track in place."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    def _seed_spring_thief_unbound(self) -> str:
        """A catalog-like track named Spring Thief (Yorushika, album, duration) with no
        apple_music_persistent_id -- the exact P12 duplicate-playback shape."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == UNBOUND_TRACK)
        artist_id = "art_00000000-0000-4000-8000-000000000009"
        album_id = "alb_00000000-0000-4000-8000-000000000009"
        model["artists"].append(
            {
                "id": artist_id,
                "external_ids": {"apple_music_persistent_id": None},
                "name": "Yorushika",
            }
        )
        model["albums"].append(
            {
                "id": album_id,
                "external_ids": {"apple_music_persistent_id": None},
                "name": "Spring Thief - Single",
                "artist_ids": [artist_id],
                "release_date": "2021-01-09",
            }
        )
        track["name"] = "Spring Thief"
        track["artist_ids"] = [artist_id]
        track["album_id"] = album_id
        track["duration_ms"] = 290279
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)
        return track["id"]

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def test_play_pause_next_previous_commands(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        for tool in ("play", "pause", "next_track", "previous_track"):
            result = client.call(tool, {})
            self.assertEqual(result.outcome.value, "ok", tool)
        self.assertEqual(
            [name for name, _ in adapter.calls],
            ["play", "pause", "next_track", "previous_track"],
        )

    def test_play_track_resolves_binding_and_commands_persistent_id(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(adapter.calls, [("play_track", ("SYNTH-TRACK-001",))])
        self.assertEqual(result.payload["persistent_id"], "SYNTH-TRACK-001")

    def test_play_track_unknown_canonical_id_fails_closed(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call(
            "play_track", {"canonical_id": "trk_00000000-0000-4000-8000-000000000000"}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "canonical_entity_not_found")
        self.assertEqual(adapter.calls, [])

    def test_play_track_missing_binding_fails_closed(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("play_track", {"canonical_id": UNBOUND_TRACK})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")
        self.assertEqual(adapter.calls, [])

    # --- P12: playback-equivalent resolver (read-only; writes nothing) -------------

    def test_play_track_bound_track_ignores_resolver(self) -> None:
        adapter = FakePlaybackAdapter()
        resolver = FakePlaybackResolver(result="SHOULD-NOT-BE-USED")
        service = self._service(AgentClientPolicy.FULL, adapter, resolver)
        result = self._client(service).call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(adapter.calls, [("play_track", ("SYNTH-TRACK-001",))])
        self.assertEqual(result.payload["resolution"], "binding")
        self.assertEqual(resolver.calls, [])

    def test_play_track_resolver_unique_hit_plays_resolved_id(self) -> None:
        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        resolver = FakePlaybackResolver(result="398490020FF165D3")
        service = self._service(AgentClientPolicy.FULL, adapter, resolver)
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(adapter.calls, [("play_track", ("398490020FF165D3",))])
        self.assertEqual(result.payload["persistent_id"], "398490020FF165D3")
        self.assertEqual(result.payload["resolution"], "playback_equivalent")
        self.assertEqual(
            resolver.calls,
            [
                {
                    "name": "Spring Thief",
                    "artist": "Yorushika",
                    "album": "Spring Thief - Single",
                    "duration_ms": 290279,
                }
            ],
        )

    def test_play_track_resolver_miss_fails_closed(self) -> None:
        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter, FakePlaybackResolver(None))
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")
        self.assertEqual(adapter.calls, [])

    def test_play_track_resolver_ambiguous_fails_closed(self) -> None:
        # The resolver contract maps multiple full matches to None: fail closed.
        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter, FakePlaybackResolver(None))
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")
        self.assertEqual(adapter.calls, [])

    def test_play_track_resolver_error_fails_closed(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        resolver = FakePlaybackResolver(error=PlaybackControlUnavailableError("AppleEvent timed out"))
        service = self._service(AgentClientPolicy.FULL, adapter, resolver)
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")
        self.assertIn("AppleEvent timed out", result.error_message or "")
        self.assertEqual(adapter.calls, [])

    def test_play_track_resolver_unwired_fails_closed(self) -> None:
        # No resolver wired: the historical fail-closed behavior is unchanged.
        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter, None)
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")
        self.assertEqual(adapter.calls, [])

    def test_play_track_resolver_writes_nothing(self) -> None:
        from music_agent.identity import EntityType, ExternalIdentityKey

        track_id = self._seed_spring_thief_unbound()
        adapter = FakePlaybackAdapter()
        resolver = FakePlaybackResolver(result="398490020FF165D3")
        service = self._service(AgentClientPolicy.FULL, adapter, resolver)
        result = self._client(service).call("play_track", {"canonical_id": track_id})
        self.assertEqual(result.outcome.value, "ok")
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            track = next(item for item in model["tracks"] if item["id"] == track_id)
            self.assertIsNone(track["external_ids"]["apple_music_persistent_id"])
            self.assertIsNone(
                repository.lookup_external_identity(
                    ExternalIdentityKey("apple_music", EntityType.TRACK, "398490020FF165D3")
                )
            )

    def test_no_playback_adapter_fails_closed(self) -> None:
        service = self._service(AgentClientPolicy.FULL, None)
        result = self._client(service).call("play", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_unavailable")

    def test_command_failure_surfaces_as_execution_error(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        adapter = FakePlaybackAdapter(
            {"play": PlaybackControlUnavailableError("Application isn't running")}
        )
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("play", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertIn("Application isn't running", result.error_message or "")

    def test_read_only_client_denied(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.READ_ONLY, adapter)
        result = self._client(service).call("play", {})
        self.assertEqual(result.outcome.value, "permission_denied")
        self.assertEqual(adapter.calls, [])

    def test_replay_protection_prevents_double_command(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        request_id = "req_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        first = client.call("next_track", {}, request_id=request_id)
        second = client.call("next_track", {}, request_id=request_id)
        self.assertEqual(first.outcome.value, "ok")
        self.assertEqual(second.outcome.value, "ok")
        self.assertTrue(second.replayed)
        self.assertEqual(len(adapter.calls), 1)  # executed exactly once

    # --- P3B: playback context judgment (in-memory channel + run-binding ownership) ---

    def test_play_track_flips_channel_to_library_and_owns_current_track(self) -> None:
        """播放 Agent 选的歌后换一首: the play_track pid becomes the ownership anchor."""
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        played = client.call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(played.outcome.value, "ok")
        snapshot = client.call("get_now_playing", {}).payload
        self.assertEqual(snapshot["context"], "agent_selected")
        self.assertEqual(
            snapshot["agent_channel"], {"state": "library", "canonical_id": BOUND_TRACK}
        )

    def test_recent_run_binding_marks_playing_track_agent_selected(self) -> None:
        """推荐后换一首（推荐上下文）: a recent run's library-bound target matches the
        playing pid, even without a play_track call in this instance."""
        self._seed_run(BOUND_TRACK)
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        snapshot = self._client(service).call("get_now_playing", {}).payload
        self.assertEqual(snapshot["context"], "agent_selected")
        self.assertEqual(snapshot["agent_channel"], {"state": "none", "canonical_id": None})

    def test_playing_non_recommended_track_is_own_queue(self) -> None:
        """普通播放换一首: the playing pid is not in any recent recommendation run."""
        self._seed_run(BOUND_TRACK)
        adapter = FakePlaybackAdapter(now_pid="USER-OWN-QUEUE-PID")
        service = self._service(AgentClientPolicy.FULL, adapter)
        snapshot = self._client(service).call("get_now_playing", {}).payload
        self.assertEqual(snapshot["context"], "own_queue")

    def test_active_context_resolves_bound_current_track_despite_own_queue(self) -> None:
        self._seed_run(UNBOUND_TRACK)
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)

        snapshot = self._client(service).call("get_active_context", {}).payload

        self.assertEqual(snapshot["context"], "own_queue")
        self.assertIsNone(snapshot["referent_canonical_id"])
        self.assertEqual(snapshot["player"]["canonical_id"], BOUND_TRACK)
        self.assertEqual(snapshot["player"]["canonical_resolution"], "binding")

    def test_active_context_resolves_one_strict_playback_equivalent_without_binding(self) -> None:
        target_id = self._seed_spring_thief_unbound()
        resolver = FakePlaybackResolver(result="SPRING-THIEF-PID")
        adapter = FakePlaybackAdapter(now_pid="SPRING-THIEF-PID")

        def read_now_playing():
            from music_agent.playback_control import NowPlaying, PlayerState

            return NowPlaying(
                state=PlayerState.PLAYING,
                persistent_id="SPRING-THIEF-PID",
                name="Spring Thief",
                artist="Yorushika",
                album="Spring Thief - Single",
            )

        adapter.read_now_playing = read_now_playing  # type: ignore[method-assign]
        service = self._service(
            AgentClientPolicy.FULL, adapter, resolver=resolver
        )

        snapshot = self._client(service).call("get_active_context", {}).payload

        self.assertEqual(snapshot["player"]["canonical_id"], target_id)
        self.assertEqual(
            snapshot["player"]["canonical_resolution"],
            "playback_equivalent",
        )
        self.assertEqual(len(resolver.calls), 1)
        with CanonicalRepository(self.database_path) as repository:
            self.assertIsNone(
                repository.lookup_external_identity(
                    ExternalIdentityKey(
                        "apple_music", EntityType.TRACK, "SPRING-THIEF-PID"
                    )
                )
            )

    def test_active_context_playback_equivalent_fails_closed_on_ambiguity(self) -> None:
        self._seed_spring_thief_unbound()
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            original = next(
                track for track in model["tracks"] if track["id"] == UNBOUND_TRACK
            )
            duplicate = json.loads(json.dumps(original))
            duplicate["id"] = "trk_99999999-9999-4999-8999-999999999999"
            model["tracks"].append(duplicate)
            repository.save_model(model)
        resolver = FakePlaybackResolver(result="SPRING-THIEF-PID")
        adapter = FakePlaybackAdapter(now_pid="SPRING-THIEF-PID")

        def read_now_playing():
            from music_agent.playback_control import NowPlaying, PlayerState

            return NowPlaying(
                state=PlayerState.PLAYING,
                persistent_id="SPRING-THIEF-PID",
                name="Spring Thief",
                artist="Yorushika",
                album="Spring Thief - Single",
            )

        adapter.read_now_playing = read_now_playing  # type: ignore[method-assign]
        service = self._service(
            AgentClientPolicy.FULL, adapter, resolver=resolver
        )

        player = self._client(service).call(
            "get_active_context", {}
        ).payload["player"]

        self.assertIsNone(player["canonical_id"])
        self.assertIsNone(player["canonical_resolution"])
        self.assertEqual(len(resolver.calls), 2)

    def test_active_context_playback_equivalent_fails_closed_on_insufficient_evidence(self) -> None:
        self._seed_spring_thief_unbound()
        resolver = FakePlaybackResolver(result="SPRING-THIEF-PID")
        adapter = FakePlaybackAdapter(now_pid="SPRING-THIEF-PID")

        def read_now_playing():
            from music_agent.playback_control import NowPlaying, PlayerState

            return NowPlaying(
                state=PlayerState.PLAYING,
                persistent_id="SPRING-THIEF-PID",
                name="Spring Thief",
                artist=None,
                album="Spring Thief - Single",
            )

        adapter.read_now_playing = read_now_playing  # type: ignore[method-assign]
        service = self._service(
            AgentClientPolicy.FULL, adapter, resolver=resolver
        )

        player = self._client(service).call(
            "get_active_context", {}
        ).payload["player"]

        self.assertIsNone(player["canonical_id"])
        self.assertIsNone(player["canonical_resolution"])
        self.assertEqual(resolver.calls, [])

    def test_context_unknown_without_evidence(self) -> None:
        """No recommendation runs and no anchor: nothing supports a claim either way."""
        adapter = FakePlaybackAdapter(now_pid="SOME-OTHER-PID")
        service = self._service(AgentClientPolicy.FULL, adapter)
        snapshot = self._client(service).call("get_now_playing", {}).payload
        self.assertEqual(snapshot["context"], "unknown")
        empty = self._client(service).call("get_now_playing", {})
        adapter.now_pid = None
        self.assertEqual(
            self._client(service).call("get_now_playing", {}).payload["context"], "unknown"
        )
        self.assertEqual(empty.outcome.value, "ok")

    def test_preview_flips_channel_to_preview(self) -> None:
        """试听后换一首: a preview execution records preview state for the snapshot.
        P3B batch 2: the result reports ``started`` immediately (non-blocking).
        P15-S1 C03: the payload carries the same ``suspended`` contract as
        ``preview_batch`` (the shared helper pauses the playing Music.app)."""
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(
            AgentClientPolicy.FULL, adapter, None, runner, FakePreviewSearchSource("https://example.test/p.m4a")
        )
        client = self._client(service)
        previewed = client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        self.assertEqual(previewed.outcome.value, "ok")
        self.assertEqual(
            previewed.payload,
            {
                "canonical_id": UNBOUND_TRACK,
                "started": True,
                "preview_url": "https://example.test/p.m4a",
                "suspended": {
                    "player_state": "playing",
                    "persistent_id": "SYNTH-TRACK-001",
                    "name": "起风了 (旧版)",
                    "pause_ok": True,
                },
            },
        )
        self.assertEqual(runner.started, ["https://example.test/p.m4a"])
        snapshot = client.call("get_now_playing", {}).payload
        self.assertEqual(snapshot["agent_channel"], {"state": "preview", "canonical_id": UNBOUND_TRACK})
        # The Music.app pid is judged independently of the preview channel; with no runs
        # and no play anchor there is no ownership evidence at all.
        self.assertEqual(snapshot["context"], "unknown")

    def test_stop_preview_stops_active_and_clears_channel(self) -> None:
        """试听中「停」: stops the sounding clip and clears the preview registration."""
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(
            AgentClientPolicy.FULL, adapter, None, runner, FakePreviewSearchSource("https://example.test/p.m4a")
        )
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        stopped = client.call("stop_preview", {})
        self.assertEqual(stopped.outcome.value, "ok")
        self.assertEqual(
            stopped.payload,
            {
                "command": "stop_preview",
                "ok": True,
                "stopped": True,
                "preview_session_cancelled": False,
            },
        )
        self.assertEqual(runner.stop_calls, 1)
        snapshot = client.call("get_now_playing", {}).payload
        self.assertEqual(snapshot["agent_channel"], {"state": "none", "canonical_id": None})

    def test_referent_survives_the_stop_through_the_play_then_preview_sequence(self) -> None:
        """P19-T14-F-R4 service-level proof: formal A -> preview B -> stop ->
        referent is still B (never A, never lost). The referent is the
        conversational target; the channel is only the action log."""
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(
            AgentClientPolicy.FULL, adapter, None, runner,
            FakePreviewSearchSource("https://example.test/p.m4a"),
        )
        client = self._client(service)
        played = client.call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(played.outcome.value, "ok")
        context = client.call("get_playback_context", {}).payload
        self.assertEqual(context["referent_canonical_id"], BOUND_TRACK)

        previewed = client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        self.assertTrue(previewed.payload["started"])
        context = client.call("get_playback_context", {}).payload
        self.assertEqual(context["referent_canonical_id"], UNBOUND_TRACK)

        stopped = client.call("stop_preview", {})
        self.assertTrue(stopped.payload["stopped"])
        after = client.call("get_playback_context", {}).payload
        self.assertEqual(
            after["channel"], {"state": "none", "canonical_id": None}
        )
        self.assertEqual(after["referent_canonical_id"], UNBOUND_TRACK)
        # get_active_context carries the same surviving referent.
        active = client.call("get_active_context", {}).payload
        self.assertEqual(active["referent_canonical_id"], UNBOUND_TRACK)

    def test_failed_preview_never_touches_the_referent(self) -> None:
        """R4 regression 4: a preview that fails before the concrete target
        interaction must not corrupt the previous referent."""
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        runner = FakePreviewRunner()
        service = self._service(AgentClientPolicy.FULL, adapter, None, runner)
        client = self._client(service)
        played = client.call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(played.outcome.value, "ok")

        failed = client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        self.assertEqual(failed.outcome.value, "execution_error")

        context = client.call("get_playback_context", {}).payload
        self.assertEqual(context["referent_canonical_id"], BOUND_TRACK)
        self.assertEqual(context["channel"], {"state": "library", "canonical_id": BOUND_TRACK})

    def test_stop_preview_idempotent_without_active(self) -> None:
        runner = FakePreviewRunner()  # never started
        service = self._service(AgentClientPolicy.FULL, FakePlaybackAdapter(), None, runner)
        stopped = self._client(service).call("stop_preview", {})
        self.assertEqual(stopped.outcome.value, "ok")
        self.assertEqual(stopped.payload["stopped"], False)

    def test_stop_preview_read_only_client_denied(self) -> None:
        runner = FakePreviewRunner()
        service = self._service(AgentClientPolicy.READ_ONLY, FakePlaybackAdapter(), None, runner)
        stopped = self._client(service).call("stop_preview", {})
        self.assertEqual(stopped.outcome.value, "permission_denied")
        self.assertEqual(runner.stop_calls, 0)

    def test_next_track_clears_channel_but_keeps_ownership_anchor(self) -> None:
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        client.call("play_track", {"canonical_id": BOUND_TRACK})
        client.call("next_track", {})
        adapter.now_pid = "USER-OWN-QUEUE-PID"
        snapshot = client.call("get_now_playing", {}).payload
        self.assertEqual(snapshot["agent_channel"], {"state": "none", "canonical_id": None})
        self.assertEqual(snapshot["context"], "own_queue")
        # The anchor survived for ownership when the queue comes back to the played pid.
        adapter.now_pid = "SYNTH-TRACK-001"
        self.assertEqual(
            client.call("get_now_playing", {}).payload["context"], "agent_selected"
        )

    def test_channel_state_is_transient_across_service_instances(self) -> None:
        """The register is in-memory only: a fresh instance starts with none, nothing is
        persisted anywhere."""
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        self._client(service).call("play_track", {"canonical_id": BOUND_TRACK})
        adapter2 = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service2 = self._service(AgentClientPolicy.FULL, adapter2)
        snapshot = self._client(service2).call("get_now_playing", {}).payload
        self.assertEqual(snapshot["agent_channel"], {"state": "none", "canonical_id": None})

    def test_get_now_playing_reads_context_and_reports_elapsed(self) -> None:
        adapter = FakePlaybackAdapter()
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("get_now_playing", {})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(result.payload["now_playing"]["name"], "起风了 (旧版)")
        self.assertEqual(result.payload["now_playing"]["state"], "playing")
        self.assertEqual(result.payload["now_playing"]["persistent_id"], "REAL-PID-1")
        self.assertIn("elapsed_ms", result.payload["now_playing"])
        self.assertIsNone(result.payload["player_canonical_id"])
        self.assertIsNone(result.payload["canonical_resolution"])

    def test_get_now_playing_projects_bound_actual_player_canonical(self) -> None:
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)

        result = self._client(service).call("get_now_playing", {})

        self.assertEqual(result.payload["player_canonical_id"], BOUND_TRACK)
        self.assertEqual(result.payload["canonical_resolution"], "binding")

    def test_get_now_playing_adapter_failure_is_typed_not_a_crash(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        adapter = FakePlaybackAdapter()
        adapter.read_now_playing = (  # type: ignore[assignment]
            lambda: (_ for _ in ()).throw(
                PlaybackControlUnavailableError("Application isn't running")
            )
        )
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("get_now_playing", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_command_failed")
        self.assertIn("Application isn't running", result.error_message or "")

    def test_playback_payloads_carry_elapsed_ms(self) -> None:
        adapter = FakePlaybackAdapter()
        adapter.last_command_elapsed_ms = 12.5  # the production adapter sets this per command
        service = self._service(AgentClientPolicy.FULL, adapter)
        result = self._client(service).call("pause", {})
        self.assertIn("elapsed_ms", result.payload)
        self.assertEqual(result.payload["elapsed_ms"], 12.5)

    def test_library_write_matrix_untouched(self) -> None:
        from music_agent.agent_permission import write_capability_summary

        summary = write_capability_summary()
        self.assertEqual(len(summary), 14)  # + P11.3 add_library_song
        self.assertFalse(any(entry["execution_ready"] for entry in summary))
        service = self._service(AgentClientPolicy.FULL, FakePlaybackAdapter())
        capabilities = self._client(service).call("get_agent_capabilities", {})
        self.assertEqual(capabilities.outcome.value, "ok")
        self.assertEqual(len(capabilities.payload["tools"]), 32)  # + P16-S4 open_in_apple_music
        self.assertEqual(len(capabilities.payload["writes"]), 14)

    def test_now_playing_parser_shapes(self) -> None:
        from music_agent.playback_control import (
            MusicPlaybackAdapter,
            NowPlaying,
            PlaybackControlUnavailableError,
            PlayerState,
        )

        class RawRunner:
            def __init__(self, raw: str) -> None:
                self.raw = raw

            def read_now_playing(self) -> str:
                return self.raw

            def read_player_state(self) -> str:
                return "stopped"

            def play(self) -> None: ...
            def pause(self) -> None: ...
            def next_track(self) -> None: ...
            def previous_track(self) -> None: ...
            def play_track(self, persistent_id: str) -> None: ...

        playing = MusicPlaybackAdapter(RawRunner("REAL-PID-1\t起风了 (旧版)\t某艺人\t某专辑\tplaying"))
        result = playing.read_now_playing()
        self.assertIsInstance(result, NowPlaying)
        self.assertEqual(result.state, PlayerState.PLAYING)
        self.assertEqual(result.persistent_id, "REAL-PID-1")
        self.assertEqual(result.name, "起风了 (旧版)")

        stopped = MusicPlaybackAdapter(RawRunner("stopped")).read_now_playing()
        self.assertEqual(stopped.state, PlayerState.STOPPED)
        self.assertIsNone(stopped.persistent_id)

        with self.assertRaises(PlaybackControlUnavailableError):
            MusicPlaybackAdapter(RawRunner("a\tb\tc")).read_now_playing()
        with self.assertRaises(PlaybackControlUnavailableError):
            MusicPlaybackAdapter(RawRunner("buffering")).read_now_playing()

    def test_now_playing_script_has_no_backslash_escapes(self) -> None:
        from music_agent.playback_control import NOW_PLAYING_SCRIPT

        self.assertNotIn('\\"', NOW_PLAYING_SCRIPT)
        self.assertIn("persistent ID of currentTrack", NOW_PLAYING_SCRIPT)

    def test_playback_scripts_are_expected_commands(self) -> None:
        with patch("music_agent.playback_control.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            runner = OsascriptPlaybackRunner(timeout_seconds=5.0)
            runner.play()
            runner.pause()
            runner.next_track()
            runner.previous_track()
            runner.play_track("REAL-ID")
        scripts = [call.kwargs.get("text") is None and call.args[0][2] for call in run.call_args_list]
        self.assertIn(PLAY_SCRIPT, [call.args[0][2] for call in run.call_args_list])
        self.assertIn(PAUSE_SCRIPT, [call.args[0][2] for call in run.call_args_list])
        self.assertIn(NEXT_TRACK_SCRIPT, [call.args[0][2] for call in run.call_args_list])
        self.assertIn(PREVIOUS_TRACK_SCRIPT, [call.args[0][2] for call in run.call_args_list])
        play_track_script = PLAY_TRACK_SCRIPT.replace("targetTrackID", '"REAL-ID"', 1)
        self.assertIn(play_track_script, [call.args[0][2] for call in run.call_args_list])

    def test_adapter_has_no_resume_surface(self) -> None:
        """The headphone-safety rule: NO automatic resume exists anywhere."""
        adapter = MusicPlaybackAdapter(
            FakePlaybackAdapter()  # type: ignore[arg-type]
        )
        for method in dir(adapter):
            self.assertNotIn("resume", method)


class ProviderLoopPlaybackTest(unittest.TestCase):
    def test_loop_flows_play_command_through_p09(self) -> None:
        from music_agent.provider_agent import ProviderAgentLoop
        from music_agent.provider_contract import (
            ProviderMessage,
            ProviderMessageRole,
            ProviderResponse,
            ProviderStopReason,
            ProviderToolCall,
        )
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            adapter = FakePlaybackAdapter()
            service = SharedAgentService(
                database_path,
                clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
                playback_adapter=adapter,
            )
            self.addCleanup(service.close)

            class FakeProvider:
                def __init__(self) -> None:
                    self.responses = [
                        ProviderResponse(
                            ProviderMessage(
                                ProviderMessageRole.ASSISTANT,
                                tool_calls=(ProviderToolCall("c1", "play", "{}"),),
                            ),
                            ProviderStopReason.TOOL_USE,
                            {},
                        ),
                        ProviderResponse(
                            ProviderMessage(ProviderMessageRole.ASSISTANT, text="已开始播放。"),
                            ProviderStopReason.END_TURN,
                            {},
                        ),
                    ]

                def chat(self, system, messages, tools):
                    return self.responses.pop(0)

            client = AgentClient(
                AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
                service,
            )
            provider = FakeProvider()
            loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
            result = loop.run("开始播放")
            self.assertEqual(result.final_text, "已继续播放。")
            self.assertEqual([name for name, _ in adapter.calls], ["play"])
            self.assertEqual(
                [item.name for item in result.tool_executions],
                ["play", "get_now_playing"],
            )
            self.assertEqual(result.tool_executions[0].outcome, "ok")
            self.assertEqual(len(provider.responses), 1)


class RuntimePlaybackWiringTest(unittest.TestCase):
    def test_runtime_wires_observer_to_the_single_safety_authority(self) -> None:
        # P15-S2 r3: the observer holds the service (sole safety decision-maker),
        # never a playback adapter; playback remains the service's exclusive surface.
        from music_agent.runtime import Runtime, RuntimeConfig

        class FakeRunner:
            def read_player_state(self) -> str:
                return "stopped"

            def read_now_playing(self) -> str:
                return "{}"

            def play(self) -> None: ...

            def pause(self) -> None: ...

            def next_track(self) -> None: ...

            def previous_track(self) -> None: ...

            def play_track(self, persistent_id: str) -> None: ...

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            runtime = Runtime(
                RuntimeConfig(database_path=database_path, audio_safety_enabled=True)
            )
            with patch(
                "music_agent.playback_control.OsascriptPlaybackRunner",
                return_value=FakeRunner(),
            ):
                runtime.start()
                try:
                    self.assertIsNotNone(runtime.agent_service._playback_adapter)
                    observer = runtime.audio_monitor
                    self.assertIsNotNone(observer)
                    self.assertFalse(hasattr(observer, "_playback"))
                    self.assertIs(observer._service, runtime.agent_service)
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()


class NowPlayingLoopBoundaryTest(unittest.TestCase):
    def test_failing_get_now_playing_does_not_crash_the_loop(self) -> None:
        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentClientIdentity
        from music_agent.provider_agent import ProviderAgentLoop
        from music_agent.provider_contract import (
            ProviderMessage,
            ProviderMessageRole,
            ProviderResponse,
            ProviderStopReason,
            ProviderToolCall,
        )
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS
        from music_agent.playback_control import PlaybackControlUnavailableError

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "store.db"
            with CanonicalRepository(database_path) as repository:
                repository.save_model(
                    {"tracks": [], "artists": [], "albums": [],
                     "playlists": [], "playlist_memberships": []}
                )
            adapter = FakePlaybackAdapter()
            adapter.read_now_playing = (  # type: ignore[assignment]
                lambda: (_ for _ in ()).throw(
                    PlaybackControlUnavailableError("Application isn't running")
                )
            )
            service = SharedAgentService(
                database_path,
                clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
                playback_adapter=adapter,
            )
            self.addCleanup(service.close)

            class FakeProvider:
                def chat(self, system, messages, tools):
                    return ProviderResponse(
                        ProviderMessage(
                            ProviderMessageRole.ASSISTANT,
                            tool_calls=(ProviderToolCall(
                                "c1", "get_now_playing",
                                "{}",
                            ),),
                        ),
                        ProviderStopReason.TOOL_USE,
                        {},
                    )

            client = AgentClient(
                AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
                service,
            )
            loop = ProviderAgentLoop(FakeProvider(), client, PROVIDER_TOOL_SCHEMAS)
            result = loop.run("现在在放什么")
            # The loop capped at the round bound with an honest last-text answer --
            # no exception escaped, the CLI would exit cleanly.
            self.assertTrue(result.rounds_capped)
            self.assertEqual(result.tool_executions[0].outcome, "execution_error")
            self.assertEqual(result.tool_executions[0].error_code, "playback_command_failed")


class PreviewSuspensionTest(unittest.TestCase):
    """P15-PC C1: suspend-don't-overlap before preview audio starts."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _service(self, adapter, runner, source) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
            playback_resolver=None,
            preview_runner=runner,
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        return service

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def _seed_binding(self, track_id: str, itunes_id: str) -> None:
        """Add an itunes_store binding to one fixture track in place."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    def _adapter(
        self,
        *,
        state: str = "playing",
        pid: str | None = "REAL-PID-1",
        name: str | None = "起风了 (旧版)",
        read_error: Exception | None = None,
        pause_error: Exception | None = None,
    ) -> FakePlaybackAdapter:
        from music_agent.playback_control import NowPlaying, PlayerState

        adapter = FakePlaybackAdapter(now_pid=pid)

        def read_now_playing():
            if read_error is not None:
                raise read_error
            return NowPlaying(
                state=PlayerState(state),
                persistent_id=pid,
                name=name,
                artist="某艺人" if pid else None,
                album="某专辑" if pid else None,
            )

        def pause() -> None:
            adapter.calls.append(("pause", ()))
            if pause_error is not None:
                raise pause_error

        adapter.read_now_playing = read_now_playing  # type: ignore[method-assign]
        adapter.pause = pause  # type: ignore[method-assign]
        return adapter

    def _preview(self, adapter, runner=None):
        self._seed_binding(UNBOUND_TRACK, "296103682")
        runner = runner or FakePreviewRunner()
        service = self._service(
            adapter,
            runner,
            FakePreviewSearchSource("https://example.test/p.m4a"),
        )
        return service, runner, adapter

    def _entry(self, value):
        from music_agent.playback_context import SuspendedPlaybackEntry

        self.assertIsInstance(value, SuspendedPlaybackEntry)
        return value

    def test_playing_music_is_paused_and_suspension_recorded(self) -> None:
        service, runner, adapter = self._preview(self._adapter())
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(runner.started, ["https://example.test/p.m4a"])
        self.assertEqual(adapter.calls, [("pause", ())])
        entry = self._entry(service._playback_context.suspension.value)
        self.assertEqual(entry.player_state, "playing")
        self.assertEqual(entry.persistent_id, "REAL-PID-1")
        self.assertEqual(entry.name, "起风了 (旧版)")
        self.assertTrue(entry.pause_ok)

    def test_non_playing_states_pause_nothing_and_record_nothing(self) -> None:
        for state in ("paused", "stopped", "unknown"):
            with self.subTest(state=state):
                service, runner, adapter = self._preview(adapter=self._adapter(state=state))
                result = self._client(service).call(
                    "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
                )
                self.assertEqual(result.outcome.value, "ok")
                self.assertEqual(runner.started, ["https://example.test/p.m4a"])
                self.assertNotIn(("pause", ()), adapter.calls)
                self.assertIsNone(service._playback_context.suspension.value)

    def test_unreadable_player_never_blocks_preview(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        service, runner, adapter = self._preview(
            adapter=self._adapter(
                read_error=PlaybackControlUnavailableError("Application isn't running")
            )
        )
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertNotIn(("pause", ()), adapter.calls)
        self.assertIsNone(service._playback_context.suspension.value)

    def test_pause_failure_records_honest_entry_and_preview_still_starts(self) -> None:
        from music_agent.playback_control import PlaybackControlUnavailableError

        service, runner, adapter = self._preview(
            adapter=self._adapter(
                pause_error=PlaybackControlUnavailableError("Music.app not running")
            )
        )
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(runner.started, ["https://example.test/p.m4a"])
        entry = self._entry(service._playback_context.suspension.value)
        self.assertEqual(entry.name, "起风了 (旧版)")
        self.assertFalse(entry.pause_ok)

    def test_no_playback_adapter_is_a_noop(self) -> None:
        service, runner, _ = self._preview(adapter=None)
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(runner.started, ["https://example.test/p.m4a"])
        self.assertIsNone(service._playback_context.suspension.value)

    def test_failed_preview_lookup_never_suspends(self) -> None:
        """Music is only suspended for a preview that will actually start."""
        self._seed_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        adapter = self._adapter()
        service = self._service(adapter, runner, FakePreviewSearchSource(url=None))
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        # FakePreviewSearchSource(url=None) resolves no URL -> typed failure before audio.
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_preview_unavailable")
        self.assertEqual(runner.started, [])
        self.assertNotIn(("pause", ()), adapter.calls)
        self.assertIsNone(service._playback_context.suspension.value)

    def test_single_preview_pauses_playing_music_and_reports_the_suspension(self) -> None:
        """P15-S1 C03 验证①: 正式播放 → 单首试听 → Music.app 暂停 + suspension
        留存, and the tool result carries the same ``suspended`` contract as
        ``preview_batch`` so every preview entry point is observable."""
        service, runner, adapter = self._preview(self._adapter())
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(runner.started, ["https://example.test/p.m4a"])
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(result.payload["suspended"], {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": True,
        })
        entry = self._entry(service._playback_context.suspension.value)
        self.assertEqual(entry.persistent_id, "REAL-PID-1")

    def test_single_preview_end_never_auto_restores(self) -> None:
        """P15-S1 C03 验证②: the preview ending (stop_preview) never resumes
        formal playback on its own -- the suspension stays recorded as the
        restore target and only an explicit 继续/play restores."""
        service, runner, adapter = self._preview(self._adapter())
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        client.call("stop_preview", {})
        self.assertEqual(adapter.calls, [("pause", ())])  # never a ("play", ())
        self.assertIsNotNone(service._playback_context.suspension.value)

    def test_continue_after_a_single_preview_still_restores_play(self) -> None:
        """P15-S1 C03 验证③: with a recorded suspension, 继续播放 = play reaches
        Music.app and resumes the interrupted formal playback."""
        service, runner, adapter = self._preview(self._adapter())
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        result = client.call("play", {})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(adapter.calls, [("pause", ()), ("play", ())])
        self.assertIsNotNone(service._playback_context.suspension.value)


class PausedAwareAdapter(FakePlaybackAdapter):
    """Authoritative-state fake: pause keeps the persistent id and reads ``paused``;
    play restores ``playing`` -- what Music.app reads across a preview pause."""

    def __init__(self, state: str = "playing") -> None:
        super().__init__(now_pid="REAL-PID-1" if state != "stopped" else None)
        self.read_state = state

    def read_now_playing(self):
        from music_agent.playback_control import NowPlaying, PlayerState

        pid = "REAL-PID-1" if self.read_state in ("playing", "paused") else None
        return NowPlaying(
            state=PlayerState(self.read_state),
            persistent_id=pid,
            name="起风了 (旧版)" if pid else None,
            artist="某艺人" if pid else None,
            album="某专辑" if pid else None,
        )

    def pause(self) -> None:
        super().pause()
        if self.read_state == "playing":
            self.read_state = "paused"

    def play(self) -> None:
        super().play()
        self.read_state = "playing"
        self.now_pid = self.now_pid or "REAL-PID-1"


class StartErrorPreviewRunner(FakePreviewRunner):
    """P17 acceptance lever: the afplay boundary fails -- the clip never sounds."""

    def start_audio(self, url: str) -> None:
        from music_agent.catalog_preview import CatalogPreviewError

        raise CatalogPreviewError("preview playback failed to start: afplay missing")


class NaturalEndResumeTest(unittest.TestCase):
    """P17 acceptance: natural-end auto-restore.

    One matrix over authoritative state: a preview that interrupted playing music
    restores it when (and only when) the clip ends naturally into a still-paused,
    same-track readback. Explicit stop, start failure, and in-band user playback
    decisions during the preview all leave playback alone (no stale auto-resume).
    """

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _preview(self, adapter=None, runner=None):
        self._seed_binding(UNBOUND_TRACK, "296103682")
        adapter = adapter or PausedAwareAdapter()
        runner = runner or FakePreviewRunner()
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
            playback_resolver=None,
            preview_runner=runner,
            catalog_search_source=FakePreviewSearchSource("https://example.test/p.m4a"),
        )
        self.addCleanup(service.close)
        return service, runner, adapter

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def _seed_binding(self, track_id: str, itunes_id: str) -> None:
        """Add an itunes_store binding to one fixture track in place."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    def test_natural_end_resumes_the_playing_track_it_interrupted(self) -> None:
        service, runner, adapter = self._preview()
        client = self._client(service)
        result = client.call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(adapter.read_state, "paused")  # interrupted, not forgotten
        active = client.call("get_playback_context", {}).payload
        self.assertEqual(active["preview_suspension"], active["suspended"])

        runner.trigger_natural_finish()  # the real natural-end signal

        self.assertEqual(adapter.calls, [("pause", ()), ("play", ())])  # restored
        self.assertEqual(adapter.read_now_playing().state.value, "playing")  # authoritative
        completed = client.call("get_playback_context", {}).payload
        self.assertIsNone(completed["preview_suspension"])
        self.assertIsNotNone(completed["suspended"])  # restore-by-intent history unchanged

    def test_natural_end_never_starts_paused_or_stopped_playback(self) -> None:
        for initial_state in ("paused", "stopped"):
            with self.subTest(initial_state=initial_state):
                service, runner, adapter = self._preview(PausedAwareAdapter(initial_state))
                result = self._client(service).call(
                    "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
                )
                self.assertEqual(result.outcome.value, "ok")
                self.assertEqual(adapter.calls, [])  # nothing playing -> nothing paused
                active = self._client(service).call("get_playback_context", {}).payload
                self.assertIsNone(active["preview_suspension"])

                runner.trigger_natural_finish()

                self.assertEqual(adapter.calls, [])  # and nothing auto-starts
                self.assertEqual(adapter.read_state, initial_state)

    def test_explicit_stop_disarms_the_auto_restore(self) -> None:
        """stop_preview (or the safety pause) ends the clip the user's way --
        the natural-end signal that follows must not restore anything."""
        service, runner, adapter = self._preview()
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        result = client.call("stop_preview", {})
        self.assertEqual(result.outcome.value, "ok")
        stopped = client.call("get_playback_context", {}).payload
        self.assertIsNone(stopped["preview_suspension"])
        self.assertIsNotNone(stopped["suspended"])  # legacy manual-resume target remains

        runner.trigger_natural_finish()

        self.assertEqual(adapter.calls, [("pause", ())])  # never a ("play", ())
        self.assertEqual(adapter.read_state, "paused")

    def test_user_play_during_preview_is_never_overridden(self) -> None:
        """"The user resumed by hand while the clip sounded: the natural end must
        not double-play, and their playing state wins."""
        service, runner, adapter = self._preview()
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        result = client.call("play", {})
        self.assertEqual(result.outcome.value, "ok")
        overridden = client.call("get_playback_context", {}).payload
        self.assertIsNone(overridden["preview_suspension"])
        self.assertEqual(adapter.calls, [("pause", ()), ("play", ())])
        self.assertEqual(adapter.read_state, "playing")

        runner.trigger_natural_finish()

        self.assertEqual(adapter.calls, [("pause", ()), ("play", ())])  # exactly theirs
        self.assertEqual(adapter.read_state, "playing")

    def test_explicit_track_play_clears_only_the_preview_note_obligation(self) -> None:
        service, _, adapter = self._preview()
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})

        result = client.call("play_track", {"canonical_id": BOUND_TRACK})

        self.assertEqual(result.outcome.value, "ok")
        observed = client.call("get_playback_context", {}).payload
        self.assertIsNone(observed["preview_suspension"])
        self.assertIsNotNone(observed["suspended"])
        self.assertIn(("play_track", ("SYNTH-TRACK-001",)), adapter.calls)

    def test_repeated_preview_never_reuses_an_old_note_obligation(self) -> None:
        service, _, adapter = self._preview()
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        client.call("stop_preview", {})
        self.assertEqual(adapter.read_state, "paused")

        second = client.call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )

        self.assertEqual(second.outcome.value, "ok")
        observed = client.call("get_playback_context", {}).payload
        self.assertIsNone(observed["preview_suspension"])
        self.assertIsNotNone(observed["suspended"])  # old memo is not a current note

    def test_user_pause_decision_during_preview_is_never_overridden(self) -> None:
        """In-band re-affirmation of pause during the preview disarms restore --
        the natural end leaves the user's paused state alone."""
        service, runner, adapter = self._preview()
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        result = client.call("pause", {})
        self.assertEqual(result.outcome.value, "ok")

        runner.trigger_natural_finish()

        self.assertEqual(adapter.calls[0][0], "pause")
        self.assertNotIn(("play", ()), adapter.calls)
        self.assertEqual(adapter.read_state, "paused")

    def test_failed_preview_start_never_arms_or_restores(self) -> None:
        """The pause still happens (suspend precedes start_audio), but the clip
        never sounds: no restore arm may survive the failure, and a later
        natural-end signal stays a no-op."""
        service, runner, adapter = self._preview(runner=StartErrorPreviewRunner())
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "catalog_preview_failed")
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(adapter.read_state, "paused")

        # The arm was disarmed by the failure path...
        self.assertIsNone(service._playback_context.suspension.take_restore_arm())
        runner.trigger_natural_finish()  # ...and the hook can find nothing.
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(adapter.read_state, "paused")

    def test_failed_pause_disables_the_restore_while_preview_still_sounds(self) -> None:
        """pause_ok=False keeps the interruption honest and never auto-restores."""
        adapter = PausedAwareAdapter()
        adapter.failures = {"pause": RuntimeError("Music.app unresponsive")}
        service, runner, _ = self._preview(adapter)
        result = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "ok")
        entry = service._playback_context.suspension.value
        self.assertIsNotNone(entry)
        self.assertFalse(entry.pause_ok)

        runner.trigger_natural_finish()

        self.assertNotIn(("play", ()), adapter.calls)


BATCH_TRACK_B = "trk_22222222-2222-4222-8222-222222222222"  # SYNTH-TRACK-002
BATCH_TRACK_D = "trk_44444444-4444-4444-8444-444444444444"  # SYNTH-TRACK-004
PREVIEW_URL = "https://example.test/preview.m4a"


class StatefulPauseAdapter(FakePlaybackAdapter):
    """Production-faithful pause: once paused, consecutive reads report paused/stopped
    (so the second clip of a session does not re-pause Music.app)."""

    def pause(self) -> None:
        super().pause()
        self.now_pid = None


class PreviewBatchTest(unittest.TestCase):
    """P15-S1: preview_batch queue assembly, suspension, auto-advance and terminal
    semantics over the real service + fakes (no afplay, no Music.app).
    """

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        self.events: list[dict] = []

    def _service(self, adapter=None, runner=None, source=None) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
            playback_resolver=None,
            preview_runner=runner or FakePreviewRunner(),
            catalog_search_source=source or FakePreviewSearchSource(PREVIEW_URL),
        )
        self.addCleanup(service.close)
        service.preview_event_handler = self.events.append
        return service

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def _seed_binding(self, track_id: str, itunes_id: str | None) -> None:
        """Set (or clear) one fixture track's itunes_store binding in place."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    def _seed_runs(self, *track_lists: list[str]) -> list[str]:
        """Persist one recommendation run per track list, newest last (derived batch
        = the last one). Direct repository write (the P3B context-judgment seam)."""
        from datetime import datetime, timezone

        from music_agent.preference_attribution import (
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.recommendation_contract import (
            RECOMMENDATION_CONTRACT_VERSION,
            Candidate,
            CandidateSourceReference,
            Eligibility,
            RecommendationContext,
            RecommendationItem,
            RecommendationRequest,
            RecommendationResult,
            RecommendedItemKind,
            ScoreBreakdown,
            ScoreComponent,
            generate_run_id,
        )
        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )

        produced = datetime(2026, 8, 16, 9, 0, 0, tzinfo=timezone.utc)
        run_ids: list[str] = []
        for index, track_ids in enumerate(track_lists):
            items = tuple(
                RecommendationItem(
                    candidate=Candidate(
                        candidate_id=self._candidate_id(index, item_index),
                        target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                        source=CandidateSourceReference("test_seed", "context_judgment"),
                        eligibility=Eligibility.ELIGIBLE,
                    ),
                    score=ScoreBreakdown(0.9, (ScoreComponent("test", 0.9),)),
                )
                for item_index, track_id in enumerate(track_ids)
            )
            run = RecommendationResult(
                run_id=generate_run_id(),
                request=RecommendationRequest(
                    context=RecommendationContext(datetime.now(timezone.utc), ()),
                    recommended_kind=RecommendedItemKind.TRACK,
                    limit=len(items),
                ),
                items=items,
                produced_at=produced,
                contract_version=RECOMMENDATION_CONTRACT_VERSION,
            )
            produced = datetime(2026, 8, 16, 9, 5 + index, 0, 0, tzinfo=timezone.utc)
            with RecommendationHistoryRepository(self.database_path) as repository:
                repository.save_result(run)
            run_ids.append(run.run_id)
        return run_ids

    @staticmethod
    def _candidate_id(run_index: int, item_index: int) -> str:
        """Deterministic, contract-valid candidate id (cnd_ + versioned UUID)."""
        import uuid

        return f"cnd_{uuid.uuid5(uuid.NAMESPACE_URL, f'p15-seed-{run_index}-{item_index}')}"

    def _preview_batch(self, service) -> dict:
        result = self._client(service).call("preview_batch", {})
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        return result.payload

    def test_batch_assembles_queue_and_auto_advances_to_completion(self) -> None:
        """A1/A7 happy path: 2 previewable items play one after another with progress
        events, then the session completes exactly once."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        payload = self._preview_batch(service)
        self.assertTrue(payload["started"])
        self.assertIsNone(payload["suspended"])
        session = payload["session"]
        self.assertEqual(session["state"], "running")
        self.assertEqual(session["total"], 2)
        self.assertEqual(session["position"], 1)
        self.assertEqual(session["current_canonical_id"], BOUND_TRACK)
        self.assertEqual(session["current_name"], "Synthetic Duet")
        self.assertEqual(session["skipped"], [])
        self.assertEqual(runner.started, [PREVIEW_URL])
        self.assertEqual([event["event"] for event in self.events], ["progress"])

        runner.trigger_natural_finish()  # first clip ends naturally
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertEqual(observed["session"]["state"], "running")
        self.assertEqual(observed["session"]["position"], 2)
        self.assertEqual(observed["session"]["current_canonical_id"], BATCH_TRACK_B)
        self.assertTrue(observed["preview_sounding"])
        self.assertEqual([event["event"] for event in self.events], ["progress", "progress"])

        runner.trigger_natural_finish()  # last clip ends -> COMPLETED, one-shot
        completed = [event for event in self.events if event["event"] == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["session"]["state"], "completed")
        self.assertEqual(completed[0]["session"]["position"], 2)
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])  # no empty reloop

        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertIsNone(observed["session"])  # terminal -> registry cleared
        self.assertFalse(observed["preview_sounding"])

    def test_auto_advance_from_a_real_reaper_thread_completes_without_errors(self) -> None:
        """P15-S1 真机修复回归: production fires the natural-finish hook on the
        reaper thread -- a REAL worker thread must be able to advance the session
        clip after clip (zero SQLite affinity on that path: the live bug failed
        every auto-advance with ProgrammingError)."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_binding(BATCH_TRACK_D, "STORE-D")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B, BATCH_TRACK_D])
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        payload = self._preview_batch(service)
        self.assertEqual(payload["session"]["total"], 3)
        callback = runner.on_natural_finish
        self.assertIsNotNone(callback)

        def finish_one_clip() -> None:
            self.worker_thread = threading.get_ident()  # type: ignore[attr-defined]
            runner.active = False
            callback()

        # One natural finish per clip: clip 1 -> clip 2 -> clip 3 -> COMPLETED.
        for _ in range(3):
            worker = threading.Thread(target=finish_one_clip)
            worker.start()
            worker.join()
        # The hook genuinely ran off the main thread -- this regression test is real.
        self.assertNotEqual(self.worker_thread, threading.get_ident())
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL, PREVIEW_URL])
        progress = [event for event in self.events if event["event"] == "progress"]
        self.assertEqual(len(progress), 3)
        completed = [event for event in self.events if event["event"] == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["session"]["state"], "completed")
        self.assertEqual(completed[0]["session"]["skipped"], [])
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertIsNone(observed["session"])

    def test_batch_natural_completion_restores_interrupted_formal_playback(self) -> None:
        """P17 acceptance: the session's LAST natural end restores the formal
        playback its first clip interrupted -- mid-session advances never do."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        adapter = PausedAwareAdapter()
        service = self._service(adapter=adapter, runner=runner)
        payload = self._preview_batch(service)
        self.assertTrue(payload["started"])
        self.assertEqual(payload["suspended"]["player_state"], "playing")
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(adapter.read_state, "paused")

        runner.trigger_natural_finish()  # clip 1 -> clip 2: mid-session, no restore
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(adapter.read_state, "paused")

        runner.trigger_natural_finish()  # last clip -> COMPLETED + one restore
        self.assertEqual(adapter.calls, [("pause", ()), ("play", ())])
        self.assertEqual(adapter.read_state, "playing")

    def test_advance_to_completion_disarms_the_auto_restore(self) -> None:
        """The user skipped through the last clip: the preview ended on their
        command, not naturally -- the formal playback must stay paused."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        adapter = PausedAwareAdapter()
        service = self._service(adapter=adapter, runner=runner)
        self._preview_batch(service)
        runner.trigger_natural_finish()  # advance onto the last clip

        result = self._client(service).call("advance_preview", {})
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertTrue(result.payload["completed"])

        self.assertEqual(adapter.calls, [("pause", ())])  # never a ("play", ())
        self.assertEqual(adapter.read_state, "paused")

    def test_system_error_fails_the_session_instead_of_masquerading_as_skips(self) -> None:
        """P15-S1 真机修复: an infrastructure failure (the live cross-thread
        ProgrammingError) terminates the session as FAILED with the reason -- no
        per-item skip, no third-clip attempt, never a dishonest COMPLETED."""
        import sqlite3

        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_binding(BATCH_TRACK_D, "STORE-D")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B, BATCH_TRACK_D])
        runner = FailingPreviewRunner(
            sqlite3.ProgrammingError(
                "SQLite objects created in a thread can only be used in that same thread."
            ),
            from_call=2,
        )
        service = self._service(runner=runner)
        self._preview_batch(service)  # clip 1 starts fine

        def finish_one_clip() -> None:
            runner.active = False
            runner.on_natural_finish()

        worker = threading.Thread(target=finish_one_clip)
        worker.start()
        worker.join()
        self.assertEqual(runner.started, [PREVIEW_URL])  # no cascade attempt
        failed = [event for event in self.events if event["event"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["session"]["state"], "failed")
        self.assertEqual(failed[0]["session"]["failure_reason"], "ProgrammingError")
        self.assertEqual(failed[0]["session"]["skipped"], [])
        self.assertEqual(
            [event for event in self.events if event["event"] == "completed"], []
        )
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertIsNone(observed["session"])

    def test_transport_class_error_is_systemic_and_fails_the_session(self) -> None:
        """The skip class is exactly the item-unavailable class; a generic
        CatalogPreviewError (transport / afplay boundary) is systemic and fails."""
        from music_agent.catalog_preview import CatalogPreviewError

        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FailingPreviewRunner(
            CatalogPreviewError("preview command failed: afplay missing")
        )
        service = self._service(runner=runner)
        self._preview_batch(service)
        runner.trigger_natural_finish()
        self.assertEqual(runner.started, [PREVIEW_URL])
        failed = [event for event in self.events if event["event"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["session"]["state"], "failed")
        self.assertEqual(
            failed[0]["session"]["failure_reason"], "catalog_preview_failed"
        )
        self.assertEqual(
            [event for event in self.events if event["event"] == "completed"], []
        )

    def test_assembly_skips_unavailable_items(self) -> None:
        """route=unavailable entries are recorded at assembly, never failing the run."""
        self._seed_binding(BOUND_TRACK, "STORE-A")  # UNBOUND_TRACK stays unbound
        self._seed_runs([BOUND_TRACK, UNBOUND_TRACK])
        service = self._service()
        payload = self._preview_batch(service)
        session = payload["session"]
        self.assertEqual(session["total"], 1)
        self.assertEqual(session["current_canonical_id"], BOUND_TRACK)
        self.assertEqual(session["skipped"], [
            {
                "canonical_id": UNBOUND_TRACK,
                "name": "Albumless Study",
                "reason": "unavailable: no playable route",
            }
        ])

    def test_batch_with_no_previewable_items_fails_closed(self) -> None:
        self._seed_runs([UNBOUND_TRACK])  # only unbound tracks in the batch
        service = self._service()
        result = self._client(service).call("preview_batch", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "preview_session_unavailable")
        self.assertEqual(self.events, [])

    def test_batch_without_active_batch_fails_closed(self) -> None:
        service = self._service()
        result = self._client(service).call("preview_batch", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "preview_session_unavailable")

    def test_batch_suspends_playing_music_once_and_reports_it(self) -> None:
        """A4: the suspension happens once at session start; later clips no-op."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        adapter = StatefulPauseAdapter(now_pid="REAL-PID-1")
        runner = FakePreviewRunner()
        service = self._service(adapter=adapter, runner=runner)
        payload = self._preview_batch(service)
        self.assertEqual(adapter.calls, [("pause", ())])
        self.assertEqual(payload["suspended"], {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": True,
        })
        runner.trigger_natural_finish()  # second clip starts: no further pause
        self.assertEqual(adapter.calls, [("pause", ())])
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertEqual(observed["suspended"], payload["suspended"])
        self.assertEqual(observed["session"]["position"], 2)

    def test_runtime_lookup_failure_skips_and_keeps_going(self) -> None:
        """A7: one clip's URL resolution fails -> skipped + the next clip plays."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(
            runner=runner,
            source=MapPreviewSearchSource({"STORE-A": PREVIEW_URL, "STORE-B": None}),
        )
        self._preview_batch(service)
        runner.trigger_natural_finish()  # first ends; second fails to resolve
        self.assertEqual(runner.started, [PREVIEW_URL])
        completed = [event for event in self.events if event["event"] == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["session"]["state"], "completed")
        self.assertEqual(completed[0]["session"]["skipped"], [
            {
                "canonical_id": BATCH_TRACK_B,
                "name": "Synthetic Solo",
                "reason": "catalog_preview_unavailable",
            }
        ])

    def test_single_failing_item_ends_honest_all_skipped_session(self) -> None:
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_runs([BOUND_TRACK])
        runner = FakePreviewRunner()
        service = self._service(
            runner=runner,
            source=MapPreviewSearchSource({"STORE-A": None}),
        )
        payload = self._preview_batch(service)
        self.assertEqual(payload["session"]["state"], "completed")
        # P17-A2: started is derived from the post-start state, so a batch whose
        # items are all unavailable never reports 已启动.
        self.assertFalse(payload["started"])
        self.assertEqual(payload["session"]["skipped"], [
            {
                "canonical_id": BOUND_TRACK,
                "name": "Synthetic Duet",
                "reason": "catalog_preview_unavailable",
            }
        ])
        self.assertEqual(runner.started, [])

    def test_fast_systemic_failure_projects_failed_not_started(self) -> None:
        """P17-A2: dying on the very first clip returns a failed session snapshot --
        never started=True, never a running snapshot (no audio ever started)."""
        from music_agent.catalog_preview import CatalogPreviewError

        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_runs([BOUND_TRACK])
        runner = FailingPreviewRunner(
            CatalogPreviewError("preview command failed: afplay missing"),
            from_call=1,
        )
        service = self._service(runner=runner)
        payload = self._preview_batch(service)
        self.assertFalse(payload["started"])
        session = payload["session"]
        self.assertEqual(session["state"], "failed")
        self.assertEqual(session["failure_reason"], "catalog_preview_failed")
        self.assertEqual(runner.started, [])
        failed = [event for event in self.events if event["event"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["session"]["state"], "failed")

    def test_stop_preview_cancels_session_and_late_finish_is_a_noop(self) -> None:
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        self._preview_batch(service)
        stopped = self._client(service).call("stop_preview", {})
        self.assertEqual(stopped.outcome.value, "ok")
        self.assertTrue(stopped.payload["stopped"])
        self.assertTrue(stopped.payload["preview_session_cancelled"])
        cancelled = [event for event in self.events if event["event"] == "cancelled"]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["session"]["state"], "cancelled")
        # The reaper callback arrives late; the register is empty -> nothing advances.
        runner.trigger_natural_finish()
        self.assertEqual(runner.started, [PREVIEW_URL])
        self.assertEqual(len([e for e in self.events if e["event"] == "completed"]), 0)

    def test_single_preview_preempts_a_running_session(self) -> None:
        """A6: preview_catalog_track during a session cancels it, single clip plays."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_binding(UNBOUND_TRACK, "STORE-C")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        self._preview_batch(service)
        single = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(single.outcome.value, "ok")
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])
        cancelled = [event for event in self.events if event["event"] == "cancelled"]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["session"]["state"], "cancelled")
        # The single clip is NOT session-managed: a natural end advances nothing.
        runner.trigger_natural_finish()
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])

    def test_second_preview_batch_replaces_the_first(self) -> None:
        """A new「都放一遍」 re-derives the queue from the batch current at start,
        so a newer batch replaces a live session with a different queue.

        ``created_at`` (second resolution) can tie two fast seeds and then breaks
        randomly by run_id -- so the run ids are injected with the later seed
        lexicographically larger: the newer run wins under both the created_at
        ordering and its run_id tie-break, deterministically.
        """
        import uuid

        from music_agent.recommendation_contract import _RUN_ID_PREFIX

        suffix_a = str(uuid.uuid5(uuid.NAMESPACE_URL, "p15-seed-run-a"))
        suffix_b = str(uuid.uuid5(uuid.NAMESPACE_URL, "p15-seed-run-b"))
        earlier_id, later_id = sorted((suffix_a, suffix_b))
        run_ids = [f"{_RUN_ID_PREFIX}{earlier_id}", f"{_RUN_ID_PREFIX}{later_id}"]
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_D, "STORE-D")
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        with patch(
            "music_agent.recommendation_contract.generate_run_id",
            side_effect=iter(run_ids),
        ):
            self._seed_runs([BOUND_TRACK])
            first = self._preview_batch(service)  # only [BOUND_TRACK] exists yet
            self.assertEqual(first["session"]["total"], 1)
            self.assertEqual(first["session"]["current_canonical_id"], BOUND_TRACK)
            self._seed_runs([BATCH_TRACK_D])  # a newer batch arrives
        payload = self._preview_batch(service)
        self.assertEqual(payload["session"]["total"], 1)
        self.assertEqual(payload["session"]["current_canonical_id"], BATCH_TRACK_D)
        cancelled = [event for event in self.events if event["event"] == "cancelled"]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["session"]["state"], "cancelled")

    def test_advance_preview_skips_to_the_next_clip_in_place(self) -> None:
        """P15-S1 C02: 下一首 during a running 连播 advances the one live session
        in place -- no new session, no cancel, Music.app and the suspension
        register untouched (the formal-playback restore target survives the skip)."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        adapter = StatefulPauseAdapter(now_pid="REAL-PID-1")
        runner = FakePreviewRunner()
        service = self._service(adapter=adapter, runner=runner)
        payload = self._preview_batch(service)
        suspended = payload["suspended"]
        self.assertEqual(suspended["persistent_id"], "REAL-PID-1")

        result = self._client(service).call("advance_preview", {})
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertTrue(result.payload["advanced"])
        self.assertFalse(result.payload["completed"])
        self.assertEqual(result.payload["session"]["state"], "running")
        self.assertEqual(result.payload["session"]["position"], 2)
        self.assertEqual(result.payload["session"]["current_canonical_id"], BATCH_TRACK_B)
        # The sounding clip was stopped exactly once; the next one started.
        self.assertEqual(runner.stop_calls, 1)
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])
        # No cancel, no completed: the session never left RUNNING.
        self.assertEqual(
            [event["event"] for event in self.events], ["progress", "progress"]
        )
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertEqual(observed["session"]["state"], "running")
        self.assertEqual(observed["session"]["position"], 2)
        self.assertEqual(observed["suspended"], suspended)  # restore target intact
        self.assertEqual(adapter.calls, [("pause", ())])  # zero new Music.app contact

    def test_advance_preview_on_the_last_clip_completes_the_session(self) -> None:
        """Skipping ahead at the last clip exhausts the queue -- an honest
        COMPLETED (once), not a stuck running session, not a cancel."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(runner=runner)
        self._preview_batch(service)
        self._client(service).call("advance_preview", {})
        result = self._client(service).call("advance_preview", {})
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertFalse(result.payload["advanced"])
        self.assertTrue(result.payload["completed"])
        self.assertEqual(runner.stop_calls, 2)
        self.assertEqual(runner.started, [PREVIEW_URL, PREVIEW_URL])  # no empty reloop
        self.assertEqual(
            [event["event"] for event in self.events],
            ["progress", "progress", "completed"],
        )
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertIsNone(observed["session"])  # terminal -> registry cleared

    def test_advance_preview_next_item_unavailable_skips_and_completes(self) -> None:
        """C02 reuses the fa95d17 start path wholesale: an unavailable next clip
        stays an item-level skip, never a systemic session failure."""
        self._seed_binding(BOUND_TRACK, "STORE-A")
        self._seed_binding(BATCH_TRACK_B, "STORE-B")
        self._seed_runs([BOUND_TRACK, BATCH_TRACK_B])
        runner = FakePreviewRunner()
        service = self._service(
            runner=runner,
            source=MapPreviewSearchSource({"STORE-A": PREVIEW_URL, "STORE-B": None}),
        )
        self._preview_batch(service)
        result = self._client(service).call("advance_preview", {})
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertTrue(result.payload["advanced"])
        self.assertFalse(result.payload["completed"])  # exhaustion, not a cancel
        failed = [event for event in self.events if event["event"] == "failed"]
        self.assertEqual(failed, [])
        completed = [event for event in self.events if event["event"] == "completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["session"]["state"], "completed")
        self.assertEqual(completed[0]["session"]["skipped"], [
            {
                "canonical_id": BATCH_TRACK_B,
                "name": "Synthetic Solo",
                "reason": "catalog_preview_unavailable",
            }
        ])
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertIsNone(observed["session"])

    def test_advance_preview_without_a_live_session_fails_closed(self) -> None:
        """C02: the tool only exists to move a live session -- no session, no
        invented one; the router races settle as a typed execution error."""
        service = self._service()
        result = self._client(service).call("advance_preview", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "advance_preview_unavailable")
        self.assertEqual(self.events, [])

    def test_get_playback_context_with_no_activity_is_all_null(self) -> None:
        service = self._service()
        observed = self._client(service).call("get_playback_context", {}).payload
        self.assertEqual(observed["channel"], {"state": "none", "canonical_id": None})
        self.assertIsNone(observed["referent_canonical_id"])
        self.assertFalse(observed["preview_sounding"])
        self.assertIsNone(observed["player"])
        self.assertIsNone(observed["session"])
        self.assertIsNone(observed["suspended"])


class OpenInAppleMusicTest(unittest.TestCase):
    """在 Apple Music 中打开 -- resolve Apple's official track identity, then
    hand the concrete song to Music.app. The model passes only a canonical id;
    library-only tracks have no link and foreign URLs fail closed."""

    REAL_URL = "https://music.apple.com/us/album/reputation/1258917041?i=1258917044"
    ITUNES_ID = "1258917044"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def _seed_binding(self, track_id: str, itunes_id: str) -> None:
        """Add an itunes_store binding to one fixture track in place."""
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    def _service(
        self, policy: AgentClientPolicy = AgentClientPolicy.FULL, source=None
    ) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: policy}),
            playback_adapter=FakePlaybackAdapter(),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        return service

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def test_open_resolves_the_real_url_and_opens_it(self) -> None:
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakeOpenSearchSource({self.ITUNES_ID: self.REAL_URL})
        with patch("subprocess.run") as mocked_open:
            result = self._client(self._service(source=source)).call(
                "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
            )
        self.assertEqual(result.outcome.value, "ok", result.error_message)
        self.assertEqual(
            result.payload,
            {
                "command": "open_in_apple_music",
                "ok": True,
                "handoff_requested": True,
                "canonical_id": UNBOUND_TRACK,
                "url": self.REAL_URL,
                "client_url": "https://music.apple.com/us/song/1258917044",
                "source": "itunes_store_lookup",
            },
        )
        self.assertEqual(source.lookup_calls, [self.ITUNES_ID])
        mocked_open.assert_called_once()
        self.assertEqual(
            mocked_open.call_args.args[0],
            [
                "/usr/bin/open",
                "-a",
                "Music",
                "https://music.apple.com/us/song/1258917044",
            ],
        )

    def test_native_target_resolution_has_no_open_side_effect(self) -> None:
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakeOpenSearchSource({self.ITUNES_ID: self.REAL_URL})
        service = self._service(source=source)
        with patch("subprocess.run") as mocked_open:
            target = service.resolve_apple_music_target(UNBOUND_TRACK)
        self.assertEqual(
            target,
            {
                "canonical_id": UNBOUND_TRACK,
                "url": self.REAL_URL,
                "client_url": "https://music.apple.com/us/song/1258917044",
                "source": "itunes_store_lookup",
            },
        )
        self.assertEqual(source.lookup_calls, [self.ITUNES_ID])
        mocked_open.assert_not_called()

    def test_library_only_track_has_no_link(self) -> None:
        """No itunes_store binding: the link is absent -- honest, never searched."""
        source = FakeOpenSearchSource({})
        with patch("subprocess.run") as mocked_open:
            result = self._client(self._service(source=source)).call(
                "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
            )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "apple_music_open_unavailable")
        self.assertEqual(source.lookup_calls, [])
        mocked_open.assert_not_called()

    def test_source_without_track_view_lookup_reports_unavailable(self) -> None:
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakePreviewSearchSource("https://example.test/p.m4a")  # preview lookup only
        result = self._client(self._service(source=source)).call(
            "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "apple_music_open_unavailable")

    def test_lookup_without_url_reports_unavailable(self) -> None:
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakeOpenSearchSource({self.ITUNES_ID: None})
        with patch("subprocess.run") as mocked_open:
            result = self._client(self._service(source=source)).call(
                "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
            )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "apple_music_open_unavailable")
        mocked_open.assert_not_called()

    def test_non_apple_url_is_refused_and_never_opened(self) -> None:
        """Fail closed at the boundary even if a source returned a foreign URL."""
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakeOpenSearchSource({self.ITUNES_ID: "https://evil.example.com/x"})
        with patch("subprocess.run") as mocked_open:
            result = self._client(self._service(source=source)).call(
                "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
            )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "apple_music_open_failed")
        mocked_open.assert_not_called()

    def test_unknown_canonical_id_is_a_typed_miss(self) -> None:
        result = self._client(self._service(source=FakeOpenSearchSource({}))).call(
            "open_in_apple_music", {"canonical_id": "trk_ffffffff-ffff-4fff-8fff-ffffffffffff"}
        )
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "canonical_entity_not_found")

    def test_read_only_client_denied(self) -> None:
        self._seed_binding(UNBOUND_TRACK, self.ITUNES_ID)
        source = FakeOpenSearchSource({self.ITUNES_ID: self.REAL_URL})
        with patch("subprocess.run") as mocked_open:
            result = self._client(self._service(AgentClientPolicy.READ_ONLY, source)).call(
                "open_in_apple_music", {"canonical_id": UNBOUND_TRACK}
            )
        self.assertEqual(result.outcome.value, "permission_denied")
        mocked_open.assert_not_called()
