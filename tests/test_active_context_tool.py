"""P14-C06.2: get_active_context observation endpoint.

The tool is a pure, additive read: it composes the in-memory channel register
(same shape as get_now_playing's ``agent_channel``), the Music.app player snapshot
(``None`` when no playback adapter is wired), the ownership judgment shared with
get_now_playing, and a projection of the active recommendation batch. The batch
projection carries the P14-C07.3 provenance: the in-memory batch pointer resolves
first (``source=register``), the newest persisted run stands in when there is no
pointer (``source=derived``), and identity facts only -- never candidate/target
copies. These tests pin the endpoint contract, the permission boundary (READ class),
payload validation, and the structural agreement with get_now_playing -- whose own
contract carries forward untouched in test_playback_tools.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.repository import CanonicalRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_99999999-9999-4999-8999-999999999999"
BOUND_TRACK = "trk_11111111-1111-4111-8111-111111111111"  # SYNTH-TRACK-001
BOUND_TRACK_2 = "trk_22222222-2222-4222-8222-222222222222"  # SYNTH-TRACK-002
UNBOUND_TRACK = "trk_33333333-3333-4333-8333-333333333333"  # no persistent id


class FakePlaybackAdapter:
    def __init__(self, now_pid: str | None = "SYNTH-TRACK-001") -> None:
        self.now_pid = now_pid
        self.played: list[str] = []

    def read_now_playing(self):
        from music_agent.playback_control import NowPlaying, PlayerState

        return NowPlaying(
            state=PlayerState.PLAYING if self.now_pid is not None else PlayerState.STOPPED,
            persistent_id=self.now_pid,
            name="起风了 (旧版)" if self.now_pid is not None else None,
            artist="某艺人" if self.now_pid is not None else None,
            album="某专辑" if self.now_pid is not None else None,
        )

    def play_track(self, persistent_id: str) -> None:
        self.played.append(persistent_id)

    # remaining playback surface: wired-adapter contract, not exercised here
    def play(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def next_track(self) -> None:
        pass

    def previous_track(self) -> None:
        pass


class ReadFailAdapter:
    """Wired adapter whose read path fails -- the tool must surface a typed error."""

    def read_now_playing(self):
        raise RuntimeError("AppleEvent timed out")

    def play_track(self, persistent_id: str) -> None:
        raise AssertionError("not under test")

    def play(self) -> None:
        raise AssertionError("not under test")

    def pause(self) -> None:
        raise AssertionError("not under test")

    def next_track(self) -> None:
        raise AssertionError("not under test")

    def previous_track(self) -> None:
        raise AssertionError("not under test")


class FakePreviewRunner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stop_calls = 0
        self.active = False

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


class FakePreviewSearchSource:
    def __init__(self, url: str) -> None:
        self.url = url

    def search(self, term: str, limit: int):
        return []

    def lookup_preview_url(self, itunes_id: str) -> str:
        return self.url


class ActiveContextToolTest(unittest.TestCase):
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
        preview_runner=None,
    ) -> SharedAgentService:
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: policy}),
            playback_adapter=adapter,
            preview_runner=preview_runner,
            catalog_search_source=FakePreviewSearchSource("https://example.test/p.m4a")
            if preview_runner is not None
            else None,
        )
        self.addCleanup(service.close)
        return service

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test", label="tests"),
            service,
        )

    def _seed_run(self, track_id: str) -> str:
        return self._seed_run_items([track_id])

    def _seed_run_items(self, track_ids: list[str]) -> str:
        from datetime import timezone

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
                limit=len(track_ids),
            ),
            items=tuple(
                RecommendationItem(
                    candidate=Candidate(
                        candidate_id=f"cnd_aaaaaaaa-aaaa-4aaa-8aaa-{index:012d}",
                        target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                        source=CandidateSourceReference("test_seed", "context_judgment"),
                        eligibility=Eligibility.ELIGIBLE,
                    ),
                    score=ScoreBreakdown(0.9, (ScoreComponent("test", 0.9),)),
                )
                for index, track_id in enumerate(track_ids)
            ),
            produced_at=datetime.now(timezone.utc),
            contract_version=RECOMMENDATION_CONTRACT_VERSION,
        )
        with RecommendationHistoryRepository(self.database_path) as repository:
            repository.save_result(run)
        return run.run_id

    def _seed_itunes_binding(self, track_id: str, itunes_id: str) -> None:
        model = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        track = next(item for item in model["tracks"] if item["id"] == track_id)
        track["external_ids"]["itunes_store_id"] = itunes_id
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)

    # --- endpoint contract ------------------------------------------------------

    def test_fresh_service_reports_neutral_context(self) -> None:
        """No activity at all: everything is neutral, player is None without an adapter,
        and there is no active batch."""
        service = self._service(AgentClientPolicy.FULL, None)
        result = self._client(service).call("get_active_context", {})
        self.assertEqual(result.outcome.value, "ok")
        self.assertEqual(
            result.payload,
            {
                "channel": {"state": "none", "canonical_id": None},
                "referent_canonical_id": None,
                "preview_sounding": False,
                "player": None,
                "context": "unknown",
                "active_batch": None,
            },
        )

    def test_preview_sounding_is_runner_truth_not_the_channel_log(self) -> None:
        """P14-C06.3b: preview_sounding asks the runner, so it diverges from the
        channel register exactly when that divergence is the point."""
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SOME-OTHER-PID"), runner
        )
        client = self._client(service)
        client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        payload = client.call("get_active_context", {}).payload
        self.assertIs(payload["preview_sounding"], True)
        self.assertEqual(payload["channel"]["state"], "preview")
        # The runner's clip ends (e.g. natural exit): truth flips to False while the
        # action-log channel still says preview.
        runner.stop_preview()
        payload = client.call("get_active_context", {}).payload
        self.assertIs(payload["preview_sounding"], False)
        self.assertEqual(payload["channel"]["state"], "preview")

    def test_after_play_track_unifies_channel_player_and_ownership(self) -> None:
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        played = client.call("play_track", {"canonical_id": BOUND_TRACK})
        self.assertEqual(played.outcome.value, "ok")
        payload = client.call("get_active_context", {}).payload
        self.assertEqual(
            payload["channel"], {"state": "library", "canonical_id": BOUND_TRACK}
        )
        self.assertEqual(
            payload["player"],
            {
                "state": "playing",
                "persistent_id": "SYNTH-TRACK-001",
                "name": "起风了 (旧版)",
                "artist": "某艺人",
                "album": "某专辑",
                "canonical_id": BOUND_TRACK,
                "canonical_resolution": "binding",
            },
        )
        self.assertEqual(payload["context"], "agent_selected")
        self.assertIsNone(payload["active_batch"])

    def test_channel_agrees_with_get_now_playing_agent_channel(self) -> None:
        """The two endpoints expose the same channel register under the same shape."""
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        service = self._service(AgentClientPolicy.FULL, adapter)
        client = self._client(service)
        client.call("play_track", {"canonical_id": BOUND_TRACK})
        via_context = client.call("get_active_context", {}).payload
        via_now_playing = client.call("get_now_playing", {}).payload
        self.assertEqual(via_context["channel"], via_now_playing["agent_channel"])
        self.assertEqual(via_context["context"], via_now_playing["context"])

    def test_preview_flips_channel_to_preview_without_crashing_player(self) -> None:
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        runner = FakePreviewRunner()
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SOME-OTHER-PID"), runner
        )
        client = self._client(service)
        previewed = client.call("preview_catalog_track", {"canonical_id": UNBOUND_TRACK})
        self.assertEqual(previewed.outcome.value, "ok")
        payload = client.call("get_active_context", {}).payload
        self.assertEqual(payload["channel"], {"state": "preview", "canonical_id": UNBOUND_TRACK})
        # Player snapshot is independent of the preview channel; with no run and no
        # anchor there is no ownership evidence either way.
        self.assertIsNotNone(payload["player"])
        self.assertEqual(payload["context"], "unknown")

    def test_active_batch_derives_newest_run_without_pointer(self) -> None:
        """P14-C07.3 derived fallback: with no runtime pointer (fresh service, as
        after a restart) the newest persisted run stands in, marked as derived."""
        run_id = self._seed_run(BOUND_TRACK)
        service = self._service(AgentClientPolicy.FULL, None)
        payload = self._client(service).call("get_active_context", {}).payload
        batch = payload["active_batch"]
        self.assertIsNotNone(batch)
        self.assertEqual(batch["run_id"], run_id)
        self.assertEqual(batch["source"], "derived")
        self.assertEqual(batch["item_count"], 1)
        # Identity facts only -- deliberately no target/candidate/score/route copies.
        self.assertEqual(set(batch), {"run_id", "source", "produced_at", "item_count"})
        # produced_at is an ISO instant, not a raw datetime (wire-safe read).
        self.assertIsInstance(batch["produced_at"], str)
        datetime.fromisoformat(batch["produced_at"])

    def test_active_batch_register_pointer_wins_over_newer_history(self) -> None:
        """The pointer names the run this service delivered, even when newer runs
        exist in history that a derived read would otherwise pick."""
        run_id = self._seed_run(BOUND_TRACK)
        service = self._service(AgentClientPolicy.FULL, None)
        service._active_context.note_recommendation_batch(run_id)
        newer_run_id = self._seed_run(BOUND_TRACK_2)
        payload = self._client(service).call("get_active_context", {}).payload
        batch = payload["active_batch"]
        self.assertEqual(batch["run_id"], run_id)
        self.assertEqual(batch["source"], "register")
        self.assertEqual(batch["item_count"], 1)
        self.assertEqual(set(batch), {"run_id", "source", "produced_at", "item_count"})
        # The newer history run is untouched and still separately visible.
        fetched = self._client(service).call(
            "get_recommendation_run", {"run_id": newer_run_id}
        )
        self.assertEqual(fetched.outcome.value, "ok")

    def test_active_batch_pointer_that_does_not_resolve_falls_back_derived(self) -> None:
        """A pointer naming a run history does not hold degrades to the derived read
        (external store swap etc.) -- never a failed read or an invented batch."""
        run_id = self._seed_run(BOUND_TRACK)
        service = self._service(AgentClientPolicy.FULL, None)
        service._active_context.note_recommendation_batch(
            "rcm_99999999-9999-4999-8999-999999999999"
        )
        payload = self._client(service).call("get_active_context", {}).payload
        batch = payload["active_batch"]
        self.assertEqual(batch["run_id"], run_id)
        self.assertEqual(batch["source"], "derived")

    # --- P14-C07.3 write boundaries (item cursor via play_track / preview) ---------

    def test_play_track_hitting_active_run_records_item_index(self) -> None:
        run_id = self._seed_run_items([BOUND_TRACK, BOUND_TRACK_2])
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SYNTH-TRACK-002")
        )
        service._active_context.note_recommendation_batch(run_id)
        played = self._client(service).call(
            "play_track", {"canonical_id": BOUND_TRACK_2}
        )
        self.assertEqual(played.outcome.value, "ok")
        self.assertEqual(service._active_context.active_run_id, run_id)
        self.assertEqual(service._active_context.active_item_index, 1)

    def test_play_track_outside_active_run_clears_item_index_keeps_pointer(self) -> None:
        run_id = self._seed_run(BOUND_TRACK)
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SYNTH-TRACK-002")
        )
        service._active_context.note_recommendation_batch(run_id)
        service._active_context.note_batch_item(0)
        played = self._client(service).call(
            "play_track", {"canonical_id": BOUND_TRACK_2}
        )
        self.assertEqual(played.outcome.value, "ok")
        self.assertEqual(service._active_context.active_run_id, run_id)
        self.assertIsNone(service._active_context.active_item_index)

    def test_play_track_without_pointer_never_invents_item_index(self) -> None:
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        )
        played = self._client(service).call(
            "play_track", {"canonical_id": BOUND_TRACK}
        )
        self.assertEqual(played.outcome.value, "ok")
        self.assertIsNone(service._active_context.active_run_id)
        self.assertIsNone(service._active_context.active_item_index)

    def test_preview_hitting_active_run_records_item_index(self) -> None:
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        run_id = self._seed_run_items([BOUND_TRACK, UNBOUND_TRACK])
        runner = FakePreviewRunner()
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SOME-OTHER-PID"), runner
        )
        service._active_context.note_recommendation_batch(run_id)
        previewed = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(previewed.outcome.value, "ok")
        self.assertEqual(service._active_context.active_run_id, run_id)
        self.assertEqual(service._active_context.active_item_index, 1)

    def test_preview_outside_active_run_clears_item_index_keeps_pointer(self) -> None:
        self._seed_itunes_binding(BOUND_TRACK_2, "111000222")
        self._seed_itunes_binding(UNBOUND_TRACK, "296103682")
        run_id = self._seed_run(BOUND_TRACK_2)
        runner = FakePreviewRunner()
        service = self._service(
            AgentClientPolicy.FULL, FakePlaybackAdapter(now_pid="SOME-OTHER-PID"), runner
        )
        service._active_context.note_recommendation_batch(run_id)
        service._active_context.note_batch_item(0)
        previewed = self._client(service).call(
            "preview_catalog_track", {"canonical_id": UNBOUND_TRACK}
        )
        self.assertEqual(previewed.outcome.value, "ok")
        self.assertEqual(service._active_context.active_run_id, run_id)
        self.assertIsNone(service._active_context.active_item_index)

    def test_next_track_stop_preview_never_touch_batch_context(self) -> None:
        """Queue navigation and preview stop are deliberately outside the C07.3 write
        boundary: the batch pointer and item cursor both survive them."""
        run_id = self._seed_run_items([BOUND_TRACK, BOUND_TRACK_2])
        adapter = FakePlaybackAdapter(now_pid="SYNTH-TRACK-001")
        runner = FakePreviewRunner()
        service = self._service(AgentClientPolicy.FULL, adapter, runner)
        service._active_context.note_recommendation_batch(run_id)
        service._active_context.note_batch_item(1)
        client = self._client(service)
        for tool in ("next_track", "previous_track", "pause", "play", "stop_preview"):
            result = client.call(tool, {})
            self.assertEqual(result.outcome.value, "ok", tool)
            self.assertEqual(service._active_context.active_run_id, run_id, tool)
            self.assertEqual(service._active_context.active_item_index, 1, tool)

    def test_stopped_player_is_readable_with_context_unknown(self) -> None:
        adapter = FakePlaybackAdapter(now_pid=None)
        service = self._service(AgentClientPolicy.FULL, adapter)
        payload = self._client(service).call("get_active_context", {}).payload
        self.assertEqual(payload["player"]["state"], "stopped")
        self.assertIsNone(payload["player"]["persistent_id"])
        self.assertEqual(payload["context"], "unknown")

    def test_adapter_read_failure_becomes_typed_error(self) -> None:
        """A wired adapter that fails to read is a typed tool error, as in
        get_now_playing -- never a silently-omitted player."""
        service = self._service(AgentClientPolicy.FULL, ReadFailAdapter())
        result = self._client(service).call("get_active_context", {})
        self.assertEqual(result.outcome.value, "execution_error")
        self.assertEqual(result.error_code, "playback_command_failed")

    # --- permission boundary ----------------------------------------------------

    def test_read_only_client_allowed(self) -> None:
        service = self._service(AgentClientPolicy.READ_ONLY, None)
        result = self._client(service).call("get_active_context", {})
        self.assertEqual(result.outcome.value, "ok")

    def test_none_client_denied(self) -> None:
        service = self._service(AgentClientPolicy.NONE, FakePlaybackAdapter())
        result = self._client(service).call("get_active_context", {})
        self.assertEqual(result.outcome.value, "permission_denied")

    # --- payload validation -----------------------------------------------------

    def test_extra_payload_keys_refused(self) -> None:
        service = self._service(AgentClientPolicy.FULL, None)
        result = self._client(service).call("get_active_context", {"foo": "bar"})
        self.assertEqual(result.outcome.value, "invalid_request")


if __name__ == "__main__":
    unittest.main()
