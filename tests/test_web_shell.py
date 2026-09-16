"""P17-B shell tests: view-model projections + the loopback HTTP surface.

Suite strategy: ``standalone`` mode (no UDS anywhere, sandbox-safe) with a
fake provider, a fake playback adapter and a fixture-seeded store. Endpoints
are exercised over real localhost HTTP; the tool semantics behind each button
stay covered by their own suites (playback tools, provider agent, preview).
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.agent_contract import AgentToolOutcome
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.output_sanitizer import _EMPTY_TEXT_FALLBACK
from music_agent.playback_control import NowPlaying, PlayerState
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.provider_agent import (
    ProviderAgentResult,
    ProviderLoopToolExecution,
    RECOMMENDATION_UNFULFILLED_FALLBACK,
)
from music_agent.provider_contract import (
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
    ProviderToolCall,
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
from music_agent.repository import CanonicalRepository
from music_agent.web_shell import (
    SHELL_COMMANDS,
    ShellConfig,
    ShellEventLog,
    ShellProjection,
    WebShellApp,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_99999999-9999-4999-8999-999999999999"
BOUND_TRACK = "trk_11111111-1111-4111-8111-111111111111"  # has persistent id
UNBOUND_CATALOG_TRACK = "trk_33333333-3333-4333-8333-333333333333"  # no bindings at all


def _seed_model(database_path: Path) -> None:
    with CanonicalRepository(database_path) as repository:
        repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _seed_recommendation_run(
    database_path: Path, canonical_id: str
) -> str:
    """Seed one recommendation run; returns its run id (P19-T14-B tests
    compare the /api/cards payload against this identity)."""
    item = RecommendationItem(
        candidate=Candidate(
            candidate_id="cnd_00000000-0000-4000-8000-000000000001",
            target=PreferenceTargetReference(
                PreferenceTargetKind.TRACK, canonical_id
            ),
            source=CandidateSourceReference("web-shell-test", "seed"),
            eligibility=Eligibility.ELIGIBLE,
        ),
        score=ScoreBreakdown(total=0.5, components=(ScoreComponent("probe", 0.5),)),
    )
    run_id = generate_run_id()
    result = RecommendationResult(
        run_id=run_id,
        request=RecommendationRequest(
            context=RecommendationContext(
                now=datetime.now(timezone.utc), preference_inputs=()
            ),
            recommended_kind=RecommendedItemKind.TRACK,
            limit=1,
        ),
        items=(item,),
        produced_at=datetime.now(timezone.utc),
        contract_version=RECOMMENDATION_CONTRACT_VERSION,
    )
    with RecommendationHistoryRepository(database_path) as history:
        history.save_result(result)
    return result.run_id


class _FakePlaybackAdapter:
    """play/pause/next_track/previous_track/play_track + read_now_playing."""

    def __init__(self, now_playing: NowPlaying | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.now_playing = now_playing

    def play(self):
        self.calls.append(("play", ()))
        self._set_state(PlayerState.PLAYING)

    def pause(self):
        self.calls.append(("pause", ()))
        self._set_state(PlayerState.PAUSED)

    def next_track(self):
        self.calls.append(("next_track", ()))

    def previous_track(self):
        self.calls.append(("previous_track", ()))

    def play_track(self, **kwargs):
        self.calls.append(("play_track", tuple(kwargs.items())))

    def read_now_playing(self):
        if self.now_playing is None:
            return NowPlaying(state=PlayerState.STOPPED)
        return self.now_playing

    def _set_state(self, state: PlayerState) -> None:
        current = self.now_playing
        if current is None:
            self.now_playing = NowPlaying(state=state)
            return
        self.now_playing = NowPlaying(
            state=state,
            persistent_id=current.persistent_id,
            name=current.name,
            artist=current.artist,
            album=current.album,
        )


class _FakeProvider:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def chat(self, system, messages, tools):
        return ProviderResponse(
            ProviderMessage(
                ProviderMessageRole.ASSISTANT, text=self.replies.pop(0)
            ),
            ProviderStopReason.END_TURN,
            {"input_tokens": 1, "output_tokens": 2},
        )


class _ProseLoop:
    """Stand-in for ProviderAgentLoop returning reply text without any
    generation tool execution -- the T14-B live failure shape (a long
    hand-enumerated pseudo-list from an ordinary analysis round)."""

    def __init__(
        self, final_text: str, *, rounds_capped: bool = False
    ) -> None:
        self._final_text = final_text
        self._rounds_capped = rounds_capped
        self.calls: list[str] = []
        self.turn_plans: list[object] = []

    def run(self, text: str, *, turn_plan=None) -> ProviderAgentResult:
        self.calls.append(text)
        self.turn_plans.append(turn_plan)
        return ProviderAgentResult(
            final_text=self._final_text,
            rounds=1,
            tool_executions=(),
            context_trimmed=False,
            rounds_capped=self._rounds_capped,
        )


class _GeneratingLoop:
    """Stand-in for ProviderAgentLoop: seeds ONE recommendation run inside
    run() (between the reply door's before/after history reads, exactly
    where a real generation would persist it) and records a successful
    generate_recommendation execution -- exercising the batch-attach path
    without a live provider or preference state."""

    def __init__(self, database_path: Path, final_text: str) -> None:
        self._database_path = database_path
        self._final_text = final_text
        self.calls: list[str] = []
        self.turn_plans: list[object] = []

    def run(self, text: str, *, turn_plan=None) -> ProviderAgentResult:
        self.calls.append(text)
        self.turn_plans.append(turn_plan)
        run_id = _seed_recommendation_run(self._database_path, BOUND_TRACK)
        return ProviderAgentResult(
            final_text=self._final_text,
            recommendation_payload={"run_id": run_id},
            rounds=1,
            tool_executions=(
                ProviderLoopToolExecution(name="generate_recommendation", outcome="ok"),
            ),
            context_trimmed=False,
            rounds_capped=False,
        )


class _StopFamilyClient:
    """Stand-in P09 client for the T14-A fast path: get_playback_context
    answers with the scripted runner truth, stop_preview answers with the
    scripted stop result, and every call is recorded. Loop.client uses this
    in standalone and routed modes alike, so the fast path's authority read
    must go through exactly this surface."""

    def __init__(
        self,
        *,
        preview_sounding: bool = False,
        session_state: str | None = None,
        stop_payload: dict | None = None,
        stop_outcome: AgentToolOutcome = AgentToolOutcome.OK,
        channel_state: str = "preview",
        channel_canonical_id: str | None = None,
        referent_canonical_id: str | None = None,
        preview_payload: dict | None = None,
        preview_outcome: AgentToolOutcome = AgentToolOutcome.OK,
        play_payload: dict | None = None,
        play_outcome: AgentToolOutcome = AgentToolOutcome.OK,
        now_playing_payload: dict | None = None,
        suspended: dict | None = None,
        navigation_outcome: AgentToolOutcome = AgentToolOutcome.OK,
        advance_payload: dict | None = None,
        advance_outcome: AgentToolOutcome = AgentToolOutcome.OK,
    ) -> None:
        self.preview_sounding = preview_sounding
        self.session_state = session_state
        self.stop_payload = stop_payload if stop_payload is not None else {"stopped": True}
        self.stop_outcome = stop_outcome
        self.channel_state = channel_state
        self.channel_canonical_id = channel_canonical_id
        self.referent_canonical_id = referent_canonical_id
        self.preview_payload = (
            preview_payload if preview_payload is not None else {"started": True}
        )
        self.preview_outcome = preview_outcome
        self.play_payload = play_payload if play_payload is not None else {"ok": True}
        self.play_outcome = play_outcome
        self.now_playing_payload = now_playing_payload
        self.suspended = suspended
        self.navigation_outcome = navigation_outcome
        self.advance_payload = (
            advance_payload
            if advance_payload is not None
            else {"advanced": True, "completed": False}
        )
        self.advance_outcome = advance_outcome
        self.played_canonical_id: str | None = None
        self.control_state: str | None = None
        self.calls: list[tuple[str, dict]] = []
        self.fail_read = False
        self.fail_stop = False

    def call(self, tool: str, payload: dict):
        self.calls.append((tool, payload))
        if self.fail_read and tool == "get_playback_context":
            raise RuntimeError("transport down")
        if self.fail_stop and tool == "stop_preview":
            raise RuntimeError("transport down")
        if tool == "get_playback_context":
            session = {"state": self.session_state} if self.session_state else {}
            return SimpleNamespace(
                payload={
                    "channel": {
                        "state": self.channel_state,
                        "canonical_id": self.channel_canonical_id,
                    },
                    "referent_canonical_id": self.referent_canonical_id,
                    "preview_sounding": self.preview_sounding,
                    "session": session,
                    "suspended": self.suspended,
                },
                outcome=AgentToolOutcome.OK,
            )
        if tool == "stop_preview":
            return SimpleNamespace(
                payload=self.stop_payload, outcome=self.stop_outcome
            )
        if tool == "preview_catalog_track":
            return SimpleNamespace(
                payload=self.preview_payload, outcome=self.preview_outcome
            )
        if tool == "play_track":
            self.played_canonical_id = payload.get("canonical_id")
            return SimpleNamespace(
                payload=self.play_payload, outcome=self.play_outcome
            )
        if tool in {"next_track", "previous_track"}:
            return SimpleNamespace(
                payload={"command": tool, "ok": True},
                outcome=self.navigation_outcome,
            )
        if tool == "advance_preview":
            return SimpleNamespace(
                payload=self.advance_payload,
                outcome=self.advance_outcome,
            )
        if tool in {"play", "pause"}:
            self.control_state = "playing" if tool == "play" else "paused"
            return SimpleNamespace(
                payload={"command": tool, "ok": True},
                outcome=AgentToolOutcome.OK,
            )
        if tool == "get_now_playing":
            target_id = self.played_canonical_id
            response = self.now_playing_payload
            if response is None:
                if target_id is None and self.control_state is not None:
                    response = {"now_playing": {"state": self.control_state}}
                else:
                    response = {
                        "now_playing": {
                            "state": "playing",
                            "name": "Synthetic Track",
                            "artist": "Synthetic Artist",
                        },
                        "agent_channel": {
                            "state": "library",
                            "canonical_id": target_id,
                        },
                        "player_canonical_id": target_id,
                        "canonical_resolution": "binding",
                    }
            return SimpleNamespace(payload=response, outcome=AgentToolOutcome.OK)
        raise AssertionError(f"unexpected tool {tool}")


class _StopFamilyLoop(_ProseLoop):
    """ProviderAgentLoop stand-in whose client carries the scripted stop
    truth; reaching run() proves the fast path fell through."""

    def __init__(self, client: _StopFamilyClient) -> None:
        super().__init__("好的。")
        self.client = client


class _PlayPreviewLoop(_ProseLoop):
    """ProviderAgentLoop stand-in whose run() returns a scripted
    tool-execution trace (the T14-E door's evidence) and whose client records
    the stop call."""

    def __init__(
        self,
        client: _StopFamilyClient,
        *,
        final_text: str = "好的。",
        executions: tuple = (),
        action_attempt=None,
    ) -> None:
        super().__init__(final_text)
        self.client = client
        self._executions = tuple(executions)
        self._action_attempt = action_attempt

    def run(self, text: str, *, turn_plan=None) -> ProviderAgentResult:
        self.calls.append(text)
        self.turn_plans.append(turn_plan)
        return ProviderAgentResult(
            final_text=self._final_text,
            rounds=1,
            tool_executions=self._executions,
            context_trimmed=False,
            rounds_capped=False,
            action_attempt=self._action_attempt,
        )


class ShellProjectionTest(unittest.TestCase):
    def test_web_shell_default_client_id_is_canonical(self) -> None:
        # Live-smoke regression: the CLI's default web client id must pass the
        # agent contract (agt_ prefix + canonical UUID suffix) or startup dies.
        from music_agent.agent_contract import validate_client_id
        from music_agent.cli import _WEB_SHELL_CLIENT_ID

        validate_client_id(_WEB_SHELL_CLIENT_ID)  # raises on violation

    def test_now_playing_projects_player_and_session(self) -> None:
        projected = ShellProjection.now_playing(
            {
                "now_playing": {
                    "state": "playing",
                    "name": "起风了 (旧版)",
                    "artist": "某艺人",
                    "album": "某专辑",
                }
            },
            {
                "session": {
                    "state": "running",
                    "position": 2,
                    "total": 5,
                    "current_name": "第二首",
                }
            },
        )
        self.assertEqual(projected["state"], "playing")
        self.assertEqual(projected["name"], "起风了 (旧版)")
        self.assertEqual(projected["preview_session"]["position"], 2)
        self.assertEqual(projected["preview_session"]["current_name"], "第二首")

    def test_formal_player_projects_only_cheap_player_fields(self) -> None:
        projected = ShellProjection.formal_player(
            {
                "now_playing": {
                    "state": "playing",
                    "name": "起风了 (旧版)",
                    "artist": "某艺人",
                    "album": "某专辑",
                },
                "agent_channel": {"canonical_id": BOUND_TRACK},
            }
        )
        self.assertEqual(
            projected,
            {
                "state": "playing",
                "name": "起风了 (旧版)",
                "artist": "某艺人",
                "album": "某专辑",
            },
        )

    def test_now_playing_projects_preview_sounding_and_suspension(self) -> None:
        # P18-S1.2: the runtime's in-flight truth + the honest interruption
        # reach the UI; the suspension projection is display-only (identity
        # facts like the persistent id never cross into the UI).
        projected = ShellProjection.now_playing(
            {"now_playing": {"state": "paused", "name": "起风了 (旧版)"}},
            {
                "preview_sounding": True,
                "preview_suspension": {
                    "player_state": "playing",
                    "persistent_id": "REAL-PID-1",
                    "name": "起风了 (旧版)",
                    "pause_ok": True,
                },
            },
        )
        self.assertTrue(projected["preview_sounding"])
        self.assertEqual(
            projected["suspended"],
            {"name": "起风了 (旧版)", "player_state": "playing", "pause_ok": True},
        )

    def test_now_playing_ignores_sticky_restore_memo_after_preview_terminal(self) -> None:
        projected = ShellProjection.now_playing(
            {"now_playing": {"state": "playing", "name": "起风了 (旧版)"}},
            {
                "preview_sounding": False,
                "suspended": {
                    "player_state": "playing",
                    "name": "起风了 (旧版)",
                    "pause_ok": True,
                },
                "preview_suspension": None,
            },
        )
        self.assertIsNone(projected["suspended"])

    def test_now_playing_hides_note_when_formal_pause_did_not_succeed(self) -> None:
        projected = ShellProjection.now_playing(
            {"now_playing": {"state": "playing", "name": "起风了 (旧版)"}},
            {
                "preview_sounding": True,
                "preview_suspension": {
                    "player_state": "playing",
                    "name": "起风了 (旧版)",
                    "pause_ok": False,
                },
            },
        )
        self.assertIsNone(projected["suspended"])

    def test_now_playing_defaults_preview_fields_without_context(self) -> None:
        projected = ShellProjection.now_playing(
            {"now_playing": {"state": "playing"}}, None
        )
        self.assertFalse(projected["preview_sounding"])
        self.assertIsNone(projected["suspended"])

    def test_now_playing_degrades_honestly_on_missing_payloads(self) -> None:
        projected = ShellProjection.now_playing(None, None)
        self.assertIsNone(projected["state"])
        self.assertIsNone(projected["name"])
        self.assertIsNone(projected["preview_session"])

    def test_cards_keep_only_ui_fields(self) -> None:
        cards = ShellProjection.cards(
            [
                {
                    "target_id": BOUND_TRACK,
                    "name": "Synthetic Duet",
                    "artist_name": "某艺人",
                    "album": "Synthetic Duet - Single",
                    "playback": {"route": "library", "label": "可正式播放"},
                    "fresh_this_request": True,
                    "candidate_id": "cnd_internal",
                    "score": 0.9,
                }
            ]
        )
        self.assertEqual(len(cards), 1)
        self.assertEqual(
            set(cards[0]),
            {"canonical_id", "name", "artist", "album", "route", "apple_music_openable"},
        )
        self.assertEqual(cards[0]["route"], "library")
        self.assertEqual(cards[0]["canonical_id"], BOUND_TRACK)
        self.assertEqual(cards[0]["album"], "Synthetic Duet - Single")

    def test_cards_apple_music_openable_follows_binding_map(self) -> None:
        # P18-S2: the action gate projects the backend's own fail-closed fact
        # (itunes_store binding presence) -- never inferred from the route.
        items = [
            {"target_id": BOUND_TRACK, "name": "有绑定", "playback": {"route": "library"}},
            {"target_id": UNBOUND_CATALOG_TRACK, "name": "无绑定", "playback": {"route": "library"}},
        ]
        cards = ShellProjection.cards(
            items, {BOUND_TRACK: True, UNBOUND_CATALOG_TRACK: False}
        )
        by_id = {card["canonical_id"]: card for card in cards}
        self.assertIs(by_id[BOUND_TRACK]["apple_music_openable"], True)
        self.assertIs(by_id[UNBOUND_CATALOG_TRACK]["apple_music_openable"], False)

    def test_cards_apple_music_openable_defaults_false_without_map(self) -> None:
        # Fail-closed: absent map means the button can never appear, so the UI
        # could not offer an action the backend would refuse.
        cards = ShellProjection.cards(
            [{"target_id": BOUND_TRACK, "name": "x", "playback": {"route": "preview_only"}}]
        )
        self.assertIs(cards[0]["apple_music_openable"], False)

    def test_cards_album_defaults_to_empty_string(self) -> None:
        cards = ShellProjection.cards(
            [{"target_id": BOUND_TRACK, "name": "x", "playback": {}}]
        )
        self.assertEqual(cards[0]["album"], "")

    def test_cards_skip_items_without_target_id(self) -> None:
        cards = ShellProjection.cards([{"name": "无标识"}])
        self.assertEqual(cards, [])

    def test_cards_default_unavailable_route(self) -> None:
        cards = ShellProjection.cards(
            [{"target_id": BOUND_TRACK, "name": "x", "playback": {}}]
        )
        self.assertEqual(cards[0]["route"], "unavailable")

    def test_command_error_none_on_ok(self) -> None:
        result = SimpleNamespace(
            outcome=AgentToolOutcome.OK, payload={"ok": True}
        )
        self.assertIsNone(ShellProjection.command_error(result))

    def test_command_error_maps_failure_envelope(self) -> None:
        result = SimpleNamespace(
            outcome=AgentToolOutcome.EXECUTION_ERROR,
            error_code="agent_runtime_offline",
            error_message="测试错误",
            payload=None,
        )
        self.assertEqual(
            ShellProjection.command_error(result),
            {"code": "agent_runtime_offline", "message": "测试错误"},
        )

    def test_event_log_tail_polling(self) -> None:
        log = ShellEventLog(capacity=2)
        log.present({"event": "progress", "session": {"position": 1}})
        log.present({"event": "progress", "session": {"position": 2}})
        log.present({"event": "completed"})
        entries = log.snapshot(1)
        self.assertEqual([entry["seq"] for entry in entries], [2, 3])
        self.assertEqual(entries[-1]["event"]["event"], "completed")


class WebShellAppTest(unittest.TestCase):
    def test_chat_uses_loop_resolved_turn_plan_and_passes_same_object_to_run(self):
        from music_agent.intent_router import (
            RecommendationTurnSemantics,
            TurnExpectedResult,
            TurnPlan,
            TurnPrimarySemantic,
            TurnSemanticSource,
            TurnTaskSurface,
        )

        real_client = self.app._loop.client

        class ResolverLoop:
            def __init__(inner):
                inner.client = real_client
                inner.resolved = None
                inner.received = None

            def resolve_turn(inner, text):
                inner.resolved = TurnPlan(
                    user_text=text,
                    primary=TurnPrimarySemantic.RECOMMENDATION,
                    expected_result=TurnExpectedResult.RECOMMENDATION_BATCH,
                    task_surface=TurnTaskSurface.RECOMMENDATION,
                    recommendation=RecommendationTurnSemantics(
                        mode="generic",
                        target=None,
                        target_kind=None,
                        scene=None,
                        requested_count=5,
                    ),
                    semantic_source=TurnSemanticSource.LLM_INTERPRETED,
                )
                return inner.resolved

            def run(inner, text, *, turn_plan=None):
                inner.received = turn_plan
                return ProviderAgentResult(
                    final_text="没有生成批次。",
                    rounds=1,
                    tool_executions=(),
                    context_trimmed=False,
                    rounds_capped=False,
                )

        loop = ResolverLoop()
        self.app._loop = loop
        status, body = self._request("POST", "/api/chat", {"text": "推荐"})
        self.assertEqual(status, 200)
        self.assertIs(loop.received, loop.resolved)
        self.assertEqual(loop.received.semantic_source.value, "llm_interpreted")
        self.assertIsNone(body["batch"])

    def test_real_generation_count_and_run_identity_survive_every_backend_layer(self):
        from unittest.mock import patch
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import PreferencePersistenceRepository
        from music_agent.source_observation import ObservedValue
        artist_alpha_second = "trk_22222222-2222-4222-8222-222222222222"
        for track_id in (BOUND_TRACK, artist_alpha_second):
            with PreferencePersistenceRepository(self.database_path) as repository:
                repository.record_observation(
                    SignalIdentity(PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                                   "apple_music", "favorited"),
                    ObservedValue.value(True), observed_at="2026-09-05T00:00:00+00:00",
                    provenance="presentation-integration")

        class Provider:
            calls = []
            def chat(inner, system, messages, tools):
                inner.calls.append((messages, tools))
                if len(inner.calls) == 1:
                    args = json.dumps({"target_ids": [BOUND_TRACK, UNBOUND_CATALOG_TRACK],
                                       "limit": 5})
                    return ProviderResponse(ProviderMessage(ProviderMessageRole.ASSISTANT,
                        tool_calls=(ProviderToolCall("gen", "generate_recommendation", args),)),
                        ProviderStopReason.TOOL_USE, {})
                return ProviderResponse(ProviderMessage(ProviderMessageRole.ASSISTANT,
                    text="<final_answer>为你推荐这两首歌。</final_answer>"),
                    ProviderStopReason.END_TURN, {})

        real_loop = self.app._loop
        provider = Provider()
        real_loop.provider = provider
        class RecordingLoop:
            client = real_loop.client
            result = None
            def run(inner, text, *, turn_plan=None):
                inner.result = real_loop.run(text, turn_plan=turn_plan)
                return inner.result
        recording = RecordingLoop()
        self.app._loop = recording

        integration_run_id = "rcm_20202020-2020-4020-8020-202020202020"
        with patch("music_agent.agent_service.generate_run_id", return_value=integration_run_id):
            status, body = self._request(
                "POST", "/api/chat", {"text": "推荐 Artist Alpha 的音乐"}
            )
        self.assertEqual(status, 200)
        delivered = next(message for message in reversed(provider.calls[1][0])
                         if message.tool_results is not None)
        tool_envelope = json.loads(delivered.tool_results[0].content)
        tool_payload = tool_envelope["payload"]
        result_payload = recording.result.recommendation_payload
        run_id = tool_payload["run_id"]
        self.assertEqual(run_id, integration_run_id)
        self.assertFalse(tool_envelope["replayed"])
        self.assertEqual(tool_payload["item_count"], 2)
        self.assertEqual(len(tool_payload["items"]), 2)
        self.assertEqual(result_payload["run_id"], run_id)
        self.assertEqual(result_payload["item_count"], 2)
        self.assertEqual(len(result_payload["items"]), 2)
        self.assertEqual(body["batch"]["run_id"], run_id)
        self.assertEqual(len(body["batch"]["cards"]), 2)

    def test_frontend_executes_attached_batches_without_latest_poll_ownership(self):
        import re
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is required for the production JavaScript execution test")
        html = (Path(__file__).parents[1] / "src/music_agent/ui/index.html").read_text()
        functions = []
        for name in ("deliverChat", "rememberBatch", "openDetail", "renderExpanded",
                     "collapseToTrace", "renderTrace", "cardButtonCard", "pollCards"):
            start = re.search(r"(?:async )?function " + name + r"\(", html).start()
            end = re.search(r"\n(?:async )?function ", html[start + 1:]).start() + start + 1
            functions.append(html[start:end])
        script = r'''
const assert = require('node:assert/strict');
function element() { return {children: [], style: {}, value: '', scrollTop: 0,
  classList: { values: new Set(['off']), add(x){this.values.add(x)}, remove(x){this.values.delete(x)},
    contains(x){return this.values.has(x)}, toggle(x,on){on ? this.add(x) : this.remove(x)} },
  append(...xs){this.children.push(...xs)}, replaceChildren(){this.children=[]},
  insertBefore(x){this.children.push(x)}, focus(){} }; }
const elements = new Map(); const $ = id => {if(!elements.has(id))elements.set(id,element());return elements.get(id)};
const document = {createElement: element, querySelector: () => null};
let batchCache=null, batchHistory=[], expandedKey=null, liveTurnSeq=null, turnSeq=0,
 stageReply=false, stageProseReply=null, lastReplyText='', lastUserText=null,
 pendingExpandCards=null, chatBusy=false, lastState={}, traceStacksEl=null,
 detailReturnScroll=0, queuedDraft=null, turnBlocks=new Map(), lastPreviewTrack=null;
const input=element(), sendBtn=element();
const errors=[]; const showError=x=>errors.push(x);
const bannerOff=()=>{}, archiveLivePair=()=>{}, appendMessage=()=>{}, setThinking=()=>{},
 renderConversation=()=>{}, revealCompletedTurn=()=>{}, pollState=()=>{}, refreshFullState=()=>{}, syncTurnTraces=()=>{},
 traceButtonFor=()=>{}, renderQueue=()=>{};
const batchKey=cards=>cards.map(c=>c.canonical_id).join('|'), requestLabel=x=>x,
 traceText=(x,n)=>`${x}: ${n}`, chatLead=x=>x, looksLikeNumberedSongList=()=>false;
const commands=[]; const markPending=()=>{}, runCommand=(...args)=>commands.push(args);
let response, polled=null;
const post=async()=>response, jsonOr=async x=>x, fetchCardData=async()=>polled;
''' + "\n".join(functions) + r'''
(async()=>{
 const cards5=Array.from({length:5},(_,i)=>({canonical_id:'track-'+i,name:'Song '+i,
   artist:'Artist',route:i===0?'library':'preview_only',apple_music_openable:true}));
 const cards2=cards5.slice(0,2);
  const batch=(id,label='')=>({run_id:id,cards,label});
 let cards=cards2;
 polled={id:'historical',key:batchKey(cards),cards};
 await pollCards(); assert.equal(batchHistory.length,0); // reload is quiet
 response={reply:'为你推荐这 2 首。',batch:batch('X','类似 Yorushika')};
 await deliverChat('推荐类似 Yorushika 的音乐');
 assert.equal(batchCache.turnSeq,liveTurnSeq);
 assert.equal(batchCache.cards.length,2);
 assert.equal($('detail-cards').children.length,2);
 assert.equal($('detail-lead').textContent,'类似 Yorushika: 2');
 assert.equal($('rec-detail').classList.contains('off'),false);
 const actions=$('detail-cards').children[1].children[0].children[2].children;
 actions[0].onclick(); actions[1].onclick();
 assert.deepEqual(commands.map(c=>c[0]),['preview_catalog_track','open_in_apple_music']);
 response={reply:'找到两条资料库记录。',batch:null};
 await deliverChat('搜索 Spring Thief Yorushika');
 assert.equal($('rec-detail').classList.contains('off'),true);
 assert.equal($('recs').classList.contains('off'),true);
 assert.equal(batchHistory.length,1);
 cards=cards5;
 response={reply:'新的推荐。',batch:batch('Y')}; await deliverChat('推荐几首歌');
 assert.equal(batchCache.id,'Y'); assert.equal($('detail-cards').children.length,5);
 // A successful replay is current-turn-owned even if its id was seen earlier.
 response={reply:'这一批推荐。',batch:batch('Y')}; await deliverChat('推荐几首歌');
 assert.equal(batchCache.turnSeq,liveTurnSeq);
 assert.equal($('rec-detail').classList.contains('off'),false);
 assert.deepEqual(errors,[]);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result = subprocess.run([node, "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_provider_loop_terminal_content_is_extracted_at_http_boundary(self):
        from music_agent.deepseek_provider import _parse_response
        answer = "找到《Spring Thief》— Yorushika，你的资料库中有两个可正式播放的版本。"
        raw = "Let me note the context...\n<final_answer>" + answer + "</final_answer>"

        class WireProvider:
            def chat(self, system, messages, tools):
                return _parse_response(200, json.dumps({"choices": [{"message": {
                    "content": raw},
                    "finish_reason": "stop"}]}))

        self.app._loop.provider = WireProvider()
        status, body = self._request("POST", "/api/chat", {"text": "搜索 Spring Thief Yorushika"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], answer)
        self.assertNotIn("Let me", body["reply_html"])
        self.assertIsNone(body["batch"])
        from music_agent.final_response_boundary import FINAL_RESPONSE_FALLBACKS
        for raw, expected in (
            ("Let me note the context...\nThis appears to be a search request...",
             FINAL_RESPONSE_FALLBACKS["default"]),
            ("I found two versions in your Apple Music library.",
             "I found two versions in your Apple Music library."),
        ):
            status, body = self._request("POST", "/api/chat", {"text": "搜索 Spring Thief Yorushika"})
            self.assertEqual(status, 200)
            self.assertEqual(body["reply"], expected)
            self.assertIsNone(body["batch"])

    def test_real_loop_generation_owns_exact_run_even_when_it_is_not_latest(self):
        from music_agent.agent_contract import AgentToolResult
        from music_agent.provider_contract import ProviderToolCall
        from unittest.mock import patch
        from uuid import uuid4
        original = self.app._loop.client.call
        owned = []

        def call(name, arguments, **kwargs):
            if name != "generate_recommendation":
                return original(name, arguments, **kwargs)
            run_id = _seed_recommendation_run(self.database_path, BOUND_TRACK)
            owned.append(run_id)
            # Another durable run must not steal this request's presentation.
            _seed_recommendation_run(self.database_path, BOUND_TRACK)
            return AgentToolResult(request_id="req_" + str(uuid4()), tool="generate_recommendation",
                outcome=AgentToolOutcome.OK, payload={"run_id": run_id, "item_count": 1,
                    "items": [{"target_id": BOUND_TRACK}]},
                error_code=None, error_message=None, completed_at=datetime.now(timezone.utc))

        class GeneratingProvider:
            rounds = 0
            def chat(self, system, messages, tools):
                self.rounds += 1
                if self.rounds == 1:
                    return ProviderResponse(ProviderMessage(ProviderMessageRole.ASSISTANT,
                        tool_calls=(ProviderToolCall("gen", "generate_recommendation", "{}"),)),
                        ProviderStopReason.TOOL_USE, {})
                return ProviderResponse(ProviderMessage(ProviderMessageRole.ASSISTANT,
                    text="<final_answer>为你推荐这几首歌。</final_answer>"), ProviderStopReason.END_TURN, {})

        self.app._loop.provider = GeneratingProvider()
        with patch.object(self.app._loop.client, "call", side_effect=call):
            status, body = self._request("POST", "/api/chat", {"text": "推荐几首歌"})
        self.assertEqual(status, 200)
        self.assertEqual(body["batch"]["run_id"], owned[0])
        self.assertNotEqual(owned[0], self._newest_run_id())
        self.assertEqual(len(body["batch"]["cards"]), 1)
        # A successful tool journal replay creates no new history row. The
        # old before/after-latest gate returned batch=null in this case.
        def replay(name, arguments, **kwargs):
            if name != "generate_recommendation":
                return original(name, arguments, **kwargs)
            return AgentToolResult(request_id="req_" + str(uuid4()), tool=name,
                outcome=AgentToolOutcome.OK, payload={"run_id": owned[0], "item_count": 1,
                    "items": [{"target_id": BOUND_TRACK}]},
                error_code=None, error_message=None, completed_at=datetime.now(timezone.utc),
                replayed=True)
        self.app._loop.provider = GeneratingProvider()
        newest_before = self._newest_run_id()
        with patch.object(self.app._loop.client, "call", side_effect=replay):
            status, body = self._request("POST", "/api/chat", {"text": "推荐几首歌"})
        self.assertEqual(status, 200)
        self.assertEqual(self._newest_run_id(), newest_before)
        self.assertEqual(body["batch"]["run_id"], owned[0])
        self.assertEqual(len(body["batch"]["cards"]), 1)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        _seed_model(self.database_path)
        playback = _FakePlaybackAdapter(
            NowPlaying(
                state=PlayerState.PLAYING,
                persistent_id="REAL-PID-1",
                name="起风了 (旧版)",
                artist="某艺人",
                album="某专辑",
            )
        )
        self.adapter = playback
        self.config = ShellConfig(
            database_path=self.database_path,
            provider_factory=lambda: _FakeProvider(["好的。"]),
            agent_client=(CLIENT_ID, "full"),
            mode="standalone",
            service_factory=lambda: SharedAgentService(
                self.database_path,
                clients=AgentClientRegistry(
                    {CLIENT_ID: AgentClientPolicy("full")}
                ),
                playback_adapter=playback,
            ),
        )
        self.app = WebShellApp(self.config)
        self.app.start()
        self.serve_thread = threading.Thread(
            target=self.app.serve_forever, daemon=True
        )
        self.serve_thread.start()
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.app.close()
        self.app.stop_httpd()
        self.serve_thread.join(timeout=5.0)

    def _request(self, method: str, path: str, body: dict | None = None):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.app.port}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def test_index_served_with_visual_tokens(self) -> None:
        request = urllib.request.Request(f"http://127.0.0.1:{self.app.port}/")
        with urllib.request.urlopen(request, timeout=30) as response:
            document = response.read().decode("utf-8")
        self.assertIn("音乐助手", document)
        self.assertIn("--bg:", document)

    def test_health_reports_mode(self) -> None:
        status, body = self._request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["mode"], "standalone")

    def test_chat_round_trip_through_provider_loop(self) -> None:
        status, body = self._request("POST", "/api/chat", {"text": "你好"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")

    def test_chat_rejects_empty_message(self) -> None:
        status, body = self._request("POST", "/api/chat", {"text": "  "})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "empty_message")

    def test_direction_shift_success_renders_deterministically(self) -> None:
        # P20-Fix10 §三 Q5 / §十七: the Fix05 direction-shift web fast path
        # presents the shifted batch through the SAME deterministic presenter
        # as a provider-loop success -- evidence reasons + closing cue under
        # the Fix05 direction note (its semantic is unchanged; the prior
        # name-only lines stay as the contract-miss fallback).
        from tests.test_direction_coach import J1, M1, M2, ScriptedClient
        from tests.test_provider_agent import evidence_item

        generated = [
            evidence_item(
                "夜风",
                artist_name="J Artist",
                basis=[{"kind": "genre", "label": "J-Pop", "provenance": "推断"}],
            ),
            evidence_item(
                "夜里",
                artist_name="J Artist",
                basis=[{"kind": "genre", "label": "J-Pop", "provenance": "推断"}],
                fresh_this_request=True,
                route="preview_only",
            ),
        ]
        client = ScriptedClient(
            true_genres=("J-Pop", "Rock"),
            true_scope=(M1, M2, J1),
            generated=generated,
        )

        class _ShiftLoop:
            def __init__(self, loop_client):
                self.client = loop_client

        self.app._loop = _ShiftLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "再来一批，换个方向。"})
        self.assertEqual(status, 200)
        reply = body["reply"]
        self.assertIn("已换到「J-Pop」方向", reply)
        self.assertIn("1. 夜风 — J Artist\n这首按 J-Pop 方向推断出来。", reply)
        self.assertTrue(reply.endswith("需要试听哪一首，直接告诉我。"), reply)
        self.assertEqual(reply.count("夜风"), 1)  # one list, no duplication
        for banned in ("novel", "fresh=", "mechanism", "provenance", "basis", "score"):
            self.assertNotIn(banned, reply)
        self.assertIsNone(body["batch"])  # the temp DB cards source has no such run

    def test_explanation_line_renders_deterministically_without_the_provider(self) -> None:
        # P20-Fix11 §十四 Q: a closed explanation line through the SAME
        # /api/chat door renders the deterministic explanation from the active
        # run through the web fast path -- the provider loop is NEVER invoked
        # (its run() would fail this test), so no SSL traceback and no
        # planning narration can exist on this path.
        from tests.test_explanation_coach import ExplanationScriptedClient

        class _ExplanationLoop:
            def __init__(self, loop_client):
                self.client = loop_client

        client = ExplanationScriptedClient()
        self.app._loop = _ExplanationLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "为什么推荐这些？"})
        self.assertEqual(status, 200)
        reply = body["reply"]
        self.assertIn("这一批主要来自 Anime 方向。", reply)
        self.assertIn("1. Anime Song — Anime Artist\n这首按 Anime 方向推断出来。", reply)
        self.assertIn("这是本次新发现，目前没有更直接的偏好匹配证据。", reply)
        self.assertIsNone(body["batch"])
        self.assertFalse(body["rounds_capped"])
        for banned in ("score", "满分", "rcm_", "cnd_", "trk_", "mechanism",
                       "provenance", "basis", "让我补充", "让我核对", "SSL"):
            self.assertNotIn(banned, reply)
        self.assertEqual(
            client.calls,
            [
                ("get_active_context", {}),
                ("get_recommendation_run", {"run_id": "rcm_active"}),
            ],
        )

    def test_full_reply_surface_never_leaks_tool_ids(self) -> None:
        # The shell hands the provider loop's final_text through verbatim; the
        # body adds no tool/internal surface of its own. P18-S1.1 adds the ONE
        # extra channel: server-rendered safe Markdown (reply_html). P19-T14-B
        # adds ``batch`` -- the run+cards the chat just generated, present as
        # an explicit null when this turn generated nothing.
        status, body = self._request("POST", "/api/chat", {"text": "测试"})
        self.assertEqual(status, 200)
        self.assertEqual(
            set(body), {"reply", "reply_html", "rounds_capped", "batch"}
        )
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(body["reply_html"], "<p>好的。</p>")
        self.assertIsNone(body["batch"])

    def test_fix04_reply_door_scrubs_internal_tokens_and_process_phrases(
        self,
    ) -> None:
        # P20 Fix 04: the web /api/chat door shares the output gate with the
        # CLI print site. The B/E doors decide on the RAW text above; the
        # presented reply + reply_html are scrubbed — internal ids, route
        # fields, count fields and process narration never reach the UI,
        # while the natural capability wording survives.
        self.app._loop = _ProseLoop(
            "用户明确说「播放第二首」，这是一个正式播放请求。\n"
            "该曲 playback.route 为 preview_only。\n"
            "本批（rcm_11111111-1111-4111-8111-111111111111）共 5 首，"
            "fresh_item_count = 5。"
        )
        status, body = self._request("POST", "/api/chat", {"text": "早上好"})
        self.assertEqual(status, 200)
        self.assertEqual(
            body["reply"],
            "该曲 只能试听 30 秒。\n"
            "本批（推荐编号）共 5 首，这 5 首都是本次新发现。",
        )
        for forbidden in (
            "rcm_",
            "playback.route",
            "preview_only",
            "fresh_item_count",
            "用户明确说",
        ):
            self.assertNotIn(forbidden, body["reply"])
            self.assertNotIn(forbidden, body["reply_html"])
        self.assertIn("只能试听 30 秒", body["reply_html"])
        self.assertIn("这 5 首都是本次新发现", body["reply_html"])

    def test_fix04_reply_door_falls_back_instead_of_empty_reply(self) -> None:
        # P20 Fix 04 G at the web door: a reply consisting only of process
        # narration scrubs to the stable honest sentence — the payload is
        # never an empty string.
        self.app._loop = _ProseLoop("用户明确说「播放」，按规则我不能自动试听。")
        status, body = self._request("POST", "/api/chat", {"text": "早上好"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], _EMPTY_TEXT_FALLBACK)
        self.assertEqual(body["reply_html"], f"<p>{_EMPTY_TEXT_FALLBACK}</p>")

    # ---- P20-Fix08: the final response boundary at the web door ----

    def test_fix08_reply_door_blocks_abbreviated_ids_and_process_block(self) -> None:
        # The UAT "为什么这些适合我？" leak verbatim-shape: abbreviated
        # internal ids plus a whole process-narration block. The web door's
        # Layer-2 validator sends the explanation-shaped fallback, and neither
        # the ids nor the process text reach reply or reply_html.
        from music_agent.final_response_boundary import FINAL_RESPONSE_FALLBACKS

        self.app._loop = _ProseLoop(
            "让我核对一下结果。\n"
            "rcm_1b4e0124 的候选 trk_a7827505 与 trk_860deec5 来自 Rock。\n"
            "向用户如实呈现这一结果。\n"
        )
        status, body = self._request("POST", "/api/chat", {"text": "为什么这一批适合我？"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], FINAL_RESPONSE_FALLBACKS["explanation"])
        for forbidden in (
            "rcm_1b4e0124", "trk_a7827505", "trk_860deec5", "rcm_", "trk_",
            "让我核对", "向用户",
        ):
            self.assertNotIn(forbidden, body["reply"])
            self.assertNotIn(forbidden, body["reply_html"])

    def test_fix08_reply_door_preserves_clean_explanation_tail(self) -> None:
        # Sec.10: the useful final content after the process block survives
        # whole -- never a word-by-word patch of broken leftovers.
        self.app._loop = _ProseLoop(
            "让我核对一下工具结果。\n"
            "根据你的偏好记录，这 5 首分别来自这些方向：\n"
            "• Every Breath You Take → Rock\n"
            "• Girls Just Want to Have Fun → Pop\n"
        )
        status, body = self._request("POST", "/api/chat", {"text": "为什么这一批适合我？"})
        self.assertEqual(status, 200)
        self.assertNotIn("让我核对", body["reply"])
        self.assertIn("Every Breath You Take → Rock", body["reply"])
        self.assertIn("Girls Just Want to Have Fun → Pop", body["reply"])
        self.assertNotIn("让我核对", body["reply_html"])

    # ---- P19-T14-B: the reply door (recommendation never degrades to prose) ----

    def test_t14b_hanataba_similar_request_gets_one_sentence_fallback(self) -> None:
        # The Hanataba live failure verbatim: 「找类似这首的」 produced long
        # numbered prose and no batch. Without a generation success this turn
        # the long prose must never reach the UI -- the answer is exactly the
        # one honest fallback sentence, and no batch pretends to exist.
        self.app._loop = _ProseLoop(
            "我根据《Hanataba》为你找了几首气质相近的歌曲：\n\n"
            "1. HAPPY BIRTHDAY — back number：活泼的摇滚情歌。\n"
            "2. 高嶺の花子さん — back number：……\n"
            "是否试听？"
        )
        status, body = self._request("POST", "/api/chat", {"text": "找类似这首的"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], RECOMMENDATION_UNFULFILLED_FALLBACK)
        self.assertEqual(body["reply_html"], "<p>" + RECOMMENDATION_UNFULFILLED_FALLBACK + "</p>")
        self.assertFalse(body["rounds_capped"])
        self.assertIsNone(body["batch"])
        self.assertEqual(self.app._loop.calls, ["找类似这首的"])

    def test_t14b_numbered_pseudo_list_suppressed_for_unenumerated_phrasing(
        self,
    ) -> None:
        # Second line of defense: a reply the model shaped like a numbered
        # song list collapses to the same one sentence even when the user's
        # wording sits outside the closed intent set (the shape probe, not
        # the intent table, triggers here).
        self.app._loop = _ProseLoop(
            "1. 夜に駆ける — YOASOBI：……\n2. 群青 — YOASOBI：……"
        )
        status, body = self._request("POST", "/api/chat", {"text": "来点适合深夜的"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], RECOMMENDATION_UNFULFILLED_FALLBACK)
        self.assertIsNone(body["batch"])

    def test_t14b_plain_prose_passes_for_non_recommendation(self) -> None:
        # The door only shuts on the recommendation surface -- ordinary
        # conversation prose (no generation, no numbered list) is untouched.
        self.app._loop = _ProseLoop("好的，我记住了。")
        status, body = self._request("POST", "/api/chat", {"text": "谢谢"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的，我记住了。")
        self.assertIsNone(body["batch"])

    def test_collection_preference_statement_is_not_misread_as_recommendation(self) -> None:
        self.app._loop = _ProseLoop("明白，你喜欢 Yorushika 的歌。")
        status, body = self._request(
            "POST", "/api/chat", {"text": "我喜欢 Yorushika 的歌"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "明白，你喜欢 Yorushika 的歌。")
        self.assertIsNone(body["batch"])

    def test_web_shell_passes_its_turn_plan_to_the_provider_consumer(self) -> None:
        from music_agent.intent_router import TurnPrimarySemantic

        loop = _ProseLoop("明白，你喜欢 Yorushika 的歌。")
        self.app._loop = loop
        status, _body = self._request(
            "POST", "/api/chat", {"text": "我喜欢 Yorushika 的歌"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(len(loop.turn_plans), 1)
        self.assertEqual(
            loop.turn_plans[0].primary,
            TurnPrimarySemantic.PREFERENCE_STATEMENT,
        )

    def test_t14b_capped_closeout_passes_through_untouched(self) -> None:
        # A rounds-capped turn already carries the loop's own short closeout;
        # the door never rewrites it into the fallback.
        self.app._loop = _ProseLoop(
            "抱歉，这一轮内容有点多，先到这里吧。", rounds_capped=True
        )
        status, body = self._request("POST", "/api/chat", {"text": "找类似这首的"})
        self.assertEqual(status, 200)
        self.assertTrue(body["rounds_capped"])
        self.assertEqual(body["reply"], "抱歉，这一轮内容有点多，先到这里吧。")
        self.assertIsNone(body["batch"])

    def test_t14b_generation_success_attaches_batch_run_id_and_cards(self) -> None:
        # Structured success: the batch travels ON the chat reply -- run id
        # identity + projected cards -- for the UI's takeover on the same
        # fetch, and the reply text itself stays untouched (the door only
        # shuts on failure).
        self.app._loop = _GeneratingLoop(self.database_path, "这几首和那首气质相近：")
        status, body = self._request("POST", "/api/chat", {"text": "找类似这首的"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "这几首和那首气质相近：")
        batch = body["batch"]
        self.assertIsNotNone(batch)
        self.assertIsInstance(batch["run_id"], str)
        self.assertEqual(batch["run_id"], self._newest_run_id())
        self.assertEqual(len(batch["cards"]), 1)
        self.assertEqual(
            set(batch["cards"][0]),
            {"canonical_id", "name", "artist", "album", "route", "apple_music_openable"},
        )
        self.assertEqual(batch["cards"][0]["canonical_id"], BOUND_TRACK)

    def test_fresh_discovery_generation_also_attaches_current_turn_cards(self) -> None:
        self.app._loop = _GeneratingLoop(self.database_path, "为你找到了这批新歌：")
        status, body = self._request(
            "POST", "/api/chat", {"text": "找点我没听过的歌"}
        )
        self.assertEqual(status, 200)
        self.assertIsNotNone(body["batch"])
        self.assertEqual(body["batch"]["run_id"], self._newest_run_id())

    def test_natural_target_recommendation_without_generation_hits_same_honest_door(self) -> None:
        self.app._loop = _ProseLoop("我可以先给你列几个名字。")
        status, body = self._request(
            "POST", "/api/chat", {"text": "推荐 Yorushika 的音乐"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], RECOMMENDATION_UNFULFILLED_FALLBACK)
        self.assertIsNone(body["batch"])

    def test_plain_recommendation_request_attaches_current_turn_cards(self) -> None:
        self.app._loop = _GeneratingLoop(self.database_path, "为你推荐这几首：")
        status, body = self._request(
            "POST", "/api/chat", {"text": "推荐几首歌"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["batch"]["cards"]), 1)

    def test_shortcut_and_natural_language_share_exact_run_ownership(self) -> None:
        seen_run_ids = []
        for text in (
            "推荐音乐",
            "推荐几首歌",
            "推荐 Yorushika 的音乐",
            "给我推荐几首 Yorushika 的歌",
            "推荐类似 Yorushika 的音乐",
            "我喜欢 Yorushika，给我推荐几首歌",
            "推荐几首适合晚上听的 Yorushika 的歌",
            "我喜欢 Yorushika，推荐几首适合晚上听的歌",
            "找几首和 Spring Thief 类似的歌",
        ):
            with self.subTest(text=text):
                self.app._loop = _GeneratingLoop(self.database_path, "为你推荐这几首：")
                status, body = self._request("POST", "/api/chat", {"text": text})
                self.assertEqual(status, 200)
                self.assertIsNotNone(body["batch"])
                self.assertEqual(body["batch"]["run_id"], self._newest_run_id())
                self.assertEqual(len(body["batch"]["cards"]), 1)
                seen_run_ids.append(body["batch"]["run_id"])
        self.assertEqual(len(set(seen_run_ids)), len(seen_run_ids))

    def test_successful_current_turn_payload_is_the_presentation_authority(self) -> None:
        # The exact payload produced by this invocation is stronger ownership
        # evidence than a second text classifier. Real search/playback surfaces
        # structurally omit generation tools; this forced loop proves only that
        # a same-turn successful run cannot be reduced to prose at the web door.
        self.app._loop = _GeneratingLoop(self.database_path, "它在资料库中。")
        status, body = self._request(
            "POST", "/api/chat", {"text": "Spring Thief 在我的 Apple Music 资料库里吗？"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "它在资料库中。")
        self.assertEqual(body["batch"]["run_id"], self._newest_run_id())

    def test_explicit_similarity_label_travels_with_exact_batch(self) -> None:
        self.app._loop = _GeneratingLoop(self.database_path, "为你推荐这几首：")
        status, body = self._request(
            "POST", "/api/chat", {"text": "推荐类似 Yorushika 的音乐"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["batch"]["run_id"], self._newest_run_id())
        self.assertEqual(body["batch"]["label"], "类似 Yorushika")

    def test_t14b_same_items_new_run_still_attaches_a_new_batch(self) -> None:
        # Identity-not-key regression: two runs whose items project to the
        # same item key are still two different batches -- the second reply
        # carries the new run id (the takeover must re-expand).
        self.app._loop = _GeneratingLoop(self.database_path, "先来这几首：")
        status, first = self._request("POST", "/api/chat", {"text": "找类似这首的"})
        self.assertEqual(status, 200)
        self.app._loop = _GeneratingLoop(self.database_path, "换一批，也是近似的：")
        status, second = self._request("POST", "/api/chat", {"text": "找类似这首的"})
        self.assertEqual(status, 200)
        self.assertIsNotNone(first["batch"])
        self.assertIsNotNone(second["batch"])
        self.assertNotEqual(second["batch"]["run_id"], first["batch"]["run_id"])
        self.assertEqual(second["batch"]["cards"], first["batch"]["cards"])

    # ---- P19-T14-A: stop-family fast path (a claimed stop must really stop) ----

    def test_t14a_stop_preview_runs_on_the_authority_and_answers_honestly(self) -> None:
        # 停止试听 never reaches the provider loop. The reply is derived from
        # the AUTHORITY's own stop result (stopped=true here): the success
        # sentence, zero provider rounds, zero fabricated fields.
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "停止试听"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已停止试听。")
        self.assertEqual(body["reply_html"], "<p>已停止试听。</p>")
        self.assertFalse(body["rounds_capped"])
        self.assertIsNone(body["batch"])
        self.assertEqual(client.calls, [("stop_preview", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14a_pause_preview_with_nothing_sounding_says_so(self) -> None:
        # 暂停试听 is a plain table form (no context read). stopped=false --
        # the authority's honest idempotent no-op -- yields the fixed
        # "nothing is playing" sentence; still zero provider rounds.
        client = _StopFamilyClient(stop_payload={"stopped": False})
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "暂停试听"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "当前没有正在播放的试听。")
        self.assertEqual(client.calls, [("stop_preview", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14a_bu_ting_le_reads_the_authority_truth_first(self) -> None:
        # 不听了 is context-sensitive: the fast path reads the live Playback
        # Context through the same routed client the loop uses (the authority
        # truth -- NOT the local chat-side snapshot) and only then routes.
        # preview_sounding=true flips it to stop_preview and executes it.
        client = _StopFamilyClient(preview_sounding=True)
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "不听了"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已停止试听。")
        self.assertEqual(
            client.calls,
            [("get_playback_context", {}), ("stop_preview", {})],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_t14a_bu_ting_le_without_preview_falls_to_the_loop(self) -> None:
        # No sounding preview: the form routes to None, and the provider loop
        # answers exactly as before -- the fast path must never invent a stop.
        client = _StopFamilyClient(preview_sounding=False)
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "不听了"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(self.app._loop.calls, ["不听了"])
        self.assertEqual(client.calls, [("get_playback_context", {})])

    def test_t14a_plain_pause_uses_authoritative_state_readback(self) -> None:
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "暂停"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已暂停播放。")
        self.assertEqual(self.app._loop.calls, [])
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("pause", {}),
                ("get_now_playing", {}),
            ],
        )

    def test_t14a_plain_pause_during_a_live_session_stops_the_preview(self) -> None:
        # Routing rule unchanged (P15-S1): while a preview session literally
        # runs, plain 暂停 ≙ stop_preview. The web shell now HONORS that route
        # exactly the way chat-session does -- the session-aware meaning was
        # always stop-the-preview; only the split-brain ever broke it.
        client = _StopFamilyClient(session_state="running")
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "暂停"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已停止试听。")
        self.assertEqual(
            client.calls,
            [("get_playback_context", {}), ("stop_preview", {})],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_continue_with_suspension_bypasses_provider_and_reads_back(self) -> None:
        client = _StopFamilyClient(
            suspended={
                "player_state": "playing",
                "persistent_id": "REAL-PID-1",
                "name": "起风了",
                "pause_ok": True,
            }
        )
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "继续播放"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已继续播放。")
        self.assertEqual(self.app._loop.calls, [])
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play", {}),
                ("get_now_playing", {}),
            ],
        )

    def test_p22_continue_without_suspension_fails_closed_without_provider(self) -> None:
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "继续播放"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "没有可恢复的播放。")
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_continue_during_running_preview_reports_progress_without_play(self) -> None:
        client = _StopFamilyClient(session_state="running")
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "继续播放"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "连播进行中。")
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_next_during_running_preview_advances_preview_not_music_app(self) -> None:
        client = _StopFamilyClient(session_state="running")
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "下一首"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已切换到下一首试听。")
        self.assertEqual(
            client.calls,
            [("get_playback_context", {}), ("advance_preview", {})],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_next_off_preview_uses_music_app_authority_without_provider(self) -> None:
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "下一首"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已切换到下一首。")
        self.assertEqual(
            client.calls,
            [("get_playback_context", {}), ("next_track", {})],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_previous_uses_music_app_authority_without_provider(self) -> None:
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "上一首"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已切换到上一首。")
        self.assertEqual(client.calls, [("previous_track", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_closed_playback_failure_never_falls_back_to_provider(self) -> None:
        client = _StopFamilyClient(
            navigation_outcome=AgentToolOutcome.EXECUTION_ERROR
        )
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "上一首"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "切换上一首失败，当前播放状态未确认。")
        self.assertEqual(client.calls, [("previous_track", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_p22_context_failure_fails_closed_without_provider(self) -> None:
        client = _StopFamilyClient()
        client.fail_read = True
        self.app._loop = _StopFamilyLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "下一首"})

        self.assertEqual(status, 200)
        self.assertEqual(
            body["reply"], "暂时无法确认当前播放或试听状态，未执行下一首。"
        )
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14a_failed_truth_read_refuses_to_fabricate(self) -> None:
        # An unreadable Playback Context degrades to the provider loop --
        # never a fabricated "已停止" success (fail-closed, CLI parity).
        client = _StopFamilyClient()
        client.fail_read = True
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "不听了"})
        self.assertEqual(status, 200)
        self.assertEqual(self.app._loop.calls, ["不听了"])
        self.assertEqual(body["reply"], "好的。")

    def test_t14a_failing_stop_execute_never_claims_success(self) -> None:
        # A non-OK stop outcome (registries fail closed) hands the line back
        # to the provider loop instead of claiming the preview stopped.
        client = _StopFamilyClient(stop_outcome=AgentToolOutcome.INVALID_REQUEST)
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "停止试听"})
        self.assertEqual(status, 200)
        self.assertEqual(self.app._loop.calls, ["停止试听"])
        self.assertEqual(client.calls, [("stop_preview", {})])
        self.assertEqual(body["reply"], "好的。")

    # ---- P19-T14-E: a play-intent turn must never leave preview audio ----

    def test_t14e_named_play_degraded_to_preview_stops_it_and_answers_honestly(self) -> None:
        # 播放 Hanataba -> play_track failed -> the model self-healed into a
        # preview: the door stops the preview through the authority and
        # replaces the reply with the contract's honest sentence.
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(
                ProviderLoopToolExecution("play_track", "error"),
                ProviderLoopToolExecution("preview_catalog_track", "ok"),
            ),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放 Hanataba"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(
            body["reply_html"], f"<p>{PLAY_PREVIEW_DOWNGRADE_FALLBACK}</p>"
        )
        self.assertFalse(body["rounds_capped"])
        self.assertIsNone(body["batch"])
        self.assertEqual(self.app._loop.calls, ["播放 Hanataba"])
        self.assertEqual(client.calls, [("stop_preview", {})])

    def test_s21_followup_structured_named_play_offer_arms_session_and_accepts_exact_target(self) -> None:
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        offer_text = "《Wendy》— Test Artist 目前无法正式播放，可以试听 30 秒。需要我开始试听吗？"
        offer = OfferedAction(
            kind=PREVIEW_TRACK,
            target_canonical_id=BOUND_TRACK,
            source="structured_named_play_resolution",
            verified_title="Wendy",
            verified_artist="Test Artist",
        )
        client = _StopFamilyClient()

        class StructuredOfferLoop(_PlayPreviewLoop):
            def run(inner, text: str, *, turn_plan=None) -> ProviderAgentResult:
                inner.calls.append(text)
                inner.turn_plans.append(turn_plan)
                return ProviderAgentResult(
                    final_text=offer_text,
                    rounds=0,
                    tool_executions=(),
                    context_trimmed=False,
                    rounds_capped=False,
                    offered_action=offer,
                )

        loop = StructuredOfferLoop(client)
        self.app._loop = loop

        status, first = self._request("POST", "/api/chat", {"text": "播放 Wendy"})
        self.assertEqual(status, 200)
        self.assertEqual(first["reply"], offer_text)
        self.assertEqual(loop.calls, ["播放 Wendy"])
        self.assertEqual(self.app._offered_actions.current, offer)
        self.assertEqual(client.calls, [])

        status, second = self._request("POST", "/api/chat", {"text": "可以"})
        self.assertEqual(status, 200)
        self.assertEqual(second["reply"], "正在试听《Wendy》— Test Artist，约 30 秒。")
        self.assertEqual(loop.calls, ["播放 Wendy"])
        self.assertEqual(
            client.calls,
            [("preview_catalog_track", {"canonical_id": BOUND_TRACK})],
        )
        self.assertIsNone(self.app._offered_actions.current)

    def test_s21_real_host_decline_override_projects_explicit_index_offer_and_accepts_exact_target(self) -> None:
        """Exercise the real ProviderAgentLoop + WebShell projection boundary.

        The previous test fabricated ProviderAgentResult(offered_action=...), so it
        could not reproduce the Owner UAT shape where a pending named offer is
        declined and replaced in the same turn by an explicit-index request.
        """
        from music_agent.agent_client import AgentClient
        from music_agent.agent_contract import AgentToolResult
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK
        from music_agent.provider_agent import ProviderAgentLoop
        from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

        run_id = "rcm_22222222-2222-4222-8222-222222222222"
        fourth_target = "trk_44444444-4444-4444-8444-444444444444"
        offer_text = (
            "《Fourth Song》— Fourth Artist 目前无法正式播放，可以试听 30 秒。"
            "需要我开始试听吗？"
        )

        def ok(tool: str, payload: dict) -> AgentToolResult:
            return AgentToolResult(
                request_id="req_44444444-4444-4444-8444-444444444444",
                tool=tool,
                outcome=AgentToolOutcome.OK,
                payload=payload,
                error_code=None,
                error_message=None,
                completed_at=datetime(2026, 9, 16, tzinfo=timezone.utc),
                replayed=False,
            )

        class RuntimeClient(AgentClient):
            def __init__(inner) -> None:
                inner.calls = []

            def call(inner, name: str, arguments: dict, **kwargs):
                inner.calls.append((name, dict(arguments)))
                if name == "get_active_context":
                    return ok(
                        name,
                        {
                            "active_batch": {
                                "run_id": run_id,
                                "source": "register",
                                "item_count": 4,
                            }
                        },
                    )
                if name == "get_recommendation_run":
                    return ok(
                        name,
                        {
                            "run_id": run_id,
                            "items": [
                                {
                                    "position": position,
                                    "target_id": (
                                        fourth_target
                                        if position == 4
                                        else f"trk_{position}{position}{position}{position}{position}{position}{position}{position}-1111-4111-8111-111111111111"
                                    ),
                                    "name": "Fourth Song" if position == 4 else f"Song {position}",
                                    "artist_name": (
                                        "Fourth Artist" if position == 4 else "Other Artist"
                                    ),
                                    "playback": {
                                        "route": "preview_only" if position == 4 else "library"
                                    },
                                }
                                for position in range(1, 5)
                            ],
                        },
                    )
                if name == "preview_catalog_track":
                    self.assertEqual(arguments, {"canonical_id": fourth_target})
                    return ok(name, {"started": True})
                raise AssertionError(f"unexpected client call: {name} {arguments}")

        class RuntimeProvider:
            supports_turn_interpreter = False

            def __init__(inner) -> None:
                inner.calls = []

            def chat(inner, system, messages, tools):
                inner.calls.append((system, messages, tools))
                index = len(inner.calls)
                if index == 1:
                    return ProviderResponse(
                        ProviderMessage(
                            ProviderMessageRole.ASSISTANT,
                            tool_calls=(
                                ProviderToolCall("ctx", "get_active_context", "{}"),
                            ),
                        ),
                        ProviderStopReason.TOOL_USE,
                        {},
                    )
                if index == 2:
                    return ProviderResponse(
                        ProviderMessage(
                            ProviderMessageRole.ASSISTANT,
                            tool_calls=(
                                ProviderToolCall(
                                    "run",
                                    "get_recommendation_run",
                                    json.dumps({"run_id": run_id}),
                                ),
                            ),
                        ),
                        ProviderStopReason.TOOL_USE,
                        {},
                    )
                # This is the old runtime-gap shape: without continuation
                # replacement the compound text stays FULL, reaches this prose
                # offer, and carries no structured OfferedAction.
                return ProviderResponse(
                    ProviderMessage(ProviderMessageRole.ASSISTANT, text=offer_text),
                    ProviderStopReason.END_TURN,
                    {},
                )

        client = RuntimeClient()
        provider = RuntimeProvider()
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        self.app._loop = loop
        old_offer = OfferedAction(
            kind=PREVIEW_TRACK,
            target_canonical_id=BOUND_TRACK,
            source="structured_named_play_resolution",
            verified_title="Wendy",
            verified_artist="Test Artist",
        )
        self.app._offered_actions.arm(old_offer)

        status, first = self._request(
            "POST", "/api/chat", {"text": "不用，播放第4首"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(first["reply"], offer_text)
        armed = self.app._offered_actions.current
        self.assertIsNotNone(armed)
        self.assertEqual(armed.target_canonical_id, fourth_target)
        self.assertEqual(armed.source, "active_recommendation_explicit_index")
        self.assertEqual(armed.verified_title, "Fourth Song")
        self.assertEqual(armed.verified_artist, "Fourth Artist")
        self.assertEqual(len(provider.calls), 2)

        status, accepted = self._request("POST", "/api/chat", {"text": "好的"})
        self.assertEqual(status, 200)
        self.assertEqual(
            accepted["reply"],
            "正在试听《Fourth Song》— Fourth Artist，约 30 秒。",
        )
        self.assertEqual(len(provider.calls), 2)  # acceptance bypasses Provider
        self.assertEqual(
            client.calls,
            [
                ("get_active_context", {}),
                ("get_recommendation_run", {"run_id": run_id}),
                ("preview_catalog_track", {"canonical_id": fourth_target}),
            ],
        )
        self.assertIsNone(self.app._offered_actions.current)

        status, replay = self._request("POST", "/api/chat", {"text": "好的"})
        self.assertEqual(status, 200)
        self.assertNotEqual(replay["reply"], accepted["reply"] )
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(
            [call for call in client.calls if call[0] == "preview_catalog_track"],
            [("preview_catalog_track", {"canonical_id": fourth_target})],
        )

    def test_s21_target_bound_preview_continuation_reuses_exact_target_without_provider(self) -> None:
        from music_agent.action_attempt import (
            create_direct_action_attempt,
            mark_action_executing,
            record_action_execution,
        )
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        attempt = record_action_execution(
            mark_action_executing(
                create_direct_action_attempt(
                    BOUND_TRACK, route="preview_only", title="Wendy", artist="Test Artist"
                )
            ),
            outcome="ok",
            preview_started=True,
        )
        client = _StopFamilyClient()
        loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
            action_attempt=attempt,
        )
        self.app._loop = loop

        status, first = self._request("POST", "/api/chat", {"text": "播放 Wendy"})
        self.assertEqual(status, 200)
        self.assertEqual(first["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(loop.calls, ["播放 Wendy"])
        self.assertEqual(client.calls, [("stop_preview", {})])

        status, second = self._request("POST", "/api/chat", {"text": "开始试听"})
        self.assertEqual(status, 200)
        self.assertEqual(second["reply"], "正在试听《Wendy》— Test Artist，约 30 秒。")
        self.assertEqual(loop.calls, ["播放 Wendy"])  # Provider bypassed on turn N+1.
        self.assertEqual(
            client.calls,
            [
                ("stop_preview", {}),
                ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
            ],
        )
        self.assertIsNone(self.app._offered_actions.current)

    def test_s21_decline_consumes_without_action_and_second_accept_cannot_replay(self) -> None:
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        client = _StopFamilyClient()
        loop = _PlayPreviewLoop(client)
        self.app._loop = loop
        self.app._offered_actions.arm(
            OfferedAction(
                kind=PREVIEW_TRACK,
                target_canonical_id=BOUND_TRACK,
                source="active_recommendation_explicit_index",
            )
        )

        status, declined = self._request("POST", "/api/chat", {"text": "不用"})
        self.assertEqual(status, 200)
        self.assertEqual(declined["reply"], "好的，不试听了。")
        self.assertEqual(client.calls, [])
        self.assertEqual(loop.calls, [])

        status, normal = self._request("POST", "/api/chat", {"text": "好的"})
        self.assertEqual(status, 200)
        self.assertEqual(normal["reply"], "好的。")
        self.assertEqual(client.calls, [])
        self.assertEqual(loop.calls, ["好的"])

    def test_s21_substantive_new_request_wins_and_expires_pending_offer(self) -> None:
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        client = _StopFamilyClient()
        loop = _PlayPreviewLoop(client)
        self.app._loop = loop
        self.app._offered_actions.arm(
            OfferedAction(
                kind=PREVIEW_TRACK,
                target_canonical_id=BOUND_TRACK,
                source="active_recommendation_explicit_index",
            )
        )

        status, body = self._request(
            "POST", "/api/chat", {"text": "不用，播放第二首"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(loop.calls, ["播放第二首"])
        self.assertEqual(client.calls, [])
        self.assertIsNone(self.app._offered_actions.current)

    def test_s21_failed_acceptance_is_consumed_and_never_replayed(self) -> None:
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        client = _StopFamilyClient(preview_outcome=AgentToolOutcome.EXECUTION_ERROR)
        loop = _PlayPreviewLoop(client)
        self.app._loop = loop
        self.app._offered_actions.arm(
            OfferedAction(
                kind=PREVIEW_TRACK,
                target_canonical_id=BOUND_TRACK,
                source="test",
            )
        )

        status, failed = self._request("POST", "/api/chat", {"text": "好的"})
        self.assertEqual(status, 200)
        self.assertEqual(failed["reply"], "这首暂时无法试听。")
        self.assertEqual(
            client.calls,
            [("preview_catalog_track", {"canonical_id": BOUND_TRACK})],
        )
        self.assertIsNone(self.app._offered_actions.current)

        status, normal = self._request("POST", "/api/chat", {"text": "好的"})
        self.assertEqual(status, 200)
        self.assertEqual(normal["reply"], "好的。")
        self.assertEqual(
            client.calls,
            [("preview_catalog_track", {"canonical_id": BOUND_TRACK})],
        )
        self.assertEqual(loop.calls, ["好的"])

    def test_s21_no_pending_start_preview_does_not_guess_target(self) -> None:
        client = _StopFamilyClient()
        loop = _PlayPreviewLoop(client)
        self.app._loop = loop

        status, body = self._request("POST", "/api/chat", {"text": "开始试听"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [])
        self.assertEqual(loop.calls, ["开始试听"])

    def test_t14e_batch_item_play_degraded_to_preview_is_stopped(self) -> None:
        # 播放第2首 with a preview_only item previewed: same degradation stop.
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_batch", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放第2首"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(client.calls, [("stop_preview", {})])

    def test_t14e_bare_play_never_starts_a_preview(self) -> None:
        # Bare 播放 answered with a preview is the forbidden silent downgrade:
        # stopped, and the reply is the honest sentence.
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(client.calls, [("stop_preview", {})])

    def test_t14e_stray_preview_next_to_formal_success_is_stopped_answer_kept(self) -> None:
        # Formal playback succeeded but the model ALSO started a preview:
        # the stray preview is stopped (single audio source), and the run's
        # own answer stands -- no honest-sentence swap.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(
                ProviderLoopToolExecution("play_track", "ok"),
                ProviderLoopToolExecution("preview_catalog_track", "ok"),
            ),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放 Hanataba"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [("stop_preview", {})])

    def test_t14e_formal_play_only_never_touches_the_client(self) -> None:
        # Acceptance A/D: formal playback available -> zero preview/stop calls.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放第2首"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [])

    def test_t14e_preview_request_runs_untouched(self) -> None:
        # Acceptance C/E: explicit preview flows stay their own surface --
        # zero door activity, zero stop calls.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听第2首"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [])

    def test_t14e_failed_preview_attempt_leaves_everything_untouched(self) -> None:
        # A failed preview started no audio: nothing to stop, nothing to swap.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(
                ProviderLoopToolExecution("play_track", "error"),
                ProviderLoopToolExecution("preview_catalog_track", "error"),
            ),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放 Hanataba"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [])

    def test_t14e_delegation_family_is_untouched(self) -> None:
        # 随便播放一首 = explicit play-OR-preview delegation (the whitelisted
        # auto-preview): not a pure play intent, so the door stays silent.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "随便播放一首"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(client.calls, [])

    def test_t14e_stop_failure_still_replaces_the_reply(self) -> None:
        # The stop is best-effort; the honest-sentence swap is fail-closed
        # and never depends on the stop succeeding.
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient()
        client.fail_stop = True
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(client.calls, [("stop_preview", {})])

    # ---- P19-T14-F: 它/他/她 equivalence at the shell boundary ----
    # ---- P19-T14-F-R2: deterministic pronoun binding (channel referent) ----

    def test_t14fr2_preview_pronoun_binds_to_the_channel_referent(self) -> None:
        # Acceptance core: 试听她 normalizes to 试听它 and binds to the ONE
        # unambiguous referent BEFORE any provider round -- the service's own
        # channel register (get_playback_context -> channel.canonical_id).
        # One context read, one preview execution with the exact target id,
        # the fixed start reply, zero loop rounds, zero stop calls.
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        client = _StopFamilyClient(channel_canonical_id=BOUND_TRACK)
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听她"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_START_REPLY)
        self.assertFalse(body["rounds_capped"])
        self.assertIsNone(body["batch"])
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_all_three_preview_spellings_bind_to_the_same_target(self) -> None:
        # Acceptance 1: 试听它/试听他/试听她 are ONE track-reference path --
        # identical target id, identical Preview execution, identical reply.
        # (The live failure: 试听它 resolved while 试听他/她 did not.)
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        for text in ("试听它", "试听他", "试听她"):
            client = _StopFamilyClient(channel_canonical_id=BOUND_TRACK)
            self.app._loop = _PlayPreviewLoop(
                client,
                executions=(
                    ProviderLoopToolExecution("preview_catalog_track", "ok"),
                ),
            )
            status, body = self._request("POST", "/api/chat", {"text": text})
            self.assertEqual(status, 200)
            self.assertEqual(body["reply"], PRONOUN_PREVIEW_START_REPLY)
            self.assertEqual(
                client.calls,
                [
                    ("get_playback_context", {}),
                    ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
                ],
            )
            self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_play_pronoun_binds_to_formal_playback_only(self) -> None:
        # Acceptance 2: 播放他 binds to the same referent and executes ONLY
        # play_track -- the T14-E Play-vs-Preview separation is preserved by
        # construction (zero preview/stop calls, formal start reply).
        from music_agent.intent_router import PRONOUN_PLAY_START_REPLY

        client = _StopFamilyClient(channel_canonical_id=BOUND_TRACK)
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放他"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PLAY_START_REPLY)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": BOUND_TRACK}),
                ("get_now_playing", {}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_play_binding_failure_never_previews(self) -> None:
        # A failing formal-play execution answers the contract's honest
        # sentence and never degrades into a preview (T14-E held even on the
        # deterministic path).
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK,
            play_outcome=AgentToolOutcome.EXECUTION_ERROR,
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "error"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": BOUND_TRACK}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_web_reference_play_tool_ok_with_player_mismatch_fails_closed(self) -> None:
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK,
            now_playing_payload={
                "now_playing": {
                    "state": "playing",
                    "name": "Different Track",
                    "artist": "Different Artist",
                },
                "agent_channel": {
                    "state": "library",
                    "canonical_id": BOUND_TRACK,
                },
                "player_canonical_id": UNBOUND_CATALOG_TRACK,
                "canonical_resolution": "binding",
            },
        )
        self.app._loop = _PlayPreviewLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "播放它"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": BOUND_TRACK}),
                ("get_now_playing", {}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_web_reference_play_unresolved_player_canonical_fails_closed(self) -> None:
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK,
            now_playing_payload={
                "now_playing": {"state": "playing", "name": "Synthetic Track"},
                "agent_channel": {
                    "state": "library",
                    "canonical_id": BOUND_TRACK,
                },
                "player_canonical_id": None,
                "canonical_resolution": None,
            },
        )
        self.app._loop = _PlayPreviewLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "播放它"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(client.calls[-1], ("get_now_playing", {}))

    def test_web_reference_preview_started_false_never_claims_success(self) -> None:
        from music_agent.intent_router import PRONOUN_PREVIEW_UNAVAILABLE_REPLY

        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK,
            preview_payload={"started": False},
        )
        self.app._loop = _PlayPreviewLoop(client)

        status, body = self._request("POST", "/api/chat", {"text": "试听它"})

        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_UNAVAILABLE_REPLY)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
            ],
        )

    def test_t14fr2_no_referent_preview_pronoun_answers_a_fixed_question(self) -> None:
        # Acceptance 3 + the regression the live failure demanded: with NO
        # unambiguous referent the turn answers ONE fixed honest question and
        # the provider loop never sees the line -- recommendation tools and
        # fallback prose (「暂时没有找到合适的推荐…」) are unreachable from
        # a pronoun turn by construction.
        from music_agent.intent_router import PRONOUN_PREVIEW_ASK

        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(
                ProviderLoopToolExecution("generate_recommendation", "ok"),
            ),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听她"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_ASK)
        self.assertFalse(body["rounds_capped"])
        self.assertIsNone(body["batch"])
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_no_referent_play_pronoun_answers_a_fixed_question(self) -> None:
        # 播放他 without a referent asks the play-form question; zero loop.
        from music_agent.intent_router import PRONOUN_PLAY_ASK

        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放他"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PLAY_ASK)
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_none_channel_state_carries_no_referent(self) -> None:
        # Defensive: a canonical_id sitting alongside state=none must NOT
        # bind (only library/preview actions are the referent register).
        from music_agent.intent_router import PRONOUN_PREVIEW_ASK

        client = _StopFamilyClient(
            channel_state="none", channel_canonical_id=BOUND_TRACK
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_ASK)
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_play_binding_requires_library_or_preview_channel(self) -> None:
        # The same register rule on the play side of the pair.
        from music_agent.intent_router import PRONOUN_PLAY_ASK

        client = _StopFamilyClient(
            channel_state="none", channel_canonical_id=BOUND_TRACK
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PLAY_ASK)
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_preview_binding_without_started_is_honest(self) -> None:
        # A bound preview whose payload does not carry started=true answers
        # the unavailable sentence -- never a claimed start.
        from music_agent.intent_router import PRONOUN_PREVIEW_UNAVAILABLE_REPLY

        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK, preview_payload={"started": False}
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_UNAVAILABLE_REPLY)
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr2_running_continuous_session_defers_to_the_loop(self) -> None:
        # A running continuous preview session makes the referent contested
        # (the single-audio rule would cancel the session): the turn defers
        # to the provider loop untouched -- the loop still sees the proven
        # 它 spelling, and nothing is executed from the fast path.
        client = _StopFamilyClient(
            channel_canonical_id=BOUND_TRACK, session_state="running"
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听她"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "好的。")
        self.assertEqual(self.app._loop.calls, ["试听它"])
        self.assertEqual(client.calls, [("get_playback_context", {})])

    def test_t14fr2_failed_truth_read_falls_back_to_the_loop(self) -> None:
        # Unreadable live truth refuses to fabricate a referent: the line
        # falls through to the provider loop (fail-closed, T14-A parity).
        client = _StopFamilyClient(channel_canonical_id=BOUND_TRACK)
        client.fail_read = True
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听她"})
        self.assertEqual(status, 200)
        self.assertEqual(self.app._loop.calls, ["试听它"])
        self.assertEqual(body["reply"], "好的。")

    def test_t14fr2_stop_object_pronoun_routes_through_the_stop_family(self) -> None:
        # 停止试听她 -> 停止试听它 -> the stop-family fast path executes
        # stop_preview through the authority with the same honest replies;
        # the provider loop is never reached.
        client = _StopFamilyClient()
        self.app._loop = _StopFamilyLoop(client)
        status, body = self._request("POST", "/api/chat", {"text": "停止试听她"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "已停止试听。")
        self.assertEqual(client.calls, [("stop_preview", {})])
        self.assertEqual(self.app._loop.calls, [])

    # ---- P19-T14-F-R4: session-local referent survives preview stops ----

    def test_t14fr4_preview_pronouns_bind_the_referent_after_a_stop(self) -> None:
        # The Owner regression, verbatim: 试听 Hanataba -> stop preview ->
        # 试听他. After the stop the channel register is none (cleared by
        # constitution) while referent_canonical_id still names the track
        # the user just targeted -- all three spellings bind THAT id, one
        # preview execution, zero loop rounds, never a question, never
        # recommendation generation.
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        for text in ("试听它", "试听他", "试听她"):
            client = _StopFamilyClient(
                channel_state="none",
                channel_canonical_id=None,
                referent_canonical_id=BOUND_TRACK,
            )
            self.app._loop = _PlayPreviewLoop(
                client,
                executions=(
                    ProviderLoopToolExecution("preview_catalog_track", "ok"),
                ),
            )
            status, body = self._request("POST", "/api/chat", {"text": text})
            self.assertEqual(status, 200)
            self.assertEqual(body["reply"], PRONOUN_PREVIEW_START_REPLY)
            self.assertFalse(body["rounds_capped"])
            self.assertIsNone(body["batch"])
            self.assertEqual(
                client.calls,
                [
                    ("get_playback_context", {}),
                    ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
                ],
            )
            self.assertEqual(self.app._loop.calls, [])

    def test_t14fr4_referent_beats_a_cleared_channel(self) -> None:
        # Regression 2's extractor core at the shell: the channel may carry
        # a cleared/foreign action -- the referent is the conversational
        # target and wins outright.
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        client = _StopFamilyClient(
            channel_state="none",
            channel_canonical_id=None,
            referent_canonical_id=BOUND_TRACK,
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_START_REPLY)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("preview_catalog_track", {"canonical_id": BOUND_TRACK}),
            ],
        )

    def test_t14fr4_play_pronoun_from_the_surviving_referent_is_formal_only(self) -> None:
        # Regression 3: 播放他 after a stop binds the surviving referent and
        # executes ONLY play_track -- T14-E held by construction.
        from music_agent.intent_router import PRONOUN_PLAY_START_REPLY

        client = _StopFamilyClient(
            channel_state="none",
            channel_canonical_id=None,
            referent_canonical_id=BOUND_TRACK,
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放他"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PLAY_START_REPLY)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": BOUND_TRACK}),
                ("get_now_playing", {}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr4_failing_play_from_the_referent_never_previews(self) -> None:
        # Regression 3's failure branch: B cannot formally play -> the
        # T14-E honest fallback sentence, zero preview execution, zero stop
        # calls -- never an automatic preview.
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        client = _StopFamilyClient(
            channel_state="none",
            channel_canonical_id=None,
            referent_canonical_id=BOUND_TRACK,
            play_outcome=AgentToolOutcome.EXECUTION_ERROR,
        )
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("play_track", "error"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "播放它"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PLAY_PREVIEW_DOWNGRADE_FALLBACK)
        self.assertEqual(
            client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": BOUND_TRACK}),
            ],
        )
        self.assertEqual(self.app._loop.calls, [])

    def test_t14fr4_unbound_pronoun_after_a_stop_still_asks_honestly(self) -> None:
        # Regression 6: no referent (fresh session, no explicit interaction
        # yet) -> the fixed honest question; the loop never sees the line.
        from music_agent.intent_router import PRONOUN_PREVIEW_ASK

        client = _StopFamilyClient(channel_state="none", channel_canonical_id=None)
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(
                ProviderLoopToolExecution("generate_recommendation", "ok"),
            ),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听他"})
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], PRONOUN_PREVIEW_ASK)
        self.assertEqual(client.calls, [("get_playback_context", {})])
        self.assertEqual(self.app._loop.calls, [])

    def test_t14f_unrelated_pronoun_phrase_is_not_rewritten_at_the_shell(self) -> None:
        # Acceptance 4: the possessive 他的歌 keeps 他 -- only the track-
        # reference object spelling is normalized.
        client = _StopFamilyClient()
        self.app._loop = _PlayPreviewLoop(
            client,
            executions=(ProviderLoopToolExecution("preview_catalog_track", "ok"),),
        )
        status, body = self._request("POST", "/api/chat", {"text": "试听他的歌"})
        self.assertEqual(status, 200)
        self.assertEqual(self.app._loop.calls, ["试听他的歌"])
        self.assertEqual(client.calls, [])

    def _newest_run_id(self) -> str:
        with RecommendationHistoryRepository(self.database_path) as history:
            run_id = history.newest_run_id()
        self.assertIsNotNone(run_id)
        return run_id

    def test_command_whitelist_rejects_unknown_tools(self) -> None:
        status, body = self._request(
            "POST", "/api/command", {"tool": "discover_catalog_tracks"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "unknown_command")
        self.assertNotIn("discover_catalog_tracks", SHELL_COMMANDS)

    def test_pause_command_executes_through_backend(self) -> None:
        status, body = self._request("POST", "/api/command", {"tool": "pause"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["message"], "已暂停播放。")
        self.assertEqual(self.adapter.calls, [("pause", ())])

    def test_pause_command_tool_ok_with_playing_readback_fails_closed(self) -> None:
        client = _StopFamilyClient(
            now_playing_payload={"now_playing": {"state": "playing"}}
        )
        self.app._ui_client = client

        status, body = self._request("POST", "/api/command", {"tool": "pause"})

        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "playback_state_mismatch")
        self.assertNotEqual(body["error"]["message"], "已暂停播放。")
        self.assertEqual(client.calls, [("pause", {}), ("get_now_playing", {})])

    def test_next_previous_keep_command_accepted_contract(self) -> None:
        for tool in ("next_track", "previous_track"):
            with self.subTest(tool=tool):
                status, body = self._request(
                    "POST", "/api/command", {"tool": tool}
                )
                self.assertEqual(status, 200)
                self.assertTrue(body["ok"])
        self.assertEqual(
            self.adapter.calls,
            [("next_track", ()), ("previous_track", ())],
        )

    def test_card_play_uses_strict_terminal_truth_and_one_audio_execution(self) -> None:
        client = _StopFamilyClient(channel_canonical_id=BOUND_TRACK)
        self.app._ui_client = client

        status, body = self._request(
            "POST",
            "/api/command",
            {"tool": "play_track", "canonical_id": BOUND_TRACK},
        )

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["result"]["status"], "completed")
        self.assertEqual(
            body["message"],
            "正在播放《Synthetic Track》— Synthetic Artist。",
        )
        self.assertEqual(
            client.calls,
            [
                ("play_track", {"canonical_id": BOUND_TRACK}),
                ("get_now_playing", {}),
            ],
        )

    def test_card_play_tool_ok_with_unresolved_readback_returns_failure(self) -> None:
        client = _StopFamilyClient(
            now_playing_payload={
                "now_playing": {"state": "playing", "name": "Synthetic Track"},
                "agent_channel": {
                    "state": "library",
                    "canonical_id": BOUND_TRACK,
                },
                "player_canonical_id": None,
                "canonical_resolution": None,
            }
        )
        self.app._ui_client = client

        status, body = self._request(
            "POST",
            "/api/command",
            {"tool": "play_track", "canonical_id": BOUND_TRACK},
        )

        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "player_canonical_unresolved")
        self.assertNotIn("正在播放", body["error"]["message"])

    def test_card_preview_requires_started_true(self) -> None:
        client = _StopFamilyClient(preview_payload={"started": False})
        self.app._ui_client = client

        status, body = self._request(
            "POST",
            "/api/command",
            {"tool": "preview_catalog_track", "canonical_id": BOUND_TRACK},
        )

        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "preview_not_started")
        self.assertEqual(
            client.calls,
            [("preview_catalog_track", {"canonical_id": BOUND_TRACK})],
        )

    def test_command_requires_canonical_id_for_track_commands(self) -> None:
        status, body = self._request(
            "POST", "/api/command", {"tool": "play_track"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "missing_track")

    def test_native_apple_music_target_resolves_without_executing_open_tool(self) -> None:
        target = {
            "canonical_id": BOUND_TRACK,
            "url": "https://music.apple.com/us/album/x/1?i=2",
            "client_url": "https://music.apple.com/us/song/2",
            "source": "itunes_store_lookup",
        }
        with patch.object(
            self.app._service_a, "resolve_apple_music_target", return_value=target
        ) as resolve:
            status, body = self._request(
                "POST", "/api/apple-music-target", {"canonical_id": BOUND_TRACK}
            )
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True, "result": target})
        resolve.assert_called_once_with(BOUND_TRACK)

    def test_native_apple_music_target_requires_canonical_id(self) -> None:
        status, body = self._request("POST", "/api/apple-music-target", {})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "missing_track")

    def test_playback_failure_surfaces_honest_banner(self) -> None:
        # A second app without a playback adapter: the command must return the
        # real backend refusal, never a fabricated ok (backend failure display).
        registry = AgentClientRegistry({CLIENT_ID: AgentClientPolicy("full")})
        config = ShellConfig(
            database_path=self.database_path,
            provider_factory=lambda: _FakeProvider(["好的。"]),
            agent_client=(CLIENT_ID, "full"),
            mode="standalone",
            service_factory=lambda: SharedAgentService(
                self.database_path, clients=registry
            ),
        )
        app = WebShellApp(config)
        app.start()
        thread = threading.Thread(target=app.serve_forever, daemon=True)
        thread.start()

        def request(port: int):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/command",
                data=json.dumps({"tool": "pause"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))

        try:
            status, body = request(app.port)
        finally:
            app.close()
            app.stop_httpd()
            thread.join(timeout=5.0)
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "playback_unavailable")

    def test_state_reads_authoritative_now_playing(self) -> None:
        status, body = self._request("GET", "/api/state")
        self.assertEqual(status, 200)
        player = body["player"]
        self.assertEqual(player["state"], "playing")
        self.assertEqual(player["name"], "起风了 (旧版)")
        self.assertIsNone(player["preview_session"])

    def test_player_state_fast_endpoint_reads_now_playing_only(self) -> None:
        client = _StopFamilyClient(
            now_playing_payload={
                "now_playing": {
                    "state": "paused",
                    "name": "Fast Track",
                    "artist": "Fast Artist",
                    "album": "Fast Album",
                }
            }
        )
        self.app._ui_client = client

        status, body = self._request("GET", "/api/player-state")

        self.assertEqual(status, 200)
        self.assertEqual(
            body["player"],
            {
                "state": "paused",
                "name": "Fast Track",
                "artist": "Fast Artist",
                "album": "Fast Album",
            },
        )
        self.assertEqual(client.calls, [("get_now_playing", {})])

    def test_cards_projection_end_to_end(self) -> None:
        run_id = _seed_recommendation_run(self.database_path, BOUND_TRACK)
        status, body = self._request("GET", "/api/cards")
        self.assertEqual(status, 200)
        cards = body["cards"]
        self.assertEqual(len(cards), 1)
        self.assertEqual(
            set(cards[0]),
            {"canonical_id", "name", "artist", "album", "route", "apple_music_openable"},
        )
        self.assertEqual(cards[0]["canonical_id"], BOUND_TRACK)
        self.assertEqual(cards[0]["route"], "library")
        self.assertNotEqual(cards[0]["name"], "")
        # P19-T14-B: the run id travels with the cards -- the freshness
        # identity, not an item-key projection.
        self.assertEqual(body["run_id"], run_id)
        # BOUND_TRACK is library-only (no itunes_store binding): the Apple
        # Music action must be off -- the backend would refuse the open.
        self.assertIs(cards[0]["apple_music_openable"], False)

    def test_cards_empty_without_runs(self) -> None:
        status, body = self._request("GET", "/api/cards")
        self.assertEqual(status, 200)
        self.assertEqual(body["cards"], [])
        self.assertIsNone(body["run_id"])

    def test_events_surface_with_tail_semantics(self) -> None:
        self.app._events.present({"event": "progress", "session": {"position": 1}})
        status, body = self._request("GET", "/api/events?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)
        self.assertEqual(body["next"], 1)
        status, body = self._request("GET", f"/api/events?after={body['next']}")
        self.assertEqual(body["events"], [])

    def test_shutdown_mark_request(self) -> None:
        self.assertFalse(self.app.shutdown_requested)
        self.app.request_shutdown()
        self.assertTrue(self.app.shutdown_requested)


class UiShellContractTest(unittest.TestCase):
    """P17 acceptance: the chat shell's layout contract (structure + styles).

    No browser framework: these pin the DOM structure and the CSS rules the
    layout depends on -- the viewport-bound shell, the internally-scrolling
    history, and the composer pinned outside the scrolling region -- so a
    regression that lets the page grow with history fails here first.
    """

    _UI_PATH = Path(__file__).resolve().parent.parent / "src" / "music_agent" / "ui" / "index.html"

    @classmethod
    def setUpClass(cls) -> None:
        cls._html = cls._UI_PATH.read_text(encoding="utf-8")

    def _css_block(self, selector: str, containing: str | None = None) -> str:
        import re

        blocks = re.findall(re.escape(selector) + r"\s*\{([^}]*)\}", self._html)
        self.assertTrue(blocks, f"missing CSS rule for {selector}")
        if containing is not None:
            for block in blocks:
                if containing in block:
                    return block
            self.fail(f"no {selector!r} rule containing {containing!r}: {blocks!r}")
        return blocks[0]

    def test_viewport_bounded_shell_never_grows_the_page(self) -> None:
        for selector in ("body", ".shell"):
            self.assertIn("overflow: hidden", self._css_block(selector, "overflow: hidden"))
        shell = self._css_block(".shell")
        self.assertIn("min-height: 0", shell)
        self.assertNotIn("min-height: 100%", shell)
        # P19-R5: the viewport bound (100vh/100dvh fit) now lives on the
        # stage that sizes the scaled shell's visual footprint, not on the
        # fixed 660x880 canvas itself.
        stage = self._css_block(".viewport-stage")
        self.assertIn("100vh", stage)
        self.assertIn("100dvh", stage)  # follows window height changes

    def test_history_region_scrolls_internally(self) -> None:
        messages = self._css_block(".messages")
        self.assertIn("overflow-y: auto", messages)
        self.assertIn("min-height: 0", messages)  # flex shrink makes it scroll
        self.assertIn("flex: 1", messages)
        # the panel itself must not pin a growth floor
        self.assertIn("min-height: 0", self._css_block("#chat-panel", "min-height: 0"))
        self.assertNotIn("min-height: 420px", self._html)

    def test_composer_stays_outside_the_scrolling_region(self) -> None:
        messages_open = self._html.index('<div class="messages" id="messages">')
        composer_open = self._html.index('<div class="composer">')
        composer_close = self._html.index("</div>", composer_open)
        panel_close = self._html.index("</section>", self._html.index('id="chat-panel"'))
        self.assertLess(messages_open, composer_open)
        self.assertLess(composer_close, panel_close)
        self.assertTrue('id="input"' in self._html[composer_open:composer_close])
        self.assertTrue('id="send"' in self._html[composer_open:composer_close])

    def test_new_messages_autoscroll_to_the_latest(self) -> None:
        # P18-S3: auto-scroll keeps the newest visible, but only follows when
        # the reader is already near the bottom -- reading history must never
        # scroll away mid-read. The reader's own message always sticks.
        self.assertIn("function scrollToLatest", self._html)
        self.assertIn("function nearBottom", self._html)
        self.assertIn("box.scrollHeight - box.scrollTop - box.clientHeight", self._html)
        self.assertIn('const shouldStick = role === "user" || nearBottom()', self._html)
        # appendMessage + the thinking indicator both keep the newest visible
        self.assertGreaterEqual(self._html.count("scrollToLatest()"), 2)

    def test_conversation_stage_scrolls_instead_of_growing_the_page(self) -> None:
        # v1 layout removed the separate right column; the bounded-scroll
        # guarantee now lives on the conversation stage view, which is the
        # region that would otherwise grow with long replies.
        column = self._css_block("#view-convo")
        self.assertIn("overflow-y: auto", column)
        self.assertIn("min-height: 0", column)

    def test_completed_turn_restores_the_visible_conversation_scrollport(self) -> None:
        """The completed live pair anchors in #view-convo, never at page top."""
        import re
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is required for the conversation scroll execution test")

        match = re.search(
            r"function revealCompletedTurn\(\) \{.*?\n\}",
            self._html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        script = r'''
const assert = require('node:assert/strict');
const box = {
  scrollTop: 0, scrollHeight: 2600, clientHeight: 500,
  getBoundingClientRect: () => ({top: 100}),
};
const current = {
  firstElementChild: {},
  getBoundingClientRect: () => ({top: 2050}),
};
const $ = id => id === 'view-convo' ? box : current;
''' + match.group(0) + r'''
// Simulate WebKit clamping the hidden scrollport to zero before the visible
// DOM handoff. The new live turn sits below a long archived conversation.
revealCompletedTurn();
assert.equal(box.scrollTop, 1942);
assert.notEqual(box.scrollTop, 0);

// A short conversation still clamps safely to its available bottom rather
// than overscrolling or depending on browser scrollIntoView behaviour.
box.scrollTop = 0;
box.scrollHeight = 420;
box.clientHeight = 500;
revealCompletedTurn();
assert.equal(box.scrollTop, 0);
'''
        result = subprocess.run(
            [node, "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        deliver = self._html[
            self._html.index("async function deliverChat") : self._html.index("function sendChat")
        ]
        self.assertIn(
            "renderConversation();\n    setThinking(false);\n    revealCompletedTurn();",
            deliver,
        )
        self.assertIn("input.focus({ preventScroll: true });", deliver)
        self.assertNotIn("input.focus();", deliver)

    def test_scroll_repair_does_not_force_every_render_to_bottom(self) -> None:
        repair = self._html[
            self._html.index("function revealCompletedTurn") : self._html.index(
                "/* ---- chat: correction-while-running"
            )
        ]
        # Search, playback and recommendation replies share one completion
        # handoff. It anchors the current turn; it never installs a global
        # scrollIntoView/scrollHeight-bottom policy that steals history reads.
        self.assertIn('const box = $("view-convo");', repair)
        self.assertIn('const currentTurn = $("convo-user");', repair)
        self.assertIn("turnRect.top - boxRect.top", repair)
        self.assertNotIn("scrollIntoView", repair)
        self.assertNotIn("box.scrollTop = box.scrollHeight", repair)

        render = self._html[
            self._html.index("function renderConversation") : self._html.index(
                "function revealCompletedTurn"
            )
        ]
        self.assertNotIn("revealCompletedTurn", render)

        # Recommendation detail continues to save/restore the position that
        # the completion handoff established, and card DOM replacement does
        # not touch the conversation scrollport.
        detail = self._html[
            self._html.index("function openDetail") : self._html.index(
                "function cardButtonCard"
            )
        ]
        self.assertIn('detailReturnScroll = $("view-convo").scrollTop;', detail)
        self.assertIn('$("view-convo").scrollTop = detailReturnScroll;', detail)
        expanded = self._html[
            self._html.index("function renderExpanded") : self._html.index(
                "function collapseToTrace"
            )
        ]
        self.assertNotIn('$("view-convo").scrollTop', expanded)

    def test_main_player_scales_unitarily_without_intermediate_reflow(self) -> None:
        import re

        # P19-R5: exactly two layout modes. Above 520px the fixed 660x880
        # Main Player scales as one unit (transform: scale) inside the
        # viewport-sized stage; nothing at intermediate widths may reflow
        # shell geometry, the composer, shortcuts, fonts or padding.
        stage = self._css_block(".viewport-stage")
        self.assertIn("min(660px", stage)
        self.assertIn("calc(100vw", stage)
        self.assertIn("aspect-ratio: 0.75", stage)
        shell = self._css_block(".shell")
        self.assertIn("width: 660px", shell)
        self.assertIn("height: 880px", shell)
        self.assertIn("transform: scale(", shell)
        # the 860px intermediate reflow is gone; the only layout-switch
        # media query left is the Mini Player at <= 520px
        self.assertIsNone(
            re.search(r"@media [^{]*max-width: (521|600|700|800|860|900|1024)px", self._html)
        )
        self.assertIn("@media (max-width: 520px)", self._html)
        # Internal one-line guarantees: shortcuts stay a compact centered
        # 1x4 strip -- every pill hugs its own text, with no fixed columns
        # or edge-to-edge distribution. Text rows and the composer
        # placeholder never wrap.
        sc = self._css_block(".shortcuts")
        self.assertIn("display: flex", sc)
        self.assertIn("flex-wrap: nowrap", sc)
        self.assertIn("justify-content: center", sc)
        self.assertIn("width: auto", sc)
        self.assertNotIn("grid-template-columns", sc)
        self.assertIn("gap: 8px", sc)
        self.assertNotIn("width:", self._css_block(".chip"))
        self.assertIn("white-space: nowrap", self._css_block(".player .track"))
        self.assertIn("white-space: nowrap", self._css_block("#view-greet .hello-sub"))
        self.assertIn("white-space: nowrap", self._css_block(".composer textarea::placeholder"))

    # ---- P18-S1 contract pins ----

    def test_assistant_markdown_renders_only_the_server_html_channel(self) -> None:
        # Exactly one innerHTML sink exists in the messaging code: the
        # assistant branch consuming reply_html (server-rendered, escape-first
        # whitelisted Markdown). User messages and all other channels are
        # textContent. (The farewell quit handler also rewrites the whole
        # page on shutdown -- unrelated to messaging, excluded by targeting
        # the appendMessage function body.)
        append_open = self._html.index("function appendMessage")
        append_end = self._html.index("function setThinking")
        function = self._html[append_open:append_end]
        self.assertEqual(function.count("msg.innerHTML = renderedHtml"), 1)
        self.assertIn("msg.textContent = text", function)
        self.assertIn('role === "assistant"', function)
        self.assertIn("msg.classList.add(\"rendered\")", function)
        # the chat flow feeds the rendered channel from the server response
        self.assertIn("appendMessage(\"assistant\", data.reply || \"\", data.reply_html)", self._html)
        # rendered typography is scoped to the rendered marker
        self.assertIn(".msg.assistant.rendered", self._html)

    def test_preview_block_is_driven_by_runtime_truth_not_a_timer(self) -> None:
        # No 30-second timer anywhere: visibility comes from the polled
        # preview_sounding / preview_session facts; the frontend never
        # computes completion and never resumes anything.
        self.assertNotIn("30000", self._html)
        self.assertNotIn("30_000", self._html)
        self.assertIn("player.preview_sounding === true", self._html)
        self.assertIn("if (!session && !sounding)", self._html)
        self.assertIn("function suspensionNote", self._html)
        self.assertIn('id="preview-now"', self._html)
        self.assertIn("renderPreviewNow(player)", self._html)

    def test_preview_priority_keeps_formal_track_secondary_and_honest(self) -> None:
        # While a preview holds priority the formal track reads secondary
        # (dimmed via the suspended class) and the interruption note is
        # carried by the player note -- display wording, never a resume
        # mechanism; the runtime restores.
        self.assertIn("已暂停，试听结束后由本地服务自动恢复", self._html)
        self.assertIn("lastPreviewTrack = null", self._html)
        self.assertIn("lastPreviewTrack && lastPreviewTrack.name", self._html)
        self.assertIn('now.classList.toggle("suspended", formalSuspended)', self._html)
        self.assertIn(".player .now.suspended", self._html)
        self.assertIn('id="player-note"', self._html)

    def test_recommendation_row_is_quiet_title_artist_actions(self) -> None:
        # P19-T9: a recommendation row is compact metadata, not a card --
        # title strongest, artist secondary, one small action cluster, and
        # no album disambiguation line anywhere.
        self.assertIn("slot.className = \"card\"", self._html)
        self.assertIn("meta.append(name, artist)", self._html)
        self.assertIn('name.className = "name"', self._html)
        self.assertIn('artist.className = "artist"', self._html)
        self.assertNotIn(".card .album", self._html)

    # ---- P18-S2 contract pins ----

    def test_recommendation_presentation_has_no_fabricated_artwork(self) -> None:
        # P19-T9: compact recommendation rows carry no artwork at all --
        # no cover slot to fill, no <img>, no background-image: nothing
        # fabricated, nothing to load. The quiet glyph stays the only
        # visual identity.
        self.assertNotIn("<img", self._html)
        self.assertNotIn("background-image", self._html)
        self.assertNotIn("function coverTile", self._html)
        self.assertNotIn("cover-sm", self._html)
        self.assertGreaterEqual(self._html.count("♪"), 3)

    def test_preview_block_prioritizes_preview_over_formal_track(self) -> None:
        # Structure: the preview block sits above the formal now block and
        # only becomes visible on the runtime's preview facts.
        preview_open = self._html.index('id="preview-now"')
        player_now_open = self._html.index('id="player-now"')
        self.assertLess(preview_open, player_now_open)
        self.assertIn("display: flex", self._css_block(".preview-now.visible"))

    def test_apple_music_action_gated_on_projected_binding(self) -> None:
        # P18-S2: the Apple Music button may only appear when the projected
        # binding fact says the backend open would succeed.
        self.assertGreaterEqual(self._html.count("if (card.apple_music_openable)"), 1)
        self.assertIn("在 Apple Music 打开", self._html)

    def test_long_titles_clamp_instead_of_overflowing(self) -> None:
        # Card names clamp to two lines; the player's title/artist stay on
        # ONE line (P19-R5: writing must never restructure the player) and
        # ellipsize; the single-line metadata ellipsizes -- nothing
        # overflows bounds.
        self.assertIn("-webkit-line-clamp", self._css_block(".card .name"))
        # P19-R5 replaced the player's 2-line clamp with single-line
        # ellipsis: a long real song name truncates, it does not reflow.
        track = self._css_block(".player .track")
        self.assertIn("white-space: nowrap", track)
        self.assertIn("text-overflow: ellipsis", track)
        self.assertIn("text-overflow: ellipsis", self._css_block(".card .artist"))
        self.assertIn("text-overflow: ellipsis", self._css_block(".player .album"))
        self.assertIn("text-overflow: ellipsis", self._css_block(".player .artist"))
        self.assertIn("flex-wrap: wrap", self._css_block(".card .actions"))

    # ---- P19-T9 contract pins ----

    def test_fresh_batch_makes_structured_rows_the_primary_result(self) -> None:
        # A batch whose run id arrived attached to this chat turn belongs to
        # THIS reply: the full raw prose never becomes the main middle
        # content; the reply renders as one short verbatim lead above the
        # rows. Freshness is run identity -- item-key equality plays no part.
        self.assertIn(
            "if (batch && batch.cards.length)",
            self._html,
        )
        self.assertIn("pendingExpandCards = turnEntry", self._html)
        self.assertIn('appendMessage("assistant", lead)', self._html)
        # the batch travels with the chat reply itself: no separate
        # /api/cards fetch inside deliverChat, no cross-fetch race
        chat = self._html[
            self._html.index("async function deliverChat") : self._html.index("function sendChat")
        ]
        self.assertNotIn("const cardData = await fetchCardData();", chat)
        self.assertIn("data.batch && data.batch.run_id", chat)
        # every other reply keeps the raw rendered path unchanged
        self.assertIn('appendMessage("assistant", data.reply || "", data.reply_html)', self._html)
        self.assertIn("stageProseReply = document.querySelector", self._html)

    def test_t14b_reply_batch_identity_drives_structured_takeover(self) -> None:
        # deliverChat takes over on the run id attached to the /api/chat
        # reply itself: identity beats item-key equality, so a new run whose
        # items re-project to an old key still expands. No separate cards
        # fetch runs inside deliverChat -- same fetch, no race.
        chat = self._html[
            self._html.index("async function deliverChat") : self._html.index("function sendChat")
        ]
        self.assertIn("data.batch && data.batch.run_id", chat)
        self.assertIn("if (batch && batch.cards.length)", chat)
        self.assertNotIn("sessionBatchRunId", self._html)
        self.assertNotIn("await fetchCardData();", chat)
        self.assertNotIn("sessionBatchKey", self._html)
        # the cards poll reads the server run id into every cached batch
        cards = self._html[
            self._html.index("async function fetchCardData") : self._html.index("async function pollCards")
        ]
        self.assertIn("data.run_id || key", cards)

    def test_structured_lead_is_a_verbatim_prefix_never_invented_copy(self) -> None:
        # The lead is a presentation choice over the AI's own reply: first
        # line, cut at the earliest sentence ender, capped with an ellipsis,
        # inline markdown markers stripped -- words verbatim, no new facts.
        self.assertIn("function chatLead", self._html)
        self.assertIn("const MAX = 72;", self._html)
        self.assertIn("const end = lead.search(/[。！？!?]/);", self._html)
        self.assertIn('line.replace(/[#*_`]/g, "")', self._html)
        self.assertIn(r'lead.slice(0, MAX).replace(/[，、：:;\s]+[^，、：:;]*$/, "")', self._html)

    def test_lead_face_is_quiet_conversational_styling(self) -> None:
        # The lead sentence reads visually quiet -- soft ink, normal
        # wrapping, smaller than body text -- never the markdown face.
        lead = self._css_block(".convo-reply .msg.assistant.lead")
        self.assertIn("var(--ink-soft)", lead)
        self.assertIn("white-space: normal", lead)
        self.assertIn("font-size: 16.5px", lead)

    def test_out_of_band_batch_cannot_claim_a_chat_turn(self) -> None:
        # /api/cards reports durable latest history and therefore cannot
        # establish presentation ownership. Unknown runs are ignored; only
        # an already turn-owned batch may have its cached rows refreshed.
        start = self._html.index("async function pollCards")
        poll = self._html[start : self._html.index("/* P18-S3", start)]
        self.assertIn("if (!known) { return; }", poll)
        self.assertIn("rememberBatch(cardData);", poll)
        self.assertNotIn("openDetail", poll)
        self.assertNotIn("batchCache =", poll)
        self.assertNotIn("convertStageReplyToLead", poll)

    def test_trace_states_the_real_cached_batch_length(self) -> None:
        # The folded trace label comes from the cached batch itself -- the
        # batch count is never hard-coded, and the P19-T17-D label carries
        # the turn's own frozen request label with the generic fallback.
        self.assertIn("traceText(batchCache.label, batchCache.cards.length)", self._html)
        self.assertIn("function renderTrace", self._html)
        self.assertIn("function fetchCardData", self._html)

    def test_conversation_region_hides_its_native_scrollbar(self) -> None:
        # Scrolling stays inside the result region with keyboard/wheel
        # intact; only the native scrollbar visual is removed.
        convo = self._css_block("#view-convo", "overflow-y: auto")
        self.assertIn("overflow-y: auto", convo)
        track = self._css_block("#view-convo", "scrollbar-width: none")
        self.assertIn("scrollbar-width: none", track)
        self.assertIn("#view-convo::-webkit-scrollbar", self._html)
        self.assertIn("display: none", self._css_block("#view-convo::-webkit-scrollbar"))

    def test_preview_and_apple_music_actions_survive_in_compact_rows(self) -> None:
        # Route-aware actions are unchanged in the compact rows: library
        # plays formally, catalog previews 30s, Apple Music only on the
        # projected binding fact; the last-preview display name is tracked.
        self.assertIn('runCommand("play_track", card.canonical_id', self._html)
        self.assertIn('runCommand("preview_catalog_track", card.canonical_id', self._html)
        self.assertIn('runCommand("open_in_apple_music", card.canonical_id', self._html)
        self.assertIn('textContent = "试听 30 秒"', self._html)
        self.assertIn(
            'lastPreviewTrack = { canonical_id: card.canonical_id, name: card.name || "" }',
            self._html,
        )
        self.assertIn(
            'runCommand("play_track", card.canonical_id, null)', self._html
        )
        self.assertNotIn(
            '"正在播放 " + (card.name || "选中的曲目")', self._html
        )
        self.assertIn("if (data.message) showNotice(data.message)", self._html)
        self.assertIn('await nativeAppleMusicOpen(result.client_url)', self._html)
        self.assertIn('showNotice("已交给 Apple Music 打开")', self._html)
        self.assertIn('window.webkit.messageHandlers.appleMusicOpen', self._html)
        self.assertNotIn('"已在浏览器打开 Apple Music 页面"', self._html)


    def test_native_apple_music_handoff_waits_for_foreground_bridge_completion(self) -> None:
        # Native mode resolves identity through a side-effect-free endpoint and
        # visible success is emitted only after the Swift bridge reports completion.
        command_start = self._html.index("async function runCommand")
        command_end = self._html.index('$("play-pause")', command_start)
        command = self._html[command_start:command_end]
        self.assertIn('tool === "open_in_apple_music" && nativeAppMode()', command)
        self.assertIn('"/api/apple-music-target"', command)
        self.assertIn('nativeAppleMusic ? "/api/apple-music-target" : "/api/command"', command)
        self.assertIn('await nativeAppleMusicOpen(result.client_url)', command)
        self.assertLess(
            command.index('await nativeAppleMusicOpen(result.client_url)'),
            command.index('showNotice("已交给 Apple Music 打开")'),
        )
        self.assertIn('? { canonical_id: canonicalId }', command)
        self.assertIn('showError(e && e.message ? e.message', command)

    def test_native_apple_music_bridge_is_fail_closed_and_uses_nsworkspace(self) -> None:
        root = Path(__file__).resolve().parent.parent
        main = (root / "app/MusicAgent/Sources/MusicAgent/MainWindowController.swift").read_text(
            encoding="utf-8"
        )
        bridge = (root / "app/MusicAgent/Sources/MusicAgent/NativeAppleMusicBridge.swift").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'config.userContentController.add(\n                appleMusicBridge, name: NativeAppleMusicBridge.handlerName)',
            main,
        )
        self.assertIn('configuration.activates = true', bridge)
        self.assertIn('configuration.requiresUniversalLinks = true', bridge)
        self.assertIn(
            'NSWorkspace.shared.open(\n'
            '            clientURL,\n'
            '            configuration: configuration\n'
            '        )',
            bridge,
        )
        self.assertIn('application?.bundleIdentifier == "com.apple.Music"', bridge)
        self.assertIn('components.host?.lowercased() == "music.apple.com"', bridge)
        self.assertIn('parts[1] == "song"', bridge)
        self.assertIn('parts[2].allSatisfy({ $0.isNumber })', bridge)
        self.assertNotIn('/usr/bin/open', bridge)
        self.assertNotIn('withApplicationAt: musicApplicationURL', bridge)
        self.assertNotIn('withBundleIdentifier: "com.apple.Music"', bridge)

    # ---- P19-T10 transition-state pins ----

    def test_thinking_exit_never_reactivates_greeting_or_convo(self) -> None:
        # Greeting is the initial idle stage ONLY. setThinking(true) hides
        # both stages under the field; setThinking(false) must not decide
        # the next stage -- renderConversation is its single owner. The old
        # exit re-activated both with toggle(off, on), which produced the
        # Thinking -> Greeting -> Recommendation flash.
        self.assertNotIn('$("view-greet").classList.toggle("off", on)', self._html)
        self.assertNotIn('$("view-convo").classList.toggle("off", on)', self._html)
        st = self._html[
            self._html.index("function setThinking") : self._html.index("function renderConversation")
        ]
        self.assertIn("if (on) {", st)
        self.assertIn('$("view-greet").classList.add("off")', st)
        self.assertIn('$("view-convo").classList.add("off")', st)

    def test_handoff_activates_target_before_thinking_recedes(self) -> None:
        # P19-T10 handoff order: response/cards ready -> populate target DOM
        # -> activate target view -> deactivate Thinking, all inside one
        # task. No setThinking(false) may sit between setThinking(true) and
        # renderConversation(), and inside the finally the target activation
        # is directly followed by the thinking exit.
        deliver = self._html[
            self._html.index("async function deliverChat") : self._html.index("function sendChat")
        ]
        pre = deliver[deliver.index("setThinking(true)") : deliver.index("renderConversation();")]
        self.assertNotIn("setThinking(false);", pre)
        self.assertIn("renderConversation();", deliver)
        self.assertIn("setThinking(false);", deliver)
        self.assertIn("renderConversation();\n    setThinking(false);", self._html)

    def test_idle_stage_classes_are_static_markup_only(self) -> None:
        # Boot state is static: greeting active, convo + thinking hidden.
        # The runtime may only ever RE-hide greeting (idle-only invariant);
        # nothing in code un-hides it except renderConversation's idle case.
        self.assertRegex(self._html, r'<div class="view" id="view-greet">')
        self.assertRegex(self._html, r'<div class="view off" id="view-convo">')
        self.assertRegex(self._html, r'<div class="view off" id="view-thinking">')

    def test_status_polling_keeps_expensive_context_off_the_idle_2s_heartbeat(self) -> None:
        # Formal player truth remains responsive, but the expensive runtime
        # playback-context read is adaptive: 10s idle, 2s only while preview
        # or suspension state is active. Request-start cadence subtracts the
        # completed request's elapsed time without introducing overlap.
        # Preview/chat actions force one full refresh rather than waiting for
        # the idle cadence.
        self.assertIn('fetch("/api/player-state")', self._html)
        self.assertIn('fetch("/api/state")', self._html)
        self.assertIn("const PLAYER_POLL_MS = 2000;", self._html)
        self.assertIn("const CONTEXT_IDLE_POLL_MS = 10000;", self._html)
        self.assertIn("const CONTEXT_ACTIVE_POLL_MS = 2000;", self._html)
        self.assertIn("function contextNeedsFastPoll", self._html)
        self.assertIn("function contextPollTargetMs", self._html)
        self.assertIn("function nextContextPollDelay", self._html)
        self.assertIn("performance.now() - startedAt", self._html)
        self.assertIn("Math.max(0, contextPollTargetMs(lastState) - elapsed)", self._html)
        self.assertIn("function scheduleContextPoll", self._html)
        self.assertIn("async function refreshFullState", self._html)
        self.assertIn("setInterval(pollPlayerState, PLAYER_POLL_MS);", self._html)
        self.assertNotIn("setInterval(pollState, 2000);", self._html)
        self.assertNotIn("setInterval(pollState", self._html)
        self.assertIn('if (tool === "preview_catalog_track") refreshFullState();', self._html)

    def test_context_poll_request_start_cadence_executes_without_overlap(self) -> None:
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is required for the polling scheduler execution test")

        helpers = self._html[
            self._html.index("function contextNeedsFastPoll") : self._html.index(
                "async function pollPlayerState"
            )
        ]
        scheduler = self._html[
            self._html.index("function scheduleContextPoll") : self._html.index(
                "/* ---- cards ---- */"
            )
        ]
        script = r'''
const assert = require('node:assert/strict');
const PLAYER_POLL_MS = 2000;
const CONTEXT_IDLE_POLL_MS = 10000;
const CONTEXT_ACTIVE_POLL_MS = 2000;
let now = 0;
const performance = {now: () => now};
let lastState = {};
let contextPollTimer = null;
let nextTimerId = 1;
const timers = new Map();
const cleared = [];
function setTimeout(fn, delay) {
  const id = nextTimerId++;
  timers.set(id, {fn, delay});
  return id;
}
function clearTimeout(id) { cleared.push(id); timers.delete(id); }
function takeTimer() {
  assert.equal(timers.size, 1);
  const [id, timer] = timers.entries().next().value;
  timers.delete(id);
  return timer;
}
let pollState = async () => {};
''' + helpers + scheduler + r'''
(async () => {
  // Idle and active targets both subtract request elapsed time.
  lastState = {};
  now = 1120;
  assert.equal(nextContextPollDelay(120), 9000);
  lastState = {preview_sounding: true};
  now = 1320;
  assert.equal(nextContextPollDelay(120), 800);

  // A request slower than its target schedules at zero, never negatively.
  now = 2500;
  assert.equal(nextContextPollDelay(0), 0);

  // The next target is selected after pollState publishes its latest state.
  timers.clear();
  contextPollTimer = null;
  lastState = {};
  now = 0;
  pollState = async () => {
    now = 11500;
    lastState = {preview_sounding: true};
  };
  scheduleContextPoll();
  const idleTimer = takeTimer();
  assert.equal(idleTimer.delay, 10000);
  now = 10000;
  await idleTimer.fn();
  assert.equal(takeTimer().delay, 500);

  // No successor exists while the current request is unresolved; even an
  // over-target request only schedules its zero-delay successor afterward.
  timers.clear();
  contextPollTimer = null;
  lastState = {preview_sounding: true};
  now = 20000;
  let activeCalls = 0;
  let resolvePoll;
  pollState = () => {
    activeCalls += 1;
    assert.equal(activeCalls, 1);
    return new Promise(resolve => {
      resolvePoll = () => { activeCalls -= 1; now = 22600; resolve(); };
    });
  };
  scheduleContextPoll(0);
  const activeTimer = takeTimer();
  const inFlight = activeTimer.fn();
  assert.equal(activeCalls, 1);
  assert.equal(timers.size, 0);
  resolvePoll();
  await inFlight;
  assert.equal(activeCalls, 0);
  assert.equal(takeTimer().delay, 0);

  // Immediate refresh cancels the old timer, polls now, and resumes from
  // that request's start using the newly published active state.
  timers.clear();
  contextPollTimer = 99;
  timers.set(99, {fn: () => {}, delay: 10000});
  lastState = {};
  now = 30000;
  let refreshCalls = 0;
  pollState = async () => {
    refreshCalls += 1;
    now = 31200;
    lastState = {preview_sounding: true};
  };
  await refreshFullState();
  assert.equal(refreshCalls, 1);
  assert.ok(cleared.includes(99));
  assert.equal(takeTimer().delay, 800);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        result = subprocess.run(
            [node, "-e", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fast_player_merge_never_clears_preview_context_by_omission(self) -> None:
        merge_start = self._html.index("function mergeFormalPlayer")
        merge_end = self._html.index("function contextNeedsFastPoll", merge_start)
        merge = self._html[merge_start:merge_end]
        for field in ("preview_session", "preview_sounding", "suspended"):
            self.assertNotIn(f"lastState.{field} =", merge)
        self.assertIn('["state", "name", "artist", "album"]', merge)

    # ---- P18-S3 contract pins ----

    def test_action_buttons_show_honest_pending_labels(self) -> None:
        # In-flight actions get plain-language labels -- no tool names, no
        # provider rounds, no progress amounts.
        self.assertIn("function markPending", self._html)
        self.assertIn("function clearPending", self._html)
        self.assertIn("正在开始试听…", self._html)
        self.assertIn("正在播放…", self._html)
        self.assertIn("正在打开…", self._html)
        self.assertIn('btn.dataset.forTool = forTool', self._html)
        # no fabricated progress -- the pending helpers contain no timing and
        # the flow never invents an amount (percentages live in CSS only)
        pending_open = self._html.index("function markPending")
        pending_end = self._html.index("async function runCommand")
        pending = self._html[pending_open:pending_end]
        self.assertNotIn("setTimeout", pending)
        self.assertNotIn("setInterval", pending)
        self.assertNotIn("Date.now", pending)

    def test_pending_label_clears_on_authoritative_truth_not_timer(self) -> None:
        # The in-flight label is released by the polled runtime truth
        # (preview_sounding / formal playing), by open's own completion, or
        # by errors -- never by a client timer.
        self.assertIn("clearPending();", self._html)
        self.assertIn('if (lastState.state === "playing") clearPending()', self._html)
        self.assertIn('dataset.forTool === "open_in_apple_music"', self._html)

    def test_apple_music_wording_consistent(self) -> None:
        # One wording on every Apple Music action; the old label is gone so
        # the treatment cannot drift between branches.
        self.assertGreaterEqual(self._html.count('textContent = "在 Apple Music 打开"'), 1)
        self.assertNotIn("在 Apple Music 中打开", self._html)

    def test_focus_visible_and_long_text_never_overlap_controls(self) -> None:
        # Keyboard focus is visible on interactive elements; long unbroken
        # tokens wrap inside bubbles instead of pushing controls.
        self.assertIn("outline", self._css_block("button:focus-visible, textarea:focus-visible, a:focus-visible"))
        self.assertIn("overflow-wrap: break-word", self._css_block(".msg"))
        self.assertIn("overflow-wrap: anywhere", self._css_block(".msg.rendered code"))
        self.assertIn("flex-wrap: wrap", self._css_block(".controls"))

    def test_faint_tier_keeps_minimum_readable_contrast(self) -> None:
        # The faint ink tier stays visually tertiary but stays legible at
        # 12px. v1 re-targeted the canvas to a deep blue-black and the tier
        # to its purple-tinted value, which measures ~5.8:1 against #0b0c1a
        # (still above the 4.5:1 minimum this test guards).
        self.assertIn("--ink-faint: #8d8aa0", self._html)

    # ---- P18 Fix 01: correction-while-running (chat interruption follow-up) ----

    def test_fix01_composer_never_locks_while_a_reply_runs(self) -> None:
        # The correction gap was caused by disabling the input and send
        # button for the whole provider-loop duration. Neither is disabled
        # anywhere now -- typing stays possible the entire time.
        self.assertNotIn("input.disabled", self._html)
        self.assertNotIn("sendBtn.disabled", self._html)

    def test_fix01_correction_queues_instead_of_overlapping_requests(self) -> None:
        # A send while busy queues the text instead of firing a second
        # request: exactly one dispatch point sets chatBusy, and the queue
        # (one item, newest wins) fires only after the running request
        # reaches a terminal state and its reply has rendered.
        self.assertEqual(self._html.count("chatBusy = true"), 1)
        self.assertIn("function deliverChat", self._html)
        self.assertIn("function queueCorrection", self._html)
        self.assertIn('if (chatBusy) { queueCorrection(text); return; }', self._html)
        self.assertIn("const next = input.value.trim() || queuedDraft;", self._html)
        self.assertIn("deliverChat(next);", self._html)
        # newest intent wins: every queued send replaces the previous draft
        self.assertIn("queuedDraft = text; // replace: the newest correction wins", self._html)

    def test_fix01_no_fake_cancellation_is_offered(self) -> None:
        # Backend cancellation does not exist (the provider loop runs to
        # completion on its serial worker), so the UI must not pretend: no
        # AbortController and no "stop generating" action anywhere.
        self.assertNotIn("AbortController", self._html)
        self.assertNotIn("abort()", self._html)
        self.assertNotIn("停止生成", self._html)
        self.assertIn("textContent = \"排队发送\"", self._html)

    def test_fix01_queue_chip_is_honest_and_cancellable(self) -> None:
        # The chip states exactly what happens (sends after the current
        # reply completes) and offers a real 取消 -- the element, its hidden
        # default, and both handlers must all exist.
        self.assertIn('<div class="composer-queue" id="queue-chip">', self._html)
        self.assertIn("（当前回复完成后自动发送）", self._html)
        self.assertIn('$("queue-cancel").onclick', self._html)
        self.assertIn("function renderQueue", self._html)
        self.assertIn("queueChip.classList.toggle(\"visible\", queuedDraft !== null)", self._html)
        css = self._css_block(".composer-queue")
        self.assertIn("display: none", css)
        self.assertIn("display: flex", self._css_block(".composer-queue.visible"))

    # ---- P18 Fix 02: IME-safe Enter (composing Enter never sends/queues) ----

    def test_fix02_composing_enter_never_sends_or_queues(self) -> None:
        # While an IME is composing, Enter only confirms the candidate. The
        # gate sits inside the keydown handler BEFORE the dispatch, so both
        # the send path (idle) and the queue path (busy, Fix 01) are
        # unreachable during composition -- three independent gate terms
        # cover the engine order differences, with no timers.
        self.assertIn("let composingText = false;", self._html)
        self.assertIn('input.addEventListener("compositionstart", () => { composingText = true; });', self._html)
        self.assertIn('input.addEventListener("compositionend", () => { composingText = false; });', self._html)
        self.assertIn("if (composingText || event.isComposing || event.keyCode === 229) return;", self._html)
        self.assertNotIn("setTimeout", self._html[self._html.index("compositionstart"): self._html.index("event.preventDefault")])

    def test_fix02_post_composition_enter_dispatches_exactly_once(self) -> None:
        # Only a genuine post-composition Enter reaches sendChat, and there
        # is exactly one keydown dispatch site in the whole file -- idle
        # sends once, busy queues once (Fix 01 routing unchanged).
        self.assertEqual(self._html.count("sendChat();"), 1)
        self.assertEqual(self._html.count("event.preventDefault(); sendChat();"), 1)
        self.assertIn('if (event.key === "Enter" && !event.shiftKey) {', self._html)

    def test_fix02_shift_enter_stays_a_newline_never_a_send(self) -> None:
        # Shift+Enter is excluded by the same guard and gets no dispatch of
        # its own -- it keeps the textarea's normal newline behavior.
        self.assertEqual(self._html.count("sendChat();"), 1)
        self.assertIn('event.key === "Enter" && !event.shiftKey', self._html)
        self.assertNotIn("!event.shiftKey) { event.preventDefault(); sendChat()", self._html)

    def test_fix02_compositionend_never_sends_or_queues_by_itself(self) -> None:
        # Duplicate-send guard: compositionend only clears the flag -- no
        # send, no queue -- so a commit Enter followed by a real Enter can
        # never double-dispatch; the transcript stays deterministic.
        end_open = self._html.index('input.addEventListener("compositionend"')
        end_close = self._html.index("});", end_open) + len("});")
        handler = self._html[end_open:end_close]
        self.assertNotIn("sendChat", handler)
        self.assertNotIn("queueCorrection", handler)

    # ---- P19-T2: WAAPI thinking particle engine (S1 + S2 + S3a) ----

    def _tp_engine(self) -> str:
        a = self._html.index("/* === thinking particle engine (WAAPI, vanilla) === */")
        b = self._html.index("/* === end thinking particle engine === */")
        return self._html[a:b]

    def test_t2_particle_layer_replaces_gradient_field_layers(self) -> None:
        # S1: exactly one engine-owned container; the fog layer survives;
        # the three painted gradient layers are gone from HTML + keyframes.
        self.assertEqual(self._html.count('class="thinking-particles"'), 1)
        self.assertIn('<div class="thinking-particles" aria-hidden="true"></div>', self._html)
        self.assertEqual(self._html.count('class="field f4"'), 1)
        self.assertNotIn('<div class="field f1">', self._html)
        self.assertNotIn('<div class="field f2">', self._html)
        self.assertNotIn('<div class="field f3">', self._html)
        self.assertIn("@keyframes think-cloud-drift", self._html)
        for gone in ("think-stream-", "think-float", "think-stray", "think-shimmer"):
            self.assertNotIn(gone, self._html)
        fog = self._css_block(".field.f4")
        self.assertIn("radial-gradient", fog)
        self.assertIn("think-cloud-drift", fog)
        layer = self._css_block(".thinking-particles")
        self.assertIn("pointer-events: none", layer)
        self.assertIn("overflow: hidden", layer)
        self.assertIn("will-change", self._css_block(".thinking-particles .tp"))

    def test_t2_particle_budget_and_constants(self) -> None:
        # T8: the field is the densified demo grid -- 19x19 at 20px pitch,
        # circular-cropped at 9 cells, exactly 253 dots = TP_MAX. The
        # stagger constants: 140ms per grid unit from the centre on one
        # shared 3400ms loop; the lane machinery is gone, so the budget
        # is structural, not incidental.
        engine = self._tp_engine()
        self.assertIn("const TP_MAX = 253;", engine)
        self.assertIn("const TP_GRID_N = 19;", engine)
        self.assertIn("const TP_GRID_CELL = 20;", engine)
        self.assertIn("const TP_GRID_KEEP = 9;", engine)
        self.assertIn("const TP_STAGGER_STEP = 140;", engine)
        self.assertIn("const TP_STAGGER_PERIOD = 3400;", engine)
        self.assertNotIn("laneCounts", engine)
        self.assertNotIn("TP_LANES", engine)
        self.assertNotIn("TP_SPIRAL", engine)
        self.assertNotIn("TP_TWIST", engine)

    def test_t8_field_is_fixed_regular_grid(self) -> None:
        # T8: same fixed grid architecture, densified -- ONE deterministic
        # double loop over a fixed 19x19 grid at 20px. Positions are exact
        # grid points (centre + dx*20px); the 360px circle comes from the
        # distance crop alone (no jitter, no spiral, no lanes). Roles
        # grade by grid distance (heart < 3 cells / body <= 6.7 / fringe
        # beyond) with twelve fixed pink grid seats. Structural counts by
        # construction -- no rejection sampling, and Math.random never
        # appears anywhere in the engine.
        engine = self._tp_engine()
        self.assertIn("for (let iy = 0; iy < TP_GRID_N; iy++) {", engine)   # one plain grid loop
        self.assertIn("const dx = ix - 9, dy = iy - 9;", engine)
        self.assertIn("const dist = Math.hypot(dx, dy);", engine)
        self.assertIn("if (dist > TP_GRID_KEEP) continue;", engine)          # circular crop, positions stay grid
        self.assertIn("const role = dist < 3 ? 2", engine)                   # heart: centre + nearest two rings
        self.assertIn('TP_PINK_SEATS[ix + "," + iy] ? 3', engine)            # pink: twelve fixed grid seats
        self.assertIn("(dist <= 6.7 ? 1 : 0));", engine)                     # body inside 6.7 cells, fringe beyond
        self.assertIn(
            "push(TP_HEART.x + dx * TP_GRID_CELL, TP_HEART.y + dy * TP_GRID_CELL, role, role === 2 ? 1 : 0);",
            engine,
        )
        self.assertIn(
            '"7,5": 1, "10,5": 1, "13,6": 1, "13,9": 1, "12,12": 1, "10,13": 1, "7,13": 1, "5,12": 1, "5,9": 1, "5,6": 1, "11,12": 1, "8,12": 1',
            engine,
        )
        self.assertIn("p[6] = 0; p[7] = 0; p[8] = 0; p[9] = 0;", engine)     # retired flow slots parked
        self.assertNotIn("bodyCount", engine)                                # no rejection build
        self.assertNotIn("Math.random", engine)                              # seeded only
        self.assertNotIn("Math.cos(ang)", engine)                            # polar machinery fully retired

    def test_t2_flip_race_hard_cancels_residual_exit(self) -> None:
        # Amendment 1: an old generation may only cancel/remove what it
        # captured; entering first hard-cancels a residual exit, and only
        # then builds a new generation.
        engine = self._tp_engine()
        self.assertIn("if (tpSystem) return;", engine)
        self.assertIn("if (tpExiting) tpTeardown(tpExiting);", engine)
        self.assertLess(engine.index("if (tpSystem) return;"),
                        engine.index("if (tpExiting) tpTeardown(tpExiting);"))
        self.assertLess(engine.index("if (tpExiting) tpTeardown(tpExiting);"),
                        engine.index("++tpGen"))
        self.assertIn("tpSystem = null", engine)

    def test_t2_teardown_touches_only_generation_captured_nodes(self) -> None:
        # Never a wholesale container clear: no replaceChildren anywhere in
        # the engine; teardown cancels and removes only sys's own captives.
        engine = self._tp_engine()
        self.assertNotIn("replaceChildren", engine)
        self.assertIn("function tpTeardown", engine)
        self.assertIn(".cancel()", engine)
        self.assertIn("box.removeChild(node)", engine)
        self.assertIn("a.onfinish = null", engine)
        self.assertIn("if (tpExiting === sys) tpExiting = null;", engine)

    def test_t2_reduced_motion_composes_static_no_persistent_animations(self) -> None:
        # Reduced motion is first-class runtime state: the static compositor
        # exists and runs before any element animation in enter; the CSS
        # side only adds a one-shot opacity transition, never a kill list.
        engine = self._tp_engine()
        self.assertIn('matchMedia("(prefers-reduced-motion: reduce)")', engine)
        self.assertIn("function tpStaticCompose", engine)
        self.assertLess(engine.index("function tpStaticCompose"), engine.index("el.animate("))
        self.assertIn("if (!TP_WAAPI || tpReduced()) { tpStaticCompose(sys); return; }", engine)
        rule = ".thinking-particles .tp { transition: opacity 400ms ease; }"
        self.assertEqual(self._html.count(rule), 1)
        self.assertNotIn(".thinking-particles .tp { animation:", self._html)

    def test_t2_missing_waapi_falls_back_without_timers(self) -> None:
        # Amendment 2: if Element.prototype.animate is unavailable or
        # throws, the engine composes statically: capability is a prototype
        # read (no probe animation at parse), no timers armed, and the only
        # callers of the lifecycle are setThinking + the motion listener.
        engine = self._tp_engine()
        self.assertIn("Element.prototype.animate", engine)
        self.assertIn("const TP_WAAPI = (function () {", engine)
        self.assertIn("catch (e) { return false; }", engine)
        self.assertNotIn("setInterval", engine)
        self.assertEqual(self._html.count("tpEnter()"), 3)  # hook + def + motion listener
        self.assertEqual(self._html.count("tpExit()"), 2)   # hook + def
        self.assertIn("on ? tpEnter() : tpExit();", self._html)

    def test_t8_motion_is_outward_swell_stagger(self) -> None:
        # T8: the T7 centred stagger stays, re-tuned as one visible
        # outward breath. phase = grid distance x 140ms / 3400ms shared
        # loop; every dot waits its slot (positive delay, backwards-
        # filled at rest) then plays the same shared swell. The keys
        # builder is a pure function of the row -- zero timing
        # randomness, no angle terms (the envelope lerp fractions ride
        # the parked dynamics draws) -- exactly one loop animation per
        # particle, never translating any dot.
        engine = self._tp_engine()
        self.assertIn("function tpStaggerKeys(p) {", engine)                 # the motion model takes no rng
        self.assertIn("const dx = (p[0] - TP_HEART.x) / TP_GRID_CELL;", engine)  # grid units, back to cell space
        self.assertIn("const dist = Math.hypot(dx, dy);", engine)
        self.assertIn("const phase = dist * TP_STAGGER_STEP / TP_STAGGER_PERIOD;", engine)
        self.assertIn(
            "return { x: tpR1(p[0]), y: tpR1(p[1]), phase: phase, period: TP_STAGGER_PERIOD, keys: keys };",
            engine,
        )
        keys_fn = engine[engine.index("function tpStaggerKeys"):engine.index("function tpStaticCompose")]
        self.assertNotIn("Math.cos", keys_fn)    # the stagger has no angle terms
        self.assertNotIn("Math.sin", keys_fn)
        self.assertNotIn("rotate(", keys_fn)
        self.assertNotIn("translate(", keys_fn)
        self.assertNotIn("TP_RADIAL_RAMP", engine)   # the custom breathing model is gone
        self.assertNotIn("TP_PHASE_JIT", engine)
        self.assertNotIn("TP_CYCLE_LO", engine)
        self.assertNotIn("TP_RADIUS_MAX", engine)
        self.assertNotIn("normRadius", engine)
        self.assertNotIn("hasCounter", engine)       # no counter-phase or angular machinery
        self.assertNotIn("composite", engine)
        self.assertNotIn("rotate(", engine)
        self.assertNotIn("vx", engine)
        self.assertEqual(engine.count("iterations: Infinity"), 1)            # the stagger loop, one per particle
        self.assertIn("delay: w2.phase * w2.period,", engine)                # waits its grid slot inside the shared loop
        self.assertIn('transform: "scale(', engine)

    def test_t8_swell_envelope_and_easings(self) -> None:
        # T8 envelope contract: ONE shared keyset per dot -- a single
        # smooth bump at rest 0.70-0.78 scale / 0.30-0.45 opacity, peak
        # 1.20-1.35 / 0.85-1.00 over 500ms on true sine inOut, back to
        # rest over 1000ms on true sine out, then a soft hold to the loop
        # end (offsets are fractions of the shared 3400ms loop: 0.147 =
        # 500ms, 0.441 = 1500ms). Per-dot envelope values are seeded from
        # the parked dynamics draws -- the builder itself is rng-free --
        # and the minimum scale anywhere is the rest scale: no keyframe
        # shrinks a dot toward zero. Anime generates its easings
        # programmatically, so the port carries fitted cubic-beziers
        # (worst |dy| <= 0.001, sub-pixel).
        engine = self._tp_engine()
        self.assertIn('const TP_EASE_IN_OUT_SINE = "cubic-bezier(0.367, 0.002, 0.634, 1.000)";', engine)
        self.assertIn('const TP_EASE_OUT_SINE = "cubic-bezier(0.387, 0.614, 0.664, 0.994)";', engine)
        self.assertNotIn("TP_EASE_IN_OUT_QUAD", engine)  # the demos shrink curve is retired with the shrink
        self.assertIn("const restS = 0.70 + 0.08 * p[11];", engine)          # rest scale, 0.70-0.78 band
        self.assertIn("const peakS = 1.20 + 0.15 * (p[12] / 6.2832);", engine)  # peak scale, 1.20-1.35 band
        self.assertIn("const restO = p[4];", engine)                         # rest opacity read from the row
        self.assertIn("const peakO = 0.85 + 0.15 * p[10];", engine)          # peak opacity, 0.85-1.00 band
        self.assertIn('{ offset: 0,     transform: "scale(" + restS.toFixed(2) + ")", opacity: restO, easing: TP_EASE_IN_OUT_SINE },', engine)
        self.assertIn('{ offset: 0.147, transform: "scale(" + peakS.toFixed(2) + ")", opacity: peakO, easing: TP_EASE_OUT_SINE },', engine)
        self.assertIn('{ offset: 0.441, transform: "scale(" + restS.toFixed(2) + ")", opacity: restO },', engine)
        self.assertIn('{ offset: 1,     transform: "scale(" + restS.toFixed(2) + ")", opacity: restO },', engine)  # soft hold settles the loop end
        self.assertNotIn("scale(0.1)", engine)            # the demo's shrink-to-nothing dip is gone
        keys_fn = engine[engine.index("function tpStaggerKeys"):engine.index("function tpStaticCompose")]
        self.assertEqual(keys_fn.count("opacity:"), 4)    # the loop carries rest/peak/rest/rest opacity only
        enter_fn = engine[engine.index("const enter = el.animate"):engine.index("sys.anims.push(enter);")]
        self.assertEqual(enter_fn.count("opacity:"), 2)   # one-shot enter still fades 0.12 -> rest opacity

    def test_t8_uniform_size_rest_opacity_band(self) -> None:
        # T8: one stated base size everywhere (5.0) -- density comes from
        # the 19x19 grid, not size tiers -- with rest opacity seeded per
        # dot into the 0.30-0.45 band so the WHOLE field reads visible
        # at rest. Background alpha stays baked at 1; the row stores the
        # rest state bare; no spatial jitter anywhere in the table build.
        engine = self._tp_engine()
        self.assertIn("function push(x, y, role, glow) {", engine)
        self.assertIn("const restO = 0.30 + r() * 0.15;", engine)            # rest opacity, 0.30-0.45 band
        self.assertIn(
            "pts.push([tpR1(x), tpR1(y), role, 5.0, restO, glow, 0, 0, 0, 0, 0, 0, 0, ci]);",
            engine,
        )
        self.assertEqual(engine.count("5.0, restO, glow"), 1)                # one uniform base size, one band
        self.assertNotIn("5.0, 1.0, glow", engine)                           # the opacity-1 rest state is retired
        self.assertIn('"background:rgba(" + c[0] + "," + c[1] + "," + c[2] + ",1);"', engine)
        self.assertNotIn("r() * 0.3", engine)    # no size envelopes / tiers
        self.assertNotIn("0.25 + r() * 0.20", engine)
        self.assertNotIn("tpJit(r,", engine)     # the table build has no spatial or opacity jitter

    def test_t6_color_mapping_center_to_edge(self) -> None:
        # T8 Music Agent palette, unchanged from T6: lavender-white heart
        # at the centre, mist purple body band (#927FE0 family), deep
        # mist purple fringe, warm pink (#D98EAF family) only on the
        # twelve accent grid seats -- restrained, no neon, no rainbow.
        engine = self._tp_engine()
        self.assertIn("[[108, 96, 188], [122, 110, 202]],", engine)                              # fringe
        self.assertIn("[[146, 127, 224], [134, 116, 214], [158, 142, 228], [168, 156, 234]],", engine)  # body mist purple
        self.assertIn("[[232, 216, 242], [245, 240, 250], [222, 204, 238]],", engine)            # heart lavender-white
        self.assertIn("[[217, 142, 175], [229, 168, 192]],", engine)                             # pink accents only

    def test_t2_enter_staggers_and_exit_releases_controlled(self) -> None:
        # Radial-out aggregation with staggered delays; exit is a controlled
        # release driven by WAAPI fill:forwards -- the two one-shot belts
        # (450 static-fade / 1200 release ceiling) are the only timers.
        # T8 rest-scale continuity: enter lands on the dot's own rest
        # scale/opacity and the release starts from that same rest state
        # (explicit first keyframe -- no growth pop at either handoff).
        engine = self._tp_engine()
        self.assertIn("delay: idx * 4 + tpJit(rng, 180)", engine)
        self.assertIn('easing: "cubic-bezier(0.22, 0.61, 0.36, 1)"', engine)
        self.assertIn('fill: "backwards"', engine)
        self.assertIn("window.setTimeout(function () { tpTeardown(sys); }, 1200);", engine)
        self.assertEqual(engine.count("window.setTimeout"), 2)  # 450 static-fade + 1200 belt
        self.assertIn('fill: "forwards"', engine)
        self.assertIn("const restT = el._f.keys[0].transform;", engine)         # enter lands on the loop's rest key
        self.assertIn('{ transform: "translate(0px, 0px) " + restT, opacity: p[4] },', engine)
        self.assertIn("const restT = node._f.keys[0].transform;", engine)       # release starts from the rest state
        self.assertIn('[ { transform: "translate(0px, 0px) " + restT, opacity: p[4] }, { transform: "translate("', engine)

    def test_t2_determinism_seeded_no_external_dependency(self) -> None:
        # Zero unseeded randomness, zero imports, zero extra script tags:
        # the whole engine is vanilla, source-only, reproducible per run.
        engine = self._tp_engine()
        self.assertIn("function tpRng", engine)
        self.assertIn("0x6D2B79F5", engine)  # mulberry32 constant
        self.assertIn("tpRng(20260824)", engine)
        self.assertNotIn("Math.random", engine)
        self.assertNotIn("import ", engine)
        self.assertEqual(self._html.count("<script src"), 0)

    def test_t2_engine_inert_until_thinking(self) -> None:
        # The Default state runs nothing: no probe animation at parse time,
        # the single internal tpEnter call site sits behind the reduced-
        # motion change guard, and nothing auto-starts.
        engine = self._tp_engine()
        self.assertNotIn("probe.animate", engine)
        self.assertEqual(self._html.count("tpEnter();"), 1)  # inside the change listener
        self.assertIn('addEventListener("change", function () {', engine)
        self.assertTrue(self._html.index("on ? tpEnter() : tpExit();") < self._html.index("/* === thinking particle engine"))
        self.assertIn('document.querySelector(".thinking-particles")', engine)

    def _deliver_chat(self) -> str:
        a = self._html.index("async function deliverChat")
        b = self._html.index("\n}\n", a)
        return self._html[a:b]

    # ---- P19-T2 runtime fix: undefined `f` + lifecycle fault containment ----

    def test_t2_motion_closure_passes_stored_wave_context(self) -> None:
        # The enter loop's onfinish IIFE once ended `})(sys, el, p, f, enter);`
        # with `f` block-scoped to the earlier layout loop -- a live
        # ReferenceError on every Thinking. The closure must consume the
        # per-particle wave context captured on the node instead.
        engine = self._tp_engine()
        self.assertIn("})(sys, el, p, el._f, enter);", engine)
        self.assertNotIn(", f, enter);", engine)
        self.assertIn("el._f = w;", engine)

    def test_t2_engine_failure_cannot_wedge_chat_busy(self) -> None:
        # Fault containment: any synchronous prefix throw (particle engine
        # or otherwise) must reach the deliverChat catch, with chatBusy
        # reset in the finally beneath it -- the dispatch can never lock.
        body = self._deliver_chat()
        self.assertLess(body.index("try {"), body.index("chatBusy = true"))
        self.assertIn('console.error("[chat] deliver failed:", e);', body)
        self.assertLess(body.index("} finally {"), body.index("chatBusy = false;"))

    def test_t2_deliverchat_prefix_inside_failure_boundary(self) -> None:
        # Every synchronous prefix step (user bubble, lastUserText, request
        # text, setThinking(true)) runs inside the try, and both exits of
        # the request phase release Thinking.
        body = self._deliver_chat()
        self.assertLess(body.index('appendMessage("user", text)'), body.index("setThinking(true);"))
        self.assertLess(body.index("setThinking(true);"), body.index("const data = await"))
        # P19-T10: the Thinking exit is a single terminal act in the finally
        # -- it runs on BOTH exits (reply delivered AND failure), so the
        # fault boundary is preserved with exactly one call site.
        self.assertEqual(body.count("setThinking(false);"), 1)
        self.assertLess(body.index("} finally {"), body.index("setThinking(false);"))

    def test_t2_queue_ordering_and_finally_semantics_unchanged(self) -> None:
        # The finally block keeps its exact contract: de-wedge first, then
        # consume the queue (newest wins) before re-dispatching, and restore
        # the send label in the no-queue branch.
        body = self._deliver_chat()
        self.assertLess(body.index("chatBusy = false;"), body.index("if (queuedDraft)"))
        self.assertIn("const next = input.value.trim() || queuedDraft;", body)
        self.assertLess(body.index("queuedDraft = null; renderQueue();"),
                        body.index("deliverChat(next);"))
        self.assertIn('sendBtn.textContent = "发送";', body)

    def test_t2_setthinking_particle_hook_isolated(self) -> None:
        # The particle hook is isolated so an engine failure can never block
        # the view switching that follows it inside setThinking.
        start = self._html.index("function setThinking")
        end = self._html.index("/* the stage's two conversation lines")
        body = self._html[start:end]
        self.assertIn("try { on ? tpEnter() : tpExit(); }", body)
        self.assertIn('console.error("[tp] engine failed:", e);', body)
        self.assertLess(body.index('console.error("[tp] engine failed:", e);'),
                        body.index('$("view-thinking").classList.toggle("off", !on);'))

    def test_t2_baseline_layout_contract_untouched(self) -> None:
        # Frozen surfaces stay pixel-identical: the Default reduced-motion
        # block, Finding-A de-emphasis rules, and composer geometry are
        # unaffected by the particle rewrite. (The Default .field kill still
        # covers the surviving fog layer -- deliberately frozen.)
        self.assertIn(".breathe, .field, .shell::before { animation: none !important; }", self._html)
        self.assertIn("body.is-thinking .composer", self._html)
        composer = self._css_block(".composer")
        self.assertIn("position: relative", composer)
        self.assertIn("margin-top: 18px", composer)

    # ---- P19-T11: bottom compactness + thinking spacing + back chevron + press language ----

    def test_t11a_composer_capsule_slims_with_frozen_anchors(self) -> None:
        # Composer 92 -> 78px (border-box) with the 31.5px line box
        # re-centered at 22px of top padding; the horizontal paddings
        # (28/96), the 34px radius, font and placeholder stay frozen.
        ta = self._css_block(".composer textarea")
        self.assertIn("height: 78px", ta)
        self.assertIn("padding: 22px 96px 22px 28px", ta)
        self.assertIn("border-radius: 34px", ta)
        self.assertIn("font-size: 21px", ta)
        # P20 Native Main: the row remains in the composer's column but the
        # fit-content chips cluster around its center; bottom margins stay
        # frozen (20px gap above, 83px breathing below).
        sc = self._css_block(".shortcuts")
        self.assertIn("margin: 20px 0 83px", sc)
        self.assertIn("display: flex", sc)
        self.assertIn("gap: 8px", sc)
        self.assertIn("justify-content: center", sc)
        self.assertIn("width: auto", sc)

    def test_t11a_chips_slim_vertically_only(self) -> None:
        # Chip 52.75 -> 40.75px (vertical padding 16 -> 10px); the 13px
        # leading-edge horizontal anchor is untouched; P19-T17-E steps the
        # font one tier up (12.5 -> 13.5px) as the only size change.
        ch = self._css_block(".chip")
        self.assertIn("padding: 10px 13px", ch)
        self.assertIn("font-size: 13.5px", ch)
        self._css_block(".chip", "transform 0.18s ease-out")

    def test_t11b_thinking_field_moves_down_only(self) -> None:
        # The 500px field drops 63px in the flow (margin-top -85 -> -22px):
        # with the frozen engine geometry (TP_HEART y=210, crop radius
        # 9 cells * 20px = 180), the particle circle's top arc sits 30px
        # into the box, so it clears the request box bottom by 18px.
        self.assertIn("margin-top: -22px", self._css_block(".thinking-field"))
        self.assertNotIn("margin-top: -85px", self._html)
        engine = self._tp_engine()
        self.assertIn("TP_HEART = { x: 250, y: 210 }", engine)
        self.assertIn("TP_GRID_KEEP = 9", engine)
        self.assertIn("TP_GRID_CELL = 20", engine)
        req = self._css_block(".thinking-request")
        self.assertIn("margin-bottom: 10px", req)
        self.assertIn("font-size: 17px", req)

    def test_t11c_back_chevron_markup_and_style(self) -> None:
        # A light, container-less chevron at the result region's top-left:
        # transparent, borderless, quiet ink; folded by default. The
        # reduced-motion kill list covers it alongside the other surfaces.
        self.assertIn(
            '<button class="rec-back off" id="rec-back" type="button" aria-label="返回对话">‹</button>',
            self._html,
        )
        self.assertIn("display: none", self._css_block(".rec-back.off"))
        rb = self._css_block(".rec-back")
        self.assertIn("border: 0; background: none", rb)
        self.assertIn("font-size: 26px", rb)
        self.assertIn(".rec-trace, .rec-back,", self._html)

    def test_t11c_back_folds_without_touching_the_batch(self) -> None:
        # renderTrace hides the chevron in EVERY folded branch (it owns the
        # trace face); renderExpanded shows it again for the expanded face.
        rt = self._html[self._html.index("function renderTrace") : self._html.index("function renderExpanded")]
        self.assertIn('$("rec-back").classList.add("off");', rt)
        rex = self._html[self._html.index("function renderExpanded") : self._html.index("function collapseToTrace")]
        self.assertIn('$("rec-back").classList.remove("off");', rex)
        # The click is ONLY the fold (collapseToTrace) + focus: no fetch, no
        # renderExpanded, no batchCache mutation -- the cached batch survives
        # for the trace click to reopen verbatim.
        handler = self._html[
            self._html.index('$("rec-back").onclick = () => {') : self._html.index('document.addEventListener("keydown"')
        ]
        self.assertIn("collapseToTrace();", handler)
        self.assertIn("input.focus({ preventScroll: true });", handler)
        self.assertNotIn("fetch", handler)
        self.assertNotIn("renderExpanded", handler)
        self.assertNotIn("batchCache =", handler)
        # Escape is a desktop extra wired to the same fold, gated on the
        # expanded state, a visible convo stage, and a non-composing IME.
        esc = self._html[
            self._html.index('document.addEventListener("keydown"') : self._html.index('$("quit").onclick')
        ]
        self.assertIn('event.key === "Escape"', esc)
        self.assertIn("!composingText", esc)
        self.assertIn("expandedKey", esc)
        self.assertIn('$("view-convo").classList.contains("off")', esc)
        self.assertIn("collapseToTrace();", esc)

    def test_t11d_unified_press_language_is_restrained(self) -> None:
        # One press language across the five families: ~90ms press-down with
        # a settle + small scale, ~180ms ease-out release; primary actions
        # (play / send / preview 试听) get a brief warm-pink glow; text-style
        # actions stay opacity-only. No ripple, bounce or big motion. The
        # chevron (.rec-back) is the Fix02-R3 exception: its press is
        # geometry-free (opacity/color only, asserted below), because a scale
        # press shrank the whole wide hitbox after pointerdown and slid the
        # left-edge glyph out from under the pointer.
        for fam, pin in (
            (".chip:active", "translateY(1px) scale(0.95)"),
            (".controls button:active:not(:disabled)", "translateY(1px) scale(0.95)"),
            (".composer button:active:not(:disabled)", "translateY(calc(-50% + 1px)) scale(0.95)"),
            (".card .actions button:active:not(:disabled)", "translateY(1px) scale(0.97)"),
            (".rec-trace:active", "opacity: 0.7"),
            (".rec-back:active", "opacity: 0.7"),
        ):
            self.assertIn(pin, self._css_block(fam, containing=pin))
        # Fix02-R3 chevron contract: press feedback never moves hit geometry
        # (root cause: :active scale retargeted mouseup/click off the button)
        back_active = self._css_block(".rec-back:active")
        self.assertNotIn("scale(", back_active)
        self.assertNotIn("transform", back_active)
        # primary glows: play-pause, send chevron, preview/play pill bloom
        self.assertIn("drop-shadow(0 0 10px rgba(217, 168, 192, 0.6))", self._html)
        self.assertIn("drop-shadow(0 0 12px rgba(240, 184, 192, 0.7))", self._html)
        self.assertIn("0 0 12px rgba(240, 183, 172, 0.35)", self._html)
        # press-in ~90ms, release ~180ms ease-out
        self.assertIn("transition: transform 0.09s ease,", self._html)
        self.assertIn("transform 0.18s ease-out", self._html)
        # nothing un-macOS-like may appear anywhere (the word "bounce" survives
        # only inside the frozen P19-T9 comment "never a springy bounce")
        for banned in ("ripple", "spring(", "shake", "scale(1.1)"):
            self.assertNotIn(banned, self._html)

    def test_t12a_back_chevron_owned_by_the_convo_view_not_the_list(self) -> None:
        # P19-T12-A: the chevron LEFT the recommendation list and is now a
        # direct child of the conversation view -- the upper navigation
        # layer right below the player block. Same element, same id, same
        # press language; only DOM ownership and geometry moved.
        btn = '<button class="rec-back off" id="rec-back" type="button" aria-label="返回对话">‹</button>'
        self.assertIn(btn, self._html)
        self.assertEqual(self._html.count(btn), 1)
        convo_open = self._html.index('<div class="view off" id="view-convo">')
        convo_user = self._html.index('<div class="convo-user" id="convo-user">')
        recs_open = self._html.index('<div id="recs" class="off">')
        detail_open = self._html.index('<div class="cards" id="detail-cards">')
        btn_at = self._html.index(btn)
        self.assertTrue(
            convo_open < btn_at < convo_user,
            "chevron must sit between the convo view and its user column",
        )
        self.assertFalse(
            recs_open < btn_at < detail_open,
            "chevron must NOT live inside the recommendation list anymore",
        )
        rb = self._css_block(".rec-back")
        self.assertIn("margin: -13px 0 12px 20px", rb)
        self.assertIn("border: 0; background: none", rb)
        # P19-T13: hitbox tightened to a fixed 32x32 glyph target; the flex
        # column can no longer stretch it (align-self:flex-start), which is
        # what produced the ~370px-wide blind strip
        self.assertIn("display: inline-flex", rb)
        self.assertIn("width: 32px", rb)
        self.assertIn("height: 32px", rb)
        self.assertIn("padding: 0", rb)
        self.assertIn("align-self: flex-start", rb)

    def test_t12b_prose_renders_without_a_panel_background(self) -> None:
        # P19-T12-B: visible assistant prose paints NO rectangular panel.
        # The 0,3,0 rule now resets the background, which beats the later
        # .msg.assistant shorthand (0,2,0) that only styles the hidden
        # .messages buffer.
        visible = self._css_block(".convo-reply .msg.assistant")
        self.assertIn("background: none", visible)
        self.assertIn("margin-left: 0; padding: 0", visible)
        # the hidden buffer keeps its own surface, untouched by the fix
        buf = self._css_block(".msg.assistant", containing="var(--surface-soft)")
        self.assertIn("background: var(--surface-soft)", buf)

    def test_t12fix01_folded_face_actually_hides_the_rows(self) -> None:
        # P19-T12-Fix01 (carried through T17): rows must never linger on
        # the folded conversation face. T17 moved the rows into the
        # in-stage detail surface, so the conversation face owns NO card
        # container at all: #recs survives solely as the trace carrier,
        # #rec-detail hides through its own .off and takes over through
        # the .detail-open gate.
        self.assertIn("display: none", self._css_block(".cards.off"))
        self.assertIn("display: none", self._css_block("#recs.off"))
        self.assertIn("display: none", self._css_block("#rec-detail.off"))
        self.assertIn("display: none", self._css_block("#view-convo.detail-open > #recs"))
        rt = self._html[
            self._html.index("function renderTrace") : self._html.index("function renderExpanded")
        ]
        self.assertNotIn('$("cards")', rt)
        self.assertIn("recs.classList.remove(\"off\");", rt)
        # pollCards' same-batch branch is a pure no-op: no renderExpanded
        # call and no expandedKey write may exist between the re-sight
        # comment and the next major block.
        poll_start = self._html.index("async function pollCards")
        same = self._html[poll_start : self._html.index("/* P18-S3", poll_start)]
        self.assertNotIn("renderExpanded", same)
        self.assertNotIn("expandedKey =", same)

    def test_t12fix03_press_feedback_never_moves_the_hitbox(self) -> None:
        # P19-T12-Fix02-R3 confirmed root cause (Owner real-browser evidence):
        # the :active scale(0.94) shrank the whole wide chevron hitbox after
        # pointerdown and slid the left-edge glyph out from under the cursor,
        # so mouseup/click retargeted to #view-convo and the handler never
        # fired. Press feedback must be geometry-free: opacity/color only.
        active = self._css_block(".rec-back:active")
        self.assertNotIn("scale(", active)
        self.assertNotIn("transform", active)
        self.assertIn("opacity", active)
        self.assertIn("color", active)
        # the allowed <=1px directional nudge lives in :hover only, where it
        # cannot carry the region away from a left-edge pointer
        hover = self._css_block(".rec-back:hover")
        self.assertIn("translateX(-1px)", hover)
        # T12 placement bytes frozen
        self.assertIn("margin: -13px 0 12px 20px", self._css_block(".rec-back"))

    def test_t12fix03_speculative_fix02_pointer_gating_is_gone(self) -> None:
        # the Fix02 hypothesis (entering rows stealing the chevron's hit-test)
        # is disproven by real-browser evidence; its three speculative
        # pointer-gating additions are reverted. Rows are ordinary live
        # elements in every visible state -- no synthetic dead window.
        reveal = self._css_block(".card.reveal")
        self.assertNotIn("pointer-events", reveal)
        self.assertIn("animation: card-in 0.5s cubic-bezier(0.22, 0.61, 0.36, 1) both", reveal)
        reduced = self._css_block(".card.reveal", containing="animation: none !important")
        self.assertNotIn("pointer-events", reduced)
        # no card-family rule in the whole cards/card region touches pointer-events
        # card-family region only (past .rec-trace waits unrelated prose that
        # merely mentions the words "pointer-events")
        cards_region = self._html[self._html.index(".cards {") : self._html.index(".rec-trace {")]
        self.assertNotIn("pointer-events", cards_region)
        # the row loop wires no listeners, no timers, no rejoin state
        re_ = self._html[
            self._html.index("function renderExpanded") : self._html.index("function collapseToTrace")
        ]
        self.assertNotIn("animationend", re_)
        self.assertNotIn("animationcancel", re_)
        self.assertNotIn("rejoin", re_)
        self.assertNotIn("addEventListener", re_)
        self.assertNotIn("setTimeout", re_)
        self.assertNotIn("setInterval", re_)
        # the stagger delays are untouched: the visual language stays frozen
        self.assertIn('Math.min(i * 70, 420) + "ms"', re_)
        # no speculative topmost-layer addition on the chevron
        back = self._css_block(".rec-back")
        self.assertNotIn("z-index", back)
        self.assertNotIn("position: relative", back)

    def test_t12fix03_temporary_diagnostic_block_is_removed(self) -> None:
        # the ?diag/#diag instrumentation served its purpose; no diagnostic
        # code may remain in the production UI.
        self.assertNotIn("P19-T12-Fix02-R2", self._html)
        self.assertNotIn("diag", self._html.lower())
        self.assertNotIn("typeof location", self._html)

    def test_t14b_r4_ownership_guard_keeps_the_structured_face(self) -> None:
        # P19-T14-B-R4 (the captured FAILURE fold, fixed): the proven seam
        # was deliverChat's no-fresh-batch path answering with the raw
        # numbered recommendation prose while the session's structured face
        # was expanded -- the prose became the stage content and the
        # completion's renderTrace() folded the rows. The ownership guard
        # folds that prose to its one-line T9 lead and keeps the batch
        # expanded, so the final state is ALWAYS: short lead + structured
        # rows -- never: long numbered prose + hidden rows. The raw path
        # survives untouched for replies that own neither a batch nor an
        # expanded face (ordinary prose, the concise fallback).
        html = self._html
        # the probe mirrors the server door's shape exactly, presentation-
        # only: no fetch, no card building, no generation anywhere near it
        self.assertIn("function looksLikeNumberedSongList", html)
        probe = html[
            html.index("function looksLikeNumberedSongList") : html.index(
                "function convertStageReplyToLead"
            )
        ]
        self.assertIn(r"/^\s*\d{1,3}\s*[.、)）]\s*/.test(line)", probe)
        self.assertIn('line.indexOf("—") >= 0', probe)
        self.assertIn("return numbered >= 2 && dashed >= 1;", probe)
        self.assertNotIn("fetch(", probe)
        # the guard wraps only the no-fresh-batch branch: the raw rendered
        # append stays the default path below it, and the guard path itself
        # stages NO full prose node (nothing for renderConversation to lift
        # into the stage and no rendered HTML on the lead line)
        chat = html[
            html.index("async function deliverChat") : html.index("function sendChat")
        ]
        self.assertIn(
            '(expandedKey !== null || batch) && looksLikeNumberedSongList(data.reply || "")',
            chat,
        )
        self.assertIn("stageProseReply = null;", chat)
        self.assertIn(
            'appendMessage("assistant", data.reply || "", data.reply_html)', chat
        )
        guard_start = chat.index("P19-T14-B-R4 ownership guard")
        guard_block = chat[
            guard_start : chat.index(
                'appendMessage("assistant", data.reply || "", data.reply_html)',
                guard_start,
            )
        ]
        self.assertNotIn("data.reply_html", guard_block)
        self.assertIn('appendMessage("assistant", lead)', guard_block)
        self.assertIn('node.classList.add("lead")', guard_block)
        # no new backend surface, no prompt work, no numbered-prose parsing
        # into cards: deliverChat stays a pure presentation consumer of the
        # frozen {reply, reply_html, rounds_capped, batch} payload
        self.assertNotIn("t14b", chat)

    def test_t14b_r4_completion_never_folds_the_expanded_face(self) -> None:
        # renderConversation no longer folds an expanded batch at
        # completion: renderTrace stays the exclusive fold path for the
        # legitimate escapes (turn start, explicit back, the empty trace),
        # never for a completion that just answered the turn.
        rc = self._html[
            self._html.index("function renderConversation") : self._html.index(
                "/* ---- chat: correction-while-running"
            )
        ]
        self.assertIn("if (!expandedKey) renderTrace();", rc)
        self.assertIn('else { $("recs").classList.add("off"); }', rc)
        self.assertLess(
            rc.index("if (!expandedKey) renderTrace();"),
            rc.index('else { $("recs").classList.add("off"); }'),
        )
        # the legitimate fold callers still call renderTrace directly:
        # collapseToTrace (turn start + explicit back) clears the key then
        # folds -- renderTrace itself is untouched, never globally banned
        collapse = self._html[
            self._html.index("function collapseToTrace") : self._html.index(
                "function cardButtonCard"
            )
        ]
        self.assertIn("if (!expandedKey) return;", collapse)
        self.assertIn("expandedKey = null;", collapse)
        self.assertIn("renderTrace();", collapse)
        self.assertIn('$("view-convo").scrollTop = detailReturnScroll;', collapse)

    def test_t13_rec_back_hitbox_is_a_fixed_32x32_glyph_target(self) -> None:
        # P19-T13: the chevron's clickable area is the glyph, not the row.
        # Real-browser diagnosis measured ~370x22; the stage grid's flex
        # column was stretching the block button across the column. Fixed
        # 32x32, centered glyph, zero padding, align-self:flex-start so the
        # parent can never widen it again; the T12 margin pin (origin and
        # left alignment) and the R3 geometry-free press stay byte-frozen.
        back = self._css_block(".rec-back")
        self.assertIn("display: inline-flex", back)
        self.assertIn("align-items: center", back)
        self.assertIn("justify-content: center", back)
        self.assertIn("width: 32px", back)
        self.assertIn("height: 32px", back)
        self.assertIn("padding: 0", back)
        self.assertIn("align-self: flex-start", back)
        self.assertIn("margin: -13px 0 12px 20px", back)
        self.assertNotIn("width: 100%", back)
        self.assertNotIn("stretch", back)
        # R3 press contract survives: opacity/color only, never geometry
        active = self._css_block(".rec-back:active")
        self.assertNotIn("scale(", active)
        self.assertNotIn("transform", active)
        self.assertIn("opacity: 0.7", active)
        # the fold-away projector is unchanged
        self.assertIn("display: none", self._css_block(".rec-back.off"))

    def test_t13_rec_trace_hitbox_hugs_the_text_not_the_row(self) -> None:
        # P19-T13: "已推荐 N 首…… ›" must only be clickable where the text
        # is. Old rule was display:block; width:100% -- the whole row was a
        # target. Now shrink-to-fit (fit-content + tiny 4px 6px padding),
        # inline so #recs's block flow still gives it its own line; the
        # right-hand whitespace after the text is dead area.
        trace = self._css_block(".rec-trace")
        self.assertIn("display: inline-block", trace)
        self.assertIn("width: fit-content", trace)
        self.assertIn("padding: 4px 6px", trace)
        self.assertNotIn("width: 100%", trace)
        self.assertNotIn("display: block", trace)
        # pressed/hidden states and the trace text path are untouched
        self.assertIn("opacity: 0.7", self._css_block(".rec-trace:active"))
        self.assertIn("display: none", self._css_block(".rec-trace.off"))
        rt = self._html[
            self._html.index("function renderTrace") : self._html.index("function renderExpanded")
        ]
        self.assertIn("traceText(batchCache.label, batchCache.cards.length)", rt)

    def test_t14c_shortcut_chips_hug_their_text(self) -> None:
        # The four shortcuts size from their own text + the frozen 13px/10px
        # padding and form one compact centered group. Fit-content chips +
        # nowrap guarantee one single line at 660x880.
        # Font stepped 12.5 -> 13.5px; icon plate and padding untouched.
        sc = self._css_block(".shortcuts")
        self.assertIn("display: flex", sc)
        self.assertIn("flex-wrap: nowrap", sc)
        self.assertIn("justify-content: center", sc)
        self.assertIn("gap: 8px", sc)
        self.assertIn("width: auto", sc)
        self.assertIn("margin: 20px 0 83px", sc)
        self.assertNotIn("grid-template-columns", sc)
        ch = self._css_block(".chip")
        self.assertNotIn("width:", ch)
        self.assertNotIn("min-width", ch)
        self.assertNotIn("flex:", ch)
        self.assertIn("padding: 10px 13px", ch)
        self.assertIn("font-size: 13.5px", ch)
        self.assertIn("white-space: nowrap", ch)
        # icon grid untouched: the 22px mask plate still precedes the label
        self.assertIn("width: 22px; height: 22px;", self._css_block(".chip::before"))

    def test_t14d_multi_turn_traces_keep_their_own_lines(self) -> None:
        # P19-T14-D: every turn's batch keeps its own trace line for the
        # whole page session. batchHistory is pure in-memory state (no
        # storage API anywhere in the trace machinery); polls re-sight keys
        # but never duplicate entries, never reorder them, never move the
        # face; restoring an older line re-renders that exact cached batch
        # with zero network; only one batch is expanded at a time and the
        # static #rec-trace stays the NEWEST line (frozen count sentence).
        def fn_slice(marker: str) -> str:
            a = self._html.index(marker)
            b = self._html.index("\n}\n", a)
            return self._html[a:b]

        state = self._html
        self.assertIn("let batchHistory = [];", state)
        self.assertIn("let traceStacksEl = null;", state)
        # session memory only -- no storage API in the whole trace subsystem
        for marker in (
            "function renderTrace",
            "function traceButtonFor",
            "async function pollCards",
            "async function deliverChat",
        ):
            self.assertNotIn("localStorage", fn_slice(marker))
            self.assertNotIn("indexedDB", fn_slice(marker))

        # rememberBatch: re-sight refreshes in place, first sight appends
        rem = self._html[self._html.index("function rememberBatch"):self._html.index("function traceButtonFor")]
        self.assertIn("entry.cards = cardData.cards; return entry;", rem)
        self.assertIn("batchHistory.push(entry);", rem)

        # renderTrace rebuilds the whole list: newest batch is the static
        # #rec-trace line, older batches become .rec-trace buttons stacked
        # ABOVE it in first-seen order
        trace = fn_slice("function renderTrace")
        self.assertIn("batchHistory.forEach((entry) => {", trace)
        self.assertIn("if (entry.id === batchCache.id) return;", trace)
        self.assertIn('older.className = "rec-trace-old";', trace)
        self.assertIn('$("recs").insertBefore(older, $("rec-trace"));', trace)
        self.assertIn('line.className = "rec-trace";', trace)
        self.assertIn("traceButtonFor(entry, line);", trace)
        self.assertIn(
            'traceText(batchCache.label, batchCache.cards.length);',
            trace,
        )

        # restoring an older line opens that exact cached batch's detail
        # face: no fetch, no chat, no regeneration -- a pure render from
        # cache through the T17 detail gate (openDetail owns the key and
        # the sync guard, the button only labels and opens)
        button = fn_slice("function traceButtonFor")
        self.assertIn("openDetail(entry);", button)
        self.assertIn("traceText(entry.label, entry.cards.length)", button)
        self.assertNotIn("fetch(", button)

        # exactly one batch expanded: the expanded face clears the built
        # older lines (renderTrace rebuilds them on the next fold)
        expanded = fn_slice("function renderExpanded")
        self.assertIn("if (traceStacksEl) traceStacksEl.replaceChildren();", expanded)

        # pollCards: a re-sighted id only refreshes its rows in place --
        # the current face never moves, no duplicate trace, no reorder,
        # while an unknown durable/out-of-band batch is ignored because
        # ownership can arrive only on the matching /api/chat response
        poll = fn_slice("async function pollCards")
        self.assertIn("batchHistory.some((entry) => entry.id === cardData.id)", poll)
        self.assertIn("if (!known) { return; }", poll)
        self.assertIn("rememberBatch(cardData);", poll)
        self.assertNotIn("renderExpanded", poll)
        self.assertNotIn("batchCache", poll)

        # deliverChat: fresh = run id never seen this session AND not just
        # shown; a re-sent id keeps its single trace entry in place (its
        # reply still folds to the lead under the R4 ownership guard when
        # it is recommendation prose). The batch arrives INSIDE the chat
        # reply (same fetch) -- no cards fetch here.
        chat = fn_slice("async function deliverChat")
        self.assertIn("if (batch && batch.cards.length)", chat)
        self.assertEqual(chat.count("rememberBatch(batch);"), 1)
        self.assertNotIn("await fetchCardData();", chat)

        # the older lines keep the T13 text-hugging hitbox (same .rec-trace
        # class) but each owns its own line: block, 6px apart, none above
        # the first
        self.assertIn("display: block", self._css_block(".rec-trace-old .rec-trace"))
        self.assertIn("margin-top: 6px", self._css_block(".rec-trace-old .rec-trace"))
        self.assertIn("margin-top: 0", self._css_block(".rec-trace-old .rec-trace:first-child"))

    def test_t16_session_turns_archive_as_ordered_blocks(self) -> None:
        # P19-T16: user requests must survive the whole frontend session in
        # true order. Each completed pair (user request + short reply) is
        # archived as an ordered .convo-turn block that stacks ABOVE the
        # live pair; the archive is plain in-memory DOM (no storage API),
        # so a reload clears it by construction.
        state = self._html
        self.assertIn("let turnSeq = 0;", state)
        self.assertIn("let liveTurnSeq = null;", state)
        self.assertIn("const turnBlocks = new Map();", state)
        # the archive container sits between the back-chevron and the live
        # pair, so archived turns read above the live pair in true order
        back = state.index('<button class="rec-back off" id="rec-back"')
        turns = state.index('<div class="convo-turns" id="convo-turns"></div>')
        user = state.index('<div class="convo-user" id="convo-user">')
        self.assertLess(back, turns)
        self.assertLess(turns, user)
        self.assertIn(".convo-turns .convo-turn { margin-bottom: 16px; }", state)
        self.assertIn(".convo-turns .turn-trace { margin-top: 4px; }", state)

    def test_t16_archive_moves_only_the_live_pair(self) -> None:
        archive = self._html[
            self._html.index("function archiveLivePair") : self._html.index(
                "/* the stage's two conversation lines"
            )
        ]
        # the fixed slots are emptied by MOVING their child nodes into a
        # fresh block -- never by re-rendering, never by touching history
        self.assertIn('$("convo-user").firstElementChild', archive)
        self.assertIn('$("convo-reply").firstElementChild', archive)
        self.assertIn('block.className = "convo-turn";', archive)
        self.assertIn("block.dataset.turnSeq = liveTurnSeq;", archive)
        self.assertIn("turnBlocks.set(liveTurnSeq, block);", archive)
        self.assertIn('$("convo-turns").append(block);', archive)
        self.assertIn("syncTurnTraces();", archive)
        # session-local metadata only -- no storage API, no network
        self.assertNotIn("localStorage", archive)
        self.assertNotIn("indexedDB", archive)
        self.assertNotIn("fetch(", archive)

    def test_t16_render_conversation_lifts_real_nodes_not_selectors(self) -> None:
        # P19-T16 root cause: the stage used the CSS selector
        # ".msg.user:last-of-type" which can never match (every .msg sits
        # in a bare div, so the user node is never the last of its type)
        # -- the user line never displayed at all. The pick now reads real
        # nodes from the hidden buffer and MOVES the last user and last
        # assistant; the buffer's greeting (index 0) can never migrate.
        def fn_slice(marker: str) -> str:
            a = self._html.index(marker)
            b = self._html.index("\n}\n", a)
            return self._html[a:b]

        conv = fn_slice("function renderConversation")
        self.assertIn('document.querySelectorAll("#messages .msg.user")', conv)
        self.assertIn("bufferUsers[bufferUsers.length - 1]", conv)
        self.assertIn("bufferReplies.length > 1", conv)
        self.assertNotIn(".msg.user:last-of-type", conv)
        self.assertIn('if (userNode) $("convo-user").append(userNode);', conv)
        self.assertIn('if (replyNode) $("convo-reply").append(replyNode);', conv)

    def test_t16_traces_live_inside_their_turn_blocks(self) -> None:
        # P19-T16 trace binding: a batch born of a chat turn keeps its trace
        # line INSIDE that turn's archived block (attached by seq at send,
        # restored by traceButtonFor with zero fetch). The detached stack
        # stays for unbound batches only; the static #rec-trace line shows
        # for the live pair and hides once that turn is archived (no
        # duplicate line); the expanded batch hides its own line.
        def fn_slice(marker: str) -> str:
            a = self._html.index(marker)
            b = self._html.index("\n}\n", a)
            return self._html[a:b]

        chat = fn_slice("async function deliverChat")
        # archival happens at the START of the next deliver -- the pair that
        # just completed belongs to its own numbered turn
        self.assertIn("archiveLivePair();", chat)
        self.assertIn("const seq = ++turnSeq;", chat)
        self.assertIn("liveTurnSeq = seq;", chat)
        self.assertIn("turnEntry.turnSeq = seq;", chat)

        trace = fn_slice("function renderTrace")
        self.assertIn("if (entry.turnSeq !== undefined) return;", trace)
        self.assertIn('$("rec-trace").classList.toggle(', trace)
        self.assertIn("turnBlocks.has(batchCache.turnSeq)", trace)
        self.assertIn("syncTurnTraces();", trace)

        sync = fn_slice("function syncTurnTraces")
        self.assertIn("turnBlocks.get(entry.turnSeq)", sync)
        self.assertIn('line.className = "rec-trace turn-trace";', sync)
        self.assertIn("traceButtonFor(entry, line);", sync)
        self.assertIn("block.append(line);", sync)
        self.assertIn("expandedKey === entry.key", sync)

        expanded = fn_slice("function renderExpanded")
        self.assertIn("syncTurnTraces();", expanded)

    def test_t16_bottom_safe_area_scroll_padding(self) -> None:
        # P19-T16 + T17-F safe area: the scroller's bottom band covers the
        # composer (measured ~88px of real canvas) plus a small rest
        # margin, so the last row of an expanded batch scrolls fully above
        # the composer. T17-F re-homes the band: it now lives on the
        # detail face only (#rec-detail), while the ordinary conversation
        # reverts to a small 8px bottom pad -- no mid-page blank for a
        # short chat, and nothing about the composer/shortcuts moves.
        convo = self._css_block("#view-convo")
        self.assertIn("overflow-y: auto", convo)
        self.assertIn("min-height: 0", convo)
        self.assertIn("padding: 30px 2px 8px", convo)
        detail = self._css_block("#rec-detail")
        self.assertIn("padding: 0 2px 100px", detail)
        self.assertIn("display: none", self._css_block("#rec-detail.off"))
        # the detail face replaces the conversation content only through
        # the .detail-open gate -- every conversation child is hidden
        # while the detail face is up (trace carrier included); the group
        # is asserted as its own rule + as raw selector lines (only the
        # last selector of the group is a valid _css_block target)
        self.assertIn("display: none", self._css_block("#view-convo.detail-open > #recs"))
        for child in (".convo-turns", ".convo-user", ".convo-reply"):
            self.assertIn("#view-convo.detail-open > " + child + ",", self._html)

    # ---- P19-T17 contract pins ----

    def test_t17a_detail_view_is_in_stage_not_a_third_window(self) -> None:
        # P19-T17-A: the recommendation detail lives INSIDE the existing
        # Main Player (inside #view-convo) -- no third .view face, no new
        # page, no new shell mode. Opening/closing it only toggles state
        # and classes: zero fetch, zero regeneration, zero /api/chat.
        import re

        state = self._html
        # exactly three stage faces (greet / convo / thinking) and no
        # fourth -- the detail surface is a child of view-convo
        self.assertEqual(len(re.findall(r'class="[^"]*\bview\b[^"]*"', state)), 3)
        convo = state.index('id="view-convo"')
        thinking = state.index('id="view-thinking"')
        detail = state.index('id="rec-detail"')
        self.assertLess(convo, detail)
        self.assertLess(detail, thinking)
        self.assertIn('id="detail-cards"', state)
        self.assertIn('id="detail-lead"', state)

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        opened = fn_slice("function renderExpanded")
        self.assertIn('$("view-convo").classList.add("detail-open");', opened)
        self.assertIn('$("rec-detail").classList.remove("off");', opened)
        self.assertIn('$("rec-back").classList.remove("off");', opened)
        self.assertIn('$("detail-lead").textContent = traceText(entry.label, entry.cards.length);', opened)
        self.assertIn("cardButtonCard(card)", opened)
        self.assertNotIn("fetch(", opened)

        folded = fn_slice("function collapseToTrace")
        self.assertIn('$("view-convo").classList.remove("detail-open");', folded)
        self.assertIn('$("rec-detail").classList.add("off");', folded)
        self.assertIn('$("view-convo").scrollTop = detailReturnScroll;', folded)

        # opening saves the convo scroll only when switching face-to-face
        # is NOT already in detail mode; opening itself triggers no fetch
        opener = fn_slice("function openDetail")
        self.assertIn("expandedKey = entry.key;", opener)
        self.assertIn("detailReturnScroll = $(\"view-convo\").scrollTop;", opener)
        self.assertIn("renderExpanded(entry);", opener)
        self.assertNotIn("fetch(", opener)

        # the trace buttons route into the detail face, never the old
        # inline expansion
        button = fn_slice("function traceButtonFor")
        self.assertIn("openDetail(entry);", button)

    def test_t17c_conversation_typography_pure_text_hierarchy(self) -> None:
        # P19-T17-C: the conversation face is a pure text hierarchy --
        # user line: no plate (rectangle backdrop stripped to none, no
        # radius/padding), weakest ink, 15px; assistant rendered prose:
        # the clear step up in size (18px), comfortable 1.8 line-height,
        # slightly brighter/warmer ink; assistant lead stays one half-step
        # below the full prose. Same font family everywhere (no new
        # family declaration entered).
        user = self._css_block(".convo-user")
        self.assertIn("font-size: 15px", user)
        self.assertIn("var(--ink-faint)", user)
        plate = self._css_block('#view-convo .convo-user [class^="msg"]')
        self.assertIn("background: none", plate)
        self.assertIn("border-radius: 0", plate)
        self.assertIn("padding: 0", plate)
        self.assertIn("max-width: 100%", plate)
        self.assertIn("color: var(--ink-faint)", plate)
        rendered = self._css_block(".convo-reply .msg.assistant.rendered")
        self.assertIn("font-size: 18px", rendered)
        self.assertIn("line-height: 1.8", rendered)
        self.assertIn("color: #f5eee8", rendered)
        lead = self._css_block(".convo-reply .msg.assistant.lead")
        self.assertIn("font-size: 16.5px", lead)
        # no new typeface: font-family only appears on base rules
        fam = self._html
        self.assertNotIn("font-family", plate + rendered + lead)

    def test_t17d_trace_labels_are_short_and_turn_local(self) -> None:
        # P19-T17-D: trace lines carry a lightweight label derived from
        # that turn's own request metadata -- frozen at send time on the
        # batch entry, never re-derived later, never the whole request
        # text. Fallback is the bare "推荐 · N 首 ›". Zero network: the
        # label path touches no fetch, no provider, no storage.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        trace_text = fn_slice("function traceText")
        self.assertIn("(label || \"推荐\")", trace_text)
        self.assertIn('+ " · "', trace_text)
        self.assertNotIn("fetch(", trace_text)

        label = state[state.index("function requestLabel"):state.index("function traceText")]
        self.assertIn("/类似|相似/", label)
        self.assertIn('"今晚推荐"', label)
        self.assertIn('"新推荐"', label)
        self.assertIn('return "";', label)
        self.assertNotIn("fetch(", label)

        # frozen at send: the live turn's entry carries the user text and
        # the label computed once from it (plus the asking-time ref name),
        # and the same values propagate onto batchCache so the static
        # newest line (and the archive-dedup) speaks this turn's identity
        chat = fn_slice("async function deliverChat")
        self.assertIn("turnEntry.userText = text;", chat)
        self.assertIn("turnEntry.label = batch.label || requestLabel(", chat)
        self.assertIn('label: data.batch.label || ""', chat)
        self.assertIn("batchCache.turnSeq = seq;", chat)
        self.assertIn("batchCache.label = turnEntry.label;", chat)
        self.assertEqual(chat.count("requestLabel("), 1)
        # nothing stuffs the whole request into a trace line
        self.assertNotIn("turnEntry.label = text;", chat)

    def test_t17f_scroll_restore_and_detail_scroll_independence(self) -> None:
        # P19-T17-A/F: back-restore keeps the exact pre-open scroll of the
        # conversation (single saved value, restored on fold) while the
        # detail face scrolls its own five rows above the composer band.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        opener = fn_slice("function openDetail")
        # face-to-face switching must NOT overwrite the saved return
        # position -- the guard reads the pre-mutation state
        self.assertIn(".classList.contains(\"detail-open\")", opener)
        self.assertIn("if (!switching) detailReturnScroll", opener)
        # detail rows render into the detail scroll container inside the
        # same scroller: the convo scrollport is the detail scrollport
        self.assertIn('entry.cards.forEach((card, i) => {', fn_slice("function renderExpanded"))

    def test_t17r1_jump_links_share_one_warm_accent(self) -> None:
        # P19-T17-R1: every genuine click-to-jump link -- rendered prose
        # anchors (live reply + archived turns) and the Apple Music jumper
        # pill in card rows -- carries ONE soft warm pink-purple accent.
        # No browser default blue anywhere; hover is a single gentle
        # brighten step; visited never falls back to the UA purple; the
        # disabled/unavailable semantics stay shared and untouched.
        state = self._html
        tokens = self._css_block(":root")
        self.assertIn("--accent: #d3a1d7", tokens)
        self.assertIn("--accent-bright: #e0b6e6", tokens)
        # prose anchors: one rest color in both containers
        self.assertIn("color: var(--accent)", self._css_block(".convo-reply .msg.rendered a", "color"))
        self.assertIn("color: var(--accent)", self._css_block(".msg.rendered a", "color"))
        # hover = one gentle brighten step; visited = pinned to the accent
        # (never the UA purple), in both the live reply and archived turns
        self.assertIn(".convo-reply .msg.rendered a:hover { color: var(--accent-bright); }", state)
        self.assertIn(".convo-reply .msg.rendered a:visited { color: var(--accent); }", state)
        self.assertIn(".msg.rendered a:hover { color: var(--accent-bright); }", state)
        self.assertIn(".msg.rendered a:visited { color: var(--accent); }", state)
        # the Apple Music jumper pill: rest carries the link accent,
        # hover takes the same one brighten step
        self.assertIn("color: var(--accent);", self._css_block(".card .actions button.open-am"))
        hover = self._css_block(".card .actions button.open-am:hover:not(:disabled)")
        self.assertIn("color: var(--accent-bright);", hover)
        # its 播放/试听 siblings keep their own rest + warm hover -- the
        # shared action-pill language is not repainted
        self.assertIn("color: var(--ink-soft)", self._css_block(".card .actions button"))
        self.assertIn("color: var(--warm)", self._css_block(".card .actions button:hover:not(:disabled)"))
        # disabled/pending semantics are the shared ones, un-mutated
        disabled = self._css_block(".card .actions button:disabled")
        self.assertIn("opacity: 0.4", disabled)
        self.assertIn("cursor: default", disabled)

    # ---- P19-T17-R2 contract pins ----

    def test_t17r2a_one_click_one_request_one_run_one_trace(self) -> None:
        # P19-T17-R2-A regression 1: ONE shortcut click -> ONE /api/chat ->
        # ONE batch registration -> ONE visible trace. The four chips share
        # the single send entry point (sendBtn.onclick); sendChat has
        # exactly one deliverChat outlet; deliverChat issues exactly one
        # chat request, carries the run identity back through ONE
        # rememberBatch (fresh-run guard), and the face rebuilds once.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        # one dispatch path for chips: propose the text through the same
        # send entry point -- no chip-local fetch/post, no second channel
        chips = state[
            state.index('document.querySelectorAll(".shortcuts .chip")') : state.index(
                '$("rec-trace").onclick'
            )
        ]
        self.assertIn("sendBtn.onclick();", chips)
        self.assertNotIn("post(", chips)
        self.assertNotIn("fetch(", chips)
        self.assertEqual(state.count("sendBtn.onclick = sendChat;"), 1)

        # sendChat: one outlet, busy clicks queue a correction instead
        send = state[state.index("function sendChat") : state.index("sendBtn.onclick = sendChat;")]
        self.assertIn("if (chatBusy) { queueCorrection(text); return; }", send)
        self.assertEqual(send.count("deliverChat("), 1)

        # deliverChat: exactly one chat request, exactly one registration
        chat = fn_slice("async function deliverChat")
        self.assertEqual(chat.count('post("/api/chat"'), 1)
        self.assertEqual(chat.count("rememberBatch(batch);"), 1)
        self.assertIn("if (batch && batch.cards.length)", chat)
        # one entry per run id: a re-sight never duplicates the trace
        rem = state[state.index("function rememberBatch") : state.index("function traceButtonFor")]
        self.assertIn("if (entry.id === cardData.id) { entry.cards = cardData.cards; return entry; }", rem)
        self.assertEqual(rem.count("batchHistory.push(entry);"), 1)

    def test_t17r2a_same_run_never_doubles_turn_trace_and_detached_trace(self) -> None:
        # P19-T17-R2-A regression 2: one run shows ONE trace. A turn-owned
        # entry keeps its line inside its turn block (syncTurnTraces) and
        # is skipped by the detached loop; once ANY batch is turn-owned,
        # the detached stack is suppressed entirely so a session-initial /
        # out-of-band entry can never project a second line. Identity is
        # the only dedupe key -- label text is never consulted.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        trace = fn_slice("function renderTrace")
        self.assertIn(
            "const hasTurnOwned = batchHistory.some((entry) => entry.turnSeq !== undefined);",
            trace,
        )
        self.assertIn("if (hasTurnOwned) return;", trace)
        # guard order inside the loop: same-run skip, then turn-owned skip,
        # then the ownership suppression -- all id/seq based
        idx_id = trace.index("if (entry.id === batchCache.id) return;")
        idx_turn = trace.index("if (entry.turnSeq !== undefined) return;")
        idx_owned = trace.index("if (hasTurnOwned) return;")
        self.assertLess(idx_id, idx_turn)
        self.assertLess(idx_turn, idx_owned)
        # the detached loop compares only batch degrees -- never labels
        detached_top = trace[trace.index("batchHistory.forEach"):idx_owned]
        self.assertNotIn("traceText", detached_top)
        self.assertNotIn("label", detached_top)
        # the archived turn path puts EXACTLY one line per turn block
        sync = fn_slice("function syncTurnTraces")
        self.assertIn('let line = block.querySelector(".turn-trace");', sync)
        self.assertEqual(sync.count("block.append(line);"), 1)

    def test_t17r2a_out_of_band_batch_stays_hidden(self) -> None:
        # P20 recommendation ownership: the newest durable run is not the
        # current answer. An out-of-band/session-initial poll cannot enter
        # batchHistory or batchCache, and the live trace requires matching
        # turnSeq ownership. Archived turn traces remain independently
        # available through syncTurnTraces().
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        trace = fn_slice("function renderTrace")
        self.assertIn(
            "batchCache && liveTurnSeq !== null && batchCache.turnSeq === liveTurnSeq",
            trace,
        )
        self.assertIn("if (!batchHistory.length || !stageReply || !currentTurnOwnsBatch)", trace)
        self.assertIn('recs.classList.add("off");', trace)
        self.assertIn("syncTurnTraces();", trace)

        poll = fn_slice("async function pollCards")
        self.assertIn("if (!known) { return; }", poll)
        self.assertNotIn("batchHistory.push", poll)
        self.assertNotIn("openDetail", poll)

    def test_t17r2a_three_sequential_clicks_three_runs_three_traces(self) -> None:
        # P19-T17-R2-A regression 4: three sequential clicks -> three
        # requests -> three runs -> three archived turns, each with its
        # OWN single trace line and no duplicate anywhere: each delivery
        # archives the previous pair at its start, so every batch lands in
        # its own turnSeq; the static line holds only the newest; the
        # detached suppression and the per-block single-line guard leave
        # exactly one line per run.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        chat = fn_slice("async function deliverChat")
        # each click archives the prior turn and takes a fresh seq
        self.assertIn("archiveLivePair();", chat)
        self.assertIn("const seq = ++turnSeq;", chat)
        self.assertEqual(chat.count('post("/api/chat"'), 1)

        sync = fn_slice("function syncTurnTraces")
        self.assertIn('let line = block.querySelector(".turn-trace");', sync)
        self.assertIn("if (!line) {", sync)
        self.assertEqual(sync.count("block.append(line);"), 1)

        # the newest line is the single static button; older stacks hold
        # one element each, and the R2-A guard prevents any second pass
        trace = fn_slice("function renderTrace")
        self.assertEqual(trace.count('$("rec-trace-text").textContent ='), 1)
        self.assertEqual(trace.count("older.append(line)"), 1)
        self.assertIn("if (hasTurnOwned) return;", trace)

    def test_t17r2b_trace_lines_carry_the_link_accent(self) -> None:
        # P19-T17-R2-B: every genuinely clickable trace line -- the live
        # static #rec-trace, archived turn traces (.rec-trace.turn-trace)
        # and the legal detached/out-of-band lines (.rec-trace-old
        # .rec-trace, the same base class) -- shares the ONE R1 link
        # accent: rest --accent, hover the single --accent-bright step.
        # No browser blue, no UA purple; geometry, font and the 2px nudge
        # stay byte-identical; body prose and 播放/试听 pills untouched.
        state = self._html
        rule = self._css_block(".rec-trace")
        self.assertIn("color: var(--accent)", rule)
        self.assertNotIn("--ink-faint", rule)
        self.assertNotIn("--warm", rule)
        hover = self._css_block(".rec-trace:hover")
        self.assertIn("color: var(--accent-bright)", hover)
        # geometry/font/hitbox untouched (T13 fit-content + padding cache)
        self.assertIn("font-size: 13px", rule)
        self.assertIn("display: inline-block", rule)
        self.assertIn("width: fit-content", rule)
        self.assertIn("padding: 4px 6px", rule)
        self.assertIn("transform: translateX(2px)", hover)
        # the three trace surfaces reuse the one class -> one accent rule
        self.assertIn('line.className = "rec-trace turn-trace";', state)
        self.assertIn('line.className = "rec-trace";', state)
        self.assertIn('class="rec-trace off" id="rec-trace"', state)
        # the detached container only overrides layout (block + margins),
        # never color -- the accent inherits from the base class
        self.assertNotIn("color", self._css_block(".rec-trace-old .rec-trace"))

    def test_t17r2b_r1_jump_link_colors_unchanged(self) -> None:
        # P19-T17-R2-B regression 6: the R1 surface (Apple Music jumper
        # pill + rendered prose anchors, live and archived) keeps its
        # exact token usage -- R2-B only extends the same accent to the
        # trace lines, it never repaints the R1 owners.
        state = self._html
        tokens = self._css_block(":root")
        self.assertIn("--accent: #d3a1d7", tokens)
        self.assertIn("--accent-bright: #e0b6e6", tokens)
        self.assertIn("color: var(--accent);", self._css_block(".card .actions button.open-am"))
        self.assertIn(
            "color: var(--accent-bright);",
            self._css_block(".card .actions button.open-am:hover:not(:disabled)"),
        )
        self.assertIn(".convo-reply .msg.rendered a { color: var(--accent); }", state)
        self.assertIn(".convo-reply .msg.rendered a:hover { color: var(--accent-bright); }", state)
        self.assertIn(".msg.rendered a:visited { color: var(--accent); }", state)
        # body 播放/试听 pills stay on their own shared colors
        self.assertIn("color: var(--ink-soft)", self._css_block(".card .actions button"))
        self.assertIn("color: var(--warm)", self._css_block(".card .actions button:hover:not(:disabled)"))

    def test_t17r2a_trace_restore_stays_zero_network(self) -> None:
        # P19-T17-R2-A regression 7: exact batch restore stays zero
        # network / zero regeneration -- the trace click only rebuilds the
        # cached rows through the T17 detail gate; the R2-A suppression is
        # a fold-time render guard, it touches no request path.
        state = self._html

        def fn_slice(marker: str) -> str:
            a = state.index(marker)
            b = state.index("\n}\n", a)
            return state[a:b]

        button = fn_slice("function traceButtonFor")
        self.assertIn("openDetail(entry);", button)
        self.assertIn("traceText(entry.label, entry.cards.length)", button)
        self.assertNotIn("fetch(", button)
        self.assertNotIn("post(", button)
        self.assertNotIn("fetch(", fn_slice("function openDetail"))
        self.assertNotIn("fetch(", fn_slice("function renderExpanded"))
