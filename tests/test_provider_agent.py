"""P10.8: Provider agent loop tests -- fake provider + the REAL P09 service on a temp store."""

import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from music_agent.final_response_boundary import FINAL_ANSWER_CONTRACT
from datetime import datetime, timezone
from pathlib import Path

from music_agent.action_attempt import (
    ActionAttemptStatus,
    ActionReadbackState,
    create_direct_action_attempt,
    mark_action_executing,
    record_action_execution,
)
from music_agent.agent_client import AgentClient
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.apple_music_catalog import CatalogTrack
from music_agent.intent_router import resolve_turn_plan
from music_agent.provider_agent import (
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_TOOL_NAMES,
    MAX_DISCOVER_PER_RUN,
    RECOMMENDATION_UNFULFILLED_FALLBACK,
    _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE,
    _EMPTY_FINAL_ANSWER_CLOSEOUT,
    _EXPLANATION_TOOL_NAMES,
    _LIBRARY_QUERY_TOOL_NAMES,
    _FEEDBACK_TOOL_NAMES,
    _FINAL_ANSWER_EXECUTIONS,
    _GENERATION_FAILURE_CLOSEOUT,
    _GENERIC_CURRENT_PLAYER_TARGET_ERROR_CODE,
    _GENERATION_RESULT_MAX_CHARS,
    _GENERATION_TOOL_NAMES,
    _MAX_TOOL_RESULT_CHARS,
    _PLAYBACK_TOOL_NAMES,
    _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES,
    _POST_GENERATION_DELEGATED_READBACK_TOOL_NAMES,
    _POST_GENERATION_CLOSEOUT_ERROR_CODE,
    _POST_GENERATION_CLOSEOUT_MESSAGE,
    _PREVIEW_TOOL_NAMES,
    _RECOMMENDATION_TOOL_NAMES,
    _ROUND_CAP_CLOSEOUT,
    _TURN_CLARIFICATION_CLOSEOUT,
    _RUN_READER_RESULT_MAX_CHARS,
    _S4_BASE_PROMPT,
    _S4_DISCOVERY_PROMPT,
    _S4_EXPLANATION_PROMPT,
    _S4_LIBRARY_QUERY_PROMPT,
    _S4_FEEDBACK_PROMPT,
    _S4_MODULE_BASE,
    _S4_MODULE_EXPLANATION,
    _S4_MODULE_REFERENT,
    _S4_ALL_MODULES,
    _S4_PLAYBACK_PROMPT,
    _S4_PREVIEW_PROMPT,
    _S4_PROMPT_CLAUSES,
    _S4_RECOMMENDATION_PROMPT,
    _S5_PREFETCH_TOOL_NAMES,
    _ResolvedRecommendationSemantics,
    action_result_play_preview_downgrade,
    action_result_preview_started,
    ProviderAgentLoop,
    ProviderError,
    ProviderLoopConfig,
    ProviderLoopToolExecution,
    _capture_fresh_promoted_ids,
    _apply_recommendation_semantics,
    _compose_system_prompt,
    _discover_budget_exhausted_message,
    _s5_prefetch_recommendation_reads_enabled,
    _select_system_prompt,
    _select_task_tools,
    _resolve_recommendation_semantics,
    _recommendation_semantics_prompt,
    _recommendation_scope_ids,
    formal_play_started,
    generation_succeeded,
    looks_like_numbered_song_list,
    play_intent_preview_downgrade,
    preview_path_started,
    PLAY_PREVIEW_DOWNGRADE_FALLBACK,
)
from music_agent.provider_contract import (
    ProviderAuthError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
    ProviderToolCall,
    ProviderToolSchema,
)
from music_agent.provider_instrumentation import measure_round_input
import music_agent.provider_recommendation_policy as recommendation_policy
from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS
from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.routed_client import RoutedAgentClient
from music_agent.track_similarity import SimilarityExecutionContext

CLIENT_ID = "agt_44444444-4444-4444-8444-444444444444"


def fixture_model() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "canonical_music_model.json").read_text(encoding="utf-8"))


class FakeProvider:
    """Scripted provider: returns queued responses, records every chat call."""

    def __init__(self, responses: list[ProviderResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if not self.responses:
            raise AssertionError("FakeProvider exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DelegatedPlaybackAdapter:
    """Deterministic Music.app boundary for the post-generation delegation test."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.now_pid: str | None = None

    def read_player_state(self) -> str:
        return "playing" if self.now_pid is not None else "stopped"

    def read_now_playing(self):
        from music_agent.playback_control import NowPlaying, PlayerState

        return NowPlaying(
            state=(
                PlayerState.PLAYING
                if self.now_pid is not None
                else PlayerState.STOPPED
            ),
            persistent_id=self.now_pid,
            name="Synthetic Duet" if self.now_pid is not None else None,
            artist="Artist Alpha" if self.now_pid is not None else None,
            album="Synthetic Album" if self.now_pid is not None else None,
        )

    def play(self) -> None:
        self.calls.append(("play", ()))

    def pause(self) -> None:
        self.calls.append(("pause", ()))

    def next_track(self) -> None:
        self.calls.append(("next_track", ()))

    def previous_track(self) -> None:
        self.calls.append(("previous_track", ()))

    def play_track(self, persistent_id: str) -> None:
        self.calls.append(("play_track", (persistent_id,)))
        self.now_pid = persistent_id


def text_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        ProviderMessage(ProviderMessageRole.ASSISTANT, text=text),
        ProviderStopReason.END_TURN,
        {"input_tokens": 1, "output_tokens": 2},
    )


def tool_response(calls: list[ProviderToolCall]) -> ProviderResponse:
    return ProviderResponse(
        ProviderMessage(ProviderMessageRole.ASSISTANT, tool_calls=tuple(calls)),
        ProviderStopReason.TOOL_USE,
        {"input_tokens": 1, "output_tokens": 2},
    )


def planning_tool_response(calls: list[ProviderToolCall], text: str) -> ProviderResponse:
    """A tool round whose assistant message carries planning preamble text.

    Mirrors the live leakage shape: the model narrates what it intends to do
    ("让我先检查一下……") in the same message that requests the tool calls.
    """
    return ProviderResponse(
        ProviderMessage(ProviderMessageRole.ASSISTANT, text=text, tool_calls=tuple(calls)),
        ProviderStopReason.TOOL_USE,
        {"input_tokens": 1, "output_tokens": 2},
    )


def empty_generation_result(tool: str) -> AgentToolResult:
    """Canned execution_error mirroring the live empty-refusal failure shape."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code="empty_recommendation",
        error_message=(
            "generation produced zero items; nothing was written to recommendation "
            "history. Relax genres/exclusions or discover new candidates and retry"
        ),
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def empty_generation_result_with_diagnostics(
    tool: str,
    *,
    reason: str,
    excluded_previous_count: int = 0,
) -> AgentToolResult:
    diagnostics = {
        "input_target_count": 6,
        "after_direction_filter_count": 6,
        "direct_evidence_count": 6,
        "positive_evidence_count": 6,
        "negative_evidence_count": 0,
        "candidate_count": 6,
        "excluded_previous_count": excluded_previous_count,
        "reason": reason,
        "recommended_next_action": "test fixture",
    }
    return replace(
        empty_generation_result(tool),
        error_message=(
            "generation produced zero items; nothing was written to recommendation "
            "history. diagnostics: "
            + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
        ),
    )




def named_play_search_result(matches: list[dict]) -> AgentToolResult:
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="search_library_tracks",
        outcome=AgentToolOutcome.OK,
        payload={"matched_count": len(matches), "matches": matches},
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 9, 16, 0, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def named_play_match(
    canonical_id: str,
    *,
    name: str = "Wendy",
    artist: str = "Test Artist",
    route: str = "preview_only",
) -> dict:
    return {
        "target_id": canonical_id,
        "name": name,
        "artist_name": artist,
        "playback": {"route": route, "label": route},
        "provenance": {"kind": "apple_music_catalog"},
        "bindings": {"apple_music_catalog_id": "catalog-test"},
    }

def ok_generation_result(tool: str) -> AgentToolResult:
    """Canned OK result for a generation call (loop-visible success)."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload={"run_id": "rcm_11111111-1111-4111-8111-111111111111", "items": []},
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_generation_batch(tool: str) -> AgentToolResult:
    """Canned OK generation result carrying a non-empty batch -- the loop-level
    fact of "successful non-empty" is outcome=ok (an empty batch never returns
    ok under the P09 contract); the payload mirrors that fact."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload={
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 1,
            "items": [
                {
                    "target_id": "trk_11111111-1111-4111-8111-111111111111",
                    "name": "夜曲",
                    "artist_name": "测试艺人",
                    "playback": {"route": "library", "label": "library"},
                }
            ],
        },
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def evidence_item(
    name: str,
    *,
    artist_name: str | None = None,
    basis: list[dict] | None = None,
    fresh_this_request: bool = False,
    route: str = "library",
    **extra: object,
) -> dict:
    """One generation-result item under the post-Fix09 display contract: a
    name, the playback fact, the fresh identity, and the shared evidence
    block (mechanism agreeing with the basis rows, P20-Fix09). ``extra``
    rides along (score_total, direct_state, label, explanation, ...) to prove
    the deterministic renderer ignores everything outside that contract."""
    entry: dict = {
        "name": name,
        "playback": {"route": route, "label": route},
        "evidence": {
            "mechanism": (
                "直接偏好"
                if any(row.get("provenance") == "直接" for row in (basis or []))
                else "推断偏好"
            ),
            "basis": list(basis) if basis is not None else [],
        },
        "fresh_this_request": fresh_this_request,
    }
    if artist_name is not None:
        entry["artist_name"] = artist_name
    entry.update(extra)
    return entry


def ok_generation_evidence_batch(tool: str, items: list[dict]) -> AgentToolResult:
    """Canned OK generation result whose items carry the shared evidence block
    -- the shape a real post-Fix09 generation success resolves into, and the
    only shape the deterministic presenter accepts."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload={
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": len(items),
            "items": items,
        },
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def failed_generation_result(tool: str) -> AgentToolResult:
    """Canned non-empty-failure execution_error for a generation call."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code="generation_engine_error",
        error_message="the engine blew up",
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_discovery_result() -> AgentToolResult:
    """Canned OK result for a discover_catalog_tracks execution."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="discover_catalog_tracks",
        outcome=AgentToolOutcome.OK,
        payload={"term": "夜晚", "discovered_count": 2},
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_already_bound_discovery_result() -> AgentToolResult:
    """A completed discovery that expands no persisted canonical supply."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="discover_catalog_tracks",
        outcome=AgentToolOutcome.OK,
        payload={
            "term": "IU",
            "returned_count": 10,
            "already_bound_count": 10,
            "staged_count": 0,
            "promoted_count": 0,
            "items": [],
        },
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def replayed_discovery_result() -> AgentToolResult:
    """Canned journal-replay answer for discover_catalog_tracks: the P09 layer
    already recorded this request identity and returns the earlier outcome
    without re-running the catalog search (replayed=True, still a dispatch)."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="discover_catalog_tracks",
        outcome=AgentToolOutcome.OK,
        payload={"term": "夜晚", "discovered_count": 2, "from_journal": True},
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=True,
    )


def failed_discovery_result() -> AgentToolResult:
    """Canned transient Catalog failure for discover_catalog_tracks. P20-PerfFix02
    failure recovery: a failed search is delivered fail-honest and stays FREE
    against the per-turn budget (only genuine OK executions charge)."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="discover_catalog_tracks",
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code="catalog_http_error",
        error_message="catalog unreachable",
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_read_result(tool: str) -> AgentToolResult:
    """Canned OK result for a read tool (loop-visible success)."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload={"ok": True},
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_tool_result(tool: str, payload: Mapping[str, object]) -> AgentToolResult:
    """Canned OK result with an explicit structured payload."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload=dict(payload),
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


def ok_delegated_play_readback(target_id: str) -> AgentToolResult:
    return ok_tool_result(
        "get_now_playing",
        {
            "now_playing": {
                "state": "playing",
                "persistent_id": "SYNTH-PID",
                "name": "Synthetic",
                "artist": "Synthetic Artist",
                "album": None,
            },
            "context": "agent_selected",
            "agent_channel": {
                "state": "library",
                "canonical_id": target_id,
            },
            "player_canonical_id": target_id,
            "canonical_resolution": "binding",
        },
    )


def ok_preview_started(target_id: str) -> AgentToolResult:
    return ok_tool_result(
        "preview_catalog_track",
        {"canonical_id": target_id, "started": True},
    )


def ok_search_result_for_artist(artist: str) -> AgentToolResult:
    """Read-only semantic-resolution result for a named artist fixture."""
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="search_library_tracks",
        outcome=AgentToolOutcome.OK,
        payload={
            "matches": [
                {
                    "target_id": "trk_11111111-1111-4111-8111-111111111111",
                    "name": "Bye, Summer",
                    "artist_name": artist,
                },
                {
                    "target_id": "trk_22222222-2222-4222-8222-222222222222",
                    "name": "Love Poem",
                    "artist_name": artist,
                },
            ]
        },
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=False,
    )


# S5 (round reduction): ordinary recommendation runs deterministically prefetch
# the three anchor reads before provider round 1. RecordingClient-queued
# expectations must account for them -- both in the call log and in the canned
# result queue (the prefetch consumes the queue first).
S5_PREFETCH_RECORDED = [
    "get_active_context",
    "list_recommendation_runs",
    "list_feedback_observations",
]


def s5_prefetch_padding() -> list[AgentToolResult]:
    """S5: three OK read results, one per deterministic prefetch read, to pad a
    RecordingClient result queue ahead of the provider-requested calls."""
    return [
        ok_read_result(name) for name in S5_PREFETCH_RECORDED
    ]


class RecordingClient(AgentClient):
    """Stub AgentClient for the generation-budget tests: records every tool the
    loop actually executes and returns canned results in queued order -- a
    supplied result per execution, defaulting to the empty-refusal error.

    P15-S3-S3D: also captures the INTERNAL ``fresh_canonical_ids`` execution
    kwarg per call (None when the loop passed none) -- the loop-level transport
    fact, never part of the model payload."""

    def __init__(
        self,
        service: SharedAgentService,
        results: list[AgentToolResult] | None = None,
    ) -> None:
        super().__init__(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
        )
        self.recorded: list[str] = []
        self.recorded_payloads: list[dict] = []
        self.fresh_kwargs: list[tuple[str, ...] | None] = []
        self.scope_kwargs: list[tuple[str, ...] | None] = []
        self.similarity_contexts: list[SimilarityExecutionContext | None] = []
        self._results = list(results) if results is not None else []

    def call(
        self,
        tool,
        payload,
        *,
        request_id=None,
        issued_at=None,
        completed_at=None,
        fresh_canonical_ids=None,
        recommendation_scope_ids=None,
        similarity_context=None,
    ):
        self.recorded.append(str(tool))
        self.recorded_payloads.append(dict(payload))
        self.fresh_kwargs.append(
            tuple(fresh_canonical_ids) if fresh_canonical_ids is not None else None
        )
        self.scope_kwargs.append(
            tuple(recommendation_scope_ids)
            if recommendation_scope_ids is not None
            else None
        )
        self.similarity_contexts.append(similarity_context)
        if self._results:
            return self._results.pop(0)
        return empty_generation_result(str(tool))


GENERATE_PAYLOAD = json.dumps(
    {"target_ids": ["trk_11111111-1111-4111-8111-111111111111"]}
)


def project_tools(names: frozenset[str]) -> list[ProviderToolSchema]:
    """S3: the registry projected to a task-tool name group (registry order).

    Exactly the list ``ProviderAgentLoop`` sends for a classified task when the
    caller configured the full registry -- the S3 expectations compare against
    this projection so the tests assert real schema objects, not names.
    """
    return [item for item in PROVIDER_TOOL_SCHEMAS if item.name in names]


class ProviderRecommendationPolicyExtractionTest(unittest.TestCase):
    def test_recommendation_policy_helpers_are_reexported_from_extracted_module(self) -> None:
        import music_agent.provider_agent as provider_agent

        helper_names = (
            "_ResolvedRecommendationSemantics",
            "_apply_recommendation_semantics",
            "_bind_current_track_similarity_seed",
            "_deterministic_generic_recovery_call",
            "_deterministic_preference_fallback_calls",
            "_generic_direct_history_exhausted",
            "_generic_recommendation_result_only_current_player",
            "_generic_recommendation_targets_only_current_player",
            "_recommendation_scope_ids",
            "_recommendation_semantics_prompt",
            "_resolve_recommendation_semantics",
        )
        for name in helper_names:
            self.assertIs(
                getattr(provider_agent, name),
                getattr(recommendation_policy, name),
                name,
            )


class ProviderAgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture_model())

    def _service(self, policy: AgentClientPolicy) -> SharedAgentService:
        return SharedAgentService(
            self.database_path, clients=AgentClientRegistry({CLIENT_ID: policy})
        )

    def _client(self, service: SharedAgentService) -> AgentClient:
        return AgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
        )

    def test_turn_resolver_deterministic_plan_never_calls_interpreter_provider(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)

        class InterpreterCapableProvider:
            supports_turn_interpreter = True

            def chat(self, system, messages, tools):
                raise AssertionError("deterministic turn must not call interpreter")

        loop = ProviderAgentLoop(
            InterpreterCapableProvider(), self._client(service), PROVIDER_TOOL_SCHEMAS
        )
        plan = loop.resolve_turn("推荐几首歌")
        self.assertEqual(plan.primary.value, "recommendation")
        self.assertEqual(plan.semantic_source.value, "deterministic")

    def test_turn_resolver_unknown_uses_interpreter_and_returns_existing_turn_plan(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)

        class InterpreterCapableProvider:
            supports_turn_interpreter = True

            def __init__(self):
                self.calls = []

            def chat(self, system, messages, tools):
                self.calls.append((system, tuple(messages), tuple(tools)))
                return ProviderResponse(
                    ProviderMessage(
                        ProviderMessageRole.ASSISTANT,
                        text=json.dumps(
                            {
                                "intent": "recommendation",
                                "recommendation": {
                                    "mode": "generic",
                                    "requested_count": 5,
                                    "scene": None,
                                    "seed": None,
                                },
                                "action": None,
                                "requires_clarification": False,
                                "reason": None,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                    ProviderStopReason.END_TURN,
                    {},
                )

        provider = InterpreterCapableProvider()
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        plan = loop.resolve_turn("推荐")
        self.assertEqual(plan.primary.value, "recommendation")
        self.assertEqual(plan.recommendation.mode, "generic")
        self.assertEqual(plan.semantic_source.value, "llm_interpreted")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][2], ())

    def test_interpreted_generic_recommendation_uses_existing_recommendation_surface(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)

        class InterpreterCapableProvider(FakeProvider):
            supports_turn_interpreter = True

        provider = InterpreterCapableProvider([
            text_response(json.dumps({
                "intent": "recommendation",
                "recommendation": {
                    "mode": "generic",
                    "requested_count": 5,
                    "scene": None,
                    "seed": None,
                },
                "action": None,
                "requires_clarification": False,
                "reason": None,
            }, ensure_ascii=False)),
            text_response("暂未生成。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)

        result = loop.run("推荐")

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(
            provider.calls[1]["tools"],
            project_tools(_RECOMMENDATION_TOOL_NAMES),
        )
        self.assertEqual(result.tool_executions, ())
        self.assertFalse(result.rounds_capped)

    def test_interpreter_failure_clarifies_without_entering_tool_loop(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)

        class InterpreterCapableProvider(FakeProvider):
            supports_turn_interpreter = True

        provider = InterpreterCapableProvider([text_response("not-json")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)

        result = loop.run("弄一下")

        self.assertEqual(result.final_text, _TURN_CLARIFICATION_CLOSEOUT)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.tool_executions, ())
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["tools"], [])

    def test_provider_loop_consumes_caller_turn_plan_without_reclassifying(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([])
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS
        )
        text = "我喜欢 Yorushika 的歌"
        turn_plan = resolve_turn_plan(text)

        with patch(
            "music_agent.provider_agent.resolve_turn_plan",
            side_effect=AssertionError("caller plan must remain the semantic source"),
        ):
            result = loop.run(text, turn_plan=turn_plan)

        self.assertEqual(result.final_text, "明白，你喜欢 Yorushika 的歌。")
        self.assertEqual(result.rounds, 0)
        self.assertEqual(provider.calls, [])

    def _seed_two_positives(self) -> None:
        """Seeds the two fixture-model tracks with real favorited positive
        observations (the standard promotion-end-to-end prelude)."""
        from music_agent.preference_attribution import (
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import (
            PreferencePersistenceRepository,
        )
        from music_agent.source_observation import ObservedValue

        for track_id in ("trk_11111111-1111-4111-8111-111111111111",
                         "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"):
            with PreferencePersistenceRepository(self.database_path) as repository:
                repository.record_observation(
                    SignalIdentity(
                        PreferenceTargetReference(
                            PreferenceTargetKind.TRACK, track_id
                        ),
                        "apple_music",
                        "favorited",
                    ),
                    ObservedValue.value(True),
                    observed_at="2026-08-16T00:00:00+00:00",
                    provenance="fixture_seed",
                )

    def _canonical_ids_for_catalog_ids(self, catalog_ids: set[str]) -> list[str]:
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
        return [
            track["id"]
            for track in model["tracks"]
            if track["external_ids"].get("apple_music_catalog_id") in catalog_ids
        ]

    def test_explicit_semantics_resolve_generic_artist_and_track_targets(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = self._client(service)
        artist = _resolve_recommendation_semantics(
            client, "推荐 Artist Alpha 的音乐"
        )
        self.assertEqual(artist.mode, "artist_constraint")
        self.assertEqual(artist.target_kind, "artist")
        self.assertEqual(
            set(artist.target_ids),
            {
                "trk_11111111-1111-4111-8111-111111111111",
                "trk_22222222-2222-4222-8222-222222222222",
            },
        )
        self.assertEqual(_recommendation_scope_ids(artist), artist.target_ids)
        track = _resolve_recommendation_semantics(
            client, "找几首和 Synthetic Solo 类似的歌"
        )
        self.assertEqual(track.target_kind, "track")
        self.assertEqual(
            track.target_ids,
            ("trk_22222222-2222-4222-8222-222222222222",),
        )

    def test_explicit_seed_replaces_model_target_but_preserves_scene_arguments(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        semantics = _resolve_recommendation_semantics(
            self._client(service),
            "我喜欢 Artist Alpha，推荐几首适合晚上听的歌",
        )
        call = ProviderToolCall(
            "g1",
            "generate_inferred_recommendation",
            json.dumps(
                {
                    "target_ids": ["trk_44444444-4444-4444-8444-444444444444"],
                    "limit": 5,
                    "genres": ["Synthetic Pop"],
                }
            ),
        )
        rewritten = _apply_recommendation_semantics(call, semantics)
        arguments = json.loads(rewritten.arguments)
        self.assertEqual(set(arguments["target_ids"]), set(semantics.target_ids))
        self.assertEqual(arguments["genres"], ["Synthetic Pop"])
        prompt = _recommendation_semantics_prompt(semantics)
        self.assertIn("mode=preference_seed", prompt)
        self.assertIn("scene=evening", prompt)
        self.assertIn("不是艺人硬约束", prompt)
        self.assertIsNone(_recommendation_scope_ids(semantics))

    def test_generic_recommendation_rejects_current_player_as_sole_target(self) -> None:
        current_id = "trk_11111111-1111-4111-8111-111111111111"
        broader_id = "trk_22222222-2222-4222-8222-222222222222"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps(
                                {"target_ids": [current_id, current_id], "limit": 1}
                            ),
                        )
                    ]
                ),
                tool_response(
                    [
                        ProviderToolCall(
                            "g2",
                            "generate_recommendation",
                            json.dumps({"target_ids": [broader_id], "limit": 1}),
                        )
                    ]
                ),
                text_response("已为你准备好推荐。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            [
                ok_tool_result(
                    "get_active_context",
                    {"player": {"canonical_id": current_id}},
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
                ok_generation_batch("generate_recommendation"),
            ],
        )
        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐几首歌")

        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        self.assertEqual(client.recorded_payloads[-1]["target_ids"], [broader_id])
        self.assertEqual(client.recorded_payloads[-1]["limit"], 5)
        self.assertIsNone(client.similarity_contexts[-1])
        first_guard = json.loads(
            provider.calls[1]["messages"][-1].tool_results[0].content
        )
        self.assertEqual(
            first_guard["error_code"], _GENERIC_CURRENT_PLAYER_TARGET_ERROR_CODE
        )
        self.assertEqual(len(result.tool_executions), 1)

    def test_generic_final_sole_current_player_recovers_to_inferred(self) -> None:
        current_id = "trk_11111111-1111-4111-8111-111111111111"
        other_id = "trk_33333333-3333-4333-8333-333333333333"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps(
                                {
                                    "target_ids": [
                                        current_id,
                                        "trk_22222222-2222-4222-8222-222222222222",
                                    ],
                                    "limit": 5,
                                }
                            ),
                        )
                    ]
                ),
                text_response("已为你准备好推荐。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                ok_tool_result(
                    "get_active_context",
                    {
                        "player": {
                            "state": "playing",
                            "canonical_id": current_id,
                            "canonical_resolution": "binding",
                        }
                    },
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
                ok_generation_evidence_batch(
                    "generate_recommendation",
                    [evidence_item("Current", target_id=current_id)],
                ),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [evidence_item("Recovered", target_id=other_id)],
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐音乐")

        self.assertEqual(
            client.recorded[-2:],
            ["generate_recommendation", "generate_inferred_recommendation"],
        )
        self.assertEqual(result.recommendation_payload["item_count"], 1)
        self.assertEqual(
            result.recommendation_payload["items"][0]["target_id"], other_id
        )
        self.assertEqual(
            [
                execution.origin
                for execution in result.tool_executions
                if execution.name in _GENERATION_TOOL_NAMES
            ],
            ["provider_requested", "policy_injected"],
        )
        self.assertEqual(len(provider.calls), 2)

    def test_generic_partial_non_current_result_remains_successful(self) -> None:
        current_id = "trk_11111111-1111-4111-8111-111111111111"
        other_id = "trk_33333333-3333-4333-8333-333333333333"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps(
                                {
                                    "target_ids": [
                                        current_id,
                                        "trk_22222222-2222-4222-8222-222222222222",
                                    ],
                                    "limit": 5,
                                }
                            ),
                        )
                    ]
                ),
                text_response("已为你准备好推荐。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                ok_tool_result(
                    "get_active_context",
                    {
                        "player": {
                            "state": "playing",
                            "canonical_id": current_id,
                            "canonical_resolution": "binding",
                        }
                    },
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
                ok_generation_evidence_batch(
                    "generate_recommendation",
                    [evidence_item("Other", target_id=other_id)],
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐音乐")

        self.assertEqual(client.recorded[-1], "generate_recommendation")
        self.assertEqual(
            client.recorded.count("generate_inferred_recommendation"), 0
        )
        self.assertEqual(result.recommendation_payload["item_count"], 1)
        self.assertEqual(
            result.recommendation_payload["items"][0]["target_id"], other_id
        )

    def test_generic_direct_history_exhaustion_forces_inferred_recovery(self) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": target_ids, "limit": 5}),
                        ),
                        ProviderToolCall(
                            "g2",
                            "generate_recommendation",
                            json.dumps({"target_ids": target_ids, "limit": 5}),
                        ),
                    ]
                ),
                text_response("已为你准备好推荐。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=s5_prefetch_padding()
            + [
                empty_generation_result_with_diagnostics(
                    "generate_recommendation",
                    reason="all_eligible_candidates_excluded",
                    excluded_previous_count=2,
                ),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [
                        evidence_item(
                            "Recovered",
                            target_id="trk_33333333-3333-4333-8333-333333333333",
                        )
                    ],
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐音乐")

        generation_calls = [
            name for name in client.recorded if name in _GENERATION_TOOL_NAMES
        ]
        self.assertEqual(
            generation_calls,
            ["generate_recommendation", "generate_inferred_recommendation"],
        )
        self.assertEqual(len(generation_calls), 2)
        self.assertEqual(ProviderLoopConfig().max_generation_attempts, 2)
        self.assertEqual(result.recommendation_payload["item_count"], 1)
        self.assertEqual(
            [
                execution.origin
                for execution in result.tool_executions
                if execution.name in _GENERATION_TOOL_NAMES
            ],
            ["provider_requested", "policy_injected"],
        )

    def test_generic_inferred_recovery_failure_uses_fixed_closeout(self) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": target_ids, "limit": 5}),
                        )
                    ]
                )
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=s5_prefetch_padding()
            + [
                empty_generation_result_with_diagnostics(
                    "generate_recommendation",
                    reason="all_eligible_candidates_excluded",
                    excluded_previous_count=2,
                ),
                empty_generation_result("generate_inferred_recommendation"),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐音乐")

        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertIsNone(result.recommendation_payload)
        self.assertEqual(
            [name for name in client.recorded if name in _GENERATION_TOOL_NAMES],
            ["generate_recommendation", "generate_inferred_recommendation"],
        )

    def test_generic_non_history_empty_does_not_force_inferred_recovery(self) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": target_ids, "limit": 5}),
                        )
                    ]
                ),
                text_response("当前没有足够的直接偏好证据。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=s5_prefetch_padding()
            + [
                empty_generation_result_with_diagnostics(
                    "generate_recommendation",
                    reason="no_direct_evidence",
                    excluded_previous_count=0,
                )
            ],
        )

        ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS).run("推荐音乐")

        self.assertEqual(client.recorded[-1], "generate_recommendation")
        self.assertNotIn("generate_inferred_recommendation", client.recorded)

    def test_non_generic_history_exhaustion_does_not_force_generic_recovery(self) -> None:
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": [target_id], "limit": 5}),
                        )
                    ]
                ),
                text_response("当前没有合适候选。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[ok_search_result_for_artist("Artist Alpha")]
            + s5_prefetch_padding()
            + [
                empty_generation_result_with_diagnostics(
                    "generate_recommendation",
                    reason="all_eligible_candidates_excluded",
                    excluded_previous_count=1,
                )
            ],
        )

        ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS).run(
            "推荐 Artist Alpha 的歌"
        )

        self.assertEqual(client.recorded[-1], "generate_recommendation")
        self.assertNotIn("generate_inferred_recommendation", client.recorded)

    def test_current_track_similarity_uses_positive_seed_only_as_inferred_basis(self) -> None:
        # Synthetic Solo is favorited/rated positive in the canonical fixture.
        # Even so, it is mandatory exclusion rather than candidate supply.
        current_id = "trk_22222222-2222-4222-8222-222222222222"
        candidate_id = "trk_11111111-1111-4111-8111-111111111111"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": [current_id], "limit": 1}),
                        )
                    ]
                ),
                text_response("已为你准备好相似推荐。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            [
                ok_tool_result(
                    "get_active_context",
                    {
                        "player": {
                            "canonical_id": current_id,
                            "canonical_resolution": "binding",
                        }
                    },
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [
                        evidence_item(
                            "Synthetic Duet",
                            artist_name="Artist Alpha",
                            target_id=candidate_id,
                        )
                    ],
                ),
            ],
        )
        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("找类似这首的")

        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED + ["generate_inferred_recommendation"],
        )
        self.assertEqual(client.recorded_payloads[-1]["target_ids"], [current_id])
        self.assertEqual(
            client.recorded_payloads[-1]["exclude_target_ids"], [current_id]
        )
        self.assertIs(client.recorded_payloads[-1]["avoid_previous_runs"], True)
        self.assertEqual(client.recorded_payloads[-1]["limit"], 5)
        self.assertEqual(
            client.similarity_contexts[-1], SimilarityExecutionContext(current_id)
        )
        self.assertIn("mode=similarity_seed", provider.calls[0]["system"])
        self.assertIn("seed_source=current_track", provider.calls[0]["system"])
        self.assertIn(current_id, provider.calls[0]["system"])
        self.assertEqual(
            [item["target_id"] for item in result.recommendation_payload["items"]],
            [candidate_id],
        )
        self.assertEqual(len(result.tool_executions), 1)

    def test_current_track_similarity_preserves_non_seed_candidates_but_excludes_seed(
        self,
    ) -> None:
        seed_id = "trk_22222222-2222-4222-8222-222222222222"
        candidate_id = "trk_11111111-1111-4111-8111-111111111111"
        semantics = _ResolvedRecommendationSemantics(
            mode="similarity_seed",
            target=None,
            target_kind="track",
            scene=None,
            target_ids=(seed_id,),
            requested_count=5,
            seed_source="current_track",
        )
        rewritten = _apply_recommendation_semantics(
            ProviderToolCall(
                "g1",
                "generate_recommendation",
                json.dumps(
                    {
                        "target_ids": [seed_id, candidate_id],
                        "limit": 1,
                        "avoid_previous_runs": False,
                    }
                ),
            ),
            semantics,
        )
        arguments = json.loads(rewritten.arguments)

        self.assertEqual(rewritten.name, "generate_inferred_recommendation")
        self.assertEqual(arguments["target_ids"], [seed_id, candidate_id])
        self.assertEqual(arguments["exclude_target_ids"], [seed_id])
        self.assertIs(arguments["avoid_previous_runs"], True)
        self.assertEqual(arguments["limit"], 5)

    def test_current_track_similarity_unresolved_seed_fails_closed(self) -> None:
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps(
                                {
                                    "target_ids": [
                                        "trk_11111111-1111-4111-8111-111111111111"
                                    ],
                                    "limit": 5,
                                }
                            ),
                        )
                    ]
                )
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            [
                ok_tool_result(
                    "get_active_context",
                    {
                        "player": {
                            "canonical_id": None,
                            "canonical_resolution": None,
                        }
                    },
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("找类似这首的")

        self.assertEqual(client.recorded, S5_PREFETCH_RECORDED)
        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertIsNone(result.recommendation_payload)

    def test_current_track_similarity_seed_in_success_payload_fails_closed(self) -> None:
        seed_id = "trk_22222222-2222-4222-8222-222222222222"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": [seed_id], "limit": 5}),
                        )
                    ]
                )
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            [
                ok_tool_result(
                    "get_active_context",
                    {
                        "player": {
                            "canonical_id": seed_id,
                            "canonical_resolution": "binding",
                        }
                    },
                ),
                ok_read_result("list_recommendation_runs"),
                ok_read_result("list_feedback_observations"),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [evidence_item("Synthetic Solo", target_id=seed_id)],
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("找类似这首的")

        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertIsNone(result.recommendation_payload)

    def test_two_consecutive_current_track_similarity_turns_never_return_seed(
        self,
    ) -> None:
        seed_id = "trk_22222222-2222-4222-8222-222222222222"
        candidates = (
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_44444444-4444-4444-8444-444444444444",
        )
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            f"g{index}",
                            "generate_recommendation",
                            json.dumps({"target_ids": [seed_id], "limit": 1}),
                        )
                    ]
                )
                if step % 2 == 0
                else text_response("已准备好相似推荐。")
                for step, index in enumerate((1, 1, 2, 2))
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        results: list[AgentToolResult] = []
        for index, candidate_id in enumerate(candidates, start=1):
            results.extend(
                [
                    ok_tool_result(
                        "get_active_context",
                        {
                            "player": {
                                "canonical_id": seed_id,
                                "canonical_resolution": "binding",
                            }
                        },
                    ),
                    ok_read_result("list_recommendation_runs"),
                    ok_read_result("list_feedback_observations"),
                    ok_generation_evidence_batch(
                        "generate_inferred_recommendation",
                        [
                            evidence_item(
                                f"Candidate {index}", target_id=candidate_id
                            )
                        ],
                    ),
                ]
            )
        client = RecordingClient(service, results)
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)

        delivered = [
            loop.run("找类似这首的"),
            loop.run("再找一些类似这首的"),
        ]

        generated = [
            (tool, payload)
            for tool, payload in zip(client.recorded, client.recorded_payloads)
            if tool in _GENERATION_TOOL_NAMES
        ]
        self.assertEqual(
            [tool for tool, _payload in generated],
            ["generate_inferred_recommendation", "generate_inferred_recommendation"],
        )
        for _tool, payload in generated:
            self.assertEqual(payload["exclude_target_ids"], [seed_id])
            self.assertIs(payload["avoid_previous_runs"], True)
            self.assertEqual(payload["limit"], 5)
        generated_contexts = [
            context
            for tool, context in zip(client.recorded, client.similarity_contexts)
            if tool in _GENERATION_TOOL_NAMES
        ]
        self.assertEqual(
            generated_contexts,
            [SimilarityExecutionContext(seed_id), SimilarityExecutionContext(seed_id)],
        )
        self.assertEqual(
            [
                result.recommendation_payload["items"][0]["target_id"]
                for result in delivered
            ],
            list(candidates),
        )

    def test_track_similarity_seed_is_excluded_from_its_own_generation(self) -> None:
        seed_id = "trk_22222222-2222-4222-8222-222222222222"
        semantics = _ResolvedRecommendationSemantics(
            mode="similarity_seed",
            target="Synthetic Solo",
            target_kind="track",
            scene=None,
            target_ids=(seed_id,),
        )
        call = ProviderToolCall(
            "g1",
            "generate_inferred_recommendation",
            json.dumps(
                {
                    "target_ids": ["trk_44444444-4444-4444-8444-444444444444"],
                    "exclude_target_ids": [
                        "trk_11111111-1111-4111-8111-111111111111"
                    ],
                    "limit": 5,
                }
            ),
        )
        arguments = json.loads(
            _apply_recommendation_semantics(call, semantics).arguments
        )
        self.assertEqual(arguments["target_ids"], [seed_id])
        self.assertEqual(
            arguments["exclude_target_ids"],
            [
                "trk_11111111-1111-4111-8111-111111111111",
                seed_id,
            ],
        )

    def test_preference_scene_catalog_query_uses_seed_not_scene_literal(self) -> None:
        semantics = _ResolvedRecommendationSemantics(
            mode="preference_seed",
            target="IU",
            target_kind="artist",
            scene="evening",
            target_ids=("trk_11111111-1111-4111-8111-111111111111",),
        )
        call = ProviderToolCall(
            "d1",
            "discover_catalog_tracks",
            json.dumps({"term": "适合晚上听的歌", "limit": 5}),
        )
        arguments = json.loads(
            _apply_recommendation_semantics(call, semantics).arguments
        )
        self.assertEqual(arguments["term"], "IU")
        self.assertEqual(arguments["limit"], 5)

    def test_artist_constraint_catalog_query_stays_artist_only(self) -> None:
        semantics = _ResolvedRecommendationSemantics(
            mode="artist_constraint",
            target="IU",
            target_kind="artist",
            scene=None,
            target_ids=("trk_11111111-1111-4111-8111-111111111111",),
        )
        call = ProviderToolCall(
            "d1",
            "discover_catalog_tracks",
            json.dumps({"term": "night pop", "limit": 5}),
        )
        arguments = json.loads(
            _apply_recommendation_semantics(call, semantics).arguments
        )
        self.assertEqual(arguments["term"], "IU")

    def test_preference_scene_empty_pool_expands_catalog_then_retries(self) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        generation = json.dumps(
            {
                "target_ids": target_ids,
                "limit": 5,
                "avoid_previous_runs": True,
            }
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_recommendation", generation)]
                ),
                text_response("已找到新的晚间候选。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[ok_search_result_for_artist("IU")]
            + s5_prefetch_padding()
            + [
                empty_generation_result("generate_recommendation"),
                ok_discovery_result(),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    items=[
                        evidence_item(
                            "Palette",
                            artist_name="IU",
                            route="preview_only",
                            target_id="trk_33333333-3333-4333-8333-333333333333",
                        ),
                        evidence_item(
                            "Through the Night",
                            artist_name="IU",
                            route="preview_only",
                            target_id="trk_44444444-4444-4444-8444-444444444444",
                        ),
                    ],
                ),
            ],
        )
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )

        result = loop.run("我喜欢 IU，推荐几首适合晚上听的歌")

        self.assertEqual(
            client.recorded,
            ["search_library_tracks"]
            + S5_PREFETCH_RECORDED
            + [
                "generate_recommendation",
                "discover_catalog_tracks",
                "generate_inferred_recommendation",
            ],
        )
        self.assertEqual(result.recommendation_payload["item_count"], 2)
        self.assertEqual(len(result.recommendation_payload["items"]), 2)
        self.assertEqual(
            [execution.outcome for execution in result.tool_executions],
            ["execution_error", "ok", "ok"],
        )
        self.assertEqual(
            [execution.origin for execution in result.tool_executions],
            ["provider_requested", "policy_injected", "policy_injected"],
        )
        # The provider only requested the first direct generation.  The next
        # provider round nevertheless sees a valid synthetic tool-call/result
        # pair for the policy-owned discovery and inferred retry.
        self.assertEqual(len(provider.calls), 2)
        policy_call_message = provider.calls[1]["messages"][-2]
        policy_result_message = provider.calls[1]["messages"][-1]
        self.assertEqual(
            [call.name for call in policy_call_message.tool_calls],
            ["discover_catalog_tracks", "generate_inferred_recommendation"],
        )
        self.assertEqual(
            [result.call_id for result in policy_result_message.tool_results],
            [call.call_id for call in policy_call_message.tool_calls],
        )
        discovery = next(
            record
            for record in result.trace.tools
            if record.name == "discover_catalog_tracks"
        )
        self.assertEqual(discovery.arguments["term"], "IU")
        retry = [
            record
            for record in result.trace.tools
            if record.name == "generate_inferred_recommendation"
        ][0]
        self.assertEqual(discovery.origin, "policy_injected")
        self.assertEqual(retry.origin, "policy_injected")
        self.assertEqual(retry.arguments["target_ids"], target_ids)
        self.assertTrue(retry.arguments["avoid_previous_runs"])
        self.assertNotIn("genres", retry.arguments)
        self.assertEqual(
            sum(
                execution.name in _GENERATION_TOOL_NAMES
                for execution in result.tool_executions
            ),
            2,
        )
        system = provider.calls[0]["system"]
        self.assertIn("mode=preference_seed", system)
        self.assertIn("scene=evening", system)
        self.assertIn("允许在本轮预算内执行一次", system)

    def test_preference_fallback_keeps_freshness_and_fails_honestly_when_inferred_empty(
        self,
    ) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        generation = json.dumps(
            {
                "target_ids": target_ids,
                "limit": 5,
                "avoid_previous_runs": True,
            }
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_recommendation", generation)]
                ),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[ok_search_result_for_artist("IU")]
            + s5_prefetch_padding()
            + [
                empty_generation_result("generate_recommendation"),
                ok_discovery_result(),
                empty_generation_result("generate_inferred_recommendation"),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("我喜欢 IU，推荐几首歌")

        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertIsNone(result.recommendation_payload)
        self.assertEqual(
            client.recorded,
            ["search_library_tracks"]
            + S5_PREFETCH_RECORDED
            + [
                "generate_recommendation",
                "discover_catalog_tracks",
                "generate_inferred_recommendation",
            ],
        )
        retry_payload = client.recorded_payloads[-1]
        self.assertEqual(retry_payload["target_ids"], target_ids)
        self.assertEqual(retry_payload["limit"], 5)
        self.assertIs(retry_payload["avoid_previous_runs"], True)
        self.assertEqual(
            [
                execution.outcome
                for execution in result.tool_executions
                if execution.name in _GENERATION_TOOL_NAMES
            ],
            ["execution_error", "execution_error"],
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            [execution.origin for execution in result.tool_executions],
            ["provider_requested", "policy_injected", "policy_injected"],
        )

    def test_preference_fallback_retries_inferred_when_discovery_promotes_zero(
        self,
    ) -> None:
        target_ids = ["trk_11111111-1111-4111-8111-111111111111"]
        generation = json.dumps(
            {"target_ids": target_ids, "limit": 5, "genres": ["K-Pop"]}
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_recommendation", generation)]
                ),
                text_response("已找到候选。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[ok_search_result_for_artist("IU")]
            + s5_prefetch_padding()
            + [
                empty_generation_result("generate_recommendation"),
                ok_already_bound_discovery_result(),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [
                        evidence_item(
                            "Palette",
                            artist_name="IU",
                            route="preview_only",
                            target_id="trk_33333333-3333-4333-8333-333333333333",
                        )
                    ],
                ),
            ],
        )

        result = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS).run(
            "我喜欢 IU，推荐几首适合晚上听的歌"
        )

        self.assertEqual(
            client.recorded[-3:],
            [
                "generate_recommendation",
                "discover_catalog_tracks",
                "generate_inferred_recommendation",
            ],
        )
        self.assertEqual(client.recorded_payloads[-2], {"limit": 10, "term": "IU"})
        self.assertEqual(client.recorded_payloads[-1]["genres"], ["K-Pop"])
        self.assertIs(client.recorded_payloads[-1]["avoid_previous_runs"], True)
        self.assertEqual(result.recommendation_payload["item_count"], 1)

    def test_provider_repeats_after_policy_fallback_are_budget_blocked(self) -> None:
        generation = json.dumps(
            {
                "target_ids": ["trk_11111111-1111-4111-8111-111111111111"],
                "limit": 5,
            }
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_recommendation", generation)]
                ),
                tool_response(
                    [
                        ProviderToolCall("g2", "generate_recommendation", generation),
                        ProviderToolCall(
                            "d2", "discover_catalog_tracks", '{"term":"IU"}'
                        ),
                    ]
                ),
                text_response("请展示已有批次。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[ok_search_result_for_artist("IU")]
            + s5_prefetch_padding()
            + [
                empty_generation_result("generate_recommendation"),
                ok_discovery_result(),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [
                        evidence_item(
                            "Palette",
                            artist_name="IU",
                            route="preview_only",
                            target_id="trk_33333333-3333-4333-8333-333333333333",
                        )
                    ],
                ),
            ],
        )
        result = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        ).run("我喜欢 IU，推荐几首歌")

        self.assertEqual(client.recorded.count("generate_recommendation"), 1)
        self.assertEqual(client.recorded.count("generate_inferred_recommendation"), 1)
        self.assertEqual(client.recorded.count("discover_catalog_tracks"), 1)
        gated = [record for record in result.trace.tools if not record.executed]
        self.assertEqual(
            {record.error_code for record in gated},
            {"post_generation_closeout"},
        )
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertEqual(result.recommendation_payload["item_count"], 1)

    def test_policy_fallback_calls_use_the_normal_durable_journal(self) -> None:
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_22222222-2222-4222-8222-222222222222",
        ]
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": target_ids, "limit": 5}),
                        )
                    ]
                )
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = self._client(service)

        result = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS).run(
            "我喜欢 Artist Alpha，推荐几首歌"
        )

        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            tools = [record.request.tool for record in journal.list()]
        self.assertEqual(
            tools[-3:],
            [
                "generate_recommendation",
                "discover_catalog_tracks",
                "generate_inferred_recommendation",
            ],
        )
        self.assertEqual(
            [execution.origin for execution in result.tool_executions[-2:]],
            ["policy_injected", "policy_injected"],
        )

    def test_policy_fallback_never_claims_artist_or_similarity_semantics(self) -> None:
        generation = json.dumps(
            {
                "target_ids": ["trk_11111111-1111-4111-8111-111111111111"],
                "limit": 5,
            }
        )
        for phrase in (
            "推荐 Artist Alpha 的歌",
            "推荐类似 Artist Alpha 的音乐",
        ):
            with self.subTest(phrase=phrase):
                provider = FakeProvider(
                    [
                        tool_response(
                            [
                                ProviderToolCall(
                                    "g1", "generate_recommendation", generation
                                )
                            ]
                        ),
                        text_response("本轮没有合适的新候选。"),
                    ]
                )
                service = self._service(AgentClientPolicy.FULL)
                self.addCleanup(service.close)
                client = RecordingClient(
                    service,
                    results=[ok_search_result_for_artist("Artist Alpha")]
                    + s5_prefetch_padding()
                    + [empty_generation_result("generate_recommendation")],
                )

                result = ProviderAgentLoop(
                    provider, client, PROVIDER_TOOL_SCHEMAS
                ).run(phrase)

                self.assertEqual(client.recorded[-1], "generate_recommendation")
                self.assertNotIn("discover_catalog_tracks", client.recorded)
                self.assertNotIn("generate_inferred_recommendation", client.recorded)
                self.assertTrue(
                    all(
                        execution.origin == "provider_requested"
                        for execution in result.tool_executions
                    )
                )

    def test_artist_constraint_refreshes_scope_after_catalog_expansion(self) -> None:
        old_id = "trk_11111111-1111-4111-8111-111111111111"
        new_id = "trk_33333333-3333-4333-8333-333333333333"

        def search_result(ids: list[str]) -> AgentToolResult:
            result = ok_search_result_for_artist("IU")
            matches = [
                {
                    "target_id": target_id,
                    "name": "IU fixture",
                    "artist_name": "IU",
                }
                for target_id in ids
            ]
            return replace(result, payload={"matches": matches})

        generation = json.dumps(
            {"target_ids": [old_id], "limit": 5, "avoid_previous_runs": True}
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_inferred_recommendation", generation)]
                ),
                tool_response(
                    [ProviderToolCall("d1", "discover_catalog_tracks", '{"term":"night"}')]
                ),
                tool_response(
                    [ProviderToolCall("g2", "generate_inferred_recommendation", generation)]
                ),
                text_response("已找到 IU 的新候选。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[search_result([old_id])]
            + s5_prefetch_padding()
            + [
                empty_generation_result("generate_inferred_recommendation"),
                ok_discovery_result(),
                search_result([old_id, new_id]),
                ok_generation_evidence_batch(
                    "generate_inferred_recommendation",
                    [
                        evidence_item(
                            "IU fixture",
                            artist_name="IU",
                            route="preview_only",
                            target_id=new_id,
                        )
                    ],
                ),
            ],
        )
        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("推荐 IU 的歌")

        self.assertEqual(result.recommendation_payload["item_count"], 1)
        self.assertEqual(client.recorded.count("discover_catalog_tracks"), 1)
        self.assertEqual(client.recorded.count("search_library_tracks"), 2)
        generation_scopes = [
            scope
            for tool, scope in zip(client.recorded, client.scope_kwargs)
            if tool == "generate_inferred_recommendation"
        ]
        self.assertEqual(generation_scopes, [(old_id,), (old_id, new_id)])
        generation_payloads = [
            payload
            for tool, payload in zip(client.recorded, client.recorded_payloads)
            if tool == "generate_inferred_recommendation"
        ]
        self.assertEqual(
            [set(payload["target_ids"]) for payload in generation_payloads],
            [{old_id}, {old_id, new_id}],
        )

    def test_internal_artist_scope_filters_generation_without_ranking_change(self) -> None:
        from music_agent.preference_attribution import (
            PreferenceTargetKind,
            PreferenceTargetReference,
        )
        from music_agent.preference_persistence import SignalIdentity
        from music_agent.preference_persistence_repository import (
            PreferencePersistenceRepository,
        )
        from music_agent.source_observation import ObservedValue

        alpha = "trk_22222222-2222-4222-8222-222222222222"
        beta = "trk_44444444-4444-4444-8444-444444444444"
        with PreferencePersistenceRepository(self.database_path) as repository:
            for target_id in (alpha, beta):
                repository.record_observation(
                    SignalIdentity(
                        PreferenceTargetReference(
                            PreferenceTargetKind.TRACK, target_id
                        ),
                        "apple_music",
                        "favorited",
                    ),
                    ObservedValue.value(True),
                    observed_at="2026-08-16T00:00:00+00:00",
                    provenance="semantic_scope_fixture",
                )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        result = self._client(service).call(
            "generate_recommendation",
            {"target_ids": [alpha, beta], "limit": 5},
            recommendation_scope_ids=(alpha,),
        )
        self.assertEqual(result.outcome, AgentToolOutcome.OK)
        self.assertEqual(
            {item["target_id"] for item in result.payload["items"]}, {alpha}
        )

    def test_explicit_similarity_seed_is_pinned_in_live_provider_prompt(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS
        )
        loop.run("推荐类似 Artist Alpha 的音乐")
        system = provider.calls[0]["system"]
        self.assertIn("mode=similarity_seed", system)
        self.assertIn("target=Artist Alpha", system)
        self.assertIn("不得用 current playback 替换本句显式 target", system)
        self.assertEqual(
            {tool.name for tool in provider.calls[0]["tools"]},
            set(_RECOMMENDATION_TOOL_NAMES),
        )

    def test_happy_path_tool_call_flows_through_p09(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "get_canonical_entity",
                                 json.dumps({"canonical_id": "trk_11111111-1111-4111-8111-111111111111"})),
            ]),
            text_response("该轨道是《Synthetic Duet》。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("这首歌是什么？")
        self.assertEqual(result.final_text, "该轨道是《Synthetic Duet》。")
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.context_trimmed)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            [execution.name for execution in result.tool_executions], ["get_canonical_entity"]
        )
        self.assertEqual(result.tool_executions[0].outcome, "ok")
        # The P09 journal recorded exactly one request (the tool call).
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(len(journal.list()), 1)

    def test_refusal_surfaces_to_the_model(self) -> None:
        # A read_only client asking for a mutate tool: P09 refuses; the model sees it.
        service = self._service(AgentClientPolicy.READ_ONLY)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "record_feedback",
                                 json.dumps({"kind": "liked", "source_system": "apple_music",
                                             "source_path": "test",
                                             "target_id": "trk_11111111-1111-4111-8111-111111111111"})),
            ]),
            text_response("抱歉，我无权记录反馈（permission_denied）。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("记录我喜欢这首歌")
        self.assertIn("permission_denied", result.final_text)
        self.assertEqual(result.tool_executions[0].outcome, "permission_denied")
        self.assertEqual(result.tool_executions[0].error_code, "permission_denied")

    def test_model_supplied_produced_at_fails_closed_at_execution(self) -> None:
        """P15 burn-down Issue 1: durable run time is service-authoritative. A
        model that hallucinates ``produced_at`` into a generation payload is
        rejected by the envelope validator before any execution -- the refusal
        surfaces to the model verbatim, the journal records the refusal, and
        nothing reaches recommendation history."""
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "g1",
                    "generate_recommendation",
                    json.dumps(
                        {
                            "target_ids": [
                                "trk_11111111-1111-4111-8111-111111111111"
                            ],
                            "limit": 5,
                            "produced_at": "2026-08-16T01:10:00+08:00",
                        }
                    ),
                )
            ]),
            text_response("该请求因未知参数被拒绝。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("推荐一首歌")
        self.assertEqual(result.tool_executions[0].outcome, "invalid_request")
        self.assertEqual(result.tool_executions[0].error_code, "validation_error")
        # The model-visible tool result carries the unknown-key refusal verbatim.
        tool_result = provider.calls[1]["messages"][-1].tool_results[0]
        self.assertIn("payload keys must be one of", tool_result.content)
        # The refusal never reached a generation handler: zero durable runs.
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )
        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            rows = journal.list()
            self.assertEqual(len(rows), 4)  # three recommendation prefetch reads + refusal
            self.assertEqual(
                sum(row.request.tool == "generate_recommendation" for row in rows), 1
            )
        with RecommendationHistoryRepository(self.database_path) as history:
            self.assertEqual(len(history.list_runs()), 0)

    def test_live_write_refuses_not_execution_ready(self) -> None:
        from music_agent.identity import EntityType, ExternalIdentityKey
        from music_agent.intent_repository import PendingIntentRepository
        from music_agent.source_observation import ObservedValue
        from music_agent.write_intent import WriteOperation, create_scalar_pending_intent

        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED,
            "trk_11111111-1111-4111-8111-111111111111",
            ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001"),
            ObservedValue.value(True),
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)

        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "execute_write_intent",
                                 json.dumps({"intent_id": intent.intent_id})),
            ]),
            text_response("该写入能力尚未就绪（not_execution_ready）。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("帮我写回 Apple Music")
        self.assertEqual(result.tool_executions[0].outcome, "not_execution_ready")
        self.assertIn("not_execution_ready", result.final_text)

    def test_per_turn_dedupe_executes_once(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        arguments = json.dumps({"canonical_id": "trk_11111111-1111-4111-8111-111111111111"})
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "get_canonical_entity", arguments),
                ProviderToolCall("c2", "get_canonical_entity", arguments),
            ]),
            text_response("完成。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("查两次同一实体")
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(len(journal.list()), 1)  # deduped, executed once
        self.assertEqual(len(result.tool_executions), 1)

    def test_same_round_multi_call_executes_sequentially_in_one_round(self) -> None:
        # P15-S4-M3-A leaves the loop's execution semantics untouched: several
        # tool calls of one provider response still run sequentially inside that
        # one round, in call order -- no parallelism, no reordering.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                ok_read_result("list_recommendation_runs"),
                ok_read_result("query_track_preference"),
            ],
        )
        loop = ProviderAgentLoop(
            FakeProvider([
                tool_response([
                    ProviderToolCall("c1", "list_recommendation_runs", "{}"),
                    ProviderToolCall(
                        "c2", "query_track_preference",
                        json.dumps({"target_id": "trk_11111111-1111-4111-8111-111111111111"}),
                    ),
                ]),
                text_response("完成。"),
            ]),
            client,
            PROVIDER_TOOL_SCHEMAS,
        )
        result = loop.run("并行读两个")
        self.assertEqual(result.rounds, 2)  # one multi-call round + the answer
        self.assertEqual(
            client.recorded, ["list_recommendation_runs", "query_track_preference"]
        )
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["list_recommendation_runs", "query_track_preference"],
        )

    def test_in_run_read_cache_reuses_identical_read_across_turns(self) -> None:
        # Two separate turns ask the identical read question: the second result comes
        # from the run-level cache, so P09 executes the tool exactly once.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        arguments = json.dumps({"canonical_id": "trk_11111111-1111-4111-8111-111111111111"})
        provider = FakeProvider([
            tool_response([ProviderToolCall("c1", "get_canonical_entity", arguments)]),
            tool_response([ProviderToolCall("c2", "get_canonical_entity", arguments)]),
            text_response("完成。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("两轮都查同一实体")
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(len(journal.list()), 1)  # cached, executed once
        self.assertEqual(len(result.tool_executions), 1)
        self.assertEqual(result.rounds, 3)

    def test_mutation_clears_the_in_run_read_cache(self) -> None:
        # read -> mutation -> identical read: the post-mutation read must be a fresh
        # execution, never a stale cached result.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        read_args = json.dumps({"canonical_id": "trk_11111111-1111-4111-8111-111111111111"})
        feedback_args = json.dumps({
            "kind": "liked",
            "source_system": "recommendation_ui",
            "source_path": "card_actions",
            "target_id": "trk_11111111-1111-4111-8111-111111111111",
        })
        provider = FakeProvider([
            tool_response([ProviderToolCall("c1", "get_canonical_entity", read_args)]),
            tool_response([ProviderToolCall("c2", "record_feedback", feedback_args)]),
            tool_response([ProviderToolCall("c3", "get_canonical_entity", read_args)]),
            text_response("完成。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("查询、写入、再查询")
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(len(journal.list()), 3)  # read, write, fresh re-read
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["get_canonical_entity", "record_feedback", "get_canonical_entity"],
        )
        self.assertEqual(
            [execution.outcome for execution in result.tool_executions],
            ["ok", "ok", "ok"],
        )

    def test_same_turn_mutation_dedupe_survives_cache_invalidation(self) -> None:
        # Two identical mutation calls inside ONE turn: the per-turn barrier still
        # prevents a double apply (the run cache never held this key at all).
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        feedback_args = json.dumps({
            "kind": "liked",
            "source_system": "recommendation_ui",
            "source_path": "card_actions",
            "target_id": "trk_11111111-1111-4111-8111-111111111111",
        })
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "record_feedback", feedback_args),
                ProviderToolCall("c2", "record_feedback", feedback_args),
            ]),
            text_response("完成。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("重复记录同一条反馈")
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(len(journal.list()), 1)  # applied exactly once
        self.assertEqual(len(result.tool_executions), 1)

    def test_invalid_json_arguments_feed_back_an_error(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([ProviderToolCall("c1", "get_canonical_entity", "not-json")]),
            text_response("参数无效。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("查实体")
        self.assertEqual(result.tool_executions[0].outcome, "invalid_arguments")
        self.assertIn("参数无效", result.final_text)

    def test_max_rounds_caps_the_loop(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        responses = [
            tool_response([ProviderToolCall(f"c{i}", "get_agent_capabilities", "{}")])
            for i in range(1, 4)
        ]
        provider = FakeProvider(responses)
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_tool_rounds=3),
        )
        result = loop.run("无限调用")
        self.assertTrue(result.rounds_capped)
        self.assertEqual(result.rounds, 3)
        # P17-A1: the cap falls back to the deterministic closeout, never scavenged text.
        self.assertEqual(result.final_text, _ROUND_CAP_CLOSEOUT)

    def test_capped_loop_never_selects_intermediate_planning_text(self) -> None:
        """P17-A1: tool-round preamble text must not become the user-facing answer.

        Live symptom was rounds-capped runs replying with 「让我如实向用户报告……」
        — the model's planning narration, scavenged from history. The fix must be
        structural: the final answer comes only from a terminal (no tool_calls)
        provider message, or the fixed closeout.
        """
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        preamble = "让我先检查一下用户的偏好画像，再决定推荐哪些歌。"
        responses = [
            planning_tool_response(
                [ProviderToolCall("c1", "get_agent_capabilities", "{}")], preamble
            ),
            planning_tool_response(
                [ProviderToolCall("c2", "get_tool_capabilities", "{}")],
                "我应该先确认候选池里有什么。",
            ),
        ]
        provider = FakeProvider(responses)
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_tool_rounds=2),
        )
        result = loop.run("无限调用")
        self.assertTrue(result.rounds_capped)
        self.assertEqual(result.final_text, _ROUND_CAP_CLOSEOUT)
        self.assertNotIn("让我先检查", result.final_text)
        self.assertNotIn("画像", result.final_text)
        self.assertNotIn("我应该", result.final_text)

    def test_empty_terminal_answer_falls_back_to_fixed_closeout(self) -> None:
        """P17-A1: an empty terminal message gets an honest fixed fallback."""
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("   ")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("测试")
        self.assertEqual(result.final_text, _EMPTY_FINAL_ANSWER_CLOSEOUT)

    def test_preview_batch_prompt_rule_is_state_driven(self) -> None:
        """P17-A2: the prompt pins the batch answer to session.state, not bare started."""
        self.assertIn("回答必须以工具返回里的 session.state 为准", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("started 只是 state=running 的派生标志", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("failed → 连播已中断并如实说明原因，绝不宣称已启动", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("started=false 时绝不宣称已启动", DEFAULT_SYSTEM_PROMPT)

    def test_context_trim_is_reported(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        responses = [
            tool_response([ProviderToolCall("c1", "get_agent_capabilities", "{}")]),
            tool_response([ProviderToolCall("c2", "get_agent_capabilities", "{}")]),
            text_response("好。"),
        ]
        provider = FakeProvider(responses)
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_tool_rounds=4, max_context_messages=4),
        )
        result = loop.run("裁剪测试")
        # History grows past the bound (user + 2×(assistant + results) = 5 > 4).
        self.assertTrue(result.context_trimmed)
        # The provider only ever saw the bounded history.
        for call in provider.calls:
            self.assertLessEqual(len(call["messages"]), 4)

    def test_provider_errors_propagate_typed(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([ProviderAuthError("bad key")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        with self.assertRaises(ProviderAuthError):
            loop.run("hello")

    def test_unknown_tool_name_refuses_at_p09(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([ProviderToolCall("c1", "not_a_real_tool", "{}")]),
            text_response("工具不存在（tool_not_supported）。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("用不存在的工具")
        self.assertEqual(result.tool_executions[0].outcome, "tool_not_supported")

    def test_bounded_payload_truncates_large_results(self) -> None:
        from music_agent.provider_agent import _bounded_payload

        payload = {"key": "x" * 5000}
        bounded = _bounded_payload(payload)
        self.assertTrue(bounded["truncated"])
        self.assertLessEqual(len(bounded["preview"]), 2000)

    def test_schemas_cover_all_p09_tools(self) -> None:
        from music_agent.agent_tools import AGENT_TOOL_REGISTRY, AgentToolName

        # P15-S1 C02: advance_preview is a CLI-routing-only tool -- the fast path
        # fires it off a live "running" session read and the provider loop never
        # sees it (session management is service-local, not a provider decision).
        self.assertEqual(
            {schema.name for schema in PROVIDER_TOOL_SCHEMAS},
            set(AGENT_TOOL_REGISTRY.tool_names)
            - {AgentToolName.ADVANCE_PREVIEW.value},
        )


class ChatCliTest(unittest.TestCase):
    def test_chat_parser_accepts_flags(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args([
            "chat", "--db", "store.db", "--message", "你好",
            "--agent-client", f"{CLIENT_ID}:full",
            "--api-key-env", "MY_KEY", "--model", "deepseek-chat",
            "--timeout", "30", "--max-rounds", "4",
        ])
        self.assertEqual(args.message, "你好")
        self.assertEqual(args.api_key_env, "MY_KEY")
        self.assertEqual(args.model, "deepseek-chat")
        self.assertEqual(args.max_rounds, 4)
        # Per-provider defaults resolve in _build_chat_provider, not in the parser.
        self.assertEqual(args.provider, "deepseek")
        self.assertIsNone(args.base_url)

    def test_chat_requires_exactly_one_client(self) -> None:
        import io
        from contextlib import redirect_stderr

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main([
                    "chat", "--db", str(Path(tmp) / "s.db"), "--message", "hi",
                ])
            self.assertEqual(exit_code, 2)
            self.assertIn("exactly one --agent-client", stderr.getvalue())


class DefaultSystemPromptRecommendationExperienceContractTest(unittest.TestCase):
    """P13 experience phase: answer-format discipline, 换一组 semantics, direction mapping."""

    def test_answer_format_forbids_internal_identifiers_and_process(self) -> None:
        self.assertIn("只列「歌名 — 艺人」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要展示 run_id", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要展示", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("过程说明", DEFAULT_SYSTEM_PROMPT)

    def test_new_batch_requires_exclusion_of_previous_batch(self) -> None:
        self.assertIn("新推荐必须与上一批不同", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("exclude_target_ids", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("avoid_previous_runs", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("禁止把上一批原样复述成新推荐", DEFAULT_SYSTEM_PROMPT)

    def test_direction_maps_to_genres_or_search_terms(self) -> None:
        self.assertIn("可选 genres 参数", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("日系→J-Pop", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("翻译成目录搜索词", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要声称系统做了无法支持的属性过滤", DEFAULT_SYSTEM_PROMPT)


class DefaultSystemPromptPlaybackContractTest(unittest.TestCase):
    """The default system prompt must carry the named-track playback routing invariant.

    These tests pin contract text only -- they do not assert anything about how any
    provider model behaves with the prompt.
    """

    def test_prompt_requires_play_track_once_canonical_id_exists(self) -> None:
        self.assertIn("必须使用 play_track(canonical_id) 选曲播放", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_forbids_generic_play_as_selection_substitute(self) -> None:
        self.assertIn("绝不能代替选曲", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_covers_target_already_paused_case(self) -> None:
        self.assertIn("即使 get_now_playing 已显示该歌曲处于暂停状态也不例外", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_forbids_success_claim_from_generic_play_ok_alone(self) -> None:
        self.assertIn("不得仅凭通用 play 返回 ok 就声称指定歌曲正在播放", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_requires_post_action_verification(self) -> None:
        self.assertIn("播放动作之后必须再用 get_now_playing 核对当前曲目", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_requires_honest_report_on_verification_mismatch(self) -> None:
        self.assertIn("不一致时如实报告实际结果", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_prefers_library_entries_among_same_name_hits(self) -> None:
        # P14-R3.2: same-name/name-variant multi-hits must not fall back to catalog previews.
        self.assertIn("同名/名称变体多命中时", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("优先艺人命中的条目", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得仅因目录候选带试听链接就放弃正式播放", DEFAULT_SYSTEM_PROMPT)


class DefaultSystemPromptPlaybackExecutionContractTest(unittest.TestCase):
    """Delegation authorization, generation closeout, and playback-route contracts."""

    def test_explicit_delegation_keeps_one_payload_bound_action_after_generation(self) -> None:
        self.assertIn("未经用户委托不得擅自替用户做决定", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("你来决定", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("首次成功批次锁定为本轮唯一推荐结果", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("禁止继续生成或发现", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("完成一次已授权的播放或试听", DEFAULT_SYSTEM_PROMPT)

    def test_playback_route_maps_to_execution_tool(self) -> None:
        self.assertIn("playback.route", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("library 表示可正式播放", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("preview_only 表示只能试听", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("unavailable 表示当前不可播放", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("preview_catalog_track 试听", DEFAULT_SYSTEM_PROMPT)

    def test_any_song_intent_never_uses_resume_or_queue_transport(self) -> None:
        self.assertIn("随便播放一首", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("禁止用通用 play 恢复当前曲目充数", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("禁止调用 next_track/previous_track 冒充选歌", DEFAULT_SYSTEM_PROMPT)

    def test_explicit_formal_constraint_never_degrades_to_preview(self) -> None:
        # P16-S3: the explicit 正式 carve-out -- a library route is the only
        # allowed pick, and a batch without one asks instead of auto-previewing.
        # This keeps the provider path (other phrasings + the fast path's
        # remit fallback) aligned with the deterministic CLI chain.
        self.assertIn("用户明确限定「正式」播放", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("只能选择 ", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("playback.route=library 的条目", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得自动降级成试听", DEFAULT_SYSTEM_PROMPT)

    def test_open_in_apple_music_uses_only_the_resolved_real_url(self) -> None:
        # P16-S4: opening locates the batch item by its target_id, hands the
        # canonical id to the tool, and treats the result url as the only real
        # link -- never composing one, never substituting on absence.
        self.assertIn("用户要「在 Apple Music 中打开」某曲", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("open_in_apple_music(canonical_id=该 target_id)", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("只有工具结果里的 url 才是真实官方链接", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("绝不自行构造或复述任何 URL", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("本地资料库曲目无目录链接", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得用搜索、试听或其他曲目替代", DEFAULT_SYSTEM_PROMPT)


class DefaultSystemPromptSwapContextContractTest(unittest.TestCase):
    """Third phase B: 「换一首」routes by get_active_context + preview_sounding truth;
    next_track is allowed only for own-queue playback, never as a selection shortcut."""

    def test_swap_requires_context_read_first(self) -> None:
        self.assertIn("用户说「换一首」先读 get_active_context", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("先看 channel（library=Agent 刚正式", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("再看 preview_sounding", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("agent_selected=当前曲目属于最近", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("own_queue=属于用户自己的播放队列", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("unknown=无法判断", DEFAULT_SYSTEM_PROMPT)

    def test_own_queue_is_the_only_next_track_carve_out(self) -> None:
        self.assertIn(
            "channel 为 none 且 context 为 own_queue 时：「换一首」翻译为 next_track",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn("这是唯一允许调用 next_track/previous_track 的情况", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("此时不得代理成选歌", DEFAULT_SYSTEM_PROMPT)

    def test_preview_and_library_channels_route_the_swap(self) -> None:
        self.assertIn("channel 为 library 且 context 为 agent_selected", DEFAULT_SYSTEM_PROMPT)
        self.assertIn(
            "preview_sounding 为 true（试听真实在响）",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn("用 preview_catalog_track 直接试听下一首", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不看 context", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("用 play_track 播放同批另一首", DEFAULT_SYSTEM_PROMPT)
        # C06.5: an ended preview (channel=preview, no sound) must not hit the direct
        # preview branch; it falls through to the channel/context matrix below.
        self.assertIn(
            "preview_sounding 为 false 且 channel 为 preview 时", DEFAULT_SYSTEM_PROMPT
        )
        self.assertIn("channel=preview 视同 none", DEFAULT_SYSTEM_PROMPT)

    def test_recommendation_state_swap_reuses_the_batch(self) -> None:
        self.assertIn("从当前推荐批换另一首展示或播放", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要重新生成推荐", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("或 context 为 unknown 时", DEFAULT_SYSTEM_PROMPT)

    def test_get_now_playing_schema_documents_context_fields(self) -> None:
        schema = next(
            item for item in PROVIDER_TOOL_SCHEMAS if item.name == "get_now_playing"
        )
        self.assertIn("context（agent_selected/own_queue/unknown", schema.description)
        self.assertIn("agent_channel（library/preview/none", schema.description)
        self.assertIn("player_canonical_id", schema.description)
        self.assertIn("canonical_resolution", schema.description)
        self.assertIn("「换一首」时先读这两项", schema.description)


class DefaultSystemPromptBatchContextContractTest(unittest.TestCase):
    """P14-C07.4: the model locates the current batch through get_active_context's
    active_batch (register/derived) and resolves ordinals / 「刚才推荐的」/ swap
    candidates against it via get_recommendation_run -- prompt alignment only."""

    def test_active_batch_is_the_unified_batch_locator(self) -> None:
        self.assertIn("当前推荐批的统一定位方式", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("source=register 表示本服务刚交付的批次", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("source=derived 表示无指针时按历史最新兜底", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("两者同样用于定位「当前批」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("仍需用 get_recommendation_run(run_id) 读取", DEFAULT_SYSTEM_PROMPT)

    def test_ordinal_requests_map_onto_batch_position(self) -> None:
        self.assertIn("「第一首/第二首/N首」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("条目顺序即推荐顺序", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("「第一首」是列表第 1 项", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("从 1 开始数，不要错位", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("按 playback.route", DEFAULT_SYSTEM_PROMPT)

    def test_recent_recommendation_prefers_active_batch_over_history(self) -> None:
        self.assertIn("「刚才推荐的/上次推荐的」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn(
            "优先用 get_active_context 的 active_batch 的 run_id", DEFAULT_SYSTEM_PROMPT
        )
        self.assertIn("active_batch 为 null 时才回退用 list_recommendation_runs", DEFAULT_SYSTEM_PROMPT)

    def test_swap_ties_into_active_batch_with_honest_fallback(self) -> None:
        self.assertIn("「换一首」与当前批的衔接", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("同样以 active_batch 定位", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("切换候选时取刚播放/试听那首之外的条目", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要虚构批次", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要为换一首擅自生成新推荐", DEFAULT_SYSTEM_PROMPT)

    def test_run_tool_descriptions_carry_the_batch_reading_rules(self) -> None:
        by_name = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}
        run_desc = by_name["get_recommendation_run"].description
        self.assertIn("第 1 个条目即「第一首」", run_desc)
        self.assertIn("用户按「第 N 首」点播时取第 N 个条目", run_desc)
        list_desc = by_name["list_recommendation_runs"].description
        self.assertIn("当前批次优先读 get_active_context 的 active_batch", list_desc)
        self.assertIn("active_batch 为 null 的老批次/兜底场景才用本工具", list_desc)


class DefaultSystemPromptReadBatchingContractTest(unittest.TestCase):
    """P15-S4-M3-A: the provider-facing READ batching discipline -- one narrow
    rule added to DEFAULT_SYSTEM_PROMPT only. These tests pin contract text;
    they assert nothing about how any provider model behaves."""

    def test_system_prompt_pins_same_round_independent_reads(self) -> None:
        self.assertIn("同一决策步骤需要多个互不依赖的只读查询时", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("优先在同一轮一次发出多个工具调用", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("定位当前批次的同时并行读取偏好与推荐历史", DEFAULT_SYSTEM_PROMPT)

    def test_system_prompt_pins_data_dependency_split_rounds(self) -> None:
        self.assertIn("参数必须来自上一轮结果", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("（如 run_id、observation_id、canonical_id）", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("必须等结果返回后分轮发出", DEFAULT_SYSTEM_PROMPT)

    def test_system_prompt_pins_mutation_readback_never_same_round(self) -> None:
        self.assertIn("任何会改变状态的工具（播放/试听/生成/记录/写入/加库）", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得与其核对读", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("在同一次输出中一并发出", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("核对读必须在变更工具结果返回之后的下一轮进行", DEFAULT_SYSTEM_PROMPT)

    def test_tool_descriptions_carry_zero_batching_change(self) -> None:
        # M3-A is provider-instruction-only: the rule must live in the system
        # prompt, never in any tool description/schema text.
        for item in PROVIDER_TOOL_SCHEMAS:
            with self.subTest(tool=item.name):
                self.assertNotIn("互不依赖的只读查询", item.description)
                self.assertNotIn("核对读必须在变更工具结果返回之后的下一轮", item.description)

    def test_m2_recommendation_discipline_is_unregressed(self) -> None:
        # M2 closeout + M2-1/M2-2 pins must survive the appended rule verbatim.
        self.assertIn("推荐生成工具一旦成功返回非空批次，本次请求即已交付完成", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("之后的目录发现请求会被拒绝、不会真正执行", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("生成失败（空结果重试后仍为空", DEFAULT_SYSTEM_PROMPT)
        # The batching paragraph sits inside the same single system prompt the
        # two providers both receive -- after the M2 rules, before the closing.
        self.assertLess(
            DEFAULT_SYSTEM_PROMPT.index("推荐生成工具一旦成功返回非空批次"),
            DEFAULT_SYSTEM_PROMPT.index("同一决策步骤需要多个互不依赖的只读查询时"),
        )
        self.assertLess(
            DEFAULT_SYSTEM_PROMPT.index("同一决策步骤需要多个互不依赖的只读查询时"),
            DEFAULT_SYSTEM_PROMPT.index("最终用中文简洁回答用户。"),
        )


class GenerationAttemptBudgetTest(ProviderAgentLoopTest):
    """P14-R4.3 + P14-R4.5: the shared generation-tool attempt budget. R4.5
    semantics: when every allowed attempt fails (or the budget is overrun with
    no success this run) the loop terminates deterministically with the fixed
    closeout -- no envelope round, no model decision, no further tool rounds.
    After the first success, the post-generation closeout supersedes the unused
    retry budget and blocks every later generation/discovery call."""

    def _budget_loop(
        self,
        provider: FakeProvider,
        results: list[AgentToolResult] | None = None,
    ) -> tuple[RecordingClient, ProviderAgentLoop]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        return client, loop

    def _generate_call(self, call_id: str) -> ProviderToolCall:
        return ProviderToolCall(call_id, "generate_recommendation", GENERATE_PAYLOAD)

    def test_final_failed_attempt_terminates_with_fixed_closeout(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            tool_response([self._generate_call("c2")]),
            tool_response([self._generate_call("c3")]),
            text_response("这一句永远不该出现。"),
        ])
        client, loop = self._budget_loop(provider)
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertIn("暂时没有找到新的推荐曲目", result.final_text)
        self.assertIn("2. 换一个方向，我重新帮你找", result.final_text)
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED
            + ["generate_recommendation", "generate_recommendation"],
        )
        self.assertEqual(len(result.tool_executions), 2)
        # The loop ended itself: no third round was ever sent to the provider.
        self.assertEqual(len(provider.calls), 2)

    def test_budget_overrun_without_success_terminates_mid_round(self) -> None:
        provider = FakeProvider([
            tool_response([
                self._generate_call("c1"),
                self._generate_call("c2"),  # identical args: same-round dedup
                self._generate_call("c3"),  # over budget, no success this run
            ]),
            tool_response([ProviderToolCall("c4", "get_active_context", "{}")]),
            text_response("这一句永远不该出现。"),
        ])
        client, loop = self._budget_loop(provider)
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertEqual(result.rounds, 1)
        self.assertFalse(result.rounds_capped)
        # c1 executed, its identical twin c2 was deduped against it (still an
        # attempt -- the gate counts attempts, not executions), and c3's overrun
        # terminated the run before P09 or any further provider round.
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.responses), 2)  # queued script untouched

    def test_successful_generation_does_not_terminate(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response("已为你生成新的一批：A — 甲。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=s5_prefetch_padding()
            + [ok_generation_result("generate_recommendation")],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text, "已为你生成新的一批：A — 甲。")
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )

    def test_evidence_less_plain_call_succeeds_without_an_inferred_retry_round(self) -> None:
        """P16-S2: a plain generate call whose target has no direct evidence
        succeeds in ONE executed tool call -- the deterministic inferred-channel
        fallback inside the same call, against the REAL service. The old chain
        (empty_recommendation error round, then a model round re-calling
        generate_inferred_recommendation) leaves no trace in the transcript:
        two provider rounds total (tool round + answer round), one generation,
        one ok outcome."""
        extra = {
            "id": "trk_ffffffff-ffff-4fff-8fff-00000000ffff",
            "external_ids": {"apple_music_persistent_id": None, "itunes_store_id": "EXTR-F"},
            "name": "Catalog Pop F",
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
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
        model["tracks"].append(extra)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(model)
        self._seed_two_positives()  # affinities exist; the target itself has none

        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "c1",
                    "generate_recommendation",
                    json.dumps({"target_ids": [extra["id"]], "limit": 5}),
                ),
            ]),
            text_response("已为你生成新的一批。"),
        ])
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("推荐几首歌")
        # P20-Fix10: the success scene is the deterministic rendering now --
        # one item of track-self inferred preference, batch preview-only (this
        # catalog track has no library binding), no provider text.
        self.assertEqual(
            result.final_text,
            "为你推荐这 1 首：\n\n"
            "1. Catalog Pop F — Artist Alpha\n"
            "这首按对这首曲目本身的偏好推断递选。\n\n"
            "这批曲目都只能试听 30 秒。需要试听哪一首，直接告诉我。",
        )
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["generate_recommendation"],
        )
        self.assertEqual(result.tool_executions[0].outcome, "ok")
        self.assertIsNone(result.tool_executions[0].error_code)
        # The provider never saw an error round to react to: exactly the tool
        # round and the answer round were sent.
        self.assertEqual(len(provider.calls), 2)

    def test_generation_after_success_is_refused_without_second_execution(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # ok
            tool_response([self._generate_call("c2")]),  # phase already closed
            text_response("这是第二批：B — 乙。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=s5_prefetch_padding()
            + [ok_generation_result("generate_recommendation")],
        )
        result = loop.run("换一组")
        self.assertEqual(result.final_text, "这是第二批：B — 乙。")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED
            + ["generate_recommendation"],
        )
        self.assertEqual(provider.calls[1]["tools"], [])
        user_message = provider.calls[2]["messages"][-1]
        injected = json.loads(user_message.tool_results[0].content)
        self.assertEqual(injected["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)

    def test_generation_budget_config_must_be_positive(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderLoopConfig(max_generation_attempts=0)

    def test_first_success_is_the_only_durable_run_and_payload_authority(self) -> None:
        self._seed_two_positives()
        arguments = json.dumps(
            {
                "target_ids": ["trk_11111111-1111-4111-8111-111111111111"],
                "limit": 1,
            }
        )
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("g1", "generate_recommendation", arguments)]
                ),
                # A non-conforming provider may emit a call despite receiving
                # zero schemas. The loop must still refuse it before P09.
                tool_response(
                    [ProviderToolCall("g2", "generate_recommendation", arguments)]
                ),
                text_response("使用第一批。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS
        )

        result = loop.run("推荐几首歌")

        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )
        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )

        with RecommendationHistoryRepository(self.database_path) as history:
            runs = history.list_runs(limit=10)
        self.assertEqual(len(runs), 1)
        self.assertEqual(result.recommendation_payload["run_id"], runs[0].run_id)
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertEqual(provider.calls[2]["tools"], [])
        with AgentRequestJournalRepository(self.database_path) as journal:
            generation_rows = [
                row
                for row in journal.list()
                if row.request.tool == "generate_recommendation"
            ]
        self.assertEqual(len(generation_rows), 1)
        self.assertEqual(
            json.loads(provider.calls[2]["messages"][-1].tool_results[0].content)[
                "error_code"
            ],
            _POST_GENERATION_CLOSEOUT_ERROR_CODE,
        )

    def test_delegated_generation_keeps_one_run_then_plays_with_readback(self) -> None:
        self._seed_two_positives()
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "g1",
                            "generate_recommendation",
                            json.dumps({"target_ids": [target_id], "limit": 1}),
                        )
                    ]
                ),
                tool_response(
                    [
                        ProviderToolCall(
                            "p1",
                            "play_track",
                            json.dumps({"canonical_id": target_id}),
                        )
                    ]
                ),
                tool_response(
                    [ProviderToolCall("n1", "get_now_playing", "{}")]
                ),
                text_response("已为你选中并播放〈Synthetic Duet〉。"),
            ]
        )
        adapter = DelegatedPlaybackAdapter()
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
        )
        self.addCleanup(service.close)

        result = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS
        ).run("随便播放一首")

        from music_agent.recommendation_history_repository import (
            RecommendationHistoryRepository,
        )

        with RecommendationHistoryRepository(self.database_path) as history:
            runs = history.list_runs(limit=10)
        self.assertEqual(len(runs), 1)
        self.assertEqual(result.recommendation_payload["run_id"], runs[0].run_id)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["generate_recommendation", "play_track", "get_now_playing"],
        )
        self.assertEqual(adapter.calls, [("play_track", ("SYNTH-TRACK-001",))])
        # A successful delegated generation enters the shared code-owned
        # SelectionGrant workflow immediately; the Provider is not asked to
        # choose the generated target or route in another round.
        self.assertEqual(len(provider.calls), 1)


class PostGenerationSoftCloseoutTest(ProviderAgentLoopTest):
    """After the first successful non-empty generation, further generation or
    catalog discovery is answered with ``post_generation_closeout`` instead of
    executing. Ordinary recommendation is final-only; explicit delegation gets
    only one payload-bound action plus a formal-play readback. The next real user
    request starts fresh."""

    def _closeout_loop(
        self,
        provider: FakeProvider,
        results: list[AgentToolResult] | None = None,
        *,
        instrument: bool = False,
    ) -> tuple[RecordingClient, ProviderAgentLoop]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=instrument),
        )
        return client, loop

    def _generate_call(self, call_id: str) -> ProviderToolCall:
        return ProviderToolCall(call_id, "generate_recommendation", GENERATE_PAYLOAD)

    def _inferred_call(self, call_id: str) -> ProviderToolCall:
        return ProviderToolCall(
            call_id,
            "generate_inferred_recommendation",
            json.dumps({"target_ids": ["trk_11111111-1111-4111-8111-111111111111"]}),
        )

    def _discover_call(self, call_id: str) -> ProviderToolCall:
        return ProviderToolCall(call_id, "discover_catalog_tracks", json.dumps({"term": "夜晚"}))

    def _delivered_envelope(self, provider: FakeProvider, round_index: int,
                            result_index: int = 0) -> dict:
        """The tool result(s) the loop fed back for a tool round: a tool call in
        round N is delivered to the model in the chat payload of round N+1."""
        user_message = provider.calls[round_index]["messages"][-1]
        return json.loads(user_message.tool_results[result_index].content)

    def test_ordinary_recommendation_blocks_discovery_and_other_tools_after_success(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # ok, non-empty (attempt 1)
            tool_response([
                self._discover_call("d1"),  # gated: success exists this run
                ProviderToolCall("c2", "get_active_context", "{}"),  # final-only: gated
            ]),
            text_response("已为你整理好这一批：夜曲 — 测试艺人。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        # The loop kept running and the model still produced a normal answer.
        self.assertEqual(result.final_text, "已为你整理好这一批：夜曲 — 测试艺人。")
        self.assertEqual(result.rounds, 3)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["generate_recommendation"])
        self.assertEqual(provider.calls[1]["tools"], [])
        # The model saw the deterministic synthetic refusal for the discovery.
        gated = self._delivered_envelope(provider, 2)
        self.assertEqual(gated["outcome"], "execution_error")
        self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)
        self.assertIsNone(gated["payload"])
        self.assertFalse(gated["replayed"])
        self.assertEqual(gated["error_message"], _POST_GENERATION_CLOSEOUT_MESSAGE)
        # A malformed call emitted despite zero schemas is also final-only.
        read_result = self._delivered_envelope(provider, 2, result_index=1)
        self.assertEqual(
            read_result["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE
        )

    def test_delegated_generation_enters_code_owned_selection_immediately(self) -> None:
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        provider = FakeProvider(
            [
                tool_response([self._generate_call("g1")]),
                tool_response(
                    [self._generate_call("g2"), self._discover_call("d1")]
                ),
                tool_response(
                    [
                        ProviderToolCall(
                            "p1",
                            "play_track",
                            json.dumps({"canonical_id": target_id}),
                        )
                    ]
                ),
                tool_response(
                    [ProviderToolCall("n1", "get_now_playing", "{}")]
                ),
                text_response("已完成播放。"),
            ]
        )
        client, loop = self._closeout_loop(
            provider,
            results=[
                ok_generation_batch("generate_recommendation"),
                ok_read_result("play_track"),
                ok_delegated_play_readback(target_id),
            ],
        )

        result = loop.run("你来决定")

        self.assertEqual(
            result.final_text,
            "正在播放《Synthetic》— Synthetic Artist。",
        )
        self.assertEqual(
            client.recorded,
            ["generate_recommendation", "play_track", "get_now_playing"],
        )
        # The Provider's queued second-round generation/discovery proposal is
        # never consumed, so it cannot replace the generated run's target or
        # route and no synthetic closeout exchange is needed.
        self.assertEqual(len(provider.calls), 1)

    def test_inferred_generation_success_gates_discover(self) -> None:
        provider = FakeProvider([
            tool_response([self._inferred_call("c1")]),  # ok, non-empty (attempt 1)
            tool_response([self._discover_call("d1")]),  # gated
            text_response("已为你整理好这一批。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_inferred_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        self.assertEqual(result.final_text, "已为你整理好这一批。")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["generate_inferred_recommendation"])
        self.assertEqual(provider.calls[1]["tools"], [])
        gated = self._delivered_envelope(provider, 2)
        self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)

    def test_generation_failure_keeps_discovery_allowed(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # failed (attempt 1 of 2)
            tool_response([self._discover_call("d1")]),  # allowed: no success this run
            text_response("先看看目录里有什么。"),
        ])
        client, loop = self._closeout_loop(provider, results=s5_prefetch_padding() + [
            failed_generation_result("generate_recommendation"),
            ok_discovery_result(),
        ])
        result = loop.run("推荐几首歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED + ["generate_recommendation", "discover_catalog_tracks"],
        )
        discovery = self._delivered_envelope(provider, 2)
        self.assertEqual(discovery["outcome"], "ok")
        self.assertEqual(discovery["payload"]["discovered_count"], 2)

    def test_empty_generation_keeps_discovery_allowed(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # empty (attempt 1 of 2)
            tool_response([self._discover_call("d1")]),  # allowed: empty is not success
            text_response("先看看目录里有什么。"),
        ])
        client, loop = self._closeout_loop(provider, results=s5_prefetch_padding() + [
            empty_generation_result("generate_recommendation"),
            ok_discovery_result(),
        ])
        result = loop.run("推荐几首歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED + ["generate_recommendation", "discover_catalog_tracks"],
        )
        discovery = self._delivered_envelope(provider, 2)
        self.assertEqual(discovery["outcome"], "ok")

    def test_discovery_before_generation_executes_normally(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # pre-generation: executes
            tool_response([self._generate_call("c1")]),  # ok, non-empty
            text_response("夜曲 — 测试艺人，适合晚上听。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_discovery_result(),
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        self.assertEqual(result.final_text, "夜曲 — 测试艺人，适合晚上听。")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "generate_recommendation"]
        )

    def test_same_round_discover_after_success_is_gated(self) -> None:
        provider = FakeProvider([
            tool_response([
                self._generate_call("c1"),  # ok sets the state...
                self._discover_call("d1"),  # ...so this later call in the round is gated
            ]),
            text_response("已为你整理好这一批。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["generate_recommendation"])
        gated = self._delivered_envelope(provider, 1, result_index=1)
        self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)

    def test_same_round_discover_before_generation_executes(self) -> None:
        provider = FakeProvider([
            tool_response([
                self._discover_call("d1"),  # before the success: executes
                self._generate_call("c1"),  # ok, non-empty
            ]),
            text_response("已为你整理好这一批。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_discovery_result(),
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "generate_recommendation"]
        )

    def test_new_run_resets_the_closeout(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # run A: ok
            tool_response([self._discover_call("d1")]),  # run A: gated
            text_response("这是第一批。"),
            tool_response([self._discover_call("d2")]),  # run B: fresh state, executes
            text_response("新的搜索也有结果。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_recommendation"),
            ok_discovery_result(),
        ])
        first = loop.run("帮我选几首适合晚上听的")
        self.assertEqual(first.final_text, "这是第一批。")
        self.assertEqual(client.recorded, ["generate_recommendation"])
        gated = self._delivered_envelope(provider, 2)
        self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)
        second = loop.run("再找找适合雨天听的歌")
        self.assertEqual(second.final_text, "新的搜索也有结果。")
        self.assertEqual(
            client.recorded, ["generate_recommendation", "discover_catalog_tracks"]
        )

    def test_gated_discover_instrumentation_records_unexecuted(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # ok, non-empty
            tool_response([self._discover_call("d1")]),  # gated
            text_response("已为你整理好这一批。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_recommendation"),
        ], instrument=True)
        result = loop.run("帮我选几首适合晚上听的")
        self.assertIsNotNone(result.trace)
        discover_records = [
            record for record in result.trace.tools
            if record.name == "discover_catalog_tracks"
        ]
        self.assertEqual(len(discover_records), 1)  # exactly the gated call
        record = discover_records[0]
        self.assertFalse(record.executed)
        self.assertIsNone(record.duration_ms)
        self.assertEqual(record.outcome, "execution_error")
        self.assertEqual(record.error_code, _POST_GENERATION_CLOSEOUT_ERROR_CODE)
        self.assertEqual(record.raw_result_chars, 0)
        self.assertGreater(record.delivered_result_chars, 0)
        # P15-S4-M2 (diagnostic): the refused query itself is in the trace.
        self.assertEqual(record.arguments, {"term": "夜晚"})
        self.assertFalse(record.truncated)
        self.assertFalse(record.replayed)
        self.assertEqual(result.trace.truncated_count, 0)
        # The generation record itself is an executed success -- not disguised.
        generate_records = [
            record for record in result.trace.tools
            if record.name == "generate_recommendation"
        ]
        self.assertEqual(len(generate_records), 1)
        self.assertTrue(generate_records[0].executed)
        self.assertEqual(generate_records[0].outcome, "ok")

    def test_gated_discover_produces_no_catalog_side_effect(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # ok, non-empty
            tool_response([
                self._discover_call("d1"),  # gated
                self._discover_call("d2"),  # gated again (deterministic, not deduped)
            ]),
            text_response("已为你整理好这一批。"),
        ])
        client, loop = self._closeout_loop(provider, results=[
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("帮我选几首适合晚上听的")
        self.assertFalse(result.rounds_capped)
        # Neither discovery ever crossed the client boundary -- no P09 call,
        # no catalog search, no staging writes.
        self.assertEqual(client.recorded, ["generate_recommendation"])
        for index in (0, 1):
            gated = self._delivered_envelope(provider, 2, result_index=index)
            self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)

    def test_first_success_closeout_supersedes_remaining_generation_budget(self) -> None:
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),  # ok (attempt 1)
            tool_response([self._generate_call("c2")]),  # phase-complete gate
            tool_response([self._discover_call("d1")]),  # same phase gate
            text_response("第一批仍然可用：夜曲 — 测试艺人。"),
        ])
        client, loop = self._closeout_loop(provider, results=s5_prefetch_padding() + [
            ok_generation_batch("generate_recommendation"),
        ])
        result = loop.run("换一组")
        self.assertEqual(result.final_text, "第一批仍然可用：夜曲 — 测试艺人。")
        self.assertEqual(result.rounds, 4)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED + ["generate_recommendation"],
        )
        for round_index in (2, 3):
            gated = self._delivered_envelope(provider, round_index)
            self.assertEqual(
                gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE
            )
            self.assertEqual(provider.calls[round_index - 1]["tools"], [])


class DiscoverBudgetTest(ProviderAgentLoopTest):
    """P15-S3-S3B: the per-run Fresh Catalog discovery budget. Real,
    non-replayed OK discover_catalog_tracks executions are capped at
    max_discover_per_run per loop run (P20-PerfFix02 default 1 -- one
    successful Catalog search per user turn); everything past the cap is
    answered with a deterministic synthetic refusal (discover_budget_exhausted)
    and never reaches P09. Only genuine OK executions charge: same-round
    dedupe, the read cache, the post-generation closeout, invalid arguments,
    journal replays, failed (execution_error) searches and the gate itself
    never do. The counter is a run() local, so a new user message re-enables
    discovery, and the M2 closeout keeps priority (success first, budget
    second -- distinct error codes). Tests that exercise the generic
    multi-execution mechanics pin max_discover=2 explicitly; the default-cap
    semantics are pinned by the P20-PerfFix02 cases."""

    def _budget_loop(
        self,
        provider: FakeProvider,
        results: list[AgentToolResult] | None = None,
        *,
        instrument: bool = False,
        max_discover: int = MAX_DISCOVER_PER_RUN,
    ) -> tuple[RecordingClient, ProviderAgentLoop]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(
                instrument=instrument, max_discover_per_run=max_discover
            ),
        )
        return client, loop

    def _discover_call(self, call_id: str, term: str = "夜晚") -> ProviderToolCall:
        return ProviderToolCall(
            call_id, "discover_catalog_tracks", json.dumps({"term": term})
        )

    def _delivered_envelope(self, provider: FakeProvider, round_index: int,
                            result_index: int = 0) -> dict:
        """The tool result(s) the loop fed back for a tool round: a tool call in
        round N is delivered to the model in the chat payload of round N+1."""
        user_message = provider.calls[round_index]["messages"][-1]
        return json.loads(user_message.tool_results[result_index].content)

    def test_first_discover_executes(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),
            text_response("目录里找到两首。"),
        ])
        client, loop = self._budget_loop(provider, results=[ok_discovery_result()])
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["discover_catalog_tracks"])
        first = self._delivered_envelope(provider, 1)
        self.assertEqual(first["outcome"], "ok")
        self.assertEqual(first["payload"]["discovered_count"], 2)

    def test_second_discover_executes_under_an_explicit_two_cap(self) -> None:
        # Generic budget mechanics: with max_discover_per_run=2 both fires are
        # legal (the P20-PerfFix02 default is 1 -- pinned separately below).
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),
            tool_response([self._discover_call("d2", term="雨天")]),
            text_response("两轮搜索都有结果。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        for round_index in (1, 2):
            envelope = self._delivered_envelope(provider, round_index)
            self.assertEqual(envelope["outcome"], "ok")

    def test_third_discover_blocked_without_execution(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1)
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (2)
            tool_response([self._discover_call("d3", term="清晨")]),  # blocked (3rd)
            text_response("这两轮搜索的结果先给你。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertEqual(result.final_text, "这两轮搜索的结果先给你。")
        self.assertEqual(result.rounds, 4)
        self.assertFalse(result.rounds_capped)
        # The third discovery never crossed the client boundary -- no P09 call,
        # no catalog search.
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        blocked = self._delivered_envelope(provider, 3)
        self.assertEqual(blocked["outcome"], "execution_error")
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)
        self.assertEqual(
            blocked["error_message"], _discover_budget_exhausted_message(2)
        )
        self.assertIsNone(blocked["payload"])
        self.assertFalse(blocked["replayed"])

    def test_fourth_discover_also_blocked(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1)
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (2)
            tool_response([self._discover_call("d3", term="清晨")]),  # blocked (3rd)
            tool_response([self._discover_call("d4", term="星夜")]),  # blocked (4th)
            text_response("没有找到更多，先把已有的给你。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertEqual(result.rounds, 5)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        for round_index in (3, 4):
            blocked = self._delivered_envelope(provider, round_index)
            self.assertEqual(
                blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE
            )

    def test_plain_recommendation_cannot_disable_recent_batch_freshness(self) -> None:
        """Two real service runs keep freshness and recover deterministically.

        The second direct generation is emptied by recent-run exclusion. Code,
        not the Provider, owns the one remaining attempt and routes it through
        inferred generation; the fixture has no alternate supply, so that retry
        also fails honestly and no duplicate run is persisted.
        """
        self._seed_two_positives()
        target_ids = [
            "trk_11111111-1111-4111-8111-111111111111",
            "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        ]
        args = json.dumps({
            "target_ids": target_ids, "limit": 5, "avoid_previous_runs": False,
        })
        provider = FakeProvider([
            tool_response([ProviderToolCall("g1", "generate_recommendation", args)]),
            text_response("第一批。"),
            tool_response([ProviderToolCall("g2", "generate_recommendation", args)]),
            text_response("暂时没有新的可推荐曲目。"),
        ])
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)

        first = loop.run("推荐几首歌")
        second = loop.run("推荐几首歌")

        self.assertEqual(first.recommendation_payload["item_count"], 2)
        self.assertEqual(len(first.recommendation_payload["items"]), 2)
        self.assertIsNone(second.recommendation_payload)
        self.assertEqual(second.tool_executions[0].error_code, "empty_recommendation")
        self.assertEqual(second.tool_executions[1].name, "generate_inferred_recommendation")
        self.assertEqual(second.tool_executions[1].error_code, "empty_recommendation")
        self.assertEqual(second.tool_executions[1].origin, "policy_injected")
        first_envelope = json.loads(provider.calls[1]["messages"][-1].tool_results[0].content)
        self.assertFalse(first_envelope["replayed"])
        self.assertEqual(len(provider.calls), 3)
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository
        from music_agent.recommendation_history_repository import RecommendationHistoryRepository
        with AgentRequestJournalRepository(self.database_path) as journal:
            generation_rows = [
                row
                for row in journal.list()
                if row.request.tool in _GENERATION_TOOL_NAMES
            ]
            self.assertEqual(
                [row.request.tool for row in generation_rows],
                [
                    "generate_recommendation",
                    "generate_recommendation",
                    "generate_inferred_recommendation",
                ],
            )
            self.assertEqual(
                [row.request.payload["avoid_previous_runs"] for row in generation_rows],
                [True, True, True],
            )
        with RecommendationHistoryRepository(self.database_path) as history:
            self.assertEqual(len(history.list_runs(limit=10)), 1)

    def test_new_run_resets_the_budget(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # run A (1)
            tool_response([self._discover_call("d2", term="雨天")]),  # run A (2)
            text_response("这是第一批。"),
            tool_response([self._discover_call("d3", term="清晨")]),  # run B: fresh
            text_response("新的搜索也有结果。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[ok_discovery_result(), ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        first = loop.run("找点新歌")
        self.assertEqual(first.final_text, "这是第一批。")
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        second = loop.run("换一个方向继续找")
        self.assertEqual(second.final_text, "新的搜索也有结果。")
        self.assertEqual(
            client.recorded,
            ["discover_catalog_tracks"] * 3,
        )
        fresh = self._delivered_envelope(provider, 4)
        self.assertEqual(fresh["outcome"], "ok")

    def test_blocked_discover_instrumentation_records_unexecuted(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1)
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (2)
            tool_response([self._discover_call("d3", term="清晨")]),  # blocked (3rd)
            text_response("这两轮搜索的结果先给你。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[ok_discovery_result(), ok_discovery_result()],
            instrument=True,
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertIsNotNone(result.trace)
        discover_records = [
            record for record in result.trace.tools
            if record.name == "discover_catalog_tracks"
        ]
        self.assertEqual(len(discover_records), 3)  # two executed + the blocked call
        executed = discover_records[:2]
        for record in executed:
            self.assertTrue(record.executed)
            self.assertEqual(record.outcome, "ok")
            self.assertIsNotNone(record.duration_ms)
        blocked = discover_records[2]
        self.assertFalse(blocked.executed)
        self.assertIsNone(blocked.duration_ms)
        self.assertEqual(blocked.outcome, "execution_error")
        self.assertEqual(blocked.error_code, _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)
        self.assertEqual(blocked.raw_result_chars, 0)
        self.assertGreater(blocked.delivered_result_chars, 0)
        self.assertFalse(blocked.truncated)
        self.assertFalse(blocked.replayed)
        self.assertEqual(blocked.arguments, {"term": "清晨"})

    def test_budget_error_code_is_distinct_from_sibling_codes(self) -> None:
        sibling_codes = {
            _POST_GENERATION_CLOSEOUT_ERROR_CODE,
            "generation_budget_exhausted",
            "agent_runtime_offline",
            "empty_recommendation",
        }
        self.assertNotIn(_DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE, sibling_codes)
        self.assertEqual(_DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE, "discover_budget_exhausted")

    def test_same_round_dedupe_does_not_double_charge(self) -> None:
        provider = FakeProvider([
            tool_response([
                self._discover_call("d1"),  # executes (1)
                self._discover_call("d1b"),  # identical args: same-round dedupe
            ]),
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (2)
            tool_response([self._discover_call("d3", term="清晨")]),  # blocked (3rd)
            text_response("好的。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        # The deduped twin never executed and never charged: the third call is
        # still only the second genuine execution... and the fourth is blocked.
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        first = self._delivered_envelope(provider, 1, result_index=0)
        duplicate = self._delivered_envelope(provider, 1, result_index=1)
        self.assertEqual(first["outcome"], "ok")
        self.assertEqual(duplicate["outcome"], "ok")  # same content, no error graft
        blocked = self._delivered_envelope(provider, 3)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_validation_failure_does_not_consume_budget(self) -> None:
        provider = FakeProvider([
            tool_response([ProviderToolCall("x1", "discover_catalog_tracks", "not-json")]),
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (1)
            tool_response([self._discover_call("d3", term="清晨")]),  # executes (2)
            tool_response([self._discover_call("d4", term="星夜")]),  # blocked (3rd)
            text_response("好的。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[ok_discovery_result(), ok_discovery_result()],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        # The broken-arguments call never dispatched to P09 and never charged.
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        invalid = self._delivered_envelope(provider, 1)
        self.assertEqual(invalid["outcome"], "invalid_arguments")
        self.assertNotIn("error_code", invalid)  # the pre-P09 refusal shape
        performed_second = self._delivered_envelope(provider, 3)
        self.assertEqual(performed_second["outcome"], "ok")
        blocked = self._delivered_envelope(provider, 4)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_journal_replay_does_not_consume_budget(self) -> None:
        # If the replayed result charged, d3 would be the blocked call; only a
        # free replay lets d3 execute and pushes the block onto d4.
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # journal replay: free
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (1)
            tool_response([self._discover_call("d3", term="清晨")]),  # executes (2)
            tool_response([self._discover_call("d4", term="星夜")]),  # blocked (3rd)
            text_response("好的。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[
                replayed_discovery_result(),
                ok_discovery_result(),
                ok_discovery_result(),
            ],
            max_discover=2,
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        # The replayed outcome returned the earlier result without re-running
        # the search, so it stayed free: the second and third calls executed.
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks"] * 3
        )
        self.assertEqual(self._delivered_envelope(provider, 1)["outcome"], "ok")
        self.assertEqual(self._delivered_envelope(provider, 2)["outcome"], "ok")
        third = self._delivered_envelope(provider, 3)
        self.assertEqual(third["outcome"], "ok")
        blocked = self._delivered_envelope(provider, 4)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_after_generation_success_m2_prevents_discover_regardless_of_budget(self) -> None:
        # M2 priority: a successful non-empty generation answers later discovery
        # with its own closeout even while discover budget remains available.
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            tool_response([
                self._discover_call("d1"),  # M2 gate (budget still 0/1)
                self._discover_call("d2", term="雨天"),  # M2 gate again
            ]),
            text_response("已有的一批先给你。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=s5_prefetch_padding()
            + [ok_generation_batch("generate_recommendation")],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.rounds, 3)
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        for index in (0, 1):
            gated = self._delivered_envelope(provider, 2, result_index=index)
            self.assertEqual(gated["error_code"], _POST_GENERATION_CLOSEOUT_ERROR_CODE)
            self.assertNotEqual(
                gated["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE
            )

    def test_max_discover_constant_and_config_are_pinned(self) -> None:
        # P20-PerfFix02: the per-turn budget is ONE successful Catalog search.
        self.assertEqual(MAX_DISCOVER_PER_RUN, 1)
        self.assertEqual(ProviderLoopConfig().max_discover_per_run, 1)
        with self.assertRaises(ProviderError):
            ProviderLoopConfig(max_discover_per_run=0)
        with self.assertRaises(ProviderError):
            ProviderLoopConfig(max_discover_per_run="2")

    def test_config_budget_value_is_honored_with_matching_message(self) -> None:
        # The config knob exists for tests only; the delivered refusal must
        # reflect the configured cap, not a hardcoded number.
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1 of 1)
            tool_response([self._discover_call("d2", term="雨天")]),  # blocked
            text_response("好的。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[ok_discovery_result()], max_discover=1
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["discover_catalog_tracks"])
        blocked = self._delivered_envelope(provider, 2)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)
        self.assertEqual(blocked["error_message"], _discover_budget_exhausted_message(1))

    def test_budget_exhaustion_does_not_limit_other_tools(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1)
            tool_response([self._discover_call("d2", term="雨天")]),  # executes (2)
            tool_response([
                self._discover_call("d3", term="清晨"),  # blocked
                ProviderToolCall(
                    "s1", "search_library_tracks", json.dumps({"term": "夜曲"})
                ),
                ProviderToolCall("a1", "get_active_context", "{}"),
            ]),
            text_response("现有的这批先给你。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[
                ok_discovery_result(),
                ok_discovery_result(),
                ok_read_result("search_library_tracks"),
                ok_read_result("get_active_context"),
            ],
            max_discover=2,
        )
        result = loop.run("换一批歌")
        self.assertFalse(result.rounds_capped)
        # Only the discovery was refused; the named-playback lookup path and
        # the context read still executed normally after the budget ran out.
        self.assertEqual(
            client.recorded,
            [
                "discover_catalog_tracks",
                "discover_catalog_tracks",
                "search_library_tracks",
                "get_active_context",
            ],
        )
        blocked = self._delivered_envelope(provider, 3, result_index=0)
        search = self._delivered_envelope(provider, 3, result_index=1)
        context = self._delivered_envelope(provider, 3, result_index=2)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)
        self.assertEqual(search["outcome"], "ok")
        self.assertEqual(context["outcome"], "ok")

    def test_empty_generation_keeps_discovery_allowed_under_budget(self) -> None:
        # No generation success this run: M2 stays off and the budget lets the
        # sanctioned post-empty discovery execute as before.
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("c1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            tool_response([self._discover_call("d1")]),  # executes (1)
            text_response("先看看目录里有什么。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=s5_prefetch_padding()
            + [
                empty_generation_result("generate_recommendation"),
                ok_discovery_result(),
            ],
        )
        result = loop.run("推荐几首歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded,
            S5_PREFETCH_RECORDED + ["generate_recommendation", "discover_catalog_tracks"],
        )
        discovery = self._delivered_envelope(provider, 2)
        self.assertEqual(discovery["outcome"], "ok")

    # ---- P20-PerfFix02: one successful Catalog search per user turn ---------

    def test_default_cap_blocks_a_second_discover_in_the_same_round(self) -> None:
        # The UAT shape: one provider round carrying two DISJOINT searches
        # (18.3s + 22.2s live). Only the first genuinely executes; the second
        # gets the deterministic budget refusal and the loop goes on to
        # generate the batch from the first search's results.
        provider = FakeProvider([
            tool_response([
                self._discover_call("d1"),
                self._discover_call("d2", term="雨天"),
            ]),
            tool_response([
                ProviderToolCall(
                    "g1", "generate_inferred_recommendation", GENERATE_PAYLOAD
                ),
            ]),
            text_response("这批新歌来了。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[
                ok_discovery_result(),
                ok_generation_batch("generate_inferred_recommendation"),
            ],
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(result.final_text, "这批新歌来了。")
        # Exactly one real search, then the generation from it.
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "generate_inferred_recommendation"]
        )
        first = self._delivered_envelope(provider, 1, result_index=0)
        self.assertEqual(first["outcome"], "ok")
        blocked = self._delivered_envelope(provider, 1, result_index=1)
        self.assertEqual(blocked["outcome"], "execution_error")
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)
        self.assertEqual(blocked["error_message"], _discover_budget_exhausted_message(1))
        self.assertIsNone(blocked["payload"])
        self.assertFalse(blocked["replayed"])

    def test_default_cap_blocks_a_second_discover_across_rounds(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # executes (1 of 1)
            tool_response([self._discover_call("d2", term="雨天")]),  # blocked
            text_response("一次搜索的结果。"),
        ])
        client, loop = self._budget_loop(provider, results=[ok_discovery_result()])
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["discover_catalog_tracks"])
        blocked = self._delivered_envelope(provider, 2)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_failed_discover_does_not_consume_the_single_budget(self) -> None:
        # Failure recovery: a transient Catalog failure is delivered fail-honest,
        # stays FREE, and the same turn may retry -- the budget caps successful
        # searches, not attempts. The retry is the one success; a third call is
        # refused.
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # fails (free)
            tool_response([self._discover_call("d2", term="雨天")]),  # retry: ok (1)
            tool_response([self._discover_call("d3", term="清晨")]),  # blocked
            text_response("重试后找到了。"),
        ])
        client, loop = self._budget_loop(
            provider, results=[failed_discovery_result(), ok_discovery_result()]
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "discover_catalog_tracks"]
        )
        failed = self._delivered_envelope(provider, 1)
        self.assertEqual(failed["outcome"], "execution_error")
        self.assertEqual(failed["error_code"], "catalog_http_error")
        retried = self._delivered_envelope(provider, 2)
        self.assertEqual(retried["outcome"], "ok")
        blocked = self._delivered_envelope(provider, 3)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_identical_term_repeat_is_refused_with_no_second_catalog_access(
        self,
    ) -> None:
        # A repeat of the EXACT same call never re-touches the Catalog. The
        # code truth: discover_catalog_tracks is _CACHE_INVALIDATING_TOOLS --
        # its own ok execution clears the per-run read cache (its promotions
        # mutate the catalog store, so cached reads of the pre-discovery state
        # must go stale), so the repeat cannot be answered by cache replay.
        # It hits the per-run budget gate instead: a deterministic synthetic
        # refusal, one real P09 call total, zero second Catalog access (§八-9).
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),
            tool_response([self._discover_call("d1a")]),  # same term, same args
            text_response("同一批结果。"),
        ])
        client, loop = self._budget_loop(provider, results=[ok_discovery_result()])
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(client.recorded, ["discover_catalog_tracks"])
        refused = self._delivered_envelope(provider, 2)
        self.assertEqual(refused["outcome"], "execution_error")
        self.assertEqual(
            refused["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE
        )
        self.assertEqual(
            refused["error_message"], _discover_budget_exhausted_message(1)
        )
        self.assertIsNone(refused["payload"])
        self.assertFalse(refused["replayed"])

    def test_default_cap_refusal_mints_no_fresh_ids(self) -> None:
        # Fresh honesty: the refused second search never touches the run-local
        # fresh-promoted set, so generate gets ONLY the first search's promoted
        # ids -- the synthetic refusal mints no fresh identity for anyone.
        provider = FakeProvider([
            tool_response([self._discover_call("d1")]),  # ok, promotes FRESH_ID_A
            tool_response([self._discover_call("d2", term="雨天")]),  # refused
            tool_response([
                ProviderToolCall(
                    "g1", "generate_inferred_recommendation", GENERATE_PAYLOAD
                ),
            ]),
            text_response("这批新歌来了。"),
        ])
        client, loop = self._budget_loop(
            provider,
            results=[
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                ok_generation_batch("generate_inferred_recommendation"),
            ],
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(
            client.recorded, ["discover_catalog_tracks", "generate_inferred_recommendation"]
        )
        # Call 0 = the genuine discover (no fresh kwarg on discovers), call 1 =
        # the generation, whose fresh transport carries exactly the first
        # search's promotion -- one id, nothing fabricated by the refusal.
        self.assertIsNone(client.fresh_kwargs[0])
        self.assertEqual(client.fresh_kwargs[1], (FRESH_ID_A,))
        blocked = self._delivered_envelope(provider, 2)
        self.assertEqual(blocked["error_code"], _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE)

    def test_discover_schema_stays_term_and_limit_without_force_refresh(self) -> None:
        schema = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}["discover_catalog_tracks"]
        self.assertEqual(schema.input_schema["required"], ["term"])
        self.assertEqual(
            set(schema.input_schema["properties"]), {"term", "limit"}
        )
        serialized = json.dumps(dict(schema.input_schema), ensure_ascii=False)
        self.assertNotIn("force_refresh", serialized)

    def test_no_new_model_visible_tool(self) -> None:
        self.assertEqual(len(PROVIDER_TOOL_SCHEMAS), 31)  # + P16-S4 open_in_apple_music
        names = [item.name for item in PROVIDER_TOOL_SCHEMAS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("discover_catalog_tracks", names)
        self.assertIn("query_catalog_discovery_state", names)
        self.assertIn("open_in_apple_music", names)
        for name in names:
            self.assertNotIn("supply", name)
            self.assertNotIn("budget", name)

    def test_schema_version_stays_v19(self) -> None:
        self.assertEqual(CURRENT_SCHEMA_VERSION, 19)


FRESH_ID_A = "trk_fbee14c2-0000-4000-8000-00000000000a"
FRESH_ID_B = "trk_fbee14c2-0000-4000-8000-00000000000b"


def promoted_discovery_result(
    promoted: list[dict],
    *,
    replayed: bool = False,
    outcome: AgentToolOutcome = AgentToolOutcome.OK,
    payload_overrides: dict | None = None,
) -> AgentToolResult:
    """Canned discover result whose envelope carries the real promoted/staged/
    skipped payload shape (P11) -- the S3-S3D capture contract reads this."""
    payload = {
        "term": "夜晚",
        "discovered_count": len(promoted),
        "promoted_count": len(promoted),
        "staged_count": 0,
        "skipped_count": 0,
        "promoted": promoted,
        "staged": [],
        "skipped": [],
    }
    if payload_overrides:
        payload.update(payload_overrides)
    return AgentToolResult(
        request_id="req_55555555-5555-4555-8555-555555555555",
        tool="discover_catalog_tracks",
        outcome=outcome,
        payload=payload,
        error_code=None,
        error_message=None,
        completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
        replayed=replayed,
    )


class FreshCaptureTest(unittest.TestCase):
    """P15-S3-S3D capture contract (test plan A, items 1-4, corrected by the
    P15-S3-S3E RAW capture patch): the run-local Fresh set takes ONLY
    promoted-status canonical ids from the RAW structured payload of a
    discover result -- everything else is under-capture, never fabrication.

    The helper receives the raw payload dict directly; outcome/replay gating
    belongs to the caller inside ``_execute_tool_call`` (proven by
    TruncatedRawCaptureRegressionTest) and can no longer be expressed at this
    level. A delivered truncation marker is presentation-only input and
    contributes nothing on its own -- but it is never the capture's input:
    when a REAL raw execution result exists with truncated delivered content,
    capture must succeed from the raw payload (see
    test_truncated_discover_keeps_full_raw_capture)."""

    def _capture(self, payload: object) -> set[str]:
        ids: set[str] = set()
        _capture_fresh_promoted_ids(payload, ids)
        return ids

    def _payload(self, **overrides: object) -> dict:
        payload: dict = {
            "term": "夜晚",
            "promoted": [
                {"status": "promoted", "canonical_id": FRESH_ID_A}
            ],
        }
        payload.update(overrides)
        return payload

    def test_promoted_status_entries_enter_the_set(self) -> None:
        self.assertEqual(self._capture(self._payload()), {FRESH_ID_A})

    def test_two_discover_payloads_union(self) -> None:
        ids: set[str] = set()
        _capture_fresh_promoted_ids(
            {"promoted": [{"status": "promoted", "canonical_id": FRESH_ID_A}]},
            ids,
        )
        _capture_fresh_promoted_ids(
            {"promoted": [{"status": "promoted", "canonical_id": FRESH_ID_B}]},
            ids,
        )
        self.assertEqual(ids, {FRESH_ID_A, FRESH_ID_B})

    def test_non_promoted_status_and_bad_ids_are_ignored(self) -> None:
        payload = {
            "promoted": [
                {"status": "staged", "canonical_id": FRESH_ID_A},
                {"status": "promoting", "canonical_id": FRESH_ID_B},
                {"canonical_id": FRESH_ID_A},  # status missing
                {"status": "promoted"},  # id missing
                {"status": "promoted", "canonical_id": ""},
                {"status": "promoted", "canonical_id": 7},
                {"status": "promoted", "canonical_id": None},
            ]
        }
        self.assertEqual(self._capture(payload), set())

    def test_staged_and_skipped_lists_never_contribute(self) -> None:
        # ALREADY_BOUND / LIBRARY_KNOWN land in ``skipped`` (not ``promoted``)
        # structurally -- defensive: even a curated promoted list carrying
        # non-promoted entries contributes nothing but the true promoted ones.
        payload = {
            "promoted": [
                {"status": "promoted", "canonical_id": FRESH_ID_A},
                {"status": "already_bound", "canonical_id": FRESH_ID_B},
            ],
            "staged": [{"status": "staged", "canonical_id": FRESH_ID_B}],
            "skipped": [
                {"status": "already_bound", "canonical_id": FRESH_ID_B},
                {"status": "library_known", "canonical_id": FRESH_ID_A},
            ],
        }
        self.assertEqual(self._capture(payload), {FRESH_ID_A})

    def test_raw_structured_shape_is_required(self) -> None:
        # Non-Mapping inputs are not a capture authority surface -- they
        # contribute nothing and never raise.
        for payload in (None, "not json", "text", [1, 2], 7):
            self.assertEqual(self._capture(payload), set())

    def test_truncated_or_marker_payload_captures_nothing_by_itself(self) -> None:
        # Reframed by the RAW capture patch: a delivered truncation marker
        # (or any payload without a promoted list) contributes nothing ON ITS
        # OWN -- it is simply not the capture input anymore, because the
        # caller passes the raw payload, never the bounded delivered envelope.
        # The distinct-authority separation is proven at loop level by
        # TruncatedRawCaptureRegressionTest: with a REAL raw execution result
        # present, capture MUST succeed even when the delivered content is
        # truncated.
        for payload in ({"truncated": True, "preview": "..."}, {"outcome": "ok"}, {}):
            self.assertEqual(self._capture(payload), set())


class FreshTransportLoopTest(ProviderAgentLoopTest):
    """P15-S3-S3D transport (test plan B, items 5-7): the loop feeds the
    authoritative same-run set into generation calls as the INTERNAL execution
    kwarg -- only genuine executions capture, replays and blocked discovers
    never rebuild Fresh, each new run() starts empty, and the model arguments
    / journal payloads never carry the ids."""

    def _loop(
        self,
        provider: FakeProvider,
        results: list[AgentToolResult],
        *,
        max_discover: int = MAX_DISCOVER_PER_RUN,
    ) -> tuple[RecordingClient, ProviderAgentLoop]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_discover_per_run=max_discover),
        )
        return client, loop

    @staticmethod
    def _discover(call_id: str, term: str = "夜晚") -> ProviderToolCall:
        return ProviderToolCall(
            call_id, "discover_catalog_tracks", json.dumps({"term": term})
        )

    def test_genuine_discover_feeds_the_generation_kwarg(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([
                ProviderToolCall(
                    "g1", "generate_inferred_recommendation", GENERATE_PAYLOAD
                )
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                ok_generation_batch("generate_inferred_recommendation"),
            ],
        )
        result = loop.run("找点新歌")
        self.assertFalse(result.rounds_capped)
        # The discover call carried no kwarg (nothing captured yet, and it is
        # not a generation tool); the generation call carried the sorted tuple.
        self.assertEqual(client.fresh_kwargs, [None, (FRESH_ID_A,)])
        self.assertEqual(client.recorded, ["discover_catalog_tracks",
                                           "generate_inferred_recommendation"])

    def test_multiple_discovers_union_and_sort_before_generation(self) -> None:
        # Transport mechanics pinned under an explicit two-cap (the generic
        # multi-execution shape -- the default turn budget is one search).
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([self._discover("d2", term="雨天")]),
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_B}]
                ),
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                ok_generation_batch("generate_recommendation"),
            ],
            max_discover=2,
        )
        loop.run("找点新歌")
        # Deterministic transport: sorted canonical ids, one kwarg tuple.
        self.assertEqual(
            client.fresh_kwargs,
            [None, None, (FRESH_ID_A, FRESH_ID_B)],
        )

    def test_replayed_discover_never_rebuilds_fresh(self) -> None:
        # prefer NOT to rebuild Fresh from replay: the journal answer carries
        # the same payload but replayed=True, so the set stays empty and the
        # generation kwarg is absent.
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}],
                    replayed=True,
                ),
                ok_generation_batch("generate_recommendation"),
            ],
        )
        loop.run("找点新歌")
        self.assertEqual(client.fresh_kwargs, [None, None])

    def test_execution_error_discover_captures_nothing(self) -> None:
        # Loop-level replacement for the former unit pin on outcome parsing:
        # the RAW capture gate requires outcome==ok, so an executed-but-failed
        # discover never feeds Fresh. (Non-ok results may not carry a payload
        # under the P09 contract -- the gate fires on the outcome, before any
        # promoted list could ever be read.)
        failed = AgentToolResult(
            request_id="req_66666666-6666-4666-8666-666666666666",
            tool="discover_catalog_tracks",
            outcome=AgentToolOutcome.EXECUTION_ERROR,
            payload=None,
            error_code="catalog_timeout",
            error_message="catalog search failed",
            completed_at=datetime(2026, 8, 21, 1, 0, 0, tzinfo=timezone.utc),
            replayed=False,
        )
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                failed,
                ok_generation_batch("generate_recommendation"),
            ],
        )
        loop.run("找点新歌")
        self.assertEqual(client.fresh_kwargs, [None, None])

    def test_each_run_starts_with_an_empty_fresh_set(self) -> None:
        provider = FakeProvider([
            # run 1: discover then generate.
            tool_response([self._discover("d1")]),
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
            # run 2: generate only -- the previous run's captures must be gone.
            tool_response([
                ProviderToolCall("g2", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                ok_generation_batch("generate_recommendation"),
                ok_generation_batch("generate_recommendation"),
            ],
        )
        first = loop.run("找点新歌")
        second = loop.run("再推荐一次")
        self.assertFalse(first.rounds_capped)
        self.assertFalse(second.rounds_capped)
        self.assertEqual(client.fresh_kwargs, [None, (FRESH_ID_A,), None])

    def test_non_generation_calls_never_receive_the_kwarg(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([
                ProviderToolCall(
                    "r1",
                    "list_recommendation_runs",
                    json.dumps({"limit": 3}),
                )
            ]),
            text_response("最近的批次如下。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                ok_read_result("list_recommendation_runs"),
            ],
        )
        loop.run("看看最近的推荐")
        self.assertEqual(client.fresh_kwargs, [None, None])

    def test_budget_blocked_discover_adds_no_fresh_ids(self) -> None:
        provider = FakeProvider([
            tool_response([self._discover("d1")]),
            tool_response([self._discover("d2", term="雨天")]),
            tool_response([self._discover("d3", term="清晨")]),  # blocked: 3rd
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider,
            [
                promoted_discovery_result(
                    [{"status": "promoted", "canonical_id": FRESH_ID_A}]
                ),
                promoted_discovery_result([], payload_overrides={"skipped": [
                    {"status": "already_bound", "canonical_id": FRESH_ID_B}
                ]}),
                ok_generation_batch("generate_recommendation"),
            ],
            max_discover=2,
        )
        loop.run("找点新歌")
        # The blocked third discover never crossed the client; only the two
        # genuine in-budget executions fed the kwarg -- and skipped entries
        # contributed nothing.
        self.assertEqual(client.recorded, ["discover_catalog_tracks",
                                           "discover_catalog_tracks",
                                           "generate_recommendation"])
        self.assertEqual(client.fresh_kwargs, [None, None, (FRESH_ID_A,)])

    def test_invalid_argument_discover_never_captures(self) -> None:
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("d1", "discover_catalog_tracks", "not-json")
            ]),
            tool_response([
                ProviderToolCall("g1", "generate_recommendation", GENERATE_PAYLOAD)
            ]),
            text_response("已推荐。"),
        ])
        client, loop = self._loop(
            provider, [ok_generation_batch("generate_recommendation")]
        )
        loop.run("找点新歌")
        # The malformed discover never reached the client at all (executed
        # False), so nothing was captured and nothing fed the generation.
        self.assertEqual(client.recorded, ["generate_recommendation"])
        self.assertEqual(client.fresh_kwargs, [None])


class LoopCatalogSource:
    """Minimal deterministic catalog transport for the promotion e2e test."""

    def __init__(self, tracks: tuple) -> None:
        self.tracks = tracks
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> tuple:
        self.calls.append((term, limit))
        return self.tracks


class _CapturingAgentClient(AgentClient):
    """AgentClient that keeps every RAW service result, so tests can assert on
    the full P09 payload -- the truth source -- even when the loop's
    model-visible envelope is subject to _MAX_TOOL_RESULT_CHARS truncation.
    Also records the internal fresh kwarg per call (None when absent)."""

    def __init__(self, identity: AgentClientIdentity, service: SharedAgentService):
        super().__init__(identity, service)
        self.raw_results: list = []
        self.fresh_kwargs: list[tuple[str, ...] | None] = []

    def call(self, *args, **kwargs):
        ids = kwargs.get("fresh_canonical_ids")
        self.fresh_kwargs.append(tuple(ids) if ids is not None else None)
        result = super().call(*args, **kwargs)
        self.raw_results.append(result)
        return result


class FreshPromotionEndToEndTest(ProviderAgentLoopTest):
    """P15-S3-S3D end to end: a REAL P09 discover promotes a REAL canonical
    track, the loop captures its id from the REAL result envelope, the REAL
    inferred handler receives it via the kwarg, the min_fresh floor swaps the
    batch, and the delivered envelope reports the truth -- the Request-2 fix in
    miniature, fully deterministic, zero paid provider."""

    FRESH_CATALOG_ID = "CATALOG-FRESH-1"

    def _fresh_catalog_track(self) -> "CatalogTrack":
        return CatalogTrack(
            catalog_id=self.FRESH_CATALOG_ID,
            name="Fresh Catalog Song",
            artist_names=("Artist Alpha",),
            album_name="Catalog Album",
            genres=("Synthetic Pop",),
            isrc="USSYN2400099",
            duration_ms=201000,
            release_date="2024-01-15",
            url=None,
            artist_catalog_ids=("CATALOG-ARTIST-1",),
            album_catalog_id="CATALOG-ALBUM-1",
        )

    def test_discover_promote_generate_min_fresh_end_to_end(self) -> None:
        self._seed_two_positives()
        source = LoopCatalogSource((self._fresh_catalog_track(),))
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        client = _CapturingAgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
        )
        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "d1", "discover_catalog_tracks", json.dumps({"term": "fresh"})
                )
            ]),
            tool_response([
                ProviderToolCall(
                    "g1",
                    "generate_inferred_recommendation",
                    json.dumps(
                        {
                            "target_ids": [
                                "trk_11111111-1111-4111-8111-111111111111",
                                "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                            ],
                            "limit": 2,
                            "min_fresh": 1,
                        }
                    ),
                )
            ]),
            text_response("已推荐。"),
        ])
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        result = loop.run("给我来点新歌")
        # P20-Fix10: the success scene's user-visible output is the
        # deterministic rendering of the authoritative batch (the provider's
        # free text is replaced wholesale); the flow facts asserted below are
        # the point of this end-to-end test and are unchanged.
        self.assertTrue(result.final_text.startswith("为你推荐这 2 首："), result.final_text)
        self.assertIn("1. Synthetic Duet — Artist Alpha, Artist Beta", result.final_text)
        self.assertIn("这首来自你对这首曲目本身的已有偏好。", result.final_text)
        self.assertIn("这是本次新发现，按 Synthetic Pop 方向推断出来。", result.final_text)
        self.assertTrue(result.final_text.endswith("需要试听哪一首，直接告诉我。"), result.final_text)
        self.assertEqual(source.calls, [("fresh", 25)])
        # The promoted canonical id: exactly the model track bound to the
        # catalog id this run genuinely discovered.
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
        promoted_id = next(
            track["id"]
            for track in model["tracks"]
            if track["external_ids"].get("apple_music_catalog_id")
            == self.FRESH_CATALOG_ID
        )
        # The generation's RAW service result (captured before the loop's
        # model-visible bounding): fresh floor swapped the batch and the truth
        # fields report 1/2 fresh.
        raw = client.raw_results[-1]
        self.assertEqual(raw.outcome.value, "ok")
        self.assertEqual(raw.payload["fresh_item_count"], 1)
        by_id = {
            item["target_id"]: item["fresh_this_request"]
            for item in raw.payload["items"]
        }
        self.assertEqual(
            by_id,
            {
                "trk_11111111-1111-4111-8111-111111111111": False,
                promoted_id: True,
            },
        )
        raw_gen = json.dumps(dict(raw.payload), ensure_ascii=False, sort_keys=False)
        # The delivered envelope (the tool result fed back to the model in the
        # round after its call) carries the outcome either unbounded or as the
        # pre-existing 2000-char truncation marker; when truncated the truth
        # summary key still sits inside the preview (it precedes the items).
        generation = json.loads(
            provider.calls[2]["messages"][-1].tool_results[0].content
        )
        self.assertEqual(generation["outcome"], "ok")
        delivered = generation["payload"]
        if delivered.get("truncated"):
            self.assertIn("fresh_item_count", delivered["preview"])
        else:
            self.assertEqual(delivered["fresh_item_count"], 1)
            self.assertEqual(json.dumps(delivered, sort_keys=True), json.dumps(json.loads(raw_gen), sort_keys=True))
        # The discover envelope itself never carries fresh identity, and the
        # journal payloads never carry the ids -- the transport stayed internal.
        discovery = json.loads(
            provider.calls[1]["messages"][-1].tool_results[0].content
        )
        self.assertIn("promoted", discovery["payload"])
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            for row in journal.list():
                encoded = json.dumps(dict(row.request.payload), ensure_ascii=False)
                self.assertNotIn(FRESH_ID_A, encoded)
                self.assertNotIn("fresh_canonical_ids", encoded)


def _raw_regression_canonical(prefix: str, n: int) -> str:
    return f"trk_{prefix}{n:04x}-0000-4000-8000-000000000000"


class TruncatedRawCaptureRegressionTest(ProviderAgentLoopTest):
    """P15-S3-S3E corrective patch (directive §6): the 2000-char delivered
    truncation can no longer starve the Fresh capture. A REAL-scale discover
    payload (>2000 chars, 19 promoted + 6 already_bound) reproduces the live
    19->0 defect shape; the corrected loop captures all 19 from the RAW payload
    while the delivered envelope stays the bounded marker -- model-visible
    behavior unchanged.

    Covers §6 regressions A (delivered truncated==true), B-D (the model-visible
    preview need not contain the complete promoted set, yet the loop captures
    the full RAW promoted list and hands the complete sorted set to the
    generation call), and F (already_bound entries never enter). Regression E
    (fresh_promoted_count == raw promoted count) needs the real P09 service and
    lives in LiveFailureDeterministicReplayTest."""

    PROMOTED_IDS = tuple(_raw_regression_canonical("fdee", i) for i in range(1, 20))
    SKIPPED_IDS = tuple(_raw_regression_canonical("fba0", i) for i in range(1, 7))

    @classmethod
    def _big_discovery_result(cls) -> "AgentToolResult":
        # Each promoted entry carries a long display name, so the serialized
        # payload comfortably exceeds _MAX_TOOL_RESULT_CHARS=2000 -- the shape
        # of every real 25-hit iTunes discover (~9.5KB in the live failure).
        padding = "声" * 130
        promoted = [
            {
                "status": "promoted",
                "canonical_id": cid,
                "catalog_id": f"CATALOG-P-{i:02d}",
                "name": f"Promoted Song {i:02d} {padding}",
                "artist_name": "周杰伦",
                "preview_url": f"https://example.com/p{i:02d}",
                "error": None,
                "blocker_codes": [],
            }
            for i, cid in enumerate(cls.PROMOTED_IDS, 1)
        ]
        skipped = [
            {
                "status": "already_bound",
                "canonical_id": cid,
                "catalog_id": f"CATALOG-S-{i:02d}",
                "name": f"Bound Song {i:02d}",
                "artist_name": "周杰伦",
                "preview_url": f"https://example.com/s{i:02d}",
                "error": None,
                "blocker_codes": [],
            }
            for i, cid in enumerate(cls.SKIPPED_IDS, 1)
        ]
        payload = {
            "term": "周杰伦",
            "limit": 25,
            "discovered_count": len(promoted) + len(skipped),
            "promoted_count": len(promoted),
            "staged_count": 0,
            "skipped_count": len(skipped),
            "promoted": promoted,
            "staged": [],
            "skipped": skipped,
        }
        assert (
            len(json.dumps(payload, ensure_ascii=False)) > 2000
        ), "fixture must exceed the truncation threshold to be meaningful"
        return AgentToolResult(
            request_id="req_77777777-7777-4777-8777-777777777777",
            tool="discover_catalog_tracks",
            outcome=AgentToolOutcome.OK,
            payload=payload,
            error_code=None,
            error_message=None,
            completed_at=datetime(2026, 8, 21, 1, 0, 0, tzinfo=timezone.utc),
            replayed=False,
        )

    def test_truncated_discover_keeps_full_raw_capture(self) -> None:
        big = self._big_discovery_result()
        generate_args = json.dumps({"limit": 5, "min_fresh": 5})
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("d1", "discover_catalog_tracks", json.dumps({"term": "周杰伦"}))
            ]),
            tool_response([
                ProviderToolCall("g1", "generate_inferred_recommendation", generate_args),
                ProviderToolCall("g2", "generate_inferred_recommendation", generate_args),
                ProviderToolCall("g3", "generate_inferred_recommendation", generate_args),
            ]),
            text_response("这一句永远不该出现。"),
        ])
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[big, empty_generation_result("generate_inferred_recommendation")],
        )
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        result = loop.run("库外新歌")
        # Every allowed generation attempt failed (the canned empty refusal),
        # so the loop terminated deterministically -- capture had already
        # happened inside _execute_tool_call, before that outcome.
        self.assertEqual(result.final_text, _GENERATION_FAILURE_CLOSEOUT)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(len(provider.calls), 2)
        # A. The delivered envelope keeps the bounded marker -- exactly what
        # the live run delivered (truncation behavior unchanged).
        delivered = json.loads(
            provider.calls[1]["messages"][-1].tool_results[0].content
        )
        self.assertEqual(delivered["outcome"], "ok")
        self.assertIs(delivered["payload"]["truncated"], True)
        # B. The model-visible payload is the marker: no promoted list, and at
        # least one promoted canonical id sits beyond the 2000-char preview cut.
        self.assertNotIn("promoted", delivered["payload"])
        preview = delivered["payload"]["preview"]
        self.assertTrue(
            any(cid not in preview for cid in self.PROMOTED_IDS),
            "fixture must place some promoted ids beyond the preview cut",
        )
        # C+D. The loop captured the complete RAW promoted set, and the
        # generation call received the complete sorted tuple as the internal
        # kwarg -- old implementation captured zero here (19 -> 0), the
        # corrected one captures all 19 (19 -> 19).
        self.assertEqual(
            client.recorded,
            ["discover_catalog_tracks", "generate_inferred_recommendation"],
        )
        self.assertEqual(
            client.fresh_kwargs,
            [None, tuple(sorted(self.PROMOTED_IDS))],
        )
        # F. already_bound entries never enter the authoritative set.
        self.assertFalse(
            any(cid in client.fresh_kwargs[1] for cid in self.SKIPPED_IDS)
        )


class TermCatalogSource:
    """Term-sensitive deterministic catalog transport: each term maps to its own
    track set, so one shared service can host the mixed bound/fresh seeding and
    the live 起风了 / 周杰伦 replay with a single source."""

    def __init__(self, by_term: dict[str, tuple]) -> None:
        self._by_term = by_term
        self.calls: list[tuple[str, int]] = []

    def search(self, term: str, limit: int) -> tuple:
        self.calls.append((term, limit))
        return self._by_term[term]


class LiveFailureDeterministicReplayTest(ProviderAgentLoopTest):
    """P15-S3-S3E corrective patch (directive §9): deterministic replay of the
    latest live request -- in-loop discover 起风了 (25 hits, ALL already_bound)
    then 周杰伦 (25 hits: 19 promoted + 6 already_bound), then
    generate_inferred_recommendation(limit=5, min_fresh=5) -- through the REAL
    P09 service with a scripted source (zero paid provider, zero network). The
    audit proved zero negative heads for the 19 promoted, so the fixed chain
    must read fresh_promoted_count=19, fresh_candidate_count=19,
    fresh_negative_rejected_count=0, and fresh_item_count >= 1.

    Also carries §8's regression: a truncated real discover RAW payload with
    promoted rows still activates the Fresh channel end to end."""

    def _track(self, catalog_id: str, name: str, index: int) -> CatalogTrack:
        return CatalogTrack(
            catalog_id=catalog_id,
            name=f"{name} {'歌' * 40}",
            artist_names=("Jay Chou",),
            album_name=f"Live Album {index}",
            genres=("Synthetic Pop",),
            isrc=f"USTEST{index:09d}",
            duration_ms=201000,
            release_date="2024-01-15",
            url=None,
            artist_catalog_ids=(f"LIVE-ARTIST-{index}",),
            album_catalog_id=f"LIVE-ALBUM-{index}",
        )

    def test_live_19_promoted_replay_reaches_min_fresh_batch(self) -> None:
        self._seed_two_positives()
        bound_25 = tuple(
            self._track(f"LIVE-B25-{i:02d}", f"Jay Chou Bound A {i:02d}", i)
            for i in range(1, 26)
        )
        bound_6 = tuple(
            self._track(f"LIVE-B06-{i:02d}", f"Jay Chou Bound B {i:02d}", 100 + i)
            for i in range(1, 7)
        )
        promoted_19 = tuple(
            self._track(f"LIVE-P19-{i:02d}", f"Jay Chou Live New {i:02d}", 200 + i)
            for i in range(1, 20)
        )
        source = TermCatalogSource({
            "seed-bound-25": bound_25,
            "seed-bound-6": bound_6,
            "起风了": bound_25,
            "周杰伦": bound_6 + promoted_19,
        })
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        client = _CapturingAgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
        )
        # Pre-run seeding through the real discover tool: binds the 31 known
        # tracks. Out-of-loop requests -- their ids must never leak into the
        # run's Fresh set (the capture lives inside the loop only).
        client.call("discover_catalog_tracks", {"term": "seed-bound-25"})
        client.call("discover_catalog_tracks", {"term": "seed-bound-6"})
        client.raw_results.clear()
        client.fresh_kwargs.clear()
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("d1", "discover_catalog_tracks", json.dumps({"term": "起风了"})),
                ProviderToolCall("d2", "discover_catalog_tracks", json.dumps({"term": "周杰伦"})),
            ]),
            tool_response([
                ProviderToolCall(
                    "g1",
                    "generate_inferred_recommendation",
                    json.dumps(
                        {
                            # target_ids is a required field; the seeded
                            # positives stand in for the live request's
                            # promoted targets (the live ids do not exist in
                            # this test DB) -- Fresh supply rides the internal
                            # kwarg, independent of the target list.
                            "target_ids": [
                                "trk_11111111-1111-4111-8111-111111111111",
                                "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                            ],
                            "limit": 5,
                            "min_fresh": 5,
                        }
                    ),
                ),
            ]),
            text_response("已推荐。"),
        ])
        # The live shape ran TWO genuine searches in one round; the default
        # turn budget is now one, so this replay pins the transport mechanics
        # under an explicit two-cap (RAW capture union, truncation honesty,
        # min_fresh supply) instead of the default-cap semantics.
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_discover_per_run=2),
        )
        result = loop.run("库外新歌，尽量找新的")
        # P20-Fix10: the success scene is the deterministic rendering of the
        # fresh batch -- 5 items, each honestly 本次新发现 with its inferred
        # direction reason (the replay/transport facts asserted below are the
        # point of this test and are unchanged).
        self.assertTrue(result.final_text.startswith("为你推荐这 5 首："), result.final_text)
        self.assertEqual(
            result.final_text.count("这是本次新发现，按 Synthetic Pop 方向推断出来。"), 5
        )
        self.assertTrue(result.final_text.endswith("需要试听哪一首，直接告诉我。"), result.final_text)
        self.assertEqual(
            source.calls,
            [
                ("seed-bound-25", 25),
                ("seed-bound-6", 25),
                ("起风了", 25),
                ("周杰伦", 25),
            ],
        )
        promoted_canonical_ids = self._canonical_ids_for_catalog_ids(
            {f"LIVE-P19-{i:02d}" for i in range(1, 20)}
        )
        bound_canonical_ids = self._canonical_ids_for_catalog_ids(
            {f"LIVE-B25-{i:02d}" for i in range(1, 26)}
            | {f"LIVE-B06-{i:02d}" for i in range(1, 7)}
        )
        self.assertEqual(len(promoted_canonical_ids), 19)
        self.assertEqual(len(bound_canonical_ids), 31)
        promoted_set = set(promoted_canonical_ids)
        bound_set = set(bound_canonical_ids)
        # A+B. Both delivered envelopes keep the bounded marker; the 周杰伦
        # one carries no promoted list and parts of the promoted set lie beyond
        # the preview cut -- yet capture must have succeeded from RAW.
        discovery_round = provider.calls[1]["messages"][-1].tool_results
        feng_delivered = json.loads(discovery_round[0].content)
        self.assertIs(feng_delivered["payload"]["truncated"], True)
        zhou_delivered = json.loads(discovery_round[1].content)
        self.assertIs(zhou_delivered["payload"]["truncated"], True)
        self.assertNotIn("promoted", zhou_delivered["payload"])
        self.assertTrue(
            any(
                cid not in zhou_delivered["payload"]["preview"]
                for cid in promoted_canonical_ids
            )
        )
        # C+D+F. The complete RAW promoted set reached the generation call as
        # the internal kwarg; discover calls carried none; the 31 already_bound
        # canonicals never enter the set.
        self.assertEqual(
            client.fresh_kwargs,
            [None, None, tuple(sorted(promoted_canonical_ids))],
        )
        self.assertFalse(
            any(cid in bound_set for cid in client.fresh_kwargs[2])
        )
        # E. The generation's RAW service result reports the true funnel:
        # 19 promoted -> 19 candidates -> 0 vetoed; min_fresh floor then put
        # fresh items into the batch (exactly the chain the live run starved).
        raw = client.raw_results[-1]
        self.assertEqual(raw.outcome.value, "ok")
        self.assertEqual(raw.payload["fresh_promoted_count"], 19)
        self.assertEqual(raw.payload["fresh_candidate_count"], 19)
        self.assertEqual(raw.payload["fresh_negative_rejected_count"], 0)
        self.assertEqual(raw.payload["item_count"], 5)
        self.assertGreaterEqual(raw.payload["fresh_item_count"], 1)
        fresh_targets = {
            item["target_id"]
            for item in raw.payload["items"]
            if item["fresh_this_request"]
        }
        self.assertTrue(fresh_targets)
        # Every fresh-flagged item targets a genuinely promoted canonical; no
        # already_bound track can ever be flagged Fresh.
        self.assertTrue(fresh_targets <= promoted_set)
        self.assertFalse(fresh_targets & bound_set)
        # The journal payloads never carry the ids -- the transport stayed
        # internal end to end.
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            for row in journal.list():
                encoded = json.dumps(dict(row.request.payload), ensure_ascii=False)
                self.assertNotIn("fresh_canonical_ids", encoded)


class _RecordingExecuteService(SharedAgentService):
    """Real shared service that also records the authoritative execution seam:
    each executed request's tool name, the internal Fresh set that arrived,
    and the RAW result -- the observation point proving internal context
    reached the real service through the real RoutedAgentClient delegate path
    (no fake client, no kwargs-swallowing subclass)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen_executions: list[tuple[str, tuple[str, ...], AgentToolResult]] = []

    def execute(
        self,
        request,
        *,
        completed_at: str | None = None,
        fresh_canonical_ids: tuple[str, ...] | None = None,
        recommendation_scope_ids: tuple[str, ...] | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> AgentToolResult:
        result = super().execute(
            request,
            completed_at=completed_at,
            fresh_canonical_ids=fresh_canonical_ids,
            recommendation_scope_ids=recommendation_scope_ids,
            similarity_context=similarity_context,
        )
        self.seen_executions.append(
            (
                str(request.tool),
                tuple(fresh_canonical_ids) if fresh_canonical_ids is not None else (),
                result,
            )
        )
        return result


class RoutedClientFreshTransportEndToEndTest(ProviderAgentLoopTest):
    """P15-S3-S3D/E live-equivalent regression for the routed-wiring gap: the
    real chat-session binds ProviderAgentLoop to a REAL RoutedAgentClient
    (cli.py), which previously raised TypeError on the injected
    ``fresh_canonical_ids`` kwarg. No RecordingClient, no kwargs-swallowing
    fake: the loop executes through the actual override, discover + generate
    ride the local delegate (they are not in REMOTE_TOOL_NAMES), and the
    authoritative service seam is observed directly.

    Covers the live chain: capture -> injection -> routed client accepts ->
    delegate forwards verbatim -> service receives the complete Fresh set ->
    inferred handler reports 19/19/0 with fresh_item_count >= 1 -> model
    arguments / journal never carry the ids, and a new run starts empty."""

    def _track(self, catalog_id: str, name: str, index: int) -> CatalogTrack:
        return CatalogTrack(
            catalog_id=catalog_id,
            name=f"{name} {'歌' * 40}",
            artist_names=("Jay Chou",),
            album_name=f"Routed Album {index}",
            genres=("Synthetic Pop",),
            isrc=f"USROUT{index:09d}",
            duration_ms=201000,
            release_date="2024-01-15",
            url=None,
            artist_catalog_ids=(f"ROUTED-ARTIST-{index}",),
            album_catalog_id=f"ROUTED-ALBUM-{index}",
        )

    def _routed_service_and_client(
        self, source: TermCatalogSource
    ) -> tuple[_RecordingExecuteService, RoutedAgentClient]:
        service = _RecordingExecuteService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            catalog_search_source=source,
        )
        self.addCleanup(service.close)
        # A dummy remote path: discover + generate are local routing entries,
        # so the socket is never dialed in this test -- itself a pinned fact
        # (Fresh provenance never needs the wire).
        client = RoutedAgentClient(
            AgentClientIdentity(client_id=CLIENT_ID, model_id="test-model", label="tests"),
            service,
            remote_socket_path=Path(self.temporary_directory.name) / "no-run-here.sock",
        )
        return service, client

    def test_fresh_chain_reaches_the_inferred_handler(self) -> None:
        self._seed_two_positives()
        bound_6 = tuple(
            self._track(f"RC-B06-{i:02d}", f"Jay Chou Bound R {i:02d}", 300 + i)
            for i in range(1, 7)
        )
        promoted_19 = tuple(
            self._track(f"RC-P19-{i:02d}", f"Jay Chou Live New R {i:02d}", 400 + i)
            for i in range(1, 20)
        )
        source = TermCatalogSource({
            "seed-bound-6": bound_6,
            "周杰伦": bound_6 + promoted_19,
        })
        service, client = self._routed_service_and_client(source)
        # A plain no-kwarg local call keeps the byte-for-byte base behavior
        # (directive item 11): it also double-checks the seam records an empty
        # internal set rather than a missing one.
        client.call("discover_catalog_tracks", {"term": "seed-bound-6"})
        service.seen_executions.clear()
        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "d1", "discover_catalog_tracks", json.dumps({"term": "周杰伦"})
                ),
            ]),
            tool_response([
                ProviderToolCall(
                    "g1",
                    "generate_inferred_recommendation",
                    json.dumps(
                        {
                            "target_ids": [
                                "trk_11111111-1111-4111-8111-111111111111",
                                "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                            ],
                            "limit": 5,
                            "min_fresh": 5,
                        }
                    ),
                ),
            ]),
            text_response("已推荐。"),
        ])
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        result = loop.run("库外新歌，尽量找新的")
        # The loop completed: no TypeError at the routed-client boundary
        # (the live failure). P20-Fix10: the success scene now ends in the
        # deterministic rendering of the fresh batch.
        self.assertTrue(result.final_text.startswith("为你推荐这 5 首："), result.final_text)
        self.assertEqual(
            result.final_text.count("这是本次新发现，按 Synthetic Pop 方向推断出来。"), 5
        )
        promoted_canonical_ids = self._canonical_ids_for_catalog_ids(
            {f"RC-P19-{i:02d}" for i in range(1, 20)}
        )
        self.assertEqual(len(promoted_canonical_ids), 19)
        # Items 5-6: the discover executed with an empty internal set, the
        # generation with the complete sorted Fresh set -- the routed client
        # forwarded the kwarg verbatim down the delegate path to the service.
        self.assertEqual(
            [(tool, ids) for tool, ids, _ in service.seen_executions],
            [
                ("discover_catalog_tracks", ()),
                (
                    "generate_inferred_recommendation",
                    tuple(sorted(promoted_canonical_ids)),
                ),
            ],
        )
        # Items 7-9: the real inferred handler saw the full set and reports the
        # true funnel -- the same chain the previous live run starved.
        generate_raw = service.seen_executions[-1][2]
        self.assertEqual(generate_raw.outcome.value, "ok")
        self.assertEqual(generate_raw.payload["fresh_promoted_count"], 19)
        self.assertEqual(generate_raw.payload["fresh_candidate_count"], 19)
        self.assertEqual(generate_raw.payload["fresh_negative_rejected_count"], 0)
        self.assertEqual(generate_raw.payload["item_count"], 5)
        self.assertGreaterEqual(generate_raw.payload["fresh_item_count"], 1)
        promoted_set = set(promoted_canonical_ids)
        fresh_targets = {
            item["target_id"]
            for item in generate_raw.payload["items"]
            if item["fresh_this_request"]
        }
        self.assertTrue(fresh_targets)
        self.assertTrue(fresh_targets <= promoted_set)
        # Item 10: the ids are structured out of the model-visible surface --
        # neither the delivered generate envelope nor any journal payload ever
        # carries them.
        generate_envelope = provider.calls[2]["messages"][-1].tool_results[0].content
        self.assertNotIn("fresh_canonical_ids", generate_envelope)
        from music_agent.agent_request_journal_repository import (
            AgentRequestJournalRepository,
        )

        with AgentRequestJournalRepository(self.database_path) as journal:
            for row in journal.list():
                encoded = json.dumps(dict(row.request.payload), ensure_ascii=False)
                self.assertNotIn("fresh_canonical_ids", encoded)

    def test_new_run_starts_with_an_empty_fresh_set(self) -> None:
        self._seed_two_positives()
        promoted_1 = (
            self._track("RC-P01-01", "Jay Chou Live New Solo", 500),
        )
        source = TermCatalogSource({"周杰伦": promoted_1})
        service, client = self._routed_service_and_client(source)
        generate_args = json.dumps(
            {
                "target_ids": [
                    "trk_11111111-1111-4111-8111-111111111111",
                    "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                ],
                "limit": 5,
                "min_fresh": 5,
                # P19-T15: this test is about the per-run fresh-set reset, not
                # recent-run dedup -- repeat the identical pool explicitly.
                "avoid_previous_runs": False,
            }
        )
        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "d1", "discover_catalog_tracks", json.dumps({"term": "周杰伦"})
                ),
            ]),
            tool_response([
                ProviderToolCall("g1", "generate_inferred_recommendation", generate_args),
            ]),
            text_response("已推荐。"),
            tool_response([
                ProviderToolCall("g2", "generate_inferred_recommendation", generate_args),
            ]),
            text_response("已推荐。"),
        ])
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        first = loop.run("库外新歌")
        # Deliberately unclassified test-only line: this test exercises the
        # per-run Fresh transport reset, not the new-recommendation freshness gate.
        second = loop.run("继续检查库外通道")
        self.assertEqual(first.final_text, "已推荐。")
        self.assertEqual(second.final_text, "已推荐。")
        generate_seams = [
            (ids, result)
            for tool, ids, result in service.seen_executions
            if tool == "generate_inferred_recommendation"
        ]
        self.assertEqual(len(generate_seams), 2)
        # Run 1: the captured id traveled; run 2: a genuinely empty set -- the
        # per-run capture state reset (directive item 13).
        self.assertEqual(len(generate_seams[0][0]), 1)
        self.assertEqual(generate_seams[1][0], ())
        self.assertEqual(generate_seams[1][1].payload["fresh_promoted_count"], 0)
        self.assertEqual(generate_seams[1][1].payload["fresh_item_count"], 0)


class DefaultSystemPromptFreshTruthContractTest(unittest.TestCase):
    """P15-S3-S3D: the fresh-truth clause appended to DEFAULT_SYSTEM_PROMPT
    (after the S3-S3B/S3-S3C rules, so every earlier pin stays in place).
    Pins contract text only; asserts nothing about provider behavior."""

    def test_fresh_truth_fields_named(self) -> None:
        self.assertIn("fresh_this_request", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("fresh_item_count", DEFAULT_SYSTEM_PROMPT)

    def test_word_use_rules_present(self) -> None:
        self.assertIn("只有 fresh_this_request=true 的条目才能称「本次新发现/刚找到/新歌」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("只有 fresh_item_count 等于批次条目总数时才能说「这一批全都是本次新发现」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("用户明确要新歌但 fresh_item_count=0 时", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("必须如实说明：本次确实进行了目录搜索", DEFAULT_SYSTEM_PROMPT)

    def test_no_manual_id_comparison(self) -> None:
        self.assertIn("绝不自行比对名称、艺人或 target_ids 推断", DEFAULT_SYSTEM_PROMPT)
        # Known-Catalog items carry the restricted vocabulary, never the fresh one.
        self.assertIn("false 的目录条目只能称已知目录/目录候选/推断候选", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得把已知目录曲目说成新发现", DEFAULT_SYSTEM_PROMPT)

    def test_min_fresh_knob_guidance_present(self) -> None:
        self.assertIn("min_fresh", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("0≤值≤limit", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("普通推荐请求不传 min_fresh", DEFAULT_SYSTEM_PROMPT)


class FreshToolSchemaContractTest(unittest.TestCase):
    """P15-S3-S3D tool schema boundary: min_fresh exists ONLY on the inferred
    generation tool (integer, 0 <= value <= limit via description), and the
    plain tool's schema has no such key -- the model can only ever name the
    count, never the fresh tracks themselves."""

    def test_inferred_schema_carries_integer_min_fresh(self) -> None:
        schema = {
            item.name: item for item in PROVIDER_TOOL_SCHEMAS
        }["generate_inferred_recommendation"]
        properties = schema.input_schema["properties"]
        self.assertIn("min_fresh", properties)
        self.assertEqual(properties["min_fresh"]["type"], "integer")
        self.assertEqual(properties["min_fresh"]["minimum"], 0)

    def test_plain_generate_schema_has_no_min_fresh(self) -> None:
        schema = {
            item.name: item for item in PROVIDER_TOOL_SCHEMAS
        }["generate_recommendation"]
        self.assertNotIn("min_fresh", schema.input_schema["properties"])

    def test_no_schema_names_fresh_ids(self) -> None:
        # Fresh identity is internal: no model-visible key anywhere.
        for item in PROVIDER_TOOL_SCHEMAS:
            self.assertNotIn(
                "fresh_canonical_ids", json.dumps(dict(item.input_schema))
            )
            self.assertNotIn("fresh_promoted_ids", json.dumps(dict(item.input_schema)))


class DefaultSystemPromptDiscoverBudgetContractTest(unittest.TestCase):
    """P15-S3-S3B: the five-rule fresh-discovery prompt contract appended to
    DEFAULT_SYSTEM_PROMPT (after the M3-A batching paragraph, so every earlier
    pin stays in place). These tests pin contract text; they assert nothing
    about how any provider model behaves."""

    def test_known_first_rule_present(self) -> None:
        self.assertIn("普通推荐请求优先复用已见过的已知目录候选", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("query_catalog_discovery_state 的长期记忆", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("空结果诊断中的已知目录供给事实", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要仅因为用户要求推荐就自动发起新的目录搜索", DEFAULT_SYSTEM_PROMPT)

    def test_explicit_fresh_intent_rule_present(self) -> None:
        self.assertIn("「新的」「没听过的」「库外的」「再找一些」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("可以优先用 discover_catalog_tracks 访问 Apple Music Catalog", DEFAULT_SYSTEM_PROMPT)

    def test_already_bound_is_never_new_discovery(self) -> None:
        self.assertIn("状态为 already_bound", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("再次遇到了 ", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("绝不能把它们描述成新发现、新歌或本次首次找到", DEFAULT_SYSTEM_PROMPT)

    def test_fresh_tracks_flow_into_inferred(self) -> None:
        self.assertIn(
            "直接交给 generate_inferred_recommendation（带 min_fresh）继续推荐",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "允许在本轮预算内执行一次 discover_catalog_tracks",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn("一次扩展后仍为空就如实结束，不得继续换词", DEFAULT_SYSTEM_PROMPT)

    def test_budget_exhausted_stop_search_rule_present(self) -> None:
        self.assertIn("目录搜索预算为 1 次成功搜索", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("错误码 discover_budget_exhausted", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("停止换搜索词继续搜索", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("改用已有的已知/推断候选", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("本轮没有找到足够的新结果", DEFAULT_SYSTEM_PROMPT)
        # P20-PerfFix02 failure recovery: a failed search never burns the
        # single budget, and the prompt says it may be retried (cross-round,
        # not confined to the same round) until one success -- one transient
        # Catalog failure must not end the turn's recovery ability (§四-4).
        self.assertIn("失败的搜索不消耗预算", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("可再次重试，直到一次成功", DEFAULT_SYSTEM_PROMPT)

    def test_m3a_batching_instruction_still_present(self) -> None:
        self.assertIn("同一决策步骤需要多个互不依赖的只读查询时", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("核对读必须在变更工具结果返回之后的下一轮进行", DEFAULT_SYSTEM_PROMPT)
        # The fresh-discovery paragraph was appended after M3-A, before the
        # closing line -- neither the batching rule nor the closing moved.
        self.assertLess(
            DEFAULT_SYSTEM_PROMPT.index("同一决策步骤需要多个互不依赖的只读查询时"),
            DEFAULT_SYSTEM_PROMPT.index("目录搜索预算为 1 次成功搜索"),
        )
        self.assertLess(
            DEFAULT_SYSTEM_PROMPT.index("目录搜索预算为 1 次成功搜索"),
            DEFAULT_SYSTEM_PROMPT.index("最终用中文简洁回答用户。"),
        )


class DefaultSystemPromptExplorationFloorContractTest(unittest.TestCase):
    """P15-S3-S3C prompt items 30-34: the exploration-floor paragraph appended
    after the S3B five rules. Ordinary recommendations must NOT force
    exploration; only explicit fresh intent after real discovery may request it;
    min_exploration is never described as a freshness guarantee; the five S3B
    rules and the M3-A batching instruction stay byte-present and ordered."""

    def test_ordinary_recommendation_does_not_force_exploration(self) -> None:
        self.assertIn("探索选择：普通推荐请求不强制混入探索歌曲", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("min_exploration 默认 0", DEFAULT_SYSTEM_PROMPT)

    def test_explicit_fresh_intent_may_request_floor_after_discovery(self) -> None:
        self.assertIn("可以在本轮已完成真实目录发现之后", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("向 generate_inferred_recommendation 传 min_exploration=1", DEFAULT_SYSTEM_PROMPT)

    def test_floor_is_never_described_as_fresh_guarantee(self) -> None:
        self.assertIn("并不是保证本次刚发现的新歌一定入批", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("新歌身份只来自 discover_catalog_tracks", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("already_bound 永不算新", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("如实说明而不是编造", DEFAULT_SYSTEM_PROMPT)

    def test_five_s3b_rules_still_present(self) -> None:
        for text in (
            "普通推荐请求优先复用已见过的已知目录候选",
            "绝不能把它们描述成新发现、新歌或本次首次找到",
            "直接交给 generate_inferred_recommendation（带 min_fresh）继续推荐",
            "目录搜索预算为 1 次成功搜索",
            "下一条用户消息会重新允许目录发现",
        ):
            self.assertIn(text, DEFAULT_SYSTEM_PROMPT)

    def test_m3a_batching_instruction_still_present_and_ordered(self) -> None:
        m3a = DEFAULT_SYSTEM_PROMPT.index("同一决策步骤需要多个互不依赖的只读查询时")
        s3b = DEFAULT_SYSTEM_PROMPT.index("目录搜索预算为 1 次成功搜索")
        s3c = DEFAULT_SYSTEM_PROMPT.index("探索选择：普通推荐请求不强制混入探索歌曲")
        close = DEFAULT_SYSTEM_PROMPT.index("最终用中文简洁回答用户。")
        self.assertLess(m3a, s3b)
        self.assertLess(s3b, s3c)
        self.assertLess(s3c, close)


class ToolArgumentSchemaSafetyTest(unittest.TestCase):
    """P15-S4-M2 (diagnostic): code-level confirmation that the current agent
    tool surface cannot put credentials into a trace -- every tool argument
    property name is checked against the redaction markers, so a future schema
    change introducing a sensitive field fails loudly instead of leaking into
    a trace file."""

    def test_no_tool_argument_property_matches_sensitive_markers(self) -> None:
        from music_agent.provider_instrumentation import _SENSITIVE_ARGUMENT_KEY_MARKERS

        structural = {
            "type", "description", "properties", "required", "items",
            "additionalProperties", "enum", "const", "format", "default",
            "anyOf", "oneOf", "allOf",
        }
        property_names: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key not in structural:
                        property_names.add(key)
                    walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        for schema in PROVIDER_TOOL_SCHEMAS:
            walk(schema.input_schema)
        offenders = [
            name
            for name in property_names
            if any(marker in name.lower() for marker in _SENSITIVE_ARGUMENT_KEY_MARKERS)
        ]
        self.assertEqual(offenders, [])


class DefaultSystemPromptOutputContractTest(unittest.TestCase):
    """P14-R1 (triage #3/#5/#6): the user-facing output contract -- presentation
    order mirrors batch order, internal identifiers/state words/process narration
    are forbidden, and one 换一组 equals one generation with an empty-result
    refusal feedback path."""

    def test_presentation_order_must_match_batch_order(self) -> None:
        self.assertIn("列表顺序必须与工具返回的条目顺序完全一致", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("第 1 个条目即「第一首」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("严禁以任何理由重排", DEFAULT_SYSTEM_PROMPT)

    def test_internal_identifiers_are_forbidden_in_user_text(self) -> None:
        self.assertIn("不要展示 run_id、candidate_id、target_id", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("rcm_/cnd_/fbk_ 开头的编号", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("playback.route 取值标签", DEFAULT_SYSTEM_PROMPT)

    def test_internal_state_words_are_forbidden_in_user_text(self) -> None:
        self.assertIn("active_batch、runs_total、preview_sounding", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("context 取值（agent_selected/own_queue/unknown）", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("一律换成面向用户的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不说 preview_only", DEFAULT_SYSTEM_PROMPT)

    def test_process_narration_is_forbidden(self) -> None:
        self.assertIn("禁止用「我如实告知用户…」「我应该向用户说明…」等句式", DEFAULT_SYSTEM_PROMPT)

    def test_process_narration_exemplars_and_batch_locator_ban(self) -> None:
        self.assertIn("「让我检查…」「我从工具结果得知…」等同类过程句", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要解释批次定位机制", DEFAULT_SYSTEM_PROMPT)

    def test_generation_attempt_cap_is_taught_alongside_retry_once(self) -> None:
        self.assertIn("生成工具每轮至多尝试两次", DEFAULT_SYSTEM_PROMPT)

    def test_new_recommendation_requests_must_generate_a_fresh_batch(self) -> None:
        self.assertIn("用户提出新的推荐请求", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("必须调用生成推荐工具生成新批次并展示新的 items", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("严禁把历史批次复述成新推荐", DEFAULT_SYSTEM_PROMPT)

    def test_generation_failure_terminates_with_labeled_choices(self) -> None:
        # P14-R4.4: on empty-after-retry or budget exhaustion the only sanctioned
        # path is an honest closeout with two labeled choices -- no more tooling.
        self.assertIn("（空结果重试后仍为空，或生成工具预算用尽）即进入收尾", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("停止调用一切推荐相关工具", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得继续读取推荐历史", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("「暂时没有新的可推荐曲目」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("①重听刚才那批", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("必须明确标注「这是刚才那批，不是新的推荐」", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("②换个方向", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("严禁把旧批次包装成新推荐", DEFAULT_SYSTEM_PROMPT)

    def test_swap_generates_at_most_once_with_empty_refusal_feedback(self) -> None:
        self.assertIn("一次「换一组」只调用一次生成推荐工具", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("空结果不会写入推荐历史", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("至多重试一次", DEFAULT_SYSTEM_PROMPT)

    def test_generate_tool_descriptions_document_the_empty_refusal(self) -> None:
        by_name = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}
        for tool_name in ("generate_recommendation", "generate_inferred_recommendation"):
            description = by_name[tool_name].description
            self.assertIn("不写入任何推荐历史", description)
            self.assertIn("一次「换一组」只应调用本工具一次", description)

    def test_generate_tool_descriptions_require_generation_for_new_requests(self) -> None:
        by_name = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}
        for tool_name in ("generate_recommendation", "generate_inferred_recommendation"):
            description = by_name[tool_name].description
            self.assertIn("任何新的推荐请求都应通过本工具生成新批次", description)
            self.assertIn("历史批次仅用于用户明确指代", description)


class DefaultSystemPromptLibraryLookupContractTest(unittest.TestCase):
    """P14-R3.1: named playback resolves library-first, catalog only as fallback."""

    def test_prompt_resolution_order_is_library_first(self) -> None:
        self.assertIn("先用 search_library_tracks 按歌名/关键词查 Music Agent 已知记录", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("只有 provenance.kind=apple_music_library 才能称为用户的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("catalog 记录不得称为用户资料库内容", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("没有可信 Library 命中或结果不确定时才用", DEFAULT_SYSTEM_PROMPT)

    def test_search_library_tracks_schema_carries_the_resolution_order(self) -> None:
        schema = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}["search_library_tracks"]
        self.assertIn("只读、无网络", schema.description)
        self.assertIn("不等同于用户的本地资料库", schema.description)
        self.assertIn("provenance", schema.description)
        self.assertIn("bindings", schema.description)
        self.assertIn("catalog provenance 的记录不得称为用户资料库内容", schema.description)
        self.assertEqual(schema.input_schema["required"], ["term"])
        self.assertIn("term", schema.input_schema["properties"])
        self.assertIn("limit", schema.input_schema["properties"])


class DefaultSystemPromptNonBlockingPreviewContractTest(unittest.TestCase):
    """Third phase B batch 2: preview starts non-blocking, stop_preview stops it, and a
    new preview replaces the sounding one without an explicit stop."""

    def test_preview_is_non_blocking_and_reports_started(self) -> None:
        self.assertIn("试听是非阻塞的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("启动后立即返回 started", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("已开始试听", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要等待播放完成", DEFAULT_SYSTEM_PROMPT)

    def test_stop_preview_handles_the_stop_request(self) -> None:
        self.assertIn("「停止/停/别放了/关掉」", DEFAULT_SYSTEM_PROMPT)
        # C06.5: the stop decision keys on the real preview_sounding truth.
        self.assertIn("先读 get_active_context 的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("preview_sounding（试听是否真实在响的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("调用 stop_preview 停止当前", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("无活动试听时幂等", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不要自动 stop_preview", DEFAULT_SYSTEM_PROMPT)

    def test_switching_preview_stops_the_previous_automatically(self) -> None:
        self.assertIn("新试听会自动停止上一首", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("无需先调用 stop_preview", DEFAULT_SYSTEM_PROMPT)

    def test_preview_and_stop_schemas_document_non_blocking_semantics(self) -> None:
        by_name = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}
        self.assertIn("stop_preview", by_name)
        stop_schema = by_name["stop_preview"]
        self.assertIn("永远不会触碰 Music.app", stop_schema.description)
        self.assertIn("幂等成功", stop_schema.description)
        self.assertIsInstance(stop_schema.input_schema.get("properties"), dict)
        preview_schema = by_name["preview_catalog_track"]
        self.assertIn("启动后立即返回 started", preview_schema.description)
        self.assertIn("自动停止上一首试听", preview_schema.description)


class RecommendationReplyDoorContractTest(unittest.TestCase):
    """P19-T14-B: the reply-door primitives -- generation-success detection,
    the numbered pseudo-list shape probe, and the ONE-sentence fallback. The
    web shell enforces the door with these; the prompt clause teaches the
    same rule so the provider mostly never trips it."""

    def test_generation_success_requires_a_generation_tool_with_ok(self) -> None:
        ok = ProviderLoopToolExecution("generate_recommendation", "ok")
        self.assertTrue(generation_succeeded((ok,)))
        inferred_ok = ProviderLoopToolExecution(
            "generate_inferred_recommendation", "ok"
        )
        self.assertTrue(generation_succeeded((inferred_ok,)))
        failed = ProviderLoopToolExecution("generate_recommendation", "failed")
        self.assertFalse(generation_succeeded((failed,)))
        unrelated = ProviderLoopToolExecution("discover_catalog_tracks", "ok")
        self.assertFalse(generation_succeeded((unrelated,)))
        self.assertFalse(generation_succeeded(()))

    def test_public_alias_is_the_same_tool_name_fact(self) -> None:
        self.assertIs(GENERATION_TOOL_NAMES, _GENERATION_TOOL_NAMES)
        self.assertIn("generate_recommendation", GENERATION_TOOL_NAMES)
        self.assertIn("generate_inferred_recommendation", GENERATION_TOOL_NAMES)

    def test_numbered_song_list_probe(self) -> None:
        pseudo = (
            "1. HAPPY BIRTHDAY — back number：活泼的摇滚情歌。\n"
            "2. 高嶺の花子さん — back number：……"
        )
        self.assertTrue(looks_like_numbered_song_list(pseudo))
        # alternate list markers count the same
        self.assertTrue(
            looks_like_numbered_song_list("1、夜に駆ける — YOASOBI\n2) 群青 — YOASOBI")
        )
        # one numbered line is not a list
        self.assertFalse(looks_like_numbered_song_list("1. 晚安。"))
        # two numbered lines without any song separator are not a song list
        self.assertFalse(looks_like_numbered_song_list("1. 第一步\n2. 第二步"))
        # plain prose and non-strings never match
        self.assertFalse(looks_like_numbered_song_list("好的，我记住了。"))
        self.assertFalse(looks_like_numbered_song_list(None))
        self.assertFalse(looks_like_numbered_song_list(42))

    def test_fallback_is_one_short_honest_sentence(self) -> None:
        self.assertEqual(
            RECOMMENDATION_UNFULFILLED_FALLBACK,
            "暂时没有找到合适的推荐，换个方向或换一首歌再试试吧。",
        )
        self.assertLessEqual(len(RECOMMENDATION_UNFULFILLED_FALLBACK), 40)
        self.assertNotEqual(RECOMMENDATION_UNFULFILLED_FALLBACK, _GENERATION_FAILURE_CLOSEOUT)

    def test_prompt_teaches_the_similar_to_current_seed_flow(self) -> None:
        # The teaching layer names the T14-B live wording explicitly and the
        # no-batch cap: one short sentence, never a hand-enumerated list.
        self.assertIn("找类似这首的", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("同样必须调用生成推荐工具", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("最终回答只能是简短一句话", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("严禁把曲目逐一写成「1. 2. 3.」编号罗列来充数", DEFAULT_SYSTEM_PROMPT)


class T14EPlayPreviewGuardTest(unittest.TestCase):
    """P19-T14-E: the pure Play-vs-Preview guard facts.

    These are the decision helpers the caller-side doors (web shell / CLI)
    pair with a stop_preview call: ``preview_path_started`` (audio really
    started this run), ``formal_play_started`` (formal playback executed ok)
    and their composition ``play_intent_preview_downgrade`` (an explicit play
    request degraded to preview). All read the public execution record --
    name + outcome strings, nothing else.
    """

    def test_preview_path_started_reads_preview_ok_only(self) -> None:
        self.assertTrue(
            preview_path_started(
                (ProviderLoopToolExecution("preview_catalog_track", "ok"),)
            )
        )
        self.assertTrue(
            preview_path_started((ProviderLoopToolExecution("preview_batch", "ok"),))
        )
        self.assertFalse(
            preview_path_started(
                (ProviderLoopToolExecution("preview_catalog_track", "error"),)
            )
        )
        self.assertFalse(
            preview_path_started(
                (ProviderLoopToolExecution("play_track", "ok"),)
            )
        )
        self.assertFalse(preview_path_started(()))

    def test_structured_preview_failure_overrides_legacy_ok_trace(self) -> None:
        attempt = mark_action_executing(
            create_direct_action_attempt(
                "trk_11111111-1111-4111-8111-111111111111",
                route="preview_only",
            )
        )
        attempt = record_action_execution(
            attempt, outcome="ok", preview_started=False
        )
        result = SimpleNamespace(
            action_attempt=attempt,
            tool_executions=(
                ProviderLoopToolExecution("preview_catalog_track", "ok"),
            ),
        )

        self.assertFalse(action_result_preview_started(result))
        self.assertFalse(
            action_result_play_preview_downgrade("播放 Hanataba", result)
        )

    def test_formal_play_started_reads_the_formal_family(self) -> None:
        self.assertTrue(
            formal_play_started((ProviderLoopToolExecution("play_track", "ok"),))
        )
        # play (resume) is formal; a failed selection is not
        self.assertTrue(formal_play_started((ProviderLoopToolExecution("play", "ok"),)))
        self.assertFalse(
            formal_play_started((ProviderLoopToolExecution("play_track", "error"),))
        )
        self.assertFalse(formal_play_started(()))

    def test_named_play_degraded_to_preview_is_detected(self) -> None:
        # THE live failure shape: 播放 Hanataba -> play_track failed -> the
        # model self-healed into a 30s preview.
        self.assertTrue(
            play_intent_preview_downgrade(
                "播放 Hanataba",
                (
                    ProviderLoopToolExecution("play_track", "error"),
                    ProviderLoopToolExecution("preview_catalog_track", "ok"),
                ),
            )
        )
        # 播放第2首 with a preview_only item previewed: same degradation.
        self.assertTrue(
            play_intent_preview_downgrade(
                "播放第2首",
                (ProviderLoopToolExecution("preview_batch", "ok"),),
            )
        )
        # bare 播放 answered with a preview: same degradation.
        self.assertTrue(
            play_intent_preview_downgrade(
                "播放", (ProviderLoopToolExecution("preview_catalog_track", "ok"),)
            )
        )

    def test_formal_success_or_preview_failure_never_flags(self) -> None:
        # Formal playback succeeded -- the door may still stop a stray preview
        # but must never swap the reply.
        self.assertFalse(
            play_intent_preview_downgrade(
                "播放 Hanataba",
                (
                    ProviderLoopToolExecution("play_track", "ok"),
                    ProviderLoopToolExecution("preview_catalog_track", "ok"),
                ),
            )
        )
        # bare 播放 resumed the current track -- formal, no degradation.
        self.assertFalse(
            play_intent_preview_downgrade(
                "播放",
                (
                    ProviderLoopToolExecution("play", "ok"),
                    ProviderLoopToolExecution("preview_catalog_track", "ok"),
                ),
            )
        )
        # a failed preview attempt started no audio: nothing to guard.
        self.assertFalse(
            play_intent_preview_downgrade(
                "播放 Hanataba",
                (ProviderLoopToolExecution("preview_catalog_track", "error"),),
            )
        )

    def test_preview_and_non_play_intents_are_never_flagged(self) -> None:
        # Explicit preview requests stay the preview family's own surface.
        for text in ("试听 Hanataba", "试听这首", "试听第2首", "都放一遍"):
            with self.subTest(text=text):
                self.assertFalse(
                    play_intent_preview_downgrade(
                        text,
                        (ProviderLoopToolExecution("preview_catalog_track", "ok"),),
                    )
                )
        # The delegation family (随便播放一首 = play-OR-preview authorized)
        # and plain prose never read as a degradation.
        for text in ("随便播放一首", "放首歌", "推荐几首歌", "好的"):
            with self.subTest(text=text):
                self.assertFalse(
                    play_intent_preview_downgrade(
                        text,
                        (ProviderLoopToolExecution("preview_catalog_track", "ok"),),
                    )
                )

    def test_fallback_is_exactly_the_contract_sentence(self) -> None:
        self.assertEqual(
            PLAY_PREVIEW_DOWNGRADE_FALLBACK,
            "这首目前无法正式播放，可以试听 30 秒。",
        )

    def test_prompt_forbids_the_silent_downgrade(self) -> None:
        # The teaching layer states the no-downgrade contract in the exact
        # honest sentence and the 播放-vs-试听 distinction.
        self.assertIn("播放意图绝不自动降级成试听", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("这首目前无法正式播放，可以试听 30 秒。", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("绝不产生试听", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("用户要求播放它时", DEFAULT_SYSTEM_PROMPT)

    def test_prompt_keeps_pronoun_resolution_out_of_the_provider(self) -> None:
        # P19-T14-F-R2/R4 hardening for the fall-through corners (session
        # running / offline / other hosts): whenever a pronoun turn reaches
        # the provider, the model must never resolve it through
        # recommendation or discovery tools and must ask honestly instead of
        # drifting into recommendation prose -- the live-failure shape
        # (「暂时没有找到合适的推荐…」) is unreachable by contract.
        # R4 corrects the claimed binding source: the session-local
        # referent_canonical_id (survives stops) first, the channel action
        # log only as its legacy fallback.
        self.assertIn("代词的解析与目标绑定由外层快捷通路负责", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("referent_canonical_id", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("回看 channel.canonical_id 动作日志", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("模型不得用推荐或目录发现工具解析代词含义", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得为代词编造曲目", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("如实反问「想试听哪首歌？」", DEFAULT_SYSTEM_PROMPT)


class S1PlainChatZeroToolTest(ProviderAgentLoopTest):
    """S1: plain chat runs with ZERO tool schemas; everything else keeps all of them.

    The gate is the closed plain-chat classifier in ``intent_router``; the loop
    must consult it once per user message and never strip a tool a real music
    request needs. ``FakeProvider`` records the exact per-round tool sequence,
    so these tests observe what the provider would have received.
    """

    def test_plain_chat_greeting_gets_zero_tools(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("你好！有什么可以帮你听的？")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("你好")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.final_text, "你好！有什么可以帮你听的？")
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(result.tool_executions, ())

    def test_thanks_and_farewell_get_zero_tools(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [text_response("不客气。"), text_response("再见，随时来找我推荐音乐。")]
        )
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        self.assertEqual(loop.run("谢谢").final_text, "不客气。")
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(loop.run("再见").final_text, "再见，随时来找我推荐音乐。")
        self.assertEqual(provider.calls[1]["tools"], [])

    def test_recommendation_request_runs_the_recommendation_group(self) -> None:
        # S3 builds on S1: a classified music request no longer carries the
        # full 31 schemas -- it runs its task group (here the recommendation
        # chain, the projection of _RECOMMENDATION_TOOL_NAMES). The 31-tool
        # registry still exists and still ships on every unclassified line;
        # S1's zero-tool plain-chat behavior is unchanged.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        loop.run("推荐几首歌")
        self.assertEqual(provider.calls[0]["tools"], project_tools(_RECOMMENDATION_TOOL_NAMES))

    def test_unlisted_chitchat_and_consent_keep_tools_fail_safe(self) -> None:
        # A false positive here would rob a real music request of its tools,
        # so every unlisted line -- including chitchat outside the closed set
        # and the consent family (好的 answers 要试听吗? and must stay able to
        # execute) -- keeps the full 30+ schemas.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [text_response("还行。"), text_response("好嘞。"), text_response("你猜。")]
        )
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        for text in ("好的", "天气不错啊", "你好，帮我推荐几首歌"):
            loop.run(text)
        for call in provider.calls:
            self.assertEqual(call["tools"], list(PROVIDER_TOOL_SCHEMAS))

    def test_zero_tool_round_measures_zero_schemas_when_instrumented(self) -> None:
        # The S1 acceptance metric comes straight from the same measurement
        # the --trace file reports: a plain-chat round must count 0 schemas
        # and 0 schema chars out of the input.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("我在。")])
        loop = ProviderAgentLoop(
            provider,
            self._client(service),
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("在吗")
        self.assertIsNotNone(result.trace)
        round_measure = result.trace.rounds[0]
        self.assertEqual(round_measure.tool_schemas_count, 0)
        self.assertEqual(round_measure.tool_schemas_chars, 0)
        # S4: the plain-chat round also "pays" only the BASE prompt, not the
        # full 8,766-char prompt -- the same measurement the --trace file
        # reports, pinned against the composed per-task constants.
        self.assertEqual(round_measure.input_chars, len(_S4_BASE_PROMPT + FINAL_ANSWER_CONTRACT) + 2)
        self.assertLess(len(_S4_BASE_PROMPT), len(DEFAULT_SYSTEM_PROMPT) // 2)
        # A tooled round on the same loop still measures the task group (S3:
        # recommendation rounds carry the recommendation chain, not the
        # full registry) AND the recommendation prompt (S4).
        provider.responses.append(text_response("好的。"))
        result = loop.run("推荐几首歌")
        self.assertEqual(
            result.trace.rounds[0].tool_schemas_count, len(_RECOMMENDATION_TOOL_NAMES)
        )
        # S5: the recommendation round pre-carries the three prefetch reads
        # (synthetic round-0 assistant tool-calls + user tool-results), so
        # input_chars = prompt + all message payloads + schemas. Recomputing
        # from the exact messages FakeProvider recorded proves the trace
        # mirrors what was actually sent -- an exact literal pin would have
        # to duplicate the temp-fixture service payloads.
        recomputed = measure_round_input(
            provider.calls[1]["system"],
            provider.calls[1]["messages"],
            provider.calls[1]["tools"],
        )
        self.assertEqual(result.trace.rounds[0].messages_count, 3)
        self.assertEqual(result.trace.rounds[0].input_chars, recomputed.input_chars)
        self.assertEqual(
            result.trace.rounds[0].tool_schemas_chars, recomputed.tool_schemas_chars
        )
        self.assertGreater(
            recomputed.input_chars,
            len(_S4_RECOMMENDATION_PROMPT) + len("推荐几首歌")
            + recomputed.tool_schemas_chars,
        )

    def test_deduped_plain_chat_yields_final_answer_without_tools(self) -> None:
        # The zero-tool decision is made once per user message: punctuated
        # greetings share the normalized classifier and never pay schemas.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("嗨！")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("你好！")
        self.assertEqual(result.final_text, "嗨！")
        self.assertEqual(provider.calls[0]["tools"], [])


class S2FinalAnswerRoundTest(ProviderAgentLoopTest):
    """S2: after a provably-terminal round every remaining round runs with
    ZERO tool schemas -- and nowhere else.

    The switch is strictly conserving: contract-pinned terminal actions and an
    ordinary successful generation shrink to zero. Explicit delegation instead
    gets one payload-bound play/preview action and the required playback readback.
    Reads, failures, discovery and failed generation keep their selected surface.
    ``FakeProvider``
    records the exact per-round tool sequence; the instrumented trace meters
    the acceptance metric (tool_schemas_count)."""

    def _call(
        self, name: str, arguments: Mapping[str, object] | None = None
    ) -> ProviderToolCall:
        return ProviderToolCall(
            call_id=f"call_{name}",
            name=name,
            arguments=json.dumps(arguments or {}),
        )

    def _loop(
        self,
        responses: list[ProviderResponse],
        results: list[AgentToolResult] | None = None,
        *,
        instrument: bool = False,
    ) -> tuple[ProviderAgentLoop, FakeProvider]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(responses)
        client = (
            RecordingClient(service, results=results)
            if results is not None
            else self._client(service)
        )
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=instrument),
        )
        return loop, provider

    def test_plain_chat_still_zero_tools_after_s2(self) -> None:
        # S1 behavior must persist unchanged under the per-round S2 selection:
        # a plain-chat line never pays a schema, on any round.
        loop, provider = self._loop([text_response("你好！有什么可以帮你听的？")])
        result = loop.run("你好")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(provider.calls[0]["tools"], [])

    def test_collection_preference_statement_is_acknowledged_without_provider_or_generation(self) -> None:
        loop, provider = self._loop(
            [text_response("provider should not be called")],
            results=[],
        )

        result = loop.run("我喜欢 Yorushika 的歌")

        self.assertEqual(result.final_text, "明白，你喜欢 Yorushika 的歌。")
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.tool_executions, ())
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(loop.client.recorded, [])
        self.assertIsNone(result.recommendation_payload)

    def test_current_track_feedback_forms_own_the_full_lifecycle(self) -> None:
        cases = {
            "我喜欢这首歌": "liked",
            "我不喜欢这首歌": "disliked",
            "我喜欢当前正在播放的这首歌": "liked",
            "我不喜欢当前正在播放的这首歌": "disliked",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                adapter = DelegatedPlaybackAdapter()
                adapter.now_pid = "SYNTH-TRACK-001"
                service = SharedAgentService(
                    self.database_path,
                    clients=AgentClientRegistry(
                        {CLIENT_ID: AgentClientPolicy.FULL}
                    ),
                    playback_adapter=adapter,
                )
                self.addCleanup(service.close)
                provider = FakeProvider([])
                loop = ProviderAgentLoop(
                    provider, self._client(service), PROVIDER_TOOL_SCHEMAS
                )

                result = loop.run(text)

                self.assertEqual(result.final_text, "好的，已记录你的反馈。")
                self.assertEqual(result.rounds, 0)
                self.assertEqual(len(provider.calls), 0)
                self.assertEqual(
                    [execution.name for execution in result.tool_executions],
                    [
                        "get_active_context",
                        "record_feedback",
                        "interpret_feedback",
                        "apply_learning",
                    ],
                )
                observations = self._client(service).call(
                    "list_feedback_observations", {}
                ).payload["observations"]
                latest = json.loads(observations[-1])
                self.assertEqual(latest["kind"], kind)
                self.assertEqual(
                    latest["target"]["target_id"],
                    "trk_11111111-1111-4111-8111-111111111111",
                )

    def test_current_track_feedback_ignores_own_queue_provenance(self) -> None:
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

        # Seed history directly: the test needs only a truthful prior Agent
        # recommendation that does NOT contain the current player track. Going
        # through generation would add unrelated preference/ranking preconditions.
        produced_at = datetime.now(timezone.utc)
        historical = RecommendationResult(
            run_id=generate_run_id(),
            request=RecommendationRequest(
                context=RecommendationContext(produced_at, ()),
                recommended_kind=RecommendedItemKind.TRACK,
                limit=1,
            ),
            items=(
                RecommendationItem(
                    candidate=Candidate(
                        candidate_id="cnd_99999999-9999-4999-8999-999999999999",
                        target=PreferenceTargetReference(
                            PreferenceTargetKind.TRACK,
                            "trk_22222222-2222-4222-8222-222222222222",
                        ),
                        source=CandidateSourceReference(
                            "test_seed", "own_queue_context"
                        ),
                        eligibility=Eligibility.ELIGIBLE,
                    ),
                    score=ScoreBreakdown(
                        0.9, (ScoreComponent("test", 0.9),)
                    ),
                ),
            ),
            produced_at=produced_at,
            contract_version=RECOMMENDATION_CONTRACT_VERSION,
        )
        with RecommendationHistoryRepository(self.database_path) as history:
            history.save_result(historical)

        adapter = DelegatedPlaybackAdapter()
        adapter.now_pid = "SYNTH-TRACK-001"
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
        )
        self.addCleanup(service.close)
        client = self._client(service)
        before = client.call("get_active_context", {}).payload
        self.assertEqual(before["context"], "own_queue")
        self.assertIsNone(before["referent_canonical_id"])

        provider = FakeProvider([])
        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("我喜欢这首歌")

        self.assertEqual(result.final_text, "好的，已记录你的反馈。")
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            [
                "get_active_context",
                "record_feedback",
                "interpret_feedback",
                "apply_learning",
            ],
        )

    def test_unresolved_current_track_feedback_clarifies_without_writing(self) -> None:
        adapter = DelegatedPlaybackAdapter()
        adapter.now_pid = "UNBOUND-AND-UNRESOLVED"
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
        )
        self.addCleanup(service.close)
        client = self._client(service)
        provider = FakeProvider([])

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("我不喜欢当前正在播放的这首歌")

        self.assertIn("无法唯一确认", result.final_text)
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["get_active_context"],
        )
        observations = client.call("list_feedback_observations", {}).payload
        self.assertEqual(observations["count"], 0)

    def test_feedback_record_success_owns_interpret_apply_and_terminal_reply(self) -> None:
        feedback_id = "fbk_11111111-1111-4111-8111-111111111111"
        record = self._call(
            "record_feedback",
            {
                "kind": "liked",
                "source_system": "recommendation_ui",
                "source_path": "chat",
                "target_id": "trk_11111111-1111-4111-8111-111111111111",
            },
        )
        # A stale model-planned apply in the SAME assistant tool message must
        # not create a second learning path. The runtime injects its own exact
        # interpret -> apply tail after the one durable observation succeeds.
        stale_apply = self._call("apply_learning", {"feedback_id": feedback_id})
        loop, provider = self._loop(
            [
                tool_response([record, stale_apply]),
                text_response("provider final should not be needed"),
            ],
            results=[
                ok_tool_result(
                    "record_feedback",
                    {"feedback_id": feedback_id, "kind": "liked"},
                ),
                ok_tool_result(
                    "interpret_feedback",
                    {
                        "feedback_id": feedback_id,
                        "direction": "positive",
                        "reason": "explicit_statement",
                        "explicitness": "explicit",
                    },
                ),
                ok_tool_result(
                    "apply_learning",
                    {"applied": True, "feedback_id": feedback_id},
                ),
            ],
        )

        result = loop.run("我喜欢第二首")

        self.assertEqual(result.final_text, "好的，已记录你的反馈。")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["record_feedback", "interpret_feedback", "apply_learning"],
        )
        self.assertEqual(
            [execution.origin for execution in result.tool_executions],
            ["provider_requested", "policy_injected", "policy_injected"],
        )
        self.assertEqual(
            loop.client.recorded,
            ["record_feedback", "interpret_feedback", "apply_learning"],
        )

    def test_feedback_learning_failure_reports_partial_success_without_more_provider_rounds(self) -> None:
        feedback_id = "fbk_22222222-2222-4222-8222-222222222222"
        loop, provider = self._loop(
            [
                tool_response(
                    [
                        self._call(
                            "record_feedback",
                            {
                                "kind": "disliked",
                                "source_system": "recommendation_ui",
                                "source_path": "chat",
                                "target_id": "trk_11111111-1111-4111-8111-111111111111",
                            },
                        )
                    ]
                ),
                text_response("provider final should not be needed"),
            ],
            results=[
                ok_tool_result(
                    "record_feedback",
                    {"feedback_id": feedback_id, "kind": "disliked"},
                ),
                AgentToolResult(
                    request_id="req_55555555-5555-4555-8555-555555555555",
                    tool="interpret_feedback",
                    outcome=AgentToolOutcome.EXECUTION_ERROR,
                    payload=None,
                    error_code="feedback_learning_error",
                    error_message="synthetic failure",
                    completed_at=datetime(
                        2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc
                    ),
                    replayed=False,
                ),
            ],
        )

        result = loop.run("我不喜欢第二首")

        self.assertEqual(result.final_text, "你的反馈已经记录，但偏好更新暂未完成。")
        self.assertFalse(result.rounds_capped)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            loop.client.recorded,
            ["record_feedback", "interpret_feedback"],
        )

    def test_play_flow_keeps_tools_on_every_round(self) -> None:
        # The prompt REQUIRES the get_now_playing check-back after every play
        # action, so play_track-ok is provably NOT a terminal state: the
        # readback round AND the final text round keep the task group (S3:
        # the play chain -- search/play/check-back/batch reads -- on every
        # round; S2 does not drop it).
        loop, provider = self._loop(
            [
                tool_response([self._call("play_track")]),
                tool_response([self._call("get_now_playing")]),
                text_response("已在播放〈夜曲〉。"),
            ],
            results=[ok_read_result("play_track"), ok_read_result("get_now_playing")],
        )
        result = loop.run("播放这首")
        self.assertEqual(result.rounds, 3)
        self.assertIn("夜曲", result.final_text)
        for call in provider.calls:
            self.assertEqual(call["tools"], project_tools(_PLAYBACK_TOOL_NAMES))

    def test_same_turn_generated_library_item_is_selected_and_verified_in_code(self) -> None:
        selected_id = "trk_11111111-1111-4111-8111-111111111111"
        injected_id = "trk_99999999-9999-4999-8999-999999999999"
        loop, provider = self._loop(
            [
                tool_response([self._call("generate_recommendation")]),
                # Never consumed: generation success immediately enters the
                # code-owned SelectionGrant workflow. The Provider cannot
                # substitute either a different canonical id or Preview route.
                tool_response(
                    [
                        self._call(
                            "preview_catalog_track",
                            {"canonical_id": injected_id},
                        )
                    ]
                ),
            ],
            results=[
                ok_generation_batch("generate_recommendation"),
                ok_read_result("play_track"),
                ok_delegated_play_readback(selected_id),
            ],
        )
        result = loop.run("你来决定")
        self.assertEqual(result.rounds, 1)
        self.assertEqual(
            result.final_text,
            "正在播放《Synthetic》— Synthetic Artist。",
        )
        self.assertEqual(provider.calls[0]["tools"], list(PROVIDER_TOOL_SCHEMAS))
        self.assertEqual(
            loop.client.recorded,
            ["generate_recommendation", "play_track", "get_now_playing"],
        )
        self.assertEqual(
            loop.client.recorded_payloads[1],
            {"canonical_id": selected_id},
        )
        self.assertNotIn(injected_id, str(loop.client.recorded_payloads))
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNotNone(result.action_attempt)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(result.action_attempt.selected_canonical_id, selected_id)
        self.assertEqual(result.action_attempt.expected_action, "play_track")
        self.assertEqual(
            result.action_attempt.readback_state,
            ActionReadbackState.VERIFIED,
        )
        verified = loop.client.service.verified_selections_for_run(
            "rcm_11111111-1111-4111-8111-111111111111"
        )
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].canonical_id, selected_id)
        self.assertEqual(verified[0].item_position, 1)
        self.assertEqual(verified[0].action_kind, "play_track")
        self.assertEqual(verified[0].playback_route, "library")

    def test_same_turn_generated_playback_mismatch_fails_closed(self) -> None:
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        other_id = "trk_22222222-2222-4222-8222-222222222222"
        loop, provider = self._loop(
            [tool_response([self._call("generate_recommendation")])],
            results=[
                ok_generation_batch("generate_recommendation"),
                ok_read_result("play_track"),
                ok_tool_result(
                    "get_now_playing",
                    {
                        "now_playing": {
                            "state": "playing",
                            "name": "Different Song",
                            "artist": "Different Artist",
                        },
                        "context": "agent_selected",
                        "agent_channel": {
                            "state": "library",
                            "canonical_id": target_id,
                        },
                        "player_canonical_id": other_id,
                        "canonical_resolution": "binding",
                    },
                ),
            ],
        )

        result = loop.run("你来决定")

        self.assertEqual(
            result.final_text,
            "播放指令已发出，但暂时无法确认当前播放状态。",
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(
            result.action_attempt.failure_reason,
            "player_canonical_mismatch",
        )
        self.assertEqual(
            loop.client.service.verified_selections_for_run(
                "rcm_11111111-1111-4111-8111-111111111111"
            ),
            (),
        )

    def test_same_turn_generated_playback_unresolved_canonical_fails_closed(self) -> None:
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        loop, provider = self._loop(
            [tool_response([self._call("generate_recommendation")])],
            results=[
                ok_generation_batch("generate_recommendation"),
                ok_read_result("play_track"),
                ok_tool_result(
                    "get_now_playing",
                    {
                        "now_playing": {
                            "state": "playing",
                            "name": "Synthetic",
                            "artist": "Synthetic Artist",
                        },
                        "context": "agent_selected",
                        "agent_channel": {
                            "state": "library",
                            "canonical_id": target_id,
                        },
                        "player_canonical_id": None,
                        "canonical_resolution": None,
                    },
                ),
            ],
        )

        result = loop.run("你来决定")

        self.assertEqual(
            result.final_text,
            "播放指令已发出，但暂时无法确认当前播放状态。",
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(
            result.action_attempt.failure_reason,
            "player_canonical_unresolved",
        )
        self.assertEqual(
            loop.client.service.verified_selections_for_run(
                "rcm_11111111-1111-4111-8111-111111111111"
            ),
            (),
        )

    def test_delegated_preview_does_not_claim_success_without_started_true(self) -> None:
        target_id = "trk_33333333-3333-4333-8333-333333333333"
        loop, provider = self._loop(
            [tool_response([self._call("generate_recommendation")])],
            results=[
                ok_generation_evidence_batch(
                    "generate_recommendation",
                    items=[
                        evidence_item(
                            "Albumless Study",
                            artist_name="Artist Alpha",
                            route="preview_only",
                            target_id=target_id,
                        )
                    ],
                ),
                ok_tool_result(
                    "preview_catalog_track",
                    {"canonical_id": target_id, "started": False},
                ),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(result.final_text, "这首暂时无法试听。")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(result.action_attempt.failure_reason, "preview_not_started")
        self.assertEqual(
            loop.client.service.verified_selections_for_run(
                "rcm_11111111-1111-4111-8111-111111111111"
            ),
            (),
        )

    def test_same_turn_generated_preview_route_is_code_owned_and_single_shot(self) -> None:
        target_id = "trk_33333333-3333-4333-8333-333333333333"
        loop, provider = self._loop(
            [
                tool_response([self._call("generate_recommendation")]),
                # Never consumed: a Provider-proposed formal-play route cannot
                # replace the generated item's authoritative preview_only route.
                tool_response(
                    [self._call("play_track", {"canonical_id": target_id})]
                ),
            ],
            results=[
                ok_generation_evidence_batch(
                    "generate_recommendation",
                    items=[
                        evidence_item(
                            "Albumless Study",
                            artist_name="Artist Alpha",
                            route="preview_only",
                            target_id=target_id,
                        )
                    ],
                ),
                ok_preview_started(target_id),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(
            result.final_text,
            "正在试听《Albumless Study》— Artist Alpha，约 30 秒。",
        )
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            ["generate_recommendation", "preview_catalog_track"],
        )
        self.assertEqual(
            loop.client.recorded_payloads[1],
            {"canonical_id": target_id},
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(result.action_attempt.expected_route, "preview_only")
        self.assertEqual(
            [entry for entry in loop.client.recorded if entry in {
                "play_track", "preview_catalog_track"
            }],
            ["preview_catalog_track"],
        )
        verified = loop.client.service.verified_selections_for_run(
            "rcm_11111111-1111-4111-8111-111111111111"
        )
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].canonical_id, target_id)
        self.assertEqual(verified[0].action_kind, "preview_catalog_track")
        self.assertEqual(verified[0].playback_route, "preview_only")

    def test_llm_current_track_similarity_with_only_referenced_item_clarifies_before_workflow(self) -> None:
        interpreter_payload = {
            "intent": "recommendation",
            "recommendation": {
                "mode": "similarity",
                "requested_count": 5,
                "scene": None,
                "seed": {"kind": "current_track", "value": None},
            },
            "action": None,
            "requires_clarification": False,
            "reason": None,
        }
        loop, provider = self._loop(
            [text_response(json.dumps(interpreter_payload, ensure_ascii=False))],
            results=[
                ok_tool_result(
                    "get_active_context",
                    {
                        "active_batch": None,
                        "player": {"state": "stopped"},
                        "referent_canonical_id": "trk_11111111-1111-4111-8111-111111111111",
                        "preview_sounding": False,
                    },
                ),
            ],
        )
        provider.supports_turn_interpreter = True

        result = loop.run("我想听类似这首的歌")

        self.assertEqual(result.final_text, _TURN_CLARIFICATION_CLOSEOUT)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.tool_executions, ())
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(loop.client.recorded, ["get_active_context"])
        self.assertNotIn("generate_recommendation", loop.client.recorded)
        self.assertNotIn("generate_inferred_recommendation", loop.client.recorded)
        self.assertNotIn("discover_catalog_tracks", loop.client.recorded)

    def test_llm_interpreted_delegation_still_enters_code_owned_selection(self) -> None:
        run_id = "rcm_44444444-4444-4444-8444-444444444444"
        target_id = "trk_33333333-3333-4333-8333-333333333333"
        interpreter_payload = {
            "intent": "delegated_selection",
            "recommendation": None,
            "action": {
                "source": "active_recommendation",
                "selection_mode": "agent_choose_one",
            },
            "requires_clarification": False,
            "reason": None,
        }
        loop, provider = self._loop(
            [
                text_response(json.dumps(interpreter_payload, ensure_ascii=False)),
                tool_response([self._call("get_active_context")]),
            ],
            results=[
                # Minimal interpreter context read. Only high-level facts are
                # exposed to the model; the run id never enters its prompt.
                ok_tool_result(
                    "get_active_context",
                    {
                        "active_batch": {"run_id": run_id, "item_count": 1},
                        "player": None,
                        "referent_canonical_id": None,
                        "preview_sounding": False,
                    },
                ),
                # Existing workflow then re-reads the authoritative context.
                ok_tool_result(
                    "get_active_context",
                    {"active_batch": {"run_id": run_id, "item_count": 1}},
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "position": 1,
                                "target_id": target_id,
                                "name": "Albumless Study",
                                "artist_name": "Artist Alpha",
                                "playback": {"route": "preview_only"},
                            }
                        ],
                    },
                ),
                ok_tool_result(
                    "preview_catalog_track",
                    {"canonical_id": target_id, "started": True},
                ),
            ],
        )
        provider.supports_turn_interpreter = True

        result = loop.run("随便来一首")

        self.assertEqual(
            result.final_text,
            "正在试听《Albumless Study》— Artist Alpha，约 30 秒。",
        )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0]["tools"], [])
        interpreter_request = json.loads(provider.calls[0]["messages"][0].text)
        self.assertNotIn("run_id", json.dumps(interpreter_request, ensure_ascii=False))
        self.assertNotIn("canonical_id", json.dumps(interpreter_request, ensure_ascii=False))
        self.assertEqual(
            loop.client.recorded,
            [
                "get_active_context",
                "get_active_context",
                "get_recommendation_run",
                "preview_catalog_track",
            ],
        )
        self.assertEqual(loop.client.recorded_payloads[-1], {"canonical_id": target_id})
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)

    def test_delegation_from_existing_batch_preview_finishes_without_provider_final(self) -> None:
        # Mirrors the Owner UAT shape: active context -> exact run read
        # -> one code-selected preview from the already-existing current batch.
        # The provider supplies no target or route after the authoritative run
        # read; code owns both and finishes without another provider round.
        run_id = "rcm_33333333-3333-4333-8333-333333333333"
        target_id = "trk_33333333-3333-4333-8333-333333333333"
        loop, provider = self._loop(
            [
                tool_response([self._call("get_active_context")]),
                # Must remain unused: code acts immediately after the exact
                # active run id becomes authoritative.
                text_response("provider final should not be needed"),
            ],
            results=[
                ok_tool_result(
                    "get_active_context",
                    {
                        "active_batch": {
                            "run_id": run_id,
                            "source": "derived",
                            "item_count": 1,
                        }
                    },
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "target_id": target_id,
                                "name": "Albumless Study",
                                "artist_name": "Artist Alpha",
                                "playback": {"route": "preview_only"},
                            }
                        ],
                    },
                ),
                ok_tool_result(
                    "preview_catalog_track",
                    {"canonical_id": target_id, "started": True},
                ),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(
            result.final_text,
            "正在试听《Albumless Study》— Artist Alpha，约 30 秒。",
        )
        self.assertIsNone(result.recommendation_payload)
        self.assertEqual(
            [execution.name for execution in result.tool_executions],
            [
                "get_active_context",
                "get_recommendation_run",
                "preview_catalog_track",
            ],
        )
        self.assertEqual(
            [execution.origin for execution in result.tool_executions],
            ["provider_requested", "policy_injected", "policy_injected"],
        )
        self.assertEqual(loop.client.recorded_payloads[-1], {"canonical_id": target_id})
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        verified = loop.client.service.verified_selections_for_run(run_id)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].canonical_id, target_id)

    def test_existing_batch_provider_proposal_cannot_replace_code_owned_action(self) -> None:
        run_id = "rcm_33333333-3333-4333-8333-333333333333"
        authoritative_id = "trk_33333333-3333-4333-8333-333333333333"
        injected_id = "trk_99999999-9999-4999-8999-999999999999"
        loop, provider = self._loop(
            [
                tool_response([self._call("get_active_context")]),
                # A malicious next response is never requested: code owns the
                # exact run read and action immediately after active context.
                tool_response(
                    [
                        self._call(
                            "play_track",
                            {"canonical_id": injected_id},
                        )
                    ]
                ),
            ],
            results=[
                ok_tool_result(
                    "get_active_context",
                    {"active_batch": {"run_id": run_id, "item_count": 1}},
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "position": 1,
                                "target_id": authoritative_id,
                                "playback": {"route": "preview_only"},
                            }
                        ],
                    },
                ),
                ok_preview_started(authoritative_id),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(result.final_text, "已开始试听，约 30 秒。")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            loop.client.recorded,
            ["get_active_context", "get_recommendation_run", "preview_catalog_track"],
        )
        self.assertEqual(
            loop.client.recorded_payloads[-1],
            {"canonical_id": authoritative_id},
        )
        self.assertNotIn(injected_id, str(loop.client.recorded_payloads))

    def test_existing_batch_library_route_constructs_only_formal_play(self) -> None:
        run_id = "rcm_11111111-1111-4111-8111-111111111111"
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        loop, provider = self._loop(
            [
                tool_response([self._call("get_active_context")]),
                text_response("provider final should not be needed"),
            ],
            results=[
                ok_tool_result(
                    "get_active_context",
                    {"active_batch": {"run_id": run_id, "item_count": 1}},
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "position": 1,
                                "target_id": target_id,
                                "playback": {"route": "library"},
                            }
                        ],
                    },
                ),
                ok_read_result("play_track"),
                ok_delegated_play_readback(target_id),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(
            result.final_text,
            "正在播放《Synthetic》— Synthetic Artist。",
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            loop.client.recorded,
            [
                "get_active_context",
                "get_recommendation_run",
                "play_track",
                "get_now_playing",
            ],
        )
        self.assertNotIn("preview_catalog_track", loop.client.recorded)
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(
            result.action_attempt.readback_state,
            ActionReadbackState.VERIFIED,
        )
        self.assertEqual(result.action_attempt.actual_player_canonical_id, target_id)

    def test_active_selection_tool_ok_but_actual_player_mismatch_fails_closed(self) -> None:
        run_id = "rcm_11111111-1111-4111-8111-111111111111"
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        other_id = "trk_22222222-2222-4222-8222-222222222222"
        loop, provider = self._loop(
            [tool_response([self._call("get_active_context")])],
            results=[
                ok_tool_result(
                    "get_active_context",
                    {"active_batch": {"run_id": run_id, "item_count": 1}},
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "position": 1,
                                "target_id": target_id,
                                "playback": {"route": "library"},
                            }
                        ],
                    },
                ),
                ok_read_result("play_track"),
                ok_tool_result(
                    "get_now_playing",
                    {
                        "now_playing": {
                            "state": "playing",
                            "name": "Different Song",
                            "artist": "Different Artist",
                        },
                        "agent_channel": {
                            "state": "library",
                            "canonical_id": target_id,
                        },
                        "player_canonical_id": other_id,
                        "canonical_resolution": "binding",
                    },
                ),
            ],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(
            result.final_text,
            "播放指令已发出，但暂时无法确认当前播放状态。",
        )
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(
            result.action_attempt.failure_reason,
            "player_canonical_mismatch",
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            loop.client.service.verified_selections_for_run(run_id), ()
        )

    def test_consecutive_existing_batch_turns_reuse_the_same_authoritative_run(self) -> None:
        self._seed_two_positives()
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        adapter = DelegatedPlaybackAdapter()
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
        )
        self.addCleanup(service.close)
        client = self._client(service)
        generated = client.call(
            "generate_recommendation",
            {"target_ids": [target_id], "limit": 1},
        )
        run_id = generated.payload["run_id"]
        provider = FakeProvider(
            [
                tool_response(
                    [ProviderToolCall("c1", "get_active_context", "{}")]
                ),
                tool_response(
                    [ProviderToolCall("c2", "get_active_context", "{}")]
                ),
            ]
        )
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)

        first = loop.run("随便播放一首")
        after_first = client.call("get_active_context", {})
        second = loop.run("随便播放一首")
        after_second = client.call("get_active_context", {})

        self.assertEqual(
            first.final_text,
            "正在播放《Synthetic Duet》— Artist Alpha。",
        )
        self.assertEqual(
            second.final_text,
            "正在播放《Synthetic Duet》— Artist Alpha。",
        )
        self.assertEqual(after_first.payload["active_batch"]["run_id"], run_id)
        self.assertEqual(after_second.payload["active_batch"]["run_id"], run_id)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            adapter.calls,
            [
                ("play_track", ("SYNTH-TRACK-001",)),
                ("play_track", ("SYNTH-TRACK-001",)),
            ],
        )
        self.assertEqual(
            [execution.name for execution in first.tool_executions],
            ["get_active_context", "get_recommendation_run", "play_track", "get_now_playing"],
        )
        self.assertEqual(
            [execution.name for execution in second.tool_executions],
            ["get_active_context", "get_recommendation_run", "play_track", "get_now_playing"],
        )
        verified = service.verified_selections_for_run(run_id)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].canonical_id, target_id)

    def test_choose_another_progresses_deterministically_then_exhausts(self) -> None:
        run_id = "rcm_44444444-4444-4444-8444-444444444444"
        track_ids = (
            "trk_44444444-4444-4444-8444-444444444441",
            "trk_44444444-4444-4444-8444-444444444442",
            "trk_44444444-4444-4444-8444-444444444443",
        )
        run_payload = {
            "run_id": run_id,
            "items": [
                {
                    "position": position,
                    "target_id": canonical_id,
                    "name": f"Track {position}",
                    "artist_name": "Artist",
                    "playback": {"route": "library"},
                }
                for position, canonical_id in enumerate(track_ids, start=1)
            ],
        }

        def context_result() -> AgentToolResult:
            return ok_tool_result(
                "get_active_context",
                {"active_batch": {"run_id": run_id, "item_count": 3}},
            )

        def readback(canonical_id: str, position: int) -> AgentToolResult:
            return ok_tool_result(
                "get_now_playing",
                {
                    "now_playing": {
                        "state": "playing",
                        "name": f"Track {position}",
                        "artist": "Artist",
                    },
                    "agent_channel": {
                        "state": "library",
                        "canonical_id": canonical_id,
                    },
                    "player_canonical_id": canonical_id,
                    "canonical_resolution": "binding",
                },
            )

        results: list[AgentToolResult] = []
        for position, canonical_id in enumerate(track_ids, start=1):
            results.extend(
                [
                    context_result(),
                    ok_tool_result("get_recommendation_run", run_payload),
                    ok_read_result("play_track"),
                    readback(canonical_id, position),
                ]
            )
        results.extend(
            [
                context_result(),
                ok_tool_result("get_recommendation_run", run_payload),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        provider = FakeProvider(
            [tool_response([self._call("get_active_context")])]
        )
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)

        first = loop.run("随便播放一首")
        second = loop.run("再换一首")
        third = loop.run("再换一首")
        exhausted = loop.run("再换一首")

        self.assertEqual(
            [
                payload["canonical_id"]
                for tool, payload in zip(client.recorded, client.recorded_payloads)
                if tool == "play_track"
            ],
            list(track_ids),
        )
        self.assertEqual(first.action_attempt.selected_canonical_id, track_ids[0])
        self.assertEqual(second.action_attempt.selected_canonical_id, track_ids[1])
        self.assertEqual(third.action_attempt.selected_canonical_id, track_ids[2])
        self.assertIsNone(exhausted.action_attempt)
        self.assertEqual(
            exhausted.final_text,
            "当前这批推荐里已经没有其他可播放或试听的曲目了。",
        )
        self.assertEqual(
            [selection.canonical_id for selection in service.verified_selections_for_run(run_id)],
            list(track_ids),
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("generate_recommendation", client.recorded)
        self.assertNotIn("generate_inferred_recommendation", client.recorded)

    def test_failed_item_remains_eligible_for_choose_another(self) -> None:
        run_id = "rcm_55555555-5555-4555-8555-555555555555"
        target_id = "trk_55555555-5555-4555-8555-555555555555"
        other_id = "trk_66666666-6666-4666-8666-666666666666"
        context = ok_tool_result(
            "get_active_context",
            {"active_batch": {"run_id": run_id, "item_count": 1}},
        )
        run = ok_tool_result(
            "get_recommendation_run",
            {
                "run_id": run_id,
                "items": [
                    {
                        "position": 1,
                        "target_id": target_id,
                        "playback": {"route": "library"},
                    }
                ],
            },
        )
        mismatch = ok_tool_result(
            "get_now_playing",
            {
                "now_playing": {"state": "playing"},
                "agent_channel": {
                    "state": "library",
                    "canonical_id": target_id,
                },
                "player_canonical_id": other_id,
                "canonical_resolution": "binding",
            },
        )
        success = ok_delegated_play_readback(target_id)
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                context,
                run,
                ok_read_result("play_track"),
                mismatch,
                context,
                run,
                ok_read_result("play_track"),
                success,
            ],
        )
        provider = FakeProvider(
            [tool_response([self._call("get_active_context")])]
        )
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)

        failed = loop.run("随便播放一首")
        retried = loop.run("再换一首")

        self.assertEqual(failed.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(retried.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(retried.action_attempt.selected_canonical_id, target_id)
        played = [
            payload["canonical_id"]
            for tool, payload in zip(client.recorded, client.recorded_payloads)
            if tool == "play_track"
        ]
        self.assertEqual(played, [target_id, target_id])
        self.assertEqual(
            [selection.canonical_id for selection in service.verified_selections_for_run(run_id)],
            [target_id],
        )
        self.assertEqual(len(provider.calls), 1)

    def test_active_item_index_update_before_failed_readback_is_not_verified(self) -> None:
        class StubbornPlaybackAdapter(DelegatedPlaybackAdapter):
            def play_track(self, persistent_id: str) -> None:
                self.calls.append(("play_track", (persistent_id,)))

        self._seed_two_positives()
        target_id = "trk_11111111-1111-4111-8111-111111111111"
        adapter = StubbornPlaybackAdapter()
        adapter.now_pid = "SYNTH-TRACK-002"
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
        )
        self.addCleanup(service.close)
        client = self._client(service)
        generated = client.call(
            "generate_recommendation",
            {"target_ids": [target_id], "limit": 1},
        )
        run_id = generated.payload["run_id"]
        provider = FakeProvider(
            [tool_response([self._call("get_active_context")])]
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("随便播放一首")

        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(result.action_attempt.failure_reason, "player_canonical_mismatch")
        self.assertEqual(service._active_context.active_item_index, 0)
        self.assertEqual(service.verified_selections_for_run(run_id), ())

    def test_existing_batch_delegation_refuses_unbound_target_without_dispatch(self) -> None:
        target_id = "trk_33333333-3333-4333-8333-333333333333"
        loop, provider = self._loop(
            [
                tool_response(
                    [
                        self._call(
                            "preview_catalog_track",
                            {"canonical_id": target_id},
                        )
                    ]
                ),
                text_response("无法执行。"),
            ],
            results=[],
        )

        result = loop.run("随便播放一首")

        self.assertEqual(
            result.final_text,
            "暂时无法从当前推荐中确认要播放或试听的曲目。",
        )
        self.assertEqual(loop.client.recorded, [])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.tool_executions, ())

    def test_recommendation_generation_ok_makes_next_round_final_only(self) -> None:
        loop, provider = self._loop(
            [
                tool_response([self._call("generate_recommendation")]),
                text_response("为你找到这几首：1. 夜曲 — 测试艺人，古典旋律。"),
            ],
            results=s5_prefetch_padding()
            + [ok_generation_batch("generate_recommendation")],
            instrument=True,
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.final_text, "为你找到这几首：1. 夜曲 — 测试艺人，古典旋律。")
        self.assertEqual(provider.calls[0]["tools"], project_tools(_RECOMMENDATION_TOOL_NAMES))
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertIsNotNone(result.trace)
        self.assertEqual(
            result.trace.rounds[0].tool_schemas_count, len(_RECOMMENDATION_TOOL_NAMES)
        )
        self.assertEqual(result.trace.rounds[1].tool_schemas_count, 0)
        self.assertEqual(result.trace.rounds[1].tool_calls_count, 0)

    def test_tools_stay_until_generation_then_final_round_is_empty(self) -> None:
        loop, provider = self._loop(
            [
                tool_response([self._call("get_active_context")]),
                tool_response([self._call("discover_catalog_tracks")]),
                tool_response([self._call("generate_recommendation")]),
                text_response("今晚这批：1. 夜曲 — 测试艺人。"),
            ],
            results=s5_prefetch_padding()
            + [
                ok_discovery_result(),
                ok_generation_batch("generate_recommendation"),
            ],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.rounds, 4)
        for call in provider.calls[:3]:
            self.assertEqual(call["tools"], project_tools(_RECOMMENDATION_TOOL_NAMES))
        self.assertEqual(provider.calls[3]["tools"], [])

    def test_preview_started_zero_tools_next_round(self) -> None:
        # THE S2 case: preview_catalog_track started -> the contract pins
        # 「已开始试听」 as the immediate answer with no confirmation calls, so
        # the next round is final-answer-only. Instrumented: the same trace
        # the --trace file reports must meter tool_schemas_count 0 on the
        # final round.
        loop, provider = self._loop(
            [
                tool_response([self._call("preview_catalog_track")]),
                text_response("已开始试听（30 秒）。"),
            ],
            results=[ok_read_result("preview_catalog_track")],
            instrument=True,
        )
        result = loop.run("试听这首")
        self.assertEqual(result.rounds, 2)
        self.assertIn("试听", result.final_text)
        self.assertEqual(provider.calls[0]["tools"], project_tools(_PREVIEW_TOOL_NAMES))
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertIsNotNone(result.trace)
        self.assertEqual(result.trace.rounds[0].tool_schemas_count, len(_PREVIEW_TOOL_NAMES))
        self.assertEqual(result.trace.rounds[0].tool_calls_count, 1)
        final_round = result.trace.rounds[1]
        self.assertEqual(final_round.tool_schemas_count, 0)
        self.assertEqual(final_round.tool_schemas_chars, 0)
        self.assertEqual(final_round.tool_calls_count, 0)

    def test_stop_and_batch_preview_and_open_ok_zero_tools_next_round(self) -> None:
        # The other three contract-pinned terminal executions switch the same
        # way: stop_preview (idempotent stop), preview_batch (session.state is
        # the answer), open_in_apple_music (the url is the answer).
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [
                tool_response([self._call("stop_preview")]),
                text_response("已停止试听。"),
                tool_response([self._call("preview_batch")]),
                text_response("已开始连续试听本批，进度会陆续播报。"),
                tool_response([self._call("open_in_apple_music")]),
                text_response("已在 Apple Music 中打开这首歌。"),
            ]
        )
        loop = ProviderAgentLoop(
            provider, RecordingClient(service, results=[
                ok_read_result("stop_preview"),
                ok_read_result("preview_batch"),
                ok_read_result("open_in_apple_music"),
            ]), PROVIDER_TOOL_SCHEMAS
        )
        self.assertEqual(loop.run("停止试听").rounds, 2)
        self.assertEqual(loop.run("把这一批都试听一遍").rounds, 2)
        self.assertEqual(loop.run("在 Apple Music 中打开刚才第 2 首").rounds, 2)
        # S3: the stop and batch-preview forms run the preview group; the open
        # request is unclassified so it keeps the full registry (its only tool
        # here). S2: each ok terminal execution still drops the next round to ().
        self.assertEqual(provider.calls[0]["tools"], project_tools(_PREVIEW_TOOL_NAMES))
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertEqual(provider.calls[2]["tools"], project_tools(_PREVIEW_TOOL_NAMES))
        self.assertEqual(provider.calls[3]["tools"], [])
        self.assertEqual(provider.calls[4]["tools"], list(PROVIDER_TOOL_SCHEMAS))
        self.assertEqual(provider.calls[5]["tools"], [])

    def test_failure_outcome_never_drops_tools(self) -> None:
        # Non-ok outcomes never switch: the model still has explaining and
        # recovery to do and must keep every tool (the switch reads only
        # outcome == ok from the execution record).
        loop, provider = self._loop(
            [
                tool_response([self._call("preview_catalog_track")]),
                text_response("抱歉，这首试听启动失败。"),
            ],
            results=[failed_generation_result("preview_catalog_track")],
        )
        result = loop.run("试听这首")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(provider.calls[1]["tools"], project_tools(_PREVIEW_TOOL_NAMES))


class DirectActionTerminalTruthTest(ProviderAgentLoopTest):
    """Named/non-delegated actions close from ActionAttempt, not model prose."""

    TARGET = "trk_11111111-1111-4111-8111-111111111111"

    def _run(self, text: str, tool: str, results: list[AgentToolResult]):
        provider = FakeProvider(
            [
                tool_response(
                    [
                        ProviderToolCall(
                            "a1",
                            tool,
                            json.dumps({"canonical_id": self.TARGET}),
                        )
                    ]
                ),
                text_response("模型自行声称成功。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        effective_results = list(results)
        if tool == "play_track":
            # P22-S2.1 follow-up: named formal play now performs one
            # code-owned library read from TurnPlan.target_text before the
            # Provider can execute play_track. Keep the existing formal-play
            # assertions, but make that new read explicit in the fixture.
            effective_results.insert(
                0,
                named_play_search_result(
                    [
                        named_play_match(
                            self.TARGET,
                            name="Synthetic",
                            artist="Synthetic Artist",
                            route="library",
                        )
                    ]
                ),
            )
        client = RecordingClient(service, results=effective_results)
        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run(text)
        return result, client, provider

    def test_formal_play_uses_strict_readback_and_skips_provider_closeout(self) -> None:
        result, client, provider = self._run(
            "播放 Synthetic",
            "play_track",
            [
                ok_read_result("play_track"),
                ok_delegated_play_readback(self.TARGET),
            ],
        )

        self.assertEqual(
            result.final_text, "正在播放《Synthetic》— Synthetic Artist。"
        )
        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(
            client.recorded,
            ["search_library_tracks", "play_track", "get_now_playing"],
        )
        self.assertEqual(len(provider.calls), 1)

    def test_formal_play_tool_ok_with_actual_mismatch_fails_closed(self) -> None:
        readback = ok_delegated_play_readback(self.TARGET)
        payload = dict(readback.payload)
        payload["player_canonical_id"] = (
            "trk_22222222-2222-4222-8222-222222222222"
        )
        result, _, provider = self._run(
            "播放 Synthetic",
            "play_track",
            [ok_read_result("play_track"), ok_tool_result("get_now_playing", payload)],
        )

        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(
            result.action_attempt.failure_reason, "player_canonical_mismatch"
        )
        self.assertNotIn("正在播放", result.final_text)
        self.assertEqual(len(provider.calls), 1)

    def test_preview_started_false_is_terminal_failure(self) -> None:
        result, client, provider = self._run(
            "试听 Synthetic",
            "preview_catalog_track",
            [
                ok_tool_result(
                    "preview_catalog_track",
                    {"canonical_id": self.TARGET, "started": False},
                )
            ],
        )

        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.FAILED)
        self.assertEqual(result.action_attempt.failure_reason, "preview_not_started")
        self.assertEqual(result.final_text, "这首暂时无法试听。")
        self.assertEqual(client.recorded, ["preview_catalog_track"])
        self.assertEqual(len(provider.calls), 1)

    def test_preview_started_true_is_terminal_success(self) -> None:
        result, client, provider = self._run(
            "试听 Synthetic",
            "preview_catalog_track",
            [ok_preview_started(self.TARGET)],
        )

        self.assertEqual(result.action_attempt.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(result.final_text, "已开始试听，约 30 秒。")
        self.assertEqual(client.recorded, ["preview_catalog_track"])
        self.assertEqual(len(provider.calls), 1)

    def test_pause_success_requires_observed_paused_state(self) -> None:
        provider = FakeProvider(
            [
                tool_response([ProviderToolCall("c1", "pause", "{}")]),
                text_response("模型自行声称已暂停。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                ok_read_result("pause"),
                ok_tool_result(
                    "get_now_playing", {"now_playing": {"state": "paused"}}
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("暂停")

        self.assertEqual(result.final_text, "已暂停播放。")
        self.assertEqual(
            result.playback_control_attempt.status,
            ActionAttemptStatus.COMPLETED,
        )
        self.assertEqual(client.recorded, ["pause", "get_now_playing"])
        self.assertEqual(len(provider.calls), 1)

    def test_pause_tool_ok_with_playing_readback_fails_closed(self) -> None:
        provider = FakeProvider(
            [
                tool_response([ProviderToolCall("c1", "pause", "{}")]),
                text_response("模型自行声称已暂停。"),
            ]
        )
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            results=[
                ok_read_result("pause"),
                ok_tool_result(
                    "get_now_playing", {"now_playing": {"state": "playing"}}
                ),
            ],
        )

        result = ProviderAgentLoop(
            provider, client, PROVIDER_TOOL_SCHEMAS
        ).run("暂停")

        self.assertEqual(
            result.playback_control_attempt.status,
            ActionAttemptStatus.FAILED,
        )
        self.assertNotEqual(result.final_text, "已暂停播放。")
        self.assertEqual(client.recorded, ["pause", "get_now_playing"])
        self.assertEqual(len(provider.calls), 1)



class P22S21StructuredNamedPlayOfferTest(unittest.TestCase):
    TARGET = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture_model())

    def _service(self, policy: AgentClientPolicy) -> SharedAgentService:
        return SharedAgentService(
            self.database_path, clients=AgentClientRegistry({CLIENT_ID: policy})
        )

    def _loop(self, results: list[AgentToolResult], responses=None):
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(list(responses or []))
        client = RecordingClient(service, results)
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        return loop, provider, client

    def test_unique_preview_only_named_search_returns_structured_offer(self) -> None:
        loop, provider, client = self._loop([
            named_play_search_result([named_play_match(self.TARGET)])
        ])

        result = loop.run("播放 Wendy")

        self.assertEqual(provider.calls, [])
        self.assertEqual(client.recorded, ["search_library_tracks"])
        self.assertEqual(client.recorded_payloads, [{"term": "Wendy"}])
        self.assertIsNotNone(result.offered_action)
        self.assertEqual(result.offered_action.target_canonical_id, self.TARGET)
        self.assertEqual(result.offered_action.verified_title, "Wendy")
        self.assertEqual(result.offered_action.verified_artist, "Test Artist")
        self.assertEqual(result.offered_action.source, "structured_named_play_resolution")
        self.assertIn("《Wendy》— Test Artist", result.final_text)
        self.assertIn("可以试听 30 秒", result.final_text)
        self.assertEqual(result.rounds, 0)

    def test_post_discovery_refresh_uses_turnplan_target_not_provider_term(self) -> None:
        empty = named_play_search_result([])
        discovered = AgentToolResult(
            request_id="req_55555555-5555-4555-8555-555555555556",
            tool="discover_catalog_tracks",
            outcome=AgentToolOutcome.OK,
            payload={"term": "model-chosen-wrong-term", "discovered_count": 1},
            error_code=None,
            error_message=None,
            completed_at=datetime(2026, 9, 16, 0, 0, 1, tzinfo=timezone.utc),
            replayed=False,
        )
        refreshed = named_play_search_result([named_play_match(self.TARGET)])
        discover_call = ProviderToolCall(
            "d1",
            "discover_catalog_tracks",
            json.dumps({"term": "model-chosen-wrong-term"}),
        )
        loop, provider, client = self._loop(
            [empty, discovered, refreshed],
            [tool_response([discover_call])],
        )

        result = loop.run("播放 Wendy")

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            client.recorded,
            ["search_library_tracks", "discover_catalog_tracks", "search_library_tracks"],
        )
        self.assertEqual(client.recorded_payloads[0], {"term": "Wendy"})
        self.assertEqual(client.recorded_payloads[1], {"term": "model-chosen-wrong-term"})
        self.assertEqual(client.recorded_payloads[2], {"term": "Wendy"})
        self.assertIsNotNone(result.offered_action)
        self.assertEqual(result.offered_action.target_canonical_id, self.TARGET)

    def test_ambiguous_same_name_preview_results_fail_closed_without_offer(self) -> None:
        loop, provider, _ = self._loop([
            named_play_search_result([
                named_play_match(self.TARGET, artist="Artist A"),
                named_play_match(
                    "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                    artist="Artist B",
                ),
            ])
        ])

        result = loop.run("播放 Wendy")

        self.assertEqual(provider.calls, [])
        self.assertIsNone(result.offered_action)
        self.assertIn("多个", result.final_text)
        self.assertIn("歌手", result.final_text)

    def test_library_route_does_not_create_preview_offer_or_steal_provider_turn(self) -> None:
        loop, provider, client = self._loop(
            [named_play_search_result([named_play_match(self.TARGET, route="library")])],
            [text_response("继续按正式播放流程处理。")],
        )

        result = loop.run("播放 Wendy")

        self.assertIsNone(result.offered_action)
        self.assertEqual(result.final_text, "继续按正式播放流程处理。")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(client.recorded, ["search_library_tracks"])

    def test_provider_prose_cannot_create_offer_when_structured_target_unresolved(self) -> None:
        loop, provider, _ = self._loop(
            [named_play_search_result([])],
            [text_response("只能试听 30 秒。需要我为你试听吗？")],
        )

        result = loop.run("播放 Wendy")

        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(result.offered_action)
        self.assertNotIn("需要我为你试听吗", result.final_text)
        self.assertIn("无法唯一确认", result.final_text)

    def test_explicit_index_preview_only_run_emits_structured_offer_for_exact_second_item(self) -> None:
        run_id = "rcm_22222222-2222-4222-8222-222222222222"
        first_id = "trk_11111111-1111-4111-8111-111111111111"
        second_id = "trk_22222222-2222-4222-8222-222222222222"
        context_call = ProviderToolCall("ctx", "get_active_context", "{}")
        run_call = ProviderToolCall(
            "run",
            "get_recommendation_run",
            json.dumps({"run_id": run_id}),
        )
        loop, provider, client = self._loop(
            [
                ok_tool_result(
                    "get_active_context",
                    {
                        "active_batch": {
                            "run_id": run_id,
                            "source": "register",
                            "item_count": 2,
                        }
                    },
                ),
                ok_tool_result(
                    "get_recommendation_run",
                    {
                        "run_id": run_id,
                        "items": [
                            {
                                "position": 1,
                                "target_id": first_id,
                                "name": "First Song",
                                "artist_name": "First Artist",
                                "playback": {"route": "library"},
                            },
                            {
                                "position": 2,
                                "target_id": second_id,
                                "name": "Second Song",
                                "artist_name": "Second Artist",
                                "playback": {"route": "preview_only"},
                            },
                        ],
                    },
                ),
            ],
            [
                tool_response([context_call]),
                tool_response([run_call]),
                text_response("provider prose offer should never be needed"),
            ],
        )

        result = loop.run("播放第二首")

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(client.recorded, ["get_active_context", "get_recommendation_run"])
        self.assertIsNotNone(result.offered_action)
        self.assertEqual(result.offered_action.target_canonical_id, second_id)
        self.assertEqual(result.offered_action.verified_title, "Second Song")
        self.assertEqual(result.offered_action.verified_artist, "Second Artist")
        self.assertEqual(
            result.offered_action.source, "active_recommendation_explicit_index"
        )
        self.assertEqual(
            result.final_text,
            "《Second Song》— Second Artist 目前无法正式播放，可以试听 30 秒。需要我开始试听吗？",
        )
        self.assertNotIn("preview_catalog_track", client.recorded)
        self.assertNotIn("play_track", client.recorded)

    def test_explicit_index_stale_or_unresolved_target_never_emits_structured_offer(self) -> None:
        run_id = "rcm_22222222-2222-4222-8222-222222222222"
        second_id = "trk_22222222-2222-4222-8222-222222222222"
        scenarios = (
            (
                "stale-derived-run",
                {"run_id": run_id, "source": "derived", "item_count": 2},
                [
                    {
                        "position": 2,
                        "target_id": second_id,
                        "playback": {"route": "preview_only"},
                    }
                ],
            ),
            (
                "unresolved-index",
                {"run_id": run_id, "source": "register", "item_count": 1},
                [
                    {
                        "position": 1,
                        "target_id": second_id,
                        "playback": {"route": "preview_only"},
                    }
                ],
            ),
            (
                "missing-canonical",
                {"run_id": run_id, "source": "register", "item_count": 2},
                [
                    {
                        "position": 2,
                        "name": "Second Song",
                        "playback": {"route": "preview_only"},
                    }
                ],
            ),
            (
                "ambiguous-position",
                {"run_id": run_id, "source": "register", "item_count": 2},
                [
                    {
                        "position": 2,
                        "target_id": second_id,
                        "playback": {"route": "preview_only"},
                    },
                    {
                        "position": 2,
                        "target_id": "trk_33333333-3333-4333-8333-333333333333",
                        "playback": {"route": "preview_only"},
                    },
                ],
            ),
            (
                "no-preview-eligibility",
                {"run_id": run_id, "source": "register", "item_count": 2},
                [
                    {
                        "position": 2,
                        "target_id": second_id,
                        "playback": {"route": "unavailable"},
                    }
                ],
            ),
        )
        for label, active_batch, items in scenarios:
            with self.subTest(label=label):
                context_call = ProviderToolCall("ctx", "get_active_context", "{}")
                run_call = ProviderToolCall(
                    "run",
                    "get_recommendation_run",
                    json.dumps({"run_id": run_id}),
                )
                loop, provider, client = self._loop(
                    [
                        ok_tool_result(
                            "get_active_context", {"active_batch": active_batch}
                        ),
                        ok_tool_result(
                            "get_recommendation_run",
                            {"run_id": run_id, "items": items},
                        ),
                    ],
                    [
                        tool_response([context_call]),
                        tool_response([run_call]),
                        text_response("普通流程继续。"),
                    ],
                )

                result = loop.run("播放第二首")

                self.assertIsNone(result.offered_action)
                self.assertEqual(result.final_text, "普通流程继续。")
                self.assertEqual(len(provider.calls), 3)
                self.assertNotIn("preview_catalog_track", client.recorded)

class S3TaskToolSurfaceTest(ProviderAgentLoopTest):
    """S3: the first round of a run carries the task's tool group; every
    unclassified line keeps the full registry, and the S1/S2 behaviors compose
    on top (plain chat zero schemas; terminal-ok rounds drop to zero).

    The groups are name projections of ``PROVIDER_TOOL_SCHEMAS`` built inside
    the loop from the caller's tool list -- no schema is created, deleted, or
    modified. ``FakeProvider`` records the exact per-round tool sequence, and
    the instrumented trace meters the acceptance metric."""

    def _call(self, name: str) -> ProviderToolCall:
        return ProviderToolCall(call_id=f"call_{name}", name=name, arguments="{}")

    def _loop(
        self,
        responses: list[ProviderResponse],
        results: list[AgentToolResult] | None = None,
        *,
        instrument: bool = False,
    ) -> tuple[ProviderAgentLoop, FakeProvider]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(responses)
        client = (
            RecordingClient(service, results=results)
            if results is not None
            else self._client(service)
        )
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=instrument),
        )
        return loop, provider

    def _first_round_tools(self, text: str) -> list[ProviderToolSchema]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        loop.run(text)
        return provider.calls[0]["tools"]

    def test_plain_chat_still_zero_schemas_after_s3(self) -> None:
        # S1 composes: the zero-tool gate fires before any task grouping.
        self.assertEqual(self._first_round_tools("你好"), [])

    def test_recommendation_family_runs_the_recommendation_group(self) -> None:
        # The closed recommendation set plus the open 推荐点<方向> form all map
        # to the recommendation chain -- provably smaller than the full 31 but
        # containing every tool the live B2 chain and the prompt's
        # recommendation clauses can reach.
        expected = project_tools(_RECOMMENDATION_TOOL_NAMES)
        for text in ("推荐几首歌", "再来一批", "换一组", "推荐点日系的", "找类似这首的"):
            self.assertEqual(self._first_round_tools(text), expected)
        self.assertLess(len(expected), len(PROVIDER_TOOL_SCHEMAS))
        names = {tool.name for tool in expected}
        for critical in (
            "generate_recommendation",
            "generate_inferred_recommendation",
            "discover_catalog_tracks",
            "query_catalog_discovery_state",
            "search_library_tracks",
            "get_active_context",
            "get_now_playing",
            "get_recommendation_run",
            "list_recommendation_runs",
            "query_track_preference",
            # The live B2 recommendation chain really consulted these reads.
            "list_feedback_observations",
        ):
            self.assertIn(critical, names)
        for unreachable in (
            "play_track", "play", "pause", "next_track", "previous_track",
            "preview_catalog_track", "preview_batch", "stop_preview",
            "get_playback_context",
            "record_feedback", "interpret_feedback", "apply_learning",
            "open_in_apple_music", "add_catalog_to_library",
            "execute_write_intent", "get_agent_capabilities",
        ):
            self.assertNotIn(unreachable, names)

    def test_fresh_discovery_forms_share_the_recommendation_group(self) -> None:
        # 找些新的-class requests need discover + inferred generation + the
        # batch/context reads -- exactly the recommendation chain, nothing more.
        expected = project_tools(_RECOMMENDATION_TOOL_NAMES)
        for text in ("找些新的", "推荐没听过的", "找库外歌曲"):
            self.assertEqual(self._first_round_tools(text), expected)
            self.assertNotIn(
                "open_in_apple_music", {t.name for t in self._first_round_tools(text)}
            )

    def test_generation_ok_makes_the_next_round_final_only(self) -> None:
        loop, provider = self._loop(
            [
                tool_response([self._call("generate_recommendation")]),
                text_response("为你找到这几首。"),
            ],
            results=s5_prefetch_padding()
            + [ok_generation_batch("generate_recommendation")],
        )
        loop.run("推荐几首歌")
        self.assertEqual(provider.calls[1]["tools"], [])

    def test_playback_family_runs_the_playback_group(self) -> None:
        # The play contract's chain: library search first, catalog fallback,
        # play_track (never play as a substitute), the mandated check-back,
        # bare-播放 resume, batch reads for 播放第N首. The formal-playback
        # boundary is structural (B ruling): a play turn is offered NO preview
        # tool at all, so self-downgrade is impossible to express -- saying
        # "只能试听" stays a natural-language offer that waits for the user's
        # own preview turn.
        expected = project_tools(_PLAYBACK_TOOL_NAMES)
        for text in ("播放夜曲", "播放第二首", "播放这首", "播放"):
            self.assertEqual(self._first_round_tools(text), expected)
        names = {tool.name for tool in expected}
        for critical in (
            "play_track", "play", "search_library_tracks", "discover_catalog_tracks",
            "get_now_playing", "get_active_context", "get_recommendation_run",
            "list_recommendation_runs",
        ):
            self.assertIn(critical, names)
        for unreachable in (
            "preview_catalog_track", "preview_batch", "stop_preview", "pause",
            "next_track", "previous_track",
            "generate_recommendation", "generate_inferred_recommendation",
            "record_feedback", "open_in_apple_music",
        ):
            self.assertNotIn(unreachable, names)

    def test_play_preview_boundary_pins(self) -> None:
        # The B ruling, pinned as four explicit properties. (1) The play
        # surface carries none of the preview execution tools -- structural,
        # not wording-only. (2) The preview surface still carries
        # preview_catalog_track. (3) Formal 播放第N首/etc. lines can never
        # surface a preview tool. (4) Mixed/preview-adjacent lines keep the
        # full registry -- the fix must not widen any classifier (the router
        # is untouched; a mixed line reaching the full set is the observable
        # proof of it).
        play_names = {tool.name for tool in project_tools(_PLAYBACK_TOOL_NAMES)}
        preview_names = {tool.name for tool in project_tools(_PREVIEW_TOOL_NAMES)}
        for preview_tool in ("preview_catalog_track", "preview_batch", "stop_preview"):
            self.assertNotIn(preview_tool, play_names)
        self.assertIn("preview_catalog_track", preview_names)
        full_names = {tool.name for tool in PROVIDER_TOOL_SCHEMAS}
        for text in ("播放第二首", "播放夜曲", "播放"):
            self.assertNotIn(
                "preview_catalog_track",
                {tool.name for tool in self._first_round_tools(text)},
            )
        for text in ("试听后再播放", "试听还是播放", "播放器怎么用"):
            self.assertEqual(
                {tool.name for tool in self._first_round_tools(text)}, full_names
            )

    def test_preview_family_runs_the_preview_group(self) -> None:
        # 试听- family, the batch-preview phrasings, and the stop residuals
        # the routing table hands back -- the preview chain plus the batch
        # reads and pause (the stop clause's music fallback). Play tools are
        # out: a preview turn never plays.
        expected = project_tools(_PREVIEW_TOOL_NAMES)
        for text in (
            "试听第二首", "试听夜曲", "试听这首", "都试听一遍",
            "把这一批都试听一遍", "停止试听", "别放了", "停",
        ):
            self.assertEqual(self._first_round_tools(text), expected)
        names = {tool.name for tool in expected}
        for critical in (
            "preview_catalog_track", "stop_preview", "preview_batch",
            "get_playback_context", "get_active_context", "get_recommendation_run",
            "list_recommendation_runs", "search_library_tracks",
            "discover_catalog_tracks", "pause",
        ):
            self.assertIn(critical, names)
        for unreachable in (
            "play_track", "play", "next_track", "previous_track",
            "generate_recommendation", "generate_inferred_recommendation",
            "record_feedback", "open_in_apple_music",
        ):
            self.assertNotIn(unreachable, names)

    def test_feedback_family_runs_the_feedback_group(self) -> None:
        # The P08 verdict chain: locate the batch referent, record, interpret,
        # apply, read back. Generation/play/preview/discovery are unreachable
        # from a verdict (the prompt forbids auto-generation on locate
        # failures, and no play/preview follows a record).
        expected = project_tools(_FEEDBACK_TOOL_NAMES)
        for text in (
            "我喜欢第二首", "这首不喜欢", "不喜欢第二首", "这个方向不错", "这首不错",
        ):
            self.assertEqual(self._first_round_tools(text), expected)
        names = {tool.name for tool in expected}
        for critical in (
            "record_feedback", "interpret_feedback", "apply_learning",
            "list_feedback_observations", "get_feedback_observation",
            "list_learning_applications", "get_learning_application",
            "get_active_context", "get_recommendation_run", "list_recommendation_runs",
        ):
            self.assertIn(critical, names)
        for unreachable in (
            "generate_recommendation", "generate_inferred_recommendation",
            "play_track", "play", "preview_catalog_track", "preview_batch",
            "stop_preview", "discover_catalog_tracks", "open_in_apple_music",
        ):
            self.assertNotIn(unreachable, names)

    def test_ambiguous_mixed_and_unclassified_keep_the_full_registry(self) -> None:
        # Fail-safe center: mixed intents, chained commands, near-miss
        # phrasings, the delegation family (whose first decision may need
        # generation OR play/preview), named-song verdicts, capability questions -- all
        # keep the full 31 so no capability is ever lost to a narrowing.
        full = list(PROVIDER_TOOL_SCHEMAS)
        for text in (
            "换一首",            # context-sensitive: next_track vs play_track vs preview
            "你来决定",          # delegation decision needs the broad first-round surface
            "随便播放一首",       # same, explicit delegation
            "好的",              # consent after 要试听吗? -- must keep preview tools
            "播放器怎么用",       # capability question
            "试听是什么意思",     # feature question (excluded by the preview markers)
            "试听哪首好",        # ambiguous preview question
            "喜欢夜曲",          # named-song verdict (needs search + feedback)
            "你好，帮我推荐几首歌",   # greeting glued to a request
            "再找找适合雨天听的歌",   # near-miss of the fresh set
            "换一批歌",          # near-miss of 换一批
            "在 Apple Music 中打开刚才第 2 首",  # open family
        ):
            self.assertEqual(self._first_round_tools(text), full)

    def test_instrumentation_meters_the_effective_group(self) -> None:
        # The acceptance metric must reflect the schemas actually sent: a
        # recommendation round counts exactly the group's schemas and chars.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(
            provider,
            self._client(service),
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("推荐几首歌")
        round_measure = result.trace.rounds[0]
        self.assertEqual(round_measure.tool_schemas_count, len(_RECOMMENDATION_TOOL_NAMES))
        expected_chars = sum(
            len(tool.name) + len(tool.description)
            + len(json.dumps(dict(tool.input_schema), ensure_ascii=False))
            for tool in project_tools(_RECOMMENDATION_TOOL_NAMES)
        )
        self.assertEqual(round_measure.tool_schemas_chars, expected_chars)
        # The full registry still exists unchanged (no schema deleted):
        # an unclassified run on the same loop meters the full 31.
        provider.responses.append(text_response("好的。"))
        unclassified = loop.run("无限调用")
        self.assertEqual(
            unclassified.trace.rounds[0].tool_schemas_count, len(PROVIDER_TOOL_SCHEMAS)
        )

    def test_custom_tool_list_intersects_with_the_group(self) -> None:
        # The groups are name filters over the CALLER's tool list: a loop
        # configured with a pruned list sends only the intersection, and an
        # intersection that misses every group tool still never fabricates.
        registry_by_name = {item.name: item for item in PROVIDER_TOOL_SCHEMAS}
        pruned = [
            registry_by_name["generate_recommendation"],
            registry_by_name["play_track"],
            registry_by_name["stop_preview"],
        ]
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。"), text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), pruned)
        loop.run("推荐几首歌")
        self.assertEqual([t.name for t in provider.calls[0]["tools"]],
                         ["generate_recommendation"])
        loop.run("试听这首")
        self.assertEqual([t.name for t in provider.calls[1]["tools"]],
                         ["stop_preview"])

    def test_s1_and_s3_compose_per_run(self) -> None:
        # The selection is made once per user message: a plain-chat run stays
        # at zero schemas, the next recommendation run on the SAME loop runs
        # the recommendation group -- no cross-run leakage either way.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("你好！"), text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        loop.run("你好")
        loop.run("推荐几首歌")
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(provider.calls[1]["tools"], project_tools(_RECOMMENDATION_TOOL_NAMES))


class P20Fix09GenerationPromptDisciplineTest(ProviderAgentLoopTest):
    """P20-Fix09 prompt side: the generation scenes (recommendation +
    discovery + the full fallback) teach the first-presentation evidence
    discipline -- per-item reasons only from the item's evidence block,
    provenance-graded wording (§六/§七 forbidden upgrades), the encyclopedia
    filler ban (§八), the now-playing false-causality ban (§九), score never a
    reason (§十二 I), and first-view == follow-up fact level (§十一). The
    explanation prompt carries its own Fix03 module and does NOT duplicate the
    generation-scene clause."""

    _GROUNDING_PHRASES = (
        "推荐理由的证据纪律",
        "只允许来自该条目的",
        "evidence 字段",
        "绝不升级成命中收藏、充足正面支撑、强烈偏好、",
        "不写音乐评论与百科式填充",
        "推荐理由只说明为什么 Music Agent 选了它",
        "当前正在播放的曲目绝不能自动成为推荐理由",
        "理由不提分数",
        "事实等级必须一致",
        "不得在首次展示先夸大、等追问再纠正",
        "带「本次新发现」身份标记的条目只决定「本次新发现」的身份措辞",
        "本次新发现，按 J-Pop 方向推断递选",
        "basis 是曲目自身（label 就是这首歌自己的名字）的",
        "本次新发现，按对这首曲目本身的偏好推断递选",
        "绝不把曲目自身的推断偏好写成某个曲风方向",
        "basis 是 genre（风格）的说「按该风格方向推断」",
        "绝不因为新发现身份把有推断证据的条目说成没有证据",
        "也不得把无新发现标记的目录条目说成新发现",
        # P20-Fix10: the success-scene presentation division of labor -- the
        # program renders the final list + reasons, the model must not
        # re-list them in its free final answer.
        "推荐生成成功后的最终推荐列表与每首理由由程序按权威结果确定性渲染展示",
    )

    def test_recommendation_prompt_teaches_presentation_grounding(self) -> None:
        for phrase in self._GROUNDING_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, _S4_RECOMMENDATION_PROMPT)

    def test_discovery_prompt_teaches_presentation_grounding(self) -> None:
        for phrase in self._GROUNDING_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, _S4_DISCOVERY_PROMPT)

    def test_full_fallback_teaches_presentation_grounding(self) -> None:
        for phrase in self._GROUNDING_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, DEFAULT_SYSTEM_PROMPT)

    def test_explanation_prompt_keeps_fix03_module_without_duplicate(self) -> None:
        # The explanation scene grounds through the Fix03 explanation module;
        # the generation-scene clause must not bleed into it (module-exclusive,
        # like every other S4 surface pin).
        self.assertNotIn("推荐理由的证据纪律", _S4_EXPLANATION_PROMPT)
        self.assertIn("证据方向混合时如实列出各方向", _S4_EXPLANATION_PROMPT)


class S4TaskPromptSurfaceTest(ProviderAgentLoopTest):
    """S4: the run's SYSTEM PROMPT is narrowed per task by the same local
    classifiers as the tool surface. Pinned here: the default is the tagged
    composition of general modules plus BASE safety invariants, each task's
    effective prompt carries exactly its modules, classifier misses keep those
    invariants, and prompts are chosen once per run (no cross-run leakage).

    Substring pins below are chosen to be module-exclusive -- each excluded
    phrase is guaranteed absent from the BASE module, so an on-the-wire
    prompt that must NOT contain it proves the module really did not load."""

    def test_extracted_prompt_system_matches_pre_extraction_snapshots_exactly(self) -> None:
        expected = {
            "_S4_BASE_PROMPT": (1328, "984fd6eb8df3fb1229b223b913c8b1fda38db85fef0d02db681f73559caab3dd"),
            "_S4_RECOMMENDATION_PROMPT": (8567, "2952530801587fc4c421522803c49e151863d80a57818309a3c313452a93d5c7"),
            "_S4_PLAYBACK_PROMPT": (5090, "b6679a51eb44e70da8b4e41f1de51a31f1e0a34f380601a6bae449ec249cf819"),
            "_S4_PREVIEW_PROMPT": (4224, "2ef938ceb2aa0b76e13f8429e784144fb7f0fe3a5a34a4b93f5f476d19a691b1"),
            "_S4_FEEDBACK_PROMPT": (3470, "7d3f33d6d8d22c4ea0f33f5bb71133aaf2f7d3eb86e13ef85f11771a383b9a2f"),
            "_S4_DISCOVERY_PROMPT": (6425, "b6e477a7cf6f7c1b894f454d32f4846b31f2007132537a41159b5da870240479"),
            "_S4_EXPLANATION_PROMPT": (4290, "d69b00677ed52675c5f0c8ef69aaee921623fa8e52fc0868eb7e92d0f6bdbc74"),
            "_S4_LIBRARY_QUERY_PROMPT": (1598, "95eb362c7261cc39c1fb6ff1f28a6c8c7a77b7cfb09f58283f1a9cf2731b28ad"),
            "DEFAULT_SYSTEM_PROMPT": (10941, "0b78ca3eec0cb1d5d3b08fd016369b0e6a9f78871a0c5083d1ce902cdd9428f0"),
        }
        prompts = {
            "_S4_BASE_PROMPT": _S4_BASE_PROMPT,
            "_S4_RECOMMENDATION_PROMPT": _S4_RECOMMENDATION_PROMPT,
            "_S4_PLAYBACK_PROMPT": _S4_PLAYBACK_PROMPT,
            "_S4_PREVIEW_PROMPT": _S4_PREVIEW_PROMPT,
            "_S4_FEEDBACK_PROMPT": _S4_FEEDBACK_PROMPT,
            "_S4_DISCOVERY_PROMPT": _S4_DISCOVERY_PROMPT,
            "_S4_EXPLANATION_PROMPT": _S4_EXPLANATION_PROMPT,
            "_S4_LIBRARY_QUERY_PROMPT": _S4_LIBRARY_QUERY_PROMPT,
            "DEFAULT_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT,
        }
        for name, prompt in prompts.items():
            expected_len, expected_sha = expected[name]
            self.assertEqual(len(prompt), expected_len, name)
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), expected_sha, name)

    def test_extracted_clause_order_membership_and_counts_match_pre_extraction_snapshot(self) -> None:
        import collections

        expected_counts = {
            "base": 6,
            "playback": 5,
            "preview": 1,
            "recommendation": 10,
            "feedback": 0,
            "discovery": 5,
            "referent": 6,
            "explanation": 1,
            "library_query": 1,
        }
        counts = collections.Counter(tag for tag, _ in _S4_PROMPT_CLAUSES)
        self.assertEqual(len(_S4_PROMPT_CLAUSES), 35)
        self.assertEqual(
            {module: counts[module] for module in expected_counts}, expected_counts
        )
        encoded = json.dumps(
            _S4_PROMPT_CLAUSES, ensure_ascii=False, separators=(",", ":")
        )
        self.assertEqual(
            hashlib.sha256(encoded.encode()).hexdigest(),
            "67e191eba14f5efd6800a1f41d805407d3b8de0d9d40bb70109494ede74e2d43",
        )

    def _first_round_prompt(self, text: str) -> str:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        loop.run(text)
        self.assertTrue(provider.calls[0]["system"].endswith(FINAL_ANSWER_CONTRACT))
        return provider.calls[0]["system"][:-len(FINAL_ANSWER_CONTRACT)]

    # ---- composition facts -------------------------------------------------

    def test_full_prompt_is_the_authoritative_safe_fallback_composition(self) -> None:
        self.assertEqual(
            DEFAULT_SYSTEM_PROMPT,
            _compose_system_prompt(_S4_ALL_MODULES),
        )
        self.assertIn("全任务安全不变量", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("不得触发播放、试听、目录发现、推荐生成", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("解释推荐原因时只能使用推荐批次的权威 evidence", DEFAULT_SYSTEM_PROMPT)

    def test_every_clause_is_tagged_with_a_declared_module(self) -> None:
        declared = {
            "base", "playback", "preview", "recommendation", "feedback",
            "discovery", "referent", "explanation", "library_query",
        }
        tags = {tag for tag, _ in _S4_PROMPT_CLAUSES}
        self.assertLessEqual(tags, declared)  # no undeclared tag may appear
        # feedback is declared-but-EMPTY by design: no feedback-specific rule
        # lives in this prompt (the P08 write discipline is carried by tool
        # schemas and sealed policy code). Its absence as a tag is the honest
        # pin of that fact -- the module exists in the composition table.
        self.assertNotIn("feedback", tags)
        # The mapping is per-clause, not per-module: the 8,766-char prompt
        # breaks into many pieces (one per rule), never just six blocks.
        self.assertGreater(len(_S4_PROMPT_CLAUSES), 20)

    def test_task_prompts_are_strictly_smaller_than_full(self) -> None:
        for prompt in (
            _S4_BASE_PROMPT, _S4_RECOMMENDATION_PROMPT, _S4_PLAYBACK_PROMPT,
            _S4_PREVIEW_PROMPT, _S4_FEEDBACK_PROMPT, _S4_DISCOVERY_PROMPT,
        ):
            self.assertLess(len(prompt), len(DEFAULT_SYSTEM_PROMPT))
        # The plain-chat win is the headline number: BASE alone is a small
        # fraction of the full prompt.
        self.assertLess(len(_S4_BASE_PROMPT), len(DEFAULT_SYSTEM_PROMPT) // 4)

    # ---- module membership -------------------------------------------------

    def test_base_module_is_identity_and_discipline_alone(self) -> None:
        prompt = _S4_BASE_PROMPT
        self.assertIn("你是用户的个人音乐推荐 Agent（Music Agent）", prompt)
        self.assertIn("不要编造状态", prompt)
        self.assertIn("不要展示 run_id、candidate_id", prompt)
        self.assertIn("推理只在工具调用中进行", prompt)
        for excluded in (
            "播放指定歌曲时", "播放意图绝不自动降级成试听", "试听是非阻塞的",
            "session.state", "推荐结果输出格式", "必须调用生成推荐工具",
            "min_fresh", "目录发现取舍", "fresh_this_request",
            "推荐生成成功后的最终推荐列表与每首理由由程序按权威结果确定性渲染展示",
        ):
            self.assertNotIn(excluded, prompt)

    def test_every_task_prompt_ships_the_base_module(self) -> None:
        for prompt in (_S4_RECOMMENDATION_PROMPT, _S4_PLAYBACK_PROMPT,
                       _S4_PREVIEW_PROMPT, _S4_FEEDBACK_PROMPT, _S4_DISCOVERY_PROMPT):
            self.assertIn("不要编造状态", prompt)
            self.assertIn("最终用中文简洁回答用户", prompt)

    def test_recommendation_prompt_loads_recommendation_discovery_and_referent(self) -> None:
        prompt = _S4_RECOMMENDATION_PROMPT
        for included in (
            "推荐结果输出格式", "必须调用生成推荐工具生成新批次",
            "一次「换一组」只调用一次生成推荐工具", "收尾", "找类似这首的",
            "当前推荐批的统一定位方式", "第一首/第二首/N首",
            "推荐生成成功后的最终推荐列表与每首理由由程序按权威结果确定性渲染展示",
            "目录发现取舍", "允许在本轮预算内执行一次", "min_fresh",
            "fresh_this_request",
        ):
            self.assertIn(included, prompt)
        for excluded in (
            "播放意图绝不自动降级成试听", "试听是非阻塞的", "session.state",
        ):
            self.assertNotIn(excluded, prompt)

    def test_playback_prompt_loads_playback_and_referent(self) -> None:
        prompt = _S4_PLAYBACK_PROMPT
        for included in (
            "播放指定歌曲时", "播放意图绝不自动降级成试听",
            "get_now_playing 核对", "你来决定/随便播放一首/放首歌",
            "当前推荐批的统一定位方式", "在 Apple Music 中打开",
        ):
            self.assertIn(included, prompt)
        for excluded in (
            "试听是非阻塞的", "session.state", "目录发现取舍", "min_fresh",
            "推荐结果输出格式", "找类似这首的",
        ):
            self.assertNotIn(excluded, prompt)

    def test_preview_prompt_loads_preview_and_referent(self) -> None:
        prompt = _S4_PREVIEW_PROMPT
        for included in (
            "试听是非阻塞的", "已开始试听", "session.state", "试听它/播放它",
            "第一首/第二首/N首",
        ):
            self.assertIn(included, prompt)
        for excluded in (
            "播放意图绝不自动降级成试听", "推荐结果输出格式", "目录发现取舍",
            "min_fresh", "必须调用生成推荐工具",
        ):
            self.assertNotIn(excluded, prompt)

    def test_feedback_prompt_is_base_plus_referent_only(self) -> None:
        # FEEDBACK/LEARNING contributes no clauses today (the P08 write
        # discipline lives in tool schemas + sealed policy code), so the
        # feedback composition is exactly BASE + referent -- and no other
        # task family's rules leak in.
        prompt = _S4_FEEDBACK_PROMPT
        self.assertIn("不要编造状态", prompt)
        self.assertIn("当前推荐批的统一定位方式", prompt)
        for excluded in (
            "播放意图绝不自动降级成试听", "试听是非阻塞的", "session.state",
            "推荐结果输出格式", "必须调用生成推荐工具", "目录发现取舍",
            "min_fresh", "fresh_this_request",
        ):
            self.assertNotIn(excluded, prompt)
        self.assertLess(len(prompt), len(_S4_RECOMMENDATION_PROMPT))

    def test_discovery_prompt_loads_recommendation_and_discovery(self) -> None:
        prompt = _S4_DISCOVERY_PROMPT
        for included in (
            "目录发现取舍", "already_bound", "min_exploration",
            "fresh_this_request", "min_fresh", "Fresh 候选通道",
            "推荐结果输出格式", "必须调用生成推荐工具",
        ):
            self.assertIn(included, prompt)
        for excluded in (
            "播放意图绝不自动降级成试听", "试听是非阻塞的", "session.state",
        ):
            self.assertNotIn(excluded, prompt)

    # ---- selector + on-the-wire prompts ------------------------------------

    def test_selector_maps_every_task_family_to_its_prompt(self) -> None:
        for text, expected in (
            ("你好", _S4_BASE_PROMPT),
            ("在吗", _S4_BASE_PROMPT),
            ("推荐几首歌", _S4_RECOMMENDATION_PROMPT),
            ("再来一批", _S4_RECOMMENDATION_PROMPT),
            ("找类似这首的", _S4_RECOMMENDATION_PROMPT),
            ("播放 起风了", _S4_PLAYBACK_PROMPT),
            ("播放第二首", _S4_PLAYBACK_PROMPT),
            ("试听这首", _S4_PREVIEW_PROMPT),
            ("都放一遍", _S4_PREVIEW_PROMPT),
            ("停止试听", _S4_PREVIEW_PROMPT),
            ("喜欢第二首", _S4_FEEDBACK_PROMPT),
            ("这首不好听", _S4_FEEDBACK_PROMPT),
            ("找些新的", _S4_DISCOVERY_PROMPT),
            ("推荐没听过的", _S4_DISCOVERY_PROMPT),
        ):
            self.assertEqual(_select_system_prompt(text, DEFAULT_SYSTEM_PROMPT), expected)

    def test_classified_runs_send_the_narrowed_prompt_on_the_wire(self) -> None:
        # The loop actually chats with the effective prompt, not the constant.
        self.assertEqual(self._first_round_prompt("你好"), _S4_BASE_PROMPT)
        generic_semantics_prompt = _recommendation_semantics_prompt(
            _ResolvedRecommendationSemantics(
                "generic", None, None, None, (), 5, None
            )
        )
        self.assertEqual(
            self._first_round_prompt("推荐几首歌"),
            _S4_RECOMMENDATION_PROMPT + generic_semantics_prompt,
        )
        self.assertEqual(self._first_round_prompt("播放 夜に駆ける"), _S4_PLAYBACK_PROMPT)

    def test_ambiguous_lines_keep_the_full_prompt(self) -> None:
        # fail-safe: mixed/chained/unnamed/unclassified inputs get the FULL
        # FALLBACK -- the same lines that keep the full tool set under S3.
        for text in ("换一首", "你来决定", "随便播放一首", "好的", "试听后再播放",
                     "试听还是播放", "播放器怎么用", "天气不错啊"):
            self.assertEqual(self._first_round_prompt(text), DEFAULT_SYSTEM_PROMPT)

    def test_library_classifier_miss_keeps_read_only_safety_in_effective_prompt(
        self,
    ) -> None:
        text = "Spring Thief 在我的 Apple Music 资料库里吗？"
        with patch(
            "music_agent.intent_router.is_read_only_library_intent",
            return_value=False,
        ):
            effective = _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(effective, DEFAULT_SYSTEM_PROMPT)
        self.assertIn("只读搜索、查询、解释或能力询问只用于回答当前问题", effective)
        self.assertIn("不得触发播放、试听、目录发现、推荐生成", effective)
        self.assertIn("Apple Music Library 与 Catalog 目录来源不得混淆", effective)

    def test_explanation_classifier_miss_keeps_evidence_safety_in_effective_prompt(
        self,
    ) -> None:
        text = "为什么推荐这些？"
        with patch(
            "music_agent.intent_router.is_recommendation_explanation_intent",
            return_value=False,
        ):
            effective = _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(effective, DEFAULT_SYSTEM_PROMPT)
        self.assertIn("解释推荐原因时只能使用推荐批次的权威 evidence", effective)
        self.assertIn("不得用常识或音乐知识补造", effective)

    def test_custom_configured_prompt_passes_through_unchanged(self) -> None:
        # Only the built-in prompt is ever decomposed: a caller-configured
        # prompt is returned verbatim for every input, classified or not
        # (fail-safe -- custom prompts are not this module's to split).
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。"), text_response("好的。")])
        loop = ProviderAgentLoop(
            provider,
            self._client(service),
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(system_prompt="CUSTOM NON-DEFAULT PROMPT"),
        )
        loop.run("推荐几首歌")
        loop.run("你好")
        self.assertEqual(provider.calls[0]["system"], "CUSTOM NON-DEFAULT PROMPT")
        self.assertEqual(provider.calls[1]["system"], "CUSTOM NON-DEFAULT PROMPT")

    def test_prompt_selection_is_made_once_per_run(self) -> None:
        # S4 composes with S3: the narrowing decisions are both per user
        # message -- a plain-chat run on one turn never mutates the prompt
        # (or tool surface) of the next classified turn on the same loop.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("你好！"), text_response("好的。")])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        loop.run("你好")
        loop.run("推荐几首歌")
        self.assertEqual(provider.calls[0]["system"], _S4_BASE_PROMPT + FINAL_ANSWER_CONTRACT)
        generic_semantics_prompt = _recommendation_semantics_prompt(
            _ResolvedRecommendationSemantics(
                "generic", None, None, None, (), 5, None
            )
        )
        self.assertEqual(
            provider.calls[1]["system"],
            _S4_RECOMMENDATION_PROMPT
            + generic_semantics_prompt
            + FINAL_ANSWER_CONTRACT,
        )

    # ---- P20-Fix02 natural-phrase coverage ---------------------------------

    def test_explanation_family_maps_to_the_explanation_prompt(self) -> None:
        for text in (
            "为什么推荐这些？", "为什么给我推荐这些？", "为什么这些适合我？",
            "这几首为什么适合我？", "这批为什么适合我？", "推荐理由是什么？",
            "为什么推荐这几首？",
            # P20-Fix06: the verb-first 这一批/这批 variants (UAT-live).
            "为什么这一批适合我？", "为什么这批适合我？", "这批推荐为什么适合我？",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    _S4_EXPLANATION_PROMPT,
                )

    def test_recommendation_surface_extensions_map_to_the_recommendation_prompt(
        self,
    ) -> None:
        for text in (
            "最近给我推荐几首歌", "帮我推荐几首歌", "来几首推荐", "再推荐一批",
            "推荐点歌", "再来一批，换个方向", "换一组换个方向",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    _S4_RECOMMENDATION_PROMPT,
                )

    def test_fresh_extensions_map_to_the_discovery_prompt(self) -> None:
        for text in (
            "推荐一些新歌", "推荐一些没听过的新歌", "推荐一些我没听过的新歌",
            "推荐几首我没有的歌", "找点我没听过的歌", "给我找些新歌",
            "推荐点库外的歌", "找一些新的",
            "来点没听过的",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    _S4_DISCOVERY_PROMPT,
                )

    def test_fresh_priority_beats_recommendation_for_overlapping_forms(self) -> None:
        # 推荐一些新歌 sits in BOTH the frozen recommendation door and the
        # fresh set: the selectors must resolve it on the catalog path.
        self.assertEqual(
            _select_system_prompt("推荐一些新歌", DEFAULT_SYSTEM_PROMPT),
            _S4_DISCOVERY_PROMPT,
        )

    def test_fix02_negatives_keep_the_full_prompt(self) -> None:
        for text in (
            "推荐系统是怎么工作的？", "你会推荐吗？", "为什么推荐算法这么慢？",
            "播放和推荐有什么区别？", "不要推荐了", "我不想听推荐",
            "刚才推荐出错了", "推荐几首然后播放第二首", "推荐还是播放？",
            "推荐完帮我直接试听", "我到底该推荐还是继续播放？",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    _select_system_prompt(text, DEFAULT_SYSTEM_PROMPT),
                    DEFAULT_SYSTEM_PROMPT,
                )

    def test_explanation_tools_are_the_recommendation_reads_only(self) -> None:
        explanation = set(_EXPLANATION_TOOL_NAMES)
        recommendation = set(_RECOMMENDATION_TOOL_NAMES)
        self.assertEqual(
            explanation,
            recommendation
            - {
                "generate_recommendation",
                "generate_inferred_recommendation",
                "discover_catalog_tracks",
            },
        )
        self.assertEqual(len(explanation), 12)
        # Everything kept really exists in the shipped registry (read-only).
        registered = {schema.name for schema in PROVIDER_TOOL_SCHEMAS}
        self.assertTrue(explanation <= registered)

    def test_explanation_family_runs_the_narrowed_read_only_surface(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        for text in (
            "为什么推荐这些？", "为什么给我推荐这些？", "为什么这些适合我？",
            "这几首为什么适合我？", "这批为什么适合我？", "推荐理由是什么？",
            "为什么推荐这几首？",
            # P20-Fix06: the verb-first 这一批/这批 variants (UAT-live).
            "为什么这一批适合我？", "为什么这批适合我？", "这批推荐为什么适合我？",
        ):
            provider = FakeProvider([text_response("好的。")])
            loop = ProviderAgentLoop(
                provider, self._client(service), PROVIDER_TOOL_SCHEMAS
            )
            loop.run(text)
            offered = {tool.name for tool in provider.calls[0]["tools"]}
            self.assertEqual(offered, set(_EXPLANATION_TOOL_NAMES))

    def test_library_queries_offer_reads_but_never_playback(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        for text in (
            "搜索 Spring Thief Yorushika",
            "Spring Thief 在我的资料库里吗？",
            "我的资料库有哪些 Yorushika 的歌？",
        ):
            provider = FakeProvider([text_response("找到了。")])
            loop = ProviderAgentLoop(
                provider, self._client(service), PROVIDER_TOOL_SCHEMAS
            )
            loop.run(text)
            offered = {tool.name for tool in provider.calls[0]["tools"]}
            self.assertEqual(offered, set(_LIBRARY_QUERY_TOOL_NAMES))
            self.assertNotIn("play_track", offered)
            self.assertNotIn("play", offered)
            self.assertNotIn("preview_catalog_track", offered)
            self.assertNotIn("generate_recommendation", offered)
            self.assertEqual(
                provider.calls[0]["system"], _S4_LIBRARY_QUERY_PROMPT + FINAL_ANSWER_CONTRACT
            )

    def test_named_play_keeps_formal_playback_surface(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        loop = ProviderAgentLoop(
            provider, self._client(service), PROVIDER_TOOL_SCHEMAS
        )
        loop.run("播放 Spring Thief — Yorushika")
        offered = {tool.name for tool in provider.calls[0]["tools"]}
        self.assertIn("play_track", offered)
        self.assertNotIn("preview_catalog_track", offered)

    def test_explanation_run_cannot_call_generation_tools(self) -> None:
        # Structural: the explanation surface does not OFFER the generation
        # tools, so even a model that would ask for a new batch cannot call
        # one. One honest read + text answer, no write, no discover, no S5.
        service = _RecordingExecuteService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
        )
        self.addCleanup(service.close)
        provider = FakeProvider([
            tool_response([
                ProviderToolCall("r1", "list_recommendation_runs", "{}"),
            ]),
            text_response("上次那批来自你的收藏偏好。"),
        ])
        loop = ProviderAgentLoop(provider, self._client(service), PROVIDER_TOOL_SCHEMAS)
        result = loop.run("为什么推荐这些？")
        self.assertEqual(result.final_text, "上次那批来自你的收藏偏好。")
        executed = [tool for tool, _, _ in service.seen_executions]
        self.assertIn("list_recommendation_runs", executed)
        self.assertNotIn("generate_recommendation", executed)
        self.assertNotIn("generate_inferred_recommendation", executed)
        self.assertNotIn("discover_catalog_tracks", executed)
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled(
                _select_system_prompt("为什么推荐这些？", DEFAULT_SYSTEM_PROMPT),
                PROVIDER_TOOL_SCHEMAS,
            )
        )

    def test_explanation_prompt_is_base_plus_referent_plus_evidence_module(self) -> None:
        # P20-Fix02: why-family = BASE + referent (current batch is the referent).
        # P20-Fix03: + the EXPLANATION grounding module (evidence-only reasons).
        # No recommendation-clause rules: those govern producing a NEW batch.
        self.assertEqual(
            _S4_EXPLANATION_PROMPT,
            _compose_system_prompt(
                (_S4_MODULE_BASE, _S4_MODULE_REFERENT, _S4_MODULE_EXPLANATION)
            ),
        )
        for excluded in (
            "推荐结果输出格式", "必须调用生成推荐工具", "目录发现取舍",
            "试听是非阻塞的", "播放意图绝不自动降级成试听",
        ):
            self.assertNotIn(excluded, _S4_EXPLANATION_PROMPT)
        self.assertLess(len(_S4_EXPLANATION_PROMPT), len(_S4_RECOMMENDATION_PROMPT))

    def test_full_fallback_stays_byte_identical_without_explanation_module(self) -> None:
        # P20-Fix03: the explanation module is deliberately ABSENT from the
        # all-modules tuple, so the grounding rules never reach the full
        # fallback (pin refreshed to 8,766 chars / sha256 e7137619... by
        # P20-PerfFix02's discovery-budget clause, not by explanation). It only
        # loads on classified why-turns, which never fall back to the full
        # prompt in practice -- and when they do (S4 composition), it joins
        # the full set additively instead of inflating every other task.
        self.assertNotIn(_S4_MODULE_EXPLANATION, _S4_ALL_MODULES)

    def test_explanation_module_grounds_and_bans_invented_reasons(self) -> None:
        module = _compose_system_prompt((_S4_MODULE_EXPLANATION,))
        # Grounding of every reason in the evidence the reader actually
        # delivered (mechanism/basis/note and queried records).
        for required in (
            "只允许来自其 evidence 字段",
            "mechanism、basis",
            "暂无偏好匹配证据",
        ):
            self.assertIn(required, module)
        # Artist identity: verbatim from the item or canonical entity; the
        # LLM must never re-translate an identity it did not look up.
        for required in (
            "严禁把艺人自行翻译成另一身份",
            "不得把 Eileen Yo 说成别的艺人",
        ):
            self.assertIn(required, module)
        # Scores are internal ranking, never reasons -- the ban wording must
        # exist verbatim so the assembled prompt forbids the misuse.
        for banned in (
            "满分", "100% 匹配", "高度吻合",
        ):
            self.assertIn(banned, module)
        # No music-encyclopedia inventions from model knowledge.
        for banned in (
            "天后天王", "代表作",
        ):
            self.assertIn(banned, module)
        # Inferred-only basis must be honestly limited, not upgraded.
        self.assertIn("偏好推断出来的", module)


class S5PrefetchGateTest(ProviderAgentLoopTest):
    """S5 (round reduction): the recommendation read prefetch fires ONLY on
    ordinary recommendation runs -- the same classification S4 already made.

    Non-recommendation families must never execute any S5 prefetch anchor read.
    Named formal play is the one intentional exception to the old "zero client
    calls" shape: it now performs one code-owned ``search_library_tracks``
    target-resolution read before provider planning. That read is not S5
    recommendation prefetch. Provider round 1 must still begin with the bare
    user text for every case covered here."""

    def _assert_no_prefetch(
        self,
        provider: FakeProvider,
        client: RecordingClient,
        text: str,
        *,
        expected_recorded_delta: tuple[str, ...] = (),
    ) -> None:
        loop = ProviderAgentLoop(provider, client, PROVIDER_TOOL_SCHEMAS)
        recorded_before = len(client.recorded)
        result = loop.run(text)
        recorded_delta = tuple(client.recorded[recorded_before:])
        self.assertEqual(recorded_delta, expected_recorded_delta)
        self.assertFalse(set(recorded_delta) & set(_S5_PREFETCH_TOOL_NAMES))
        call = provider.calls[-1]
        messages = call["messages"]
        if not expected_recorded_delta:
            self.assertEqual(len(messages), 1)
        else:
            self.assertGreaterEqual(len(messages), 1)
        self.assertEqual(messages[0].text, text)
        self.assertEqual(messages[0].tool_calls, None)
        self.assertIsNone(result.trace)

    def test_prefetch_gate_accepts_exactly_the_s4_recommendation_run(self) -> None:
        # Direct private-function pin: same prompt + full rec group is the only
        # accepted combination (the rec group membership itself is separately
        # covered below by the wire-shape tests).
        rec_tools = project_tools(_RECOMMENDATION_TOOL_NAMES)
        self.assertTrue(
            _s5_prefetch_recommendation_reads_enabled(
                _S4_RECOMMENDATION_PROMPT, rec_tools
            )
        )
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled(
                DEFAULT_SYSTEM_PROMPT, rec_tools
            )
        )
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled(
                _S4_RECOMMENDATION_PROMPT, project_tools(_PLAYBACK_TOOL_NAMES)
            )
        )
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled(_S4_RECOMMENDATION_PROMPT, [])
        )
        subset = [tool for tool in rec_tools if tool.name != "list_feedback_observations"]
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled(_S4_RECOMMENDATION_PROMPT, subset)
        )
        self.assertFalse(
            _s5_prefetch_recommendation_reads_enabled("CUSTOM NON-DEFAULT PROMPT", rec_tools)
        )
        for name in _S5_PREFETCH_TOOL_NAMES:
            self.assertIn(name, _RECOMMENDATION_TOOL_NAMES)

    def test_non_recommendation_families_never_prefetch(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        # One FakeProvider response per run. Assert on each run's client-call
        # delta so the named formal-play resolution read cannot contaminate the
        # zero-call expectation for the families that follow it.
        provider = FakeProvider(
            [text_response("好的。") for _ in range(5)]
        )
        client = RecordingClient(service, [])
        cases = (
            ("你好", ()),                         # plain chat
            ("播放 夜曲", ("search_library_tracks",)),  # named formal play
            ("试听这首", ()),                     # preview family
            ("喜欢第二首", ()),                   # feedback family
            ("找一些新的歌", ()),                 # discovery family
        )
        for text, expected_recorded_delta in cases:
            self._assert_no_prefetch(
                provider,
                client,
                text,
                expected_recorded_delta=expected_recorded_delta,
            )

    def test_custom_system_prompt_never_prefetches(self) -> None:
        # A caller-configured prompt passes through S4's selector untouched and
        # can never equal the module constant -- the gate is prompt-identity,
        # not text classification alone.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        client = RecordingClient(service, [])
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(system_prompt="CUSTOM NON-DEFAULT PROMPT"),
        )
        loop.run("推荐几首歌")
        self.assertEqual(client.recorded, [])
        self.assertEqual(len(provider.calls[0]["messages"]), 1)


class _FailOnceClient(RecordingClient):
    """S5 fail-safe probe: raises on the first N executions, then behaves like
    a plain RecordingClient (empty queue)."""

    def __init__(self, service: SharedAgentService, fail_count: int = 1) -> None:
        super().__init__(service, [])
        self.remaining_failures = fail_count

    def call(self, tool, payload, **kwargs):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("prefetch boom")
        return super().call(tool, payload, **kwargs)


class S5PrefetchWireTest(ProviderAgentLoopTest):
    """S5: the prefetch's exact wire shape -- a synthetic round-0 assistant
    tool-call message + user tool-result message, the same envelopes a real
    read round produces, before provider round 1."""

    def _call(self, name: str) -> ProviderToolCall:
        return ProviderToolCall(call_id=f"call_{name}", name=name, arguments="{}")

    def test_wire_shape_and_execution_order(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider([text_response("好的。")])
        client = RecordingClient(service, s5_prefetch_padding())
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("推荐几首歌")
        # Executions: exactly the three deterministic reads, in order, once.
        self.assertEqual(client.recorded, S5_PREFETCH_RECORDED)
        self.assertEqual(result.rounds, 1)
        # The provider-requested execution record stays empty: the prefetch is
        # loop orchestration, not something the model asked for.
        self.assertEqual(result.tool_executions, ())
        messages = provider.calls[0]["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].text, "推荐几首歌")
        self.assertIsNone(messages[0].tool_calls)
        self.assertIsNone(messages[0].tool_results)
        # Synthetic assistant round: the three reads as tool calls with the
        # same quiet {} payload the provider would send.
        self.assertEqual(messages[1].text, None)
        self.assertIsNone(messages[1].tool_results)
        self.assertEqual(
            [call.name for call in messages[1].tool_calls], S5_PREFETCH_RECORDED
        )
        self.assertEqual(
            [call.arguments for call in messages[1].tool_calls], ["{}", "{}", "{}"]
        )
        self.assertEqual(
            [call.call_id for call in messages[1].tool_calls],
            [f"s5_prefetch_{index}_{name}" for index, name in enumerate(S5_PREFETCH_RECORDED, start=1)],
        )
        # The paired tool-result message answers each call with the SAME
        # envelope a provider-requested execution would have delivered.
        self.assertEqual(messages[2].text, None)
        self.assertIsNone(messages[2].tool_calls)
        self.assertEqual(
            [item.call_id for item in messages[2].tool_results],
            [call.call_id for call in messages[1].tool_calls],
        )
        for content in (item.content for item in messages[2].tool_results):
            envelope = json.loads(content)
            self.assertEqual(envelope["outcome"], "ok")
            self.assertIsNone(envelope["error_code"])
            self.assertIsNone(envelope["error_message"])
            self.assertFalse(envelope["replayed"])
            self.assertEqual(envelope["payload"], {"ok": True})
        # Trace: the three read executions are present with round_index=0
        # (provider rounds are 1-based), ordered, executed, and never hidden.
        self.assertIsNotNone(result.trace)
        self.assertEqual(len(result.trace.rounds), 1)
        self.assertEqual(result.trace.rounds[0].messages_count, 3)
        self.assertEqual(result.trace.rounds[0].tool_calls_count, 0)
        measures = result.trace.tools
        self.assertEqual(len(measures), 3)
        for measure, name in zip(measures, S5_PREFETCH_RECORDED):
            self.assertEqual(measure.round_index, 0)
            self.assertEqual(measure.name, name)
            self.assertTrue(measure.executed)
            self.assertEqual(measure.outcome, "ok")
            self.assertEqual(measure.arguments, {})
            self.assertFalse(measure.truncated)
            self.assertFalse(measure.replayed)

    def test_prefetch_failure_is_all_or_nothing_and_fail_open(self) -> None:
        # A client failure on ANY prefetch read aborts the whole prefetch: no
        # synthetic pair is injected, nothing is cached, and the provider
        # collects the reads itself exactly as before S5.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [tool_response([self._call("get_active_context")]), text_response("好的。")]
        )
        client = _FailOnceClient(service)
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("推荐几首歌")
        # Exactly one execution: the provider's own get_active_context in
        # round 1 -- the aborted prefetch left nothing behind.
        self.assertEqual(client.recorded, ["get_active_context"])
        self.assertEqual(result.rounds, 2)
        # Round 1 went to the provider with ONLY the user text; the round-2
        # conversation carries the provider's real tool round.
        self.assertEqual(len(provider.calls[0]["messages"]), 1)
        self.assertEqual(provider.calls[0]["messages"][0].text, "推荐几首歌")
        self.assertEqual(len(provider.calls[1]["messages"]), 3)
        # Trace: no round-0 records -- the abort left no measurement either.
        self.assertEqual([record.round_index for record in result.trace.tools], [1])
        self.assertTrue(result.trace.tools[0].executed)

    def test_provider_rereads_are_answered_from_the_read_cache(self) -> None:
        # The provider re-requests all three reads plus a generation in its
        # first round: the three reads are served from the prefetch cache with
        # identical content and ZERO second executions -- only the generation
        # reaches the client.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [
                tool_response(
                    [
                        self._call("get_active_context"),
                        self._call("list_recommendation_runs"),
                        self._call("list_feedback_observations"),
                        self._call("generate_recommendation"),
                    ]
                ),
                text_response("已经为你准备好了这批推荐。"),
            ]
        )
        client = RecordingClient(
            service, s5_prefetch_padding() + [ok_generation_batch("generate_recommendation")]
        )
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.final_text, "已经为你准备好了这批推荐。")
        measures = result.trace.tools
        self.assertEqual(
            [record.name for record in measures],
            S5_PREFETCH_RECORDED
            + S5_PREFETCH_RECORDED
            + ["generate_recommendation"],
        )
        for index, name in enumerate(S5_PREFETCH_RECORDED):
            cached = measures[3 + index]
            self.assertEqual(cached.round_index, 1)
            self.assertEqual(cached.name, name)
            self.assertEqual(cached.outcome, "cached")
            self.assertFalse(cached.executed)
            self.assertIsNone(cached.duration_ms)
            # Byte-identical delivery: the cache answer IS the prefetch answer.
            self.assertEqual(
                cached.delivered_result_chars,
                measures[index].delivered_result_chars,
            )
        # The generation execution came after the cache answers in round 1.
        self.assertEqual(measures[6].round_index, 1)
        self.assertEqual(measures[6].outcome, "ok")
        self.assertTrue(measures[6].executed)

    def test_generation_in_the_first_provider_round_yields_two_rounds(self) -> None:
        # The target shape live B should achieve: reads already answered, the
        # provider generates in round 1 and answers in round 2 -- no read
        # round in between.
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        provider = FakeProvider(
            [
                tool_response([self._call("generate_recommendation")]),
                text_response("为你准备好了这批推荐。"),
            ]
        )
        client = RecordingClient(
            service, s5_prefetch_padding() + [ok_generation_batch("generate_recommendation")]
        )
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(instrument=True),
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.final_text, "为你准备好了这批推荐。")
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        self.assertEqual([record.round_index for record in result.trace.rounds], [1, 2])


class S5GenerationTailPolicyTest(ProviderAgentLoopTest):
    """S5: generation-family tool results get their own 8,000-char bound with a
    deterministic tail policy -- oversized generation payloads drop the
    ``encoded_result`` scoring/provenance tail from the MODEL-VISIBLE payload
    (items survive whole), still-oversized payloads fall back to the standard
    head-preview marker, and the trace's raw size stays honest on the true
    service payload. Non-generation tools keep the 2,000-char bound."""

    def _execute(
        self, result: AgentToolResult, tool: str = "generate_recommendation"
    ):
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, [result])
        loop = ProviderAgentLoop(provider=FakeProvider([]), client=client, tools=PROVIDER_TOOL_SCHEMAS)
        call = ProviderToolCall(call_id="call_1", name=tool, arguments="{}")
        return loop._execute_tool_call(call)

    def test_small_batch_with_encoded_result_delivers_whole(self) -> None:
        payload = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 2,
            "items": [
                {"name": "夜曲", "artist_name": "周杰伦"},
                {"name": "七里香", "artist_name": "周杰伦"},
            ],
            "encoded_result": json.dumps(
                {"scoring": {"orders": [{"route": "direct_positive"}]}}
            ),
        }
        raw_text = json.dumps(payload, ensure_ascii=False)
        self.assertLess(len(raw_text), _GENERATION_RESULT_MAX_CHARS)
        measure = self._execute(
            AgentToolResult(
                request_id="req_55555555-5555-4555-8555-555555555555",
                tool="generate_recommendation",
                outcome=AgentToolOutcome.OK,
                payload=payload,
                error_code=None,
                error_message=None,
                completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
                replayed=False,
            )
        )
        self.assertFalse(measure.truncated)
        self.assertEqual(measure.raw_result_chars, len(raw_text))
        # The whole payload -- including the scoring tail -- reaches the model.
        self.assertIn('"encoded_result"', measure.content)
        self.assertIn("夜曲", measure.content)

    def test_oversized_generation_drops_encoded_result_but_keeps_items(self) -> None:
        head = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 3,
            "items": [
                {"name": "夜曲", "artist_name": "周杰伦", "playback": {"route": "preview_only"}}
                for _ in range(3)
            ],
        }
        payload = dict(head)
        payload["encoded_result"] = "x" * 20000
        raw_text = json.dumps(payload, ensure_ascii=False)
        head_text = json.dumps(head, ensure_ascii=False)
        self.assertGreater(len(raw_text), _GENERATION_RESULT_MAX_CHARS)
        self.assertLess(len(head_text), _GENERATION_RESULT_MAX_CHARS)
        measure = self._execute(
            AgentToolResult(
                request_id="req_55555555-5555-4555-8555-555555555555",
                tool="generate_recommendation",
                outcome=AgentToolOutcome.OK,
                payload=payload,
                error_code=None,
                error_message=None,
                completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
                replayed=False,
            )
        )
        # Honest raw size on the TRUE payload, even though the model saw less.
        self.assertEqual(measure.raw_result_chars, len(raw_text))
        self.assertFalse(measure.truncated)
        # The scoring tail is gone from the model-visible text; items survive.
        self.assertNotIn("encoded_result", measure.content)
        self.assertIn("夜曲", measure.content)

    def test_still_oversized_generation_falls_back_to_head_preview(self) -> None:
        # Items alone overflow the family bound: standard head-preview marker.
        long_name = (
            "夜曲与它的现场版本以及在十一月发布的混音加长豪华特别版录音"
            "以及那场在淡江体育场举行的从未发行过的暖场即兴段落" * 8
        )
        head = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 20,
            "items": [
                {"name": long_name, "artist_name": "周杰伦"} for _ in range(20)
            ],
        }
        head_text = json.dumps(head, ensure_ascii=False)
        self.assertGreater(len(head_text), _GENERATION_RESULT_MAX_CHARS)
        measure = self._execute(
            AgentToolResult(
                request_id="req_55555555-5555-4555-8555-555555555555",
                tool="generate_recommendation",
                outcome=AgentToolOutcome.OK,
                payload=head,
                error_code=None,
                error_message=None,
                completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
                replayed=False,
            )
        )
        self.assertTrue(measure.truncated)
        self.assertEqual(measure.raw_result_chars, len(head_text))
        envelope = json.loads(measure.content)
        self.assertEqual(
            envelope["payload"],
            {"truncated": True, "preview": head_text[: _GENERATION_RESULT_MAX_CHARS]},
        )

    def test_generation_error_envelope_is_untouched(self) -> None:
        # Failure payloads (None) keep their error envelope byte-for-byte: the
        # tail policy only applies to successful dict payloads.
        measure = self._execute(failed_generation_result("generate_recommendation"))
        self.assertFalse(measure.truncated)
        self.assertEqual(measure.raw_result_chars, 0)
        self.assertEqual(measure.outcome, "execution_error")
        self.assertEqual(measure.error_code, "generation_engine_error")
        self.assertIn('"payload": null', measure.content)
        self.assertIn("generation_engine_error", measure.content)

    def test_non_generation_tools_keep_the_2000_char_bound(self) -> None:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        big = {"runs": ["x" * 60 for _ in range(50)]}  # ~3.2KB raw, > 2000
        raw_text = json.dumps(big, ensure_ascii=False)
        self.assertGreater(len(raw_text), 2000)
        client = RecordingClient(
            service,
            [ok_read_result("list_recommendation_runs")]  # placeholder replaced below
        )
        client._results = [
            AgentToolResult(
                request_id="req_55555555-5555-4555-8555-555555555555",
                tool="list_recommendation_runs",
                outcome=AgentToolOutcome.OK,
                payload=big,
                error_code=None,
                error_message=None,
                completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
                replayed=False,
            )
        ]
        loop = ProviderAgentLoop(provider=FakeProvider([]), client=client, tools=PROVIDER_TOOL_SCHEMAS)
        measure = loop._execute_tool_call(
            ProviderToolCall(call_id="call_1", name="list_recommendation_runs", arguments="{}")
        )
        self.assertTrue(measure.truncated)
        self.assertEqual(measure.raw_result_chars, len(raw_text))
        envelope = json.loads(measure.content)
        self.assertEqual(
            envelope["payload"], {"truncated": True, "preview": raw_text[:2000]}
        )


class P20Fix03RunReaderTailPolicyTest(ProviderAgentLoopTest):
    """P20-Fix03: the run detail reader (get_recommendation_run) joins the
    generation-family tail policy at a READER-bound 6,000 chars (targeted
    policy, not a global raise of the 2,000-char default). Oversized reader
    payloads drop the encoded_result tail from the MODEL-VISIBLE payload so
    all per-item evidence heads survive whole; still-oversized heads fall
    back to the standard head-preview marker; the service/MCP payload keeps
    encoded_result untouched (this class only shapes what the model sees)."""

    def _execute(self, result: AgentToolResult, tool: str = "get_recommendation_run"):
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, [result])
        loop = ProviderAgentLoop(provider=FakeProvider([]), client=client, tools=PROVIDER_TOOL_SCHEMAS)
        call = ProviderToolCall(call_id="call_1", name=tool, arguments="{}")
        return loop._execute_tool_call(call)

    def _ok(self, payload: dict) -> AgentToolResult:
        return AgentToolResult(
            request_id="req_55555555-5555-4555-8555-555555555555",
            tool="get_recommendation_run",
            outcome=AgentToolOutcome.OK,
            payload=payload,
            error_code=None,
            error_message=None,
            completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
            replayed=False,
        )

    def _evidence_items(self, count: int) -> list[dict]:
        names = ["夜曲", "七里香", "Futarino Onna", "白鸟", "栋青南"]
        return [
            {
                "position": pos,
                "name": names[pos - 1],
                "artist_name": "周杰伦",
                "target_kind": "track",
                "score_total": 1.0,
                "evidence": {
                    "mechanism": "目录推断",
                    "basis": [{"kind": "genre", "label": "J-Pop", "provenance": "推断"}],
                },
            }
            for pos in range(1, count + 1)
        ]

    def test_five_evidence_items_survive_whole_with_encoded_tail(self) -> None:
        # The P20-Fix03 scene: 5 evidence-bearing items under 6,000 chars of
        # head, but an encoded_result tail pushes the raw payload past the
        # reader bound -- the tail is dropped so every item + its evidence
        # reaches the model intact, and truncated stays False (no fallback).
        head = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 5,
            "items": self._evidence_items(5),
        }
        payload = dict(head)
        payload["encoded_result"] = "x" * 12000
        raw_text = json.dumps(payload, ensure_ascii=False)
        head_text = json.dumps(head, ensure_ascii=False)
        self.assertGreater(len(raw_text), _RUN_READER_RESULT_MAX_CHARS)
        self.assertLess(len(head_text), _RUN_READER_RESULT_MAX_CHARS)
        measure = self._execute(self._ok(payload))
        self.assertEqual(measure.raw_result_chars, len(raw_text))  # honest raw
        self.assertFalse(measure.truncated)
        self.assertNotIn("encoded_result", measure.content)
        for name in ("夜曲", "七里香", "Futarino Onna", "白鸟", "栋青南"):
            self.assertIn(name, measure.content)
        # Evidence is delivered per item, not partially cut: every mechanism
        # and basis label survives the projection.
        self.assertEqual(measure.content.count('"mechanism": "目录推断"'), 5)
        self.assertEqual(measure.content.count('"label": "J-Pop"'), 5)

    def test_small_reader_payload_keeps_encoded_result(self) -> None:
        """Under the reader bound everything -- encoded_result included --
        reaches the model (deliver-whole semantics, same as generation)."""
        payload = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 2,
            "items": self._evidence_items(2),
            "encoded_result": "z" * 500,
        }
        raw_text = json.dumps(payload, ensure_ascii=False)
        self.assertLess(len(raw_text), _RUN_READER_RESULT_MAX_CHARS)
        measure = self._execute(self._ok(payload))
        self.assertFalse(measure.truncated)
        self.assertEqual(measure.raw_result_chars, len(raw_text))
        self.assertIn('"encoded_result"', measure.content)
        self.assertIn("夜曲", measure.content)

    def test_reader_head_overflow_falls_back_to_head_preview(self) -> None:
        long_note = "这是一段超长的探索性说明" * 60
        head = {
            "run_id": "rcm_11111111-1111-4111-8111-111111111111",
            "item_count": 20,
            "items": [
                {
                    "position": pos,
                    "name": long_note,
                    "artist_name": "周杰伦",
                    "evidence": {"mechanism": "探索性新发现",
                                 "basis": [], "note": long_note},
                }
                for pos in range(1, 21)
            ],
        }
        head_text = json.dumps(head, ensure_ascii=False)
        self.assertGreater(len(head_text), _RUN_READER_RESULT_MAX_CHARS)
        measure = self._execute(self._ok(head))
        self.assertTrue(measure.truncated)
        self.assertEqual(measure.raw_result_chars, len(head_text))
        envelope = json.loads(measure.content)
        self.assertEqual(
            envelope["payload"],
            {
                "truncated": True,
                "preview": head_text[: _RUN_READER_RESULT_MAX_CHARS],
            },
        )

    def test_reader_error_envelope_is_untouched(self) -> None:
        failed = AgentToolResult(
            request_id="req_55555555-5555-4555-8555-555555555555",
            tool="get_recommendation_run",
            outcome=AgentToolOutcome.EXECUTION_ERROR,
            payload=None,
            error_code="history_read_error",
            error_message=None,
            completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
            replayed=False,
        )
        measure = self._execute(failed)
        self.assertFalse(measure.truncated)
        self.assertEqual(measure.outcome, "execution_error")
        self.assertEqual(measure.error_code, "history_read_error")
        self.assertIn('"payload": null', measure.content)

    def test_reader_bound_is_targeted_not_global(self) -> None:
        # The 6,000-char reader bound coexists with the 8,000-char generation
        # bound and the untouched 2,000-char default for every other tool:
        # this is a targeted policy for explanation-needed results only.
        self.assertEqual(_RUN_READER_RESULT_MAX_CHARS, 6000)
        self.assertEqual(_GENERATION_RESULT_MAX_CHARS, 8000)
        self.assertEqual(_MAX_TOOL_RESULT_CHARS, 2000)
        # The listener-heavy reader is never in the generation family; the
        # generation/everything-else bound can only be validated end-to-end
        # by the family tests above (its 2,000 default is pinned next).
        big = {"runs": ["x" * 60 for _ in range(50)]}  # ~3.2KB raw, > 2000
        raw_text = json.dumps(big, ensure_ascii=False)
        self.assertGreater(len(raw_text), 2000)
        self.assertLess(len(raw_text), _RUN_READER_RESULT_MAX_CHARS)
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(
            service,
            [
                AgentToolResult(
                    request_id="req_55555555-5555-4555-8555-555555555555",
                    tool="list_recommendation_runs",
                    outcome=AgentToolOutcome.OK,
                    payload=big,
                    error_code=None,
                    error_message=None,
                    completed_at=datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc),
                    replayed=False,
                )
            ],
        )
        loop = ProviderAgentLoop(provider=FakeProvider([]), client=client, tools=PROVIDER_TOOL_SCHEMAS)
        measure = loop._execute_tool_call(
            ProviderToolCall(call_id="call_1", name="list_recommendation_runs", arguments="{}")
        )
        self.assertTrue(measure.truncated)
        self.assertEqual(
            json.loads(measure.content)["payload"],
            {"truncated": True, "preview": raw_text[:2000]},
        )


class DeterministicRecommendationPresentationTest(ProviderAgentLoopTest):
    """P20-Fix10: after a successful generation the user-visible final answer
    IS the deterministic presenter's rendering of the authoritative payload --
    the provider's final free text is dropped wholesale on that scene (mandate
    §十三), so planning narration, internal terms, duplicate lists,
    encyclopedia filler, and current-playing causality in the model's answer
    are structurally unreachable (§十九 A--F). Failure scenes keep the
    provider's fail-honest text and the explanation scene is untouched
    (§十四/§十六); the renderer is pure (§十五) and the loop's final text is
    the single choke point both CLI chat/chat-session and web /api/chat
    present verbatim (§十七)."""

    ROCK_INFERRED = {"kind": "genre", "label": "Rock", "provenance": "推断"}
    ROCK_DIRECT = {"kind": "genre", "label": "Rock", "provenance": "直接"}
    TRACK_SELF_DIRECT = {"kind": "track", "label": "夜曲", "provenance": "直接"}

    def _presentation_loop(
        self,
        provider: FakeProvider,
        results: list[AgentToolResult] | None = None,
        *,
        max_tool_rounds: int = 8,
    ) -> tuple[RecordingClient, ProviderAgentLoop]:
        service = self._service(AgentClientPolicy.FULL)
        self.addCleanup(service.close)
        client = RecordingClient(service, results=results)
        loop = ProviderAgentLoop(
            provider,
            client,
            PROVIDER_TOOL_SCHEMAS,
            config=ProviderLoopConfig(max_tool_rounds=max_tool_rounds),
        )
        return client, loop

    @staticmethod
    def _generate_call(call_id: str, tool: str = "generate_recommendation") -> ProviderToolCall:
        return ProviderToolCall(
            call_id, tool,
            json.dumps({"target_ids": ["trk_11111111-1111-4111-8111-111111111111"]}),
        )

    def _batch_with(self, items: list[dict], tool: str = "generate_recommendation") -> AgentToolResult:
        return ok_generation_evidence_batch(tool, items)

    def test_plain_success_output_is_the_renderer_not_the_provider_text(self) -> None:
        # A: five items, a provider final text full of narration -- the user
        # output is the deterministic rendering alone, produced with zero
        # extra tool calls beyond the single generation (§十五 purity).
        batch = [
            evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT]),
            evidence_item("City Lights", basis=[self.ROCK_INFERRED]),
            evidence_item("Mono", basis=[self.ROCK_DIRECT]),
            evidence_item("Old Favorite", basis=[self.ROCK_DIRECT]),
            evidence_item("Note Only"),
        ]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response(
                "推荐已成功生成。让我根据结果整理最终回答。\n\n"
                "本批包含 5 首，其中部分是本次新发现……\n"
                "为你找到了这几首：\n1. 夜曲 — 测试艺人\n2. City Lights\n"
                "3. Mono\n4. Old Favorite\n5. Note Only"
            ),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertTrue(result.final_text.startswith("为你推荐这 5 首："), result.final_text)
        self.assertNotIn("让我根据结果整理最终回答", result.final_text)
        self.assertNotIn("推荐已成功生成", result.final_text)
        self.assertNotIn("为你找到了这几首", result.final_text)
        self.assertNotIn("本批包含", result.final_text)
        # The presenter executed zero further tools; only the generation ran.
        self.assertEqual(
            client.recorded, S5_PREFETCH_RECORDED + ["generate_recommendation"]
        )
        self.assertEqual(result.rounds, 2)
        self.assertFalse(result.rounds_capped)

    def test_fresh_batch_marks_exactly_the_fresh_items(self) -> None:
        # B: 3 fresh + 2 non-fresh via the inferred channel; fresh identity is
        # per item, internal fresh terms never surface, and an all-preview_only
        # batch carries the 30-second cue. Extra inferred-channel fields
        # (score_total / direct_state / label) are outside the contract and
        # must not leak either.
        batch = [
            evidence_item("City Lights", basis=[self.ROCK_INFERRED],
                          fresh_this_request=True, route="preview_only",
                          label="Rock", score_total=0.87, direct_state="unknown"),
            evidence_item("Summer Rain", basis=[self.ROCK_INFERRED],
                          fresh_this_request=True, route="preview_only"),
            evidence_item("Night Unknown", fresh_this_request=True, route="preview_only"),
            evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT],
                          route="preview_only"),
            evidence_item("Old Favorite", basis=[self.ROCK_DIRECT], route="preview_only"),
        ]
        provider = FakeProvider([
            tool_response([self._generate_call("c1", "generate_inferred_recommendation")]),
            text_response("都整理好了。"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding()
            + [self._batch_with(batch, "generate_inferred_recommendation")],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text.count("这是本次新发现"), 3, result.final_text)
        for banned in ("fresh=true", "fresh=false", "novel", "fresh_this_request"):
            self.assertNotIn(banned, result.final_text)
        for banned in ("score_total", "0.87", "direct_state"):
            self.assertNotIn(banned, result.final_text)
        self.assertTrue(
            result.final_text.endswith(
                "这批曲目都只能试听 30 秒。需要试听哪一首，直接告诉我。"
            ),
            result.final_text,
        )

    def test_planning_leakage_is_structurally_unreachable(self) -> None:
        # C: the exact live-UAT leakage phrases in the provider's final text
        # never reach the user -- not via sanitizer deletion but because the
        # provider final text is not the output at all on a success scene.
        batch = [
            evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT]),
            evidence_item("City Lights", basis=[self.ROCK_INFERRED],
                          fresh_this_request=True),
        ]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response(
                "推荐已成功生成。让我根据结果整理最终回答。\n"
                "让我按照证据纪律整理理由：\n"
                "Surrealila ... novel，fresh。\n"
                "夜聊 ... fresh=false\n"
                "推荐依据 依据 是……\n"
                "让我按照证据纪律整理理由。"
            ),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        for leaked in (
            "让我根据结果整理最终回答",
            "让我按照证据纪律整理理由",
            "novel，fresh",
            "fresh=false",
            "推荐依据 依据 是",
            "Surrealila",
            "夜聊",
        ):
            self.assertNotIn(leaked, result.final_text)
        self.assertTrue(result.final_text.startswith("为你推荐这 2 首："), result.final_text)

    def test_duplicate_list_is_impossible(self) -> None:
        # D: the provider lists the same tracks once; the renderer lists them
        # once -- each track name appears exactly once in the final output.
        names = ["甲夜曲", "乙城市", "丙摇滚", "丁蓝调", "戊爵士"]
        batch = [evidence_item(name, basis=[self.ROCK_INFERRED]) for name in names]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response(
                "为你找到了这几首：\n" + "\n".join(
                    f"{position}. {name}" for position, name in enumerate(names, 1)
                )
            ),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        for name in names:
            self.assertEqual(result.final_text.count(name), 1, result.final_text)

    def test_current_playing_causality_from_the_model_is_unreachable(self) -> None:
        # E: the model's answer claims a current-playing basis the evidence
        # never recorded -- dropped wholesale with the model text.
        batch = [
            evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT]),
            evidence_item("City Lights", basis=[self.ROCK_INFERRED]),
        ]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response("我基于当前播放的 YOASOBI 曲目和你的偏好生成了这批推荐：……"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertNotIn("当前播放", result.final_text)
        self.assertNotIn("YOASOBI", result.final_text)
        self.assertNotIn("基于你正在听", result.final_text)

    def test_encyclopedia_filler_from_the_model_is_unreachable(self) -> None:
        # F: 经典金曲 / 动漫主题曲 / 治愈 style filler in the model's answer
        # never reaches the rendered surface.
        batch = [evidence_item("夜曲", basis=[self.ROCK_INFERRED])]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response("这些都是经典金曲和动漫主题曲，非常治愈，适合放空。"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        for banned in ("经典金曲", "动漫主题曲", "治愈", "适合放空"):
            self.assertNotIn(banned, result.final_text)

    def test_direct_evidence_copy_renders(self) -> None:
        # G: a real direct row renders the direct claim, nothing upgraded.
        batch = [evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT])]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("好了。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertIn("这首来自你对这首曲目本身的已有偏好。", result.final_text)

    def test_inferred_genre_copy_renders_without_upgrade(self) -> None:
        # H: inferred Rock reads only 「按 Rock 方向推断出来」 -- never 已有偏好.
        batch = [evidence_item("City Lights", basis=[self.ROCK_INFERRED])]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("好了。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertIn("这首按 Rock 方向推断出来。", result.final_text)
        self.assertNotIn("已有偏好", result.final_text)

    def test_fresh_without_evidence_is_honest(self) -> None:
        # I: fresh + zero basis reads the fixed honest sentence.
        batch = [evidence_item("Night Unknown", fresh_this_request=True)]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("好了。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertIn("这是本次新发现，目前没有更直接的偏好匹配证据。", result.final_text)

    def test_one_item_success_shows_one_item_honestly(self) -> None:
        # K: a 1-item success renders exactly one item (§十四: no padding).
        batch = [evidence_item("Only One", basis=[self.ROCK_DIRECT])]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("只有一首也列出来了。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertTrue(result.final_text.startswith("为你推荐这 1 首："), result.final_text)
        self.assertEqual(result.final_text.count("1. "), 1)
        self.assertNotIn("2. ", result.final_text)

    def test_failure_scene_keeps_the_provider_fail_honest_text(self) -> None:
        # L: no successful generation -> no success renderer; the provider's
        # fail-honest answer is the user output unchanged.
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response("抱歉，目前没有新的可推荐曲目。"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [empty_generation_result("generate_recommendation")],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text, "抱歉，目前没有新的可推荐曲目。")
        self.assertNotIn("为你推荐这", result.final_text)

    def test_ordering_is_exactly_the_renderer_order(self) -> None:
        # M: the rendered order is the payload's selected order, pinned by
        # relative positions (fresh and non-fresh mixed).
        batch = [
            evidence_item("第三首", basis=[self.ROCK_INFERRED]),
            evidence_item("第一首", basis=[self.ROCK_DIRECT],
                          fresh_this_request=True),
            evidence_item("第二首", basis=[]),
        ]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("顺序我重新排了一下：第二首、第一首、第三首。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        first = result.final_text.find("1. 第三首")
        second = result.final_text.find("2. 第一首")
        third = result.final_text.find("3. 第二首")
        self.assertLess(first, second)
        self.assertLess(second, third)
        self.assertNotIn("顺序我重新排了一下", result.final_text)

    def test_rendered_text_is_the_single_choke_point_for_all_surfaces(self) -> None:
        # N/O/P: CLI chat, CLI chat-session, and web /api/chat all present
        # ``final_text`` verbatim; this pins the exact deterministic string the
        # shared upstream produces -- one renderer, byte-identical output.
        batch = [
            evidence_item("夜曲", artist_name="测试艺人", basis=[self.TRACK_SELF_DIRECT]),
            evidence_item("City Lights", basis=[self.ROCK_INFERRED]),
        ]
        provider = FakeProvider([tool_response([self._generate_call("c1")]),
                                 text_response("你的推荐好了。")])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [self._batch_with(batch)],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(
            result.final_text,
            "为你推荐这 2 首：\n\n"
            "1. 夜曲 — 测试艺人\n这首来自你对这首曲目本身的已有偏好。\n\n"
            "2. City Lights\n这首按 Rock 方向推断出来。\n\n"
            "需要试听哪一首，直接告诉我。",
        )

    def test_round_cap_after_success_renders_instead_of_the_closeout(self) -> None:
        # A capped run that DID generate successfully ends with the rendered
        # batch, not _ROUND_CAP_CLOSEOUT (the success scene overrides the cap
        # text the same way it overrides a terminal answer).
        batch = [evidence_item("夜曲", basis=[self.TRACK_SELF_DIRECT])]
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            tool_response([ProviderToolCall("c2", "get_agent_capabilities", "{}")]),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [
                self._batch_with(batch),
                ok_read_result("get_agent_capabilities"),
            ],
            max_tool_rounds=2,
        )
        result = loop.run("推荐几首歌")
        self.assertTrue(result.rounds_capped)
        self.assertNotIn(_ROUND_CAP_CLOSEOUT, result.final_text)
        self.assertTrue(result.final_text.startswith("为你推荐这 1 首："), result.final_text)

    def test_explanation_scene_keeps_the_explanation_path(self) -> None:
        # Q: an explanation follow-up (no generation tool call) keeps the
        # provider's explanation text -- Fix03/Fix06 path untouched.
        provider = FakeProvider([
            tool_response([
                ProviderToolCall(
                    "c1", "get_recommendation_run",
                    json.dumps({"run_id": "rcm_11111111-1111-4111-8111-111111111111"}),
                ),
            ]),
            text_response("这批推荐来自你的直接偏好和 Rock 方向的推断。"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=[ok_read_result("get_recommendation_run")],
        )
        result = loop.run("为什么推荐这些？")
        self.assertEqual(result.final_text, "这批推荐来自你的直接偏好和 Rock 方向的推断。")
        self.assertNotIn("为你推荐这", result.final_text)

    def test_legacy_no_evidence_payload_keeps_the_provider_text_path(self) -> None:
        # The strict contract: a generation success whose payload items lack
        # evidence blocks (legacy replayed shapes) renders None -- the loop
        # keeps the provider's own answer instead of guessing reasons.
        provider = FakeProvider([
            tool_response([self._generate_call("c1")]),
            text_response("已为你整理好这一批：夜曲 — 测试艺人。"),
        ])
        client, loop = self._presentation_loop(
            provider,
            results=s5_prefetch_padding() + [ok_generation_batch("generate_recommendation")],
        )
        result = loop.run("推荐几首歌")
        self.assertEqual(result.final_text, "已为你整理好这一批：夜曲 — 测试艺人。")
        self.assertNotIn("为你推荐这", result.final_text)


if __name__ == "__main__":
    unittest.main()
