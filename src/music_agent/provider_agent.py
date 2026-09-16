"""P10.8: The provider tool-call loop through the sealed P09 Shared Agent boundary.

The loop turns one natural-language request into: provider round trips, P09 tool
executions through the client-owned :class:`AgentClient`, and a final natural-language
answer. P09 is never bypassed -- every tool execution is a normal AgentClient call, so
contract-version checks, replay journaling, payload validation, permission enforcement,
and the sealed capability gate all apply exactly as designed. Provider/model identity
stays metadata; no provider-specific state is created anywhere.

Loop safety:

- **Bounded rounds**: at most ``max_tool_rounds`` provider round trips; hitting the bound
  ends the loop with a deterministic fixed closeout (P17-A1: intermediate assistant
  planning text is never scavenged from history as the final answer).
- **Bounded context**: conversation history is trimmed to the most recent messages
  (system prompt always kept whole); the trim is reported.
- **Per-turn dedupe**: an identical (tool, arguments) pair inside one turn reuses the
  earlier result instead of re-executing -- a model that repeats a mutation call cannot
  apply it twice within the turn (cross-turn replay remains governed by P09's journal,
  which is the authority for request identities).
- **In-run read cache**: an identical (tool, arguments) read re-executed in a *later*
  turn of the same run returns the earlier result without another P09 call, so a model
  that re-probes a fact it already saw stops costing provider round trips and context.
  Every state-changing tool (see ``_CACHE_INVALIDATING_TOOLS``) clears the cache when it
  executes, so a cached read can never go stale across a mutation inside the run.
- **Post-generation closeout**: once a generation call returns ``ok`` this run
  (non-empty by the P09 contract -- empty batches raise and never persist), its payload
  becomes the run's single presentation authority. Further generation or Catalog
  discovery calls are answered with a deterministic synthetic refusal instead of
  executing. Ordinary recommendation runs then carry zero tool schemas. An existing
  explicit play-or-preview delegation may instead consume exactly one item from that
  payload via ``play_track`` or ``preview_catalog_track``; successful formal playback
  gets its required single ``get_now_playing`` readback, then the run becomes final-only.
  The state is per-run, so the next user message starts fresh and pre-generation
  discovery is never blocked.
- **Final-answer-only / code-owned termination (S2/P20)**: once a round provably
  completes the run's tool phase, the model cannot reopen it. Contract-pinned terminal
  actions switch to zero schemas; ordinary generation success closes generation;
  delegated playback/preview success now returns a deterministic runtime-backed reply
  (formal play only after its get_now_playing readback); and a closed feedback-verdict
  turn hands interpret -> apply -> closeout to code immediately after its one successful
  record_feedback. Failures and non-terminal reads keep their selected tool surface.
  The round count never expands merely to let the model restate a fact the runtime has
  already proved.
- **Per-task tool surfaces (S3)**: the first round no longer always carries the
  full tool set -- the run starts from a task group projected from one TurnPlan.
  Deterministic intent-router classifiers resolve stable forms first; P22 may use the
  narrow no-tool Conversation Interpreter only when that parser returns UNKNOWN.
  Plain chat (S1) gets (), the
  closed feedback forms get the feedback/learning chain + batch locating, the
  preview forms (试听- family, batch-preview phrasings and the stop residuals the
  routing table hands back) get the preview chain + batch locating, explicit
  play-intent gets the search -> play -> check-back chain with NO preview
  tools (the formal-playback boundary is structural, not wording-only: the
  model cannot self-downgrade into a preview it was never offered; the
  caller-side T14-E door remains the outer backstop), and the
  recommendation forms (closed set + 推荐点<方向>) and fresh-discovery forms share
  the recommendation chain (context reads, both generation tools, catalog
  discovery + discovery memory, preference queries, and the feedback/learning
  READS the live B2 recommendation run demonstrably consults). Every other line
  -- mixed, chained, named-song verdicts, delegation (你来决定/随便播放), 换一首,
  capability questions, every near-miss -- keeps the full set fail-safe. Group
  membership is a name-filter over the caller's tool list on the run's first
  round only; the S2 per-round switch then only ever shrinks it further. No
  schema is deleted: all 31 ``PROVIDER_TOOL_SCHEMAS`` still exist and the full
  set ships on any unclassified request.
- **Per-task system prompts (S4)**: the run's system prompt is chosen by the
  same local classifiers (``_select_system_prompt``): plain chat runs the BASE
  identity module alone, each task family runs BASE + its task modules
  (referent/batch ships with the point-at-a-track families and with
  recommendation; fresh/catalog adds the discovery module), and every
  unclassifiable line keeps the full prompt. Safety invariants that must survive a
  classifier miss live in BASE, while task-specific Library and explanation detail
  stays in its narrow module. Only the built-in
  ``DEFAULT_SYSTEM_PROMPT`` is decomposed -- a caller-configured prompt passes
  through unchanged. The full fallback is the tagged composition of the general
  modules plus those always-on BASE invariants; it is a behavioral/safety boundary,
  not merely a token optimization.
- **Recommendation read prefetch (S5)**: ordinary recommendation runs pre-read
  the three anchor reads (``get_active_context`` + ``list_recommendation_runs``
  + ``list_feedback_observations``) deterministically, ONCE, before provider
  round 1, and inject the results as a synthetic round-0 assistant/tool
  message pair -- the exact wire shape a real read round produces. The
  provider can therefore go straight to ``generate_recommendation`` in its
  first round (2 rounds total) instead of spending S4's two read rounds first
  (4). All-or-nothing and fail-open (a prefetch failure injects nothing), and
  the results land in the read cache so a provider that re-reads the same
  facts is answered without a second execution. Generation results also get
  their own 8000-char bound with ``encoded_result`` dropped first when
  oversized, so the full ordered item list of every legal batch (limit <= 20)
  reaches the model -- no re-read of the durable run is needed to obtain item
  names/order the generation itself produced.
- **Per-run discover budget**: ``discover_catalog_tracks`` is capped at
  ``max_discover_per_run`` successful executions per run (default 1). Only a real,
  non-replayed OK execution charges the budget -- same-round dedupe, the read cache,
  the post-generation closeout, invalid arguments, failed executions and the gate
  itself never do. Once the budget is spent, further discover calls get a
  deterministic synthetic refusal (``discover_budget_exhausted``) and no real Catalog
  search; the counter is per-run, so every new user message re-enables discovery.
- **Run-local Fresh provenance (S3-S3D)**: the canonical ids of every PROMOTED entry
  in a genuinely executed, non-replayed, ``ok`` discover result this run form the
  authoritative ``fresh_promoted_ids`` set -- a ``run()`` local, so each new user
  message starts empty. It is passed to the two generation tools as an internal
  execution kwarg (never into model arguments, the journal payload, or any tool
  schema), so the model cannot forge Fresh identity. Staged/skipped entries
  (including ALREADY_BOUND), synthetic refusals (budget/closeout) and failures
  never contribute. Capture reads the RAW structured result payload from the
  execution itself -- the delivered envelope (subject to the 2000-char
  ``_MAX_TOOL_RESULT_CHARS`` truncation marker) is never an authority.
- **Refusals become text**: a P09 refusal (unknown_client / permission_denied /
  not_execution_ready / invalid_request / execution_error / tool_not_supported) is fed
  back to the model as the tool result, so the final answer can explain it honestly.
- **Unparseable arguments** are fed back as an error text instead of crashing the loop.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from music_agent.action_attempt import (
    ActionAttempt,
    ActionAttemptStatus,
    PlaybackControlAttempt,
    create_playback_control_attempt,
    create_action_attempt,
    create_direct_action_attempt,
    mark_action_executing,
    record_action_execution,
    record_playback_control_execution,
    render_verified_action_result,
    render_verified_playback_control_result,
    verify_formal_play_readback,
    verify_playback_control_readback,
)
from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentToolOutcome, AgentToolResult
from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK
from music_agent.intent_router import (
    RecommendationTurnSemantics,
    TurnPlan,
    TurnPrimarySemantic,
    TurnTaskSurface,
    resolve_turn_plan,
)
from music_agent.provider_contract import (
    ChatProvider,
    ProviderError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderToolCall,
    ProviderToolResult,
    ProviderToolSchema,
)
from music_agent.provider_recommendation_policy import (
    _ResolvedRecommendationSemantics,
    _apply_recommendation_semantics,
    _artist_name_matches,
    _bind_current_track_similarity_seed,
    _cached_current_player_canonical_id,
    _cached_strict_current_player_canonical_id,
    _deterministic_generic_recovery_call,
    _deterministic_preference_fallback_calls,
    _empty_generation_diagnostics,
    _enforce_new_recommendation_freshness,
    _entity_name_key,
    _generation_payload_contains_similarity_seed,
    _generic_direct_history_exhausted,
    _generic_recommendation_result_only_current_player,
    _generic_recommendation_targets_only_current_player,
    _recommendation_scope_ids,
    _recommendation_semantics_prompt,
    _resolve_recommendation_semantics,
)
from music_agent.provider_instrumentation import (
    ProviderRoundMeasure,
    ProviderRunTrace,
    ProviderToolMeasure,
    measure_round_input,
    safe_tool_arguments,
)
from music_agent.prompts.workflow import (
    DEFAULT_SYSTEM_PROMPT,
    _S4_ALL_MODULES,
    _S4_BASE_PROMPT,
    _S4_DISCOVERY_PROMPT,
    _S4_EXPLANATION_PROMPT,
    _S4_FEEDBACK_PROMPT,
    _S4_LIBRARY_QUERY_PROMPT,
    _S4_MODULE_BASE,
    _S4_MODULE_DISCOVERY,
    _S4_MODULE_EXPLANATION,
    _S4_MODULE_FEEDBACK,
    _S4_MODULE_LIBRARY_QUERY,
    _S4_MODULE_PLAYBACK,
    _S4_MODULE_PREVIEW,
    _S4_MODULE_RECOMMENDATION,
    _S4_MODULE_REFERENT,
    _S4_PLAYBACK_PROMPT,
    _S4_PREVIEW_PROMPT,
    _S4_PROMPT_CLAUSES,
    _S4_RECOMMENDATION_PROMPT,
    _compose_system_prompt,
)
# P20-Fix10: deterministic presentation of a successful recommendation batch
# (see recommendation_presenter.py) -- the loop imports the renderer here so
# every consumer of ``final_text`` (CLI chat/chat-session, web /api/chat)
# shares the one rendered presentation at the common upstream. The renderer
# is a near-pure function of the authoritative generation payload; it fails
# closed to None and the loop keeps its ordinary text path.
from music_agent.recommendation_presenter import render_recommendation_for_user
from music_agent.selection_grant import (
    DelegatedAudioAction,
    build_selection_grant,
    delegated_action_is_authorized,
    select_delegated_audio_action,
    selection_grant_is_exhausted,
)
from music_agent.track_similarity import SimilarityExecutionContext
from music_agent.turn_interpreter import (
    TurnInterpreterContext,
    interpret_turn,
)

logger = logging.getLogger("music_agent.provider_loop")

_MAX_TOOL_RESULT_CHARS = 2000
# S5 (token-cost optimization): family-specific result bound for the two
# generation tools. Their ok payload head (run identity + counts + the full
# ordered item list with name/artist/album/playback, ~300 chars per item)
# is the only part the model consumes for the final presentation; the
# ``encoded_result`` tail is scoring/provenance internals. The bound is set
# so the head of EVERY legal batch (limit <= 20) fits undamaged: a payload
# that overflows has its ``encoded_result`` tail dropped first (see
# ``_execute_tool_call``), and only a still-oversized remainder falls back
# to the standard head-preview marker. Baseline's 2000-char bound truncated
# a 5-item payload at ~3.5 items and the model still had to present the
# rest; 8000 makes a truncated generation payload impossible in practice
# while still capping the pathological case.
_GENERATION_RESULT_MAX_CHARS = 8000
# P20-Fix03: family-specific result bound for the recommendation-run detail
# reader. An explanation turn ("为什么推荐这些？") must receive EVERY item's
# per-item evidence, never a list cut by the global 2000-char bound: the
# reader's compact head (position + identity + evidence, ~300 chars per item)
# fits whole for every legal batch (limit <= 20). The oversized
# ``encoded_result`` scoring/provenance tail is dropped from the MODEL-VISIBLE
# payload exactly like the generation family (the service payload keeps it;
# MCP consumers are untouched), and only a still-oversized head falls back to
# the standard preview marker. Not a global raise -- the 2000-char bound for
# every other non-generation tool is untouched.
_RUN_READER_RESULT_MAX_CHARS = 6000
# Every tool that changes durable/observable state in some way. Executing one of these
# invalidates all cached read results for the rest of the run, so a re-read after a
# write is always a fresh P09 call. Pure reads (queries, listings, get_* tools and
# get_now_playing) are NOT in this set and are cached per (name, arguments).
_CACHE_INVALIDATING_TOOLS: frozenset[str] = frozenset({
    "generate_recommendation",
    "generate_inferred_recommendation",
    "record_feedback",
    "apply_learning",
    "execute_write_intent",
    "discover_catalog_tracks",
    "add_catalog_to_library",
    "preview_catalog_track",
    "stop_preview",
    "play",
    "pause",
    "next_track",
    "previous_track",
    "play_track",
})
# Recommendation-producing tools share one per-run attempt budget
# (``max_generation_attempts``): a run that keeps failing to build a batch
# must stop exploring and answer instead of burning slow discovery rounds.
# P14-R4.3: attempts are counted before the cache/dedup paths, so identical
# retries count too -- the budget limits attempts, not executions.
_GENERATION_TOOL_NAMES: frozenset[str] = frozenset({
    "generate_recommendation",
    "generate_inferred_recommendation",
})
# P19 compatibility alias. Production presentation now calls
# ``generation_succeeded`` instead; keep the alias until the real repository
# is searched for external imports before deletion. Identical object, not a copy.
GENERATION_TOOL_NAMES: frozenset[str] = _GENERATION_TOOL_NAMES
# P14-R4.5: deterministic closeout returned when every allowed generation
# attempt failed (budget exhausted, or the final attempt returned empty).
# Live R4.4 verified that leaving the termination to the model after the
# budget runs out stalls the loop (rule explanation instead of an answer),
# so the loop itself ends the run with this fixed user-facing text.
_GENERATION_FAILURE_CLOSEOUT = (
    "暂时没有找到新的推荐曲目。\n"
    "\n"
    "你可以：\n"
    "1. 继续听刚才那批（这是刚才那批，不是新的推荐）\n"
    "2. 换一个方向，我重新帮你找"
)
# P19-T14-B: the ONE-sentence fallback the reply door (web shell) substitutes
# when a recommendation-shaped request finished WITHOUT any successful
# generation this run. The model's own prose -- including a hand-enumerated
# "1. 2. 3." song pseudo-list -- must never reach the user in that state.
# Distinct from ``_GENERATION_FAILURE_CLOSEOUT``: that one is the loop's
# deterministic double-failure termination (reachable only when BOTH allowed
# generation attempts failed); this one covers the gap the loop cannot see --
# the model answering a recommendation request without ever producing a batch
# (or after a single failed attempt followed by prose).
RECOMMENDATION_UNFULFILLED_FALLBACK = "暂时没有找到合适的推荐，换个方向或换一首歌再试试吧。"
# P17-A1: the deterministic closeout returned when the loop hits the round bound
# while the provider kept requesting tools. The run has no terminal answer
# message, so the history must NOT be scavenged -- scanning assistant history
# would select a tool-round preamble (intermediate planning content) as the
# final user response. The fixed text closes the run honestly instead.
_ROUND_CAP_CLOSEOUT = (
    "本轮请求处理步骤过多，没有整理出最终回复。"
    "请换一种更简单的说法再试一次（例如直接说「推荐几首歌」"
    "「播放一首正式歌曲」「暂停」或「继续播放」）。"
)
# P17-A1: a terminating message whose text is empty would otherwise surface an
# empty answer on the user-facing door; the fixed fallback keeps it honest.
_EMPTY_FINAL_ANSWER_CLOSEOUT = "这次没有生成有效回复，请再说一次。"
# P15-S4-M2-1/P20 cleanup: one deterministic refusal for every generation or
# discovery call requested after the first successful non-empty generation. The
# payload already captured is the run's single presentation authority; the next user
# message starts a new run and re-enables both phases.
_POST_GENERATION_CLOSEOUT_ERROR_CODE = "post_generation_closeout"
_POST_GENERATION_CLOSEOUT_MESSAGE = (
    "本次请求已经成功生成非空推荐批次，不需要继续生成或搜索目录；"
    "如果原始请求明确委托播放或试听，只能从该批次选择一首执行一次已授权动作；"
    "除此之外请直接基于已生成的批次整理最终用户响应；"
    "如果用户下一条消息明确提出新的搜索或方向，"
    "新的一轮请求会重新允许目录发现。"
)
_GENERIC_CURRENT_PLAYER_TARGET_ERROR_CODE = "generic_current_player_target"
_GENERIC_CURRENT_PLAYER_TARGET_MESSAGE = (
    "本轮是无显式 seed 的通用推荐；当前播放曲目只能作为上下文，"
    "不能成为唯一 target_id。请从已有偏好/推荐事实选择更广的目标范围，"
    "或先发现候选，再以 requested_count=5 重新调用生成工具。"
)
# P15-S3-S3B: the per-run Fresh Catalog discovery budget. A discover call charges
# the budget only when it genuinely executes (a real P09 call that is not a journal
# replay) AND succeeds -- same-round dedupe, the read cache, the post-generation
# closeout, invalid arguments, failed executions and the gate itself never charge.
# The counter is a run() local, so every new user message starts fresh -- the budget
# is a hard cap on real successful Apple Music Catalog searches and nothing else (no
# term cache, no cooldown, no caches at all). P20-PerfFix02: the cap is 1 successful
# search per user turn -- the live UAT fresh request ran two disjoint searches in one
# round (18.3s + 22.2s) and the second contributed zero of the final five
# recommendations. Failed searches stay free so one transient Catalog failure does
# not burn the single budget (§四-4 failure recovery).
MAX_DISCOVER_PER_RUN = 1
_DISCOVER_TOOL_NAMES: frozenset[str] = frozenset({"discover_catalog_tracks"})
_DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE = "discover_budget_exhausted"
# S2 behavioral boundary: tools whose "ok" execution provably ends the
# run's tool phase. Their provider-prompt contracts pin the immediate answer
# and forbid any follow-up call in the same run -- preview_catalog_track
# started (「直接回答已开始试听…不要重复调用确认」), preview_batch (answer from
# the returned session.state, no per-track calls), stop_preview (idempotent
# stop, answer from the result), and open_in_apple_music (the result carries
# the only official url; the answer just relays it). Ordinary recommendation
# generation is handled separately because explicit delegation has one narrowly
# bounded post-generation action state. After a round where any member here
# executes ok the loop sends every remaining round with ZERO tool schemas.
#
# Deliberately absent: play/play_track (the
# prompt REQUIRES the get_now_playing check-back after every play action),
# pause/next_track/previous_track (no explicit final-answer contract --
# chained-command phrasings keep their tools), the feedback/learning and
# write tools (follow-ups are legitimate), all reads, and every non-ok
# outcome.
_FINAL_ANSWER_EXECUTIONS: frozenset[str] = frozenset(
    {
        "preview_catalog_track",
        "preview_batch",
        "stop_preview",
        "open_in_apple_music",
    }
)

# P20 post-generation delegation: these are the only actions an already-authorized
# play-or-preview delegation may take from the current run's exact payload. Formal
# playback then narrows again to its mandatory readback; no search, batch lookup,
# generation, discovery, transport resume or queue navigation survives the boundary.
_POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES: frozenset[str] = frozenset(
    {"play_track", "preview_catalog_track"}
)
_POST_GENERATION_DELEGATED_READBACK_TOOL_NAMES: frozenset[str] = frozenset(
    {"get_now_playing"}
)

# S3 behavioral/safety routing: per-task tool-surface groups, selected once per
# run on its first round. Membership is evidence-driven, not speculative:

# The recommendation chain. Established by the live B2 baseline run (the real
# 推荐几首歌 chain: get_active_context + list_feedback_observations ->
# list_recommendation_runs + query_track_preference -> generate_recommendation)
# plus every prompt clause a recommendation-family turn can reach: the
# similar-to-current flow (get_active_context / get_now_playing +
# search_library_tracks -> discover fallback -> either generation tool), the
# 换一组/换一批 exclusion flow (the batch reads), the direction flow (genre
# params or a catalog search term), the fresh policy (query_catalog_discovery_state
# + discover + generate_inferred_recommendation with min_fresh/min_exploration),
# and the M2 empty-result steering (retry the generation, discover new terms --
# BOTH tools must stay visible so the deterministic refusal channel can steer
# instead of a hallucinated envelope). Feedback/learning members are READS only:
# B2 really consulted list_feedback_observations, so the reads stay; the writes
# (record/interpret/apply) need a user verdict that a pure recommendation
# request cannot carry. Plays, previews, pause/next/previous, open/write tools
# are provably unreachable in recommendation turns (post-generation the prompt
# pins the final answer; play/preview needs its own user turn).
_RECOMMENDATION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_active_context",
        "get_now_playing",
        "get_recommendation_run",
        "list_recommendation_runs",
        "get_canonical_entity",
        "search_library_tracks",
        "query_track_preference",
        "list_feedback_observations",
        "get_feedback_observation",
        "list_learning_applications",
        "get_learning_application",
        "query_catalog_discovery_state",
        "discover_catalog_tracks",
        "generate_recommendation",
        "generate_inferred_recommendation",
    }
)

# P20-Fix02: the recommendation-EXPLANATION surface -- the reads a
# why-question about the current batch needs (active batch, recent runs,
# preference/feedback/learning evidence, catalog state) without anything that
# can PRODUCE a new recommendation. Both generation tools are absent (an
# explanation run reads, never generates -- the two generate_* tools are
# structurally unreachable), and discover_catalog_tracks is absent because it
# persists staging rows (a write). Everything kept is an existing read-only
# tool; no group definition above was touched.
_EXPLANATION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_active_context",
        "get_now_playing",
        "get_recommendation_run",
        "list_recommendation_runs",
        "get_canonical_entity",
        "search_library_tracks",
        "query_track_preference",
        "list_feedback_observations",
        "get_feedback_observation",
        "list_learning_applications",
        "get_learning_application",
        "query_catalog_discovery_state",
    }
)

# Read/search questions may inspect the mixed known-track projection and, when
# needed, the public catalog, but they can never execute playback or generate
# recommendations.  This is the structural guard behind the prompt wording:
# ``play_track`` is absent rather than merely discouraged.
_LIBRARY_QUERY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "search_library_tracks",
        "get_canonical_entity",
    }
)

# S5 (token-cost optimization): the deterministic read prefetch of the ordinary
# recommendation chain -- the three anchor reads the live baseline spent its
# first two rounds collecting (get_active_context in a round of its own, then
# list_recommendation_runs + list_feedback_observations with a junk
# query_track_preference probe). They are mutually independent and independent
# of any provider decision, so the loop executes them once BEFORE provider
# round 1 and injects the results as a synthetic round-0 message pair (see
# ``_s5_prefetch_recommendation_reads``); the provider can then go straight to
# ``generate_recommendation`` in its first round. Order is the dependency
# order the S3 group documents (context first, then batch history and
# feedback anchors); each runs at most once per request, with the provider's
# usual ``{}`` quiet payload, so the results are byte-identical to what the
# provider would have received had it called them itself.
_S5_PREFETCH_TOOL_NAMES: tuple[str, ...] = (
    "get_active_context",
    "list_recommendation_runs",
    "list_feedback_observations",
)

# The formal-playback chain. The prompt's play contract: search_library_tracks
# first, discover_catalog_tracks only as the fallback, play_track to select
# (never play as a substitute), the mandated get_now_playing check-back, play
# for a bare 播放 resume, and the batch reads for 播放第N首 (get_active_context
# for active_batch, get_recommendation_run for the entries,
# list_recommendation_runs as the no-active-batch fallback).
# NO preview tools (S3 boundary ruling): a formal play turn must be able only
# to walk the formal path. Removing preview_catalog_track (and preview_batch /
# stop_preview, which were never here) makes self-downgrade structurally
# impossible -- the model cannot call what it is not offered -- instead of
# relying on the prompt's wording alone. When formal playback is unavailable
# the play group still has everything needed to say so in natural language and
# offer an explicit preview, which then arrives as its own 试听 turn through
# the preview group. The T14-E caller-side downgrade door remains untouched as
# the outer backstop (it owns verdicts on any residual run state, and it is
# not part of the tool-surface contract). STOP/queue tools stay out (换一首 is
# its own turn and routes context-sensitively; generation is forbidden on play
# turns by the 播放第N首 contract).
_PLAYBACK_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "search_library_tracks",
        "discover_catalog_tracks",
        "play_track",
        "play",
        "get_now_playing",
        "get_active_context",
        "get_recommendation_run",
        "list_recommendation_runs",
    }
)

# The preview chain. 试听<N首/歌名>: batch reads to locate the target, library
# search with catalog fallback for a named track, preview_catalog_track to
# start; the batch-preview phrasings: preview_batch and stop_preview, progress
# via get_playback_context; the stop residuals (停/别放了/关掉 that the routing
# table hands back to the loop): the prompt's stop clause reads
# get_active_context's preview_sounding and needs stop_preview for a sounding
# preview and pause when the user actually means the music -- the player
# snapshot already rides get_active_context, so get_now_playing stays out.
# Play tools are out: a preview turn never plays (T14-E forbids the reverse
# direction, and 播放 requests classify as play, not preview).
_PREVIEW_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "preview_catalog_track",
        "stop_preview",
        "preview_batch",
        "get_playback_context",
        "get_active_context",
        "get_recommendation_run",
        "list_recommendation_runs",
        "search_library_tracks",
        "discover_catalog_tracks",
        "pause",
    }
)

# The feedback/learning chain. The provider may use the batch-locating reads to
# resolve 第N首/这首 and may issue the one record_feedback write. After that write
# succeeds, P20 code owns interpret -> apply -> terminal reply; generation,
# discovery and playback are structurally absent from this task surface.
_FEEDBACK_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "record_feedback",
        "interpret_feedback",
        "apply_learning",
        "list_feedback_observations",
        "get_feedback_observation",
        "list_learning_applications",
        "get_learning_application",
        "get_active_context",
        "get_recommendation_run",
        "list_recommendation_runs",
    }
)

# P20 intent/control consolidation: once a closed feedback-verdict turn has
# successfully created its one durable observation, the runtime owns the rest
# of the P08 lifecycle.  The provider may locate the referent and formulate the
# record_feedback payload, but it does not get to create a second observation,
# spin on interpret/apply reads, or decide when the learning turn is finished.
_FEEDBACK_TURN_CLOSED_ERROR_CODE = "feedback_turn_closed"
_FEEDBACK_TURN_CLOSED_MESSAGE = (
    "本轮反馈已经成功记录；后续解释与学习由运行时完成，不再执行其他模型工具调用。"
)
_FEEDBACK_RECORDED_CLOSEOUT = "好的，已记录你的反馈。"
_FEEDBACK_PARTIAL_CLOSEOUT = "你的反馈已经记录，但偏好更新暂未完成。"
_CURRENT_TRACK_FEEDBACK_CLARIFICATION = (
    "我能看到你在指当前播放的歌曲，但暂时无法唯一确认它对应的音乐记录。"
    "请告诉我歌名和艺人。"
)
_CURRENT_TRACK_FEEDBACK_RECORD_FAILED = "这次没能记录你对当前歌曲的反馈，请稍后再试。"

# P20 delegated-action closeout: one authorized audio action per turn. Success
# and failure both close deterministically from the runtime result instead of
# spending a free-form provider final round that can later be rejected by the
# final-response boundary.
_DELEGATED_ACTION_CLOSED_ERROR_CODE = "delegated_action_closed"
_DELEGATED_ACTION_CLOSED_MESSAGE = (
    "本轮委托播放/试听已经执行过一次；不得再执行第二个音频动作。"
)
_DELEGATED_ACTION_UNBOUND_ERROR_CODE = "delegated_action_unbound"

_TURN_CLARIFICATION_CLOSEOUT = "我还不能确定你想让我做什么，可以说得更具体一点吗？"
_DELEGATED_ACTION_UNBOUND_MESSAGE = (
    "委托播放/试听的目标必须来自本轮生成批次或本轮已读取确认的当前推荐批。"
)
_DELEGATED_PREVIEW_OK_CLOSEOUT = "已开始试听，约 30 秒。"
_DELEGATED_PREVIEW_FAILED_CLOSEOUT = "这首暂时无法试听。"
_DELEGATED_PLAY_OK_CLOSEOUT = "已开始播放。"
_DELEGATED_PLAY_FAILED_CLOSEOUT = "这首暂时无法正式播放。"
_DELEGATED_PLAY_READBACK_FAILED_CLOSEOUT = "播放指令已发出，但暂时无法确认当前播放状态。"
_DELEGATED_SELECTION_EXHAUSTED_CLOSEOUT = "当前这批推荐里已经没有其他可播放或试听的曲目了。"


def _tools_named(
    tools: Sequence[ProviderToolSchema], names: frozenset[str]
) -> tuple[ProviderToolSchema, ...]:
    """S3: the caller-provided tools whose names are in ``names``, in order.

    A pure name-filter over the loop's tool list -- the loop never fabricates
    or re-sorts schemas, so a non-default tool list (tests, future callers)
    simply intersects with the group.
    """
    return tuple(tool for tool in tools if tool.name in names)


def _post_generation_delegated_action_names(payload: Mapping | None) -> frozenset[str]:
    """Action schemas justified by playback routes in this exact generation payload."""
    if not isinstance(payload, Mapping):
        return frozenset()
    items = payload.get("items")
    if not isinstance(items, list):
        return frozenset()
    names: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("target_id"), str):
            continue
        playback = item.get("playback")
        route = playback.get("route") if isinstance(playback, Mapping) else None
        if route == "library":
            names.add("play_track")
        elif route == "preview_only":
            names.add("preview_catalog_track")
    return frozenset(names)


def _delegated_action_targets_payload(
    call: ProviderToolCall, payload: Mapping | None
) -> bool:
    """Fail closed unless the selected id belongs to the payload with that action route."""
    if call.name not in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES:
        return False
    try:
        arguments = json.loads(call.arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(arguments, Mapping):
        return False
    canonical_id = arguments.get("canonical_id")
    if not isinstance(canonical_id, str) or not canonical_id:
        return False
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        return False
    expected_route = "library" if call.name == "play_track" else "preview_only"
    return any(
        isinstance(item, Mapping)
        and item.get("target_id") == canonical_id
        and isinstance(item.get("playback"), Mapping)
        and item["playback"].get("route") == expected_route
        for item in items
    )


def _delegated_action_target_id(call: ProviderToolCall) -> str | None:
    if call.name not in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES:
        return None
    try:
        arguments = json.loads(call.arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(arguments, Mapping):
        return None
    canonical_id = arguments.get("canonical_id")
    return canonical_id if isinstance(canonical_id, str) and canonical_id else None


def _delegated_play_readback_confirms(
    payload: Mapping | None, target_id: str | None
) -> bool:
    """Compatibility predicate backed by the strict ActionAttempt verifier."""
    if not target_id:
        return False
    action = DelegatedAudioAction(
        recommendation_run_id="compatibility",
        item_position=1,
        canonical_id=target_id,
        playback_route="library",
        tool_name="play_track",
    )
    attempt = mark_action_executing(create_action_attempt(action))
    attempt = record_action_execution(attempt, outcome="ok")
    return (
        verify_formal_play_readback(attempt, payload).status
        is ActionAttemptStatus.COMPLETED
    )


def _existing_batch_payload_from_tool_result(
    *,
    tool_name: str,
    payload: Mapping | None,
    active_run_id: str | None,
) -> tuple[str | None, Mapping | None]:
    """Capture the current/derived recommendation batch from reads this turn.

    Delegated selection from an existing batch must be bound to facts the
    runtime actually returned in THIS turn, not to a canonical id remembered
    in free-form model context. ``get_active_context`` establishes the current
    run identity; ``get_recommendation_run`` or ``list_recommendation_runs``
    supplies its item/route payload. When no active pointer exists, the service
    contract permits newest-history fallback, so the first listed run becomes
    the bound batch.
    """
    if not isinstance(payload, Mapping):
        return active_run_id, None
    if tool_name == "get_active_context":
        active = payload.get("active_batch")
        run_id = active.get("run_id") if isinstance(active, Mapping) else None
        return (run_id if isinstance(run_id, str) and run_id else None), None
    if tool_name == "get_recommendation_run":
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return active_run_id, None
        if active_run_id is not None and run_id != active_run_id:
            return active_run_id, None
        return run_id, payload
    if tool_name == "list_recommendation_runs":
        runs = payload.get("runs")
        if not isinstance(runs, list):
            return active_run_id, None
        selected = None
        if active_run_id is not None:
            selected = next(
                (
                    run
                    for run in runs
                    if isinstance(run, Mapping)
                    and run.get("run_id") == active_run_id
                ),
                None,
            )
        elif runs and isinstance(runs[0], Mapping):
            selected = runs[0]
        if not isinstance(selected, Mapping):
            return active_run_id, None
        run_id = selected.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return active_run_id, None
        return run_id, selected
    return active_run_id, None


def _select_task_tools(
    user_text: str, tools: Sequence[ProviderToolSchema]
) -> tuple[ProviderToolSchema, ...]:
    """S3: the run's FIRST-round tool surface for ``user_text``.

    Compatibility helper for callers that only have raw text: it reuses the
    deterministic S1/S3 classifiers and keeps every unclassified line on the full set.
    Production ``run()`` projects from its already-resolved TurnPlan instead.
    BASE prompt invariants remain active on that wider fallback, but precise
    least-privilege still depends on correct classification. Plain chat (S1)
    stays zero schemas.
    """
    return _select_task_tools_from_plan(resolve_turn_plan(user_text), tools)


def _select_task_tools_from_plan(
    turn_plan: TurnPlan, tools: Sequence[ProviderToolSchema]
) -> tuple[ProviderToolSchema, ...]:
    """Project one already-resolved TurnPlan onto the existing S3 tool groups."""
    surface = turn_plan.task_surface
    if surface is TurnTaskSurface.PLAIN_CHAT:
        return ()
    if surface is TurnTaskSurface.FEEDBACK:
        return _tools_named(tools, _FEEDBACK_TOOL_NAMES)
    if surface is TurnTaskSurface.PREVIEW:
        return _tools_named(tools, _PREVIEW_TOOL_NAMES)
    if surface is TurnTaskSurface.PLAYBACK:
        return _tools_named(tools, _PLAYBACK_TOOL_NAMES)
    if surface is TurnTaskSurface.LIBRARY_QUERY:
        return _tools_named(tools, _LIBRARY_QUERY_TOOL_NAMES)
    if surface is TurnTaskSurface.RECOMMENDATION_EXPLANATION:
        return _tools_named(tools, _EXPLANATION_TOOL_NAMES)
    if surface in (
        TurnTaskSurface.RECOMMENDATION,
        TurnTaskSurface.FRESH_DISCOVERY,
    ):
        return _tools_named(tools, _RECOMMENDATION_TOOL_NAMES)
    return tuple(tools)


def _discover_budget_exhausted_message(max_discover: int) -> str:
    """P15-S3-S3B: the deterministic synthetic refusal for a discover call
    requested after the per-run Fresh discovery budget ran out. It directs the
    model to stop re-term searching and settle with what it has -- distinct
    from the M2 post-generation closeout, which is about an already-delivered
    batch rather than about remaining supply."""
    return (
        f"本轮请求已达到 {max_discover} 次目录发现（discover_catalog_tracks）上限，"
        "请不要再换搜索词继续搜索目录；"
        "优先使用已有候选（已有推荐历史、推断候选与已知目录供给）回答用户；"
        "若确实没有足够的可用候选，请如实说明本轮没有找到足够的新结果；"
        "下一条用户消息会开始新的一轮请求、重新允许目录发现。"
    )


def _capture_fresh_promoted_ids(result_payload: Mapping | None, fresh_promoted_ids: set[str]) -> None:
    """P15-S3-S3D (corrected): authoritative same-run Fresh capture from the RAW
    structured payload of a genuine discover execution.

    P15-S3-S3E corrective patch (live-failure audit): the input is the RAW tool
    result payload ``result.payload`` -- the dict the service produced, BEFORE
    ``_bound_payload_with_measure`` replaces oversized payloads with the
    ``{"truncated": True, "preview": ...}`` model-visible marker. The delivered
    text is presentation, never authority: a real catalog discover serializes
    ~9.5KB (> ``_MAX_TOOL_RESULT_CHARS``), so parsing the delivered envelope
    previously captured nothing, silently, on every real discovery.

    Fresh identity is exactly the canonical id of entries the service itself
    classified PROMOTED:

    * only the ``promoted`` list is iterated, and only entries with
      ``status == "promoted"`` and a non-empty string canonical id enter --
      staged/skipped entries (ALREADY_BOUND and LIBRARY_KNOWN land in
      ``skipped``) never do;
    * a malformed / marker-only / unparseable payload contributes nothing
      (under-capture, never fabrication -- the Fresh floor is best effort);
    * the caller gates execution: only a genuinely executed, non-replayed,
      ``ok`` discover calls this helper -- FAILED / blocked / budget-exhausted /
      closeout-refused / replay-only executions never reach it.

    The ids then travel ONLY into the generation execution kwarg -- never into
    model arguments, the journal request payload, or any tool schema.
    """
    if not isinstance(result_payload, Mapping):
        return
    promoted = result_payload.get("promoted")
    if not isinstance(promoted, list):
        return
    for entry in promoted:
        if not isinstance(entry, Mapping) or entry.get("status") != "promoted":
            continue
        canonical_id = entry.get("canonical_id")
        if isinstance(canonical_id, str) and canonical_id:
            fresh_promoted_ids.add(canonical_id)
def _select_system_prompt(user_text: str, system_prompt: str) -> str:
    """S4: the run's effective system prompt for ``user_text``.

    Reuses the S1/S3 local intent-router classifiers -- the exact same
    deterministic text judgments that choose the tool surface, no second LLM
    router. Only the built-in ``DEFAULT_SYSTEM_PROMPT`` is ever decomposed: a
    caller-configured prompt passes through unchanged, and every input the
    classifiers cannot pin down (mixed/chained/ambiguous/unknown) keeps the
    full prompt. The BASE invariants make that fallback safety-complete without
    importing conflicting task-only Library/explanation instructions.
    """
    return _select_system_prompt_from_plan(
        resolve_turn_plan(user_text), system_prompt
    )


def _select_system_prompt_from_plan(
    turn_plan: TurnPlan, system_prompt: str
) -> str:
    """Project one already-resolved TurnPlan onto the existing S4 prompt set."""
    if system_prompt != DEFAULT_SYSTEM_PROMPT:
        return system_prompt
    surface = turn_plan.task_surface
    if surface is TurnTaskSurface.PLAIN_CHAT:
        return _S4_BASE_PROMPT
    if surface is TurnTaskSurface.FEEDBACK:
        return _S4_FEEDBACK_PROMPT
    if surface is TurnTaskSurface.PREVIEW:
        return _S4_PREVIEW_PROMPT
    if surface is TurnTaskSurface.PLAYBACK:
        return _S4_PLAYBACK_PROMPT
    if surface is TurnTaskSurface.LIBRARY_QUERY:
        return _S4_LIBRARY_QUERY_PROMPT
    # P20-Fix02: explanation uses the read-only explanation prompt. The
    # families are disjoint, so the position between play and fresh is purely
    # documentary; the constraint that matters is the pair BELOW: fresh is
    # checked before recommendation so fresh-intent forms that also sit in
    # the recommendation door (推荐一些新歌) always travel the catalog path.
    if surface is TurnTaskSurface.RECOMMENDATION_EXPLANATION:
        return _S4_EXPLANATION_PROMPT
    if surface is TurnTaskSurface.FRESH_DISCOVERY:
        return _S4_DISCOVERY_PROMPT
    if surface is TurnTaskSurface.RECOMMENDATION:
        return _S4_RECOMMENDATION_PROMPT
    return system_prompt


def _s5_prefetch_recommendation_reads_enabled(
    system_prompt: str, tools: Sequence[ProviderToolSchema]
) -> bool:
    """S5: prefetch gate -- exactly the ordinary-recommendation runs.

    True only when (a) the run travels with the S4 recommendation prompt --
    the same classifier result S4 produced, so fresh/play/feedback/preview/
    plain/ambiguous lines and any caller-configured prompt (custom prompts
    pass through the selector untouched and never equal the module prompt)
    are all excluded -- and (b) every prefetch read is present in the run's
    tool surface (a narrowed or caller-supplied tool list must fail the gate
    fail-safe: skipping the prefetch only costs the rounds S4 already paid,
    injecting results the tools could not have produced would lie)."""
    if system_prompt != _S4_RECOMMENDATION_PROMPT:
        return False
    names = {tool.name for tool in tools}
    return all(name in names for name in _S5_PREFETCH_TOOL_NAMES)


@dataclass(frozen=True, slots=True)
class ProviderLoopConfig:
    """Bounds and prompts for one provider agent loop."""

    max_tool_rounds: int = 8
    max_context_messages: int = 24
    max_generation_attempts: int = 2
    # P15-S3-S3B: hard per-run cap on genuine Fresh Catalog discoveries (real,
    # non-replayed executions of discover_catalog_tracks). The prompt teaches
    # the same number; a non-default value exists for tests only.
    max_discover_per_run: int = MAX_DISCOVER_PER_RUN
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # P15-S4-M1: cost instrumentation. Off by default: with it disabled the loop
    # accumulates no measurements and the result carries ``trace=None``.
    instrument: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_tool_rounds, int) or self.max_tool_rounds <= 0:
            raise ProviderError("max_tool_rounds must be a positive integer")
        if not isinstance(self.max_context_messages, int) or self.max_context_messages < 4:
            raise ProviderError("max_context_messages must be an integer >= 4")
        if (
            not isinstance(self.max_generation_attempts, int)
            or self.max_generation_attempts <= 0
        ):
            raise ProviderError("max_generation_attempts must be a positive integer")
        if (
            not isinstance(self.max_discover_per_run, int)
            or self.max_discover_per_run <= 0
        ):
            raise ProviderError("max_discover_per_run must be a positive integer")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ProviderError("system_prompt must be a non-empty string")
        if not isinstance(self.instrument, bool):
            raise ProviderError("instrument must be a bool")


@dataclass(frozen=True, slots=True)
class ProviderLoopToolExecution:
    """One tool execution record (observability for the CLI/status surface).

    ``origin`` is non-durable execution metadata.  Policy-injected calls still
    execute through P09 and therefore receive ordinary journal rows, while the
    origin makes their control-plane source explicit in the live result/trace.
    """

    name: str
    outcome: str
    error_code: str | None = None
    elapsed_ms: float | None = None
    origin: str = "provider_requested"


@dataclass(frozen=True, slots=True)
class ToolExecutionMeasure:
    """P15-S4-M1: measurement facts of one provider-requested tool call.

    ``content`` is exactly the pre-instrumentation tool result string sent back
    to the provider. ``error_message`` retains the raw service error text for
    narrow code-owned recovery decisions; the remaining measurement fields do
    not affect what the provider receives. ``executed`` is
    False only for the invalid-arguments refusal (dedupe/cache/closeout/
    budget-gate answers are recorded by the loop directly).

    ``payload`` carries the RAW structured service payload of any successful
    execution. P20-Fix10 uses generation payloads as the authoritative batch
    presentation source; the P20 feedback closeout uses record_feedback's raw
    ``feedback_id`` to finish interpret -> apply in code. Non-ok executions and
    invalid-arguments refusals carry None.
    """

    content: str
    outcome: str
    error_code: str | None
    error_message: str | None
    raw_result_chars: int
    delivered_result_chars: int
    truncated: bool
    replayed: bool
    executed: bool
    payload: Mapping | None = None


@dataclass(frozen=True, slots=True)
class ProviderAgentResult:
    """The outcome of one provider agent run."""

    final_text: str
    rounds: int
    tool_executions: tuple[ProviderLoopToolExecution, ...]
    context_trimmed: bool
    rounds_capped: bool
    total_elapsed_ms: float | None = None
    # P15-S4-M1: cost measurements for this run. Present only when the loop was
    # built with ``ProviderLoopConfig(instrument=True)`` -- never by default.
    trace: ProviderRunTrace | None = None
    # The successful tool payload belongs to this invocation, including replays.
    # Presentation must never infer that ownership from global history order.
    recommendation_payload: Mapping[str, Any] | None = None
    # Terminal in-memory truth for a code-owned selected or directly targeted
    # track action. Non-track controls use the separate state attempt below.
    action_attempt: ActionAttempt | None = None
    # Direct play/pause controls have no selected canonical target. Their
    # typed terminal truth is therefore carried separately rather than being
    # forced into ActionAttempt's canonical-equality contract.
    playback_control_attempt: PlaybackControlAttempt | None = None
    # S2.1 follow-up: an assistant-visible offer may only exist when code has
    # already resolved one authoritative canonical target from structured
    # facts. The conversation host owns the pending lifecycle; this per-turn
    # result merely carries the immutable action to arm atomically with the
    # deterministic offer text.
    offered_action: OfferedAction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.final_text, str):
            raise ProviderError("final_text must be a string")
        object.__setattr__(self, "tool_executions", tuple(self.tool_executions))


class ProviderAgentLoop:
    """Runs one user request through provider + P09, bounded and fail-closed."""

    def __init__(
        self,
        provider: ChatProvider,
        client: AgentClient,
        tools: Sequence[ProviderToolSchema],
        *,
        config: ProviderLoopConfig | None = None,
    ) -> None:
        if not callable(getattr(provider, "chat", None)):
            raise ProviderError("provider must implement ChatProvider")
        if not isinstance(client, AgentClient):
            raise ProviderError("client must be an AgentClient")
        if not tools or not all(isinstance(tool, ProviderToolSchema) for tool in tools):
            raise ProviderError("tools must be a non-empty sequence of ProviderToolSchema")
        self.provider = provider
        self.client = client
        self.tools = tuple(tools)
        self.config = config or ProviderLoopConfig()

    def _run_current_track_feedback(self, kind: str) -> ProviderAgentResult:
        """Own the current-player feedback lifecycle without a provider round."""
        run_started = time.monotonic()
        executions: list[ProviderLoopToolExecution] = []
        tool_records: list[ProviderToolMeasure] = []

        def execute(name: str, arguments: Mapping[str, object]) -> ToolExecutionMeasure:
            call_index = len(executions) + 1
            call = ProviderToolCall(
                call_id=f"policy_current_feedback_{call_index}_{name}",
                name=name,
                arguments=json.dumps(
                    dict(arguments),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            started = time.monotonic()
            measure = self._execute_tool_call(call)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            executions.append(
                self._execution_record(
                    call,
                    measure.content,
                    elapsed_ms,
                    origin="policy_injected",
                )
            )
            if self.config.instrument:
                tool_records.append(
                    ProviderToolMeasure(
                        round_index=0,
                        call_index=call_index,
                        name=name,
                        outcome=measure.outcome,
                        error_code=measure.error_code,
                        duration_ms=elapsed_ms if measure.executed else None,
                        raw_result_chars=measure.raw_result_chars,
                        delivered_result_chars=measure.delivered_result_chars,
                        truncated=measure.truncated,
                        replayed=measure.replayed,
                        executed=measure.executed,
                        arguments=safe_tool_arguments(call.arguments),
                        origin="policy_injected",
                    )
                )
            return measure

        def finish(text: str) -> ProviderAgentResult:
            return self._finish_result(
                ProviderAgentResult(
                    final_text=text,
                    recommendation_payload=None,
                    rounds=0,
                    tool_executions=tuple(executions),
                    context_trimmed=False,
                    rounds_capped=False,
                    total_elapsed_ms=round(
                        (time.monotonic() - run_started) * 1000.0, 1
                    ),
                ),
                [],
                tool_records,
            )

        context = execute("get_active_context", {})
        player = (
            context.payload.get("player")
            if context.outcome == "ok" and isinstance(context.payload, Mapping)
            else None
        )
        canonical_id = (
            player.get("canonical_id") if isinstance(player, Mapping) else None
        )
        if not isinstance(canonical_id, str) or not canonical_id:
            return finish(_CURRENT_TRACK_FEEDBACK_CLARIFICATION)

        recorded = execute(
            "record_feedback",
            {
                "kind": kind,
                "source_system": "apple_music",
                "source_path": "now_playing",
                "target_id": canonical_id,
            },
        )
        feedback_id = (
            recorded.payload.get("feedback_id")
            if recorded.outcome == "ok" and isinstance(recorded.payload, Mapping)
            else None
        )
        if not isinstance(feedback_id, str) or not feedback_id:
            return finish(_CURRENT_TRACK_FEEDBACK_RECORD_FAILED)

        for name in ("interpret_feedback", "apply_learning"):
            if execute(name, {"feedback_id": feedback_id}).outcome != "ok":
                return finish(_FEEDBACK_PARTIAL_CLOSEOUT)
        return finish(_FEEDBACK_RECORDED_CLOSEOUT)

    def _run_selection_grant(
        self,
        turn_plan: TurnPlan,
        *,
        authoritative_payload: Mapping | None = None,
        active_run_id: str | None = None,
        recommendation_payload: Mapping | None = None,
        round_index: int,
        next_call_index: int,
        executions: list[ProviderLoopToolExecution],
        context_trimmed: bool,
        run_started: float,
        round_records: list[ProviderRoundMeasure],
        tool_records: list[ProviderToolMeasure],
    ) -> ProviderAgentResult:
        """Execute one code-owned SelectionGrant action from an authoritative run.

        Existing-active delegation supplies ``active_run_id`` and this helper
        reads that exact durable run. Same-turn generation supplies the exact
        successful generation payload directly. Both then share one grant,
        selector, route, ActionAttempt, readback and result-rendering workflow.
        """

        def finish(
            text: str, action_attempt: ActionAttempt | None = None
        ) -> ProviderAgentResult:
            return self._finish_result(
                ProviderAgentResult(
                    final_text=text,
                    recommendation_payload=recommendation_payload,
                    action_attempt=action_attempt,
                    rounds=round_index,
                    tool_executions=tuple(executions),
                    context_trimmed=context_trimmed,
                    rounds_capped=False,
                    total_elapsed_ms=round(
                        (time.monotonic() - run_started) * 1000.0, 1
                    ),
                ),
                round_records,
                tool_records,
            )

        def execute(policy_call: ProviderToolCall) -> ToolExecutionMeasure:
            nonlocal next_call_index
            next_call_index += 1
            started = time.monotonic()
            measure = self._execute_tool_call(policy_call)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            executions.append(
                self._execution_record(
                    policy_call,
                    measure.content,
                    elapsed_ms,
                    origin="policy_injected",
                )
            )
            if self.config.instrument:
                tool_records.append(
                    ProviderToolMeasure(
                        round_index=round_index,
                        call_index=next_call_index,
                        name=policy_call.name,
                        outcome=measure.outcome,
                        error_code=measure.error_code,
                        duration_ms=elapsed_ms if measure.executed else None,
                        raw_result_chars=measure.raw_result_chars,
                        delivered_result_chars=measure.delivered_result_chars,
                        truncated=measure.truncated,
                        replayed=measure.replayed,
                        executed=measure.executed,
                        arguments=safe_tool_arguments(policy_call.arguments),
                        origin="policy_injected",
                    )
                )
            return measure

        if authoritative_payload is None and active_run_id is not None:
            run_call = ProviderToolCall(
                call_id=f"policy_selection_grant_{round_index}_get_run",
                name="get_recommendation_run",
                arguments=json.dumps(
                    {"run_id": active_run_id},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            run_result = execute(run_call)
            authoritative_payload = (
                run_result.payload
                if run_result.outcome == "ok"
                and isinstance(run_result.payload, Mapping)
                else None
            )
        authoritative_run_id = (
            authoritative_payload.get("run_id")
            if isinstance(authoritative_payload, Mapping)
            else None
        )
        if (
            not isinstance(authoritative_run_id, str)
            or not authoritative_run_id
            or (
                active_run_id is not None
                and authoritative_run_id != active_run_id
            )
        ):
            return finish("暂时无法从当前推荐中确认要播放或试听的曲目。")

        verified_canonical_ids = frozenset(
            selection.canonical_id
            for selection in self.client.service.verified_selections_for_run(
                authoritative_run_id
            )
        )
        grant = build_selection_grant(
            turn_plan,
            authoritative_payload,
            verified_canonical_ids=verified_canonical_ids,
        )
        action = select_delegated_audio_action(grant)
        if action is None:
            if selection_grant_is_exhausted(grant):
                return finish(_DELEGATED_SELECTION_EXHAUSTED_CLOSEOUT)
            return finish("暂时无法从当前推荐中确认要播放或试听的曲目。")

        call = ProviderToolCall(
            call_id=f"policy_selection_grant_{round_index}_{action.tool_name}",
            name=action.tool_name,
            arguments=json.dumps(
                {"canonical_id": action.canonical_id},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        # The old payload validator remains as a defensive assertion over the
        # code-created action. It is no longer the mechanism that chooses
        # whether to trust a Provider-proposed canonical id or route.
        if not (
            delegated_action_is_authorized(action, grant)
            and _delegated_action_targets_payload(call, authoritative_payload)
        ):
            logger.error("code-owned delegated action failed its SelectionGrant invariant")
            return finish("暂时无法从当前推荐中确认要播放或试听的曲目。")

        attempt = mark_action_executing(create_action_attempt(action))
        action_result = execute(call)
        preview_started = (
            action_result.payload.get("started")
            if isinstance(action_result.payload, Mapping)
            else None
        )
        attempt = record_action_execution(
            attempt,
            outcome=action_result.outcome,
            preview_started=(
                preview_started if isinstance(preview_started, bool) else None
            ),
            error_code=action_result.error_code,
        )
        if action.tool_name == "preview_catalog_track":
            if attempt.status is ActionAttemptStatus.COMPLETED:
                self.client.service.record_verified_selection(
                    run_id=action.recommendation_run_id,
                    canonical_id=action.canonical_id,
                    item_position=action.item_position,
                    action_kind=action.tool_name,
                    playback_route=action.playback_route,
                )
            return finish(render_verified_action_result(attempt), attempt)
        if attempt.status is ActionAttemptStatus.FAILED:
            return finish(render_verified_action_result(attempt), attempt)
        readback_call = ProviderToolCall(
            call_id=f"policy_selection_grant_{round_index}_get_now_playing",
            name="get_now_playing",
            arguments="{}",
        )
        readback = execute(readback_call)
        attempt = verify_formal_play_readback(
            attempt,
            readback.payload if readback.outcome == "ok" else None,
        )
        if attempt.status is ActionAttemptStatus.COMPLETED:
            self.client.service.record_verified_selection(
                run_id=action.recommendation_run_id,
                canonical_id=action.canonical_id,
                item_position=action.item_position,
                action_kind=action.tool_name,
                playback_route=action.playback_route,
            )
        return finish(render_verified_action_result(attempt), attempt)

    def _run_choose_another(self, turn_plan: TurnPlan) -> ProviderAgentResult:
        """Resolve choose-another from session context without a Provider round.

        The semantic is already explicit in ``TurnPlan``. Code reads the current
        authoritative run and enters the shared SelectionGrant workflow directly, so
        a Provider cannot generate a replacement batch or choose a target when the
        current run is exhausted.
        """
        run_started = time.monotonic()
        call = ProviderToolCall(
            call_id="policy_choose_another_get_active_context",
            name="get_active_context",
            arguments="{}",
        )
        started = time.monotonic()
        measure = self._execute_tool_call(call)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        executions = [
            self._execution_record(
                call,
                measure.content,
                elapsed_ms,
                origin="policy_injected",
            )
        ]
        tool_records: list[ProviderToolMeasure] = []
        if self.config.instrument:
            tool_records.append(
                ProviderToolMeasure(
                    round_index=0,
                    call_index=1,
                    name=call.name,
                    outcome=measure.outcome,
                    error_code=measure.error_code,
                    duration_ms=elapsed_ms if measure.executed else None,
                    raw_result_chars=measure.raw_result_chars,
                    delivered_result_chars=measure.delivered_result_chars,
                    truncated=measure.truncated,
                    replayed=measure.replayed,
                    executed=measure.executed,
                    arguments=safe_tool_arguments(call.arguments),
                )
            )
        active_run_id, _payload = _existing_batch_payload_from_tool_result(
            tool_name=call.name,
            payload=measure.payload,
            active_run_id=None,
        )
        if measure.outcome != "ok" or active_run_id is None:
            return self._finish_result(
                ProviderAgentResult(
                    final_text="暂时无法从当前推荐中确认要播放或试听的曲目。",
                    recommendation_payload=None,
                    action_attempt=None,
                    rounds=0,
                    tool_executions=tuple(executions),
                    context_trimmed=False,
                    rounds_capped=False,
                    total_elapsed_ms=round(
                        (time.monotonic() - run_started) * 1000.0, 1
                    ),
                ),
                [],
                tool_records,
            )
        return self._run_selection_grant(
            turn_plan,
            active_run_id=active_run_id,
            round_index=0,
            next_call_index=1,
            executions=executions,
            context_trimmed=False,
            run_started=run_started,
            round_records=[],
            tool_records=tool_records,
        )

    def _turn_interpreter_context(self) -> TurnInterpreterContext:
        """Read only the high-level context needed for semantic disambiguation.

        The interpreter never sees run ids, canonical ids, routes, tool schemas, or
        database rows.  A failed read yields unknown (``None``) facts rather than a
        guessed false value.
        """
        try:
            result = self.client.call("get_active_context", {})
        except Exception:
            return TurnInterpreterContext()
        if result.outcome is not AgentToolOutcome.OK or not isinstance(
            result.payload, Mapping
        ):
            return TurnInterpreterContext()
        payload = result.payload
        player = payload.get("player")
        has_current_playback = None
        if isinstance(player, Mapping):
            state = player.get("state")
            if isinstance(state, str):
                has_current_playback = state in {"playing", "paused"}
        active_batch = payload.get("active_batch")
        has_active_recommendation = None
        active_item_count = None
        if active_batch is None:
            has_active_recommendation = False
        elif isinstance(active_batch, Mapping):
            has_active_recommendation = True
            count = active_batch.get("item_count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                active_item_count = count
        referent = payload.get("referent_canonical_id")
        has_referenced_item = (
            True
            if isinstance(referent, str) and bool(referent)
            else False
            if referent is None
            else None
        )
        preview_sounding = payload.get("preview_sounding")
        return TurnInterpreterContext(
            has_current_playback=has_current_playback,
            has_active_recommendation=has_active_recommendation,
            active_recommendation_item_count=active_item_count,
            has_referenced_item=has_referenced_item,
            preview_active=(
                preview_sounding if isinstance(preview_sounding, bool) else None
            ),
        )

    def resolve_turn(self, user_text: str) -> TurnPlan:
        """Resolve one semantic plan: deterministic first, LLM only for UNKNOWN.

        Web/CLI/direct-loop callers all converge here.  A valid interpreted plan is
        returned as the same existing ``TurnPlan`` type consumed by every downstream
        workflow.
        """
        deterministic_plan = resolve_turn_plan(user_text)
        if deterministic_plan.primary is not TurnPrimarySemantic.UNKNOWN:
            return deterministic_plan
        # Only provider adapters that explicitly opt into the strict no-tool
        # structured interpreter path use it. Third-party/test ChatProvider
        # implementations keep the established FULL compatibility behavior
        # rather than having an extra opaque provider round inserted beneath
        # them. Production DeepSeek/Codex adapters both opt in.
        if not getattr(self.provider, "supports_turn_interpreter", False):
            return deterministic_plan
        interpretation = interpret_turn(
            self.provider,
            user_text,
            deterministic_plan,
            self._turn_interpreter_context(),
        )
        logger.debug(
            "turn semantic resolution source=%s interpreter_status=%s error=%s",
            interpretation.plan.semantic_source.value,
            interpretation.status,
            interpretation.error_code,
        )
        return interpretation.plan

    def run(
        self, user_text: str, *, turn_plan: TurnPlan | None = None
    ) -> ProviderAgentResult:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ProviderError("user_text must be a non-empty string")
        if turn_plan is None:
            turn_plan = self.resolve_turn(user_text)
        elif not isinstance(turn_plan, TurnPlan) or turn_plan.user_text != user_text:
            raise ProviderError("turn_plan must describe the same user_text")

        if turn_plan.requires_clarification:
            return self._finish_result(
                ProviderAgentResult(
                    final_text=_TURN_CLARIFICATION_CLOSEOUT,
                    recommendation_payload=None,
                    rounds=0,
                    tool_executions=(),
                    context_trimmed=False,
                    rounds_capped=False,
                    total_elapsed_ms=0.0,
                ),
                [],
                [],
            )

        preference_statement = turn_plan.preference_statement
        if preference_statement is not None:
            # Current domain writes are track/recommendation-feedback based; there
            # is no honest artist-level preference mutation on this provider
            # surface. Do not fan an artist statement out into synthetic likes
            # on individual tracks, and do not let the full fallback reinterpret
            # it as a recommendation request. Acknowledge only what the user
            # actually said, without claiming persistence.
            verb = "喜欢" if preference_statement.polarity == "positive" else "不喜欢"
            text = f"明白，你{verb} {preference_statement.target} 的歌。"
            return self._finish_result(
                ProviderAgentResult(
                    final_text=text,
                    recommendation_payload=None,
                    rounds=0,
                    tool_executions=(),
                    context_trimmed=False,
                    rounds_capped=False,
                    total_elapsed_ms=0.0,
                ),
                [],
                [],
            )

        current_track_feedback = turn_plan.current_track_feedback
        if current_track_feedback is not None:
            return self._run_current_track_feedback(current_track_feedback.kind)

        playback_action = turn_plan.playback_action
        if (
            playback_action is not None
            and playback_action.source == "active_recommendation"
            and playback_action.selection_mode == "choose_another"
        ):
            return self._run_choose_another(turn_plan)

        delegated_action_authorized = turn_plan.delegated_action_authorized
        feedback_lifecycle_owned = turn_plan.task_surface is TurnTaskSurface.FEEDBACK
        # S1+S3 behavioral/safety routing: the first round's tool surface is
        # chosen from the one resolved TurnPlan -- deterministic forms stay local;
        # only deterministic UNKNOWN may have been translated by P22's no-tool
        # interpreter. Plain chat runs zero schemas (S1), closed/interpreted task
        # families use their narrowed groups, and unresolved compatibility turns
        # keep the caller's full tool set fail-safe. No schema is
        # ever deleted; the full set still ships on any unclassified request.
        tools = _select_task_tools_from_plan(turn_plan, self.tools)
        # S4 behavioral/safety routing: the run's effective system prompt is
        # narrowed per task by the same classifiers -- see _select_system_prompt.
        # Chosen once per run; S2 only ever shrinks the per-round tool list
        # from here, never the prompt.
        system_prompt = _select_system_prompt_from_plan(
            turn_plan, self.config.system_prompt
        )
        # Explicit recommendation semantics are resolved once, before any
        # provider round.  This read-only canonical lookup gives the model and
        # the generation boundary the same current-turn target; neither the
        # now-playing snapshot nor historical context may replace it later.
        recommendation_semantics = _resolve_recommendation_semantics(
            self.client, turn_plan.recommendation
        )
        # S2 behavioral boundary: the task selection evolves per round. It may
        # shrink to final-only, or -- solely after generation for an existing
        # explicit delegation -- to one payload-bound action and the formal-play
        # readback. It never expands beyond the first-round surface.
        round_tools: tuple[ProviderToolSchema, ...] = tools
        run_started = time.monotonic()
        messages: list[ProviderMessage] = [
            ProviderMessage(ProviderMessageRole.USER, text=user_text)
        ]
        executions: list[ProviderLoopToolExecution] = []
        # P15-S4-M1: measurement accumulation. Instrumentation is opt-in; with it
        # off these stay empty and no trace is ever built.
        round_records: list[ProviderRoundMeasure] = []
        tool_records: list[ProviderToolMeasure] = []
        results_cache: dict[tuple[str, str], str] = {}
        context_trimmed = False
        rounds_capped = False
        final_text = ""
        generation_attempts = 0
        generation_succeeded = False
        delegated_action_attempted = False
        delegated_playback_readback_pending = False
        delegated_formal_target_id: str | None = None
        # Successful delegated playback/preview is runtime-owned completion.
        # Once the requested action (and formal-play readback) succeeds, the
        # user-facing closeout is deterministic; do not spend another provider
        # round asking free-form prose to restate a fact the runtime already
        # proved.  This also covers delegation from an existing active batch,
        # where no generation occurs in the current turn.
        delegated_terminal_text: str | None = None
        # Closed feedback-verdict turns are allowed exactly one successful
        # durable observation. After that point interpret/apply + termination
        # are runtime-owned, eliminating the model-driven learning loop that
        # could exhaust the provider round cap after the write already landed.
        feedback_record_id: str | None = None
        # Existing-batch delegation is bound to recommendation facts read in
        # THIS turn. This closes the safety gap exposed by the real UAT where
        # "随便播放一首" reused a two-hour-old active batch without generating a
        # new payload: the action may still use that batch, but only after its
        # run/items/routes have been read and captured here.
        delegated_existing_run_id: str | None = None
        delegated_existing_batch_payload: Mapping | None = None
        # P20 preference-seed fallback handoff.  Once injected, the discovery
        # and inferred retry consume the only remaining policy opportunities;
        # the provider cannot earn another fallback by asking again.
        preference_fallback_injected = False
        # P20-Fix10/P20 cleanup: the authoritative payload of this run's first
        # successful generation (either generation tool). Success closes the
        # generation phase in code, so this single capture is exact and cannot
        # be overwritten by a second durable run. None until success.
        final_generation_payload: Mapping | None = None
        # P15-S3-S3B: per-run counter of genuine Fresh Catalog discoveries. A
        # run() local, so every new user message re-enables discovery.
        discover_executions = 0
        # P15-S3-S3D: authoritative same-run Fresh provenance -- canonical ids of
        # PROMOTED entries in genuinely executed, non-replayed, ok discover
        # results this run. A run() local (auto-cleared per new user request),
        # never persisted, and passed to generation tools ONLY as an internal
        # execution kwarg -- the model can never write or forge it.
        fresh_promoted_ids: set[str] = set()

        # P22-S2.1 follow-up: named formal-play turns now project the user
        # language target into TurnPlan and perform one code-owned read before
        # Provider planning. The search term is therefore the user's semantic
        # target, never a model-authored query. A unique preview_only result can
        # become an OfferedAction immediately; ambiguous targets fail closed.
        named_play_target_text = (
            playback_action.target_text
            if playback_action is not None
            else None
        )
        named_play_resolution_status: str | None = None
        explicit_index_offer_requested = _explicit_index_preview_offer_requested(turn_plan)
        explicit_index_offer_run_id: str | None = None
        if named_play_target_text is not None:
            try:
                (
                    named_search_call,
                    named_search_measure,
                    named_search_execution,
                    named_search_elapsed_ms,
                ) = self._execute_named_play_search(
                    named_play_target_text, call_id="policy_named_play_search_0"
                )
            except Exception:
                logger.exception("code-owned named-play search failed")
                named_play_resolution_status = "unresolved"
            else:
                executions.append(named_search_execution)
                messages.extend(
                    (
                        ProviderMessage(
                            ProviderMessageRole.ASSISTANT,
                            tool_calls=(named_search_call,),
                        ),
                        ProviderMessage(
                            ProviderMessageRole.USER,
                            tool_results=(
                                ProviderToolResult(
                                    named_search_call.call_id,
                                    named_search_measure.content,
                                ),
                            ),
                        ),
                    )
                )
                results_cache[(
                    named_search_call.name,
                    named_search_call.arguments,
                )] = named_search_measure.content
                if self.config.instrument:
                    tool_records.append(
                        ProviderToolMeasure(
                            round_index=0,
                            call_index=1,
                            name=named_search_call.name,
                            outcome=named_search_measure.outcome,
                            error_code=named_search_measure.error_code,
                            duration_ms=(
                                named_search_elapsed_ms
                                if named_search_measure.executed
                                else None
                            ),
                            raw_result_chars=named_search_measure.raw_result_chars,
                            delivered_result_chars=(
                                named_search_measure.delivered_result_chars
                            ),
                            truncated=named_search_measure.truncated,
                            replayed=named_search_measure.replayed,
                            executed=named_search_measure.executed,
                            arguments=safe_tool_arguments(
                                named_search_call.arguments
                            ),
                            origin="policy_injected",
                        )
                    )
                resolution = (
                    _resolve_named_play_search_payload(
                        named_search_measure.payload, named_play_target_text
                    )
                    if named_search_measure.outcome == "ok"
                    else _NamedPlayResolution("unresolved")
                )
                named_play_resolution_status = resolution.status
                if resolution.offer is not None:
                    return self._finish_result(
                        ProviderAgentResult(
                            final_text=_render_exact_target_preview_offer(
                                resolution.offer
                            ),
                            offered_action=resolution.offer,
                            rounds=0,
                            tool_executions=tuple(executions),
                            context_trimmed=False,
                            rounds_capped=False,
                            total_elapsed_ms=round(
                                (time.monotonic() - run_started) * 1000.0, 1
                            ),
                        ),
                        round_records,
                        tool_records,
                    )
                if resolution.status == "ambiguous":
                    return self._finish_result(
                        ProviderAgentResult(
                            final_text=_NAMED_PLAY_AMBIGUOUS_CLOSEOUT,
                            rounds=0,
                            tool_executions=tuple(executions),
                            context_trimmed=False,
                            rounds_capped=False,
                            total_elapsed_ms=round(
                                (time.monotonic() - run_started) * 1000.0, 1
                            ),
                        ),
                        round_records,
                        tool_records,
                    )

        # S5 (token-cost optimization): deterministic read prefetch. Ordinary
        # recommendation runs pay for the three anchor reads ONCE, before any
        # provider round, as a synthetic round-0 message pair -- the exact
        # wire shape a real read round produces. This is what lets the
        # provider generate in its first round (2 rounds total) instead of
        # spending S4's two read rounds first (4 rounds). All-or-nothing and
        # fail-open: a prefetch failure leaves ``messages`` untouched and the
        # provider collects the reads itself, exactly as before S5.
        if _s5_prefetch_recommendation_reads_enabled(system_prompt, tools):
            prefetch_calls, prefetch_results, prefetch_records = (
                self._s5_prefetch_recommendation_reads(results_cache)
            )
            if prefetch_calls:
                messages.append(
                    ProviderMessage(
                        ProviderMessageRole.ASSISTANT,
                        tool_calls=tuple(prefetch_calls),
                    )
                )
                messages.append(
                    ProviderMessage(
                        ProviderMessageRole.USER,
                        tool_results=tuple(prefetch_results),
                    )
                )
                if self.config.instrument:
                    tool_records.extend(prefetch_records)

        # ``seed_source=current_track`` is abstract until the existing player
        # resolver has proved one canonical identity. Bind only that strict
        # projection; an unresolved player remains empty and is rejected at
        # the generation boundary below.
        recommendation_semantics = _bind_current_track_similarity_seed(
            recommendation_semantics, results_cache
        )

        from music_agent.final_response_boundary import FINAL_ANSWER_CONTRACT

        # Keep caller-supplied system prompts byte-for-byte authoritative.
        # Turn semantics augmentation is part of the built-in prompt pipeline only;
        # code-level recommendation guards still enforce the same semantics for
        # custom-prompt runs without silently mutating the caller's prompt.
        semantics_prompt = (
            _recommendation_semantics_prompt(recommendation_semantics)
            if self.config.system_prompt == DEFAULT_SYSTEM_PROMPT
            else ""
        )
        delivery_system = system_prompt + semantics_prompt + (
            FINAL_ANSWER_CONTRACT if self.config.system_prompt == DEFAULT_SYSTEM_PROMPT else ""
        )
        for round_index in range(1, self.config.max_tool_rounds + 1):
            messages, trimmed = self._trim_history(messages)
            context_trimmed = context_trimmed or trimmed
            if self.config.instrument:
                input_size = measure_round_input(
                    delivery_system, messages, round_tools
                )
                round_started = time.monotonic()
            response = self.provider.chat(delivery_system, messages, round_tools)
            messages.append(response.message)
            calls = response.message.tool_calls or ()
            if self.config.instrument:
                round_records.append(
                    ProviderRoundMeasure(
                        round_index=round_index,
                        provider=type(self.provider).__name__,
                        provider_latency_ms=(time.monotonic() - round_started) * 1000.0,
                        input_chars=input_size.input_chars,
                        tool_schemas_chars=input_size.tool_schemas_chars,
                        tool_schemas_count=input_size.tool_schemas_count,
                        messages_count=input_size.messages_count,
                        tool_calls_count=len(calls),
                        usage=dict(response.usage),
                    )
                )
            if not calls:
                # No tool calls: this is the final answer (any preamble text is part
                # of the history, never silently discarded). A terminating message
                # with empty text fails honest instead of surfacing a blank answer
                # (P17-A1).
                final_text = response.message.text or ""
                if not final_text.strip():
                    final_text = _EMPTY_FINAL_ANSWER_CLOSEOUT
                if named_play_target_text is not None:
                    # A named-play assistant response may not invent a Preview
                    # offer in prose. Only the structured result above can arm
                    # one. If code still lacks a unique target after the tool
                    # phase, fail closed instead of relaying model wording that
                    # the next turn could not bind safely.
                    if named_play_resolution_status == "unresolved":
                        final_text = _NAMED_PLAY_UNRESOLVED_CLOSEOUT
                    elif named_play_resolution_status == "unavailable":
                        final_text = _NAMED_PLAY_UNAVAILABLE_CLOSEOUT
                if (
                    generation_succeeded
                    and final_generation_payload is not None
                    and not delegated_action_attempted
                ):
                    # P20-Fix10: a successful generation owns the presentation.
                    # The provider's free final text is not the user output --
                    # the deterministic renderer of the authoritative batch is
                    # (one ordered list, reasons from the shared evidence,
                    # never the model's planning narration). The renderer fails
                    # closed to None on any payload outside the post-Fix09 item
                    # contract, and the model text path (still closed by the
                    # Fix08 boundary downstream) stands exactly as before.
                    rendered = render_recommendation_for_user(
                        final_generation_payload
                    )
                    if rendered is not None:
                        final_text = rendered
                return self._finish_result(
                    ProviderAgentResult(
                        final_text=final_text,
                        recommendation_payload=final_generation_payload,
                        rounds=round_index,
                        tool_executions=tuple(executions),
                        context_trimmed=context_trimmed,
                        rounds_capped=False,
                        total_elapsed_ms=round((time.monotonic() - run_started) * 1000.0, 1),
                    ),
                    round_records,
                    tool_records,
                )
            # Tool calls present (with or without preamble text): execute them all
            # through P09 and continue the loop -- a mixed content+tool_calls response
            # is real provider behavior and must complete the tool round.

            seen: dict[tuple[str, str], str] = {}
            results: list[ProviderToolResult] = []
            # Policy-owned calls are represented as a real synthetic assistant
            # tool-call message followed by matching tool results.  This keeps
            # both provider wire protocols valid: no tool result ever refers
            # to an id the provider history has not first seen as a tool call.
            policy_messages: list[ProviderMessage] = []
            policy_fallback_failed = False
            # S2: round-local final-only switch. Contract-pinned terminal tools
            # set it directly; ordinary generation success sets it explicitly,
            # while delegated success first enters its restricted action state.
            terminal_ok = False
            call_index = 0
            for raw_call in calls:
                active_selection_run_id: str | None = None
                call = _enforce_new_recommendation_freshness(turn_plan, raw_call)
                recommendation_semantics = _bind_current_track_similarity_seed(
                    recommendation_semantics, results_cache
                )
                if (
                    call.name in _GENERATION_TOOL_NAMES
                    and recommendation_semantics is not None
                    and recommendation_semantics.mode == "similarity_seed"
                    and recommendation_semantics.seed_source == "current_track"
                    and not recommendation_semantics.target_ids
                ):
                    logger.warning(
                        "%s refused: current-track similarity seed has no strict "
                        "canonical resolution",
                        call.name,
                    )
                    return self._generation_failure_result(
                        round_index=round_index,
                        executions=executions,
                        context_trimmed=context_trimmed,
                        run_started=run_started,
                        round_records=round_records,
                        tool_records=tool_records,
                    )
                call = _apply_recommendation_semantics(
                    call, recommendation_semantics
                )
                call_index += 1
                if feedback_lifecycle_owned and feedback_record_id is not None:
                    # The observation is already durable. Every remaining call
                    # in this provider message is stale planning: code will run
                    # the exact interpret/apply tail once after this round.
                    logger.warning(
                        "%s gated by the feedback closeout after %s",
                        call.name,
                        feedback_record_id,
                    )
                    closeout_content = json.dumps(
                        {
                            "outcome": "execution_error",
                            "error_code": _FEEDBACK_TURN_CLOSED_ERROR_CODE,
                            "error_message": _FEEDBACK_TURN_CLOSED_MESSAGE,
                            "payload": None,
                            "replayed": False,
                        },
                        ensure_ascii=False,
                    )
                    results.append(ProviderToolResult(call.call_id, closeout_content))
                    if self.config.instrument:
                        tool_records.append(
                            self._unexecuted_tool_record(
                                round_index=round_index,
                                call_index=call_index,
                                call=call,
                                outcome="execution_error",
                                error_code=_FEEDBACK_TURN_CLOSED_ERROR_CODE,
                                delivered_result_chars=len(closeout_content),
                            )
                        )
                    continue
                if (
                    delegated_action_authorized
                    and delegated_action_attempted
                    and call.name in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES
                ):
                    logger.warning(
                        "%s gated after the one delegated audio action", call.name
                    )
                    closeout_content = json.dumps(
                        {
                            "outcome": "execution_error",
                            "error_code": _DELEGATED_ACTION_CLOSED_ERROR_CODE,
                            "error_message": _DELEGATED_ACTION_CLOSED_MESSAGE,
                            "payload": None,
                            "replayed": False,
                        },
                        ensure_ascii=False,
                    )
                    results.append(ProviderToolResult(call.call_id, closeout_content))
                    if self.config.instrument:
                        tool_records.append(
                            self._unexecuted_tool_record(
                                round_index=round_index,
                                call_index=call_index,
                                call=call,
                                outcome="execution_error",
                                error_code=_DELEGATED_ACTION_CLOSED_ERROR_CODE,
                                delivered_result_chars=len(closeout_content),
                            )
                        )
                    continue
                if (
                    delegated_action_authorized
                    and not generation_succeeded
                    and call.name in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES
                    and not _delegated_action_targets_payload(
                        call, delegated_existing_batch_payload
                    )
                ):
                    logger.warning(
                        "%s refused: existing-batch delegated target was not bound "
                        "by recommendation facts read this turn",
                        call.name,
                    )
                    closeout_content = json.dumps(
                        {
                            "outcome": "execution_error",
                            "error_code": _DELEGATED_ACTION_UNBOUND_ERROR_CODE,
                            "error_message": _DELEGATED_ACTION_UNBOUND_MESSAGE,
                            "payload": None,
                            "replayed": False,
                        },
                        ensure_ascii=False,
                    )
                    results.append(ProviderToolResult(call.call_id, closeout_content))
                    delegated_terminal_text = (
                        "暂时无法从当前推荐中确认要播放或试听的曲目。"
                    )
                    terminal_ok = True
                    if self.config.instrument:
                        tool_records.append(
                            self._unexecuted_tool_record(
                                round_index=round_index,
                                call_index=call_index,
                                call=call,
                                outcome="execution_error",
                                error_code=_DELEGATED_ACTION_UNBOUND_ERROR_CODE,
                                delivered_result_chars=len(closeout_content),
                            )
                        )
                    continue
                if generation_succeeded:
                    allowed_after_generation: frozenset[str] = frozenset()
                    if delegated_action_authorized:
                        if delegated_playback_readback_pending:
                            allowed_after_generation = (
                                _POST_GENERATION_DELEGATED_READBACK_TOOL_NAMES
                            )
                        elif not delegated_action_attempted:
                            allowed_after_generation = (
                                _post_generation_delegated_action_names(
                                    final_generation_payload
                                )
                            )
                    action_targets_payload = (
                        call.name not in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES
                        or _delegated_action_targets_payload(
                            call, final_generation_payload
                        )
                    )
                    if (
                        call.name not in allowed_after_generation
                        or not action_targets_payload
                    ):
                        # The first successful non-empty generation closes only
                        # generation/discovery for every run. Ordinary recommendation
                        # closes every tool; explicit delegation retains the narrow
                        # state above. This guard also handles non-conforming calls
                        # emitted despite their schema being absent.
                        logger.warning(
                            "%s gated by the post-generation closeout; returning "
                            "the synthetic refusal without a P09 call",
                            call.name,
                        )
                        closeout_content = json.dumps(
                            {
                                "outcome": "execution_error",
                                "error_code": _POST_GENERATION_CLOSEOUT_ERROR_CODE,
                                "error_message": _POST_GENERATION_CLOSEOUT_MESSAGE,
                                "payload": None,
                                "replayed": False,
                            },
                            ensure_ascii=False,
                        )
                        results.append(
                            ProviderToolResult(call.call_id, closeout_content)
                        )
                        if self.config.instrument:
                            tool_records.append(
                                self._unexecuted_tool_record(
                                    round_index=round_index,
                                    call_index=call_index,
                                    call=call,
                                    outcome="execution_error",
                                    error_code=_POST_GENERATION_CLOSEOUT_ERROR_CODE,
                                    delivered_result_chars=len(closeout_content),
                                )
                            )
                        continue
                if _generic_recommendation_targets_only_current_player(
                    call, recommendation_semantics, results_cache
                ):
                    logger.warning(
                        "%s refused: generic recommendation promoted the current "
                        "player into its sole target",
                        call.name,
                    )
                    guard_content = json.dumps(
                        {
                            "outcome": "execution_error",
                            "error_code": _GENERIC_CURRENT_PLAYER_TARGET_ERROR_CODE,
                            "error_message": _GENERIC_CURRENT_PLAYER_TARGET_MESSAGE,
                            "payload": None,
                            "replayed": False,
                        },
                        ensure_ascii=False,
                    )
                    results.append(ProviderToolResult(call.call_id, guard_content))
                    if self.config.instrument:
                        tool_records.append(
                            self._unexecuted_tool_record(
                                round_index=round_index,
                                call_index=call_index,
                                call=call,
                                outcome="execution_error",
                                error_code=_GENERIC_CURRENT_PLAYER_TARGET_ERROR_CODE,
                                delivered_result_chars=len(guard_content),
                            )
                        )
                    continue
                if call.name in _GENERATION_TOOL_NAMES:
                    generation_attempts += 1
                    if generation_attempts > self.config.max_generation_attempts:
                        # P14-R4.5: every allowed generation attempt failed --
                        # terminate the loop deterministically with the fixed
                        # closeout. Live R4.4 verified that leaving the decision
                        # to the model stalls the run (rule narration, no answer).
                        logger.warning(
                            "generation tool budget (%d attempts) exhausted "
                            "without a success; terminating the loop with the "
                            "fixed closeout",
                            self.config.max_generation_attempts,
                        )
                        return self._generation_failure_result(
                            round_index=round_index,
                            executions=executions,
                            context_trimmed=context_trimmed,
                            run_started=run_started,
                            round_records=round_records,
                            tool_records=tool_records,
                        )
                key = (call.name, call.arguments)
                if key in seen or key in results_cache:
                    result_content = seen[key] if key in seen else results_cache[key]
                    results.append(ProviderToolResult(call.call_id, result_content))
                    if self.config.instrument:
                        tool_records.append(self._unexecuted_tool_record(
                            round_index=round_index,
                            call_index=call_index,
                            call=call,
                            outcome="deduped" if key in seen else "cached",
                            error_code=None,
                            delivered_result_chars=len(result_content),
                        ))
                    continue
                if (
                    call.name in _DISCOVER_TOOL_NAMES
                    and discover_executions >= self.config.max_discover_per_run
                ):
                    # P15-S3-S3B: per-run Fresh discovery budget guard. Only a
                    # would-execute call reaches here (dedupe/cache and the M2
                    # closeout answer earlier), so this is the single point where
                    # a third+ discovery is turned into a deterministic refusal
                    # instead of a real Apple Music Catalog search. The loop
                    # keeps running -- the model gets this refusal as the tool
                    # result and composes the honest user-facing answer.
                    logger.warning(
                        "catalog discovery budget (%d executions) exhausted; "
                        "returning the deterministic synthetic refusal without "
                        "a P09 call",
                        self.config.max_discover_per_run,
                    )
                    budget_content = json.dumps(
                        {
                            "outcome": "execution_error",
                            "error_code": _DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE,
                            "error_message": _discover_budget_exhausted_message(
                                self.config.max_discover_per_run
                            ),
                            "payload": None,
                            "replayed": False,
                        },
                        ensure_ascii=False,
                    )
                    results.append(ProviderToolResult(call.call_id, budget_content))
                    if self.config.instrument:
                        tool_records.append(self._unexecuted_tool_record(
                            round_index=round_index,
                            call_index=call_index,
                            call=call,
                            outcome="execution_error",
                            error_code=_DISCOVER_BUDGET_EXHAUSTED_ERROR_CODE,
                            delivered_result_chars=len(budget_content),
                        ))
                    continue
                tool_started = time.monotonic()
                strict_scope = _recommendation_scope_ids(recommendation_semantics)
                similarity_context = None
                if (
                    call.name == "generate_inferred_recommendation"
                    and recommendation_semantics is not None
                    and recommendation_semantics.mode == "similarity_seed"
                    and recommendation_semantics.seed_source == "current_track"
                    and recommendation_semantics.target_kind == "track"
                    and len(recommendation_semantics.target_ids) == 1
                ):
                    similarity_context = SimilarityExecutionContext(
                        recommendation_semantics.target_ids[0]
                    )
                strict_current_player_id = (
                    _cached_strict_current_player_canonical_id(results_cache)
                    if call.name in _GENERATION_TOOL_NAMES
                    and recommendation_semantics is not None
                    and recommendation_semantics.mode == "generic"
                    else None
                )
                tool_measure = self._execute_tool_call(
                    call,
                    fresh_promoted_ids,
                    recommendation_scope_ids=strict_scope,
                    similarity_context=similarity_context,
                )
                tool_elapsed_ms = (time.monotonic() - tool_started) * 1000.0
                if (
                    call.name in _DISCOVER_TOOL_NAMES
                    and tool_measure.executed
                    and not tool_measure.replayed
                    and tool_measure.outcome == "ok"
                ):
                    # P15-S3-S3B: only a genuine OK execution charges the
                    # budget. A journal replay returned the earlier recorded
                    # outcome without re-running the search, so it stays free;
                    # invalid arguments never executed and never charge; and a
                    # failed (execution_error) search stays free too -- one
                    # transient Catalog failure must not burn the single
                    # successful-search budget (P20-PerfFix02 failure recovery).
                    # (Fresh capture itself runs inside ``_execute_tool_call``
                    # on the RAW payload -- see its S3-S3D note.)
                    discover_executions += 1
                if (
                    call.name in _DISCOVER_TOOL_NAMES
                    and tool_measure.outcome == "ok"
                    and recommendation_semantics is not None
                    and recommendation_semantics.mode == "artist_constraint"
                ):
                    # Catalog promotion can add canonical tracks for the hard-
                    # constrained artist. Refresh through the same read-only
                    # identity projection used at turn start; otherwise the
                    # pre-discovery id allow-list would make expansion a no-op.
                    # Only exact artist matches enter the refreshed scope.
                    refreshed = _resolve_recommendation_semantics(
                        self.client, turn_plan.recommendation
                    )
                    if refreshed is not None and refreshed.target_ids:
                        recommendation_semantics = refreshed
                result_content = tool_measure.content
                seen[key] = result_content
                if call.name in _CACHE_INVALIDATING_TOOLS:
                    # A state-changing execution makes every earlier cached read stale.
                    results_cache.clear()
                else:
                    results_cache[key] = result_content
                results.append(ProviderToolResult(call.call_id, result_content))
                execution = self._execution_record(call, result_content, tool_elapsed_ms)
                executions.append(execution)
                if (
                    named_play_target_text is not None
                    and call.name == "discover_catalog_tracks"
                    and execution.outcome == "ok"
                ):
                    # Discovery does not itself carry the authoritative
                    # playback.route contract. Re-read through the existing
                    # read-only search using TurnPlan target_text, then resolve
                    # from that complete structured payload only.
                    try:
                        (
                            refreshed_call,
                            refreshed_measure,
                            refreshed_execution,
                            refreshed_elapsed_ms,
                        ) = self._execute_named_play_search(
                            named_play_target_text,
                            call_id=(
                                f"policy_named_play_refresh_{round_index}_{call_index}"
                            ),
                        )
                    except Exception:
                        logger.exception("named-play post-discovery search failed")
                        named_play_resolution_status = "unresolved"
                    else:
                        executions.append(refreshed_execution)
                        results_cache[(
                            refreshed_call.name,
                            refreshed_call.arguments,
                        )] = refreshed_measure.content
                        policy_messages.extend(
                            (
                                ProviderMessage(
                                    ProviderMessageRole.ASSISTANT,
                                    tool_calls=(refreshed_call,),
                                ),
                                ProviderMessage(
                                    ProviderMessageRole.USER,
                                    tool_results=(
                                        ProviderToolResult(
                                            refreshed_call.call_id,
                                            refreshed_measure.content,
                                        ),
                                    ),
                                ),
                            )
                        )
                        if self.config.instrument:
                            tool_records.append(
                                ProviderToolMeasure(
                                    round_index=round_index,
                                    call_index=call_index + 1,
                                    name=refreshed_call.name,
                                    outcome=refreshed_measure.outcome,
                                    error_code=refreshed_measure.error_code,
                                    duration_ms=(
                                        refreshed_elapsed_ms
                                        if refreshed_measure.executed
                                        else None
                                    ),
                                    raw_result_chars=(
                                        refreshed_measure.raw_result_chars
                                    ),
                                    delivered_result_chars=(
                                        refreshed_measure.delivered_result_chars
                                    ),
                                    truncated=refreshed_measure.truncated,
                                    replayed=refreshed_measure.replayed,
                                    executed=refreshed_measure.executed,
                                    arguments=safe_tool_arguments(
                                        refreshed_call.arguments
                                    ),
                                    origin="policy_injected",
                                )
                            )
                        resolution = (
                            _resolve_named_play_search_payload(
                                refreshed_measure.payload,
                                named_play_target_text,
                            )
                            if refreshed_measure.outcome == "ok"
                            else _NamedPlayResolution("unresolved")
                        )
                        named_play_resolution_status = resolution.status
                        if resolution.offer is not None:
                            return self._finish_result(
                                ProviderAgentResult(
                                    final_text=_render_exact_target_preview_offer(
                                        resolution.offer
                                    ),
                                    offered_action=resolution.offer,
                                    rounds=round_index,
                                    tool_executions=tuple(executions),
                                    context_trimmed=context_trimmed,
                                    rounds_capped=False,
                                    total_elapsed_ms=round(
                                        (time.monotonic() - run_started) * 1000.0,
                                        1,
                                    ),
                                ),
                                round_records,
                                tool_records,
                            )
                        if resolution.status == "ambiguous":
                            return self._finish_result(
                                ProviderAgentResult(
                                    final_text=_NAMED_PLAY_AMBIGUOUS_CLOSEOUT,
                                    rounds=round_index,
                                    tool_executions=tuple(executions),
                                    context_trimmed=context_trimmed,
                                    rounds_capped=False,
                                    total_elapsed_ms=round(
                                        (time.monotonic() - run_started) * 1000.0,
                                        1,
                                    ),
                                ),
                                round_records,
                                tool_records,
                            )
                        if resolution.status in {"unresolved", "unavailable"}:
                            return self._finish_result(
                                ProviderAgentResult(
                                    final_text=(
                                        _NAMED_PLAY_UNAVAILABLE_CLOSEOUT
                                        if resolution.status == "unavailable"
                                        else _NAMED_PLAY_UNRESOLVED_CLOSEOUT
                                    ),
                                    rounds=round_index,
                                    tool_executions=tuple(executions),
                                    context_trimmed=context_trimmed,
                                    rounds_capped=False,
                                    total_elapsed_ms=round(
                                        (time.monotonic() - run_started) * 1000.0,
                                        1,
                                    ),
                                ),
                                round_records,
                                tool_records,
                            )
                if (
                    feedback_lifecycle_owned
                    and call.name == "record_feedback"
                    and execution.outcome == "ok"
                    and isinstance(tool_measure.payload, Mapping)
                ):
                    recorded_id = tool_measure.payload.get("feedback_id")
                    if isinstance(recorded_id, str) and recorded_id:
                        feedback_record_id = recorded_id
                if delegated_action_authorized and execution.outcome == "ok":
                    captured_run_id, captured_payload = (
                        _existing_batch_payload_from_tool_result(
                            tool_name=call.name,
                            payload=tool_measure.payload,
                            active_run_id=delegated_existing_run_id,
                        )
                    )
                    if call.name == "get_active_context":
                        delegated_existing_run_id = captured_run_id
                        delegated_existing_batch_payload = None
                        active_selection_run_id = captured_run_id
                    elif captured_payload is not None:
                        delegated_existing_run_id = captured_run_id
                        delegated_existing_batch_payload = captured_payload
                if execution.outcome == "ok" and call.name in _FINAL_ANSWER_EXECUTIONS:
                    terminal_ok = True
                if self.config.instrument:
                    tool_records.append(
                        ProviderToolMeasure(
                            round_index=round_index,
                            call_index=call_index,
                            name=call.name,
                            outcome=tool_measure.outcome,
                            error_code=tool_measure.error_code,
                            duration_ms=tool_elapsed_ms if tool_measure.executed else None,
                            raw_result_chars=tool_measure.raw_result_chars,
                            delivered_result_chars=tool_measure.delivered_result_chars,
                            truncated=tool_measure.truncated,
                            replayed=tool_measure.replayed,
                            executed=tool_measure.executed,
                            arguments=safe_tool_arguments(call.arguments),
                        )
                    )
                if explicit_index_offer_requested and execution.outcome == "ok":
                    if call.name == "get_active_context":
                        assert playback_action is not None
                        assert playback_action.explicit_index is not None
                        explicit_index_offer_run_id = _registered_active_run_id_for_index(
                            tool_measure.payload, playback_action.explicit_index
                        )
                    elif (
                        call.name == "get_recommendation_run"
                        and explicit_index_offer_run_id is not None
                    ):
                        offer = _resolve_explicit_index_preview_offer(
                            turn_plan,
                            tool_measure.payload,
                            expected_run_id=explicit_index_offer_run_id,
                        )
                        if offer is not None:
                            return self._finish_result(
                                ProviderAgentResult(
                                    final_text=_render_exact_target_preview_offer(offer),
                                    offered_action=offer,
                                    recommendation_payload=final_generation_payload,
                                    rounds=round_index,
                                    tool_executions=tuple(executions),
                                    context_trimmed=context_trimmed,
                                    rounds_capped=False,
                                    total_elapsed_ms=round(
                                        (time.monotonic() - run_started) * 1000.0, 1
                                    ),
                                ),
                                round_records,
                                tool_records,
                            )
                if (
                    playback_action is not None
                    and call.name in {"play", "pause"}
                ):
                    # Non-targeted transport controls cannot satisfy the
                    # canonical identity gate used by play_track. They still
                    # have one code-owned terminal truth: an OK mutation must
                    # be followed by one authoritative player-state read, and
                    # only the expected observed state may be presented as
                    # success. The Provider does not get a closeout round in
                    # which it could reinterpret the raw OK envelope.
                    playback_control_attempt = create_playback_control_attempt(
                        call.name,
                        expected_state=(
                            "playing" if call.name == "play" else "paused"
                        ),
                    )
                    playback_control_attempt = (
                        record_playback_control_execution(
                            playback_control_attempt,
                            outcome=tool_measure.outcome,
                            error_code=tool_measure.error_code,
                        )
                    )
                    if (
                        playback_control_attempt.status
                        is ActionAttemptStatus.AWAITING_READBACK
                    ):
                        readback_call = ProviderToolCall(
                            call_id=(
                                f"policy_playback_control_{round_index}_"
                                "get_now_playing"
                            ),
                            name="get_now_playing",
                            arguments="{}",
                        )
                        readback_started = time.monotonic()
                        readback_measure = self._execute_tool_call(readback_call)
                        readback_elapsed_ms = (
                            time.monotonic() - readback_started
                        ) * 1000.0
                        executions.append(
                            self._execution_record(
                                readback_call,
                                readback_measure.content,
                                readback_elapsed_ms,
                                origin="policy_injected",
                            )
                        )
                        if self.config.instrument:
                            tool_records.append(
                                ProviderToolMeasure(
                                    round_index=round_index,
                                    call_index=call_index + 1,
                                    name=readback_call.name,
                                    outcome=readback_measure.outcome,
                                    error_code=readback_measure.error_code,
                                    duration_ms=(
                                        readback_elapsed_ms
                                        if readback_measure.executed
                                        else None
                                    ),
                                    raw_result_chars=(
                                        readback_measure.raw_result_chars
                                    ),
                                    delivered_result_chars=(
                                        readback_measure.delivered_result_chars
                                    ),
                                    truncated=readback_measure.truncated,
                                    replayed=readback_measure.replayed,
                                    executed=readback_measure.executed,
                                    arguments={},
                                    origin="policy_injected",
                                )
                            )
                        playback_control_attempt = (
                            verify_playback_control_readback(
                                playback_control_attempt,
                                (
                                    readback_measure.payload
                                    if readback_measure.outcome == "ok"
                                    else None
                                ),
                            )
                        )
                    return self._finish_result(
                        ProviderAgentResult(
                            final_text=(
                                render_verified_playback_control_result(
                                    playback_control_attempt
                                )
                            ),
                            recommendation_payload=final_generation_payload,
                            playback_control_attempt=playback_control_attempt,
                            rounds=round_index,
                            tool_executions=tuple(executions),
                            context_trimmed=context_trimmed,
                            rounds_capped=False,
                            total_elapsed_ms=round(
                                (time.monotonic() - run_started) * 1000.0, 1
                            ),
                        ),
                        round_records,
                        tool_records,
                    )
                if (
                    not delegated_action_authorized
                    and call.name in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES
                ):
                    # Named/reference play and preview do not need a
                    # SelectionGrant, but they share the same terminal truth:
                    # started=true for Preview and strict canonical readback
                    # for formal play. The action already executed once above;
                    # code now closes from ActionAttempt instead of asking the
                    # Provider to reinterpret its OK envelope.
                    route = (
                        "library"
                        if call.name == "play_track"
                        else "preview_only"
                    )
                    target_id = _delegated_action_target_id(call)
                    if target_id is None:
                        continue
                    action_attempt = mark_action_executing(
                        create_direct_action_attempt(target_id, route=route)
                    )
                    preview_started = (
                        tool_measure.payload.get("started")
                        if isinstance(tool_measure.payload, Mapping)
                        and isinstance(tool_measure.payload.get("started"), bool)
                        else None
                    )
                    action_attempt = record_action_execution(
                        action_attempt,
                        outcome=tool_measure.outcome,
                        preview_started=preview_started,
                        error_code=tool_measure.error_code,
                    )
                    if (
                        action_attempt.status
                        is ActionAttemptStatus.AWAITING_READBACK
                    ):
                        readback_call = ProviderToolCall(
                            call_id=(
                                f"policy_direct_action_{round_index}_"
                                "get_now_playing"
                            ),
                            name="get_now_playing",
                            arguments="{}",
                        )
                        readback_started = time.monotonic()
                        readback_measure = self._execute_tool_call(readback_call)
                        readback_elapsed_ms = (
                            time.monotonic() - readback_started
                        ) * 1000.0
                        executions.append(
                            self._execution_record(
                                readback_call,
                                readback_measure.content,
                                readback_elapsed_ms,
                                origin="policy_injected",
                            )
                        )
                        if self.config.instrument:
                            tool_records.append(
                                ProviderToolMeasure(
                                    round_index=round_index,
                                    call_index=call_index + 1,
                                    name=readback_call.name,
                                    outcome=readback_measure.outcome,
                                    error_code=readback_measure.error_code,
                                    duration_ms=(
                                        readback_elapsed_ms
                                        if readback_measure.executed
                                        else None
                                    ),
                                    raw_result_chars=(
                                        readback_measure.raw_result_chars
                                    ),
                                    delivered_result_chars=(
                                        readback_measure.delivered_result_chars
                                    ),
                                    truncated=readback_measure.truncated,
                                    replayed=readback_measure.replayed,
                                    executed=readback_measure.executed,
                                    arguments={},
                                    origin="policy_injected",
                                )
                            )
                        action_attempt = verify_formal_play_readback(
                            action_attempt,
                            (
                                readback_measure.payload
                                if readback_measure.outcome == "ok"
                                else None
                            ),
                        )
                    return self._finish_result(
                        ProviderAgentResult(
                            final_text=render_verified_action_result(action_attempt),
                            recommendation_payload=final_generation_payload,
                            action_attempt=action_attempt,
                            rounds=round_index,
                            tool_executions=tuple(executions),
                            context_trimmed=context_trimmed,
                            rounds_capped=False,
                            total_elapsed_ms=round(
                                (time.monotonic() - run_started) * 1000.0, 1
                            ),
                        ),
                        round_records,
                        tool_records,
                    )
                if active_selection_run_id is not None:
                    # The active run identity is now authoritative. Read that
                    # exact run, select and dispatch immediately in code before
                    # the Provider can propose a run id, canonical id or route.
                    return self._run_selection_grant(
                        turn_plan,
                        active_run_id=active_selection_run_id,
                        round_index=round_index,
                        next_call_index=call_index,
                        executions=executions,
                        context_trimmed=context_trimmed,
                        run_started=run_started,
                        round_records=round_records,
                        tool_records=tool_records,
                    )
                if call.name in _GENERATION_TOOL_NAMES:
                    generic_current_player_only = (
                        execution.outcome == "ok"
                        and _generic_recommendation_result_only_current_player(
                            tool_measure.payload,
                            recommendation_semantics,
                            strict_current_player_id,
                        )
                    )
                    generic_history_exhausted = _generic_direct_history_exhausted(
                        call,
                        recommendation_semantics,
                        error_code=tool_measure.error_code,
                        error_message=tool_measure.error_message,
                    )
                    if (
                        call.name == "generate_recommendation"
                        and generation_attempts == 1
                        and generation_attempts < self.config.max_generation_attempts
                        and (generic_current_player_only or generic_history_exhausted)
                    ):
                        recovery_call = _deterministic_generic_recovery_call(
                            call,
                            round_index=round_index,
                            call_index=call_index,
                        )
                        logger.info(
                            "[policy-injected] generic direct generation %s; "
                            "executing inferred recovery",
                            (
                                "returned only the strict current player"
                                if generic_current_player_only
                                else "was exhausted by recent-run exclusion"
                            ),
                        )
                        generation_attempts += 1
                        recovery_started = time.monotonic()
                        recovery_measure = self._execute_tool_call(
                            recovery_call,
                            fresh_promoted_ids,
                        )
                        recovery_elapsed_ms = (
                            time.monotonic() - recovery_started
                        ) * 1000.0
                        recovery_execution = self._execution_record(
                            recovery_call,
                            recovery_measure.content,
                            recovery_elapsed_ms,
                            origin="policy_injected",
                        )
                        executions.append(recovery_execution)
                        policy_messages.extend(
                            (
                                ProviderMessage(
                                    ProviderMessageRole.ASSISTANT,
                                    tool_calls=(recovery_call,),
                                ),
                                ProviderMessage(
                                    ProviderMessageRole.USER,
                                    tool_results=(
                                        ProviderToolResult(
                                            recovery_call.call_id,
                                            recovery_measure.content,
                                        ),
                                    ),
                                ),
                            )
                        )
                        if self.config.instrument:
                            tool_records.append(
                                ProviderToolMeasure(
                                    round_index=round_index,
                                    call_index=call_index + 1,
                                    name=recovery_call.name,
                                    outcome=recovery_measure.outcome,
                                    error_code=recovery_measure.error_code,
                                    duration_ms=recovery_elapsed_ms,
                                    raw_result_chars=recovery_measure.raw_result_chars,
                                    delivered_result_chars=(
                                        recovery_measure.delivered_result_chars
                                    ),
                                    truncated=recovery_measure.truncated,
                                    replayed=recovery_measure.replayed,
                                    executed=recovery_measure.executed,
                                    arguments=safe_tool_arguments(
                                        recovery_call.arguments
                                    ),
                                    origin="policy_injected",
                                )
                            )
                        call_index += 1
                        recovery_current_player_only = (
                            recovery_execution.outcome == "ok"
                            and _generic_recommendation_result_only_current_player(
                                recovery_measure.payload,
                                recommendation_semantics,
                                strict_current_player_id,
                            )
                        )
                        if (
                            recovery_execution.outcome == "ok"
                            and not recovery_current_player_only
                        ):
                            generation_succeeded = True
                            terminal_ok = not delegated_action_authorized
                            if recovery_measure.payload is not None:
                                final_generation_payload = recovery_measure.payload
                        else:
                            policy_fallback_failed = True
                        continue
                    if generic_current_player_only:
                        logger.error(
                            "generic recommendation returned only the strict current "
                            "player; failing closed without delivering the batch"
                        )
                        return self._generation_failure_result(
                            round_index=round_index,
                            executions=executions,
                            context_trimmed=context_trimmed,
                            run_started=run_started,
                            round_records=round_records,
                            tool_records=tool_records,
                        )
                    if (
                        execution.outcome == "ok"
                        and _generation_payload_contains_similarity_seed(
                            tool_measure.payload, recommendation_semantics
                        )
                    ):
                        logger.error(
                            "similarity generation returned its strict seed; "
                            "failing closed without delivering the batch"
                        )
                        return self._generation_failure_result(
                            round_index=round_index,
                            executions=executions,
                            context_trimmed=context_trimmed,
                            run_started=run_started,
                            round_records=round_records,
                            tool_records=tool_records,
                        )
                    if execution.outcome == "ok":
                        generation_succeeded = True
                        if not delegated_action_authorized:
                            terminal_ok = True
                        # P20-Fix10: capture the ok execution's raw payload for
                        # the deterministic renderer (the measure already holds
                        # it; captured here because `execution.outcome` is the
                        # per-execution success record). A payload-less ok
                        # generation (legacy replay shape) leaves the capture
                        # empty. Delegated turns fail closed through the shared
                        # grant workflow instead of asking the Provider to
                        # invent a target or route from that missing payload.
                        if tool_measure.payload is not None:
                            final_generation_payload = tool_measure.payload
                        if delegated_action_authorized:
                            return self._run_selection_grant(
                                turn_plan,
                                authoritative_payload=final_generation_payload,
                                recommendation_payload=final_generation_payload,
                                round_index=round_index,
                                next_call_index=call_index,
                                executions=executions,
                                context_trimmed=context_trimmed,
                                run_started=run_started,
                                round_records=round_records,
                                tool_records=tool_records,
                            )
                    elif (
                        call.name == "generate_recommendation"
                        and generation_attempts == 1
                        and tool_measure.executed
                        and not tool_measure.replayed
                        and execution.error_code == "empty_recommendation"
                        and recommendation_semantics is not None
                        and recommendation_semantics.mode == "preference_seed"
                    ):
                        if (
                            not preference_fallback_injected
                            and discover_executions
                            < self.config.max_discover_per_run
                            and generation_attempts
                            < self.config.max_generation_attempts
                        ):
                            preference_fallback_injected = True
                            discovery_call, inferred_call = (
                                _deterministic_preference_fallback_calls(
                                    call,
                                    recommendation_semantics,
                                    round_index=round_index,
                                    call_index=call_index,
                                )
                            )
                            logger.info(
                                "[policy-injected] preference direct generation "
                                "returned empty; executing seed-only Catalog "
                                "discovery followed by inferred retry"
                            )

                            discovery_started = time.monotonic()
                            discovery_measure = self._execute_tool_call(
                                discovery_call, fresh_promoted_ids
                            )
                            discovery_elapsed_ms = (
                                time.monotonic() - discovery_started
                            ) * 1000.0
                            if (
                                discovery_measure.executed
                                and not discovery_measure.replayed
                                and discovery_measure.outcome == "ok"
                            ):
                                discover_executions += 1
                            # A real discovery dispatch invalidates cached reads
                            # even when every returned item was already bound.
                            results_cache.clear()
                            discovery_execution = self._execution_record(
                                discovery_call,
                                discovery_measure.content,
                                discovery_elapsed_ms,
                                origin="policy_injected",
                            )
                            executions.append(discovery_execution)

                            generation_attempts += 1
                            inferred_started = time.monotonic()
                            inferred_measure = self._execute_tool_call(
                                inferred_call, fresh_promoted_ids
                            )
                            inferred_elapsed_ms = (
                                time.monotonic() - inferred_started
                            ) * 1000.0
                            inferred_execution = self._execution_record(
                                inferred_call,
                                inferred_measure.content,
                                inferred_elapsed_ms,
                                origin="policy_injected",
                            )
                            executions.append(inferred_execution)

                            policy_messages.extend(
                                (
                                    ProviderMessage(
                                        ProviderMessageRole.ASSISTANT,
                                        tool_calls=(
                                            discovery_call,
                                            inferred_call,
                                        ),
                                    ),
                                    ProviderMessage(
                                        ProviderMessageRole.USER,
                                        tool_results=(
                                            ProviderToolResult(
                                                discovery_call.call_id,
                                                discovery_measure.content,
                                            ),
                                            ProviderToolResult(
                                                inferred_call.call_id,
                                                inferred_measure.content,
                                            ),
                                        ),
                                    ),
                                )
                            )
                            # Keep trace call indices unique and in actual
                            # execution order if the provider supplied more
                            # calls in this same assistant message.
                            policy_call_index = call_index
                            if self.config.instrument:
                                for injected_index, (
                                    injected_call,
                                    injected_measure,
                                    injected_elapsed,
                                ) in enumerate(
                                    (
                                        (
                                            discovery_call,
                                            discovery_measure,
                                            discovery_elapsed_ms,
                                        ),
                                        (
                                            inferred_call,
                                            inferred_measure,
                                            inferred_elapsed_ms,
                                        ),
                                    ),
                                    start=1,
                                ):
                                    tool_records.append(
                                        ProviderToolMeasure(
                                            round_index=round_index,
                                            call_index=(
                                                policy_call_index
                                                + injected_index
                                            ),
                                            name=injected_call.name,
                                            outcome=injected_measure.outcome,
                                            error_code=(
                                                injected_measure.error_code
                                            ),
                                            duration_ms=injected_elapsed,
                                            raw_result_chars=(
                                                injected_measure.raw_result_chars
                                            ),
                                            delivered_result_chars=(
                                                injected_measure.delivered_result_chars
                                            ),
                                            truncated=injected_measure.truncated,
                                            replayed=injected_measure.replayed,
                                            executed=injected_measure.executed,
                                            arguments=safe_tool_arguments(
                                                injected_call.arguments
                                            ),
                                            origin="policy_injected",
                                        )
                                    )
                            call_index += 2
                            if inferred_execution.outcome == "ok":
                                generation_succeeded = True
                                terminal_ok = not delegated_action_authorized
                                if inferred_measure.payload is not None:
                                    final_generation_payload = (
                                        inferred_measure.payload
                                    )
                                if delegated_action_authorized:
                                    return self._run_selection_grant(
                                        turn_plan,
                                        authoritative_payload=(
                                            final_generation_payload
                                        ),
                                        recommendation_payload=(
                                            final_generation_payload
                                        ),
                                        round_index=round_index,
                                        next_call_index=call_index,
                                        executions=executions,
                                        context_trimmed=context_trimmed,
                                        run_started=run_started,
                                        round_records=round_records,
                                        tool_records=tool_records,
                                    )
                            else:
                                policy_fallback_failed = True
                    elif (
                        not generation_succeeded
                        and generation_attempts >= self.config.max_generation_attempts
                    ):
                        # P14-R4.5: the final allowed attempt failed (empty
                        # generation) with no success this run -- retry is
                        # impossible, so the loop terminates with the fixed
                        # closeout instead of feeding the failure back.
                        logger.warning(
                            "generation failed on the final allowed attempt; "
                            "terminating the loop with the fixed closeout"
                        )
                        return self._generation_failure_result(
                            round_index=round_index,
                            executions=executions,
                            context_trimmed=context_trimmed,
                            run_started=run_started,
                            round_records=round_records,
                            tool_records=tool_records,
                        )
                if (
                    delegated_action_authorized
                    and call.name in _POST_GENERATION_DELEGATED_ACTION_TOOL_NAMES
                ):
                    delegated_action_attempted = True
                    if call.name == "play_track":
                        if execution.outcome == "ok":
                            delegated_formal_target_id = _delegated_action_target_id(call)
                            delegated_playback_readback_pending = True
                        else:
                            delegated_formal_target_id = None
                            delegated_playback_readback_pending = False
                            delegated_terminal_text = _DELEGATED_PLAY_FAILED_CLOSEOUT
                            terminal_ok = True
                    elif call.name == "preview_catalog_track":
                        delegated_playback_readback_pending = False
                        preview_started = (
                            execution.outcome == "ok"
                            and isinstance(tool_measure.payload, Mapping)
                            and tool_measure.payload.get("started") is True
                        )
                        delegated_terminal_text = (
                            _DELEGATED_PREVIEW_OK_CLOSEOUT
                            if preview_started
                            else _DELEGATED_PREVIEW_FAILED_CLOSEOUT
                        )
                        terminal_ok = True
                elif (
                    delegated_action_authorized
                    and delegated_playback_readback_pending
                    and call.name == "get_now_playing"
                ):
                    delegated_playback_readback_pending = False
                    readback_confirmed = (
                        execution.outcome == "ok"
                        and _delegated_play_readback_confirms(
                            tool_measure.payload, delegated_formal_target_id
                        )
                    )
                    delegated_terminal_text = (
                        _DELEGATED_PLAY_OK_CLOSEOUT
                        if readback_confirmed
                        else _DELEGATED_PLAY_READBACK_FAILED_CLOSEOUT
                    )
                    delegated_formal_target_id = None
                    terminal_ok = True
            messages.append(
                ProviderMessage(ProviderMessageRole.USER, tool_results=tuple(results))
            )
            messages.extend(policy_messages)
            if policy_fallback_failed:
                logger.warning(
                    "policy-injected inferred retry failed on the final "
                    "allowed generation attempt; terminating with the fixed "
                    "closeout"
                )
                return self._generation_failure_result(
                    round_index=round_index,
                    executions=executions,
                    context_trimmed=context_trimmed,
                    run_started=run_started,
                    round_records=round_records,
                    tool_records=tool_records,
                )
            if feedback_lifecycle_owned and feedback_record_id is not None:
                # The provider's job ended at one successful record. Execute the
                # sealed P08 tail through the same AgentClient/P09 boundary, once,
                # then return a deterministic user closeout. No extra provider
                # final round exists, so a completed feedback mutation cannot
                # later degrade into a round-cap or sanitizer fallback.
                feedback_closeout = _FEEDBACK_RECORDED_CLOSEOUT
                for injected_offset, injected_name in enumerate(
                    ("interpret_feedback", "apply_learning"), start=1
                ):
                    injected_call = ProviderToolCall(
                        call_id=(
                            f"policy_feedback_{injected_name}_{round_index}_"
                            f"{call_index + injected_offset}"
                        ),
                        name=injected_name,
                        arguments=json.dumps(
                            {"feedback_id": feedback_record_id},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    injected_started = time.monotonic()
                    injected_measure = self._execute_tool_call(
                        injected_call, fresh_promoted_ids
                    )
                    injected_elapsed_ms = (
                        time.monotonic() - injected_started
                    ) * 1000.0
                    injected_execution = self._execution_record(
                        injected_call,
                        injected_measure.content,
                        injected_elapsed_ms,
                        origin="policy_injected",
                    )
                    executions.append(injected_execution)
                    if self.config.instrument:
                        tool_records.append(
                            ProviderToolMeasure(
                                round_index=round_index,
                                call_index=call_index + injected_offset,
                                name=injected_call.name,
                                outcome=injected_measure.outcome,
                                error_code=injected_measure.error_code,
                                duration_ms=(
                                    injected_elapsed_ms
                                    if injected_measure.executed
                                    else None
                                ),
                                raw_result_chars=injected_measure.raw_result_chars,
                                delivered_result_chars=(
                                    injected_measure.delivered_result_chars
                                ),
                                truncated=injected_measure.truncated,
                                replayed=injected_measure.replayed,
                                executed=injected_measure.executed,
                                arguments=safe_tool_arguments(
                                    injected_call.arguments
                                ),
                                origin="policy_injected",
                            )
                        )
                    if injected_execution.outcome != "ok":
                        feedback_closeout = _FEEDBACK_PARTIAL_CLOSEOUT
                        break
                return self._finish_result(
                    ProviderAgentResult(
                        final_text=feedback_closeout,
                        recommendation_payload=final_generation_payload,
                        rounds=round_index,
                        tool_executions=tuple(executions),
                        context_trimmed=context_trimmed,
                        rounds_capped=False,
                        total_elapsed_ms=round(
                            (time.monotonic() - run_started) * 1000.0, 1
                        ),
                    ),
                    round_records,
                    tool_records,
                )
            if delegated_terminal_text is not None:
                return self._finish_result(
                    ProviderAgentResult(
                        final_text=delegated_terminal_text,
                        recommendation_payload=final_generation_payload,
                        rounds=round_index,
                        tool_executions=tuple(executions),
                        context_trimmed=context_trimmed,
                        rounds_capped=False,
                        total_elapsed_ms=round(
                            (time.monotonic() - run_started) * 1000.0, 1
                        ),
                    ),
                    round_records,
                    tool_records,
                )

            # S2: ordinary generation success and completed delegated actions
            # become final-only. An incomplete delegated action gets only the
            # payload-supported action schema, or only get_now_playing after a
            # successful formal play. The guard above refuses every other call.
            if terminal_ok:
                round_tools = ()
            elif delegated_action_authorized and delegated_playback_readback_pending:
                # Formal delegated playback always requires exactly one
                # get_now_playing readback before we claim success, including
                # the existing-active-batch path where this turn generated no
                # recommendation payload of its own.
                round_tools = _tools_named(
                    tools, _POST_GENERATION_DELEGATED_READBACK_TOOL_NAMES
                )
            elif generation_succeeded and delegated_action_authorized:
                if not delegated_action_attempted:
                    round_tools = _tools_named(
                        tools,
                        _post_generation_delegated_action_names(
                            final_generation_payload
                        ),
                    )
                else:
                    round_tools = ()
        # The round bound was reached while the provider kept requesting tools.
        # There is no terminal answer message, so the intermediate history must
        # not be scanned: an assistant tool-round preamble is planning content
        # and must never be selected as the final user response (P17-A1).
        # Close out deterministically with the fixed text instead.
        rounds_capped = True
        # P20-Fix10: a successful generation still owns the presentation even
        # when the model never produced a terminal answer -- the deterministic
        # batch render is the honest closeout (same renderer, same payload
        # contract; None remits to the fixed closeout unchanged).
        closeout_text = _ROUND_CAP_CLOSEOUT
        if generation_succeeded and final_generation_payload is not None:
            rendered = render_recommendation_for_user(final_generation_payload)
            if rendered is not None:
                closeout_text = rendered
        logger.warning(
            "provider agent loop capped at %d rounds; returning the deterministic round-cap closeout",
            self.config.max_tool_rounds,
        )
        return self._finish_result(
            ProviderAgentResult(
                final_text=closeout_text,
                recommendation_payload=final_generation_payload,
                rounds=self.config.max_tool_rounds,
                tool_executions=tuple(executions),
                context_trimmed=context_trimmed,
                rounds_capped=True,
                total_elapsed_ms=round((time.monotonic() - run_started) * 1000.0, 1),
            ),
            round_records,
            tool_records,
        )

    def _unexecuted_tool_record(
        self,
        *,
        round_index: int,
        call_index: int,
        call: ProviderToolCall,
        outcome: str,
        error_code: str | None,
        delivered_result_chars: int,
    ) -> ProviderToolMeasure:
        """P15-S4-M1: the measurement record for a tool call the loop answered
        without a P09 execution (dedupe, read cache, generation/closeout/
        discover-budget gates)."""
        return ProviderToolMeasure(
            round_index=round_index,
            call_index=call_index,
            name=call.name,
            outcome=outcome,
            error_code=error_code,
            duration_ms=None,
            raw_result_chars=0,
            delivered_result_chars=delivered_result_chars,
            truncated=False,
            replayed=False,
            executed=False,
            arguments=safe_tool_arguments(call.arguments),
        )

    def _finish_result(
        self,
        result: ProviderAgentResult,
        round_records: list[ProviderRoundMeasure],
        tool_records: list[ProviderToolMeasure],
    ) -> ProviderAgentResult:
        """Close provider terminal content before any consumer receives it,
        then attach the optional instrumentation trace.
        """
        from music_agent.final_response_boundary import extract_provider_final

        result = replace(result, final_text=extract_provider_final(result.final_text))
        if not self.config.instrument:
            return result
        trace = ProviderRunTrace(
            total_ms=result.total_elapsed_ms or 0.0,
            rounds_capped=result.rounds_capped,
            context_trimmed=result.context_trimmed,
            rounds=tuple(round_records),
            tools=tuple(tool_records),
        )
        return replace(result, trace=trace)

    def _generation_failure_result(
        self,
        *,
        round_index: int,
        executions: list[ProviderLoopToolExecution],
        context_trimmed: bool,
        run_started: float,
        round_records: list[ProviderRoundMeasure],
        tool_records: list[ProviderToolMeasure],
    ) -> ProviderAgentResult:
        """Deterministic failure termination (P14-R4.5): fixed user closeout.

        The loop ends the run itself with the fixed closeout text instead of
        asking the model to decide how to answer a generation failure -- live
        R4.4 showed the model stalls (rule narration, further tool musing) and
        produces no user output. ``rounds_capped`` stays False: this is a
        deliberate failure outcome, not a round-bound truncation.
        """
        return self._finish_result(
            ProviderAgentResult(
                final_text=_GENERATION_FAILURE_CLOSEOUT,
                rounds=round_index,
                tool_executions=tuple(executions),
                context_trimmed=context_trimmed,
                rounds_capped=False,
                total_elapsed_ms=round((time.monotonic() - run_started) * 1000.0, 1),
            ),
            round_records,
            tool_records,
        )

    def _s5_prefetch_recommendation_reads(
        self, results_cache: dict[tuple[str, str], str]
    ) -> tuple[
        list[ProviderToolCall], list[ProviderToolResult], list[ProviderToolMeasure]
    ]:
        """S5: deterministic pre-read of the ordinary recommendation chain's
        three anchor reads (see ``_S5_PREFETCH_TOOL_NAMES``).

        Each read executes once, with the same ``{}`` quiet payload the
        provider typically sends, through the normal client boundary (journal/
        replay/validation all apply). Results go through the same bounding and
        envelope pipeline as provider-requested calls, so the provider sees
        exactly the texts it would have received -- just already answered. The
        loop returns them as a synthetic assistant tool-call message pair that
        ``run()`` injects before round 1; the provider's later duplicate reads
        of the same (name, arguments) are served from the read cache without a
        second execution.

        All-or-nothing: any client failure aborts the entire prefetch --
        nothing is injected and the run falls back to provider-driven reads, so
        the prefetch can never break a request. The executions are NOT part of
        ``ProviderAgentResult.tool_executions`` (that list reports what the
        provider requested); under instrumentation they ARE recorded in the
        trace with ``round_index=0`` (provider rounds are 1-based), so the
        token cost of the extra context is never hidden.
        """
        calls: list[ProviderToolCall] = []
        tool_results: list[ProviderToolResult] = []
        records: list[ProviderToolMeasure] = []
        for index, name in enumerate(_S5_PREFETCH_TOOL_NAMES, start=1):
            started = time.monotonic()
            try:
                result = self.client.call(name, {})
            except Exception as error:
                logger.warning(
                    "S5 prefetch aborted: %s failed (%s); falling back to "
                    "provider-driven reads",
                    name,
                    error,
                )
                return [], [], []
            elapsed_ms = (time.monotonic() - started) * 1000.0
            payload_text, raw_result_chars, truncated = _bound_payload_with_measure(
                result.payload
            )
            content = _tool_execution_envelope(result, payload_text)
            call_id = f"s5_prefetch_{index}_{name}"
            calls.append(
                ProviderToolCall(call_id=call_id, name=name, arguments="{}")
            )
            tool_results.append(ProviderToolResult(call_id=call_id, content=content))
            if self.config.instrument:
                records.append(
                    ProviderToolMeasure(
                        round_index=0,
                        call_index=index,
                        name=name,
                        outcome=result.outcome.value,
                        error_code=result.error_code,
                        duration_ms=elapsed_ms,
                        raw_result_chars=raw_result_chars,
                        delivered_result_chars=len(content),
                        truncated=truncated,
                        replayed=result.replayed,
                        executed=True,
                        arguments=safe_tool_arguments("{}"),
                        origin="prefetch",
                    )
                )
        for name, tool_result in zip(_S5_PREFETCH_TOOL_NAMES, tool_results):
            # The provider may still re-read any of these facts; the read
            # cache answers that repeat deterministically (identical content,
            # no second execution -- the same interplay the loop already has
            # for provider-requested reads).
            results_cache[(name, "{}")] = tool_result.content
        return calls, tool_results, records

    def _execute_named_play_search(
        self, target_text: str, *, call_id: str
    ) -> tuple[ProviderToolCall, ToolExecutionMeasure, ProviderLoopToolExecution, float]:
        """Execute the code-owned named-play read using TurnPlan text only."""
        call = ProviderToolCall(
            call_id=call_id,
            name="search_library_tracks",
            arguments=json.dumps(
                {"term": target_text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        started = time.monotonic()
        measure = self._execute_tool_call(call)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        execution = self._execution_record(
            call, measure.content, elapsed_ms, origin="policy_injected"
        )
        return call, measure, execution, elapsed_ms

    def _execute_tool_call(
        self,
        call: ProviderToolCall,
        fresh_promoted_ids: set[str] | None = None,
        *,
        recommendation_scope_ids: tuple[str, ...] | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> ToolExecutionMeasure:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as error:
            content = json.dumps(
                {"outcome": "invalid_arguments", "error": f"arguments are not JSON: {error}"},
                ensure_ascii=False,
            )
            return ToolExecutionMeasure(
                content=content,
                outcome="invalid_arguments",
                error_code=None,
                error_message=None,
                raw_result_chars=0,
                delivered_result_chars=len(content),
                truncated=False,
                replayed=False,
                executed=False,
            )
        if not isinstance(arguments, dict):
            content = json.dumps(
                {"outcome": "invalid_arguments", "error": "arguments must be a JSON object"},
                ensure_ascii=False,
            )
            return ToolExecutionMeasure(
                content=content,
                outcome="invalid_arguments",
                error_code=None,
                error_message=None,
                raw_result_chars=0,
                delivered_result_chars=len(content),
                truncated=False,
                replayed=False,
                executed=False,
            )
        call_kwargs: dict[str, object] = {}
        if call.name in _GENERATION_TOOL_NAMES and fresh_promoted_ids:
            # P15-S3-S3D: pass the authoritative same-run Fresh provenance to the
            # generation handler as the internal execution kwarg. Deterministic
            # order; never touches the model arguments, the journal request
            # payload, or any tool schema.
            call_kwargs["fresh_canonical_ids"] = tuple(sorted(fresh_promoted_ids))
        if call.name in _GENERATION_TOOL_NAMES and recommendation_scope_ids is not None:
            # Current-turn artist ownership is execution context, not a public
            # model field: it can only narrow candidates to canonical tracks
            # resolved by the application and never changes scoring/ranking.
            call_kwargs["recommendation_scope_ids"] = recommendation_scope_ids
        if similarity_context is not None:
            # The strict current-player canonical identity is code-owned turn
            # context.  It is never serialized into Provider arguments and the
            # service accepts it only for inferred generation.
            call_kwargs["similarity_context"] = similarity_context
        result = self.client.call(call.name, arguments, **call_kwargs)
        if (
            fresh_promoted_ids is not None
            and call.name in _DISCOVER_TOOL_NAMES
            and not result.replayed
            and result.outcome is AgentToolOutcome.OK
        ):
            # P15-S3-S3D corrected: Fresh capture reads the RAW structured
            # payload right here, before any bounding -- the delivered envelope
            # (and its 2000-char truncation marker) is never an authority
            # source. Live audit: real discovers serialize ~9.5KB, so the old
            # delivered-text capture silently took zero ids on every run.
            _capture_fresh_promoted_ids(result.payload, fresh_promoted_ids)
        # S5 (token-cost optimization): generation results get their own bound
        # and a deterministic tail policy. RAW size is always measured on the
        # true service payload (the trace must report what the service sent);
        # when a generation payload overflows the family bound, the
        # ``encoded_result`` scoring/provenance tail is dropped from the
        # MODEL-VISIBLE payload while items (ordered, name/artist/album/
        # playback, fresh flags) survive whole -- the only part the final
        # presentation consumes. A payload that still overflows after the
        # drop falls back to the standard head-preview marker; non-generation
        # tools keep the 2000-char bound untouched.
        # P20-Fix03: the recommendation-run detail reader joins the tail-drop
        # family with its own (smaller) bound -- explanation turns need every
        # item's compact evidence whole, never the raw encoded tail. Its
        # service payload and MCP consumers keep ``encoded_result`` unchanged.
        result_bound = _MAX_TOOL_RESULT_CHARS
        raw_text = None
        model_payload: Mapping | None = result.payload
        tail_drop_family = (
            call.name in _GENERATION_TOOL_NAMES or call.name == "get_recommendation_run"
        )
        if tail_drop_family:
            result_bound = (
                _GENERATION_RESULT_MAX_CHARS
                if call.name in _GENERATION_TOOL_NAMES
                else _RUN_READER_RESULT_MAX_CHARS
            )
            # AgentToolResult normalizes payloads to a read-only mappingproxy
            # (never a plain dict), so the drop gate tests Mapping -- the
            # mutable dict isinstance would silently skip the tail policy for
            # every real service result.
            if isinstance(result.payload, Mapping) and "encoded_result" in result.payload:
                raw_text = json.dumps(dict(result.payload), ensure_ascii=False)
                if len(raw_text) > result_bound:
                    model_payload = {
                        key: value
                        for key, value in result.payload.items()
                        if key != "encoded_result"
                    }
        payload_text, raw_result_chars, truncated = _bound_payload_with_measure(
            model_payload, max_chars=result_bound
        )
        if raw_text is not None:
            raw_result_chars = len(raw_text)
        content = _tool_execution_envelope(result, payload_text)
        # Preserve the successful RAW structured payload for code-owned
        # follow-up policies. Generation uses it for deterministic batch
        # presentation; feedback uses record_feedback.feedback_id for the
        # deterministic interpret -> apply tail. The model-visible envelope
        # remains independently bounded above.
        raw_payload: Mapping | None = None
        if (
            result.outcome is AgentToolOutcome.OK
            and isinstance(result.payload, Mapping)
        ):
            raw_payload = result.payload
        return ToolExecutionMeasure(
            content=content,
            outcome=result.outcome.value,
            error_code=result.error_code,
            error_message=result.error_message,
            raw_result_chars=raw_result_chars,
            delivered_result_chars=len(content),
            truncated=truncated,
            replayed=result.replayed,
            executed=True,
            payload=raw_payload,
        )

    def _execution_record(
        self,
        call: ProviderToolCall,
        result_content: str,
        elapsed_ms: float | None = None,
        *,
        origin: str = "provider_requested",
    ) -> ProviderLoopToolExecution:
        outcome = "ok"
        error_code = None
        try:
            parsed = json.loads(result_content)
            if isinstance(parsed, dict):
                outcome = str(parsed.get("outcome", "ok"))
                code = parsed.get("error_code")
                error_code = str(code) if code is not None else None
        except json.JSONDecodeError:
            pass
        return ProviderLoopToolExecution(
            name=call.name,
            outcome=outcome,
            error_code=error_code,
            elapsed_ms=round(elapsed_ms, 1) if elapsed_ms is not None else None,
            origin=origin,
        )

    def _trim_history(self, messages: list[ProviderMessage]) -> tuple[list[ProviderMessage], bool]:
        """Keep the most recent messages; system text is not part of history here."""
        if len(messages) <= self.config.max_context_messages:
            return messages, False
        return messages[-self.config.max_context_messages :], True


def generation_succeeded(
    tool_executions: Sequence[object],
) -> bool:
    """P19-T14-B: True when at least one generation tool executed "ok" this run.

    ``tool_executions`` is the run result's public per-execution record
    (``ProviderAgentResult.tool_executions``): name + outcome strings, exactly
    as ``_execution_record`` derived them from the tool envelopes. A journal
    replay still counts (its recorded outcome is ok); the reply door pairs this
    test with its own before/after latest-run comparison to decide whether a
    NEW batch exists for this turn.
    """
    for execution in tool_executions:
        if (
            getattr(execution, "name", None) in _GENERATION_TOOL_NAMES
            and getattr(execution, "outcome", None) == "ok"
        ):
            return True
    return False


# A numbered list item as models write them ("1. ", "2、", "3)") -- any digit
# count, punctuation-tolerant leading whitespace.
_NUMBERED_ITEM_LINE = re.compile(r"^\s*\d{1,3}\s*[.、)）]\s*")


# P19-T14-E: Play-vs-Preview guard facts. A play-intent turn must never end
# with preview audio; these pure helpers read the run's public execution
# record so the caller-side doors (web shell / CLI) can stop a stray preview
# and replace the reply honestly. They execute nothing -- ``tool_executions``
# is ``ProviderAgentResult.tool_executions``, the same name+outcome record
# ``generation_succeeded`` reads ("ok" is exactly ``_execution_record``'s
# success outcome).
_PREVIEW_EXECUTION_NAMES: frozenset[str] = frozenset(
    ("preview_catalog_track", "preview_batch")
)

# The formal-playback family: play_track selects and plays; play resumes the
# current formal context (the only formal meaning a bare 播放 can have).
_FORMAL_PLAY_EXECUTION_NAMES: frozenset[str] = frozenset(("play_track", "play"))

# The exact honest sentence the product contract requires instead of a silent
# preview downgrade (「这首目前无法正式播放，可以试听 30 秒。」).
PLAY_PREVIEW_DOWNGRADE_FALLBACK = "这首目前无法正式播放，可以试听 30 秒。"
_NAMED_PLAY_AMBIGUOUS_CLOSEOUT = "找到多个可能的同名版本，请告诉我歌手或更具体的版本。"
_NAMED_PLAY_UNRESOLVED_CLOSEOUT = "暂时无法唯一确认你要播放的歌曲，请告诉我歌手或更具体的版本。"
_NAMED_PLAY_UNAVAILABLE_CLOSEOUT = "这首目前没有可正式播放或试听的版本。"
_NAMED_PLAY_OFFER_SOURCE = "structured_named_play_resolution"
_EXPLICIT_INDEX_OFFER_SOURCE = "active_recommendation_explicit_index"


@dataclass(frozen=True, slots=True)
class _NamedPlayResolution:
    status: str
    offer: OfferedAction | None = None


def _named_play_match_key(value: object) -> str | None:
    """Conservative user-text/item matcher; never an identity resolver by itself."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"\s+", "", value.strip().casefold())
    paired = (("《", "》"), ("〈", "〉"), ("“", "”"), ('"', '"'), ("'", "'"))
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in paired:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : len(text) - len(right)]
                changed = True
                break
    return text or None


def _named_play_item_matches_target(item: Mapping, target_text: str) -> bool:
    target = _named_play_match_key(target_text)
    title = _named_play_match_key(item.get("name"))
    artist = _named_play_match_key(item.get("artist_name"))
    if target is None or title is None:
        return False
    if target == title:
        return True
    # Support an explicit artist+title phrase without turning fuzzy search into
    # authority. Both structured facts must literally occur in the user's own
    # target text; the Provider's search term is never consulted here.
    return bool(artist and title in target and artist in target)


def _build_exact_target_preview_offer(
    *,
    canonical_id: object,
    playback_route: object,
    source: str,
    verified_title: object = None,
    verified_artist: object = None,
    formal_play_requested: bool,
    confirmation_required: bool = True,
) -> OfferedAction | None:
    """Produce the one structured Preview offer from an exact code-owned target.

    Target resolution is intentionally outside this seam: named search, an active
    RecommendationRun index, a referent, or a future SelectionGrant may establish
    exact authority differently.  Once exact authority exists, however, a formal
    play request may offer Preview only when the authoritative route is
    ``preview_only`` and confirmation is required.  Missing identity/eligibility
    fails closed; assistant prose is never consulted.
    """

    if (
        not formal_play_requested
        or not confirmation_required
        or playback_route != "preview_only"
        or not isinstance(canonical_id, str)
        or not canonical_id
    ):
        return None
    title = verified_title if isinstance(verified_title, str) and verified_title else None
    artist = verified_artist if isinstance(verified_artist, str) and verified_artist else None
    return OfferedAction(
        kind=PREVIEW_TRACK,
        target_canonical_id=canonical_id,
        source=source,
        verified_title=title,
        verified_artist=artist,
    )


def _explicit_index_preview_offer_requested(turn_plan: TurnPlan) -> bool:
    action = turn_plan.playback_action
    return bool(
        action
        and action.kind == "play"
        and action.source == "active_recommendation"
        and action.selection_mode == "explicit_index"
        and isinstance(action.explicit_index, int)
        and not isinstance(action.explicit_index, bool)
        and action.explicit_index > 0
        and action.explicit_play
    )


def _registered_active_run_id_for_index(
    payload: Mapping | None, explicit_index: int
) -> str | None:
    """Return only a live session-register run that can contain the index.

    ``source=derived`` is historical fallback, not sufficient authority for
    arming a next-turn action.  This deliberately makes stale/restarted session
    context fail closed for the structured continuation.
    """

    if not isinstance(payload, Mapping):
        return None
    active = payload.get("active_batch")
    if not isinstance(active, Mapping) or active.get("source") != "register":
        return None
    run_id = active.get("run_id")
    item_count = active.get("item_count")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(item_count, int)
        or isinstance(item_count, bool)
        or item_count < explicit_index
    ):
        return None
    return run_id


def _resolve_explicit_index_preview_offer(
    turn_plan: TurnPlan, payload: Mapping | None, *, expected_run_id: str
) -> OfferedAction | None:
    """Resolve one exact RecommendationRun position into the common offer seam.

    The run id comes from the live active-batch register read in this same turn.
    Position, canonical identity, title/artist and route come only from the exact
    ``get_recommendation_run`` payload.  Duplicate/missing positions are
    ambiguous and therefore fail closed.
    """

    if not _explicit_index_preview_offer_requested(turn_plan):
        return None
    if not isinstance(payload, Mapping) or payload.get("run_id") != expected_run_id:
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    action = turn_plan.playback_action
    assert action is not None and action.explicit_index is not None
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("position") == action.explicit_index
    ]
    if len(matches) != 1:
        return None
    item = matches[0]
    playback = item.get("playback")
    route = playback.get("route") if isinstance(playback, Mapping) else None
    return _build_exact_target_preview_offer(
        canonical_id=item.get("target_id"),
        playback_route=route,
        source=_EXPLICIT_INDEX_OFFER_SOURCE,
        verified_title=item.get("name"),
        verified_artist=item.get("artist_name"),
        formal_play_requested=True,
        confirmation_required=True,
    )


def _render_exact_target_preview_offer(action: OfferedAction) -> str:
    """Render only from the same structured exact-target offer the host arms."""
    if action.verified_title and action.verified_artist:
        subject = f"《{action.verified_title}》— {action.verified_artist}"
    elif action.verified_title:
        subject = f"《{action.verified_title}》"
    else:
        subject = "这首歌"
    return f"{subject} 目前无法正式播放，可以试听 30 秒。需要我开始试听吗？"


def _resolve_named_play_search_payload(
    payload: Mapping | None, target_text: str
) -> _NamedPlayResolution:
    """Resolve one named-play target solely from complete structured search facts.

    ``search_library_tracks`` reports the total ``matched_count`` separately
    from the displayed ``matches``.  Authority is granted only when the payload
    covers the full result set, so a Provider-chosen small limit can never hide
    a second same-name candidate and manufacture false uniqueness.
    """
    if not isinstance(payload, Mapping):
        return _NamedPlayResolution("unresolved")
    matches = payload.get("matches")
    matched_count = payload.get("matched_count")
    if not isinstance(matches, list) or not isinstance(matched_count, int):
        return _NamedPlayResolution("unresolved")
    if matched_count != len(matches):
        return _NamedPlayResolution("unresolved")

    by_id: dict[str, Mapping] = {}
    for item in matches:
        if not isinstance(item, Mapping) or not _named_play_item_matches_target(
            item, target_text
        ):
            continue
        target_id = item.get("target_id")
        if isinstance(target_id, str) and target_id:
            by_id[target_id] = item
    if not by_id:
        return _NamedPlayResolution("unresolved")
    if len(by_id) != 1:
        return _NamedPlayResolution("ambiguous")

    canonical_id, item = next(iter(by_id.items()))
    playback = item.get("playback")
    route = playback.get("route") if isinstance(playback, Mapping) else None
    if route == "library":
        return _NamedPlayResolution("library")
    if route != "preview_only":
        return _NamedPlayResolution("unavailable")
    title = item.get("name") if isinstance(item.get("name"), str) else None
    artist = (
        item.get("artist_name")
        if isinstance(item.get("artist_name"), str)
        else None
    )
    offer = _build_exact_target_preview_offer(
        canonical_id=canonical_id,
        playback_route=route,
        source=_NAMED_PLAY_OFFER_SOURCE,
        verified_title=title,
        verified_artist=artist,
        formal_play_requested=True,
        confirmation_required=True,
    )
    return _NamedPlayResolution("preview_only", offer)


def preview_path_started(tool_executions: Sequence[object]) -> bool:
    """True when any preview tool executed "ok" this run (audio really started)."""
    return any(
        getattr(execution, "name", None) in _PREVIEW_EXECUTION_NAMES
        and getattr(execution, "outcome", None) == "ok"
        for execution in tool_executions
    )


def formal_play_started(tool_executions: Sequence[object]) -> bool:
    """True when formal playback (play_track or play) executed "ok" this run."""
    return any(
        getattr(execution, "name", None) in _FORMAL_PLAY_EXECUTION_NAMES
        and getattr(execution, "outcome", None) == "ok"
        for execution in tool_executions
    )


def play_intent_preview_downgrade(
    user_text: str, tool_executions: Sequence[object]
) -> bool:
    """True when an explicit play request degraded to preview (T14-E door).

    The composition of three facts: the user asked for formal playback, the
    run opened the preview path successfully, and no formal playback executed
    "ok". Callers replace the run's answer with
    ``PLAY_PREVIEW_DOWNGRADE_FALLBACK`` (after stopping the preview through
    their authoritative client). The residual case -- formal playback OK and
    a preview also started -- returns False (the answer stands) but still
    needs the stray preview stopped: callers pair this test with
    ``preview_path_started`` for that branch.
    """
    from music_agent.intent_router import is_explicit_play_intent

    return (
        is_explicit_play_intent(user_text)
        and preview_path_started(tool_executions)
        and not formal_play_started(tool_executions)
    )


def action_result_preview_started(result: object) -> bool:
    """Read preview start truth from ActionAttempt, with a legacy trace fallback."""

    attempt = getattr(result, "action_attempt", None)
    if isinstance(attempt, ActionAttempt) and attempt.expected_route == "preview_only":
        return attempt.status is ActionAttemptStatus.COMPLETED
    executions = getattr(result, "tool_executions", ())
    return preview_path_started(executions)


def action_result_play_preview_downgrade(
    user_text: str, result: object
) -> bool:
    """Apply the play/preview guard using structured terminal truth when present."""

    from music_agent.intent_router import is_explicit_play_intent

    if not is_explicit_play_intent(user_text):
        return False
    if not action_result_preview_started(result):
        return False
    attempt = getattr(result, "action_attempt", None)
    if isinstance(attempt, ActionAttempt) and attempt.expected_route == "library":
        formal_completed = attempt.status is ActionAttemptStatus.COMPLETED
    else:
        executions = getattr(result, "tool_executions", ())
        formal_completed = formal_play_started(executions)
    return not formal_completed


def looks_like_numbered_song_list(text: str) -> bool:
    """P19-T14-B: a conservative shape probe for the model's song pseudo-lists.

    True only when the reply contains at least TWO numbered lines AND at least
    one of those numbered lines uses the product's own track separator (``—``,
    as in 歌名 — 艺人). It is a SUPPRESSION gate for the gap case (a
    recommendation that finished with no generated batch), never a card parser
    and never a list renderer: the reply door replaces the whole text with the
    fixed one-sentence fallback, it does not extract items from it.

    Deliberately narrow so legitimate prose survives: an options list without
    track separators (loop closeouts, capability lists) never matches, and a
    single track mention cannot match on its own.
    """
    if not isinstance(text, str):
        return False
    numbered = [line for line in text.splitlines() if _NUMBERED_ITEM_LINE.match(line)]
    return len(numbered) >= 2 and any("—" in line for line in numbered)


def _bound_payload_with_measure(
    payload: Mapping | None,
    max_chars: int = _MAX_TOOL_RESULT_CHARS,
) -> tuple[dict | None, int, bool]:
    """P15-S4-M1: bound one tool payload and measure its raw serialized size.

    ``bounded`` is byte-identical to the pre-instrumentation ``_bounded_payload``
    output (S5 added only the per-call bound, defaulting to the module constant
    so every existing call site is unchanged); ``raw_chars`` counts the full
    JSON text before bounding (0 when the payload was None -- nothing was
    serialized); ``truncated`` reports whether the provider received the
    preview marker instead of the payload.
    """
    if payload is None:
        return None, 0, False
    text = json.dumps(dict(payload), ensure_ascii=False)
    if len(text) > max_chars:
        return {"truncated": True, "preview": text[:max_chars]}, len(text), True
    return dict(payload), len(text), False


def _tool_execution_envelope(
    result: AgentToolResult, payload_text: dict | None
) -> str:
    """The model-visible result envelope of one service execution.

    S5 extracted the construction out of ``_execute_tool_call`` so the loop's
    prefetch path (``_s5_prefetch_recommendation_reads``) delivers results in
    exactly the same shape, byte-for-byte, as a provider-requested call.
    """
    return json.dumps(
        {
            "outcome": result.outcome.value,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "payload": payload_text,
            "replayed": result.replayed,
        },
        ensure_ascii=False,
    )


def _bounded_payload(payload: Mapping | None) -> dict | None:
    bounded, _, _ = _bound_payload_with_measure(payload)
    return bounded
