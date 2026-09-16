"""P09.5: SharedAgentService -- the shared agent/service layer composition root.

One :class:`SharedAgentService` owns every domain repository over a single SQLite store and
exposes the registered agent tools to model clients. Models are clients of this layer, never
independent sources of user-state truth: every canonical music, preference, recommendation,
feedback, and learning fact is read from -- and every durable mutation is written through -- the
existing P06/P07/P08 repositories the service composes. No per-model state exists anywhere.

Execution semantics for one request (fixed order, all fail closed):

1. contract version check -- anything other than the current agent contract refuses;
2. replay check -- a journaled request id returns its recorded outcome (``replayed=True``) for
   an identical payload, and refuses with ``replay_conflict`` for a different payload;
3. tool lookup -- unknown tools refuse with ``tool_not_supported``;
4. payload envelope validation -- malformed payloads refuse with ``invalid_request``;
5. permission decision -- unregistered clients refuse with ``unknown_client``; client policy
   refusals with ``permission_denied``; live writes delegate to the sealed capability matrix and
   refuse with ``not_execution_ready`` (nothing is execution-ready today);
6. execution -- domain errors surface as ``execution_error`` carrying the domain's stable
   ``error_code``; unexpected non-domain exceptions propagate (never silently succeed);
7. journaling -- every completed request/refusal (except replay answers, whose row already
   exists) is recorded append-only under its ``req_`` identity.

Domain-layer calibration: P06 owns no policy defaults, so the service owns the production
calibration values below (the same thresholds the phase tests have used as canonical constants).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import unicodedata

logger = logging.getLogger(__name__)

from music_agent.active_music_context import ActiveMusicContext, VerifiedSelection
from music_agent.audio_safety import device_safety_trace, device_safety_trace_enabled
from music_agent.device_context import AudioOutputSnapshot
from music_agent.playback_context import PlaybackContext, SuspendedPlaybackEntry
from music_agent.preview_session import (
    PreviewSessionItem,
    PreviewSessionSnapshot,
    PreviewSessionState,
)
from music_agent.safety_pause import SafetyPauseAction, SafetyPausePolicy
from music_agent.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
    encode_agent_payload,
)
from music_agent.agent_permission import (
    AgentClientRegistry,
    decide_agent_permission,
    write_capability_summary,
)
from music_agent.agent_request_journal_repository import AgentRequestJournalRepository
from music_agent.agent_tools import AGENT_TOOL_REGISTRY, AgentToolName
from music_agent.write_intent import WRITE_CAPABILITY_MATRIX, WriteOperation
from music_agent.direct_track_preference import DirectPreferenceMagnitudePolicy
from music_agent.confidence_derivation import (
    ConfidenceDerivationPolicy,
    ConservativeConfidencePolicy,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.familiarity import FamiliarityNormalizationPolicy
from music_agent.feedback_contract import (
    AttributionRelation,
    FeedbackAttribution,
    FeedbackKind,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
    encode_feedback_observation,
    generate_feedback_id,
)
from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.feedback_interpretation import InterpretationPolicy, interpret_observation
from music_agent.intent_repository import PendingIntentRepository
from music_agent.learning_application import LearningApplicationRepository
from music_agent.learning_effect import LearningEffectPolicy, derive_learning_effect
from music_agent.learning_policy import LearningPolicy, propose_preference_update
from music_agent.preference_attribution import (
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_propagation import canonicalize_genre_key
from music_agent.preference_query import TrackPreferenceState, query_track_preference
from music_agent.preference_strength import PreferenceState
from music_agent.preference_signal import RatingBandPolicy, SignalDirection
from music_agent.catalog_track_state_repository import (
    CatalogTrackState,
    CatalogTrackStateRepository,
    normalize_discovery_term,
)
from music_agent.known_catalog_supply import summarize_known_catalog_supply
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    assemble_recommendation_result,
    encode_recommendation_result,
    generate_candidate_id,
    generate_run_id,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.recommendation_execution_service import RecommendationExecutionService
from music_agent.recommendation_quality import (
    REASON_PREVIOUSLY_RECOMMENDED,
    QualityEvidence,
    collect_previous_targets,
)
from music_agent.recommendation_ranking import (
    RankingOutcome,
    build_recommendation,
    rank_recommendations,
)
from music_agent.repository import CanonicalRepository
from music_agent.sibling_dedupe import (
    exclude_historical_siblings,
    has_sibling_duplicate,
    select_distinct_works,
)
from music_agent.track_similarity import (
    SIMILARITY_SOURCE_PATH,
    SimilarityExecutionContext,
    TrackSimilarityEvidence,
    score_track_similarity,
)
from music_agent.write_execution_repository import WriteExecutionRepository
from music_agent.write_orchestrator import WriteCommandAdapter, WriteOrchestrator

# P3B: how many latest recommendation runs count as "recent" when judging whether the
# Music.app current track belongs to Agent recommendations (get_now_playing context field).
_NOW_PLAYING_CONTEXT_RUN_WINDOW = 3
# P14-R4.2: how many latest recommendation runs avoid_previous_runs folds in as
# repeat-control evidence. Bounded on purpose: repeats are suppressed in the short
# term, while recommendations beyond the window may re-enter the candidate pool.
_AVOID_PREVIOUS_RUNS_WINDOW = 5
# P14-R3.1: cap for search_library_tracks when the caller omits limit.
_LIBRARY_SEARCH_DEFAULT_LIMIT = 10
# P14-R3.2: playback-capability rank for the same-name sort (library > preview_only >
# unavailable), mirroring the _playback_annotation view without touching its rules.
_LIBRARY_ROUTE_RANK = {"library": 0, "preview_only": 1, "unavailable": 2}
# P15-S3-S3D: the two recommendation-producing handlers are the only tool handlers
# that accept the internal same-run Fresh provenance kwarg. Everything else keeps the
# payload-only handler signature.
_FRESH_AWARE_GENERATION_TOOL_NAMES: frozenset[AgentToolName] = frozenset(
    {
        AgentToolName.GENERATE_RECOMMENDATION,
        AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
    }
)

# Durable replay discriminator for code-owned execution context.  This key is
# never accepted by a public tool schema and never passed to a handler; it lives
# only in the append-only request journal so one request id cannot replay a run
# produced for another strict similarity seed.
_JOURNALED_EXECUTION_CONTEXT_KEY = "__music_agent_execution_context"


class SharedAgentServiceError(ValueError):
    code = "shared_agent_service_error"


class SharedAgentServiceValidationError(SharedAgentServiceError):
    code = "validation_error"


class SimilaritySeedUnavailableError(SharedAgentServiceError):
    code = "similarity_seed_unavailable"


class CanonicalEntityNotFoundError(SharedAgentServiceError):
    code = "canonical_entity_not_found"


class RecommendationRunNotFoundError(SharedAgentServiceError):
    code = "recommendation_run_not_found"


class FeedbackNotFoundError(SharedAgentServiceError):
    code = "feedback_not_found"


class LearningApplicationNotFoundError(SharedAgentServiceError):
    code = "learning_application_not_found"


class WriteIntentNotFoundError(SharedAgentServiceError):
    code = "write_intent_not_found"


class WriteAdapterUnavailableError(SharedAgentServiceError):
    code = "write_adapter_unavailable"


class PlaybackUnavailableError(SharedAgentServiceError):
    code = "playback_unavailable"


class PlaybackCommandFailedError(SharedAgentServiceError):
    code = "playback_command_failed"


class CatalogDiscoveryUnavailableError(SharedAgentServiceError):
    code = "catalog_discovery_unavailable"


class CatalogDiscoveryError(SharedAgentServiceError):
    """A catalog transport failure surfaced as a typed service error.

    The transport's stable code (e.g. ``catalog_credentials_missing``) is preserved so the
    provider envelope can act on the specific cause.
    """

    code = "catalog_discovery_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class PreviewSessionUnavailableError(SharedAgentServiceError):
    """P15-S1: ``preview_batch`` cannot assemble any queue -- no active batch, or none
    of its items has a previewable route. Fail closed: a session is never half-built.
    """

    code = "preview_session_unavailable"


class AdvancePreviewUnavailableError(SharedAgentServiceError):
    """P15-S1 C02: ``advance_preview`` found no live session to advance.

    Terminal sessions are cleared from the register at their end, so this is the
    no-session (or raced-to-terminal) answer -- the tool never invents a session
    and never falls back to Music.app's next_track.
    """

    code = "advance_preview_unavailable"


_EMPTY_GENERATION_PREFIX = (
    "generation produced zero items; nothing was written to recommendation history."
)

# P15-S4-M2-2: machine reasons for an empty generation, chosen by the funnel's
# real execution order (direction filter -> direct evidence directionality ->
# candidate pool -> quality policy). The provider loop reads the emitted
# diagnostics to pick its next move instead of guessing between "relax genres"
# and "discover more".
_REASON_DIRECTION_FILTERED_ALL_TARGETS = "direction_filtered_all_targets"
_REASON_NO_DIRECT_EVIDENCE = "no_direct_evidence"
_REASON_ALL_TARGETS_NEGATIVE = "all_targets_negative_evidence"
_REASON_ALL_EXCLUDED = "all_eligible_candidates_excluded"
_REASON_NO_CANDIDATES_IN_POOL = "no_candidates_in_pool"
_REASON_EMPTY_GENERATION = "empty_generation"


class EmptyRecommendationError(SharedAgentServiceError):
    """A generation that produced zero items is refused before persistence (P14-R2).

    Empty runs must never enter recommendation history -- they pollute the
    history and could become the active batch pointer. The refusal surfaces to
    the model as a tool error so it can relax criteria and retry.

    P15-S4-M2-2: ``error_message`` now carries ``diagnostics`` -- real counts
    from this execution's funnel plus a machine ``reason`` and a
    ``recommended_next_action`` for the provider loop (see
    :func:`_empty_generation_diagnostics`). The payload stays None and the
    error code stays ``empty_recommendation``; only the delivered message gains
    the structured suffix.
    """

    code = "empty_recommendation"

    def __init__(self, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        message = _EMPTY_GENERATION_PREFIX
        if diagnostics:
            message += " diagnostics: " + json.dumps(
                diagnostics, ensure_ascii=False, sort_keys=True
            )
        super().__init__(message)
        self.diagnostics = diagnostics


# --- production calibration (owned by the service layer, not by P06) -----

DEFAULT_SOURCE_SYSTEM = "apple_music"

PRODUCTION_RATING_POLICY = RatingBandPolicy(positive_threshold=70, negative_threshold=30)
PRODUCTION_MAGNITUDE_POLICY = DirectPreferenceMagnitudePolicy(0.9, 0.8)


@dataclass(frozen=True, slots=True)
class _ProductionFamiliarityPolicy:
    """Deterministic production familiarity calibration: saturates at ``cap`` play counts."""

    cap: float

    def normalize(self, play_count: int) -> float:
        return min(1.0, play_count / self.cap)


PRODUCTION_FAMILIARITY_POLICY: FamiliarityNormalizationPolicy = _ProductionFamiliarityPolicy(10.0)

# P13-C02/C03: read-side reliability. The derivation policy maps durable evidence onto the
# seven S5 components; the aggregation policy folds a claim into a bounded score. Both are
# immutable stateless values owned by the service layer, exactly like the other PRODUCTION_*
# calibrations. Freshness stays descriptive (weight 0): no temporal decay.
PRODUCTION_CONFIDENCE_DERIVATION_POLICY = ConfidenceDerivationPolicy()
PRODUCTION_CONFIDENCE_AGGREGATION_POLICY = ConservativeConfidencePolicy()


class SharedAgentService:
    """The shared agent layer: one store, one registry, one permission gate, all tools."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clients: AgentClientRegistry,
        write_adapter: WriteCommandAdapter | None = None,
        playback_adapter=None,
        playback_resolver=None,
        catalog_library_transport=None,
        preview_runner=None,
        catalog_search_source=None,
        capability_matrix=None,
    ) -> None:
        if not isinstance(clients, AgentClientRegistry):
            raise SharedAgentServiceValidationError("clients must be an AgentClientRegistry")
        if playback_adapter is not None and not all(
            callable(getattr(playback_adapter, method, None))
            for method in ("play", "pause", "next_track", "previous_track", "play_track")
        ):
            raise SharedAgentServiceValidationError(
                "playback_adapter must provide play/pause/next_track/previous_track/play_track"
            )
        if playback_resolver is not None and not callable(
            getattr(playback_resolver, "resolve_playback_track", None)
        ):
            raise SharedAgentServiceValidationError(
                "playback_resolver must provide resolve_playback_track"
            )
        if catalog_library_transport is not None and not all(
            callable(getattr(catalog_library_transport, method, None))
            for method in ("add_song", "search_library_songs")
        ):
            raise SharedAgentServiceValidationError(
                "catalog_library_transport must provide add_song and search_library_songs"
            )
        if preview_runner is not None and not all(
            callable(getattr(preview_runner, method, None))
            for method in ("start_audio", "stop_preview", "is_preview_active")
        ):
            raise SharedAgentServiceValidationError(
                "preview_runner must provide start_audio, stop_preview and is_preview_active"
            )
        if preview_runner is None:
            # P11-T4: the production preview boundary is the local afplay runner; callers
            # that need a deterministic boundary inject one explicitly.
            from music_agent.catalog_preview import AfplayPreviewRunner

            preview_runner = AfplayPreviewRunner()
        if catalog_search_source is not None and not callable(
            getattr(catalog_search_source, "search", None)
        ):
            raise SharedAgentServiceValidationError(
                "catalog_search_source must provide search(term, limit)"
            )
        self.database_path = Path(database_path)
        self._clients = clients
        self._write_adapter = write_adapter
        self._playback_adapter = playback_adapter
        self._playback_resolver = playback_resolver
        self._catalog_library_transport = catalog_library_transport
        self._preview_runner = preview_runner
        self._catalog_search_source = catalog_search_source
        self._capability_matrix = capability_matrix or WRITE_CAPABILITY_MATRIX
        # P14-C06.1: the transient music-interaction register is one in-memory,
        # single-writer ActiveMusicContext (see that module for the boundaries).
        # Semantics are unchanged from the former ad-hoc dict: channel records the
        # service's last audio action (``none``/``library``/``preview``), canonical_id
        # names that track, and persistent_id anchors the last library pid actually
        # commanded (the strongest ownership evidence). Queue navigation and
        # stop_preview clear the channel but keep the anchor; a new service instance
        # always starts at ``none``. Preview-active truth is deliberately not mirrored
        # here -- the preview runner owns it.
        self._active_context = ActiveMusicContext()
        # P15-PC: the playback-continuity coordination holder (the suspension
        # register now; the P15-S1 preview-session register joins it in that slice).
        self._playback_context = PlaybackContext()
        # P15-S2 r2/r3: deterministic safety-pause policy (classifier + one-effect
        # dedup). Pure decision state -- no playback register, no resume. As of r3
        # this service is the single safety authority: the runtime's audio-output
        # observer forwards every default-output transition to
        # handle_default_output_transition, and the former P10.5 monitor's own
        # transport heuristic + pause path is gone.
        self._safety_pause_policy = SafetyPausePolicy()
        # P15-S1: opt-in presenter hook. The CLI registers a callback to print
        # progress/completion/cancellation notices for continuous preview sessions;
        # unattended services leave it None. Listener failures never break audio.
        self.preview_event_handler = None
        self._wire_preview_finish_hook()
        self._canonical = CanonicalRepository(self.database_path)
        self._preference = PreferencePersistenceRepository(self.database_path)
        self._recommendation_history = RecommendationHistoryRepository(self.database_path)
        # P15-S3: derived long-term memory per canonical Catalog track (migration 0019).
        # Written by the catalog ingestion orchestrator (discovery occurrences) and by the
        # two generate tool handlers right after their save_result commits (recommendation
        # projection); read by query_catalog_discovery_state.
        self._catalog_track_state = CatalogTrackStateRepository(self.database_path)
        self._feedback_history = FeedbackHistoryRepository(self.database_path)
        self._learning_application = LearningApplicationRepository(self.database_path)
        self._journal = AgentRequestJournalRepository(self.database_path)
        self._intents = PendingIntentRepository(self.database_path)
        self._write_execution = WriteExecutionRepository(self.database_path)
        self._recommendation_execution = RecommendationExecutionService(
            canonical=self._canonical,
            preference=self._preference,
            recommendation_history=self._recommendation_history,
            catalog_track_state=self._catalog_track_state,
            active_context=self._active_context,
            rating_policy=PRODUCTION_RATING_POLICY,
            magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
            familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
            empty_recommendation_error_type=EmptyRecommendationError,
            similarity_seed_unavailable_error_type=SimilaritySeedUnavailableError,
            preference_inputs_by_target=self._preference_inputs_by_target,
            item_evidence_entry=self._item_evidence_entry,
            playback_annotation=self._playback_annotation,
            item_playback_summary=self._item_playback_summary,
            run_id_factory=self._generate_recommendation_run_id,
        )
        self._handlers: dict[AgentToolName, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
            AgentToolName.GET_CANONICAL_ENTITY: self._execute_get_canonical_entity,
            AgentToolName.QUERY_TRACK_PREFERENCE: self._execute_query_track_preference,
            AgentToolName.LIST_RECOMMENDATION_RUNS: self._execute_list_recommendation_runs,
            AgentToolName.GET_RECOMMENDATION_RUN: self._execute_get_recommendation_run,
            AgentToolName.GENERATE_RECOMMENDATION: self._execute_generate_recommendation,
            AgentToolName.LIST_FEEDBACK_OBSERVATIONS: self._execute_list_feedback_observations,
            AgentToolName.GET_FEEDBACK_OBSERVATION: self._execute_get_feedback_observation,
            AgentToolName.RECORD_FEEDBACK: self._execute_record_feedback,
            AgentToolName.INTERPRET_FEEDBACK: self._execute_interpret_feedback,
            AgentToolName.LIST_LEARNING_APPLICATIONS: self._execute_list_learning_applications,
            AgentToolName.GET_LEARNING_APPLICATION: self._execute_get_learning_application,
            AgentToolName.APPLY_LEARNING: self._execute_apply_learning,
            AgentToolName.GET_AGENT_CAPABILITIES: self._execute_get_agent_capabilities,
            AgentToolName.EXECUTE_WRITE_INTENT: self._execute_execute_write_intent,
            AgentToolName.PLAY: self._execute_play,
            AgentToolName.PAUSE: self._execute_pause,
            AgentToolName.NEXT_TRACK: self._execute_next_track,
            AgentToolName.PREVIOUS_TRACK: self._execute_previous_track,
            AgentToolName.PLAY_TRACK: self._execute_play_track,
            AgentToolName.GET_NOW_PLAYING: self._execute_get_now_playing,
            AgentToolName.GET_ACTIVE_CONTEXT: self._execute_get_active_context,
            AgentToolName.GET_PLAYBACK_CONTEXT: self._execute_get_playback_context,
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION: self._execute_generate_inferred_recommendation,
            AgentToolName.PREVIEW_CATALOG_TRACK: self._execute_preview_catalog_track,
            AgentToolName.PREVIEW_BATCH: self._execute_preview_batch,
            AgentToolName.ADVANCE_PREVIEW: self._execute_advance_preview,
            AgentToolName.STOP_PREVIEW: self._execute_stop_preview,
            AgentToolName.ADD_CATALOG_TO_LIBRARY: self._execute_add_catalog_to_library,
            AgentToolName.OPEN_IN_APPLE_MUSIC: self._execute_open_in_apple_music,
            AgentToolName.DISCOVER_CATALOG_TRACKS: self._execute_discover_catalog_tracks,
            AgentToolName.SEARCH_LIBRARY_TRACKS: self._execute_search_library_tracks,
            AgentToolName.QUERY_CATALOG_DISCOVERY_STATE: self._execute_query_catalog_discovery_state,
        }

    def _generate_recommendation_run_id(self) -> str:
        """Resolve the module-level run-id generator at execution time.

        Keeping this indirection in the SharedAgentService preserves the existing
        patch/instrumentation seam while RecommendationExecutionService owns the
        generation implementation.
        """
        return generate_run_id()

    def close(self) -> None:
        for repository in (
            self._canonical,
            self._preference,
            self._recommendation_history,
            self._catalog_track_state,
            self._feedback_history,
            self._learning_application,
            self._journal,
            self._intents,
            self._write_execution,
        ):
            repository.close()

    def __enter__(self) -> SharedAgentService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def execute(
        self,
        request: AgentRequest,
        *,
        completed_at: str | None = None,
        fresh_canonical_ids: tuple[str, ...] | None = None,
        recommendation_scope_ids: tuple[str, ...] | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> AgentToolResult:
        """Execute one agent request through the full fail-closed flow.

        P15-S3-S3D: ``fresh_canonical_ids`` is the internal same-run Fresh
        provenance transport for the two generation handlers (see the dispatch
        below). It never touches the P09 envelope, the journal's request payload,
        or validation -- only that one handler fork consumes it.

        True Similarity V1 accepts a separate strict ``similarity_context``. Its
        canonical seed discriminator is copied only into the durable journal
        request so request-id replay cannot cross seed identities; the reserved
        field is never validated as public payload or passed to a handler.
        """
        if not isinstance(request, AgentRequest):
            raise SharedAgentServiceValidationError("request must be an AgentRequest")
        if similarity_context is not None and not isinstance(
            similarity_context, SimilarityExecutionContext
        ):
            raise SharedAgentServiceValidationError(
                "similarity_context must be a SimilarityExecutionContext or None"
            )
        completed_dt = _resolve_completed_at(completed_at)
        journal_request = request
        if similarity_context is not None:
            if _JOURNALED_EXECUTION_CONTEXT_KEY in request.payload:
                return self._refuse(
                    request,
                    AgentToolOutcome.INVALID_REQUEST,
                    "reserved_execution_context",
                    "public payload must not contain internal execution context",
                    completed_dt,
                )
            journal_payload = dict(request.payload)
            journal_payload[_JOURNALED_EXECUTION_CONTEXT_KEY] = {
                "similarity_seed_canonical_id": (
                    similarity_context.seed_canonical_id
                )
            }
            journal_request = AgentRequest(
                request_id=request.request_id,
                client=request.client,
                tool=request.tool,
                payload=journal_payload,
                issued_at=request.issued_at,
                contract_version=request.contract_version,
            )
        if request.contract_version != AGENT_CONTRACT_VERSION:
            return self._refuse(
                journal_request,
                AgentToolOutcome.INVALID_REQUEST,
                "unsupported_contract_version",
                (
                    f"agent contract version {request.contract_version} is not supported "
                    f"(current: {AGENT_CONTRACT_VERSION})"
                ),
                completed_dt,
            )
        existing = self._journal.get(request.request_id)
        if existing is not None:
            if encode_agent_payload(existing.request.payload) == encode_agent_payload(
                journal_request.payload
            ):
                stored = existing.result
                return AgentToolResult(
                    request_id=stored.request_id,
                    tool=stored.tool,
                    outcome=stored.outcome,
                    payload=stored.payload,
                    error_code=stored.error_code,
                    error_message=stored.error_message,
                    completed_at=stored.completed_at,
                    contract_version=stored.contract_version,
                    replayed=True,
                )
            return AgentToolResult(
                request_id=request.request_id,
                tool=request.tool,
                outcome=AgentToolOutcome.REPLAY_CONFLICT,
                payload=None,
                error_code="replay_conflict",
                error_message=(
                    f"request {request.request_id} was already executed with a different payload"
                ),
                completed_at=completed_dt,
            )
        spec = AGENT_TOOL_REGISTRY.lookup(request.tool)
        if spec is None:
            return self._refuse(
                journal_request,
                AgentToolOutcome.TOOL_NOT_SUPPORTED,
                "tool_not_supported",
                f"tool {request.tool!r} is not in the agent tool registry",
                completed_dt,
            )
        if (
            similarity_context is not None
            and spec.name is not AgentToolName.GENERATE_INFERRED_RECOMMENDATION
        ):
            return self._refuse(
                journal_request,
                AgentToolOutcome.INVALID_REQUEST,
                "invalid_similarity_execution_context",
                "similarity execution context is valid only for inferred generation",
                completed_dt,
            )
        try:
            spec.validate(request.payload)
        except ValueError as error:
            return self._refuse(
                journal_request,
                AgentToolOutcome.INVALID_REQUEST,
                getattr(error, "code", "invalid_request"),
                str(error),
                completed_dt,
            )
        live_write_operation = None
        if spec.name is AgentToolName.EXECUTE_WRITE_INTENT:
            intent = self._intents.get_intent(request.payload["intent_id"])
            if intent is None:
                return self._refuse(
                    journal_request,
                    AgentToolOutcome.EXECUTION_ERROR,
                    "write_intent_not_found",
                    f"write intent {request.payload['intent_id']} does not exist",
                    completed_dt,
                )
            live_write_operation = intent.operation
        elif spec.name is AgentToolName.ADD_CATALOG_TO_LIBRARY:
            live_write_operation = WriteOperation.ADD_LIBRARY_SONG
        denial = decide_agent_permission(
            spec.permission_class,
            self._clients.policy_for(request.client.client_id),
            live_write_operation=live_write_operation,
            capability_matrix=self._capability_matrix,
        )
        if denial is not None:
            return self._refuse(
                journal_request,
                denial.outcome,
                denial.code,
                denial.message,
                completed_dt,
            )
        try:
            handler = self._handlers[spec.name]
            if spec.name in _FRESH_AWARE_GENERATION_TOOL_NAMES:
                # P15-S3-S3D: the two recommendation-producing handlers accept the
                # INTERNAL same-run Fresh provenance (captured by the provider loop
                # from genuinely executed discoveries). It is request execution
                # context only: never part of the model payload or any tool
                # schema -- the model cannot write it. The similarity seed alone
                # is journaled under a reserved discriminator for durable replay;
                # Fresh and artist-scope context remain unjournaled. Absent
                # (direct/non-loop callers) == empty == no Fresh,
                # which fails closed to the pre-S3-S3D behavior.
                # P15 burn-down Issue 1: the same context seam carries the
                # AUTHORITATIVE produced_at -- the service's own execution instant
                # (completed_dt). The model has no tool-schema key and no payload
                # path for run time; deterministic tests inject completed_at
                # through the trusted execute kwarg.
                generation_kwargs: dict[str, Any] = {
                    "fresh_canonical_ids": tuple(fresh_canonical_ids or ()),
                    "recommendation_scope_ids": recommendation_scope_ids,
                    "produced_at": completed_dt,
                }
                if spec.name is AgentToolName.GENERATE_INFERRED_RECOMMENDATION:
                    generation_kwargs["similarity_context"] = similarity_context
                payload = handler(request.payload, **generation_kwargs)
            elif spec.name is AgentToolName.RECORD_FEEDBACK:
                # P16-S1: feedback observation time is service-authoritative --
                # the request's own execution instant. The model has no payload
                # key for observed_at (rejected at the validation boundary); the
                # trusted completed_at kwarg is the deterministic test/replay
                # seam, following the produced_at precedent.
                payload = handler(request.payload, observed_at=completed_dt)
            elif spec.name is AgentToolName.APPLY_LEARNING:
                # P16-S1: learning application time is service-authoritative
                # (same trusted seam; no model path exists for applied_at).
                payload = handler(request.payload, applied_at=completed_dt)
            else:
                payload = handler(request.payload)
            result = AgentToolResult(
                request_id=request.request_id,
                tool=request.tool,
                outcome=AgentToolOutcome.OK,
                payload=payload,
                error_code=None,
                error_message=None,
                completed_at=completed_dt,
            )
        except ValueError as error:
            result = AgentToolResult(
                request_id=request.request_id,
                tool=request.tool,
                outcome=AgentToolOutcome.EXECUTION_ERROR,
                payload=None,
                error_code=getattr(error, "code", "shared_agent_service_error"),
                error_message=str(error),
                completed_at=completed_dt,
            )
        self._journal.record(journal_request, result)
        return result

    def _refuse(
        self,
        request: AgentRequest,
        outcome: AgentToolOutcome,
        code: str,
        message: str,
        completed_at: datetime,
    ) -> AgentToolResult:
        result = AgentToolResult(
            request_id=request.request_id,
            tool=request.tool,
            outcome=outcome,
            payload=None,
            error_code=code,
            error_message=message,
            completed_at=completed_at,
        )
        self._journal.record(request, result)
        return result

    # --- tool handlers (read) -------------------------------------------------

    def _execute_get_canonical_entity(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        canonical_id = payload["canonical_id"]
        model = self._canonical.load_model()
        for key in ("tracks", "artists", "albums", "playlists"):
            for entity in model[key]:
                if entity.get("id") == canonical_id:
                    return {"entity_type": key[:-1], "entity": entity}
        raise CanonicalEntityNotFoundError(f"no canonical entity {canonical_id} in the store")

    def _execute_query_track_preference(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        source_system = payload.get("source_system") or DEFAULT_SOURCE_SYSTEM
        state = query_track_preference(
            self._preference,
            PreferenceTargetReference(PreferenceTargetKind.TRACK, payload["target_id"]),
            rating_policy=PRODUCTION_RATING_POLICY,
            magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
            familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
            source_system=source_system,
            confidence_policy=PRODUCTION_CONFIDENCE_DERIVATION_POLICY,
        )
        claim = state.confidence_claim
        if claim is None:
            confidence = None
        else:
            components = claim.components
            confidence = {
                "score": PRODUCTION_CONFIDENCE_AGGREGATION_POLICY.aggregate(claim),
                "quality": components.quality,
                "quantity": components.quantity,
                "freshness": components.freshness,
                "consistency": components.consistency,
                "contradiction": components.contradiction.value,
                "source_reliability": components.source_reliability,
                "inference_distance": components.inference_distance,
            }
        explanation = state.explanation
        return {
            "target_kind": state.target.kind.value,
            "target_id": state.target.target_id,
            "source_system": source_system,
            "preference_state": state.direct_preference.strength.state.value,
            "magnitude": state.direct_preference.strength.magnitude,
            "familiarity_level": state.familiarity.level.value,
            "familiarity_reason": state.familiarity.reason.value,
            "familiarity_magnitude": state.familiarity.magnitude,
            "confidence": confidence,
            "explanation": {
                "derivation": explanation.derivation.value,
                "signals": [
                    {
                        "signal": entry.contribution.signal.value,
                        "direction": entry.contribution.direction.value,
                        "reason": entry.contribution.reason.value,
                        "explicitness": entry.contribution.explicitness.value,
                    }
                    for entry in explanation.signals
                    if entry.contribution.direction is not SignalDirection.NO_CLAIM
                ],
                "conflicts": [
                    {
                        "kind": conflict.kind.value,
                        "first_direction": conflict.first_direction.value,
                        "second_direction": conflict.second_direction.value,
                    }
                    for conflict in explanation.conflicts
                ],
            },
        }

    def _execute_list_recommendation_runs(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        # Newest-first (list_runs contract); items carry hydrated display names so a
        # run can be resolved from natural-language references like "刚才推荐的 X".
        # Only the most recent ``limit`` runs are expanded (default 5) so repeated
        # history probes stay cheap in provider context; ``runs_total`` reports the
        # full count so the model knows more history exists without fetching it.
        limit = payload.get("limit", 5)
        results = self._recommendation_history.list_runs()
        track_by_id, artist_by_id = self._canonical_display_maps()
        return {
            "runs": [
                {
                    "run_id": result.run_id,
                    "produced_at": result.produced_at.isoformat(),
                    "contract_version": result.contract_version,
                    "item_count": len(result.items),
                    "items": [
                        self._history_item_entry(item, track_by_id, artist_by_id)
                        for item in result.items
                    ],
                }
                for result in results[:limit]
            ],
            "runs_total": len(results),
        }

    def _execute_get_recommendation_run(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = self._recommendation_history.get_result(payload["run_id"])
        if result is None:
            raise RecommendationRunNotFoundError(
                f"recommendation run {payload['run_id']} does not exist"
            )
        track_by_id, artist_by_id = self._canonical_display_maps()
        # P20-Fix03: the detail reader also delivers each item's DURABLE
        # evidence (mechanism / basis / provenance / weak-evidence note), so an
        # explanation turn ("为什么推荐这些？") can ground every claim in what
        # the run actually recorded instead of the model's own music knowledge.
        # Everything comes from the persisted result's frozen primitives --
        # source_path, basis_targets and the run's own preference_inputs --
        # resolved to display labels; nothing is inferred at read time. The
        # list reader keeps the identity-only entry (token discipline).
        inputs_by_target = SharedAgentService._preference_inputs_by_target(
            result.request.context.preference_inputs
        )
        return {
            "run_id": result.run_id,
            "produced_at": result.produced_at.isoformat(),
            "contract_version": result.contract_version,
            "item_count": len(result.items),
            "items": [
                self._run_detail_item_entry(
                    item, position, track_by_id, artist_by_id, inputs_by_target
                )
                for position, item in enumerate(result.items, start=1)
            ],
            "encoded_result": encode_recommendation_result(result),
        }

    # P20-Fix03: frozen display vocabulary over the candidate source_path
    # namespace (preference_driven / catalog_driven / fresh_driven). An unknown
    # path is passed through raw -- the projection must fail honest, never
    # guess a mechanism for a source it does not know.
    _RUN_READER_MECHANISMS: dict[str, str] = {
        "preference_driven": "直接偏好",
        "catalog_driven": "目录推断",
        "fresh_driven": "探索性新发现",
        SIMILARITY_SOURCE_PATH: "曲目元数据相似",
    }

    @staticmethod
    def _preference_inputs_by_target(
        preference_inputs,
    ) -> dict[tuple[str, str], set[str]]:
        """Index a run context's preference inputs by (kind, target id) -> provenances.

        Shared by the run detail reader and the generation result projections
        (P20-Fix09) so first presentation and follow-up explanation resolve
        provenance against the exact same facts. Only DIRECTIONAL inputs
        (POSITIVE / NEGATIVE) count as evidence: a context also records
        non-directional direct snapshots (strength UNKNOWN etc.) for every
        target the executor scanned, and those carry no preference claim --
        indexing them here would license 「直接」 provenance for targets whose
        only directional evidence is inferred (the Fix09 live-UAT inflation:
        a novel item's evidence claimed 直接偏好 against its own unknown
        direct state).
        """
        inputs_by_target: dict[tuple[str, str], set[str]] = {}
        for input_ in preference_inputs:
            # The frozen P06 directionality rule: non-directional states carry
            # no claim to act on, so they carry no evidence to project either.
            if input_.strength.state not in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                continue
            key = (input_.target.kind.value, input_.target.target_id)
            inputs_by_target.setdefault(key, set()).add(input_.provenance.value)
        return inputs_by_target

    @classmethod
    def _item_evidence_entry(
        cls,
        item,
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
        inputs_by_target: Mapping[tuple[str, str], set[str]],
    ) -> dict[str, Any]:
        """One item's durable evidence block (P20-Fix03 projection core).

        P20-Fix09: this builder is SHARED by the run detail reader and the two
        generation result projections, so the first presentation of a batch and
        a later "为什么推荐这些？" explanation read one fact source -- exactly
        what the persisted run recorded: ``mechanism`` (the candidate source
        translated through the frozen vocabulary), ``basis`` (the motivating
        preference targets resolved to display labels with their recorded
        provenance) and, for zero-legitimate-basis items, an honest ``note``.
        A basis target with no matching preference input in the run context, or
        whose canonical display entity no longer exists, is dropped fail-closed:
        the evidence shrinks rather than guessing.
        """
        if item.candidate.source.source_path == SIMILARITY_SOURCE_PATH:
            return cls._similarity_item_evidence(
                item, track_by_id, artist_by_id
            )

        from music_agent.preference_propagation import canonicalize_genre_key

        basis: list[dict[str, str]] = []
        track = track_by_id.get(item.candidate.target.target_id)
        track_genres = (
            {canonicalize_genre_key(genre): genre for genre in track["genres"]}
            if track is not None
            else {}
        )
        for target in item.candidate.basis_targets:
            provenances = inputs_by_target.get((target.kind.value, target.target_id))
            if not provenances:
                continue
            if target.kind.value == "genre":
                # The stored key is already canonicalized; the map carries the
                # track's ORIGINAL genre string for canonoical keys it owns.
                label = track_genres.get(target.target_id, target.target_id)
            elif target.kind.value == "artist":
                artist = artist_by_id.get(target.target_id)
                if artist is None:
                    continue
                label = artist["name"]
            else:
                ref = track_by_id.get(target.target_id)
                if ref is None:
                    continue
                label = ref["name"]
            basis.append(
                {
                    "kind": target.kind.value,
                    "label": label,
                    "provenance": "直接" if "direct" in provenances else "推断",
                }
            )
        # P20-Fix09: the mechanism must agree with the basis it presents.
        # ``preference_driven`` translates to 「直接偏好」, but the frozen P07
        # generator also admits a target whose only directional conclusion is
        # INFERRED (a novel track with an unknown direct state) -- evidence
        # claiming 直接 there is a fact upgrade the run never recorded. Such
        # an item is provenance-honest only as 「推断偏好」 (its basis rows
        # all read 推断); any direct-directional basis keeps 「直接偏好」.
        mechanism = cls._RUN_READER_MECHANISMS.get(
            item.candidate.source.source_path, item.candidate.source.source_path
        )
        if (
            item.candidate.source.source_path == "preference_driven"
            and basis
            and all(row["provenance"] == "推断" for row in basis)
        ):
            mechanism = "推断偏好"
        evidence: dict[str, Any] = {
            "mechanism": mechanism,
            "basis": basis,
        }
        if not basis:
            # Same honest wording the inferred generation boundary already uses
            # for zero-basis fresh-driven items -- an item without recorded
            # evidence gets exactly this, never an invented suitability story.
            evidence["note"] = "本次目录搜索的新发现（暂无偏好匹配证据）"
        return evidence

    @classmethod
    def _similarity_item_evidence(
        cls,
        item,
        track_by_id: Mapping[str, Mapping[str, Any]],
        artist_by_id: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Truthful V1 evidence reconstructed from the persisted seed role.

        ``similarity_driven`` candidates persist exactly one Track basis: the
        strict canonical seed.  The component values persist the scored axes;
        this projection reuses the same pure scorer over those stable canonical
        identities and never reinterprets preference as similarity.
        """

        candidate = track_by_id.get(item.candidate.target.target_id)
        seed_refs = [
            target
            for target in item.candidate.basis_targets
            if target.kind is PreferenceTargetKind.TRACK
            and target.target_id != item.candidate.target.target_id
        ]
        seed = track_by_id.get(seed_refs[0].target_id) if len(seed_refs) == 1 else None
        if candidate is None or seed is None:
            return {
                "mechanism": cls._RUN_READER_MECHANISMS[SIMILARITY_SOURCE_PATH],
                "basis": [],
                "note": "相似度 seed 的 canonical metadata 当前不可解析",
            }
        evidence = score_track_similarity(seed, candidate)
        basis: list[dict[str, str]] = []
        basis.extend(
            {"kind": "genre", "label": genre, "provenance": "相似"}
            for genre in evidence.shared_genres
        )
        basis.extend(
            {
                "kind": "artist",
                "label": (
                    artist_by_id.get(artist_id, {}).get("name") or artist_id
                ),
                "provenance": "相似",
            }
            for artist_id in evidence.shared_artist_ids
        )
        if evidence.shared_composer is not None:
            basis.append(
                {
                    "kind": "composer",
                    "label": evidence.shared_composer,
                    "provenance": "相似",
                }
            )
        basis.extend(
            {"kind": "tag", "label": tag, "provenance": "相似"}
            for tag in evidence.shared_tags
        )
        return {
            "mechanism": cls._RUN_READER_MECHANISMS[SIMILARITY_SOURCE_PATH],
            "seed": {"target_id": seed["id"], "name": seed["name"]},
            "basis": basis,
        }

    @classmethod
    def _run_detail_item_entry(
        cls,
        item,
        position: int,
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
        inputs_by_target: Mapping[tuple[str, str], set[str]],
    ) -> dict[str, Any]:
        """One detail-reader item: identity/hydration facts plus the durable evidence block.

        P20-Fix03: ``evidence`` exposes exactly what the persisted run recorded for
        this item -- ``mechanism`` (the candidate source translated through the frozen
        vocabulary), ``basis`` (the motivating preference targets resolved to display
        labels with their recorded provenance) and, for zero-legitimate-basis items, an
        honest ``note``. A basis target with no matching preference input in the run
        context, or whose canonical display entity no longer exists, is dropped
        fail-closed: the evidence shrinks rather than guessing. ``score_total`` stays a
        machine field (internal ranking, never a suitability reason).
        """
        entry = SharedAgentService._history_item_entry(
            item, track_by_id, artist_by_id
        )
        return {
            "position": position,
            **entry,
            "evidence": cls._item_evidence_entry(
                item, track_by_id, artist_by_id, inputs_by_target
            ),
        }

    def _canonical_display_maps(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Track and Artist lookup maps loaded from the canonical model for display hydration."""
        model = self._canonical.load_model()
        return (
            {track["id"]: track for track in model["tracks"]},
            {artist["id"]: artist for artist in model["artists"]},
        )

    @staticmethod
    def _history_item_entry(
        item,
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """One compact history item with display names hydrated from the canonical Track -> Artist relation."""
        target_id = item.candidate.target.target_id
        track = track_by_id.get(target_id)
        artist_names = (
            [artist_by_id[artist_id]["name"] for artist_id in track["artist_ids"] if artist_id in artist_by_id]
            if track is not None else []
        )
        return {
            "candidate_id": item.candidate.candidate_id,
            "target_kind": item.candidate.target.kind.value,
            "target_id": target_id,
            "name": track["name"] if track is not None else None,
            "artist_name": ", ".join(artist_names) if artist_names else None,
            "score_total": item.score.total,
            "playback": SharedAgentService._playback_annotation(track),
        }

    def _execute_list_feedback_observations(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        observations = self._feedback_history.list_observations()
        return {
            "count": len(observations),
            "observations": [
                encode_feedback_observation(observation) for observation in observations
            ],
        }

    def _execute_get_feedback_observation(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        observation = self._feedback_history.get_observation(payload["feedback_id"])
        if observation is None:
            raise FeedbackNotFoundError(
                f"feedback observation {payload['feedback_id']} does not exist"
            )
        return {"observation": encode_feedback_observation(observation)}

    def _execute_interpret_feedback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        observation = self._feedback_history.get_observation(payload["feedback_id"])
        if observation is None:
            raise FeedbackNotFoundError(
                f"feedback observation {payload['feedback_id']} does not exist"
            )
        interpretation = interpret_observation(observation, InterpretationPolicy(1))
        return {
            "feedback_id": interpretation.feedback_id,
            "direction": interpretation.direction.value,
            "reason": interpretation.reason.value,
            "explicitness": interpretation.explicitness.value,
            "policy_version": interpretation.policy_version,
            "contract_version": interpretation.contract_version,
        }

    def _execute_list_learning_applications(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        applications = self._learning_application.list_applications()
        return {"applications": [_application_projection(record) for record in applications]}

    def _execute_get_learning_application(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = self._learning_application.get_application(payload["feedback_id"])
        if record is None:
            raise LearningApplicationNotFoundError(
                f"learning application for {payload['feedback_id']} does not exist"
            )
        return {"application": _application_projection(record)}

    def _execute_get_agent_capabilities(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "agent_contract_version": AGENT_CONTRACT_VERSION,
            "schema_version": self._journal.schema_version,
            "tools": [
                {
                    "name": name,
                    "permission_class": AGENT_TOOL_REGISTRY.lookup(name).permission_class.value,
                }
                for name in AGENT_TOOL_REGISTRY.tool_names
            ],
            "writes": list(write_capability_summary()),
        }

    # --- tool handlers (mutate) ----------------------------------------------

    def direction_shift_inputs(
        self, source_system: str | None = None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Read-only enumeration of the user's real positive direction evidence (Fix05).

        Returns ``(positive genre keys, positive track ids)`` from ONE
        model-wide pass over durable state -- the direction and the scope of a
        direction shift at once:

        * every canonical track's directional direct state propagates through
          the frozen P10 genre split and reduces through
          ``build_genre_affinities``; net-positive genres come back in
          preference order (strongest affinity first, deterministic key-order
          tie-break);
        * the track scope is the set of tracks whose resolved direct state is
          POSITIVE -- the same real evidence a direction generation may build
          candidates from (the generate tools narrow it further by genre, so
          the scope carries the full pool the requested direction selects
          from, never just one batch).

        Nothing is written, cached or guessed, and no direction without
        net-positive evidence ever appears. This is a service-level read (never
        a registered agent tool -- the agent surface is untouched); the
        direction-shift coach consumes it.
        """
        from music_agent.catalog_recommendation import EvenSplitPolicy
        from music_agent.genre_affinity import (
            GenreAffinityPolicy,
            SourcedContribution,
            build_genre_affinities,
        )
        from music_agent.preference_propagation import (
            PropagationKind,
            propagate_track_preference,
        )

        source = source_system or DEFAULT_SOURCE_SYSTEM
        model = self._canonical.load_model()
        sourced: list[SourcedContribution] = []
        positive_track_ids: list[str] = []
        split_policy = EvenSplitPolicy()
        for track in model["tracks"]:
            state = query_track_preference(
                self._preference,
                PreferenceTargetReference(PreferenceTargetKind.TRACK, track["id"]),
                rating_policy=PRODUCTION_RATING_POLICY,
                magnitude_policy=PRODUCTION_MAGNITUDE_POLICY,
                familiarity_policy=PRODUCTION_FAMILIARITY_POLICY,
                source_system=source,
            )
            if state.direct_preference.strength.state not in (
                PreferenceState.POSITIVE,
                PreferenceState.NEGATIVE,
            ):
                continue
            if state.direct_preference.strength.state is PreferenceState.POSITIVE:
                positive_track_ids.append(track["id"])
            for contribution in propagate_track_preference(
                state.direct_preference,
                artist_ids=track.get("artist_ids", []),
                genres=track.get("genres", []),
                artist_split=split_policy,
                genre_split=split_policy,
            ):
                if contribution.kind is PropagationKind.GENRE:
                    sourced.append(SourcedContribution(source, contribution))
        affinities = build_genre_affinities(sourced, GenreAffinityPolicy())
        positives = sorted(
            (affinity for affinity in affinities if affinity.affinity > 0),
            key=lambda affinity: (-affinity.affinity, affinity.genre_key),
        )
        seen: set[str] = set()
        ordered: list[str] = []
        for affinity in positives:
            key = affinity.genre_key
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return (tuple(ordered), tuple(positive_track_ids))

    def positive_direction_affinities(
        self, source_system: str | None = None
    ) -> tuple[str, ...]:
        """The genre half of :meth:`direction_shift_inputs` (Fix05 coach input)."""
        return self.direction_shift_inputs(source_system)[0]

    def canonical_genre_keys(
        self, track_ids: Iterable[str]
    ) -> dict[str, tuple[str, ...]]:
        """Read-only canonical genre keys of the given canonical tracks (Fix05).

        A batch's direction derives from its items' DURABLE basis: genre-kind
        basis entries carry labels, but track-kind entries expose only display
        names on the tool surface -- the item's own canonical genres are the
        honest direction signal. The keys use the frozen canonicalizer
        (first-occurrence order, deduped, case preserved); unknown ids are
        omitted. Nothing is written; the direction-shift coach consumes this.
        """
        from music_agent.preference_propagation import canonicalize_genre_key

        wanted = {
            track_id
            for track_id in track_ids
            if isinstance(track_id, str) and track_id
        }
        if not wanted:
            return {}
        model = self._canonical.load_model()
        result: dict[str, tuple[str, ...]] = {}
        for track in model["tracks"]:
            track_id = track.get("id")
            if track_id not in wanted:
                continue
            keys: list[str] = []
            seen: set[str] = set()
            for genre in track.get("genres", ()):
                key = (
                    canonicalize_genre_key(genre)
                    if isinstance(genre, str)
                    else ""
                )
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
            result[track_id] = tuple(keys)
        return result

    @staticmethod
    def _playback_annotation(track: Mapping[str, Any] | None) -> dict[str, Any]:
        """Playback capability of one canonical Track, projected from its bindings.

        View-layer fact for the model (never a recommendation or scoring input):
        ``library`` -- apple_music_persistent_id bound, full playback via play_track;
        ``preview_only`` -- itunes_store binding only, 30-second preview route;
        ``unavailable`` -- no executable route. Playback tool logic is untouched.
        """
        if track is None:
            return {"route": "unavailable", "label": "不可用"}
        external_ids = track.get("external_ids", {})
        if external_ids.get("apple_music_persistent_id"):
            return {"route": "library", "label": "可正式播放"}
        if external_ids.get("itunes_store_id"):
            return {"route": "preview_only", "label": "只能试听"}
        return {"route": "unavailable", "label": "不可用"}

    @staticmethod
    def _item_playback_summary(
        item,
        track_by_id: dict[str, Mapping[str, Any]],
        artist_by_id: dict[str, Mapping[str, Any]],
        album_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compact generate-tool item: display names + playback capability (view-only).

        P18-S1.3: ``album`` is the version/source disambiguation display fact
        (single vs album release); additive, identity stays the target id.
        """
        target_id = item.candidate.target.target_id
        track = track_by_id.get(target_id)
        artist_names = (
            [
                artist_by_id[artist_id]["name"]
                for artist_id in track["artist_ids"]
                if artist_id in artist_by_id
            ]
            if track is not None
            else []
        )
        album_name: str | None = None
        if (
            track is not None
            and album_by_id is not None
            and track.get("album_id") is not None
        ):
            album = album_by_id.get(track["album_id"])
            if album is not None:
                album_name = album.get("name")
        return {
            "target_id": target_id,
            "name": track["name"] if track is not None else None,
            "artist_name": ", ".join(artist_names) if artist_names else None,
            "album": album_name,
            "playback": SharedAgentService._playback_annotation(track),
        }

    @staticmethod
    def _normalize_lookup_text(value: str) -> str:
        """Library-lookup normalization: NFKC + whitespace collapse + casefold.

        NFKC (rather than playback's NFC) so fullwidth user input also matches;
        on the canonical store's regular names it behaves identically to NFC.
        Semantics otherwise mirror ``playback_control._normalize_text``.
        """
        return unicodedata.normalize("NFKC", " ".join(value.split())).casefold()

    @staticmethod
    def _library_track_artist_names(
        track: Mapping[str, Any], artist_by_id: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        """Resolved artist display names of one canonical track (missing links skipped)."""
        return [
            artist_by_id[artist_id]["name"]
            for artist_id in track.get("artist_ids", ())
            if artist_id in artist_by_id and artist_by_id[artist_id].get("name")
        ]

    @staticmethod
    def _library_search_provenance(track: Mapping[str, Any]) -> dict[str, Any]:
        """Source identity projected for mixed canonical-store search results.

        ``search_library_tracks`` is a historical tool name: the underlying scan is
        over every known canonical Track, not only Music.app Library membership.
        Keep that distinction machine-visible so a catalog row can never be narrated
        as belonging to the user's Apple Music Library.
        """
        external_ids = track.get("external_ids", {})
        source_systems = [
            source_system
            for source_system, key in (
                ("apple_music", "apple_music_persistent_id"),
                ("apple_music_catalog", "apple_music_catalog_id"),
                ("itunes_store", "itunes_store_id"),
                ("isrc", "isrc"),
            )
            if external_ids.get(key)
        ]
        if "apple_music" in source_systems:
            kind = "apple_music_library"
            label = "Apple Music Library"
        elif "apple_music_catalog" in source_systems or "itunes_store" in source_systems:
            kind = "catalog"
            label = "Catalog（非 Apple Music Library）"
        else:
            kind = "music_agent_record"
            label = "Music Agent 已知记录"
        return {"kind": kind, "label": label, "source_systems": source_systems}

    @staticmethod
    def _library_search_bindings(track: Mapping[str, Any]) -> dict[str, Any]:
        """Stable identity evidence for one search result (read-only projection)."""
        external_ids = track.get("external_ids", {})
        return {
            "apple_music_persistent_id": external_ids.get("apple_music_persistent_id"),
            "apple_music_catalog_id": external_ids.get("apple_music_catalog_id"),
            "itunes_store_id": external_ids.get("itunes_store_id"),
        }

    @staticmethod
    def _library_lookup_matches(
        model: Mapping[str, Any], term: str
    ) -> list[Mapping[str, Any]]:
        """Name/artist match over the synced canonical model (P14-R3.1, R3.2 sort).

        Match level 0 is whole-term exact title equality. Level 1 is a full title
        token sequence contained in a longer query (for example title + artist).
        Level 2 is the existing name/artist token substring fallback.

        Ranking preserves title relevance first. Within levels 0/1, an explicit
        artist mismatch is demoted, then playback capability ranks library above
        preview_only/unavailable; missing Library metadata is neutral rather than a
        reason to truncate a formally playable exact-title result. Level 2 retains
        artist-hit priority before capability. Canonical ID is the final stable tie.
        """
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        normalized_term = SharedAgentService._normalize_lookup_text(term)
        if not normalized_term:
            return []
        tokens = normalized_term.split()
        ranked: list[tuple[int, int, int, int, int, Mapping[str, Any]]] = []
        for track in model["tracks"]:
            normalized_name = SharedAgentService._normalize_lookup_text(track.get("name") or "")
            if not normalized_name:
                continue
            name_tokens = normalized_name.split()
            artist_blob = " ".join(
                SharedAgentService._library_track_artist_names(track, artist_by_id)
            )
            normalized_artists = (
                SharedAgentService._normalize_lookup_text(artist_blob) if artist_blob else ""
            )
            artist_hit = bool(normalized_artists) and any(
                token in normalized_artists for token in tokens
            )
            if normalized_name == normalized_term:
                level = 0
            elif name_tokens and all(token in tokens for token in name_tokens):
                level = 1
            elif any(token in normalized_name for token in tokens) or artist_hit:
                level = 2
            else:
                continue
            route = SharedAgentService._playback_annotation(track)["route"]
            route_rank = _LIBRARY_ROUTE_RANK.get(route, len(_LIBRARY_ROUTE_RANK))
            if level < 2:
                extra_tokens = [token for token in tokens if token not in name_tokens]
                artist_mismatch = bool(
                    extra_tokens
                    and normalized_artists
                    and not any(token in normalized_artists for token in extra_tokens)
                )
                secondary = (
                    1 if artist_mismatch else 0,
                    route_rank,
                    0 if artist_hit else 1,
                )
            else:
                secondary = (0 if artist_hit else 1, route_rank, 0)
            ranked.append(
                (
                    level,
                    -len(normalized_name) if level == 1 else 0,
                    *secondary,
                    track,
                )
            )
        ranked.sort(
            key=lambda entry: (
                entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]["id"]
            )
        )
        return [entry[5] for entry in ranked]

    def _execute_generate_recommendation(
        self,
        payload: Mapping[str, Any],
        fresh_canonical_ids: tuple[str, ...] = (),
        recommendation_scope_ids: tuple[str, ...] | None = None,
        produced_at: datetime | None = None,
    ) -> dict[str, Any]:
        return self._recommendation_execution._execute_generate_recommendation(
            payload,
            fresh_canonical_ids=fresh_canonical_ids,
            recommendation_scope_ids=recommendation_scope_ids,
            produced_at=produced_at,
        )

    def _execute_record_feedback(
        self, payload: Mapping[str, Any], observed_at: datetime | None = None
    ) -> dict[str, Any]:
        # P16-S1: the durable observation instant is service-authoritative. The
        # execute path stamps this request's execution instant (``completed_dt``)
        # through the trusted context seam; the model payload has no key for it.
        # The kwarg is internal-only (direct Python callers / deterministic
        # tests/replay); the provider-facing path can never reach it.
        observed_dt = observed_at or datetime.now(timezone.utc)
        target = None
        if payload.get("target_id") is not None:
            target = PreferenceTargetReference(
                PreferenceTargetKind.TRACK, payload["target_id"]
            )
        recommendation = None
        if payload.get("run_id") is not None:
            recommendation = FeedbackRecommendationReference(
                payload["run_id"], payload["candidate_id"]
            )
        attribution = None
        if payload.get("attribution") is not None:
            attribution_data = payload["attribution"]
            attribution = FeedbackAttribution(
                PreferenceTargetReference(
                    PreferenceTargetKind(attribution_data["aspect_kind"]),
                    attribution_data["aspect_id"],
                ),
                AttributionRelation(attribution_data["relation"]),
            )
        observation = assemble_feedback_observation(
            feedback_id=payload.get("feedback_id") or generate_feedback_id(),
            kind=FeedbackKind(payload["kind"]),
            source=FeedbackSourceReference(
                payload["source_system"], payload["source_path"]
            ),
            observed_at=observed_dt,
            target=target,
            recommendation=recommendation,
            attribution=attribution,
            event_at=(
                datetime.fromisoformat(payload["event_at"])
                if payload.get("event_at")
                else None
            ),
            source_event_id=payload.get("source_event_id"),
        )
        self._feedback_history.save_observation(observation)
        return {"feedback_id": observation.feedback_id, "kind": observation.kind.value}

    def _execute_apply_learning(
        self, payload: Mapping[str, Any], applied_at: datetime | None = None
    ) -> dict[str, Any]:
        # P16-S1: application time is service-authoritative (the execute-path
        # execution instant); the payload has no applied_at key. The internal
        # kwarg is the trusted deterministic seam only.
        applied_iso = (applied_at or datetime.now(timezone.utc)).isoformat()
        feedback_id = payload["feedback_id"]
        observation = self._feedback_history.get_observation(feedback_id)
        if observation is None:
            raise FeedbackNotFoundError(f"feedback observation {feedback_id} does not exist")
        resolved_target = None
        if observation.target is None and observation.recommendation is not None:
            # P10 cross-phase amendment: recommendation-feedback learning bridge.
            # Strict persisted-provenance resolution; the observation stays untouched.
            from music_agent.recommendation_feedback_bridge import (
                resolve_recommendation_feedback_target,
            )

            resolved_target = resolve_recommendation_feedback_target(
                self._recommendation_history, observation.recommendation
            )
        interpretation = interpret_observation(observation, InterpretationPolicy(1))
        effect = derive_learning_effect(
            interpretation, LearningEffectPolicy(1), resolved_target=resolved_target
        )
        proposal = propose_preference_update(effect, LearningPolicy(2))
        if proposal is None:
            return {
                "applied": False,
                "feedback_id": feedback_id,
                "reason": "no_proposal",
            }
        record = self._learning_application.apply(
            proposal, applied_at=applied_iso
        )
        return {
            "applied": True,
            "feedback_id": record.feedback_id,
            "proposal_kind": record.proposal_kind.value,
            "provenance": record.provenance,
        }

    def _execute_true_similarity_recommendation(
        self,
        payload: Mapping[str, Any],
        *,
        similarity_context: SimilarityExecutionContext,
        produced_at: datetime,
        fresh_canonical_ids: tuple[str, ...],
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
        album_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return self._recommendation_execution._execute_true_similarity_recommendation(
            payload,
            similarity_context=similarity_context,
            produced_at=produced_at,
            fresh_canonical_ids=fresh_canonical_ids,
            track_by_id=track_by_id,
            artist_by_id=artist_by_id,
            album_by_id=album_by_id,
        )

    def _execute_generate_inferred_recommendation(
        self,
        payload: Mapping[str, Any],
        fresh_canonical_ids: tuple[str, ...] = (),
        recommendation_scope_ids: tuple[str, ...] | None = None,
        produced_at: datetime | None = None,
        similarity_context: SimilarityExecutionContext | None = None,
    ) -> dict[str, Any]:
        return self._recommendation_execution._execute_generate_inferred_recommendation(
            payload,
            fresh_canonical_ids=fresh_canonical_ids,
            recommendation_scope_ids=recommendation_scope_ids,
            produced_at=produced_at,
            similarity_context=similarity_context,
        )

    # --- tool handlers (P11.3 catalog usage path) ------------------------------

    def _require_catalog_binding(self, canonical_id: str) -> str:
        """Resolve one canonical Track to its durable Catalog Song ID, failing closed."""
        catalog_id = self._canonical.get_bound_external_id(
            "apple_music_catalog", EntityType.TRACK, canonical_id
        )
        if catalog_id is None:
            raise CanonicalEntityNotFoundError(
                f"no canonical entity {canonical_id} in the store"
                if self._canonical.get_entity_type(canonical_id) is None
                else f"track {canonical_id} has no Apple Music Catalog binding"
            )
        return catalog_id

    def _suspend_music_for_preview(self) -> SuspendedPlaybackEntry | None:
        """P15-PC P2: pause formal playback before preview audio starts, fail closed.

        Reads Music.app once (read-only snapshot): only the ``playing`` state is
        interrupted -- paused/stopped/unknown/unreadable states cause no pause and no
        record (never guessed). A failing pause command never blocks the preview and
        records the interruption with ``pause_ok=False``. Nothing here ever resumes.
        """
        adapter = self._playback_adapter
        if adapter is None:
            device_safety_trace(
                "suspend: guard hit -- no playback adapter wired -> no pause, no record"
            )
            return None
        try:
            now_playing = adapter.read_now_playing()
        except Exception as error:
            device_safety_trace(
                f"suspend: guard hit -- read_now_playing raised {error!r} "
                "-> no pause, no record"
            )
            return None
        state = getattr(now_playing, "state", None)
        if getattr(state, "value", state) != "playing":
            device_safety_trace(
                f"suspend: guard hit -- Music.app state={getattr(state, 'value', state)!r} "
                "(not playing) -> no pause, no record"
            )
            return None
        memo = {
            "player_state": "playing",
            "persistent_id": getattr(now_playing, "persistent_id", None),
            "name": getattr(now_playing, "name", None),
        }
        pause_ok = False
        device_safety_trace(
            f"suspend: Music.app playing; attempting pause via {type(adapter).__name__}"
        )
        try:
            adapter.pause()
            pause_ok = True
            device_safety_trace("suspend: pause() returned OK")
        except Exception as error:
            device_safety_trace(
                f"suspend: pause() raised {error!r} -> recording pause_ok=False"
            )
        if device_safety_trace_enabled():
            try:
                readback = adapter.read_now_playing()
                rb_state = getattr(readback, "state", None)
                rb_value = getattr(rb_state, "value", rb_state)
                device_safety_trace(f"suspend: readback after pause -> state={rb_value!r}")
            except Exception as error:
                device_safety_trace(f"suspend: readback after pause raised {error!r}")
        entry = SuspendedPlaybackEntry(pause_ok=pause_ok, **memo)
        self._playback_context.suspension.record(entry)
        device_safety_trace(f"suspend: SuspendedPlaybackEntry recorded (pause_ok={pause_ok})")
        return entry

    def _wire_preview_finish_hook(self) -> None:
        """P15-S1: hand the runner the natural-finish hook (duck-typed boundary).

        The production AfplayPreviewRunner validates the callback through a property
        setter; test fakes accept a plain attribute. A runner that exposes neither
        simply never auto-advances -- the session stays observable, nothing breaks.
        """
        try:
            self._preview_runner.on_natural_finish = (  # type: ignore[attr-defined]
                self._on_preview_clip_finished
            )
        except Exception:
            logger.exception("preview runner rejected the natural-finish hook")

    def _execute_preview_catalog_track(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P11-T4: start one canonical Track's 30-second audio preview, fail closed.

        P15-S1 pre-emption: a single preview command ends any running continuous
        session first (CANCELLED + notice), then starts exactly as before -- the
        single-audio-channel rule (a new preview replaces the sounding one) is the
        session-level rule too.

        P15-S1 C03: the result carries the same ``suspended`` contract as
        ``preview_batch`` -- every preview entry point reports the formal-playback
        interruption it recorded (None when nothing was playing), so the provider
        sees one uniform suspension shape across all 试听 requests.
        """
        canonical_id = payload["canonical_id"]
        cancelled = self._playback_context.sessions.cancel()
        if cancelled is not None:
            self._notify_preview_event("cancelled", cancelled)
        try:
            preview_url, suspended_entry = self._start_preview_for(canonical_id)
        except BaseException:
            # P17 acceptance: a preview that never started must not inherit any
            # earlier preview's restore arm (fail closed -- no auto-restore).
            self._playback_context.suspension.disarm_restore()
            raise
        # P17 acceptance: this preview's own interruption (None when nothing was
        # playing) is the only entry eligible for a natural-end auto-restore.
        self._playback_context.suspension.arm_restore(suspended_entry)
        return {
            "canonical_id": canonical_id,
            "started": True,
            "preview_url": preview_url,
            "suspended": self._suspended_entry_dict(suspended_entry),
        }

    def _start_preview_for(
        self, canonical_id: str
    ) -> tuple[str, SuspendedPlaybackEntry | None]:
        """P11-T4: the resolve→suspend→start→register path for ONE preview clip.

        Returns ``(preview_url, suspended_entry)``: the URL that started and the
        suspension the shared helper recorded (None when nothing was playing or
        the player could not be read -- never invented). The executor surfaces
        the entry in its tool contract; the batch path records the same shape.

        The single-tool handler (``preview_catalog_track``) uses this path on the
        agent-loop thread; continuous-session clips use ``_start_session_clip``
        instead (assembly-time facts, zero repository reads -- the reaper thread
        may call it). Because ``_suspend_music_for_preview`` only acts while
        Music.app reads as *playing*, a preview that starts while nothing plays
        records no suspension -- nothing is ever invented.

        The preview URL is re-resolved at preview time from the track's durable
        ``itunes_store`` binding (the signed URL rotates and is deliberately not
        persisted). No preview URL, no lookup route, or a missing binding all fail
        closed; nothing here mutates the library.
        """
        from music_agent.apple_music_catalog import CatalogMappingError, CatalogTransportError
        from music_agent.catalog_preview import CatalogPreviewError, CatalogPreviewUnavailableError

        model = self._canonical.load_model()
        track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
        if track is None:
            raise CanonicalEntityNotFoundError(f"canonical Track {canonical_id} does not exist")
        itunes_id = track["external_ids"].get("itunes_store_id")
        if not itunes_id:
            raise CatalogPreviewUnavailableError(
                f"track {canonical_id} has no itunes_store binding; "
                "no credential-free preview URL is resolvable"
            )
        lookup = getattr(self._catalog_search_source, "lookup_preview_url", None)
        if not callable(lookup):
            raise CatalogPreviewUnavailableError(
                "the wired catalog source provides no credential-free preview lookup"
            )
        try:
            preview_url = lookup(itunes_id)
        except (CatalogMappingError, CatalogTransportError) as error:
            raise CatalogPreviewError(f"iTunes preview lookup failed: {error}") from error
        if not preview_url:
            raise CatalogPreviewUnavailableError(
                f"iTunes lookup resolved no preview URL for {itunes_id}"
            )
        # P15-PC (suspend-don't-overlap): when Music.app is playing, pause it and
        # record an honest suspension before preview audio starts (never overlaps).
        suspended_entry = self._suspend_music_for_preview()
        self._preview_runner.start_audio(preview_url)
        self._active_context.note_preview(canonical_id)
        self._update_batch_item_position(canonical_id)
        return preview_url, suspended_entry

    def resolve_apple_music_target(self, canonical_id: str) -> dict[str, Any]:
        """Resolve one trusted song-level Apple Music target without opening anything.

        This is an application-internal read boundary for the native WebShell. It is not
        registered as an Agent tool and creates no OS side effect: canonical identity and
        the official Apple ``trackViewUrl`` stay backend-owned, while the foreground native
        shell remains the single owner of the external-app handoff.
        """
        target, _itunes_id = self._resolve_apple_music_target(canonical_id)
        return target

    def _resolve_apple_music_target(
        self, canonical_id: str
    ) -> tuple[dict[str, Any], str]:
        from music_agent import apple_music_open
        from music_agent.apple_music_catalog import CatalogMappingError, CatalogTransportError

        if not isinstance(canonical_id, str) or not canonical_id.strip():
            raise SharedAgentServiceValidationError(
                "canonical_id must be a non-empty string"
            )
        model = self._canonical.load_model()
        track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
        if track is None:
            raise CanonicalEntityNotFoundError(f"canonical Track {canonical_id} does not exist")
        itunes_id = track["external_ids"].get("itunes_store_id")
        if not itunes_id:
            raise apple_music_open.AppleMusicOpenUnavailableError(
                f"track {canonical_id} has no itunes_store binding; "
                "a library-only track has no Apple Music catalog link"
            )
        lookup = getattr(self._catalog_search_source, "lookup_track_view_url", None)
        if not callable(lookup):
            raise apple_music_open.AppleMusicOpenUnavailableError(
                "the wired catalog source provides no track-view lookup"
            )
        try:
            url = lookup(itunes_id)
        except (CatalogMappingError, CatalogTransportError) as error:
            raise apple_music_open.AppleMusicOpenError(
                f"iTunes track-view lookup failed: {error}"
            ) from error
        if not url:
            raise apple_music_open.AppleMusicOpenUnavailableError(
                f"iTunes lookup resolved no trackViewUrl for {itunes_id}"
            )
        client_url = apple_music_open.client_song_url(url, itunes_id)
        return (
            {
                "canonical_id": canonical_id,
                "url": url,
                "client_url": client_url,
                "source": "itunes_store_lookup",
            },
            itunes_id,
        )

    def _execute_open_in_apple_music(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Open one canonical Track through the non-native cross-surface handoff.

        The native WebShell does not call this execution path: it uses
        :meth:`resolve_apple_music_target` and lets its foreground AppKit bridge own the
        only OS handoff. CLI/MCP/browser-compatible callers keep the existing tool
        semantics here.
        """
        from music_agent import apple_music_open

        target, itunes_id = self._resolve_apple_music_target(payload["canonical_id"])
        apple_music_open.open_music_app(target["url"], itunes_id)
        return {
            "command": "open_in_apple_music",
            "ok": True,
            "handoff_requested": True,
            **target,
        }

    def _stop_sounding_preview(self) -> tuple[bool, bool]:
        """P3B/P15-S1: stop the currently sounding preview -- idempotent, never
        touches Music.app. Returns ``(runner_stopped, live_session_cancelled)``.

        The single visible stop semantics, shared by the user stop command and
        the P15-S2 safety pause so the two paths cannot drift apart: the
        transient preview registration is cleared on the way out, while the
        persistent_id anchor (last library playback) is deliberately untouched.
        """
        stopped = self._preview_runner.stop_preview()
        cancelled = self._playback_context.sessions.cancel()
        if cancelled is not None:
            self._notify_preview_event("cancelled", cancelled)
        self._active_context.clear_channel()
        # P17 acceptance: an explicit stop (or the P15-S2 safety pause that shares
        # this path) means the preview did not end naturally -- never auto-restore.
        self._playback_context.suspension.disarm_restore()
        return stopped, cancelled is not None

    def _execute_stop_preview(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P3B batch 2: stop the currently sounding preview through the shared
        stop semantics (stop ≙ end-of-连播), reported through the appended
        ``preview_session_cancelled`` key while every pre-existing key keeps
        its exact meaning.
        """
        stopped, had_session = self._stop_sounding_preview()
        return {
            "command": "stop_preview",
            "ok": True,
            "stopped": stopped,
            "preview_session_cancelled": had_session,
        }

    def handle_default_output_transition(
        self,
        before: AudioOutputSnapshot,
        after: AudioOutputSnapshot,
    ) -> SafetyPauseAction | None:
        """P15-S2 r2: the r3 event pump's entry point for one authoritative
        default-output change (baseline / default_device_changed facts only).
        Not wired to any event source yet; production behavior is unchanged
        until r3.

        Deterministic end to end: classify (uid facts, fail closed) -> decide
        from the sounding controlled paths -> execute exclusively through the
        unified P15-S1 control semantics (``_suspend_music_for_preview`` /
        ``_stop_sounding_preview``). No LLM participates, nothing resumes, no
        second playback state is created, and the P10.5 polling monitor is
        intentionally untouched. Returns the executed action for observability;
        None when nothing was classified, needed, or acted on.
        """
        if not isinstance(before, AudioOutputSnapshot) or not isinstance(
            after, AudioOutputSnapshot
        ):
            device_safety_trace(
                "service: type-guard rejected the forwarded pair -> returning None"
            )
            return None
        device_safety_trace(
            f"service: handle_default_output_transition entered: "
            f"{before.device_uid!r} -> {after.device_uid!r}"
        )
        classification = self._safety_pause_policy.on_default_output_change(
            before, after
        )
        device_safety_trace(
            f"service: classification={classification!r} "
            f"policy_dedup_key={self._safety_pause_policy._last_acted}"
        )
        if classification is None:
            device_safety_trace("service: classification/duplicate -> returning None")
            return None
        formal_sounding = self._formal_playback_sounding()
        preview_active = self._live_preview_session_active()
        device_safety_trace(
            f"service: _formal_playback_sounding={formal_sounding} "
            f"_live_preview_session_active={preview_active}"
        )
        action = self._safety_pause_policy.decide(
            classification,
            formal_sounding=formal_sounding,
            preview_active=preview_active,
        )
        if action is None:
            device_safety_trace(
                "service: decide returned None (no controlled audio sounding) -> "
                "no pause executed"
            )
            return None
        device_safety_trace(
            f"service: decide -> SafetyPauseAction(pause_formal_playback="
            f"{action.pause_formal_playback}, pause_preview={action.pause_preview})"
        )
        if action.pause_formal_playback:
            device_safety_trace(
                "service: executing _suspend_music_for_preview (formal pause path)"
            )
            # The unified suspend semantics: pause + an honest
            # SuspendedPlaybackEntry. 继续播放 keeps working through the
            # existing restore-by-intent path; nothing here ever auto-restores.
            self._suspend_music_for_preview()
        if action.pause_preview:
            device_safety_trace(
                "service: executing _stop_sounding_preview (preview stop path)"
            )
            self._stop_sounding_preview()
        return action

    def _formal_playback_sounding(self) -> bool:
        """Fail closed: an unreadable player state is unknown, and unknown is
        never treated as sounding (the same reading idiom the preview
        suspension uses; guessed state never pauses anything)."""
        adapter = self._playback_adapter
        if adapter is None:
            return False
        try:
            now_playing = adapter.read_now_playing()
        except Exception:
            return False
        state = getattr(now_playing, "state", None)
        return getattr(state, "value", state) == "playing"

    def _live_preview_session_active(self) -> bool:
        """A live continuous session exists (the registry only ever holds the
        RUNNING one; terminal sessions are cleared)."""
        return self._playback_context.sessions.current() is not None

    def _execute_preview_batch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P15-S1: preview the whole active recommendation batch -- one 30-second clip
        after another, switching automatically, with no further agent turns.

        Queue assembly is service-side (the model never lists canonical ids): the
        active batch projects from the register (generate boundary) or derived
        history, items stay in persisted rank order, and every item without a
        playable route is skipped at assembly -- reported, never failing the run.
        No batch, or a batch whose items all lack a route, fails closed.

        P17-A2: one authoritative state is projected per batch. The start may
        consume the whole queue (every clip skipped -> COMPLETED) or hit a
        systemic failure (FAILED) before this handler returns; ``started`` is
        therefore derived from the post-start snapshot (``state == running``)
        and never asserted unconditionally, so a terminal session can never be
        presented as "started". The terminal snapshot survives on the session
        object even though the registry is cleared by the terminal transition.
        """
        items, skipped = self._assemble_preview_queue()
        if not items:
            raise PreviewSessionUnavailableError("当前推荐批次的曲目全部不可试听")
        # One suspension per session: pause formal playback before the first clip
        # (fail-closed; later clips no-op because Music.app no longer reads playing).
        suspended_entry = self._suspend_music_for_preview()
        # P17 acceptance: the session's own interruption (None when nothing was
        # playing) is the only entry eligible for a natural-end auto-restore.
        self._playback_context.suspension.arm_restore(suspended_entry)
        # P15-S1 单会话: a new batch replaces any live session, which closes out as
        # CANCELLED (same shape as the single-preview pre-emption path).
        cancelled = self._playback_context.sessions.cancel()
        if cancelled is not None:
            self._notify_preview_event("cancelled", cancelled)
        session, _ = self._playback_context.sessions.start(items, skipped)
        first = session.current_item()
        if first is not None:
            self._start_session_item(session, first)
        snapshot = session.snapshot()
        return {
            "started": snapshot.state == PreviewSessionState.RUNNING.value,
            "suspended": self._suspended_entry_dict(suspended_entry),
            "session": self._session_snapshot_dict(snapshot),
        }

    def _execute_advance_preview(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P15-S1 C02: skip to the next clip of the RUNNING continuous session.

        The user's 下一首/下一首试听 during a 连播: the sounding clip is stopped
        and the queue advances through the same start path the reaper uses,
        so the P15 真机修复 classifications hold verbatim -- an unavailable next
        clip is still an item-level skip, a systemic start failure is still a
        session-level FAILED. The advance never creates or replaces a session,
        never cancels one, and never touches Music.app or the suspension
        register: the formal-playback restore target survives the skip exactly
        as it was recorded.

        On the LAST clip the advance exhausts the queue and completes the
        session honestly (the user skipped ahead; nothing further to sound).
        Without a live session this fails closed -- the router only routes the
        tool off a literal "running" read, so a hit here is a race already
        settled against us.
        """
        session = self._playback_context.sessions.current()
        if session is None:
            raise AdvancePreviewUnavailableError("当前没有正在进行的试听连播")
        # Stop the sounding clip first: the runner's stop flag suppresses the
        # natural-finish hook, so nothing double-advances behind this skip.
        self._preview_runner.stop_preview()
        next_item = session.advance()
        if next_item is None:
            self._finish_session_if_completed()
            # P17 acceptance: the user skipped to the end -- the last clip was
            # stopped, not naturally finished, so no auto-restore.
            self._playback_context.suspension.disarm_restore()
            return {"advanced": False, "completed": True}
        self._start_session_item(session, next_item)
        return {
            "advanced": True,
            "completed": False,
            "session": self._session_snapshot_dict(
                self._playback_context.sessions.snapshot()
            ),
        }

    def _assemble_preview_queue(
        self,
    ) -> tuple[list[PreviewSessionItem], list[dict[str, Any]]]:
        """P15-S1: derive the continuous-preview queue from the active batch.

        Rank order stays the persisted order of the run's items (the order the model
        saw); items without a playable route are skipped here -- recorded, never
        failed. The recommendation model itself is never consulted for the queue.

        P15 真机修复: the durable per-item facts the reaper-thread auto-advance
        needs (owning run, ``itunes_store`` binding, batch-item index) are resolved
        HERE on the agent-loop thread, so the advance path never touches a
        repository -- SQLite access stays single-threaded by construction.
        """
        projection = self._active_batch_projection()
        if projection is None:
            raise PreviewSessionUnavailableError(
                "当前没有可连播的推荐批次：请先让 Agent 生成一批推荐"
            )
        result = self._recommendation_history.get_result(projection["run_id"])
        if result is None:
            raise PreviewSessionUnavailableError(
                "active_batch 指向的推荐批次已不在推荐历史中"
            )
        track_by_id, artist_by_id = self._canonical_display_maps()
        items: list[PreviewSessionItem] = []
        skipped: list[dict[str, Any]] = []
        for item_index, recommendation_item in enumerate(result.items):
            summary = self._item_playback_summary(
                recommendation_item, track_by_id, artist_by_id
            )
            if summary["playback"]["route"] == "unavailable":
                skipped.append(
                    {
                        "canonical_id": summary["target_id"],
                        "name": summary["name"],
                        "reason": "unavailable: no playable route",
                    }
                )
                continue
            track = track_by_id.get(summary["target_id"])
            items.append(
                PreviewSessionItem(
                    canonical_id=summary["target_id"],
                    name=summary["name"],
                    artist_name=summary["artist_name"],
                    route=summary["playback"]["route"],
                    run_id=projection["run_id"],
                    itunes_id=(
                        track["external_ids"].get("itunes_store_id")
                        if track is not None
                        else None
                    ),
                    batch_item_index=item_index,
                )
            )
        return items, skipped

    def _start_session_item(self, session, item) -> None:
        """P15-S1: start one session clip and route failures by class.

        Item-level unavailability (no ``itunes_store`` binding, no resolvable
        preview URL, unknown track) skips that clip and continues the run; any
        other failure -- transport, afplay boundary, the cross-thread
        ProgrammingError of the live bug -- is systemic: the whole session fails
        with the reason exposed, and the remaining clips are NOT cascaded into
        per-item skips (a masquerade the 真机 run exposed). This may be called
        from the reaper thread (session advance) or the agent-loop thread (batch
        start).
        """
        from music_agent.catalog_preview import CatalogPreviewUnavailableError

        pending = item
        while pending is not None:
            try:
                self._start_session_clip(pending)
                self._notify_preview_event("progress", session.snapshot())
                return
            except (CatalogPreviewUnavailableError, CanonicalEntityNotFoundError) as error:
                reason = getattr(error, "code", None) or type(error).__name__
                pending = session.advance(
                    skipped={
                        "canonical_id": pending.canonical_id,
                        "name": pending.name,
                        "reason": reason,
                    }
                )
            except Exception as error:  # systemic -- fail the run, never cascade
                reason = getattr(error, "code", None) or type(error).__name__
                # Only fail the session this loop was started for; a concurrent
                # replacement/cancel already closed it out with its own terminal.
                if self._playback_context.sessions.current() is session:
                    failed = self._playback_context.sessions.fail(reason)
                    if failed is not None:
                        # P17 acceptance: a failed preview never auto-restores.
                        self._playback_context.suspension.disarm_restore()
                        self._notify_preview_event("failed", failed)
                return
        self._finish_session_if_completed()

    def _start_session_clip(self, item: PreviewSessionItem) -> str:
        """P15 真机修复: start one queued clip with ZERO repository reads.

        The reaper thread calls this for auto-advance, so the path must not touch
        SQLite (production connections are created on and bound to the agent-loop
        thread -- the live bug). Everything durable was resolved at assembly time
        (``run_id``/``itunes_id``/``batch_item_index`` on the item); only the
        rotating preview URL is re-resolved here, over HTTP (thread-safe), and
        only in-memory registers are mutated.
        """
        from music_agent.apple_music_catalog import CatalogMappingError, CatalogTransportError
        from music_agent.catalog_preview import CatalogPreviewError, CatalogPreviewUnavailableError

        if not item.itunes_id:
            raise CatalogPreviewUnavailableError(
                f"track {item.canonical_id} has no itunes_store binding; "
                "no credential-free preview URL is resolvable"
            )
        lookup = getattr(self._catalog_search_source, "lookup_preview_url", None)
        if not callable(lookup):
            raise CatalogPreviewUnavailableError(
                "the wired catalog source provides no credential-free preview lookup"
            )
        try:
            preview_url = lookup(item.itunes_id)
        except (CatalogMappingError, CatalogTransportError) as error:
            raise CatalogPreviewError(f"iTunes preview lookup failed: {error}") from error
        if not preview_url:
            raise CatalogPreviewUnavailableError(
                f"iTunes lookup resolved no preview URL for {item.itunes_id}"
            )
        self._preview_runner.start_audio(preview_url)
        self._active_context.note_preview(item.canonical_id)
        # Member-gated cursor move (the same contract as _update_batch_item_position,
        # without its repository read): only while this session's own run is still
        # the active one.
        if (
            item.batch_item_index is not None
            and item.run_id is not None
            and self._active_context.active_run_id == item.run_id
        ):
            self._active_context.note_batch_item(item.batch_item_index)
        return preview_url

    def _on_preview_clip_finished(self) -> None:
        """P15-S1 runner hook (reaper thread): one clip's natural end advances the
        continuous session to the next clip; the last one completes it. A session
        cancelled by a stop/pre-empt raced here finds a terminal (already cleared)
        register and no-ops. Nothing here runs on the agent loop and nothing here
        must raise -- reaper cleanup outranks any session surprise.

        P17 acceptance: the natural end of the preview as a whole -- the session's
        last clip, or a session-less single preview -- may auto-restore the formal
        playback the preview had interrupted (see ``_auto_restore_formal_playback``).
        Mid-session advances never restore: only the final natural end does.
        """
        try:
            session = self._playback_context.sessions.current()
            if session is None:
                # Session-less single-track preview's natural end (P15-S1 shape):
                # nothing to advance; restore eligibility is consumed here.
                self._auto_restore_formal_playback()
                return
            next_item = session.advance()
            if next_item is None:
                self._finish_session_if_completed()
                self._auto_restore_formal_playback()
                return
            self._start_session_item(session, next_item)
        except Exception:
            logger.exception("preview session advance failed; session left as-is")

    def _auto_restore_formal_playback(self) -> None:
        """P17 acceptance: one natural preview end may restore the formal playback
        it interrupted -- fail closed, never from stale or user-overridden state.

        Eligibility (all required): the restore arm names this preview's own
        interruption (armed at preview start; disarmed by explicit stop, failure,
        advance-to-completion, or any user playback command); its pause succeeded;
        Music.app still reads ``paused`` on the SAME persistent id. Anything else
        -- including a user decision made out of band -- leaves playback alone.

        Runs on the reaper thread and must never raise: continuity restoration is
        best effort and cleanup outranks it. The restore uses the adapter's own
        ``play`` primitive -- the exact boundary ``_execute_play`` uses -- and
        deliberately does not touch the action-log register: this is continuity
        restoration, not a user command, so intent routing is not misled.
        """
        entry = self._playback_context.suspension.take_restore_arm()
        if entry is None or not entry.pause_ok:
            device_safety_trace(
                f"auto-restore: arm={entry!r} -> no restore "
                "(nothing armed / pause did not succeed)"
            )
            return
        adapter = self._playback_adapter
        if adapter is None:
            device_safety_trace("auto-restore: no playback adapter wired -> no restore")
            return
        try:
            now_playing = adapter.read_now_playing()
        except Exception as error:
            device_safety_trace(
                f"auto-restore: read_now_playing raised {error!r} -> no restore"
            )
            return
        state = getattr(now_playing, "state", None)
        state_value = getattr(state, "value", state)
        if state_value != "paused":
            device_safety_trace(
                f"auto-restore: Music.app state={state_value!r} (not paused) "
                "-> no restore (user acted, or the pause never held)"
            )
            return
        if (
            entry.persistent_id is not None
            and getattr(now_playing, "persistent_id", None) != entry.persistent_id
        ):
            device_safety_trace(
                "auto-restore: now playing a different persistent id -> no restore"
            )
            return
        try:
            device_safety_trace("auto-restore: resuming formal playback via adapter.play()")
            adapter.play()
        except Exception as error:
            device_safety_trace(f"auto-restore: adapter.play() raised {error!r} -> no restore")

    def _finish_session_if_completed(self) -> None:
        """One-shot COMPLETED notice + registry clear -- terminal states are final."""
        snapshot = self._playback_context.sessions.finish_completed()
        if snapshot is not None:
            self._notify_preview_event("completed", snapshot)

    def _notify_preview_event(self, event: str, snapshot: PreviewSessionSnapshot) -> None:
        """P15-S1: deliver one session event to the optional presenter; listener
        failures are swallowed (presentation may never break audio)."""
        handler = self.preview_event_handler
        logger.debug(
            "[preview-ipc] run: event generated kind=%s session_state=%s consumer=%s",
            event,
            snapshot.state if snapshot is not None else None,
            type(handler).__name__ if handler is not None else "none",
        )
        if handler is None:
            return
        try:
            handler(
                {
                    "event": event,
                    "session": self._session_snapshot_dict(snapshot),
                    "suspended": self._suspended_entry_dict(
                        self._playback_context.suspension.value
                    ),
                }
            )
        except Exception:
            logger.exception("preview event presenter raised; event dropped")

    def _session_snapshot_dict(
        self, snapshot: PreviewSessionSnapshot | None
    ) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        return {
            "state": snapshot.state,
            "total": snapshot.total,
            "position": snapshot.position,
            "current_canonical_id": snapshot.current_canonical_id,
            "current_name": snapshot.current_name,
            "skipped": list(snapshot.skipped),
            "failure_reason": snapshot.failure_reason,
        }

    def _suspended_entry_dict(
        self, entry: SuspendedPlaybackEntry | None
    ) -> dict[str, Any] | None:
        if entry is None:
            return None
        return {
            "player_state": entry.player_state,
            "persistent_id": entry.persistent_id,
            "name": entry.name,
            "pause_ok": entry.pause_ok,
        }

    def _update_batch_item_position(self, canonical_id: str) -> None:
        """P14-C07.3: point the batch-item cursor at the audio action just taken.

        Member-gated: the cursor moves only when the acted track is an item of the run
        this service last delivered (the in-memory pointer, resolved against durable
        history so rank order stays authoritative). A track outside the run -- or a
        pointer that no longer resolves -- clears the cursor while the batch pointer
        survives; no pointer means nothing to record.
        """
        run_id = self._active_context.active_run_id
        if run_id is None:
            return
        result = self._recommendation_history.get_result(run_id)
        if result is None:
            self._active_context.clear_item_index()
            return
        for item_index, item in enumerate(result.items):
            if item.candidate.target.target_id == canonical_id:
                self._active_context.note_batch_item(item_index)
                return
        self._active_context.clear_item_index()

    def _execute_add_catalog_to_library(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from music_agent.catalog_library import (
            CatalogLibraryUnavailableError,
            add_catalog_song_to_library,
        )

        canonical_id = payload["canonical_id"]
        catalog_id = self._require_catalog_binding(canonical_id)
        if self._catalog_library_transport is None:
            raise CatalogLibraryUnavailableError(
                "no catalog library transport is wired; library add stays unavailable"
            )
        result = add_catalog_song_to_library(
            self._canonical, self._catalog_library_transport, catalog_id, canonical_id
        )
        return {
            "catalog_id": result.catalog_id,
            "canonical_id": result.canonical_id,
            "add_status": result.add_status,
            "readback_status": result.readback_status,
            "bound_persistent_id": result.bound_persistent_id,
            "bound_isrc": result.bound_isrc,
            "error": result.error,
        }

    def _execute_discover_catalog_tracks(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """P11-T1/T2: real catalog search -> staging -> relation resolution -> promotion.

        The injected source searches the read-only catalog API; each hit is deduped /
        known-library excluded, staged durably, then its relations are resolved through
        authoritative Catalog Artist / Album identities and promotion is attempted through the
        sealed path. Blocked hits stay staged and report why.
        """
        from music_agent.apple_music_catalog import (
            CatalogCredentialsError,
            CatalogMappingError,
            CatalogTransportError,
        )
        from music_agent.candidate_staging import CandidateStagingRepository
        from music_agent.catalog_ingestion import (
            CatalogIngestionOrchestrator,
            CatalogIngestStatus,
        )
        from music_agent.catalog_track_state_repository import CatalogTrackStateRepository

        source = self._catalog_search_source
        if source is None:
            raise CatalogDiscoveryUnavailableError(
                "no catalog search source is wired; catalog discovery stays unavailable"
            )
        term = payload["term"]
        limit = payload.get("limit", 25)
        try:
            tracks = source.search(term, limit)
            with CandidateStagingRepository(self.database_path) as staging, CatalogTrackStateRepository(
                self.database_path
            ) as track_state:
                orchestrator = CatalogIngestionOrchestrator(
                    self._canonical, staging, source, track_state=track_state
                )
                outcomes = orchestrator.ingest(tracks, term=term)
        except (CatalogCredentialsError, CatalogTransportError, CatalogMappingError) as error:
            raise CatalogDiscoveryError(str(error), code=error.code) from error
        promoted: list[dict[str, Any]] = []
        staged: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for track, outcome in zip(tracks, outcomes):
            entry = {
                "catalog_id": outcome.catalog_id,
                "name": track.name,
                "artist_name": ", ".join(track.artist_names),
                "status": outcome.status.value,
                "canonical_id": outcome.canonical_id,
                "error": outcome.error,
                "blocker_codes": list(outcome.blocker_codes),
                "preview_url": track.preview_url,
            }
            if outcome.status is CatalogIngestStatus.PROMOTED:
                promoted.append(entry)
            elif outcome.status in (
                CatalogIngestStatus.STAGED,
                CatalogIngestStatus.STAGED_BLOCKED,
                CatalogIngestStatus.IDENTITY_CONFLICT,
            ):
                staged.append(entry)
            else:
                skipped.append(entry)
        return {
            "term": term,
            "limit": limit,
            "discovered_count": len(tracks),
            "promoted_count": len(promoted),
            "staged_count": len(staged),
            "skipped_count": len(skipped),
            "promoted": promoted,
            "staged": staged,
            "skipped": skipped,
        }

    # --- tool handlers (transient playback; P10.12) ----------------------------

    def _require_playback(self):
        if self._playback_adapter is None:
            raise PlaybackUnavailableError(
                "no playback adapter is wired; playback stays unavailable"
            )
        return self._playback_adapter

    def _playback_elapsed_ms(self) -> float | None:
        adapter = self._playback_adapter
        elapsed = getattr(adapter, "last_command_elapsed_ms", None)
        return round(elapsed, 1) if isinstance(elapsed, (int, float)) else None

    def _execute_play(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._run_playback_command("play", lambda adapter: adapter.play())
        return {"command": "play", "ok": True, "elapsed_ms": self._playback_elapsed_ms()}

    def _execute_pause(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._run_playback_command("pause", lambda adapter: adapter.pause())
        return {"command": "pause", "ok": True, "elapsed_ms": self._playback_elapsed_ms()}

    def _execute_search_library_tracks(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Mixed canonical-store Track lookup (historical tool name; P20 Slice 1).

        Read-only over all known canonical Tracks: Music.app Library, catalog and
        otherwise unbound records. Provenance and identity evidence prevent this
        mixed result set from being mistaken for Library membership.
        """
        term = payload["term"]
        limit = payload.get("limit") or _LIBRARY_SEARCH_DEFAULT_LIMIT
        model = self._canonical.load_model()
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        ordered = SharedAgentService._library_lookup_matches(model, term)
        shown = ordered[:limit]
        return {
            "matched_count": len(ordered),
            "matches": [
                {
                    "target_id": track["id"],
                    "name": track.get("name"),
                    "artist_name": (
                        ", ".join(
                            SharedAgentService._library_track_artist_names(track, artist_by_id)
                        )
                        or None
                    ),
                    "playback": SharedAgentService._playback_annotation(track),
                    "provenance": SharedAgentService._library_search_provenance(track),
                    "bindings": SharedAgentService._library_search_bindings(track),
                }
                for track in shown
            ],
        }

    def _execute_query_catalog_discovery_state(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """P15-S3-S2: read-only long-term catalog-track memory.

        canonical_id mode: one track's durable discovery/recommendation facts plus the
        derived never_recommended / previously_recommended labels. term mode: the rows whose
        bounded discovery-term memory contains the normalized term (per-track remembered
        yields only -- this is a memory/fact view, never a freshness verdict on the live
        catalog). No ranking, no score, no exploration eligibility: the exploration
        combination read lands in S3-S3.
        """
        track_by_id, artist_by_id = self._canonical_display_maps()
        if payload.get("canonical_id") is not None:
            state = self._catalog_track_state.get_state(payload["canonical_id"])
            if state is None:
                return {"found": False, "canonical_id": payload["canonical_id"]}
            entry = SharedAgentService._catalog_state_entry(
                state, track_by_id, artist_by_id
            )
            entry["discovery_terms"] = state.discovery_terms
            return {"found": True, "state": entry}
        limit = payload.get("limit", 25)
        normalized = normalize_discovery_term(payload["term"])
        matches: list[dict[str, Any]] = []
        for state in self._catalog_track_state.find_states_by_term(payload["term"]):
            entry = SharedAgentService._catalog_state_entry(
                state, track_by_id, artist_by_id
            )
            # A matched row carries the normalized key by construction (the repository
            # matched on json_each.key); KeyError here would be corruption, keep it loud.
            entry["term_count"] = state.discovery_terms[normalized]
            matches.append(entry)
        return {
            "term": payload["term"],
            "match_count": len(matches),
            "matches": matches[:limit],
        }

    @staticmethod
    def _catalog_state_entry(
        state: CatalogTrackState,
        track_by_id: dict[str, dict[str, Any]],
        artist_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """One long-term memory entry: durable facts + derived recommendation labels."""
        track = track_by_id.get(state.canonical_id)
        artist_names = (
            [
                artist_by_id[artist_id].get("name")
                for artist_id in track.get("artist_ids", ())
                if artist_id in artist_by_id
            ]
            if track is not None
            else []
        )
        return {
            "canonical_id": state.canonical_id,
            "name": track.get("name") if track is not None else None,
            "artist_name": (
                ", ".join(name for name in artist_names if name) if artist_names else None
            ),
            "source_system": state.source_system,
            "first_discovered_at": state.first_discovered_at,
            "last_discovered_at": state.last_discovered_at,
            "discovery_count": state.discovery_count,
            "first_recommended_at": state.first_recommended_at,
            "last_recommended_at": state.last_recommended_at,
            "recommendation_count": state.recommendation_count,
            "never_recommended": state.recommendation_count == 0,
            "previously_recommended": state.recommendation_count > 0,
        }

    def _execute_next_track(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._run_playback_command("next_track", lambda adapter: adapter.next_track())
        # Player-queue navigation: the agent's own channel anchor no longer describes
        # what is playing; the persistent_id anchor stays for ownership judgment.
        self._active_context.clear_channel()
        return {"command": "next_track", "ok": True, "elapsed_ms": self._playback_elapsed_ms()}

    def _execute_previous_track(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._run_playback_command("previous_track", lambda adapter: adapter.previous_track())
        self._active_context.clear_channel()
        return {"command": "previous_track", "ok": True, "elapsed_ms": self._playback_elapsed_ms()}

    def _run_playback_command(self, command: str, invoke) -> None:
        """Run one transient playback command; failures map to the typed domain error."""
        try:
            invoke(self._require_playback())
        except PlaybackUnavailableError:
            raise
        except Exception as error:
            raise PlaybackCommandFailedError(str(error)) from error
        # P17 acceptance: a user's playback decision (play/pause/next/previous/
        # play_track) cancels any pending natural-end auto-restore -- the user
        # has taken control; a stale suspension must never override it.
        self._playback_context.suspension.disarm_restore()

    def _execute_play_track(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        canonical_id = payload["canonical_id"]
        model = self._canonical.load_model()
        track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
        if track is None:
            raise CanonicalEntityNotFoundError(f"no canonical entity {canonical_id} in the store")
        persistent_id = track["external_ids"]["apple_music_persistent_id"]
        resolution = "binding"
        if persistent_id is None:
            # P12: playback-equivalent resolution (read-only, evidence-gated). Writes
            # nothing -- the resolved id is live Music.app evidence, never a binding claim.
            persistent_id = self._resolve_playback_equivalent(model, track)
            resolution = "playback_equivalent"
            if persistent_id is None:
                raise PlaybackUnavailableError(
                    f"track {canonical_id} has no Apple Music binding and no unique "
                    "playback-equivalent library track; play_track fails closed"
                )
        else:
            key = ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
            if self._canonical.lookup_external_identity(key) != canonical_id:
                raise PlaybackUnavailableError(
                    f"Apple Music binding for {canonical_id} does not resolve back; play_track fails closed"
                )
        self._run_playback_command(
            "play_track", lambda adapter: adapter.play_track(persistent_id)
        )
        self._active_context.note_library_playback(canonical_id, persistent_id)
        self._update_batch_item_position(canonical_id)
        return {
            "command": "play_track",
            "ok": True,
            "persistent_id": persistent_id,
            "resolution": resolution,
            "elapsed_ms": self._playback_elapsed_ms(),
        }

    def _resolve_playback_equivalent(self, model: Mapping[str, Any], track: Mapping[str, Any]) -> str | None:
        """Resolve one binding-less track to a playback-equivalent library persistent ID.

        Playback resolution only: no canonical write, no binding, no identity fact.
        Returns None when the resolver is unwired or no unique match exists; resolver
        errors surface as a typed PlaybackUnavailableError (fail closed).
        """
        resolver = self._playback_resolver
        if resolver is None:
            return None
        artist_names = [
            artist["name"]
            for artist in model["artists"]
            if artist["id"] in track["artist_ids"] and artist.get("name")
        ]
        album_name = next(
            (
                album["name"]
                for album in model["albums"]
                if album["id"] == track["album_id"] and album.get("name")
            ),
            None,
        )
        try:
            return resolver.resolve_playback_track(
                name=track["name"],
                artist=", ".join(artist_names) if artist_names else None,
                album=album_name,
                duration_ms=track["duration_ms"],
            )
        except Exception as error:
            raise PlaybackUnavailableError(
                f"playback-equivalent lookup failed: {error}"
            ) from error

    @staticmethod
    def _playback_identity_text(value: str | None) -> str | None:
        """Exact playback-identity comparison form (never fuzzy or ranked)."""
        if not isinstance(value, str) or not value.strip():
            return None
        return unicodedata.normalize("NFC", " ".join(value.split())).casefold()

    def _resolve_current_player_canonical(
        self, now_playing: Any
    ) -> tuple[str | None, str | None]:
        """Resolve the current Music.app track without creating identity facts.

        An existing persistent-ID binding is authoritative. With no binding, only
        canonical tracks having exact normalized title/artist evidence are sent
        through the existing duration-gated playback-equivalent resolver. Exactly
        one canonical track must resolve back to the current persistent ID.
        """
        def field(name: str) -> Any:
            if isinstance(now_playing, Mapping):
                return now_playing.get(name)
            return getattr(now_playing, name, None)

        persistent_id = field("persistent_id")
        if not isinstance(persistent_id, str) or not persistent_id.strip():
            return None, None
        persistent_id = persistent_id.strip()
        bound = self._canonical.lookup_external_identity(
            ExternalIdentityKey("apple_music", EntityType.TRACK, persistent_id)
        )
        if bound is not None:
            return bound, "binding"

        current_name = self._playback_identity_text(
            field("name")
        )
        current_artist = self._playback_identity_text(
            field("artist")
        )
        if current_name is None or current_artist is None:
            return None, None

        model = self._canonical.load_model()
        artist_by_id = {
            artist["id"]: artist
            for artist in model["artists"]
        }
        resolved: list[str] = []
        for track in model["tracks"]:
            # A different exact binding is contrary evidence; playback-equivalent
            # fallback is only for canonical tracks that lack a Music.app binding.
            if track["external_ids"].get("apple_music_persistent_id") is not None:
                continue
            artist_names = self._library_track_artist_names(track, artist_by_id)
            if (
                self._playback_identity_text(track.get("name")) != current_name
                or self._playback_identity_text(", ".join(artist_names))
                != current_artist
            ):
                continue
            if self._resolve_playback_equivalent(model, track) == persistent_id:
                resolved.append(track["id"])
                if len(resolved) > 1:
                    return None, None
        if len(resolved) == 1:
            return resolved[0], "playback_equivalent"
        return None, None

    def _now_playing_context(self, persistent_id: str | None) -> str:
        """Ownership of the Music.app current track relative to this service's output.

        ``agent_selected`` -- the playing track is the one the agent last commanded via
        play_track (in-memory persistent_id anchor), or its pid matches a library binding
        inside the latest ``_NOW_PLAYING_CONTEXT_RUN_WINDOW`` recommendation runs.
        ``own_queue`` -- recent recommendation runs carry track targets but none matches
        the playing pid: the user is listening to their own queue.
        ``unknown`` -- no usable evidence (nothing playing, no anchor, no run yet, or no
        track targets in the window). Computed per read from the same durable sources the
        model can see; never persisted and never guessed.
        """
        if persistent_id is None:
            return "unknown"
        pid = persistent_id.strip()
        if not pid:
            return "unknown"
        anchor = self._active_context.persistent_id
        if anchor is not None and anchor == pid:
            return "agent_selected"
        # Evidence that this service HAS produced selectable output: the in-memory anchor
        # or at least one track target inside the run window. With such evidence and no
        # match, the playing pid belongs to the user's own queue; without any evidence,
        # nothing can be claimed either way.
        agent_output_exists = anchor is not None
        results = self._recommendation_history.list_runs()
        model = self._canonical.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        for result in results[:_NOW_PLAYING_CONTEXT_RUN_WINDOW]:
            for item in result.items:
                target = item.candidate.target
                if target.kind is not PreferenceTargetKind.TRACK:
                    continue
                agent_output_exists = True
                track = track_by_id.get(target.target_id)
                if track is None:
                    continue
                binding = track["external_ids"].get("apple_music_persistent_id")
                if binding == pid:
                    return "agent_selected"
        return "own_queue" if agent_output_exists else "unknown"

    def _execute_get_now_playing(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Typed now-playing read: adapter failures become a typed tool error, never a
        crash -- the provider loop turns the envelope into an honest final answer.

        P3B: the snapshot is enriched with two view-layer, in-memory-only fields --
        ``context`` (agent_selected/own_queue/unknown ownership of the current track)
        and ``agent_channel`` (this service's last playback/preview action). Neither is
        persisted and neither changes playback tool behavior.
        """
        adapter = self._require_playback()
        try:
            now_playing = adapter.read_now_playing()
        except PlaybackUnavailableError:
            raise
        except Exception as error:
            raise PlaybackCommandFailedError(str(error)) from error
        player_canonical_id, canonical_resolution = (
            self._resolve_current_player_canonical(now_playing)
        )
        return {
            "now_playing": {
                "state": now_playing.state.value,
                "persistent_id": now_playing.persistent_id,
                "name": now_playing.name,
                "artist": now_playing.artist,
                "album": now_playing.album,
                "elapsed_ms": self._playback_elapsed_ms(),
            },
            "context": self._now_playing_context(now_playing.persistent_id),
            "agent_channel": {
                "state": self._active_context.channel,
                "canonical_id": self._active_context.canonical_id,
            },
            "player_canonical_id": player_canonical_id,
            "canonical_resolution": canonical_resolution,
        }

    def _active_batch_projection(self) -> dict[str, Any] | None:
        """Projection of the active recommendation batch, or ``None`` when no run exists.

        P14-C07.3 provenance: ``register`` when the in-memory batch pointer (set at the
        generate boundary) resolves in history -- the run this service actually
        delivered; ``derived`` when there is no pointer (a fresh instance or another
        process) and the newest persisted run stands in. Identity facts only: candidate
        content, scores, playback routes and target lists are deliberately never copied
        here -- consumers fetch the full run on demand via ``get_recommendation_run``.
        """
        run_id = self._active_context.active_run_id
        if run_id is not None:
            result = self._recommendation_history.get_result(run_id)
            if result is not None:
                return {
                    "run_id": result.run_id,
                    "source": "register",
                    "produced_at": result.produced_at.isoformat(),
                    "item_count": len(result.items),
                }
        results = self._recommendation_history.list_runs()
        if not results:
            return None
        result = results[0]
        return {
            "run_id": result.run_id,
            "source": "derived",
            "produced_at": result.produced_at.isoformat(),
            "item_count": len(result.items),
        }

    def verified_selections_for_run(
        self, run_id: str
    ) -> tuple[VerifiedSelection, ...]:
        """Read the session projection for one exact RecommendationRun.

        This is an internal code-owned boundary, not an Agent tool or Provider
        payload. ``ActiveMusicContext`` remains the sole state owner.
        """
        return self._active_context.verified_selections_for_run(run_id)

    def record_verified_selection(
        self,
        *,
        run_id: str,
        canonical_id: str,
        item_position: int,
        action_kind: str,
        playback_route: str,
    ) -> VerifiedSelection:
        """Write one completed ActionAttempt fact to the session owner.

        The Provider loop calls this only after ``ActionAttempt.COMPLETED``. A fresh
        service using the established derived-active fallback may bind that exact run;
        a conflicting live run still fails closed inside ``ActiveMusicContext``.
        """
        return self._active_context.note_verified_selection(
            run_id=run_id,
            canonical_id=canonical_id,
            item_position=item_position,
            action_kind=action_kind,
            playback_route=playback_route,
        )

    def _execute_get_active_context(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P14-C06.2: unified runtime-only observation of the service's transient music
        context. ``get_now_playing`` keeps its exact contract; this is an additive,
        read-only companion that composes it with the in-memory channel register and the
        newest recommendation batch.

        The context facts are read by the shared ``_context_observation`` (see there
        for the truth semantics; adds the P19 conversational referent alongside the
        channel register); ``active_batch`` projects the active recommendation
        batch -- the in-memory pointer resolves first (``source="register"``), the
        newest persisted run stands in when there is no pointer (``source="derived"``);
        identity facts only, no candidate copies. Nothing here is written, cached, or
        guessed.
        """
        observation = self._context_observation()
        player = observation["player"]
        if isinstance(player, Mapping):
            player = dict(player)
            canonical_id, canonical_resolution = (
                self._resolve_current_player_canonical(player)
            )
            player["canonical_id"] = canonical_id
            player["canonical_resolution"] = canonical_resolution
        return {
            "channel": observation["channel"],
            "referent_canonical_id": observation["referent_canonical_id"],
            "preview_sounding": observation["preview_sounding"],
            "player": player,
            "context": observation["context"],
            "active_batch": self._active_batch_projection(),
        }

    def _context_observation(self) -> dict[str, Any]:
        """P15-PC shared read: one consistent point-in-time observation.

        ``channel`` mirrors get_now_playing's ``agent_channel`` (snapshot of the
        single-writer register), ``referent_canonical_id`` is the conversational
        track target (P19-T14-F-R4: the last successfully resolved explicit track
        interaction; survives preview stops by design), ``preview_sounding`` is the
        real runner truth
        (``is_preview_active`` -- independent of the channel action log, and
        deliberately NOT stored in the register), ``player`` is the same adapter
        read without position info (``None`` when no playback adapter is wired;
        adapter failures become a typed tool error, as in get_now_playing), and
        ``context`` reuses the same ownership judgment.
        """
        snapshot = self._active_context.snapshot()
        preview_sounding = False
        if self._preview_runner is not None:
            try:
                preview_sounding = self._preview_runner.is_preview_active()
            except Exception:
                # Truth unreadable -> report not sounding -> routing defers to the
                # provider loop instead of ever claiming a false positive.
                preview_sounding = False
        player: dict[str, Any] | None = None
        context = "unknown"
        if self._playback_adapter is not None:
            try:
                now_playing = self._playback_adapter.read_now_playing()
            except PlaybackUnavailableError:
                raise
            except Exception as error:
                raise PlaybackCommandFailedError(str(error)) from error
            player = {
                "state": now_playing.state.value,
                "persistent_id": now_playing.persistent_id,
                "name": now_playing.name,
                "artist": now_playing.artist,
                "album": now_playing.album,
            }
            context = self._now_playing_context(now_playing.persistent_id)
        return {
            "channel": {
                "state": snapshot.channel,
                "canonical_id": snapshot.canonical_id,
            },
            "referent_canonical_id": snapshot.referent_canonical_id,
            "preview_sounding": preview_sounding,
            "player": player,
            "context": context,
        }

    def _execute_get_playback_context(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """P15-PC: unified playback-continuity observation -- the frozen context facts
        composed with the two P15 coordination registers.

        ``session`` is the live continuous-preview snapshot (or null -- terminal
        sessions are cleared), ``suspended`` the historical restore-by-intent memo,
        ``preview_suspension`` the current preview's pending restore obligation,
        and ``referent_canonical_id`` the P19 conversational track target (survives
        preview stops; see ``_context_observation``).
        Read-only; nothing is written and nothing here resumes anything.
        """
        observation = self._context_observation()
        return {
            "channel": observation["channel"],
            "referent_canonical_id": observation["referent_canonical_id"],
            "preview_sounding": observation["preview_sounding"],
            "player": observation["player"],
            "session": self._session_snapshot_dict(
                self._playback_context.sessions.snapshot()
            ),
            "suspended": self._suspended_entry_dict(
                self._playback_context.suspension.value
            ),
            # P20 UI lifecycle: ``suspended`` remains the historical
            # restore-by-intent memo. This companion is the current preview's
            # still-pending auto-restore obligation and disappears on every
            # terminal/cancel/override path through the existing restore arm.
            "preview_suspension": self._suspended_entry_dict(
                self._playback_context.suspension.pending_restore
            ),
        }

    # --- tool handlers (live write) -------------------------------------------

    def _execute_execute_write_intent(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._write_adapter is None:
            raise WriteAdapterUnavailableError(
                "no write command adapter is wired; live writes stay unavailable"
            )
        orchestrator = WriteOrchestrator(self._write_execution, self._write_adapter)
        intent = orchestrator.execute_pending_intent(payload["intent_id"])
        return {"intent_id": intent.intent_id, "state": intent.state.value}


def _resolve_completed_at(completed_at: str | None) -> datetime:
    if completed_at is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(completed_at)
    except (TypeError, ValueError) as error:
        raise SharedAgentServiceValidationError(
            "completed_at must be a timezone-aware ISO datetime"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SharedAgentServiceValidationError(
            "completed_at must be a timezone-aware ISO datetime"
        )
    return parsed


def _application_projection(record: object) -> dict[str, Any]:
    target = record.target
    attribution = record.attribution
    return {
        "feedback_id": record.feedback_id,
        "proposal_kind": record.proposal_kind.value,
        "target_kind": None if target is None else target.kind.value,
        "target_id": None if target is None else target.target_id,
        "signal_source_system": record.signal_source_system,
        "signal_path": record.signal_path,
        "provenance": record.provenance,
        "attribution_relation": None if attribution is None else attribution.relation.value,
        "applied_at": record.applied_at,
    }
