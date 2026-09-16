"""P09.2: the agent tool registry -- the stable callable surface over existing capabilities.

Model clients never touch repositories, tables, migrations, or domain internals. They invoke one
of the tools registered here by its stable name with a bounded JSON payload. Each
:class:`AgentToolSpec` declares:

* ``name`` -- the stable tool identity (part of the agent-facing surface, versioned by the agent
  contract, not by the tool set);
* ``permission_class`` -- ``read`` (no durable mutation), ``mutate`` (durable internal mutation
  whose safety/readiness semantics are established by the owning phase: append-only journals,
  dedup keys, readback conventions), or ``live_write`` (Apple Music writes governed by the P02--
  P05 capability matrix -- none is execution-ready today, so these always fail closed at the
  permission gate);
* ``validate_payload`` -- a hand-rolled fail-closed validator for the payload envelope (required
  keys, types, ID namespaces, ISO timestamps). Deep semantic validation stays in the owning
  domain contract; this layer rejects malformed envelopes before any execution.

Payload validation is deliberately structural, never semantic: it rejects unknown keys, wrong
types, foreign ID namespaces, and naive timestamps, and leaves everything else to the domain
layers the tools delegate to.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from music_agent.feedback_contract import AttributionRelation, FeedbackKind, validate_feedback_id
from music_agent.identity import EntityType, validate_canonical_id
from music_agent.preference_attribution import PreferenceTargetKind
from music_agent.recommendation_contract import validate_candidate_id, validate_run_id
from music_agent.write_intent import validate_intent_id


class AgentToolError(ValueError):
    code = "agent_tool_error"


class AgentToolValidationError(AgentToolError):
    code = "validation_error"


class AgentToolName(StrEnum):
    """The stable tool identities exposed by the shared agent service."""

    # canonical music facts
    GET_CANONICAL_ENTITY = "get_canonical_entity"
    # P06 preference state
    QUERY_TRACK_PREFERENCE = "query_track_preference"
    # P07 recommendation state
    LIST_RECOMMENDATION_RUNS = "list_recommendation_runs"
    GET_RECOMMENDATION_RUN = "get_recommendation_run"
    GENERATE_RECOMMENDATION = "generate_recommendation"
    # P08 feedback / learning state
    LIST_FEEDBACK_OBSERVATIONS = "list_feedback_observations"
    GET_FEEDBACK_OBSERVATION = "get_feedback_observation"
    RECORD_FEEDBACK = "record_feedback"
    INTERPRET_FEEDBACK = "interpret_feedback"
    LIST_LEARNING_APPLICATIONS = "list_learning_applications"
    GET_LEARNING_APPLICATION = "get_learning_application"
    APPLY_LEARNING = "apply_learning"
    # capability / readiness surface
    GET_AGENT_CAPABILITIES = "get_agent_capabilities"
    # live Apple Music writes (capability-gated; never execution-ready today)
    EXECUTE_WRITE_INTENT = "execute_write_intent"
    # transient playback control (P10.12; NOT a library-write capability)
    PLAY = "play"
    PAUSE = "pause"
    NEXT_TRACK = "next_track"
    PREVIOUS_TRACK = "previous_track"
    PLAY_TRACK = "play_track"
    GET_NOW_PLAYING = "get_now_playing"
    # P14-C06.2: unified runtime-only observation of the service's transient music context
    GET_ACTIVE_CONTEXT = "get_active_context"
    # P10.17b: source-scoped direct + genre-inferred recommendation (additive tool)
    GENERATE_INFERRED_RECOMMENDATION = "generate_inferred_recommendation"
    # P11.3: catalog usage path -- preview (transient, no library change) and
    # library add (a real library mutation, LIVE_WRITE class)
    PREVIEW_CATALOG_TRACK = "preview_catalog_track"
    # P3B batch 2: stop the agent-launched non-blocking preview (afplay boundary only)
    STOP_PREVIEW = "stop_preview"
    ADD_CATALOG_TO_LIBRARY = "add_catalog_to_library"
    # P16-S4: open one catalog track's official Apple Music page in the browser
    # (real trackViewUrl resolved service-side from the durable itunes_store binding)
    OPEN_IN_APPLE_MUSIC = "open_in_apple_music"
    # P11-T1/T2: catalog discovery -> staging -> relation resolution -> auto promotion
    DISCOVER_CATALOG_TRACKS = "discover_catalog_tracks"
    # P14-R3.1: store-side library name lookup (read-only; library-first playback resolution)
    SEARCH_LIBRARY_TRACKS = "search_library_tracks"
    # P15-S1: continuous preview of the active recommendation batch (session)
    PREVIEW_BATCH = "preview_batch"
    # P15-S1 C02: advance the RUNNING session one clip (CLI-routing-only tool)
    ADVANCE_PREVIEW = "advance_preview"
    # P15-PC: unified playback-continuity observation (session + suspension composed)
    GET_PLAYBACK_CONTEXT = "get_playback_context"
    # P15-S3-S2: long-term catalog-track memory read (derived facts only; no ranking/score)
    QUERY_CATALOG_DISCOVERY_STATE = "query_catalog_discovery_state"


class AgentToolPermissionClass(StrEnum):
    """The permission class of a tool, evaluated before any execution."""

    READ = "read"
    MUTATE = "mutate"
    LIVE_WRITE = "live_write"


class AgentToolSpec:
    """One registered agent tool: name, permission class, and payload-envelope validator."""

    __slots__ = ("name", "permission_class", "validate_payload")

    def __init__(
        self,
        name: AgentToolName,
        permission_class: AgentToolPermissionClass,
        validate_payload: Callable[[Mapping[str, Any]], None],
    ) -> None:
        if not isinstance(name, AgentToolName):
            raise AgentToolValidationError("tool name must be an AgentToolName")
        if not isinstance(permission_class, AgentToolPermissionClass):
            raise AgentToolValidationError(
                "permission_class must be an AgentToolPermissionClass"
            )
        if not callable(validate_payload):
            raise AgentToolValidationError("validate_payload must be callable")
        self.name = name
        self.permission_class = permission_class
        self.validate_payload = validate_payload

    def validate(self, payload: Mapping[str, Any]) -> None:
        """Fail closed on a payload envelope outside this tool's shape."""
        self.validate_payload(payload)


class AgentToolRegistry:
    """The frozen registry of agent-facing tools, keyed by stable tool name."""

    def __init__(self, specs: Mapping[str, AgentToolSpec]) -> None:
        copied: dict[str, AgentToolSpec] = {}
        for name, spec in specs.items():
            if not isinstance(name, str) or name == "":
                raise AgentToolValidationError("registry keys must be non-empty strings")
            if name != spec.name.value:
                raise AgentToolValidationError(
                    f"registry key {name!r} must match spec name {spec.name.value!r}"
                )
            copied[name] = spec
        self._specs: Mapping[str, AgentToolSpec] = MappingProxyType(copied)

    def lookup(self, name: str | AgentToolName) -> AgentToolSpec | None:
        """Return the spec for a tool name, or ``None`` for an unknown tool (fail closed upstream)."""
        if isinstance(name, AgentToolName):
            name = name.value
        if not isinstance(name, str):
            return None
        return self._specs.get(name)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._specs)


# --- payload-envelope validators ------------------------------------------

_NOTHING = object()


def _require_exact_keys(payload: Mapping[str, Any], *names: str) -> None:
    if set(payload) != set(names):
        raise AgentToolValidationError(f"payload keys must be exactly {sorted(names)}")


def _require_keys(
    payload: Mapping[str, Any], required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> None:
    allowed = set(required) | set(optional)
    unknown = set(payload) - allowed
    if unknown:
        raise AgentToolValidationError(f"payload keys must be one of {sorted(allowed)}")
    missing = set(required) - set(payload)
    if missing:
        raise AgentToolValidationError(f"payload is missing required keys {sorted(missing)}")


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key, _NOTHING)
    if not isinstance(value, str) or value == "":
        raise AgentToolValidationError(f"{key} must be a non-empty string")
    return value


def _require_optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _require_string(payload, key)


def _require_aware_iso(payload: Mapping[str, Any], key: str) -> None:
    value = payload.get(key, _NOTHING)
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise AgentToolValidationError(f"{key} must be an ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AgentToolValidationError(f"{key} must be a timezone-aware ISO datetime")


def _require_optional_aware_iso(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None:
        _require_aware_iso(payload, key)


def _require_optional_bool(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None:
        if not isinstance(payload[key], bool):
            raise AgentToolValidationError(f"{key} must be a boolean")


def _require_optional_track_id_list(payload: Mapping[str, Any], key: str) -> None:
    if key not in payload or payload[key] is None:
        return
    value = payload[key]
    if not isinstance(value, list) or len(value) == 0:
        raise AgentToolValidationError(f"{key} must be a non-empty list of track ids")
    for entry in value:
        try:
            validate_canonical_id(EntityType.TRACK, entry)
        except ValueError as error:
            raise AgentToolValidationError(f"{key} contains an invalid id: {error}") from error


def _require_optional_genres(payload: Mapping[str, Any], key: str) -> None:
    if key not in payload or payload[key] is None:
        return
    value = payload[key]
    if not isinstance(value, list) or len(value) == 0:
        raise AgentToolValidationError(f"{key} must be a non-empty list of genre strings")
    for entry in value:
        if not isinstance(entry, str) or entry.strip() == "":
            raise AgentToolValidationError(f"{key} entries must be non-empty strings")


def _require_id(
    payload: Mapping[str, Any], key: str, validator: Callable[[str], None]
) -> str:
    """Require a string in the given ID namespace, preserving the owning module's rules."""
    value = _require_string(payload, key)
    try:
        validator(value)
    except ValueError as error:
        raise AgentToolValidationError(f"{key} is invalid: {error}") from error
    return value


def _require_track_target_id(payload: Mapping[str, Any], key: str) -> str:
    return _require_id(
        payload, key, lambda value: validate_canonical_id(EntityType.TRACK, value)
    )


def _require_feedback_id(payload: Mapping[str, Any], key: str) -> str:
    return _require_id(payload, key, validate_feedback_id)


def _require_feedback_kind(payload: Mapping[str, Any], key: str) -> str:
    value = _require_string(payload, key)
    known = {kind.value for kind in FeedbackKind}
    if value not in known:
        raise AgentToolValidationError(f"{key} must be one of {sorted(known)}")
    return value


def _validate_get_canonical_entity(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "canonical_id")
    _require_string(payload, "canonical_id")


def _validate_query_track_preference(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, ("target_id",), ("source_system",))
    _require_track_target_id(payload, "target_id")
    _require_optional_string(payload, "source_system")


def _validate_list_recommendation_runs(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, (), ("limit",))
    limit = payload.get("limit", _NOTHING)
    if limit is _NOTHING:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 20:
        raise AgentToolValidationError("limit must be an integer between 1 and 20")


def _validate_get_recommendation_run(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "run_id")
    _require_id(payload, "run_id", validate_run_id)


def _validate_generate_recommendation(
    payload: Mapping[str, Any], extra_optional: tuple[str, ...] = ()
) -> None:
    # Shared payload envelope for generate_recommendation and
    # generate_inferred_recommendation (registry comment below): exclusion args apply to
    # both; ``genres`` narrows a directed request's candidate space.
    # P15-S3-S3C: ``extra_optional`` lets the inferred tool widen ITS OWN envelope
    # without widening the plain tool's -- ``min_exploration`` is inferred-only and
    # remains an unknown key (rejected) for generate_recommendation.
    # P15 burn-down Issue 1: ``produced_at`` is deliberately NOT part of the
    # envelope -- the durable run time is service-authoritative (the trusted
    # ``completed_at`` execution context, never model input). A model that sends
    # ``produced_at`` anyway is rejected here as an unknown key (fail-closed).
    _require_keys(
        payload, ("target_ids", "limit"),
        (
            "source_system",
            "exclude_target_ids",
            "avoid_previous_runs",
            "genres",
        )
        + extra_optional,
    )
    target_ids = payload.get("target_ids", _NOTHING)
    if not isinstance(target_ids, list) or len(target_ids) == 0:
        raise AgentToolValidationError("target_ids must be a non-empty list of track ids")
    for value in target_ids:
        try:
            validate_canonical_id(EntityType.TRACK, value)
        except ValueError as error:
            raise AgentToolValidationError(f"target_ids contains an invalid id: {error}") from error
    limit = payload.get("limit", _NOTHING)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AgentToolValidationError("limit must be an integer >= 1")
    _require_optional_string(payload, "source_system")
    _require_optional_track_id_list(payload, "exclude_target_ids")
    _require_optional_bool(payload, "avoid_previous_runs")
    _require_optional_genres(payload, "genres")


def _validate_generate_inferred_recommendation(payload: Mapping[str, Any]) -> None:
    # P15-S3-S3C: the shared envelope plus the inferred-only exploration-floor knob.
    # ``min_exploration`` is an integer with ``0 <= value <= limit``, defaulting to 0
    # (absent/None == 0); it only shapes FINAL SELECTION and never scoring, affinity,
    # eligibility, or persistence. The plain tool never accepts this key.
    # P15-S3-S3D: same shape for ``min_fresh`` -- the best-effort same-run Fresh
    # floor. The min_fresh/min_exploration RELATIONSHIP is pinned in the service
    # (effective_min_exploration = max(...), normalization not refusal), not here:
    # the validator only enforces each knob's own 0 <= value <= limit bound, and
    # fresh identity itself is internal (the model can only ever name the count,
    # never the tracks).
    _validate_generate_recommendation(
        payload, extra_optional=("min_exploration", "min_fresh")
    )
    for key in ("min_exploration", "min_fresh"):
        value = payload.get(key, _NOTHING)
        if value is _NOTHING or value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise AgentToolValidationError(f"{key} must be an integer")
        if value < 0:
            raise AgentToolValidationError(f"{key} must be >= 0")
        if value > payload["limit"]:
            raise AgentToolValidationError(f"{key} must be <= limit")


def _validate_list_feedback_observations(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload)


def _validate_get_feedback_observation(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "feedback_id")
    _require_feedback_id(payload, "feedback_id")


def _validate_record_feedback(payload: Mapping[str, Any]) -> None:
    # P16-S1: observed_at is service-authoritative -- the model has no key for it
    # (a payload carrying one fails as an unknown key and nothing is written).
    _require_keys(
        payload,
        ("kind", "source_system", "source_path"),
        (
            "target_id",
            "run_id",
            "candidate_id",
            "attribution",
            "event_at",
            "source_event_id",
            "feedback_id",
        ),
    )
    _require_feedback_kind(payload, "kind")
    _require_string(payload, "source_system")
    _require_string(payload, "source_path")
    target_id = payload.get("target_id")
    run_id = payload.get("run_id")
    candidate_id = payload.get("candidate_id")
    has_target = target_id is not None
    has_recommendation = run_id is not None or candidate_id is not None
    if has_target == has_recommendation:
        raise AgentToolValidationError(
            "exactly one of target_id or the run_id/candidate_id pair is required"
        )
    if has_target:
        _require_track_target_id(payload, "target_id")
    else:
        if run_id is None or candidate_id is None:
            raise AgentToolValidationError("run_id and candidate_id must be provided together")
        _require_id(payload, "run_id", validate_run_id)
        _require_id(payload, "candidate_id", validate_candidate_id)
    attribution = payload.get("attribution")
    if attribution is not None:
        if not isinstance(attribution, dict) or set(attribution) != {
            "aspect_kind",
            "aspect_id",
            "relation",
        }:
            raise AgentToolValidationError(
                "attribution must be an object with exactly aspect_kind, aspect_id, relation"
            )
        kind = attribution.get("aspect_kind", _NOTHING)
        known_kinds = {item.value for item in PreferenceTargetKind}
        if kind not in known_kinds:
            raise AgentToolValidationError(
                f"attribution.aspect_kind must be one of {sorted(known_kinds)}"
            )
        if not isinstance(attribution.get("aspect_id"), str) or attribution["aspect_id"] == "":
            raise AgentToolValidationError("attribution.aspect_id must be a non-empty string")
        known_relations = {item.value for item in AttributionRelation}
        if attribution.get("relation") not in known_relations:
            raise AgentToolValidationError(
                f"attribution.relation must be one of {sorted(known_relations)}"
            )
    _require_optional_aware_iso(payload, "event_at")
    _require_optional_string(payload, "source_event_id")
    feedback_id = payload.get("feedback_id")
    if feedback_id is not None:
        _require_feedback_id(payload, "feedback_id")


def _validate_interpret_feedback(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "feedback_id")
    _require_feedback_id(payload, "feedback_id")


def _validate_list_learning_applications(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload)


def _validate_get_learning_application(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "feedback_id")
    _require_feedback_id(payload, "feedback_id")


def _validate_apply_learning(payload: Mapping[str, Any]) -> None:
    # P16-S1: applied_at is service-authoritative -- no model key, no payload path.
    _require_exact_keys(payload, "feedback_id")
    _require_feedback_id(payload, "feedback_id")


def _validate_get_agent_capabilities(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload)


def _validate_playback_no_args(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload)


def _validate_play_track(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "canonical_id")
    _require_track_target_id(payload, "canonical_id")


def _validate_execute_write_intent(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "intent_id")
    _require_id(payload, "intent_id", validate_intent_id)


def _validate_preview_catalog_track(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "canonical_id")
    _require_track_target_id(payload, "canonical_id")


def _validate_add_catalog_to_library(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "canonical_id")
    _require_track_target_id(payload, "canonical_id")


def _validate_open_in_apple_music(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, "canonical_id")
    _require_track_target_id(payload, "canonical_id")


def _validate_discover_catalog_tracks(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, ("term",), ("limit",))
    _require_string(payload, "term")
    if "limit" in payload and payload["limit"] is not None:
        limit = payload["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AgentToolValidationError("limit must be an integer >= 1")


def _validate_search_library_tracks(payload: Mapping[str, Any]) -> None:
    # Same envelope as discover_catalog_tracks: term required non-empty, optional limit >= 1.
    _require_keys(payload, ("term",), ("limit",))
    _require_string(payload, "term")
    if "limit" in payload and payload["limit"] is not None:
        limit = payload["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AgentToolValidationError("limit must be an integer >= 1")


def _validate_query_catalog_discovery_state(payload: Mapping[str, Any]) -> None:
    _require_keys(payload, (), ("canonical_id", "term", "limit"))
    canonical_id = payload.get("canonical_id")
    term = payload.get("term")
    if canonical_id is not None and term is not None:
        raise AgentToolValidationError("canonical_id and term are mutually exclusive")
    if canonical_id is None and term is None:
        raise AgentToolValidationError("exactly one of canonical_id or term is required")
    if canonical_id is not None:
        _require_track_target_id(payload, "canonical_id")
    else:
        _require_string(payload, "term")
    limit = payload.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AgentToolValidationError("limit must be an integer >= 1")


_READ = AgentToolPermissionClass.READ
_MUTATE = AgentToolPermissionClass.MUTATE
_LIVE_WRITE = AgentToolPermissionClass.LIVE_WRITE

AGENT_TOOL_REGISTRY = AgentToolRegistry(
    {
        AgentToolName.GET_CANONICAL_ENTITY.value: AgentToolSpec(
            AgentToolName.GET_CANONICAL_ENTITY, _READ, _validate_get_canonical_entity
        ),
        AgentToolName.QUERY_TRACK_PREFERENCE.value: AgentToolSpec(
            AgentToolName.QUERY_TRACK_PREFERENCE, _READ, _validate_query_track_preference
        ),
        AgentToolName.LIST_RECOMMENDATION_RUNS.value: AgentToolSpec(
            AgentToolName.LIST_RECOMMENDATION_RUNS, _READ, _validate_list_recommendation_runs
        ),
        AgentToolName.GET_RECOMMENDATION_RUN.value: AgentToolSpec(
            AgentToolName.GET_RECOMMENDATION_RUN, _READ, _validate_get_recommendation_run
        ),
        AgentToolName.GENERATE_RECOMMENDATION.value: AgentToolSpec(
            AgentToolName.GENERATE_RECOMMENDATION, _MUTATE, _validate_generate_recommendation
        ),
        AgentToolName.LIST_FEEDBACK_OBSERVATIONS.value: AgentToolSpec(
            AgentToolName.LIST_FEEDBACK_OBSERVATIONS, _READ, _validate_list_feedback_observations
        ),
        AgentToolName.GET_FEEDBACK_OBSERVATION.value: AgentToolSpec(
            AgentToolName.GET_FEEDBACK_OBSERVATION, _READ, _validate_get_feedback_observation
        ),
        AgentToolName.RECORD_FEEDBACK.value: AgentToolSpec(
            AgentToolName.RECORD_FEEDBACK, _MUTATE, _validate_record_feedback
        ),
        AgentToolName.INTERPRET_FEEDBACK.value: AgentToolSpec(
            AgentToolName.INTERPRET_FEEDBACK, _READ, _validate_interpret_feedback
        ),
        AgentToolName.LIST_LEARNING_APPLICATIONS.value: AgentToolSpec(
            AgentToolName.LIST_LEARNING_APPLICATIONS, _READ, _validate_list_learning_applications
        ),
        AgentToolName.GET_LEARNING_APPLICATION.value: AgentToolSpec(
            AgentToolName.GET_LEARNING_APPLICATION, _READ, _validate_get_learning_application
        ),
        AgentToolName.APPLY_LEARNING.value: AgentToolSpec(
            AgentToolName.APPLY_LEARNING, _MUTATE, _validate_apply_learning
        ),
        AgentToolName.GET_AGENT_CAPABILITIES.value: AgentToolSpec(
            AgentToolName.GET_AGENT_CAPABILITIES, _READ, _validate_get_agent_capabilities
        ),
        AgentToolName.EXECUTE_WRITE_INTENT.value: AgentToolSpec(
            AgentToolName.EXECUTE_WRITE_INTENT, _LIVE_WRITE, _validate_execute_write_intent
        ),
        # P10.12 transient playback: MUTATE class (client-policy gated), deliberately
        # NOT LIVE_WRITE -- the sealed library-write capability matrix is untouched.
        AgentToolName.PLAY.value: AgentToolSpec(
            AgentToolName.PLAY, _MUTATE, _validate_playback_no_args
        ),
        AgentToolName.PAUSE.value: AgentToolSpec(
            AgentToolName.PAUSE, _MUTATE, _validate_playback_no_args
        ),
        AgentToolName.NEXT_TRACK.value: AgentToolSpec(
            AgentToolName.NEXT_TRACK, _MUTATE, _validate_playback_no_args
        ),
        AgentToolName.PREVIOUS_TRACK.value: AgentToolSpec(
            AgentToolName.PREVIOUS_TRACK, _MUTATE, _validate_playback_no_args
        ),
        AgentToolName.PLAY_TRACK.value: AgentToolSpec(
            AgentToolName.PLAY_TRACK, _MUTATE, _validate_play_track
        ),
        AgentToolName.GET_NOW_PLAYING.value: AgentToolSpec(
            AgentToolName.GET_NOW_PLAYING, _READ, _validate_playback_no_args
        ),
        # P14-C06.2: unified active-context observation is a pure read of in-memory,
        # view-layer state plus on-demand adapter/history reads -- no mutation.
        AgentToolName.GET_ACTIVE_CONTEXT.value: AgentToolSpec(
            AgentToolName.GET_ACTIVE_CONTEXT, _READ, _validate_playback_no_args
        ),
        AgentToolName.GENERATE_INFERRED_RECOMMENDATION.value: AgentToolSpec(
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
            _MUTATE,
            # P15-S3-S3C: shared envelope + inferred-only optional ``min_exploration``.
            _validate_generate_inferred_recommendation,
        ),
        # P11.3 catalog usage path: preview is transient (MUTATE, like playback);
        # library add is a real external mutation (LIVE_WRITE, capability-gated).
        AgentToolName.PREVIEW_CATALOG_TRACK.value: AgentToolSpec(
            AgentToolName.PREVIEW_CATALOG_TRACK, _MUTATE, _validate_preview_catalog_track
        ),
        # P3B batch 2: stopping the agent's own preview is transient (MUTATE, like playback);
        # it never touches Music.app or durable state.
        AgentToolName.STOP_PREVIEW.value: AgentToolSpec(
            AgentToolName.STOP_PREVIEW, _MUTATE, _validate_playback_no_args
        ),
        AgentToolName.ADD_CATALOG_TO_LIBRARY.value: AgentToolSpec(
            AgentToolName.ADD_CATALOG_TO_LIBRARY, _LIVE_WRITE, _validate_add_catalog_to_library
        ),
        # P16-S4: opening Apple's official track page is a desktop side effect, never a
        # library/playback mutation (MUTATE, like preview) -- the URL is resolved
        # service-side from the durable binding, the model never composes one.
        AgentToolName.OPEN_IN_APPLE_MUSIC.value: AgentToolSpec(
            AgentToolName.OPEN_IN_APPLE_MUSIC, _MUTATE, _validate_open_in_apple_music
        ),
        # P11-T1: catalog discovery stages durable candidates (MUTATE), never a library write.
        AgentToolName.DISCOVER_CATALOG_TRACKS.value: AgentToolSpec(
            AgentToolName.DISCOVER_CATALOG_TRACKS,
            _MUTATE,
            _validate_discover_catalog_tracks,
        ),
        # P14-R3.1: store-only name lookup over the synced canonical model (READ, zero writes).
        AgentToolName.SEARCH_LIBRARY_TRACKS.value: AgentToolSpec(
            AgentToolName.SEARCH_LIBRARY_TRACKS,
            _READ,
            _validate_search_library_tracks,
        ),
        # P15-S1: starting a continuous preview session is transient audio (MUTATE, like
        # playback); the queue is assembled service-side from the active batch.
        AgentToolName.PREVIEW_BATCH.value: AgentToolSpec(
            AgentToolName.PREVIEW_BATCH, _MUTATE, _validate_playback_no_args
        ),
        # P15-S1 C02: advancing a running session is transient audio routing
        # (MUTATE, like playback); never touches Music.app or durable state.
        AgentToolName.ADVANCE_PREVIEW.value: AgentToolSpec(
            AgentToolName.ADVANCE_PREVIEW, _MUTATE, _validate_playback_no_args
        ),
        # P15-PC: the playback-continuity observation is a pure read of in-memory
        # registers plus on-demand adapter/runner reads -- no mutation.
        AgentToolName.GET_PLAYBACK_CONTEXT.value: AgentToolSpec(
            AgentToolName.GET_PLAYBACK_CONTEXT, _READ, _validate_playback_no_args
        ),
        # P15-S3-S2: pure long-term memory read -- no ranking, scoring, or eligibility.
        AgentToolName.QUERY_CATALOG_DISCOVERY_STATE.value: AgentToolSpec(
            AgentToolName.QUERY_CATALOG_DISCOVERY_STATE,
            _READ,
            _validate_query_catalog_discovery_state,
        ),
    }
)
