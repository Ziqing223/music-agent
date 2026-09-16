"""P22-S1: deterministic-first LLM conversation interpreter.

This module is deliberately a semantic translator, not a workflow agent.  It may
convert an unresolved deterministic :class:`TurnPlan` into another TurnPlan, but it
never receives tools and it cannot author canonical ids, recommendation run ids,
playback routes, tool names, selected tracks, or action outcomes.

The provider boundary is intentionally narrow:

    deterministic UNKNOWN
        -> provider.chat(..., tools=())
        -> strict JSON object
        -> structural + semantic validation
        -> existing TurnPlan

Malformed or contradictory output fails closed to a clarification TurnPlan.  A valid
``unsupported`` result preserves the unresolved plan so the caller may use the existing
FULL compatibility path when appropriate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping

from music_agent.intent_router import (
    PlaybackActionTurnSemantics,
    RecommendationTurnSemantics,
    TurnExpectedResult,
    TurnPlan,
    TurnPrimarySemantic,
    TurnSemanticSource,
    TurnTaskSurface,
)
from music_agent.provider_contract import (
    ChatProvider,
    ProviderMessage,
    ProviderMessageRole,
)
from music_agent.prompts.conversation_interpreter import INTERPRETER_SYSTEM_PROMPT

logger = logging.getLogger("music_agent.turn_interpreter")





class TurnInterpreterValidationError(ValueError):
    """Structured interpreter output violated the P22 semantic contract."""


@dataclass(frozen=True, slots=True)
class TurnInterpreterContext:
    """Minimal high-level context exposed to the semantic translator.

    ``None`` means the fact was unavailable, not false.  No durable identity or route
    is carried here by construction.
    """

    has_current_playback: bool | None = None
    has_active_recommendation: bool | None = None
    active_recommendation_item_count: int | None = None
    has_referenced_item: bool | None = None
    preview_active: bool | None = None

    def to_prompt_payload(self) -> dict[str, bool | int | None]:
        return {
            "has_current_playback": self.has_current_playback,
            "has_active_recommendation": self.has_active_recommendation,
            "active_recommendation_item_count": self.active_recommendation_item_count,
            "has_referenced_item": self.has_referenced_item,
            "preview_active": self.preview_active,
        }


@dataclass(frozen=True, slots=True)
class TurnInterpreterResult:
    """One observable interpreter attempt.

    ``raw_output`` is transient diagnostic material only.  It never becomes business
    authority or durable state.
    """

    status: str
    plan: TurnPlan
    raw_output: str | None = None
    error_code: str | None = None


def interpret_turn(
    provider: ChatProvider,
    user_text: str,
    deterministic_plan: TurnPlan,
    minimal_context: TurnInterpreterContext | None = None,
) -> TurnInterpreterResult:
    """Interpret one unresolved turn through the existing provider abstraction.

    The caller must run the deterministic parser first.  Resolved deterministic plans
    bypass the model entirely; calling this function with one is a contract error.
    """
    if deterministic_plan.primary is not TurnPrimarySemantic.UNKNOWN:
        raise TurnInterpreterValidationError(
            "interpret_turn requires a deterministic UNKNOWN plan"
        )
    if deterministic_plan.user_text != user_text:
        raise TurnInterpreterValidationError(
            "deterministic_plan must describe the same user_text"
        )
    context = minimal_context or TurnInterpreterContext()
    prompt_payload = {
        "user_text": user_text,
        "context": context.to_prompt_payload(),
    }
    try:
        response = provider.chat(
            INTERPRETER_SYSTEM_PROMPT,
            (
                ProviderMessage(
                    ProviderMessageRole.USER,
                    text=json.dumps(
                        prompt_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ),
            (),
        )
    except Exception as error:  # Provider boundary: exceptions become fail-closed semantics.
        logger.warning("turn interpreter provider failure: %s", type(error).__name__)
        return TurnInterpreterResult(
            status="failed",
            plan=_clarification_plan(
                deterministic_plan, reason="interpreter_provider_failure"
            ),
            error_code="provider_error",
        )

    message = response.message
    if message.tool_calls is not None or message.text is None:
        return TurnInterpreterResult(
            status="failed",
            plan=_clarification_plan(
                deterministic_plan, reason="interpreter_non_text_response"
            ),
            raw_output=message.text,
            error_code="invalid_response_shape",
        )
    raw_output = message.text.strip()
    try:
        payload = json.loads(raw_output)
        plan, status = _validated_plan_from_payload(
            payload, deterministic_plan=deterministic_plan, context=context
        )
    except (json.JSONDecodeError, TurnInterpreterValidationError, TypeError, ValueError):
        return TurnInterpreterResult(
            status="failed",
            plan=_clarification_plan(
                deterministic_plan, reason="interpreter_invalid_structured_output"
            ),
            raw_output=raw_output,
            error_code="invalid_structured_output",
        )
    return TurnInterpreterResult(status=status, plan=plan, raw_output=raw_output)


def _validated_plan_from_payload(
    payload: object,
    *,
    deterministic_plan: TurnPlan,
    context: TurnInterpreterContext,
) -> tuple[TurnPlan, str]:
    root = _require_object(payload, "root")
    _require_exact_keys(
        root,
        {"intent", "recommendation", "action", "requires_clarification", "reason"},
        "root",
    )
    intent = root["intent"]
    if intent not in {
        "recommendation",
        "delegated_selection",
        "clarification",
        "unsupported",
    }:
        raise TurnInterpreterValidationError("invalid intent enum")
    requires_clarification = root["requires_clarification"]
    if not isinstance(requires_clarification, bool):
        raise TurnInterpreterValidationError("requires_clarification must be boolean")
    reason = _optional_text(root["reason"], "reason", limit=300)
    recommendation = root["recommendation"]
    action = root["action"]

    if intent == "clarification":
        if not requires_clarification or recommendation is not None or action is not None:
            raise TurnInterpreterValidationError("contradictory clarification semantics")
        return _clarification_plan(
            deterministic_plan, reason=reason or "interpreter_requested_clarification"
        ), "clarification"

    if requires_clarification:
        raise TurnInterpreterValidationError(
            "non-clarification intent cannot require clarification"
        )

    if intent == "unsupported":
        if recommendation is not None or action is not None:
            raise TurnInterpreterValidationError("unsupported intent must carry no action")
        return replace(
            deterministic_plan,
            semantic_source=TurnSemanticSource.UNRESOLVED,
            requires_clarification=False,
            clarification_reason=reason,
        ), "unsupported"

    if intent == "recommendation":
        if action is not None:
            raise TurnInterpreterValidationError("recommendation cannot carry action")
        semantics = _validate_recommendation(recommendation)
        if (
            semantics.mode == "similarity_seed"
            and semantics.seed_source == "current_track"
            and context.has_current_playback is False
        ):
            return _clarification_plan(
                deterministic_plan, reason="no_similarity_referent"
            ), "clarification"
        return TurnPlan(
            user_text=deterministic_plan.user_text,
            primary=TurnPrimarySemantic.RECOMMENDATION,
            expected_result=TurnExpectedResult.RECOMMENDATION_BATCH,
            task_surface=TurnTaskSurface.RECOMMENDATION,
            recommendation=semantics,
            semantic_source=TurnSemanticSource.LLM_INTERPRETED,
        ), "interpreted"

    # delegated_selection
    if recommendation is not None:
        raise TurnInterpreterValidationError(
            "delegated selection cannot carry recommendation semantics"
        )
    _validate_delegated_action(action)
    if context.has_active_recommendation is False:
        return _clarification_plan(
            deterministic_plan, reason="no_active_recommendation"
        ), "clarification"
    return TurnPlan(
        user_text=deterministic_plan.user_text,
        primary=TurnPrimarySemantic.PLAYBACK_ACTION,
        expected_result=TurnExpectedResult.ACTION_RESULT,
        task_surface=TurnTaskSurface.PLAYBACK,
        playback_action=PlaybackActionTurnSemantics(
            kind="play_or_preview",
            source="active_recommendation",
            selection_mode="agent_choose_one",
            delegated=True,
        ),
        semantic_source=TurnSemanticSource.LLM_INTERPRETED,
    ), "interpreted"


def _validate_recommendation(value: object) -> RecommendationTurnSemantics:
    obj = _require_object(value, "recommendation")
    _require_exact_keys(obj, {"mode", "requested_count", "scene", "seed"}, "recommendation")
    mode = obj["mode"]
    if mode not in {"generic", "similarity"}:
        raise TurnInterpreterValidationError("invalid recommendation mode")
    requested_count = obj["requested_count"]
    if requested_count is not None:
        if isinstance(requested_count, bool) or not isinstance(requested_count, int):
            raise TurnInterpreterValidationError("requested_count must be integer or null")
        if requested_count < 1 or requested_count > 20:
            raise TurnInterpreterValidationError("requested_count out of range")
    requested_count = requested_count or 5
    scene = _optional_text(obj["scene"], "scene", limit=200)
    seed = obj["seed"]

    if mode == "generic":
        if seed is not None:
            raise TurnInterpreterValidationError("generic recommendation cannot carry seed")
        return RecommendationTurnSemantics(
            mode="generic",
            target=None,
            target_kind=None,
            scene=scene,
            requested_count=requested_count,
            seed_source=None,
        )

    seed_obj = _require_object(seed, "seed")
    _require_exact_keys(seed_obj, {"kind", "value"}, "seed")
    kind = seed_obj["kind"]
    if kind == "current_track":
        if seed_obj["value"] is not None:
            raise TurnInterpreterValidationError("current_track seed value must be null")
        return RecommendationTurnSemantics(
            mode="similarity_seed",
            target=None,
            target_kind="track",
            scene=scene,
            requested_count=requested_count,
            seed_source="current_track",
        )
    if kind != "free_text":
        raise TurnInterpreterValidationError("invalid seed kind")
    target = _optional_text(seed_obj["value"], "seed.value", limit=300)
    if target is None:
        raise TurnInterpreterValidationError("free_text similarity seed requires value")
    return RecommendationTurnSemantics(
        mode="similarity_seed",
        target=target,
        target_kind=None,
        scene=scene,
        requested_count=requested_count,
        seed_source=None,
    )


def _validate_delegated_action(value: object) -> None:
    obj = _require_object(value, "action")
    _require_exact_keys(obj, {"source", "selection_mode"}, "action")
    if obj["source"] != "active_recommendation":
        raise TurnInterpreterValidationError("invalid delegated action source")
    if obj["selection_mode"] != "agent_choose_one":
        raise TurnInterpreterValidationError("invalid delegated selection mode")


def _clarification_plan(plan: TurnPlan, *, reason: str) -> TurnPlan:
    return TurnPlan(
        user_text=plan.user_text,
        primary=TurnPrimarySemantic.UNKNOWN,
        expected_result=TurnExpectedResult.CLARIFICATION,
        task_surface=TurnTaskSurface.FULL,
        semantic_source=TurnSemanticSource.UNRESOLVED,
        requires_clarification=True,
        clarification_reason=reason,
    )


def _require_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TurnInterpreterValidationError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TurnInterpreterValidationError(f"{label} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise TurnInterpreterValidationError(
            f"{label} keys must equal {sorted(keys)!r}; got {sorted(actual)!r}"
        )


def _optional_text(value: object, label: str, *, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TurnInterpreterValidationError(f"{label} must be string or null")
    text = value.strip()
    if not text or len(text) > limit:
        raise TurnInterpreterValidationError(f"{label} is empty or too long")
    return text
