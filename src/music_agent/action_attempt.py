"""In-memory execution truth for one code-owned delegated audio action."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Callable
from typing import Mapping

from music_agent.selection_grant import DelegatedAudioAction


class ActionAttemptStatus(str, Enum):
    REQUESTED = "requested"
    EXECUTING = "executing"
    AWAITING_READBACK = "awaiting_readback"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionExecutionResult(str, Enum):
    NOT_STARTED = "not_started"
    OK = "ok"
    ERROR = "error"


class ActionReadbackState(str, Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class ActionAttempt:
    """One selected action's request, execution and verification state."""

    recommendation_run_id: str | None
    selected_canonical_id: str
    expected_canonical_id: str
    expected_route: str
    expected_action: str
    selected_title: str | None
    selected_artist: str | None
    item_position: int | None = None
    status: ActionAttemptStatus = ActionAttemptStatus.REQUESTED
    execution_result: ActionExecutionResult = ActionExecutionResult.NOT_STARTED
    readback_state: ActionReadbackState = ActionReadbackState.NOT_STARTED
    observed_playback_state: str | None = None
    agent_channel_canonical_id: str | None = None
    actual_player_canonical_id: str | None = None
    canonical_resolution: str | None = None
    observed_title: str | None = None
    observed_artist: str | None = None
    failure_reason: str | None = None
    execution_error_code: str | None = None
    execution_error_message: str | None = None


class ActionAttemptTransitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlaybackControlAttempt:
    """One non-targeted play/pause command's observed terminal state."""

    command: str
    expected_playback_state: str
    status: ActionAttemptStatus = ActionAttemptStatus.REQUESTED
    execution_result: ActionExecutionResult = ActionExecutionResult.NOT_STARTED
    readback_state: ActionReadbackState = ActionReadbackState.NOT_STARTED
    observed_playback_state: str | None = None
    failure_reason: str | None = None
    execution_error_code: str | None = None
    execution_error_message: str | None = None


def create_action_attempt(action: DelegatedAudioAction) -> ActionAttempt:
    """Create the requested state from a code-authorized action."""

    return ActionAttempt(
        recommendation_run_id=action.recommendation_run_id,
        selected_canonical_id=action.canonical_id,
        expected_canonical_id=action.canonical_id,
        expected_route=action.playback_route,
        expected_action=action.tool_name,
        selected_title=action.title,
        selected_artist=action.artist,
        item_position=action.item_position,
    )


def create_direct_action_attempt(
    canonical_id: str,
    *,
    route: str,
    title: str | None = None,
    artist: str | None = None,
) -> ActionAttempt:
    """Create the same terminal-truth object for a non-batch track action.

    Direct references and Web card clicks have an authoritative canonical id
    but no RecommendationRun selection position.  They therefore leave the
    batch fields empty instead of fabricating a run identity.  Route still
    determines the only permissible tool; execution and success remain unset.
    """

    tool_name = {
        "library": "play_track",
        "preview_only": "preview_catalog_track",
    }.get(route)
    if not isinstance(canonical_id, str) or not canonical_id or tool_name is None:
        raise ValueError("direct action requires a canonical id and supported route")
    return ActionAttempt(
        recommendation_run_id=None,
        item_position=None,
        selected_canonical_id=canonical_id,
        expected_canonical_id=canonical_id,
        expected_route=route,
        expected_action=tool_name,
        selected_title=title,
        selected_artist=artist,
    )


def mark_action_executing(attempt: ActionAttempt) -> ActionAttempt:
    if attempt.status is not ActionAttemptStatus.REQUESTED:
        raise ActionAttemptTransitionError("only requested actions can start executing")
    return replace(attempt, status=ActionAttemptStatus.EXECUTING)


def record_action_execution(
    attempt: ActionAttempt,
    *,
    outcome: str,
    preview_started: bool | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ActionAttempt:
    """Record tool truth and either terminate or await formal-play readback."""

    if attempt.status is not ActionAttemptStatus.EXECUTING:
        raise ActionAttemptTransitionError("execution requires an executing attempt")
    if outcome != "ok":
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            execution_result=ActionExecutionResult.ERROR,
            failure_reason="execution_error",
            execution_error_code=error_code,
            execution_error_message=error_message,
        )
    if attempt.expected_route == "preview_only":
        if preview_started is True:
            return replace(
                attempt,
                status=ActionAttemptStatus.COMPLETED,
                execution_result=ActionExecutionResult.OK,
                readback_state=ActionReadbackState.NOT_REQUIRED,
            )
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            execution_result=ActionExecutionResult.OK,
            readback_state=ActionReadbackState.NOT_REQUIRED,
            failure_reason="preview_not_started",
        )
    if attempt.expected_route == "library":
        return replace(
            attempt,
            status=ActionAttemptStatus.AWAITING_READBACK,
            execution_result=ActionExecutionResult.OK,
            readback_state=ActionReadbackState.PENDING,
        )
    return replace(
        attempt,
        status=ActionAttemptStatus.FAILED,
        execution_result=ActionExecutionResult.OK,
        failure_reason="unsupported_route",
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def verify_formal_play_readback(
    attempt: ActionAttempt, payload: Mapping | None
) -> ActionAttempt:
    """Apply the strict formal-play equality gate to actual player readback."""

    if (
        attempt.status is not ActionAttemptStatus.AWAITING_READBACK
        or attempt.expected_route != "library"
    ):
        raise ActionAttemptTransitionError(
            "formal readback requires an awaiting library attempt"
        )
    now_playing = payload.get("now_playing") if isinstance(payload, Mapping) else None
    agent_channel = payload.get("agent_channel") if isinstance(payload, Mapping) else None
    observed_state = (
        now_playing.get("state") if isinstance(now_playing, Mapping) else None
    )
    agent_id = (
        agent_channel.get("canonical_id")
        if isinstance(agent_channel, Mapping)
        else None
    )
    agent_state = (
        agent_channel.get("state") if isinstance(agent_channel, Mapping) else None
    )
    player_id = payload.get("player_canonical_id") if isinstance(payload, Mapping) else None
    resolution = payload.get("canonical_resolution") if isinstance(payload, Mapping) else None
    common = {
        "observed_playback_state": _optional_text(observed_state),
        "agent_channel_canonical_id": _optional_text(agent_id),
        "actual_player_canonical_id": _optional_text(player_id),
        "canonical_resolution": _optional_text(resolution),
        "observed_title": (
            _optional_text(now_playing.get("name"))
            if isinstance(now_playing, Mapping)
            else None
        ),
        "observed_artist": (
            _optional_text(now_playing.get("artist"))
            if isinstance(now_playing, Mapping)
            else None
        ),
    }
    if not isinstance(now_playing, Mapping):
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.UNRESOLVED,
            failure_reason="readback_unavailable",
            **common,
        )
    if observed_state != "playing":
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.MISMATCH,
            failure_reason="player_not_playing",
            **common,
        )
    if not isinstance(player_id, str) or not player_id or not isinstance(resolution, str) or not resolution:
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.UNRESOLVED,
            failure_reason="player_canonical_unresolved",
            **common,
        )
    if (
        attempt.selected_canonical_id != attempt.expected_canonical_id
        or agent_state != "library"
        or agent_id != attempt.expected_canonical_id
    ):
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.MISMATCH,
            failure_reason="agent_channel_mismatch",
            **common,
        )
    if player_id != attempt.expected_canonical_id:
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.MISMATCH,
            failure_reason="player_canonical_mismatch",
            **common,
        )
    return replace(
        attempt,
        status=ActionAttemptStatus.COMPLETED,
        readback_state=ActionReadbackState.VERIFIED,
        failure_reason=None,
        **common,
    )


def _tool_result_parts(result: object) -> tuple[str, Mapping | None]:
    outcome = getattr(result, "outcome", None)
    outcome = getattr(outcome, "value", outcome)
    payload = getattr(result, "payload", None)
    return (
        outcome if isinstance(outcome, str) else "execution_error",
        payload if isinstance(payload, Mapping) else None,
    )


def run_action_attempt(
    attempt: ActionAttempt,
    invoke: Callable[[str, Mapping[str, object]], object],
) -> ActionAttempt:
    """Execute one track action and return its sole structured terminal truth.

    ``invoke`` is the existing AgentClient/Provider execution boundary.  This
    helper owns only the action lifecycle for the current turn: one action
    dispatch, and for formal play exactly one ``get_now_playing`` readback.
    Music.app / Preview Runtime remain the external authorities represented by
    those tool results.
    """

    executing = mark_action_executing(attempt)
    try:
        result = invoke(
            executing.expected_action,
            {"canonical_id": executing.expected_canonical_id},
        )
    except Exception:
        return record_action_execution(executing, outcome="execution_error")
    outcome, payload = _tool_result_parts(result)
    terminal = record_action_execution(
        executing,
        outcome=outcome,
        preview_started=(
            payload.get("started")
            if isinstance(payload, Mapping)
            and isinstance(payload.get("started"), bool)
            else None
        ),
        error_code=(
            getattr(result, "error_code", None)
            if isinstance(getattr(result, "error_code", None), str)
            else None
        ),
        error_message=(
            getattr(result, "error_message", None)
            if isinstance(getattr(result, "error_message", None), str)
            else None
        ),
    )
    if terminal.status is not ActionAttemptStatus.AWAITING_READBACK:
        return terminal
    try:
        readback = invoke("get_now_playing", {})
    except Exception:
        return verify_formal_play_readback(terminal, None)
    readback_outcome, readback_payload = _tool_result_parts(readback)
    return verify_formal_play_readback(
        terminal,
        readback_payload if readback_outcome == "ok" else None,
    )


def run_playback_control_attempt(
    command: str,
    *,
    expected_state: str,
    invoke: Callable[[str, Mapping[str, object]], object],
) -> PlaybackControlAttempt:
    """Execute play/pause once and verify its terminal player state by readback."""

    attempt = create_playback_control_attempt(command, expected_state=expected_state)
    try:
        executed = invoke(command, {})
    except Exception:
        return record_playback_control_execution(
            attempt, outcome="execution_error"
        )
    outcome, _payload = _tool_result_parts(executed)
    attempt = record_playback_control_execution(
        attempt,
        outcome=outcome,
        error_code=(
            getattr(executed, "error_code", None)
            if isinstance(getattr(executed, "error_code", None), str)
            else None
        ),
        error_message=(
            getattr(executed, "error_message", None)
            if isinstance(getattr(executed, "error_message", None), str)
            else None
        ),
    )
    if attempt.status is not ActionAttemptStatus.AWAITING_READBACK:
        return attempt
    try:
        readback = invoke("get_now_playing", {})
    except Exception:
        return verify_playback_control_readback(attempt, None)
    readback_outcome, payload = _tool_result_parts(readback)
    return verify_playback_control_readback(
        attempt, payload if readback_outcome == "ok" else None
    )


def create_playback_control_attempt(
    command: str, *, expected_state: str
) -> PlaybackControlAttempt:
    if (command, expected_state) not in {
        ("play", "playing"),
        ("pause", "paused"),
    }:
        raise ValueError("state-verified control must be play/playing or pause/paused")
    return PlaybackControlAttempt(
        command=command,
        expected_playback_state=expected_state,
    )


def record_playback_control_execution(
    attempt: PlaybackControlAttempt,
    *,
    outcome: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> PlaybackControlAttempt:
    if attempt.status is not ActionAttemptStatus.REQUESTED:
        raise ActionAttemptTransitionError(
            "control execution requires a requested attempt"
        )
    if outcome != "ok":
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            execution_result=ActionExecutionResult.ERROR,
            failure_reason="execution_error",
            execution_error_code=error_code,
            execution_error_message=error_message,
        )
    return replace(
        attempt,
        status=ActionAttemptStatus.AWAITING_READBACK,
        execution_result=ActionExecutionResult.OK,
        readback_state=ActionReadbackState.PENDING,
    )


def verify_playback_control_readback(
    attempt: PlaybackControlAttempt, payload: Mapping | None
) -> PlaybackControlAttempt:
    if attempt.status is not ActionAttemptStatus.AWAITING_READBACK:
        raise ActionAttemptTransitionError(
            "control readback requires an awaiting attempt"
        )
    now_playing = payload.get("now_playing") if isinstance(payload, Mapping) else None
    observed = (
        now_playing.get("state") if isinstance(now_playing, Mapping) else None
    )
    if not isinstance(observed, str) or not observed:
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            readback_state=ActionReadbackState.UNRESOLVED,
            failure_reason="readback_unavailable",
        )
    if observed != attempt.expected_playback_state:
        return replace(
            attempt,
            status=ActionAttemptStatus.FAILED,
            execution_result=ActionExecutionResult.OK,
            readback_state=ActionReadbackState.MISMATCH,
            observed_playback_state=observed,
            failure_reason="playback_state_mismatch",
        )
    return replace(
        attempt,
        status=ActionAttemptStatus.COMPLETED,
        execution_result=ActionExecutionResult.OK,
        readback_state=ActionReadbackState.VERIFIED,
        observed_playback_state=observed,
    )


def render_verified_playback_control_result(
    attempt: PlaybackControlAttempt,
) -> str:
    """Render play/pause only from the typed observed-state verdict."""

    if attempt.status is ActionAttemptStatus.COMPLETED:
        return "已暂停播放。" if attempt.command == "pause" else "已继续播放。"
    if attempt.status is not ActionAttemptStatus.FAILED:
        raise ActionAttemptTransitionError("only terminal attempts can be rendered")
    if attempt.command == "pause":
        return "暂停指令已发出，但暂时无法确认当前已暂停。"
    return "播放指令已发出，但暂时无法确认当前已继续播放。"


def render_verified_action_result(attempt: ActionAttempt) -> str:
    """Render a deterministic terminal result from ActionAttempt truth."""

    if attempt.status is ActionAttemptStatus.COMPLETED:
        if attempt.expected_route == "preview_only":
            if attempt.selected_title and attempt.selected_artist:
                return (
                    f"正在试听《{attempt.selected_title}》— "
                    f"{attempt.selected_artist}，约 30 秒。"
                )
            if attempt.selected_title:
                return f"正在试听《{attempt.selected_title}》，约 30 秒。"
            return "已开始试听，约 30 秒。"
        if attempt.observed_title and attempt.observed_artist:
            return f"正在播放《{attempt.observed_title}》— {attempt.observed_artist}。"
        if attempt.observed_title:
            return f"正在播放《{attempt.observed_title}》。"
        return "已开始播放。"
    if attempt.status is not ActionAttemptStatus.FAILED:
        raise ActionAttemptTransitionError("only terminal attempts can be rendered")
    if attempt.expected_route == "preview_only":
        return "这首暂时无法试听。"
    if attempt.failure_reason == "execution_error":
        return "这首暂时无法正式播放。"
    return "播放指令已发出，但暂时无法确认当前播放状态。"
