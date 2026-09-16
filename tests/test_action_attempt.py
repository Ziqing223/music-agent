from __future__ import annotations

import unittest
from types import SimpleNamespace

from music_agent.action_attempt import (
    ActionAttemptStatus,
    ActionReadbackState,
    create_action_attempt,
    create_direct_action_attempt,
    mark_action_executing,
    record_action_execution,
    render_verified_action_result,
    run_action_attempt,
    run_playback_control_attempt,
    verify_formal_play_readback,
)
from music_agent.selection_grant import DelegatedAudioAction


RUN_ID = "rcm_11111111-1111-4111-8111-111111111111"
EXPECTED_ID = "trk_11111111-1111-4111-8111-111111111111"
OTHER_ID = "trk_22222222-2222-4222-8222-222222222222"


def _action(
    *,
    route: str = "library",
    tool_name: str = "play_track",
) -> DelegatedAudioAction:
    return DelegatedAudioAction(
        recommendation_run_id=RUN_ID,
        item_position=1,
        canonical_id=EXPECTED_ID,
        playback_route=route,
        tool_name=tool_name,
        title="Synthetic Duet",
        artist="Artist Alpha",
    )


def _readback(
    *,
    state: str = "playing",
    agent_id: str | None = EXPECTED_ID,
    player_id: str | None = EXPECTED_ID,
    resolution: str | None = "binding",
) -> dict:
    return {
        "now_playing": {
            "state": state,
            "persistent_id": "SYNTH-TRACK-001",
            "name": "Synthetic Duet",
            "artist": "Artist Alpha",
            "album": "Synthetic Album",
        },
        "agent_channel": {
            "state": "library",
            "canonical_id": agent_id,
        },
        "player_canonical_id": player_id,
        "canonical_resolution": resolution,
    }


def _awaiting_formal_attempt():
    attempt = mark_action_executing(create_action_attempt(_action()))
    return record_action_execution(attempt, outcome="ok")


class ActionAttemptContractTest(unittest.TestCase):
    def test_formal_lifecycle_completes_only_on_three_way_canonical_agreement(self) -> None:
        requested = create_action_attempt(_action())
        executing = mark_action_executing(requested)
        awaiting = record_action_execution(executing, outcome="ok")
        completed = verify_formal_play_readback(awaiting, _readback())

        self.assertEqual(requested.status, ActionAttemptStatus.REQUESTED)
        self.assertEqual(executing.status, ActionAttemptStatus.EXECUTING)
        self.assertEqual(awaiting.status, ActionAttemptStatus.AWAITING_READBACK)
        self.assertEqual(completed.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(completed.readback_state, ActionReadbackState.VERIFIED)
        self.assertEqual(completed.expected_canonical_id, EXPECTED_ID)
        self.assertEqual(completed.agent_channel_canonical_id, EXPECTED_ID)
        self.assertEqual(completed.actual_player_canonical_id, EXPECTED_ID)
        self.assertEqual(
            render_verified_action_result(completed),
            "正在播放《Synthetic Duet》— Artist Alpha。",
        )

    def test_expected_not_equal_agent_channel_fails_closed(self) -> None:
        failed = verify_formal_play_readback(
            _awaiting_formal_attempt(),
            _readback(agent_id=OTHER_ID),
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.readback_state, ActionReadbackState.MISMATCH)
        self.assertEqual(failed.failure_reason, "agent_channel_mismatch")

    def test_expected_not_equal_actual_player_fails_closed(self) -> None:
        failed = verify_formal_play_readback(
            _awaiting_formal_attempt(),
            _readback(player_id=OTHER_ID),
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.readback_state, ActionReadbackState.MISMATCH)
        self.assertEqual(failed.failure_reason, "player_canonical_mismatch")

    def test_actual_player_canonical_unresolved_fails_closed(self) -> None:
        failed = verify_formal_play_readback(
            _awaiting_formal_attempt(),
            _readback(player_id=None, resolution=None),
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.readback_state, ActionReadbackState.UNRESOLVED)
        self.assertEqual(failed.failure_reason, "player_canonical_unresolved")

    def test_execution_ok_does_not_override_actual_player_mismatch(self) -> None:
        awaiting = _awaiting_formal_attempt()
        self.assertEqual(awaiting.status, ActionAttemptStatus.AWAITING_READBACK)

        failed = verify_formal_play_readback(
            awaiting,
            _readback(player_id=OTHER_ID),
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)

    def test_not_playing_fails_closed(self) -> None:
        failed = verify_formal_play_readback(
            _awaiting_formal_attempt(),
            _readback(state="paused"),
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.failure_reason, "player_not_playing")

    def test_unavailable_readback_fails_with_structured_reason(self) -> None:
        failed = verify_formal_play_readback(_awaiting_formal_attempt(), None)

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.readback_state, ActionReadbackState.UNRESOLVED)
        self.assertEqual(failed.failure_reason, "readback_unavailable")

    def test_preview_ok_and_started_true_completes(self) -> None:
        attempt = mark_action_executing(
            create_action_attempt(
                _action(route="preview_only", tool_name="preview_catalog_track")
            )
        )

        completed = record_action_execution(
            attempt,
            outcome="ok",
            preview_started=True,
        )

        self.assertEqual(completed.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(completed.readback_state, ActionReadbackState.NOT_REQUIRED)
        self.assertEqual(
            render_verified_action_result(completed),
            "正在试听《Synthetic Duet》— Artist Alpha，约 30 秒。",
        )

    def test_preview_ok_and_started_false_fails(self) -> None:
        attempt = mark_action_executing(
            create_action_attempt(
                _action(route="preview_only", tool_name="preview_catalog_track")
            )
        )

        failed = record_action_execution(
            attempt,
            outcome="ok",
            preview_started=False,
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.failure_reason, "preview_not_started")

    def test_preview_execution_error_fails(self) -> None:
        attempt = mark_action_executing(
            create_action_attempt(
                _action(route="preview_only", tool_name="preview_catalog_track")
            )
        )

        failed = record_action_execution(attempt, outcome="execution_error")

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.failure_reason, "execution_error")

    def test_direct_action_has_no_fabricated_recommendation_identity(self) -> None:
        attempt = create_direct_action_attempt(EXPECTED_ID, route="library")

        self.assertIsNone(attempt.recommendation_run_id)
        self.assertIsNone(attempt.item_position)
        self.assertEqual(attempt.expected_action, "play_track")
        self.assertEqual(attempt.expected_canonical_id, EXPECTED_ID)

    def test_shared_runner_formal_play_executes_once_then_strict_readback(self) -> None:
        calls: list[tuple[str, dict]] = []

        def invoke(tool: str, payload: dict):
            calls.append((tool, payload))
            return SimpleNamespace(
                outcome="ok",
                payload=(
                    {"command": "play_track", "ok": True}
                    if tool == "play_track"
                    else _readback()
                ),
            )

        completed = run_action_attempt(
            create_direct_action_attempt(EXPECTED_ID, route="library"),
            invoke,
        )

        self.assertEqual(completed.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(
            calls,
            [
                ("play_track", {"canonical_id": EXPECTED_ID}),
                ("get_now_playing", {}),
            ],
        )

    def test_shared_runner_never_upgrades_tool_ok_after_player_mismatch(self) -> None:
        def invoke(tool: str, payload: dict):
            return SimpleNamespace(
                outcome="ok",
                payload=(
                    {"command": "play_track", "ok": True}
                    if tool == "play_track"
                    else _readback(player_id=OTHER_ID)
                ),
            )

        failed = run_action_attempt(
            create_direct_action_attempt(EXPECTED_ID, route="library"),
            invoke,
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.failure_reason, "player_canonical_mismatch")

    def test_shared_runner_preview_requires_started_true(self) -> None:
        calls: list[tuple[str, dict]] = []

        def invoke(tool: str, payload: dict):
            calls.append((tool, payload))
            return SimpleNamespace(outcome="ok", payload={"started": False})

        failed = run_action_attempt(
            create_direct_action_attempt(EXPECTED_ID, route="preview_only"),
            invoke,
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.failure_reason, "preview_not_started")
        self.assertEqual(
            calls,
            [("preview_catalog_track", {"canonical_id": EXPECTED_ID})],
        )

    def test_pause_completes_only_after_observed_paused_state(self) -> None:
        calls: list[tuple[str, dict]] = []

        def invoke(tool: str, payload: dict):
            calls.append((tool, payload))
            return SimpleNamespace(
                outcome="ok",
                payload=(
                    {"command": "pause", "ok": True}
                    if tool == "pause"
                    else {"now_playing": {"state": "paused"}}
                ),
            )

        completed = run_playback_control_attempt(
            "pause", expected_state="paused", invoke=invoke
        )

        self.assertEqual(completed.status, ActionAttemptStatus.COMPLETED)
        self.assertEqual(completed.readback_state, ActionReadbackState.VERIFIED)
        self.assertEqual(calls, [("pause", {}), ("get_now_playing", {})])

    def test_pause_tool_ok_with_playing_readback_fails_closed(self) -> None:
        def invoke(tool: str, payload: dict):
            return SimpleNamespace(
                outcome="ok",
                payload=(
                    {"command": "pause", "ok": True}
                    if tool == "pause"
                    else {"now_playing": {"state": "playing"}}
                ),
            )

        failed = run_playback_control_attempt(
            "pause", expected_state="paused", invoke=invoke
        )

        self.assertEqual(failed.status, ActionAttemptStatus.FAILED)
        self.assertEqual(failed.readback_state, ActionReadbackState.MISMATCH)
        self.assertEqual(failed.failure_reason, "playback_state_mismatch")


if __name__ == "__main__":
    unittest.main()
