"""P15-S2 round 2: transition classifier + safety-pause policy tests.

Track C pins the V1 classification -- exactly one safety transition shape
(known-non-builtin default output -> BuiltInSpeakerDevice uid), decided on
stable uid / raw device facts only, with names and transports never consulted
and unknown facts failing closed. Track D pins the deterministic policy --
no LLM, at most one pause effect per output transition, idempotent once
paused, never resuming (reconnect is silent; restore stays the user's
继续播放 intent through the existing suspension record). The integration
class exercises the agent-service entry through the P15-S1 unified control
semantics (``_suspend_music_for_preview`` / preview-session cancel) with the
same deterministic fakes the playback tests use. These tests are written
against the round-2 boundary: the service entry is NOT fed by any event
source yet, and the P10.5 polling monitor is untouched.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.device_context import AudioOutputEventType, AudioOutputSnapshot
from music_agent.preview_session import PreviewSessionItem
from music_agent.safety_pause import (
    BUILTIN_SPEAKER_DEVICE_UID,
    AudioTransitionClassification,
    SafetyPauseAction,
    SafetyPausePolicy,
    classify_output_fallback,
)
from test_playback_tools import FakePlaybackAdapter, FakePreviewRunner

CLIENT_ID = "agt_99999999-9999-4999-8999-999999999999"

AIRPODS_UID = "A0:B1:C2:D3:E4:F5"
WIRED_UID = "BuiltInHeadphoneOutputDevice"


def _snapshot(
    *,
    uid: str | None,
    event_type: str = "default_device_changed",
    transport_type: str | None = None,
    device_name: str | None = None,
    **facts: object,
) -> AudioOutputSnapshot:
    return AudioOutputSnapshot(
        event_type=event_type,
        timestamp="2026-08-19T09:00:00.000Z",
        device_uid=uid,
        transport_type=transport_type,
        device_name=device_name,
        **facts,
    )


class OutputFallbackClassifierTest(unittest.TestCase):
    """Track C: exactly one safety shape; uid facts only; fail closed elsewhere."""

    def test_airpods_like_external_to_builtin_speaker_classifies_safety(self) -> None:
        classification = classify_output_fallback(
            _snapshot(uid=AIRPODS_UID, transport_type="blue", device_name="AirPods Pro"),
            _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
        )
        self.assertIs(
            classification,
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )

    def test_wired_headphone_to_builtin_speaker_classifies_safety(self) -> None:
        classification = classify_output_fallback(
            _snapshot(uid=WIRED_UID, transport_type="hdpn"),
            _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
        )
        self.assertIs(
            classification,
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )

    def test_builtin_to_airpods_is_not_a_safety_fallback(self) -> None:
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
                _snapshot(uid=AIRPODS_UID),
            )
        )

    def test_builtin_to_wired_is_not_a_safety_fallback(self) -> None:
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
                _snapshot(uid=WIRED_UID),
            )
        )

    def test_external_to_external_is_not_a_v1_safety_fallback(self) -> None:
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=AIRPODS_UID),
                _snapshot(uid=WIRED_UID),
            )
        )

    def test_same_device_never_triggers(self) -> None:
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=WIRED_UID),
                _snapshot(uid=WIRED_UID),
            )
        )
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
            )
        )

    def test_missing_previous_uid_fails_closed(self) -> None:
        """A transition whose 'before' cannot be identified is never a fallback."""
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=None),
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
            )
        )

    def test_missing_current_uid_fails_closed(self) -> None:
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=AIRPODS_UID),
                _snapshot(uid=None),
            )
        )

    def test_device_list_and_alive_facts_never_trigger(self) -> None:
        """Fact-update events (device_list/alive) are diagnostics, not decisions:
        the classifier only accepts snapshots the default-output observation
        produced (baseline / default_device_changed)."""
        for event_type in ("device_list_changed", "alive_changed"):
            self.assertIsNone(
                classify_output_fallback(
                    _snapshot(uid=AIRPODS_UID),
                    _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID, event_type=event_type),
                ),
                event_type,
            )
            self.assertIsNone(
                classify_output_fallback(
                    _snapshot(uid=AIRPODS_UID, event_type=event_type),
                    _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
                ),
                event_type,
            )

    def test_startup_baseline_is_an_acceptable_before_side(self) -> None:
        """First change after startup: before carries the baseline facts."""
        self.assertIs(
            classify_output_fallback(
                _snapshot(uid=AIRPODS_UID, event_type="baseline"),
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
            ),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )

    def test_names_and_transports_never_decide(self) -> None:
        """Names/transports are raw facts: a speaker-shaped current side with a
        non-speaker uid is NOT a fallback, and a bluetooth-shaped before side
        whose uid differs from the speaker still IS. The uid alone decides."""
        self.assertIsNone(
            classify_output_fallback(
                _snapshot(uid=AIRPODS_UID, transport_type="blue"),
                _snapshot(
                    uid="SomeVirtualUID",
                    transport_type="bltn",
                    device_name="BuiltInSpeakerDevice",
                ),
            )
        )
        self.assertIs(
            classify_output_fallback(
                _snapshot(
                    uid="SomeBluetoothUID",
                    transport_type="blue",
                    device_name="MacBook Pro扬声器",
                ),
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
            ),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )


class SafetyPausePolicyTest(unittest.TestCase):
    """Track D: deterministic decide + one-effect dedup; the policy never resumes."""

    def setUp(self) -> None:
        self.policy = SafetyPausePolicy()

    def _fallback(self) -> tuple[AudioOutputSnapshot, AudioOutputSnapshot]:
        return (
            _snapshot(uid=AIRPODS_UID),
            _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
        )

    def test_duplicate_notification_yields_one_effect(self) -> None:
        """Repeated Core Audio notifications of the same transition classify once."""
        before, after = self._fallback()
        self.assertIs(
            self.policy.on_default_output_change(before, after),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )
        self.assertIsNone(self.policy.on_default_output_change(before, after))

    def test_fallback_rearms_after_output_leaves_the_speaker(self) -> None:
        """A later fallback (reconnect then drop) is a new hazard: it acts again."""
        before, after = self._fallback()
        self.assertIs(
            self.policy.on_default_output_change(before, after),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )
        self.assertIsNone(
            self.policy.on_default_output_change(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID), _snapshot(uid=AIRPODS_UID)
            )
        )
        self.assertIs(
            self.policy.on_default_output_change(before, after),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )

    def test_unknown_after_never_rearms_but_a_new_external_fallback_still_acts(
        self,
    ) -> None:
        """Fail-safe direction: an unreadable current default never clamps a
        *distinct* later fallback; it only fails itself closed."""
        before, after = self._fallback()
        self.assertIs(
            self.policy.on_default_output_change(before, after),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )
        self.assertIsNone(
            self.policy.on_default_output_change(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID), _snapshot(uid=None)
            )
        )
        self.assertIs(
            self.policy.on_default_output_change(
                _snapshot(uid=WIRED_UID), _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID)
            ),
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
        )

    def test_decide_without_sounding_playback_is_a_noop(self) -> None:
        action = self.policy.decide(
            AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER,
            formal_sounding=False,
            preview_active=False,
        )
        self.assertIsNone(action)

    def test_decide_requests_pause_only_for_the_sounding_path(self) -> None:
        classification = AudioTransitionClassification.OUTPUT_FALLBACK_TO_BUILTIN_SPEAKER
        formal_only = self.policy.decide(
            classification, formal_sounding=True, preview_active=False
        )
        self.assertIsNotNone(formal_only)
        self.assertTrue(formal_only.pause_formal_playback)
        self.assertFalse(formal_only.pause_preview)

        preview_only = self.policy.decide(
            classification, formal_sounding=False, preview_active=True
        )
        self.assertIsNotNone(preview_only)
        self.assertFalse(preview_only.pause_formal_playback)
        self.assertTrue(preview_only.pause_preview)

        both = self.policy.decide(
            classification, formal_sounding=True, preview_active=True
        )
        self.assertIsNotNone(both)
        self.assertTrue(both.pause_formal_playback)
        self.assertTrue(both.pause_preview)

    def test_decide_fails_closed_on_no_classification(self) -> None:
        self.assertIsNone(
            self.policy.decide(None, formal_sounding=True, preview_active=True)
        )

    def test_policy_has_no_resume_surface(self) -> None:
        """reconnect never resumes: the policy only classifies and decides,
        and nothing it returns or holds can start audio."""
        self.assertFalse(hasattr(self.policy, "resume"))
        self.assertFalse(hasattr(self.policy, "restore"))


class SafetyPauseIntegrationTest(unittest.TestCase):
    """The agent-service entry: unified P15-S1 control semantics, no new state."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def _service(
        self, *, playing: bool = True, with_live_preview: bool = False
    ) -> tuple[SharedAgentService, FakePlaybackAdapter, FakePreviewRunner]:
        adapter = FakePlaybackAdapter(now_pid="REAL-PID-1" if playing else None)
        runner = FakePreviewRunner()
        if with_live_preview:
            runner.active = True
        service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
            playback_adapter=adapter,
            playback_resolver=None,
            preview_runner=runner,
            catalog_search_source=None,
        )
        self.addCleanup(service.close)
        if with_live_preview:
            service._playback_context.sessions.start(
                [
                    PreviewSessionItem(
                        canonical_id="trk_safety_test",
                        name="测试曲目",
                        artist_name=None,
                        route="preview_only",
                    )
                ]
            )
        self.events: list[dict] = []
        service.preview_event_handler = lambda event: self.events.append(event)
        return service, adapter, runner

    def _fallback(
        self, previous_uid: str = AIRPODS_UID
    ) -> tuple[AudioOutputSnapshot, AudioOutputSnapshot]:
        return (
            _snapshot(uid=previous_uid),
            _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
        )

    def test_formal_playback_is_safety_paused(self) -> None:
        service, adapter, runner = self._service(playing=True)
        action = service.handle_default_output_transition(*self._fallback())
        self.assertIsNotNone(action)
        self.assertTrue(action.pause_formal_playback)
        self.assertFalse(action.pause_preview)
        self.assertIn(("pause", ()), adapter.calls)
        self.assertNotIn(("play", ()), adapter.calls)
        self.assertEqual(runner.stop_calls, 0)
        # The unified suspend semantics record the interruption: the restored
        # 继续播放 message keeps an honest basis (restore-by-intent untouched).
        suspension = service._playback_context.suspension.value
        self.assertIsNotNone(suspension)
        self.assertTrue(suspension.pause_ok)
        self.assertEqual(suspension.name, "起风了 (旧版)")

    def test_active_preview_session_is_safety_paused(self) -> None:
        service, adapter, runner = self._service(playing=False, with_live_preview=True)
        action = service.handle_default_output_transition(*self._fallback())
        self.assertIsNotNone(action)
        self.assertFalse(action.pause_formal_playback)
        self.assertTrue(action.pause_preview)
        self.assertEqual(runner.stop_calls, 1)
        self.assertNotIn(("pause", ()), adapter.calls)
        # Unified stop semantics: the live session is cancelled and reported.
        self.assertIsNone(service._playback_context.sessions.current())
        self.assertEqual([event["event"] for event in self.events], ["cancelled"])
        self.assertEqual(service._playback_context.suspension.value, None)

    def test_duplicate_transition_causes_one_pause_effect(self) -> None:
        service, adapter, runner = self._service(playing=True)
        service.handle_default_output_transition(*self._fallback())
        second = service.handle_default_output_transition(*self._fallback())
        self.assertIsNone(second)
        self.assertEqual([name for name, _ in adapter.calls].count("pause"), 1)
        self.assertEqual(runner.stop_calls, 0)
        self.assertNotIn(("play", ()), adapter.calls)

    def test_already_paused_is_idempotent(self) -> None:
        """No sounding controlled path -> the entry is a no-op."""
        service, adapter, runner = self._service(playing=False)
        action = service.handle_default_output_transition(*self._fallback())
        self.assertIsNone(action)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(runner.stop_calls, 0)

    def test_reconnect_never_resumes(self) -> None:
        service, adapter, runner = self._service(playing=True)
        service.handle_default_output_transition(*self._fallback())
        # Music now reads paused (the safety pause took effect).
        adapter.now_pid = None
        self.assertIsNone(
            service.handle_default_output_transition(
                _snapshot(uid=BUILTIN_SPEAKER_DEVICE_UID),
                _snapshot(uid=AIRPODS_UID),
            )
        )
        self.assertIsNone(service.handle_default_output_transition(*self._fallback()))
        call_names = [name for name, _ in adapter.calls]
        self.assertNotIn("play", call_names)
        self.assertNotIn("play_track", call_names)
        # The recorded suspension survives: restore stays the user's intent.
        self.assertIsNotNone(service._playback_context.suspension.value)

    def test_no_controlled_playback_is_a_noop(self) -> None:
        service, adapter, runner = self._service(playing=False)
        action = service.handle_default_output_transition(*self._fallback())
        self.assertIsNone(action)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(runner.stop_calls, 0)


if __name__ == "__main__":
    unittest.main()