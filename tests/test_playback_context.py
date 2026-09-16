"""P15-PC C1: Playback Context unit tests (suspension register + coordinator holder)."""

import unittest

from music_agent.playback_context import (
    AudioSuspension,
    PlaybackContext,
    SuspendedPlaybackEntry,
)

ENTRY = SuspendedPlaybackEntry("playing", "REAL-PID-1", "起风了 (旧版)", True)


class AudioSuspensionTest(unittest.TestCase):
    def test_starts_empty(self) -> None:
        self.assertIsNone(AudioSuspension().value)

    def test_record_and_read(self) -> None:
        suspension = AudioSuspension()
        suspension.record(ENTRY)
        self.assertEqual(suspension.value, ENTRY)

    def test_record_overwrites_previous(self) -> None:
        suspension = AudioSuspension()
        suspension.record(ENTRY)
        fresh = SuspendedPlaybackEntry("playing", "REAL-PID-2", "晴天", True)
        suspension.record(fresh)
        self.assertEqual(suspension.value, fresh)

    def test_clear_empties(self) -> None:
        suspension = AudioSuspension()
        suspension.record(ENTRY)
        suspension.clear()
        self.assertIsNone(suspension.value)

    def test_record_rejects_foreign_types(self) -> None:
        with self.assertRaises(ValueError):
            AudioSuspension().record({"player_state": "playing"})  # type: ignore[arg-type]

    def test_pause_failed_entry_keeps_interruption_facts(self) -> None:
        entry = SuspendedPlaybackEntry("playing", "REAL-PID-1", None, False)
        self.assertFalse(entry.pause_ok)
        self.assertEqual(entry.player_state, "playing")
        self.assertEqual(entry.persistent_id, "REAL-PID-1")

    # P17 acceptance: the natural-end restore arm (armed per preview start,
    # consumed one-shot on the natural end, disarmed by stop/failure/user).

    def test_restore_arm_starts_empty(self) -> None:
        suspension = AudioSuspension()
        self.assertIsNone(suspension.pending_restore)
        self.assertIsNone(suspension.take_restore_arm())

    def test_restore_arm_take_is_one_shot(self) -> None:
        suspension = AudioSuspension()
        suspension.arm_restore(ENTRY)
        self.assertEqual(suspension.pending_restore, ENTRY)
        self.assertEqual(suspension.take_restore_arm(), ENTRY)
        self.assertIsNone(suspension.pending_restore)
        self.assertIsNone(suspension.take_restore_arm())

    def test_disarm_empties_the_restore_arm(self) -> None:
        suspension = AudioSuspension()
        suspension.record(ENTRY)
        suspension.arm_restore(ENTRY)
        suspension.disarm_restore()
        self.assertIsNone(suspension.pending_restore)
        self.assertEqual(suspension.value, ENTRY)
        self.assertIsNone(suspension.take_restore_arm())

    def test_arming_none_clears_a_stale_arm(self) -> None:
        """A preview that interrupts nothing must re-arm to None so an earlier
        preview's arm can never leak forward."""
        suspension = AudioSuspension()
        suspension.arm_restore(ENTRY)
        suspension.arm_restore(None)
        self.assertIsNone(suspension.take_restore_arm())

    def test_arm_lifecycle_never_drops_the_recorded_entry(self) -> None:
        """The arm is eligibility plumbing; the messaging entry survives arm,
        take, and disarm (restore-by-intent messaging stays intact)."""
        suspension = AudioSuspension()
        suspension.record(ENTRY)
        suspension.arm_restore(ENTRY)
        suspension.take_restore_arm()
        self.assertEqual(suspension.value, ENTRY)
        suspension.arm_restore(ENTRY)
        suspension.disarm_restore()
        self.assertEqual(suspension.value, ENTRY)

    def test_arm_restore_rejects_foreign_types(self) -> None:
        with self.assertRaises(ValueError):
            AudioSuspension().arm_restore({"player_state": "playing"})  # type: ignore[arg-type]


class PlaybackContextTest(unittest.TestCase):
    def test_default_context_exposes_empty_suspension(self) -> None:
        context = PlaybackContext()
        self.assertIsNone(context.suspension.value)

    def test_each_instance_is_isolated(self) -> None:
        first = PlaybackContext()
        second = PlaybackContext()
        first.suspension.record(ENTRY)
        self.assertIsNone(second.suspension.value)


if __name__ == "__main__":
    unittest.main()
