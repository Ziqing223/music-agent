"""P15-S1: Preview Session unit tests -- state machine, queue cursor, terminal races."""

import unittest

from music_agent.preview_session import (
    PreviewSession,
    PreviewSessionItem,
    PreviewSessionRegistry,
)

FIRST = PreviewSessionItem("trk_11111111-1111-4111-8111-111111111111", "Synthetic Duet", "Artist Alpha", "preview_only")
SECOND = PreviewSessionItem("trk_22222222-2222-4222-8222-222222222222", "Synthetic Solo", "Artist Alpha", "library")
THIRD = PreviewSessionItem("trk_44444444-4444-4444-8444-444444444444", "Year Precision", "Artist Gamma", "preview_only")


class PreviewSessionTest(unittest.TestCase):
    def test_session_requires_at_least_one_item(self) -> None:
        with self.assertRaises(ValueError):
            PreviewSession([])

    def test_session_rejects_foreign_items(self) -> None:
        with self.assertRaises(ValueError):
            PreviewSession([{"canonical_id": "trk_x"}])  # type: ignore[list-item]

    def test_initial_snapshot_running_at_first_item(self) -> None:
        session = PreviewSession([FIRST, SECOND, THIRD])
        snapshot = session.snapshot()
        self.assertEqual(snapshot.state, "running")
        self.assertEqual(snapshot.total, 3)
        self.assertEqual(snapshot.position, 1)
        self.assertEqual(snapshot.current_canonical_id, FIRST.canonical_id)
        self.assertEqual(snapshot.current_name, FIRST.name)
        self.assertEqual(snapshot.skipped, ())
        self.assertEqual(session.current_item(), FIRST)

    def test_assembly_skips_are_preserved_in_order(self) -> None:
        skipped = [{"canonical_id": "trk_0", "name": "X", "reason": "unavailable: no playable route"}]
        session = PreviewSession([FIRST], skipped)
        self.assertEqual(session.snapshot().skipped, tuple(skipped))

    def test_advance_walks_the_queue_then_completes(self) -> None:
        session = PreviewSession([FIRST, SECOND])
        self.assertEqual(session.advance(), SECOND)
        snapshot = session.snapshot()
        self.assertEqual(snapshot.state, "running")
        self.assertEqual(snapshot.position, 2)
        self.assertEqual(snapshot.current_canonical_id, SECOND.canonical_id)
        self.assertEqual(snapshot.current_name, SECOND.name)
        # The last clip's natural end exhausts the queue: COMPLETED, one-shot.
        self.assertIsNone(session.advance())
        completed = session.snapshot()
        self.assertEqual(completed.state, "completed")
        self.assertEqual(completed.position, 2)  # honest: the last clip was #2
        self.assertEqual(completed.current_canonical_id, SECOND.canonical_id)
        # Terminal no-ops: further advances change nothing.
        self.assertIsNone(session.advance())
        self.assertEqual(session.snapshot().state, "completed")

    def test_advance_skipped_moves_past_the_failed_clip(self) -> None:
        session = PreviewSession([FIRST, SECOND, THIRD])
        # FIRST finishes; SECOND fails to start -> skipped record moves us to THIRD.
        self.assertEqual(session.advance(), SECOND)
        self.assertEqual(
            session.advance(
                skipped={"canonical_id": SECOND.canonical_id, "name": SECOND.name, "reason": "catalog_preview_unavailable"}
            ),
            THIRD,
        )
        snapshot = session.snapshot()
        self.assertEqual(snapshot.position, 3)
        self.assertEqual(snapshot.skipped, (
            {"canonical_id": SECOND.canonical_id, "name": SECOND.name, "reason": "catalog_preview_unavailable"},
        ))

    def test_all_runtime_failures_end_in_honest_completed_all_skipped(self) -> None:
        session = PreviewSession([FIRST])
        self.assertIsNone(
            session.advance(
                skipped={"canonical_id": FIRST.canonical_id, "name": FIRST.name, "reason": "catalog_preview_unavailable"}
            )
        )
        snapshot = session.snapshot()
        self.assertEqual(snapshot.state, "completed")
        self.assertEqual(snapshot.position, 1)
        self.assertEqual(len(snapshot.skipped), 1)

    def test_cancel_is_terminal_and_idempotent(self) -> None:
        session = PreviewSession([FIRST, SECOND])
        self.assertTrue(session.cancel())
        snapshot = session.snapshot()
        self.assertEqual(snapshot.state, "cancelled")
        # A cancelled session never advances and never un-cancels.
        self.assertFalse(session.cancel())
        self.assertIsNone(session.advance())
        self.assertIsNone(session.current_item())
        self.assertEqual(session.snapshot().state, "cancelled")

    def test_advance_after_skipped_return_rejects_non_dict(self) -> None:
        with self.assertRaises(ValueError):
            PreviewSession([FIRST]).advance(skipped=("bad",))  # type: ignore[arg-type]

    def test_fail_is_terminal_one_shot_and_carries_the_reason(self) -> None:
        """P15-S1 真机修复: a systemic failure ends the run with FAILED -- one-shot,
        reason exposed, no advance/cancel can resurrect it."""
        session = PreviewSession([FIRST, SECOND])
        self.assertTrue(session.fail("store thread error"))
        snapshot = session.snapshot()
        self.assertEqual(snapshot.state, "failed")
        self.assertEqual(snapshot.failure_reason, "store thread error")
        self.assertFalse(session.fail("no-op later"))
        self.assertIsNone(session.advance())
        self.assertIsNone(session.current_item())
        self.assertFalse(session.cancel())
        self.assertEqual(session.snapshot().state, "failed")

    def test_fail_rejects_a_blank_or_non_string_reason(self) -> None:
        with self.assertRaises(ValueError):
            PreviewSession([FIRST]).fail("")
        with self.assertRaises(ValueError):
            PreviewSession([FIRST]).fail(7)  # type: ignore[arg-type]

    def test_failure_reason_is_none_outside_a_failed_session(self) -> None:
        session = PreviewSession([FIRST])
        self.assertIsNone(session.snapshot().failure_reason)
        session.cancel()
        self.assertIsNone(session.snapshot().failure_reason)


class PreviewSessionRegistryTest(unittest.TestCase):
    def test_starts_empty(self) -> None:
        registry = PreviewSessionRegistry()
        self.assertIsNone(registry.current())
        self.assertIsNone(registry.snapshot())
        self.assertIsNone(registry.cancel())
        self.assertIsNone(registry.finish_completed())

    def test_start_returns_new_session_and_none_when_nothing_replaced(self) -> None:
        registry = PreviewSessionRegistry()
        session, replaced = registry.start([FIRST])
        self.assertIsNone(replaced)
        self.assertIs(registry.current(), session)
        self.assertEqual(registry.snapshot().state, "running")

    def test_start_replaces_a_live_session(self) -> None:
        registry = PreviewSessionRegistry()
        first, _ = registry.start([FIRST])
        second, replaced = registry.start([SECOND])
        self.assertIsNotNone(replaced)
        self.assertIs(registry.current(), second)
        self.assertEqual(replaced.snapshot().current_canonical_id, FIRST.canonical_id)

    def test_cancel_clears_the_register_and_returns_the_terminal_snapshot(self) -> None:
        registry = PreviewSessionRegistry()
        registry.start([FIRST])
        cancelled = registry.cancel()
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.state, "cancelled")
        self.assertIsNone(registry.current())
        self.assertIsNone(registry.snapshot())
        # One-shot: a second cancel finds nothing.
        self.assertIsNone(registry.cancel())

    def test_finish_completed_clears_once_and_only_on_completion(self) -> None:
        registry = PreviewSessionRegistry()
        session, _ = registry.start([FIRST])
        # Not exhausted yet: no clear, no snapshot returned.
        self.assertIsNone(registry.finish_completed())
        self.assertIs(registry.current(), session)
        # Exhaust naturally through the session's own advance...
        self.assertIsNone(session.advance())
        completed = registry.finish_completed()
        self.assertIsNotNone(completed)
        self.assertEqual(completed.state, "completed")
        self.assertIsNone(registry.current())
        # One-shot terminal: nothing to clear a second time.
        self.assertIsNone(registry.finish_completed())

    def test_cancel_after_natural_exhaustion_is_still_cancel_first_wins_style(self) -> None:
        """A stop that lands while the last clip is finishing: once the queue ended,
        the registry treats the completion as the terminal fact (cancel finds nothing
        -- the session was never RUNNING when the stop arrived).
        """
        registry = PreviewSessionRegistry()
        session, _ = registry.start([FIRST, SECOND])
        self.assertEqual(session.advance(), SECOND)
        cancelled = registry.cancel()  # stop lands during the last clip
        self.assertEqual(cancelled.state, "cancelled")
        # The natural finish then arrives late: the registry is already empty.
        self.assertIsNone(registry.finish_completed())

    def test_fail_clears_the_register_and_returns_the_terminal_snapshot(self) -> None:
        """P15-S1 真机修复: FAILED is a third one-shot terminal -- clear the
        register, expose the snapshot exactly once; later cleanups find nothing."""
        registry = PreviewSessionRegistry()
        registry.start([FIRST])
        failed = registry.fail("wire broke")
        self.assertIsNotNone(failed)
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.failure_reason, "wire broke")
        self.assertIsNone(registry.current())
        self.assertIsNone(registry.snapshot())
        # One-shot: a second fail (or a late finish) finds nothing.
        self.assertIsNone(registry.fail("again"))
        self.assertIsNone(registry.finish_completed())

    def test_fail_does_not_clear_a_nothing_state(self) -> None:
        registry = PreviewSessionRegistry()
        self.assertIsNone(registry.fail("nothing live"))


if __name__ == "__main__":
    unittest.main()