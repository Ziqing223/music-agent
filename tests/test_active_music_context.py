"""P14-C06.1: runtime-only ActiveMusicContext register.

The register replaced the service's ad-hoc playback-state dict with strictly the same
observable semantics: channel transitions happen at the same five boundaries, queue
navigation and stop_preview keep the ownership anchor, and a fresh instance starts at
``none`` (transient across service instances -- the service-level pin lives in
``test_playback_tools``). These tests pin the register itself: its initial state, its
mutators, its validation, the deliberate absence of any preview-active mirror, and the
P14-C07.3 batch context pair (pointer + item index, see BatchContextTest).
"""

from __future__ import annotations

import unittest

from music_agent.active_music_context import (
    CHANNEL_LIBRARY,
    CHANNEL_NONE,
    CHANNEL_PREVIEW,
    ActiveMusicContext,
    ActiveMusicContextError,
    ActiveMusicContextSnapshot,
    ActiveMusicContextValidationError,
    VerifiedSelection,
)

CANONICAL_ID = "trk_11111111-1111-4111-8111-111111111111"
PERSISTENT_ID = "777777777"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
SECOND_RUN_ID = "rcm_33333333-3333-4333-8333-333333333333"


class InitialStateTest(unittest.TestCase):
    def test_fresh_instance_is_neutral(self) -> None:
        context = ActiveMusicContext()
        self.assertEqual(context.channel, CHANNEL_NONE)
        self.assertIsNone(context.canonical_id)
        self.assertIsNone(context.persistent_id)
        self.assertIsNone(context.active_run_id)
        self.assertIsNone(context.active_item_index)
        self.assertIsNone(context.verified_selection_run_id)
        self.assertEqual(context.verified_selections, ())
        self.assertIsNone(context.referent_canonical_id)

    def test_instances_are_independent(self) -> None:
        first = ActiveMusicContext()
        second = ActiveMusicContext()
        first.note_preview(CANONICAL_ID)
        self.assertEqual(second.channel, CHANNEL_NONE)
        self.assertIsNone(second.canonical_id)
        self.assertIsNone(second.referent_canonical_id)


class MutatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ActiveMusicContext()

    def test_library_playback_sets_channel_track_and_anchor(self) -> None:
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.assertEqual(self.context.channel, CHANNEL_LIBRARY)
        self.assertEqual(self.context.canonical_id, CANONICAL_ID)
        self.assertEqual(self.context.persistent_id, PERSISTENT_ID)

    def test_preview_sets_channel_and_track_but_never_touches_anchor(self) -> None:
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.context.note_preview(CANONICAL_ID)
        self.assertEqual(self.context.channel, CHANNEL_PREVIEW)
        self.assertEqual(self.context.canonical_id, CANONICAL_ID)
        self.assertEqual(self.context.persistent_id, PERSISTENT_ID)

    def test_clear_channel_resets_channel_but_keeps_anchor(self) -> None:
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.context.clear_channel()
        self.assertEqual(self.context.channel, CHANNEL_NONE)
        self.assertIsNone(self.context.canonical_id)
        self.assertEqual(self.context.persistent_id, PERSISTENT_ID)

    def test_clear_channel_is_idempotent(self) -> None:
        self.context.clear_channel()
        self.context.clear_channel()
        self.assertEqual(self.context.channel, CHANNEL_NONE)
        self.assertIsNone(self.context.canonical_id)
        self.assertIsNone(self.context.persistent_id)

    def test_registered_track_carries_through_channel_transitions(self) -> None:
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.context.note_preview(CANONICAL_ID)
        self.context.clear_channel()
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.assertEqual(self.context.channel, CHANNEL_LIBRARY)
        self.assertEqual(self.context.persistent_id, PERSISTENT_ID)


class ReferentTest(unittest.TestCase):
    """P19-T14-F-R4: the session-local conversational referent.

    ``referent_canonical_id`` is the last successfully resolved explicit track
    target (试听 X / 播放 X), decoupled from the action-log channel: it is
    written by the same two mutators, replaced whole by each new target, and
    deliberately survives every channel clear -- a preview stop keeps the
    target conversational. A fresh instance (session reset) has none.
    """

    OTHER_CANONICAL_ID = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

    def setUp(self) -> None:
        self.context = ActiveMusicContext()

    def test_explicit_play_sets_the_referent(self) -> None:
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_explicit_preview_sets_the_referent(self) -> None:
        self.context.note_preview(CANONICAL_ID)
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_a_new_explicit_target_replaces_the_previous_referent(self) -> None:
        # Regression 5: explicit track C (here OTHER_CANONICAL_ID) becomes the
        # referent wholesale; no cross-contamination of the previous target.
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.context.note_preview(self.OTHER_CANONICAL_ID)
        self.assertEqual(self.context.referent_canonical_id, self.OTHER_CANONICAL_ID)

    def test_referent_survives_channel_clears(self) -> None:
        # The Owner regression: 试听 X → stop preview → 试听他 must still
        # target X -- clear_channel closes the action log only.
        self.context.note_preview(CANONICAL_ID)
        self.context.clear_channel()
        self.assertEqual(self.context.channel, CHANNEL_NONE)
        self.assertIsNone(self.context.canonical_id)
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_referent_survives_queue_navigation_clears(self) -> None:
        # next/previous also route through clear_channel; the conversational
        # target must not be lost to unrelated queue movement.
        self.context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.context.clear_channel()
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_failed_requests_never_touch_the_referent(self) -> None:
        # Regression 4: a failed validation leaves the previous referent
        # intact -- the mutator refused before any state moved.
        self.context.note_preview(CANONICAL_ID)
        with self.assertRaises(ActiveMusicContextValidationError):
            self.context.note_preview("")
        with self.assertRaises(ActiveMusicContextValidationError):
            self.context.note_library_playback(CANONICAL_ID, "")
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_batch_cursor_operations_never_touch_the_referent(self) -> None:
        # note_batch_item / clear_item_index move only the batch cursor; the
        # conversational target is independent of recommendation bookkeeping.
        self.context.note_recommendation_batch("rcm_00000000-0000-4000-8000-000000000000")
        self.context.note_preview(CANONICAL_ID)
        self.context.note_batch_item(0)
        self.context.clear_item_index()
        self.assertEqual(self.context.referent_canonical_id, CANONICAL_ID)

    def test_snapshot_mirrors_the_referent(self) -> None:
        self.context.note_preview(CANONICAL_ID)
        snapshot = self.context.snapshot()
        self.assertEqual(snapshot.referent_canonical_id, CANONICAL_ID)
        self.context.clear_channel()
        self.assertEqual(snapshot.referent_canonical_id, CANONICAL_ID)


class ValidationTest(unittest.TestCase):
    def test_empty_canonical_id_refused(self) -> None:
        for bad in ("", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ActiveMusicContextValidationError):
                    ActiveMusicContext().note_preview(bad)

    def test_library_playback_requires_non_empty_persistent_id(self) -> None:
        for bad in ("", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ActiveMusicContextValidationError):
                    ActiveMusicContext().note_library_playback(CANONICAL_ID, bad)

    def test_failed_validation_leaves_state_untouched(self) -> None:
        context = ActiveMusicContext()
        with self.assertRaises(ActiveMusicContextValidationError):
            context.note_preview("")
        self.assertEqual(context.channel, CHANNEL_NONE)
        self.assertIsNone(context.canonical_id)


class BatchContextTest(unittest.TestCase):
    """P14-C07.3: the recommendation batch pointer / item index pair.

    The pair obeys the state model: (None, None), (run, None) and (run, index) are
    legal; (None, index) is a forbidden state the mutators refuse to produce. The
    pointer is a reference -- nothing here mirrors batch content, and the batch
    boundary always restarts the index.
    """

    def setUp(self) -> None:
        self.context = ActiveMusicContext()

    def test_note_recommendation_batch_sets_pointer_and_restarts_index(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.assertEqual(self.context.active_run_id, RUN_ID)
        self.assertIsNone(self.context.active_item_index)

    def test_recommendation_batch_replaces_previous_and_restarts_index(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.context.note_batch_item(0)
        self.context.note_recommendation_batch(SECOND_RUN_ID)
        self.assertEqual(self.context.active_run_id, SECOND_RUN_ID)
        self.assertIsNone(self.context.active_item_index)

    def test_note_batch_item_sets_zero_based_index(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.context.note_batch_item(2)
        self.assertEqual(self.context.active_item_index, 2)

    def test_item_index_without_run_is_refused(self) -> None:
        with self.assertRaises(ActiveMusicContextError):
            self.context.note_batch_item(0)
        # The refusal happens before any mutation: the forbidden (None, index) state
        # can never be produced.
        self.assertIsNone(self.context.active_run_id)
        self.assertIsNone(self.context.active_item_index)

    def test_item_index_must_be_non_negative_int(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        for bad in (-1, 1.5, "0", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ActiveMusicContextValidationError):
                    self.context.note_batch_item(bad)

    def test_batch_run_id_must_be_non_empty_string(self) -> None:
        for bad in ("", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ActiveMusicContextValidationError):
                    self.context.note_recommendation_batch(bad)

    def test_clear_item_index_keeps_pointer_and_is_idempotent(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.context.note_batch_item(1)
        self.context.clear_item_index()
        self.assertEqual(self.context.active_run_id, RUN_ID)
        self.assertIsNone(self.context.active_item_index)
        self.context.clear_item_index()
        self.assertEqual(self.context.active_run_id, RUN_ID)
        self.assertIsNone(self.context.active_item_index)

    def test_failed_batch_validation_leaves_state_untouched(self) -> None:
        with self.assertRaises(ActiveMusicContextValidationError):
            self.context.note_recommendation_batch("")
        self.assertIsNone(self.context.active_run_id)

    def test_snapshot_mirrors_batch_fields(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.context.note_batch_item(0)
        snapshot = self.context.snapshot()
        self.assertEqual(snapshot.active_run_id, RUN_ID)
        self.assertEqual(snapshot.active_item_index, 0)


class VerifiedSelectionTest(unittest.TestCase):
    """P21 Slice 3: completed-action facts are distinct from the old cursor."""

    def setUp(self) -> None:
        self.context = ActiveMusicContext()

    def _record(
        self,
        canonical_id: str = CANONICAL_ID,
        *,
        run_id: str = RUN_ID,
        item_position: int = 1,
        action_kind: str = "play_track",
        playback_route: str = "library",
    ) -> VerifiedSelection:
        return self.context.note_verified_selection(
            run_id=run_id,
            canonical_id=canonical_id,
            item_position=item_position,
            action_kind=action_kind,
            playback_route=playback_route,
        )

    def test_completed_fact_is_scoped_to_the_active_run(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)

        recorded = self._record()

        self.assertEqual(
            recorded,
            VerifiedSelection(CANONICAL_ID, 1, "play_track", "library"),
        )
        self.assertEqual(self.context.verified_selection_run_id, RUN_ID)
        self.assertEqual(self.context.verified_selections_for_run(RUN_ID), (recorded,))
        self.assertEqual(self.context.verified_selections_for_run(SECOND_RUN_ID), ())

    def test_fresh_derived_session_can_bind_the_authoritative_run(self) -> None:
        recorded = self._record()

        self.assertEqual(self.context.active_run_id, RUN_ID)
        self.assertEqual(self.context.verified_selection_run_id, RUN_ID)
        self.assertEqual(self.context.verified_selections, (recorded,))

    def test_new_run_clears_old_verified_selections(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self._record()

        self.context.note_recommendation_batch(SECOND_RUN_ID)

        self.assertEqual(self.context.active_run_id, SECOND_RUN_ID)
        self.assertEqual(self.context.verified_selection_run_id, SECOND_RUN_ID)
        self.assertEqual(self.context.verified_selections, ())
        self.assertEqual(self.context.verified_selections_for_run(RUN_ID), ())

    def test_different_run_cannot_write_into_the_active_projection(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)

        with self.assertRaises(ActiveMusicContextError):
            self._record(run_id=SECOND_RUN_ID)

        self.assertEqual(self.context.verified_selections, ())

    def test_same_completed_fact_is_idempotent_but_conflict_is_refused(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        first = self._record()
        second = self._record()

        self.assertIs(first, second)
        self.assertEqual(len(self.context.verified_selections), 1)
        with self.assertRaises(ActiveMusicContextError):
            self._record(action_kind="preview_catalog_track")
        self.assertEqual(self.context.verified_selections, (first,))

    def test_active_item_index_is_not_verified_truth(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        self.context.note_batch_item(0)

        self.assertEqual(self.context.active_item_index, 0)
        self.assertEqual(self.context.verified_selections, ())

    def test_verified_fact_validation_is_fail_closed(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        cases = (
            {"canonical_id": ""},
            {"item_position": 0},
            {"action_kind": ""},
            {"playback_route": ""},
        )
        for replacement in cases:
            arguments = {
                "canonical_id": CANONICAL_ID,
                "item_position": 1,
                "action_kind": "play_track",
                "playback_route": "library",
            }
            arguments.update(replacement)
            with self.subTest(replacement=replacement):
                with self.assertRaises(ActiveMusicContextValidationError):
                    self.context.note_verified_selection(
                        run_id=RUN_ID,
                        **arguments,
                    )
                self.assertEqual(self.context.verified_selections, ())

    def test_snapshot_is_an_immutable_copy_of_verified_facts(self) -> None:
        self.context.note_recommendation_batch(RUN_ID)
        first = self._record()
        snapshot = self.context.snapshot()

        self.context.note_recommendation_batch(SECOND_RUN_ID)

        self.assertEqual(snapshot.verified_selection_run_id, RUN_ID)
        self.assertEqual(snapshot.verified_selections, (first,))


class SnapshotTest(unittest.TestCase):
    """P14-C06.2: snapshot() is an immutable copy -- later mutations never change it."""

    def test_snapshot_mirrors_register_state(self) -> None:
        context = ActiveMusicContext()
        context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        snapshot = context.snapshot()
        self.assertIsInstance(snapshot, ActiveMusicContextSnapshot)
        self.assertEqual(snapshot.channel, CHANNEL_LIBRARY)
        self.assertEqual(snapshot.canonical_id, CANONICAL_ID)
        self.assertEqual(snapshot.persistent_id, PERSISTENT_ID)
        self.assertEqual(snapshot.referent_canonical_id, CANONICAL_ID)

    def test_snapshot_is_immune_to_later_mutations(self) -> None:
        context = ActiveMusicContext()
        context.note_preview(CANONICAL_ID)
        snapshot = context.snapshot()
        context.clear_channel()
        context.note_library_playback(CANONICAL_ID, PERSISTENT_ID)
        self.assertEqual(snapshot.channel, CHANNEL_PREVIEW)
        self.assertEqual(snapshot.canonical_id, CANONICAL_ID)
        self.assertIsNone(snapshot.persistent_id)

    def test_snapshot_is_frozen(self) -> None:
        snapshot = ActiveMusicContext().snapshot()
        with self.assertRaises(Exception):
            snapshot.channel = CHANNEL_PREVIEW  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
