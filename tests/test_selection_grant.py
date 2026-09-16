from __future__ import annotations

import unittest

from music_agent.intent_router import resolve_turn_plan
from music_agent.selection_grant import (
    DelegatedAudioAction,
    build_selection_grant,
    delegated_action_is_authorized,
    select_delegated_audio_action,
    selection_grant_is_exhausted,
)


RUN_ID = "rcm_11111111-1111-4111-8111-111111111111"
LIBRARY_ID = "trk_11111111-1111-4111-8111-111111111111"
PREVIEW_ID = "trk_22222222-2222-4222-8222-222222222222"


def _payload(*items: dict) -> dict:
    return {"run_id": RUN_ID, "items": list(items)}


def _item(
    canonical_id: str,
    route: str | None,
    *,
    position: int | None = None,
    name: str = "Song",
    artist: str = "Artist",
) -> dict:
    result = {
        "target_id": canonical_id,
        "name": name,
        "artist_name": artist,
        "playback": {} if route is None else {"route": route},
    }
    if position is not None:
        result["position"] = position
    return result


class SelectionGrantContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.turn_plan = resolve_turn_plan("随便播放一首")

    def test_authorized_items_are_exactly_the_authoritative_run_items(self) -> None:
        payload = _payload(
            _item(LIBRARY_ID, "library", position=1, name="Library Song"),
            _item(PREVIEW_ID, "preview_only", position=2, name="Preview Song"),
        )

        grant = build_selection_grant(self.turn_plan, payload)

        self.assertIsNotNone(grant)
        self.assertEqual(grant.recommendation_run_id, RUN_ID)
        self.assertEqual(grant.selection_mode, "agent_choose_one")
        self.assertEqual(grant.max_audio_actions, 1)
        self.assertEqual(
            {(item.position, item.canonical_id, item.playback_route) for item in grant.authorized_items},
            {
                (1, LIBRARY_ID, "library"),
                (2, PREVIEW_ID, "preview_only"),
            },
        )
        self.assertEqual(
            grant.allowed_actions,
            frozenset({"play_track", "preview_catalog_track"}),
        )

    def test_one_library_item_selects_formal_play_from_authoritative_route(self) -> None:
        grant = build_selection_grant(
            self.turn_plan,
            _payload(_item(LIBRARY_ID, "library")),
        )

        action = select_delegated_audio_action(grant)

        self.assertEqual(action.canonical_id, LIBRARY_ID)
        self.assertEqual(action.playback_route, "library")
        self.assertEqual(action.tool_name, "play_track")
        self.assertTrue(delegated_action_is_authorized(action, grant))

    def test_multiple_items_select_the_first_eligible_item_stably(self) -> None:
        grant = build_selection_grant(
            self.turn_plan,
            _payload(
                _item("trk_00000000-0000-4000-8000-000000000000", "unavailable"),
                _item(PREVIEW_ID, "preview_only"),
                _item(LIBRARY_ID, "library"),
            ),
        )

        first = select_delegated_audio_action(grant)
        second = select_delegated_audio_action(grant)

        self.assertEqual(first, second)
        self.assertEqual(first.canonical_id, PREVIEW_ID)
        self.assertEqual(first.tool_name, "preview_catalog_track")

    def test_provider_shaped_action_cannot_inject_identity_or_change_route(self) -> None:
        grant = build_selection_grant(
            self.turn_plan,
            _payload(_item(LIBRARY_ID, "library")),
        )
        injected_identity = DelegatedAudioAction(
            recommendation_run_id=RUN_ID,
            item_position=1,
            canonical_id=PREVIEW_ID,
            playback_route="library",
            tool_name="play_track",
        )
        changed_route = DelegatedAudioAction(
            recommendation_run_id=RUN_ID,
            item_position=1,
            canonical_id=LIBRARY_ID,
            playback_route="preview_only",
            tool_name="preview_catalog_track",
        )

        self.assertFalse(delegated_action_is_authorized(injected_identity, grant))
        self.assertFalse(delegated_action_is_authorized(changed_route, grant))

    def test_unsupported_or_unresolved_route_fails_closed(self) -> None:
        for route in ("unavailable", "future_route", None):
            with self.subTest(route=route):
                grant = build_selection_grant(
                    self.turn_plan,
                    _payload(_item(LIBRARY_ID, route)),
                )
                self.assertIsNone(select_delegated_audio_action(grant))

    def test_non_delegated_turn_cannot_create_a_selection_grant(self) -> None:
        plan = resolve_turn_plan("播放第二首")
        payload = _payload(_item(LIBRARY_ID, "library"))

        self.assertIsNone(build_selection_grant(plan, payload))

    def test_choose_another_excludes_only_verified_canonical_ids(self) -> None:
        plan = resolve_turn_plan("再换一首")
        another_id = "trk_33333333-3333-4333-8333-333333333333"
        grant = build_selection_grant(
            plan,
            _payload(
                _item(LIBRARY_ID, "library", position=1),
                _item(PREVIEW_ID, "preview_only", position=2),
                _item(another_id, "library", position=3),
            ),
            verified_canonical_ids=frozenset({LIBRARY_ID}),
        )

        action = select_delegated_audio_action(grant)

        self.assertEqual(grant.selection_mode, "choose_another")
        self.assertEqual(action.canonical_id, PREVIEW_ID)
        self.assertEqual(action.item_position, 2)
        self.assertFalse(selection_grant_is_exhausted(grant))

    def test_choose_another_is_deterministic_across_progressive_exclusions(self) -> None:
        plan = resolve_turn_plan("再换一首")
        payload = _payload(
            _item(LIBRARY_ID, "library", position=1),
            _item(PREVIEW_ID, "preview_only", position=2),
        )
        first = select_delegated_audio_action(
            build_selection_grant(plan, payload)
        )
        second = select_delegated_audio_action(
            build_selection_grant(
                plan,
                payload,
                verified_canonical_ids=frozenset({LIBRARY_ID}),
            )
        )

        self.assertEqual(first.canonical_id, LIBRARY_ID)
        self.assertEqual(second.canonical_id, PREVIEW_ID)

    def test_choose_another_reports_exhausted_only_after_all_executable_items(self) -> None:
        plan = resolve_turn_plan("再换一首")
        payload = _payload(
            _item(LIBRARY_ID, "library"),
            _item(PREVIEW_ID, "preview_only"),
        )
        grant = build_selection_grant(
            plan,
            payload,
            verified_canonical_ids=frozenset({LIBRARY_ID, PREVIEW_ID}),
        )

        self.assertIsNone(select_delegated_audio_action(grant))
        self.assertTrue(selection_grant_is_exhausted(grant))

    def test_unsupported_only_batch_is_not_misreported_as_exhausted(self) -> None:
        grant = build_selection_grant(
            resolve_turn_plan("再换一首"),
            _payload(_item(LIBRARY_ID, "unavailable")),
            verified_canonical_ids=frozenset({LIBRARY_ID}),
        )

        self.assertIsNone(select_delegated_audio_action(grant))
        self.assertFalse(selection_grant_is_exhausted(grant))

    def test_agent_choose_one_keeps_existing_behavior_despite_verified_projection(self) -> None:
        grant = build_selection_grant(
            self.turn_plan,
            _payload(
                _item(LIBRARY_ID, "library"),
                _item(PREVIEW_ID, "preview_only"),
            ),
            verified_canonical_ids=frozenset({LIBRARY_ID}),
        )

        action = select_delegated_audio_action(grant)

        self.assertEqual(action.canonical_id, LIBRARY_ID)
        self.assertFalse(selection_grant_is_exhausted(grant))


if __name__ == "__main__":
    unittest.main()
