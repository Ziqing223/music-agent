import unittest

from music_agent.conversation_continuation import (
    PREVIEW_TRACK,
    OfferedAction,
    OfferedActionRegister,
)


TRACK_A = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TRACK_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _offer(target=TRACK_A, *, source="play_preview_downgrade"):
    return OfferedAction(
        kind=PREVIEW_TRACK,
        target_canonical_id=target,
        source=source,
        verified_title="Wendy" if target == TRACK_A else "Other",
        verified_artist="Artist",
    )


class OfferedActionRegisterTest(unittest.TestCase):
    def test_generic_and_preview_specific_acceptance_consume_exact_offer(self):
        for text in ("好的", "好", "可以", "行", "开始试听", "试听吧", "好的！"):
            with self.subTest(text=text):
                register = OfferedActionRegister()
                register.arm(_offer())
                decision = register.resolve(text)
                self.assertEqual(decision.outcome, "accepted")
                self.assertEqual(decision.action, _offer())
                self.assertIsNone(register.current)

    def test_pure_decline_consumes_without_action(self):
        for text in ("不用", "不用了", "算了", "不要", "不用。"):
            with self.subTest(text=text):
                register = OfferedActionRegister()
                register.arm(_offer())
                decision = register.resolve(text)
                self.assertEqual(decision.outcome, "declined")
                self.assertIsNone(decision.action)
                self.assertIsNone(register.current)

    def test_substantive_new_request_overrides_instead_of_pure_decline(self):
        register = OfferedActionRegister()
        register.arm(_offer())
        decision = register.resolve("不用，播放第二首")
        self.assertEqual(decision.outcome, "override")
        self.assertEqual(decision.replacement_text, "播放第二首")
        self.assertIsNone(register.current)

    def test_new_offer_supersedes_old_offer(self):
        register = OfferedActionRegister()
        register.arm(_offer(TRACK_A))
        register.arm(_offer(TRACK_B))
        decision = register.resolve("开始试听")
        self.assertEqual(decision.outcome, "accepted")
        self.assertEqual(decision.action.target_canonical_id, TRACK_B)
        self.assertIsNone(register.current)

    def test_accepted_offer_is_consumed_exactly_once(self):
        register = OfferedActionRegister()
        register.arm(_offer())
        first = register.resolve("好的")
        second = register.resolve("好的")
        self.assertEqual(first.outcome, "accepted")
        self.assertEqual(second.outcome, "none")

    def test_no_pending_offer_never_claims_preview_target(self):
        register = OfferedActionRegister()
        decision = register.resolve("开始试听")
        self.assertEqual(decision.outcome, "none")
        self.assertIsNone(decision.action)

    def test_contract_rejects_missing_target_and_future_action_kind(self):
        with self.assertRaises(ValueError):
            OfferedAction(kind=PREVIEW_TRACK, target_canonical_id="", source="x")
        with self.assertRaises(ValueError):
            OfferedAction(kind="resume_playback", target_canonical_id=TRACK_A, source="x")


if __name__ == "__main__":
    unittest.main()
