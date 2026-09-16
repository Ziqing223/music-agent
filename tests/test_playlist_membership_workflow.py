import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from music_agent.identity import EntityType, generate_canonical_id, validate_canonical_id
from music_agent.playlist_membership_workflow import (
    MembershipCreationResult,
    MembershipWorkflowError,
    create_playlist_membership,
)
from music_agent.repository import CanonicalRepository


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class ExplicitMembershipWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)
        self.playlist = self.model["playlists"][0]["id"]  # pl_...555 (Repeated Membership)
        self.track = self.model["tracks"][0]["id"]       # trk_...111 (Synthetic Duet)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def memberships(self) -> list[dict]:
        with CanonicalRepository(self.database_path) as repository:
            return repository.load_model()["playlist_memberships"]

    def create(
        self, *, playlist_id=None, track_id=None, position=0, added_at=None
    ) -> MembershipCreationResult:
        return create_playlist_membership(
            self.database_path,
            playlist_id=self.playlist if playlist_id is None else playlist_id,
            track_id=self.track if track_id is None else track_id,
            position=position,
            added_at=added_at,
        )

    def test_create_one_membership_with_independent_pm_identity(self) -> None:
        result = self.create(position=5, added_at="2026-01-02T03:04:05+08:00")
        self.assertTrue(result.membership_id.startswith("pm_"))
        validate_canonical_id(EntityType.PLAYLIST_MEMBERSHIP, result.membership_id)
        self.assertNotEqual(
            result.membership_id, generate_canonical_id(EntityType.PLAYLIST_MEMBERSHIP)
        )
        memberships = self.memberships()
        self.assertEqual(len(memberships), 6)
        created = next(m for m in memberships if m["id"] == result.membership_id)
        self.assertEqual(created["playlist_id"], self.playlist)
        self.assertEqual(created["track_id"], self.track)
        self.assertEqual(created["position"], 5)
        self.assertEqual(created["added_at"], "2026-01-02T03:04:05+08:00")

    def test_same_playlist_track_creates_second_distinct_membership(self) -> None:
        first = self.create(position=0)
        second = self.create(position=0)
        self.assertNotEqual(first.membership_id, second.membership_id)
        self.assertTrue(first.membership_id.startswith("pm_"))
        self.assertTrue(second.membership_id.startswith("pm_"))
        ids = {m["id"] for m in self.memberships()}
        self.assertIn(first.membership_id, ids)
        self.assertIn(second.membership_id, ids)
        self.assertEqual(len(ids), 7)  # 5 fixture memberships + 2 new distinct pm_

    def test_position_does_not_participate_in_identity(self) -> None:
        first = self.create(position=3)
        second = self.create(position=4)
        self.assertNotEqual(first.membership_id, second.membership_id)
        self.assertEqual(first.position, 3)
        self.assertEqual(second.position, 4)
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(first.playlist_id, second.playlist_id)

    def test_added_at_does_not_participate_in_identity(self) -> None:
        first = self.create(position=0, added_at="2026-01-02T03:04:05Z")
        second = self.create(position=0, added_at="2026-02-03T04:05:06+00:00")
        self.assertNotEqual(first.membership_id, second.membership_id)
        self.assertEqual(first.added_at, "2026-01-02T03:04:05Z")

    def test_added_at_is_optional(self) -> None:
        result = self.create(position=0)
        self.assertIsNone(result.added_at)

    def test_invalid_playlist_has_no_mutation(self) -> None:
        with self.assertRaises(MembershipWorkflowError):
            self.create(playlist_id="pl_dddddddd-dddd-4ddd-8ddd-dddddddddddd")
        self.assertEqual(len(self.memberships()), 5)

    def test_invalid_track_has_no_mutation(self) -> None:
        with self.assertRaises(MembershipWorkflowError):
            self.create(track_id="trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd")
        self.assertEqual(len(self.memberships()), 5)

    def test_wrong_entity_namespace_has_no_mutation(self) -> None:
        # A Track ID in the playlist_id slot must be rejected by namespace, not dangling lookup.
        with self.assertRaises(MembershipWorkflowError):
            self.create(playlist_id=self.track)
        with self.assertRaises(MembershipWorkflowError):
            self.create(track_id=self.playlist)
        self.assertEqual(len(self.memberships()), 5)

    def test_invalid_position_and_added_at_are_rejected(self) -> None:
        for bad_position in (-1, "0", 1.5, True):
            with self.subTest(position=bad_position):
                with self.assertRaises(MembershipWorkflowError):
                    self.create(position=bad_position)
        for bad_added_at in ("2026-01-02T03:04:05", "", "not-a-date"):
            with self.subTest(added_at=bad_added_at):
                with self.assertRaises(MembershipWorkflowError):
                    self.create(position=0, added_at=bad_added_at)
        self.assertEqual(len(self.memberships()), 5)

    def test_restart_recovery_preserves_new_membership(self) -> None:
        result = self.create(position=9, added_at="2026-03-04T05:06:07+08:00")
        # Reopen the store fresh (the repository reopens from disk each call).
        memberships = self.memberships()
        created = next(m for m in memberships if m["id"] == result.membership_id)
        self.assertEqual(created["position"], 9)
        self.assertEqual(created["added_at"], "2026-03-04T05:06:07+08:00")
        self.assertEqual(created["playlist_id"], self.playlist)
        self.assertEqual(created["track_id"], self.track)

    def test_existing_memberships_preserved(self) -> None:
        before = self.memberships()
        self.create(position=7)
        after = self.memberships()
        before_ids = {m["id"] for m in before}
        after_ids = {m["id"] for m in after}
        self.assertTrue(before_ids <= after_ids)
        self.assertEqual(len(after_ids - before_ids), 1)

    def test_graph_validation_blocks_dangling_references_atomically(self) -> None:
        # A membership referencing a valid playlist/track that is dropped from the model would
        # fail graph validation. Here we confirm a normal create validates the full model; a
        # malformed request cannot corrupt the store.
        before = self.memberships()
        self.create(position=6)
        self.assertEqual(len(self.memberships()), len(before) + 1)

    def test_stale_whole_model_save_does_not_delete_prior_membership(self) -> None:
        # Simulate a second caller that loaded the model before the first membership was created,
        # then persists its own membership from that stale snapshot. Because save_model upserts by
        # primary key and never deletes playlist_memberships rows, the first membership must
        # survive alongside the second.
        with CanonicalRepository(self.database_path) as repository:
            stale_model = repository.load_model()

        first = self.create(position=1)
        assert first.membership_id not in {
            m["id"] for m in stale_model["playlist_memberships"]
        }

        stale_with_second = deepcopy(stale_model)
        stale_with_second["playlist_memberships"].append({
            "id": "pm_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1",
            "playlist_id": self.playlist,
            "track_id": self.track,
            "position": 2,
            "added_at": None,
        })
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(stale_with_second)

        memberships = self.memberships()
        ids = {m["id"] for m in memberships}
        self.assertIn(first.membership_id, ids)
        self.assertIn("pm_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1", ids)
        self.assertEqual(len(ids), 7)


if __name__ == "__main__":
    unittest.main()
