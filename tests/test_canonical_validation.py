import copy
import json
import unittest
from pathlib import Path

from music_agent.validation import (
    GraphValidationError,
    StructuralValidationError,
    validate_fixture,
    validate_graph,
    validate_structure,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class CanonicalValidationTest(unittest.TestCase):
    def assert_structural_error(self, fixture: dict, validator: str) -> None:
        with self.assertRaises(StructuralValidationError) as raised:
            validate_structure(fixture)
        self.assertEqual(raised.exception.issue.validator, validator)

    def assert_graph_error(self, fixture: dict, code: str) -> None:
        validate_structure(fixture)
        with self.assertRaises(GraphValidationError) as raised:
            validate_graph(fixture)
        self.assertEqual(raised.exception.code, code)

    def test_complete_fixture_is_valid(self) -> None:
        validate_fixture(load_fixture())

    def test_null_false_zero_empty_array_and_multi_artist_are_valid(self) -> None:
        fixture = load_fixture()
        first, third = fixture["tracks"][0], fixture["tracks"][2]
        self.assertIs(first["library_state"]["favorited"], False)
        self.assertEqual(first["library_state"]["play_count"], 0)
        self.assertEqual(first["agent_metadata"]["tags"], [])
        self.assertIsNone(third["album_id"])
        self.assertIsNone(third["external_ids"]["apple_music_persistent_id"])
        self.assertEqual(len(first["artist_ids"]), 2)
        validate_fixture(fixture)

    def test_duplicate_track_membership_pair_with_distinct_ids_is_valid(self) -> None:
        fixture = load_fixture()
        first, second = fixture["playlist_memberships"][:2]
        self.assertEqual((first["playlist_id"], first["track_id"]), (second["playlist_id"], second["track_id"]))
        self.assertNotEqual(first["id"], second["id"])
        validate_fixture(fixture)

    def test_missing_required_key_is_invalid(self) -> None:
        fixture = load_fixture()
        del fixture["tracks"][0]["name"]
        self.assert_structural_error(fixture, "required")

    def test_wrong_field_type_is_invalid(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["duration_ms"] = "210000"
        self.assert_structural_error(fixture, "type")

    def test_scalar_field_cannot_be_array(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["composer"] = ["Composer"]
        self.assert_structural_error(fixture, "type")

    def test_multi_reference_cannot_be_scalar(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["artist_ids"] = fixture["tracks"][0]["artist_ids"][0]
        self.assert_structural_error(fixture, "type")

    def test_invalid_entity_id_shape_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["id"] = "track_from_name"
        self.assert_structural_error(fixture, "pattern")

    def test_invalid_datetime_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["library_state"]["added_to_library_at"] = "not-a-date"
        self.assert_structural_error(fixture, "format")

    def test_datetime_without_timezone_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["library_state"]["added_to_library_at"] = "2025-01-01T10:00:00"
        self.assert_structural_error(fixture, "format")

    def test_invalid_release_date_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["release_date"] = "2024-02-30"
        self.assert_structural_error(fixture, "format")

    def test_release_date_precision_uses_calendar_semantics(self) -> None:
        valid_dates = ("0001", "2026", "9999", "2026-01", "2026-12", "2026-01-31", "2024-02-29")
        invalid_dates = (
            "0000",
            "10000",
            "026",
            "２０２６",
            "２０２６-01",
            "2026-０１",
            "２０２６-０１",
            "２０２６-01-31",
            "2026-０１-31",
            "2026-01-３１",
            "２０２６-０１-３１",
            "2026-00",
            "2026-13",
            "2026-02-30",
            "2026-04-31",
            "2025-02-29",
        )

        for release_date in valid_dates:
            with self.subTest(release_date=release_date, expected="valid"):
                fixture = load_fixture()
                fixture["tracks"][0]["release_date"] = release_date
                validate_structure(fixture)

        for release_date in invalid_dates:
            with self.subTest(release_date=release_date, expected="invalid"):
                fixture = load_fixture()
                fixture["tracks"][0]["release_date"] = release_date
                self.assert_structural_error(fixture, "format")

    def test_rating_out_of_range_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["library_state"]["rating"] = 101
        self.assert_structural_error(fixture, "maximum")

    def test_playlist_track_ids_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["playlists"][0]["track_ids"] = [fixture["tracks"][0]["id"]]
        self.assert_structural_error(fixture, "additionalProperties")

    def test_unexpected_field_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["artists"][0]["unexpected"] = True
        self.assert_structural_error(fixture, "additionalProperties")

    def test_duplicate_canonical_id_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"].append(copy.deepcopy(fixture["tracks"][0]))
        self.assert_graph_error(fixture, "duplicate_canonical_id")

    def test_dangling_artist_id_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["artist_ids"] = ["art_dddddddd-dddd-4ddd-8ddd-dddddddddddd"]
        self.assert_graph_error(fixture, "dangling_artist_id")

    def test_dangling_album_id_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["tracks"][0]["album_id"] = "alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        self.assert_graph_error(fixture, "dangling_album_id")

    def test_dangling_playlist_id_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["playlist_memberships"][0]["playlist_id"] = "pl_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        self.assert_graph_error(fixture, "dangling_playlist_id")

    def test_dangling_track_id_is_rejected(self) -> None:
        fixture = load_fixture()
        fixture["playlist_memberships"][0]["track_id"] = "trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        self.assert_graph_error(fixture, "dangling_track_id")


if __name__ == "__main__":
    unittest.main()
