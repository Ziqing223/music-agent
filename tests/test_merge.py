import copy
import json
import tempfile
import unittest
from pathlib import Path

from music_agent.identity import EntityType
from music_agent.merge import (
    Authority,
    FieldDisposition,
    MergeValidationError,
    OWNERSHIP_POLICY,
    merge_observations,
)
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import (
    ObservationState,
    ObservationValidationError,
    ObservedValue,
    SourceObservation,
    SourcePresence,
)
from music_agent.validation import validate_fixture


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def observe(
    entity_type: EntityType,
    canonical_id: str,
    fields: dict[str, ObservedValue] | None = None,
    presence: SourcePresence = SourcePresence.PRESENT,
) -> SourceObservation:
    return SourceObservation(entity_type, canonical_id, "apple_music", fields or {}, presence)


class SourceObservationTest(unittest.TestCase):
    def test_missing_null_and_false_zero_empty_array_values_are_distinct(self) -> None:
        missing = ObservedValue.missing()
        null = ObservedValue.null()
        values = [ObservedValue.value(False), ObservedValue.value(0), ObservedValue.value([])]
        self.assertIs(missing.state, ObservationState.MISSING)
        self.assertIs(null.state, ObservationState.NULL)
        self.assertEqual([value.payload for value in values], [False, 0, []])
        self.assertTrue(all(value.state is ObservationState.VALUE for value in values))

    def test_invalid_tagged_state_combinations_are_rejected(self) -> None:
        invalid = (
            lambda: ObservedValue(ObservationState.MISSING, "payload"),
            lambda: ObservedValue(ObservationState.NULL, "payload"),
            lambda: ObservedValue(ObservationState.VALUE),
            lambda: ObservedValue.value(None),
        )
        for factory in invalid:
            with self.subTest(factory=factory):
                with self.assertRaises(ObservationValidationError):
                    factory()

    def test_non_present_observation_cannot_carry_fields(self) -> None:
        track_id = load_fixture()["tracks"][0]["id"]
        with self.assertRaises(ObservationValidationError):
            observe(
                EntityType.TRACK,
                track_id,
                {"name": ObservedValue.value("ambiguous")},
                SourcePresence.MISSING,
            )

    def test_empty_source_system_is_rejected(self) -> None:
        track_id = load_fixture()["tracks"][0]["id"]
        with self.assertRaises(ObservationValidationError) as context:
            SourceObservation(EntityType.TRACK, track_id, "", {})
        self.assertEqual(context.exception.code, "validation_error")


class OwnershipAwareMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_fixture()
        self.track_id = self.model["tracks"][0]["id"]

    def test_central_ownership_policy_has_shared_tags_and_apple_music_fields(self) -> None:
        self.assertIs(
            OWNERSHIP_POLICY[(EntityType.TRACK, "agent_metadata.tags")], Authority.SHARED_MODEL
        )
        self.assertIs(OWNERSHIP_POLICY[(EntityType.TRACK, "name")], Authority.APPLE_MUSIC)
        self.assertIs(
            OWNERSHIP_POLICY[(EntityType.TRACK, "library_state.play_count")],
            Authority.APPLE_MUSIC,
        )

    def test_apple_music_source_can_update_apple_music_owned_field(self) -> None:
        result = merge_observations(
            self.model,
            [observe(
                EntityType.TRACK,
                self.track_id,
                {"name": ObservedValue.value("Apple Music Name")},
            )],
        )
        self.assertEqual(result.model["tracks"][0]["name"], "Apple Music Name")
        self.assertIs(result.field_outcomes[0].disposition, FieldDisposition.UPDATED)

    def test_unsupported_source_cannot_update_any_authority_and_input_is_unchanged(self) -> None:
        original = copy.deepcopy(self.model)
        cases = (
            ("name", ObservedValue.value("Other Source Name")),
            ("agent_metadata.tags", ObservedValue.value(["other-source-tag"])),
        )
        for field_path, value in cases:
            with self.subTest(field_path=field_path):
                observation = SourceObservation(
                    EntityType.TRACK,
                    self.track_id,
                    "synthetic_other_source",
                    {field_path: value},
                )
                with self.assertRaisesRegex(MergeValidationError, "unsupported source system") as context:
                    merge_observations(self.model, [observation])
                self.assertEqual(context.exception.code, "validation_error")
                self.assertEqual(self.model, original)

    def test_track_scalar_relations_boolean_zero_array_and_nullable_fields_update(self) -> None:
        observation = observe(EntityType.TRACK, self.track_id, {
            "name": ObservedValue.value("Refreshed Track"),
            "artist_ids": ObservedValue.value([self.model["artists"][2]["id"]]),
            "album_id": ObservedValue.null(),
            "genres": ObservedValue.value([]),
            "composer": ObservedValue.null(),
            "library_state.favorited": ObservedValue.value(False),
            "library_state.play_count": ObservedValue.value(0),
            "library_state.rating": ObservedValue.null(),
        })
        self.model["tracks"][0]["library_state"]["favorited"] = True
        self.model["tracks"][0]["library_state"]["play_count"] = 9
        result = merge_observations(self.model, [observation])
        track = result.model["tracks"][0]
        self.assertEqual(track["name"], "Refreshed Track")
        self.assertEqual(track["artist_ids"], [self.model["artists"][2]["id"]])
        self.assertIsNone(track["album_id"])
        self.assertEqual(track["genres"], [])
        self.assertIsNone(track["composer"])
        self.assertIs(track["library_state"]["favorited"], False)
        self.assertEqual(track["library_state"]["play_count"], 0)
        self.assertIsNone(track["library_state"]["rating"])
        self.assertEqual(len(result.changed_fields), 7)

    def test_missing_preserves_and_shared_owned_value_is_explicitly_preserved(self) -> None:
        self.model["tracks"][0]["agent_metadata"]["tags"] = ["local-tag"]
        observation = observe(EntityType.TRACK, self.track_id, {
            "name": ObservedValue.missing(),
            "agent_metadata.tags": ObservedValue.value(["source-tag"]),
        })
        result = merge_observations(self.model, [observation])
        self.assertEqual(result.model["tracks"][0]["name"], self.model["tracks"][0]["name"])
        self.assertEqual(result.model["tracks"][0]["agent_metadata"]["tags"], ["local-tag"])
        self.assertEqual(result.changed_fields, ())
        self.assertEqual(
            [outcome.disposition for outcome in result.field_outcomes],
            [FieldDisposition.MISSING_PRESERVED, FieldDisposition.NOT_AUTHORITATIVE_PRESERVED],
        )

    def test_same_value_is_unchanged_and_repeat_is_idempotent(self) -> None:
        observation = observe(
            EntityType.TRACK, self.track_id, {"name": ObservedValue.value("Refreshed Track")}
        )
        first = merge_observations(self.model, [observation])
        second = merge_observations(first.model, [observation])
        self.assertEqual(first.changed_fields, (f"track:{self.track_id}:name",))
        self.assertEqual(second.changed_fields, ())
        self.assertIs(second.field_outcomes[0].disposition, FieldDisposition.UNCHANGED)
        self.assertEqual(second.model, first.model)

    def test_changed_fields_reports_net_batch_change_once(self) -> None:
        original_name = self.model["tracks"][0]["name"]
        result = merge_observations(self.model, [
            observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.value("Temporary")}),
            observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.value(original_name)}),
        ])
        self.assertEqual(result.model, self.model)
        self.assertEqual(result.changed_fields, ())

    def test_artist_album_playlist_and_membership_fields_update(self) -> None:
        artist_id = self.model["artists"][0]["id"]
        album_id = self.model["albums"][0]["id"]
        playlist_id = self.model["playlists"][0]["id"]
        membership_id = self.model["playlist_memberships"][0]["id"]
        observations = [
            observe(EntityType.ARTIST, artist_id, {"name": ObservedValue.value("Artist Updated")}),
            observe(EntityType.ALBUM, album_id, {
                "artist_ids": ObservedValue.value([self.model["artists"][2]["id"]]),
                "release_date": ObservedValue.value("2026-02"),
            }),
            observe(EntityType.PLAYLIST, playlist_id, {"name": ObservedValue.value("Playlist Updated")}),
            observe(EntityType.PLAYLIST_MEMBERSHIP, membership_id, {
                "playlist_id": ObservedValue.value(self.model["playlists"][1]["id"]),
                "track_id": ObservedValue.value(self.model["tracks"][1]["id"]),
                "position": ObservedValue.value(8),
                "added_at": ObservedValue.value("2026-01-02T03:04:05Z"),
            }),
        ]
        result = merge_observations(self.model, observations)
        self.assertEqual(result.model["artists"][0]["name"], "Artist Updated")
        self.assertEqual(result.model["albums"][0]["release_date"], "2026-02")
        self.assertEqual(result.model["playlists"][0]["name"], "Playlist Updated")
        self.assertEqual(result.model["playlist_memberships"][0]["position"], 8)
        validate_fixture(result.model)

    def test_identity_fields_and_unknown_paths_are_rejected(self) -> None:
        cases = ("id", "external_ids", "external_ids.apple_music_persistent_id", "unknown_field", "library_state.unknown")
        for path in cases:
            with self.subTest(path=path):
                with self.assertRaises(MergeValidationError) as context:
                    merge_observations(
                        self.model,
                        [observe(EntityType.TRACK, self.track_id, {path: ObservedValue.value("x")})],
                    )
                self.assertEqual(context.exception.code, "validation_error")

    def test_invalid_null_and_dangling_relation_fail_without_mutating_input(self) -> None:
        original = copy.deepcopy(self.model)
        observations = (
            observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.null()}),
            observe(EntityType.TRACK, self.track_id, {
                "album_id": ObservedValue.value("alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd")
            }),
        )
        for observation in observations:
            with self.subTest(observation=observation):
                with self.assertRaises(MergeValidationError):
                    merge_observations(self.model, [observation])
                self.assertEqual(self.model, original)

    def test_batch_failure_is_atomic_and_unresolved_target_is_rejected(self) -> None:
        original = copy.deepcopy(self.model)
        valid = observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.value("Changed")})
        invalid = observe(EntityType.TRACK, self.model["tracks"][1]["id"], {"name": ObservedValue.null()})
        with self.assertRaises(MergeValidationError):
            merge_observations(self.model, [valid, invalid])
        self.assertEqual(self.model, original)

        unknown_id = "trk_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        with self.assertRaisesRegex(MergeValidationError, "unresolved target"):
            merge_observations(self.model, [observe(EntityType.TRACK, unknown_id)])

    def test_all_source_presence_states_preserve_non_present_entities(self) -> None:
        for presence in (
            SourcePresence.MISSING, SourcePresence.UNKNOWN, SourcePresence.CONFIRMED_DELETED
        ):
            with self.subTest(presence=presence):
                result = merge_observations(
                    self.model, [observe(EntityType.TRACK, self.track_id, presence=presence)]
                )
                self.assertEqual(result.model, self.model)
                self.assertEqual(
                    result.source_presence[f"track:{self.track_id}"], presence
                )
        present = merge_observations(
            self.model,
            [observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.value("Present")})],
        )
        self.assertEqual(present.model["tracks"][0]["name"], "Present")

    def test_omitted_entity_is_not_deleted(self) -> None:
        omitted_id = self.model["tracks"][1]["id"]
        result = merge_observations(
            self.model,
            [observe(EntityType.TRACK, self.track_id, {"name": ObservedValue.value("Only A")})],
        )
        self.assertIn(omitted_id, {track["id"] for track in result.model["tracks"]})


class MergeRepositoryIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.model["tracks"][0]["agent_metadata"]["tags"] = ["local-tag"]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_merge_save_reopen_preserves_identity_graph_and_shared_owned_data(self) -> None:
        track_id = self.model["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)
            current = repository.load_model()
            result = merge_observations(current, [observe(EntityType.TRACK, track_id, {
                "name": ObservedValue.value("Persisted Refresh"),
                "artist_ids": ObservedValue.value([self.model["artists"][2]["id"]]),
                "agent_metadata.tags": ObservedValue.value(["source-tag"]),
            })])
            repository.save_model(result.model)
        with CanonicalRepository(self.database_path) as repository:
            loaded = repository.load_model()
        self.assertEqual(loaded["tracks"][0]["id"], track_id)
        self.assertEqual(loaded["tracks"][0]["name"], "Persisted Refresh")
        self.assertEqual(loaded["tracks"][0]["agent_metadata"]["tags"], ["local-tag"])
        validate_fixture(loaded)

    def test_failed_merge_is_not_persisted(self) -> None:
        track_id = self.model["tracks"][0]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)
            current = repository.load_model()
            with self.assertRaises(MergeValidationError):
                merge_observations(
                    current, [observe(EntityType.TRACK, track_id, {"name": ObservedValue.null()})]
                )
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), self.model)


if __name__ == "__main__":
    unittest.main()
