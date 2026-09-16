import copy
import json
import unittest
from pathlib import Path
from uuid import UUID

from music_agent.identity import (
    ENTITY_ID_PREFIX,
    EntityType,
    ExternalIdentityKey,
    IdentityConflictError,
    IdentityRegistry,
    IdentityValidationError,
    generate_canonical_id,
    validate_canonical_id,
)
from music_agent.validation import validate_fixture


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"


class StableIdentityTest(unittest.TestCase):
    def test_all_entity_types_generate_uuid4_in_correct_namespace(self) -> None:
        for entity_type in EntityType:
            with self.subTest(entity_type=entity_type):
                canonical_id = generate_canonical_id(entity_type)
                prefix = ENTITY_ID_PREFIX[entity_type]
                self.assertTrue(canonical_id.startswith(prefix))
                self.assertEqual(UUID(canonical_id[len(prefix) :]).version, 4)
                validate_canonical_id(entity_type, canonical_id)

    def test_repeated_generation_does_not_reuse_ids_in_process(self) -> None:
        generated = {generate_canonical_id(EntityType.TRACK) for _ in range(1000)}
        self.assertEqual(len(generated), 1000)

    def test_type_namespace_mismatch_is_rejected(self) -> None:
        track_id = generate_canonical_id(EntityType.TRACK)
        validate_canonical_id(EntityType.TRACK, track_id)
        with self.assertRaises(IdentityValidationError) as raised:
            validate_canonical_id(EntityType.ARTIST, track_id)
        self.assertEqual(raised.exception.code, "validation_error")

    def test_first_bind_repeat_lookup_and_idempotent_bind(self) -> None:
        registry = IdentityRegistry()
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "external_001")
        track_id = registry.create_canonical_id(EntityType.TRACK)
        self.assertEqual(registry.bind(key, track_id), track_id)
        self.assertEqual(registry.lookup(key), track_id)
        self.assertEqual(registry.bind(key, track_id), track_id)
        self.assertEqual(len(registry), 1)

    def test_lookup_returns_none_when_external_identity_is_not_found(self) -> None:
        registry = IdentityRegistry()
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "not_bound")
        self.assertIsNone(registry.lookup(key))

    def test_conflict_is_explicit_and_preserves_original_binding(self) -> None:
        registry = IdentityRegistry()
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "external_001")
        original = registry.create_canonical_id(EntityType.TRACK)
        attempted = registry.create_canonical_id(EntityType.TRACK)
        registry.bind(key, original)
        with self.assertRaises(IdentityConflictError) as raised:
            registry.bind(key, attempted)
        self.assertEqual(raised.exception.code, "identity_conflict")
        self.assertEqual(raised.exception.existing_canonical_id, original)
        self.assertEqual(raised.exception.attempted_canonical_id, attempted)
        self.assertEqual(registry.lookup(key), original)

    def test_different_external_ids_remain_distinct_without_metadata(self) -> None:
        registry = IdentityRegistry()
        first_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "external_001")
        second_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "external_002")
        first_id = registry.create_canonical_id(EntityType.TRACK)
        second_id = registry.create_canonical_id(EntityType.TRACK)
        registry.bind(first_key, first_id)
        registry.bind(second_key, second_id)
        self.assertNotEqual(registry.lookup(first_key), registry.lookup(second_key))

    def test_multiple_external_ids_can_bind_to_one_canonical_entity(self) -> None:
        registry = IdentityRegistry()
        track_id = registry.create_canonical_id(EntityType.TRACK)
        first_key = ExternalIdentityKey("apple_music", EntityType.TRACK, "external_001")
        second_key = ExternalIdentityKey("secondary_source", EntityType.TRACK, "external_001")
        registry.bind(first_key, track_id)
        registry.bind(second_key, track_id)
        self.assertEqual(registry.lookup(first_key), track_id)
        self.assertEqual(registry.lookup(second_key), track_id)

    def test_external_ids_are_opaque_and_case_sensitive(self) -> None:
        registry = IdentityRegistry()
        lower = ExternalIdentityKey("apple_music", EntityType.TRACK, "abc")
        upper = ExternalIdentityKey("apple_music", EntityType.TRACK, "ABC")
        lower_id = registry.create_canonical_id(EntityType.TRACK)
        upper_id = registry.create_canonical_id(EntityType.TRACK)
        registry.bind(lower, lower_id)
        registry.bind(upper, upper_id)
        self.assertEqual(registry.lookup(lower), lower_id)
        self.assertEqual(registry.lookup(upper), upper_id)

    def test_external_ids_are_not_unicode_normalized(self) -> None:
        registry = IdentityRegistry()
        composed = ExternalIdentityKey("source", EntityType.TRACK, "é")
        decomposed = ExternalIdentityKey("source", EntityType.TRACK, "é")
        composed_id = registry.create_canonical_id(EntityType.TRACK)
        decomposed_id = registry.create_canonical_id(EntityType.TRACK)
        registry.bind(composed, composed_id)
        registry.bind(decomposed, decomposed_id)
        self.assertEqual(registry.lookup(composed), composed_id)
        self.assertEqual(registry.lookup(decomposed), decomposed_id)

    def test_binding_rejects_canonical_id_from_another_entity_namespace(self) -> None:
        registry = IdentityRegistry()
        artist_key = ExternalIdentityKey("apple_music", EntityType.ARTIST, "artist_001")
        track_id = registry.create_canonical_id(EntityType.TRACK)
        with self.assertRaises(IdentityValidationError) as raised:
            registry.bind(artist_key, track_id)
        self.assertEqual(raised.exception.code, "validation_error")
        self.assertIsNone(registry.lookup(artist_key))

    def test_null_and_empty_external_ids_are_rejected(self) -> None:
        for external_id in (None, ""):
            with self.subTest(external_id=external_id):
                with self.assertRaises(IdentityValidationError):
                    ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)  # type: ignore[arg-type]

    def test_artist_and_album_can_exist_without_external_binding(self) -> None:
        registry = IdentityRegistry()
        artist_id = registry.create_canonical_id(EntityType.ARTIST)
        album_id = registry.create_canonical_id(EntityType.ALBUM)
        validate_canonical_id(EntityType.ARTIST, artist_id)
        validate_canonical_id(EntityType.ALBUM, album_id)
        self.assertEqual(len(registry), 0)

    def test_membership_ids_are_independent_new_identities(self) -> None:
        first = generate_canonical_id(EntityType.PLAYLIST_MEMBERSHIP)
        second = generate_canonical_id(EntityType.PLAYLIST_MEMBERSHIP)
        self.assertNotEqual(first, second)

    def test_generated_ids_integrate_with_canonical_fixture_schema(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        generated = copy.deepcopy(fixture)
        mapping: dict[str, str] = {}
        collections = (
            ("tracks", EntityType.TRACK),
            ("artists", EntityType.ARTIST),
            ("albums", EntityType.ALBUM),
            ("playlists", EntityType.PLAYLIST),
            ("playlist_memberships", EntityType.PLAYLIST_MEMBERSHIP),
        )
        for collection, entity_type in collections:
            for entity in generated[collection]:
                old_id = entity["id"]
                entity["id"] = generate_canonical_id(entity_type)
                mapping[old_id] = entity["id"]
        for album in generated["albums"]:
            album["artist_ids"] = [mapping[artist_id] for artist_id in album["artist_ids"]]
        for track in generated["tracks"]:
            track["artist_ids"] = [mapping[artist_id] for artist_id in track["artist_ids"]]
            if track["album_id"] is not None:
                track["album_id"] = mapping[track["album_id"]]
        for membership in generated["playlist_memberships"]:
            membership["playlist_id"] = mapping[membership["playlist_id"]]
            membership["track_id"] = mapping[membership["track_id"]]
        validate_fixture(generated)


if __name__ == "__main__":
    unittest.main()
