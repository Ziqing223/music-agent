import copy
import json
import tempfile
import unittest
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.repository import CanonicalRepository
from music_agent.snapshot import (
    SnapshotCompleteness,
    SnapshotRecord,
    SnapshotValidationError,
    SourceSnapshot,
    SourceSnapshotScope,
    apply_snapshot,
)
from music_agent.source_observation import ObservedValue, SourcePresence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
SCOPE_KEY = "library_tracks"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str) -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


def scope(
    completeness: SnapshotCompleteness = SnapshotCompleteness.COMPLETE,
    deletion_authoritative: bool = True,
    source_system: str = "apple_music",
    entity_type: EntityType = EntityType.TRACK,
    scope_key: str = SCOPE_KEY,
) -> SourceSnapshotScope:
    return SourceSnapshotScope(
        source_system, entity_type, scope_key, completeness, deletion_authoritative
    )


def record(external_id: str, fields: dict | None = None) -> SnapshotRecord:
    return SnapshotRecord(track_key(external_id), fields or {})


class SnapshotContractTest(unittest.TestCase):
    def test_invalid_scope_duplicate_identity_and_scope_mismatch_are_rejected(self) -> None:
        invalid_scopes = (
            lambda: SourceSnapshotScope("", EntityType.TRACK, SCOPE_KEY, SnapshotCompleteness.COMPLETE, True),
            lambda: SourceSnapshotScope("apple_music", EntityType.TRACK, "", SnapshotCompleteness.COMPLETE, True),
            lambda: SourceSnapshotScope("apple_music", EntityType.TRACK, SCOPE_KEY, "complete", True),
        )
        for factory in invalid_scopes:
            with self.subTest(factory=factory):
                with self.assertRaises(SnapshotValidationError):
                    factory()
        duplicate = record("SYNTH-TRACK-001")
        with self.assertRaisesRegex(SnapshotValidationError, "duplicate source identity"):
            SourceSnapshot(scope(), (duplicate, duplicate))
        wrong = SnapshotRecord(
            ExternalIdentityKey("other", EntityType.TRACK, "external"), {}
        )
        with self.assertRaisesRegex(SnapshotValidationError, "outside the declared scope"):
            SourceSnapshot(scope(), (wrong,))


class SnapshotApplicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()
        self.model["tracks"][0]["agent_metadata"]["tags"] = ["local-tag"]
        self.bound = {
            "SYNTH-TRACK-001": self.model["tracks"][0]["id"],
            "SYNTH-TRACK-002": self.model["tracks"][1]["id"],
            "SYNTH-TRACK-004": self.model["tracks"][3]["id"],
        }
        self.unbound_id = self.model["tracks"][2]["id"]
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def apply(self, snapshot: SourceSnapshot):
        with CanonicalRepository(self.database_path) as repository:
            return apply_snapshot(repository, snapshot)

    def presence(self, canonical_id: str) -> SourcePresence | None:
        with CanonicalRepository(self.database_path) as repository:
            return repository.get_source_presence(
                "apple_music", EntityType.TRACK, canonical_id, SCOPE_KEY
            )

    def test_complete_authoritative_omission_confirms_deletion_but_preserves_entity_and_binding(self) -> None:
        observed_id = self.bound["SYNTH-TRACK-001"]
        omitted_id = self.bound["SYNTH-TRACK-002"]
        self.apply(SourceSnapshot(scope(), tuple(record(external_id) for external_id in self.bound)))
        result = self.apply(SourceSnapshot(scope(), (record("SYNTH-TRACK-001"),)))
        self.assertIs(result.presence_outcomes[observed_id], SourcePresence.PRESENT)
        self.assertIs(result.presence_outcomes[omitted_id], SourcePresence.CONFIRMED_DELETED)
        self.assertIn(omitted_id, result.confirmed_deleted_canonical_ids)
        self.assertIs(self.presence(omitted_id), SourcePresence.CONFIRMED_DELETED)
        self.assertIsNone(self.presence(self.unbound_id))
        with CanonicalRepository(self.database_path) as repository:
            self.assertIn(omitted_id, {item["id"] for item in repository.load_model()["tracks"]})
            self.assertEqual(
                repository.lookup_external_identity(track_key("SYNTH-TRACK-002")), omitted_id
            )

    def test_binding_alone_is_not_scope_membership_or_deletion_evidence(self) -> None:
        omitted_id = self.bound["SYNTH-TRACK-002"]
        result = self.apply(SourceSnapshot(scope(), (record("SYNTH-TRACK-001"),)))
        self.assertNotIn(omitted_id, result.presence_outcomes)
        self.assertNotIn(omitted_id, result.confirmed_deleted_canonical_ids)
        self.assertIsNone(self.presence(omitted_id))

    def test_presence_membership_is_isolated_between_scope_keys(self) -> None:
        canonical_id = self.bound["SYNTH-TRACK-002"]
        scope_a = "library_tracks_a"
        scope_b = "library_tracks_b"
        self.apply(SourceSnapshot(scope(scope_key=scope_a), (record("SYNTH-TRACK-002"),)))
        result = self.apply(SourceSnapshot(scope(scope_key=scope_b), ()))
        with CanonicalRepository(self.database_path) as repository:
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, canonical_id, scope_a
                ),
                SourcePresence.PRESENT,
            )
            self.assertIsNone(repository.get_source_presence(
                "apple_music", EntityType.TRACK, canonical_id, scope_b
            ))
        self.assertNotIn(canonical_id, result.confirmed_deleted_canonical_ids)

    def test_partial_unknown_and_complete_non_authoritative_omission_preserve_durable_present(self) -> None:
        all_records = tuple(record(external_id) for external_id in self.bound)
        self.apply(SourceSnapshot(scope(), all_records))
        omitted_id = self.bound["SYNTH-TRACK-002"]
        cases = (
            (SnapshotCompleteness.PARTIAL, True, SourcePresence.MISSING),
            (SnapshotCompleteness.UNKNOWN, True, SourcePresence.UNKNOWN),
            (SnapshotCompleteness.COMPLETE, False, SourcePresence.MISSING),
        )
        for completeness, authoritative, runtime_presence in cases:
            with self.subTest(completeness=completeness, authoritative=authoritative):
                result = self.apply(SourceSnapshot(
                    scope(completeness, authoritative), (record("SYNTH-TRACK-001"),)
                ))
                self.assertIs(result.presence_outcomes[omitted_id], runtime_presence)
                self.assertIs(self.presence(omitted_id), SourcePresence.PRESENT)
                self.assertNotIn(omitted_id, result.confirmed_deleted_canonical_ids)

    def test_transient_omission_does_not_overwrite_durable_confirmed_deleted(self) -> None:
        omitted_id = self.bound["SYNTH-TRACK-002"]
        self.apply(SourceSnapshot(scope(), tuple(record(external_id) for external_id in self.bound)))
        self.apply(SourceSnapshot(scope(), (record("SYNTH-TRACK-001"),)))
        self.assertIs(self.presence(omitted_id), SourcePresence.CONFIRMED_DELETED)
        self.apply(SourceSnapshot(
            scope(SnapshotCompleteness.PARTIAL, True), (record("SYNTH-TRACK-001"),)
        ))
        self.assertIs(self.presence(omitted_id), SourcePresence.CONFIRMED_DELETED)

    def test_unresolved_source_record_does_not_create_identity_or_disable_deletion_inference(self) -> None:
        unknown_key = track_key("UNBOUND-SOURCE-ID")
        omitted_id = self.bound["SYNTH-TRACK-002"]
        self.apply(SourceSnapshot(scope(), tuple(record(external_id) for external_id in self.bound)))
        result = self.apply(SourceSnapshot(
            scope(), (record("SYNTH-TRACK-001"), SnapshotRecord(unknown_key, {}))
        ))
        self.assertEqual(result.unresolved_source_records, (unknown_key,))
        self.assertIn(omitted_id, result.confirmed_deleted_canonical_ids)
        with CanonicalRepository(self.database_path) as repository:
            self.assertIsNone(repository.lookup_external_identity(unknown_key))
            self.assertEqual(repository.counts()["canonical_entities"], 16)

    def test_metadata_uses_merge_shared_tags_preserve_and_unchanged_still_confirms_present(self) -> None:
        canonical_id = self.bound["SYNTH-TRACK-001"]
        changed = self.apply(SourceSnapshot(scope(), (record("SYNTH-TRACK-001", {
            "name": ObservedValue.value("Snapshot Name"),
            "agent_metadata.tags": ObservedValue.value(["source-tag"]),
        }),)))
        self.assertEqual(changed.resulting_model["tracks"][0]["name"], "Snapshot Name")
        self.assertEqual(changed.resulting_model["tracks"][0]["agent_metadata"]["tags"], ["local-tag"])
        self.assertTrue(changed.changed_fields)
        unchanged = self.apply(SourceSnapshot(scope(), (record("SYNTH-TRACK-001", {
            "name": ObservedValue.value("Snapshot Name"),
        }),)))
        self.assertEqual(unchanged.changed_fields, ())
        self.assertIs(self.presence(canonical_id), SourcePresence.PRESENT)

    def test_present_deleted_present_reappearance_uses_same_canonical_identity(self) -> None:
        external_id = "SYNTH-TRACK-001"
        canonical_id = self.bound[external_id]
        first = self.apply(SourceSnapshot(scope(), (record(external_id),)))
        self.assertIs(first.presence_outcomes[canonical_id], SourcePresence.PRESENT)
        self.apply(SourceSnapshot(scope(), ()))
        self.assertIs(self.presence(canonical_id), SourcePresence.CONFIRMED_DELETED)
        third = self.apply(SourceSnapshot(scope(), (record(external_id),)))
        self.assertIs(self.presence(canonical_id), SourcePresence.PRESENT)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.lookup_external_identity(track_key(external_id)), canonical_id)
            self.assertEqual(
                [item["id"] for item in repository.load_model()["tracks"]].count(canonical_id), 1
            )
        self.assertIs(third.presence_outcomes[canonical_id], SourcePresence.PRESENT)

    def test_invalid_merge_is_atomic_for_model_and_presence(self) -> None:
        original = copy.deepcopy(self.model)
        self.apply(SourceSnapshot(scope(), tuple(record(external_id) for external_id in self.bound)))
        before_presence = {
            canonical_id: self.presence(canonical_id) for canonical_id in self.bound.values()
        }
        invalid = SourceSnapshot(scope(), (
            record("SYNTH-TRACK-001", {"name": ObservedValue.value("Must Not Persist")}),
            record("SYNTH-TRACK-002", {
                "album_id": ObservedValue.value("alb_dddddddd-dddd-4ddd-8ddd-dddddddddddd")
            }),
        ))
        with CanonicalRepository(self.database_path) as repository:
            with self.assertRaises(SnapshotValidationError):
                apply_snapshot(repository, invalid)
            self.assertEqual(repository.load_model(), original)
        self.assertEqual(
            {canonical_id: self.presence(canonical_id) for canonical_id in self.bound.values()},
            before_presence,
        )

    def test_same_snapshot_is_idempotent_and_restart_recovers_model_and_presence(self) -> None:
        snapshot = SourceSnapshot(scope(), (record("SYNTH-TRACK-001", {
            "name": ObservedValue.value("Durable Snapshot Name")
        }),))
        first = self.apply(snapshot)
        second = self.apply(snapshot)
        self.assertTrue(first.changed_fields)
        self.assertEqual(second.changed_fields, ())
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model()["tracks"][0]["name"], "Durable Snapshot Name")
            records = repository.list_source_presence(
                "apple_music", EntityType.TRACK, SCOPE_KEY
            )
            self.assertEqual(
                [item.canonical_id for item in records],
                [self.bound["SYNTH-TRACK-001"]],
            )
            self.assertEqual(repository.counts()["external_identity_bindings"], 7)

    def test_unsupported_source_and_entity_type_fail_without_persistence(self) -> None:
        original = copy.deepcopy(self.model)
        cases = (
            SourceSnapshot(scope(source_system="other"), (SnapshotRecord(
                ExternalIdentityKey("other", EntityType.TRACK, "external"), {}
            ),)),
            SourceSnapshot(scope(entity_type=EntityType.ARTIST), (SnapshotRecord(
                ExternalIdentityKey("apple_music", EntityType.ARTIST, "external"), {}
            ),)),
        )
        for snapshot in cases:
            with self.subTest(scope=snapshot.scope):
                with CanonicalRepository(self.database_path) as repository:
                    with self.assertRaises(SnapshotValidationError):
                        apply_snapshot(repository, snapshot)
                    self.assertEqual(repository.load_model(), original)
                    self.assertEqual(
                        repository.list_source_presence(
                            "apple_music", EntityType.TRACK, SCOPE_KEY
                        ),
                        (),
                    )


if __name__ == "__main__":
    unittest.main()
