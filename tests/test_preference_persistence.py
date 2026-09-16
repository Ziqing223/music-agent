"""P06 S10: durable preference persistence (head + immutable evidence revisions).

These tests prove the durable source-of-truth for preference signals: one mutable
``preference_signal_heads`` row per :class:`SignalIdentity` and an append-only
``preference_evidence_revisions`` history. They cover the frozen three-state boundary, the
BASELINE / TRANSITION / confirmation revision rules, atomic idempotency (retry- and
restart-safe), the optimistic CAS guard, revision immutability, migration, and the isolation of
persistence from the canonical entity tables and from every query-time derivation.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.direct_track_preference import (
    DirectPreferenceMagnitudePolicy,
    resolve_direct_track_preference,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.preference_attribution import (
    PreferenceAttributionError,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    PREFERENCE_EVIDENCE_CONTRACT_VERSION,
    EvidenceRevision,
    PreferencePersistenceValidationError,
    RecordObservationOutcome,
    RevisionKind,
    SignalHead,
    SignalIdentity,
    decode_semantic_value,
    encode_semantic_value,
)
from music_agent.preference_persistence_repository import (
    PreferencePersistenceRepository,
    PreferencePersistenceRepositoryError,
    StaleHeadStateError,
)
from music_agent.preference_signal import PreferenceSignal, normalize_preference_signal
from music_agent.preference_strength import PreferenceState
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS, CanonicalRepository
from music_agent.source_observation import ObservationState, ObservedValue

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
ALBUM_ID = "alb_11111111-1111-4111-8111-111111111111"
OBSERVED_AT = "2026-08-16T00:00:00+00:00"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def signal_identity(
    path: str = "favorited",
    source: str = "apple_music",
    target: PreferenceTargetReference | None = None,
) -> SignalIdentity:
    return SignalIdentity(target or track_target(), source, path)


def value(payload: object) -> ObservedValue:
    return ObservedValue.value(payload)


class PreferencePersistenceDomainTest(unittest.TestCase):
    # --- SignalIdentity ----------------------------------------------------

    def test_signal_identity_is_stable_and_equal_for_same_fields(self) -> None:
        first = signal_identity()
        second = SignalIdentity(track_target(), "apple_music", "favorited")
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(first.target.kind, PreferenceTargetKind.TRACK)
        self.assertEqual(first.target.target_id, TRACK_ID)

    def test_signal_identity_rejects_non_target(self) -> None:
        with self.assertRaises(PreferencePersistenceValidationError):
            SignalIdentity("not-a-target", "apple_music", "favorited")

    def test_signal_identity_rejects_empty_source_and_path(self) -> None:
        for source, path in (("", "favorited"), ("apple_music", "")):
            with self.assertRaises(PreferencePersistenceValidationError):
                SignalIdentity(track_target(), source, path)

    def test_signal_identity_rejects_non_canonical_track_key(self) -> None:
        with self.assertRaises(PreferenceAttributionError):
            SignalIdentity(
                PreferenceTargetReference(PreferenceTargetKind.TRACK, "not-a-track"),
                "apple_music",
                "favorited",
            )

    def test_signal_identity_accepts_genre_string_key(self) -> None:
        genre = PreferenceTargetReference(PreferenceTargetKind.GENRE, "ambient")
        self.assertEqual(
            SignalIdentity(genre, "apple_music", "rating").target.target_id, "ambient"
        )

    def test_signal_identity_accepts_artist_and_album_canonical_keys(self) -> None:
        for kind, target_id in (
            (PreferenceTargetKind.ARTIST, ARTIST_ID),
            (PreferenceTargetKind.ALBUM, ALBUM_ID),
        ):
            with self.subTest(kind=kind.value):
                SignalIdentity(PreferenceTargetReference(kind, target_id), "apple_music", "rating")

    # --- semantic value encoding -------------------------------------------

    def test_semantic_value_encoding_preserves_bool_int_str(self) -> None:
        for payload in (True, False, 0, 7, 100, "text"):
            with self.subTest(payload=payload):
                self.assertEqual(decode_semantic_value(encode_semantic_value(payload)), payload)

    def test_semantic_value_encoding_distinguishes_true_from_one(self) -> None:
        self.assertNotEqual(encode_semantic_value(True), encode_semantic_value(1))
        self.assertIs(decode_semantic_value(encode_semantic_value(True)), True)
        self.assertIs(decode_semantic_value(encode_semantic_value(1)), 1)

    def test_semantic_value_rejects_unsupported_types(self) -> None:
        for payload in (1.5, None, [True], {"a": 1}):
            with self.subTest(payload=payload):
                with self.assertRaises(PreferencePersistenceValidationError):
                    encode_semantic_value(payload)

    def test_decode_semantic_value_rejects_non_string_and_non_scalar(self) -> None:
        with self.assertRaises(PreferencePersistenceValidationError):
            decode_semantic_value(1)
        with self.assertRaises(PreferencePersistenceValidationError):
            decode_semantic_value("[1, 2]")

    # --- EvidenceRevision / SignalHead validation --------------------------

    def test_revision_kind_sequence_invariant(self) -> None:
        base = dict(
            identity=signal_identity(),
            revision_sequence=1,
            revision_kind=RevisionKind.BASELINE,
            semantic_value=True,
            observed_at=OBSERVED_AT,
            event_at=None,
            provenance=DIRECT_OBSERVATION_PROVENANCE,
            evidence_contract_version=PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        EvidenceRevision(**base)  # valid baseline
        with self.assertRaises(PreferencePersistenceValidationError):
            EvidenceRevision(**{**base, "revision_sequence": 2})  # baseline at seq 2
        with self.assertRaises(PreferencePersistenceValidationError):
            EvidenceRevision(**{
                **base, "revision_kind": RevisionKind.TRANSITION, "revision_sequence": 1,
            })

    def test_revision_rejects_non_scalar_semantic_value_and_empty_provenance(self) -> None:
        base = dict(
            identity=signal_identity(),
            revision_sequence=1,
            revision_kind=RevisionKind.BASELINE,
            semantic_value=True,
            observed_at=OBSERVED_AT,
            event_at=None,
            provenance=DIRECT_OBSERVATION_PROVENANCE,
            evidence_contract_version=PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        with self.assertRaises(PreferencePersistenceValidationError):
            EvidenceRevision(**{**base, "semantic_value": None})
        with self.assertRaises(PreferencePersistenceValidationError):
            EvidenceRevision(**{**base, "provenance": ""})
        with self.assertRaises(PreferencePersistenceValidationError):
            EvidenceRevision(**{**base, "evidence_contract_version": 0})

    def test_head_requires_value_for_value_state(self) -> None:
        base = dict(
            identity=signal_identity(),
            current_semantic_value=None,
            last_observed_state=ObservationState.VALUE,
            first_observed_at=OBSERVED_AT,
            last_observed_at=OBSERVED_AT,
            current_revision_sequence=0,
            evidence_contract_version=PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        with self.assertRaises(PreferencePersistenceValidationError):
            SignalHead(**base)

    def test_head_allows_value_retained_under_missing_state(self) -> None:
        SignalHead(
            identity=signal_identity(),
            current_semantic_value=True,
            last_observed_state=ObservationState.MISSING,
            first_observed_at=OBSERVED_AT,
            last_observed_at=OBSERVED_AT,
            current_revision_sequence=1,
            evidence_contract_version=PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )

    def test_record_observation_outcome_validates(self) -> None:
        head = SignalHead(
            identity=signal_identity(),
            current_semantic_value=True,
            last_observed_state=ObservationState.VALUE,
            first_observed_at=OBSERVED_AT,
            last_observed_at=OBSERVED_AT,
            current_revision_sequence=1,
            evidence_contract_version=PREFERENCE_EVIDENCE_CONTRACT_VERSION,
        )
        RecordObservationOutcome(head, None)
        with self.assertRaises(PreferencePersistenceValidationError):
            RecordObservationOutcome("not-a-head", None)


class PreferencePersistenceRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def head_rows(self, identity: SignalIdentity) -> list[sqlite3.Row]:
        with PreferencePersistenceRepository(self.database_path) as repository:
            return list(
                repository._connection.execute(
                    """SELECT * FROM preference_signal_heads
                    WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?""",
                    (
                        identity.target.kind.value,
                        identity.target.target_id,
                        identity.source_system,
                        identity.signal_path,
                    ),
                )
            )

    def revision_rows(self, identity: SignalIdentity) -> list[sqlite3.Row]:
        with PreferencePersistenceRepository(self.database_path) as repository:
            return list(
                repository._connection.execute(
                    """SELECT * FROM preference_evidence_revisions
                    WHERE target_kind=? AND target_key=? AND source_system=? AND signal_path=?
                    ORDER BY revision_sequence""",
                    (
                        identity.target.kind.value,
                        identity.target.target_id,
                        identity.source_system,
                        identity.signal_path,
                    ),
                )
            )

    # --- schema / migration ------------------------------------------------

    def test_fresh_database_reaches_current_version_with_persistence_tables(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("preference_signal_heads", tables)
            self.assertIn("preference_evidence_revisions", tables)

    def test_head_table_has_only_the_frozen_columns(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(preference_signal_heads)"
                )
            }
        self.assertEqual(
            columns,
            {
                "target_kind",
                "target_key",
                "source_system",
                "signal_path",
                "current_semantic_value_json",
                "last_observed_state",
                "first_observed_at",
                "last_observed_at",
                "current_revision_sequence",
                "evidence_contract_version",
                "created_at",
                "updated_at",
            },
        )

    def test_revision_table_has_only_the_frozen_columns(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(preference_evidence_revisions)"
                )
            }
        self.assertEqual(
            columns,
            {
                "target_kind",
                "target_key",
                "source_system",
                "signal_path",
                "revision_sequence",
                "revision_kind",
                "semantic_value_json",
                "observed_at",
                "event_at",
                "provenance",
                "evidence_contract_version",
            },
        )

    def test_v10_store_upgrades_to_v11_without_changing_prior_state(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        key = ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:10]):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model(fixture)
                self.assertEqual(repository.schema_version, 10)

        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(key), track_id)

    # --- identity uniqueness -----------------------------------------------

    def test_one_identity_yields_one_head_and_two_signal_paths_yield_two(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
        self.assertEqual(len(self.head_rows(identity)), 1)

        rating_identity = signal_identity(path="rating")
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(rating_identity, value(80), observed_at=OBSERVED_AT)
        self.assertEqual(len(self.head_rows(identity)), 1)
        self.assertEqual(len(self.head_rows(rating_identity)), 1)

    def test_head_identity_round_trips_exactly(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            head = repository.get_head(identity)
        self.assertIsNotNone(head)
        self.assertEqual(head.identity, identity)
        self.assertEqual(head.identity.target.kind, PreferenceTargetKind.TRACK)
        self.assertEqual(head.identity.target.target_id, TRACK_ID)

    # --- revision rules ----------------------------------------------------

    def test_first_sighting_creates_one_baseline(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            outcome = repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
        self.assertIsNotNone(outcome.revision)
        self.assertIs(outcome.revision.revision_kind, RevisionKind.BASELINE)
        self.assertEqual(outcome.revision.revision_sequence, 1)
        self.assertIs(outcome.revision.semantic_value, True)
        self.assertEqual(len(self.revision_rows(identity)), 1)

    def test_repeated_equal_observation_creates_no_second_revision(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            second = repository.record_observation(
                identity, value(True), observed_at="2026-08-16T00:00:01+00:00"
            )
        self.assertIsNone(second.revision)
        self.assertEqual(second.head.current_revision_sequence, 1)
        self.assertEqual(len(self.revision_rows(identity)), 1)

    def test_value_change_creates_one_transition(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            outcome = repository.record_observation(
                identity, value(False), observed_at="2026-08-16T00:00:01+00:00"
            )
        self.assertIs(outcome.revision.revision_kind, RevisionKind.TRANSITION)
        self.assertEqual(outcome.revision.revision_sequence, 2)
        self.assertIs(outcome.revision.semantic_value, False)
        self.assertEqual(len(self.revision_rows(identity)), 2)

    def test_type_change_is_a_value_change(self) -> None:
        identity = signal_identity(path="rating")
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            outcome = repository.record_observation(
                identity, value(1), observed_at="2026-08-16T00:00:01+00:00"
            )
        self.assertIs(outcome.revision.revision_kind, RevisionKind.TRANSITION)
        self.assertEqual(len(self.revision_rows(identity)), 2)

    def test_false_null_and_value_are_three_distinct_states(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(False), observed_at="t0")
            null_outcome = repository.record_observation(
                identity, ObservedValue.null(), observed_at="t1"
            )
            self.assertIsNone(null_outcome.revision)
            self.assertIs(null_outcome.head.last_observed_state, ObservationState.NULL)
            self.assertIs(null_outcome.head.current_semantic_value, False)
            value_outcome = repository.record_observation(
                identity, value(True), observed_at="t2"
            )
            self.assertIs(value_outcome.head.last_observed_state, ObservationState.VALUE)
        self.assertEqual(len(self.revision_rows(identity)), 2)  # False baseline, True transition

    def test_missing_creates_no_revision_and_first_value_is_still_baseline(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            missing = repository.record_observation(
                identity, ObservedValue.missing(), observed_at="t0"
            )
            self.assertIsNone(missing.revision)
            self.assertEqual(missing.head.current_revision_sequence, 0)
            self.assertIs(missing.head.last_observed_state, ObservationState.MISSING)

            baseline = repository.record_observation(identity, value(True), observed_at="t1")
            self.assertIs(baseline.revision.revision_kind, RevisionKind.BASELINE)
            self.assertEqual(baseline.revision.revision_sequence, 1)
        self.assertEqual(len(self.revision_rows(identity)), 1)

    # --- idempotency / crash safety ----------------------------------------

    def test_retry_after_commit_is_confirmation_noop(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
        with PreferencePersistenceRepository(self.database_path) as repository:
            retry = repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
        self.assertIsNone(retry.revision)
        self.assertEqual(retry.head.current_revision_sequence, 1)
        self.assertEqual(len(self.revision_rows(identity)), 1)

    def test_failed_transition_rolls_back_and_retry_produces_one_revision(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            repository._connection.execute(
                """CREATE TRIGGER fail_revision_insert BEFORE INSERT ON preference_evidence_revisions
                BEGIN SELECT RAISE(ABORT, 'simulated crash'); END"""
            )
            with self.assertRaises(sqlite3.IntegrityError):
                repository.record_observation(
                    identity, value(False), observed_at="2026-08-16T00:00:01+00:00"
                )
            # The whole transaction rolled back: head and revisions are unchanged.
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertIs(head.current_semantic_value, True)
            self.assertEqual(len(self.revision_rows(identity)), 1)

            repository._connection.execute("DROP TRIGGER fail_revision_insert")
            retry = repository.record_observation(
                identity, value(False), observed_at="2026-08-16T00:00:01+00:00"
            )
            self.assertIs(retry.revision.revision_kind, RevisionKind.TRANSITION)
            self.assertEqual(retry.revision.revision_sequence, 2)
        self.assertEqual(len(self.revision_rows(identity)), 2)

    # --- ordering / immutability / CAS -------------------------------------

    def test_revisions_list_in_deterministic_sequence_order(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at="t0")
            repository.record_observation(identity, value(False), observed_at="t1")
            repository.record_observation(identity, value(True), observed_at="t2")
            revisions = repository.list_revisions(identity)
        self.assertEqual([r.revision_sequence for r in revisions], [1, 2, 3])
        self.assertEqual(
            [r.revision_kind for r in revisions],
            [RevisionKind.BASELINE, RevisionKind.TRANSITION, RevisionKind.TRANSITION],
        )
        self.assertEqual([r.semantic_value for r in revisions], [True, False, True])

    def test_repository_has_no_revision_update_or_delete_path(self) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            self.assertFalse(hasattr(repository, "update_revision"))
            self.assertFalse(hasattr(repository, "delete_revision"))
            self.assertFalse(hasattr(repository, "update_head"))

    def test_immutable_revision_cannot_be_rewritten_or_deleted(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE preference_evidence_revisions SET semantic_value_json='false'"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "DELETE FROM preference_evidence_revisions"
                )
        self.assertEqual(len(self.revision_rows(identity)), 1)

    def test_stale_expected_sequence_fails_closed(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            with self.assertRaises(StaleHeadStateError):
                repository.record_observation(
                    identity, value(False), observed_at="t1", expected_revision_sequence=0
                )
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertIs(head.current_semantic_value, True)
        self.assertEqual(len(self.revision_rows(identity)), 1)

    def test_matching_expected_sequence_proceeds(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            outcome = repository.record_observation(
                identity, value(False), observed_at="t1", expected_revision_sequence=1
            )
            self.assertIs(outcome.revision.revision_kind, RevisionKind.TRANSITION)

    # --- reload ------------------------------------------------------------

    def test_head_and_revisions_reload_after_reopen(self) -> None:
        identity = signal_identity()
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at="t0")
            repository.record_observation(identity, value(False), observed_at="t1")

        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            revisions = repository.list_revisions(identity)

        self.assertEqual(head.current_revision_sequence, 2)
        self.assertIs(head.current_semantic_value, False)
        self.assertIs(head.last_observed_state, ObservationState.VALUE)
        self.assertEqual(len(revisions), 2)
        self.assertEqual(revisions[1].semantic_value, False)

    # --- isolation from canonical + derived state --------------------------

    def test_preference_persistence_touches_no_canonical_tables(self) -> None:
        fixture = load_fixture()
        track_id = fixture["tracks"][0]["id"]
        identity = SignalIdentity(
            PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
            "apple_music",
            "favorited",
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
            before_counts = repository.counts()
            before_model = repository.load_model()

        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(identity, value(True), observed_at=OBSERVED_AT)
            repository.record_observation(identity, value(False), observed_at="t1")

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)
            self.assertEqual(repository.load_model(), before_model)

    def test_no_query_time_derivations_are_materialized(self) -> None:
        forbidden = (
            "recency",
            "influence",
            "recent_preference",
            "current_preference",
            "confidence",
            "familiarity",
            "temporal_decay",
        )
        with PreferencePersistenceRepository(self.database_path) as repository:
            tables = {
                str(row[0])
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            columns = {
                str(row[1])
                for table in ("preference_signal_heads", "preference_evidence_revisions")
                for row in repository._connection.execute(f"PRAGMA table_info({table})")
            }
        for name in forbidden:
            self.assertNotIn(name, tables)
            self.assertFalse(any(name in column for column in columns), name)

    # --- S1--S9 derivation unchanged ---------------------------------------

    def test_s1_s3_derivation_chain_is_unchanged(self) -> None:
        observed = ObservedValue.value(True)
        contribution = normalize_preference_signal(PreferenceSignal.FAVORITED, observed)
        conclusion = resolve_direct_track_preference(
            [contribution], DirectPreferenceMagnitudePolicy(0.9, 0.9)
        )
        self.assertIs(conclusion.state, PreferenceState.POSITIVE)
        self.assertIs(contribution.direction.value, "positive")


if __name__ == "__main__":
    unittest.main()
