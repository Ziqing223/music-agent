"""E2-C: durable probe-derived verification evidence (record-only).

These tests prove that a clean ``VERIFIED`` + ``RESTORED`` ``set_favorited`` probe promotes into one
immutable historical ``CapabilityVerificationEvidence`` record, that the promotion predicate
re-derives the complete evidence shape rather than trusting two enum values, and that promotion is
idempotent, concurrency-safe, fail-closed, and record-only: it never mutates the probe row, the
canonical model, bindings, presence, intents, or attempts, performs no Music.app command/read, and
never changes the P01 capability matrix or ``is_execution_ready``.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from music_agent.capability_probe import (
    CapabilityProbe,
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    finalize,
    generate_probe_id,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_promotion import (
    CapabilityProbePromotionService,
    ProbeNotFoundError,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.capability_verification_evidence import (
    CapabilityVerificationEvidence,
    CapabilityVerificationEvidenceValidationError,
    PromotionNotEligibleError,
    SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION,
    evidence_from_probe,
    is_promotion_eligible,
)
from music_agent.capability_verification_evidence_repository import (
    CapabilityVerificationEvidenceRepository,
    EvidenceConflictError,
)
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import (
    CURRENT_SCHEMA_VERSION,
    CanonicalRepository,
    SourcePresenceRecord,
)
from music_agent.source_observation import ObservedValue, SourcePresence
from music_agent.write_execution import AttemptState
from music_agent.write_execution_repository import WriteExecutionRepository
from music_agent.write_intent import (
    DomainPermission,
    IntentState,
    WRITE_CAPABILITY_MATRIX,
    WriteOperation,
    create_scalar_pending_intent,
    is_execution_ready,
    resolve_capability,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
OTHER_TRACK_ID = "trk_22222222-2222-4222-8222-222222222222"
OTHER_TRACK_PID = "SYNTH-TRACK-002"
V8_MIGRATIONS = (
    (1, "0001_canonical_store.sql"),
    (2, "0002_source_presence.sql"),
    (3, "0003_ingestion_candidates.sql"),
    (4, "0004_pending_write_intents.sql"),
    (5, "0005_write_execution_attempts.sql"),
    (6, "0006_pending_write_intent_requirements.sql"),
    (7, "0007_capability_probes.sql"),
    (8, "0008_capability_probe_recovery_attempts.sql"),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def value(payload: bool) -> ObservedValue:
    return ObservedValue.value(payload)


def track_key() -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)


def favorited_intent():
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), value(True)
    )


def verified_probe(
    track_id: str = TRACK_ID, pid: str = TRACK_PID, f0: bool = False, d0: bool = False
) -> CapabilityProbe:
    """A clean VERIFIED + RESTORED probe for an arbitrary coverage class (F0, D0)."""
    probe = create_probe(track_id, pid, f0, d0)
    probe = mark_forward_started(probe)
    probe = observe_forward(probe, CommandOutcome.SUCCESS, value(not f0), value(d0))
    probe = mark_restore_started(probe)
    probe = observe_restore(probe, CommandOutcome.SUCCESS, value(f0), value(d0))
    return finalize(probe)


def failed_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID) -> CapabilityProbe:
    probe = mark_forward_started(create_probe(track_id, pid, False, False))
    return observe_forward(probe, CommandOutcome.FAILED, value(True), value(False))


def inconclusive_restored_probe() -> CapabilityProbe:
    probe = mark_inconclusive(create_probe(TRACK_ID, TRACK_PID, False, False))
    return mark_recovery_status(probe, RecoveryStatus.RESTORED)


def verified_non_restored_probe() -> CapabilityProbe:
    return mark_recovery_status(verified_probe(), RecoveryStatus.NEEDS_MANUAL_CHECK)


def malformed_verified_probe() -> CapabilityProbe:
    """VERIFIED + RESTORED but with the forward/restore observations missing (constructible)."""
    return CapabilityProbe(
        probe_id=generate_probe_id(),
        operation=WriteOperation.SET_FAVORITED,
        target_canonical_id=TRACK_ID,
        target_persistent_id=TRACK_PID,
        baseline_favorited=False,
        baseline_disliked=False,
        step_state=ProbeStepState.RESTORE_OBSERVED,
        recovery_status=RecoveryStatus.RESTORED,
        verification_verdict=VerificationVerdict.VERIFIED,
    )


class CapabilityVerificationEvidenceDomainTest(unittest.TestCase):
    # --- contract version ---------------------------------------------------

    def test_contract_version_is_the_initial_version_one(self) -> None:
        self.assertEqual(SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION, 1)

    # --- promotion predicate ------------------------------------------------

    def test_verified_restored_is_eligible(self) -> None:
        self.assertTrue(is_promotion_eligible(verified_probe()))

    def test_verified_d0_true_is_eligible_and_records_its_class(self) -> None:
        probe = verified_probe(f0=False, d0=True)
        self.assertTrue(is_promotion_eligible(probe))
        self.assertIs(probe.baseline_disliked, True)

    def test_failed_is_not_eligible(self) -> None:
        self.assertFalse(is_promotion_eligible(failed_probe()))

    def test_inconclusive_restored_is_not_eligible(self) -> None:
        self.assertFalse(is_promotion_eligible(inconclusive_restored_probe()))

    def test_verified_non_restored_is_not_eligible(self) -> None:
        probe = verified_non_restored_probe()
        self.assertIs(probe.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(probe.recovery_status, RecoveryStatus.NEEDS_MANUAL_CHECK)
        self.assertFalse(is_promotion_eligible(probe))

    def test_malformed_verified_is_not_eligible(self) -> None:
        probe = malformed_verified_probe()
        self.assertIs(probe.verification_verdict, VerificationVerdict.VERIFIED)
        self.assertIs(probe.recovery_status, RecoveryStatus.RESTORED)
        self.assertFalse(is_promotion_eligible(probe))

    def test_non_probe_is_not_eligible(self) -> None:
        self.assertFalse(is_promotion_eligible(None))
        self.assertFalse(is_promotion_eligible("verified"))

    # --- evidence model + builder ------------------------------------------

    def test_evidence_from_probe_copies_identity_and_coverage_class(self) -> None:
        probe = verified_probe(f0=False, d0=True)
        evidence = evidence_from_probe(probe, "2026-08-15T00:00:00+00:00")
        self.assertEqual(evidence.probe_id, probe.probe_id)
        self.assertIs(evidence.operation, WriteOperation.SET_FAVORITED)
        self.assertEqual(evidence.target_canonical_id, TRACK_ID)
        self.assertEqual(evidence.target_persistent_id, TRACK_PID)
        self.assertIs(evidence.baseline_favorited, False)  # F0
        self.assertIs(evidence.baseline_disliked, True)  # D0
        self.assertEqual(
            evidence.verification_contract_version,
            SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION,
        )
        self.assertEqual(evidence.verified_at, "2026-08-15T00:00:00+00:00")

    def test_evidence_from_probe_rejects_non_eligible(self) -> None:
        with self.assertRaises(PromotionNotEligibleError):
            evidence_from_probe(failed_probe(), "2026-08-15T00:00:00+00:00")

    def test_evidence_model_validation_fails_closed(self) -> None:
        good = evidence_from_probe(verified_probe(), "2026-08-15T00:00:00+00:00")
        base = {
            "probe_id": good.probe_id,
            "operation": good.operation,
            "target_canonical_id": good.target_canonical_id,
            "target_persistent_id": good.target_persistent_id,
            "baseline_favorited": good.baseline_favorited,
            "baseline_disliked": good.baseline_disliked,
            "verification_contract_version": good.verification_contract_version,
            "verified_at": good.verified_at,
        }
        bad_cases = [
            ("operation", WriteOperation.SET_DISLIKED),
            ("target_persistent_id", ""),
            ("baseline_favorited", 1),
            ("baseline_disliked", "yes"),
            ("verification_contract_version", 0),
            ("verification_contract_version", True),
            ("verified_at", ""),
        ]
        for field, bad in bad_cases:
            with self.assertRaises(CapabilityVerificationEvidenceValidationError, msg=field):
                CapabilityVerificationEvidence(**{**base, field: bad})


class CapabilityVerificationEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"
        self.model = load_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def save_probe(self, probe: CapabilityProbe) -> CapabilityProbe:
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
        return probe

    def promote(self, probe_id: str) -> CapabilityVerificationEvidence:
        with (
            CapabilityProbeRepository(self.database_path) as probe_repository,
            CapabilityVerificationEvidenceRepository(self.database_path) as evidence_repository,
        ):
            service = CapabilityProbePromotionService(probe_repository, evidence_repository)
            return service.promote_probe(probe_id)

    def evidence_rows(self) -> list[sqlite3.Row]:
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            return list(
                repository._connection.execute(
                    "SELECT * FROM capability_verification_evidence ORDER BY probe_id"
                )
            )

    # --- schema / migration -------------------------------------------------

    def test_fresh_database_reaches_v9_with_evidence_table(self) -> None:
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("capability_verification_evidence", tables)
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(capability_verification_evidence)"
                )
            }
            self.assertEqual(
                columns,
                {
                    "probe_id",
                    "operation",
                    "target_canonical_id",
                    "target_persistent_id",
                    "baseline_favorited",
                    "baseline_disliked",
                    "verification_contract_version",
                    "verified_at",
                },
            )

    def test_real_v8_store_upgrades_to_v9_without_changing_prior_state(self) -> None:
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        probe = verified_probe(track_id, TRACK_PID, False, False)
        with patch("music_agent.repository.MIGRATIONS", V8_MIGRATIONS):
            with CanonicalRepository(self.database_path) as repository:
                repository.save_model_with_source_presence(fixture, [presence])
                self.assertEqual(repository.schema_version, 8)
            with CapabilityProbeRepository(self.database_path) as repository:
                repository.save_probe(probe)
                self.assertEqual(repository.schema_version, 8)

        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), fixture)
            self.assertEqual(repository.lookup_external_identity(binding_key), track_id)
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, track_id, "library_tracks"
                ),
                SourcePresence.PRESENT,
            )
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(probe.probe_id), probe)

    # --- promotion: happy path + idempotency --------------------------------

    def test_verified_restored_promotes_to_evidence_row(self) -> None:
        probe = self.save_probe(verified_probe())
        evidence = self.promote(probe.probe_id)
        self.assertEqual(evidence.probe_id, probe.probe_id)
        self.assertIs(evidence.operation, WriteOperation.SET_FAVORITED)
        self.assertIs(evidence.baseline_favorited, False)
        self.assertIs(evidence.baseline_disliked, False)
        self.assertEqual(
            evidence.verification_contract_version,
            SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION,
        )
        self.assertTrue(evidence.verified_at)
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertEqual(repository.get_by_probe_id(probe.probe_id), evidence)

    def test_promotion_records_correct_f0_d0_coverage_class(self) -> None:
        probe = self.save_probe(verified_probe(f0=False, d0=True))
        evidence = self.promote(probe.probe_id)
        self.assertIs(evidence.baseline_favorited, False)  # F0
        self.assertIs(evidence.baseline_disliked, True)  # D0

    def test_promotion_records_correct_contract_version(self) -> None:
        probe = self.save_probe(verified_probe())
        evidence = self.promote(probe.probe_id)
        self.assertEqual(evidence.verification_contract_version, 1)
        self.assertEqual(
            evidence.verification_contract_version,
            SET_FAVORITED_PROBE_EVIDENCE_CONTRACT_VERSION,
        )

    def test_same_probe_promoted_twice_produces_one_row(self) -> None:
        probe = self.save_probe(verified_probe())
        first = self.promote(probe.probe_id)
        second = self.promote(probe.probe_id)
        self.assertEqual(first, second)
        self.assertEqual(len(self.evidence_rows()), 1)

    def test_get_by_probe_id_returns_none_for_unknown(self) -> None:
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertIsNone(repository.get_by_probe_id("prb_ffffffff-ffff-4fff-8fff-ffffffffffff"))

    def test_promote_unknown_probe_fails_closed(self) -> None:
        with self.assertRaises(ProbeNotFoundError):
            self.promote("prb_ffffffff-ffff-4fff-8fff-ffffffffffff")

    # --- promotion: rejection ----------------------------------------------

    def test_failed_probe_is_rejected(self) -> None:
        probe = self.save_probe(failed_probe())
        with self.assertRaises(PromotionNotEligibleError):
            self.promote(probe.probe_id)
        self.assertEqual(len(self.evidence_rows()), 0)

    def test_inconclusive_restored_probe_is_rejected(self) -> None:
        probe = self.save_probe(inconclusive_restored_probe())
        with self.assertRaises(PromotionNotEligibleError):
            self.promote(probe.probe_id)
        self.assertEqual(len(self.evidence_rows()), 0)

    def test_verified_non_restored_probe_is_rejected(self) -> None:
        probe = self.save_probe(verified_non_restored_probe())
        with self.assertRaises(PromotionNotEligibleError):
            self.promote(probe.probe_id)
        self.assertEqual(len(self.evidence_rows()), 0)

    def test_malformed_verified_probe_is_rejected(self) -> None:
        probe = self.save_probe(malformed_verified_probe())
        with self.assertRaises(PromotionNotEligibleError):
            self.promote(probe.probe_id)
        self.assertEqual(len(self.evidence_rows()), 0)

    # --- immutability / conflict -------------------------------------------

    def test_conflicting_evidence_signature_fails_closed(self) -> None:
        probe = self.save_probe(verified_probe(f0=False, d0=False))
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            repository._connection.execute(
                """INSERT INTO capability_verification_evidence(
                    probe_id, operation, target_canonical_id, target_persistent_id,
                    baseline_favorited, baseline_disliked, verification_contract_version, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    probe.probe_id,
                    "set_favorited",
                    TRACK_ID,
                    TRACK_PID,
                    0,
                    1,  # conflicting D0
                    1,
                    "2026-08-15T00:00:00+00:00",
                ),
            )
        with self.assertRaises(EvidenceConflictError):
            self.promote(probe.probe_id)

    # --- coverage query -----------------------------------------------------

    def test_evidence_query_distinguishes_d0_false_vs_d0_true(self) -> None:
        false_probe = self.save_probe(verified_probe(TRACK_ID, TRACK_PID, f0=False, d0=False))
        true_probe = self.save_probe(
            verified_probe(OTHER_TRACK_ID, OTHER_TRACK_PID, f0=False, d0=True)
        )
        false_evidence = self.promote(false_probe.probe_id)
        true_evidence = self.promote(true_probe.probe_id)

        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            d0_false = repository.list_evidence(baseline_disliked=False)
            d0_true = repository.list_evidence(baseline_disliked=True)
            all_evidence = repository.list_evidence()

        self.assertEqual([e.probe_id for e in d0_false], [false_evidence.probe_id])
        self.assertEqual([e.probe_id for e in d0_true], [true_evidence.probe_id])
        self.assertEqual(
            {e.probe_id for e in all_evidence},
            {false_evidence.probe_id, true_evidence.probe_id},
        )

    # --- concurrency --------------------------------------------------------

    def test_concurrent_same_probe_promotion_produces_one_row(self) -> None:
        probe = self.save_probe(verified_probe())
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, CURRENT_SCHEMA_VERSION)

        results = []
        errors = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            with (
                CapabilityProbeRepository(self.database_path) as probe_repository,
                CapabilityVerificationEvidenceRepository(self.database_path) as evidence_repository,
            ):
                service = CapabilityProbePromotionService(probe_repository, evidence_repository)
                try:
                    barrier.wait()
                    results.append(service.promote_probe(probe.probe_id))
                except Exception as error:
                    errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 2)
        self.assertEqual(len(errors), 0)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.evidence_rows()), 1)

    # --- isolation: no mutation of probe/canonical/operational state --------

    def seed_isolated_store(self):
        fixture = copy.deepcopy(self.model)
        track_id = fixture["tracks"][0]["id"]
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, track_id, "library_tracks", SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])
            before_counts = repository.counts()
            before_model = repository.load_model()
        intent = favorited_intent()
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
        probe = verified_probe(track_id, TRACK_PID, False, False)
        with CapabilityProbeRepository(self.database_path) as repository:
            repository.save_probe(probe)
        return {
            "track_id": track_id,
            "binding_key": binding_key,
            "intent": intent,
            "attempt": attempt,
            "probe": probe,
            "before_counts": before_counts,
            "before_model": before_model,
        }

    def test_promotion_mutates_no_probe_canonical_or_operational_state(self) -> None:
        seed = self.seed_isolated_store()
        evidence = self.promote(seed["probe"].probe_id)

        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), seed["before_counts"])
            self.assertEqual(repository.load_model(), seed["before_model"])
            self.assertEqual(
                repository.lookup_external_identity(seed["binding_key"]), seed["track_id"]
            )
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, seed["track_id"], "library_tracks"
                ),
                SourcePresence.PRESENT,
            )
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(seed["intent"].intent_id).state, IntentState.PENDING)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_attempt(seed["attempt"].attempt_id).state, AttemptState.STARTED)
        with CapabilityProbeRepository(self.database_path) as repository:
            self.assertEqual(repository.get_probe(seed["probe"].probe_id), seed["probe"])
        with CapabilityVerificationEvidenceRepository(self.database_path) as repository:
            self.assertEqual(repository.get_by_probe_id(seed["probe"].probe_id), evidence)

    # --- no external I/O ----------------------------------------------------

    def test_promotion_performs_no_music_app_command_or_read(self) -> None:
        probe = self.save_probe(verified_probe())
        # Promotion must not shell out to Music.app. Patch every subprocess/osascript entry point
        # so any attempted external command fails the test instead of touching the real system.
        with (
            patch("subprocess.run", side_effect=AssertionError("no subprocess during promotion")),
            patch("subprocess.Popen", side_effect=AssertionError("no subprocess during promotion")),
            patch("os.system", side_effect=AssertionError("no os.system during promotion")),
        ):
            evidence = self.promote(probe.probe_id)
        self.assertEqual(evidence.probe_id, probe.probe_id)

    # --- capability isolation -----------------------------------------------

    def test_matrix_unchanged_and_set_favorited_still_not_execution_ready(self) -> None:
        matrix_before = dict(WRITE_CAPABILITY_MATRIX)
        probe = self.save_probe(verified_probe())
        self.promote(probe.probe_id)

        self.assertEqual(dict(WRITE_CAPABILITY_MATRIX), matrix_before)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.adapter_implemented, True)
        self.assertIs(capability.readback_verified, False)
        self.assertIs(capability.readback_implemented, True)
        self.assertFalse(is_execution_ready(capability))
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))

    def test_d0_false_evidence_does_not_change_readiness(self) -> None:
        probe = self.save_probe(verified_probe(f0=False, d0=False))
        self.promote(probe.probe_id)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertFalse(is_execution_ready(capability))
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.readback_verified, False)


if __name__ == "__main__":
    unittest.main()
