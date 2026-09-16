import json
import tempfile
import unittest
from pathlib import Path

from music_agent.apple_music import AppleMusicReadError, AppleMusicSourceAdapter
from music_agent.capability_probe import (
    CommandOutcome,
    ProbeStepState,
    RecoveryStatus,
    VerificationVerdict,
    create_probe,
    finalize,
    mark_forward_started,
    mark_inconclusive,
    mark_recovery_status,
    mark_restore_started,
    observe_forward,
    observe_restore,
)
from music_agent.capability_probe_preflight import (
    CapabilityProbePreflightService,
    PreflightRejectionReason,
)
from music_agent.capability_probe_recovery_attempt_repository import (
    CapabilityProbeRecoveryAttemptRepository,
)
from music_agent.capability_probe_repository import CapabilityProbeRepository
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import CanonicalRepository, SourcePresenceRecord
from music_agent.source_observation import ObservedValue, SourcePresence
from music_agent.write_execution import AttemptState
from music_agent.write_execution_repository import WriteExecutionRepository
from music_agent.write_intent import (
    DomainPermission,
    IntentState,
    WriteOperation,
    create_scalar_pending_intent,
    is_execution_ready,
    resolve_capability,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
SCOPE_KEY = "library_tracks"

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_PID = "SYNTH-TRACK-001"
OTHER_TRACK_ID = "trk_22222222-2222-4222-8222-222222222222"
NO_BINDING_TRACK_ID = "trk_33333333-3333-4333-8333-333333333333"
PRESENTLESS_TRACK_ID = "trk_44444444-4444-4444-8444-444444444444"
ALBUM_ID = "alb_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
MISSING_TRACK_ID = "trk_99999999-9999-4999-8999-999999999999"

FOUND_FALSE_FALSE = (
    '{"status":"found","fields":{"name":"Synthetic Duet","favorited":false,"disliked":false,'
    '"rating":0,"played_count":0}}'
)
FOUND_FALSE_TRUE = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":true,'
    '"rating":0,"played_count":0}}'
)
FOUND_TRUE_FALSE = (
    '{"status":"found","fields":{"name":"X","favorited":true,"disliked":false,'
    '"rating":0,"played_count":0}}'
)
FOUND_FAVORITED_NULL = (
    '{"status":"found","fields":{"name":"X","favorited":null,"disliked":false}}'
)
FOUND_DISLIKED_NULL = (
    '{"status":"found","fields":{"name":"X","favorited":false,"disliked":null}}'
)
FOUND_FAVORITED_NON_BOOL = (
    '{"status":"found","fields":{"name":"X","favorited":123,"disliked":false}}'
)
CONFIRMED_NOT_FOUND = '{"status":"confirmed_not_found"}'


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeReadRunner:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, persistent_id: str) -> str:
        self.calls.append(persistent_id)
        if self.error is not None:
            raise self.error
        if self.output is None:
            raise AppleMusicReadError("no read output configured")
        return self.output


class EmptyBindingCanonicalRepository:
    """Fake canonical read port whose binding yields an empty persistent ID."""

    def get_entity_type(self, canonical_id: str) -> EntityType | None:
        return EntityType.TRACK if canonical_id == TRACK_ID else None

    def get_bound_external_id(
        self, source_system: str, entity_type: EntityType, canonical_id: str
    ) -> str | None:
        return ""

    def get_source_presence(
        self, source_system: str, entity_type: EntityType, canonical_id: str, scope_key: str
    ) -> SourcePresence | None:
        return SourcePresence.PRESENT


def pending_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    return create_probe(track_id, pid, False, False)


def inconclusive_needs_manual_check(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = mark_inconclusive(create_probe(track_id, pid, False, False))
    return mark_recovery_status(probe, RecoveryStatus.NEEDS_MANUAL_CHECK)


def inconclusive_restored(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = mark_inconclusive(create_probe(track_id, pid, False, False))
    return mark_recovery_status(probe, RecoveryStatus.RESTORED)


def inconclusive_baseline_confirmed(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    return mark_inconclusive(create_probe(track_id, pid, False, False))


def failed_probe(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = mark_forward_started(create_probe(track_id, pid, False, False))
    # Forward command succeeded but the readback did not flip -> deterministic FAILED.
    return observe_forward(
        probe, CommandOutcome.SUCCESS, ObservedValue.value(False), ObservedValue.value(False)
    )


def verified_restored(track_id: str = TRACK_ID, pid: str = TRACK_PID):
    probe = create_probe(track_id, pid, False, False)
    probe = mark_forward_started(probe)
    probe = observe_forward(
        probe, CommandOutcome.SUCCESS, ObservedValue.value(True), ObservedValue.value(False)
    )
    probe = mark_restore_started(probe)
    probe = observe_restore(
        probe, CommandOutcome.SUCCESS, ObservedValue.value(False), ObservedValue.value(False)
    )
    return finalize(probe)


class PreflightTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _seed_canonical(self) -> None:
        fixture = load_fixture()
        presence = SourcePresenceRecord(
            "apple_music", EntityType.TRACK, TRACK_ID, SCOPE_KEY, SourcePresence.PRESENT
        )
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model_with_source_presence(fixture, [presence])

    def _check_target(
        self,
        canonical_track_id: str,
        *,
        read_output: str | None = None,
        read_error: Exception | None = None,
        seed_probes: tuple = (),
        canonical_repository=None,
    ):
        source_adapter = AppleMusicSourceAdapter(FakeReadRunner(read_output, read_error))
        with CapabilityProbeRepository(self.database_path) as probes, CapabilityProbeRecoveryAttemptRepository(
            self.database_path
        ) as attempts:
            for probe in seed_probes:
                probes.save_probe(probe)
            if canonical_repository is None:
                with CanonicalRepository(self.database_path) as canonical:
                    service = CapabilityProbePreflightService(
                        canonical, probes, attempts, source_adapter, SCOPE_KEY
                    )
                    return service.check_target(canonical_track_id)
            service = CapabilityProbePreflightService(
                canonical_repository, probes, attempts, source_adapter, SCOPE_KEY
            )
            return service.check_target(canonical_track_id)


class CapabilityProbePreflightEligibilityTest(PreflightTestBase):
    def setUp(self) -> None:
        super().setUp()
        self._seed_canonical()

    def test_valid_track_is_eligible_with_persistent_id_and_baseline(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        self.assertTrue(result.eligible)
        self.assertEqual(result.canonical_track_id, TRACK_ID)
        self.assertEqual(result.target_persistent_id, TRACK_PID)
        self.assertIs(result.baseline_favorited, False)
        self.assertIs(result.baseline_disliked, False)
        self.assertIsNone(result.rejection_reason)

    def test_baseline_favorited_false_is_exact_value(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_FALSE_TRUE)
        self.assertTrue(result.eligible)
        self.assertIs(result.baseline_favorited, False)
        self.assertIs(result.baseline_disliked, True)

    def test_baseline_disliked_false_is_exact_value(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_TRUE_FALSE)
        self.assertTrue(result.eligible)
        self.assertIs(result.baseline_favorited, True)
        self.assertIs(result.baseline_disliked, False)

    def test_canonical_entity_missing_rejects(self) -> None:
        result = self._check_target(MISSING_TRACK_ID)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.CANONICAL_ENTITY_NOT_FOUND)

    def test_wrong_entity_type_rejects(self) -> None:
        result = self._check_target(ALBUM_ID)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.NOT_A_TRACK)

    def test_binding_missing_rejects(self) -> None:
        result = self._check_target(NO_BINDING_TRACK_ID)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.APPLE_MUSIC_BINDING_MISSING)

    def test_empty_binding_rejects(self) -> None:
        result = self._check_target(
            TRACK_ID, canonical_repository=EmptyBindingCanonicalRepository()
        )
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.PERSISTENT_ID_EMPTY)

    def test_presence_not_present_rejects(self) -> None:
        # PRESENTLESS_TRACK_ID has a binding but no source-presence record in this seed.
        result = self._check_target(PRESENTLESS_TRACK_ID)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.SOURCE_PRESENCE_NOT_PRESENT)

    def test_live_read_not_found_rejects(self) -> None:
        result = self._check_target(TRACK_ID, read_output=CONFIRMED_NOT_FOUND)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.LIVE_READ_NOT_FOUND)

    def test_live_lookup_failed_rejects(self) -> None:
        result = self._check_target(TRACK_ID, read_error=AppleMusicReadError("boom"))
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.LIVE_LOOKUP_FAILED)

    def test_favorited_missing_rejects(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_FAVORITED_NULL)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.FAVORITED_MISSING)

    def test_disliked_missing_rejects(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_DISLIKED_NULL)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.DISLIKED_MISSING)

    def test_non_bool_field_rejects(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_FAVORITED_NON_BOOL)
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.FAVORITED_NOT_BOOL)


class CapabilityProbePreflightExistingProbeTest(PreflightTestBase):
    def setUp(self) -> None:
        super().setUp()
        self._seed_canonical()

    def test_pending_same_target_blocks(self) -> None:
        result = self._check_target(TRACK_ID, seed_probes=(pending_probe(),))
        self.assertFalse(result.eligible)
        self.assertIs(result.rejection_reason, PreflightRejectionReason.EXISTING_PROBE_PENDING)

    def test_inconclusive_needs_manual_check_blocks(self) -> None:
        result = self._check_target(
            TRACK_ID, seed_probes=(inconclusive_needs_manual_check(),)
        )
        self.assertFalse(result.eligible)
        self.assertIs(
            result.rejection_reason, PreflightRejectionReason.EXISTING_PROBE_UNRESOLVED_SOURCE
        )

    def test_failed_same_target_blocks(self) -> None:
        result = self._check_target(TRACK_ID, seed_probes=(failed_probe(),))
        self.assertFalse(result.eligible)
        self.assertIs(
            result.rejection_reason, PreflightRejectionReason.EXISTING_PROBE_UNRESOLVED_SOURCE
        )

    def test_started_recovery_attempt_blocks(self) -> None:
        probe = pending_probe()
        with CapabilityProbeRepository(self.database_path) as probes, CapabilityProbeRecoveryAttemptRepository(
            self.database_path
        ) as attempts:
            probes.save_probe(probe)
            attempts.begin_attempt(probe.probe_id)
        result = self._check_target(TRACK_ID, seed_probes=())
        self.assertFalse(result.eligible)
        self.assertIs(
            result.rejection_reason, PreflightRejectionReason.EXISTING_PROBE_STARTED_ATTEMPT
        )

    def test_verified_restored_does_not_block(self) -> None:
        result = self._check_target(
            TRACK_ID, read_output=FOUND_FALSE_FALSE, seed_probes=(verified_restored(),)
        )
        self.assertTrue(result.eligible)

    def test_safe_inconclusive_restored_does_not_block(self) -> None:
        result = self._check_target(
            TRACK_ID, read_output=FOUND_FALSE_FALSE, seed_probes=(inconclusive_restored(),)
        )
        self.assertTrue(result.eligible)

    def test_safe_inconclusive_baseline_confirmed_does_not_block(self) -> None:
        result = self._check_target(
            TRACK_ID,
            read_output=FOUND_FALSE_FALSE,
            seed_probes=(inconclusive_baseline_confirmed(),),
        )
        self.assertTrue(result.eligible)

    def test_different_target_probe_does_not_block(self) -> None:
        other_probe = pending_probe(track_id=OTHER_TRACK_ID, pid="SYNTH-TRACK-002")
        result = self._check_target(
            TRACK_ID, read_output=FOUND_FALSE_FALSE, seed_probes=(other_probe,)
        )
        self.assertTrue(result.eligible)


class CapabilityProbePreflightIsolationTest(PreflightTestBase):
    def setUp(self) -> None:
        super().setUp()
        self._seed_canonical()

    def _seed_intent_and_attempt(self) -> dict:
        binding_key = ExternalIdentityKey("apple_music", EntityType.TRACK, TRACK_PID)
        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED, TRACK_ID, binding_key, ObservedValue.value(True)
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)
        with WriteExecutionRepository(self.database_path) as repository:
            attempt = repository.begin_execution(intent.intent_id)
        return {"intent": intent, "attempt": attempt, "binding_key": binding_key}

    def test_identity_resolved_from_durable_binding_only(self) -> None:
        result = self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        # The persistent ID equals the durable binding's external_id, not the track name or any
        # canonical external_ids projection the caller might otherwise guess from.
        self.assertEqual(result.target_persistent_id, TRACK_PID)

    def test_no_name_or_fuzzy_lookup(self) -> None:
        read_runner = FakeReadRunner(output=FOUND_FALSE_FALSE)
        source_adapter = AppleMusicSourceAdapter(read_runner)
        with CanonicalRepository(self.database_path) as canonical, CapabilityProbeRepository(
            self.database_path
        ) as probes, CapabilityProbeRecoveryAttemptRepository(self.database_path) as attempts:
            service = CapabilityProbePreflightService(
                canonical, probes, attempts, source_adapter, SCOPE_KEY
            )
            service.check_target(TRACK_ID)
        # The source read is keyed solely on the persistent ID from the binding, never a name.
        self.assertEqual(read_runner.calls, [TRACK_PID])

    def test_check_target_performs_no_db_writes(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            before_counts = repository.counts()
        self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.counts(), before_counts)

    def test_no_command_invocation_dependency(self) -> None:
        source_adapter = AppleMusicSourceAdapter(FakeReadRunner(output=FOUND_FALSE_FALSE))
        with CanonicalRepository(self.database_path) as canonical, CapabilityProbeRepository(
            self.database_path
        ) as probes, CapabilityProbeRecoveryAttemptRepository(self.database_path) as attempts:
            service = CapabilityProbePreflightService(
                canonical, probes, attempts, source_adapter, SCOPE_KEY
            )
            attributes = set(vars(service))
        # Only read dependencies plus the scope key; no command runner, no write repository.
        self.assertEqual(
            attributes,
            {
                "_canonical_repository",
                "_probe_repository",
                "_recovery_attempt_repository",
                "_source_adapter",
                "_scope_key",
            },
        )

    def test_canonical_binding_presence_unchanged(self) -> None:
        with CanonicalRepository(self.database_path) as repository:
            before_model = repository.load_model()
            before_binding = repository.get_bound_external_id(
                "apple_music", EntityType.TRACK, TRACK_ID
            )
            before_presence = repository.get_source_presence(
                "apple_music", EntityType.TRACK, TRACK_ID, SCOPE_KEY
            )
        self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(repository.load_model(), before_model)
            self.assertEqual(
                repository.get_bound_external_id("apple_music", EntityType.TRACK, TRACK_ID),
                before_binding,
            )
            self.assertIs(
                repository.get_source_presence(
                    "apple_music", EntityType.TRACK, TRACK_ID, SCOPE_KEY
                ),
                before_presence,
            )

    def test_intents_attempts_probes_unchanged(self) -> None:
        seed = self._seed_intent_and_attempt()
        probe = pending_probe()
        with CapabilityProbeRepository(self.database_path) as probes:
            probes.save_probe(probe)
        self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        with PendingIntentRepository(self.database_path) as repository:
            self.assertIs(repository.get_intent(seed["intent"].intent_id).state, IntentState.PENDING)
        with WriteExecutionRepository(self.database_path) as repository:
            self.assertIs(repository.get_attempt(seed["attempt"].attempt_id).state, AttemptState.STARTED)
        with CapabilityProbeRepository(self.database_path) as probes:
            self.assertEqual(probes.get_probe(probe.probe_id), probe)

    def test_capability_matrix_unchanged(self) -> None:
        self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        capability = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(capability.domain_permission, DomainPermission.ALLOWED)
        self.assertIs(capability.capability_verified, False)
        self.assertIs(capability.readback_verified, False)
        self.assertTrue(capability.adapter_implemented)
        self.assertTrue(capability.readback_implemented)

    def test_execution_ready_remains_false(self) -> None:
        self._check_target(TRACK_ID, read_output=FOUND_FALSE_FALSE)
        self.assertFalse(is_execution_ready(resolve_capability(WriteOperation.SET_FAVORITED)))


if __name__ == "__main__":
    unittest.main()
