import copy
import dataclasses
import json
import unittest
from pathlib import Path

from music_agent.identity import EntityType, ExternalIdentityKey, validate_canonical_id
from music_agent.source_observation import ObservedValue, ObservationState
from music_agent.write_intent import (
    RELATION_WRITE_VALUE,
    DomainPermission,
    IntentState,
    PendingIntent,
    ReadbackDecision,
    ReadbackStrategy,
    RequirementRole,
    UnsupportedWriteOperation,
    WriteDomainPermissionUnspecified,
    WriteEvent,
    WriteIntentValidationError,
    WriteNotDomainWritable,
    WriteOperation,
    WriteRequirement,
    WriteTransitionError,
    advance_intent,
    apply_readback,
    create_pending_intent,
    create_scalar_pending_intent,
    evaluate_readback,
    generate_intent_id,
    is_execution_ready,
    mark_command_failed,
    mark_command_succeeded,
    resolve_capability,
    validate_intent_id,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
PLAYLIST_ID = "pl_55555555-5555-4555-8555-555555555555"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def track_key(external_id: str = "SYNTH-TRACK-001") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.TRACK, external_id)


def playlist_key(external_id: str = "SYNTH-PLAYLIST-1") -> ExternalIdentityKey:
    return ExternalIdentityKey("apple_music", EntityType.PLAYLIST, external_id)


def playlist_requirement(
    canonical_id: str = PLAYLIST_ID, external_id: str = "SYNTH-PLAYLIST-1"
) -> WriteRequirement:
    return WriteRequirement(
        RequirementRole.PLAYLIST,
        canonical_id,
        ExternalIdentityKey("apple_music", EntityType.PLAYLIST, external_id),
    )


def track_requirement(
    canonical_id: str = TRACK_ID, external_id: str = "SYNTH-TRACK-001"
) -> WriteRequirement:
    return WriteRequirement(
        RequirementRole.TRACK,
        canonical_id,
        ExternalIdentityKey("apple_music", EntityType.TRACK, external_id),
    )


def membership_intent(
    playlist: WriteRequirement | None = None, track: WriteRequirement | None = None
) -> PendingIntent:
    return create_pending_intent(
        WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
        (playlist or playlist_requirement(), track or track_requirement()),
        RELATION_WRITE_VALUE,
    )


def favorited_intent(value: object = True) -> PendingIntent:
    return create_scalar_pending_intent(
        WriteOperation.SET_FAVORITED,
        TRACK_ID,
        track_key(),
        ObservedValue.value(value),
    )


class WriteCapabilityMatrixTest(unittest.TestCase):
    def test_allowed_domain_and_capability_verified_are_independent(self) -> None:
        favorited = resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertIs(favorited.domain_permission, DomainPermission.ALLOWED)
        self.assertFalse(favorited.capability_verified)
        self.assertEqual(favorited.entity_type, EntityType.TRACK)
        self.assertEqual(favorited.field_path, "library_state.favorited")

    def test_playlist_writes_are_historically_capability_verified(self) -> None:
        for operation in (
            WriteOperation.CREATE_PLAYLIST,
            WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            WriteOperation.DELETE_PLAYLIST,
        ):
            capability = resolve_capability(operation)
            self.assertTrue(capability.capability_verified, operation.value)
        # Only the add-membership command adapter is implemented; create/delete stay unimplemented.
        self.assertTrue(resolve_capability(WriteOperation.ADD_PLAYLIST_MEMBERSHIP).adapter_implemented)
        self.assertFalse(resolve_capability(WriteOperation.CREATE_PLAYLIST).adapter_implemented)
        self.assertFalse(resolve_capability(WriteOperation.DELETE_PLAYLIST).adapter_implemented)

    def test_track_field_writes_are_not_capability_verified(self) -> None:
        for operation in (
            WriteOperation.SET_FAVORITED,
            WriteOperation.SET_DISLIKED,
            WriteOperation.SET_RATING,
        ):
            self.assertFalse(
                resolve_capability(operation).capability_verified,
                f"{operation.value} was only read, never written, by P01",
            )

    def test_remove_membership_is_not_capability_verified(self) -> None:
        self.assertFalse(resolve_capability(WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP).capability_verified)

    def test_adapter_implemented_set_is_membership_and_favorited(self) -> None:
        implemented = {
            WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            WriteOperation.SET_FAVORITED,
            # P11.3: the MusicKit library-add transport is implemented and deterministic-tested.
            WriteOperation.ADD_LIBRARY_SONG,
        }
        for operation in WriteOperation:
            capability = resolve_capability(operation)
            self.assertEqual(
                capability.adapter_implemented,
                operation in implemented,
                operation.value,
            )

    def test_only_favorited_readback_is_implemented(self) -> None:
        for operation in WriteOperation:
            capability = resolve_capability(operation)
            self.assertEqual(
                capability.readback_implemented,
                operation in {
                    WriteOperation.SET_FAVORITED,
                    # P11.3: catalog-id relationship readback is implemented and tested.
                    WriteOperation.ADD_LIBRARY_SONG,
                },
                operation.value,
            )

    def test_no_write_is_execution_ready(self) -> None:
        for operation in WriteOperation:
            self.assertFalse(is_execution_ready(resolve_capability(operation)))

    def test_allowed_operations_are_explicitly_allowed(self) -> None:
        for operation in (
            WriteOperation.SET_FAVORITED,
            WriteOperation.SET_DISLIKED,
            WriteOperation.SET_RATING,
            WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP,
        ):
            self.assertIs(
                resolve_capability(operation).domain_permission,
                DomainPermission.ALLOWED,
                operation.value,
            )

    def test_disallowed_fields_are_explicitly_disallowed(self) -> None:
        for operation in (
            WriteOperation.SET_PLAY_COUNT,
            WriteOperation.SET_SKIP_COUNT,
            WriteOperation.SET_LAST_PLAYED_AT,
            WriteOperation.SET_ADDED_TO_LIBRARY_AT,
            WriteOperation.SET_ARTIST_IDS,
            WriteOperation.SET_ALBUM_ID,
        ):
            self.assertIs(
                resolve_capability(operation).domain_permission,
                DomainPermission.DISALLOWED,
                operation.value,
            )

    def test_playlist_entity_create_delete_are_domain_unspecified(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.DELETE_PLAYLIST):
            capability = resolve_capability(operation)
            self.assertIs(capability.domain_permission, DomainPermission.UNSPECIFIED)
            self.assertTrue(capability.capability_verified)

    def test_playlist_readback_is_historically_verified(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.ADD_PLAYLIST_MEMBERSHIP):
            capability = resolve_capability(operation)
            self.assertTrue(capability.readback_verified, operation.value)
            self.assertEqual(capability.readback_strategy, ReadbackStrategy.READ_PLAYLIST_CONTENTS)

    def test_delete_playlist_readback_is_not_verified(self) -> None:
        delete = resolve_capability(WriteOperation.DELETE_PLAYLIST)
        self.assertTrue(delete.capability_verified)
        self.assertFalse(delete.readback_verified)
        self.assertEqual(delete.readback_strategy, ReadbackStrategy.READ_PLAYLIST_ABSENCE)

    def test_track_field_write_readback_is_not_verified(self) -> None:
        for operation in (
            WriteOperation.SET_FAVORITED,
            WriteOperation.SET_DISLIKED,
            WriteOperation.SET_RATING,
        ):
            capability = resolve_capability(operation)
            self.assertFalse(capability.readback_verified)
            self.assertEqual(capability.readback_strategy, ReadbackStrategy.READ_FIELD)

    def test_writable_track_fields_have_read_field_readback(self) -> None:
        for operation, field in (
            (WriteOperation.SET_FAVORITED, "library_state.favorited"),
            (WriteOperation.SET_DISLIKED, "library_state.disliked"),
            (WriteOperation.SET_RATING, "library_state.rating"),
        ):
            capability = resolve_capability(operation)
            self.assertEqual(capability.readback_strategy, ReadbackStrategy.READ_FIELD)
            self.assertEqual(capability.readback_field_path, field)

    def test_unverifiable_track_writes_have_no_readback(self) -> None:
        for operation in (
            WriteOperation.SET_SKIP_COUNT,
            WriteOperation.SET_ARTIST_IDS,
            WriteOperation.SET_ALBUM_ID,
        ):
            capability = resolve_capability(operation)
            self.assertEqual(capability.readback_strategy, ReadbackStrategy.UNAVAILABLE)
            self.assertIsNone(capability.readback_field_path)

    def test_unsupported_operation_fails_closed(self) -> None:
        with self.assertRaises(UnsupportedWriteOperation):
            resolve_capability("set_name")
        with self.assertRaises(UnsupportedWriteOperation):
            create_scalar_pending_intent(
                "set_name", TRACK_ID, track_key(), ObservedValue.value(True)
            )

    def test_disallowed_operation_cannot_form_an_intent(self) -> None:
        with self.assertRaises(WriteNotDomainWritable):
            create_scalar_pending_intent(
                WriteOperation.SET_PLAY_COUNT, TRACK_ID, track_key(), ObservedValue.value(5)
            )

    def test_unspecified_domain_operation_cannot_form_an_intent(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.DELETE_PLAYLIST):
            with self.subTest(operation=operation.value):
                with self.assertRaises(WriteDomainPermissionUnspecified):
                    create_scalar_pending_intent(
                        operation,
                        PLAYLIST_ID,
                        playlist_key(),
                        ObservedValue.value(True),
                    )

    def test_unspecified_is_never_execution_ready_even_if_verified_and_implemented(self) -> None:
        for operation in (WriteOperation.CREATE_PLAYLIST, WriteOperation.DELETE_PLAYLIST):
            capability = resolve_capability(operation)
            hypothetical = dataclasses.replace(
                capability,
                capability_verified=True,
                adapter_implemented=True,
                readback_implemented=True,
            )
            self.assertIs(hypothetical.domain_permission, DomainPermission.UNSPECIFIED)
            self.assertTrue(hypothetical.capability_verified)
            self.assertTrue(hypothetical.adapter_implemented)
            self.assertFalse(is_execution_ready(hypothetical))

    def test_execution_ready_requires_readback_implementation(self) -> None:
        favorited = resolve_capability(WriteOperation.SET_FAVORITED)
        without_readback = dataclasses.replace(
            favorited,
            capability_verified=True,
            adapter_implemented=True,
            readback_implemented=False,
        )
        self.assertIs(without_readback.domain_permission, DomainPermission.ALLOWED)
        self.assertTrue(without_readback.capability_verified)
        self.assertTrue(without_readback.adapter_implemented)
        self.assertFalse(without_readback.readback_implemented)
        self.assertFalse(is_execution_ready(without_readback))

        with_readback = dataclasses.replace(without_readback, readback_implemented=True)
        self.assertTrue(with_readback.readback_implemented)
        self.assertTrue(is_execution_ready(with_readback))


class RequirementContractTest(unittest.TestCase):
    def test_scalar_intent_carries_exactly_one_target_requirement(self) -> None:
        intent = favorited_intent(True)
        self.assertEqual(len(intent.requirements), 1)
        self.assertIs(intent.requirements[0].role, RequirementRole.TARGET)
        self.assertEqual(intent.requirements[0].canonical_id, TRACK_ID)

    def test_add_membership_intent_carries_playlist_and_track(self) -> None:
        intent = membership_intent()
        self.assertEqual(
            {requirement.role for requirement in intent.requirements},
            {RequirementRole.PLAYLIST, RequirementRole.TRACK},
        )

    def test_missing_playlist_role_fails_closed(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (track_requirement(),),
                RELATION_WRITE_VALUE,
            )

    def test_missing_track_role_fails_closed(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (playlist_requirement(),),
                RELATION_WRITE_VALUE,
            )

    def test_duplicate_role_fails_closed(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (playlist_requirement(), playlist_requirement(), track_requirement()),
                RELATION_WRITE_VALUE,
            )

    def test_extra_role_fails_closed(self) -> None:
        extra = WriteRequirement(
            RequirementRole.TARGET, TRACK_ID, ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001")
        )
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (playlist_requirement(), track_requirement(), extra),
                RELATION_WRITE_VALUE,
            )

    def test_playlist_role_with_track_entity_fails_closed(self) -> None:
        bad_playlist = WriteRequirement(
            RequirementRole.PLAYLIST,
            TRACK_ID,
            ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001"),
        )
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (bad_playlist, track_requirement()),
                RELATION_WRITE_VALUE,
            )

    def test_track_role_with_playlist_entity_fails_closed(self) -> None:
        bad_track = WriteRequirement(
            RequirementRole.TRACK,
            PLAYLIST_ID,
            ExternalIdentityKey("apple_music", EntityType.PLAYLIST, "SYNTH-PLAYLIST-1"),
        )
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
                (playlist_requirement(), bad_track),
                RELATION_WRITE_VALUE,
            )

    def test_scalar_intent_rejects_relation_roles(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.SET_FAVORITED,
                (track_requirement(),),
                ObservedValue.value(True),
            )

    def test_scalar_intent_rejects_multiple_requirements(self) -> None:
        target = WriteRequirement(
            RequirementRole.TARGET, TRACK_ID, track_key()
        )
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.SET_FAVORITED,
                (target, target),
                ObservedValue.value(True),
            )

    def test_requirement_order_does_not_affect_semantics(self) -> None:
        forward = PendingIntent(
            intent_id="int_11111111-1111-4111-8111-111111111111",
            operation=WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            requirements=(playlist_requirement(), track_requirement()),
            requested_value=RELATION_WRITE_VALUE,
        )
        reversed_order = PendingIntent(
            intent_id="int_11111111-1111-4111-8111-111111111111",
            operation=WriteOperation.ADD_PLAYLIST_MEMBERSHIP,
            requirements=(track_requirement(), playlist_requirement()),
            requested_value=RELATION_WRITE_VALUE,
        )
        self.assertEqual(forward, reversed_order)
        self.assertEqual(
            [requirement.role for requirement in forward.requirements],
            [RequirementRole.PLAYLIST, RequirementRole.TRACK],
        )

    def test_relation_requirements_keep_canonical_and_external_identity_distinct(self) -> None:
        intent = membership_intent()
        playlist = next(r for r in intent.requirements if r.role is RequirementRole.PLAYLIST)
        self.assertEqual(playlist.canonical_id, PLAYLIST_ID)
        self.assertEqual(playlist.external_identity.external_id, "SYNTH-PLAYLIST-1")
        self.assertNotEqual(playlist.canonical_id, playlist.external_identity.external_id)

    def test_remove_membership_relation_payload_is_deferred(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_pending_intent(
                WriteOperation.REMOVE_PLAYLIST_MEMBERSHIP,
                (playlist_requirement(), track_requirement()),
                RELATION_WRITE_VALUE,
            )


class PendingIntentIdentityTest(unittest.TestCase):
    def test_intent_id_is_independent_of_canonical_and_external_identity(self) -> None:
        intent = favorited_intent(True)
        requirement = intent.requirements[0]
        self.assertNotEqual(intent.intent_id, requirement.canonical_id)
        self.assertNotEqual(intent.intent_id, requirement.external_identity.external_id)
        self.assertTrue(intent.intent_id.startswith("int_"))
        self.assertNotIn("trk_", intent.intent_id)
        self.assertNotIn("pm_", intent.intent_id)

    def test_intent_and_canonical_namespaces_are_disjoint(self) -> None:
        intent_id = generate_intent_id()
        validate_intent_id(intent_id)
        with self.assertRaises(WriteIntentValidationError):
            validate_intent_id(TRACK_ID)  # canonical trk_ id is not an intent id
        with self.assertRaises(Exception):
            validate_canonical_id(EntityType.TRACK, intent_id)  # intent id is not a canonical id

    def test_two_intents_have_distinct_identities(self) -> None:
        self.assertNotEqual(favorited_intent(True).intent_id, favorited_intent(False).intent_id)

    def test_wrong_entity_type_external_identity_fails_closed(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_scalar_pending_intent(
                WriteOperation.SET_FAVORITED,
                PLAYLIST_ID,
                ExternalIdentityKey("apple_music", EntityType.PLAYLIST, "SYNTH-PLAYLIST-1"),
                ObservedValue.value(True),
            )

    def test_non_apple_music_external_identity_fails_closed(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_scalar_pending_intent(
                WriteOperation.SET_FAVORITED,
                TRACK_ID,
                ExternalIdentityKey("spotify", EntityType.TRACK, "x"),
                ObservedValue.value(True),
            )

    def test_missing_requested_value_is_rejected(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            create_scalar_pending_intent(
                WriteOperation.SET_FAVORITED, TRACK_ID, track_key(), ObservedValue.missing()
            )


class TransitionTest(unittest.TestCase):
    def test_command_success_moves_to_awaiting_readback_not_confirmed(self) -> None:
        intent = mark_command_succeeded(favorited_intent(True))
        self.assertIs(intent.state, IntentState.AWAITING_READBACK)

    def test_command_failure_moves_to_execution_failed(self) -> None:
        intent = mark_command_failed(favorited_intent(True))
        self.assertIs(intent.state, IntentState.EXECUTION_FAILED)

    def test_command_unknown_moves_to_outcome_unknown(self) -> None:
        intent = advance_intent(favorited_intent(True), WriteEvent.COMMAND_UNKNOWN)
        self.assertIs(intent.state, IntentState.OUTCOME_UNKNOWN)

    def test_outcome_unknown_reconciles_on_matching_readback(self) -> None:
        unknown = advance_intent(favorited_intent(True), WriteEvent.COMMAND_UNKNOWN)
        confirmed = advance_intent(unknown, WriteEvent.READBACK_MATCHED)
        self.assertIs(confirmed.state, IntentState.CONFIRMED)

    def test_outcome_unknown_rejects_mismatched_readback(self) -> None:
        unknown = advance_intent(favorited_intent(True), WriteEvent.COMMAND_UNKNOWN)
        with self.assertRaises(WriteTransitionError):
            advance_intent(unknown, WriteEvent.READBACK_MISMATCHED)

    def test_outcome_unknown_rejects_command_events(self) -> None:
        unknown = advance_intent(favorited_intent(True), WriteEvent.COMMAND_UNKNOWN)
        for event in (
            WriteEvent.COMMAND_SUCCEEDED,
            WriteEvent.COMMAND_FAILED,
            WriteEvent.COMMAND_UNKNOWN,
        ):
            with self.subTest(event=event):
                with self.assertRaises(WriteTransitionError):
                    advance_intent(unknown, event)

    def test_matching_readback_confirms(self) -> None:
        intent = apply_readback(
            mark_command_succeeded(favorited_intent(False)), ObservedValue.value(False)
        )
        self.assertIs(intent.state, IntentState.CONFIRMED)

    def test_mismatching_readback_is_mismatch_not_failure(self) -> None:
        intent = apply_readback(
            mark_command_succeeded(favorited_intent(True)), ObservedValue.value(False)
        )
        self.assertIs(intent.state, IntentState.READBACK_MISMATCH)

    def test_null_requested_value_confirms_on_null_readback(self) -> None:
        intent = apply_readback(
            mark_command_succeeded(
                create_scalar_pending_intent(
                    WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.null()
                )
            ),
            ObservedValue.null(),
        )
        self.assertIs(intent.state, IntentState.CONFIRMED)

    def test_no_readback_cannot_confirm(self) -> None:
        intent = mark_command_succeeded(favorited_intent(True))
        with self.assertRaises(WriteTransitionError):
            apply_readback(intent, ObservedValue.missing())

    def test_readback_before_command_fails_closed(self) -> None:
        with self.assertRaises(WriteTransitionError):
            apply_readback(favorited_intent(True), ObservedValue.value(True))
        with self.assertRaises(WriteTransitionError):
            advance_intent(favorited_intent(True), WriteEvent.READBACK_MATCHED)

    def test_terminal_states_accept_no_further_events(self) -> None:
        confirmed = apply_readback(
            mark_command_succeeded(favorited_intent(True)), ObservedValue.value(True)
        )
        for event in WriteEvent:
            with self.subTest(state="confirmed", event=event):
                with self.assertRaises(WriteTransitionError):
                    advance_intent(confirmed, event)

        failed = mark_command_failed(favorited_intent(True))
        for event in WriteEvent:
            with self.subTest(state="execution_failed", event=event):
                with self.assertRaises(WriteTransitionError):
                    advance_intent(failed, event)

    def test_transition_does_not_mutate_input_intent(self) -> None:
        intent = favorited_intent(True)
        original = copy.deepcopy(intent)
        advanced = mark_command_succeeded(intent)
        self.assertIs(intent.state, IntentState.PENDING)
        self.assertEqual(intent, original)
        self.assertIsNot(advanced, intent)

    def test_transition_and_evaluation_do_not_mutate_model(self) -> None:
        model = load_fixture()
        snapshot = copy.deepcopy(model)
        intent = mark_command_succeeded(favorited_intent(True))
        apply_readback(intent, ObservedValue.value(True))
        evaluate_readback(ObservedValue.value(0), ObservedValue.value(0))
        resolve_capability(WriteOperation.SET_FAVORITED)
        self.assertEqual(model, snapshot)


class ReadbackEvaluationTest(unittest.TestCase):
    def test_false_zero_and_null_are_explicit_not_missing(self) -> None:
        self.assertEqual(
            evaluate_readback(ObservedValue.value(False), ObservedValue.value(False)),
            ReadbackDecision.MATCHED,
        )
        self.assertEqual(
            evaluate_readback(ObservedValue.value(0), ObservedValue.value(0)),
            ReadbackDecision.MATCHED,
        )
        self.assertEqual(
            evaluate_readback(ObservedValue.null(), ObservedValue.null()),
            ReadbackDecision.MATCHED,
        )

    def test_false_is_not_mistaken_for_missing(self) -> None:
        self.assertEqual(
            evaluate_readback(ObservedValue.value(False), ObservedValue.missing()),
            ReadbackDecision.UNAVAILABLE,
        )

    def test_value_vs_null_disagreement_is_mismatch(self) -> None:
        self.assertEqual(
            evaluate_readback(ObservedValue.value(0), ObservedValue.null()),
            ReadbackDecision.MISMATCHED,
        )
        self.assertEqual(
            evaluate_readback(ObservedValue.null(), ObservedValue.value(0)),
            ReadbackDecision.MISMATCHED,
        )

    def test_missing_requested_value_is_rejected(self) -> None:
        with self.assertRaises(WriteIntentValidationError):
            evaluate_readback(ObservedValue.missing(), ObservedValue.value(True))


class IntentValueContractTest(unittest.TestCase):
    def test_false_zero_and_null_requested_values_are_preserved(self) -> None:
        false_intent = favorited_intent(False)
        self.assertIs(false_intent.requested_value.state, ObservationState.VALUE)
        self.assertIs(false_intent.requested_value.payload, False)

        zero_intent = create_scalar_pending_intent(
            WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.value(0)
        )
        self.assertEqual(zero_intent.requested_value.payload, 0)

        null_intent = create_scalar_pending_intent(
            WriteOperation.SET_RATING, TRACK_ID, track_key(), ObservedValue.null()
        )
        self.assertIs(null_intent.requested_value.state, ObservationState.NULL)


if __name__ == "__main__":
    unittest.main()
