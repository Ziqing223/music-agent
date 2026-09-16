"""P08.1: the feedback observation contract for the feedback-learning loop.

These tests prove the pure domain contract only: stable ``fbk_`` identity, the kind vocabulary
with its frozen explicitness/direction mappings, the target / recommendation-reference /
attribution axes, provenance, the injected timezone-aware timestamps, the duplicate-key
semantics, the assembly seam, and the canonical JSON round-trip. No interpretation, no learning
effect, no P06 mutation, and no persistence is implemented here, and nothing touches SQLite.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import UUID

from music_agent.feedback_contract import (
    FEEDBACK_CONTRACT_VERSION,
    AttributionRelation,
    FeedbackAttribution,
    FeedbackContractValidationError,
    FeedbackDirection,
    FeedbackExplicitness,
    FeedbackKind,
    FeedbackObservation,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
    decode_feedback_observation,
    encode_feedback_observation,
    feedback_duplicate_key,
    generate_feedback_id,
    validate_feedback_id,
)
from music_agent.identity import ENTITY_ID_PREFIX
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def artist_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def source(
    system: str = "recommendation_ui", path: str = "card_actions"
) -> FeedbackSourceReference:
    return FeedbackSourceReference(system, path)


def recommendation_ref() -> FeedbackRecommendationReference:
    return FeedbackRecommendationReference(RUN_ID, CANDIDATE_ID)


def observation(
    *,
    kind: FeedbackKind = FeedbackKind.LIKED,
    target: PreferenceTargetReference | None = None,
    recommendation: FeedbackRecommendationReference | None = None,
    attribution: FeedbackAttribution | None = None,
    event_at: datetime | None = None,
    source_event_id: str | None = None,
) -> FeedbackObservation:
    return assemble_feedback_observation(
        feedback_id=generate_feedback_id(),
        kind=kind,
        source=source(),
        observed_at=NOW,
        target=target,
        recommendation=recommendation,
        attribution=attribution,
        event_at=event_at,
        source_event_id=source_event_id,
    )


class FeedbackIdentityTest(unittest.TestCase):
    def test_feedback_id_uses_fbk_namespace_and_uuid4(self) -> None:
        feedback_id = generate_feedback_id()
        self.assertTrue(feedback_id.startswith("fbk_"))
        self.assertEqual(UUID(feedback_id[len("fbk_") :]).version, 4)
        validate_feedback_id(feedback_id)

    def test_generated_ids_do_not_reuse_in_process(self) -> None:
        feedback_ids = {generate_feedback_id() for _ in range(1000)}
        self.assertEqual(len(feedback_ids), 1000)

    def test_namespace_is_outside_every_existing_namespace(self) -> None:
        feedback_id = generate_feedback_id()
        for prefix in ENTITY_ID_PREFIX.values():
            self.assertFalse(feedback_id.startswith(prefix))
        for prefix in ("int_", "att_", "prb_", "rec_", "rcm_", "cnd_"):
            self.assertFalse(feedback_id.startswith(prefix))

    def test_validate_rejects_wrong_namespace(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            validate_feedback_id(f"rcm_{feedback_id_suffix()}")

    def test_validate_rejects_non_uuid_suffix(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            validate_feedback_id("fbk_not-a-uuid")

    def test_validate_rejects_non_string(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            validate_feedback_id(12345)


def feedback_id_suffix() -> str:
    return generate_feedback_id()[len("fbk_") :]


class FeedbackVocabularyTest(unittest.TestCase):
    EXPLICIT_DIRECTIONS = {
        FeedbackKind.LIKED: FeedbackDirection.POSITIVE,
        FeedbackKind.DIRECTION_GOOD: FeedbackDirection.POSITIVE,
        FeedbackKind.DISLIKED: FeedbackDirection.NEGATIVE,
        FeedbackKind.CORRECTED: FeedbackDirection.NONE,
        FeedbackKind.ATTRIBUTION_CORRECTION: FeedbackDirection.NONE,
    }
    IMPLICIT_KINDS = (
        FeedbackKind.FAVORITED,
        FeedbackKind.SKIPPED,
        FeedbackKind.REPLAYED,
        FeedbackKind.COMPLETED,
        FeedbackKind.PLAYED,
    )

    def test_explicit_kinds_are_explicit_with_frozen_direction(self) -> None:
        for kind, direction in self.EXPLICIT_DIRECTIONS.items():
            with self.subTest(kind=kind):
                attribution = (
                    FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED)
                    if kind is FeedbackKind.ATTRIBUTION_CORRECTION
                    else None
                )
                self.assertEqual(
                    observation(kind=kind, target=track_target(), attribution=attribution).explicitness,
                    FeedbackExplicitness.EXPLICIT,
                )
                self.assertEqual(
                    observation(kind=kind, target=track_target(), attribution=attribution).direction,
                    direction,
                )

    def test_implicit_kinds_are_implicit_and_carry_no_directional_claim(self) -> None:
        for kind in self.IMPLICIT_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(observation(kind=kind, target=track_target()).explicitness,
                                 FeedbackExplicitness.IMPLICIT)
                self.assertEqual(observation(kind=kind, target=track_target()).direction,
                                 FeedbackDirection.NONE)


class FeedbackObservationValidationTest(unittest.TestCase):
    def test_kind_is_required_and_typed(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind="liked",
                source=source(),
                observed_at=NOW,
                target=track_target(),
            )

    def test_source_is_required_and_typed(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind=FeedbackKind.LIKED,
                source="recommendation_ui",
                observed_at=NOW,
                target=track_target(),
            )

    def test_observed_at_must_be_timezone_aware(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind=FeedbackKind.LIKED,
                source=source(),
                observed_at=datetime(2026, 8, 16, 0, 0, 0),
                target=track_target(),
            )

    def test_requires_target_or_recommendation(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            observation(kind=FeedbackKind.LIKED)

    def test_target_alone_is_enough(self) -> None:
        obs = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        self.assertIsNotNone(obs.target)
        self.assertIsNone(obs.recommendation)

    def test_recommendation_alone_is_enough(self) -> None:
        obs = observation(kind=FeedbackKind.DIRECTION_GOOD, recommendation=recommendation_ref())
        self.assertIsNone(obs.target)
        self.assertIsNotNone(obs.recommendation)

    def test_target_and_recommendation_can_both_be_present(self) -> None:
        obs = observation(
            kind=FeedbackKind.SKIPPED,
            target=track_target(),
            recommendation=recommendation_ref(),
        )
        self.assertIsNotNone(obs.target)
        self.assertIsNotNone(obs.recommendation)

    def test_target_is_type_checked(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind=FeedbackKind.LIKED,
                source=source(),
                observed_at=NOW,
                target=TRACK_ID,
            )

    def test_genre_target_uses_non_canonical_string_key(self) -> None:
        genre = PreferenceTargetReference(PreferenceTargetKind.GENRE, "jazz")
        obs = observation(kind=FeedbackKind.DIRECTION_GOOD, target=genre)
        self.assertEqual(obs.target, genre)

    def test_event_at_must_be_timezone_aware_when_present(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            observation(
                kind=FeedbackKind.PLAYED,
                target=track_target(),
                event_at=datetime(2026, 8, 15, 23, 0, 0),
            )

    def test_source_event_id_must_be_non_empty_when_present(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            observation(kind=FeedbackKind.PLAYED, target=track_target(), source_event_id="")

    def test_contract_version_must_be_positive_int(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind=FeedbackKind.LIKED,
                source=source(),
                observed_at=NOW,
                target=track_target(),
                contract_version=0,
            )
        with self.assertRaises(FeedbackContractValidationError):
            FeedbackObservation(
                feedback_id=generate_feedback_id(),
                kind=FeedbackKind.LIKED,
                source=source(),
                observed_at=NOW,
                target=track_target(),
                contract_version=True,
            )

    def test_observation_is_immutable(self) -> None:
        obs = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        with self.assertRaises(FrozenInstanceError):
            obs.target = track_target()

    def test_equal_observations_are_hashable_records(self) -> None:
        first = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        second = FeedbackObservation(
            feedback_id=first.feedback_id,
            kind=first.kind,
            source=first.source,
            observed_at=first.observed_at,
            target=first.target,
            contract_version=first.contract_version,
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))


class FeedbackAttributionTest(unittest.TestCase):
    def test_attributed_relation_claims_the_aspect(self) -> None:
        attribution = FeedbackAttribution(artist_target(), AttributionRelation.ATTRIBUTED)
        obs = observation(
            kind=FeedbackKind.LIKED, target=track_target(), attribution=attribution
        )
        self.assertEqual(obs.attribution, attribution)

    def test_excluded_relation_disclaims_the_aspect(self) -> None:
        attribution = FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED)
        obs = observation(
            kind=FeedbackKind.ATTRIBUTION_CORRECTION,
            target=track_target(),
            attribution=attribution,
        )
        self.assertEqual(obs.attribution.relation, AttributionRelation.EXCLUDED)

    def test_attribution_correction_requires_attribution(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            observation(kind=FeedbackKind.ATTRIBUTION_CORRECTION, target=track_target())

    def test_attribution_aspect_validates_canonical_identity(self) -> None:
        with self.assertRaises(ValueError):
            FeedbackAttribution(
                PreferenceTargetReference(PreferenceTargetKind.ARTIST, "not-an-artist-id"),
                AttributionRelation.EXCLUDED,
            )

    def test_attribution_is_orthogonal_to_direction(self) -> None:
        obs = observation(
            kind=FeedbackKind.LIKED,
            target=track_target(),
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.EXCLUDED),
        )
        self.assertEqual(obs.direction, FeedbackDirection.POSITIVE)


class FeedbackRecommendationReferenceTest(unittest.TestCase):
    def test_reference_preserves_p07_identity_pair(self) -> None:
        ref = recommendation_ref()
        self.assertEqual(ref.run_id, RUN_ID)
        self.assertEqual(ref.candidate_id, CANDIDATE_ID)

    def test_invalid_run_id_is_rejected_by_the_p07_validator(self) -> None:
        with self.assertRaises(ValueError):
            FeedbackRecommendationReference("trk_11111111-1111-4111-8111-111111111111", CANDIDATE_ID)

    def test_invalid_candidate_id_is_rejected_by_the_p07_validator(self) -> None:
        with self.assertRaises(ValueError):
            FeedbackRecommendationReference(RUN_ID, "rcm_22222222-2222-4222-8222-222222222222")


class FeedbackAssemblyTest(unittest.TestCase):
    def test_assembly_stamps_the_current_contract_version(self) -> None:
        obs = observation(kind=FeedbackKind.LIKED, target=track_target())
        self.assertEqual(obs.contract_version, FEEDBACK_CONTRACT_VERSION)

    def test_assembly_is_deterministic_for_the_same_inputs(self) -> None:
        def build() -> FeedbackObservation:
            return assemble_feedback_observation(
                feedback_id="fbk_44444444-4444-4444-8444-444444444444",
                kind=FeedbackKind.LIKED,
                source=source(),
                observed_at=NOW,
                target=track_target(),
            )

        self.assertEqual(build(), build())


class FeedbackDuplicateKeyTest(unittest.TestCase):
    def test_source_event_id_is_the_strong_dedupe_anchor(self) -> None:
        first = observation(
            kind=FeedbackKind.SKIPPED,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        second = observation(
            kind=FeedbackKind.SKIPPED,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        self.assertEqual(feedback_duplicate_key(first), feedback_duplicate_key(second))

    def test_source_scopes_the_source_event_id(self) -> None:
        first = observation(
            kind=FeedbackKind.SKIPPED,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        other_source = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=FeedbackKind.SKIPPED,
            source=FeedbackSourceReference("watch_app", "card_actions"),
            observed_at=NOW,
            target=track_target(),
            source_event_id="playback_evt_7",
        )
        self.assertNotEqual(feedback_duplicate_key(first), feedback_duplicate_key(other_source))

    def test_without_source_event_id_the_full_fields_are_the_key(self) -> None:
        first = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        same_fields = FeedbackObservation(
            feedback_id=generate_feedback_id(),
            kind=FeedbackKind.SKIPPED,
            source=first.source,
            observed_at=first.observed_at,
            target=first.target,
            contract_version=first.contract_version,
        )
        self.assertEqual(feedback_duplicate_key(first), feedback_duplicate_key(same_fields))

    def test_distinct_instants_are_distinct_events(self) -> None:
        first = observation(kind=FeedbackKind.PLAYED, target=track_target())
        later = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=FeedbackKind.PLAYED,
            source=first.source,
            observed_at=datetime(2026, 8, 16, 1, 0, 0, tzinfo=timezone.utc),
            target=track_target(),
        )
        self.assertNotEqual(feedback_duplicate_key(first), feedback_duplicate_key(later))

    def test_distinct_kinds_are_distinct_events(self) -> None:
        skipped = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        completed = observation(kind=FeedbackKind.COMPLETED, target=track_target())
        self.assertNotEqual(
            feedback_duplicate_key(skipped), feedback_duplicate_key(completed)
        )

    def test_rejects_non_observation(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            feedback_duplicate_key("not an observation")


class FeedbackSerializationTest(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        obs = observation(
            kind=FeedbackKind.LIKED,
            target=track_target(),
            recommendation=recommendation_ref(),
            attribution=FeedbackAttribution(artist_target(), AttributionRelation.ATTRIBUTED),
            event_at=datetime(2026, 8, 15, 23, 0, 0, tzinfo=timezone.utc),
            source_event_id="ui_evt_9",
        )
        decoded = decode_feedback_observation(encode_feedback_observation(obs))
        self.assertEqual(decoded, obs)

    def test_round_trip_preserves_optional_none_fields(self) -> None:
        obs = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        decoded = decode_feedback_observation(encode_feedback_observation(obs))
        self.assertIsNone(decoded.recommendation)
        self.assertIsNone(decoded.attribution)
        self.assertIsNone(decoded.event_at)
        self.assertIsNone(decoded.source_event_id)
        self.assertEqual(decoded, obs)

    def test_encoding_is_deterministic_and_valid_json(self) -> None:
        obs = observation(kind=FeedbackKind.LIKED, target=track_target())
        self.assertEqual(encode_feedback_observation(obs), encode_feedback_observation(obs))
        json.loads(encode_feedback_observation(obs))

    def test_derived_properties_are_not_encoded(self) -> None:
        text = encode_feedback_observation(observation(kind=FeedbackKind.LIKED, target=track_target()))
        self.assertNotIn("explicitness", text)
        self.assertNotIn("direction", text)

    def test_round_trip_recomputes_derived_properties(self) -> None:
        obs = observation(kind=FeedbackKind.SKIPPED, target=track_target())
        decoded = decode_feedback_observation(encode_feedback_observation(obs))
        self.assertEqual(decoded.explicitness, FeedbackExplicitness.IMPLICIT)
        self.assertEqual(decoded.direction, FeedbackDirection.NONE)

    def test_decode_rejects_non_string(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            decode_feedback_observation(12345)

    def test_decode_rejects_unparseable_json(self) -> None:
        with self.assertRaises(FeedbackContractValidationError):
            decode_feedback_observation("{not json")

    def test_decode_rejects_naive_timestamp(self) -> None:
        obs = observation(kind=FeedbackKind.LIKED, target=track_target())
        data = json.loads(encode_feedback_observation(obs))
        data["observed_at"] = "2026-08-16T00:00:00"
        with self.assertRaises(FeedbackContractValidationError):
            decode_feedback_observation(json.dumps(data))

    def test_decode_rejects_unknown_kind(self) -> None:
        obs = observation(kind=FeedbackKind.LIKED, target=track_target())
        data = json.loads(encode_feedback_observation(obs))
        data["kind"] = "super_liked"
        with self.assertRaises(FeedbackContractValidationError):
            decode_feedback_observation(json.dumps(data))

    def test_decode_rejects_missing_fields(self) -> None:
        obs = observation(kind=FeedbackKind.LIKED, target=track_target())
        data = json.loads(encode_feedback_observation(obs))
        del data["source"]
        with self.assertRaises(FeedbackContractValidationError):
            decode_feedback_observation(json.dumps(data))


if __name__ == "__main__":
    unittest.main()
