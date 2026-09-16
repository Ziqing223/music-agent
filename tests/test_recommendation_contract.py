"""P07.1: the recommendation contract shared by candidate generation, scoring, and persistence.

These tests prove the pure domain contract only: stable run/candidate identity, the read-only
preference-input reference, candidate eligibility/rejection placement, the score/score-breakdown
boundary, request/context validation, the assembly seam, and the canonical JSON round-trip. No
candidate generation, scoring, ranking, or persistence is implemented here, and nothing touches
SQLite.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from music_agent.identity import ENTITY_ID_PREFIX
from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    RECOMMENDATION_CONTRACT_VERSION,
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendationContractValidationError,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResult,
    RecommendedItemKind,
    Rejection,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
    decode_recommendation_result,
    encode_recommendation_result,
    generate_candidate_id,
    generate_run_id,
    validate_candidate_id,
    validate_run_id,
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


def positive_strength(magnitude: float = 0.9) -> PreferenceStrength:
    return PreferenceStrength(PreferenceState.POSITIVE, magnitude)


def direct_input() -> PreferenceInput:
    return PreferenceInput.from_direct(DerivedPreference(track_target(), positive_strength()))


def inferred_input() -> PreferenceInput:
    return PreferenceInput.from_inferred(
        InferredAffinity(artist_target(), positive_strength(0.7))
    )


def source(system: str = "candidate_gen", path: str = "preference_match") -> CandidateSourceReference:
    return CandidateSourceReference(system, path)


def candidate(
    *,
    candidate_id: str = CANDIDATE_ID,
    target: PreferenceTargetReference | None = None,
    basis: tuple[PreferenceTargetReference, ...] = (),
    eligibility: Eligibility = Eligibility.ELIGIBLE,
    rejection: Rejection | None = None,
) -> Candidate:
    return Candidate(
        candidate_id,
        target or track_target(),
        source(),
        basis,
        eligibility,
        rejection,
    )


def score(*, total: float = 0.9, components: tuple[ScoreComponent, ...] | None = None) -> ScoreBreakdown:
    return ScoreBreakdown(
        total,
        components or (ScoreComponent("preference_match", 0.9),),
    )


def item(cand: Candidate | None = None) -> RecommendationItem:
    return RecommendationItem(cand or candidate(), score())


def context(now: datetime = NOW, inputs: tuple[PreferenceInput, ...] | None = None) -> RecommendationContext:
    return RecommendationContext(now, inputs or (direct_input(),))


def request(*, recommended_kind: RecommendedItemKind = RecommendedItemKind.TRACK, limit: int = 5) -> RecommendationRequest:
    return RecommendationRequest(context(), recommended_kind, limit)


class RecommendationIdentityTest(unittest.TestCase):
    def test_run_id_uses_rcm_namespace_and_uuid4(self) -> None:
        run_id = generate_run_id()
        self.assertTrue(run_id.startswith("rcm_"))
        self.assertEqual(UUID(run_id[len("rcm_") :]).version, 4)
        validate_run_id(run_id)

    def test_candidate_id_uses_cnd_namespace_and_uuid4(self) -> None:
        candidate_id = generate_candidate_id()
        self.assertTrue(candidate_id.startswith("cnd_"))
        self.assertEqual(UUID(candidate_id[len("cnd_") :]).version, 4)
        validate_candidate_id(candidate_id)

    def test_generated_ids_do_not_reuse_in_process(self) -> None:
        run_ids = {generate_run_id() for _ in range(1000)}
        candidate_ids = {generate_candidate_id() for _ in range(1000)}
        self.assertEqual(len(run_ids), 1000)
        self.assertEqual(len(candidate_ids), 1000)

    def test_namespaces_are_outside_canonical_and_operational_prefixes(self) -> None:
        run_id = generate_run_id()
        candidate_id = generate_candidate_id()
        for prefix in (*ENTITY_ID_PREFIX.values(), "int_", "att_", "prb_", "rec_"):
            self.assertFalse(run_id.startswith(prefix), prefix)
            self.assertFalse(candidate_id.startswith(prefix), prefix)

    def test_validate_rejects_wrong_namespace_and_bad_suffix(self) -> None:
        for bad in ("trk_11111111-1111-4111-8111-111111111111", "rcm_not-a-uuid", "rcm_"):
            with self.assertRaises(RecommendationContractValidationError):
                validate_run_id(bad)
        for bad in ("trk_11111111-1111-4111-8111-111111111111", "cnd_not-a-uuid"):
            with self.assertRaises(RecommendationContractValidationError):
                validate_candidate_id(bad)


class PreferenceInputReferenceTest(unittest.TestCase):
    def test_from_direct_preserves_target_strength_and_direct_provenance(self) -> None:
        input_ = direct_input()
        self.assertEqual(input_.target, track_target())
        self.assertEqual(input_.provenance, PreferenceProvenance.DIRECT)
        self.assertEqual(input_.strength.state, PreferenceState.POSITIVE)
        self.assertEqual(input_.strength.magnitude, 0.9)

    def test_from_inferred_preserves_inferred_provenance(self) -> None:
        input_ = inferred_input()
        self.assertEqual(input_.provenance, PreferenceProvenance.INFERRED)
        self.assertEqual(input_.target, artist_target())

    def test_from_direct_rejects_inferred_type(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            PreferenceInput.from_direct(InferredAffinity(track_target(), positive_strength()))  # type: ignore[arg-type]

    def test_constructor_rejects_bad_provenance_and_strength(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            PreferenceInput(track_target(), "direct", positive_strength())  # type: ignore[arg-type]
        with self.assertRaises(RecommendationContractValidationError):
            PreferenceInput(track_target(), PreferenceProvenance.DIRECT, "strong")  # type: ignore[arg-type]

    def test_read_only_snapshot_does_not_alias_p06(self) -> None:
        original = DerivedPreference(track_target(), positive_strength())
        input_ = PreferenceInput.from_direct(original)
        self.assertEqual(input_.target, original.target)
        self.assertEqual(input_.strength, original.strength)
        # The snapshot carries the same frozen value; P06 has no mutation path to expose.
        self.assertEqual(original.provenance, PreferenceProvenance.DIRECT)


class CandidateTest(unittest.TestCase):
    def test_eligible_candidate_defaults_without_rejection(self) -> None:
        c = candidate()
        self.assertIs(c.eligibility, Eligibility.ELIGIBLE)
        self.assertIsNone(c.rejection)

    def test_rejected_candidate_requires_rejection(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            candidate(eligibility=Eligibility.REJECTED)
        rejected = candidate(eligibility=Eligibility.REJECTED, rejection=Rejection("already_in_library"))
        self.assertIs(rejected.eligibility, Eligibility.REJECTED)
        self.assertEqual(rejected.rejection.reason, "already_in_library")

    def test_eligible_candidate_must_not_carry_rejection(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            candidate(rejection=Rejection("already_in_library"))

    def test_candidate_rejects_genre_target(self) -> None:
        genre = PreferenceTargetReference(PreferenceTargetKind.GENRE, "ambient")
        with self.assertRaises(RecommendationContractValidationError):
            candidate(target=genre)

    def test_candidate_accepts_artist_and_album_targets(self) -> None:
        Candidate(
            "cnd_44444444-4444-4444-8444-444444444444",
            artist_target(),
            source(),
        )
        Candidate(
            "cnd_55555555-5555-4555-8555-555555555555",
            PreferenceTargetReference(PreferenceTargetKind.ALBUM, "alb_11111111-1111-4111-8111-111111111111"),
            source(),
        )

    def test_candidate_rejects_duplicate_basis_targets(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            candidate(basis=(track_target(), track_target()))

    def test_candidate_basis_targets_coerced_to_tuple(self) -> None:
        c = Candidate(CANDIDATE_ID, track_target(), source(), [track_target()])
        self.assertIsInstance(c.basis_targets, tuple)

    def test_candidate_rejects_invalid_id_and_source(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            Candidate("bad-id", track_target(), source())
        with self.assertRaises(RecommendationContractValidationError):
            Candidate(CANDIDATE_ID, track_target(), "not-a-source")  # type: ignore[arg-type]


class ScoreTest(unittest.TestCase):
    def test_score_breakdown_accepts_bounded_components(self) -> None:
        s = ScoreBreakdown(0.8, (ScoreComponent("preference_match", 0.5), ScoreComponent("novelty", 0.3)))
        self.assertEqual(s.total, 0.8)
        self.assertEqual(len(s.components), 2)

    def test_score_rejects_out_of_range_and_non_finite(self) -> None:
        for bad in (1.1, -0.1, float("nan"), float("inf")):
            with self.assertRaises(RecommendationContractValidationError):
                ScoreBreakdown(bad, (ScoreComponent("a", 0.5),))
            with self.assertRaises(RecommendationContractValidationError):
                ScoreComponent("a", bad)

    def test_score_rejects_bool_and_str_values(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            ScoreBreakdown(True, (ScoreComponent("a", 0.5),))  # type: ignore[arg-type]
        with self.assertRaises(RecommendationContractValidationError):
            ScoreComponent("a", "0.5")  # type: ignore[arg-type]

    def test_score_breakdown_requires_at_least_one_component(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            ScoreBreakdown(0.5, ())

    def test_score_rejects_duplicate_component_names(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            ScoreBreakdown(0.5, (ScoreComponent("a", 0.2), ScoreComponent("a", 0.3)))


class RecommendationItemTest(unittest.TestCase):
    def test_eligible_candidate_can_be_scored(self) -> None:
        i = item()
        self.assertIs(i.candidate.eligibility, Eligibility.ELIGIBLE)

    def test_rejected_candidate_cannot_be_scored(self) -> None:
        rejected = candidate(eligibility=Eligibility.REJECTED, rejection=Rejection("already_in_library"))
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationItem(rejected, score())

    def test_item_requires_candidate_and_score_types(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationItem("not-a-candidate", score())  # type: ignore[arg-type]
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationItem(candidate(), "not-a-score")  # type: ignore[arg-type]


class ContextAndRequestTest(unittest.TestCase):
    def test_context_requires_aware_now(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationContext(datetime(2026, 8, 16), (direct_input(),))

    def test_context_rejects_duplicate_input_target_and_provenance(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationContext(NOW, (direct_input(), direct_input()))

    def test_context_allows_same_target_with_distinct_provenance(self) -> None:
        inferred_track = PreferenceInput(
            track_target(), PreferenceProvenance.INFERRED, positive_strength(0.5)
        )
        RecommendationContext(NOW, (direct_input(), inferred_track))

    def test_request_requires_positive_limit_and_valid_kind(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationRequest(context(), RecommendedItemKind.TRACK, 0)
        with self.assertRaises(RecommendationContractValidationError):
            RecommendationRequest(context(), "track", 5)  # type: ignore[arg-type]

    def test_request_validates(self) -> None:
        r = request()
        self.assertEqual(r.recommended_kind, RecommendedItemKind.TRACK)
        self.assertEqual(r.limit, 5)


class AssembleResultTest(unittest.TestCase):
    def test_assemble_stamps_current_contract_version(self) -> None:
        result = assemble_recommendation_result(
            request(), [item()], run_id=RUN_ID, produced_at=NOW
        )
        self.assertEqual(result.contract_version, RECOMMENDATION_CONTRACT_VERSION)
        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(result.items[0].candidate.candidate_id, CANDIDATE_ID)

    def test_assemble_preserves_item_order(self) -> None:
        second = item(
            candidate(candidate_id="cnd_66666666-6666-4666-8666-666666666666")
        )
        result = assemble_recommendation_result(
            request(), [item(), second], run_id=RUN_ID, produced_at=NOW
        )
        self.assertEqual(
            [i.candidate.candidate_id for i in result.items],
            [CANDIDATE_ID, "cnd_66666666-6666-4666-8666-666666666666"],
        )

    def test_assemble_allows_empty_items(self) -> None:
        result = assemble_recommendation_result(
            request(), [], run_id=RUN_ID, produced_at=NOW
        )
        self.assertEqual(result.items, ())

    def test_result_rejects_item_kind_mismatch(self) -> None:
        artist_item = RecommendationItem(
            Candidate(
                "cnd_77777777-7777-4777-8777-777777777777",
                artist_target(),
                source(),
            ),
            score(),
        )
        with self.assertRaises(RecommendationContractValidationError):
            assemble_recommendation_result(
                request(), [artist_item], run_id=RUN_ID, produced_at=NOW
            )

    def test_result_rejects_duplicate_candidate(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            assemble_recommendation_result(
                request(), [item(), item()], run_id=RUN_ID, produced_at=NOW
            )

    def test_result_requires_valid_run_id_and_aware_produced_at(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            assemble_recommendation_result(
                request(), [item()], run_id="trk_bad", produced_at=NOW
            )
        with self.assertRaises(RecommendationContractValidationError):
            assemble_recommendation_result(
                request(), [item()], run_id=RUN_ID, produced_at=datetime(2026, 8, 16)
            )


class SerializationTest(unittest.TestCase):
    def sample_result(self) -> RecommendationResult:
        return assemble_recommendation_result(
            request(), [item()], run_id=RUN_ID, produced_at=NOW
        )

    def test_round_trip_preserves_everything(self) -> None:
        result = self.sample_result()
        decoded = decode_recommendation_result(encode_recommendation_result(result))
        self.assertEqual(decoded, result)
        self.assertEqual(decoded.items[0].score.components[0].name, "preference_match")
        self.assertEqual(decoded.request.context.preference_inputs[0].provenance, PreferenceProvenance.DIRECT)

    def test_encoding_is_deterministic(self) -> None:
        result = self.sample_result()
        self.assertEqual(
            encode_recommendation_result(result),
            encode_recommendation_result(result),
        )

    def test_round_trip_preserves_item_order(self) -> None:
        second = item(candidate(candidate_id="cnd_66666666-6666-4666-8666-666666666666"))
        result = assemble_recommendation_result(
            request(), [item(), second], run_id=RUN_ID, produced_at=NOW
        )
        decoded = decode_recommendation_result(encode_recommendation_result(result))
        self.assertEqual(
            [i.candidate.candidate_id for i in decoded.items],
            [CANDIDATE_ID, "cnd_66666666-6666-4666-8666-666666666666"],
        )

    def test_round_trip_preserves_preference_strength(self) -> None:
        result = assemble_recommendation_result(
            request(), [item()], run_id=RUN_ID, produced_at=NOW
        )
        decoded = decode_recommendation_result(encode_recommendation_result(result))
        self.assertEqual(decoded.request.context.preference_inputs[0].strength.state, PreferenceState.POSITIVE)

    def test_decode_rejects_non_string_and_garbage(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            decode_recommendation_result(1)  # type: ignore[arg-type]
        with self.assertRaises(RecommendationContractValidationError):
            decode_recommendation_result("not json")
        with self.assertRaises(RecommendationContractValidationError):
            decode_recommendation_result('{"run_id": 1}')

    def test_encode_rejects_non_result(self) -> None:
        with self.assertRaises(RecommendationContractValidationError):
            encode_recommendation_result("not-a-result")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
