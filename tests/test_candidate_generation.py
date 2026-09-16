"""P07.2 (Track A): preference-driven candidate generation.

These tests prove the candidate-generation slice only: kind matching, positive motivation,
negative rejection with the machine-code reason, non-directional skipping, the per-target merge
with deduplicated basis targets, deterministic ordering, the empty outcome, contract
conformance of every emitted candidate, and fail-closed input validation. Nothing touches
SQLite, the clock, or P06 persistence; only the frozen contract and P06 domain types are
imported alongside the module under test.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from music_agent.candidate_generation import (
    CANDIDATE_SOURCE_PATH,
    CANDIDATE_SOURCE_SYSTEM,
    REJECTION_NEGATIVE_PREFERENCE,
    CandidateGenerationError,
    CandidateGenerationValidationError,
    generate_candidates,
)
from music_agent.preference_attribution import (
    DerivedPreference,
    InferredAffinity,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    PreferenceInput,
    RecommendationContext,
    RecommendedItemKind,
    validate_candidate_id,
)

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
TRACK_ID_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_ID_C = "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
ARTIST_ID = "art_11111111-1111-4111-8111-111111111111"
ALBUM_ID = "alb_11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

PREFERENCE_SOURCE = CandidateSourceReference(CANDIDATE_SOURCE_SYSTEM, CANDIDATE_SOURCE_PATH)


def track(target_id: str = TRACK_ID) -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id)


def artist() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_ID)


def album() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.ALBUM, ALBUM_ID)


def genre() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.GENRE, "ambient")


def strength(state: PreferenceState, magnitude: float | None = None) -> PreferenceStrength:
    if magnitude is None:
        if state is PreferenceState.NEUTRAL:
            magnitude = 0.0
        elif state in (PreferenceState.POSITIVE, PreferenceState.NEGATIVE):
            magnitude = 0.9
    return PreferenceStrength(state, magnitude)


def direct_input(
    target: PreferenceTargetReference,
    state: PreferenceState = PreferenceState.POSITIVE,
    magnitude: float | None = None,
) -> PreferenceInput:
    return PreferenceInput.from_direct(DerivedPreference(target, strength(state, magnitude)))


def inferred_input(
    target: PreferenceTargetReference,
    state: PreferenceState = PreferenceState.POSITIVE,
    magnitude: float | None = None,
) -> PreferenceInput:
    return PreferenceInput.from_inferred(InferredAffinity(target, strength(state, magnitude)))


def context(*inputs: PreferenceInput, now: datetime = NOW) -> RecommendationContext:
    return RecommendationContext(now, inputs)


def generate(
    *inputs: PreferenceInput,
    kind: RecommendedItemKind = RecommendedItemKind.TRACK,
) -> tuple[Candidate, ...]:
    return generate_candidates(context(*inputs), kind)


class CandidateGenerationValidationTest(unittest.TestCase):
    def test_rejects_non_context(self) -> None:
        with self.assertRaises(CandidateGenerationValidationError):
            generate_candidates("not a context", RecommendedItemKind.TRACK)  # type: ignore[arg-type]

    def test_rejects_non_kind(self) -> None:
        with self.assertRaises(CandidateGenerationValidationError):
            generate_candidates(context(direct_input(track())), "track")  # type: ignore[arg-type]

    def test_rejects_bool_and_none_kind(self) -> None:
        with self.assertRaises(CandidateGenerationValidationError):
            generate_candidates(context(), True)  # type: ignore[arg-type]
        with self.assertRaises(CandidateGenerationValidationError):
            generate_candidates(context(), None)  # type: ignore[arg-type]

    def test_validation_error_is_a_candidate_generation_error(self) -> None:
        with self.assertRaises(CandidateGenerationError) as raised:
            generate_candidates(None, RecommendedItemKind.TRACK)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, "validation_error")


class KindMatchingTest(unittest.TestCase):
    def test_track_request_motivates_track_input(self) -> None:
        candidates = generate(direct_input(track()), kind=RecommendedItemKind.TRACK)
        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0].target.kind, PreferenceTargetKind.TRACK)

    def test_artist_request_motivates_artist_input(self) -> None:
        candidates = generate(direct_input(artist()), kind=RecommendedItemKind.ARTIST)
        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0].target.kind, PreferenceTargetKind.ARTIST)

    def test_album_request_motivates_album_input(self) -> None:
        candidates = generate(direct_input(album()), kind=RecommendedItemKind.ALBUM)
        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0].target.kind, PreferenceTargetKind.ALBUM)

    def test_non_matching_kind_inputs_are_skipped(self) -> None:
        for kind, other in (
            (RecommendedItemKind.TRACK, artist()),
            (RecommendedItemKind.TRACK, album()),
            (RecommendedItemKind.ARTIST, track()),
            (RecommendedItemKind.ALBUM, track()),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(generate(direct_input(other), kind=kind), ())

    def test_genre_inputs_never_motivate_a_candidate(self) -> None:
        for kind in RecommendedItemKind:
            with self.subTest(kind=kind):
                self.assertEqual(generate(direct_input(genre()), kind=kind), ())

    def test_mixed_kinds_only_requested_kind_survives(self) -> None:
        candidates = generate(
            direct_input(track()),
            direct_input(artist()),
            direct_input(genre()),
            kind=RecommendedItemKind.TRACK,
        )
        self.assertEqual([candidate.target for candidate in candidates], [track()])


class PositiveMotivationTest(unittest.TestCase):
    def test_direct_positive_yields_eligible_candidate(self) -> None:
        (candidate,) = generate(direct_input(track()))
        self.assertIs(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertIsNone(candidate.rejection)
        self.assertEqual(candidate.target, track())
        self.assertEqual(candidate.basis_targets, (track(),))
        self.assertEqual(candidate.source, PREFERENCE_SOURCE)
        self.assertEqual(candidate.source.source_system, CANDIDATE_SOURCE_SYSTEM)
        self.assertEqual(candidate.source.source_path, CANDIDATE_SOURCE_PATH)
        self.assertTrue(candidate.candidate_id.startswith("cnd_"))
        validate_candidate_id(candidate.candidate_id)

    def test_inferred_positive_yields_eligible_candidate(self) -> None:
        (candidate,) = generate(inferred_input(track()))
        self.assertIs(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidate.basis_targets, (track(),))

    def test_magnitude_is_not_consulted_for_eligibility(self) -> None:
        low = generate(direct_input(track(), magnitude=0.1))
        high = generate(direct_input(track(), magnitude=1.0))
        self.assertIs(low[0].eligibility, Eligibility.ELIGIBLE)
        self.assertIs(high[0].eligibility, Eligibility.ELIGIBLE)


class NegativeRejectionTest(unittest.TestCase):
    def test_direct_negative_yields_rejected_candidate(self) -> None:
        (candidate,) = generate(direct_input(track(), state=PreferenceState.NEGATIVE))
        self.assertIs(candidate.eligibility, Eligibility.REJECTED)
        self.assertIsNotNone(candidate.rejection)
        self.assertEqual(candidate.rejection.reason, REJECTION_NEGATIVE_PREFERENCE)
        self.assertEqual(candidate.rejection.reason, "negative_preference")
        self.assertEqual(candidate.basis_targets, (track(),))

    def test_inferred_negative_yields_rejected_candidate(self) -> None:
        (candidate,) = generate(inferred_input(track(), state=PreferenceState.NEGATIVE))
        self.assertIs(candidate.eligibility, Eligibility.REJECTED)
        self.assertEqual(candidate.rejection.reason, REJECTION_NEGATIVE_PREFERENCE)

    def test_rejection_reason_is_a_non_empty_machine_code(self) -> None:
        (candidate,) = generate(direct_input(track(), state=PreferenceState.NEGATIVE))
        reason = candidate.rejection.reason
        self.assertIsInstance(reason, str)
        self.assertNotEqual(reason, "")
        self.assertNotIn(" ", reason)

    def test_rejected_candidate_carries_rejection_and_eligible_never_does(self) -> None:
        candidates = generate(
            direct_input(track(), state=PreferenceState.NEGATIVE),
            direct_input(track(TRACK_ID_B)),
        )
        by_target = {candidate.target: candidate for candidate in candidates}
        self.assertIs(by_target[track()].eligibility, Eligibility.REJECTED)
        self.assertIsNotNone(by_target[track()].rejection)
        self.assertIs(by_target[track(TRACK_ID_B)].eligibility, Eligibility.ELIGIBLE)
        self.assertIsNone(by_target[track(TRACK_ID_B)].rejection)


class NonDirectionalHandlingTest(unittest.TestCase):
    def test_non_directional_states_produce_no_candidate(self) -> None:
        for state in (
            PreferenceState.UNKNOWN,
            PreferenceState.INSUFFICIENT,
            PreferenceState.NEUTRAL,
            PreferenceState.CONFLICT,
        ):
            with self.subTest(state=state):
                self.assertEqual(generate(direct_input(track(), state=state)), ())

    def test_non_directional_input_contributes_nothing_beside_a_positive(self) -> None:
        (candidate,) = generate(
            direct_input(track(), state=PreferenceState.UNKNOWN),
            inferred_input(track(), state=PreferenceState.POSITIVE),
        )
        self.assertIs(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidate.basis_targets, (track(),))


class MergeAndDedupTest(unittest.TestCase):
    def test_direct_and_inferred_positive_merge_into_one_candidate(self) -> None:
        (candidate,) = generate(direct_input(track()), inferred_input(track()))
        self.assertIs(candidate.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(candidate.basis_targets, (track(),))

    def test_direct_and_inferred_negative_merge_into_one_rejected_candidate(self) -> None:
        (candidate,) = generate(
            direct_input(track(), state=PreferenceState.NEGATIVE),
            inferred_input(track(), state=PreferenceState.NEGATIVE),
        )
        self.assertIs(candidate.eligibility, Eligibility.REJECTED)
        self.assertEqual(candidate.rejection.reason, REJECTION_NEGATIVE_PREFERENCE)
        self.assertEqual(candidate.basis_targets, (track(),))

    def test_negative_vetoes_positive_for_same_target(self) -> None:
        cases = (
            (direct_input(track(), state=PreferenceState.POSITIVE), inferred_input(track(), state=PreferenceState.NEGATIVE)),
            (direct_input(track(), state=PreferenceState.NEGATIVE), inferred_input(track(), state=PreferenceState.POSITIVE)),
        )
        for inputs in cases:
            with self.subTest(inputs=inputs):
                (candidate,) = generate(*inputs)
                self.assertIs(candidate.eligibility, Eligibility.REJECTED)
                self.assertEqual(candidate.rejection.reason, REJECTION_NEGATIVE_PREFERENCE)
                self.assertEqual(candidate.basis_targets, (track(),))

    def test_distinct_targets_produce_distinct_candidates(self) -> None:
        candidates = generate(direct_input(track()), direct_input(track(TRACK_ID_B)))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.target for candidate in candidates},
            {track(), track(TRACK_ID_B)},
        )


class OrderingTest(unittest.TestCase):
    def test_candidates_sorted_by_target_id_regardless_of_input_order(self) -> None:
        a, b, c = track(), track(TRACK_ID_B), track(TRACK_ID_C)
        candidates = generate(direct_input(c), direct_input(a), direct_input(b))
        self.assertEqual([candidate.target for candidate in candidates], [a, b, c])

    def test_rejected_and_eligible_interleave_in_sorted_order(self) -> None:
        a, b, c = track(), track(TRACK_ID_B), track(TRACK_ID_C)
        candidates = generate(
            direct_input(b),
            direct_input(c, state=PreferenceState.NEGATIVE),
            direct_input(a),
        )
        self.assertEqual([candidate.target for candidate in candidates], [a, b, c])
        self.assertEqual(
            [candidate.eligibility for candidate in candidates],
            [Eligibility.ELIGIBLE, Eligibility.ELIGIBLE, Eligibility.REJECTED],
        )


class EmptyContextTest(unittest.TestCase):
    def test_empty_inputs_yield_empty_candidates(self) -> None:
        self.assertEqual(generate(), ())

    def test_only_non_matching_inputs_yield_empty_candidates(self) -> None:
        self.assertEqual(generate(direct_input(artist()), kind=RecommendedItemKind.TRACK), ())


class ContractConformanceTest(unittest.TestCase):
    def test_every_emitted_candidate_passes_contract_validation(self) -> None:
        candidates = generate(
            direct_input(track()),
            inferred_input(track(TRACK_ID_B), state=PreferenceState.NEGATIVE),
            direct_input(track(TRACK_ID_C)),
        )
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            validate_candidate_id(candidate.candidate_id)
            # Reconstructing through the contract's validated constructor proves conformance.
            Candidate(
                candidate.candidate_id,
                candidate.target,
                candidate.source,
                candidate.basis_targets,
                candidate.eligibility,
                candidate.rejection,
            )

    def test_eligibility_rejection_invariants_hold_for_every_candidate(self) -> None:
        candidates = generate(
            direct_input(track(), state=PreferenceState.NEGATIVE),
            direct_input(track(TRACK_ID_B)),
        )
        for candidate in candidates:
            if candidate.eligibility is Eligibility.REJECTED:
                self.assertIsNotNone(candidate.rejection)
            else:
                self.assertIsNone(candidate.rejection)


class DeterminismTest(unittest.TestCase):
    def test_same_context_yields_same_shape_and_order(self) -> None:
        inputs = (
            direct_input(track(TRACK_ID_C)),
            inferred_input(track(TRACK_ID_B), state=PreferenceState.NEGATIVE),
            direct_input(track()),
        )
        first = generate(*inputs)
        second = generate(*inputs)
        self.assertEqual([c.target for c in first], [c.target for c in second])
        self.assertEqual([c.eligibility for c in first], [c.eligibility for c in second])
        self.assertEqual(
            [c.rejection.reason if c.rejection is not None else None for c in first],
            [c.rejection.reason if c.rejection is not None else None for c in second],
        )
        self.assertEqual([c.basis_targets for c in first], [c.basis_targets for c in second])
        # Identities are minted fresh per run, never derived from the target.
        self.assertNotEqual([c.candidate_id for c in first], [c.candidate_id for c in second])

    def test_generation_does_not_depend_on_now(self) -> None:
        inputs = (
            direct_input(track()),
            direct_input(track(TRACK_ID_B), state=PreferenceState.NEGATIVE),
        )
        earlier = generate_candidates(
            RecommendationContext(datetime(2026, 1, 1, tzinfo=timezone.utc), inputs),
            RecommendedItemKind.TRACK,
        )
        later = generate_candidates(
            RecommendationContext(datetime(2026, 12, 31, tzinfo=timezone.utc), inputs),
            RecommendedItemKind.TRACK,
        )
        self.assertEqual([c.target for c in earlier], [c.target for c in later])
        self.assertEqual([c.eligibility for c in earlier], [c.eligibility for c in later])


if __name__ == "__main__":
    unittest.main()
