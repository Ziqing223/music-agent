"""P15-S3-S3A: known-catalog supply facts for the empty-diagnostics envelope.

The pure counts are the "known supply" exposure from P15-S3-S3 decision 2: real
facts from one execution's local inputs (pool + durable memory + frozen P11.2
candidate verdicts), never a freshness or trigger verdict. These tests pin the
counting semantics -- missing memory rows are "no memory", never "never
recommended" -- and the fail-closed validation.
"""

import unittest

from music_agent.catalog_track_state_repository import CatalogTrackState
from music_agent.known_catalog_supply import (
    KnownCatalogSupplyError,
    summarize_known_catalog_supply,
)
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    Eligibility,
    Rejection,
)

TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_C = "trk_cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def _candidate(
    track_id: str,
    *,
    eligibility: Eligibility = Eligibility.ELIGIBLE,
    index: int = 0,
) -> Candidate:
    """Minimal catalog candidate carrying only the eligibility it needs to convey."""
    return Candidate(
        candidate_id=f"cnd_aaaaaaaa-aaaa-4aaa-8aaa-{index:012d}",
        target=PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
        source=CandidateSourceReference("test_seed", "known_supply"),
        basis_targets=(),
        eligibility=eligibility,
        rejection=(
            Rejection("negative_preference")
            if eligibility is Eligibility.REJECTED
            else None
        ),
    )


def _state(track_id: str, recommendation_count: int = 0) -> CatalogTrackState:
    return CatalogTrackState(
        canonical_id=track_id,
        source_system="itunes_store",
        first_discovered_at=None,
        last_discovered_at=None,
        discovery_count=0,
        discovery_terms={},
        first_recommended_at=None,
        last_recommended_at=None,
        recommendation_count=recommendation_count,
        updated_at="2026-08-20T00:00:00+00:00",
    )


def _pool(*track_ids: str) -> list[dict]:
    return [{"id": track_id} for track_id in track_ids]


class KnownCatalogSupplyCountingTest(unittest.TestCase):
    """The five counts come from the caller's real facts, never invented ones."""

    def test_empty_inputs_report_all_zero_supply(self) -> None:
        supply = summarize_known_catalog_supply(
            _pool(), states_by_track_id={}, candidates=()
        )
        self.assertEqual(supply.known_catalog_track_count, 0)
        self.assertEqual(supply.known_never_recommended_count, 0)
        self.assertEqual(supply.known_eligible_count, 0)
        self.assertEqual(supply.known_rejected_count, 0)
        self.assertEqual(supply.known_never_recommended_eligible_count, 0)

    def test_pool_and_never_recommended_use_durable_memory_only(self) -> None:
        """Pool size is the passed pool; never-recommended only counts rows whose
        memory shows zero recommendation events -- a missing row is no memory."""
        supply = summarize_known_catalog_supply(
            _pool(TRACK_A, TRACK_B, TRACK_C),
            states_by_track_id={TRACK_A: _state(TRACK_A, 0), TRACK_B: _state(TRACK_B, 2)},
            candidates=(),
        )
        self.assertEqual(supply.known_catalog_track_count, 3)
        self.assertEqual(supply.known_never_recommended_count, 1)

    def test_eligible_and_rejected_are_recounted_never_rederived(self) -> None:
        eligible_a = _candidate(TRACK_A, index=0)
        eligible_b = _candidate(TRACK_B, index=1)
        rejected_c = _candidate(TRACK_C, eligibility=Eligibility.REJECTED, index=2)
        supply = summarize_known_catalog_supply(
            _pool(TRACK_A, TRACK_B, TRACK_C),
            states_by_track_id={},
            candidates=(eligible_a, eligible_b, rejected_c),
        )
        self.assertEqual(supply.known_eligible_count, 2)
        self.assertEqual(supply.known_rejected_count, 1)
        self.assertEqual(supply.known_never_recommended_eligible_count, 0)

    def test_never_recommended_eligible_intersects_memory_with_verdict(self) -> None:
        """Only eligible candidates whose memory row shows zero recommendation
        events count as the true never-shown supply."""
        supply = summarize_known_catalog_supply(
            _pool(TRACK_A, TRACK_B, TRACK_C),
            states_by_track_id={
                TRACK_A: _state(TRACK_A, 0),
                TRACK_B: _state(TRACK_B, 3),
            },
            candidates=(
                _candidate(TRACK_A, index=0),
                _candidate(TRACK_B, index=1),
                _candidate(TRACK_C, index=2),
            ),
        )
        self.assertEqual(supply.known_eligible_count, 3)
        self.assertEqual(supply.known_never_recommended_count, 1)
        self.assertEqual(supply.known_never_recommended_eligible_count, 1)

    def test_states_for_ids_outside_the_pool_are_ignored(self) -> None:
        supply = summarize_known_catalog_supply(
            _pool(TRACK_A),
            states_by_track_id={TRACK_B: _state(TRACK_B, 0)},
            candidates=(),
        )
        self.assertEqual(supply.known_catalog_track_count, 1)
        self.assertEqual(supply.known_never_recommended_count, 0)


class KnownCatalogSupplyValidationTest(unittest.TestCase):
    """Wrong argument types fail closed, mirroring the surrounding modules."""

    def _fill(self, *, catalog_tracks=(), states_by_track_id=None, candidates=()):
        return summarize_known_catalog_supply(
            catalog_tracks,
            states_by_track_id={} if states_by_track_id is None else states_by_track_id,
            candidates=candidates,
        )

    def test_catalog_tracks_must_be_a_sequence_of_mappings(self) -> None:
        for bad in ("trk_string", 7, None, {TRACK_A: 1}, [TRACK_A]):
            with self.subTest(bad=bad):
                with self.assertRaises(KnownCatalogSupplyError):
                    self._fill(catalog_tracks=bad)

    def test_catalog_tracks_require_valid_canonical_track_ids(self) -> None:
        for bad in ({}, {"id": None}, {"id": "alb_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}, {"id": "not-an-id"}):
            with self.subTest(bad=bad):
                with self.assertRaises(KnownCatalogSupplyError):
                    self._fill(catalog_tracks=[bad])

    def test_duplicate_pool_tracks_fail_closed(self) -> None:
        with self.assertRaises(KnownCatalogSupplyError):
            self._fill(catalog_tracks=_pool(TRACK_A, TRACK_A))

    def test_states_must_be_a_mapping_of_state_values(self) -> None:
        with self.assertRaises(KnownCatalogSupplyError):
            self._fill(states_by_track_id=[TRACK_A])
        with self.assertRaises(KnownCatalogSupplyError):
            self._fill(states_by_track_id={TRACK_A: "not-a-state"})
        with self.assertRaises(KnownCatalogSupplyError):
            self._fill(states_by_track_id={"bad": _state(TRACK_A, 0)})

    def test_candidates_must_be_a_sequence_of_candidates(self) -> None:
        for bad in ("candidate", 7, None, [_candidate(TRACK_A), "not-a-candidate"]):
            with self.subTest(bad=bad):
                with self.assertRaises(KnownCatalogSupplyError):
                    self._fill(candidates=bad)


if __name__ == "__main__":
    unittest.main()