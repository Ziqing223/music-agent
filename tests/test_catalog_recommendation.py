"""P11.2: unified affinity inputs and catalog recommendations (orchestration tests)."""

import unittest
from datetime import datetime, timezone

from music_agent.artist_affinity import ArtistAffinity, build_artist_affinities
from music_agent.catalog_recommendation import (
    EvenSplitPolicy,
    affinity_inputs,
    build_affinity_inputs,
    build_catalog_recommendation,
)
from music_agent.genre_affinity import GenreAffinityPolicy, GenreAffinity
from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    PreferenceInput,
    generate_run_id,
)

NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)

ARTIST_A = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ARTIST_B = "art_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TRACK_L1 = "trk_11111111-1111-4111-8111-111111111111"
TRACK_L2 = "trk_22222222-2222-4222-8222-222222222222"
TRACK_L3 = "trk_33333333-3333-4333-8333-333333333333"
TRACK_L4 = "trk_44444444-4444-4444-8444-444444444444"
TRACK_C1 = "trk_c1c1c1c1-c1c1-4c1c-8c1c-c1c1c1c1c1c1"
TRACK_C2 = "trk_c2c2c2c2-c2c2-4c2c-8c2c-c2c2c2c2c2c2"
TRACK_C3 = "trk_c3c3c3c3-c3c3-4c3c-8c3c-c3c3c3c3c3c3"

SOURCE = "apple_music"


def direct(target_id: str, state: PreferenceState, magnitude: float) -> DerivedPreference:
    return DerivedPreference(
        PreferenceTargetReference(PreferenceTargetKind.TRACK, target_id),
        PreferenceStrength(state, magnitude),
    )


def catalog_track(track_id: str, *, genres=(), artist_ids=()) -> dict:
    return {"id": track_id, "genres": list(genres), "artist_ids": list(artist_ids)}


class AffinityInputsTest(unittest.TestCase):
    def test_positive_direct_preference_yields_genre_and_artist_inputs(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
        }
        inputs = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        self.assertEqual(len(inputs), 2)
        by_target = {input_.target: input_ for input_ in inputs}
        rock = by_target[PreferenceTargetReference(PreferenceTargetKind.GENRE, "Rock")]
        self.assertEqual(rock.strength.state, PreferenceState.POSITIVE)
        self.assertEqual(rock.strength.magnitude, 1.0)
        artist = by_target[PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_A)]
        self.assertEqual(artist.strength.magnitude, 1.0)

    def test_negative_direct_preference_yields_negative_inputs(self) -> None:
        tracks = {TRACK_L3: {"id": TRACK_L3, "genres": ["Metal"], "artist_ids": [ARTIST_B]}}
        inputs = build_affinity_inputs(
            [direct(TRACK_L3, PreferenceState.NEGATIVE, 0.9)], tracks, SOURCE
        )
        self.assertEqual(len(inputs), 2)
        self.assertTrue(
            all(input_.strength.state is PreferenceState.NEGATIVE for input_ in inputs)
        )
        self.assertTrue(all(input_.strength.magnitude == 0.9 for input_ in inputs))

    def test_multi_genre_track_splits_evenly(self) -> None:
        tracks = {TRACK_L4: {"id": TRACK_L4, "genres": ["Rock", "Jazz"], "artist_ids": []}}
        inputs = build_affinity_inputs(
            [direct(TRACK_L4, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        by_target = {input_.target.target_id: input_ for input_ in inputs}
        self.assertEqual(by_target["Rock"].strength.magnitude, 0.5)
        self.assertEqual(by_target["Jazz"].strength.magnitude, 0.5)

    def test_below_threshold_affinities_are_excluded(self) -> None:
        tracks = {TRACK_L4: {"id": TRACK_L4, "genres": ["Rock", "Jazz"], "artist_ids": []}}
        inputs = build_affinity_inputs(
            [direct(TRACK_L4, PreferenceState.POSITIVE, 1.0)],
            tracks,
            SOURCE,
            policy=GenreAffinityPolicy(affinity_threshold=0.6),
        )
        self.assertEqual(inputs, ())

    def test_non_directional_preferences_contribute_nothing(self) -> None:
        tracks = {TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]}}
        inputs = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.NEUTRAL, 0.0)], tracks, SOURCE
        )
        self.assertEqual(inputs, ())

    def test_even_split_policy_divides_magnitudes(self) -> None:
        policy = EvenSplitPolicy()
        self.assertEqual(policy.split_artist_magnitude(1.0, 2), 0.5)
        self.assertEqual(policy.split_genre_magnitude(0.8, 4), 0.2)


class CatalogRecommendationTest(unittest.TestCase):
    def test_artist_affinity_affects_new_song_ranking(self) -> None:
        # Two library preferences, same genre, different artists and magnitudes.
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
            TRACK_L2: {"id": TRACK_L2, "genres": ["Rock"], "artist_ids": [ARTIST_B]},
        }
        affinity = build_affinity_inputs(
            [
                direct(TRACK_L1, PreferenceState.POSITIVE, 1.0),
                direct(TRACK_L2, PreferenceState.POSITIVE, 0.5),
            ],
            tracks,
            SOURCE,
        )
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=(),
            affinity_inputs=affinity,
            catalog_tracks=[
                catalog_track(TRACK_C2, genres=("Rock",), artist_ids=[ARTIST_B]),
                catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A]),
            ],
            limit=10,
        )
        ranked = [item.candidate.target.target_id for item in outcome.result.items]
        self.assertEqual(ranked, [TRACK_C1, TRACK_C2])
        scores = {item.candidate.target.target_id: item.score.total for item in outcome.result.items}
        self.assertGreater(scores[TRACK_C1], scores[TRACK_C2])

    def test_genre_affinity_affects_new_song_ranking(self) -> None:
        # Same artist, two genres with different magnitudes.
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
            TRACK_L2: {"id": TRACK_L2, "genres": ["Jazz"], "artist_ids": [ARTIST_A]},
        }
        affinity = build_affinity_inputs(
            [
                direct(TRACK_L1, PreferenceState.POSITIVE, 1.0),
                direct(TRACK_L2, PreferenceState.POSITIVE, 0.5),
            ],
            tracks,
            SOURCE,
        )
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=(),
            affinity_inputs=affinity,
            catalog_tracks=[
                catalog_track(TRACK_C2, genres=("Jazz",), artist_ids=[ARTIST_A]),
                catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A]),
            ],
            limit=10,
        )
        ranked = [item.candidate.target.target_id for item in outcome.result.items]
        self.assertEqual(ranked, [TRACK_C1, TRACK_C2])

    def test_negative_preference_suppresses_relevant_candidates(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
            TRACK_L3: {"id": TRACK_L3, "genres": ["Metal"], "artist_ids": [ARTIST_B]},
        }
        affinity = build_affinity_inputs(
            [
                direct(TRACK_L1, PreferenceState.POSITIVE, 1.0),
                direct(TRACK_L3, PreferenceState.NEGATIVE, 0.9),
            ],
            tracks,
            SOURCE,
        )
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=(),
            affinity_inputs=affinity,
            catalog_tracks=[
                catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A]),
                catalog_track(TRACK_C3, genres=("Metal",), artist_ids=[ARTIST_B]),
            ],
            limit=10,
        )
        ranked = [item.candidate.target.target_id for item in outcome.result.items]
        self.assertIn(TRACK_C1, ranked)
        self.assertNotIn(TRACK_C3, ranked)  # rejected at the candidate layer, never scored

    def test_unrelated_ambient_preferences_pass_through_and_rank(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
        }
        affinity = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        unrelated = direct(TRACK_L4, PreferenceState.POSITIVE, 0.6)
        ambient = (PreferenceInput.from_direct(unrelated),)
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=ambient,
            affinity_inputs=affinity,
            catalog_tracks=[catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A])],
            limit=10,
        )
        targets = [item.candidate.target.target_id for item in outcome.result.items]
        self.assertIn(TRACK_L4, targets)  # unrelated preference still ranks
        self.assertIn(TRACK_C1, targets)  # catalog candidate still ranks
        # The unrelated input survives intact in the persisted result context.
        context_inputs = outcome.result.request.context.preference_inputs
        self.assertIn(ambient[0], context_inputs)

    def test_preference_driven_candidate_wins_on_duplicate_target(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
        }
        affinity = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        ambient = (PreferenceInput.from_direct(direct(TRACK_C1, PreferenceState.POSITIVE, 0.7)),)
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=ambient,
            affinity_inputs=affinity,
            catalog_tracks=[catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A])],
            limit=10,
        )
        matching = [item for item in outcome.result.items if item.candidate.target.target_id == TRACK_C1]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].candidate.source.source_path, "preference_driven")

    def test_catalog_candidates_carry_catalog_driven_source(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
        }
        affinity = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=(),
            affinity_inputs=affinity,
            catalog_tracks=[catalog_track(TRACK_C1, genres=("Rock",), artist_ids=[ARTIST_A])],
            limit=10,
        )
        item = outcome.result.items[0]
        self.assertEqual(item.candidate.source.source_system, "music_agent")
        self.assertEqual(item.candidate.source.source_path, "catalog_driven")
        self.assertGreater(item.score.total, 0)  # never a zero-score insert

    def test_unjudged_catalog_track_never_competes(self) -> None:
        tracks = {
            TRACK_L1: {"id": TRACK_L1, "genres": ["Rock"], "artist_ids": [ARTIST_A]},
        }
        affinity = build_affinity_inputs(
            [direct(TRACK_L1, PreferenceState.POSITIVE, 1.0)], tracks, SOURCE
        )
        outcome = build_catalog_recommendation(
            run_id=generate_run_id(),
            produced_at=NOW,
            ambient_inputs=(),
            affinity_inputs=affinity,
            catalog_tracks=[catalog_track(TRACK_C2, genres=("Jazz",), artist_ids=[ARTIST_B])],
            limit=10,
        )
        self.assertEqual(outcome.result.items, ())


class ArtistAffinityReducerTest(unittest.TestCase):
    def test_artist_affinity_reduction_matches_genre_reducer_semantics(self) -> None:
        from music_agent.preference_propagation import InferredAffinityContribution, PropagationKind
        from music_agent.artist_affinity import SourcedArtistContribution
        from music_agent.preference_signal import SignalDirection

        contribution = InferredAffinityContribution(
            source_track=PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_L1),
            target=PreferenceTargetReference(PreferenceTargetKind.ARTIST, ARTIST_A),
            direction=SignalDirection.POSITIVE,
            input_magnitude=1.0,
            derived_magnitude=1.0,
            kind=PropagationKind.ARTIST,
            constraint=None,
            split_count=1,
        )
        result = build_artist_affinities(
            [SourcedArtistContribution(SOURCE, contribution)], GenreAffinityPolicy()
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ArtistAffinity)
        self.assertEqual(result[0].artist_id, ARTIST_A)
        self.assertEqual(result[0].affinity, 1.0)

    def test_affinity_inputs_project_genre_and_artist(self) -> None:
        genre = GenreAffinity(
            source_system=SOURCE, genre_key="Rock",
            positive_count=1, negative_count=0, positive_sum=0.8, negative_sum=0.0,
            net=0.8, affinity=0.8, source_track_ids=(TRACK_L1,),
        )
        artist = ArtistAffinity(
            source_system=SOURCE, artist_id=ARTIST_A,
            positive_count=1, negative_count=0, positive_sum=0.6, negative_sum=0.0,
            net=0.6, affinity=0.6, source_track_ids=(TRACK_L1,),
        )
        inputs = affinity_inputs([genre], [artist], GenreAffinityPolicy())
        self.assertEqual(len(inputs), 2)
        kinds = {input_.target.kind for input_ in inputs}
        self.assertEqual(kinds, {PreferenceTargetKind.GENRE, PreferenceTargetKind.ARTIST})


if __name__ == "__main__":
    unittest.main()
