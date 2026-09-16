import inspect
import unittest
from dataclasses import FrozenInstanceError, fields

from music_agent.preference_attribution import (
    AttributionConstraint,
    ConstraintMode,
    DerivedPreference,
    PreferenceProvenance,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import (
    AlbumAttenuationPolicy,
    ArtistSplitPolicy,
    AttributionConstraintBinding,
    DiscountAttenuationPolicy,
    GenreSplitPolicy,
    InferredAffinityContribution,
    PropagationKind,
    PropagationValidationError,
    canonicalize_genre_key,
    propagate_track_preference,
)
from music_agent.preference_signal import SignalDirection
from music_agent.preference_strength import PreferenceState, PreferenceStrength

TRACK_ID = "trk_00000000-0000-4000-8000-000000000000"
ARTIST_A = "art_00000000-0000-4000-8000-000000000001"
ARTIST_B = "art_00000000-0000-4000-8000-000000000002"
ALBUM_ID = "alb_00000000-0000-4000-8000-000000000000"


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def ref(kind: PreferenceTargetKind, target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(kind, target_id)


def direct(state: PreferenceState, magnitude: float | None = None) -> DerivedPreference:
    return DerivedPreference(track_target(), PreferenceStrength(state, magnitude))


class _EvenSplit:
    """Split a magnitude evenly across N siblings, for both artist and genre seams."""

    def split_artist_magnitude(self, magnitude: float, artist_count: int) -> float:
        return magnitude / artist_count

    def split_genre_magnitude(self, magnitude: float, genre_count: int) -> float:
        return magnitude / genre_count


class _FactorAlbum:
    def __init__(self, factor: float) -> None:
        self._factor = factor

    def attenuate_album(self, magnitude: float) -> float:
        return magnitude * self._factor


class _FactorDiscount:
    def __init__(self, factor: float) -> None:
        self._factor = factor

    def attenuate_discount(self, magnitude: float) -> float:
        return magnitude * self._factor


class _AmplifyingArtistSplit:
    def split_artist_magnitude(self, magnitude: float, artist_count: int) -> float:
        return magnitude * 2


class _NaNArtistSplit:
    def split_artist_magnitude(self, magnitude: float, artist_count: int) -> float:
        return float("nan")


def even_split() -> _EvenSplit:
    return _EvenSplit()


def propagate(
    state: PreferenceState,
    magnitude: float,
    *,
    artist_ids=(),
    album_id=None,
    genres=(),
    constraints=(),
    artist_split=None,
    album_attenuation=None,
    genre_split=None,
    discount_attenuation=None,
):
    return propagate_track_preference(
        direct(state, magnitude),
        artist_ids=artist_ids,
        album_id=album_id,
        genres=genres,
        constraints=constraints,
        artist_split=artist_split,
        album_attenuation=album_attenuation,
        genre_split=genre_split,
        discount_attenuation=discount_attenuation,
    )


def positive(artist_ids=(), album_id=None, genres=(), **kwargs):
    return propagate(
        PreferenceState.POSITIVE, 0.9, artist_ids=artist_ids, album_id=album_id, genres=genres, **kwargs
    )


class PropagationKindTest(unittest.TestCase):
    def test_enum_values_are_stable(self) -> None:
        expected = {
            PropagationKind.ARTIST: "artist",
            PropagationKind.ALBUM: "album",
            PropagationKind.GENRE: "genre",
        }
        self.assertEqual(list(PropagationKind), list(expected))
        for member, value in expected.items():
            self.assertEqual(member.value, value)

    def test_three_kinds_are_distinct(self) -> None:
        self.assertEqual(len({PropagationKind.ARTIST, PropagationKind.ALBUM, PropagationKind.GENRE}), 3)


class CanonicalizeGenreKeyTest(unittest.TestCase):
    def test_trim(self) -> None:
        self.assertEqual(canonicalize_genre_key("  Hip Hop  "), "Hip Hop")

    def test_collapse_internal_whitespace(self) -> None:
        self.assertEqual(canonicalize_genre_key("Hip   Hop"), "Hip Hop")
        self.assertEqual(canonicalize_genre_key("Hip\t\tHop"), "Hip Hop")
        self.assertEqual(canonicalize_genre_key("Hip\n Hop"), "Hip Hop")

    def test_unicode_nfc(self) -> None:
        self.assertEqual(canonicalize_genre_key("café"), "café")

    def test_preserve_case(self) -> None:
        self.assertEqual(canonicalize_genre_key("hip hop"), "hip hop")
        self.assertEqual(canonicalize_genre_key("Hip Hop"), "Hip Hop")

    def test_preserve_punctuation(self) -> None:
        self.assertEqual(canonicalize_genre_key("Hip-Hop"), "Hip-Hop")
        self.assertEqual(canonicalize_genre_key("indie/rock"), "indie/rock")

    def test_no_alias_merge(self) -> None:
        self.assertNotEqual(canonicalize_genre_key("Hip Hop"), canonicalize_genre_key("Hip-Hop"))
        self.assertNotEqual(canonicalize_genre_key("Hip-Hop"), canonicalize_genre_key("hip-hop"))

    def test_non_string_fails_closed(self) -> None:
        for bad in (None, 5, True, ["Hip Hop"]):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    canonicalize_genre_key(bad)


class ArtistPropagationTest(unittest.TestCase):
    def test_one_artist_passes_through_unsplit(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A])
        self.assertEqual(len(contributions), 1)
        contribution = contributions[0]
        self.assertIs(contribution.kind, PropagationKind.ARTIST)
        self.assertEqual(contribution.target, ref(PreferenceTargetKind.ARTIST, ARTIST_A))
        self.assertEqual(contribution.direction, SignalDirection.POSITIVE)
        self.assertEqual(contribution.input_magnitude, 0.9)
        self.assertEqual(contribution.derived_magnitude, 0.9)
        self.assertEqual(contribution.split_count, 1)
        self.assertIsNone(contribution.constraint)

    def test_multiple_artists_are_split(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split=even_split())
        self.assertEqual(len(contributions), 2)
        self.assertEqual(contributions[0].derived_magnitude, 0.45)
        self.assertEqual(contributions[1].derived_magnitude, 0.45)
        self.assertEqual(contributions[0].split_count, 2)
        self.assertEqual(contributions[1].split_count, 2)

    def test_no_double_amplification(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split=even_split())
        total = sum(contribution.derived_magnitude for contribution in contributions)
        self.assertEqual(total, 0.9)

    def test_negative_direction_propagates(self) -> None:
        contributions = propagate(PreferenceState.NEGATIVE, 0.8, artist_ids=[ARTIST_A])
        self.assertEqual(len(contributions), 1)
        self.assertEqual(contributions[0].direction, SignalDirection.NEGATIVE)
        self.assertEqual(contributions[0].derived_magnitude, 0.8)

    def test_block_removes_that_artist(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_B), AttributionConstraint(ConstraintMode.BLOCK)
        )
        contributions = positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split=even_split(), constraints=[binding])
        self.assertEqual([c.target.target_id for c in contributions], [ARTIST_A])

    def test_discount_reduces_and_marks(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        contributions = positive(
            artist_ids=[ARTIST_A], constraints=[binding], discount_attenuation=_FactorDiscount(0.5)
        )
        self.assertEqual(len(contributions), 1)
        self.assertEqual(contributions[0].derived_magnitude, 0.45)
        self.assertIs(contributions[0].constraint, ConstraintMode.DISCOUNT)

    def test_discount_without_policy_fails_closed(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A], constraints=[binding])

    def test_multi_artist_without_split_policy_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A, ARTIST_B])

    def test_invalid_artist_id_fails_closed(self) -> None:
        for bad in (
            "artist-from-name",
            "trk_00000000-0000-4000-8000-000000000000",
            "art_not-a-uuid",
            5,
            None,
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    positive(artist_ids=[bad])

    def test_duplicate_artist_id_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A, ARTIST_A], artist_split=even_split())


class AlbumPropagationTest(unittest.TestCase):
    def test_album_present_contributes(self) -> None:
        contributions = positive(album_id=ALBUM_ID)
        self.assertEqual(len(contributions), 1)
        contribution = contributions[0]
        self.assertIs(contribution.kind, PropagationKind.ALBUM)
        self.assertEqual(contribution.target, ref(PreferenceTargetKind.ALBUM, ALBUM_ID))
        self.assertEqual(contribution.derived_magnitude, 0.9)
        self.assertEqual(contribution.split_count, 1)

    def test_album_missing_contributes_nothing(self) -> None:
        self.assertEqual(positive(album_id=None), [])

    def test_album_attenuation_policy_applies(self) -> None:
        contributions = positive(album_id=ALBUM_ID, album_attenuation=_FactorAlbum(0.5))
        self.assertEqual(contributions[0].derived_magnitude, 0.45)
        self.assertEqual(contributions[0].input_magnitude, 0.9)

    def test_album_block_removes_contribution(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ALBUM, ALBUM_ID), AttributionConstraint(ConstraintMode.BLOCK)
        )
        self.assertEqual(positive(album_id=ALBUM_ID, constraints=[binding]), [])

    def test_album_discount_reduces_and_marks(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ALBUM, ALBUM_ID), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        contributions = positive(
            album_id=ALBUM_ID, constraints=[binding], discount_attenuation=_FactorDiscount(0.5)
        )
        self.assertEqual(contributions[0].derived_magnitude, 0.45)
        self.assertIs(contributions[0].constraint, ConstraintMode.DISCOUNT)

    def test_invalid_album_id_fails_closed(self) -> None:
        for bad in ("album-from-name", 5, ""):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    positive(album_id=bad)


class GenrePropagationTest(unittest.TestCase):
    def test_single_genre_is_canonicalized(self) -> None:
        contributions = positive(genres=["  Hip   Hop  "])
        self.assertEqual(len(contributions), 1)
        self.assertIs(contributions[0].kind, PropagationKind.GENRE)
        self.assertEqual(contributions[0].target.target_id, "Hip Hop")
        self.assertEqual(contributions[0].derived_magnitude, 0.9)
        self.assertEqual(contributions[0].split_count, 1)

    def test_no_alias_merge(self) -> None:
        contributions = positive(genres=["Hip Hop", "Hip-Hop"], genre_split=even_split())
        self.assertEqual([c.target.target_id for c in contributions], ["Hip Hop", "Hip-Hop"])

    def test_multiple_genres_are_split_without_amplification(self) -> None:
        contributions = positive(genres=["rock", "jazz", "blues"], genre_split=even_split())
        self.assertEqual(len(contributions), 3)
        self.assertEqual([c.derived_magnitude for c in contributions], [0.3, 0.3, 0.3])
        total = sum(c.derived_magnitude for c in contributions)
        self.assertAlmostEqual(total, 0.9)

    def test_multi_genre_without_split_policy_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(genres=["rock", "jazz"])

    def test_genre_block_removes_that_genre(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.GENRE, "jazz"), AttributionConstraint(ConstraintMode.BLOCK)
        )
        contributions = positive(genres=["rock", "jazz"], genre_split=even_split(), constraints=[binding])
        self.assertEqual([c.target.target_id for c in contributions], ["rock"])

    def test_genre_discount_reduces_and_marks(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.GENRE, "rock"), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        contributions = positive(
            genres=["rock"], constraints=[binding], discount_attenuation=_FactorDiscount(0.5)
        )
        self.assertEqual(contributions[0].derived_magnitude, 0.45)
        self.assertIs(contributions[0].constraint, ConstraintMode.DISCOUNT)

    def test_empty_genre_fails_closed(self) -> None:
        for bad in ("", "   ", "\t\n"):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    positive(genres=[bad])

    def test_non_string_genre_fails_closed(self) -> None:
        for bad in (None, 5, ["rock"]):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    positive(genres=[bad])

    def test_duplicate_genre_after_canonicalization_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(genres=["Hip Hop", "  Hip   Hop  "], genre_split=even_split())


class PreferenceStatePropagationTest(unittest.TestCase):
    def test_positive_propagates(self) -> None:
        self.assertEqual(len(positive(artist_ids=[ARTIST_A])), 1)

    def test_negative_propagates(self) -> None:
        self.assertEqual(len(propagate(PreferenceState.NEGATIVE, 0.8, artist_ids=[ARTIST_A])), 1)

    def test_unknown_does_not_propagate(self) -> None:
        self.assertEqual(propagate(PreferenceState.UNKNOWN, None, artist_ids=[ARTIST_A]), [])

    def test_insufficient_does_not_propagate(self) -> None:
        self.assertEqual(propagate(PreferenceState.INSUFFICIENT, None, artist_ids=[ARTIST_A]), [])

    def test_neutral_does_not_propagate(self) -> None:
        self.assertEqual(propagate(PreferenceState.NEUTRAL, 0.0, artist_ids=[ARTIST_A]), [])

    def test_conflict_does_not_propagate(self) -> None:
        self.assertEqual(propagate(PreferenceState.CONFLICT, None, artist_ids=[ARTIST_A]), [])


class AttributionConstraintBindingTest(unittest.TestCase):
    def test_valid_binding(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.BLOCK)
        )
        self.assertIs(binding.constraint.mode, ConstraintMode.BLOCK)

    def test_non_track_source_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            AttributionConstraintBinding(
                ref(PreferenceTargetKind.ARTIST, ARTIST_A),
                ref(PreferenceTargetKind.ARTIST, ARTIST_A),
                AttributionConstraint(ConstraintMode.BLOCK),
            )

    def test_track_target_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            AttributionConstraintBinding(
                track_target(),
                track_target(),
                AttributionConstraint(ConstraintMode.BLOCK),
            )

    def test_invalid_constraint_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            AttributionConstraintBinding(
                track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), "block"
            )

    def test_frozen(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.BLOCK)
        )
        with self.assertRaises(FrozenInstanceError):
            binding.constraint = AttributionConstraint(ConstraintMode.DISCOUNT)


class DeterminismAndOrderTest(unittest.TestCase):
    def test_full_track_order_is_stable(self) -> None:
        contributions = positive(
            artist_ids=[ARTIST_A, ARTIST_B],
            album_id=ALBUM_ID,
            genres=["rock", "jazz"],
            artist_split=even_split(),
            genre_split=even_split(),
        )
        kinds = [c.kind for c in contributions]
        self.assertEqual(
            kinds,
            [PropagationKind.ARTIST, PropagationKind.ARTIST, PropagationKind.ALBUM, PropagationKind.GENRE, PropagationKind.GENRE],
        )

    def test_deterministic_result(self) -> None:
        def run():
            return positive(artist_ids=[ARTIST_A, ARTIST_B], genres=["rock"], artist_split=even_split(), genre_split=even_split())

        self.assertEqual(run(), run())


class SafetyTest(unittest.TestCase):
    def test_direct_preference_unchanged(self) -> None:
        preference = direct(PreferenceState.POSITIVE, 0.9)
        before = DerivedPreference(preference.target, PreferenceStrength(PreferenceState.POSITIVE, 0.9))
        propagate_track_preference(preference, artist_ids=[ARTIST_A])
        self.assertEqual(preference, before)
        self.assertEqual(preference.strength.magnitude, 0.9)

    def test_provenance_is_fixed_inferred(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A])
        self.assertIs(contributions[0].provenance, PreferenceProvenance.INFERRED)

    def test_contribution_has_no_confidence_field(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A])
        for field in ("confidence", "score", "quality", "quantity"):
            self.assertFalse(hasattr(contributions[0], field), field)

    def test_contribution_has_no_temporal_field(self) -> None:
        contributions = positive(artist_ids=[ARTIST_A])
        for field in ("observed_at", "event_at", "now", "decay", "recency"):
            self.assertFalse(hasattr(contributions[0], field), field)

    def test_no_hardcoded_production_attenuation_constant(self) -> None:
        import music_agent.preference_propagation as module

        for name, value in vars(module).items():
            if name.isupper():
                self.assertNotIsInstance(value, (int, float), f"numeric constant {name}")

    def test_module_reads_no_clock(self) -> None:
        import music_agent.preference_propagation as module

        source = inspect.getsource(module)
        for banned in ("datetime.now(", "datetime.utcnow(", "datetime.today(", "time.time(", "time.monotonic("):
            self.assertNotIn(banned, source, banned)

    def test_module_has_no_aggregation_function(self) -> None:
        import music_agent.preference_propagation as module

        for name, value in vars(module).items():
            if callable(value) and not inspect.isclass(value):
                self.assertNotIn("aggregat", name, f"aggregation helper {name} must not exist")
                self.assertNotIn("reduc", name, f"reducer helper {name} must not exist")

    def test_module_performs_no_persistence(self) -> None:
        import music_agent.preference_propagation as module

        source = inspect.getsource(module)
        for banned in ("open(", "sqlite", "connect(", "write(", "insert("):
            self.assertNotIn(banned, source, banned)


class PolicyValidationTest(unittest.TestCase):
    def test_non_policy_object_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split="split")

    def test_amplifying_split_result_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split=_AmplifyingArtistSplit())

    def test_nan_split_result_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A, ARTIST_B], artist_split=_NaNArtistSplit())

    def test_discount_that_amplifies_fails_closed(self) -> None:
        binding = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A], constraints=[binding], discount_attenuation=_FactorDiscount(2.0))


class ValidationTest(unittest.TestCase):
    def test_non_derived_preference_fails_closed(self) -> None:
        for bad in (None, "preference", 5, PreferenceStrength(PreferenceState.POSITIVE, 0.9)):
            with self.subTest(bad=bad):
                with self.assertRaises(PropagationValidationError):
                    propagate_track_preference(bad, artist_ids=[ARTIST_A])

    def test_non_track_target_fails_closed(self) -> None:
        artist_preference = DerivedPreference(
            ref(PreferenceTargetKind.ARTIST, ARTIST_A), PreferenceStrength(PreferenceState.POSITIVE, 0.9)
        )
        with self.assertRaises(PropagationValidationError):
            propagate_track_preference(artist_preference, artist_ids=[ARTIST_A])

    def test_conflicting_constraints_fail_closed(self) -> None:
        first = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.BLOCK)
        )
        second = AttributionConstraintBinding(
            track_target(), ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.DISCOUNT)
        )
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A], constraints=[first, second])

    def test_constraint_for_wrong_track_fails_closed(self) -> None:
        other_track = PreferenceTargetReference(
            PreferenceTargetKind.TRACK, "trk_00000000-0000-4000-8000-000000000001"
        )
        binding = AttributionConstraintBinding(
            other_track, ref(PreferenceTargetKind.ARTIST, ARTIST_A), AttributionConstraint(ConstraintMode.BLOCK)
        )
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=[ARTIST_A], constraints=[binding])

    def test_string_is_not_a_relationship_list(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=ARTIST_A)
        with self.assertRaises(PropagationValidationError):
            positive(genres="rock")

    def test_non_iterable_relationship_list_fails_closed(self) -> None:
        with self.assertRaises(PropagationValidationError):
            positive(artist_ids=5)


class ImmutabilityEqualityTest(unittest.TestCase):
    def test_contribution_is_frozen(self) -> None:
        contribution = positive(artist_ids=[ARTIST_A])[0]
        with self.assertRaises(FrozenInstanceError):
            contribution.derived_magnitude = 0.1

    def test_contribution_equality_and_hash(self) -> None:
        first = positive(artist_ids=[ARTIST_A])[0]
        second = positive(artist_ids=[ARTIST_A])[0]
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_contribution_fields_are_exactly_declared(self) -> None:
        expected = [
            "source_track",
            "target",
            "direction",
            "input_magnitude",
            "derived_magnitude",
            "kind",
            "constraint",
            "split_count",
        ]
        self.assertEqual([f.name for f in fields(InferredAffinityContribution)], expected)

    def test_invalid_contribution_fields_fail_closed(self) -> None:
        artist_target = ref(PreferenceTargetKind.ARTIST, ARTIST_A)
        with self.assertRaises(PropagationValidationError):
            InferredAffinityContribution(
                source_track=track_target(),
                target=artist_target,
                direction=SignalDirection.POSITIVE,
                input_magnitude=0.9,
                derived_magnitude=0.95,
                kind=PropagationKind.ARTIST,
                constraint=None,
                split_count=1,
            )
        with self.assertRaises(PropagationValidationError):
            InferredAffinityContribution(
                source_track=artist_target,
                target=artist_target,
                direction=SignalDirection.POSITIVE,
                input_magnitude=0.9,
                derived_magnitude=0.9,
                kind=PropagationKind.ARTIST,
                constraint=None,
                split_count=1,
            )
        with self.assertRaises(PropagationValidationError):
            InferredAffinityContribution(
                source_track=track_target(),
                target=artist_target,
                direction=SignalDirection.NO_CLAIM,
                input_magnitude=0.9,
                derived_magnitude=0.9,
                kind=PropagationKind.ARTIST,
                constraint=None,
                split_count=1,
            )
        with self.assertRaises(PropagationValidationError):
            InferredAffinityContribution(
                source_track=track_target(),
                target=artist_target,
                direction=SignalDirection.POSITIVE,
                input_magnitude=0.9,
                derived_magnitude=0.9,
                kind=PropagationKind.ALBUM,
                constraint=None,
                split_count=1,
            )


class PolicySeamAbstractionTest(unittest.TestCase):
    def test_policy_protocols_are_abstract_seams(self) -> None:
        for protocol in (
            ArtistSplitPolicy,
            AlbumAttenuationPolicy,
            GenreSplitPolicy,
            DiscountAttenuationPolicy,
        ):
            with self.subTest(protocol=protocol.__name__):
                with self.assertRaises(TypeError):
                    protocol()  # type: ignore[misc]


class ErrorHierarchyTest(unittest.TestCase):
    def test_validation_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(PropagationValidationError, ValueError))

    def test_error_code_is_stable(self) -> None:
        self.assertEqual(PropagationValidationError.code, "validation_error")


if __name__ == "__main__":
    unittest.main()
